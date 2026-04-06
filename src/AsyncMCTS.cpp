// AsyncMCTS.cpp — Phase 4: async MCTS implementation
//
// select(), expand(), backpropagate() mirror MCTS.cpp exactly.
//
// Key design change (performance):
//   run_simulations() pipelines GPU evaluation with CPU selection — while the
//   CentralEvaluator processes batch N, this worker already selects batch N+1
//   and pushes it to the queue.  This hides GPU latency behind CPU work and
//   keeps both CPU and GPU continuously busy.
//
//   run_simulation_batch() is kept for test compatibility; it is now just
//   select_and_submit() + flush_pending() with no overlap.

#include "AsyncMCTS.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>  // memcpy
#include <future>
#include <iostream>
#include <limits>

AsyncMCTS::AsyncMCTS(EvalQueue* queue)
    : queue_(queue)
{}

// ── select ────────────────────────────────────────────────────────────────────
// Mirrors MCTS::select() exactly: UCB tree walk with virtual loss.

Node* AsyncMCTS::select(Node& root, std::vector<PathEntry>& path) {
    Node* node = &root;

    while (node->is_expanded) {
        const float sqrt_n = std::sqrt(static_cast<float>(node->total_N + 1));

        float best_score  = -std::numeric_limits<float>::infinity();
        int   best_action = -1;

        const auto& legal = *node->legal_moves_cache;
        const auto& Qv    = *node->Q;
        const auto& Pv    = *node->P;
        const auto& Nv    = *node->N;

        for (int a = 0; a < ACTION_DIM; ++a) {
            if (!legal[a]) continue;
            const float ucb = Qv[a] + C_PUCT * Pv[a] * sqrt_n / (1.0f + Nv[a]);
            if (ucb > best_score) {
                best_score  = ucb;
                best_action = a;
            }
        }

        if (best_action < 0) break;  // terminal node — no legal moves

        path.push_back({node, best_action});

        // Virtual loss: discourage other sims in the same batch from this edge.
        (*node->W)[best_action] -= 1.0f;
        (*node->Q)[best_action]  = (*node->W)[best_action]
                                    / std::max((*node->N)[best_action], 1.0f);

        // Traverse to child (create it if this is the first visit).
        auto it = node->children.find(best_action);
        if (it == node->children.end()) {
            chess::Board child_board = node->board;
            auto res = ChessEnv::apply_action(best_action, child_board);
            if (res.first) {
                auto child_node = std::make_unique<Node>(child_board);
                Node* child_ptr = child_node.get();
                node->children[best_action] = std::move(child_node);
                node = child_ptr;
            }
            break;  // new child is always unexpanded
        }
        node = it->second.get();
    }

    return node;
}

// ── expand ────────────────────────────────────────────────────────────────────
// Mirrors MCTS::expand() but takes a pre-fetched EvalResult instead of raw ptrs.

float AsyncMCTS::expand(Node& leaf, const EvalResult& result) {
    leaf.allocate_stats();

    *leaf.legal_moves_cache = ChessEnv::get_action_mask(leaf.board);
    const auto& legal = *leaf.legal_moves_cache;

    // Masked softmax over raw logits for numerical stability.
    float max_p    = -std::numeric_limits<float>::infinity();
    bool  any_legal = false;
    for (int a = 0; a < ACTION_DIM; ++a) {
        if (legal[a]) {
            if (result.policy[a] > max_p) max_p = result.policy[a];
            any_legal = true;
        }
    }

    auto&  Pv      = *leaf.P;
    float  sum_exp = 0.0f;
    for (int a = 0; a < ACTION_DIM; ++a) {
        if (any_legal && legal[a]) {
            const float e = std::exp(result.policy[a] - max_p);
            Pv[a]    = e;
            sum_exp += e;
        } else {
            Pv[a] = 0.0f;
        }
    }

    const float inv = 1.0f / (sum_exp + 1e-10f);
    for (int a = 0; a < ACTION_DIM; ++a) Pv[a] *= inv;

    leaf.is_expanded = true;
    return result.value;
}

// ── backpropagate ─────────────────────────────────────────────────────────────
// Mirrors MCTS::backpropagate() exactly.

void AsyncMCTS::backpropagate(const std::vector<PathEntry>& path, float value) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Node* node   = it->first;
        int   action = it->second;

        value = -value;  // flip perspective before assignment

        (*node->N)[action] += 1.0f;
        (*node->W)[action] += value + 1.0f;  // +1 cancels virtual loss
        (*node->Q)[action]  = (*node->W)[action] / (*node->N)[action];
        node->total_N      += 1;
    }
}

// ── select_and_submit ─────────────────────────────────────────────────────────
// Selects leaf_batch_size leaves (applying virtual loss), encodes each board
// state, and pushes EvalRequests to the queue.  Returns immediately — the
// caller is responsible for awaiting the futures via flush_pending().

AsyncMCTS::PendingBatch
AsyncMCTS::select_and_submit(Node& root, int leaf_batch_size) {
    PendingBatch pending;
    pending.leaves.resize(leaf_batch_size, nullptr);
    pending.paths.resize(leaf_batch_size);
    pending.futures.reserve(leaf_batch_size);

    // Selection (with virtual loss applied per leaf).
    for (int i = 0; i < leaf_batch_size; ++i) {
        pending.leaves[i] = select(root, pending.paths[i]);
    }

    // Encode and push to queue — non-blocking.
    // encode_state_into writes directly into req.state (no torch allocation).
    for (int i = 0; i < leaf_batch_size; ++i) {
        EvalRequest req;
        ChessEnv::encode_state_into(pending.leaves[i]->board, req.state.data());
        pending.futures.push_back(req.promise.get_future());
        queue_->push(std::move(req));
    }

    return pending;
}

// ── flush_pending ─────────────────────────────────────────────────────────────
// Blocks until all futures in a PendingBatch are fulfilled, then expands and
// backpropagates each leaf using the returned NN results.

void AsyncMCTS::flush_pending(PendingBatch& pending) {
    const int n = static_cast<int>(pending.leaves.size());

    std::vector<EvalResult> results;
    results.reserve(n);
    for (int fi = 0; fi < (int)pending.futures.size(); ++fi) {
        auto& f = pending.futures[fi];
        // Wait with timeout — if stuck for >30 s, print a diagnostic.
        constexpr int WARN_SECS = 30;
        auto status = f.wait_for(std::chrono::seconds(WARN_SECS));
        if (status == std::future_status::timeout) {
            std::cerr << "[DEBUG flush_pending] STUCK waiting for future " << fi
                      << "/" << n << " for >" << WARN_SECS << "s"
                      << "  queue_pending=" << queue_->pending_count()
                      << "  active_workers=" << queue_->active_worker_count()
                      << " — possible deadlock!\n"
                      << std::flush;
            // Keep waiting (now without timeout) so we don't lose the future.
        }
        results.push_back(f.get());
    }

    for (int i = 0; i < n; ++i) {
        Node* leaf  = pending.leaves[i];
        float value = 0.0f;

        if (!leaf->is_expanded) {
            auto go = ChessEnv::game_over(leaf->board);
            if (go.first) {
                value = go.second ? -1.0f : 0.0f;
                leaf->allocate_stats();
                leaf->is_expanded = true;
            } else {
                value = expand(*leaf, results[i]);
            }
        } else {
            // Duplicate leaf in same batch — NN value is still usable.
            auto go = ChessEnv::game_over(leaf->board);
            value = go.first ? (go.second ? -1.0f : 0.0f) : results[i].value;
        }

        backpropagate(pending.paths[i], value);
    }
}

// ── run_simulations ───────────────────────────────────────────────────────────
// Pipelined simulation: while the GPU evaluates batch N, the CPU selects and
// submits batch N+1.  This eliminates the idle CPU time present in the naive
// select→submit→wait→process loop.
//
// Safety: batch N+1 selection is done after N's virtual losses are applied but
// before N's expand+backprop.  This is the same trade-off as ordinary batched
// virtual loss and does not introduce data races (each worker owns its tree).

void AsyncMCTS::run_simulations(Node& root, int num_simulations, int leaf_batch_size) {
    const int num_batches = num_simulations / leaf_batch_size;
    if (num_batches <= 0) return;

    // Seed: submit first batch (no CPU work to overlap yet).
    PendingBatch cur = select_and_submit(root, leaf_batch_size);

    for (int b = 1; b < num_batches; ++b) {
        // Overlap: select + submit next batch while GPU evaluates current.
        PendingBatch nxt = select_and_submit(root, leaf_batch_size);
        flush_pending(cur);          // wait for current, expand + backprop
        cur = std::move(nxt);
    }

    flush_pending(cur);              // drain final batch
}

// ── run_simulation_batch ──────────────────────────────────────────────────────
// Single-batch helper kept for backward compatibility.
// For performance-critical paths use run_simulations() instead.

void AsyncMCTS::run_simulation_batch(Node& root, int leaf_batch_size) {
    auto pending = select_and_submit(root, leaf_batch_size);
    flush_pending(pending);
}

// ── best_action ───────────────────────────────────────────────────────────────

int AsyncMCTS::best_action(const Node& root) const {
    if (!root.N) return -1;
    const auto& n = *root.N;
    return static_cast<int>(
        std::max_element(n.begin(), n.end()) - n.begin()
    );
}
