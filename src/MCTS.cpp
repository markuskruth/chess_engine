// MCTS.cpp — Phase 3 implementation
//
// Mirrors Python's MCTS class in Agent.py.
// Key design decisions:
//   - select()         applies virtual loss during tree walk (diversifies batch paths).
//   - expand()         runs a masked softmax over raw NN logits to produce priors P.
//   - backpropagate()  flips value sign at each step (perspective change) and adds +1
//                      to W to cancel the -1 virtual loss applied during selection.
//   - run_simulation_batch()  batches all leaf NN evaluations into a single forward pass.

#include "MCTS.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

// ── Node ─────────────────────────────────────────────────────────────────────

void Node::allocate_stats() {
    P                 = std::make_unique<std::array<float, ACTION_DIM>>();
    N                 = std::make_unique<std::array<float, ACTION_DIM>>();
    W                 = std::make_unique<std::array<float, ACTION_DIM>>();
    Q                 = std::make_unique<std::array<float, ACTION_DIM>>();
    legal_moves_cache = std::make_unique<ActionMask>();
    P->fill(0.0f);
    N->fill(0.0f);
    W->fill(0.0f);
    Q->fill(0.0f);
    legal_moves_cache->fill(false);
}

// ── MCTS ─────────────────────────────────────────────────────────────────────

MCTS::MCTS(torch::jit::Module model, torch::Device device)
    : model_(std::move(model)), device_(device)
{
    model_.to(device_);
    model_.eval();
}

// ── select ────────────────────────────────────────────────────────────────────
//
// Walk the tree using UCB until we reach an unexpanded node.
// At each step:
//   1. Choose argmax UCB over legal moves.
//   2. Record the step in `path`.
//   3. Apply virtual loss (W -= 1) so later sims in the same batch explore
//      different branches.
//   4. Follow the child, creating it if necessary.
// Returns the leaf node (which is unexpanded, or a terminal already-expanded node).

Node* MCTS::select(Node& root, std::vector<PathEntry>& path) {
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

        // Virtual loss: make this edge look worse to other sims in the same batch
        (*node->W)[best_action] -= 1.0f;
        (*node->Q)[best_action]  = (*node->W)[best_action]
                                    / std::max((*node->N)[best_action], 1.0f);

        // Traverse to child (create it if this is the first visit to this edge)
        auto it = node->children.find(best_action);
        if (it == node->children.end()) {
            chess::Board child_board = node->board;
            auto res = ChessEnv::apply_action(best_action, child_board);
            if (res.first) {  // valid action
                auto child_node = std::make_unique<Node>(child_board);
                Node* child_ptr = child_node.get();
                node->children[best_action] = std::move(child_node);
                node = child_ptr;
            }
            break;  // new child is always unexpanded — stop selecting
        }
        node = it->second.get();
    }

    return node;
}

// ── expand ────────────────────────────────────────────────────────────────────
//
// Initialise a leaf node's statistics from raw NN policy logits via masked softmax.
// Mirrors Python's expand() in Agent.py.

float MCTS::expand(Node& leaf, float p_raw[], float v_raw) {
    leaf.allocate_stats();

    // Cache legal move mask for this node
    *leaf.legal_moves_cache = ChessEnv::get_action_mask(leaf.board);
    const auto& legal = *leaf.legal_moves_cache;

    // Masked softmax over p_raw:
    //   1. Find max logit over legal actions (for numerical stability).
    //   2. exp(p - max) for legal, 0 for illegal.
    //   3. Normalise.
    float max_p    = -std::numeric_limits<float>::infinity();
    bool  any_legal = false;
    for (int a = 0; a < ACTION_DIM; ++a) {
        if (legal[a]) {
            if (p_raw[a] > max_p) max_p = p_raw[a];
            any_legal = true;
        }
    }

    auto&  Pv     = *leaf.P;
    float  sum_exp = 0.0f;
    for (int a = 0; a < ACTION_DIM; ++a) {
        if (any_legal && legal[a]) {
            const float e = std::exp(p_raw[a] - max_p);
            Pv[a]    = e;
            sum_exp += e;
        } else {
            Pv[a] = 0.0f;
        }
    }

    const float inv = 1.0f / (sum_exp + 1e-10f);
    for (int a = 0; a < ACTION_DIM; ++a) Pv[a] *= inv;

    leaf.is_expanded = true;
    return v_raw;
}

// ── backpropagate ─────────────────────────────────────────────────────────────
//
// Walk `path` in reverse, flipping the value at each step (perspective change)
// and updating N, W, Q, total_N.
//
// W[a] += value + 1.0  — the +1 cancels the virtual loss of -1 from select().
// Value is flipped BEFORE the assignment (critical fix from the migration plan).

void MCTS::backpropagate(const std::vector<PathEntry>& path, float value) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Node* node   = it->first;
        int   action = it->second;

        value = -value;  // flip perspective BEFORE assignment

        (*node->N)[action] += 1.0f;
        (*node->W)[action] += value + 1.0f;  // +1 cancels virtual loss
        (*node->Q)[action]  = (*node->W)[action] / (*node->N)[action];
        node->total_N      += 1;
    }
}

// ── run_simulation_batch ──────────────────────────────────────────────────────
//
// Run `leaf_batch_size` simulations in a single batched NN forward pass.
// Mirrors Python's run_simulation_batch() in Agent.py, including virtual loss
// and the guard for duplicate paths hitting the same leaf.

void MCTS::run_simulation_batch(Node& root, int leaf_batch_size) {
    std::vector<Node*>                  leaves(leaf_batch_size, nullptr);
    std::vector<std::vector<PathEntry>> all_paths(leaf_batch_size);

    // ── SELECTION (with virtual loss) ─────────────────────────────────────────
    for (int i = 0; i < leaf_batch_size; ++i) {
        leaves[i] = select(root, all_paths[i]);
    }

    // ── BATCHED NN FORWARD ────────────────────────────────────────────────────
    // Encode all leaf boards → stack into (B, 20, 8, 8) tensor
    auto states = torch::zeros(
        {leaf_batch_size, STATE_CHANNELS, BOARD_SZ, BOARD_SZ}, torch::kFloat32);
    for (int i = 0; i < leaf_batch_size; ++i) {
        states[i] = ChessEnv::encode_state(leaves[i]->board);
    }
    states = states.to(device_);

    torch::Tensor p_batch, v_batch;
    {
        torch::NoGradGuard no_grad;
        auto out = model_.forward({states}).toTuple();
        // policy: (B, 8, 8, 73) → (B, 4672)  [C-order matches our action index]
        p_batch = out->elements()[0].toTensor()
                     .to(torch::kCPU).contiguous()
                     .reshape({leaf_batch_size, ACTION_DIM});
        // value:  (B, 1) → (B,)
        v_batch = out->elements()[1].toTensor()
                     .to(torch::kCPU).contiguous()
                     .reshape({leaf_batch_size});
    }

    float* p_data = p_batch.data_ptr<float>();
    float* v_data = v_batch.data_ptr<float>();

    // ── EXPAND + BACKPROP ─────────────────────────────────────────────────────
    for (int i = 0; i < leaf_batch_size; ++i) {
        Node*  leaf  = leaves[i];
        float  value = 0.0f;

        if (!leaf->is_expanded) {
            auto go = ChessEnv::game_over(leaf->board);
            if (go.first) {
                // Terminal node: exact value, skip NN output
                value = go.second ? -1.0f : 0.0f;
                leaf->allocate_stats();   // zeroed-out P, N, W, Q, mask
                leaf->is_expanded = true;
            } else {
                // Normal leaf: expand with NN priors
                value = expand(*leaf, p_data + i * ACTION_DIM, v_data[i]);
            }
        } else {
            // Duplicate path in same batch: re-use NN value (or exact for terminals)
            auto go = ChessEnv::game_over(leaf->board);
            value = go.first ? (go.second ? -1.0f : 0.0f) : v_data[i];
        }

        backpropagate(all_paths[i], value);
    }
}

// ── best_action ───────────────────────────────────────────────────────────────

int MCTS::best_action(const Node& root) const {
    if (!root.N) return -1;
    const auto& n = *root.N;
    return static_cast<int>(
        std::max_element(n.begin(), n.end()) - n.begin()
    );
}
