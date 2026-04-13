// ParallelSelfPlay.cpp — Phase 4 implementation
//
// Orchestration:
//   1. Load TorchScript model → CentralEvaluator (GPU thread).
//   2. Launch num_workers game threads, each with its own AsyncMCTS.
//   3. Join workers; all games complete; notify_worker_done() called by each.
//   4. Shutdown EvalQueue; join CentralEvaluator thread.

#include "ParallelSelfPlay.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

// ── Curriculum FENs ───────────────────────────────────────────────────────────
// Used for episodes 1–5 instead of the standard start position.
// Goals: generate decisive games early in training so the value head receives
// non-zero reward signal from the very first episode.
//
// Mix of mate-in-1 positions (verified) and overwhelming material advantages
// (K+2Q/K+2R/K+Q+R vs lone king, both White-wins and Black-wins variants).

static constexpr int CURRICULUM_EPISODES = 5;

static const std::vector<std::string> CURRICULUM_FENS = {
    // ── Mate-in-1 (White to move) ─────────────────────────────────────────────
    "1k6/8/1KQ5/8/8/8/8/8 w - - 0 1",   // Kb6 + Qc6 vs kb8  →  Qc8#
    "6k1/5Q2/6K1/8/8/8/8/8 w - - 0 1",  // Kg6 + Qf7 vs kg8  →  Qf8#
    "k7/2Q5/2K5/8/8/8/8/8 w - - 0 1",   // Kc6 + Qc7 vs ka8  →  Qa7#
    // ── Mate-in-1 (Black to move) ─────────────────────────────────────────────
    "8/8/8/8/8/8/1q6/1k5K b - - 0 1",   // kb1 + qb2 vs Kh1  →  Qh2#
    "K7/8/kq6/8/8/8/8/8 b - - 0 1",     // ka6 + qb6 vs Ka8  →  Qa7#
    "8/8/8/8/8/1k6/2q5/K7 b - - 0 1",   // kb3 + qc2 vs Ka1  →  Qb2#
    // ── K + 2 Queens vs lone king (White wins) ────────────────────────────────
    "8/8/3k4/8/8/8/8/2QKQ3 w - - 0 1",
    // ── K + 2 Rooks vs lone king (White wins) ────────────────────────────────
    "8/8/3k4/8/8/8/8/2RKR3 w - - 0 1",
    // ── K + Queen + Rook vs lone king (White wins) ────────────────────────────
    "8/8/3k4/8/8/8/8/2QKR3 w - - 0 1",
    // ── k + 2 queens vs lone King (Black wins) ────────────────────────────────
    "3qkq2/8/8/4K3/8/8/8/8 b - - 0 1",
    // ── k + 2 rooks vs lone King (Black wins) ────────────────────────────────
    "3rkr2/8/8/4K3/8/8/8/8 b - - 0 1",
    // ── k + queen + rook vs lone King (Black wins) ────────────────────────────
    "3qkr2/8/8/4K3/8/8/8/8 b - - 0 1",
};

ParallelSelfPlay::ParallelSelfPlay(const ParallelSelfPlayConfig& cfg)
    : cfg_(cfg)
{}

// ── run ───────────────────────────────────────────────────────────────────────

std::pair<std::vector<Sample>, std::vector<GameMeta>>
ParallelSelfPlay::run() {
    // --- ADD THIS LINE TO PREVENT THE OPENMP DEADLOCK ---
    torch::set_num_threads(1);

    // Open output file and write a placeholder header.
    // The real num_samples is written after all workers complete.
    if (!cfg_.output_path.empty()) {
        out_file_.open(cfg_.output_path, std::ios::binary);
        if (!out_file_)
            throw std::runtime_error("ParallelSelfPlay: cannot open output file: " + cfg_.output_path);
        const uint32_t zero = 0, ch = STATE_CHANNELS, bsz = BOARD_SZ;
        out_file_.write(reinterpret_cast<const char*>(&zero), sizeof(uint32_t));
        out_file_.write(reinterpret_cast<const char*>(&ch),   sizeof(uint32_t));
        out_file_.write(reinterpret_cast<const char*>(&bsz),  sizeof(uint32_t));
    }

    // Load model for the central evaluator.
    torch::jit::Module model;
    try {
        model = torch::jit::load(cfg_.model_path);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "ParallelSelfPlay: cannot load model: " + std::string(e.what()));
    }
    const auto device = torch::cuda::is_available()
                        ? torch::Device(torch::kCUDA)
                        : torch::Device(torch::kCPU);

    // Shared eval queue (num_workers producers, 1 consumer).
    EvalQueue queue(cfg_.num_workers);

    // Start central evaluator thread.
    CentralEvaluator evaluator(
        std::move(model), &queue, device,
        cfg_.min_batch(), cfg_.max_batch());
    evaluator.start();

    // Distribute games across workers (last worker takes the remainder).
    const int base    = cfg_.num_games / cfg_.num_workers;
    const int extra   = cfg_.num_games % cfg_.num_workers;

    std::vector<std::thread> workers;
    workers.reserve(cfg_.num_workers);
    for (int w = 0; w < cfg_.num_workers; ++w) {
        const int games_this_worker = base + (w < extra ? 1 : 0);
        if (games_this_worker == 0) {
            // No games for this worker — still need to signal done.
            queue.notify_worker_done();
            continue;
        }
        workers.emplace_back([this, games_this_worker, &queue] {
            worker_fn(games_this_worker, &queue);
        });
    }

    // Wait for all workers to finish.
    for (auto& t : workers) t.join();

    // Signal evaluator to exit (workers have all called notify_worker_done,
    // which drives active_workers_ to 0 and unblocks any partial batch;
    // shutdown() ensures the evaluator exits cleanly if the queue is already empty).
    queue.shutdown();
    evaluator.join();

    // Seek back to position 0 and write the real sample count into the header.
    if (out_file_.is_open()) {
        out_file_.flush();
        out_file_.seekp(0, std::ios::beg);
        out_file_.write(reinterpret_cast<const char*>(&total_samples_written_),
                        sizeof(uint32_t));
        out_file_.close();
    }

    return {{}, all_meta_};  // samples are already on disk; return empty vector
}

// ── worker_fn ─────────────────────────────────────────────────────────────────

void ParallelSelfPlay::worker_fn(int num_games, EvalQueue* queue) {
    auto tid = std::this_thread::get_id();
    std::cerr << "[DEBUG worker " << tid << "] started, num_games=" << num_games << "\n" << std::flush;
    AsyncMCTS mcts(queue);
    try {
        for (int g = 0; g < num_games; ++g) {
            auto [samples, meta] = play_game(mcts, cfg_.temperature,
                                              cfg_.dirichlet_alpha,
                                              cfg_.dirichlet_epsilon);
            std::cerr << "[DEBUG worker " << tid << "] game " << (g+1) << "/" << num_games
                      << "  moves=" << meta.moves << "  outcome=" << meta.outcome << "\n" << std::flush;
            {
                std::lock_guard<std::mutex> lk(results_mutex_);
                // Stream samples directly to disk — no RAM accumulation.
                if (out_file_.is_open()) {
                    for (const auto& s : samples) {
                        out_file_.write(reinterpret_cast<const char*>(s.state.data()),
                                        s.state.size() * sizeof(float));
                        out_file_.write(reinterpret_cast<const char*>(s.pi.data()),
                                        s.pi.size() * sizeof(float));
                        out_file_.write(reinterpret_cast<const char*>(&s.z), sizeof(float));
                    }
                    total_samples_written_ += static_cast<uint32_t>(samples.size());
                }
                all_meta_.push_back(std::move(meta));
            }
        }
    } catch (const std::exception& ex) {
        // Log and continue — other workers should not be affected.
        // The exception does not propagate past the thread boundary.
        std::cerr << "[ParallelSelfPlay] worker " << tid << " EXCEPTION: " << ex.what() << "\n" << std::flush;
    }
    std::cerr << "[DEBUG worker " << tid << "] done, calling notify_worker_done\n" << std::flush;
    queue->notify_worker_done();
    std::cerr << "[DEBUG worker " << tid << "] exited\n" << std::flush;
}

// ── play_game ─────────────────────────────────────────────────────────────────
// Mirrors SelfPlay::play_game() exactly, replacing MCTS with AsyncMCTS.

std::pair<std::vector<Sample>, GameMeta>
ParallelSelfPlay::play_game(AsyncMCTS& mcts, float temperature,
                             float dirichlet_alpha, float dirichlet_epsilon) {
    static thread_local std::mt19937 rng(std::random_device{}());

    // Select starting position: curriculum FENs for early episodes, normal start after.
    chess::Board board;
    if (cfg_.episode <= CURRICULUM_EPISODES) {
        std::uniform_int_distribution<int> fen_dist(
            0, static_cast<int>(CURRICULUM_FENS.size()) - 1);
        board = chess::Board(CURRICULUM_FENS[fen_dist(rng)]);
    }

    struct TrajEntry {
        std::array<float, STATE_CHANNELS * BOARD_SZ * BOARD_SZ> state;
        std::array<float, ACTION_DIM>                            pi;
        chess::Color                                             turn;
    };
    std::vector<TrajEntry> traj;
    traj.reserve(cfg_.max_moves);

    if (cfg_.num_simulations < cfg_.leaf_batch_size)
        throw std::runtime_error("ParallelSelfPlay: num_simulations must be >= leaf_batch_size");

    for (int step = 0; step < cfg_.max_moves; ++step) {
        if (ChessEnv::game_over(board).first) break;

        const chess::Color turn = board.sideToMove();
        Node root(board);

        // Pipelined: overlaps GPU evaluation with CPU selection for next batch.
        // Dirichlet noise is injected into root.P inside run_simulations after
        // the first NN evaluation, so all subsequent batches use the noisy prior.
        mcts.run_simulations(root, cfg_.num_simulations, cfg_.leaf_batch_size,
                             dirichlet_alpha, dirichlet_epsilon);

        // Build policy π from visit counts.
        // When num_simulations == leaf_batch_size (num_batches == 1), the root is
        // expanded but no child is ever visited (all 64 paths are empty), so every
        // N[a] stays 0. In that case fall back to the raw NN prior P as the policy
        // target — it still provides a valid gradient for the policy head.
        std::array<float, ACTION_DIM> pi{};
        float sum_n = 0.0f;
        if (root.N) {
            for (int a = 0; a < ACTION_DIM; ++a) sum_n += (*root.N)[a];
            if (sum_n > 0.0f) {
                const float inv = 1.0f / sum_n;
                for (int a = 0; a < ACTION_DIM; ++a) pi[a] = (*root.N)[a] * inv;
            } else if (root.P) {
                // Single-batch fallback: use NN prior as policy target.
                pi = *root.P;
            }
        }

        // Sample action: stochastic for the first 30 plies, greedy thereafter.
        const float eff_temp = (step < 30) ? temperature : 0.0f;
        int action_idx = 0;
        if (sum_n > 0.0f) {
            if (eff_temp < 1e-3f) {
                action_idx = mcts.best_action(root);
            } else {
                std::discrete_distribution<int> dist(pi.begin(), pi.end());
                action_idx = dist(rng);
            }
        } else if (root.is_expanded && root.P && root.legal_moves_cache) {
            // Single-batch fallback: no child was visited, use NN prior to pick a move.
            const auto& P     = *root.P;
            const auto& legal = *root.legal_moves_cache;
            if (eff_temp < 1e-3f) {
                float best_p = -1.0f;
                for (int a = 0; a < ACTION_DIM; ++a)
                    if (legal[a] && P[a] > best_p) { best_p = P[a]; action_idx = a; }
            } else {
                std::vector<float> probs(ACTION_DIM, 0.0f);
                for (int a = 0; a < ACTION_DIM; ++a)
                    if (legal[a]) probs[a] = P[a];
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                action_idx = dist(rng);
            }
        }

        // Encode and store state for this ply (no torch allocation).
        {
            TrajEntry entry;
            ChessEnv::encode_state_into(board, entry.state.data());
            entry.pi   = pi;
            entry.turn = turn;
            traj.push_back(entry);
        }

        ChessEnv::apply_action(action_idx, board);
    }

    // ── Terminal value from White's perspective ───────────────────────────────
    float       z       = 0.0f;
    std::string outcome = "limit";

    auto go = ChessEnv::game_over(board);
    if (go.first) {
        if (go.second) {  // checkmate: sideToMove is the mated player
            z       = (board.sideToMove() == chess::Color::BLACK) ?  1.0f : -1.0f;
            outcome = (board.sideToMove() == chess::Color::BLACK) ? "white" : "black";
        } else {
            z       = 0.0f;
            outcome = "draw";
        }
    } else {
        // Move limit reached — recorded as a draw with zero reward.
        // z = 0.0f, outcome = "limit" (already set above)
    }

    // Build samples: flip z so it is always from the perspective of the mover.
    std::vector<Sample> samples;
    samples.reserve(traj.size());
    for (const auto& e : traj) {
        Sample s;
        s.state = e.state;
        s.pi    = e.pi;
        s.z     = (e.turn == chess::Color::WHITE) ? z : -z;
        samples.push_back(s);
    }

    GameMeta meta;
    meta.moves   = static_cast<int>(traj.size());
    meta.outcome = outcome;

    return {samples, meta};
}

// ── write_data ────────────────────────────────────────────────────────────────

void ParallelSelfPlay::write_data(const std::string& path,
                                   const std::vector<Sample>& samples) {
    SelfPlay::write_data(path, samples);
}
