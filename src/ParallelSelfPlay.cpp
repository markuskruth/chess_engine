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

ParallelSelfPlay::ParallelSelfPlay(const ParallelSelfPlayConfig& cfg)
    : cfg_(cfg)
{}

// ── run ───────────────────────────────────────────────────────────────────────

std::pair<std::vector<Sample>, std::vector<GameMeta>>
ParallelSelfPlay::run() {
    // --- ADD THIS LINE TO PREVENT THE OPENMP DEADLOCK ---
    torch::set_num_threads(1);
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

    return {all_samples_, all_meta_};
}

// ── worker_fn ─────────────────────────────────────────────────────────────────

void ParallelSelfPlay::worker_fn(int num_games, EvalQueue* queue) {
    auto tid = std::this_thread::get_id();
    std::cerr << "[DEBUG worker " << tid << "] started, num_games=" << num_games << "\n" << std::flush;
    AsyncMCTS mcts(queue);
    try {
        for (int g = 0; g < num_games; ++g) {
            auto [samples, meta] = play_game(mcts, cfg_.temperature);
            std::cerr << "[DEBUG worker " << tid << "] game " << (g+1) << "/" << num_games
                      << "  moves=" << meta.moves << "  outcome=" << meta.outcome << "\n" << std::flush;
            std::lock_guard<std::mutex> lk(results_mutex_);
            all_samples_.insert(all_samples_.end(),
                                samples.begin(), samples.end());
            all_meta_.push_back(std::move(meta));
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
ParallelSelfPlay::play_game(AsyncMCTS& mcts, float temperature) {
    chess::Board board;

    struct TrajEntry {
        std::array<float, STATE_CHANNELS * BOARD_SZ * BOARD_SZ> state;
        std::array<float, ACTION_DIM>                            pi;
        chess::Color                                             turn;
    };
    std::vector<TrajEntry> traj;
    traj.reserve(cfg_.max_moves);

    static thread_local std::mt19937 rng(std::random_device{}());

    if (cfg_.num_simulations < cfg_.leaf_batch_size)
        throw std::runtime_error("ParallelSelfPlay: num_simulations must be >= leaf_batch_size");

    for (int step = 0; step < cfg_.max_moves; ++step) {
        if (ChessEnv::game_over(board).first) break;

        const chess::Color turn = board.sideToMove();
        Node root(board);

        // Pipelined: overlaps GPU evaluation with CPU selection for next batch.
        mcts.run_simulations(root, cfg_.num_simulations, cfg_.leaf_batch_size);

        // Build policy π from visit counts.
        std::array<float, ACTION_DIM> pi{};
        float sum_n = 0.0f;
        if (root.N) {
            for (int a = 0; a < ACTION_DIM; ++a) sum_n += (*root.N)[a];
            if (sum_n > 0.0f) {
                const float inv = 1.0f / sum_n;
                for (int a = 0; a < ACTION_DIM; ++a) pi[a] = (*root.N)[a] * inv;
            }
        }

        // Sample action (stochastic at temperature > 0, greedy otherwise).
        int action_idx = 0;
        if (sum_n > 0.0f) {
            if (temperature < 1e-3f) {
                action_idx = mcts.best_action(root);
            } else {
                std::discrete_distribution<int> dist(pi.begin(), pi.end());
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
        // Move limit reached — heuristic tiebreaker (mirrors Python's threshold 0.3)
        float pot = ChessEnv::get_evaluation(board);
        if      (pot >  0.3f) { z =  1.0f; outcome = "white"; }
        else if (pot < -0.3f) { z = -1.0f; outcome = "black"; }
        // else z = 0.0f, outcome = "limit"
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
