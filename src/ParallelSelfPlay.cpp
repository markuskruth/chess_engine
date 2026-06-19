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

// ── Curriculum (Tier 1: strength-tracking difficulty ramp) ─────────────────────
//
// Goal: keep the VALUE HEAD fed with TRUE decisive outcomes (real checkmates,
// z = ±1) without ever putting the hand-made heuristic into the value target.
// We do this by starting games from positions the network can convert AT ITS
// CURRENT STRENGTH, ramping difficulty by *mate distance* as training proceeds:
//
//   T0  mate-in-1            — guaranteed decisive: MCTS sees the mating child
//                              immediately, so even an untrained net earns z = ±1.
//   T1  mate-in-2            — a short forced mate; needs a few plies of search.
//   T2  winning material     — KQ/KR/KRR vs lone king; decisive but requires real
//                              conversion technique (no <=2 forced mate), so it
//                              only yields decisive games once the net has some skill.
//   T3  standard opening     — the AlphaZero target distribution.
//
// Each game samples a tier from an episode-dependent distribution (a blend, not a
// hard switch), so the easy tiers fade gradually and the full opening grows in.
// A small permanent floor of forced mates remains forever as cheap insurance
// against value-head collapse late in training (set the last row to {0,0,0,1} for
// pure self-play once the net reliably produces its own decisive games).
//
// NOTE: every FEN below is a VERIFIED forced mate (minimax solver) / decisive
// position — see tools/gen_curriculum.py. The previous curriculum (a) was only 5
// episodes long, (b) leaned on K+2Q-vs-k positions a weak net just shuffles into a
// 50-move draw (z = 0, defeating the curriculum's purpose), and (c) contained a
// mislabeled "mate-in-1" that was not mate at all.

// Tier 0 — verified mate-in-1 (color-balanced).
static const std::vector<std::string> MATE_IN_1_FENS = {
    "8/8/3k4/8/4Q3/8/2Q5/4K3 w - - 0 1",
    "1q6/3K4/8/4q3/5k2/8/8/8 b - - 0 1",
    "8/8/3K4/8/8/6Q1/7Q/2k5 w - - 0 1",
    "1K6/8/q7/6q1/8/8/8/7k b - - 0 1",
    "4K3/7Q/4k3/6Q1/8/8/8/8 w - - 0 1",
    "K7/2q4q/8/8/8/k7/8/8 b - - 0 1",
    "2R5/8/8/8/7R/k7/8/1K6 w - - 0 1",
    "7K/8/8/8/8/8/5q1k/6q1 b - - 0 1",
};

// Tier 1 — verified mate-in-2 (color-balanced).
static const std::vector<std::string> MATE_IN_2_FENS = {
    "1R6/8/8/4Q3/1K6/8/8/1k6 w - - 0 1",
    "2q5/1k5K/8/8/8/8/8/4r3 b - - 0 1",
    "2k5/8/8/8/8/6Q1/4Q1K1/8 w - - 0 1",
    "8/8/8/8/4K3/8/4k3/2q1q3 b - - 0 1",
    "1k6/8/4R3/8/3Q4/5K2/8/8 w - - 0 1",
    "8/5K2/8/1q6/3k4/8/8/7q b - - 0 1",
    "7k/6R1/8/8/4R3/3K4/8/8 w - - 0 1",
    "8/8/8/5r2/8/1qk5/8/7K b - - 0 1",
};

// Tier 2 — verified decisive material, NOT a <=2 forced mate (needs conversion).
static const std::vector<std::string> WINNING_MATERIAL_FENS = {
    "4K3/8/8/8/1k6/8/8/R7 w - - 0 1",
    "8/8/6k1/8/K7/8/5r2/8 b - - 0 1",
    "1R6/8/8/6k1/8/8/7K/8 w - - 0 1",
    "8/6r1/8/4k3/1K6/8/8/r7 b - - 0 1",
    "8/8/8/5k2/3R3K/8/8/R7 w - - 0 1",
    "6k1/8/2q5/8/1K6/8/8/8 b - - 0 1",
};

// Episode-indexed tier-sampling weights: {mate-in-1, mate-in-2, material, opening}.
// The first row whose max_episode >= episode applies.
struct CurriculumStage {
    int                   max_episode;
    std::array<double, 4> weights;
};
static const std::vector<CurriculumStage> CURRICULUM_SCHEDULE = {
    {  10, {0.55, 0.30, 0.15, 0.00}},   // bootstrap: almost all guaranteed-decisive
    {  25, {0.30, 0.30, 0.25, 0.15}},
    {  50, {0.15, 0.20, 0.25, 0.40}},
    { 100, {0.07, 0.10, 0.13, 0.70}},
    {1 << 30, {0.01, 0.02, 0.02, 0.95}},// steady state: ~5% decisive-signal floor
};

// Pick a starting board for the given episode using the curriculum schedule.
static chess::Board select_curriculum_board(int episode, std::mt19937& rng) {
    const std::array<double, 4>* w = &CURRICULUM_SCHEDULE.back().weights;
    for (const auto& stage : CURRICULUM_SCHEDULE) {
        if (episode <= stage.max_episode) { w = &stage.weights; break; }
    }

    std::discrete_distribution<int> tier_dist(w->begin(), w->end());
    const int tier = tier_dist(rng);

    const std::vector<std::string>* pool = nullptr;
    switch (tier) {
        case 0: pool = &MATE_IN_1_FENS;        break;
        case 1: pool = &MATE_IN_2_FENS;        break;
        case 2: pool = &WINNING_MATERIAL_FENS; break;
        default: return chess::Board();        // tier 3 — standard start position
    }
    std::uniform_int_distribution<int> fen_dist(0, static_cast<int>(pool->size()) - 1);
    return chess::Board((*pool)[fen_dist(rng)]);
}

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
                        out_file_.write(reinterpret_cast<const char*>(&s.z),           sizeof(float));
                        out_file_.write(reinterpret_cast<const char*>(&s.eval_target), sizeof(float));
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

    // Select starting position via the curriculum schedule (Tier 1 difficulty ramp):
    // forced-mate / decisive positions early, blending toward the standard opening.
    chess::Board board = select_curriculum_board(cfg_.episode, rng);

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

        // Encode and store state + policy for this ply (no torch allocation).
        // The moves-left target is derived from the trajectory length after the game ends.
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

    // Build samples. z is flipped to be mover-relative. The auxiliary slot now holds
    // moves-left = plies remaining to game end (traj.size() - i): a perspective-
    // independent count, so it is NOT flipped per side to move.
    std::vector<Sample> samples;
    samples.reserve(traj.size());
    const float total_plies = static_cast<float>(traj.size());
    for (size_t i = 0; i < traj.size(); ++i) {
        const auto& e = traj[i];
        Sample s;
        s.state       = e.state;
        s.pi          = e.pi;
        s.z           = (e.turn == chess::Color::WHITE) ? z : -z;
        s.eval_target = total_plies - static_cast<float>(i);   // moves-left (raw plies)
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
