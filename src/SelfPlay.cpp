// SelfPlay.cpp — Phase 4/5 implementation
//
// SelfPlay::play_game() mirrors Python's MCTS.play_game() in Agent.py.
// SelfPlay::run()        runs cfg_.num_games games sequentially.
// SelfPlay::write_data() serialises samples to the flat binary format.

#include "SelfPlay.h"
#include <fstream>
#include <random>
#include <stdexcept>
#include <numeric>

// ── SelfPlay constructor ──────────────────────────────────────────────────────

SelfPlay::SelfPlay(const SelfPlayConfig& cfg)
    : cfg_(cfg),
      mcts_(
          torch::jit::load(cfg.model_path),
          torch::cuda::is_available()
              ? torch::Device(torch::kCUDA)
              : torch::Device(torch::kCPU)
      )
{}

// ── run ───────────────────────────────────────────────────────────────────────

std::pair<std::vector<Sample>, std::vector<GameMeta>> SelfPlay::run() {
    std::vector<Sample>   all_samples;
    std::vector<GameMeta> all_meta;
    for (int i = 0; i < cfg_.num_games; ++i) {
        auto [samples, meta] = play_game(cfg_.temperature);
        all_samples.insert(all_samples.end(), samples.begin(), samples.end());
        all_meta.push_back(meta);
    }
    return {all_samples, all_meta};
}

// ── play_game ─────────────────────────────────────────────────────────────────
//
// Plays one complete game from the starting position using MCTS.
// Returns a list of (state, policy, value) samples and game metadata.
//
// Value assignment (mirrors Agent.py play_game):
//   - Terminal by checkmate: sideToMove is mated, so z = +1 if Black is mated
//     (White wins), -1 if White is mated (Black wins), from White's perspective.
//   - Terminal by draw rule:  z = 0.
//   - Move limit hit:         z = ±1 if get_evaluation() > 0.3 (mirrors Python's
//     threshold heuristic), z = 0 otherwise ("limit").
//
// Per-ply value flip: each sample stores z from the perspective of the player
// who moved at that ply (matching the flipped board encoding).

std::pair<std::vector<Sample>, GameMeta>
SelfPlay::play_game(float temperature) {
    chess::Board board;  // starting position

    struct TrajEntry {
        std::array<float, STATE_CHANNELS * BOARD_SZ * BOARD_SZ> state;
        std::array<float, ACTION_DIM>                            pi;
        chess::Color                                             turn;
    };
    std::vector<TrajEntry> traj;
    traj.reserve(cfg_.max_moves);

    // Thread-local RNG so parallel workers don't share state.
    static thread_local std::mt19937 rng(std::random_device{}());

    const int num_batches = cfg_.num_simulations / cfg_.leaf_batch_size;
    if (num_batches <= 0)
        throw std::runtime_error("SelfPlay: num_simulations must be >= leaf_batch_size");

    for (int step = 0; step < cfg_.max_moves; ++step) {
        if (ChessEnv::game_over(board).first) break;

        const chess::Color turn = board.sideToMove();

        // Fresh MCTS root for this position
        Node root(board);

        for (int b = 0; b < num_batches; ++b) {
            mcts_.run_simulation_batch(root, cfg_.leaf_batch_size);
        }

        // Build policy π from raw visit counts
        std::array<float, ACTION_DIM> pi{};
        float sum_n = 0.0f;
        if (root.N) {
            for (int a = 0; a < ACTION_DIM; ++a) sum_n += (*root.N)[a];
            if (sum_n > 0.0f) {
                const float inv = 1.0f / sum_n;
                for (int a = 0; a < ACTION_DIM; ++a) pi[a] = (*root.N)[a] * inv;
            }
        }

        // Sample action from π (stochastic) or take argmax (deterministic)
        int action_idx = 0;
        if (sum_n > 0.0f) {
            if (temperature < 1e-3f) {
                action_idx = mcts_.best_action(root);
            } else {
                std::discrete_distribution<int> dist(pi.begin(), pi.end());
                action_idx = dist(rng);
            }
        }

        // Store encoded state + policy for this ply
        {
            auto state_t = ChessEnv::encode_state(board)
                               .to(torch::kCPU).contiguous();
            TrajEntry entry;
            std::copy(state_t.data_ptr<float>(),
                      state_t.data_ptr<float>()
                          + STATE_CHANNELS * BOARD_SZ * BOARD_SZ,
                      entry.state.begin());
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
        if (go.second) {  // checkmate: the player to move is mated
            z       = (board.sideToMove() == chess::Color::BLACK) ? 1.0f : -1.0f;
            outcome = (board.sideToMove() == chess::Color::BLACK) ? "white" : "black";
        } else {
            z       = 0.0f;
            outcome = "draw";
        }
    } else {
        // Move limit reached — use heuristic evaluation as tiebreaker
        // (mirrors Python's Agent.py play_game tiebreaker with threshold 0.3)
        float pot = ChessEnv::get_evaluation(board);
        if      (pot >  0.3f) { z =  1.0f; outcome = "white"; }
        else if (pot < -0.3f) { z = -1.0f; outcome = "black"; }
        // else z = 0.0f, outcome = "limit" (already set)
    }

    // Build samples: flip z for Black's plies so it is always current-player-relative
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
//
// Binary format (matches Python's load_binary_game_data()):
//   Header:   uint32 num_samples | uint32 state_channels(20) | uint32 board_size(8)
//   Per sample: float32[1280] state | float32[4672] pi | float32[1] z

void SelfPlay::write_data(const std::string& path,
                           const std::vector<Sample>& samples) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open output file: " + path);

    const uint32_t n   = static_cast<uint32_t>(samples.size());
    const uint32_t ch  = STATE_CHANNELS;
    const uint32_t bsz = BOARD_SZ;
    f.write(reinterpret_cast<const char*>(&n),   sizeof(n));
    f.write(reinterpret_cast<const char*>(&ch),  sizeof(ch));
    f.write(reinterpret_cast<const char*>(&bsz), sizeof(bsz));

    for (const auto& s : samples) {
        f.write(reinterpret_cast<const char*>(s.state.data()),
                s.state.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(s.pi.data()),
                s.pi.size() * sizeof(float));
        f.write(reinterpret_cast<const char*>(&s.z), sizeof(float));
    }
}
