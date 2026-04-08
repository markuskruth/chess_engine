#pragma once
// SelfPlay.h — self-play loop and data serialization (Phase 4/5 target)
//
// Generates game data using MCTS and writes it to a flat binary file
// that Python's training loop can load.

#include "MCTS.h"
#include <string>
#include <vector>

// ── Binary data format ────────────────────────────────────────────────────────
// Written by SelfPlay::write_data(), read by Python's load_binary_game_data().
//
// File layout:
//   Header:   uint32 num_samples
//             uint32 state_channels   (always 20)
//             uint32 board_size       (always 8)
//   Per sample (repeated num_samples times):
//             float32[20*8*8]  encoded state     (1280 floats)
//             float32[4672]    policy target pi  (4672 floats)
//             float32[1]       value target z

struct Sample {
    std::array<float, STATE_CHANNELS * BOARD_SZ * BOARD_SZ> state;  // 1280
    std::array<float, ACTION_DIM>                            pi;     // 4672
    float                                                    z;
};

struct GameMeta {
    int    moves;
    std::string outcome;  // "white", "black", "draw", "limit"
};

// ── Configuration ─────────────────────────────────────────────────────────────

struct SelfPlayConfig {
    int num_games        = 10;
    int num_simulations  = 400;   // total sims per position (must be divisible by leaf_batch)
    int leaf_batch_size  = 8;
    int max_moves        = 400;   // plies; 400 = 200 full moves
    float temperature    = 1.0f;
    std::string model_path;       // path to TorchScript model file
    std::string output_path;      // path for binary game data output
};

// ── SelfPlay ──────────────────────────────────────────────────────────────────

class SelfPlay {
public:
    explicit SelfPlay(const SelfPlayConfig& cfg);

    // Run cfg.num_games self-play games.
    // Returns collected samples and per-game metadata.
    std::pair<std::vector<Sample>, std::vector<GameMeta>> run();

    // Serialize samples to binary file (cfg.output_path).
    static void write_data(const std::string& path,
                           const std::vector<Sample>& samples);

private:
    SelfPlayConfig cfg_;
    MCTS           mcts_;

    // Play a single game from the starting position.
    std::pair<std::vector<Sample>, GameMeta>
    play_game(float temperature = 1.0f);
};
