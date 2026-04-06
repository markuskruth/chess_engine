// main.cpp — chess_rl self-play runner (Phase 5)
//
// Generates self-play game data and writes it to a binary file that can be
// loaded by Python's load_binary_game_data() in data_loader.py.
//
// Usage:
//   selfplay.exe <model_path> [options]
//
// Options:
//   --games   N    Games to play             (default: 10)
//   --workers W    Parallel worker threads   (default: 1, sequential SelfPlay)
//   --sims    S    MCTS simulations/move     (default: 400)
//   --batch   B    Leaf batch size           (default: 8)
//   --moves   M    Max moves per game        (default: 200)
//   --temp    T    Temperature               (default: 1.0)
//   --output  P    Output binary file        (default: game_data.bin)
//
// With --workers 1 (default) the single-threaded SelfPlay class is used.
// With --workers > 1 the multi-threaded ParallelSelfPlay class is used.
//
// Binary format written (readable by data_loader.py):
//   Header:   uint32 num_samples | uint32 state_channels (20) | uint32 board_size (8)
//   Per sample: float32[1280] state | float32[4672] pi | float32[1] z

#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include "SelfPlay.h"
#include "ParallelSelfPlay.h"

// ── Argument parsing ──────────────────────────────────────────────────────────

struct RunConfig {
    std::string model_path;
    std::string output_path = "game_data.bin";
    int   num_games        = 10;
    int   num_workers      = 1;
    int   num_simulations  = 400;
    int   leaf_batch_size  = 8;
    int   max_moves        = 200;
    float temperature      = 1.0f;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <model_path> [options]\n"
        << "   or: " << prog << " --model <model_path> [options]\n"
        << "\nOptions:\n"
        << "  --model   P    TorchScript model file    (or positional first arg)\n"
        << "  --games   N    Games to play             (default: 10)\n"
        << "  --workers W    Parallel worker threads   (default: 1)\n"
        << "  --threads W    Synonym for --workers\n"
        << "  --sims    S    MCTS simulations/move     (default: 400)\n"
        << "  --batch   B    Leaf batch size           (default: 8)\n"
        << "  --moves   M    Max moves per game        (default: 200)\n"
        << "  --temp    T    Temperature               (default: 1.0)\n"
        << "  --output  P    Output binary file        (default: game_data.bin)\n";
}

static RunConfig parse_args(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        std::exit(1);
    }

    RunConfig cfg;

    // argv[1] is the model path if it doesn't start with '--'
    int start = 2;
    if (std::string(argv[1]).rfind("--", 0) != 0) {
        cfg.model_path = argv[1];
    } else {
        start = 1;  // all args are named; --model must appear somewhere
    }

    for (int i = start; i < argc; ++i) {
        std::string key = argv[i];
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << key << "\n";
            std::exit(1);
        }
        const char* val = argv[++i];
        if      (key == "--model")                 cfg.model_path      = val;
        else if (key == "--games")                 cfg.num_games       = std::atoi(val);
        else if (key == "--workers" ||
                 key == "--threads")               cfg.num_workers     = std::atoi(val);
        else if (key == "--sims")                  cfg.num_simulations = std::atoi(val);
        else if (key == "--batch")                 cfg.leaf_batch_size = std::atoi(val);
        else if (key == "--moves")                 cfg.max_moves       = std::atoi(val);
        else if (key == "--temp")                  cfg.temperature     = static_cast<float>(std::atof(val));
        else if (key == "--output")                cfg.output_path     = val;
        else {
            std::cerr << "Unknown option: " << key << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (cfg.model_path.empty()) {
        std::cerr << "Model path is required (first positional arg or --model).\n";
        print_usage(argv[0]);
        std::exit(1);
    }

    if (cfg.num_games <= 0 || cfg.num_workers <= 0 || cfg.num_simulations <= 0 ||
        cfg.leaf_batch_size <= 0 || cfg.max_moves <= 0) {
        std::cerr << "All numeric options must be positive.\n";
        std::exit(1);
    }
    if (cfg.num_simulations % cfg.leaf_batch_size != 0) {
        std::cerr << "--sims must be divisible by --batch.\n";
        std::exit(1);
    }

    return cfg;
}

// ── Summary printing ──────────────────────────────────────────────────────────

static void print_summary(const std::vector<GameMeta>& meta) {
    int whites = 0, blacks = 0, draws = 0, limits = 0, total_moves = 0;
    for (const auto& m : meta) {
        if      (m.outcome == "white") ++whites;
        else if (m.outcome == "black") ++blacks;
        else if (m.outcome == "draw")  ++draws;
        else                           ++limits;
        total_moves += m.moves;
    }
    const int n = static_cast<int>(meta.size());
    const float avg = n > 0 ? static_cast<float>(total_moves) / n : 0.0f;

    std::cout << "\n── Game summary ────────────────────────────────────\n";
    std::cout << "  Games      : " << n << "\n";
    std::cout << "  Avg moves  : " << std::fixed << std::setprecision(1) << avg << "\n";
    std::cout << "  White wins : " << whites
              << "  (" << std::setprecision(1) << (100.0f * whites / std::max(n, 1)) << "%)\n";
    std::cout << "  Black wins : " << blacks
              << "  (" << (100.0f * blacks / std::max(n, 1)) << "%)\n";
    std::cout << "  Draws      : " << draws
              << "  (" << (100.0f * draws / std::max(n, 1)) << "%)\n";
    if (limits > 0)
        std::cout << "  Move limit : " << limits
                  << "  (" << (100.0f * limits / std::max(n, 1)) << "%)\n";
    std::cout << "────────────────────────────────────────────────────\n";
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    const RunConfig cfg = parse_args(argc, argv);

    std::cout << "=== chess_rl self-play runner ===\n";
    std::cout << "  model    : " << cfg.model_path    << "\n";
    std::cout << "  output   : " << cfg.output_path   << "\n";
    std::cout << "  games    : " << cfg.num_games     << "\n";
    std::cout << "  workers  : " << cfg.num_workers   << "\n";
    std::cout << "  sims     : " << cfg.num_simulations << "\n";
    std::cout << "  batch    : " << cfg.leaf_batch_size  << "\n";
    std::cout << "  moves    : " << cfg.max_moves     << "\n";
    std::cout << "  temp     : " << cfg.temperature   << "\n\n";

    std::vector<Sample>   samples;
    std::vector<GameMeta> meta;

    try {
        if (cfg.num_workers <= 1) {
            // ── Sequential self-play ──────────────────────────────────────────
            std::cout << "[SelfPlay] running " << cfg.num_games << " game(s)...\n";
            SelfPlayConfig sp_cfg;
            sp_cfg.model_path      = cfg.model_path;
            sp_cfg.output_path     = cfg.output_path;
            sp_cfg.num_games       = cfg.num_games;
            sp_cfg.num_simulations = cfg.num_simulations;
            sp_cfg.leaf_batch_size = cfg.leaf_batch_size;
            sp_cfg.max_moves       = cfg.max_moves;
            sp_cfg.temperature     = cfg.temperature;

            SelfPlay sp(sp_cfg);
            auto [s, m] = sp.run();
            samples = std::move(s);
            meta    = std::move(m);
        } else {
            // ── Parallel self-play ────────────────────────────────────────────
            std::cout << "[ParallelSelfPlay] running " << cfg.num_games
                      << " game(s) across " << cfg.num_workers << " worker(s)...\n";
            ParallelSelfPlayConfig pp_cfg;
            pp_cfg.model_path      = cfg.model_path;
            pp_cfg.output_path     = cfg.output_path;
            pp_cfg.num_games       = cfg.num_games;
            pp_cfg.num_workers     = cfg.num_workers;
            pp_cfg.num_simulations = cfg.num_simulations;
            pp_cfg.leaf_batch_size = cfg.leaf_batch_size;
            pp_cfg.max_moves       = cfg.max_moves;
            pp_cfg.temperature     = cfg.temperature;

            ParallelSelfPlay psp(pp_cfg);
            auto [s, m] = psp.run();
            samples = std::move(s);
            meta    = std::move(m);
        }
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << "\n";
        return 1;
    }

    // ── Write binary data ─────────────────────────────────────────────────────
    std::cout << "\n[write_data] " << samples.size() << " sample(s) → "
              << cfg.output_path << "\n";
    try {
        SelfPlay::write_data(cfg.output_path, samples);
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] write_data: " << ex.what() << "\n";
        return 1;
    }

    print_summary(meta);
    std::cout << "Done.\n";
    return 0;
}
