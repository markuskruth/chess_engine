#pragma once
// ParallelSelfPlay.h — Phase 4: multi-game parallel self-play
//
// Runs num_workers games concurrently.  All leaf NN evaluations are routed
// through a single CentralEvaluator GPU thread for efficient batching.
//
// Binary output format is identical to SelfPlay::write_data() so the same
// Python training loop can load either source without modification.

#include "AsyncMCTS.h"
#include "CentralEvaluator.h"
#include "SelfPlay.h"     // reuses Sample, GameMeta, SelfPlayConfig
#include <mutex>
#include <vector>

// ── ParallelSelfPlayConfig ────────────────────────────────────────────────────

struct ParallelSelfPlayConfig {
    int   num_games        = 100;
    int   num_workers      = 4;     // concurrent game threads
    int   num_simulations  = 400;   // MCTS simulations per position
    int   leaf_batch_size  = 8;     // leaves per run_simulation_batch call
    int   max_moves        = 200;   // move limit per game
    float temperature      = 1.0f;
    std::string model_path;         // path to TorchScript model
    std::string output_path;        // path for binary game data (empty = no write)

    // Evaluator batch tunables (auto-computed if left at 0).
    // min_batch: evaluator fires once this many leaves are queued.
    //   MUST be <= leaf_batch_size to avoid deadlock: if the last active worker
    //   is blocked waiting for its futures, it cannot call notify_worker_done(),
    //   so the evaluator must fire on the items already in the queue.
    // max_batch: maximum leaves per GPU forward pass (caps GPU memory).
    //   With pipelining each worker has up to 2 batches in-flight, so the
    //   natural queue depth is num_workers * leaf_batch_size * 2.
    int min_batch_override = 0;  // 0 = leaf_batch_size
    int max_batch_override = 0;  // 0 = num_workers * leaf_batch_size * 2

    int min_batch() const {
        return min_batch_override > 0
               ? min_batch_override
               : leaf_batch_size;  // fire as soon as one worker's batch is ready
    }
    int max_batch() const {
        return max_batch_override > 0
               ? max_batch_override
               : num_workers * leaf_batch_size * 2;  // capture full pipeline depth
    }
};

// ── ParallelSelfPlay ──────────────────────────────────────────────────────────

class ParallelSelfPlay {
public:
    explicit ParallelSelfPlay(const ParallelSelfPlayConfig& cfg);

    // Run cfg.num_games games in parallel.
    // Blocks until all games complete and the evaluator thread exits.
    std::pair<std::vector<Sample>, std::vector<GameMeta>> run();

    // Serialize samples to binary file (identical format to SelfPlay::write_data).
    static void write_data(const std::string& path,
                           const std::vector<Sample>& samples);

private:
    ParallelSelfPlayConfig cfg_;

    // Accumulated across worker threads; protected by results_mutex_.
    std::vector<Sample>   all_samples_;
    std::vector<GameMeta> all_meta_;
    std::mutex            results_mutex_;

    // Worker thread entry point: plays num_games games, then signals queue.
    void worker_fn(int num_games, EvalQueue* queue);

    // Play a single game using async MCTS (called from worker threads).
    std::pair<std::vector<Sample>, GameMeta>
    play_game(AsyncMCTS& mcts, float temperature);
};
