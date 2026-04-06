#pragma once
// AsyncMCTS.h — Phase 4: MCTS tree search with async leaf evaluation
//
// Mirrors MCTS (Phase 3) but delegates NN evaluation to a CentralEvaluator
// via EvalQueue instead of calling the model directly.
// One AsyncMCTS instance per worker thread — NOT thread-safe.
//
// The Node struct (from MCTS.h) is reused unchanged.

#include "MCTS.h"       // for Node, ACTION_DIM, C_PUCT, PathEntry shape
#include "Evaluator.h"  // for EvalQueue, EvalResult, EvalRequest
#include <future>

class AsyncMCTS {
public:
    // C_PUCT exploration constant — same as MCTS.
    static constexpr float C_PUCT = 1.0f;

    using PathEntry = std::pair<Node*, int>;

    // queue: shared with CentralEvaluator (non-owning pointer, must outlive this).
    explicit AsyncMCTS(EvalQueue* queue);

    // Pipelined simulation run: overlaps GPU evaluation of batch N with CPU
    // selection of batch N+1.  Prefer this over calling run_simulation_batch
    // in a loop — it hides GPU latency and keeps CPU + GPU busy simultaneously.
    // Requires: num_simulations % leaf_batch_size == 0.
    void run_simulations(Node& root, int num_simulations, int leaf_batch_size);

    // Single-batch helper (kept for test compatibility and the sequential path).
    // Selects leaf_batch_size leaves, submits to the queue, blocks until all
    // futures are resolved, then expands and backpropagates.
    void run_simulation_batch(Node& root, int leaf_batch_size = 8);

    // Greedy best action by visit count (identical to MCTS::best_action).
    int best_action(const Node& root) const;

private:
    // A submitted batch whose GPU results have not yet been awaited.
    struct PendingBatch {
        std::vector<Node*>                   leaves;
        std::vector<std::vector<PathEntry>>  paths;
        std::vector<std::future<EvalResult>> futures;
    };

    EvalQueue* queue_;  // non-owning

    // These methods are logically identical to the private methods of MCTS.
    // They are duplicated here to keep MCTS.h independent of Evaluator.h.
    Node* select(Node& root, std::vector<PathEntry>& path);
    float expand(Node& leaf, const EvalResult& result);
    void  backpropagate(const std::vector<PathEntry>& path, float value);

    // Select leaf_batch_size leaves, encode, submit to queue — returns immediately.
    PendingBatch select_and_submit(Node& root, int leaf_batch_size);

    // Wait for all futures in a PendingBatch, then expand + backprop each leaf.
    void flush_pending(PendingBatch& pending);
};
