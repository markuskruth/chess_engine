#pragma once
// MCTS.h — C++ MCTS tree search (Phase 3 implementation target)
//
// Mirrors the Python Node + MCTS class in Agent.py.
// Key design notes:
//   - Node owns its children via unique_ptr (automatic cleanup).
//   - N/W/Q/P are heap-allocated only after expansion to save memory on
//     the many leaf nodes that are never expanded.
//   - The central evaluator pattern (Phase 4) will use a shared leaf queue;
//     for now the interface assumes synchronous NN calls.

#include "ChessEnv.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <torch/script.h>  // torch::jit::Module (TorchScript model)

// ── Node ─────────────────────────────────────────────────────────────────────

struct Node {
    chess::Board board;

    // Statistics — heap-allocated on first expansion.
    // Size = ACTION_DIM = 4672 floats each.
    std::unique_ptr<std::array<float, ACTION_DIM>> P;  // prior probabilities
    std::unique_ptr<std::array<float, ACTION_DIM>> N;  // visit counts
    std::unique_ptr<std::array<float, ACTION_DIM>> W;  // total value
    std::unique_ptr<std::array<float, ACTION_DIM>> Q;  // mean value
    std::unique_ptr<ActionMask>                    legal_moves_cache;

    std::unordered_map<int, std::unique_ptr<Node>> children;
    bool is_expanded = false;
    int  total_N     = 0;  // cached sum of N[*], used in UCB numerator

    explicit Node(const chess::Board& b) : board(b) {}

    // Allocate stat arrays on first expand call.
    void allocate_stats();
};

// ── MCTS ─────────────────────────────────────────────────────────────────────

class MCTS {
public:
    // c_puct exploration constant — matches Python self.c = 1.0
    static constexpr float C_PUCT = 1.0f;

    // Construct with a loaded TorchScript model and device.
    explicit MCTS(torch::jit::Module model,
                  torch::Device      device = torch::kCPU);

    // Run `leaf_batch_size` simulations from root in a single batched NN call.
    // Mirrors Python's run_simulation_batch().
    void run_simulation_batch(Node& root, int leaf_batch_size = 8);

    // Choose the best action from root after simulations (greedy: argmax N).
    int best_action(const Node& root) const;

private:
    torch::jit::Module model_;
    torch::Device      device_;

    // Selection: walk the tree applying virtual loss; returns leaf + path.
    // path = vector of {node*, action_idx} from root to leaf's parent.
    using PathEntry = std::pair<Node*, int>;
    Node* select(Node& root, std::vector<PathEntry>& path);

    // Expansion: evaluate leaf with NN, fill P, legal_moves_cache.
    // Returns the leaf value (exact -1/0 for terminals, NN output otherwise).
    float expand(Node& leaf, float p_raw[], float v_raw);

    // Backpropagation: undo virtual loss and credit real value along path.
    // Value is already from the leaf's perspective; sign is flipped at each step.
    void backpropagate(const std::vector<PathEntry>& path, float value);
};
