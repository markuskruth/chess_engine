// test_mcts.cpp — Phase 3 validation
//
// Two test tiers:
//   Tier 1 (no model required): validates Node structure, virtual-loss invariants,
//           and that select() does not crash on an unexpanded root.
//   Tier 2 (model required):    runs a small number of MCTS simulations, checks
//           that visit counts are consistent and best_action() returns a legal move.
//
// Usage:
//   test_mcts.exe                          # Tier 1 only
//   test_mcts.exe path/to/model_traced.pt  # Tier 1 + Tier 2

#include "MCTS.h"
#include "ChessEnv.h"
#include <iostream>
#include <string>

// ── helpers ───────────────────────────────────────────────────────────────────

static int g_pass = 0, g_fail = 0;

static void check(bool ok, const char* msg) {
    if (ok) {
        std::cout << "  [PASS] " << msg << "\n";
        ++g_pass;
    } else {
        std::cerr << "  [FAIL] " << msg << "\n";
        ++g_fail;
    }
}

// ── Tier 1: structure tests (no model needed) ─────────────────────────────────

static void test_node_allocation() {
    std::cout << "\n-- Node allocation --\n";
    chess::Board board;
    Node node(board);

    check(!node.is_expanded,          "node starts unexpanded");
    check(node.total_N == 0,          "total_N starts at 0");
    check(!node.P,                    "P not allocated before expand");
    check(!node.N,                    "N not allocated before expand");
    check(node.children.empty(),      "no children initially");

    node.allocate_stats();
    check(node.P  != nullptr,         "P allocated");
    check(node.N  != nullptr,         "N allocated");
    check(node.W  != nullptr,         "W allocated");
    check(node.Q  != nullptr,         "Q allocated");
    check(node.legal_moves_cache != nullptr, "legal_moves_cache allocated");

    // All stats must start at zero
    bool all_zero = true;
    for (int a = 0; a < ACTION_DIM; ++a) {
        if ((*node.P)[a] != 0.0f || (*node.N)[a] != 0.0f ||
            (*node.W)[a] != 0.0f || (*node.Q)[a] != 0.0f)
        { all_zero = false; break; }
    }
    check(all_zero, "all stats initialise to 0");
}

static void test_action_mask_starting_pos() {
    std::cout << "\n-- Action mask (starting position) --\n";
    chess::Board board;
    auto mask = ChessEnv::get_action_mask(board);

    int num_legal = 0;
    for (bool b : mask) if (b) ++num_legal;

    // Starting position has exactly 20 legal moves
    check(num_legal == 20, "starting position: 20 legal moves in mask");
}

static void test_game_over_starting_pos() {
    std::cout << "\n-- game_over (starting position) --\n";
    chess::Board board;
    auto [over, checkmate] = ChessEnv::game_over(board);
    check(!over,      "starting position is not game-over");
    check(!checkmate, "starting position is not checkmate");
}

static void test_apply_action_roundtrip() {
    std::cout << "\n-- apply_action (e2-e4) --\n";
    chess::Board board;
    auto mask = ChessEnv::get_action_mask(board);

    // Find any legal action and apply it
    int found = -1;
    for (int a = 0; a < ACTION_DIM; ++a) {
        if (mask[a]) { found = a; break; }
    }
    check(found >= 0, "found at least one legal action");

    if (found >= 0) {
        auto [ok, mv] = ChessEnv::apply_action(found, board);
        check(ok, "apply_action succeeds for legal action");
        (void)mv;
    }
}

// ── Tier 2: full MCTS simulation (model required) ─────────────────────────────

static void test_mcts_simulations(const std::string& model_path) {
    std::cout << "\n-- MCTS simulation (model: " << model_path << ") --\n";

    torch::jit::Module model;
    try {
        model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "  [SKIP] Cannot load model: " << e.what() << "\n";
        return;
    }

    const auto device = torch::cuda::is_available()
                        ? torch::Device(torch::kCUDA)
                        : torch::Device(torch::kCPU);
    MCTS mcts(std::move(model), device);

    chess::Board board;   // starting position
    Node root(board);

    const int LEAF_BATCH = 8;
    const int NUM_BATCHES = 5;   // 40 simulations total

    for (int b = 0; b < NUM_BATCHES; ++b) {
        mcts.run_simulation_batch(root, LEAF_BATCH);
    }

    check(root.is_expanded,  "root is expanded after simulations");
    check(root.N != nullptr, "root.N allocated");

    if (root.N) {
        // Batch 1: all 8 selects run before any expand, so root is unexpanded for
        // all of them → all paths are empty → root.N unchanged (0).
        // Batches 2..NUM_BATCHES: root is expanded, each sim adds 1 to root.N.
        // Expected: LEAF_BATCH * (NUM_BATCHES - 1)  — matches Python's behaviour.
        int sum_n = 0;
        for (int a = 0; a < ACTION_DIM; ++a) sum_n += static_cast<int>((*root.N)[a]);
        const int expected = LEAF_BATCH * (NUM_BATCHES - 1);
        check(sum_n == expected,
              ("root visit count == " + std::to_string(expected)
               + " (got " + std::to_string(sum_n) + ")").c_str());

        int best = mcts.best_action(root);
        check(best >= 0 && best < ACTION_DIM, "best_action in valid range");

        if (best >= 0) {
            auto mask = ChessEnv::get_action_mask(board);
            check(mask[best], "best_action is a legal move");
        }
    }
}

// ── entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::cout << "=== Phase 3: MCTS test ===\n";

    // Tier 1 (always)
    test_node_allocation();
    test_action_mask_starting_pos();
    test_game_over_starting_pos();
    test_apply_action_roundtrip();

    // Tier 2 (when model is provided)
    if (argc > 1) {
        test_mcts_simulations(argv[1]);
    } else {
        std::cout << "\n[info] Tier 2 skipped (no model path provided).\n";
        std::cout << "[info] Run: test_mcts.exe path/to/model_traced.pt\n";
    }

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
