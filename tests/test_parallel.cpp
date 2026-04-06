// test_parallel.cpp — Phase 4 validation
//
// Tier 1 (no model): EvalQueue unit tests — push/pop, shutdown, partial flush,
//                    promise lifecycle.
// Tier 2 (model required): AsyncMCTS driven by a stub evaluator thread.
// Tier 3 (model required): full ParallelSelfPlay integration — N workers,
//                          real CentralEvaluator, data integrity checks.
//
// Usage:
//   test_parallel.exe                          # Tier 1 only
//   test_parallel.exe path/to/model_traced.pt  # Tier 1 + Tier 2 + Tier 3

#include "Evaluator.h"
#include "AsyncMCTS.h"
#include "CentralEvaluator.h"
#include "ParallelSelfPlay.h"
#include <iostream>
#include <string>
#include <thread>

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

// ── Tier 1: EvalQueue unit tests (no model) ───────────────────────────────────

// Helper: build an EvalRequest that doesn't need a live future.
static EvalRequest make_dummy_request() {
    EvalRequest req;
    req.state.fill(0.0f);
    // Obtain and discard the future — dropping a future without get() is safe.
    (void)req.promise.get_future();
    return req;
}

static void test_evalqueue_push_pop() {
    std::cout << "\n-- EvalQueue: push / pop --\n";
    EvalQueue q(1);

    q.push(make_dummy_request());
    q.push(make_dummy_request());
    q.push(make_dummy_request());

    check(q.pending_count() == 3, "3 items pending after 3 pushes");

    // min_batch=2 → should return immediately (3 >= 2)
    bool ok = q.wait_for_batch(2);
    check(ok, "wait_for_batch returns true when queue >= min_batch");

    auto batch = q.pop_batch(2);
    check(static_cast<int>(batch.size()) == 2, "pop_batch(2) returns 2 items");
    check(q.pending_count() == 1, "1 item remaining after pop of 2");

    // Pop the last item
    ok = q.wait_for_batch(1);
    check(ok, "wait_for_batch returns true for remaining item");
    batch = q.pop_batch(10);
    check(static_cast<int>(batch.size()) == 1, "second pop returns 1 remaining item");
    check(q.pending_count() == 0, "queue empty after draining");
}

static void test_evalqueue_shutdown_empty() {
    std::cout << "\n-- EvalQueue: shutdown with empty queue --\n";
    EvalQueue q(1);

    q.notify_worker_done();  // active_workers → 0
    q.shutdown();

    bool ok = q.wait_for_batch(1);
    check(!ok, "wait_for_batch returns false when shutdown + empty");
}

static void test_evalqueue_partial_flush() {
    std::cout << "\n-- EvalQueue: partial-batch flush when workers done --\n";
    EvalQueue q(1);

    // Push fewer items than min_batch
    q.push(make_dummy_request());
    q.push(make_dummy_request());

    // Signal worker done — active_workers drops to 0, which unblocks wait
    // even though 2 < min_batch_size=100.
    q.notify_worker_done();

    bool ok = q.wait_for_batch(100);  // min_batch larger than queue size
    check(ok, "partial flush: wait_for_batch returns true when workers done");

    auto batch = q.pop_batch(10);
    check(static_cast<int>(batch.size()) == 2, "partial flush: all 2 items returned");

    // Now empty + shutdown
    q.shutdown();
    ok = q.wait_for_batch(1);
    check(!ok, "partial flush: wait_for_batch returns false after drain + shutdown");
}

static void test_evalqueue_promise_lifecycle() {
    std::cout << "\n-- EvalQueue: promise / future lifecycle --\n";
    EvalQueue q(1);

    EvalRequest req;
    req.state.fill(1.0f);
    auto future = req.promise.get_future();
    q.push(std::move(req));

    check(q.pending_count() == 1, "1 item queued");

    q.notify_worker_done();
    bool ok = q.wait_for_batch(1);
    check(ok, "wait returns true for queued item");

    auto batch = q.pop_batch(10);
    check(static_cast<int>(batch.size()) == 1, "popped 1 item");

    // Manually fulfil the promise from "outside" (simulates evaluator)
    EvalResult res{};
    res.value = 0.42f;
    res.policy.fill(0.0f);
    batch[0].promise.set_value(std::move(res));

    // Future should now be ready
    auto result = future.get();
    check(std::abs(result.value - 0.42f) < 1e-6f, "future returns correct value");
}

// ── Tier 2: AsyncMCTS + stub evaluator thread ─────────────────────────────────

// A minimal evaluator that drains EvalQueue and returns constant results.
// Runs in a background thread; stopped via notify_worker_done + shutdown.
static void stub_evaluator(EvalQueue* queue, int min_batch) {
    while (queue->wait_for_batch(min_batch)) {
        auto batch = queue->pop_batch(64);
        for (auto& req : batch) {
            EvalResult res{};
            res.policy.fill(0.0f);
            res.value = 0.0f;
            req.promise.set_value(std::move(res));
        }
    }
}

static void test_async_mcts_simulations(const std::string& model_path) {
    std::cout << "\n-- AsyncMCTS single-threaded with stub evaluator --\n";
    (void)model_path;  // model not needed for this tier

    constexpr int LEAF_BATCH  = 8;
    constexpr int NUM_BATCHES = 5;

    EvalQueue queue(1);
    std::thread stub([&] { stub_evaluator(&queue, 1); });

    chess::Board board;
    Node root(board);

    AsyncMCTS mcts(&queue);
    for (int b = 0; b < NUM_BATCHES; ++b) {
        mcts.run_simulation_batch(root, LEAF_BATCH);
    }

    check(root.is_expanded, "root is expanded after simulations");
    check(root.N != nullptr, "root.N allocated");

    if (root.N) {
        int sum_n = 0;
        for (int a = 0; a < ACTION_DIM; ++a) sum_n += static_cast<int>((*root.N)[a]);
        // Same invariant as test_mcts: first batch leaves root.N unchanged.
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

    // Shut down the stub evaluator thread cleanly.
    queue.notify_worker_done();
    queue.shutdown();
    stub.join();
}

// ── Tier 3: full ParallelSelfPlay integration ─────────────────────────────────

static void test_parallel_selfplay(const std::string& model_path) {
    std::cout << "\n-- ParallelSelfPlay integration (model: " << model_path << ") --\n";

    ParallelSelfPlayConfig cfg;
    cfg.model_path        = model_path;
    cfg.num_games         = 2;
    cfg.num_workers       = 2;
    cfg.num_simulations   = 16;   // 2 batches of 8
    cfg.leaf_batch_size   = 8;
    cfg.max_moves         = 1;
    cfg.temperature       = 1.0f;
    cfg.min_batch_override = 8;   // standard: 1 full worker's batch per evaluator call
    cfg.max_batch_override = 16;

    try {
        ParallelSelfPlay psp(cfg);
        auto [samples, meta] = psp.run();

        check(static_cast<int>(meta.size()) == cfg.num_games,
              "correct number of games produced");
        check(!samples.empty(), "at least one sample produced");

        // Every π must sum to ~1 (valid probability distribution).
        bool pi_ok = true;
        for (const auto& s : samples) {
            float sum = 0.0f;
            for (float v : s.pi) sum += v;
            if (sum < 0.99f || sum > 1.01f) { pi_ok = false; break; }
        }
        check(pi_ok, "all policy vectors sum to ~1");

        // z values must be in [-1, 1].
        bool z_ok = true;
        for (const auto& s : samples) {
            if (s.z < -1.001f || s.z > 1.001f) { z_ok = false; break; }
        }
        check(z_ok, "all z values in [-1, 1]");

        // Game outcomes must be valid.
        bool outcome_ok = true;
        for (const auto& m : meta) {
            const auto& o = m.outcome;
            if (o != "white" && o != "black" && o != "draw" && o != "limit")
                { outcome_ok = false; break; }
        }
        check(outcome_ok, "all game outcomes are valid strings");

        int total_moves = 0;
        for (const auto& m : meta) total_moves += m.moves;
        check(total_moves > 0, "games have moves");

        std::cout << "  [info] games=" << meta.size()
                  << "  samples=" << samples.size()
                  << "  total_moves=" << total_moves << "\n";
        for (const auto& m : meta) {
            std::cout << "    game: moves=" << m.moves
                      << "  outcome=" << m.outcome << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "  [FAIL] Exception: " << ex.what() << "\n";
        ++g_fail;
    }
}

// ── entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::cout << "=== Phase 4: parallel self-play test ===\n";

    // Tier 1 (always)
    test_evalqueue_push_pop();
    test_evalqueue_shutdown_empty();
    test_evalqueue_partial_flush();
    test_evalqueue_promise_lifecycle();

    // Tier 2 + 3 (when a model is supplied)
    if (argc > 1) {
        const std::string model_path = argv[1];
        test_async_mcts_simulations(model_path);
        test_parallel_selfplay(model_path);
    } else {
        std::cout << "\n[info] Tier 2 + 3 skipped (no model path provided).\n";
        std::cout << "[info] Run: test_parallel.exe path/to/model_traced.pt\n";
    }

    std::cout << "\n=== Results: " << g_pass << " passed, "
              << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
