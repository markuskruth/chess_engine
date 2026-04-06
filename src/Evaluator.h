#pragma once
// Evaluator.h — Phase 4: thread-safe leaf evaluation queue
//
// MCTS worker threads submit leaf states as EvalRequests; the CentralEvaluator
// thread batches them for efficient GPU inference and fulfils each promise.

#include "ChessEnv.h"
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <vector>

// ── EvalResult ────────────────────────────────────────────────────────────────
// Returned to each AsyncMCTS worker via std::future<EvalResult>.
// policy[] contains raw NN logits (ACTION_DIM values, before masked softmax).
// value is the scalar board evaluation output by the network.

struct EvalResult {
    std::array<float, ACTION_DIM> policy;  // raw logits, size 4672
    float                         value;
};

// ── EvalRequest ───────────────────────────────────────────────────────────────
// One leaf evaluation request.  The state is pre-encoded on the worker thread
// (CPU) to avoid per-request GPU work inside the queue drain loop.
// EvalRequest is move-only (holds a std::promise).

struct EvalRequest {
    // Pre-encoded board: flat float[STATE_CHANNELS * BOARD_SZ * BOARD_SZ = 1280]
    std::array<float, STATE_CHANNELS * BOARD_SZ * BOARD_SZ> state;
    std::promise<EvalResult> promise;
};

// ── EvalQueue ─────────────────────────────────────────────────────────────────
// Thread-safe MPSC queue: N worker threads push(); 1 evaluator thread drains.
//
// Shutdown protocol:
//   1. Each worker calls notify_worker_done() when it finishes all its games.
//   2. When active_workers_ reaches 0, the evaluator's wait is unblocked so
//      any partial batch is flushed immediately (no indefinite stall).
//   3. ParallelSelfPlay calls shutdown() after joining workers, which causes
//      wait_for_batch() to return false on the next (empty) drain attempt.

class EvalQueue {
public:
    // total_workers: number of worker threads that will call push().
    explicit EvalQueue(int total_workers);

    // Push one request from a worker thread.  Thread-safe.
    void push(EvalRequest req);

    // Called by the evaluator thread.
    // Blocks until either:
    //   (a) queue_.size() >= min_batch_size, or
    //   (b) active_workers_ == 0 and queue is non-empty (partial-batch flush), or
    //   (c) shutdown() was called and the queue is empty.
    // Returns true  if items may be available (call pop_batch next).
    // Returns false if shutdown() was called and the queue is empty (exit loop).
    bool wait_for_batch(int min_batch_size);

    // Drain up to max_batch items from the front of the queue.
    // Must be called only after wait_for_batch() returned true.
    std::vector<EvalRequest> pop_batch(int max_batch);

    // Called by each worker thread exactly once when it finishes all games.
    // Decrements active_workers_; if it hits 0, notifies the evaluator so any
    // remaining partial batch is processed without waiting for min_batch_size.
    void notify_worker_done();

    // Called by ParallelSelfPlay after all workers have finished.
    // Causes wait_for_batch() to return false once the queue is empty.
    void shutdown();

    // Approximate diagnostics (non-blocking).
    int pending_count() const;
    int active_worker_count() const;

private:
    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    std::deque<EvalRequest> queue_;
    std::atomic<int>        active_workers_;
    bool                    shutdown_ = false;
};
