#pragma once
// CentralEvaluator.h — Phase 4: batched GPU inference thread
//
// Owns the torch::jit::Module.  Drains EvalQueue in batches, runs one forward
// pass per batch, and fulfils each EvalRequest's promise with an EvalResult.
// Must be the *only* object that ever calls model_.forward() — workers never
// touch the model directly.

#include "Evaluator.h"
#include <atomic>
#include <thread>
#include <torch/script.h>

class CentralEvaluator {
public:
    // model         — loaded TorchScript model (moved in, owned here).
    // queue         — shared EvalQueue (non-owning pointer, must outlive this).
    // device        — kCUDA or kCPU.
    // min_batch     — evaluator waits until this many leaves are queued.
    //                 Set to 1 for low latency; num_workers*leaf_batch for throughput.
    // max_batch     — maximum leaves per forward pass (caps GPU memory usage).
    CentralEvaluator(torch::jit::Module model,
                     EvalQueue*         queue,
                     torch::Device      device,
                     int                min_batch,
                     int                max_batch);

    // Non-copyable, non-movable (owns std::thread).
    CentralEvaluator(const CentralEvaluator&)            = delete;
    CentralEvaluator& operator=(const CentralEvaluator&) = delete;

    // Starts the evaluator thread.  Call before launching worker threads.
    void start();

    // Blocks until the evaluator thread exits.
    // Call only after EvalQueue::shutdown() has been called.
    void join();

    bool is_running() const { return running_.load(); }

private:
    torch::jit::Module  model_;
    EvalQueue*          queue_;   // non-owning
    torch::Device       device_;
    int                 min_batch_;
    int                 max_batch_;
    std::thread         thread_;
    std::atomic<bool>   running_{false};

    // Pre-allocated input tensors (reused every batch to avoid per-call allocation).
    // Initialized in run() after the CUDA context is set up.
    torch::Tensor       states_cpu_;  // {max_batch_, 20, 8, 8} pinned CPU memory
    torch::Tensor       states_gpu_;  // {max_batch_, 20, 8, 8} on device_

    void run();
    void process_batch(std::vector<EvalRequest>& batch);
};
