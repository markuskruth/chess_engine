// CentralEvaluator.cpp — Phase 4 implementation

#include "CentralEvaluator.h"
#include <ATen/cuda/CUDAContext.h>
#include <chrono>
#include <cstring>  // memcpy
#include <iostream>

CentralEvaluator::CentralEvaluator(torch::jit::Module model,
                                   EvalQueue*         queue,
                                   torch::Device      device,
                                   int                min_batch,
                                   int                max_batch)
    : model_(std::move(model)),
      queue_(queue),
      device_(device),
      min_batch_(min_batch),
      max_batch_(max_batch)
{
    // Move cuda initialization from here to inside the execution thread
    //model_.to(device_);
    //model_.eval();
}

void CentralEvaluator::start() {
    running_ = true;
    thread_  = std::thread([this] { run(); });
}

void CentralEvaluator::join() {
    if (thread_.joinable()) thread_.join();
}

// ── run ───────────────────────────────────────────────────────────────────────

void CentralEvaluator::run() {
    try {
        // GPU thread now owns the CUDA context — all model ops happen here.
        model_.to(device_);
        model_.eval();

        // Disable cuDNN auto-benchmark: prevents cuDNN from hanging while it
        // profiles kernel candidates for this input shape.
        at::globalContext().setBenchmarkCuDNN(false);

        // Pre-allocate fixed-size input tensors (reused every batch).
        // Using pinned (page-locked) CPU memory for fast async DMA transfers.
        const bool use_cuda = (device_.type() == torch::kCUDA);
        states_cpu_ = torch::zeros(
            {max_batch_, STATE_CHANNELS, BOARD_SZ, BOARD_SZ},
            torch::TensorOptions().dtype(torch::kFloat32)
                                  .pinned_memory(use_cuda));
        states_gpu_ = torch::zeros(
            {max_batch_, STATE_CHANNELS, BOARD_SZ, BOARD_SZ},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_));

        // Warmup: run one forward pass before the main loop so that cuDNN
        // selects and caches its kernel implementation upfront.  This prevents
        // the first real batch from paying a large latency spike.
        {
            torch::NoGradGuard no_grad;
            model_.forward({states_gpu_});
            if (use_cuda) {
                c10::cuda::device_synchronize();
            }
        }

        int  batch_count    = 0;
        auto last_heartbeat = std::chrono::steady_clock::now();
        auto last_forward   = std::chrono::steady_clock::now();

        std::cerr << "[DEBUG evaluator] started (warmup done), min_batch=" << min_batch_
                  << " max_batch=" << max_batch_ << "\n" << std::flush;

        while (queue_->wait_for_batch(min_batch_)) {
            auto batch = queue_->pop_batch(max_batch_);
            if (!batch.empty()) {
                batch_count++;
                auto now = std::chrono::steady_clock::now();

                // Print at startup and then every 10 seconds.
                auto secs_since_hb = std::chrono::duration_cast<std::chrono::seconds>(
                                         now - last_heartbeat).count();
                if (batch_count <= 3 || secs_since_hb >= 10) {
                    std::cerr << "[DEBUG evaluator] batch #" << batch_count
                              << "  size=" << batch.size()
                              << "  active_workers=" << queue_->active_worker_count()
                              << "  queue_after_pop=" << queue_->pending_count()
                              << "\n" << std::flush;
                    last_heartbeat = now;
                }

                last_forward = std::chrono::steady_clock::now();
                process_batch(batch);  // GPU forward pass inside here

                // Detect if forward() itself was suspiciously slow.
                auto forward_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now() - last_forward).count();
                if (forward_ms > 5000) {
                    std::cerr << "[DEBUG evaluator] WARNING: process_batch took "
                              << forward_ms << "ms for batch #" << batch_count
                              << " (size=" << batch.size() << ")\n" << std::flush;
                }
            } else {
                // Queue empty but workers still active — print periodic heartbeat.
                auto now  = std::chrono::steady_clock::now();
                auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                                now - last_heartbeat).count();
                if (secs >= 5) {
                    std::cerr << "[DEBUG evaluator] idle: active_workers="
                              << queue_->active_worker_count()
                              << "  queue_pending=" << queue_->pending_count()
                              << "  batches_processed=" << batch_count
                              << "\n" << std::flush;
                    last_heartbeat = now;
                }
            }
        }
        std::cerr << "[DEBUG evaluator] exiting, batches_processed="
                  << batch_count << "\n" << std::flush;
    } catch (const std::exception& e) {
        // If the GPU throws an OOM or shape error, print it loudly!
        std::cerr << "\n[FATAL ERROR] GPU Evaluator thread crashed: " << e.what() << "\n";
        // Instantly kill the entire C++ program so the workers don't deadlock
        std::exit(1);
    }
    running_ = false;
}

// ── process_batch ─────────────────────────────────────────────────────────────

void CentralEvaluator::process_batch(std::vector<EvalRequest>& batch) {
    const int B = static_cast<int>(batch.size());
    constexpr int STATE_FLOATS = STATE_CHANNELS * BOARD_SZ * BOARD_SZ;  // 1280

    // 1. Fill the pre-allocated pinned CPU buffer (zero tail to keep shape fixed).
    float* dst = states_cpu_.data_ptr<float>();
    // Fill valid entries; zero the tail so the GPU always sees the same shape.
    for (int i = 0; i < B; ++i) {
        std::memcpy(dst + i * STATE_FLOATS,
                    batch[i].state.data(),
                    STATE_FLOATS * sizeof(float));
    }
    if (B < max_batch_) {
        std::memset(dst + B * STATE_FLOATS, 0,
                    (max_batch_ - B) * STATE_FLOATS * sizeof(float));
    }

    // 2. Async DMA from pinned CPU → GPU (non-blocking; CUDA stream ensures
    //    the copy completes before the forward pass uses the data).
    states_gpu_.copy_(states_cpu_, /*non_blocking=*/true);

    // 3. Forward pass — shape {max_batch_, 20, 8, 8} NEVER changes.
    torch::Tensor p_batch, v_batch;
    {
        torch::NoGradGuard no_grad;
        auto out = model_.forward({states_gpu_}).toTuple();

        // Slice only the first B valid results before copying to CPU.
        p_batch = out->elements()[0].toTensor()
                     .slice(0, 0, B)
                     .to(torch::kCPU).contiguous()
                     .reshape({B, ACTION_DIM});

        v_batch = out->elements()[1].toTensor()
                     .slice(0, 0, B)
                     .to(torch::kCPU).contiguous()
                     .reshape({B});
    }

    const float* p_data = p_batch.data_ptr<float>();
    const float* v_data = v_batch.data_ptr<float>();

    // 3. Fulfill the promises
    for (int i = 0; i < B; ++i) {
        EvalResult res;
        std::memcpy(res.policy.data(),
                    p_data + i * ACTION_DIM,
                    ACTION_DIM * sizeof(float));
        res.value = v_data[i];
        batch[i].promise.set_value(std::move(res));
    }
}
