// CentralEvaluator.cpp — Phase 4 implementation

#include "CentralEvaluator.h"
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
        // Initialize here instead
        // The GPU thread now securely owns the CUDA memory context
        model_.to(device_);
        model_.eval();

        int  batch_count   = 0;
        auto last_heartbeat = std::chrono::steady_clock::now();
        auto last_forward   = std::chrono::steady_clock::now();

        std::cerr << "[DEBUG evaluator] started, min_batch=" << min_batch_
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

    // 1. ALWAYS allocate a tensor of size max_batch_ to prevent cuDNN deadlocks
    auto states = torch::zeros({max_batch_, STATE_CHANNELS, BOARD_SZ, BOARD_SZ},
                               torch::kFloat32);
                               
    float* dst = states.data_ptr<float>();
    for (int i = 0; i < B; ++i) {
        std::memcpy(dst + i * STATE_FLOATS,
                    batch[i].state.data(),
                    STATE_FLOATS * sizeof(float));
    }
    states = states.to(device_);

    // 2. Forward pass — the shape {max_batch_, 20, 8, 8} NEVER changes.
    torch::Tensor p_batch, v_batch;
    {
        torch::NoGradGuard no_grad;
        auto out = model_.forward({states}).toTuple();
        
        // Extract policy and slice only the first 'B' valid results
        p_batch = out->elements()[0].toTensor()
                     .slice(0, 0, B) // <--- Slice here!
                     .to(torch::kCPU).contiguous()
                     .reshape({B, ACTION_DIM});
                     
        // Extract value and slice only the first 'B' valid results
        v_batch = out->elements()[1].toTensor()
                     .slice(0, 0, B) // <--- Slice here!
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
