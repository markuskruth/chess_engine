// Evaluator.cpp — Phase 4: EvalQueue implementation

#include "Evaluator.h"
#include <chrono>
#include <iostream>

EvalQueue::EvalQueue(int total_workers)
    : active_workers_(total_workers)
{}

void EvalQueue::push(EvalRequest req) {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        queue_.push_back(std::move(req));
    }
    cv_.notify_one();
}

bool EvalQueue::wait_for_batch(int min_batch_size) {
    std::unique_lock<std::mutex> lk(mutex_);
    
    cv_.wait_for(lk, std::chrono::milliseconds(2), [&] {
        return static_cast<int>(queue_.size()) >= min_batch_size || 
               active_workers_.load() == 0 || 
               shutdown_;
    });

    // The ONLY time we want the Evaluator thread to exit (return false)
    // is if all workers are finished/shutting down AND there is no data left to process.
    if (queue_.empty() && (active_workers_.load() == 0 || shutdown_)) {
        return false;
    }

    // Otherwise, keep the Evaluator alive! 
    // Even if the queue is temporarily empty, the workers are just thinking.
    return true;
}

std::vector<EvalRequest> EvalQueue::pop_batch(int max_batch) {
    std::lock_guard<std::mutex> lk(mutex_);
    int n = std::min(max_batch, static_cast<int>(queue_.size()));
    std::vector<EvalRequest> batch;
    batch.reserve(n);
    for (int i = 0; i < n; ++i) {
        batch.push_back(std::move(queue_.front()));
        queue_.pop_front();
    }
    return batch;
}

void EvalQueue::notify_worker_done() {
    int remaining = --active_workers_;
    std::cerr << "[DEBUG] notify_worker_done: active_workers=" << remaining
              << "  queue_size=" << pending_count() << "\n" << std::flush;
    if (remaining == 0) {
        cv_.notify_all();  // wake evaluator to flush partial batch
    }
}

void EvalQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
}

int EvalQueue::pending_count() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<int>(queue_.size());
}

int EvalQueue::active_worker_count() const {
    return active_workers_.load();
}
