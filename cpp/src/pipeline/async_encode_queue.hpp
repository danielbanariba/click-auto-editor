#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <cuda_runtime.h>

#include "ffmpeg_encoder.hpp"

namespace vhs {
namespace pipeline {

struct EncodeRequest {
    const unsigned char* d_frame;
    int64_t pts;
    cudaEvent_t ready_event;
};

/**
 * Async encoding queue — runs NVENC on a dedicated thread so that
 * CUDA effect processing on the main thread can overlap with encoding.
 *
 * Backpressure: submit() blocks when the queue already holds one item,
 * ensuring at most one frame is queued ahead (double-buffer depth = 1).
 */
class AsyncEncodeQueue {
public:
    explicit AsyncEncodeQueue(FFmpegEncoder& encoder)
        : encoder_(encoder)
        , error_(false)
        , stop_(false)
        , running_(false)
    {}

    ~AsyncEncodeQueue() {
        stop();
    }

    void start() {
        stop_ = false;
        error_ = false;
        running_ = true;
        thread_ = std::thread(&AsyncEncodeQueue::encoder_loop, this);
    }

    /**
     * Submit a frame for encoding.
     * Blocks if the queue already has MAX_DEPTH items (backpressure).
     * Returns false if the encoder thread has reported an error.
     */
    bool submit(const unsigned char* d_frame, int64_t pts, cudaEvent_t ready_event) {
        std::unique_lock<std::mutex> lock(mutex_);
        // Wait until there's room or an error / stop occurred
        can_submit_.wait(lock, [this] {
            return queue_.size() < MAX_DEPTH || error_ || stop_;
        });

        if (error_ || stop_) return false;

        queue_.push({d_frame, pts, ready_event});
        lock.unlock();
        can_consume_.notify_one();
        return true;
    }

    /**
     * Wait for all queued frames to finish encoding, then join the thread.
     */
    void flush_and_stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        can_consume_.notify_one();

        if (thread_.joinable()) {
            thread_.join();
        }
        running_ = false;
    }

    /**
     * Immediate stop — discards pending frames.
     */
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            // Drain the queue so events can be cleaned up by the caller
            while (!queue_.empty()) queue_.pop();
        }
        can_consume_.notify_one();
        can_submit_.notify_all();

        if (thread_.joinable()) {
            thread_.join();
        }
        running_ = false;
    }

    bool has_error() const { return error_.load(); }
    bool is_running() const { return running_.load(); }

private:
    static constexpr size_t MAX_DEPTH = 1;

    void encoder_loop() {
        while (true) {
            EncodeRequest req;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                can_consume_.wait(lock, [this] {
                    return !queue_.empty() || stop_;
                });

                if (queue_.empty()) {
                    // stop_ was set and queue is drained
                    break;
                }

                req = queue_.front();
                queue_.pop();
            }
            // Unblock submit() now that we've consumed an item
            can_submit_.notify_one();

            // Wait for GPU effects to finish for this specific frame
            cudaError_t cuda_err = cudaEventSynchronize(req.ready_event);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "[AsyncEncoder] cudaEventSynchronize failed: %s\n",
                        cudaGetErrorString(cuda_err));
                error_ = true;
                can_submit_.notify_all();
                break;
            }

            // Encode the frame
            if (!encoder_.encode_frame(req.d_frame, req.pts)) {
                fprintf(stderr, "[AsyncEncoder] encode_frame failed at pts=%ld\n", req.pts);
                error_ = true;
                can_submit_.notify_all();
                break;
            }
        }
    }

    FFmpegEncoder& encoder_;
    std::atomic<bool> error_;
    std::atomic<bool> stop_;
    std::atomic<bool> running_;

    std::mutex mutex_;
    std::condition_variable can_submit_;
    std::condition_variable can_consume_;
    std::queue<EncodeRequest> queue_;

    std::thread thread_;
};

} // namespace pipeline
} // namespace vhs
