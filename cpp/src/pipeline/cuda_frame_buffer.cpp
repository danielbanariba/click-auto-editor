/**
 * CUDA Frame Buffer Implementation
 */

#include "cuda_frame_buffer.hpp"
#include "ffmpeg_decoder.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/cancel_flag.hpp"
#include <cstdio>
#include <algorithm>

namespace vhs {
namespace pipeline {

// ============================================
// CudaFrameBuffer Implementation
// ============================================

CudaFrameBuffer::CudaFrameBuffer(int width, int height, int num_buffers)
    : width_(width)
    , height_(height)
    , frame_size_(width * height * 3)
    , num_buffers_(num_buffers)
{
    buffers_.resize(num_buffers, nullptr);
    in_use_.resize(num_buffers, false);

    // Allocate GPU buffers
    for (int i = 0; i < num_buffers; i++) {
        CUDA_CHECK(cudaMalloc(&buffers_[i], frame_size_));
    }

    printf("[CudaFrameBuffer] Allocated %d buffers (%zu MB total)\n",
           num_buffers, (frame_size_ * num_buffers) / (1024 * 1024));
}

CudaFrameBuffer::~CudaFrameBuffer() {
    for (auto& buffer : buffers_) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
}

unsigned char* CudaFrameBuffer::acquire() {
    for (int i = 0; i < num_buffers_; i++) {
        if (!in_use_[i]) {
            in_use_[i] = true;
            return buffers_[i];
        }
    }
    // No free buffer - this shouldn't happen with proper synchronization
    fprintf(stderr, "[CudaFrameBuffer] Warning: No free buffers available!\n");
    return nullptr;
}

void CudaFrameBuffer::release(unsigned char* buffer) {
    for (int i = 0; i < num_buffers_; i++) {
        if (buffers_[i] == buffer) {
            in_use_[i] = false;
            return;
        }
    }
    fprintf(stderr, "[CudaFrameBuffer] Warning: Buffer not found in pool!\n");
}

unsigned char* CudaFrameBuffer::get(int index) {
    if (index >= 0 && index < num_buffers_) {
        return buffers_[index];
    }
    return nullptr;
}

// ============================================
// VHSOverlayBuffer Implementation
// ============================================

VHSOverlayBuffer::VHSOverlayBuffer()
    : frame_count_(0)
    , width_(0)
    , height_(0)
    , frame_size_(0)
    , is_loaded_(false)
{
}

VHSOverlayBuffer::~VHSOverlayBuffer() {
    for (auto& frame : d_frames_) {
        if (frame) {
            cudaFree(frame);
        }
    }
    d_frames_.clear();
}

bool VHSOverlayBuffer::load(const std::string& path, int target_width, int target_height) {
    // Clear existing frames
    for (auto& frame : d_frames_) {
        if (frame) cudaFree(frame);
    }
    d_frames_.clear();
    is_loaded_ = false;

    // Open VHS overlay video - prefer software decoding to avoid format issues
    FFmpegDecoder decoder;
    if (!decoder.open(path, false)) {  // Force software decoding
        fprintf(stderr, "[VHSOverlay] Failed to open: %s\n", path.c_str());
        return false;
    }

    width_ = target_width;
    height_ = target_height;
    frame_size_ = width_ * height_ * 3;

    // Limit frames to avoid VRAM exhaustion
    // At 4K (24MB/frame), 40 frames ≈ 1 GB
    constexpr int MAX_OVERLAY_FRAMES = 150;

    // Calculate VRAM budget (cap to ~1 GB, and adapt to free VRAM)
    size_t max_vram_bytes = 1ULL * 1024 * 1024 * 1024;
    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        size_t dynamic_cap = free_mem / 6;  // conservative per-process budget
        if (dynamic_cap < max_vram_bytes) {
            max_vram_bytes = dynamic_cap;
        }
    }
    if (max_vram_bytes < frame_size_) {
        max_vram_bytes = frame_size_;
    }
    int max_frames_by_vram = static_cast<int>(max_vram_bytes / frame_size_);
    int max_frames = std::min(MAX_OVERLAY_FRAMES, max_frames_by_vram);

    printf("[VHSOverlay] Loading up to %d frames (%.1f MB VRAM budget)...\n",
           max_frames, (frame_size_ * max_frames) / (1024.0 * 1024.0));

    // Allocate temporary buffer for decoding
    size_t decode_frame_size = static_cast<size_t>(decoder.width()) * decoder.height() * 3;
    unsigned char* d_temp = nullptr;
    cudaError_t temp_err = cudaMalloc(&d_temp, decode_frame_size);
    if (temp_err != cudaSuccess) {
        fprintf(stderr, "[VHSOverlay] Failed to allocate temp buffer: %s\n",
                cudaGetErrorString(temp_err));
        decoder.close();
        return false;
    }

    int frames_loaded = 0;
    int64_t pts;

    while (frames_loaded < max_frames) {
        if (vhs::utils::is_cancel_requested()) {
            fprintf(stderr, "[VHSOverlay] Cancel requested, stopping overlay load\n");
            break;
        }
        // Try to decode next frame
        if (!decoder.decode_frame(d_temp, &pts)) {
            break;  // End of video or error
        }

        // Allocate GPU memory for this frame
        unsigned char* d_frame = nullptr;
        cudaError_t err = cudaMalloc(&d_frame, frame_size_);
        if (err != cudaSuccess || d_frame == nullptr) {
            fprintf(stderr, "[VHSOverlay] Failed to allocate frame %d\n", frames_loaded);
            break;
        }

        // For simplicity, if sizes match, direct copy
        // Otherwise would need resize kernel (omitted for brevity)
        if (decoder.width() == target_width && decoder.height() == target_height) {
            err = cudaMemcpy(d_frame, d_temp, frame_size_, cudaMemcpyDeviceToDevice);
        } else {
            // TODO: Add resize kernel or use NPP resize
            // For now, just fill with gray
            err = cudaMemset(d_frame, 128, frame_size_);
        }

        if (err != cudaSuccess) {
            cudaFree(d_frame);
            fprintf(stderr, "[VHSOverlay] Failed to copy frame %d\n", frames_loaded);
            break;
        }

        d_frames_.push_back(d_frame);
        frames_loaded++;

        // Progress every 50 frames
        if (frames_loaded % 50 == 0) {
            printf("[VHSOverlay] Loaded %d/%d frames...\n", frames_loaded, max_frames);
        }
    }

    cudaFree(d_temp);
    decoder.close();

    // Sync and clear any CUDA errors
    cudaDeviceSynchronize();
    cudaGetLastError();  // Clear error state

    frame_count_ = frames_loaded;
    is_loaded_ = (frame_count_ > 0);

    if (is_loaded_) {
        printf("[VHSOverlay] Loaded %d frames (%.1f MB VRAM)\n",
               frame_count_, (frame_size_ * frame_count_) / (1024.0 * 1024.0));
    } else {
        printf("[VHSOverlay] No frames loaded\n");
    }

    return is_loaded_;
}

unsigned char* VHSOverlayBuffer::get_frame(int frame_index) const {
    if (!is_loaded_ || d_frames_.empty()) {
        return nullptr;
    }
    // Loop through frames
    int idx = frame_index % frame_count_;
    return d_frames_[idx];
}

} // namespace pipeline
} // namespace vhs
