#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace vhs {
namespace pipeline {

/**
 * CUDA Frame Buffer Pool
 *
 * Manages GPU memory allocation for video frames.
 * Pre-allocates buffers to avoid allocation overhead during processing.
 */
class CudaFrameBuffer {
public:
    CudaFrameBuffer(int width, int height, int num_buffers = 3);
    ~CudaFrameBuffer();

    // Get a free buffer for writing
    unsigned char* acquire();

    // Release buffer back to pool
    void release(unsigned char* buffer);

    // Get buffer at specific index
    unsigned char* get(int index);

    // Properties
    int width() const { return width_; }
    int height() const { return height_; }
    size_t frame_size() const { return frame_size_; }
    int num_buffers() const { return num_buffers_; }

private:
    int width_;
    int height_;
    size_t frame_size_;
    int num_buffers_;

    std::vector<unsigned char*> buffers_;
    std::vector<bool> in_use_;
};

/**
 * VHS Overlay Buffer
 *
 * Loads VHS noise video frames into GPU memory for overlay blending.
 * Pre-decodes all frames for maximum performance during render.
 */
class VHSOverlayBuffer {
public:
    VHSOverlayBuffer();
    ~VHSOverlayBuffer();

    // Load VHS overlay video into GPU memory
    bool load(const std::string& path, int target_width, int target_height);

    // Get frame at index (loops automatically)
    unsigned char* get_frame(int frame_index) const;

    // Properties
    int frame_count() const { return frame_count_; }
    bool is_loaded() const { return is_loaded_; }

private:
    std::vector<unsigned char*> d_frames_;
    int frame_count_;
    int width_;
    int height_;
    size_t frame_size_;
    bool is_loaded_;
};

} // namespace pipeline
} // namespace vhs
