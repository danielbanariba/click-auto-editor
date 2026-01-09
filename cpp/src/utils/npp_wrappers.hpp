#pragma once

#include <npp.h>
#include <nppi.h>
#include <cuda_runtime.h>

namespace vhs {
namespace npp {

/**
 * NPP Gaussian Blur Wrapper
 *
 * Simplified wrapper for NPP Gaussian filter operations.
 * Used by color_bleeding and sharpness_reduction effects.
 */

// Initialize NPP stream context
void init_npp_stream(cudaStream_t stream);

// Gaussian blur for single channel (float)
bool gaussian_blur_32f_c1(
    const float* d_src,
    float* d_dst,
    int width,
    int height,
    int src_step,
    int dst_step,
    int kernel_size,
    cudaStream_t stream = nullptr
);

// Gaussian blur for RGB image (unsigned char)
bool gaussian_blur_8u_c3(
    const unsigned char* d_src,
    unsigned char* d_dst,
    int width,
    int height,
    int src_step,
    int dst_step,
    int kernel_size,
    cudaStream_t stream = nullptr
);

// Box filter (faster alternative to Gaussian)
bool box_filter_8u_c3(
    const unsigned char* d_src,
    unsigned char* d_dst,
    int width,
    int height,
    int src_step,
    int dst_step,
    int kernel_size,
    cudaStream_t stream = nullptr
);

// Resize image using NPP
bool resize_8u_c3(
    const unsigned char* d_src,
    int src_width,
    int src_height,
    int src_step,
    unsigned char* d_dst,
    int dst_width,
    int dst_height,
    int dst_step,
    int interpolation = NPPI_INTER_LINEAR,
    cudaStream_t stream = nullptr
);

// Get required scratch buffer size for Gaussian
size_t get_gaussian_buffer_size(int width, int height, int kernel_size);

} // namespace npp
} // namespace vhs
