#pragma once

#include "cuda_utils.hpp"

namespace vhs {
namespace cuda {

// Convert RGB frame to YIQ color space
// Input: d_rgb - device pointer to RGB frame (uchar3)
// Output: d_yiq - device pointer to YIQ frame (float3)
void rgb_to_yiq(const unsigned char* d_rgb, float* d_yiq,
                int width, int height, cudaStream_t stream = 0);

// Convert YIQ frame back to RGB
void yiq_to_rgb(const float* d_yiq, unsigned char* d_rgb,
                int width, int height, cudaStream_t stream = 0);

// BGR to RGB in-place conversion (for OpenCV compatibility)
void bgr_to_rgb(unsigned char* d_frame, int width, int height,
                cudaStream_t stream = 0);

} // namespace cuda
} // namespace vhs
