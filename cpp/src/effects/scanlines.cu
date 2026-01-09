/**
 * Scanlines Effect
 *
 * Simulates interlaced video (480i/576i) with visible scan lines.
 * Every other line is darkened to create the characteristic look.
 */

#include "utils/cuda_utils.hpp"

namespace vhs {
namespace effects {

__global__ void scanlines_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height,
    float darkness  // 0.7-0.85 based on intensity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // Even lines are darker
    float factor = (y % 2 == 0) ? darkness : 1.0f;

    // Apply darkness factor to all channels
    output[idx + 0] = static_cast<unsigned char>(input[idx + 0] * factor);
    output[idx + 1] = static_cast<unsigned char>(input[idx + 1] * factor);
    output[idx + 2] = static_cast<unsigned char>(input[idx + 2] * factor);
}

void apply_scanlines(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    float darkness,
    cudaStream_t stream
) {
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    scanlines_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height, darkness
    );
}

} // namespace effects
} // namespace vhs
