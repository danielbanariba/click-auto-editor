/**
 * Chromatic Aberration Effect
 *
 * Simulates lens chromatic aberration where color channels
 * focus at different planes, creating color fringing.
 *
 * Algorithm:
 * - Blue channel: shift left (-offset)
 * - Red channel: shift right (+offset)
 * - Green channel: no change
 */

#include "utils/cuda_utils.hpp"

namespace vhs {
namespace effects {

__global__ void chromatic_aberration_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height,
    int offset  // 2-6 pixels based on intensity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // Blue channel: shift left (sample from right)
    int blue_src_x = min(x + offset, width - 1);
    int blue_src_idx = (y * width + blue_src_x) * 3;

    // Red channel: shift right (sample from left)
    int red_src_x = max(x - offset, 0);
    int red_src_idx = (y * width + red_src_x) * 3;

    // Green channel: no change
    int green_src_idx = idx;

    // Write output (BGR order for OpenCV compatibility)
    output[idx + 0] = input[blue_src_idx + 0];   // B
    output[idx + 1] = input[green_src_idx + 1];  // G
    output[idx + 2] = input[red_src_idx + 2];    // R
}

void apply_chromatic_aberration(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    int offset,
    cudaStream_t stream
) {
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    chromatic_aberration_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height, offset
    );
}

} // namespace effects
} // namespace vhs
