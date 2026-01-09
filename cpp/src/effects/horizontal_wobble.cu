/**
 * Horizontal Wobble Effect
 *
 * Simulates the "gelatin effect" caused by tape speed variations
 * and tracking errors. Each scan line is displaced independently
 * based on a sinusoidal function.
 */

#include "utils/cuda_utils.hpp"
#include <math.h>

namespace vhs {
namespace effects {

__global__ void horizontal_wobble_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height,
    float frequency,    // 30-60 vertical undulations
    float amplitude,    // 0-15 pixels
    float time_offset   // time_t * speed (speed = 2.0)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate sinusoidal offset for this line
    float normalized_y = static_cast<float>(y) / static_cast<float>(height);
    int offset = static_cast<int>(sinf(normalized_y * frequency + time_offset) * amplitude);

    // Calculate source X with clamping (not wrap)
    int src_x = x - offset;
    src_x = max(0, min(width - 1, src_x));

    int dst_idx = (y * width + x) * 3;
    int src_idx = (y * width + src_x) * 3;

    // Copy pixel
    output[dst_idx + 0] = input[src_idx + 0];
    output[dst_idx + 1] = input[src_idx + 1];
    output[dst_idx + 2] = input[src_idx + 2];
}

void apply_horizontal_wobble(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    float frequency,
    float amplitude,
    float time_offset,
    cudaStream_t stream
) {
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    horizontal_wobble_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height,
        frequency, amplitude, time_offset
    );
}

} // namespace effects
} // namespace vhs
