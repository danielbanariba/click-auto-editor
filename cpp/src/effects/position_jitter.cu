/**
 * Position Jitter Effect
 *
 * Simulates mechanical instabilities in VHS playback.
 * The entire frame shifts periodically in X and Y.
 */

#include "utils/cuda_utils.hpp"

namespace vhs {
namespace effects {

__global__ void position_jitter_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height,
    int offset_x,  // -8 to +8 pixels
    int offset_y   // -12 to +12 pixels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate source position with wrap-around
    int src_x = (x - offset_x + width) % width;
    int src_y = (y - offset_y + height) % height;

    int dst_idx = (y * width + x) * 3;
    int src_idx = (src_y * width + src_x) * 3;

    // Copy pixel
    output[dst_idx + 0] = input[src_idx + 0];
    output[dst_idx + 1] = input[src_idx + 1];
    output[dst_idx + 2] = input[src_idx + 2];
}

void apply_position_jitter(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    int offset_x,
    int offset_y,
    cudaStream_t stream
) {
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    position_jitter_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height, offset_x, offset_y
    );
}

// Vertical jitter is a special case of position jitter (X = 0)
void apply_vertical_jitter(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    int offset_y,
    cudaStream_t stream
) {
    apply_position_jitter(d_input, d_output, width, height, 0, offset_y, stream);
}

} // namespace effects
} // namespace vhs
