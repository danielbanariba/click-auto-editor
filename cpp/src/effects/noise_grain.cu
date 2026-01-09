/**
 * Noise/Grain Effect
 *
 * Adds analog sensor noise characteristic of VHS:
 * - Luminance noise: affects all channels equally
 * - Color noise: independent per channel (only if intensity > 0.3)
 */

#include "utils/cuda_utils.hpp"
#include <curand_kernel.h>

namespace vhs {
namespace effects {

// Initialize cuRAND states
__global__ void init_curand_states(
    curandState* states,
    unsigned long seed,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void noise_grain_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height,
    curandState* states,
    float noise_luma,      // 10-30
    float noise_color,     // 0-5
    bool apply_color_noise
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int pixel_idx = idx * 3;

    // Get local RNG state
    curandState local_state = states[idx];

    // Read BGR
    float b = static_cast<float>(input[pixel_idx + 0]);
    float g = static_cast<float>(input[pixel_idx + 1]);
    float r = static_cast<float>(input[pixel_idx + 2]);

    // Luminance noise (same for all channels)
    float luma_noise = curand_normal(&local_state) * noise_luma;
    b += luma_noise;
    g += luma_noise;
    r += luma_noise;

    // Color noise (independent per channel)
    if (apply_color_noise && noise_color > 0.0f) {
        b += curand_normal(&local_state) * noise_color;
        g += curand_normal(&local_state) * noise_color;
        r += curand_normal(&local_state) * noise_color;
    }

    // Clamp and write
    output[pixel_idx + 0] = static_cast<unsigned char>(fminf(fmaxf(b, 0.0f), 255.0f));
    output[pixel_idx + 1] = static_cast<unsigned char>(fminf(fmaxf(g, 0.0f), 255.0f));
    output[pixel_idx + 2] = static_cast<unsigned char>(fminf(fmaxf(r, 0.0f), 255.0f));

    // Save RNG state
    states[idx] = local_state;
}

// Host-side cuRAND state management
static curandState* d_curand_states = nullptr;
static int curand_width = 0;
static int curand_height = 0;

void init_noise_rng(int width, int height, unsigned long seed) {
    if (d_curand_states != nullptr && curand_width == width && curand_height == height) {
        return;  // Already initialized
    }

    // Free previous states if any
    if (d_curand_states != nullptr) {
        cudaFree(d_curand_states);
    }

    // Allocate new states
    size_t state_size = width * height * sizeof(curandState);
    CUDA_CHECK(cudaMalloc(&d_curand_states, state_size));

    // Initialize states
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();
    init_curand_states<<<grid, block>>>(d_curand_states, seed, width, height);

    curand_width = width;
    curand_height = height;

    cuda::sync_check();
}

void cleanup_noise_rng() {
    if (d_curand_states != nullptr) {
        cudaFree(d_curand_states);
        d_curand_states = nullptr;
    }
}

void apply_noise_grain(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    float noise_luma,
    float noise_color,
    bool apply_color_noise,
    cudaStream_t stream
) {
    // Ensure RNG is initialized
    if (d_curand_states == nullptr) {
        init_noise_rng(width, height, 42);
    }

    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    noise_grain_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height,
        d_curand_states, noise_luma, noise_color, apply_color_noise
    );
}

} // namespace effects
} // namespace vhs
