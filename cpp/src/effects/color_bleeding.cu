/**
 * Color Bleeding Effect
 *
 * Simulates VHS "color-under" where chrominance has ~8x lower
 * resolution than luminance. This creates horizontal color smearing.
 *
 * Algorithm:
 * 1. Convert RGB to YIQ color space
 * 2. Apply horizontal Gaussian blur to I and Q channels (chrominance)
 * 3. Keep Y channel (luminance) untouched
 * 4. Convert back to RGB
 *
 * Note: Uses NPP for the Gaussian blur operation.
 */

#include "utils/cuda_utils.hpp"
#include "utils/color_conversion.hpp"
#include "utils/npp_wrappers.hpp"

namespace vhs {
namespace effects {

// Temporary buffers for YIQ processing
static float* d_yiq_buffer = nullptr;
static float* d_i_channel = nullptr;
static float* d_q_channel = nullptr;
static float* d_i_blurred = nullptr;
static float* d_q_blurred = nullptr;
static int buffer_width = 0;
static int buffer_height = 0;

// Kernel to extract I and Q channels from YIQ buffer
__global__ void extract_iq_kernel(
    const float* __restrict__ yiq,
    float* __restrict__ i_channel,
    float* __restrict__ q_channel,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int yiq_idx = idx * 3;

    i_channel[idx] = yiq[yiq_idx + 1];
    q_channel[idx] = yiq[yiq_idx + 2];
}

// Kernel to merge blurred I/Q back into YIQ buffer
__global__ void merge_iq_kernel(
    float* __restrict__ yiq,
    const float* __restrict__ i_blurred,
    const float* __restrict__ q_blurred,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int yiq_idx = idx * 3;

    yiq[yiq_idx + 1] = i_blurred[idx];
    yiq[yiq_idx + 2] = q_blurred[idx];
}

void init_color_bleeding_buffers(int width, int height) {
    if (d_yiq_buffer != nullptr && buffer_width == width && buffer_height == height) {
        return;
    }

    // Free previous buffers
    if (d_yiq_buffer != nullptr) {
        cudaFree(d_yiq_buffer);
        cudaFree(d_i_channel);
        cudaFree(d_q_channel);
        cudaFree(d_i_blurred);
        cudaFree(d_q_blurred);
    }

    size_t yiq_size = width * height * 3 * sizeof(float);
    size_t channel_size = width * height * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_yiq_buffer, yiq_size));
    CUDA_CHECK(cudaMalloc(&d_i_channel, channel_size));
    CUDA_CHECK(cudaMalloc(&d_q_channel, channel_size));
    CUDA_CHECK(cudaMalloc(&d_i_blurred, channel_size));
    CUDA_CHECK(cudaMalloc(&d_q_blurred, channel_size));

    buffer_width = width;
    buffer_height = height;
}

void cleanup_color_bleeding_buffers() {
    if (d_yiq_buffer != nullptr) {
        cudaFree(d_yiq_buffer);
        cudaFree(d_i_channel);
        cudaFree(d_q_channel);
        cudaFree(d_i_blurred);
        cudaFree(d_q_blurred);
        d_yiq_buffer = nullptr;
    }
}

void apply_color_bleeding(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    int blur_amount,  // 5-25 pixels
    cudaStream_t stream
) {
    // Ensure buffers are allocated
    init_color_bleeding_buffers(width, height);

    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    // Step 1: Convert RGB to YIQ
    cuda::rgb_to_yiq(d_input, d_yiq_buffer, width, height, stream);

    // Step 2: Extract I and Q channels
    extract_iq_kernel<<<grid, block, 0, stream>>>(
        d_yiq_buffer, d_i_channel, d_q_channel, width, height
    );

    // Step 3: Apply Gaussian blur using NPP wrapper
    int step = width * static_cast<int>(sizeof(float));
    size_t channel_size = static_cast<size_t>(width) * height * sizeof(float);

    bool blur_i_ok = npp::gaussian_blur_32f_c1(
        d_i_channel, d_i_blurred,
        width, height,
        step, step,
        blur_amount,
        stream
    );
    if (!blur_i_ok) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_i_blurred, d_i_channel, channel_size,
            cudaMemcpyDeviceToDevice, stream
        ));
    }

    bool blur_q_ok = npp::gaussian_blur_32f_c1(
        d_q_channel, d_q_blurred,
        width, height,
        step, step,
        blur_amount,
        stream
    );
    if (!blur_q_ok) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_q_blurred, d_q_channel, channel_size,
            cudaMemcpyDeviceToDevice, stream
        ));
    }

    // Step 4: Merge blurred I/Q back
    merge_iq_kernel<<<grid, block, 0, stream>>>(
        d_yiq_buffer, d_i_blurred, d_q_blurred, width, height
    );

    // Step 5: Convert back to RGB
    cuda::yiq_to_rgb(d_yiq_buffer, d_output, width, height, stream);
}

} // namespace effects
} // namespace vhs
