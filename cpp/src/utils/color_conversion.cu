#include "color_conversion.hpp"
#include "config/render_config.hpp"

namespace vhs {
namespace cuda {

// YIQ conversion matrices in constant memory for fast access
__constant__ float c_rgb_to_yiq[9] = {
    0.299f,  0.587f,  0.114f,   // Y
    0.596f, -0.274f, -0.322f,   // I
    0.211f, -0.523f,  0.312f    // Q
};

__constant__ float c_yiq_to_rgb[9] = {
    1.0f,  0.956f,  0.621f,     // R
    1.0f, -0.272f, -0.647f,     // G
    1.0f, -1.106f,  1.703f      // B
};

// RGB to YIQ kernel
__global__ void rgb_to_yiq_kernel(
    const unsigned char* __restrict__ rgb,
    float* __restrict__ yiq,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int rgb_idx = idx * 3;
    int yiq_idx = idx * 3;

    // Read RGB (normalized to 0-1)
    float r = rgb[rgb_idx + 0] / 255.0f;
    float g = rgb[rgb_idx + 1] / 255.0f;
    float b = rgb[rgb_idx + 2] / 255.0f;

    // Convert to YIQ using constant memory matrices
    float Y = c_rgb_to_yiq[0] * r + c_rgb_to_yiq[1] * g + c_rgb_to_yiq[2] * b;
    float I = c_rgb_to_yiq[3] * r + c_rgb_to_yiq[4] * g + c_rgb_to_yiq[5] * b;
    float Q = c_rgb_to_yiq[6] * r + c_rgb_to_yiq[7] * g + c_rgb_to_yiq[8] * b;

    // Write YIQ
    yiq[yiq_idx + 0] = Y;
    yiq[yiq_idx + 1] = I;
    yiq[yiq_idx + 2] = Q;
}

// YIQ to RGB kernel
__global__ void yiq_to_rgb_kernel(
    const float* __restrict__ yiq,
    unsigned char* __restrict__ rgb,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int yiq_idx = idx * 3;
    int rgb_idx = idx * 3;

    // Read YIQ
    float Y = yiq[yiq_idx + 0];
    float I = yiq[yiq_idx + 1];
    float Q = yiq[yiq_idx + 2];

    // Convert to RGB using constant memory matrices
    float r = c_yiq_to_rgb[0] * Y + c_yiq_to_rgb[1] * I + c_yiq_to_rgb[2] * Q;
    float g = c_yiq_to_rgb[3] * Y + c_yiq_to_rgb[4] * I + c_yiq_to_rgb[5] * Q;
    float b = c_yiq_to_rgb[6] * Y + c_yiq_to_rgb[7] * I + c_yiq_to_rgb[8] * Q;

    // Clamp to 0-255 and write
    rgb[rgb_idx + 0] = static_cast<unsigned char>(fminf(fmaxf(r * 255.0f, 0.0f), 255.0f));
    rgb[rgb_idx + 1] = static_cast<unsigned char>(fminf(fmaxf(g * 255.0f, 0.0f), 255.0f));
    rgb[rgb_idx + 2] = static_cast<unsigned char>(fminf(fmaxf(b * 255.0f, 0.0f), 255.0f));
}

// BGR to RGB swap kernel
__global__ void bgr_to_rgb_kernel(
    unsigned char* __restrict__ frame,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // Swap B and R channels
    unsigned char tmp = frame[idx + 0];
    frame[idx + 0] = frame[idx + 2];
    frame[idx + 2] = tmp;
}

// Host functions
void rgb_to_yiq(const unsigned char* d_rgb, float* d_yiq,
                int width, int height, cudaStream_t stream) {
    dim3 grid = calc_grid_dim(width, height);
    dim3 block = block_dim();

    rgb_to_yiq_kernel<<<grid, block, 0, stream>>>(d_rgb, d_yiq, width, height);
}

void yiq_to_rgb(const float* d_yiq, unsigned char* d_rgb,
                int width, int height, cudaStream_t stream) {
    dim3 grid = calc_grid_dim(width, height);
    dim3 block = block_dim();

    yiq_to_rgb_kernel<<<grid, block, 0, stream>>>(d_yiq, d_rgb, width, height);
}

void bgr_to_rgb(unsigned char* d_frame, int width, int height,
                cudaStream_t stream) {
    dim3 grid = calc_grid_dim(width, height);
    dim3 block = block_dim();

    bgr_to_rgb_kernel<<<grid, block, 0, stream>>>(d_frame, width, height);
}

} // namespace cuda
} // namespace vhs
