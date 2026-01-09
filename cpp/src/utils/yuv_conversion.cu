/**
 * BGR to YUV420 conversion on GPU.
 */

#include "yuv_conversion.hpp"
#include "cuda_utils.hpp"
#include <cmath>

namespace vhs {
namespace cuda {

__global__ void bgr_to_y_kernel(
    const unsigned char* bgr,
    unsigned char* y_plane,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float b = static_cast<float>(bgr[idx + 0]);
    float g = static_cast<float>(bgr[idx + 1]);
    float r = static_cast<float>(bgr[idx + 2]);

    float y_val = 0.299f * r + 0.587f * g + 0.114f * b;
    y_val = fminf(fmaxf(y_val, 0.0f), 255.0f);
    y_plane[y * width + x] = static_cast<unsigned char>(y_val);
}

__global__ void bgr_to_uv_kernel(
    const unsigned char* bgr,
    unsigned char* u_plane,
    unsigned char* v_plane,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int uv_width = width / 2;
    int uv_height = height / 2;

    if (x >= uv_width || y >= uv_height) return;

    int src_x = x * 2;
    int src_y = y * 2;

    float r_sum = 0.0f;
    float g_sum = 0.0f;
    float b_sum = 0.0f;

    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int px = src_x + dx;
            int py = src_y + dy;
            int idx = (py * width + px) * 3;
            b_sum += static_cast<float>(bgr[idx + 0]);
            g_sum += static_cast<float>(bgr[idx + 1]);
            r_sum += static_cast<float>(bgr[idx + 2]);
        }
    }

    float r = r_sum * 0.25f;
    float g = g_sum * 0.25f;
    float b = b_sum * 0.25f;

    float u = 128.0f - 0.169f * r - 0.331f * g + 0.5f * b;
    float v = 128.0f + 0.5f * r - 0.419f * g - 0.081f * b;

    u = fminf(fmaxf(u, 0.0f), 255.0f);
    v = fminf(fmaxf(v, 0.0f), 255.0f);

    int uv_idx = y * uv_width + x;
    u_plane[uv_idx] = static_cast<unsigned char>(u);
    v_plane[uv_idx] = static_cast<unsigned char>(v);
}

__global__ void bgr_to_uv_interleaved_kernel(
    const unsigned char* bgr,
    unsigned char* uv_plane,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int uv_width = width / 2;
    int uv_height = height / 2;

    if (x >= uv_width || y >= uv_height) return;

    int src_x = x * 2;
    int src_y = y * 2;

    float r_sum = 0.0f;
    float g_sum = 0.0f;
    float b_sum = 0.0f;

    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int px = src_x + dx;
            int py = src_y + dy;
            int idx = (py * width + px) * 3;
            b_sum += static_cast<float>(bgr[idx + 0]);
            g_sum += static_cast<float>(bgr[idx + 1]);
            r_sum += static_cast<float>(bgr[idx + 2]);
        }
    }

    float r = r_sum * 0.25f;
    float g = g_sum * 0.25f;
    float b = b_sum * 0.25f;

    float u = 128.0f - 0.169f * r - 0.331f * g + 0.5f * b;
    float v = 128.0f + 0.5f * r - 0.419f * g - 0.081f * b;

    u = fminf(fmaxf(u, 0.0f), 255.0f);
    v = fminf(fmaxf(v, 0.0f), 255.0f);

    int uv_idx = (y * uv_width + x) * 2;
    uv_plane[uv_idx] = static_cast<unsigned char>(u);
    uv_plane[uv_idx + 1] = static_cast<unsigned char>(v);
}

void bgr_to_yuv420(
    const unsigned char* d_bgr,
    unsigned char* d_y,
    unsigned char* d_u,
    unsigned char* d_v,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 block = cuda::block_dim();
    dim3 grid_y(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    dim3 grid_uv(
        (width / 2 + block.x - 1) / block.x,
        (height / 2 + block.y - 1) / block.y
    );

    bgr_to_y_kernel<<<grid_y, block, 0, stream>>>(d_bgr, d_y, width, height);
    bgr_to_uv_kernel<<<grid_uv, block, 0, stream>>>(d_bgr, d_u, d_v, width, height);
}

void bgr_to_nv12(
    const unsigned char* d_bgr,
    unsigned char* d_y,
    unsigned char* d_uv,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 block = cuda::block_dim();
    dim3 grid_y(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    dim3 grid_uv(
        (width / 2 + block.x - 1) / block.x,
        (height / 2 + block.y - 1) / block.y
    );

    bgr_to_y_kernel<<<grid_y, block, 0, stream>>>(d_bgr, d_y, width, height);
    bgr_to_uv_interleaved_kernel<<<grid_uv, block, 0, stream>>>(
        d_bgr, d_uv, width, height
    );
}

__device__ __forceinline__ unsigned char clamp_u8(int value) {
    return static_cast<unsigned char>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

__device__ __forceinline__ void yuv_to_bgr(int y, int u, int v,
                                           unsigned char& b,
                                           unsigned char& g,
                                           unsigned char& r) {
    int c = y - 16;
    int d = u - 128;
    int e = v - 128;

    int r_val = (298 * c + 409 * e + 128) >> 8;
    int g_val = (298 * c - 100 * d - 208 * e + 128) >> 8;
    int b_val = (298 * c + 516 * d + 128) >> 8;

    b = clamp_u8(b_val);
    g = clamp_u8(g_val);
    r = clamp_u8(r_val);
}

__global__ void nv12_to_bgr_kernel(
    const unsigned char* y_plane,
    const unsigned char* uv_plane,
    unsigned char* bgr,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int y_idx = y * width + x;
    int uv_idx = (y / 2) * width + (x / 2) * 2;

    int Y = y_plane[y_idx];
    int U = uv_plane[uv_idx];
    int V = uv_plane[uv_idx + 1];

    unsigned char b, g, r;
    yuv_to_bgr(Y, U, V, b, g, r);

    int out_idx = (y * width + x) * 3;
    bgr[out_idx + 0] = b;
    bgr[out_idx + 1] = g;
    bgr[out_idx + 2] = r;
}

__global__ void yuv420_to_bgr_kernel(
    const unsigned char* y_plane,
    const unsigned char* u_plane,
    const unsigned char* v_plane,
    unsigned char* bgr,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int y_idx = y * width + x;
    int uv_idx = (y / 2) * (width / 2) + (x / 2);

    int Y = y_plane[y_idx];
    int U = u_plane[uv_idx];
    int V = v_plane[uv_idx];

    unsigned char b, g, r;
    yuv_to_bgr(Y, U, V, b, g, r);

    int out_idx = (y * width + x) * 3;
    bgr[out_idx + 0] = b;
    bgr[out_idx + 1] = g;
    bgr[out_idx + 2] = r;
}

__global__ void yuv444_to_bgr_kernel(
    const unsigned char* y_plane,
    const unsigned char* u_plane,
    const unsigned char* v_plane,
    unsigned char* bgr,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int Y = y_plane[idx];
    int U = u_plane[idx];
    int V = v_plane[idx];

    unsigned char b, g, r;
    yuv_to_bgr(Y, U, V, b, g, r);

    int out_idx = idx * 3;
    bgr[out_idx + 0] = b;
    bgr[out_idx + 1] = g;
    bgr[out_idx + 2] = r;
}

void nv12_to_bgr(
    const unsigned char* d_y,
    const unsigned char* d_uv,
    unsigned char* d_bgr,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    nv12_to_bgr_kernel<<<grid, block, 0, stream>>>(
        d_y, d_uv, d_bgr, width, height
    );
}

void yuv420_to_bgr(
    const unsigned char* d_y,
    const unsigned char* d_u,
    const unsigned char* d_v,
    unsigned char* d_bgr,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    yuv420_to_bgr_kernel<<<grid, block, 0, stream>>>(
        d_y, d_u, d_v, d_bgr, width, height
    );
}

void yuv444_to_bgr(
    const unsigned char* d_y,
    const unsigned char* d_u,
    const unsigned char* d_v,
    unsigned char* d_bgr,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    yuv444_to_bgr_kernel<<<grid, block, 0, stream>>>(
        d_y, d_u, d_v, d_bgr, width, height
    );
}

} // namespace cuda
} // namespace vhs
