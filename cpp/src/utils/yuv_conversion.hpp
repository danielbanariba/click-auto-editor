#pragma once

#include <cuda_runtime.h>

namespace vhs {
namespace cuda {

void bgr_to_yuv420(
    const unsigned char* d_bgr,
    unsigned char* d_y,
    unsigned char* d_u,
    unsigned char* d_v,
    int width,
    int height,
    cudaStream_t stream = 0
);

void bgr_to_nv12(
    const unsigned char* d_bgr,
    unsigned char* d_y,
    unsigned char* d_uv,
    int width,
    int height,
    cudaStream_t stream = 0
);

void nv12_to_bgr(
    const unsigned char* d_y,
    const unsigned char* d_uv,
    unsigned char* d_bgr,
    int width,
    int height,
    cudaStream_t stream = 0
);

void yuv420_to_bgr(
    const unsigned char* d_y,
    const unsigned char* d_u,
    const unsigned char* d_v,
    unsigned char* d_bgr,
    int width,
    int height,
    cudaStream_t stream = 0
);

void yuv444_to_bgr(
    const unsigned char* d_y,
    const unsigned char* d_u,
    const unsigned char* d_v,
    unsigned char* d_bgr,
    int width,
    int height,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace vhs
