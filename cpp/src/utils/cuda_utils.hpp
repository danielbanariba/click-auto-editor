#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <stdexcept>

// Error checking macro (defined outside namespace - macros don't respect namespaces)
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            throw std::runtime_error(cudaGetErrorString(err));              \
        }                                                                   \
    } while(0)

namespace vhs {
namespace cuda {

// Default block dimensions for 2D kernels
constexpr int BLOCK_DIM_X = 16;
constexpr int BLOCK_DIM_Y = 16;

// Calculate grid dimensions
inline dim3 calc_grid_dim(int width, int height) {
    return dim3(
        (width + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
    );
}

inline dim3 block_dim() {
    return dim3(BLOCK_DIM_X, BLOCK_DIM_Y);
}

// RGB pixel structure (matches OpenCV BGR order for compatibility)
struct __align__(4) Pixel {
    unsigned char b;
    unsigned char g;
    unsigned char r;
    unsigned char a;  // padding for alignment
};

// Float3 for intermediate calculations
struct __align__(16) PixelF {
    float b;
    float g;
    float r;
    float a;
};

// YIQ color space
struct __align__(16) YIQ {
    float y;
    float i;
    float q;
    float pad;
};

// Initialize CUDA device
inline void init_device(int device_id = 0) {
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("[CUDA] Device: %s\n", prop.name);
    printf("[CUDA] Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("[CUDA] Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

// Synchronize and check for errors
inline void sync_check() {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace vhs
