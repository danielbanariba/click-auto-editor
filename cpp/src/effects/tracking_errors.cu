/**
 * Tracking Errors Effect
 *
 * Simulates misalignment between video heads and tape tracks.
 * Creates horizontal bands that are randomly displaced.
 */

#include "vhs_effect_chain.hpp"  // For TrackingBand struct
#include "utils/cuda_utils.hpp"
#include <cstdint>

namespace vhs {
namespace effects {

// Maximum number of tracking bands
constexpr int MAX_TRACKING_BANDS = 8;

__constant__ TrackingBand c_tracking_bands[MAX_TRACKING_BANDS];

__global__ void tracking_errors_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height,
    int num_bands
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int offset = 0;

    // Find active band for this row
    for (int b = 0; b < num_bands; b++) {
        if (c_tracking_bands[b].active &&
            y >= c_tracking_bands[b].y_start &&
            y < c_tracking_bands[b].y_end) {
            offset = c_tracking_bands[b].offset;
            break;
        }
    }

    // Calculate source X with wrap
    int64_t src_x = static_cast<int64_t>(x) - static_cast<int64_t>(offset);
    src_x %= width;
    if (src_x < 0) {
        src_x += width;
    }

    int dst_idx = (y * width + x) * 3;
    int src_idx = (y * width + src_x) * 3;

    output[dst_idx + 0] = input[src_idx + 0];
    output[dst_idx + 1] = input[src_idx + 1];
    output[dst_idx + 2] = input[src_idx + 2];
}

void apply_tracking_errors(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    const TrackingBand* bands,
    int num_bands,
    cudaStream_t stream
) {
    // Copy band parameters to constant memory
    int safe_num_bands = num_bands;
    if (safe_num_bands < 0) {
        safe_num_bands = 0;
    } else if (safe_num_bands > MAX_TRACKING_BANDS) {
        safe_num_bands = MAX_TRACKING_BANDS;
    }

    if (safe_num_bands == 0) {
        size_t frame_size = static_cast<size_t>(width) * height * 3;
        CUDA_CHECK(cudaMemcpyAsync(d_output, d_input, frame_size,
                                   cudaMemcpyDeviceToDevice, stream));
        return;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(
        c_tracking_bands, bands,
        safe_num_bands * sizeof(TrackingBand)
    ));

    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    tracking_errors_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height, safe_num_bands
    );
    CUDA_CHECK(cudaGetLastError());
}

// Helper to generate random tracking bands (call from host)
void generate_tracking_bands(
    TrackingBand* bands,
    int& num_bands,
    int height,
    float probability,  // 0.3 * intensity
    float sigma         // 5 * intensity
) {
    // Random number of bands (2-5)
    num_bands = 2 + (rand() % 4);
    if (num_bands > MAX_TRACKING_BANDS) {
        num_bands = MAX_TRACKING_BANDS;
    }

    int band_height = height / num_bands;

    for (int b = 0; b < num_bands; b++) {
        bands[b].y_start = b * band_height;
        bands[b].y_end = (b == num_bands - 1) ? height : (b + 1) * band_height;

        // Random activation (30% probability per band)
        bands[b].active = (static_cast<float>(rand()) / RAND_MAX) < probability;

        if (bands[b].active && sigma > 0.0f) {
            // Gaussian offset with guard against log(0)
            float u1 = static_cast<float>(rand()) / RAND_MAX;
            float u2 = static_cast<float>(rand()) / RAND_MAX;
            if (u1 < 1.0e-6f) {
                u1 = 1.0e-6f;
            }
            float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
            int offset = static_cast<int>(z * sigma);
            int max_offset = static_cast<int>(sigma * 6.0f);
            if (max_offset < 1) {
                max_offset = 1;
            }
            if (offset > max_offset) {
                offset = max_offset;
            } else if (offset < -max_offset) {
                offset = -max_offset;
            }
            bands[b].offset = offset;
        } else {
            bands[b].offset = 0;
        }
    }
}

} // namespace effects
} // namespace vhs
