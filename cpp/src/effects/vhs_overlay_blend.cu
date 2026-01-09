/**
 * VHS Overlay Blend Effect
 *
 * Blends VHS noise video over the processed frame using
 * softlight blend mode with configurable opacity.
 */

#include "utils/cuda_utils.hpp"

namespace vhs {
namespace effects {

// Softlight blend function (per channel)
__device__ __forceinline__ float softlight_blend(float base, float overlay) {
    // Normalize to 0-1
    float a = base / 255.0f;
    float b = overlay / 255.0f;

    float result;
    if (b < 0.5f) {
        result = 2.0f * a * b + a * a * (1.0f - 2.0f * b);
    } else {
        result = 2.0f * a * (1.0f - b) + sqrtf(a) * (2.0f * b - 1.0f);
    }

    return result * 255.0f;
}

__global__ void vhs_overlay_blend_kernel(
    const unsigned char* __restrict__ base,
    const unsigned char* __restrict__ overlay,
    unsigned char* __restrict__ output,
    int width, int height,
    float opacity  // 0.35 default
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // Read base and overlay pixels
    float base_b = static_cast<float>(base[idx + 0]);
    float base_g = static_cast<float>(base[idx + 1]);
    float base_r = static_cast<float>(base[idx + 2]);

    float over_b = static_cast<float>(overlay[idx + 0]);
    float over_g = static_cast<float>(overlay[idx + 1]);
    float over_r = static_cast<float>(overlay[idx + 2]);

    // Apply softlight blend
    float blend_b = softlight_blend(base_b, over_b);
    float blend_g = softlight_blend(base_g, over_g);
    float blend_r = softlight_blend(base_r, over_r);

    // Apply opacity
    float inv_opacity = 1.0f - opacity;
    float out_b = base_b * inv_opacity + blend_b * opacity;
    float out_g = base_g * inv_opacity + blend_g * opacity;
    float out_r = base_r * inv_opacity + blend_r * opacity;

    // Write output
    output[idx + 0] = static_cast<unsigned char>(fminf(fmaxf(out_b, 0.0f), 255.0f));
    output[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(out_g, 0.0f), 255.0f));
    output[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(out_r, 0.0f), 255.0f));
}

void apply_vhs_overlay_blend(
    const unsigned char* d_base,
    const unsigned char* d_overlay,
    unsigned char* d_output,
    int width, int height,
    float opacity,
    cudaStream_t stream
) {
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    vhs_overlay_blend_kernel<<<grid, block, 0, stream>>>(
        d_base, d_overlay, d_output, width, height, opacity
    );
}

} // namespace effects
} // namespace vhs
