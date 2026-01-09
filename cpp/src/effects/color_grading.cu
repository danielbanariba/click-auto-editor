/**
 * VHS Color Grading Effect
 *
 * Applies characteristic VHS color look:
 * 1. Elevated blacks (VHS never has pure black)
 * 2. Reduced contrast
 * 3. Slight desaturation
 * 4. Warm/yellow tint
 */

#include "utils/cuda_utils.hpp"
#include "config/render_config.hpp"

namespace vhs {
namespace effects {

__global__ void color_grading_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height,
    float black_lift,      // 10-20
    float contrast_factor, // 0.7-0.9
    float desaturation,    // 0.15-0.30
    float blue_factor,     // 0.95
    float red_factor       // 1.05
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;

    // Read BGR
    float b = static_cast<float>(input[idx + 0]);
    float g = static_cast<float>(input[idx + 1]);
    float r = static_cast<float>(input[idx + 2]);

    // 1. Lift blacks
    b += black_lift;
    g += black_lift;
    r += black_lift;

    // 2. Reduce contrast
    constexpr float mid_gray = 127.5f;
    b = mid_gray + (b - mid_gray) * contrast_factor;
    g = mid_gray + (g - mid_gray) * contrast_factor;
    r = mid_gray + (r - mid_gray) * contrast_factor;

    // 3. Desaturate (blend with grayscale)
    // NTSC grayscale weights
    float gray = 0.114f * b + 0.587f * g + 0.299f * r;
    b = b * (1.0f - desaturation) + gray * desaturation;
    g = g * (1.0f - desaturation) + gray * desaturation;
    r = r * (1.0f - desaturation) + gray * desaturation;

    // 4. Warm tint
    b *= blue_factor;
    r *= red_factor;

    // Clamp and write
    output[idx + 0] = static_cast<unsigned char>(fminf(fmaxf(b, 0.0f), 255.0f));
    output[idx + 1] = static_cast<unsigned char>(fminf(fmaxf(g, 0.0f), 255.0f));
    output[idx + 2] = static_cast<unsigned char>(fminf(fmaxf(r, 0.0f), 255.0f));
}

void apply_color_grading(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    float black_lift,
    float contrast_factor,
    float desaturation,
    float blue_factor,
    float red_factor,
    cudaStream_t stream
) {
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    color_grading_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, width, height,
        black_lift, contrast_factor, desaturation,
        blue_factor, red_factor
    );
}

} // namespace effects
} // namespace vhs
