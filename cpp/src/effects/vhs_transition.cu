/**
 * VHS Transition Effect
 *
 * Slide-down transition with VHS noise/distortion.
 */

#include "vhs_transition.hpp"
#include "config/render_config.hpp"
#include "utils/cuda_utils.hpp"

#include <cmath>

namespace vhs {
namespace effects {

__device__ __forceinline__ unsigned int hash_u32(unsigned int x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbU;
    x ^= x >> 11;
    x *= 0xac4c1b51U;
    x ^= x >> 15;
    x *= 0x31848babU;
    x ^= x >> 14;
    return x;
}

__device__ __forceinline__ float rand01(unsigned int seed) {
    return (hash_u32(seed) & 0x00FFFFFFU) / 16777215.0f;
}

__device__ __forceinline__ int wrap_int(int v, int max) {
    if (max <= 0) return 0;
    v %= max;
    if (v < 0) v += max;
    return v;
}

__global__ void vhs_transition_kernel(
    const unsigned char* __restrict__ intro,
    const unsigned char* __restrict__ main_frame,
    unsigned char* __restrict__ output,
    int width,
    int height,
    float progress,
    int frame_index
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float clamped = fminf(fmaxf(progress, 0.0f), 1.0f);
    float strength = 1.0f - clamped;

    float slide_y = (1.0f - clamped) * static_cast<float>(height);

    float jitter_max_x = VHS_TRANSITION_FRAME_JITTER * static_cast<float>(width);
    float jitter_max_y = VHS_TRANSITION_FRAME_JITTER * static_cast<float>(height);

    unsigned int frame_seed = static_cast<unsigned int>(frame_index) * 747796405U;
    float jitter_x = (rand01(frame_seed ^ 0x68bc21U) * 2.0f - 1.0f) * jitter_max_x * strength;
    float jitter_y = (rand01(frame_seed ^ 0x02e5beU) * 2.0f - 1.0f) * jitter_max_y * strength;

    float distort_amp = (0.02f + 0.05f * VHS_TRANSITION_TAPE_DISTORTION) *
                        static_cast<float>(width) * strength;
    float wrinkle_amp = VHS_TRANSITION_WRINKLE_SIZE * 0.5f *
                        static_cast<float>(height) * strength;

    int band_height = static_cast<int>(fmaxf(2.0f, static_cast<float>(height) * VHS_TRANSITION_WRINKLE_SIZE));
    int band = (band_height > 0) ? (y / band_height) : 0;
    float band_rand = rand01(frame_seed ^ (band * 0x9e3779b9U));
    int band_offset = static_cast<int>((band_rand * 2.0f - 1.0f) * distort_amp);

    float line_rand = rand01(frame_seed ^ (y * 0x27d4eb2dU));
    int line_offset = static_cast<int>((line_rand * 2.0f - 1.0f) * distort_amp * 0.4f);

    float wrinkle_phase = static_cast<float>(y) * 0.12f + frame_index * 0.35f;
    int wrinkle_offset = static_cast<int>(sinf(wrinkle_phase) * wrinkle_amp);

    int offset_x = static_cast<int>(jitter_x) + band_offset + line_offset + wrinkle_offset;

    int src_x = wrap_int(x + offset_x, width);

    int main_y = static_cast<int>(static_cast<float>(y) - slide_y + jitter_y);
    bool use_main = (main_y >= 0 && main_y < height);
    int src_y = use_main ? main_y : y;
    int dst_idx = (y * width + x) * 3;

    const unsigned char* src = use_main ? main_frame : intro;

    int chroma_offset = static_cast<int>(VHS_TRANSITION_CHROMA_OFFSET * strength);
    int x_b = wrap_int(src_x - chroma_offset, width);
    int x_g = src_x;
    int x_r = wrap_int(src_x + chroma_offset, width);

    int idx_b = (src_y * width + x_b) * 3;
    int idx_g = (src_y * width + x_g) * 3;
    int idx_r = (src_y * width + x_r) * 3;

    float b = static_cast<float>(src[idx_b + 0]);
    float g = static_cast<float>(src[idx_g + 1]);
    float r = static_cast<float>(src[idx_r + 2]);

    if (VHS_TRANSITION_CHROMA_BLUR > 0.0f) {
        int x_b2 = wrap_int(x_b + 1, width);
        int x_g2 = wrap_int(x_g + 1, width);
        int x_r2 = wrap_int(x_r + 1, width);
        int idx_b2 = (src_y * width + x_b2) * 3;
        int idx_g2 = (src_y * width + x_g2) * 3;
        int idx_r2 = (src_y * width + x_r2) * 3;
        b = b * (1.0f - VHS_TRANSITION_CHROMA_BLUR) + static_cast<float>(src[idx_b2 + 0]) * VHS_TRANSITION_CHROMA_BLUR;
        g = g * (1.0f - VHS_TRANSITION_CHROMA_BLUR) + static_cast<float>(src[idx_g2 + 1]) * VHS_TRANSITION_CHROMA_BLUR;
        r = r * (1.0f - VHS_TRANSITION_CHROMA_BLUR) + static_cast<float>(src[idx_r2 + 2]) * VHS_TRANSITION_CHROMA_BLUR;
    }

    float base_noise = 20.0f + 60.0f * VHS_TRANSITION_TAPE_NOISE;
    float noise_strength = base_noise * (0.4f + 0.6f * strength);
    float flicker = (rand01(frame_seed ^ 0xa5a5a5U) * 2.0f - 1.0f);
    noise_strength *= (1.0f + VHS_TRANSITION_RANDOM_NOISE * flicker);

    unsigned int nseed = (frame_seed ^ (y * 73856093U) ^ (x * 19349663U));
    float n_b = (rand01(nseed ^ 0x111111U) * 2.0f - 1.0f) * noise_strength;
    float n_g = (rand01(nseed ^ 0x222222U) * 2.0f - 1.0f) * noise_strength;
    float n_r = (rand01(nseed ^ 0x333333U) * 2.0f - 1.0f) * noise_strength;

    float line_glitch = (rand01(frame_seed ^ (y * 2654435761U)) < (0.003f * VHS_TRANSITION_TAPE_DISTORTION)) ? 60.0f : 0.0f;

    b = fminf(fmaxf(b + n_b + line_glitch, 0.0f), 255.0f);
    g = fminf(fmaxf(g + n_g + line_glitch, 0.0f), 255.0f);
    r = fminf(fmaxf(r + n_r + line_glitch, 0.0f), 255.0f);

    output[dst_idx + 0] = static_cast<unsigned char>(b);
    output[dst_idx + 1] = static_cast<unsigned char>(g);
    output[dst_idx + 2] = static_cast<unsigned char>(r);
}

void apply_vhs_transition(
    const unsigned char* d_intro,
    const unsigned char* d_main,
    unsigned char* d_output,
    int width,
    int height,
    float progress,
    int frame_index,
    cudaStream_t stream
) {
    dim3 grid = cuda::calc_grid_dim(width, height);
    dim3 block = cuda::block_dim();

    vhs_transition_kernel<<<grid, block, 0, stream>>>(
        d_intro, d_main, d_output,
        width, height,
        progress, frame_index
    );
}

} // namespace effects
} // namespace vhs
