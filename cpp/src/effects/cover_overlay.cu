/**
 * Cover Overlay - copy cover image onto center of frame.
 */

#include "cover_overlay.hpp"
#include "utils/cuda_utils.hpp"

namespace vhs {
namespace effects {

__global__ void cover_overlay_kernel(
    unsigned char* frame,
    const unsigned char* cover_bgra,
    int frame_width,
    int frame_height,
    int cover_width,
    int cover_height,
    int offset_x,
    int offset_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cover_width || y >= cover_height) return;

    int dst_x = offset_x + x;
    int dst_y = offset_y + y;

    if (dst_x < 0 || dst_y < 0 || dst_x >= frame_width || dst_y >= frame_height) {
        return;
    }

    int dst_idx = (dst_y * frame_width + dst_x) * 3;
    int src_idx = (y * cover_width + x) * 4;
    float alpha = static_cast<float>(cover_bgra[src_idx + 3]) / 255.0f;
    if (alpha <= 0.0f) {
        return;
    }

    float inv_alpha = 1.0f - alpha;
    frame[dst_idx + 0] = static_cast<unsigned char>(
        cover_bgra[src_idx + 0] * alpha + frame[dst_idx + 0] * inv_alpha
    );
    frame[dst_idx + 1] = static_cast<unsigned char>(
        cover_bgra[src_idx + 1] * alpha + frame[dst_idx + 1] * inv_alpha
    );
    frame[dst_idx + 2] = static_cast<unsigned char>(
        cover_bgra[src_idx + 2] * alpha + frame[dst_idx + 2] * inv_alpha
    );
}

void apply_cover_overlay(
    unsigned char* d_frame,
    const unsigned char* d_cover_bgra,
    int frame_width,
    int frame_height,
    int cover_width,
    int cover_height,
    cudaStream_t stream
) {
    int offset_x = (frame_width - cover_width) / 2;
    int offset_y = (frame_height - cover_height) / 2;

    dim3 block = cuda::block_dim();
    dim3 grid(
        (cover_width + block.x - 1) / block.x,
        (cover_height + block.y - 1) / block.y
    );

    cover_overlay_kernel<<<grid, block, 0, stream>>>(
        d_frame, d_cover_bgra,
        frame_width, frame_height,
        cover_width, cover_height,
        offset_x, offset_y
    );
}

} // namespace effects
} // namespace vhs
