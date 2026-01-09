#pragma once

#include <cuda_runtime.h>

namespace vhs {
namespace effects {

void apply_cover_overlay(
    unsigned char* d_frame,
    const unsigned char* d_cover_bgra,
    int frame_width,
    int frame_height,
    int cover_width,
    int cover_height,
    cudaStream_t stream = 0
);

} // namespace effects
} // namespace vhs
