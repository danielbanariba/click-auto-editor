#pragma once

#include <cuda_runtime.h>

namespace vhs {
namespace effects {

void apply_vhs_transition(
    const unsigned char* d_intro,
    const unsigned char* d_main,
    unsigned char* d_output,
    int width,
    int height,
    float progress,
    int frame_index,
    cudaStream_t stream = 0
);

} // namespace effects
} // namespace vhs
