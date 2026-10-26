/**
 * Arsenal VHS - the transition from the held last frame of the intro to the main part.
 *
 * Ported from the Arsenal VHS effect of a FilmCraft build (MIT OR Apache-2.0).
 */

#pragma once

#include <cuda_runtime.h>

namespace vhs {
namespace effects {

/**
 * Draws the transition at `progress` (0..1) into `d_output`: `d_intro` (A, the held last intro frame) rolls
 * out downward while `d_main` (B, the main part) rolls in from the top, with the tape's damage over the
 * seam. All three are packed 8-bit BGR frames of width x height. At progress 0 the result is A and at 1 it
 * is B; a frame depends only on its progress, so a repeated frame is identical. `frame_index` is not used.
 * The device memory it keeps between frames is let go when the progress reaches 1.
 * Throws std::runtime_error when a CUDA call fails.
 */
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
