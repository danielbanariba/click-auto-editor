#pragma once

#include "config/render_config.hpp"
#include "effects/arsenal_vhs.hpp"
#include <cuda_runtime.h>
#include <memory>

namespace vhs {
namespace effects {

// Soft-light blend of a VHS noise frame over a frame (optional, off unless an overlay is given)
void apply_vhs_overlay_blend(const unsigned char* d_base, const unsigned char* d_overlay,
                             unsigned char* d_output, int width, int height,
                             float opacity, cudaStream_t stream = 0);

/**
 * VHS Effect Chain
 *
 * Draws the Arsenal VHS look (see arsenal_vhs.hpp) on a packed 8-bit BGR frame. It runs on the
 * background only: the sharp cover and tracklist overlays are composited after it.
 *
 * The output depends only on the frame's index and the look's fixed seed, never on how many frames
 * were drawn before or on a global random generator, so a frame renders to the same bytes every time.
 */
class VHSEffectChain {
public:
    VHSEffectChain(int width, int height, const ::vhs::VHSParams& params);
    ~VHSEffectChain();

    // Draw the look onto one frame; the result is in d_output (d_input may be the same buffer).
    // frame_time is the frame's time in seconds at FPS frames per second.
    // Throws std::runtime_error when a CUDA call fails.
    void process_frame(
        const unsigned char* d_input,
        unsigned char* d_output,
        const unsigned char* d_vhs_overlay,  // VHS noise frame (can be nullptr)
        float frame_time,                     // Current time in seconds
        cudaStream_t stream = 0
    );

    // Initialize resources
    void init();

    // Cleanup resources
    void cleanup();

private:
    int width_;
    int height_;
    ::vhs::VHSParams params_;

    std::unique_ptr<arsenal::Engine> engine_;
    unsigned char* d_overlay_scratch_;  // only allocated when an overlay is given

    bool initialized_;
};

} // namespace effects
} // namespace vhs
