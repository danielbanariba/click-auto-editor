/**
 * VHS Effect Chain Implementation
 *
 * One pass of the Arsenal VHS look per frame (see arsenal_vhs.cu), then the optional VHS noise
 * overlay. No global random generator: the look is a function of the frame index and a fixed seed.
 */

#include "vhs_effect_chain.hpp"
#include "utils/cuda_utils.hpp"
#include <cmath>
#include <cstdint>

namespace vhs {
namespace effects {

namespace {

/** The frame index of a time in seconds at FPS (0 for a time that is not a usable number). */
std::int64_t frame_index_of(float frame_time) {
    const double frames = static_cast<double>(frame_time) * static_cast<double>(::vhs::FPS);
    if (!std::isfinite(frames) || frames < 0.0 || frames > 9.0e15) {
        return 0;
    }
    return static_cast<std::int64_t>(std::llround(frames));
}

} // namespace

VHSEffectChain::VHSEffectChain(int width, int height, const ::vhs::VHSParams& params)
    : width_(width)
    , height_(height)
    , params_(params)
    , d_overlay_scratch_(nullptr)
    , initialized_(false)
{
}

VHSEffectChain::~VHSEffectChain() {
    cleanup();
}

void VHSEffectChain::init() {
    if (initialized_) return;
    if (width_ <= 0 || height_ <= 0) return;  // nothing to draw on

    engine_.reset(new arsenal::Engine(width_, height_));
    engine_->init();
    initialized_ = true;
}

void VHSEffectChain::cleanup() {
    if (d_overlay_scratch_) {
        cudaFree(d_overlay_scratch_);
        d_overlay_scratch_ = nullptr;
    }
    if (engine_) {
        engine_->cleanup();
        engine_.reset();
    }
    initialized_ = false;
}

void VHSEffectChain::process_frame(
    const unsigned char* d_input,
    unsigned char* d_output,
    const unsigned char* d_vhs_overlay,
    float frame_time,
    cudaStream_t stream
) {
    if (width_ <= 0 || height_ <= 0 || d_input == nullptr || d_output == nullptr) {
        return;
    }
    if (!initialized_) {
        init();
    }

    const arsenal::FrameInfo frame = arsenal::frame_info(frame_index_of(frame_time), static_cast<double>(::vhs::FPS));
    const arsenal::Controls controls;  // the calibrated look

    if (d_vhs_overlay == nullptr) {
        engine_->render(d_input, d_output, controls, frame, stream);
        return;
    }

    if (d_overlay_scratch_ == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_overlay_scratch_, static_cast<size_t>(width_) * height_ * 3));
    }
    engine_->render(d_input, d_overlay_scratch_, controls, frame, stream);
    apply_vhs_overlay_blend(d_overlay_scratch_, d_vhs_overlay, d_output, width_, height_,
                            VHS_OVERLAY_OPACITY, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace effects
} // namespace vhs
