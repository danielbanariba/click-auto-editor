/**
 * VHS Effect Chain Implementation
 *
 * Orchestrates all 10 VHS effects in the correct order.
 * Uses ping-pong buffering for efficient in-place processing.
 */

#include "vhs_effect_chain.hpp"
#include "utils/cuda_utils.hpp"
#include <cstdlib>
#include <cmath>

namespace vhs {
namespace effects {

static bool env_flag_enabled(const char* name) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0' || env[0] == '0') {
        return false;
    }
    return true;
}

static bool safe_mode_enabled() {
    static int cached = -1;
    if (cached == -1) {
        cached = env_flag_enabled("VHS_SAFE_MODE") ? 1 : 0;
    }
    return cached == 1;
}

static bool tracking_errors_disabled() {
    static int cached = -1;
    if (cached == -1) {
        cached = (safe_mode_enabled() || env_flag_enabled("VHS_DISABLE_TRACKING_ERRORS")) ? 1 : 0;
    }
    return cached == 1;
}

static bool color_bleeding_disabled() {
    static int cached = -1;
    if (cached == -1) {
        cached = (safe_mode_enabled() || env_flag_enabled("VHS_DISABLE_COLOR_BLEEDING")) ? 1 : 0;
    }
    return cached == 1;
}

static bool noise_disabled() {
    static int cached = -1;
    if (cached == -1) {
        cached = (safe_mode_enabled() || env_flag_enabled("VHS_DISABLE_NOISE")) ? 1 : 0;
    }
    return cached == 1;
}

static bool overlay_disabled() {
    static int cached = -1;
    if (cached == -1) {
        cached = (safe_mode_enabled() || env_flag_enabled("VHS_DISABLE_OVERLAY")) ? 1 : 0;
    }
    return cached == 1;
}

VHSEffectChain::VHSEffectChain(int width, int height, const ::vhs::VHSParams& params)
    : width_(width)
    , height_(height)
    , params_(params)
    , d_buffer_a_(nullptr)
    , d_buffer_b_(nullptr)
    , last_jitter_time_(0.0f)
    , current_jitter_x_(0)
    , current_jitter_y_(0)
    , initialized_(false)
{
}

VHSEffectChain::~VHSEffectChain() {
    cleanup();
}

void VHSEffectChain::init() {
    if (initialized_) return;

    // Allocate ping-pong buffers
    size_t frame_size = width_ * height_ * 3 * sizeof(unsigned char);
    CUDA_CHECK(cudaMalloc(&d_buffer_a_, frame_size));
    CUDA_CHECK(cudaMalloc(&d_buffer_b_, frame_size));

    // Initialize effect-specific resources
    if (!color_bleeding_disabled()) {
        init_color_bleeding_buffers(width_, height_);
    }
    if (!noise_disabled()) {
        init_noise_rng(width_, height_, 42);
    }

    initialized_ = true;
}

void VHSEffectChain::cleanup() {
    if (!initialized_) return;

    if (d_buffer_a_) {
        cudaFree(d_buffer_a_);
        d_buffer_a_ = nullptr;
    }
    if (d_buffer_b_) {
        cudaFree(d_buffer_b_);
        d_buffer_b_ = nullptr;
    }

    cleanup_color_bleeding_buffers();
    cleanup_noise_rng();

    initialized_ = false;
}

void VHSEffectChain::process_frame(
    const unsigned char* d_input,
    unsigned char* d_output,
    const unsigned char* d_vhs_overlay,
    float frame_time,
    cudaStream_t stream
) {
    if (!initialized_) {
        init();
    }

    // Ping-pong buffer pointers
    unsigned char* src = d_buffer_a_;
    unsigned char* dst = d_buffer_b_;

    // Copy input to buffer A
    size_t frame_size = width_ * height_ * 3;
    CUDA_CHECK(cudaMemcpyAsync(d_buffer_a_, d_input, frame_size,
                                      cudaMemcpyDeviceToDevice, stream));

    // ========================================
    // Effect 1: Color Bleeding
    // ========================================
    if (!color_bleeding_disabled()) {
        apply_color_bleeding(src, dst, width_, height_,
                            params_.color_bleeding_blur(), stream);
        std::swap(src, dst);
    }

    // ========================================
    // Effect 2: Chromatic Aberration
    // ========================================
    apply_chromatic_aberration(src, dst, width_, height_,
                               params_.chromatic_offset(), stream);
    std::swap(src, dst);

    // ========================================
    // Effect 3: Horizontal Wobble
    // ========================================
    float wobble_time = frame_time * 2.0f;  // speed = 2.0
    apply_horizontal_wobble(src, dst, width_, height_,
                           params_.wobble_frequency(),
                           params_.wobble_amplitude(),
                           wobble_time, stream);
    std::swap(src, dst);

    // ========================================
    // Effect 4: Tracking Errors (occasional)
    // ========================================
    if (!tracking_errors_disabled() &&
        static_cast<float>(rand()) / RAND_MAX < params_.tracking_prob()) {
        TrackingBand bands[8];
        int num_bands;
        generate_tracking_bands(bands, num_bands, height_,
                               0.3f, params_.tracking_sigma());
        apply_tracking_errors(src, dst, width_, height_, bands, num_bands, stream);
        std::swap(src, dst);
    }

    // ========================================
    // Effect 5: Position Jitter
    // ========================================
    float jitter_freq = params_.jitter_freq();
    float jitter_interval = 1.0f / jitter_freq;

    if (frame_time - last_jitter_time_ >= jitter_interval) {
        // Generate new jitter offset
        float amp_x = params_.jitter_amp_x();
        float amp_y = params_.jitter_amp_y();
        current_jitter_x_ = static_cast<int>((static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * amp_x);
        current_jitter_y_ = static_cast<int>((static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f) * amp_y);
        last_jitter_time_ = frame_time;
    }

    if (current_jitter_x_ != 0 || current_jitter_y_ != 0) {
        apply_position_jitter(src, dst, width_, height_,
                             current_jitter_x_, current_jitter_y_, stream);
        std::swap(src, dst);
    }

    // ========================================
    // Effect 6: Vertical Jitter (occasional)
    // ========================================
    float vertical_prob = params_.intensity * 0.1f;
    if (static_cast<float>(rand()) / RAND_MAX < vertical_prob) {
        int jump = 1 + static_cast<int>(params_.intensity * 4);
        int direction = (rand() % 2) * 2 - 1;
        apply_vertical_jitter(src, dst, width_, height_, jump * direction, stream);
        std::swap(src, dst);
    }

    // ========================================
    // Effect 7: Scanlines
    // ========================================
    apply_scanlines(src, dst, width_, height_,
                   params_.scanline_darkness(), stream);
    std::swap(src, dst);

    // ========================================
    // Effect 8: Noise/Grain
    // ========================================
    if (!noise_disabled()) {
        apply_noise_grain(src, dst, width_, height_,
                         params_.noise_luma(),
                         params_.noise_color(),
                         params_.apply_color_noise(), stream);
        std::swap(src, dst);
    }

    // ========================================
    // Effect 9: Color Grading
    // ========================================
    apply_color_grading(src, dst, width_, height_,
                       params_.black_lift(),
                       params_.contrast_factor(),
                       params_.desaturation(),
                       grading::BLUE_TINT,
                       grading::RED_TINT, stream);
    std::swap(src, dst);

    // ========================================
    // Effect 10: Sharpness Reduction (using NPP blur)
    // ========================================
    // For simplicity, we skip this as NPP blur was applied in color_bleeding
    // Could add another blur pass here with smaller kernel

    // ========================================
    // VHS Overlay Blend (optional)
    // ========================================
    if (d_vhs_overlay != nullptr && !overlay_disabled()) {
        apply_vhs_overlay_blend(src, d_vhs_overlay, dst, width_, height_,
                               VHS_OVERLAY_OPACITY, stream);
        std::swap(src, dst);
    }

    // Copy result to output
    CUDA_CHECK(cudaMemcpyAsync(d_output, src, frame_size,
                                      cudaMemcpyDeviceToDevice, stream));
}

} // namespace effects
} // namespace vhs
