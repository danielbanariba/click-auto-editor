#pragma once

#include "config/render_config.hpp"
#include <cuda_runtime.h>

namespace vhs {
namespace effects {

// Forward declarations for effect functions
void apply_color_bleeding(const unsigned char* d_input, unsigned char* d_output,
                          int width, int height, int blur_amount, cudaStream_t stream = 0);

void apply_chromatic_aberration(const unsigned char* d_input, unsigned char* d_output,
                                int width, int height, int offset, cudaStream_t stream = 0);

void apply_horizontal_wobble(const unsigned char* d_input, unsigned char* d_output,
                             int width, int height, float frequency, float amplitude,
                             float time_offset, cudaStream_t stream = 0);

// Tracking band parameters structure
struct TrackingBand {
    int y_start;
    int y_end;
    int offset;
    bool active;
};

void apply_tracking_errors(const unsigned char* d_input, unsigned char* d_output,
                           int width, int height, const TrackingBand* bands,
                           int num_bands, cudaStream_t stream = 0);
void generate_tracking_bands(TrackingBand* bands, int& num_bands, int height,
                             float probability, float sigma);

void apply_position_jitter(const unsigned char* d_input, unsigned char* d_output,
                           int width, int height, int offset_x, int offset_y,
                           cudaStream_t stream = 0);

void apply_vertical_jitter(const unsigned char* d_input, unsigned char* d_output,
                           int width, int height, int offset_y, cudaStream_t stream = 0);

void apply_scanlines(const unsigned char* d_input, unsigned char* d_output,
                     int width, int height, float darkness, cudaStream_t stream = 0);

void apply_noise_grain(const unsigned char* d_input, unsigned char* d_output,
                       int width, int height, float noise_luma, float noise_color,
                       bool apply_color_noise, cudaStream_t stream = 0);
void init_noise_rng(int width, int height, unsigned long seed = 42);
void cleanup_noise_rng();

void apply_color_grading(const unsigned char* d_input, unsigned char* d_output,
                         int width, int height, float black_lift, float contrast_factor,
                         float desaturation, float blue_factor, float red_factor,
                         cudaStream_t stream = 0);

void apply_vhs_overlay_blend(const unsigned char* d_base, const unsigned char* d_overlay,
                             unsigned char* d_output, int width, int height,
                             float opacity, cudaStream_t stream = 0);

// Cleanup functions
void init_color_bleeding_buffers(int width, int height);
void cleanup_color_bleeding_buffers();

/**
 * VHS Effect Chain
 *
 * Orchestrates all 10 VHS effects in the correct order:
 * 1. Color Bleeding
 * 2. Chromatic Aberration
 * 3. Horizontal Wobble
 * 4. Tracking Errors
 * 5. Position Jitter
 * 6. Vertical Jitter
 * 7. Scanlines
 * 8. Noise/Grain
 * 9. Color Grading
 * 10. Sharpness Reduction
 * + VHS Overlay Blend
 */
class VHSEffectChain {
public:
    VHSEffectChain(int width, int height, const ::vhs::VHSParams& params);
    ~VHSEffectChain();

    // Process a frame with all VHS effects
    // Returns output in d_output
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

    // Double buffer for ping-pong processing
    unsigned char* d_buffer_a_;
    unsigned char* d_buffer_b_;

    // Jitter state
    float last_jitter_time_;
    int current_jitter_x_;
    int current_jitter_y_;

    bool initialized_;
};

} // namespace effects
} // namespace vhs
