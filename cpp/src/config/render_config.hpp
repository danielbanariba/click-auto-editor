#pragma once

#include <string>
#include <cstdint>

namespace vhs {

// Video specifications
constexpr int VIDEO_WIDTH = 3840;
constexpr int VIDEO_HEIGHT = 2160;
constexpr int FPS = 24;
constexpr float INTRO_DURATION = 7.0f;
constexpr float TRANSITION_DURATION = 1.0f;

// NVENC settings
constexpr int VIDEO_CQ = 20;
constexpr int VIDEO_BITRATE = 45000000;
constexpr int VIDEO_MAXRATE = 45000000;
constexpr int VIDEO_BUFSIZE = 90000000;
constexpr int VIDEO_B_FRAMES = 2;
constexpr const char* VIDEO_PRESET = "p1";
constexpr const char* AUDIO_BITRATE = "320k";

// VHS Effect parameters (based on intensity 0.0-1.0)
struct VHSParams {
    float intensity = 0.5f;

    // Derived parameters (calculated from intensity)
    int color_bleeding_blur() const { return static_cast<int>(5 + intensity * 20); }
    int chromatic_offset() const { return static_cast<int>(2 + intensity * 4); }
    float wobble_frequency() const { return 30.0f + intensity * 30.0f; }
    float wobble_amplitude() const { return intensity * 15.0f; }
    float jitter_freq() const { return 7.0f + intensity * 5.0f; }
    float jitter_amp_x() const { return 2.0f + intensity * 6.0f; }
    float jitter_amp_y() const { return 3.0f + intensity * 9.0f; }
    float tracking_prob() const { return 0.3f * intensity; }
    float tracking_sigma() const { return 5.0f * intensity; }
    float scanline_darkness() const { return 0.7f + intensity * 0.15f; }
    float noise_luma() const { return 10.0f + intensity * 20.0f; }
    float noise_color() const { return 5.0f * intensity; }
    bool apply_color_noise() const { return intensity > 0.3f; }
    float black_lift() const { return 1.5f + intensity * 2.5f; }
    float contrast_factor() const { return 1.2f - intensity * 0.005f; }
    float desaturation() const { return 0.15f + intensity * 0.15f; }
    int blur_amount() const { return 1 + static_cast<int>(intensity * 2); }
};

// YIQ color conversion matrices (NTSC standard)
namespace yiq {
    // RGB to YIQ
    constexpr float R_TO_Y = 0.299f;
    constexpr float G_TO_Y = 0.587f;
    constexpr float B_TO_Y = 0.114f;

    constexpr float R_TO_I = 0.596f;
    constexpr float G_TO_I = -0.274f;
    constexpr float B_TO_I = -0.322f;

    constexpr float R_TO_Q = 0.211f;
    constexpr float G_TO_Q = -0.523f;
    constexpr float B_TO_Q = 0.312f;

    // YIQ to RGB
    constexpr float Y_TO_R = 1.0f;
    constexpr float I_TO_R = 0.956f;
    constexpr float Q_TO_R = 0.621f;

    constexpr float Y_TO_G = 1.0f;
    constexpr float I_TO_G = -0.272f;
    constexpr float Q_TO_G = -0.647f;

    constexpr float Y_TO_B = 1.0f;
    constexpr float I_TO_B = -1.106f;
    constexpr float Q_TO_B = 1.703f;
}

// Color grading constants
namespace grading {
    constexpr float BLUE_TINT = 1.0f;
    constexpr float RED_TINT = 1.0f;
    constexpr float MID_GRAY = 127.5f;
}

// VHS overlay blend
constexpr float VHS_OVERLAY_OPACITY = 0.6f;
constexpr float VHS_TRANSITION_TAPE_NOISE = 1.0f;      // 100%
constexpr float VHS_TRANSITION_TAPE_DISTORTION = 1.0f; // 100%
constexpr float VHS_TRANSITION_RANDOM_NOISE = 0.25f;   // 25%
constexpr float VHS_TRANSITION_WRINKLE_SIZE = 0.05f;   // 5%
constexpr float VHS_TRANSITION_FRAME_JITTER = 0.02f;   // 2%
constexpr float VHS_TRANSITION_CHROMA_OFFSET = 6.0f;   // pixels
constexpr float VHS_TRANSITION_CHROMA_BLUR = 0.35f;    // blend factor

// Render configuration
struct RenderConfig {
    std::string input_folder;
    std::string output_path;
    std::string intro_path;
    std::string vhs_overlay_path;
    VHSParams vhs_params;
    int width = VIDEO_WIDTH;
    int height = VIDEO_HEIGHT;
    int fps = FPS;
    int cq = VIDEO_CQ;
    bool show_progress = true;
};

namespace config {

// Extended configuration with file I/O
class RenderConfig {
public:
    RenderConfig();

    bool load_from_file(const std::string& path);
    bool save_to_file(const std::string& path) const;

    int width;
    int height;
    double fps;
    int cq;
    std::string preset;
    bool use_hw_accel;
    float vhs_intensity;
};

} // namespace config

} // namespace vhs
