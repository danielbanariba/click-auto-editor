#pragma once

#include <string>
#include <cstdint>

namespace vhs {

// Video specifications
constexpr int VIDEO_WIDTH = 3840;
constexpr int VIDEO_HEIGHT = 2160;
constexpr int FPS = 24;
constexpr float INTRO_DURATION = 7.0f;
constexpr float TRANSITION_DURATION = 2.0f;  // seconds the main part rolls in over the held intro frame

// NVENC settings
constexpr int VIDEO_CQ = 20;
constexpr int VIDEO_BITRATE = 45000000;
constexpr int VIDEO_MAXRATE = 45000000;
constexpr int VIDEO_BUFSIZE = 90000000;
constexpr int VIDEO_B_FRAMES = 2;
constexpr const char* VIDEO_PRESET = "p1";
constexpr const char* AUDIO_BITRATE = "320k";

// VHS effect parameters. The Arsenal VHS look has fixed, calibrated settings (see
// effects/arsenal_vhs.hpp); the intensity is kept only so the command line stays compatible and
// no longer changes the picture.
struct VHSParams {
    float intensity = 0.5f;
};

// VHS overlay blend
constexpr float VHS_OVERLAY_OPACITY = 0.6f;

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
