#pragma once

#include <string>

#include "config/render_config.hpp"

namespace vhs {
namespace pipeline {

class CoverPipeline {
public:
    CoverPipeline();
    ~CoverPipeline();

    bool process(
        const std::string& intro_path,
        const std::string& cover_path,
        const std::string& cover_overlay_path,
        const std::string& tracklist_path,
        const std::string& track_overlays_path,
        double main_duration,
        const std::string& output_path,
        const ::vhs::VHSParams& params,
        int width,
        int height,
        double fps,
        int cq,
        const std::string& preset,
        bool use_hw_accel,
        const std::string& vhs_overlay_path = ""
    );
};

} // namespace pipeline
} // namespace vhs
