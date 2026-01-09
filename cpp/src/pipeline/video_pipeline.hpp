#pragma once

#include <string>
#include <memory>
#include <functional>
#include <cuda_runtime.h>

#include "ffmpeg_decoder.hpp"
#include "ffmpeg_encoder.hpp"
#include "cuda_frame_buffer.hpp"
#include "effects/vhs_effect_chain.hpp"
#include "config/render_config.hpp"

namespace vhs {
namespace pipeline {

/**
 * Progress callback type
 * Parameters: current_frame, total_frames, fps
 */
using ProgressCallback = std::function<void(int64_t, int64_t, double)>;

/**
 * Video Processing Pipeline
 *
 * Orchestrates the complete video rendering pipeline:
 * 1. Decode input video (NVDEC)
 * 2. Apply VHS effects (CUDA)
 * 3. Blend VHS overlay (CUDA)
 * 4. Encode output video (NVENC)
 */
class VideoPipeline {
public:
    VideoPipeline();
    ~VideoPipeline();

    // Set VHS effect parameters
    void set_params(const ::vhs::VHSParams& params);

    // Set progress callback
    void set_progress_callback(ProgressCallback callback);

    // Load VHS overlay video
    bool load_vhs_overlay(const std::string& path);

    // Process single video file
    bool process(
        const std::string& input_path,
        const std::string& output_path,
        int cq = 20,
        const std::string& preset = "p1"
    );

    // Process folder of videos
    bool process_folder(
        const std::string& input_folder,
        const std::string& output_folder,
        int cq = 20,
        const std::string& preset = "p1"
    );

private:
    ::vhs::VHSParams params_;
    ProgressCallback progress_callback_;

    std::unique_ptr<VHSOverlayBuffer> vhs_overlay_;
    std::string vhs_overlay_path_;  // Store path for deferred loading
    cudaStream_t stream_;

    bool process_frames(
        FFmpegDecoder& decoder,
        FFmpegEncoder& encoder,
        effects::VHSEffectChain& effect_chain
    );
};

} // namespace pipeline
} // namespace vhs
