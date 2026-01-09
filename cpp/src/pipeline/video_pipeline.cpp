/**
 * Video Pipeline Implementation
 */

#include "video_pipeline.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/cancel_flag.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace vhs {
namespace pipeline {

VideoPipeline::VideoPipeline()
    : vhs_overlay_(nullptr)
    , stream_(nullptr)
{
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

VideoPipeline::~VideoPipeline() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void VideoPipeline::set_params(const ::vhs::VHSParams& params) {
    params_ = params;
}

void VideoPipeline::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = callback;
}

bool VideoPipeline::load_vhs_overlay(const std::string& path) {
    if (!fs::exists(path)) {
        fprintf(stderr, "[Overlay] File not found: %s\n", path.c_str());
        return false;
    }
    // Store path for deferred loading (we need video dimensions first)
    vhs_overlay_path_ = path;
    return true;
}

bool VideoPipeline::process(
    const std::string& input_path,
    const std::string& output_path,
    int cq,
    const std::string& preset
) {
    printf("\n[Pipeline] Processing: %s\n", input_path.c_str());

    // Open decoder
    FFmpegDecoder decoder;
    if (!decoder.open(input_path, true)) {
        fprintf(stderr, "[Pipeline] Failed to open input: %s\n", input_path.c_str());
        return false;
    }

    int width = decoder.width();
    int height = decoder.height();
    double fps = decoder.fps();

    printf("[Pipeline] Video: %dx%d @ %.2f fps\n", width, height, fps);

    // Open encoder
    FFmpegEncoder encoder;
    if (!encoder.open(output_path, width, height, fps, cq, preset, true)) {
        fprintf(stderr, "[Pipeline] Failed to open output: %s\n", output_path.c_str());
        return false;
    }

    // Load VHS overlay if path was provided
    if (!vhs_overlay_path_.empty()) {
        printf("[Overlay] Loading VHS noise overlay...\n");
        vhs_overlay_ = std::make_unique<VHSOverlayBuffer>();
        if (!vhs_overlay_->load(vhs_overlay_path_, width, height)) {
            fprintf(stderr, "[Overlay] Warning: Failed to load overlay, continuing without it\n");
            vhs_overlay_.reset();
        }

        // Ensure CUDA state is clean before proceeding
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[Overlay] CUDA error after loading: %s\n",
                    cudaGetErrorString(err));
            vhs_overlay_.reset();
        }

        // If no frames were loaded, clear the overlay
        if (vhs_overlay_ && !vhs_overlay_->is_loaded()) {
            vhs_overlay_.reset();
        }
    }

    // Initialize effect chain
    effects::VHSEffectChain effect_chain(width, height, params_);
    effect_chain.init();

    // Process frames
    bool success = process_frames(decoder, encoder, effect_chain);

    // Cleanup
    effect_chain.cleanup();
    encoder.close();
    decoder.close();

    if (success) {
        printf("[Pipeline] Completed: %s\n", output_path.c_str());
    }

    return success;
}

bool VideoPipeline::process_frames(
    FFmpegDecoder& decoder,
    FFmpegEncoder& encoder,
    effects::VHSEffectChain& effect_chain
) {
    int width = decoder.width();
    int height = decoder.height();
    size_t frame_size = width * height * 3;

    // Allocate GPU buffers
    unsigned char* d_input;
    unsigned char* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, frame_size));
    CUDA_CHECK(cudaMalloc(&d_output, frame_size));

    int64_t total_frames = decoder.frame_count();
    int64_t frame_number = 0;
    int64_t pts;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_progress_time = start_time;

    printf("[Pipeline] Processing %ld frames...\n", total_frames);

    while (decoder.decode_frame(d_input, &pts)) {
        if (vhs::utils::is_cancel_requested()) {
            fprintf(stderr, "[Pipeline] Cancel requested, stopping\n");
            break;
        }
        // Calculate frame time for effects
        float frame_time = static_cast<float>(frame_number) / decoder.fps();

        // Get VHS overlay frame if available
        unsigned char* d_overlay = nullptr;
        if (vhs_overlay_ && vhs_overlay_->is_loaded()) {
            d_overlay = vhs_overlay_->get_frame(static_cast<int>(frame_number));
        }

        // Apply VHS effects
        effect_chain.process_frame(d_input, d_output, d_overlay, frame_time, stream_);

        // Synchronize before encoding
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        // Encode frame
        if (!encoder.encode_frame(d_output, pts)) {
            fprintf(stderr, "[Pipeline] Encoding failed at frame %ld\n", frame_number);
            break;
        }

        frame_number++;

        // Progress update every 30 frames
        if (frame_number % 30 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(now - start_time).count();
            double fps_actual = frame_number / elapsed;

            if (progress_callback_) {
                progress_callback_(frame_number, total_frames, fps_actual);
            } else {
                // Default progress output
                double progress = (total_frames > 0) ?
                    (100.0 * frame_number / total_frames) : 0.0;
                printf("\r[Pipeline] Frame %ld/%ld (%.1f%%) @ %.1f fps   ",
                       frame_number, total_frames, progress, fps_actual);
                fflush(stdout);
            }
        }
    }

    // Final progress
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
    double avg_fps = frame_number / total_elapsed;

    printf("\n[Pipeline] Processed %ld frames in %.1f seconds (%.1f fps avg)\n",
           frame_number, total_elapsed, avg_fps);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return true;
}

bool VideoPipeline::process_folder(
    const std::string& input_folder,
    const std::string& output_folder,
    int cq,
    const std::string& preset
) {
    // Create output folder if needed
    fs::create_directories(output_folder);

    int processed = 0;
    int failed = 0;

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        // Convert to lowercase
        for (auto& c : ext) c = std::tolower(c);

        // Check for video files
        if (ext != ".mp4" && ext != ".avi" && ext != ".mkv" && ext != ".mov") {
            continue;
        }

        std::string input_path = entry.path().string();
        std::string output_path = (fs::path(output_folder) / entry.path().filename()).string();

        if (process(input_path, output_path, cq, preset)) {
            processed++;
        } else {
            failed++;
        }
    }

    printf("\n[Pipeline] Folder complete: %d processed, %d failed\n", processed, failed);
    return (failed == 0);
}

} // namespace pipeline
} // namespace vhs
