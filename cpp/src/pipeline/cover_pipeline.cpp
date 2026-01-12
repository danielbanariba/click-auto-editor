/**
 * Cover pipeline - generate VHS main segment from cover, then prepend intro.
 */

#include "cover_pipeline.hpp"

#include "ffmpeg_decoder.hpp"
#include "ffmpeg_encoder.hpp"
#include "cuda_frame_buffer.hpp"
#include "effects/vhs_effect_chain.hpp"
#include "effects/cover_overlay.hpp"
#include "effects/vhs_transition.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/npp_wrappers.hpp"
#include "utils/image_loader.hpp"
#include "utils/cancel_flag.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace vhs {
namespace pipeline {

CoverPipeline::CoverPipeline() = default;
CoverPipeline::~CoverPipeline() = default;

static bool upload_image_to_gpu(
    const utils::ImageData& image,
    int bytes_per_pixel,
    unsigned char** d_output
) {
    if (image.pixels.empty() || image.width <= 0 || image.height <= 0) {
        return false;
    }

    size_t size = static_cast<size_t>(image.width) * image.height * bytes_per_pixel;
    CUDA_CHECK(cudaMalloc(d_output, size));
    CUDA_CHECK(cudaMemcpy(*d_output, image.pixels.data(), size, cudaMemcpyHostToDevice));
    return true;
}

static utils::ImageData convert_bgr_to_bgra(const utils::ImageData& src) {
    utils::ImageData out;
    out.width = src.width;
    out.height = src.height;
    out.pixels.resize(static_cast<size_t>(src.width) * src.height * 4);

    const unsigned char* src_ptr = src.pixels.data();
    unsigned char* dst_ptr = out.pixels.data();
    size_t total_pixels = static_cast<size_t>(src.width) * src.height;

    for (size_t i = 0; i < total_pixels; i++) {
        dst_ptr[i * 4 + 0] = src_ptr[i * 3 + 0];
        dst_ptr[i * 4 + 1] = src_ptr[i * 3 + 1];
        dst_ptr[i * 4 + 2] = src_ptr[i * 3 + 2];
        dst_ptr[i * 4 + 3] = 255;
    }

    return out;
}

bool CoverPipeline::process(
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
    const std::string& vhs_overlay_path
) {
    if (main_duration <= 0.0) {
        fprintf(stderr, "[CoverPipeline] Invalid main duration\n");
        return false;
    }

    struct TrackEntry {
        double start = 0.0;
        double end = 0.0;
    };

    auto load_tracklist = [](const std::string& path, std::vector<TrackEntry>& tracks) -> bool {
        tracks.clear();
        if (path.empty()) {
            return false;
        }
        std::ifstream file(path);
        if (!file.is_open()) {
            fprintf(stderr, "[CoverPipeline] Failed to open tracklist: %s\n", path.c_str());
            return false;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            std::stringstream ss(line);
            std::string idx_str;
            std::string start_str;
            std::string end_str;
            if (!std::getline(ss, idx_str, '\t')) {
                continue;
            }
            if (!std::getline(ss, start_str, '\t')) {
                continue;
            }
            if (!std::getline(ss, end_str, '\t')) {
                continue;
            }
            try {
                double start = std::stod(start_str);
                double end = std::stod(end_str);
                if (end <= start) {
                    continue;
                }
                TrackEntry entry;
                entry.start = start;
                entry.end = end;
                tracks.push_back(entry);
            } catch (const std::exception&) {
                continue;
            }
        }
        return !tracks.empty();
    };

    auto load_overlay_images = [](const std::string& dir_path, int target_width, int target_height,
                                  std::vector<unsigned char*>& d_overlays) -> bool {
        d_overlays.clear();
        if (dir_path.empty()) {
            return false;
        }
        std::filesystem::path dir(dir_path);
        if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
            fprintf(stderr, "[CoverPipeline] Overlay dir not found: %s\n", dir_path.c_str());
            return false;
        }

        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            return false;
        }

        for (const auto& path : files) {
            utils::ImageData overlay;
            if (!utils::load_image_bgra(path.string(), overlay)) {
                fprintf(stderr, "[CoverPipeline] Failed to load overlay: %s\n", path.string().c_str());
                continue;
            }
            if (overlay.width != target_width || overlay.height != target_height) {
                utils::ImageData resized;
                if (!utils::resize_image_bgra(overlay, target_width, target_height, resized)) {
                    fprintf(stderr, "[CoverPipeline] Failed to resize overlay: %s\n", path.string().c_str());
                    continue;
                }
                overlay = std::move(resized);
            }

            unsigned char* d_overlay = nullptr;
            size_t size = static_cast<size_t>(overlay.width) * overlay.height * 4;
            cudaError_t err = cudaMalloc(&d_overlay, size);
            if (err != cudaSuccess) {
                fprintf(stderr, "[CoverPipeline] Failed to allocate overlay GPU buffer: %s\n",
                        cudaGetErrorString(err));
                break;
            }
            err = cudaMemcpy(d_overlay, overlay.pixels.data(), size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "[CoverPipeline] Failed to upload overlay: %s\n",
                        cudaGetErrorString(err));
                cudaFree(d_overlay);
                break;
            }
            d_overlays.push_back(d_overlay);
        }

        return !d_overlays.empty();
    };

    auto free_overlay_images = [](std::vector<unsigned char*>& d_overlays) {
        for (auto* ptr : d_overlays) {
            if (ptr) {
                cudaFree(ptr);
            }
        }
        d_overlays.clear();
    };

    utils::ImageData cover;
    if (!utils::load_image_bgr(cover_path, cover)) {
        fprintf(stderr, "[CoverPipeline] Failed to load cover: %s\n", cover_path.c_str());
        return false;
    }

    // Prepare background (scale to fill, crop center)
    double scale = std::max(
        static_cast<double>(width) / cover.width,
        static_cast<double>(height) / cover.height
    );
    int bg_w = std::max(1, static_cast<int>(std::round(cover.width * scale)));
    int bg_h = std::max(1, static_cast<int>(std::round(cover.height * scale)));

    utils::ImageData cover_resized;
    if (!utils::resize_image_bgr(cover, bg_w, bg_h, cover_resized)) {
        fprintf(stderr, "[CoverPipeline] Failed to resize background\n");
        return false;
    }

    int crop_x = std::max(0, (bg_w - width) / 2);
    int crop_y = std::max(0, (bg_h - height) / 2);
    utils::ImageData bg_image;
    if (!utils::crop_image_bgr(cover_resized, crop_x, crop_y, width, height, bg_image)) {
        fprintf(stderr, "[CoverPipeline] Failed to crop background\n");
        return false;
    }

    // Prepare cover overlay (with alpha)
    utils::ImageData cover_overlay_src;
    if (!cover_overlay_path.empty()) {
        if (!utils::load_image_bgra(cover_overlay_path, cover_overlay_src)) {
            fprintf(stderr, "[CoverPipeline] Failed to load cover overlay: %s\n",
                    cover_overlay_path.c_str());
        }
    }
    if (cover_overlay_src.pixels.empty()) {
        cover_overlay_src = convert_bgr_to_bgra(cover);
    }

    int cover_height = std::max(1, static_cast<int>(std::round(height * 0.99)));
    double aspect = static_cast<double>(cover_overlay_src.width) / cover_overlay_src.height;
    int cover_width = std::max(1, static_cast<int>(std::round(cover_height * aspect)));

    utils::ImageData cover_overlay;
    if (!utils::resize_image_bgra(cover_overlay_src, cover_width, cover_height, cover_overlay)) {
        fprintf(stderr, "[CoverPipeline] Failed to resize cover overlay\n");
        return false;
    }

    // Upload images to GPU
    unsigned char* d_bg = nullptr;
    unsigned char* d_bg_blur = nullptr;
    unsigned char* d_cover = nullptr;
    if (!upload_image_to_gpu(bg_image, 3, &d_bg)) {
        fprintf(stderr, "[CoverPipeline] Failed to upload background\n");
        return false;
    }
    if (!upload_image_to_gpu(cover_overlay, 4, &d_cover)) {
        fprintf(stderr, "[CoverPipeline] Failed to upload cover overlay\n");
        cudaFree(d_bg);
        return false;
    }

    std::vector<TrackEntry> tracklist;
    std::vector<unsigned char*> track_overlays;
    bool use_track_overlays = false;
    if (!tracklist_path.empty() && !track_overlays_path.empty()) {
        if (load_tracklist(tracklist_path, tracklist) &&
            load_overlay_images(track_overlays_path, width, height, track_overlays)) {
            use_track_overlays = true;
        } else {
            free_overlay_images(track_overlays);
            tracklist.clear();
        }
    }

    // Blur background on GPU
    size_t frame_size = static_cast<size_t>(width) * height * 3;
    CUDA_CHECK(cudaMalloc(&d_bg_blur, frame_size));
    int step = width * 3;
    bool blur_ok = npp::gaussian_blur_8u_c3(
        d_bg, d_bg_blur,
        width, height,
        step, step,
        15,  // max kernel size
        0
    );
    unsigned char* d_bg_final = blur_ok ? d_bg_blur : d_bg;

    // Setup encoder
    FFmpegEncoder encoder;
    if (!encoder.open(output_path, width, height, fps, cq, preset, use_hw_accel)) {
        fprintf(stderr, "[CoverPipeline] Failed to open encoder\n");
        cudaFree(d_bg);
        cudaFree(d_bg_blur);
        cudaFree(d_cover);
        return false;
    }

    // Overlay noise (optional)
    std::unique_ptr<VHSOverlayBuffer> vhs_overlay;
    if (!vhs_overlay_path.empty()) {
        vhs_overlay = std::make_unique<VHSOverlayBuffer>();
        if (!vhs_overlay->load(vhs_overlay_path, width, height)) {
            vhs_overlay.reset();
        }
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate buffers (retry without overlay if VRAM is tight)
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    auto allocate_io_buffers = [&]() -> bool {
        cudaError_t err = cudaMalloc(&d_input, frame_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CoverPipeline] Failed to allocate input buffer: %s\n",
                    cudaGetErrorString(err));
            d_input = nullptr;
            return false;
        }
        err = cudaMalloc(&d_output, frame_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CoverPipeline] Failed to allocate output buffer: %s\n",
                    cudaGetErrorString(err));
            cudaFree(d_input);
            d_input = nullptr;
            d_output = nullptr;
            return false;
        }
        return true;
    };

    if (!allocate_io_buffers()) {
        if (vhs_overlay) {
            fprintf(stderr, "[CoverPipeline] VRAM low, disabling overlay and retrying\n");
            vhs_overlay.reset();
            cudaGetLastError();  // Clear error state before retry
        }
        if (!allocate_io_buffers()) {
            encoder.close();
            cudaFree(d_bg);
            if (d_bg_blur) cudaFree(d_bg_blur);
            cudaFree(d_cover);
            cudaStreamDestroy(stream);
            return false;
        }
    }

    // Process intro frames (no VHS effect)
    int64_t pts = 0;
    int64_t intro_frames = static_cast<int64_t>(std::round(INTRO_DURATION * fps));
    unsigned char* d_intro_last = nullptr;

    if (!intro_path.empty()) {
        CUDA_CHECK(cudaMalloc(&d_intro_last, frame_size));
    }

    if (!intro_path.empty()) {
        FFmpegDecoder decoder;
        if (decoder.open(intro_path, use_hw_accel)) {
            int src_w = decoder.width();
            int src_h = decoder.height();
            size_t src_size = static_cast<size_t>(src_w) * src_h * 3;

            unsigned char* d_intro = nullptr;
            CUDA_CHECK(cudaMalloc(&d_intro, src_size));

            unsigned char* d_intro_resized = nullptr;
            if (src_w != width || src_h != height) {
                CUDA_CHECK(cudaMalloc(&d_intro_resized, frame_size));
            }

            int64_t frame_idx = 0;
            while (frame_idx < intro_frames && decoder.decode_frame(d_intro, nullptr)) {
                if (vhs::utils::is_cancel_requested()) {
                    fprintf(stderr, "[CoverPipeline] Cancel requested, stopping intro\n");
                    break;
                }
                const unsigned char* d_frame = d_intro;

                if (d_intro_resized) {
                    npp::resize_8u_c3(
                        d_intro,
                        src_w,
                        src_h,
                        src_w * 3,
                        d_intro_resized,
                        width,
                        height,
                        width * 3,
                        NPPI_INTER_LINEAR,
                        stream
                    );
                    d_frame = d_intro_resized;
                }

                if (d_intro_last) {
                    CUDA_CHECK(cudaMemcpyAsync(d_intro_last, d_frame, frame_size,
                                               cudaMemcpyDeviceToDevice, stream));
                }

                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (!encoder.encode_frame(d_frame, pts)) {
                    fprintf(stderr, "[CoverPipeline] Failed to encode intro frame\n");
                    break;
                }

                pts += 1000;
                frame_idx++;
            }

            cudaFree(d_intro);
            if (d_intro_resized) cudaFree(d_intro_resized);
            decoder.close();
        } else {
            fprintf(stderr, "[CoverPipeline] Warning: intro not loaded, skipping\n");
        }
    }

    // VHS effect chain
    effects::VHSEffectChain effect_chain(width, height, params);
    effect_chain.init();

    int64_t main_frames = static_cast<int64_t>(std::round(main_duration * fps));
    int64_t transition_frames = static_cast<int64_t>(std::round(TRANSITION_DURATION * fps));
    if (transition_frames < 1) {
        transition_frames = 1;
    }
    for (int64_t i = 0; i < main_frames; i++) {
        if (vhs::utils::is_cancel_requested()) {
            fprintf(stderr, "[CoverPipeline] Cancel requested, stopping\n");
            break;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_input, d_bg_final, frame_size,
                                   cudaMemcpyDeviceToDevice, stream));

        float frame_time = static_cast<float>(i / fps);
        unsigned char* d_overlay = nullptr;
        if (vhs_overlay && vhs_overlay->is_loaded()) {
            d_overlay = vhs_overlay->get_frame(static_cast<int>(i));
        }

        effect_chain.process_frame(d_input, d_output, d_overlay, frame_time, stream);
        if (use_track_overlays) {
            int track_idx = 0;
            double current_time = static_cast<double>(i) / fps;
            if (current_time >= tracklist.back().end) {
                track_idx = static_cast<int>(tracklist.size()) - 1;
            } else {
                for (size_t t = 0; t < tracklist.size(); t++) {
                    if (current_time >= tracklist[t].start && current_time < tracklist[t].end) {
                        track_idx = static_cast<int>(t);
                        break;
                    }
                }
            }
            if (!track_overlays.empty()) {
                if (track_idx < 0) {
                    track_idx = 0;
                }
                if (track_idx >= static_cast<int>(track_overlays.size())) {
                    track_idx = static_cast<int>(track_overlays.size()) - 1;
                }
                effects::apply_cover_overlay(
                    d_output,
                    track_overlays[track_idx],
                    width,
                    height,
                    width,
                    height,
                    stream
                );
            }
        } else {
            effects::apply_cover_overlay(d_output, d_cover, width, height, cover_width, cover_height, stream);
        }

        if (d_intro_last && i < transition_frames) {
            float progress = static_cast<float>(i + 1) / static_cast<float>(transition_frames);
            effects::apply_vhs_transition(
                d_intro_last,
                d_output,
                d_input,
                width,
                height,
                progress,
                static_cast<int>(i),
                stream
            );
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (!encoder.encode_frame(d_input, pts)) {
                fprintf(stderr, "[CoverPipeline] Failed to encode transition frame %ld\n", i);
                break;
            }
        } else {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (!encoder.encode_frame(d_output, pts)) {
                fprintf(stderr, "[CoverPipeline] Failed to encode frame %ld\n", i);
                break;
            }
        }

        pts += 1000;
    }

    effect_chain.cleanup();
    encoder.close();

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bg);
    if (d_bg_blur) cudaFree(d_bg_blur);
    cudaFree(d_cover);
    free_overlay_images(track_overlays);
    if (d_intro_last) cudaFree(d_intro_last);
    cudaStreamDestroy(stream);

    return true;
}

} // namespace pipeline
} // namespace vhs
