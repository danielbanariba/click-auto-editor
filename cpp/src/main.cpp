/**
 * VHS Video Renderer - Main Entry Point
 *
 * CLI tool for applying VHS effects to videos using CUDA acceleration.
 *
 * Usage:
 *   ./vhs_render --input <path> --output <path> [options]
 *
 * Options:
 *   --input, -i     Input video file or folder
 *   --output, -o    Output video file or folder
 *   --intensity     VHS effect intensity (0.0-1.0, default: 0.5)
 *   --vhs-overlay   Path to VHS noise overlay video
 *   --cq            Constant quality (0-51, default: 20)
 *   --preset        NVENC preset (p1-p7, default: p1)
 *   --no-gpu        Disable GPU acceleration
 *   --help, -h      Show help message
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <cuda_runtime.h>
#include <exception>

#include "pipeline/video_pipeline.hpp"
#include "pipeline/cover_pipeline.hpp"
#include "config/render_config.hpp"
#include "utils/cancel_flag.hpp"

namespace fs = std::filesystem;

void print_usage(const char* prog_name) {
    std::cout << "\nVHS Video Renderer - CUDA Accelerated\n";
    std::cout << "======================================\n\n";
    std::cout << "Usage: " << prog_name << " --input <path> --output <path> [options]\n\n";
    std::cout << "Required:\n";
    std::cout << "  --input, -i     Input video file or folder\n";
    std::cout << "  --output, -o    Output video file or folder\n\n";
    std::cout << "Options:\n";
    std::cout << "  --intensity     VHS effect intensity (0.0-1.0, default: 0.5)\n";
    std::cout << "  --vhs-overlay   Path to VHS noise overlay video\n";
    std::cout << "  --intro         Intro video path (cover mode)\n";
    std::cout << "  --cover         Cover image path (enables cover mode)\n";
    std::cout << "  --cover-overlay Cover overlay path with alpha (optional)\n";
    std::cout << "  --tracklist     Tracklist file (optional)\n";
    std::cout << "  --track-overlays Directory with track overlay images (optional)\n";
    std::cout << "  --duration      Main segment duration in seconds (cover mode)\n";
    std::cout << "  --cq            Constant quality (0-51, default: 20)\n";
    std::cout << "  --preset        NVENC preset (p1-p7, default: p1)\n";
    std::cout << "  --no-gpu        Disable GPU acceleration\n";
    std::cout << "  --help, -h      Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << prog_name << " -i video.mp4 -o output.mp4\n";
    std::cout << "  " << prog_name << " -i ./videos/ -o ./output/ --intensity 0.7\n";
    std::cout << "  " << prog_name << " -i video.mp4 -o out.mp4 --vhs-overlay noise.mp4\n\n";
    std::cout << "  " << prog_name << " --cover cover.jpg --intro intro.mp4 --duration 1800 -o out.mp4\n\n";
}

void print_gpu_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "[GPU] No CUDA devices found!\n";
        return;
    }

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

        std::cout << "[GPU " << i << "] " << props.name << "\n";
        std::cout << "        Compute: " << props.major << "." << props.minor << "\n";
        std::cout << "        Memory:  " << (props.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "        SMs:     " << props.multiProcessorCount << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    vhs::utils::install_signal_handlers();

    // Parse arguments
    std::string input_path;
    std::string output_path;
    std::string vhs_overlay_path;
    std::string intro_path;
    std::string cover_path;
    std::string cover_overlay_path;
    std::string tracklist_path;
    std::string track_overlays_path;
    double main_duration = 0.0;
    float intensity = 0.5f;
    int cq = 20;
    std::string preset = "p1";
    bool use_gpu = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--input" || arg == "-i") {
            if (i + 1 < argc) input_path = argv[++i];
        }
        else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) output_path = argv[++i];
        }
        else if (arg == "--intensity") {
            if (i + 1 < argc) intensity = std::atof(argv[++i]);
        }
        else if (arg == "--vhs-overlay") {
            if (i + 1 < argc) vhs_overlay_path = argv[++i];
        }
        else if (arg == "--intro") {
            if (i + 1 < argc) intro_path = argv[++i];
        }
        else if (arg == "--cover") {
            if (i + 1 < argc) cover_path = argv[++i];
        }
        else if (arg == "--cover-overlay") {
            if (i + 1 < argc) cover_overlay_path = argv[++i];
        }
        else if (arg == "--tracklist") {
            if (i + 1 < argc) tracklist_path = argv[++i];
        }
        else if (arg == "--track-overlays") {
            if (i + 1 < argc) track_overlays_path = argv[++i];
        }
        else if (arg == "--duration") {
            if (i + 1 < argc) main_duration = std::atof(argv[++i]);
        }
        else if (arg == "--cq") {
            if (i + 1 < argc) cq = std::atoi(argv[++i]);
        }
        else if (arg == "--preset") {
            if (i + 1 < argc) preset = argv[++i];
        }
        else if (arg == "--no-gpu") {
            use_gpu = false;
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    bool cover_mode = !cover_path.empty();
    if (output_path.empty()) {
        std::cerr << "Error: --output is required\n";
        print_usage(argv[0]);
        return 1;
    }

    if (!cover_mode && input_path.empty()) {
        std::cerr << "Error: --input is required when not using --cover\n";
        print_usage(argv[0]);
        return 1;
    }

    if (cover_mode && main_duration <= 0.0) {
        std::cerr << "Error: --duration is required when using --cover\n";
        print_usage(argv[0]);
        return 1;
    }

    // Clamp intensity
    if (intensity < 0.0f) intensity = 0.0f;
    if (intensity > 1.0f) intensity = 1.0f;

    // Clamp CQ
    if (cq < 0) cq = 0;
    if (cq > 51) cq = 51;

    // Print GPU info
    std::cout << "\n=== VHS Video Renderer ===\n\n";
    print_gpu_info();

    // Configure VHS parameters
    vhs::VHSParams params;
    params.intensity = intensity;

    std::cout << "[Config] Intensity: " << intensity << "\n";
    std::cout << "[Config] CQ: " << cq << "\n";
    std::cout << "[Config] Preset: " << preset << "\n";
    std::cout << "[Config] GPU: " << (use_gpu ? "enabled" : "disabled") << "\n\n";

    bool success = false;
    try {
        if (cover_mode) {
            std::cout << "[Cover] Intro: " << (intro_path.empty() ? "(none)" : intro_path) << "\n";
            std::cout << "[Cover] Cover: " << cover_path << "\n";
            if (!cover_overlay_path.empty()) {
                std::cout << "[Cover] Overlay: " << cover_overlay_path << "\n";
            }
            std::cout << "[Cover] Main duration: " << main_duration << "s\n";

            vhs::pipeline::CoverPipeline pipeline;
            success = pipeline.process(
                intro_path,
                cover_path,
                cover_overlay_path,
                tracklist_path,
                track_overlays_path,
                main_duration,
                output_path,
                params,
                ::vhs::VIDEO_WIDTH,
                ::vhs::VIDEO_HEIGHT,
                ::vhs::FPS,
                cq,
                preset,
                use_gpu,
                vhs_overlay_path
            );
        } else {
            // Create pipeline
            vhs::pipeline::VideoPipeline pipeline;
            pipeline.set_params(params);

            // Load VHS overlay if provided
            if (!vhs_overlay_path.empty()) {
                std::cout << "[Overlay] Loading: " << vhs_overlay_path << "\n";
                if (!pipeline.load_vhs_overlay(vhs_overlay_path)) {
                    std::cerr << "[Overlay] Warning: Failed to load VHS overlay\n";
                }
            }

            // Set progress callback with percentage bar
            pipeline.set_progress_callback([](int64_t current, int64_t total, double fps) {
                if (total <= 0) return;

                double progress = 100.0 * current / total;
                int bar_width = 40;
                int filled = static_cast<int>(progress / 100.0 * bar_width);

                std::cout << "\r[";
                for (int i = 0; i < bar_width; i++) {
                    if (i < filled) std::cout << "=";
                    else if (i == filled) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << static_cast<int>(progress) << "% @ "
                          << static_cast<int>(fps) << " fps   " << std::flush;
            });

            // Check if input is file or directory
            bool is_directory = fs::is_directory(input_path);

            if (is_directory) {
                success = pipeline.process_folder(input_path, output_path, cq, preset);
            } else {
                success = pipeline.process(input_path, output_path, cq, preset);
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "\n\n[Error] Caught exception: " << ex.what() << "\n";
        cudaDeviceReset();
        return 1;
    } catch (...) {
        std::cerr << "\n\n[Error] Unknown exception\n";
        cudaDeviceReset();
        return 1;
    }

    if (vhs::utils::is_cancel_requested()) {
        std::cerr << "\n\n[Cancel] Rendering cancelled by user\n";
        cudaDeviceReset();
        return 130;
    }

    if (success) {
        std::cout << "\n\n[Done] Rendering complete!\n";
        return 0;
    } else {
        std::cerr << "\n\n[Error] Rendering failed!\n";
        return 1;
    }
}
