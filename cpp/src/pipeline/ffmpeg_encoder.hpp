#pragma once

#include <string>
#include <cuda_runtime.h>

// Forward declarations
struct AVFormatContext;
struct AVCodecContext;
struct AVBufferRef;
struct AVFrame;
struct AVPacket;
struct AVStream;

namespace vhs {
namespace pipeline {

/**
 * FFmpeg NVENC Encoder
 *
 * Encodes video using NVIDIA hardware encoder (NVENC).
 * Accepts frames from CUDA device memory.
 */
class FFmpegEncoder {
public:
    FFmpegEncoder();
    ~FFmpegEncoder();

    // Open output file for encoding
    bool open(
        const std::string& path,
        int width,
        int height,
        double fps,
        int cq = 20,
        const std::string& preset = "p1",
        bool use_hw_accel = true
    );

    // Close and finalize output
    void close();

    // Encode frame from device memory
    bool encode_frame(const unsigned char* d_input, int64_t pts);

    // Flush encoder and finalize
    bool flush();

    // Properties
    bool is_open() const { return is_open_; }

private:
    bool open_internal(
        const std::string& path,
        int width,
        int height,
        double fps,
        int cq,
        const std::string& preset,
        bool use_hw_accel
    );
    bool init_hw_frames();

    AVFormatContext* format_ctx_;
    AVCodecContext* codec_ctx_;
    AVBufferRef* hw_device_ctx_;
    AVBufferRef* hw_frames_ctx_;
    AVFrame* frame_;
    AVPacket* packet_;
    AVStream* stream_;

    int width_;
    int height_;
    double fps_;
    int64_t frame_number_;
    bool is_open_;
    bool use_hw_accel_;
    bool use_hw_frames_;
    cudaStream_t cuda_stream_;

    unsigned char* d_y_plane_;
    unsigned char* d_u_plane_;
    unsigned char* d_v_plane_;
    size_t y_plane_size_;
    size_t uv_plane_size_;

    bool write_frame(AVFrame* frame);
};

} // namespace pipeline
} // namespace vhs
