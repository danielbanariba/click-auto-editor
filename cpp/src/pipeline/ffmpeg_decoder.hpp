#pragma once

#include <string>
#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

// Forward declarations for FFmpeg types
struct AVFormatContext;
struct AVCodecContext;
struct AVBufferRef;
struct AVFrame;
struct AVPacket;

namespace vhs {
namespace pipeline {

/**
 * FFmpeg NVDEC Decoder
 *
 * Decodes video using NVIDIA hardware decoder (NVDEC).
 * Frames are decoded directly to CUDA device memory.
 */
class FFmpegDecoder {
public:
    FFmpegDecoder();
    ~FFmpegDecoder();

    // Open video file for decoding
    bool open(const std::string& path, bool use_hw_accel = true);

    // Close and release resources
    void close();

    // Decode next frame to device memory
    // Returns false if no more frames
    bool decode_frame(unsigned char* d_output, int64_t* pts = nullptr);

    // Seek to specific timestamp
    bool seek(double timestamp_seconds);

    // Properties
    int width() const { return width_; }
    int height() const { return height_; }
    double fps() const { return fps_; }
    double duration() const { return duration_; }
    int64_t frame_count() const { return frame_count_; }
    bool is_open() const { return is_open_; }

private:
    AVFormatContext* format_ctx_;
    AVCodecContext* codec_ctx_;
    AVBufferRef* hw_device_ctx_;
    AVFrame* frame_;
    AVFrame* sw_frame_;
    AVPacket* packet_;

    int video_stream_idx_;
    int width_;
    int height_;
    double fps_;
    double duration_;
    int64_t frame_count_;
    bool is_open_;
    bool use_hw_accel_;
    cudaStream_t cuda_stream_;

    unsigned char* d_y_;
    unsigned char* d_u_;
    unsigned char* d_v_;
    unsigned char* d_uv_;
    size_t y_size_;
    size_t u_size_;
    size_t v_size_;
    size_t uv_size_;
    int cached_pix_fmt_;

    bool init_hw_decoder();
    void release_device_buffers();
    bool ensure_device_buffers(int pix_fmt);
};

} // namespace pipeline
} // namespace vhs
