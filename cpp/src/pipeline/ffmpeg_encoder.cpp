/**
 * FFmpeg NVENC Encoder Implementation
 */

#include "ffmpeg_encoder.hpp"
#include "config/render_config.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/yuv_conversion.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixfmt.h>
}

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cstdio>

namespace vhs {
namespace pipeline {

FFmpegEncoder::FFmpegEncoder()
    : format_ctx_(nullptr)
    , codec_ctx_(nullptr)
    , hw_device_ctx_(nullptr)
    , hw_frames_ctx_(nullptr)
    , frame_(nullptr)
    , packet_(nullptr)
    , stream_(nullptr)
    , width_(0)
    , height_(0)
    , fps_(0.0)
    , frame_number_(0)
    , is_open_(false)
    , use_hw_accel_(true)
    , use_hw_frames_(true)
    , cuda_stream_(nullptr)
    , d_y_plane_(nullptr)
    , d_u_plane_(nullptr)
    , d_v_plane_(nullptr)
    , y_plane_size_(0)
    , uv_plane_size_(0)
{
}

FFmpegEncoder::~FFmpegEncoder() {
    close();
}

bool FFmpegEncoder::open(
    const std::string& path,
    int width,
    int height,
    double fps,
    int cq,
    const std::string& preset,
    bool use_hw_accel
) {
    close();
    width_ = width;
    height_ = height;
    fps_ = fps;

    if (!open_internal(path, width, height, fps, cq, preset, use_hw_accel)) {
        return false;
    }

    return true;
}

bool FFmpegEncoder::open_internal(
    const std::string& path,
    int width,
    int height,
    double fps,
    int cq,
    const std::string& preset,
    bool use_hw_accel
) {
    use_hw_accel_ = use_hw_accel;
    use_hw_frames_ = use_hw_accel_;

    const char* no_hw_frames = std::getenv("VHS_NVENC_NO_HWFRAMES");
    const char* safe_mode = std::getenv("VHS_SAFE_MODE");
    bool force_sw_frames = false;
    if ((no_hw_frames && no_hw_frames[0] != '\0' && no_hw_frames[0] != '0') ||
        (safe_mode && safe_mode[0] != '\0' && safe_mode[0] != '0')) {
        force_sw_frames = true;
    }
    if (force_sw_frames) {
        use_hw_frames_ = false;
    }

    // Allocate output context
    if (avformat_alloc_output_context2(&format_ctx_, nullptr, nullptr, path.c_str()) < 0) {
        fprintf(stderr, "[FFmpeg] Failed to create output context\n");
        return false;
    }

    // Find encoder
    const AVCodec* codec = nullptr;
    if (use_hw_accel_) {
        codec = avcodec_find_encoder_by_name("h264_nvenc");
        if (!codec) {
            fprintf(stderr, "[FFmpeg] h264_nvenc not available, falling back to libx264\n");
            use_hw_accel_ = false;
        }
    }

    if (!use_hw_accel_) {
        codec = avcodec_find_encoder_by_name("libx264");
        use_hw_frames_ = false;
    }

    if (!codec) {
        fprintf(stderr, "[FFmpeg] No encoder found\n");
        close();
        return false;
    }

    // Create stream
    stream_ = avformat_new_stream(format_ctx_, codec);
    if (!stream_) {
        fprintf(stderr, "[FFmpeg] Failed to create stream\n");
        close();
        return false;
    }

    // Allocate codec context
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        fprintf(stderr, "[FFmpeg] Failed to allocate codec context\n");
        close();
        return false;
    }

    // Configure codec
    codec_ctx_->width = width;
    codec_ctx_->height = height;
    codec_ctx_->time_base = AVRational{1, static_cast<int>(fps * 1000)};
    codec_ctx_->framerate = AVRational{static_cast<int>(fps * 1000), 1000};
    codec_ctx_->gop_size = std::max(1, static_cast<int>(std::round(fps / 2.0)));
    codec_ctx_->max_b_frames = ::vhs::VIDEO_B_FRAMES;
    codec_ctx_->bit_rate = ::vhs::VIDEO_BITRATE;
    codec_ctx_->rc_max_rate = ::vhs::VIDEO_MAXRATE;
    codec_ctx_->rc_buffer_size = ::vhs::VIDEO_BUFSIZE;
    codec_ctx_->profile = AV_PROFILE_H264_HIGH;
    codec_ctx_->color_primaries = AVCOL_PRI_BT709;
    codec_ctx_->color_trc = AVCOL_TRC_BT709;
    codec_ctx_->colorspace = AVCOL_SPC_BT709;
    codec_ctx_->color_range = AVCOL_RANGE_MPEG;

    if (format_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    codec_ctx_->flags |= AV_CODEC_FLAG_CLOSED_GOP;

    if (use_hw_accel_) {
        if (use_hw_frames_) {
            if (!init_hw_frames()) {
                fprintf(stderr, "[FFmpeg] HW frames init failed, using NVENC with system memory\n");
                use_hw_frames_ = false;
            }
        }
        if (!use_hw_frames_) {
            codec_ctx_->pix_fmt = AV_PIX_FMT_NV12;
        }
    } else {
        codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    }

    // Set encoder options
    if (use_hw_accel_) {
        av_opt_set(codec_ctx_->priv_data, "preset", preset.c_str(), 0);
        av_opt_set(codec_ctx_->priv_data, "tune", "hq", 0);
        av_opt_set(codec_ctx_->priv_data, "rc", "vbr", 0);
        av_opt_set_int(codec_ctx_->priv_data, "cq", cq, 0);
        av_opt_set(codec_ctx_->priv_data, "spatial_aq", "1", 0);
        av_opt_set(codec_ctx_->priv_data, "temporal_aq", "1", 0);
        av_opt_set(codec_ctx_->priv_data, "profile", "high", 0);
        av_opt_set_int(codec_ctx_->priv_data, "bf", ::vhs::VIDEO_B_FRAMES, 0);
        av_opt_set_int(codec_ctx_->priv_data, "g", codec_ctx_->gop_size, 0);
        av_opt_set_int(codec_ctx_->priv_data, "bitrate", ::vhs::VIDEO_BITRATE, 0);
        av_opt_set_int(codec_ctx_->priv_data, "maxrate", ::vhs::VIDEO_MAXRATE, 0);
        av_opt_set_int(codec_ctx_->priv_data, "bufsize", ::vhs::VIDEO_BUFSIZE, 0);
    } else {
        av_opt_set(codec_ctx_->priv_data, "preset", "fast", 0);
        av_opt_set_int(codec_ctx_->priv_data, "crf", cq, 0);
        av_opt_set(codec_ctx_->priv_data, "profile", "high", 0);
        av_opt_set_int(codec_ctx_->priv_data, "bf", ::vhs::VIDEO_B_FRAMES, 0);
        av_opt_set_int(codec_ctx_->priv_data, "g", codec_ctx_->gop_size, 0);
        av_opt_set(codec_ctx_->priv_data, "x264-params", "open-gop=0", 0);
    }

    // Open codec
    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        if (use_hw_accel_) {
            fprintf(stderr, "[FFmpeg] Failed to open NVENC, falling back to libx264\n");
            close();
            return open_internal(path, width, height, fps, cq, preset, false);
        }
        fprintf(stderr, "[FFmpeg] Failed to open encoder\n");
        close();
        return false;
    }

    // Copy codec params to stream
    if (avcodec_parameters_from_context(stream_->codecpar, codec_ctx_) < 0) {
        fprintf(stderr, "[FFmpeg] Failed to copy codec params\n");
        close();
        return false;
    }

    stream_->time_base = codec_ctx_->time_base;

    // Open output file
    if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&format_ctx_->pb, path.c_str(), AVIO_FLAG_WRITE) < 0) {
            fprintf(stderr, "[FFmpeg] Failed to open output file\n");
            close();
            return false;
        }
    }

    av_opt_set(format_ctx_->priv_data, "movflags", "+faststart", 0);

    // Write header
    if (avformat_write_header(format_ctx_, nullptr) < 0) {
        fprintf(stderr, "[FFmpeg] Failed to write header\n");
        close();
        return false;
    }

    // Allocate frame and packet
    frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();

    if (!frame_ || !packet_) {
        fprintf(stderr, "[FFmpeg] Failed to allocate frame/packet\n");
        close();
        return false;
    }

    frame_->format = codec_ctx_->pix_fmt;
    frame_->width = width;
    frame_->height = height;

    if (!use_hw_accel_ || !use_hw_frames_) {
        if (av_frame_get_buffer(frame_, 0) < 0) {
            fprintf(stderr, "[FFmpeg] Failed to allocate frame buffer\n");
            close();
            return false;
        }
    }

    cudaError_t stream_err = cudaStreamCreate(&cuda_stream_);
    if (stream_err != cudaSuccess) {
        fprintf(stderr, "[FFmpeg] Warning: failed to create CUDA stream: %s\n",
                cudaGetErrorString(stream_err));
        cuda_stream_ = nullptr;
    }

    // Allocate GPU YUV planes for fast conversion
    y_plane_size_ = static_cast<size_t>(width) * height;
    if (use_hw_accel_) {
        uv_plane_size_ = static_cast<size_t>(width) * (height / 2);
    } else {
        uv_plane_size_ = static_cast<size_t>(width / 2) * (height / 2);
    }
    CUDA_CHECK(cudaMalloc(&d_y_plane_, y_plane_size_));
    CUDA_CHECK(cudaMalloc(&d_u_plane_, uv_plane_size_));
    if (!use_hw_accel_) {
        CUDA_CHECK(cudaMalloc(&d_v_plane_, uv_plane_size_));
    }

    is_open_ = true;
    frame_number_ = 0;

    printf("[FFmpeg] Encoder opened: %dx%d @ %.2f fps, %s\n",
           width, height, fps,
           use_hw_accel_ ? (use_hw_frames_ ? "NVENC(CUDA)" : "NVENC(SW frames)") : "libx264");

    return true;
}

bool FFmpegEncoder::init_hw_frames() {
    if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA,
                               nullptr, nullptr, AV_CUDA_USE_CURRENT_CONTEXT) < 0) {
        return false;
    }

    hw_frames_ctx_ = av_hwframe_ctx_alloc(hw_device_ctx_);
    if (!hw_frames_ctx_) {
        return false;
    }

    auto* frames_ctx = reinterpret_cast<AVHWFramesContext*>(hw_frames_ctx_->data);
    frames_ctx->format = AV_PIX_FMT_CUDA;
    frames_ctx->sw_format = AV_PIX_FMT_NV12;
    frames_ctx->width = width_;
    frames_ctx->height = height_;
    frames_ctx->initial_pool_size = 8;

    if (av_hwframe_ctx_init(hw_frames_ctx_) < 0) {
        av_buffer_unref(&hw_frames_ctx_);
        return false;
    }

    codec_ctx_->pix_fmt = AV_PIX_FMT_CUDA;
    codec_ctx_->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);
    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);

    return true;
}

void FFmpegEncoder::close() {
    if (is_open_) {
        flush();
        av_write_trailer(format_ctx_);
    }

    if (packet_) {
        av_packet_free(&packet_);
        packet_ = nullptr;
    }
    if (frame_) {
        av_frame_free(&frame_);
        frame_ = nullptr;
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }
    if (hw_frames_ctx_) {
        av_buffer_unref(&hw_frames_ctx_);
        hw_frames_ctx_ = nullptr;
    }
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
    if (format_ctx_) {
        if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx_->pb);
        }
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
    }

    if (d_y_plane_) {
        cudaFree(d_y_plane_);
        d_y_plane_ = nullptr;
    }
    if (d_u_plane_) {
        cudaFree(d_u_plane_);
        d_u_plane_ = nullptr;
    }
    if (d_v_plane_) {
        cudaFree(d_v_plane_);
        d_v_plane_ = nullptr;
    }

    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }

    is_open_ = false;
}

bool FFmpegEncoder::encode_frame(const unsigned char* d_input, int64_t pts) {
    if (!is_open_) return false;

    if (use_hw_accel_) {
        cudaStream_t stream = cuda_stream_ ? cuda_stream_ : 0;
        if (!use_hw_frames_) {
            if (av_frame_make_writable(frame_) < 0) {
                return false;
            }

            if (!d_y_plane_ || !d_u_plane_) {
                fprintf(stderr, "[FFmpeg] GPU NV12 planes not initialized\n");
                return false;
            }

            vhs::cuda::bgr_to_nv12(
                d_input,
                d_y_plane_,
                d_u_plane_,
                width_,
                height_,
                stream
            );

            CUDA_CHECK(cudaMemcpy2DAsync(
                frame_->data[0],
                frame_->linesize[0],
                d_y_plane_,
                width_,
                width_,
                height_,
                cudaMemcpyDeviceToHost,
                stream
            ));
            CUDA_CHECK(cudaMemcpy2DAsync(
                frame_->data[1],
                frame_->linesize[1],
                d_u_plane_,
                width_,
                width_,
                height_ / 2,
                cudaMemcpyDeviceToHost,
                stream
            ));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            frame_->pts = pts;
            frame_number_++;
            return write_frame(frame_);
        }

        if (!d_y_plane_ || !d_u_plane_) {
            fprintf(stderr, "[FFmpeg] GPU NV12 planes not initialized\n");
            return false;
        }

        if (!frame_) {
            frame_ = av_frame_alloc();
            if (!frame_) {
                fprintf(stderr, "[FFmpeg] Failed to allocate frame\n");
                return false;
            }
        }

        av_frame_unref(frame_);
        frame_->format = codec_ctx_->pix_fmt;
        frame_->width = width_;
        frame_->height = height_;

        if (av_hwframe_get_buffer(codec_ctx_->hw_frames_ctx, frame_, 0) < 0) {
            fprintf(stderr, "[FFmpeg] Failed to get HW frame buffer\n");
            return false;
        }

        vhs::cuda::bgr_to_nv12(
            d_input,
            d_y_plane_,
            d_u_plane_,
            width_,
            height_,
            stream
        );

            // Copy NV12 planes into HW frame (device-to-device)
            CUDA_CHECK(cudaMemcpy2DAsync(
                frame_->data[0],
                frame_->linesize[0],
                d_y_plane_,
            width_,
            width_,
            height_,
            cudaMemcpyDeviceToDevice,
            stream
        ));
        CUDA_CHECK(cudaMemcpy2DAsync(
            frame_->data[1],
            frame_->linesize[1],
            d_u_plane_,
            width_,
            width_,
            height_ / 2,
            cudaMemcpyDeviceToDevice,
            stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        frame_->pts = pts;
        frame_number_++;

        return write_frame(frame_);
    }

    // Make frame writable
    if (av_frame_make_writable(frame_) < 0) {
        return false;
    }

    if (!d_y_plane_ || !d_u_plane_ || !d_v_plane_) {
        fprintf(stderr, "[FFmpeg] GPU YUV planes not initialized\n");
        return false;
    }

    vhs::cuda::bgr_to_yuv420(
        d_input,
        d_y_plane_,
        d_u_plane_,
        d_v_plane_,
        width_,
        height_,
        cuda_stream_ ? cuda_stream_ : 0
    );

    // Copy Y plane
    CUDA_CHECK(cudaMemcpy2DAsync(
        frame_->data[0],
        frame_->linesize[0],
        d_y_plane_,
        width_,
        width_,
        height_,
        cudaMemcpyDeviceToHost,
        cuda_stream_ ? cuda_stream_ : 0
    ));

    // Copy U plane
    CUDA_CHECK(cudaMemcpy2DAsync(
        frame_->data[1],
        frame_->linesize[1],
        d_u_plane_,
        width_ / 2,
        width_ / 2,
        height_ / 2,
        cudaMemcpyDeviceToHost,
        cuda_stream_ ? cuda_stream_ : 0
    ));

    // Copy V plane
    CUDA_CHECK(cudaMemcpy2DAsync(
        frame_->data[2],
        frame_->linesize[2],
        d_v_plane_,
        width_ / 2,
        width_ / 2,
        height_ / 2,
        cudaMemcpyDeviceToHost,
        cuda_stream_ ? cuda_stream_ : 0
    ));
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_ ? cuda_stream_ : 0));

    frame_->pts = pts;
    frame_number_++;

    return write_frame(frame_);
}

bool FFmpegEncoder::write_frame(AVFrame* frame) {
    // Send frame to encoder
    if (avcodec_send_frame(codec_ctx_, frame) < 0) {
        return false;
    }

    // Receive and write packets
    while (true) {
        int ret = avcodec_receive_packet(codec_ctx_, packet_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            return false;
        }

        // Rescale timestamp
        av_packet_rescale_ts(packet_, codec_ctx_->time_base, stream_->time_base);
        packet_->stream_index = stream_->index;

        // Write packet
        if (av_interleaved_write_frame(format_ctx_, packet_) < 0) {
            return false;
        }

        av_packet_unref(packet_);
    }

    return true;
}

bool FFmpegEncoder::flush() {
    if (!is_open_) return false;
    return write_frame(nullptr);
}

} // namespace pipeline
} // namespace vhs
