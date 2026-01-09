/**
 * FFmpeg NVDEC Decoder Implementation
 */

#include "ffmpeg_decoder.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/yuv_conversion.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

namespace vhs {
namespace pipeline {

FFmpegDecoder::FFmpegDecoder()
    : format_ctx_(nullptr)
    , codec_ctx_(nullptr)
    , hw_device_ctx_(nullptr)
    , frame_(nullptr)
    , sw_frame_(nullptr)
    , packet_(nullptr)
    , video_stream_idx_(-1)
    , width_(0)
    , height_(0)
    , fps_(0.0)
    , duration_(0.0)
    , frame_count_(0)
    , is_open_(false)
    , use_hw_accel_(true)
    , cuda_stream_(nullptr)
    , d_y_(nullptr)
    , d_u_(nullptr)
    , d_v_(nullptr)
    , d_uv_(nullptr)
    , y_size_(0)
    , u_size_(0)
    , v_size_(0)
    , uv_size_(0)
    , cached_pix_fmt_(-1)
{
}

FFmpegDecoder::~FFmpegDecoder() {
    close();
}

bool FFmpegDecoder::open(const std::string& path, bool use_hw_accel) {
    close();
    use_hw_accel_ = use_hw_accel;
    cached_pix_fmt_ = -1;

    cudaError_t stream_err = cudaStreamCreate(&cuda_stream_);
    if (stream_err != cudaSuccess) {
        fprintf(stderr, "[FFmpeg] Warning: failed to create CUDA stream: %s\n",
                cudaGetErrorString(stream_err));
        cuda_stream_ = nullptr;
    }

    // Open input file
    if (avformat_open_input(&format_ctx_, path.c_str(), nullptr, nullptr) < 0) {
        fprintf(stderr, "[FFmpeg] Failed to open: %s\n", path.c_str());
        return false;
    }

    // Find stream info
    if (avformat_find_stream_info(format_ctx_, nullptr) < 0) {
        fprintf(stderr, "[FFmpeg] Failed to find stream info\n");
        close();
        return false;
    }

    // Find video stream
    video_stream_idx_ = -1;
    for (unsigned int i = 0; i < format_ctx_->nb_streams; i++) {
        if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx_ = i;
            break;
        }
    }

    if (video_stream_idx_ < 0) {
        fprintf(stderr, "[FFmpeg] No video stream found\n");
        close();
        return false;
    }

    AVStream* video_stream = format_ctx_->streams[video_stream_idx_];
    AVCodecParameters* codecpar = video_stream->codecpar;

    // Find decoder
    const AVCodec* codec = nullptr;
    if (use_hw_accel_) {
        // Try hardware decoder first
        codec = avcodec_find_decoder_by_name("h264_cuvid");
        if (!codec) {
            fprintf(stderr, "[FFmpeg] h264_cuvid not available, falling back to software\n");
            use_hw_accel_ = false;
        }
    }

    if (!use_hw_accel_) {
        codec = avcodec_find_decoder(codecpar->codec_id);
    }

    if (!codec) {
        fprintf(stderr, "[FFmpeg] No decoder found\n");
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

    // Copy codec parameters
    if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
        fprintf(stderr, "[FFmpeg] Failed to copy codec parameters\n");
        close();
        return false;
    }
    codec_ctx_->pkt_timebase = video_stream->time_base;

    // Initialize hardware acceleration if available
    if (use_hw_accel_ && !init_hw_decoder()) {
        fprintf(stderr, "[FFmpeg] Hardware acceleration init failed, falling back\n");
        use_hw_accel_ = false;
    }

    // Open codec
    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        // If hardware codec failed, try software fallback
        if (use_hw_accel_) {
            fprintf(stderr, "[FFmpeg] Hardware codec failed, trying software fallback\n");
            avcodec_free_context(&codec_ctx_);

            // Find software decoder
            codec = avcodec_find_decoder(codecpar->codec_id);
            if (!codec) {
                fprintf(stderr, "[FFmpeg] No software decoder found\n");
                close();
                return false;
            }

            codec_ctx_ = avcodec_alloc_context3(codec);
            if (!codec_ctx_ || avcodec_parameters_to_context(codec_ctx_, codecpar) < 0) {
                fprintf(stderr, "[FFmpeg] Failed to setup software decoder\n");
                close();
                return false;
            }

            use_hw_accel_ = false;

            if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
                fprintf(stderr, "[FFmpeg] Failed to open software codec\n");
                close();
                return false;
            }
        } else {
            fprintf(stderr, "[FFmpeg] Failed to open codec\n");
            close();
            return false;
        }
    }

    // Allocate frames and packet
    frame_ = av_frame_alloc();
    sw_frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();

    if (!frame_ || !sw_frame_ || !packet_) {
        fprintf(stderr, "[FFmpeg] Failed to allocate frame/packet\n");
        close();
        return false;
    }

    // Store video properties
    width_ = codecpar->width;
    height_ = codecpar->height;
    fps_ = av_q2d(video_stream->avg_frame_rate);
    duration_ = static_cast<double>(format_ctx_->duration) / AV_TIME_BASE;
    frame_count_ = video_stream->nb_frames;
    if (frame_count_ <= 0) {
        frame_count_ = static_cast<int64_t>(duration_ * fps_);
    }

    is_open_ = true;
    printf("[FFmpeg] Opened: %dx%d @ %.2f fps, %s\n",
           width_, height_, fps_,
           use_hw_accel_ ? "NVDEC" : "CPU");

    return true;
}

bool FFmpegDecoder::init_hw_decoder() {
    // Create CUDA device context
    if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA,
                                nullptr, nullptr, AV_CUDA_USE_CURRENT_CONTEXT) < 0) {
        return false;
    }

    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    return true;
}

void FFmpegDecoder::release_device_buffers() {
    if (d_y_) {
        cudaFree(d_y_);
        d_y_ = nullptr;
    }
    if (d_u_) {
        cudaFree(d_u_);
        d_u_ = nullptr;
    }
    if (d_v_) {
        cudaFree(d_v_);
        d_v_ = nullptr;
    }
    if (d_uv_) {
        cudaFree(d_uv_);
        d_uv_ = nullptr;
    }
    y_size_ = 0;
    u_size_ = 0;
    v_size_ = 0;
    uv_size_ = 0;
    cached_pix_fmt_ = -1;
}

bool FFmpegDecoder::ensure_device_buffers(int pix_fmt) {
    if (cached_pix_fmt_ == pix_fmt && d_y_) {
        return true;
    }

    release_device_buffers();

    const size_t frame_pixels = static_cast<size_t>(width_) * height_;
    y_size_ = frame_pixels;

    cudaError_t err = cudaMalloc(&d_y_, y_size_);
    if (err != cudaSuccess) {
        fprintf(stderr, "[FFmpeg] Failed to allocate Y plane: %s\n",
                cudaGetErrorString(err));
        release_device_buffers();
        return false;
    }

    if (pix_fmt == AV_PIX_FMT_NV12) {
        uv_size_ = frame_pixels / 2;
        err = cudaMalloc(&d_uv_, uv_size_);
        if (err != cudaSuccess) {
            fprintf(stderr, "[FFmpeg] Failed to allocate UV plane: %s\n",
                    cudaGetErrorString(err));
            release_device_buffers();
            return false;
        }
    } else if (pix_fmt == AV_PIX_FMT_YUV420P || pix_fmt == AV_PIX_FMT_YUVJ420P) {
        u_size_ = (frame_pixels / 4);
        v_size_ = u_size_;
        err = cudaMalloc(&d_u_, u_size_);
        if (err != cudaSuccess) {
            fprintf(stderr, "[FFmpeg] Failed to allocate U plane: %s\n",
                    cudaGetErrorString(err));
            release_device_buffers();
            return false;
        }
        err = cudaMalloc(&d_v_, v_size_);
        if (err != cudaSuccess) {
            fprintf(stderr, "[FFmpeg] Failed to allocate V plane: %s\n",
                    cudaGetErrorString(err));
            release_device_buffers();
            return false;
        }
    } else if (pix_fmt == AV_PIX_FMT_YUV444P || pix_fmt == AV_PIX_FMT_YUVJ444P) {
        u_size_ = frame_pixels;
        v_size_ = frame_pixels;
        err = cudaMalloc(&d_u_, u_size_);
        if (err != cudaSuccess) {
            fprintf(stderr, "[FFmpeg] Failed to allocate U plane: %s\n",
                    cudaGetErrorString(err));
            release_device_buffers();
            return false;
        }
        err = cudaMalloc(&d_v_, v_size_);
        if (err != cudaSuccess) {
            fprintf(stderr, "[FFmpeg] Failed to allocate V plane: %s\n",
                    cudaGetErrorString(err));
            release_device_buffers();
            return false;
        }
    } else {
        release_device_buffers();
        return false;
    }

    cached_pix_fmt_ = pix_fmt;
    return true;
}

void FFmpegDecoder::close() {
    if (packet_) {
        av_packet_free(&packet_);
        packet_ = nullptr;
    }
    if (sw_frame_) {
        av_frame_free(&sw_frame_);
        sw_frame_ = nullptr;
    }
    if (frame_) {
        av_frame_free(&frame_);
        frame_ = nullptr;
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
    if (format_ctx_) {
        avformat_close_input(&format_ctx_);
        format_ctx_ = nullptr;
    }

    release_device_buffers();

    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }

    is_open_ = false;
}

bool FFmpegDecoder::decode_frame(unsigned char* d_output, int64_t* pts) {
    if (!is_open_) return false;

    while (true) {
        // Read packet
        if (av_read_frame(format_ctx_, packet_) < 0) {
            return false;  // End of file
        }

        if (packet_->stream_index != video_stream_idx_) {
            av_packet_unref(packet_);
            continue;
        }

        // Send packet to decoder
        if (avcodec_send_packet(codec_ctx_, packet_) < 0) {
            av_packet_unref(packet_);
            continue;
        }

        // Receive frame
        int ret = avcodec_receive_frame(codec_ctx_, frame_);
        av_packet_unref(packet_);

        if (ret == AVERROR(EAGAIN)) {
            continue;  // Need more data
        } else if (ret < 0) {
            return false;
        }

        // Got a frame
        if (pts) {
            *pts = frame_->pts;
        }

        // Determine source format and where the planes live (CPU or CUDA)
        AVFrame* src_frame = frame_;
        AVPixelFormat pix_fmt = static_cast<AVPixelFormat>(frame_->format);
        AVPixelFormat sw_fmt = pix_fmt;
        bool frame_on_device = (pix_fmt == AV_PIX_FMT_CUDA);

        if (frame_on_device) {
            if (frame_->hw_frames_ctx && frame_->hw_frames_ctx->data) {
                auto* frames_ctx = reinterpret_cast<AVHWFramesContext*>(frame_->hw_frames_ctx->data);
                sw_fmt = static_cast<AVPixelFormat>(frames_ctx->sw_format);
            } else {
                sw_fmt = AV_PIX_FMT_NV12;
            }
        }

        cudaStream_t stream = cuda_stream_ ? cuda_stream_ : 0;

        if (sw_fmt == AV_PIX_FMT_NV12 ||
            sw_fmt == AV_PIX_FMT_YUV420P || sw_fmt == AV_PIX_FMT_YUVJ420P ||
            sw_fmt == AV_PIX_FMT_YUV444P || sw_fmt == AV_PIX_FMT_YUVJ444P) {
            if (!ensure_device_buffers(sw_fmt)) {
                fprintf(stderr, "[FFmpeg] Failed to allocate GPU buffers for YUV conversion\n");
                return false;
            }

            cudaMemcpyKind y_kind = frame_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
            CUDA_CHECK(cudaMemcpy2DAsync(
                d_y_,
                width_,
                frame_->data[0],
                frame_->linesize[0],
                width_,
                height_,
                y_kind,
                stream
            ));

            if (sw_fmt == AV_PIX_FMT_NV12) {
                cudaMemcpyKind uv_kind = frame_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
                CUDA_CHECK(cudaMemcpy2DAsync(
                    d_uv_,
                    width_,
                    frame_->data[1],
                    frame_->linesize[1],
                    width_,
                    height_ / 2,
                    uv_kind,
                    stream
                ));
                vhs::cuda::nv12_to_bgr(d_y_, d_uv_, d_output, width_, height_, stream);
            } else if (sw_fmt == AV_PIX_FMT_YUV420P || sw_fmt == AV_PIX_FMT_YUVJ420P) {
                int uv_width = width_ / 2;
                int uv_height = height_ / 2;
                cudaMemcpyKind uv_kind = frame_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
                CUDA_CHECK(cudaMemcpy2DAsync(
                    d_u_,
                    uv_width,
                    frame_->data[1],
                    frame_->linesize[1],
                    uv_width,
                    uv_height,
                    uv_kind,
                    stream
                ));
                CUDA_CHECK(cudaMemcpy2DAsync(
                    d_v_,
                    uv_width,
                    frame_->data[2],
                    frame_->linesize[2],
                    uv_width,
                    uv_height,
                    uv_kind,
                    stream
                ));
                vhs::cuda::yuv420_to_bgr(d_y_, d_u_, d_v_, d_output, width_, height_, stream);
            } else {
                cudaMemcpyKind uv_kind = frame_on_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
                CUDA_CHECK(cudaMemcpy2DAsync(
                    d_u_,
                    width_,
                    frame_->data[1],
                    frame_->linesize[1],
                    width_,
                    height_,
                    uv_kind,
                    stream
                ));
                CUDA_CHECK(cudaMemcpy2DAsync(
                    d_v_,
                    width_,
                    frame_->data[2],
                    frame_->linesize[2],
                    width_,
                    height_,
                    uv_kind,
                    stream
                ));
                vhs::cuda::yuv444_to_bgr(d_y_, d_u_, d_v_, d_output, width_, height_, stream);
            }

            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            // Fallback: CPU conversion for unknown formats
            if (frame_on_device) {
                if (av_hwframe_transfer_data(sw_frame_, frame_, 0) < 0) {
                    return false;
                }
                src_frame = sw_frame_;
            }
            size_t rgb_size = static_cast<size_t>(width_) * height_ * 3;
            static std::vector<unsigned char> rgb_buffer;
            if (rgb_buffer.size() != rgb_size) {
                rgb_buffer.resize(rgb_size);
            }

            for (int y = 0; y < height_; y++) {
                for (int x = 0; x < width_; x++) {
                    int Y = src_frame->data[0][y * src_frame->linesize[0] + x];
                    int U = 128;
                    int V = 128;

                    int C = Y - 16;
                    int D = U - 128;
                    int E = V - 128;

                    int R = (298 * C + 409 * E + 128) >> 8;
                    int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
                    int B = (298 * C + 516 * D + 128) >> 8;

                    R = R < 0 ? 0 : (R > 255 ? 255 : R);
                    G = G < 0 ? 0 : (G > 255 ? 255 : G);
                    B = B < 0 ? 0 : (B > 255 ? 255 : B);

                    int idx = (y * width_ + x) * 3;
                    rgb_buffer[idx + 0] = static_cast<unsigned char>(B);
                    rgb_buffer[idx + 1] = static_cast<unsigned char>(G);
                    rgb_buffer[idx + 2] = static_cast<unsigned char>(R);
                }
            }

            CUDA_CHECK(cudaMemcpy(d_output, rgb_buffer.data(), rgb_size,
                                 cudaMemcpyHostToDevice));
        }

        av_frame_unref(frame_);
        return true;
    }
}

bool FFmpegDecoder::seek(double timestamp_seconds) {
    if (!is_open_) return false;

    int64_t timestamp = static_cast<int64_t>(timestamp_seconds * AV_TIME_BASE);
    return av_seek_frame(format_ctx_, -1, timestamp, AVSEEK_FLAG_BACKWARD) >= 0;
}

} // namespace pipeline
} // namespace vhs
