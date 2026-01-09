/**
 * Image loader and basic processing helpers (BGR).
 */

#include "image_loader.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

#include <cstdio>
#include <cstring>

namespace vhs {
namespace utils {

static bool decode_first_frame(
    const std::string& path,
    ImageData& out,
    AVPixelFormat dst_format,
    int bytes_per_pixel
) {
    AVFormatContext* format_ctx = nullptr;
    if (avformat_open_input(&format_ctx, path.c_str(), nullptr, nullptr) < 0) {
        fprintf(stderr, "[Image] Failed to open: %s\n", path.c_str());
        return false;
    }

    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        fprintf(stderr, "[Image] Failed to read stream info: %s\n", path.c_str());
        avformat_close_input(&format_ctx);
        return false;
    }

    int stream_idx = -1;
    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            stream_idx = static_cast<int>(i);
            break;
        }
    }

    if (stream_idx < 0) {
        fprintf(stderr, "[Image] No video stream found: %s\n", path.c_str());
        avformat_close_input(&format_ctx);
        return false;
    }

    AVCodecParameters* codecpar = format_ctx->streams[stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        fprintf(stderr, "[Image] No decoder for: %s\n", path.c_str());
        avformat_close_input(&format_ctx);
        return false;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx || avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
        fprintf(stderr, "[Image] Failed to init codec context\n");
        avformat_close_input(&format_ctx);
        if (codec_ctx) avcodec_free_context(&codec_ctx);
        return false;
    }

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        fprintf(stderr, "[Image] Failed to open codec\n");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return false;
    }

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* dst_frame = av_frame_alloc();
    if (!packet || !frame || !dst_frame) {
        fprintf(stderr, "[Image] Failed to allocate frames\n");
        if (packet) av_packet_free(&packet);
        if (frame) av_frame_free(&frame);
        if (dst_frame) av_frame_free(&dst_frame);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return false;
    }

    bool got_frame = false;
    while (!got_frame && av_read_frame(format_ctx, packet) >= 0) {
        if (packet->stream_index != stream_idx) {
            av_packet_unref(packet);
            continue;
        }

        if (avcodec_send_packet(codec_ctx, packet) < 0) {
            av_packet_unref(packet);
            continue;
        }

        int ret = avcodec_receive_frame(codec_ctx, frame);
        av_packet_unref(packet);

        if (ret == AVERROR(EAGAIN)) {
            continue;
        } else if (ret < 0) {
            break;
        }

        got_frame = true;
    }

    if (!got_frame) {
        fprintf(stderr, "[Image] Failed to decode frame: %s\n", path.c_str());
        av_packet_free(&packet);
        av_frame_free(&frame);
        av_frame_free(&dst_frame);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return false;
    }

    int width = frame->width;
    int height = frame->height;

    dst_frame->format = dst_format;
    dst_frame->width = width;
    dst_frame->height = height;

    if (av_frame_get_buffer(dst_frame, 0) < 0) {
        fprintf(stderr, "[Image] Failed to allocate image buffer\n");
        av_packet_free(&packet);
        av_frame_free(&frame);
        av_frame_free(&dst_frame);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return false;
    }

    SwsContext* sws = sws_getContext(
        width, height, static_cast<AVPixelFormat>(frame->format),
        width, height, dst_format,
        SWS_BICUBIC, nullptr, nullptr, nullptr
    );

    if (!sws) {
        fprintf(stderr, "[Image] Failed to init sws context\n");
        av_packet_free(&packet);
        av_frame_free(&frame);
        av_frame_free(&dst_frame);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        return false;
    }

    sws_scale(sws,
              frame->data, frame->linesize,
              0, height,
              dst_frame->data, dst_frame->linesize);

    out.width = width;
    out.height = height;
    out.pixels.resize(static_cast<size_t>(width) * height * bytes_per_pixel);

    for (int y = 0; y < height; y++) {
        std::memcpy(
            out.pixels.data() + static_cast<size_t>(y) * width * bytes_per_pixel,
            dst_frame->data[0] + y * dst_frame->linesize[0],
            static_cast<size_t>(width) * bytes_per_pixel
        );
    }

    sws_freeContext(sws);
    av_packet_free(&packet);
    av_frame_free(&frame);
    av_frame_free(&dst_frame);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);

    return true;
}

bool load_image_bgr(const std::string& path, ImageData& out) {
    return decode_first_frame(path, out, AV_PIX_FMT_BGR24, 3);
}

bool load_image_bgra(const std::string& path, ImageData& out) {
    return decode_first_frame(path, out, AV_PIX_FMT_BGRA, 4);
}

bool resize_image_bgr(const ImageData& src, int dst_width, int dst_height, ImageData& out) {
    if (src.width <= 0 || src.height <= 0 || src.pixels.empty()) {
        return false;
    }

    out.width = dst_width;
    out.height = dst_height;
    out.pixels.resize(static_cast<size_t>(dst_width) * dst_height * 3);

    SwsContext* sws = sws_getContext(
        src.width, src.height, AV_PIX_FMT_BGR24,
        dst_width, dst_height, AV_PIX_FMT_BGR24,
        SWS_BICUBIC, nullptr, nullptr, nullptr
    );

    if (!sws) {
        fprintf(stderr, "[Image] Failed to init resize sws\n");
        return false;
    }

    const uint8_t* src_slices[] = { src.pixels.data() };
    int src_stride[] = { src.width * 3 };
    uint8_t* dst_slices[] = { out.pixels.data() };
    int dst_stride[] = { dst_width * 3 };

    sws_scale(sws, src_slices, src_stride, 0, src.height, dst_slices, dst_stride);
    sws_freeContext(sws);

    return true;
}

bool resize_image_bgra(const ImageData& src, int dst_width, int dst_height, ImageData& out) {
    if (src.width <= 0 || src.height <= 0 || src.pixels.empty()) {
        return false;
    }

    out.width = dst_width;
    out.height = dst_height;
    out.pixels.resize(static_cast<size_t>(dst_width) * dst_height * 4);

    SwsContext* sws = sws_getContext(
        src.width, src.height, AV_PIX_FMT_BGRA,
        dst_width, dst_height, AV_PIX_FMT_BGRA,
        SWS_BICUBIC, nullptr, nullptr, nullptr
    );

    if (!sws) {
        fprintf(stderr, "[Image] Failed to init resize sws (BGRA)\n");
        return false;
    }

    const uint8_t* src_slices[] = { src.pixels.data() };
    int src_stride[] = { src.width * 4 };
    uint8_t* dst_slices[] = { out.pixels.data() };
    int dst_stride[] = { dst_width * 4 };

    sws_scale(sws, src_slices, src_stride, 0, src.height, dst_slices, dst_stride);
    sws_freeContext(sws);

    return true;
}

bool crop_image_bgr(const ImageData& src, int x, int y, int dst_width, int dst_height, ImageData& out) {
    if (src.width <= 0 || src.height <= 0 || src.pixels.empty()) {
        return false;
    }

    int start_x = x < 0 ? 0 : x;
    int start_y = y < 0 ? 0 : y;
    int end_x = start_x + dst_width;
    int end_y = start_y + dst_height;

    if (end_x > src.width || end_y > src.height) {
        return false;
    }

    out.width = dst_width;
    out.height = dst_height;
    out.pixels.resize(static_cast<size_t>(dst_width) * dst_height * 3);

    for (int row = 0; row < dst_height; row++) {
        const unsigned char* src_ptr = src.pixels.data() + ((start_y + row) * src.width + start_x) * 3;
        unsigned char* dst_ptr = out.pixels.data() + static_cast<size_t>(row) * dst_width * 3;
        std::memcpy(dst_ptr, src_ptr, static_cast<size_t>(dst_width) * 3);
    }

    return true;
}

bool crop_image_bgra(const ImageData& src, int x, int y, int dst_width, int dst_height, ImageData& out) {
    if (src.width <= 0 || src.height <= 0 || src.pixels.empty()) {
        return false;
    }

    int start_x = x < 0 ? 0 : x;
    int start_y = y < 0 ? 0 : y;
    int end_x = start_x + dst_width;
    int end_y = start_y + dst_height;

    if (end_x > src.width || end_y > src.height) {
        return false;
    }

    out.width = dst_width;
    out.height = dst_height;
    out.pixels.resize(static_cast<size_t>(dst_width) * dst_height * 4);

    for (int row = 0; row < dst_height; row++) {
        const unsigned char* src_ptr = src.pixels.data() + ((start_y + row) * src.width + start_x) * 4;
        unsigned char* dst_ptr = out.pixels.data() + static_cast<size_t>(row) * dst_width * 4;
        std::memcpy(dst_ptr, src_ptr, static_cast<size_t>(dst_width) * 4);
    }

    return true;
}

} // namespace utils
} // namespace vhs
