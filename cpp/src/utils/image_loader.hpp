#pragma once

#include <string>
#include <vector>

namespace vhs {
namespace utils {

struct ImageData {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;  // BGR or BGRA depending on loader
};

bool load_image_bgr(const std::string& path, ImageData& out);
bool load_image_bgra(const std::string& path, ImageData& out);
bool resize_image_bgr(const ImageData& src, int dst_width, int dst_height, ImageData& out);
bool resize_image_bgra(const ImageData& src, int dst_width, int dst_height, ImageData& out);
bool crop_image_bgr(const ImageData& src, int x, int y, int dst_width, int dst_height, ImageData& out);
bool crop_image_bgra(const ImageData& src, int x, int y, int dst_width, int dst_height, ImageData& out);

} // namespace utils
} // namespace vhs
