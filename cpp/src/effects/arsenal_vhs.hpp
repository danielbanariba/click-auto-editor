/**
 * Arsenal VHS - the look of the per-frame effect chain.
 *
 * Ported from the Arsenal VHS effect of a FilmCraft build (MIT OR Apache-2.0).
 *
 * The colour stage reads a packed 8-bit BGR frame, takes it to Y'CbCr (ITU-R BT.601), filters luma and
 * the two colour planes apart (separable Gaussians as three box passes, O(1) per pixel), delays the
 * planes, sharpens luma, mixes the colour back with the clean colour, and grades the result.
 *
 * The damage stage (arsenal_vhs_damage.hpp) adds the tape's noise, dust and flecks, pop lines, Frame
 * Jitter, the Wrinkle band, the tape frame and Post Noise.
 *
 * Lengths are pixels of the 2560-wide frame the look was calibrated on, times the frame's
 * width / 2560 (1.5 at 3840x2160). Controls and defaults are those of the calibrated look; the
 * controls that are neutral in that look (Chroma Offset Y, Chroma Weave X / Y, Speed X / Y, Adjust
 * Noise Scale, Interlace Error, Image Offset, Tape Style) are not ported and read as neutral.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>

namespace vhs {
namespace effects {
namespace arsenal {

/**
 * Width, in pixels, at which every length constant of the look is defined: the 2560 x 1440 frame of
 * the FilmCraft reference. A frame of width w scales all lengths by w / kReferenceWidth (1.5 at
 * 3840x2160); the height plays no part.
 */
constexpr double kReferenceWidth = 2560.0;

/** The controls of the look, in the units and ranges of the effect (the defaults are the calibrated look). */
struct Controls {
    // Color Controls: Black Level, Saturation, the two blurs and Sharpen are switched by this.
    bool color_controls = true;
    float black_level = 10.0f;
    float saturation = 100.0f;
    float luma_blur = 2.0f;
    float chroma_blur = 5.0f;
    float sharpen_radius = 5.0f;
    float sharpen_mix = 5.0f;
    // Color Offset
    float chroma_offset_x = 10.0f;
    float color_mix = 70.0f;
    // Channel Controls
    float red_offset = 10.0f;
    float green_offset = 0.0f;
    float blue_offset = -10.0f;
    float red_blur = 25.0f;
    float green_blur = 55.0f;
    float blue_blur = 45.0f;
    // Tape Damage: Enable Tape Damage switches the tape noise (Type 5, the look), the dust specks and
    // flecks, Pop Line Frequency and Frame Jitter. Tape Noise Type 5 is the only type ported.
    bool tape_damage = true;
    float pop_line_freq = 25.0f;
    float frame_jitter = 2.0f;
    // Tape Wrinkle
    bool wrinkle = true;
    float wrinkle_speed = 5.0f;
    float wrinkle_chance = 40.0f;
    float wrinkle_radius = 5.0f;
    // Output
    bool tape_frame = true;  // Frame Style 0 draws the tape frame, Style 1 draws nothing
    float post_noise = 3.0f;
    float seed = 9300.0f;  // Random Seed
};

/**
 * Which frame is drawn. Everything that varies in time is a hash of these, so a repeated frame is
 * identical and no state carries from one frame to the next.
 */
struct FrameInfo {
    std::int64_t frame_number = 0;  // position in the clip
    std::int64_t clip_key = 0;      // what makes one clip differ from another
    std::uint64_t tick = 0;         // the frame's exact timestamp (seeds the per-frame hash)
    double seconds = 0.0;           // clip time
};

/** Ticks per second of the FilmCraft timeline, which the per-frame seed is counted in. */
constexpr std::uint64_t kTicksPerSecond = 254016000000ULL;

/** The frame at `frame_number` of a clip that runs at `fps` frames per second. */
FrameInfo frame_info(std::int64_t frame_number, double fps);

/**
 * Renders the look onto device frames. Owns the device scratch it needs; one instance per frame size.
 * Not thread safe. Every CUDA failure is reported by throwing std::runtime_error.
 */
class Engine {
public:
    Engine(int width, int height);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /** Allocates the scratch. Throws if the frame cannot be handled (see max_dimension()). */
    void init();
    void cleanup() noexcept;

    /** Largest width or height the filters accept. */
    static int max_dimension();

    /**
     * d_in and d_out are packed 8-bit BGR frames of width x height (d_out may equal d_in).
     * A frame of no pixels is left alone.
     */
    void render(const std::uint8_t* d_in, std::uint8_t* d_out, const Controls& controls,
                const FrameInfo& frame, cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int width_;
    int height_;
};

} // namespace arsenal
} // namespace effects
} // namespace vhs
