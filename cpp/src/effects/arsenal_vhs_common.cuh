/**
 * Arsenal VHS - helpers shared by the colour stage, the damage stage and the transition.
 *
 * Ported from the Arsenal VHS effect of a FilmCraft build (MIT OR Apache-2.0).
 *
 * Everything here is usable on the host and on the device, so the host-side planning of a frame
 * (which dust specks fall where, where the wrinkle band is) and the kernels hash with the same code.
 * Nothing draws from a random generator: every varying quantity is a hash of the clip key, the
 * Random Seed and the frame's tick, so the same frame renders to the same bytes on every run.
 */

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#define ARSENAL_HD __host__ __device__ __forceinline__

namespace vhs {
namespace effects {
namespace arsenal {

// ---------------------------------------------------------------------------------------------
// Hashing

/** 64-bit hash of three integer coordinates and a seed. */
ARSENAL_HD std::uint64_t hash_u(std::int64_t x, std::int64_t y, std::int64_t z, std::uint64_t seed) {
    std::uint64_t h = (static_cast<std::uint64_t>(x) * 0x9E3779B97F4A7C15ULL) ^
                      (static_cast<std::uint64_t>(y) * 0xC2B2AE3D27D4EB4FULL) ^
                      (static_cast<std::uint64_t>(z) * 0x165667B19E3779F9ULL) ^
                      (seed * 0x27D4EB2F165667C5ULL);
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 29;
    h *= 0x94D049BB133111EBULL;
    return h ^ (h >> 32);
}

/** Uniform number in [0, 1) from integer coordinates. */
ARSENAL_HD float hash01(std::int64_t x, std::int64_t y, std::int64_t z, std::uint64_t seed) {
    return static_cast<float>(hash_u(x, y, z, seed) >> 40) / 16777216.0f;
}

// ---------------------------------------------------------------------------------------------
// Small maths

ARSENAL_HD float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

ARSENAL_HD float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

/** Smooth step of t clamped to 0..1. */
ARSENAL_HD float smooth01(float t) {
    t = clampf(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/** True for a finite number (NaN and infinities are false). */
ARSENAL_HD bool finitef(float v) {
    return isfinite(v);
}

// ---------------------------------------------------------------------------------------------
// Colour: display-encoded R'G'B' and Y'CbCr (ITU-R BT.601, Y' 0..1, Cb and Cr -0.5..0.5)

constexpr float kKr = 0.299f;
constexpr float kKb = 0.114f;

ARSENAL_HD void rgb_to_ycc(float r, float g, float b, float& y, float& cb, float& cr) {
    const float kg = 1.0f - kKr - kKb;
    y = kKr * r + kg * g + kKb * b;
    cb = (b - y) / (2.0f * (1.0f - kKb));
    cr = (r - y) / (2.0f * (1.0f - kKr));
}

ARSENAL_HD void ycc_to_rgb(float y, float cb, float cr, float& r, float& g, float& b) {
    const float kg = 1.0f - kKr - kKb;
    r = y + 2.0f * (1.0f - kKr) * cr;
    b = y + 2.0f * (1.0f - kKb) * cb;
    g = (y - kKr * r - kKb * b) / kg;
}

ARSENAL_HD float srgb_to_linear(float v) {
    return v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
}

ARSENAL_HD float linear_to_srgb(float v) {
    return v <= 0.0031308f ? v * 12.92f : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
}

} // namespace arsenal
} // namespace effects
} // namespace vhs
