/**
 * Arsenal VHS - the transition.
 *
 * Ported from the Arsenal VHS effect of a FilmCraft build (MIT OR Apache-2.0).
 *
 * The held last frame of the intro (A) rolls out downward while the main part (B) rolls in from the top;
 * a bar marks the seam. The settings are the calibrated defaults of the look (Transition Type 2, Tape
 * Noise Type 1, Tape Noise 100 %, Tape Distortion 100 %, Random Noise 25 %, Stroke, Wrinkle Distortion,
 * the channel offsets and blurs, Frame Jitter 2, Frame Style 0); the other types, Skew and Monochrome
 * are not ported.
 *
 * # Pipeline
 *
 *   1. composite  the two pictures rolling over, the rows displaced (Frame Jitter, the bend of the
 *                 Wrinkle Distortion), torn bands (shifted, or holding one row: vertical streaks),
 *                 colour-bar fragments, rainbow stripes, and the red, green and blue planes split,
 *                 drawn at one sample per pixel in linear light
 *   2. look       the Arsenal VHS look (colour stage and tape frame) over that picture, set from the
 *                 transition's colour controls and faded in and out by the transition's envelope
 *   3. stroke     the bar at the seam, after the look (which would grey the navy out)
 *   4. finish     Tape Noise, Random Noise and the tape's light, on the display-encoded picture
 *
 * Everything random is a hash of the frame (the progress quantised to 1/4096), the Random Seed and a
 * stream, so a frame depends only on its progress. Lengths are pixels of the 2560-wide frame the look
 * was calibrated on times width / 2560. The pictures are opaque 8-bit BGR.
 *
 * At progress 0 and 1 the result is A and B exactly.
 */

#include "vhs_transition.hpp"

#include "arsenal_vhs.hpp"
#include "arsenal_vhs_common.cuh"
#include "utils/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

namespace vhs {
namespace effects {

namespace {

using arsenal::clampf;
using arsenal::finitef;
using arsenal::linear_to_srgb;
using arsenal::srgb_to_linear;

// ---- hash streams ----
constexpr std::uint32_t kStreamFrame = 0x56540001u;
constexpr std::uint32_t kStreamTear = 0x56540002u;
constexpr std::uint32_t kStreamWobble = 0x56540003u;
constexpr std::uint32_t kStreamBend = 0x56540004u;
constexpr std::uint32_t kStreamEdge = 0x56540005u;
constexpr std::uint32_t kStreamLuma = 0x56540006u;
constexpr std::uint32_t kStreamCb = 0x56540007u;
constexpr std::uint32_t kStreamCr = 0x56540008u;
constexpr std::uint32_t kStreamRandom = 0x56540009u;
constexpr std::uint32_t kStreamBars = 0x5654000au;
constexpr std::uint32_t kStreamPatch = 0x5654000bu;
constexpr std::uint32_t kStreamStripe = 0x5654000cu;
constexpr std::uint32_t kStreamJitter = 0x5654000du;
constexpr std::uint32_t kStreamBlot = 0x5654000eu;

// ---- the controls (the calibrated defaults) ----
constexpr float kInput = -3.0f;           // Input
constexpr float kRandomSeed = 1.0f;       // Random Seed
constexpr float kTapeNoise = 1.0f;        // Tape Noise 100 %
constexpr float kTapeDistortion = 1.0f;   // Tape Distortion 100 %
constexpr float kRandomNoise = 0.25f;     // Random Noise 25 %
constexpr float kStrokeWidth = 25.0f;     // Stroke Width
constexpr float kWrinkleSize = 5.0f;      // Wrinkle Size
constexpr float kFrameJitter = 2.0f;      // Frame Jitter
constexpr float kOffset[3] = {-50.0f, 0.0f, 20.0f};  // Red, Green, Blue Offset

// ---- timing (fractions of the transition) ----
constexpr float kEnvRise0 = 0.03f, kEnvRise1 = 0.22f;
constexpr float kEnvFall0 = 0.78f, kEnvFall1 = 0.97f;
constexpr float kSeam0 = 0.04f, kSeam1 = 0.95f;
constexpr float kInputBias = 0.0025f;
/** The loop of the tape frame's left edge is counted at this rate over the transition's progress. */
constexpr double kFrameRate = 30.0;

// ---- Stroke (full-size px, times the scale) ----
constexpr float kStrokePx = 2.8f;
constexpr float kStrokeFeather = 3.0f;
constexpr float kStrokeLift = 0.012f;
constexpr float kWrinkleEdgePx = 0.8f;
constexpr float kWrinkleCell = 40.0f;
constexpr float kWrinkleBendPx = 2.0f;
constexpr float kWrinkleBendRows = 120.0f;
constexpr float kWrinkleBendCell = 6.0f;

// ---- the look, set from the colour controls ----
constexpr float kLumaBlurShare = 0.5f;
constexpr float kGreenBlurShare = 2.0f;
constexpr float kChannelBlurShare = 0.15f;
constexpr float kJitterPx = 0.5f;
constexpr float kSplitPx = 1.22f;

// ---- Tape Distortion ----
constexpr float kTearRate = 2.0f;
constexpr int kMaxTears = 12;
constexpr float kTearHeight0 = 40.0f, kTearHeight1 = 140.0f;
constexpr float kTearShift0 = 40.0f, kTearShift1 = 200.0f;
constexpr float kTearWobble = 6.0f;
constexpr float kTearFeather = 10.0f;
constexpr float kTearFull = 0.45f;
constexpr float kTearWidth0 = 300.0f, kTearWidth1 = 1000.0f;
constexpr float kTearHold = 0.8f;
constexpr float kTearBoost = 1.6f;
constexpr float kTearGain = 1.4f;
constexpr int kHoldRows = 5;
constexpr int kHoldColumns = 16;
constexpr float kBarsRate = 1.5f;
constexpr int kMaxBars = 4;
constexpr float kBarsCount0 = 3.0f, kBarsCount1 = 6.0f;
constexpr float kBarsWidth0 = 30.0f, kBarsWidth1 = 60.0f;
constexpr float kBarsHeight0 = 40.0f, kBarsHeight1 = 110.0f;
constexpr float kBarsAlpha = 0.9f;
constexpr float kBarsFeather = 4.0f;
constexpr float kStripeRate = 0.6f;
constexpr int kMaxStripes = 2;
constexpr float kStripeWidth0 = 40.0f, kStripeWidth1 = 90.0f;
constexpr float kStripeHeight0 = 300.0f, kStripeHeight1 = 900.0f;
constexpr float kStripeHueSpan = 0.6f;
constexpr float kStripeAlpha = 0.8f;
constexpr float kStripeFeather = 4.0f;
constexpr int kBarColours = 7;

// ---- Tape Noise (type 1, line snow) and Random Noise ----
constexpr float kValueNorm = 4.0f;
constexpr float kTapeLift = 0.072f;
constexpr float kLumaNoise = 0.016f, kLumaCellX = 30.0f, kLumaCellY = 1.0f;
constexpr float kBlotNoise = 0.016f, kBlotCellX = 10.0f, kBlotCellY = 6.0f;
constexpr float kChromaNoise = 0.0008f, kChromaCellX = 60.0f, kChromaCellY = 1.0f;
constexpr float kPatchNoise = 0.12f;
constexpr float kSnowRate = 1.0f;
constexpr int kMaxPatches = 3;
constexpr float kSnowWidth0 = 200.0f, kSnowWidth1 = 700.0f;
constexpr float kSnowHeight0 = 60.0f, kSnowHeight1 = 200.0f;
constexpr float kSnowCell = 40.0f;
constexpr float kRandomLuma = 0.004f;
constexpr float kRandomChroma = 0.0016f;

constexpr float kPi = 3.14159265358979323846f;
constexpr int kBlockX = 32;
constexpr int kBlockY = 8;

// =============================================================================================
// Hashing and noise (host and device: the plan and the kernels draw from the same numbers)

/** Integer hash to [0, 1). */
ARSENAL_HD float thash(int x, int y, std::uint32_t s) {
    std::uint32_t h = (static_cast<std::uint32_t>(x) * 0x8DA6B343u) ^ (static_cast<std::uint32_t>(y) * 0xD8163841u) ^
                      (s * 0xCB1AB31Fu) ^ 0x9E3779B9u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return static_cast<float>(h >> 8) / 16777216.0f;
}

ARSENAL_HD std::uint32_t mix_seed(std::uint32_t a, std::uint32_t b) {
    std::uint32_t h = (a * 0x9E3779B1u) ^ (b * 0x85EBCA6Bu) ^ 0xC2B2AE35u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    return h ^ (h >> 15);
}

ARSENAL_HD float sstep(float e0, float e1, float x) {
    if (e1 <= e0) return x < e0 ? 0.0f : 1.0f;
    const float t = clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/** Smooth value noise in [0, 1) with unit cells. */
ARSENAL_HD float vnoise(float x, float y, std::uint32_t s) {
    const float xi = floorf(x), yi = floorf(y);
    const float fx = x - xi, fy = y - yi;
    const float ux = fx * fx * (3.0f - 2.0f * fx), uy = fy * fy * (3.0f - 2.0f * fy);
    const int i = static_cast<int>(xi), j = static_cast<int>(yi);
    const float a = thash(i, j, s), b = thash(i + 1, j, s), c = thash(i, j + 1, s), d = thash(i + 1, j + 1, s);
    const float top = a + (b - a) * ux;
    const float bot = c + (d - c) * ux;
    return top + (bot - top) * uy;
}

/** Value noise of unit standard deviation, cells of `cell` full-size px along x on `row`. */
ARSENAL_HD float streak(float x, int row, float cell, std::uint32_t seed) {
    const float u = x / fmaxf(cell, 0.5f);
    const float i = floorf(u);
    const float f = u - i;
    const float s = f * f * (3.0f - 2.0f * f);
    const float a = thash(static_cast<int>(i), row, seed), b = thash(static_cast<int>(i) + 1, row, seed);
    return (a + (b - a) * s - 0.5f) * kValueNorm;
}

ARSENAL_HD float luma_of(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// =============================================================================================
// What a frame holds: planned on the host, read by the kernels

struct Tear {
    float y0, y1, x0, x1;
    float dx;
    float height, width;
    int hold;             // 1 when the band holds one row of the picture
    int seed;
    float row_u[kHoldRows];  // the draws that pick the candidate rows
};

struct Bars {
    float x0, x1, y0, y1;
    float bar;
    int first;
};

struct Stripe {
    float x0, x1, y0, y1;
    float hue;
};

struct Patch {
    float x0, x1, y0, y1;
};

struct Plan {
    int w, h;
    float wf, hf, scale;
    // the frame's timeline
    float seam, env, dmg, bar_env;
    std::uint32_t seed;
    // composite
    int split;
    float shift[3];
    float jitter_amp;   // 0 when Frame Jitter is off
    float bend_amp;     // 0 when the Wrinkle Distortion is off
    float bend_reach;
    float wobble, tear_feather, bars_soft, stripe_soft;
    int n_tears, n_bars, n_stripes;
    Tear tears[kMaxTears];
    Bars bars[kMaxBars];
    Stripe stripes[kMaxStripes];
    float bar_lin[kBarColours * 3];  // the colour-bar signal, linear
    // stroke
    float half, edge, feather;
    float stroke_lin[3];
    // finish
    float tape, random, lift;
    int n_patches;
    Patch patches[kMaxPatches];
};

/** How many of a thing there are this frame: `rate` per unit of `strength`, the fraction by the hash, at most `max`. */
int count_of(const Plan& p, float rate, float strength, int max, std::uint32_t stream) {
    if (!(strength > 0.0f)) return 0;
    const float n = rate * strength + thash(0, 0, p.seed ^ stream);
    return std::min(std::max(static_cast<int>(n), 0), max);
}

/** The rolling picture; the seam is where B ends and A begins. */
void plan_frame(Plan& p, float progress, int w, int h) {
    p = Plan{};
    p.w = w;
    p.h = h;
    p.wf = static_cast<float>(w);
    p.hf = static_cast<float>(h);
    p.scale = static_cast<float>(static_cast<double>(w) / arsenal::kReferenceWidth);

    // timeline
    const float pb = clampf(progress + kInput * kInputBias, 0.0f, 1.0f);
    p.env = sstep(kEnvRise0, kEnvRise1, pb) * (1.0f - sstep(kEnvFall0, kEnvFall1, pb));
    p.bar_env = sstep(0.04f, 0.32f, pb) * (1.0f - 0.3f * sstep(0.6f, 0.9f, pb)) * (1.0f - sstep(0.9f, 0.96f, pb));
    p.seam = p.hf * sstep(kSeam0, kSeam1, pb);
    const float s = sinf(kPi * pb);
    p.dmg = s * s;
    const std::uint32_t key = static_cast<std::uint32_t>(std::lround(progress * 4096.0f));
    p.seed = mix_seed(mix_seed(key, static_cast<std::uint32_t>(kRandomSeed)), kStreamFrame);

    const float scale = p.scale;
    const float dist = kTapeDistortion * p.dmg;

    // channel split and row displacement
    for (int k = 0; k < 3; ++k) {
        p.shift[k] = kOffset[k] * kSplitPx * scale * p.env;
        p.split |= p.shift[k] != 0.0f;
    }
    p.jitter_amp = kFrameJitter * kJitterPx * scale * p.env;
    if (p.bar_env > 0.0f) {
        p.bend_amp = kWrinkleSize * kWrinkleBendPx * scale * p.bar_env;
        p.bend_reach = kWrinkleBendRows * scale;
    }
    p.wobble = kTearWobble * scale;
    p.tear_feather = std::max(kTearFeather * scale, 0.5f);
    p.bars_soft = kBarsFeather * scale;
    p.stripe_soft = kStripeFeather * scale;

    // torn bands
    p.n_tears = count_of(p, kTearRate, dist, kMaxTears, kStreamTear);
    for (int k = 0; k < p.n_tears; ++k) {
        auto u = [&](int n) { return thash(k, n, p.seed ^ kStreamTear); };
        Tear& t = p.tears[k];
        t.height = (kTearHeight0 + (kTearHeight1 - kTearHeight0) * u(2)) * scale;
        t.y0 = u(1) * std::max(p.hf - t.height, 0.0f);
        const bool full = u(8) < kTearFull;
        t.width = full ? p.wf : std::min((kTearWidth0 + (kTearWidth1 - kTearWidth0) * u(9)) * scale, p.wf);
        t.x0 = full ? 0.0f : u(10) * std::max(p.wf - t.width, 0.0f);
        const float sign = u(6) < 0.5f ? -1.0f : 1.0f;
        t.dx = sign * (kTearShift0 + (kTearShift1 - kTearShift0) * u(4)) * scale;
        t.y1 = t.y0 + t.height;
        t.x1 = t.x0 + t.width;
        t.hold = u(3) < kTearHold;
        t.seed = k;
        for (int i = 0; i < kHoldRows; ++i) t.row_u[i] = u(20 + i);
    }

    // colour-bar fragments
    p.n_bars = count_of(p, kBarsRate, dist, kMaxBars, kStreamBars);
    for (int k = 0; k < p.n_bars; ++k) {
        auto u = [&](int n) { return thash(k, n, p.seed ^ kStreamBars); };
        const float bar = (kBarsWidth0 + (kBarsWidth1 - kBarsWidth0) * u(1)) * scale;
        const float n = std::max(std::round(kBarsCount0 + (kBarsCount1 - kBarsCount0) * u(2)), 1.0f);
        const float height = (kBarsHeight0 + (kBarsHeight1 - kBarsHeight0) * u(3)) * scale;
        Bars& b = p.bars[k];
        b.x0 = u(4) * std::max(p.wf - bar * n, 0.0f);
        b.y0 = u(5) * std::max(p.hf - height, 0.0f);
        b.x1 = b.x0 + bar * n;
        b.y1 = b.y0 + height;
        b.bar = std::max(bar, 1.0f);
        b.first = std::min(static_cast<int>(u(6) * static_cast<float>(kBarColours)), kBarColours - 1);
    }

    // rainbow stripes
    p.n_stripes = count_of(p, kStripeRate, dist, kMaxStripes, kStreamStripe);
    for (int k = 0; k < p.n_stripes; ++k) {
        auto u = [&](int n) { return thash(k, n, p.seed ^ kStreamStripe); };
        const float width = (kStripeWidth0 + (kStripeWidth1 - kStripeWidth0) * u(1)) * scale;
        const float height = std::min((kStripeHeight0 + (kStripeHeight1 - kStripeHeight0) * u(2)) * scale, p.hf);
        Stripe& st = p.stripes[k];
        st.x0 = u(3) * std::max(p.wf - width, 0.0f);
        st.y0 = u(4) * std::max(p.hf - height, 0.0f);
        st.x1 = st.x0 + width;
        st.y1 = st.y0 + height;
        st.hue = u(5);
    }

    // the seven bars of the colour-bar test signal at 75 %: white, yellow, cyan, green, magenta, red, blue
    const float bars[kBarColours][3] = {{0.75f, 0.75f, 0.75f}, {0.75f, 0.75f, 0.0f}, {0.0f, 0.75f, 0.75f}, {0.0f, 0.75f, 0.0f},
                                        {0.75f, 0.0f, 0.75f}, {0.75f, 0.0f, 0.0f},  {0.0f, 0.0f, 0.75f}};
    for (int i = 0; i < kBarColours; ++i) {
        for (int c = 0; c < 3; ++c) p.bar_lin[i * 3 + c] = srgb_to_linear(bars[i][c]);
    }

    // the stroke bar: navy #0C0C26, lifted by the tape's light, in linear light
    p.half = 0.5f * kStrokeWidth * kStrokePx * scale * p.bar_env;
    p.edge = kWrinkleEdgePx * kWrinkleSize * scale * p.bar_env;
    p.feather = std::min(kStrokeFeather * scale, p.half);
    const float navy[3] = {12.0f / 255.0f, 12.0f / 255.0f, 38.0f / 255.0f};
    for (int c = 0; c < 3; ++c) p.stroke_lin[c] = srgb_to_linear(std::min(navy[c] + kStrokeLift, 1.0f));

    // tape noise, random noise and the tape's light
    p.tape = kTapeNoise * p.dmg;
    p.random = kRandomNoise * p.dmg;
    p.lift = kTapeLift * kTapeNoise * p.dmg;
    p.n_patches = count_of(p, kSnowRate, p.tape, kMaxPatches, kStreamPatch);
    for (int k = 0; k < p.n_patches; ++k) {
        auto u = [&](int n) { return thash(k, n, p.seed ^ kStreamPatch); };
        const float pw = std::min((kSnowWidth0 + (kSnowWidth1 - kSnowWidth0) * u(1)) * scale, p.wf);
        const float ph = std::min((kSnowHeight0 + (kSnowHeight1 - kSnowHeight0) * u(2)) * scale, p.hf);
        Patch& pa = p.patches[k];
        pa.x0 = u(3) * std::max(p.wf - pw, 0.0f);
        pa.y0 = u(4) * std::max(p.hf - ph, 0.0f);
        pa.x1 = pa.x0 + pw;
        pa.y1 = pa.y0 + ph;
    }
}

// =============================================================================================
// Kernels

/** Linear light of the pixel (x, y) of a packed BGR frame. */
__device__ __forceinline__ void load_linear(const unsigned char* img, int w, int x, int y, float c[3]) {
    const size_t i = (static_cast<size_t>(y) * w + x) * 3;
    c[0] = srgb_to_linear(img[i + 2] * (1.0f / 255.0f));
    c[1] = srgb_to_linear(img[i + 1] * (1.0f / 255.0f));
    c[2] = srgb_to_linear(img[i + 0] * (1.0f / 255.0f));
}

/** Bilinear sample at (x, y) in pixel-centre coordinates, the edge repeated; linear light. */
__device__ void sample_linear(const unsigned char* img, int w, int h, float x, float y, float out[3]) {
    const float fx = finitef(x) ? clampf(x - 0.5f, 0.0f, static_cast<float>(w - 1)) : 0.0f;
    const float fy = finitef(y) ? clampf(y - 0.5f, 0.0f, static_cast<float>(h - 1)) : 0.0f;
    const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0);
    float a[3], b[3], c[3], d[3];
    load_linear(img, w, x0, y0, a);
    load_linear(img, w, x1, y0, b);
    load_linear(img, w, x0, y1, c);
    load_linear(img, w, x1, y1, d);
    for (int k = 0; k < 3; ++k) {
        const float top = a[k] + (b[k] - a[k]) * tx;
        const float bot = c[k] + (d[k] - c[k]) * tx;
        out[k] = top + (bot - top) * ty;
    }
}

/** Display RGB of hue `h` (turns) at the saturation and value of the colour-bar signal. */
__device__ void hue_rgb(float h, float out[3]) {
    h = (h - floorf(h)) * 6.0f;
    const float n[3] = {5.0f, 3.0f, 1.0f};
    for (int c = 0; c < 3; ++c) {
        const float k = fmodf(n[c] + h, 6.0f);
        out[c] = 0.75f - 0.75f * clampf(fminf(k, 4.0f - k), 0.0f, 1.0f);
    }
}

/** The picture before the displacement: the two pictures, the colour-bar fragments and the stripes. */
__device__ void picture(const Plan& P, const unsigned char* a, const unsigned char* b, float x, float y, float px[3]) {
    // A rolls out downward as B rolls in from the top: the seam is where B ends and A begins
    if (y >= P.seam) {
        sample_linear(a, P.w, P.h, x, y - P.seam, px);
    } else {
        sample_linear(b, P.w, P.h, x, y - P.seam + P.hf, px);
    }
    for (int f = 0; f < P.n_bars; ++f) {
        const Bars& s = P.bars[f];
        if (y >= s.y0 && y < s.y1 && x >= s.x0 && x < s.x1) {
            const int i = (static_cast<int>((x - s.x0) / s.bar) + s.first) % kBarColours;
            const float k = kBarsAlpha * P.dmg * sstep(0.0f, P.bars_soft, fminf(x - s.x0, s.x1 - x)) *
                            sstep(0.0f, P.bars_soft, fminf(y - s.y0, s.y1 - y));
            for (int c = 0; c < 3; ++c) px[c] += (P.bar_lin[i * 3 + c] - px[c]) * k;
        }
    }
    for (int f = 0; f < P.n_stripes; ++f) {
        const Stripe& s = P.stripes[f];
        if (y >= s.y0 && y < s.y1 && x >= s.x0 && x < s.x1) {
            float hue[3];
            hue_rgb(s.hue + kStripeHueSpan * (y - s.y0) / fmaxf(s.y1 - s.y0, 1.0f), hue);
            const float k = kStripeAlpha * P.dmg * sstep(0.0f, P.stripe_soft, fminf(x - s.x0, s.x1 - x));
            for (int c = 0; c < 3; ++c) px[c] += (srgb_to_linear(hue[c]) - px[c]) * k;
        }
    }
}

/** The row a band holds is the one of kHoldRows candidates whose columns are brightest and most varied. */
__global__ void hold_kernel(Plan P, const unsigned char* a, const unsigned char* b, float* holds) {
    const int k = threadIdx.x;
    if (k >= P.n_tears || !P.tears[k].hold) return;
    const Tear& t = P.tears[k];
    float best = -1.0f, row = t.y0;
    for (int i = 0; i < kHoldRows; ++i) {
        const float y = clampf(t.y0 + (static_cast<float>(i) + t.row_u[i]) * t.height / static_cast<float>(kHoldRows), 0.0f,
                               P.hf - 1.0f) + 0.5f;
        float v[kHoldColumns];
        float mean = 0.0f;
        for (int j = 0; j < kHoldColumns; ++j) {
            float px[3];
            picture(P, a, b, t.x0 + t.width * (static_cast<float>(j) + 0.5f) / static_cast<float>(kHoldColumns), y, px);
            v[j] = luma_of(px[0], px[1], px[2]);
            mean += v[j];
        }
        mean /= static_cast<float>(kHoldColumns);
        float var = 0.0f;
        for (int j = 0; j < kHoldColumns; ++j) var += (v[j] - mean) * (v[j] - mean);
        const float score = mean + 2.0f * sqrtf(var / static_cast<float>(kHoldColumns));
        if (score > best) {
            best = score;
            row = y;
        }
    }
    holds[k] = row;
}

__device__ __forceinline__ unsigned char to_byte(float display) {
    return static_cast<unsigned char>(__float2int_rn(clampf(display, 0.0f, 1.0f) * 255.0f));
}

/** The picture: the rolling pictures with the rows displaced and torn and the colour planes split. */
__global__ void composite_kernel(Plan P, const unsigned char* a, const unsigned char* b, const float* holds,
                                 unsigned char* out) {
    const int xi = blockIdx.x * blockDim.x + threadIdx.x;
    const int yi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= P.w || yi >= P.h) return;
    const float x = static_cast<float>(xi) + 0.5f;
    const float y = static_cast<float>(yi) + 0.5f;

    // Frame Jitter and the bend of the Wrinkle Distortion around the seam
    float dx = 0.0f;
    if (P.jitter_amp != 0.0f) dx += (thash(yi, 0, P.seed ^ kStreamJitter) - 0.5f) * 2.0f * P.jitter_amp;
    if (P.bend_amp != 0.0f) {
        const float d = fabsf(y - P.seam);
        if (d < P.bend_reach) {
            const float window = 0.5f + 0.5f * cosf(kPi * d / P.bend_reach);
            const float wiggle = (vnoise(y / (kWrinkleBendCell * P.scale), 0.5f, P.seed ^ kStreamBend) - 0.5f) * 2.0f;
            dx += P.bend_amp * window * wiggle;
        }
    }

    float xs = x - dx, ys = y, boost = 1.0f, gain = 1.0f;
    for (int k = 0; k < P.n_tears; ++k) {
        const Tear& t = P.tears[k];
        if (y >= t.y0 && y < t.y1) {
            const float w = sstep(t.x0 - P.tear_feather, t.x0 + P.tear_feather, x) *
                            (1.0f - sstep(t.x1 - P.tear_feather, t.x1 + P.tear_feather, x));
            if (w > 0.0f) {
                xs -= (t.dx + (thash(yi, t.seed + 100, P.seed ^ kStreamWobble) - 0.5f) * 2.0f * P.wobble) * w;
                if (t.hold) {
                    ys += (holds[k] - ys) * w;
                    boost += (kTearBoost - 1.0f) * w;
                    gain += (kTearGain - 1.0f) * w;
                }
            }
        }
    }

    float px[3];
    if (P.split) {
        float r[3], g[3], bl[3];
        picture(P, a, b, xs - P.shift[0], ys, r);
        picture(P, a, b, xs - P.shift[1], ys, g);
        picture(P, a, b, xs - P.shift[2], ys, bl);
        px[0] = r[0];
        px[1] = g[1];
        px[2] = bl[2];
    } else {
        picture(P, a, b, xs, ys, px);
    }
    if (boost != 1.0f) {
        const float l = luma_of(px[0], px[1], px[2]);
        for (int c = 0; c < 3; ++c) px[c] = clampf((l + (px[c] - l) * boost) * gain, 0.0f, 1.0f);
    }
    const size_t i = (static_cast<size_t>(yi) * P.w + xi) * 3;
    out[i + 0] = to_byte(linear_to_srgb(fmaxf(px[2], 0.0f)));
    out[i + 1] = to_byte(linear_to_srgb(fmaxf(px[1], 0.0f)));
    out[i + 2] = to_byte(linear_to_srgb(fmaxf(px[0], 0.0f)));
}

/**
 * The look faded by the envelope over the picture it filtered, the stroke bar at the seam, and the tape's
 * noise and light, written as display-encoded 8-bit BGR. `look` is null where the look is off.
 */
__global__ void finish_kernel(Plan P, const unsigned char* comp, const unsigned char* look, unsigned char* out) {
    const int xi = blockIdx.x * blockDim.x + threadIdx.x;
    const int yi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= P.w || yi >= P.h) return;
    const float xc = static_cast<float>(xi) + 0.5f;
    const float yc = static_cast<float>(yi) + 0.5f;

    float px[3];
    load_linear(comp, P.w, xi, yi, px);
    if (look) {
        float lk[3];
        load_linear(look, P.w, xi, yi, lk);
        for (int c = 0; c < 3; ++c) px[c] += (lk[c] - px[c]) * P.env;
    }

    // the stroke bar: a band over the picture, with a ragged edge
    if (P.half > 0.0f) {
        const float d = fabsf(yc - P.seam);
        if (d <= P.half + P.edge + P.feather) {
            const float ragged = P.edge > 0.0f
                                     ? (vnoise(xc / (kWrinkleCell * P.scale), 0.5f, P.seed ^ kStreamEdge) - 0.5f) * 2.0f * P.edge
                                     : 0.0f;
            const float hb = fmaxf(P.half + ragged, 0.0f);
            const float f = fminf(P.feather, hb);
            const float k = 1.0f - sstep(hb - f, hb + f, d);
            if (k > 0.0f) {
                for (int c = 0; c < 3; ++c) px[c] += (P.stroke_lin[c] - px[c]) * k;
            }
        }
    }

    // Tape Noise, Random Noise and the tape's light on the display-encoded picture
    float s[3];
    for (int c = 0; c < 3; ++c) s[c] = linear_to_srgb(fmaxf(px[c], 0.0f));
    const float xf = xc / P.scale;
    const int line = static_cast<int>(yc / P.scale);
    float n_y = P.lift, n_cb = 0.0f, n_cr = 0.0f;
    if (P.tape > 0.0f) {
        const int ly = static_cast<int>(floorf(yc / (kLumaCellY * P.scale)));
        const int by = static_cast<int>(floorf(yc / (kBlotCellY * P.scale)));
        const int cy = static_cast<int>(floorf(yc / (kChromaCellY * P.scale)));
        n_y += P.tape * (kLumaNoise * streak(xf, ly, kLumaCellX, P.seed ^ kStreamLuma) +
                         kBlotNoise * streak(xf, by, kBlotCellX, P.seed ^ kStreamBlot));
        n_cb += P.tape * kChromaNoise * streak(xf, cy, kChromaCellX, P.seed ^ kStreamCb);
        n_cr += P.tape * kChromaNoise * streak(xf, cy, kChromaCellX, P.seed ^ kStreamCr);
        for (int k = 0; k < P.n_patches; ++k) {
            const Patch& pa = P.patches[k];
            if (yc >= pa.y0 && yc < pa.y1 && xc >= pa.x0 && xc < pa.x1) {
                n_y += P.tape * kPatchNoise * fmaxf(streak(xf, line, kSnowCell, P.seed ^ kStreamPatch), 0.0f);
            }
        }
    }
    if (P.random > 0.0f) {
        const int fx = static_cast<int>(xf);
        auto g = [&](std::uint32_t stream) { return (thash(fx, line, P.seed ^ stream) - 0.5f) * 3.464f; };
        n_y += P.random * kRandomLuma * g(kStreamRandom);
        n_cb += P.random * kRandomChroma * g(kStreamRandom ^ 1u);
        n_cr += P.random * kRandomChroma * g(kStreamRandom ^ 2u);
    }
    s[0] += n_y + 1.402f * n_cr;
    s[1] += n_y - 0.344f * n_cb - 0.714f * n_cr;
    s[2] += n_y + 1.772f * n_cb;

    const size_t i = (static_cast<size_t>(yi) * P.w + xi) * 3;
    out[i + 0] = to_byte(s[2]);
    out[i + 1] = to_byte(s[1]);
    out[i + 2] = to_byte(s[0]);
}

// =============================================================================================
// Host

void check_launch() {
    CUDA_CHECK(cudaGetLastError());
}

/** The look as the transition sets it: the calibrated look with its own colour controls, no damage. */
arsenal::Controls transition_look() {
    arsenal::Controls c;
    c.black_level = 10.0f;
    c.saturation = 125.0f;
    c.luma_blur = 4.0f * kLumaBlurShare;
    c.chroma_blur = 0.0f;
    c.chroma_offset_x = 10.0f;
    c.red_blur = 25.0f * kChannelBlurShare;
    c.green_blur = 55.0f * kGreenBlurShare;
    c.blue_blur = 45.0f * kChannelBlurShare;
    // the channel offsets are the transition's own split, drawn with the picture
    c.red_offset = 0.0f;
    c.green_offset = 0.0f;
    c.blue_offset = 0.0f;
    // the damage (noise, dust, pop lines, jitter, wrinkle) is the transition's own
    c.tape_damage = false;
    c.wrinkle = false;
    c.frame_jitter = 0.0f;
    c.pop_line_freq = 0.0f;
    c.post_noise = 0.0f;
    c.tape_frame = true;
    c.seed = 1000.0f + kRandomSeed;
    return c;
}

/** The device memory the transition keeps between its frames (it lives only while a transition runs). */
struct State {
    int w = 0;
    int h = 0;
    unsigned char* comp = nullptr;
    unsigned char* look = nullptr;
    float* holds = nullptr;
    std::unique_ptr<arsenal::Engine> engine;

    ~State() { release(); }

    /** Errors are ignored on purpose: this also runs at exit, after the CUDA context may be gone. */
    void release() noexcept {
        if (comp) cudaFree(comp);
        if (look) cudaFree(look);
        if (holds) cudaFree(holds);
        comp = nullptr;
        look = nullptr;
        holds = nullptr;
        if (engine) engine->cleanup();
        engine.reset();
        w = h = 0;
    }

    void ensure(int width, int height) {
        if (comp && w == width && h == height) return;
        release();
        try {
            const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
            CUDA_CHECK(cudaMalloc(&comp, bytes));
            CUDA_CHECK(cudaMalloc(&look, bytes));
            CUDA_CHECK(cudaMalloc(&holds, kMaxTears * sizeof(float)));
            engine.reset(new arsenal::Engine(width, height));
            engine->init();
        } catch (...) {
            release();
            throw;
        }
        w = width;
        h = height;
    }
};

State& state() {
    static State s;
    return s;
}

void copy_frame(const unsigned char* src, unsigned char* dst, int width, int height, cudaStream_t stream) {
    if (!src || !dst || src == dst) return;
    CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(width) * height * 3, cudaMemcpyDeviceToDevice, stream));
}

} // namespace

void apply_vhs_transition(
    const unsigned char* d_intro,
    const unsigned char* d_main,
    unsigned char* d_output,
    int width,
    int height,
    float progress,
    int frame_index,
    cudaStream_t stream
) {
    (void)frame_index;  // a frame is identified by its progress
    if (width <= 0 || height <= 0 || !d_output) return;
    const float p = std::isfinite(progress) ? clampf(progress, 0.0f, 1.0f) : 0.0f;

    // before it starts the picture is A, and once it is over it is B (the memory is let go then)
    if (!d_intro || !d_main || p <= 0.0f || p >= 1.0f) {
        copy_frame((p >= 1.0f || !d_intro) ? d_main : d_intro, d_output, width, height, stream);
        if (p >= 1.0f) state().release();
        return;
    }

    State& st = state();
    st.ensure(width, height);

    Plan plan;
    plan_frame(plan, p, width, height);
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((width + kBlockX - 1) / kBlockX, (height + kBlockY - 1) / kBlockY);

    if (plan.n_tears > 0) {
        hold_kernel<<<1, kMaxTears, 0, stream>>>(plan, d_intro, d_main, st.holds);
        check_launch();
    }
    composite_kernel<<<grid, block, 0, stream>>>(plan, d_intro, d_main, st.holds, st.comp);
    check_launch();

    // the look, faded in and out by the envelope; its tape frame moves on a 30-step loop over the progress
    const unsigned char* look = nullptr;
    if (plan.env > 0.0f) {
        const std::uint32_t key = static_cast<std::uint32_t>(std::lround(p * 4096.0f));
        arsenal::FrameInfo frame;
        frame.seconds = static_cast<double>(key) / 4096.0;
        frame.frame_number = static_cast<std::int64_t>(std::llround(frame.seconds * kFrameRate));
        frame.tick = key;
        st.engine->render(st.comp, st.look, transition_look(), frame, stream);
        look = st.look;
    }

    finish_kernel<<<grid, block, 0, stream>>>(plan, st.comp, look, d_output);
    check_launch();
}

} // namespace effects
} // namespace vhs
