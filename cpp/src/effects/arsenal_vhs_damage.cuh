/**
 * Arsenal VHS - the damage stage, device side: what one pixel adds to the signal and what the output
 * stage does to it, plus the plain data the host plans for a frame and hands to the kernels.
 *
 * Ported from the Arsenal VHS effect of a FilmCraft build (MIT OR Apache-2.0).
 *
 * # What the stage does (see arsenal_vhs_damage.cu for how a frame is planned)
 *
 *  - Geometry: one displacement per row (Frame Jitter) moves the three planes together, and the Tape
 *    Wrinkle band redraws the rows it covers from held rows and sideways chunks.
 *  - Signal, added to Y', Cb and Cr before the grade: tape noise (fine snow and soft mottle in
 *    brightness, colour mottle), soft coloured flecks, dust specks, pop lines and the noise of the
 *    head-switching band at the bottom.
 *  - Output, on the graded R'G'B': the tape frame (a black bar and a glow on the left, a soft edge on
 *    the right, a foggy or grooved top) and Post Noise.
 *
 * Lengths are pixels of the 2560 x 1440 picture the look was calibrated on, times the frame's
 * scale. Amplitudes are display units (0 to 1). Nothing draws from a random generator: every varying
 * quantity is a hash of the clip, Random Seed and the frame's timestamp.
 */

#pragma once

#include "arsenal_vhs_common.cuh"

namespace vhs {
namespace effects {
namespace arsenal {

// ---------------------------------------------------------------------------------------------
// Hash streams (these name what a hash draws, so two uses never share numbers)

constexpr std::int64_t S_FRAME = 0x44414d31;
constexpr std::int64_t S_LUMA = 0x44414c31;
constexpr std::int64_t S_CB = 0x44414342;
constexpr std::int64_t S_CR = 0x44414352;
constexpr std::int64_t S_GRAIN = 0x44414752;
constexpr std::int64_t S_POST = 0x44415053;
constexpr std::int64_t S_HEAD_Y = 0x44414859;
constexpr std::int64_t S_HEAD_C = 0x44414843;
constexpr std::int64_t S_SLOW = 0x4441534c;
constexpr std::int64_t S_EDGE = 0x44414553;
constexpr std::int64_t S_FAST = 0x44414641;
constexpr std::int64_t S_WRINKLE = 0x44415752;
constexpr std::int64_t S_TILE = 0x44415449;
constexpr std::int64_t S_SPECK = 0x44415350;
constexpr std::int64_t S_POP = 0x4441504f;
constexpr std::int64_t S_TOP = 0x4441544f;
constexpr std::int64_t S_GLOW = 0x4441474c;
constexpr std::int64_t S_BEAD = 0x44414245;

// ---------------------------------------------------------------------------------------------
// Constants of the stage (lengths in pixels of the full-size picture, amplitudes in display units)

/** The noise of the look: fine snow and a soft mottle in brightness, a colour mottle, flecks and dust. */
constexpr float kLookLuma = 0.011f;
constexpr float kLookLumaSigmaX = 2.0f;   // Gaussian widths in lattice cells
constexpr float kLookLumaSigmaY = 3.0f;
constexpr float kLookFine = 0.0075f;
constexpr float kFineSigmaX = 1.1f;
constexpr float kFineSigmaY = 1.7f;
constexpr float kLookChroma = 0.012f;
constexpr float kLookChromaSigma = 5.5f;
constexpr float kCrShare = 0.2f;          // Cr mottle as a share of Cb's
constexpr float kLookSpecks = 1.5f;       // dust specks per Mpx
constexpr float kLookFlecks = 2100.0f;    // soft flecks per Mpx
/** The noise falls to this share of its amplitude (flecks to the other) on picture brighter than kNoiseFall. */
constexpr float kNoiseFloor = 0.4f;
constexpr float kFleckFloor = 0.5f;
constexpr float kNoiseFall = 0.085f;

constexpr float kSpeckRx = 3.0f;
constexpr float kSpeckRy = 2.2f;
constexpr float kSpeckAmpLo = 0.25f;
constexpr float kSpeckAmpHi = 0.70f;
constexpr float kSpeckTint = 0.10f;
constexpr float kFleckRLo = 3.5f;
constexpr float kFleckRHi = 6.5f;
constexpr float kFleckAspect = 0.75f;
constexpr float kFleckAmp = 0.035f;
constexpr float kFleckChroma = 0.065f;

constexpr float kPopPerPercent = 0.05f;
constexpr float kPopLenLo = 50.0f;
constexpr float kPopLenHi = 400.0f;
constexpr float kPopSigmaY = 1.3f;
constexpr float kPopAmpLo = 0.35f;
constexpr float kPopAmpHi = 0.80f;

constexpr float kJitterPx = 1.0f;
constexpr float kEdgeRefJitter = 2.0f;
constexpr float kEdgeKnot = 32.0f;
constexpr float kEdgeStillCell = 48.0f;
constexpr float kEdgeStillAmp = 0.42f;
constexpr float kEdgeOwnCell = 20.0f;
constexpr float kEdgeOwnAmp = 0.88f;
constexpr float kJitterSlowShare = 0.97f;
constexpr float kJitterSlowCell = 120.0f;
constexpr float kJitterFastCell = 2.0f;

constexpr float kPostSigma = 0.0019f;
constexpr float kPostFieldSigma = 0.8f;
constexpr float kPostCommon = 0.7f;

constexpr float kWrSpeed = 60.0f;
constexpr float kWrHalf = 16.0f;
constexpr float kWrCore = 0.4f;
constexpr float kWrGroupLo = 4.0f;
constexpr float kWrGroupHi = 14.0f;
constexpr float kWrChunkLo = 24.0f;
constexpr float kWrChunkHi = 150.0f;
constexpr float kWrShift = 48.0f;
constexpr float kWrShiftShare = 0.7f;
constexpr float kWrChroma = 24.0f;

constexpr float kLeftBar = 31.2f;
constexpr float kBarMid = -6.24f;
constexpr float kBarWidth = 11.4f;
constexpr float kBarBlack = 0.008f;
constexpr float kGlowR = 0.55f;  // the glow is a bluish white
constexpr float kGlowG = 0.59f;
constexpr float kGlowB = 0.72f;
constexpr float kGlowPeak = 0.638f;
constexpr float kGlowOnset = -0.2f;
constexpr float kGlowWidth = 7.57f;
constexpr float kGlowAt = 5.89f;
constexpr float kGlowTau = 7.3f;
constexpr float kGlowGrain = 0.14f;
constexpr float kBeadSd = 0.21f;
constexpr float kBeadCell0 = 10.0f;
constexpr float kBeadCell1 = 40.0f;
constexpr float kRightCentre = 10.4f;
constexpr float kRightSoft = 5.3f;
constexpr float kRightVignette = 0.09f;
constexpr float kTopVignette = 0.45f;
constexpr float kTopVignetteTau = 175.0f;
constexpr float kFogFloor = 0.053f;
constexpr float kFogTau = 82.3f;
constexpr float kFogPower = 1.7f;
constexpr float kFogEdge = 0.18f;
constexpr float kFogEdgeLen = 2.4f;
constexpr float kGrooveLeftDepth = 0.76f;
constexpr float kGrooveLeftCentre = 15.8f;
constexpr float kGrooveLeftWidth = 5.5f;
constexpr float kGrooveRightDepth = 0.69f;
constexpr float kGrooveRightCentre = 10.0f;
constexpr float kGrooveRightWidth = 5.0f;
constexpr float kGrooveStep = 480.0f;
constexpr float kGrooveJitter = 1.1f;
constexpr float kTopFog = 0.5f;
constexpr float kTopHalf = 0.2f;

constexpr float kHeadRows = 24.0f;
constexpr float kHeadAmpLo = 0.028f;
constexpr float kHeadAmpHi = 0.109f;
constexpr float kHeadCellX = 36.0f;
constexpr float kHeadCellY = 3.0f;
constexpr float kHeadChroma = 0.10f;
constexpr float kHeadLastRows = 2.0f;

/** Interpolation of the value noise (the head-switching band's noise has visible cells on purpose). */
constexpr float kValueNoiseNorm = 1.16f;
constexpr float kMinCell = 0.5f;
/** Most dust specks (and flecks) and pop lines one frame draws, and rows the Wrinkle band redraws. */
constexpr int kMaxSpecks = 20000;
constexpr int kMaxPops = 64;
constexpr int kMaxBandRows = 4096;

/** The noise fields live on a lattice with this spacing (full-size px) and at most this many points. */
constexpr float kLatticePx = 2.0f;
constexpr float kMaxCells = 4000000.0f;
constexpr float kMinFieldSigma = 0.7f;
constexpr float kMaxFieldSigma = 24.0f;

/** The planes of noise a frame reads: Y', Cb, Cr, the three Post Noise channels and the glow's grain. */
constexpr int kNoisePlanes = 7;
constexpr int kPlaneY = 0;
constexpr int kPlaneCb = 1;
constexpr int kPlaneCr = 2;
constexpr int kPlanePost = 3;  // three planes: R, G, B
constexpr int kPlaneGlow = 6;

/** Dust specks and flecks are looked up by tile. */
constexpr int kTileW = 64;
constexpr int kTileH = 32;

// ---------------------------------------------------------------------------------------------
// Data the host plans for a frame

/** A dust speck or a fleck: a soft Gaussian blob of brightness with a little colour, in working px. */
struct Event {
    float x, y;
    float amp, cb, cr;
    float sx, sy;  // Gaussian sigmas
    int x0, x1;    // columns it touches, [x0, x1)
    int fleck;     // 1 for a fleck (soft, weak, coloured, many), 0 for dust (bright, white, few)
};

/** A pop line: a short bright streak on one row that has lost its colour. */
struct Pop {
    float y, x0, x1, amp;
    int ix0, ix1;  // columns it touches, [ix0, ix1)
};

/** What one row needs besides the noise: its displacement, left edge, top gains and glow brightness. */
struct RowData {
    float shift;      // displacement to the right of the whole row
    float edge;       // how far the left edge of the tape frame is from where it rests
    float top_left;   // gain of the top of the picture left of the groove's step
    float top_right;  // and right of it
    float bead;       // brightness of the glow on this row against its mean
    float pad_;
};

/** One frame's worth of plain data for the final kernel (device pointers and the numbers around them). */
struct DamageParams {
    int w, h;
    float scale;
    std::uint64_t seed;
    int signal;    // the signal gets additions
    int noise_on;  // the tape noise fields are drawn
    int output;  // the output stage runs
    int frame;   // the tape frame is drawn
    int post_on;
    int head_on;
    int fall;    // the noise falls on bright picture
    float post;  // standard deviation of Post Noise per channel
    float fleck_dc[3];

    // events binned in tiles of kTileW x kTileH
    int tiles_x, tiles_y;
    const Event* events;
    const int* tile_start;
    const int* tile_events;
    float reach;  // rows above and below an event's centre that it reaches
    int n_pops;
    const Pop* pops;

    // noise planes, read by cubic interpolation: row y of plane p is lines[p] + y * stride
    const float* lines[kNoisePlanes];
    int stride;
    int pad;
    float inv;  // lattice cells per working pixel

    const RowData* rows;
};

// ---------------------------------------------------------------------------------------------
// Device functions

ARSENAL_HD void cubic_weights(float t, float w[4]) {
    const float t2 = t * t;
    const float t3 = t2 * t;
    w[0] = -0.5f * t3 + t2 - 0.5f * t;
    w[1] = 1.5f * t3 - 2.5f * t2 + 1.0f;
    w[2] = -1.5f * t3 + 2.0f * t2 + 0.5f * t;
    w[3] = 0.5f * t3 - 0.5f * t2;
}

/** A lattice value of unit variance. */
ARSENAL_HD float lattice_value(std::int64_t i, std::int64_t j, std::int64_t stream, std::uint64_t seed) {
    return (hash01(i, j, stream, seed) - 0.5f) * 3.464102f;
}

/**
 * Value noise of unit standard deviation at pixel (x, y) of a frame `w` wide: a lattice of
 * `cell_x` x `cell_y` full-size px laid over the picture, smoothly interpolated. It has cells on
 * purpose: it draws the streaky noise of the head-switching band.
 */
__device__ __forceinline__ float value_noise(int x, int y, int w, float scale, float cell_x, float cell_y,
                                             std::int64_t stream, std::uint64_t seed) {
    const float hi = 4.0f * static_cast<float>(w > 1 ? w : 1);
    const float cx = clampf(cell_x * scale, kMinCell, hi);
    const float cy = clampf(cell_y * scale, kMinCell, hi);
    const float v = (static_cast<float>(y) + 0.5f) / cy;
    const float jf = floorf(v);
    const float tv = smooth01(v - jf);
    const std::int64_t j = static_cast<std::int64_t>(jf);
    const float u = (static_cast<float>(x) + 0.5f) / cx;
    const float uf = floorf(u);
    const float t = smooth01(u - uf);
    const std::int64_t i = static_cast<std::int64_t>(uf);
    const float a = lerpf(lattice_value(i, j, stream, seed), lattice_value(i, j + 1, stream, seed), tv);
    const float b = lerpf(lattice_value(i + 1, j, stream, seed), lattice_value(i + 1, j + 1, stream, seed), tv);
    return lerpf(a, b, t) * kValueNoiseNorm;
}

/** The four lattice columns and weights a cubic read at working pixel x uses. */
struct NoiseTaps {
    int i[4];
    float w[4];
};

__device__ __forceinline__ NoiseTaps noise_taps(const DamageParams& d, int x) {
    NoiseTaps t;
    const float u = (static_cast<float>(x) + 0.5f) * d.inv - 0.5f + static_cast<float>(d.pad);
    const float uf = floorf(u);
    cubic_weights(u - uf, t.w);
    const int i = static_cast<int>(uf);
    const int last = d.stride - 1;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        t.i[k] = min(max(i + k - 1, 0), last);
    }
    return t;
}

__device__ __forceinline__ float read_noise(const float* row, const NoiseTaps& t) {
    return t.w[0] * row[t.i[0]] + t.w[1] * row[t.i[1]] + t.w[2] * row[t.i[2]] + t.w[3] * row[t.i[3]];
}

/**
 * Adds the damage to the signal of pixel (x, y): `ycc` is the display Y'CbCr of the picture, after the
 * blur, the delays and the sharpen, before the grade. `taps` are the noise lattice columns of x.
 */
__device__ __forceinline__ void damage_signal(const DamageParams& d, int x, int y, const NoiseTaps& taps,
                                              float ycc[3]) {
    const float sc = d.scale;
    const float xc = static_cast<float>(x) + 0.5f;
    const float yc = static_cast<float>(y) + 0.5f;

    // dust specks (added as they are) and flecks (which weaken on bright picture) that reach this pixel
    float dust[3] = {0.0f, 0.0f, 0.0f};
    float fleck[3] = {0.0f, 0.0f, 0.0f};
    if (d.tiles_x > 0) {
        const int tile = (y / kTileH) * d.tiles_x + (x / kTileW);
        const int end = d.tile_start[tile + 1];
        for (int k = d.tile_start[tile]; k < end; ++k) {
            const Event e = d.events[d.tile_events[k]];
            if (fabsf(e.y - yc) > d.reach || x < e.x0 || x >= e.x1) continue;
            const float ay = (yc - e.y) / e.sy;
            const float ax = (xc - e.x) / e.sx;
            const float g = expf(-0.5f * ay * ay) * expf(-0.5f * ax * ax);
            float* sum = e.fleck ? fleck : dust;
            sum[0] += e.amp * g;
            sum[1] += e.cb * g;
            sum[2] += e.cr * g;
        }
    }

    // pop lines: a short bright streak that loses its colour
    float keep = 1.0f;
    const float pop_sy = fmaxf(kPopSigmaY * sc, 0.3f);
    const float pop_edge = 6.0f * sc;
    for (int k = 0; k < d.n_pops; ++k) {
        const Pop p = d.pops[k];
        const float ay = (yc - p.y) / pop_sy;
        if (ay * ay > 8.0f) continue;  // the line's fall-off is under 2 % beyond 2.8 sigma
        const float g = expf(-0.5f * ay * ay);
        if (g < 0.02f) continue;
        if (x < p.ix0 || x >= p.ix1) continue;
        const float f = g * smooth01((xc - p.x0) / fmaxf(pop_edge, 0.5f)) *
                        (1.0f - smooth01((xc - p.x1 + pop_edge) / fmaxf(pop_edge, 0.5f)));
        dust[0] += p.amp * f;
        keep = fminf(keep, 1.0f - 0.9f * f);
    }

    // head-switching band: noise on the last rows of the picture
    const float head = kHeadRows * sc;
    const float depth = (yc - (static_cast<float>(d.h) - head)) / fmaxf(head, 0.5f);
    if (d.head_on && depth > 0.0f) {
        const float t = clampf(depth, 0.0f, 1.0f);
        const float n = value_noise(x, y, d.w, sc, kHeadCellX, kHeadCellY, S_HEAD_Y, d.seed);
        const float amp = lerpf(kHeadAmpLo, kHeadAmpHi, t);
        const float last = (static_cast<float>(y + 1) > static_cast<float>(d.h) - kHeadLastRows * sc) ? 0.0f : 1.0f;
        const float pos = fmaxf(n, 0.0f);
        dust[0] += last * (amp * 2.0f * pos * pos);
        dust[1] += kHeadChroma * t * value_noise(x, y, d.w, sc, kHeadCellX * 0.4f, kHeadCellY, S_HEAD_C, d.seed);
        dust[2] += kHeadChroma * t * 0.6f *
                   value_noise(x, y, d.w, sc, kHeadCellX * 0.4f, kHeadCellY, S_HEAD_C ^ 1, d.seed);
    }

    // the tape's noise dies away on black
    float ny = 0.0f, ncb = 0.0f, ncr = 0.0f;
    if (d.noise_on) {
        ny = read_noise(d.lines[kPlaneY] + static_cast<size_t>(y) * d.stride, taps);
        ncb = read_noise(d.lines[kPlaneCb] + static_cast<size_t>(y) * d.stride, taps);
        ncr = read_noise(d.lines[kPlaneCr] + static_cast<size_t>(y) * d.stride, taps);
    }
    const float dark = d.fall ? 1.0f - smooth01(ycc[0] / kNoiseFall) : 1.0f;
    const float g = lerpf(kNoiseFloor, 1.0f, dark);
    const float gf = lerpf(kFleckFloor, 1.0f, dark);
    const float y0 = ycc[0] + dust[0] + g * ny + gf * (fleck[0] - d.fleck_dc[0]);
    const float cb = (ycc[1] + dust[1] + g * ncb + gf * (fleck[1] - d.fleck_dc[1])) * keep;
    const float cr = (ycc[2] + dust[2] + g * ncr + gf * (fleck[2] - d.fleck_dc[2])) * keep;
    ycc[0] = y0;
    ycc[1] = cb;
    ycc[2] = cr;
}

/**
 * The output stage on the graded display R'G'B' of pixel (x, y): the tape frame and Post Noise. The
 * frame is on the tape, so it does not follow the picture's rows (only the black the shifted picture
 * leaves at the sides).
 */
__device__ __forceinline__ void damage_finish(const DamageParams& d, int x, int y, const NoiseTaps& taps,
                                              float rgb[3]) {
    const float sc = d.scale;
    const float xc = static_cast<float>(x) + 0.5f;
    const RowData row = d.rows[y];
    const float u = (xc - row.shift) / sc;
    const float full_w = static_cast<float>(d.w) / sc;
    if (u < 0.0f || u > full_w) {
        rgb[0] = rgb[1] = rgb[2] = 0.0f;
    } else if (d.frame) {
        const float uf = xc / sc;
        const float top = uf < kGrooveStep ? row.top_left : row.top_right;
        const float e_right = (u - (full_w - kRightCentre)) / kRightSoft;
        const float right = 1.0f / (1.0f + expf(1.7f * e_right));
        const float shade = 1.0f - kRightVignette * smooth01((u - (full_w - 160.0f)) / 160.0f);
        // the bar on the left, and the glow that lights the picture beside it
        const float e = uf - row.edge / sc - kLeftBar;
        const float inside = smooth01((e - kBarMid) / kBarWidth + 0.5f);
        const float onset = smooth01((e - kGlowOnset) / kGlowWidth + 0.5f);
        const float fade = expf(-fmaxf(e - kGlowAt, 0.0f) / kGlowTau);
        const float glow_noise =
            read_noise(d.lines[kPlaneGlow] + static_cast<size_t>(y) * d.stride, taps);
        const float grain = 1.0f + kGlowGrain * glow_noise;
        const float glow[3] = {kGlowR, kGlowG, kGlowB};
        const float k = clampf(kGlowPeak * onset * fade * row.bead * fmaxf(grain, 0.0f) * fminf(top, 1.0f), 0.0f, 1.0f);
#pragma unroll
        for (int ch = 0; ch < 3; ++ch) {
            const float picture = rgb[ch] * top * shade;
            const float lit = picture + (glow[ch] - picture) * k;
            rgb[ch] = lerpf(kBarBlack, lit, inside * right);
        }
    }
    if (d.post_on) {
#pragma unroll
        for (int ch = 0; ch < 3; ++ch) {
            const float n = read_noise(d.lines[kPlanePost + ch] + static_cast<size_t>(y) * d.stride, taps) * d.post;
            rgb[ch] = clampf(rgb[ch] + n, 0.0f, 1.0f);
        }
    }
}

} // namespace arsenal
} // namespace effects
} // namespace vhs
