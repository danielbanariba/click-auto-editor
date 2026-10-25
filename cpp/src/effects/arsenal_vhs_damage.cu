/**
 * Arsenal VHS - the damage stage, host side and noise fields.
 *
 * Ported from the Arsenal VHS effect of a FilmCraft build (MIT OR Apache-2.0).
 *
 * # Noise
 *
 * The tape noise is a Gaussian random field with no cells: white values hashed on a lattice laid over
 * the full-size picture (one point per 2 px, a value per point from the clip, Random Seed and the
 * frame's timestamp), smoothed by a Gaussian of the layer's sigma along the lattice, and read at the
 * pixel centres by cubic (Catmull-Rom) interpolation. A field smoothed by a sigma of 0.7 cells or more
 * has no energy at the lattice's own frequency to speak of, so it is the same statistically
 * everywhere: no grid, no tiles and no patches at any zoom. A frame at another scale reads the same
 * field at other positions, so it looks like the full-size one shrunk.
 *
 * The planes of noise a frame needs are sums of fields (for example brightness = soft mottle + fine
 * snow), and reading is linear, so the fields are summed on the lattice and each plane is read once.
 *
 * # Tape Wrinkle
 *
 * A horizontal band moves down the picture (60 px per second per unit of Scroll Speed, one pass takes
 * the picture height over the speed) and a pass is on with the probability Percent Chance, decided by
 * a hash of the clip, Seed and pass number. Inside it the rows are held (each is replaced by the first
 * row of its group, weighted by a cosine window) and cut into chunks that are shifted sideways, with
 * the colour planes shifted differently, so tile edges show as coloured fringes.
 *
 * # The left edge in motion
 *
 * The left edge of the tape frame runs through a loop of 30 frames of edge shapes (a tear that jumps
 * the top of the edge left every 11, 11 and 8 frames, and bulges to the right); the frame's number
 * picks the state. Below the shapes' reach two smooth value noises are added.
 */

#include "arsenal_vhs_damage.hpp"
#include "utils/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace vhs {
namespace effects {
namespace arsenal {

namespace {

// ---- the left edge's loop ----

/** The state of the left edge in each frame of its loop of 30 (an index into kEdgeStates). */
constexpr int kEdgeLoop[30] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 9, 0, 1, 2, 3, 4, 8, 3, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 9};

/**
 * The shapes of the left edge's top, one per state: how far the edge is to the right of where it rests
 * (full-size px, at Frame Jitter 2), at knots every 32 px down from the top of the picture, 0 below the
 * last. State 1 is the tear: the whole top of the edge jumps left.
 */
constexpr float kEdgeStates[10][16] = {
    {-0.6f, -0.3f, 3.0f, 8.7f, 9.4f, 8.0f, 6.3f, 3.8f, 3.4f, 1.4f, -0.1f, -0.5f, 0.0f, -0.1f, -0.2f, 0.2f},
    {-12.2f, -13.5f, -13.7f, -15.1f, -14.8f, -13.2f, -13.7f, -14.8f, -11.1f, -11.2f, -12.1f, -11.9f, -10.1f, -2.2f, 1.0f, 0.4f},
    {1.7f, 2.3f, 11.3f, 14.2f, 12.3f, 10.2f, 6.9f, 4.7f, 6.3f, 4.8f, 1.9f, 0.4f, -0.2f, 0.2f, -0.6f, 0.2f},
    {-0.5f, -1.2f, -2.4f, 5.3f, 8.3f, 7.8f, 5.1f, 3.7f, 2.8f, 1.4f, 2.2f, 1.9f, -0.3f, -0.3f, -0.7f, -0.3f},
    {-5.2f, -5.2f, -6.7f, -4.4f, -0.5f, 2.6f, 2.9f, 2.0f, 2.4f, 3.7f, 2.7f, 0.7f, 1.2f, 0.9f, 0.3f, -0.2f},
    {3.3f, 3.3f, 0.2f, 1.1f, 7.7f, 7.0f, 5.1f, 4.6f, 3.6f, 2.2f, 2.0f, 1.3f, 0.5f, 0.7f, -0.4f, 1.0f},
    {7.6f, 4.6f, 3.0f, 2.6f, -0.5f, -0.5f, -0.6f, -1.6f, -0.6f, -0.8f, 0.0f, 0.1f, -0.8f, -1.0f, -1.5f, 0.5f},
    {7.6f, 8.1f, 7.5f, 14.3f, 15.2f, 13.2f, 9.0f, 6.4f, 5.2f, 2.5f, 1.0f, 0.6f, 0.2f, -0.1f, -0.6f, -0.1f},
    {-5.6f, -5.6f, -7.0f, -6.8f, 1.3f, 3.3f, 1.7f, 2.9f, 2.2f, 0.5f, 0.4f, -0.2f, -0.9f, -1.3f, -0.4f, -1.0f},
    {8.4f, 16.7f, 20.5f, 18.5f, 13.9f, 11.5f, 10.2f, 6.3f, 4.9f, 5.0f, 3.4f, 2.2f, 0.7f, -0.4f, -0.3f, -0.5f},
};

/** The shape of `state` at `knot` knots down from the top, interpolated between knots; 0 past the last. */
float edge_state_offset(int state, float knot) {
    if (state < 0 || state >= 10 || !(knot == knot)) return 0.0f;
    const float k = fmaxf(knot, 0.0f);
    if (k >= 16.0f) return 0.0f;
    const int i = static_cast<int>(floorf(k));
    const float t = k - static_cast<float>(i);
    const float a = kEdgeStates[state][i];
    const float b = i + 1 < 16 ? kEdgeStates[state][i + 1] : 0.0f;
    return lerpf(a, b, t);
}

// ---- value noise along one axis ----

/** Value noise of unit standard deviation along one axis (rows), at `pos` cells. */
float noise_1d(float pos, std::int64_t stream, std::uint64_t seed) {
    const float i = floorf(pos);
    const float t = smooth01(pos - i);
    const std::int64_t ii = static_cast<std::int64_t>(i);
    return lerpf(lattice_value(ii, 0, stream, seed), lattice_value(ii + 1, 0, stream, seed), t) * kValueNoiseNorm;
}

float control_of(float v, float lo, float hi, float def) {
    return std::isnan(v) ? def : std::min(std::max(v, lo), hi);
}

// ---- the plan of one frame ----

struct BandRow {
    int src;          // the row it is redrawn from
    int chunk_start;  // its chunks in the plan's list
    int chunk_count;
};

struct BandChunk {
    int end;      // working px, exclusive: the chunk runs from the previous chunk's end to here
    float dx[3];  // how far the Y', Cb and Cr planes move to the right
};

struct Plan {
    bool on = false;      // Enable Tape Damage
    bool frame = false;   // Frame Style 0: the tape frame
    bool post_on = false;
    float post = 0.0f;
    std::uint64_t seed = 0;

    std::vector<RowData> rows;
    std::vector<Event> events;
    std::vector<Pop> pops;
    float reach = 0.0f;
    float fleck_dc[3] = {0.0f, 0.0f, 0.0f};
    int tiles_x = 0;
    int tiles_y = 0;
    std::vector<int> tile_start;
    std::vector<int> tile_events;

    int band_y0 = 0;
    std::vector<BandRow> band_rows;
    std::vector<BandChunk> band_chunks;

    bool active() const { return on || frame || post_on; }
};

/** Everything of a frame that depends on its place in the clip and the seed, in working pixels. */
class Planner {
public:
    Planner(const Controls& c, const FrameInfo& f, int w, int h, float scale)
        : c_(c), f_(f), w_(w), h_(h), scale_(scale) {
        clip_ = f.clip_key;
        user_seed_ = static_cast<std::int64_t>(control_of(c.seed, 1000.0f, 50000.0f, 9300.0f));
        seed_ = hash_u(clip_, user_seed_, S_FRAME, f.tick);
    }

    Plan plan() {
        Plan p;
        p.seed = seed_;
        p.on = c_.tape_damage;
        p.frame = c_.tape_frame;
        p.post = control_of(c_.post_noise, 0.0f, 100.0f, 3.0f) * kPostSigma;
        p.post_on = p.post > 0.0f;

        const float area_mpx = static_cast<float>(w_) * static_cast<float>(h_) / (scale_ * scale_) / 1e6f;
        const float pop = p.on ? control_of(c_.pop_line_freq, 0.0f, 100.0f, 25.0f) / 100.0f : 0.0f;
        const float jitter = p.on ? control_of(c_.frame_jitter, 0.0f, 20.0f, 2.0f) : 0.0f;

        // the top of the picture is foggy in some frames and grooved in others
        const float pick = hash01(0, 0, S_TOP, seed_);
        top_mix_ = pick < kTopFog ? 0.0f : (pick < kTopFog + kTopHalf ? 0.5f : 1.0f);
        groove_shift_ = kGrooveJitter * (2.0f * hash01(0, 1, S_TOP, seed_) - 1.0f);
        edge_seed_ = hash_u(clip_, user_seed_, S_EDGE, 0);
        const std::int64_t n = f_.frame_number;
        edge_state_ = kEdgeLoop[((n % 30) + 30) % 30];

        p.reach = 3.0f * fmaxf(fmaxf(kSpeckRy, kFleckRHi * kFleckAspect) * scale_, 0.3f);
        if (p.on) {
            draw_specks(p, kLookSpecks * area_mpx, false);
            draw_specks(p, kLookFlecks * area_mpx, true);
            std::stable_sort(p.events.begin(), p.events.end(), [](const Event& a, const Event& b) { return a.y < b.y; });
            draw_pops(p, pop * kPopPerPercent * 100.0f);
            bin_events(p);
        }
        p.rows = rows(p, jitter);
        if (c_.wrinkle) {
            draw_band(p, control_of(c_.wrinkle_speed, 0.0f, 200.0f, 5.0f),
                      control_of(c_.wrinkle_chance, 0.0f, 100.0f, 40.0f) / 100.0f,
                      control_of(c_.wrinkle_radius, 0.0f, 20.0f, 5.0f), f_.seconds);
        }
        return p;
    }

private:
    const Controls& c_;
    const FrameInfo& f_;
    int w_, h_;
    float scale_;
    std::int64_t clip_ = 0;
    std::int64_t user_seed_ = 0;
    std::uint64_t seed_ = 0;
    std::uint64_t edge_seed_ = 0;
    int edge_state_ = 0;
    float top_mix_ = 0.0f;
    float groove_shift_ = 0.0f;

    // ---- dust, flecks and pop lines ----

    void draw_specks(Plan& p, float expected, bool fleck) {
        if (std::isnan(expected) || expected <= 0.0f) return;
        const std::int64_t stream = fleck ? (S_SPECK ^ 0x464c) : S_SPECK;
        const float count = expected + hash01(0, 1, stream, seed_);
        const int n = static_cast<int>(std::min(count, static_cast<float>(kMaxSpecks)));
        const float fw = static_cast<float>(w_);
        const float fh = static_cast<float>(h_);
        const float area = static_cast<float>(std::max<std::int64_t>(static_cast<std::int64_t>(w_) * h_, 1));
        for (int k = 0; k < n && static_cast<int>(p.events.size()) < 2 * kMaxSpecks; ++k) {
            auto u = [&](std::int64_t s) { return hash01(k, s, stream, seed_); };
            const float kind = u(5);
            Event e{};
            e.x = u(2) * fw;
            e.y = u(3) * fh;
            float rx, ry;
            if (fleck) {
                // white, violet, blue and yellow-green, as the reference's
                float cb, cr;
                if (kind < 0.30f) {
                    cb = -0.4f; cr = 0.0f;
                } else if (kind < 0.55f) {
                    cb = 0.4f; cr = 0.2f;
                } else if (kind < 0.88f) {
                    cb = 0.4f; cr = -0.2f;
                } else {
                    cb = -1.2f; cr = -0.4f;
                }
                const float t = u(4);
                rx = lerpf(kFleckRLo, kFleckRHi, u(6));
                ry = rx * kFleckAspect;
                e.amp = fminf(kFleckAmp + kFleckAmp * -logf(1.0f - 0.999f * t), 0.6f);
                e.cb = cb * kFleckChroma;
                e.cr = cr * kFleckChroma;
            } else {
                // a speck is mostly white, sometimes green, blue or violet, rarely red
                float cb, cr;
                if (kind < 0.5f) {
                    cb = 0.0f; cr = 0.0f;
                } else if (kind < 0.7f) {
                    cb = -0.5f; cr = -0.8f;
                } else if (kind < 0.9f) {
                    cb = 0.9f; cr = 0.0f;
                } else {
                    cb = 0.1f; cr = 0.9f;
                }
                rx = kSpeckRx;
                ry = kSpeckRy;
                e.amp = lerpf(kSpeckAmpLo, kSpeckAmpHi, u(4));
                e.cb = cb * kSpeckTint;
                e.cr = cr * kSpeckTint;
            }
            e.fleck = fleck ? 1 : 0;
            e.sx = fmaxf(rx * scale_, 0.3f);
            e.sy = fmaxf(ry * scale_, 0.3f);
            e.x0 = static_cast<int>(floorf(fmaxf(e.x - 3.0f * e.sx, 0.0f)));
            e.x1 = std::min(static_cast<int>(ceilf(fmaxf(e.x + 3.0f * e.sx, 0.0f))), w_);
            if (fleck) {
                // what the flecks add to the mean of the picture, to give it back (the tape's snow adds no light on average)
                const float g = 6.2831853f * e.sx * e.sy / area;
                p.fleck_dc[0] += e.amp * g;
                p.fleck_dc[1] += e.cb * g;
                p.fleck_dc[2] += e.cr * g;
            }
            p.events.push_back(e);
        }
    }

    void draw_pops(Plan& p, float expected) {
        if (std::isnan(expected) || expected <= 0.0f) return;
        const float count = expected + hash01(0, 1, S_POP, seed_);
        const int n = static_cast<int>(std::min(count, static_cast<float>(kMaxPops)));
        const float fw = static_cast<float>(w_);
        const float fh = static_cast<float>(h_);
        const float edge = 6.0f * scale_;
        for (int k = 0; k < n; ++k) {
            auto u = [&](std::int64_t s) { return hash01(k, s, S_POP, seed_); };
            const float len = fminf((kPopLenLo + (kPopLenHi - kPopLenLo) * u(4) * u(4)) * scale_, fw);
            Pop pop{};
            pop.x0 = u(2) * fmaxf(fw - len, 0.0f);
            pop.x1 = pop.x0 + len;
            pop.y = u(3) * fh;
            pop.amp = lerpf(kPopAmpLo, kPopAmpHi, u(5));
            pop.ix0 = static_cast<int>(floorf(fmaxf(pop.x0 - edge, 0.0f)));
            pop.ix1 = std::min(static_cast<int>(ceilf(fmaxf(pop.x1 + edge, 0.0f))), w_);
            p.pops.push_back(pop);
        }
    }

    /** Lists, for each tile of the picture, the events that can reach it (in the order of the sorted events). */
    void bin_events(Plan& p) const {
        p.tiles_x = (w_ + kTileW - 1) / kTileW;
        p.tiles_y = (h_ + kTileH - 1) / kTileH;
        const size_t tiles = static_cast<size_t>(p.tiles_x) * p.tiles_y;
        std::vector<int> count(tiles + 1, 0);
        auto range = [&](const Event& e, int& tx0, int& tx1, int& ty0, int& ty1) {
            tx0 = e.x0 / kTileW;
            tx1 = std::min((std::max(e.x1, e.x0 + 1) - 1) / kTileW, p.tiles_x - 1);
            ty0 = std::max(static_cast<int>(floorf((e.y - p.reach - 0.5f) / kTileH)), 0);
            ty1 = std::min(static_cast<int>(floorf((e.y + p.reach - 0.5f) / kTileH)), p.tiles_y - 1);
        };
        for (int pass = 0; pass < 2; ++pass) {
            if (pass == 1) {
                p.tile_start.assign(tiles + 1, 0);
                for (size_t t = 0; t < tiles; ++t) p.tile_start[t + 1] = p.tile_start[t] + count[t];
                p.tile_events.assign(static_cast<size_t>(p.tile_start[tiles]), 0);
                std::fill(count.begin(), count.end(), 0);
            }
            for (size_t k = 0; k < p.events.size(); ++k) {
                const Event& e = p.events[k];
                int tx0, tx1, ty0, ty1;
                range(e, tx0, tx1, ty0, ty1);
                if (e.x1 <= e.x0) continue;
                for (int ty = ty0; ty <= ty1; ++ty) {
                    for (int tx = tx0; tx <= tx1; ++tx) {
                        const size_t t = static_cast<size_t>(ty) * p.tiles_x + tx;
                        if (pass == 0) {
                            ++count[t];
                        } else {
                            p.tile_events[static_cast<size_t>(p.tile_start[t] + count[t]++)] = static_cast<int>(k);
                        }
                    }
                }
            }
        }
    }

    // ---- rows ----

    /** Where the left edge of the tape frame is at row `full` (full-size px), to the right of where it rests. */
    float edge_offset(float full) const {
        const float top = edge_state_offset(edge_state_, full / kEdgeKnot);
        const float still = kEdgeStillAmp * noise_1d(full / kEdgeStillCell, S_EDGE, edge_seed_);
        const float own = kEdgeOwnAmp * noise_1d(full / kEdgeOwnCell, S_EDGE ^ (1 + edge_state_), edge_seed_);
        return top + still + own;
    }

    /** Gain of the top of the picture on row `y` for one side of the groove's step. */
    float top_gain(int y, float depth, float centre, float width) const {
        const float v = (static_cast<float>(y) + 0.5f) / scale_;
        const float vignette = 1.0f - kTopVignette * expf(-v / kTopVignetteTau);
        const float fog = (kFogFloor + (1.0f - kFogFloor) * (1.0f - expf(-powf(v / kFogTau, kFogPower))) +
                           kFogEdge * expf(-powf(v / kFogEdgeLen, 2.0f))) *
                          vignette;
        const float d = (v - centre - groove_shift_) / width;
        const float thin = vignette * (1.0f - depth * expf(-d * d));
        return lerpf(fog, thin, top_mix_);
    }

    /** How bright the glow of the left edge is on row `y` against its mean: two smooth scales of noise. */
    float bead(int y) const {
        const float full = (static_cast<float>(y) + 0.5f) / scale_;
        const float n = 0.8f * noise_1d(full / kBeadCell0, S_BEAD, seed_) + 0.6f * noise_1d(full / kBeadCell1, S_BEAD ^ 1, seed_);
        return fmaxf(1.0f + kBeadSd * n, 0.0f);
    }

    std::vector<RowData> rows(const Plan& p, float jitter) const {
        std::vector<RowData> out(static_cast<size_t>(h_));
        for (int y = 0; y < h_; ++y) {
            const float full = (static_cast<float>(y) + 0.5f) / scale_;
            RowData r{};
            r.top_left = 1.0f;
            r.top_right = 1.0f;
            r.bead = 1.0f;
            if (jitter > 0.0f) {
                const float share = kJitterSlowShare;
                const float rest = sqrtf(fmaxf(1.0f - share * share, 0.0f));
                const float slow = noise_1d(full / kJitterSlowCell, S_SLOW, seed_);
                const float fast = lattice_value(static_cast<std::int64_t>(floorf(full / kJitterFastCell)), 0, S_FAST, seed_);
                r.shift = jitter * kJitterPx * scale_ * (share * slow + rest * fast);
                if (p.frame) r.edge = jitter / kEdgeRefJitter * scale_ * edge_offset(full);
            }
            if (p.frame) {
                r.top_left = top_gain(y, kGrooveLeftDepth, kGrooveLeftCentre, kGrooveLeftWidth);
                r.top_right = top_gain(y, kGrooveRightDepth, kGrooveRightCentre, kGrooveRightWidth);
                r.bead = bead(y);
            }
            out[static_cast<size_t>(y)] = r;
        }
        return out;
    }

    // ---- the Wrinkle band ----

    /** The chunks of row group `group` at full strength: cut at hashed places, some shifted sideways. */
    std::vector<BandChunk> chunks(std::int64_t group) const {
        auto u = [&](std::int64_t k, std::int64_t n) { return hash01(group, k * 8 + n, S_TILE, seed_); };
        const float mean = lerpf(kWrChunkLo, kWrChunkHi, u(-1, 0)) * scale_;
        float x = -u(-1, 1) * mean;
        std::vector<BandChunk> out;
        std::int64_t k = 0;
        while (x < static_cast<float>(w_) && static_cast<int>(out.size()) < w_ / 2 + 2) {
            x += fmaxf(mean * lerpf(0.5f, 1.5f, u(k, 0)), 2.0f);
            const float moved = u(k, 1) < kWrShiftShare ? (2.0f * u(k, 2) - 1.0f) * kWrShift * scale_ : 0.0f;
            BandChunk ch{};
            ch.end = static_cast<int>(ceilf(fmaxf(x, 0.0f)));
            ch.dx[0] = moved;
            ch.dx[1] = moved + (2.0f * u(k, 3) - 1.0f) * kWrChroma * scale_;
            ch.dx[2] = moved + (2.0f * u(k, 4) - 1.0f) * kWrChroma * scale_;
            out.push_back(ch);
            ++k;
        }
        if (!out.empty()) out.back().end = w_;
        return out;
    }

    /** The Wrinkle band at `seconds`: where it is, whether this pass is on, and how its rows are redrawn. */
    void draw_band(Plan& p, float speed, float chance, float radius, double seconds) const {
        const float s = scale_;
        const float full_h = static_cast<float>(h_) / s;
        const float half = fminf(radius * kWrHalf, full_h);
        if (radius <= 0.0f || half < 1.0f || !std::isfinite(seconds) || !std::isfinite(full_h) || full_h <= 0.0f) return;
        const std::uint64_t key0 = hash_u(clip_, user_seed_, S_WRINKLE, 0);
        const double phase = static_cast<double>(hash01(0, 1, S_WRINKLE, key0) * full_h);
        const double run = phase + seconds * static_cast<double>(speed * kWrSpeed);
        const double pass = std::floor(run / static_cast<double>(full_h));
        const float centre = static_cast<float>(run - pass * static_cast<double>(full_h));
        const std::uint64_t key = hash_u(clip_, user_seed_, S_WRINKLE, static_cast<std::uint64_t>(static_cast<std::int64_t>(pass)));
        if (hash01(0, 0, S_WRINKLE, key) >= chance) return;
        // grows in at the top of the picture and out at the bottom, so a pass has no visible start or end
        const float fade = speed > 0.0f ? smooth01(centre / half) * smooth01((full_h - centre) / half) : 1.0f;
        const int first = static_cast<int>(fmaxf(floorf((centre - half) * s), 0.0f));
        const int last = std::min(static_cast<int>(fmaxf(ceilf((centre + half) * s), 0.0f)), h_);
        if (last <= first || fade <= 0.0f) return;
        const float group_h = lerpf(kWrGroupLo, kWrGroupHi, hash01(0, 2, S_TILE, seed_));
        p.band_y0 = first;
        std::int64_t cached_group = INT64_MIN;
        std::vector<BandChunk> base;
        for (int y = first; y < std::min(last, first + kMaxBandRows); ++y) {
            const float full = (static_cast<float>(y) + 0.5f) / s;
            const float r = fminf(fabsf(full - centre) / half, 1.0f);
            const float taper = r <= kWrCore ? 1.0f : powf(cosf(0.5f * 3.14159265f * (r - kWrCore) / (1.0f - kWrCore)), 2.0f);
            const float strength = fade * taper;
            const std::int64_t group = static_cast<std::int64_t>(floorf(full / group_h));
            const int start = static_cast<int>(fmaxf(ceilf(static_cast<float>(group) * group_h * s), 0.0f));
            const int src = std::min(static_cast<int>(fmaxf(roundf(lerpf(static_cast<float>(y), static_cast<float>(start), strength)), 0.0f)), y);
            if (group != cached_group) {
                cached_group = group;
                base = chunks(group);
            }
            BandRow row{};
            row.src = src;
            row.chunk_start = static_cast<int>(p.band_chunks.size());
            row.chunk_count = static_cast<int>(base.size());
            for (const BandChunk& ch : base) {
                BandChunk scaled = ch;
                for (int k = 0; k < 3; ++k) scaled.dx[k] *= strength;
                p.band_chunks.push_back(scaled);
            }
            p.band_rows.push_back(row);
        }
    }
};

// =============================================================================================
// Kernels

constexpr int kMaxTaps = 2 * 72 + 1;  // sigma 24 cells reaches 72
constexpr int kFieldThreads = 128;
constexpr int kResampleRows = 8;  // picture rows one thread of the resampling walks

/** A normalised Gaussian kernel. */
struct Taps {
    int r;
    float k[kMaxTaps];
};

/** A field's vertical pass writes (or adds) its scaled result to up to three planes. */
struct OutBatch {
    float* dst[3];
    float gain[3];
    int accumulate[3];
};

struct RowBatch {
    const float* src[kNoisePlanes];
    float* dst[kNoisePlanes];
};

struct BandBatch {
    const float* in[3];
    float* out[3];
};

/** White values of the lattice, unit variance; the lattice is padded by `pad` points all round. */
__global__ void lattice_white_kernel(float* out, int stride, int rows, int pad, std::int64_t stream, std::uint64_t seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (i >= stride || j >= rows) return;
    out[static_cast<size_t>(j) * stride + i] = lattice_value(i - pad, j - pad, stream, seed);
}

/** Smooths every lattice row along the line; the ends are repeated. */
__global__ void conv_x_kernel(const float* __restrict__ white, float* __restrict__ out, int stride, int rows, Taps taps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (x >= stride || j >= rows) return;
    const float* row = white + static_cast<size_t>(j) * stride;
    float sum = 0.0f;
    for (int m = 0; m <= 2 * taps.r; ++m) {
        sum += taps.k[m] * row[min(max(x + m - taps.r, 0), stride - 1)];
    }
    out[static_cast<size_t>(j) * stride + x] = sum;
}

/** Smooths down the lines, scales to a unit standard deviation and puts the result in the planes. */
__global__ void conv_y_kernel(const float* __restrict__ in, int stride, int rows, Taps taps, float norm, OutBatch out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (x >= stride || j >= rows) return;
    float sum = 0.0f;
    for (int m = 0; m <= 2 * taps.r; ++m) {
        sum += taps.k[m] * in[static_cast<size_t>(min(max(j + m - taps.r, 0), rows - 1)) * stride + x];
    }
    const float v = sum * norm;
    const size_t at = static_cast<size_t>(j) * stride + x;
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        if (!out.dst[q]) continue;
        out.dst[q][at] = out.accumulate[q] ? out.dst[q][at] + out.gain[q] * v : out.gain[q] * v;
    }
}

/**
 * Reads the lattice planes down to one row per picture row (the cubic read along the columns). A thread
 * walks kResampleRows rows of one column, so neighbouring rows reuse the lattice rows it has just read.
 */
__global__ void resample_rows_kernel(RowBatch batch, int stride, int lat_rows, int h, float inv, int pad) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = blockIdx.z;
    if (x >= stride || !batch.src[p]) return;
    const float* src = batch.src[p];
    float* dst = batch.dst[p];
    const int first = blockIdx.y * kResampleRows;
    const int last = min(first + kResampleRows, h);
    for (int y = first; y < last; ++y) {
        const float pos = (static_cast<float>(y) + 0.5f) * inv - 0.5f + static_cast<float>(pad);
        const float jf = floorf(pos);
        float w[4];
        cubic_weights(pos - jf, w);
        const int j = static_cast<int>(jf);
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            sum += w[k] * src[static_cast<size_t>(min(max(j + k - 1, 0), lat_rows - 1)) * stride + x];
        }
        dst[static_cast<size_t>(y) * stride + x] = sum;
    }
}

/** Redraws the rows of the Wrinkle band: each from its held row, chunk by chunk shifted sideways. */
__global__ void band_warp_kernel(BandBatch batch, const BandRow* rows, const BandChunk* chunks, int y0, int n_rows, int w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    const int which = blockIdx.z;
    if (x >= w || r >= n_rows) return;
    const BandRow row = rows[r];
    const float* in = batch.in[which];
    float* out = batch.out[which];
    // the first chunk that ends after x (chunks are in order and the last ends at the line's end)
    int lo = 0, hi = row.chunk_count;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (chunks[row.chunk_start + mid].end > x) hi = mid; else lo = mid + 1;
    }
    const size_t at = static_cast<size_t>(y0 + r) * w + x;
    if (lo >= row.chunk_count) {
        out[at] = in[at];
        return;
    }
    const float dx = chunks[row.chunk_start + lo].dx[which];
    const float* src = in + static_cast<size_t>(row.src) * w;
    const float p = clampf(static_cast<float>(x) - dx, 0.0f, static_cast<float>(w - 1));
    const int i = static_cast<int>(p);
    const float t = p - static_cast<float>(i);
    const float a = src[i];
    const float b = src[min(i + 1, w - 1)];
    out[at] = a + (b - a) * t;
}

void check_launch() {
    CUDA_CHECK(cudaGetLastError());
}

/** Taps of a Gaussian of `sigma` cells. */
Taps make_taps(float sigma, float& sum_sq) {
    Taps t{};
    const int r = std::min(std::max(static_cast<int>(std::ceil(3.0f * sigma)), 1), (kMaxTaps - 1) / 2);
    t.r = r;
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) {
        const double v = std::exp(-0.5 * (static_cast<double>(i) / sigma) * (static_cast<double>(i) / sigma));
        t.k[i + r] = static_cast<float>(v);
        sum += v;
    }
    double sq = 0.0;
    for (int i = 0; i <= 2 * r; ++i) {
        t.k[i] = static_cast<float>(t.k[i] / sum);
        sq += static_cast<double>(t.k[i]) * t.k[i];
    }
    sum_sq = static_cast<float>(sq);
    return t;
}

size_t align16(size_t n) {
    return (n + 15) & ~size_t(15);
}

} // namespace

// =============================================================================================
// DamageStage

struct DamageStage::Impl {
    int w, h;
    float scale;

    // the lattice of the noise fields
    float spacing = kLatticePx;
    int cols = 0, lat_rows = 0, pad = 0, stride = 0, rows_padded = 0;
    float* white = nullptr;
    float* smoothed = nullptr;
    float* plane[kNoisePlanes] = {};
    float* lines[kNoisePlanes] = {};

    // what one frame uploads
    unsigned char* d_blob = nullptr;
    size_t d_cap = 0;
    struct Slot {
        unsigned char* host = nullptr;
        size_t cap = 0;
        cudaEvent_t done = nullptr;
    } slots[3];
    int next_slot = 0;
    cudaStream_t last_stream = nullptr;
    bool used = false;

    // the frame in flight
    int band_rows = 0;
    int band_y0 = 0;
    const BandRow* d_band_rows = nullptr;
    const BandChunk* d_band_chunks = nullptr;
    bool ready = false;

    Impl(int width, int height) : w(width), h(height) {
        const double s = static_cast<double>(w) / kReferenceWidth;
        scale = static_cast<float>(s);
    }

    void make_grid() {
        const float sc = fmaxf(scale, 1e-4f);
        const float full_w = static_cast<float>(w) / sc;
        const float full_h = static_cast<float>(h) / sc;
        // the padding the widest field of the look needs so that its smoothing reaches the picture's edge
        pad = static_cast<int>(std::ceil(3.0f * kLookChromaSigma)) + 2;
        auto size = [&](float spacing_px, float& c, float& r) {
            c = std::ceil(full_w / spacing_px) + 1.0f;
            r = std::ceil(full_h / spacing_px) + 1.0f;
        };
        spacing = kLatticePx;
        for (int i = 0; i < 200; ++i) {
            float c, r;
            size(spacing, c, r);
            if ((c + 2.0f * pad) * (r + 2.0f * pad) <= kMaxCells) break;
            spacing *= 1.25f;
        }
        float c, r;
        size(spacing, c, r);
        cols = static_cast<int>(c);
        lat_rows = static_cast<int>(r);
        stride = cols + 2 * pad;
        rows_padded = lat_rows + 2 * pad;
    }

    void allocate() {
        make_grid();
        const size_t lattice_bytes = static_cast<size_t>(stride) * rows_padded * sizeof(float);
        const size_t line_bytes = static_cast<size_t>(stride) * h * sizeof(float);
        CUDA_CHECK(cudaMalloc(&white, lattice_bytes));
        CUDA_CHECK(cudaMalloc(&smoothed, lattice_bytes));
        for (int p = 0; p < kNoisePlanes; ++p) {
            CUDA_CHECK(cudaMalloc(&plane[p], lattice_bytes));
            CUDA_CHECK(cudaMalloc(&lines[p], line_bytes));
        }
        for (Slot& s : slots) {
            CUDA_CHECK(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
        }
    }

    void release() noexcept {
        // Errors are ignored on purpose: this also runs after the device was reset.
        if (white) cudaFree(white);
        if (smoothed) cudaFree(smoothed);
        white = smoothed = nullptr;
        for (int p = 0; p < kNoisePlanes; ++p) {
            if (plane[p]) cudaFree(plane[p]);
            if (lines[p]) cudaFree(lines[p]);
            plane[p] = lines[p] = nullptr;
        }
        if (d_blob) cudaFree(d_blob);
        d_blob = nullptr;
        d_cap = 0;
        for (Slot& s : slots) {
            if (s.host) cudaFreeHost(s.host);
            if (s.done) cudaEventDestroy(s.done);
            s = Slot{};
        }
        ready = false;
    }

    /** One field: white values on the lattice, smoothed, scaled to unit deviation, added to the planes. */
    void field(std::int64_t stream_id, std::uint64_t seed, float sigma_x, float sigma_y, const OutBatch& out,
               cudaStream_t stream) {
        sigma_x = fminf(fmaxf(sigma_x, kMinFieldSigma), kMaxFieldSigma);
        sigma_y = fminf(fmaxf(sigma_y, kMinFieldSigma), kMaxFieldSigma);
        float sq_x, sq_y;
        const Taps kx = make_taps(sigma_x, sq_x);
        const Taps ky = make_taps(sigma_y, sq_y);
        const float norm = 1.0f / sqrtf(sq_x * sq_y);
        const dim3 grid((stride + kFieldThreads - 1) / kFieldThreads, rows_padded);
        lattice_white_kernel<<<grid, kFieldThreads, 0, stream>>>(white, stride, rows_padded, pad, stream_id, seed);
        check_launch();
        conv_x_kernel<<<grid, kFieldThreads, 0, stream>>>(white, smoothed, stride, rows_padded, kx);
        check_launch();
        conv_y_kernel<<<grid, kFieldThreads, 0, stream>>>(smoothed, stride, rows_padded, ky, norm, out);
        check_launch();
    }

    static OutBatch to(float* a, float ga, bool acc_a, float* b = nullptr, float gb = 0.0f, float* c = nullptr, float gc = 0.0f) {
        OutBatch o{};
        o.dst[0] = a;
        o.gain[0] = ga;
        o.accumulate[0] = acc_a ? 1 : 0;
        o.dst[1] = b;
        o.gain[1] = gb;
        o.dst[2] = c;
        o.gain[2] = gc;
        return o;
    }

    /** Uploads `bytes` to the device blob through a pinned staging slot; returns the slot's host memory. */
    unsigned char* stage(size_t bytes, cudaStream_t stream) {
        Slot& s = slots[next_slot];
        next_slot = (next_slot + 1) % 3;
        CUDA_CHECK(cudaEventSynchronize(s.done));  // the copy that last used this slot is over
        if (s.cap < bytes) {
            if (s.host) CUDA_CHECK(cudaFreeHost(s.host));
            s.host = nullptr;
            s.cap = 0;
            const size_t cap = bytes + bytes / 2 + 4096;
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&s.host), cap));
            s.cap = cap;
        }
        if (d_cap < bytes) {
            CUDA_CHECK(cudaStreamSynchronize(stream));  // kernels of the last frame may still read the old blob
            if (d_blob) CUDA_CHECK(cudaFree(d_blob));
            d_blob = nullptr;
            d_cap = 0;
            const size_t cap = bytes + bytes / 2 + 4096;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_blob), cap));
            d_cap = cap;
        }
        return s.host;
    }

    void commit(size_t bytes, cudaStream_t stream) {
        Slot& s = slots[(next_slot + 2) % 3];  // the slot stage() handed out
        CUDA_CHECK(cudaMemcpyAsync(d_blob, s.host, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(s.done, stream));
    }
};

DamageStage::DamageStage(int width, int height) : impl_(new Impl(width, height)) {}

DamageStage::~DamageStage() {
    cleanup();
}

void DamageStage::init() {
    if (impl_->ready || impl_->w <= 0 || impl_->h <= 0) return;
    try {
        impl_->allocate();
    } catch (...) {
        impl_->release();
        throw;
    }
    impl_->ready = true;
}

void DamageStage::cleanup() noexcept {
    if (impl_) impl_->release();
}

bool DamageStage::has_band() const {
    return impl_->band_rows > 0;
}

DamageParams DamageStage::prepare(const Controls& controls, const FrameInfo& frame, cudaStream_t stream) {
    Impl& m = *impl_;
    DamageParams d{};
    m.band_rows = 0;
    if (!m.ready) init();
    if (!m.ready) return d;

    // frames are drawn in order on one stream: the device blob is reused without a fence
    if (m.used && stream != m.last_stream) {
        CUDA_CHECK(cudaStreamSynchronize(m.last_stream));
    }
    m.used = true;
    m.last_stream = stream;

    const Plan plan = Planner(controls, frame, m.w, m.h, m.scale).plan();
    if (!plan.active() && plan.band_rows.empty()) {
        return d;
    }

    // ---- upload ----
    const size_t n_rows = plan.rows.size();
    const size_t off_rows = 0;
    const size_t off_events = align16(off_rows + n_rows * sizeof(RowData));
    const size_t off_tile_start = align16(off_events + plan.events.size() * sizeof(Event));
    const size_t off_tile_events = align16(off_tile_start + plan.tile_start.size() * sizeof(int));
    const size_t off_pops = align16(off_tile_events + plan.tile_events.size() * sizeof(int));
    const size_t off_band_rows = align16(off_pops + plan.pops.size() * sizeof(Pop));
    const size_t off_band_chunks = align16(off_band_rows + plan.band_rows.size() * sizeof(BandRow));
    const size_t total = align16(off_band_chunks + plan.band_chunks.size() * sizeof(BandChunk)) + 16;
    unsigned char* host = m.stage(total, stream);
    auto put = [&](size_t offset, const void* data, size_t bytes) {
        if (bytes) std::memcpy(host + offset, data, bytes);
    };
    put(off_rows, plan.rows.data(), n_rows * sizeof(RowData));
    put(off_events, plan.events.data(), plan.events.size() * sizeof(Event));
    put(off_tile_start, plan.tile_start.data(), plan.tile_start.size() * sizeof(int));
    put(off_tile_events, plan.tile_events.data(), plan.tile_events.size() * sizeof(int));
    put(off_pops, plan.pops.data(), plan.pops.size() * sizeof(Pop));
    put(off_band_rows, plan.band_rows.data(), plan.band_rows.size() * sizeof(BandRow));
    put(off_band_chunks, plan.band_chunks.data(), plan.band_chunks.size() * sizeof(BandChunk));
    m.commit(total, stream);

    // ---- noise ----
    const bool noise_on = plan.on;
    float* planes[kNoisePlanes] = {};
    if (noise_on) {
        // brightness: a soft mottle and the fine snow; colour: a mottle in Cb and a fifth of it in Cr
        m.field(S_LUMA, plan.seed, kLookLumaSigmaX, kLookLumaSigmaY, Impl::to(m.plane[kPlaneY], kLookLuma, false), stream);
        m.field(S_GRAIN, plan.seed, kFineSigmaX, kFineSigmaY, Impl::to(m.plane[kPlaneY], kLookFine, true), stream);
        m.field(S_CB, plan.seed, kLookChromaSigma, kLookChromaSigma, Impl::to(m.plane[kPlaneCb], kLookChroma, false), stream);
        m.field(S_CR, plan.seed, kLookChromaSigma, kLookChromaSigma,
                Impl::to(m.plane[kPlaneCr], kLookChroma * kCrShare, false), stream);
        planes[kPlaneY] = m.plane[kPlaneY];
        planes[kPlaneCb] = m.plane[kPlaneCb];
        planes[kPlaneCr] = m.plane[kPlaneCr];
    }
    if (plan.post_on) {
        // Post Noise: 70 % of every channel is shared, the rest is its own
        const float a = kPostCommon;
        const float b = sqrtf(fmaxf(1.0f - kPostCommon * kPostCommon, 0.0f));
        float* const* pp = m.plane + kPlanePost;
        m.field(S_POST ^ 0, plan.seed, kPostFieldSigma, kPostFieldSigma, Impl::to(pp[0], a, false, pp[1], a, pp[2], a), stream);
        // the channels' accumulation flags: the common field assigned, the three own fields add
        for (int k = 0; k < 3; ++k) {
            m.field(S_POST ^ (1 + k), plan.seed, kPostFieldSigma, kPostFieldSigma, Impl::to(pp[k], b, true), stream);
            planes[kPlanePost + k] = pp[k];
        }
    }
    if (plan.frame) {
        m.field(S_GLOW, plan.seed, kPostFieldSigma, kPostFieldSigma, Impl::to(m.plane[kPlaneGlow], 1.0f, false), stream);
        planes[kPlaneGlow] = m.plane[kPlaneGlow];
    }
    RowBatch batch{};
    bool any = false;
    for (int p = 0; p < kNoisePlanes; ++p) {
        if (planes[p]) {
            batch.src[p] = planes[p];
            batch.dst[p] = m.lines[p];
            any = true;
        }
    }
    const float inv = 1.0f / (fmaxf(m.scale, 1e-4f) * m.spacing);
    if (any) {
        const dim3 grid((m.stride + kFieldThreads - 1) / kFieldThreads, (m.h + kResampleRows - 1) / kResampleRows,
                        kNoisePlanes);
        resample_rows_kernel<<<grid, kFieldThreads, 0, stream>>>(batch, m.stride, m.rows_padded, m.h, inv, m.pad);
        check_launch();
    }

    // ---- what the final kernel reads ----
    d.w = m.w;
    d.h = m.h;
    d.scale = m.scale;
    d.seed = plan.seed;
    d.noise_on = noise_on ? 1 : 0;
    d.signal = (noise_on || !plan.events.empty() || !plan.pops.empty() || plan.frame) ? 1 : 0;
    d.output = (plan.frame || plan.post_on) ? 1 : 0;
    d.frame = plan.frame ? 1 : 0;
    d.post_on = plan.post_on ? 1 : 0;
    d.head_on = plan.frame ? 1 : 0;
    d.fall = noise_on ? 1 : 0;
    d.post = plan.post;
    for (int k = 0; k < 3; ++k) d.fleck_dc[k] = plan.fleck_dc[k];
    d.tiles_x = plan.tiles_x;
    d.tiles_y = plan.tiles_y;
    d.events = reinterpret_cast<const Event*>(m.d_blob + off_events);
    d.tile_start = reinterpret_cast<const int*>(m.d_blob + off_tile_start);
    d.tile_events = reinterpret_cast<const int*>(m.d_blob + off_tile_events);
    d.reach = plan.reach;
    d.n_pops = static_cast<int>(plan.pops.size());
    d.pops = reinterpret_cast<const Pop*>(m.d_blob + off_pops);
    for (int p = 0; p < kNoisePlanes; ++p) d.lines[p] = m.lines[p];
    d.stride = m.stride;
    d.pad = m.pad;
    d.inv = inv;
    d.rows = reinterpret_cast<const RowData*>(m.d_blob + off_rows);

    m.band_rows = static_cast<int>(plan.band_rows.size());
    m.band_y0 = plan.band_y0;
    m.d_band_rows = reinterpret_cast<const BandRow*>(m.d_blob + off_band_rows);
    m.d_band_chunks = reinterpret_cast<const BandChunk*>(m.d_blob + off_band_chunks);
    return d;
}

void DamageStage::warp_band(float* const plane[3], float* const spare[3], int width, cudaStream_t stream) {
    Impl& m = *impl_;
    if (m.band_rows <= 0 || width <= 0) return;
    BandBatch batch{};
    for (int k = 0; k < 3; ++k) {
        batch.in[k] = plane[k];
        batch.out[k] = spare[k];
    }
    const dim3 grid((width + kFieldThreads - 1) / kFieldThreads, m.band_rows, 3);
    band_warp_kernel<<<grid, kFieldThreads, 0, stream>>>(batch, m.d_band_rows, m.d_band_chunks, m.band_y0, m.band_rows, width);
    check_launch();
    for (int k = 0; k < 3; ++k) {
        const size_t offset = static_cast<size_t>(m.band_y0) * width;
        CUDA_CHECK(cudaMemcpyAsync(plane[k] + offset, spare[k] + offset, static_cast<size_t>(m.band_rows) * width * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

} // namespace arsenal
} // namespace effects
} // namespace vhs
