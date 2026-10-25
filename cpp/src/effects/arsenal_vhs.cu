/**
 * Arsenal VHS - the colour stage.
 *
 * Ported from the Arsenal VHS effect of a FilmCraft build (MIT OR Apache-2.0).
 *
 * # Pipeline
 *
 * Display-encoded R'G'B' goes to Y'CbCr (ITU-R BT.601) and the three planes are filtered apart,
 * because VHS keeps colour in a far narrower band than brightness. Every filter is a separable
 * Gaussian made of three box passes of equal variance with fractional end weights, O(1) per pixel
 * (Wells, IEEE PAMI 1986; Gwosdek et al., SSVM 2011). The planes are then delayed (Chroma Offset
 * and the Channel Offsets), luma is sharpened with an unsharp mask, the colour is mixed back with the
 * clean colour and scaled, converted to R'G'B' and graded by Black Level.
 *
 * Lengths are pixels of the 2560-wide frame the look was calibrated on times width / 2560.
 *
 * # Control mapping (full-size px; "per unit" is per unit of the control)
 *
 *   Luminance Blur   Gaussian on Y', 0.8 x 1.3 px per unit (along x, across lines)
 *   Green Blur       adds to the luma blur, 0.0128 px per unit (in quadrature)
 *   Chroma Blur      Gaussian along the line on Cb and Cr, 0.3 px per unit
 *   Red / Blue Blur  round Gaussian on Cr / on Cb, 0.515 px per unit
 *   Chroma Offset X  delay of both colour planes to the right, 1.46 px per unit
 *   Red / Blue / Green Offset  extra delay of Cr / Cb / luma, 0.72 px per unit
 *   Color Mix        share of the filtered colour against the clean colour
 *   Saturation       gain on both colour planes, 100 is neutral
 *   Black Level      fades the picture toward the tape's tone curve (see setup_weight)
 *   Sharpen Radius / Mix   unsharp mask on Y' after the blur, 0.4 px per unit, 100 %
 *
 * Enable Color Controls switches Black Level, Saturation, the two blurs and Sharpen; Color Offset and
 * Channel Controls always act.
 */

#include "arsenal_vhs.hpp"
#include "arsenal_vhs_common.cuh"
#include "utils/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace vhs {
namespace effects {
namespace arsenal {

namespace {

// ---- constants of the calibrated look ----
constexpr float kLumaSigmaX = 0.8f;
constexpr float kLumaSigmaY = 1.3f;
constexpr float kGreenSigma = 0.0128f;
constexpr float kChromaSigmaX = 0.3f;
constexpr float kChannelSigma = 0.515f;
constexpr float kChromaOffset = 1.46f;
constexpr float kChannelOffset = 0.72f;
constexpr float kSharpenSigma = 0.4f;
/** The tape's tone curve, display-encoded: where black and white land at Black Level 10. */
constexpr float kTapeBlack[3] = {0.0555f, 0.0612f, 0.0683f};
constexpr float kTapeWhite[3] = {0.738f, 0.807f, 0.915f};
/** Black Level at which the tone curve applies in full, and the scale of its approach to twice that. */
constexpr float kSetupLevel = 10.0f;
constexpr float kSetupTau = 15.0f;
/** A blur narrower than this changes nothing a sampled image holds. */
constexpr float kMinSigma = 0.05f;

// ---- launch geometry ----
constexpr int kBoxThreads = 128;
constexpr int kBoxSegment = 128;  // outputs one thread produces before it restarts its running sum
constexpr int kTile = 32;
constexpr int kTileRows = 8;
constexpr int kMaxDimension = 16384;

/** One axis of a Gaussian: three box passes of half-width r and end weight alpha. */
struct BoxShape {
    int active;
    int r;
    float alpha;
    float norm;
};

struct BoxBatch {
    const float* src[3];
    float* dst[3];
    BoxShape shape[3];
};

struct TransposeBatch {
    const float* src[3];
    float* dst[3];
};

/** A plane and the buffer next to it: the data is in `cur`, `oth` is free. */
struct Ping {
    float* cur;
    float* oth;
};

/** The settings of one frame in working-image pixels. */
struct Look {
    int w, h;
    float luma_sigma_x, luma_sigma_y;
    float cb_sigma_x, cb_sigma_y;
    float cr_sigma_x, cr_sigma_y;
    float cb_dx, cr_dx, y_dx;
    float mix, saturation;
    float sharpen_sigma, sharpen_amount;
    float grade_gain[3], grade_lift[3];
};

/** A control as a finite number in lo..hi (NaN gives the default). */
float control(float v, float lo, float hi, float def) {
    return std::isnan(v) ? def : std::min(std::max(v, lo), hi);
}

/** Weight of the tape tone curve at `black_level`: 1 at kSetupLevel, 0 at 0, twice that at most. */
float setup_weight(float black_level) {
    const float top = 1.0f / (1.0f - std::exp(-kSetupLevel / kSetupTau));
    return top * (1.0f - std::exp(-black_level / kSetupTau));
}

Look make_look(const Controls& c, int w, int h) {
    const float scale = static_cast<float>(static_cast<double>(w) / kReferenceWidth);
    const float cap = 4.0f * static_cast<float>(std::max(w, h));
    auto sigma = [&](float per_unit, float v) { return std::min(per_unit * v * scale, cap); };
    const bool on = c.color_controls;
    auto gated = [&](float v, float neutral) { return on ? v : neutral; };

    Look l{};
    l.w = w;
    l.h = h;

    const float luma_blur = gated(control(c.luma_blur, 0.0f, 20.0f, 2.0f), 0.0f);
    const float green = sigma(kGreenSigma, control(c.green_blur, 0.0f, 300.0f, 55.0f));
    l.luma_sigma_x = std::hypot(sigma(kLumaSigmaX, luma_blur), green);
    l.luma_sigma_y = std::hypot(sigma(kLumaSigmaY, luma_blur), green);

    auto offset = [&](float v) { return control(v, -100.0f, 100.0f, 0.0f) * kChannelOffset * scale; };
    const float chroma_x = control(c.chroma_offset_x, -200.0f, 200.0f, 0.0f) * kChromaOffset * scale;
    l.cb_dx = chroma_x + offset(c.blue_offset);
    l.cr_dx = chroma_x + offset(c.red_offset);
    l.y_dx = offset(c.green_offset);

    const float black_level = gated(control(c.black_level, 0.0f, 100.0f, 10.0f), 0.0f);
    const float weight = setup_weight(black_level);
    for (int k = 0; k < 3; ++k) {
        l.grade_gain[k] = 1.0f - weight * (1.0f - (kTapeWhite[k] - kTapeBlack[k]));
        l.grade_lift[k] = weight * kTapeBlack[k];
    }

    const float chroma_sigma_x = sigma(kChromaSigmaX, gated(control(c.chroma_blur, 0.0f, 100.0f, 5.0f), 0.0f));
    const float cb_sigma = sigma(kChannelSigma, control(c.blue_blur, 0.0f, 300.0f, 45.0f));
    const float cr_sigma = sigma(kChannelSigma, control(c.red_blur, 0.0f, 300.0f, 25.0f));
    l.cb_sigma_x = std::hypot(chroma_sigma_x, cb_sigma);
    l.cb_sigma_y = cb_sigma;
    l.cr_sigma_x = std::hypot(chroma_sigma_x, cr_sigma);
    l.cr_sigma_y = cr_sigma;

    l.mix = control(c.color_mix, 0.0f, 100.0f, 70.0f) / 100.0f;
    l.saturation = gated(control(c.saturation, 0.0f, 200.0f, 100.0f), 100.0f) / 100.0f;
    l.sharpen_sigma = sigma(kSharpenSigma, gated(control(c.sharpen_radius, 0.0f, 100.0f, 5.0f), 0.0f));
    l.sharpen_amount = gated(control(c.sharpen_mix, 0.0f, 100.0f, 5.0f), 0.0f) / 100.0f;
    return l;
}

/**
 * Half-width r and end weight alpha of the box [alpha, 1 x (2r + 1), alpha] (normalised) whose variance
 * is `v`: r is the largest with r (r + 1) / 3 <= v, and alpha makes up the rest.
 */
BoxShape box_shape(float sigma) {
    BoxShape s{0, 0, 0.0f, 1.0f};
    if (!(sigma >= kMinSigma)) {
        return s;
    }
    const double v = static_cast<double>(sigma) * static_cast<double>(sigma) / 3.0;
    const double r = std::max(0.0, std::floor((std::sqrt(12.0 * v + 1.0) - 1.0) / 2.0));
    const double inner = r * (r + 1.0) * (2.0 * r + 1.0) / 3.0;
    const double alpha =
        std::min(1.0, std::max(0.0, ((2.0 * r + 1.0) * v - inner) / (2.0 * ((r + 1.0) * (r + 1.0) - v))));
    s.active = 1;
    s.r = static_cast<int>(r);
    s.alpha = static_cast<float>(alpha);
    s.norm = static_cast<float>(1.0 / ((2.0 * r + 1.0) + 2.0 * alpha));
    return s;
}

// =============================================================================================
// Kernels

/** Packed BGR8 to the three display-encoded Y'CbCr planes. */
__global__ void bgr_to_planes_kernel(const unsigned char* __restrict__ in, float* __restrict__ y_plane,
                                     float* __restrict__ cb_plane, float* __restrict__ cr_plane, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const size_t i = static_cast<size_t>(y) * w + x;
    const float b = in[i * 3 + 0] * (1.0f / 255.0f);
    const float g = in[i * 3 + 1] * (1.0f / 255.0f);
    const float r = in[i * 3 + 2] * (1.0f / 255.0f);
    float yy, cb, cr;
    rgb_to_ycc(r, g, b, yy, cb, cr);
    y_plane[i] = yy;
    cb_plane[i] = cb;
    cr_plane[i] = cr;
}

/**
 * One box pass along the slow axis of up to three planes. A thread owns one line (a pixel of the fast
 * axis) and a segment of it, and slides a running window down it; neighbouring threads read
 * neighbouring pixels, so the loads coalesce. The line is extended past both ends by repeating the end
 * pixel. `len` is the number of pixels along the blurred axis and `lines` the number of lines.
 */
__global__ void __launch_bounds__(kBoxThreads) box_pass_kernel(BoxBatch batch, int lines, int len, int segment) {
    const int z = blockIdx.z;
    const float* src = batch.src[z];
    if (!src) return;
    const BoxShape shape = batch.shape[z];
    const int line = blockIdx.x * kBoxThreads + threadIdx.x;
    const int first = blockIdx.y * segment;
    if (line >= lines || first >= len) return;
    const int last = min(first + segment, len);
    float* dst = batch.dst[z];

    const int r = shape.r;
    const float alpha = shape.alpha;
    const float norm = shape.norm;
    const int top = len - 1;
    auto at = [&](int l) -> float { return src[static_cast<size_t>(min(max(l, 0), top)) * lines + line]; };

    // window of the first output
    float sum = 0.0f;
    for (int k = -r; k <= r; ++k) sum += at(first + k);
    sum += alpha * (at(first - r - 1) + at(first + r + 1));

    float u = at(first + r + 1);  // enters the window after this output
    float p = at(first - r);      // leaves it
    float q = at(first - r - 1);  // left it before
    for (int l = first; l < last; ++l) {
        dst[static_cast<size_t>(l) * lines + line] = sum * norm;
        const float v = at(l + r + 2);
        sum += (u - p) + alpha * ((v + p) - (q + u));
        u = v;
        q = p;
        p = at(l + 1 - r);
    }
}

/** dst (cols lines of `rows`) = transpose of src (rows lines of `cols`), for up to three planes. */
__global__ void transpose3_kernel(TransposeBatch batch, int cols, int rows) {
    __shared__ float tile[kTile][kTile + 1];
    const float* src = batch.src[blockIdx.z];
    float* dst = batch.dst[blockIdx.z];
    if (!src) return;
    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y0 = blockIdx.y * kTile;
    for (int j = threadIdx.y; j < kTile; j += kTileRows) {
        const int y = y0 + j;
        if (x < cols && y < rows) tile[j][threadIdx.x] = src[static_cast<size_t>(y) * cols + x];
    }
    __syncthreads();
    const int ox = y0 + threadIdx.x;
    const int oy0 = blockIdx.x * kTile;
    for (int j = threadIdx.y; j < kTile; j += kTileRows) {
        const int oy = oy0 + j;
        if (ox < rows && oy < cols) dst[static_cast<size_t>(oy) * rows + ox] = tile[threadIdx.x][j];
    }
}

/** dst = src with every row moved `dx` pixels to the right (linear interpolation, ends repeated). */
__global__ void shift_rows_kernel(const float* __restrict__ src, float* __restrict__ dst, float dx, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float* row = src + static_cast<size_t>(y) * w;
    const float last = static_cast<float>(w - 1);
    const float p = clampf(static_cast<float>(x) - dx, 0.0f, last);
    const int i = static_cast<int>(p);
    const float t = p - static_cast<float>(i);
    const float a = row[i];
    const float b = row[min(i + 1, w - 1)];
    dst[static_cast<size_t>(y) * w + x] = a + (b - a) * t;
}

struct FinalParams {
    const unsigned char* in;
    unsigned char* out;
    const float* y;
    const float* cb;
    const float* cr;
    const float* soft;  // blurred luma for the unsharp mask, or null
    Look look;
};

/** Mixes the planes back, grades, and writes the packed BGR8 frame. */
__global__ void final_kernel(FinalParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const Look& look = p.look;
    if (x >= look.w || y >= look.h) return;
    const size_t i = static_cast<size_t>(y) * look.w + x;

    const float ib = p.in[i * 3 + 0] * (1.0f / 255.0f);
    const float ig = p.in[i * 3 + 1] * (1.0f / 255.0f);
    const float ir = p.in[i * 3 + 2] * (1.0f / 255.0f);
    float own_y, own_cb, own_cr;
    rgb_to_ycc(ir, ig, ib, own_y, own_cb, own_cr);

    float luma = p.y[i];
    if (p.soft) luma = luma + look.sharpen_amount * (luma - p.soft[i]);
    const float cb = (own_cb + (p.cb[i] - own_cb) * look.mix) * look.saturation;
    const float cr = (own_cr + (p.cr[i] - own_cr) * look.mix) * look.saturation;

    float rgb[3];
    ycc_to_rgb(luma, cb, cr, rgb[0], rgb[1], rgb[2]);
    for (int k = 0; k < 3; ++k) {
        float v = clampf(rgb[k], 0.0f, 1.0f);
        rgb[k] = clampf(v * look.grade_gain[k] + look.grade_lift[k], 0.0f, 1.0f);
    }
    p.out[i * 3 + 0] = static_cast<unsigned char>(rgb[2] * 255.0f + 0.5f);
    p.out[i * 3 + 1] = static_cast<unsigned char>(rgb[1] * 255.0f + 0.5f);
    p.out[i * 3 + 2] = static_cast<unsigned char>(rgb[0] * 255.0f + 0.5f);
}

void check_launch() {
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// =============================================================================================
// Engine

struct Engine::Impl {
    size_t pixels = 0;
    float* plane[3] = {nullptr, nullptr, nullptr};   // Y', Cb, Cr
    float* spare[3] = {nullptr, nullptr, nullptr};   // transposes and shifts
    float* soft = nullptr;                           // blurred luma for the unsharp mask
    bool ready = false;
};

FrameInfo frame_info(std::int64_t frame_number, double fps) {
    FrameInfo f;
    f.frame_number = frame_number;
    const double per_frame = (fps > 0.0 && std::isfinite(fps)) ? static_cast<double>(kTicksPerSecond) / fps : 0.0;
    f.tick = static_cast<std::uint64_t>(std::llround(static_cast<double>(frame_number) * per_frame));
    f.seconds = (fps > 0.0 && std::isfinite(fps)) ? static_cast<double>(frame_number) / fps : 0.0;
    return f;
}

Engine::Engine(int width, int height) : impl_(new Impl()), width_(width), height_(height) {}

Engine::~Engine() {
    cleanup();
}

int Engine::max_dimension() {
    return kMaxDimension;
}

void Engine::init() {
    if (impl_->ready || width_ <= 0 || height_ <= 0) {
        return;
    }
    if (width_ > kMaxDimension || height_ > kMaxDimension) {
        fprintf(stderr, "[ArsenalVhs] %dx%d is larger than the %d px the filters accept per side\n", width_, height_,
                kMaxDimension);
        throw std::runtime_error("Arsenal VHS: frame too large");
    }

    impl_->pixels = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    const size_t bytes = impl_->pixels * sizeof(float);
    try {
        for (int k = 0; k < 3; ++k) {
            CUDA_CHECK(cudaMalloc(&impl_->plane[k], bytes));
            CUDA_CHECK(cudaMalloc(&impl_->spare[k], bytes));
        }
        CUDA_CHECK(cudaMalloc(&impl_->soft, bytes));
    } catch (...) {
        cleanup();
        throw;
    }
    impl_->ready = true;
}

void Engine::cleanup() noexcept {
    if (!impl_) return;
    // Errors are ignored on purpose: this also runs after the device was reset.
    for (int k = 0; k < 3; ++k) {
        if (impl_->plane[k]) cudaFree(impl_->plane[k]);
        if (impl_->spare[k]) cudaFree(impl_->spare[k]);
        impl_->plane[k] = nullptr;
        impl_->spare[k] = nullptr;
    }
    if (impl_->soft) cudaFree(impl_->soft);
    impl_->soft = nullptr;
    impl_->ready = false;
}

namespace {

/** Three box passes along the slow axis of the planes whose shape is active; each pass swaps cur and oth. */
void box_axis(Ping p[3], const BoxShape shape[3], int lines, int len, cudaStream_t stream) {
    int widest = 0;
    bool any = false;
    for (int k = 0; k < 3; ++k) {
        if (p[k].cur && shape[k].active) {
            any = true;
            widest = std::max(widest, shape[k].r);
        }
    }
    if (!any) return;
    // A segment restarts the running sum, which costs one window: keep segments longer than that.
    const int segment = std::max(kBoxSegment, 2 * widest + 1);
    const dim3 grid((lines + kBoxThreads - 1) / kBoxThreads, (len + segment - 1) / segment, 3);
    for (int pass = 0; pass < 3; ++pass) {
        BoxBatch batch{};
        for (int k = 0; k < 3; ++k) {
            if (p[k].cur && shape[k].active) {
                batch.src[k] = p[k].cur;
                batch.dst[k] = p[k].oth;
                batch.shape[k] = shape[k];
            }
        }
        box_pass_kernel<<<grid, kBoxThreads, 0, stream>>>(batch, lines, len, segment);
        check_launch();
        for (int k = 0; k < 3; ++k) {
            if (p[k].cur && shape[k].active) std::swap(p[k].cur, p[k].oth);
        }
    }
}

/** Moves the planes whose shape is active from a `cols` x `rows` layout to a `rows` x `cols` one. */
void transpose_planes(Ping p[3], const BoxShape shape[3], int cols, int rows, cudaStream_t stream) {
    TransposeBatch batch{};
    bool any = false;
    for (int k = 0; k < 3; ++k) {
        if (p[k].cur && shape[k].active) {
            batch.src[k] = p[k].cur;
            batch.dst[k] = p[k].oth;
            any = true;
        }
    }
    if (!any) return;
    const dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile, 3);
    transpose3_kernel<<<grid, dim3(kTile, kTileRows), 0, stream>>>(batch, cols, rows);
    check_launch();
    for (int k = 0; k < 3; ++k) {
        if (p[k].cur && shape[k].active) std::swap(p[k].cur, p[k].oth);
    }
}

/**
 * Gaussian-blurs up to three w x h planes: down the columns with `sy`, then along the rows with `sx`
 * (the box passes are linear and separable, so the order only moves rounding). The results are in `cur`.
 */
void blur_planes(Ping p[3], const BoxShape sx[3], const BoxShape sy[3], int w, int h, cudaStream_t stream) {
    box_axis(p, sy, w, h, stream);  // w lines of h pixels, x is the fast axis
    // along the rows: lay the planes out with y as the fast axis, blur, and lay them back
    transpose_planes(p, sx, w, h, stream);
    box_axis(p, sx, h, w, stream);
    transpose_planes(p, sx, h, w, stream);
}

} // namespace

void Engine::render(const std::uint8_t* d_in, std::uint8_t* d_out, const Controls& controls, const FrameInfo& frame,
                    cudaStream_t stream) {
    (void)frame;  // the colour stage has nothing that varies in time
    if (width_ <= 0 || height_ <= 0 || !d_in || !d_out) {
        return;
    }
    if (!impl_->ready) {
        init();
    }
    const int w = width_;
    const int h = height_;
    const Look look = make_look(controls, w, h);
    const dim3 block(32, 8);
    const dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    // 1. planes of the picture
    bgr_to_planes_kernel<<<grid, block, 0, stream>>>(d_in, impl_->plane[0], impl_->plane[1], impl_->plane[2], w, h);
    check_launch();

    // 2. luma and colour in their own bands
    const BoxShape across[3] = {box_shape(look.luma_sigma_x), box_shape(look.cb_sigma_x), box_shape(look.cr_sigma_x)};
    const BoxShape down[3] = {box_shape(look.luma_sigma_y), box_shape(look.cb_sigma_y), box_shape(look.cr_sigma_y)};
    Ping planes[3] = {{impl_->plane[0], impl_->spare[0]}, {impl_->plane[1], impl_->spare[1]}, {impl_->plane[2], impl_->spare[2]}};
    blur_planes(planes, across, down, w, h, stream);
    for (int k = 0; k < 3; ++k) {
        impl_->plane[k] = planes[k].cur;
        impl_->spare[k] = planes[k].oth;
    }

    // 3. delays: a channel offset moves its plane, Chroma Offset moves both colour planes
    const float delays[3] = {look.y_dx, look.cb_dx, look.cr_dx};
    for (int k = 0; k < 3; ++k) {
        if (std::fabs(delays[k]) < 1e-4f) continue;
        shift_rows_kernel<<<grid, block, 0, stream>>>(impl_->plane[k], impl_->spare[k], delays[k], w, h);
        check_launch();
        std::swap(impl_->plane[k], impl_->spare[k]);
    }

    // 4. the detail the sharpening adds back is what the blur took out
    const float* soft = nullptr;
    if (look.sharpen_amount > 0.0f && look.sharpen_sigma >= kMinSigma) {
        CUDA_CHECK(cudaMemcpyAsync(impl_->soft, impl_->plane[0], impl_->pixels * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        Ping blurred[3] = {{impl_->soft, impl_->spare[0]}, {nullptr, nullptr}, {nullptr, nullptr}};
        const BoxShape round_blur[3] = {box_shape(look.sharpen_sigma), BoxShape{}, BoxShape{}};
        blur_planes(blurred, round_blur, round_blur, w, h, stream);
        impl_->soft = blurred[0].cur;
        impl_->spare[0] = blurred[0].oth;
        soft = impl_->soft;
    }

    // 5. mix the colour back, grade, write
    FinalParams params{};
    params.in = d_in;
    params.out = d_out;
    params.y = impl_->plane[0];
    params.cb = impl_->plane[1];
    params.cr = impl_->plane[2];
    params.soft = soft;
    params.look = look;
    final_kernel<<<grid, block, 0, stream>>>(params);
    check_launch();
}

} // namespace arsenal
} // namespace effects
} // namespace vhs
