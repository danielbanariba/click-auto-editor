/**
 * Arsenal VHS - the damage stage, host side: plans a frame and draws its noise.
 *
 * Ported from the Arsenal VHS effect of a FilmCraft build (MIT OR Apache-2.0).
 *
 * Everything that varies from frame to frame is decided here from a hash of the clip, the Random Seed
 * and the frame's timestamp (which dust specks fall where, where the Wrinkle band is, how far the left
 * edge of the tape frame is on every row), uploaded to the device, and read by the final kernel
 * (arsenal_vhs_damage.cuh). The noise is a Gaussian random field on a lattice, drawn on the device.
 */

#pragma once

#include "arsenal_vhs.hpp"
#include "arsenal_vhs_damage.cuh"

#include <cuda_runtime.h>

#include <memory>

namespace vhs {
namespace effects {
namespace arsenal {

class DamageStage {
public:
    DamageStage(int width, int height);
    ~DamageStage();
    DamageStage(const DamageStage&) = delete;
    DamageStage& operator=(const DamageStage&) = delete;

    /** Allocates the device memory (throws std::runtime_error when a CUDA call fails). */
    void init();
    void cleanup() noexcept;

    /**
     * Plans the frame, uploads what it needs and draws its noise on `stream`. The result points into
     * device memory that stays valid until the next call; frames must be drawn in order on one stream.
     */
    DamageParams prepare(const Controls& controls, const FrameInfo& frame, cudaStream_t stream);

    /** Whether the frame has a Wrinkle band, and the rows to move (null when no row moves). */
    bool has_band() const;

    /**
     * Redraws the rows of the Wrinkle band in the three planes (Y', Cb, Cr), each read from `plane[k]`
     * and written through `spare[k]`; the result is in `plane[k]`.
     */
    void warp_band(float* const plane[3], float* const spare[3], int width, cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace arsenal
} // namespace effects
} // namespace vhs
