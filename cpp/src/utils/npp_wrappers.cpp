/**
 * NPP Wrappers Implementation
 *
 * Note: NPP API varies between CUDA versions. This uses the standard
 * mask-size based Gaussian filter available in CUDA 12+.
 */

#include "npp_wrappers.hpp"
#include "cuda_utils.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>

namespace vhs {
namespace npp {

// Thread-local NPP stream context
static NppStreamContext npp_ctx;
static bool npp_ctx_initialized = false;

void init_npp_stream(cudaStream_t stream) {
    // Always refresh context fields in case device/stream changed.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    if (!npp_ctx_initialized) {
        std::memset(&npp_ctx, 0, sizeof(npp_ctx));
        npp_ctx_initialized = true;
    }

    npp_ctx.hStream = stream;
    npp_ctx.nCudaDeviceId = device;
    npp_ctx.nMultiProcessorCount = props.multiProcessorCount;
    npp_ctx.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    npp_ctx.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
    npp_ctx.nSharedMemPerBlock = props.sharedMemPerBlock;

    int cc_major = 0;
    int cc_minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device));
    npp_ctx.nCudaDevAttrComputeCapabilityMajor = cc_major;
    npp_ctx.nCudaDevAttrComputeCapabilityMinor = cc_minor;

    unsigned int stream_flags = 0;
    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamGetFlags(stream, &stream_flags));
    }
    npp_ctx.nStreamFlags = stream_flags;
}

// Helper to convert kernel size to NppiMaskSize
static NppiMaskSize kernel_to_mask_size(int kernel_size) {
    switch (kernel_size) {
        case 3:  return NPP_MASK_SIZE_3_X_3;
        case 5:  return NPP_MASK_SIZE_5_X_5;
        case 7:  return NPP_MASK_SIZE_7_X_7;
        case 9:  return NPP_MASK_SIZE_9_X_9;
        case 11: return NPP_MASK_SIZE_11_X_11;
        case 13: return NPP_MASK_SIZE_13_X_13;
        case 15: return NPP_MASK_SIZE_15_X_15;
        default: return NPP_MASK_SIZE_5_X_5;
    }
}

bool gaussian_blur_32f_c1(
    const float* d_src,
    float* d_dst,
    int width,
    int height,
    int src_step,
    int dst_step,
    int kernel_size,
    cudaStream_t stream
) {
    init_npp_stream(stream);

    // Ensure odd kernel size
    if (kernel_size % 2 == 0) kernel_size++;
    if (kernel_size < 3) kernel_size = 3;
    if (kernel_size > 15) kernel_size = 15;

    NppiSize roi = { width, height };
    NppiMaskSize mask = kernel_to_mask_size(kernel_size);

    // Apply Gaussian filter using mask-size variant
    NppStatus status = nppiFilterGauss_32f_C1R_Ctx(
        d_src, src_step,
        d_dst, dst_step,
        roi, mask,
        npp_ctx
    );

    if (status != NPP_SUCCESS) {
        fprintf(stderr, "[NPP] Gaussian blur 32f_C1 failed: %d\n", status);
        return false;
    }

    return true;
}

bool gaussian_blur_8u_c3(
    const unsigned char* d_src,
    unsigned char* d_dst,
    int width,
    int height,
    int src_step,
    int dst_step,
    int kernel_size,
    cudaStream_t stream
) {
    init_npp_stream(stream);

    // Ensure odd kernel size
    if (kernel_size % 2 == 0) kernel_size++;
    if (kernel_size < 3) kernel_size = 3;
    if (kernel_size > 15) kernel_size = 15;

    NppiSize roi = { width, height };
    NppiMaskSize mask;

    // Map kernel size to NPP mask size
    switch (kernel_size) {
        case 3:  mask = NPP_MASK_SIZE_3_X_3; break;
        case 5:  mask = NPP_MASK_SIZE_5_X_5; break;
        case 7:  mask = NPP_MASK_SIZE_7_X_7; break;
        case 9:  mask = NPP_MASK_SIZE_9_X_9; break;
        case 11: mask = NPP_MASK_SIZE_11_X_11; break;
        case 13: mask = NPP_MASK_SIZE_13_X_13; break;
        case 15: mask = NPP_MASK_SIZE_15_X_15; break;
        default: mask = NPP_MASK_SIZE_5_X_5; break;
    }

    // Apply Gaussian filter
    NppStatus status = nppiFilterGauss_8u_C3R_Ctx(
        d_src, src_step,
        d_dst, dst_step,
        roi, mask,
        npp_ctx
    );

    if (status != NPP_SUCCESS) {
        fprintf(stderr, "[NPP] Gaussian blur 8u_C3 failed: %d\n", status);
        return false;
    }

    return true;
}

bool box_filter_8u_c3(
    const unsigned char* d_src,
    unsigned char* d_dst,
    int width,
    int height,
    int src_step,
    int dst_step,
    int kernel_size,
    cudaStream_t stream
) {
    init_npp_stream(stream);

    // Ensure odd kernel size
    if (kernel_size % 2 == 0) kernel_size++;
    if (kernel_size < 3) kernel_size = 3;

    NppiSize roi = { width, height };
    NppiSize mask = { kernel_size, kernel_size };
    NppiPoint anchor = { kernel_size / 2, kernel_size / 2 };

    // Apply box filter (simpler non-border version)
    NppStatus status = nppiFilterBox_8u_C3R_Ctx(
        d_src, src_step,
        d_dst, dst_step,
        roi, mask, anchor,
        npp_ctx
    );

    if (status != NPP_SUCCESS) {
        fprintf(stderr, "[NPP] Box filter failed: %d\n", status);
        return false;
    }

    return true;
}

bool resize_8u_c3(
    const unsigned char* d_src,
    int src_width,
    int src_height,
    int src_step,
    unsigned char* d_dst,
    int dst_width,
    int dst_height,
    int dst_step,
    int interpolation,
    cudaStream_t stream
) {
    init_npp_stream(stream);

    NppiSize src_size = { src_width, src_height };
    NppiRect src_roi = { 0, 0, src_width, src_height };
    NppiSize dst_size = { dst_width, dst_height };
    NppiRect dst_roi = { 0, 0, dst_width, dst_height };

    NppStatus status = nppiResize_8u_C3R_Ctx(
        d_src, src_step, src_size, src_roi,
        d_dst, dst_step, dst_size, dst_roi,
        interpolation,
        npp_ctx
    );

    if (status != NPP_SUCCESS) {
        fprintf(stderr, "[NPP] Resize failed: %d\n", status);
        return false;
    }

    return true;
}

size_t get_gaussian_buffer_size(int width, int height, int kernel_size) {
    // Standard Gaussian filter doesn't need external buffer with mask-size API
    // Return estimated buffer size based on kernel size
    (void)width;
    (void)height;
    (void)kernel_size;
    return 0;  // No external buffer needed for nppiFilterGauss with mask size
}

} // namespace npp
} // namespace vhs
