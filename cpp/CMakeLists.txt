cmake_minimum_required(VERSION 3.18)
project(vhs_renderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find FFmpeg using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswscale
)

# NPP libraries from CUDA Toolkit
set(NPP_LIBS
    CUDA::nppc
    CUDA::nppif
    CUDA::nppig
    CUDA::npps
)

# Source files
set(CUDA_SOURCES
    src/effects/color_bleeding.cu
    src/effects/chromatic_aberration.cu
    src/effects/horizontal_wobble.cu
    src/effects/tracking_errors.cu
    src/effects/position_jitter.cu
    src/effects/scanlines.cu
    src/effects/noise_grain.cu
    src/effects/color_grading.cu
    src/effects/vhs_overlay_blend.cu
    src/effects/cover_overlay.cu
    src/effects/vhs_effect_chain.cu
    src/utils/cuda_utils.cu
    src/utils/color_conversion.cu
    src/utils/yuv_conversion.cu
)

set(CPP_SOURCES
    src/main.cpp
    src/pipeline/video_pipeline.cpp
    src/pipeline/ffmpeg_decoder.cpp
    src/pipeline/ffmpeg_encoder.cpp
    src/pipeline/cuda_frame_buffer.cpp
    src/pipeline/cover_pipeline.cpp
    src/utils/npp_wrappers.cpp
    src/utils/image_loader.cpp
    src/config/render_config.cpp
)

# Main executable
add_executable(vhs_render ${CPP_SOURCES} ${CUDA_SOURCES})

target_include_directories(vhs_render PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(vhs_render PRIVATE
    PkgConfig::LIBAV
    ${NPP_LIBS}
    CUDA::cudart
    CUDA::curand
)

# CUDA architecture for RTX 3090 Ti (Ampere SM 8.6)
set_target_properties(vhs_render PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
)

# Compiler flags
target_compile_options(vhs_render PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -fPIC>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fPIC>
)

# Installation
install(TARGETS vhs_render DESTINATION bin)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: 86 (RTX 3090 Ti)")
