/*
 * GSCP 2D Gaussian Splatting Rasterizer
 * Configuration constants for tile-based rasterization.
 */

#ifndef GSCP_RASTERIZER_CONFIG_H_INCLUDED
#define GSCP_RASTERIZER_CONFIG_H_INCLUDED

#include <cstdint>

#define NUM_CHANNELS 2   // Real and imaginary channels
#define BLOCK_X 16
#define BLOCK_Y 16

// SnugBox optimization: use tight axis-aligned bounding boxes instead of
// circular AABB. Set to 0 to revert to the legacy circular AABB for benchmarking.
#ifndef USE_SNUGBOX
#define USE_SNUGBOX 0
#endif

#endif
