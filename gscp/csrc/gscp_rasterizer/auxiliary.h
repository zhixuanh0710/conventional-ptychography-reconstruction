/*
 * GSCP 2D Gaussian Splatting Rasterizer
 * Auxiliary device functions and macros for tile-based rasterization.
 */

#ifndef GSCP_RASTERIZER_AUXILIARY_H_INCLUDED
#define GSCP_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include <stdio.h>
#include <iostream>
#include <stdexcept>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)

// Compute tile bounding rectangle for a Gaussian centered at p with given radius.
// (Legacy circular AABB — kept for reference / fallback.)
__forceinline__ __device__ void getRect(
    const float2 p, int max_radius,
    uint2& rect_min, uint2& rect_max,
    dim3 grid)
{
    rect_min = {
        min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
    };
    rect_max = {
        min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
    };
}

// SnugBox: tight axis-aligned bounding box with separate x/y extents.
// Reduces tile overlap for anisotropic Gaussians compared to the circular AABB.
// Inspired by Speedy-Splat (Hanson et al., 2024).
__forceinline__ __device__ void getSnugRect(
    const float2 p, int x_ext, int y_ext,
    uint2& rect_min, uint2& rect_max,
    dim3 grid)
{
    rect_min = {
        min(grid.x, max((int)0, (int)((p.x - x_ext) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y - y_ext) / BLOCK_Y)))
    };
    rect_max = {
        min(grid.x, max((int)0, (int)((p.x + x_ext + BLOCK_X - 1) / BLOCK_X))),
        min(grid.y, max((int)0, (int)((p.y + y_ext + BLOCK_Y - 1) / BLOCK_Y)))
    };
}

// SnugBox from conic: compute tight AABB directly from the inverse covariance.
// conic = {Σ⁻¹[1,1], -Σ⁻¹[0,1], Σ⁻¹[0,0]} = {c/det, -b/det, a/det}
// x_ext = ceil(3 * sqrt(Σ[0,0])), y_ext = ceil(3 * sqrt(Σ[1,1]))
__forceinline__ __device__ void getSnugRectFromConic(
    const float2 p, const float3 con,
    uint2& rect_min, uint2& rect_max,
    dim3 grid)
{
    float det_con = con.x * con.z - con.y * con.y;
    int x_ext = (int)ceilf(sqrtf(9.0f * con.z / det_con));
    int y_ext = (int)ceilf(sqrtf(9.0f * con.x / det_con));
    getSnugRect(p, x_ext, y_ext, rect_min, rect_max, grid);
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
