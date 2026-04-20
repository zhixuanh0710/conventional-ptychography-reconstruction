/*
 * GSCP 2D Gaussian Splatting Rasterizer — Forward pass declarations.
 */

#ifndef GSCP_RASTERIZER_FORWARD_H_INCLUDED
#define GSCP_RASTERIZER_FORWARD_H_INCLUDED

#include "config.h"
#include <cuda_runtime.h>

namespace FORWARD
{
    void preprocess(
        int N,
        const float* xy,           // [N, 2] normalized positions
        const float* scaling,      // [N, 2] log-space scales
        const float* rotation,     // [N, 1] rotation angles
        int W, int H,
        int max_patch_radius,
        float min_scale,           // lower bound on activated sigma (pixels)
        int* radii,                // [N] output bounding radii
        float2* means2D,           // [N] output pixel-space means
        float3* conics,            // [N] output (conic_a, conic_b, conic_c)
        const dim3 tile_grid,
        uint32_t* tiles_touched    // [N] output tile overlap count
    );

    void preprocess_cholesky(
        int N,
        const float* xy,           // [N, 2] pixel-space positions
        const float* log_L_diag,   // [N, 2] log of Cholesky diagonal
        const float* L_offdiag,    // [N] lower-triangular off-diagonal
        int W, int H,
        int max_patch_radius,
        float min_scale,           // lower bound on activated L_diag (pixels)
        int* radii,                // [N] output bounding radii
        float2* means2D,           // [N] output pixel-space means
        float3* conics,            // [N] output (conic_a, conic_b, conic_c)
        const dim3 tile_grid,
        uint32_t* tiles_touched    // [N] output tile overlap count
    );

    void render(
        const dim3 grid, dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        const float2* means2D,
        const float3* conics,
        const float* weights,      // [N, 2] (w_real, w_imag) as flat float*
        float* out_color           // [2, H, W]
    );
}

#endif
