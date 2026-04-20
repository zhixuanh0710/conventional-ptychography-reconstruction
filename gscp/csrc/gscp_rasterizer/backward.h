/*
 * GSCP 2D Gaussian Splatting Rasterizer — Backward pass declarations.
 */

#ifndef GSCP_RASTERIZER_BACKWARD_H_INCLUDED
#define GSCP_RASTERIZER_BACKWARD_H_INCLUDED

#include "config.h"
#include <cuda_runtime.h>

namespace BACKWARD
{
    void render(
        const dim3 grid, const dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        const float2* means2D,
        const float3* conics,
        const float* weights,          // [N, 2]
        const float* dL_dpixels,       // [2, H, W]
        float2* dL_dmean2D,            // [N]
        float3* dL_dconic,             // [N]
        float* dL_dweight              // [N, 2]
    );

    void preprocess(
        int N,
        const float* xy,              // [N, 2]
        const float* scaling,         // [N, 2]
        const float* rotation,        // [N, 1]
        int W, int H,
        int max_patch_radius,
        float min_scale,              // lower bound on activated sigma (pixels)
        const int* radii,
        const float2* dL_dmean2D,     // [N]
        const float3* dL_dconic,      // [N]
        float* dL_dxy,                // [N, 2]
        float* dL_dscaling,           // [N, 2]
        float* dL_drotation           // [N, 1]
    );

    void preprocess_cholesky(
        int N,
        const float* xy,              // [N, 2]
        const float* log_L_diag,      // [N, 2]
        const float* L_offdiag,       // [N]
        int W, int H,
        int max_patch_radius,
        float min_scale,              // lower bound on activated L_diag (pixels)
        const int* radii,
        const float2* dL_dmean2D,     // [N]
        const float3* dL_dconic,      // [N]
        float* dL_dxy,                // [N, 2]
        float* dL_dlog_L_diag,        // [N, 2]
        float* dL_dL_offdiag          // [N]
    );
}

#endif
