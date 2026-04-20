/*
 * GSCP 2D Gaussian Splatting Rasterizer — Backward pass CUDA kernels.
 *
 * Implements:
 *   1. renderCUDA_backward: per-pixel gradient accumulation to per-Gaussian gradients
 *   2. preprocessCUDA_backward: conic gradients -> parameter gradients
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// -----------------------------------------------------------------------
// Backward render: re-iterate Gaussians, accumulate gradients.
// Uses warp-level reduction to minimize global atomicAdd contention:
// 256 threads -> 8 warp sums -> 8 atomicAdds (32x fewer than naive).
// -----------------------------------------------------------------------
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_backward(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ means2D,
    const float3* __restrict__ conics,
    const float* __restrict__ weights,       // [N, 2]
    const float* __restrict__ dL_dpixels,    // [2, H, W]
    float2* __restrict__ dL_dmean2D,         // [N]
    float3* __restrict__ dL_dconic,          // [N]
    float* __restrict__ dL_dweight)          // [N, 2]
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = { (float)pix.x, (float)pix.y };

    const bool inside = pix.x < W && pix.y < H;
    const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float3 collected_conic[BLOCK_SIZE];
    __shared__ float2 collected_weight[BLOCK_SIZE];

    // Load upstream gradients for this pixel
    float dL_dpixel[CHANNELS] = { 0 };
    if (inside)
        for (int i = 0; i < CHANNELS; i++)
            dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

    // Forward iteration (order-independent for summation)
    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        block.sync();
        const int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            const int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = means2D[coll_id];
            collected_conic[block.thread_rank()] = conics[coll_id];
            collected_weight[block.thread_rank()] = *((float2*)(weights + 2 * coll_id));
        }
        block.sync();

        for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
        {
            const float2 xy = collected_xy[j];
            const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            const float3 con = collected_conic[j];

            // Per-pixel gradient contributions (zero if outside or below threshold)
            float gw0 = 0, gw1 = 0, gmx = 0, gmy = 0, gc0 = 0, gc1 = 0, gc2 = 0;

            if (inside)
            {
                const float power = -0.5f * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;
                if (power <= 0.0f)
                {
                    const float G = expf(power);
                    if (G >= 1e-7f)
                    {
                        const float2 w = collected_weight[j];
                        const float dL_dG = w.x * dL_dpixel[0] + w.y * dL_dpixel[1];

                        gw0 = G * dL_dpixel[0];
                        gw1 = G * dL_dpixel[1];

                        const float dG_ddx = G * (-con.x * d.x - con.y * d.y);
                        const float dG_ddy = G * (-con.z * d.y - con.y * d.x);
                        gmx = dL_dG * dG_ddx;
                        gmy = dL_dG * dG_ddy;

                        const float gdx = G * d.x;
                        const float gdy = G * d.y;
                        gc0 = -0.5f * gdx * d.x * dL_dG;
                        gc1 = -gdx * d.y * dL_dG;
                        gc2 = -0.5f * gdy * d.y * dL_dG;
                    }
                }
            }

            // Warp-level reduction: 32 threads -> 1 partial sum
            gw0 = cg::reduce(warp, gw0, cg::plus<float>());
            gw1 = cg::reduce(warp, gw1, cg::plus<float>());
            gmx = cg::reduce(warp, gmx, cg::plus<float>());
            gmy = cg::reduce(warp, gmy, cg::plus<float>());
            gc0 = cg::reduce(warp, gc0, cg::plus<float>());
            gc1 = cg::reduce(warp, gc1, cg::plus<float>());
            gc2 = cg::reduce(warp, gc2, cg::plus<float>());

            // Lane 0 of each warp writes to global memory (8 atomicAdds instead of 256)
            if (warp.thread_rank() == 0)
            {
                const int global_id = collected_id[j];
                atomicAdd(&dL_dweight[2 * global_id + 0], gw0);
                atomicAdd(&dL_dweight[2 * global_id + 1], gw1);
                atomicAdd(&dL_dmean2D[global_id].x, gmx);
                atomicAdd(&dL_dmean2D[global_id].y, gmy);
                atomicAdd(&dL_dconic[global_id].x, gc0);
                atomicAdd(&dL_dconic[global_id].y, gc1);
                atomicAdd(&dL_dconic[global_id].z, gc2);
            }
        }
    }
}

// -----------------------------------------------------------------------
// Backward preprocess: conic grads -> parameter grads, one thread per Gaussian
// -----------------------------------------------------------------------
__global__ void preprocessCUDA_backward(
    int N,
    const float* __restrict__ xy,          // [N, 2]
    const float* __restrict__ scaling,     // [N, 2]
    const float* __restrict__ rotation,    // [N, 1]
    int W, int H,
    int max_patch_radius,
    float min_scale,
    const int* __restrict__ radii,
    const float2* __restrict__ dL_dmean2D, // [N]
    const float3* __restrict__ dL_dconic,  // [N]
    float* __restrict__ dL_dxy,            // [N, 2]
    float* __restrict__ dL_dscaling,       // [N, 2]
    float* __restrict__ dL_drotation)      // [N, 1]
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N || radii[idx] <= 0)
        return;

    // Recompute forward values with same scale clamping as forward pass.
    // min_scale is a runtime parameter passed from Python (must match self.min_scale).
    const float MAX_SCALE = (float)max_patch_radius;
    float theta = rotation[idx];
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    float sx_raw = expf(scaling[2 * idx + 0]);
    float sy_raw = expf(scaling[2 * idx + 1]);
    float sx = fminf(MAX_SCALE, fmaxf(min_scale, sx_raw));
    float sy = fminf(MAX_SCALE, fmaxf(min_scale, sy_raw));
    float sx2 = sx * sx;
    float sy2 = sy * sy;

    // Covariance elements
    float a = cos_t * cos_t * sx2 + sin_t * sin_t * sy2;
    float b = cos_t * sin_t * (sx2 - sy2);
    float c = sin_t * sin_t * sx2 + cos_t * cos_t * sy2;
    float det = a * c - b * b;
    float det2_inv = 1.0f / (det * det + 1e-7f);

    // conic = (c/det, -b/det, a/det) -> dL_d(a,b,c) from dL_d(conic)
    float3 dL_dcon = dL_dconic[idx];
    float dL_da = det2_inv * (-c * c * dL_dcon.x + b * c * dL_dcon.y + (det - a * c) * dL_dcon.z);
    float dL_db = det2_inv * (2.0f * b * c * dL_dcon.x - (det + 2.0f * b * b) * dL_dcon.y + 2.0f * a * b * dL_dcon.z);
    float dL_dc = det2_inv * ((det - a * c) * dL_dcon.x + a * b * dL_dcon.y - a * a * dL_dcon.z);

    // d(a,b,c)/d(sx2, sy2, theta)
    float dL_dsx2 = cos_t * cos_t * dL_da + cos_t * sin_t * dL_db + sin_t * sin_t * dL_dc;
    float dL_dsy2 = sin_t * sin_t * dL_da - cos_t * sin_t * dL_db + cos_t * cos_t * dL_dc;
    float dL_dtheta = 2.0f * cos_t * sin_t * (sy2 - sx2) * dL_da
                    + (cos_t * cos_t - sin_t * sin_t) * (sx2 - sy2) * dL_db
                    + 2.0f * sin_t * cos_t * (sx2 - sy2) * dL_dc;

    // Chain through exp + clamp: zero gradient when scale is at clamp boundary
    float in_range_x = (sx_raw >= min_scale && sx_raw <= MAX_SCALE) ? 1.0f : 0.0f;
    float in_range_y = (sy_raw >= min_scale && sy_raw <= MAX_SCALE) ? 1.0f : 0.0f;
    dL_dscaling[2 * idx + 0] = dL_dsx2 * 2.0f * sx2 * in_range_x;
    dL_dscaling[2 * idx + 1] = dL_dsy2 * 2.0f * sy2 * in_range_y;
    dL_drotation[idx] = dL_dtheta;

    // Mean gradient: positions are already in pixel space, pass through directly
    float2 dL_dm = dL_dmean2D[idx];
    dL_dxy[2 * idx + 0] = dL_dm.x;
    dL_dxy[2 * idx + 1] = dL_dm.y;
}

// -----------------------------------------------------------------------
// Backward preprocess (Cholesky): conic grads -> Cholesky param grads
// -----------------------------------------------------------------------
__global__ void preprocessCUDA_backward_cholesky(
    int N,
    const float* __restrict__ xy,              // [N, 2]
    const float* __restrict__ log_L_diag,      // [N, 2]
    const float* __restrict__ L_offdiag,       // [N]
    int W, int H,
    int max_patch_radius,
    float min_scale,
    const int* __restrict__ radii,
    const float2* __restrict__ dL_dmean2D,     // [N]
    const float3* __restrict__ dL_dconic,      // [N]
    float* __restrict__ dL_dxy,                // [N, 2]
    float* __restrict__ dL_dlog_L_diag,        // [N, 2]
    float* __restrict__ dL_dL_offdiag)         // [N]
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N || radii[idx] <= 0)
        return;

    // Recompute forward values with same clamping as forward pass
    const float MAX_SCALE = (float)max_patch_radius;
    float l11_raw = expf(log_L_diag[2 * idx + 0]);
    float l22_raw = expf(log_L_diag[2 * idx + 1]);
    float l11 = fminf(MAX_SCALE, fmaxf(min_scale, l11_raw));
    float l22 = fminf(MAX_SCALE, fmaxf(min_scale, l22_raw));
    float l21 = L_offdiag[idx];

    // Covariance elements: Sigma = L @ L^T
    float a = l11 * l11;
    float b = l11 * l21;
    float c = l21 * l21 + l22 * l22;
    float det = a * l22 * l22;  // l11^2 * l22^2
    float det2_inv = 1.0f / (det * det + 1e-7f);

    // Step 1: conic = (c/det, -b/det, a/det) -> dL_d(a,b,c)
    // IDENTICAL Jacobian to RS backward
    float3 dL_dcon = dL_dconic[idx];
    float dL_da = det2_inv * (-c * c * dL_dcon.x + b * c * dL_dcon.y + (det - a * c) * dL_dcon.z);
    float dL_db = det2_inv * (2.0f * b * c * dL_dcon.x - (det + 2.0f * b * b) * dL_dcon.y + 2.0f * a * b * dL_dcon.z);
    float dL_dc = det2_inv * ((det - a * c) * dL_dcon.x + a * b * dL_dcon.y - a * a * dL_dcon.z);

    // Step 2: covariance -> Cholesky factor grads
    // a = l11^2          -> dL_dl11 += 2*l11*dL_da
    // b = l11*l21        -> dL_dl11 += l21*dL_db, dL_dl21 += l11*dL_db
    // c = l21^2 + l22^2  -> dL_dl21 += 2*l21*dL_dc, dL_dl22 += 2*l22*dL_dc
    float dL_dl11 = 2.0f * l11 * dL_da + l21 * dL_db;
    float dL_dl21 = l11 * dL_db + 2.0f * l21 * dL_dc;
    float dL_dl22 = 2.0f * l22 * dL_dc;

    // Step 3: chain through exp + floor clamp
    float in_range_11 = (l11_raw >= min_scale && l11_raw <= MAX_SCALE) ? 1.0f : 0.0f;
    float in_range_22 = (l22_raw >= min_scale && l22_raw <= MAX_SCALE) ? 1.0f : 0.0f;
    dL_dlog_L_diag[2 * idx + 0] = dL_dl11 * l11 * in_range_11;
    dL_dlog_L_diag[2 * idx + 1] = dL_dl22 * l22 * in_range_22;
    dL_dL_offdiag[idx] = dL_dl21;  // unconstrained, no chain rule factor

    // Step 4: position grads (pixel-space, pass through directly)
    float2 dL_dm = dL_dmean2D[idx];
    dL_dxy[2 * idx + 0] = dL_dm.x;
    dL_dxy[2 * idx + 1] = dL_dm.y;
}

// -----------------------------------------------------------------------
// C++ wrappers
// -----------------------------------------------------------------------
void BACKWARD::render(
    const dim3 grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float3* conics,
    const float* weights,
    const float* dL_dpixels,
    float2* dL_dmean2D,
    float3* dL_dconic,
    float* dL_dweight)
{
    renderCUDA_backward<NUM_CHANNELS><<<grid, block>>>(
        ranges, point_list,
        W, H,
        means2D, conics, weights,
        dL_dpixels,
        dL_dmean2D, dL_dconic, dL_dweight);
}

void BACKWARD::preprocess(
    int N,
    const float* xy,
    const float* scaling,
    const float* rotation,
    int W, int H,
    int max_patch_radius,
    float min_scale,
    const int* radii,
    const float2* dL_dmean2D,
    const float3* dL_dconic,
    float* dL_dxy,
    float* dL_dscaling,
    float* dL_drotation)
{
    preprocessCUDA_backward<<<(N + 255) / 256, 256>>>(
        N, xy, scaling, rotation,
        W, H, max_patch_radius, min_scale, radii,
        dL_dmean2D, dL_dconic,
        dL_dxy, dL_dscaling, dL_drotation);
}

void BACKWARD::preprocess_cholesky(
    int N,
    const float* xy,
    const float* log_L_diag,
    const float* L_offdiag,
    int W, int H,
    int max_patch_radius,
    float min_scale,
    const int* radii,
    const float2* dL_dmean2D,
    const float3* dL_dconic,
    float* dL_dxy,
    float* dL_dlog_L_diag,
    float* dL_dL_offdiag)
{
    preprocessCUDA_backward_cholesky<<<(N + 255) / 256, 256>>>(
        N, xy, log_L_diag, L_offdiag,
        W, H, max_patch_radius, min_scale, radii,
        dL_dmean2D, dL_dconic,
        dL_dxy, dL_dlog_L_diag, dL_dL_offdiag);
}
