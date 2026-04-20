/*
 * GSCP 2D Gaussian Splatting Rasterizer — Forward pass CUDA kernels.
 *
 * Implements:
 *   1. preprocessCUDA: per-Gaussian preprocessing (covariance -> conic, bounding box)
 *   2. renderCUDA: tile-based 2-channel rendering with shared memory batching
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// -----------------------------------------------------------------------
// Preprocess: one thread per Gaussian
// -----------------------------------------------------------------------
__global__ void preprocessCUDA(
    int N,
    const float* __restrict__ xy,         // [N, 2]
    const float* __restrict__ scaling,    // [N, 2]
    const float* __restrict__ rotation,   // [N, 1]
    int W, int H,
    int max_patch_radius,
    float min_scale,
    int* radii,
    float2* means2D,
    float3* conics,
    const dim3 tile_grid,
    uint32_t* tiles_touched)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N)
        return;

    // Initialize to zero — if unchanged, this Gaussian won't be processed
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // Read pixel-space position and clamp to canvas bounds
    float mean_x = min((float)(W - 1), max(0.0f, xy[2 * idx + 0]));
    float mean_y = min((float)(H - 1), max(0.0f, xy[2 * idx + 1]));

    // Activated scales (exp of log-space), clamped to [min_scale, max_patch_radius]
    // to prevent sub-pixel noise fitting and giant Gaussians.
    // min_scale is a runtime parameter passed from Python (must match self.min_scale).
    const float MAX_SCALE = (float)max_patch_radius;
    float sx = fminf(MAX_SCALE, fmaxf(min_scale, expf(scaling[2 * idx + 0])));
    float sy = fminf(MAX_SCALE, fmaxf(min_scale, expf(scaling[2 * idx + 1])));
    float sx2 = sx * sx;
    float sy2 = sy * sy;

    // Rotation
    float theta = rotation[idx];
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    // 2D covariance matrix: Sigma = R @ diag(sx2, sy2) @ R^T = [[a, b], [b, c]]
    float a = cos_t * cos_t * sx2 + sin_t * sin_t * sy2;
    float b = cos_t * sin_t * (sx2 - sy2);
    float c = sin_t * sin_t * sx2 + cos_t * cos_t * sy2;

    // Determinant and inverse (conic form)
    float det = a * c - b * b;
    if (det <= 0.0f)
        return;
    float det_inv = 1.0f / det;
    float3 conic = { c * det_inv, -b * det_inv, a * det_inv };

#if USE_SNUGBOX
    // SnugBox: axis-aligned bounding box from marginal standard deviations.
    // x_ext = ceil(3 * σ_x), y_ext = ceil(3 * σ_y)
    // Tighter than circular AABB for anisotropic Gaussians.
    int x_ext = (int)ceilf(3.0f * sqrtf(a));
    int y_ext = (int)ceilf(3.0f * sqrtf(c));
    int my_radius = max(x_ext, y_ext);
#else
    // Legacy circular AABB from max eigenvalue at 3σ.
    float mid = 0.5f * (a + c);
    float lambda_max = mid + sqrtf(max(0.1f, mid * mid - det));
    int my_radius = (int)ceilf(3.0f * sqrtf(lambda_max));
#endif
    if (my_radius <= 0)
        return;

    float2 point = { mean_x, mean_y };
    means2D[idx] = point;
    conics[idx] = conic;
    radii[idx] = my_radius;

    uint2 rect_min, rect_max;
#if USE_SNUGBOX
    getSnugRect(point, x_ext, y_ext, rect_min, rect_max, tile_grid);
#else
    getRect(point, my_radius, rect_min, rect_max, tile_grid);
#endif
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// -----------------------------------------------------------------------
// Preprocess (Cholesky): one thread per Gaussian
// -----------------------------------------------------------------------
__global__ void preprocessCUDA_cholesky(
    int N,
    const float* __restrict__ xy,            // [N, 2]
    const float* __restrict__ log_L_diag,    // [N, 2]
    const float* __restrict__ L_offdiag,     // [N]
    int W, int H,
    int max_patch_radius,
    float min_scale,
    int* radii,
    float2* means2D,
    float3* conics,
    const dim3 tile_grid,
    uint32_t* tiles_touched)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N)
        return;

    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // Pixel-space position clamped to canvas
    float mean_x = min((float)(W - 1), max(0.0f, xy[2 * idx + 0]));
    float mean_y = min((float)(H - 1), max(0.0f, xy[2 * idx + 1]));

    // Cholesky diagonal: exp + clamp to [min_scale, max_patch_radius]
    const float MAX_SCALE = (float)max_patch_radius;
    float l11 = fminf(MAX_SCALE, fmaxf(min_scale, expf(log_L_diag[2 * idx + 0])));
    float l22 = fminf(MAX_SCALE, fmaxf(min_scale, expf(log_L_diag[2 * idx + 1])));
    float l21 = L_offdiag[idx];

    // Covariance: Sigma = L @ L^T = [[a, b], [b, c]]
    float a = l11 * l11;
    float b = l11 * l21;
    float c = l21 * l21 + l22 * l22;

    // Determinant (guaranteed positive: l11, l22 > 0)
    float det = a * l22 * l22;  // = l11^2 * l22^2
    if (det <= 0.0f)
        return;
    float det_inv = 1.0f / det;
    float3 conic = { c * det_inv, -b * det_inv, a * det_inv };

#if USE_SNUGBOX
    int x_ext = (int)ceilf(3.0f * sqrtf(a));
    int y_ext = (int)ceilf(3.0f * sqrtf(c));
    int my_radius = max(x_ext, y_ext);
#else
    float mid = 0.5f * (a + c);
    float lambda_max = mid + sqrtf(max(0.1f, mid * mid - det));
    int my_radius = (int)ceilf(3.0f * sqrtf(lambda_max));
#endif
    if (my_radius <= 0)
        return;

    float2 point = { mean_x, mean_y };
    means2D[idx] = point;
    conics[idx] = conic;
    radii[idx] = my_radius;

    uint2 rect_min, rect_max;
#if USE_SNUGBOX
    getSnugRect(point, x_ext, y_ext, rect_min, rect_max, tile_grid);
#else
    getRect(point, my_radius, rect_min, rect_max, tile_grid);
#endif
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// -----------------------------------------------------------------------
// Render: one block per tile, one thread per pixel
// -----------------------------------------------------------------------
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ means2D,
    const float3* __restrict__ conics,
    const float* __restrict__ weights,     // [N, 2] stored as [w0_real, w0_imag, w1_real, ...]
    float* __restrict__ out_color)         // [CHANNELS, H, W]
{
    // Identify current tile and pixel
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = { (float)pix.x, (float)pix.y };

    bool inside = pix.x < W && pix.y < H;
    bool done = !inside;

    // Load tile range
    uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Shared memory for cooperative loading
    __shared__ int collected_id[BLOCK_SIZE];
    __shared__ float2 collected_xy[BLOCK_SIZE];
    __shared__ float3 collected_conic[BLOCK_SIZE];
    __shared__ float2 collected_weight[BLOCK_SIZE];

    // Per-pixel accumulators
    float C[CHANNELS] = { 0 };

    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
    {
        // Early exit if entire block is done
        int num_done = __syncthreads_count(done);
        if (num_done == BLOCK_SIZE)
            break;

        // Cooperative load into shared memory
        int progress = i * BLOCK_SIZE + block.thread_rank();
        if (range.x + progress < range.y)
        {
            int coll_id = point_list[range.x + progress];
            collected_id[block.thread_rank()] = coll_id;
            collected_xy[block.thread_rank()] = means2D[coll_id];
            collected_conic[block.thread_rank()] = conics[coll_id];
            // Load weights as float2 from interleaved [N, 2] layout
            collected_weight[block.thread_rank()] = *((float2*)(weights + 2 * coll_id));
        }
        block.sync();

        // Process current batch
        for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
        {
            float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            float3 con = collected_conic[j];

            float power = -0.5f * (con.x * d.x * d.x + con.z * d.y * d.y) - con.y * d.x * d.y;
            if (power > 0.0f)
                continue;

            float G = expf(power);
            if (G < 1e-7f)
                continue;

            float2 w = collected_weight[j];
            C[0] += w.x * G;   // real channel
            C[1] += w.y * G;   // imag channel
        }
    }

    // Write output
    if (inside)
    {
        for (int ch = 0; ch < CHANNELS; ch++)
            out_color[ch * H * W + pix_id] = C[ch];
    }
}

// -----------------------------------------------------------------------
// C++ wrappers
// -----------------------------------------------------------------------
void FORWARD::preprocess(
    int N,
    const float* xy,
    const float* scaling,
    const float* rotation,
    int W, int H,
    int max_patch_radius,
    float min_scale,
    int* radii,
    float2* means2D,
    float3* conics,
    const dim3 tile_grid,
    uint32_t* tiles_touched)
{
    preprocessCUDA<<<(N + 255) / 256, 256>>>(
        N, xy, scaling, rotation,
        W, H, max_patch_radius, min_scale,
        radii, means2D, conics,
        tile_grid, tiles_touched);
}

void FORWARD::preprocess_cholesky(
    int N,
    const float* xy,
    const float* log_L_diag,
    const float* L_offdiag,
    int W, int H,
    int max_patch_radius,
    float min_scale,
    int* radii,
    float2* means2D,
    float3* conics,
    const dim3 tile_grid,
    uint32_t* tiles_touched)
{
    preprocessCUDA_cholesky<<<(N + 255) / 256, 256>>>(
        N, xy, log_L_diag, L_offdiag,
        W, H, max_patch_radius, min_scale,
        radii, means2D, conics,
        tile_grid, tiles_touched);
}

void FORWARD::render(
    const dim3 grid, dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    const float2* means2D,
    const float3* conics,
    const float* weights,
    float* out_color)
{
    renderCUDA<NUM_CHANNELS><<<grid, block>>>(
        ranges, point_list,
        W, H,
        means2D, conics, weights,
        out_color);
}
