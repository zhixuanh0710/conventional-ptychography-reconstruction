/*
 * GSCP 2D Gaussian Splatting Rasterizer — Forward/backward orchestration.
 *
 * This is the main pipeline that connects preprocessing, binning, and rendering.
 * Adapted from R2-Gaussian's rasterizer_impl.cu, simplified for 2D (no depth sorting).
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper: find the next-highest bit of the MSB on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

// Generate (tile_id | gaussian_idx) key/value pairs for all Gaussian-tile overlaps.
__global__ void duplicateWithKeys(
    int N,
    const float2* points_xy,
    const float3* conics,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    int* radii,
    dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= N)
        return;

    if (radii[idx] > 0)
    {
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
        uint2 rect_min, rect_max;
#if USE_SNUGBOX
        getSnugRectFromConic(points_xy[idx], conics[idx], rect_min, rect_max, grid);
#else
        getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
#endif

        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                uint64_t key = y * grid.x + x;
                key <<= 32;
                key |= (uint32_t)idx;
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++;
            }
        }
    }
}

// Identify per-tile start/end ranges in the sorted list.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    uint64_t key = point_list_keys[idx];
    uint32_t currtile = key >> 32;
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}

// -----------------------------------------------------------------------
// State chunk allocation
// -----------------------------------------------------------------------
GscpRasterizer::GeometryState GscpRasterizer::GeometryState::fromChunk(char*& chunk, size_t N)
{
    GeometryState geom;
    obtain(chunk, geom.internal_radii, N, 128);
    obtain(chunk, geom.means2D, N, 128);
    obtain(chunk, geom.conics, N, 128);
    obtain(chunk, geom.tiles_touched, N, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, N);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, N, 128);
    return geom;
}

GscpRasterizer::ImageState GscpRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
    ImageState img;
    obtain(chunk, img.ranges, N, 128);
    return img;
}

GscpRasterizer::BinningState GscpRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.point_list_keys, P, 128);
    obtain(chunk, binning.point_list_keys_unsorted, P, 128);
    cub::DeviceRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.point_list_keys_unsorted, binning.point_list_keys,
        binning.point_list_unsorted, binning.point_list, P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}

// -----------------------------------------------------------------------
// Forward pass orchestration
// -----------------------------------------------------------------------
int GscpRasterizer::Rasterizer::forward(
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    const int N,
    const int W, int H,
    const float* xy,
    const float* scaling,
    const float* rotation,
    const float* weights,
    const int max_patch_radius,
    const float min_scale,
    float* out_color,
    int* radii,
    bool debug)
{
    // Allocate geometry state
    size_t chunk_size = required<GeometryState>(N);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, N);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Image state (tile ranges)
    size_t img_chunk_size = required<ImageState>(tile_grid.x * tile_grid.y);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, tile_grid.x * tile_grid.y);

    // Step 1: Preprocess — compute means, conics, radii, tile counts
    CHECK_CUDA(FORWARD::preprocess(
        N, xy, scaling, rotation,
        W, H, max_patch_radius, min_scale,
        radii, geomState.means2D, geomState.conics,
        tile_grid, geomState.tiles_touched), debug)

    // Step 2: Prefix sum of tiles_touched -> offsets
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(
        geomState.scanning_space, geomState.scan_size,
        geomState.tiles_touched, geomState.point_offsets, N), debug)

    // Get total number of Gaussian-tile instances
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + N - 1,
        sizeof(int), cudaMemcpyDeviceToHost), debug);

    if (num_rendered == 0)
        return 0;

    // Step 3: Binning — duplicate keys, sort by tile
    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    duplicateWithKeys<<<(N + 255) / 256, 256>>>(
        N,
        geomState.means2D,
        geomState.conics,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii,
        tile_grid);
    CHECK_CUDA(, debug)

    // Sort only on tile bits (upper 32 bits of key)
    int bit = getHigherMsb(tile_grid.x * tile_grid.y);
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 32, 32 + bit), debug)

    // Step 4: Identify tile ranges
    CHECK_CUDA(cudaMemset(imgState.ranges, 0,
        tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    if (num_rendered > 0)
        identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered,
            binningState.point_list_keys,
            imgState.ranges);
    CHECK_CUDA(, debug)

    // Step 5: Render — tile-based 2-channel accumulation
    CHECK_CUDA(FORWARD::render(
        tile_grid, block,
        imgState.ranges,
        binningState.point_list,
        W, H,
        geomState.means2D,
        geomState.conics,
        weights,
        out_color), debug)

    return num_rendered;
}

// -----------------------------------------------------------------------
// Forward pass orchestration (Cholesky)
// -----------------------------------------------------------------------
int GscpRasterizer::Rasterizer::forward_cholesky(
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    const int N,
    const int W, int H,
    const float* xy,
    const float* log_L_diag,
    const float* L_offdiag,
    const float* weights,
    const int max_patch_radius,
    const float min_scale,
    float* out_color,
    int* radii,
    bool debug)
{
    size_t chunk_size = required<GeometryState>(N);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, N);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    size_t img_chunk_size = required<ImageState>(tile_grid.x * tile_grid.y);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, tile_grid.x * tile_grid.y);

    // Step 1: Cholesky preprocess
    CHECK_CUDA(FORWARD::preprocess_cholesky(
        N, xy, log_L_diag, L_offdiag,
        W, H, max_patch_radius, min_scale,
        radii, geomState.means2D, geomState.conics,
        tile_grid, geomState.tiles_touched), debug)

    // Steps 2-5: identical to RS (binning, sorting, rendering are parameterization-agnostic)
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(
        geomState.scanning_space, geomState.scan_size,
        geomState.tiles_touched, geomState.point_offsets, N), debug)

    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + N - 1,
        sizeof(int), cudaMemcpyDeviceToHost), debug);

    if (num_rendered == 0)
        return 0;

    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    duplicateWithKeys<<<(N + 255) / 256, 256>>>(
        N, geomState.means2D, geomState.conics,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii, tile_grid);
    CHECK_CUDA(, debug)

    int bit = getHigherMsb(tile_grid.x * tile_grid.y);
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        binningState.list_sorting_space, binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 32, 32 + bit), debug)

    CHECK_CUDA(cudaMemset(imgState.ranges, 0,
        tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    if (num_rendered > 0)
        identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
            num_rendered, binningState.point_list_keys, imgState.ranges);
    CHECK_CUDA(, debug)

    CHECK_CUDA(FORWARD::render(
        tile_grid, block,
        imgState.ranges, binningState.point_list,
        W, H, geomState.means2D, geomState.conics,
        weights, out_color), debug)

    return num_rendered;
}

// -----------------------------------------------------------------------
// Backward pass orchestration
// -----------------------------------------------------------------------
void GscpRasterizer::Rasterizer::backward(
    const int N, int R,
    const int W, int H,
    const float* xy,
    const float* scaling,
    const float* rotation,
    const float* weights,
    const int max_patch_radius,
    const float min_scale,
    const int* radii,
    char* geom_buffer,
    char* binning_buffer,
    char* img_buffer,
    const float* dL_dpix,
    float* dL_dmean2D,
    float* dL_dconic,
    float* dL_dweight,
    float* dL_dxy,
    float* dL_dscaling,
    float* dL_drotation,
    bool debug)
{
    GeometryState geomState = GeometryState::fromChunk(geom_buffer, N);
    BinningState binningState = BinningState::fromChunk(binning_buffer, R);

    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    ImageState imgState = ImageState::fromChunk(img_buffer, tile_grid.x * tile_grid.y);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Step 1: Backward render — pixel gradients -> per-Gaussian gradients
    CHECK_CUDA(BACKWARD::render(
        tile_grid, block,
        imgState.ranges,
        binningState.point_list,
        W, H,
        geomState.means2D,
        geomState.conics,
        weights,
        dL_dpix,
        (float2*)dL_dmean2D,
        (float3*)dL_dconic,
        dL_dweight), debug)

    // Step 2: Backward preprocess — conic/mean grads -> parameter grads
    CHECK_CUDA(BACKWARD::preprocess(
        N, xy, scaling, rotation,
        W, H, max_patch_radius, min_scale, radii,
        (float2*)dL_dmean2D,
        (float3*)dL_dconic,
        dL_dxy, dL_dscaling, dL_drotation), debug)
}

// -----------------------------------------------------------------------
// Backward pass orchestration (Cholesky)
// -----------------------------------------------------------------------
void GscpRasterizer::Rasterizer::backward_cholesky(
    const int N, int R,
    const int W, int H,
    const float* xy,
    const float* log_L_diag,
    const float* L_offdiag,
    const float* weights,
    const int max_patch_radius,
    const float min_scale,
    const int* radii,
    char* geom_buffer,
    char* binning_buffer,
    char* img_buffer,
    const float* dL_dpix,
    float* dL_dmean2D,
    float* dL_dconic,
    float* dL_dweight,
    float* dL_dxy,
    float* dL_dlog_L_diag,
    float* dL_dL_offdiag,
    bool debug)
{
    GeometryState geomState = GeometryState::fromChunk(geom_buffer, N);
    BinningState binningState = BinningState::fromChunk(binning_buffer, R);

    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    ImageState imgState = ImageState::fromChunk(img_buffer, tile_grid.x * tile_grid.y);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Step 1: Backward render (shared with RS — parameterization-agnostic)
    CHECK_CUDA(BACKWARD::render(
        tile_grid, block,
        imgState.ranges,
        binningState.point_list,
        W, H,
        geomState.means2D,
        geomState.conics,
        weights,
        dL_dpix,
        (float2*)dL_dmean2D,
        (float3*)dL_dconic,
        dL_dweight), debug)

    // Step 2: Backward preprocess (Cholesky-specific)
    CHECK_CUDA(BACKWARD::preprocess_cholesky(
        N, xy, log_L_diag, L_offdiag,
        W, H, max_patch_radius, min_scale, radii,
        (float2*)dL_dmean2D,
        (float3*)dL_dconic,
        dL_dxy, dL_dlog_L_diag, dL_dL_offdiag), debug)
}
