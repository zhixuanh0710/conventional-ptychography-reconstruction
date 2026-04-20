/*
 * GSCP 2D Gaussian Splatting Rasterizer — PyTorch tensor bridge.
 *
 * Converts torch::Tensor inputs to raw pointers and calls the CUDA rasterizer.
 * Returns torch::Tensor outputs.
 */

#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <memory>
#include <functional>
#include "gscp_rasterizer/config.h"
#include "gscp_rasterizer/rasterizer.h"

// Utility: create a resizable byte tensor and return a function that grows it.
static std::function<char*(size_t)> resizeFunctional(torch::Tensor& t)
{
    auto handleResize = [&t](size_t N) -> char* {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return handleResize;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeForward(
    const torch::Tensor& xy,           // [N, 2]
    const torch::Tensor& scaling,      // [N, 2]
    const torch::Tensor& rotation,     // [N, 1]
    const torch::Tensor& weights,      // [N, 2]
    const int image_height,
    const int image_width,
    const int max_patch_radius,
    const float min_scale,
    const bool debug)
{
    if (xy.ndimension() != 2 || xy.size(1) != 2)
        AT_ERROR("xy must have dimensions (num_points, 2)");

    const int N = xy.size(0);
    const int H = image_height;
    const int W = image_width;

    auto float_opts = xy.options().dtype(torch::kFloat32);
    auto int_opts = xy.options().dtype(torch::kInt32);

    torch::Tensor out_color = torch::zeros({NUM_CHANNELS, H, W}, float_opts);
    torch::Tensor radii = torch::zeros({N}, int_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions byte_opts(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, byte_opts.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, byte_opts.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, byte_opts.device(device));
    auto geomFunc = resizeFunctional(geomBuffer);
    auto binningFunc = resizeFunctional(binningBuffer);
    auto imgFunc = resizeFunctional(imgBuffer);

    int rendered = 0;
    if (N != 0)
    {
        rendered = GscpRasterizer::Rasterizer::forward(
            geomFunc, binningFunc, imgFunc,
            N, W, H,
            xy.contiguous().data_ptr<float>(),
            scaling.contiguous().data_ptr<float>(),
            rotation.contiguous().data_ptr<float>(),
            weights.contiguous().data_ptr<float>(),
            max_patch_radius,
            min_scale,
            out_color.contiguous().data_ptr<float>(),
            radii.contiguous().data_ptr<int>(),
            debug);
    }

    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeBackward(
    const torch::Tensor& xy,
    const torch::Tensor& scaling,
    const torch::Tensor& rotation,
    const torch::Tensor& weights,
    const torch::Tensor& radii,
    const int image_height,
    const int image_width,
    const int max_patch_radius,
    const float min_scale,
    const int num_rendered,
    const torch::Tensor& geomBuffer,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const torch::Tensor& dL_dout_color,
    const bool debug)
{
    const int N = xy.size(0);
    const int H = image_height;
    const int W = image_width;

    torch::Tensor dL_dxy = torch::zeros({N, 2}, xy.options());
    torch::Tensor dL_dscaling = torch::zeros({N, 2}, xy.options());
    torch::Tensor dL_drotation = torch::zeros({N, 1}, xy.options());
    torch::Tensor dL_dweight = torch::zeros({N, 2}, xy.options());

    // Intermediate gradient buffers
    torch::Tensor dL_dmean2D = torch::zeros({N, 2}, xy.options());
    torch::Tensor dL_dconic = torch::zeros({N, 3}, xy.options());

    if (N != 0)
    {
        GscpRasterizer::Rasterizer::backward(
            N, num_rendered,
            W, H,
            xy.contiguous().data_ptr<float>(),
            scaling.contiguous().data_ptr<float>(),
            rotation.contiguous().data_ptr<float>(),
            weights.contiguous().data_ptr<float>(),
            max_patch_radius,
            min_scale,
            radii.contiguous().data_ptr<int>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            dL_dout_color.contiguous().data_ptr<float>(),
            dL_dmean2D.contiguous().data_ptr<float>(),
            dL_dconic.contiguous().data_ptr<float>(),
            dL_dweight.contiguous().data_ptr<float>(),
            dL_dxy.contiguous().data_ptr<float>(),
            dL_dscaling.contiguous().data_ptr<float>(),
            dL_drotation.contiguous().data_ptr<float>(),
            debug);
    }

    return std::make_tuple(dL_dxy, dL_dscaling, dL_drotation, dL_dweight);
}

// -----------------------------------------------------------------------
// Cholesky parameterization bridge functions
// -----------------------------------------------------------------------

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeForwardCholesky(
    const torch::Tensor& xy,           // [N, 2]
    const torch::Tensor& log_L_diag,   // [N, 2]
    const torch::Tensor& L_offdiag,    // [N]
    const torch::Tensor& weights,      // [N, 2]
    const int image_height,
    const int image_width,
    const int max_patch_radius,
    const float min_scale,
    const bool debug)
{
    if (xy.ndimension() != 2 || xy.size(1) != 2)
        AT_ERROR("xy must have dimensions (num_points, 2)");

    const int N = xy.size(0);
    const int H = image_height;
    const int W = image_width;

    auto float_opts = xy.options().dtype(torch::kFloat32);
    auto int_opts = xy.options().dtype(torch::kInt32);

    torch::Tensor out_color = torch::zeros({NUM_CHANNELS, H, W}, float_opts);
    torch::Tensor radii = torch::zeros({N}, int_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions byte_opts(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, byte_opts.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, byte_opts.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, byte_opts.device(device));
    auto geomFunc = resizeFunctional(geomBuffer);
    auto binningFunc = resizeFunctional(binningBuffer);
    auto imgFunc = resizeFunctional(imgBuffer);

    int rendered = 0;
    if (N != 0)
    {
        rendered = GscpRasterizer::Rasterizer::forward_cholesky(
            geomFunc, binningFunc, imgFunc,
            N, W, H,
            xy.contiguous().data_ptr<float>(),
            log_L_diag.contiguous().data_ptr<float>(),
            L_offdiag.contiguous().data_ptr<float>(),
            weights.contiguous().data_ptr<float>(),
            max_patch_radius,
            min_scale,
            out_color.contiguous().data_ptr<float>(),
            radii.contiguous().data_ptr<int>(),
            debug);
    }

    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeBackwardCholesky(
    const torch::Tensor& xy,
    const torch::Tensor& log_L_diag,
    const torch::Tensor& L_offdiag,
    const torch::Tensor& weights,
    const torch::Tensor& radii,
    const int image_height,
    const int image_width,
    const int max_patch_radius,
    const float min_scale,
    const int num_rendered,
    const torch::Tensor& geomBuffer,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const torch::Tensor& dL_dout_color,
    const bool debug)
{
    const int N = xy.size(0);
    const int H = image_height;
    const int W = image_width;

    torch::Tensor dL_dxy = torch::zeros({N, 2}, xy.options());
    torch::Tensor dL_dlog_L_diag = torch::zeros({N, 2}, xy.options());
    torch::Tensor dL_dL_offdiag = torch::zeros({N}, xy.options());
    torch::Tensor dL_dweight = torch::zeros({N, 2}, xy.options());

    // Intermediate gradient buffers
    torch::Tensor dL_dmean2D = torch::zeros({N, 2}, xy.options());
    torch::Tensor dL_dconic = torch::zeros({N, 3}, xy.options());

    if (N != 0)
    {
        GscpRasterizer::Rasterizer::backward_cholesky(
            N, num_rendered,
            W, H,
            xy.contiguous().data_ptr<float>(),
            log_L_diag.contiguous().data_ptr<float>(),
            L_offdiag.contiguous().data_ptr<float>(),
            weights.contiguous().data_ptr<float>(),
            max_patch_radius,
            min_scale,
            radii.contiguous().data_ptr<int>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            dL_dout_color.contiguous().data_ptr<float>(),
            dL_dmean2D.contiguous().data_ptr<float>(),
            dL_dconic.contiguous().data_ptr<float>(),
            dL_dweight.contiguous().data_ptr<float>(),
            dL_dxy.contiguous().data_ptr<float>(),
            dL_dlog_L_diag.contiguous().data_ptr<float>(),
            dL_dL_offdiag.contiguous().data_ptr<float>(),
            debug);
    }

    return std::make_tuple(dL_dxy, dL_dlog_L_diag, dL_dL_offdiag, dL_dweight);
}
