/*
 * GSCP 2D Gaussian Splatting Rasterizer — PyBind11 module registration.
 */

#include <torch/extension.h>

// Declarations from rasterize_gaussians.cu — RS parameterization
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeForward(
    const torch::Tensor& xy,
    const torch::Tensor& scaling,
    const torch::Tensor& rotation,
    const torch::Tensor& weights,
    const int image_height,
    const int image_width,
    const int max_patch_radius,
    const float min_scale,
    const bool debug);

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
    const bool debug);

// Declarations from rasterize_gaussians.cu — Cholesky parameterization
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeForwardCholesky(
    const torch::Tensor& xy,
    const torch::Tensor& log_L_diag,
    const torch::Tensor& L_offdiag,
    const torch::Tensor& weights,
    const int image_height,
    const int image_width,
    const int max_patch_radius,
    const float min_scale,
    const bool debug);

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
    const bool debug);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward", &RasterizeForward);
    m.def("rasterize_backward", &RasterizeBackward);
    m.def("rasterize_forward_cholesky", &RasterizeForwardCholesky);
    m.def("rasterize_backward_cholesky", &RasterizeBackwardCholesky);
}
