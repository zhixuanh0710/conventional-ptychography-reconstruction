/*
 * GSCP 2D Gaussian Splatting Rasterizer — Public API.
 */

#ifndef GSCP_RASTERIZER_H_INCLUDED
#define GSCP_RASTERIZER_H_INCLUDED

#include <functional>

namespace GscpRasterizer
{
    class Rasterizer
    {
    public:
        static int forward(
            std::function<char* (size_t)> geometryBuffer,
            std::function<char* (size_t)> binningBuffer,
            std::function<char* (size_t)> imageBuffer,
            const int N,           // number of Gaussians
            const int W, int H,    // image dimensions
            const float* xy,       // [N, 2] normalized positions
            const float* scaling,  // [N, 2] log-space scales
            const float* rotation, // [N, 1] rotation angles
            const float* weights,  // [N, 2] (w_real, w_imag)
            const int max_patch_radius,
            const float min_scale, // lower bound on activated sigma (pixels)
            float* out_color,      // [2, H, W]
            int* radii,            // [N]
            bool debug);

        static void backward(
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
            bool debug);

        // Cholesky parameterization: L = [[l11, 0], [l21, l22]]
        static int forward_cholesky(
            std::function<char* (size_t)> geometryBuffer,
            std::function<char* (size_t)> binningBuffer,
            std::function<char* (size_t)> imageBuffer,
            const int N,
            const int W, int H,
            const float* xy,           // [N, 2] pixel-space positions
            const float* log_L_diag,   // [N, 2] log of Cholesky diagonal
            const float* L_offdiag,    // [N] lower-triangular off-diagonal
            const float* weights,      // [N, 2] (w_real, w_imag)
            const int max_patch_radius,
            const float min_scale,
            float* out_color,          // [2, H, W]
            int* radii,                // [N]
            bool debug);

        static void backward_cholesky(
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
            bool debug);
    };
}

#endif
