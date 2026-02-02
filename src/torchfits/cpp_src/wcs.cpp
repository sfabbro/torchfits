#define _USE_MATH_DEFINES
#include <cmath>
#include "torchfits_torch.h"
#include <wcslib/wcs.h>
#include <wcslib/wcshdr.h>
#include <wcslib/wcsfix.h>
#include <wcslib/cel.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// GPU kernel for batch WCS transformations (when CUDA is available)
// REMOVED in Refactor

namespace torchfits {

class WCS {
private:
    struct wcsprm* wcs_;
    int nwcs_;  // Store the actual number of WCS structs returned by wcspih

public:
    WCS(const std::unordered_map<std::string, std::string>& header) : wcs_(nullptr), nwcs_(0) {
        int nkey = header.size();
        char* h = (char*)malloc(nkey * 81); // Allocate one extra byte for safety
        if (!h) {
            throw std::runtime_error("Failed to allocate memory for header");
        }
        
        char* p = h;
        for (const auto& [key, value] : header) {
            snprintf(p, 81, "%-8s= %-70s", key.c_str(), value.c_str());
            p += 80;
        }

        int nreject;
        int ctrl = 0;
        struct wcsprm* wcs_array = nullptr;
        int status = wcspih(h, nkey, WCSHDR_all, ctrl, &nreject, &nwcs_, &wcs_array);
        free(h); // Free the header buffer immediately after use
        
        if (status != 0 || nwcs_ <= 0) {
            // Clean up and throw error
            if (wcs_array) {
                wcsvfree(&nwcs_, &wcs_array);
            }
            if (status != 0) {
                throw std::runtime_error("Failed to parse WCS from header");
            } else {
                throw std::runtime_error("No WCS found in header");
            }
        }
        
        // Use the first wcs struct
        wcs_ = wcs_array;

        // Shift CRPIX by -1.0 to support 0-based indexing (PyTorch/Python convention)
        // FITS uses 1-based indexing, so CRPIX is 1-based.
        // We want: world = f(pixel_0based)
        // wcslib computes: world = f(pixel_input - (crpix - 1) + 1 - 1) ... ?
        // wcslib computes: intermediate = (pixel_input - crpix)
        // We want intermediate = (pixel_0based - (crpix_1based - 1))
        // So we set wcs->crpix = crpix_1based - 1.
        for (int i = 0; i < wcs_->naxis; i++) {
            wcs_->crpix[i] -= 1.0;
        }

        // Apply standard corrections (e.g. date formats, TPV aliases, etc.)
        int stat_fix[NWCSFIX]; // Status return for wcsfix
        // ctrl=1 (CDCXX), naxis=0 (default)
        // Note: checking return status might be noisy for minor fixes, but usually safe to ignore usually unless critical
        wcsfix(1, 0, wcs_, stat_fix);

        status = wcsset(wcs_);
        if (status != 0) {
            // Clean up and throw error
            // Note: we've already transferred ownership to wcs_, so we need to free it properly
            if (wcs_) {
                wcsvfree(&nwcs_, &wcs_);
                wcs_ = nullptr;
                nwcs_ = 0;
            }
            throw std::runtime_error("wcsset failed");
        }
        
        // Pre-compute matrices for GPU optimization
        precompute_matrices();
    }
    
    ~WCS() {
        if (wcs_) {
            // Use the actual number of WCS structs that were allocated
            wcsvfree(&nwcs_, &wcs_);
            wcs_ = nullptr;
            nwcs_ = 0;
        }
    }
    
    torch::Tensor pixel_to_world(const torch::Tensor& pixels) {
        auto device = pixels.device();
        auto dtype = pixels.dtype();
        int naxis = wcs_->naxis;
        
        // GPU-optimized path for CUDA tensors (only for 2D for now)
#ifdef TORCH_CUDA_AVAILABLE
        if (device.is_cuda() && is_simple_projection() && naxis == 2) {
            return pixel_to_world_gpu(pixels);
        }
#endif
        
        // CPU path with optimizations
        auto cpu_pixels = pixels.contiguous().cpu().to(torch::kFloat64);
        auto shape = cpu_pixels.sizes();
        int ncoord = shape[0];
        
        // Ensure input has correct shape (N, naxis)
        if (shape.size() < 2 || shape[1] != naxis) {
            throw std::runtime_error("Input pixels must have shape (N, " + std::to_string(naxis) + ")");
        }
        
        auto world = torch::empty_like(cpu_pixels);
        
        double* pixcrd = cpu_pixels.data_ptr<double>();
        double* worldcrd = world.data_ptr<double>();
        
        // Temporarily disable OpenMP path to debug
        // #ifdef HAS_OPENMP
        // if (ncoord > 1000) {
        //     return pixel_to_world_parallel(cpu_pixels, world, ncoord);
        // }
        // #endif
        
        // Standard wcslib transformation
        double* imgcrd = (double*)malloc(ncoord * naxis * sizeof(double));
        if (!imgcrd) throw std::runtime_error("Failed to allocate memory for imgcrd");
        
        double* phi = (double*)malloc(ncoord * sizeof(double));
        if (!phi) { free(imgcrd); throw std::runtime_error("Failed to allocate memory for phi"); }
        
        double* theta = (double*)malloc(ncoord * sizeof(double));
        if (!theta) { free(imgcrd); free(phi); throw std::runtime_error("Failed to allocate memory for theta"); }
        
        int* stat = (int*)malloc(ncoord * sizeof(int));
        if (!stat) { free(imgcrd); free(phi); free(theta); throw std::runtime_error("Failed to allocate memory for stat"); }
        
        int wcs_status = wcsp2s(wcs_, ncoord, naxis, pixcrd, imgcrd, phi, theta, worldcrd, stat);
        if (wcs_status != 0) {
            free(imgcrd); free(phi); free(theta); free(stat);
            throw std::runtime_error("WCS transformation failed");
        }
        
        free(imgcrd);
        free(phi);
        free(theta);
        free(stat);
        
        return world.to(dtype).to(device);
    }
    
    torch::Tensor world_to_pixel(const torch::Tensor& world) {
        auto device = world.device();
        auto dtype = world.dtype();
        int naxis = wcs_->naxis;
        
        // GPU-optimized path for CUDA tensors (only for 2D for now)
#ifdef TORCH_CUDA_AVAILABLE
        if (device.is_cuda() && is_simple_projection() && naxis == 2) {
            return world_to_pixel_gpu(world);
        }
#endif
        
        // CPU path with optimizations
        auto cpu_world = world.contiguous().cpu().to(torch::kFloat64);
        auto shape = cpu_world.sizes();
        int ncoord = shape[0];
        
        // Ensure input has correct shape (N, naxis)
        if (shape.size() < 2 || shape[1] != naxis) {
            throw std::runtime_error("Input world coordinates must have shape (N, " + std::to_string(naxis) + ")");
        }
        
        auto pixels = torch::empty_like(cpu_world);
        
        double* worldcrd = cpu_world.data_ptr<double>();
        double* pixcrd = pixels.data_ptr<double>();
        
        // Temporarily disable OpenMP path to debug
        // #ifdef HAS_OPENMP
        // if (ncoord > 1000) {
        //     return world_to_pixel_parallel(cpu_world, pixels, ncoord);
        // }
        // #endif
        
        // Standard wcslib transformation
        double* imgcrd = (double*)malloc(ncoord * naxis * sizeof(double));
        if (!imgcrd) throw std::runtime_error("Failed to allocate memory for imgcrd");
        
        double* phi = (double*)malloc(ncoord * sizeof(double));
        if (!phi) { free(imgcrd); throw std::runtime_error("Failed to allocate memory for phi"); }
        
        double* theta = (double*)malloc(ncoord * sizeof(double));
        if (!theta) { free(imgcrd); free(phi); throw std::runtime_error("Failed to allocate memory for theta"); }
        
        int* stat = (int*)malloc(ncoord * sizeof(int));
        if (!stat) { free(imgcrd); free(phi); free(theta); throw std::runtime_error("Failed to allocate memory for stat"); }
        
        int wcs_status = wcss2p(wcs_, ncoord, naxis, worldcrd, phi, theta, imgcrd, pixcrd, stat);
        if (wcs_status != 0) {
            free(imgcrd); free(phi); free(theta); free(stat);
            throw std::runtime_error("WCS transformation failed");
        }
        
        free(imgcrd);
        free(phi);
        free(theta);
        free(stat);
        
        return pixels.to(dtype).to(device);
    }
    
    torch::Tensor get_footprint() {
        int naxis = wcs_->naxis;
        if (naxis != 2) {
            throw std::runtime_error("get_footprint only supported for 2D WCS");
        }
        auto corners = torch::empty({4, 2}, torch::kFloat64);
        double* data = corners.data_ptr<double>();
        
        data[0] = 0.5; data[1] = 0.5;
        data[2] = wcs_->crpix[0] * 2 - 0.5; data[3] = 0.5;
        data[4] = wcs_->crpix[0] * 2 - 0.5; data[5] = wcs_->crpix[1] * 2 - 0.5;
        data[6] = 0.5; data[7] = wcs_->crpix[1] * 2 - 0.5;
        
        return pixel_to_world(corners);
    }

    int naxis() const { return wcs_->naxis; }
    int test_method() const { return 42; }
    torch::Tensor crpix() const { 
        auto tensor = torch::empty({wcs_->naxis}, torch::kFloat64);
        for(int i=0; i<wcs_->naxis; i++) tensor[i] = wcs_->crpix[i];
        return tensor;
    }
    torch::Tensor crval() const { 
        auto tensor = torch::empty({wcs_->naxis}, torch::kFloat64);
        for(int i=0; i<wcs_->naxis; i++) tensor[i] = wcs_->crval[i];
        return tensor;
    }
    torch::Tensor cdelt() const { 
        auto tensor = torch::empty({wcs_->naxis}, torch::kFloat64);
        for(int i=0; i<wcs_->naxis; i++) tensor[i] = wcs_->cdelt[i];
        return tensor;
    }

private:
    // Cached matrices for GPU optimization
    double cd_matrix_[4];
    double cd_matrix_inv_[4];
    double rotation_matrix_[9];
    double crpix_[2];
    bool is_tan_projection_ = false;
    
    // Helper to compute inverse of 2x2 matrix
    void invert_2x2(const double* m, double* inv) {
        double det = m[0]*m[3] - m[1]*m[2];
        if (std::abs(det) < 1e-10) {
            inv[0] = inv[1] = inv[2] = inv[3] = 0;
            return;
        }
        double inv_det = 1.0 / det;
        inv[0] =  m[3] * inv_det;
        inv[1] = -m[1] * inv_det;
        inv[2] = -m[2] * inv_det;
        inv[3] =  m[0] * inv_det;
    }

    void precompute_matrices() {
        // Only for 2D
        if (wcs_->naxis != 2) return;

        // Check for TAN projection
        bool is_tan = false;
        const char* ctype1 = wcs_->ctype[0];
        const char* ctype2 = wcs_->ctype[1];

        // Basic check for -TAN suffix
        if (strlen(ctype1) >= 8 && strncmp(ctype1 + 4, "-TAN", 4) == 0 &&
            strlen(ctype2) >= 8 && strncmp(ctype2 + 4, "-TAN", 4) == 0) {
            is_tan = true;
        }

        if (!is_tan) {
            is_tan_projection_ = false;
            return;
        }

        // 1. Extract CD Matrix (PC * CDELT)
        // wcs->pc is naxis * naxis (2x2)
        // cd[i][j] = pc[i][j] * cdelt[i]
        // Note: pc is row-major in wcslib logic for access but let's verify storage
        // wcslib: pc is double*, element (i,j) is pc[i*naxis + j]

        cd_matrix_[0] = wcs_->pc[0] * wcs_->cdelt[0];
        cd_matrix_[1] = wcs_->pc[1] * wcs_->cdelt[0];
        cd_matrix_[2] = wcs_->pc[2] * wcs_->cdelt[1];
        cd_matrix_[3] = wcs_->pc[3] * wcs_->cdelt[1];

        // Invert CD matrix
        invert_2x2(cd_matrix_, cd_matrix_inv_);

        // 2. Extract CRPIX
        crpix_[0] = wcs_->crpix[0];
        crpix_[1] = wcs_->crpix[1];

        // 3. Compute Rotation Matrix via Probe
        // We probe 3 native unit vectors: (1,0,0), (0,1,0), (0,0,1)
        // Corresponding native spherical (phi, theta) in degrees:
        // (1,0,0) -> phi=0, theta=0
        // (0,1,0) -> phi=90, theta=0
        // (0,0,1) -> phi=0, theta=90

        double native_phi[3] = {0.0, 90.0, 0.0};
        double native_theta[3] = {0.0, 0.0, 90.0};

        double cel_lng[3];
        double cel_lat[3];
        int stat[3];

        // Use wcs->cel structure to transform native spherical to celestial spherical
        // stride = 1
        if (cels2s(&wcs_->cel, 3, 1, native_phi, native_theta, cel_lng, cel_lat, stat) != 0) {
            is_tan_projection_ = false;
            return;
        }

        // Convert celestial results to Cartesian unit vectors (columns of R)
        for (int i = 0; i < 3; i++) {
            double lng_rad = cel_lng[i] * M_PI / 180.0;
            double lat_rad = cel_lat[i] * M_PI / 180.0;

            double x = cos(lat_rad) * cos(lng_rad);
            double y = cos(lat_rad) * sin(lng_rad);
            double z = sin(lat_rad);

            // Store in column i
            rotation_matrix_[0 * 3 + i] = x;
            rotation_matrix_[1 * 3 + i] = y;
            rotation_matrix_[2 * 3 + i] = z;
        }

        is_tan_projection_ = true;
    }

    bool is_simple_projection() const {
        return is_tan_projection_;
    }

    torch::Tensor pixel_to_world_gpu(const torch::Tensor& pixels) {
        auto device = pixels.device();
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);

        // 1. Apply Linear Transform
        auto crpix = torch::from_blob(crpix_, {2}, torch::kFloat64).to(device);
        auto p = pixels - crpix;

        auto cd = torch::from_blob(cd_matrix_, {2, 2}, torch::kFloat64).to(device);
        // intermediate = p @ cd.T
        auto intermediate = torch::matmul(p, cd.t());

        auto x = intermediate.select(1, 0);
        auto y = intermediate.select(1, 1);

        // 2. Deproject TAN
        auto r = torch::sqrt(x*x + y*y);

        // Avoid division by zero or singularity at r=0
        // For r=0, theta should be 90 deg.
        // atan2(180/pi, 0) = pi/2. Correct.
        double rad_deg = 180.0 / M_PI;
        auto theta_native = torch::atan2(torch::tensor(rad_deg, options), r);
        auto phi_native = torch::atan2(x, -y);

        // 3. Convert Native Spherical to Cartesian Unit Vector
        auto cos_theta = torch::cos(theta_native);
        auto sin_theta = torch::sin(theta_native);
        auto cos_phi = torch::cos(phi_native);
        auto sin_phi = torch::sin(phi_native);

        auto ux = cos_theta * cos_phi;
        auto uy = cos_theta * sin_phi;
        auto uz = sin_theta;

        auto u_native = torch::stack({ux, uy, uz}, 1);

        // 4. Rotate to Celestial
        auto R = torch::from_blob(rotation_matrix_, {3, 3}, torch::kFloat64).to(device);
        auto u_cel = torch::matmul(u_native, R.t());

        // 5. Convert Celestial Unit Vector to Spherical
        auto cx = u_cel.select(1, 0);
        auto cy = u_cel.select(1, 1);
        auto cz = u_cel.select(1, 2);

        cz = torch::clamp(cz, -1.0, 1.0);

        auto alpha_rad = torch::atan2(cy, cx);
        auto delta_rad = torch::asin(cz);

        auto alpha = alpha_rad * rad_deg;
        auto delta = delta_rad * rad_deg;

        alpha = torch::remainder(alpha, 360.0);

        return torch::stack({alpha, delta}, 1);
    }

    torch::Tensor world_to_pixel_gpu(const torch::Tensor& world) {
        auto device = world.device();
        auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
        double rad_deg = 180.0 / M_PI;
        double deg_rad = M_PI / 180.0;

        auto alpha = world.select(1, 0);
        auto delta = world.select(1, 1);

        auto alpha_rad = alpha * deg_rad;
        auto delta_rad = delta * deg_rad;

        // 1. Celestial Spherical -> Cartesian
        auto cx = torch::cos(delta_rad) * torch::cos(alpha_rad);
        auto cy = torch::cos(delta_rad) * torch::sin(alpha_rad);
        auto cz = torch::sin(delta_rad);

        auto u_cel = torch::stack({cx, cy, cz}, 1);

        // 2. Rotate to Native
        // u_native = u_cel @ R (since R maps Native -> Cel, and is orthogonal)
        auto R = torch::from_blob(rotation_matrix_, {3, 3}, torch::kFloat64).to(device);
        auto u_native = torch::matmul(u_cel, R);

        auto ux = u_native.select(1, 0);
        auto uy = u_native.select(1, 1);
        auto uz = u_native.select(1, 2);

        // 3. Native Cartesian -> Spherical
        uz = torch::clamp(uz, -1.0, 1.0);
        auto theta_native = torch::asin(uz);
        auto phi_native = torch::atan2(uy, ux);

        // 4. Project TAN
        // r = (180/pi) / tan(theta)
        // Check for theta=pi/2 (pole) -> r=0
        // tan(pi/2) is inf.
        // We use cot(theta) = tan(pi/2 - theta)?
        // r = (180/pi) * cot(theta)
        // cot(theta) = cos(theta)/sin(theta)
        auto r = (rad_deg * torch::cos(theta_native)) / torch::sin(theta_native);

        // If theta is very close to 90 deg, r -> 0.
        // If theta is 0 (equator), r -> inf.
        // TAN diverges at equator (theta=0).

        auto x = r * torch::sin(phi_native);
        auto y = -r * torch::cos(phi_native); // Note minus sign in FITS TAN definition

        auto intermediate = torch::stack({x, y}, 1);

        // 5. Inverse Linear Transform
        auto cd_inv = torch::from_blob(cd_matrix_inv_, {2, 2}, torch::kFloat64).to(device);
        // d = intermediate @ cd_inv.T
        auto d = torch::matmul(intermediate, cd_inv.t());

        auto crpix = torch::from_blob(crpix_, {2}, torch::kFloat64).to(device);
        auto pixels = d + crpix;

        return pixels;
    }
    
public:
    // Getters for WCS properties
    
    // CTYPE (Coordinate types)
    std::vector<std::string> ctype() const {
        std::vector<std::string> result;
        for (int i = 0; i < wcs_->naxis; i++) {
            result.push_back(std::string(wcs_->ctype[i]));
        }
        return result;
    }
    
    // CUNIT (Coordinate units)
    std::vector<std::string> cunit() const {
        std::vector<std::string> result;
        for (int i = 0; i < wcs_->naxis; i++) {
            result.push_back(std::string(wcs_->cunit[i]));
        }
        return result;
    }

    // PC Matrix (Linear transformation) - returned as 1D tensor for simplicity
    torch::Tensor pc() const {
        // wcs->pc is naxis * naxis
        int n = wcs_->naxis;
        auto tensor = torch::empty({n, n}, torch::kFloat64);
        double* data = tensor.data_ptr<double>();
        for (int i = 0; i < n * n; i++) {
            data[i] = wcs_->pc[i];
        }
        return tensor;
    }
    
    // CD Matrix (if present, otherwise derived/unused by wcslib normalized PC)
    // Note: wcslib often normalizes CD into PC + CDELT. 
    // We expose PC as the primary linear matrix.
    
    double lonpole() const { return wcs_->lonpole; }
    double latpole() const { return wcs_->latpole; }
    
};
}