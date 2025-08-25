#include <torch/torch.h>
#include <wcslib/wcs.h>
#include <wcslib/wcshdr.h>
#include <omp.h>

namespace torchfits {

class WCS {
public:
    WCS(const std::unordered_map<std::string, std::string>& header) {
        int nkey = header.size();
        char* h = (char*)malloc(nkey * 80);
        char* p = h;
        for (const auto& [key, value] : header) {
            snprintf(p, 80, "%-8s= %-70s", key.c_str(), value.c_str());
            p += 80;
        }

        int nreject;
        int ctrl = 0;
        int nwcs;
        struct wcsprm* wcs_array;
        int status = wcspih(h, nkey, WCSHDR_all, ctrl, &nreject, &nwcs, &wcs_array);
        if (nwcs > 0) {
            wcs_ = wcs_array;
        } else {
            throw std::runtime_error("No WCS found in header");
        }
        free(h);

        if (status) {
            throw std::runtime_error("Failed to parse WCS from header");
        }

        status = wcsset(wcs_);
        if (status) {
            throw std::runtime_error("wcsset failed");
        }
    }
    
    ~WCS() {
        if (wcs_) {
            wcsfree(wcs_);
            free(wcs_);
        }
    }
    
    torch::Tensor pixel_to_world(const torch::Tensor& pixels) {
        auto cpu_pixels = pixels.contiguous().cpu().to(torch::kFloat64);
        auto shape = cpu_pixels.sizes();
        int ncoord = shape[0];
        auto world = torch::empty_like(cpu_pixels);
        
        double* pixcrd = cpu_pixels.data_ptr<double>();
        double* worldcrd = world.data_ptr<double>();
        double* imgcrd = (double*)malloc(ncoord * 2 * sizeof(double));
        double* phi = (double*)malloc(ncoord * sizeof(double));
        double* theta = (double*)malloc(ncoord * sizeof(double));
        int* stat = (int*)malloc(ncoord * sizeof(int));
        
        wcsp2s(wcs_, ncoord, 2, pixcrd, imgcrd, phi, theta, worldcrd, stat);
        
        free(imgcrd);
        free(phi);
        free(theta);
        free(stat);
        
        return world.to(pixels.dtype()).to(pixels.device());
    }
    
    torch::Tensor world_to_pixel(const torch::Tensor& world) {
        auto cpu_world = world.contiguous().cpu().to(torch::kFloat64);
        auto shape = cpu_world.sizes();
        int ncoord = shape[0];
        auto pixels = torch::empty_like(cpu_world);
        
        double* worldcrd = cpu_world.data_ptr<double>();
        double* pixcrd = pixels.data_ptr<double>();
        double* imgcrd = (double*)malloc(ncoord * 2 * sizeof(double));
        double* phi = (double*)malloc(ncoord * sizeof(double));
        double* theta = (double*)malloc(ncoord * sizeof(double));
        int* stat = (int*)malloc(ncoord * sizeof(int));
        
        wcss2p(wcs_, ncoord, 2, worldcrd, phi, theta, imgcrd, pixcrd, stat);
        
        free(imgcrd);
        free(phi);
        free(theta);
        free(stat);
        
        return pixels.to(world.dtype()).to(world.device());
    }
    
    torch::Tensor get_footprint() {
        double corners[4][2] = {
            {0.5, 0.5},
            {wcs_->crpix[0] * 2 - 0.5, 0.5},
            {wcs_->crpix[0] * 2 - 0.5, wcs_->crpix[1] * 2 - 0.5},
            {0.5, wcs_->crpix[1] * 2 - 0.5}
        };
        
        auto corner_pixels = torch::from_blob(corners, {4, 2}, torch::kFloat64);
        return pixel_to_world(corner_pixels);
    }

    int naxis() const { return wcs_->naxis; }
    torch::Tensor crpix() const { return torch::tensor({wcs_->crpix[0], wcs_->crpix[1]}, torch::kFloat64); }
    torch::Tensor crval() const { return torch::tensor({wcs_->crval[0], wcs_->crval[1]}, torch::kFloat64); }
    torch::Tensor cdelt() const { return torch::tensor({wcs_->cdelt[0], wcs_->cdelt[1]}, torch::kFloat64); }

private:
    struct wcsprm* wcs_;
};

}