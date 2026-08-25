#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <array>
#include <tuple>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <ATen/ATen.h>
#include <fitsio.h>

#include "torchfits_torch.h"

namespace torchfits {
namespace detail {
struct SharedReadMeta;
struct RawFdHolder;
}

class FITSFile {
public:
    struct ScaleInfo {
        bool scaled = false;
        bool trusted = true;
        double bscale = 1.0;
        double bzero = 0.0;
    };

    FITSFile(const char* filename, int mode);
    ~FITSFile();
    void close();

    fitsfile* get_fptr() const { return fptr_; }
    int get_start_hdu() const { return start_hdu_; }

    void ensure_hdu(int hdu_num, int* status);

    const ScaleInfo& get_scale_info(int hdu_num, int bitpix);
    ScaleInfo get_scale_info_for_hdu(int hdu_num);
    bool is_compressed_image_cached(int hdu_num);
    const std::tuple<int, int, std::array<LONGLONG, 9>>& get_image_info(int hdu_num);

    torch::Tensor read_tensor(int hdu_num, bool use_mmap = true);
    torch::Tensor read_image_raw(int hdu_num, bool use_mmap = true);
    bool write_image(nb::ndarray<> tensor, int hdu_num, double bscale, double bzero);
    std::vector<std::tuple<std::string, std::string, std::string>> get_header(int hdu_num);
    std::vector<long> get_shape(int hdu_num);
    int get_dtype(int hdu_num);
    torch::Tensor read_subset(int hdu_num, long x1, long y1, long x2, long y2);
    int get_num_hdus();
    std::string get_hdu_type(int hdu_num);
    bool write_hdus(nb::list hdus, bool overwrite);
    bool write_hdus_compressed_images(nb::list hdus, int compression_type);
    std::string read_header_to_string(int hdu_num);

private:
    bool ensure_raw_fd(size_t required_end);
    void close_raw_fd();
    std::string filename_;
    int mode_;
    fitsfile* fptr_ = nullptr;
    int start_hdu_ = 1;
    int current_hdu_ = 1;
    int raw_fd_ = -1;
    off_t raw_file_size_ = 0;
    bool raw_fd_ready_ = false;
    std::unordered_map<int, ScaleInfo> scale_cache_;
    std::unordered_map<int, bool> compressed_cache_;
    std::unordered_map<int, std::tuple<int, int, std::array<LONGLONG, 9>>> image_info_cache_;
    std::shared_ptr<detail::SharedReadMeta> shared_meta_;
};

class SubsetReader {
public:
    SubsetReader(const std::string& filename, int hdu_num = 0);
    ~SubsetReader();
    torch::Tensor read(long x1, long y1, long x2, long y2);
    void close();
    long width() const { return max_x_; }
    long height() const { return max_y_; }
    int hdu() const { return hdu_num_; }
private:
    void init_from_hdu();
    bool ensure_data_mmap();
    bool try_read_via_mmap(long x1, long y1, long x2, long y2, torch::Tensor& out);
    void release_data_mmap();
    FITSFile file_;
    std::string filename_;
    int hdu_num_ = 0;
    long max_x_ = 0;
    long max_y_ = 0;
    int naxis_ = 0;
    std::vector<long> naxes_;
    torch::ScalarType dtype_ = torch::kFloat32;
    int datatype_ = TFLOAT;
    int bitpix_ = FLOAT_IMG;
    LONGLONG data_offset_ = 0;
    size_t elem_bytes_ = 0;
    bool raw_fast_ok_ = false;
    bool closed_ = false;
    // FITS integer conventions applied after mmap endian swap (0 = none).
    // signed-byte: XOR 0x80; uint16: +32768; uint32: +2147483648.
    enum class MmapConv : uint8_t { None, SignedByte, UInt16, UInt32 };
    MmapConv mmap_conv_ = MmapConv::None;
    // Uncompressed 2D mosaic fast path: map data once, slice+bswap per cutout.
    void* map_ptr_ = nullptr;
    size_t map_len_ = 0;
    off_t map_page_offset_ = 0;
    const uint8_t* pixel_base_ = nullptr;
    // Refcounted fd keeps the backing file descriptor alive while mapped and
    // guards against invalidation closing it mid-mmap.
    std::shared_ptr<detail::RawFdHolder> raw_fd_holder_;
};

} // namespace torchfits
