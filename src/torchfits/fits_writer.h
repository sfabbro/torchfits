#pragma once
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <fitsio.h>
#include <string>
#include <vector>
#include <map>

namespace py = pybind11;

/// FITS writing functionality for TorchFits v1.0
namespace torchfits_writer {

/// Write a PyTorch tensor to a FITS file as an image HDU
void write_tensor_to_fits(const std::string& filename,
                         torch::Tensor data,
                         const std::map<std::string, std::string>& header = {},
                         bool overwrite = false);

/// Write multiple tensors to a multi-extension FITS file
void write_tensors_to_mef(const std::string& filename,
                         const std::vector<torch::Tensor>& tensors,
                         const std::vector<std::map<std::string, std::string>>& headers = {},
                         const std::vector<std::string>& extnames = {},
                         bool overwrite = false);

/// Write a dictionary of tensors (table data) to a FITS table
void write_table_to_fits(const std::string& filename,
                        const py::dict& table_data,
                        const std::map<std::string, std::string>& header = {},
                        const std::vector<std::string>& column_units = {},
                        const std::vector<std::string>& column_descriptions = {},
                        bool overwrite = false);

/// Write a FitsTable object to a FITS file
void write_fits_table(const std::string& filename,
                     const py::object& fits_table,
                     bool overwrite = false);

/// Append an HDU to an existing FITS file
void append_hdu_to_fits(const std::string& filename,
                       torch::Tensor data,
                       const std::map<std::string, std::string>& header = {},
                       const std::string& extname = "");

/// Update header keywords in an existing FITS file
void update_fits_header(const std::string& filename,
                       int hdu_num,
                       const std::map<std::string, std::string>& updates);

// === Phase 2: Enhanced Writing Capabilities ===

/// Compression options for FITS files
enum class CompressionType {
    None,
    GZIP,
    RICE,
    HCOMPRESS,
    PLIO
};

struct CompressionConfig {
    CompressionType type = CompressionType::None;
    int quantize_level = 0;      // For lossy compression (HCOMPRESS)
    float quantize_dither = 0.0f; // Dithering for quantization
    int tile_dimensions[2] = {0, 0}; // Tile size for tiled compression
    bool preserve_zeros = true;   // Preserve zero values in compression
};

/// Enhanced tensor writing with compression and advanced options
void write_tensor_to_fits_advanced(const std::string& filename,
                                  torch::Tensor data,
                                  const std::map<std::string, std::string>& header = {},
                                  const CompressionConfig& compression = {},
                                  bool overwrite = false,
                                  bool checksum = false);

/// Write tensor with variable-length array support
void write_variable_length_array(const std::string& filename,
                                const std::vector<torch::Tensor>& arrays,
                                const std::map<std::string, std::string>& header = {},
                                bool overwrite = false);

/// Advanced table writing with compression and optimizations
void write_table_to_fits_advanced(const std::string& filename,
                                 const py::dict& table_data,
                                 const std::map<std::string, std::string>& header = {},
                                 const std::vector<std::string>& column_units = {},
                                 const std::vector<std::string>& column_descriptions = {},
                                 const CompressionConfig& compression = {},
                                 bool overwrite = false,
                                 bool checksum = false);

/// Streaming writer for large datasets
class StreamingWriter {
public:
    StreamingWriter(const std::string& filename, 
                   const std::vector<long>& dimensions,
                   torch::ScalarType dtype = torch::kFloat32,
                   const CompressionConfig& compression = {},
                   bool overwrite = false);
    ~StreamingWriter();
    
    /// Write a chunk of data at specified position
    void write_chunk(const torch::Tensor& chunk, 
                    const std::vector<long>& start_position);
    
    /// Write data sequentially (streaming mode)
    void write_sequential(const torch::Tensor& data);
    
    /// Finalize the file (write headers, checksums, etc.)
    void finalize(const std::map<std::string, std::string>& header = {});
    
    /// Get current write position
    size_t get_position() const { return current_position_; }

private:
    std::string filename_;
    std::vector<long> dimensions_;
    torch::ScalarType dtype_;
    CompressionConfig compression_;
    size_t current_position_;
    fitsfile* fptr_;
    bool finalized_;
};

/// Parallel writer for multi-threaded writing
class ParallelWriter {
public:
    /// Write multiple tensors in parallel
    static void write_parallel_tensors(const std::string& filename,
                                      const std::vector<torch::Tensor>& tensors,
                                      const std::vector<std::map<std::string, std::string>>& headers = {},
                                      const std::vector<std::string>& extnames = {},
                                      size_t num_threads = 0,
                                      bool overwrite = false);
    
    /// Write large tensor in parallel chunks
    static void write_large_tensor_parallel(const std::string& filename,
                                           const torch::Tensor& data,
                                           const std::map<std::string, std::string>& header = {},
                                           const CompressionConfig& compression = {},
                                           size_t chunk_size = 1024 * 1024,
                                           size_t num_threads = 0,
                                           bool overwrite = false);
};

/// Format conversion utilities
class FormatConverter {
public:
    /// Convert between different FITS data types with optimizations
    static torch::Tensor convert_data_type(const torch::Tensor& input,
                                          torch::ScalarType target_type,
                                          bool preserve_range = true);
    
    /// Optimize data layout for FITS writing
    static torch::Tensor optimize_for_fits(const torch::Tensor& input);
    
    /// Convert PyTorch tensor to FITS-compatible format
    static torch::Tensor to_fits_format(const torch::Tensor& input,
                                       int& fits_type_out,
                                       int& bits_per_pixel_out);
};

/// Update data in an existing FITS file (in-place modification)
void update_fits_data(const std::string& filename,
                     int hdu_num,
                     torch::Tensor new_data,
                     const std::vector<long>& start = {},
                     const std::vector<long>& shape = {});

/// Get FITS data type code from PyTorch tensor
int get_fits_datatype(torch::Tensor tensor);

/// Get FITS table column format from PyTorch tensor
std::string get_fits_column_format(torch::Tensor tensor);

/// Helper to convert PyTorch tensor to FITS image
void tensor_to_fits_image(fitsfile* fptr, torch::Tensor tensor);

/// Helper to write header keywords to FITS file
void write_header_keywords(fitsfile* fptr, const std::map<std::string, std::string>& header);

} // namespace torchfits_writer
