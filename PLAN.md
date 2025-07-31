# TorchFits v0.2 Implementation Plan

## Core Objectives
1. **Performance**: Match or exceed fitsio speed (building on v0.1's 2.3x gains)
2. **DataFrame-friendly**: Native PyTorch table operations with optional PyTorch-Frame integration
3. **Remote + Cache**: Robust remote file access with intelligent caching

---

## Phase 1: Enhanced Table Operations (Pure PyTorch)

### Goal: Make FITS tables first-class PyTorch citizens without external dependencies

#### 1.1 Enhanced Table API
```python
# Current v0.1 returns dict of tensors
table_data = torchfits.read("catalog.fits", hdu="CATALOG")
# {'RA': tensor([...]), 'DEC': tensor([...]), 'MAG_G': tensor([...])}

# New v0.2 enhanced table operations (pure PyTorch)
class FitsTable:
    def __init__(self, data_dict: Dict[str, torch.Tensor], metadata: Dict):
        self.data = data_dict
        self.columns = list(data_dict.keys())
        self.metadata = metadata
        
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]  # Column access
        elif isinstance(key, slice):
            return self.slice_rows(key)  # Row slicing
        elif isinstance(key, torch.Tensor):
            return self.filter(key)  # Boolean indexing
            
    def to(self, device):
        """Move all columns to device"""
        return FitsTable({k: v.to(device) for k, v in self.data.items()}, self.metadata)
        
    def select(self, columns):
        """Select specific columns"""
        return FitsTable({k: self.data[k] for k in columns}, self.metadata)
        
    def filter(self, mask):
        """Filter rows using boolean mask"""
        return FitsTable({k: v[mask] for k, v in self.data.items()}, self.metadata)
        
    def sort(self, column, descending=False):
        """Sort by column"""
        indices = torch.argsort(self.data[column], descending=descending)
        return FitsTable({k: v[indices] for k, v in self.data.items()}, self.metadata)
        
    def groupby(self, column):
        """Simple groupby operations"""
        # Return GroupedTable for aggregations
        
    @property
    def shape(self):
        first_col = next(iter(self.data.values()))
        return (first_col.shape[0], len(self.columns))
```

#### 1.2 Enhanced Read Function
```python
def read(filename_or_url, 
         hdu=1, 
         start=None, 
         shape=None, 
         columns=None,
         start_row=0,
         num_rows=None,
         cache_capacity=0,
         device='cpu',
         # NEW: Enhanced table options
         format='auto',           # 'tensor', 'table', 'dataframe' (if torch_frame available)
         return_metadata=False,   # Include column metadata
         **kwargs):
    
    # Auto-detect format based on HDU type
    if format == 'auto':
        hdu_type = get_hdu_type(filename_or_url, hdu)
        if hdu_type in ['BINTABLE', 'TABLE']:
            format = 'table'  # Return FitsTable
        else:
            format = 'tensor'  # Return tensor + header
```

#### 1.3 C++ Optimizations for Tables
```cpp
// Enhanced table reading with metadata preservation
struct ColumnMetadata {
    std::string name;
    std::string unit;
    std::string description;
    torch::Dtype dtype;
    std::vector<std::string> categorical_values;  // For enum columns
};

py::object read_table_enhanced(fitsfile* fptr, torch::Device device,
                             const py::object& columns_obj,
                             long start_row, const py::object& num_rows_obj,
                             bool return_metadata) {
    // ... existing optimized reading code ...
    
    if (return_metadata) {
        py::dict metadata;
        for (const auto& meta : column_metadata) {
            py::dict col_meta;
            col_meta["unit"] = meta.unit;
            col_meta["description"] = meta.description;
            col_meta["dtype"] = py::cast(meta.dtype);
            metadata[meta.name] = col_meta;
        }
        
        return py::make_tuple(result_dict, metadata);
    }
    
    return result_dict;
}
```

---

## Phase 2: Integrate Remote File Support

### Goal: Seamless remote file access with intelligent caching

#### 2.1 Integrate Existing Remote Code
```cpp
// In fits_reader.cpp - modify read_impl()
pybind11::object read_impl(
    pybind11::object filename_or_url,
    // ... other params ...
) {
    std::string filename;
    
    // Handle remote URLs and fsspec dicts
    if (py::isinstance<py::dict>(filename_or_url)) {
        // Convert fsspec dict to URL
        filename = RemoteFetcher::fsspec_to_url(filename_or_url.cast<py::dict>());
        filename = RemoteFetcher::ensure_local(filename);
    } else {
        std::string url_str = py::str(filename_or_url).cast<std::string>();
        filename = RemoteFetcher::ensure_local(url_str);
    }
    
    // Continue with existing local file processing...
    FITSFileWrapper f(filename);
    // ... rest of function unchanged ...
}
```

#### 2.2 Enhanced Remote Fetcher
```cpp
class RemoteFetcher {
public:
    // Convert fsspec dict to URL
    static std::string fsspec_to_url(const py::dict& fsspec_params) {
        std::string protocol = fsspec_params["protocol"].cast<std::string>();
        
        if (protocol == "s3") {
            return build_s3_url(fsspec_params);
        } else if (protocol == "gs") {
            return build_gcs_url(fsspec_params);
        } else if (protocol == "https" || protocol == "http") {
            return build_http_url(fsspec_params);
        }
        // ... other protocols
    }
    
    // Enhanced caching with metadata
    static std::string ensure_local_with_cache(const std::string& url, 
                                             const CacheOptions& options = {}) {
        std::string cache_key = generate_cache_key(url, options);
        
        if (is_cached_and_valid(cache_key)) {
            return get_cached_path(cache_key);
        }
        
        std::string local_path = download_with_progress(url, options);
        update_cache_metadata(cache_key, local_path);
        return local_path;
    }
};
```

#### 2.3 Smart Caching Strategy
```cpp
struct CacheOptions {
    bool headers_only = false;      // Download headers first
    bool use_range_requests = true; // HTTP range requests for cutouts
    size_t chunk_size = 10 * 1024 * 1024; // 10MB chunks
    int timeout_seconds = 300;
    bool verify_checksum = true;
};

class SmartCache {
    // Cache headers separately for quick metadata access
    std::map<std::string, std::string> get_headers_fast(const std::string& url, int hdu);
    
    // Range request for specific data regions
    std::string get_region(const std::string& url, 
                          const std::vector<long>& start,
                          const std::vector<long>& shape);
                          
    // Progressive download strategy
    void prefetch_likely_data(const std::string& url, const AccessPattern& pattern);
};
```

---

## Phase 3: Performance Optimizations

### Goal: Exceed fitsio performance across all operations

#### 3.1 Parallel Table Reading
```cpp
class ParallelTableReader {
private:
    std::vector<std::thread> worker_threads;
    std::queue<ColumnTask> task_queue;
    std::mutex queue_mutex;
    
public:
    py::dict read_columns_parallel(fitsfile* fptr, 
                                 const std::vector<std::string>& columns,
                                 torch::Device device) {
        // Distribute columns across threads
        // Use thread-safe CFITSIO operations
        // Merge results efficiently
    }
};
```

#### 3.2 GPU-Direct Transfers
```cpp
// Direct GPU memory allocation for large tensors
torch::Tensor allocate_gpu_tensor_direct(const std::vector<long>& shape, 
                                        torch::Dtype dtype,
                                        torch::Device device) {
    if (device.is_cuda()) {
        // Allocate directly on GPU
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        return torch::empty(shape, options);
    }
    // CPU allocation with later transfer
    return torch::empty(shape, torch::TensorOptions().dtype(dtype));
}
```

#### 3.3 Memory Pool for Frequent Operations
```cpp
class TensorMemoryPool {
private:
    std::unordered_map<std::string, std::queue<torch::Tensor>> pools;
    std::mutex pool_mutex;
    
public:
    torch::Tensor get_tensor(const std::vector<long>& shape, torch::Dtype dtype) {
        std::string key = shape_dtype_key(shape, dtype);
        
        std::lock_guard<std::mutex> lock(pool_mutex);
        if (!pools[key].empty()) {
            auto tensor = pools[key].front();
            pools[key].pop();
            return tensor;
        }
        
        return torch::empty(shape, torch::TensorOptions().dtype(dtype));
    }
    
    void return_tensor(torch::Tensor tensor) {
        // Return to pool for reuse
    }
};
```

---

## Phase 4: Optional PyTorch-Frame Integration

### Goal: Seamless upgrade path to PyTorch-Frame without breaking existing code

#### 4.1 Conditional Import and API
```python
# In __init__.py
try:
    import torch_frame
    _TORCH_FRAME_AVAILABLE = True
except ImportError:
    _TORCH_FRAME_AVAILABLE = False

def read_dataframe(filename_or_url, **kwargs):
    """Read FITS table as PyTorch-Frame DataFrame (optional dependency)"""
    if not _TORCH_FRAME_AVAILABLE:
        raise ImportError("PyTorch-Frame is required for dataframe functionality. "
                         "Install with: pip install pytorch-frame")
    
    # Read as FitsTable first
    table = read(filename_or_url, format='table', **kwargs)
    
    # Convert to PyTorch-Frame format
    return _fits_table_to_torch_frame(table)

def _fits_table_to_torch_frame(fits_table):
    """Convert FitsTable to torch_frame.DataFrame"""
    # Map FITS column types to torch_frame semantic types
    stype_mapping = {
        'RA': torch_frame.stype.numerical,
        'DEC': torch_frame.stype.numerical, 
        'MAG_': torch_frame.stype.numerical,  # Any magnitude column
        'TYPE': torch_frame.stype.categorical,
        # ... astronomy-specific mappings
    }
    
    # Create torch_frame.DataFrame with proper semantic types
```

#### 4.2 Astronomy-Specific Enhancements
```python
# Optional astronomy utilities when torch_frame is available
if _TORCH_FRAME_AVAILABLE:
    def register_astronomy_stypes():
        """Register astronomy-specific semantic types"""
        
        @torch_frame.stype_register('coordinate')
        class CoordinateType(torch_frame.SType):
            """RA/Dec coordinates with proper handling"""
            
        @torch_frame.stype_register('magnitude') 
        class MagnitudeType(torch_frame.SType):
            """Astronomical magnitudes with color calculations"""
            
        @torch_frame.stype_register('redshift')
        class RedshiftType(torch_frame.SType):
            """Cosmological redshift with distance calculations"""
```

---

## Implementation Sequence (Agent Actions)

### Week 1: Core Table Enhancements
1. **Implement FitsTable class** in Python with PyTorch operations
2. **Enhance C++ table reading** to include metadata extraction
3. **Update read() function** to support format parameter
4. **Add comprehensive tests** for table operations

### Week 2: Remote Integration
1. **Integrate remote.cpp** into main read path
2. **Add fsspec dict parsing** in C++
3. **Implement smart caching** with metadata tracking
4. **Add range request support** for HTTP downloads

### Week 3: Performance Optimizations
1. **Implement parallel table reading** with thread pool
2. **Add GPU-direct memory transfers** for CUDA tensors
3. **Create memory pool system** for frequent allocations
4. **Benchmark and tune** all optimizations

### Week 4: Optional PyTorch-Frame + Polish
1. **Add conditional PyTorch-Frame integration**
2. **Create astronomy-specific semantic types**
3. **Comprehensive testing** with real datasets
4. **Documentation and examples** for all new features

---

## Success Metrics

### Performance Targets
- **Tables**: 2x faster than v0.1 (total 4.6x vs original), competitive with fitsio
- **Images**: 3x faster than v0.1 (total 7-9x vs original)
- **Remote files**: <15% overhead vs local (cached), <5 sec first access
- **Memory efficiency**: 40% reduction in peak usage via pooling

### API Goals
- **Backward compatible**: All v0.1 code continues to work
- **DataFrame friendly**: Rich table operations with pure PyTorch
- **Optional dependencies**: PyTorch-Frame enhances but isn't required
- **Remote transparent**: URL/fsspec works exactly like local files

### Robustness
- **Network resilience**: Automatic retry, graceful degradation
- **Cache reliability**: Checksums, corruption detection, cleanup
- **Error handling**: Clear messages, proper exception types
- **Memory safety**: No leaks, proper resource cleanup

This plan delivers on all three core objectives while maintaining torchfits' philosophy of PyTorch-native astronomy data access with optional advanced features.
