#pragma once
#include "torchfits_torch.h"
#include <string>

// Parallel Rice Decompression
// Reads a Rice-compressed image HDU directly from file (mmap) 
// using multiple threads, bypassing CFITSIO's internal sequential processing.
torch::Tensor read_rice_parallel(const std::string& path, int hdu, int num_threads = -1);
