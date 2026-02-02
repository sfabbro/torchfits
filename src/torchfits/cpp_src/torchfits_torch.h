#pragma once

#if defined(__has_include)
#if __has_include(<torch/torch.h>)
#include <torch/torch.h>
#else
#include <ATen/ATen.h>

namespace torch {
using Tensor = at::Tensor;
using TensorOptions = at::TensorOptions;
using ScalarType = at::ScalarType;

using at::empty;
using at::empty_like;
using at::from_blob;

using at::kBool;
using at::kByte;
using at::kShort;
using at::kInt;
using at::kLong;
using at::kHalf;
using at::kFloat;
using at::kDouble;
using at::kBFloat16;

constexpr auto kUInt8 = at::kByte;
constexpr auto kInt16 = at::kShort;
constexpr auto kInt32 = at::kInt;
constexpr auto kInt64 = at::kLong;
constexpr auto kFloat32 = at::kFloat;
constexpr auto kFloat64 = at::kDouble;
} // namespace torch
#endif
#else
#include <torch/torch.h>
#endif
