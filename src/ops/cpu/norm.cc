// Highway-vectorized CPU norms. This file is compiled once per SIMD target
// via foreach_target.h; the kernels live in per-target namespaces and the
// HWY_ONCE section below defines the public entries that dispatch to the
// best target at runtime. All of it stays inside the CPU backend.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/cpu/norm.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace inferx {
namespace ops {
namespace cpu {

namespace HWY_NAMESPACE {

/// \brief Normalizes one fp32 row into `out`.
///
/// `bias` is 0 for the standard scale-by-weight form and 1 for Gemma's
/// (1 + weight). Loads beyond `dim` never happen: vector blocks stop at the
/// alignment boundary and LoadN/StoreN cover the tail.
void RmsNormRowF32(const float* x, const float* w, float* out, size_t dim, float eps,
                   float bias) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;
  const size_t lanes = hn::Lanes(d);

  V sum_sq = hn::Zero(d);
  size_t i = 0;
  for (; i + lanes <= dim; i += lanes) {
    const V v = hn::LoadU(d, x + i);
    sum_sq = hn::MulAdd(v, v, sum_sq);
  }
  if (i < dim) {
    const V v = hn::LoadN(d, x + i, dim - i);
    sum_sq = hn::MulAdd(v, v, sum_sq);
  }
  const float mean_sq = hn::ReduceSum(d, sum_sq) / static_cast<float>(dim);
  const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

  const V scale = hn::Set(d, inv_rms);
  const V scale_bias = hn::Set(d, bias);
  for (i = 0; i + lanes <= dim; i += lanes) {
    const V v = hn::Mul(hn::LoadU(d, x + i), scale);
    const V wv = hn::Add(hn::LoadU(d, w + i), scale_bias);
    hn::StoreU(hn::Mul(v, wv), d, out + i);
  }
  if (i < dim) {
    const V v = hn::Mul(hn::LoadN(d, x + i, dim - i), scale);
    const V wv = hn::Add(hn::LoadN(d, w + i, dim - i), scale_bias);
    hn::StoreN(hn::Mul(v, wv), d, out + i, dim - i);
  }
}

}  // namespace HWY_NAMESPACE

}  // namespace cpu
}  // namespace ops
}  // namespace inferx
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

#include <optional>

#include "inferx/ops/cpu/norm.h"

namespace inferx {
namespace ops {
namespace cpu {

// The dispatch table must be generated in the same namespace as the
// per-target kernels above, and before the helpers that call it.
HWY_EXPORT(RmsNormRowF32);

namespace {

static_assert(sizeof(_Float16) == 2, "host fp16 conversions need 16-bit _Float16");

float Bf16BitsToFloat(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

/// \brief Rounds an fp32 to the nearest bfloat16 bit pattern.
uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  const uint32_t rounding = 0x7fffu + lsb;
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

float F16BitsToFloat(uint16_t h) {
  _Float16 v = 0;
  std::memcpy(&v, &h, sizeof(v));
  return static_cast<float>(v);
}

uint16_t FloatToF16Bits(float value) {
  const _Float16 v = static_cast<_Float16>(value);
  uint16_t h = 0;
  std::memcpy(&h, &v, sizeof(h));
  return h;
}

/// \brief Half-precision RMS path: convert each row to fp32, normalize with
///        the vector kernel, convert back. When `residual` is set, each row
///        first accumulates x + residual with a single rounding into
///        `residual`, and the normalization reads the rounded sum.
Status RmsNormConverted(const uint16_t* x, const uint16_t* w, uint16_t* out, int64_t rows,
                        int64_t dim, float eps, float bias, float (*decode)(uint16_t),
                        uint16_t (*encode)(float), uint16_t* residual = nullptr,
                        bool round_before_weight = false) {
  std::vector<float> w_row(dim), x_row(dim), out_row(dim);
  for (int64_t j = 0; j < dim; ++j) w_row[j] = decode(w[j]);
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* x_src = x + r * dim;
    uint16_t* res_dst = residual != nullptr ? residual + r * dim : nullptr;
    if (res_dst != nullptr) {
      for (int64_t j = 0; j < dim; ++j) {
        res_dst[j] = encode(decode(x_src[j]) + decode(res_dst[j]));
        x_row[j] = decode(res_dst[j]);
      }
    } else {
      for (int64_t j = 0; j < dim; ++j) x_row[j] = decode(x_src[j]);
    }
    uint16_t* out_dst = out + r * dim;
    if (round_before_weight) {
      // Reference rounding: the normalized activation rounds to the input
      // dtype before the scale multiply, and that product rounds again.
      float sum_sq = 0.0f;
      for (int64_t j = 0; j < dim; ++j) sum_sq += x_row[j] * x_row[j];
      const float inv = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
      for (int64_t j = 0; j < dim; ++j) {
        const float normalized = decode(encode(x_row[j] * inv));
        out_dst[j] = encode(normalized * (w_row[j] + bias));
      }
      continue;
    }
    HWY_DYNAMIC_DISPATCH(RmsNormRowF32)
    (x_row.data(), w_row.data(), out_row.data(), static_cast<size_t>(dim), eps, bias);
    for (int64_t j = 0; j < dim; ++j) out_dst[j] = encode(out_row[j]);
  }
  return OkStatus();
}

/// \brief Scalar two-pass fp32 LayerNorm over one row.
///
/// The CPU backend is the accuracy fallback, so moments accumulate in double;
/// the fused CUDA kernel stays fp32 end to end.
void LayerNormRowF32(const float* x, const float* w, const float* bias, float* out, int64_t dim,
                     float eps) {
  double sum = 0.0;
  for (int64_t j = 0; j < dim; ++j) sum += x[j];
  const double mean = sum / static_cast<double>(dim);
  double sum_sq = 0.0;
  for (int64_t j = 0; j < dim; ++j) {
    const double d = x[j] - mean;
    sum_sq += d * d;
  }
  const float inv_std =
      1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(dim)) + eps);
  for (int64_t j = 0; j < dim; ++j) {
    float y = static_cast<float>(x[j] - mean) * inv_std * w[j];
    if (bias != nullptr) y += bias[j];
    out[j] = y;
  }
}

/// \brief Half-precision Layer path: convert each row to fp32, normalize,
///        convert back.
Status LayerNormConverted(const uint16_t* x, const uint16_t* w, const uint16_t* b, uint16_t* out,
                          int64_t rows, int64_t dim, float eps, float (*decode)(uint16_t),
                          uint16_t (*encode)(float)) {
  std::vector<float> w_row(dim), x_row(dim), out_row(dim);
  std::vector<float> bias_row;
  const float* bias = nullptr;
  if (b != nullptr) {
    bias_row.resize(dim);
    for (int64_t j = 0; j < dim; ++j) bias_row[j] = decode(b[j]);
    bias = bias_row.data();
  }
  for (int64_t j = 0; j < dim; ++j) w_row[j] = decode(w[j]);
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* x_src = x + r * dim;
    for (int64_t j = 0; j < dim; ++j) x_row[j] = decode(x_src[j]);
    LayerNormRowF32(x_row.data(), w_row.data(), bias, out_row.data(), dim, eps);
    uint16_t* out_dst = out + r * dim;
    for (int64_t j = 0; j < dim; ++j) out_dst[j] = encode(out_row[j]);
  }
  return OkStatus();
}

}  // namespace

Status RmsNorm(OpContext& /*ctx*/, const Tensor& x, const Tensor& weight, Tensor& out,
               const NormConfig& config) {
  // Host memory is synchronous: the op completes before returning.
  const int64_t rows = x.Dim(0);
  const int64_t dim = x.Dim(1);
  const float bias = config.plus_one ? 1.0f : 0.0f;
  switch (x.GetDataType()) {
    case DataType::kFloat32:
      for (int64_t r = 0; r < rows; ++r) {
        HWY_DYNAMIC_DISPATCH(RmsNormRowF32)
        (static_cast<const float*>(x.Data()) + r * dim,
         static_cast<const float*>(weight.Data()), static_cast<float*>(out.Data()) + r * dim,
         static_cast<size_t>(dim), config.eps, bias);
      }
      return OkStatus();
    case DataType::kFloat16:
      return RmsNormConverted(static_cast<const uint16_t*>(x.Data()),
                              static_cast<const uint16_t*>(weight.Data()),
                              static_cast<uint16_t*>(out.Data()), rows, dim, config.eps, bias,
                              F16BitsToFloat, FloatToF16Bits, nullptr,
                              config.round_before_weight);
    case DataType::kBFloat16:
      return RmsNormConverted(static_cast<const uint16_t*>(x.Data()),
                              static_cast<const uint16_t*>(weight.Data()),
                              static_cast<uint16_t*>(out.Data()), rows, dim, config.eps, bias,
                              Bf16BitsToFloat, FloatToBf16Bits, nullptr,
                              config.round_before_weight);
    default:
      return UnimplementedError("RmsNorm on CPU does not support dtype ",
                                DataTypeName(x.GetDataType()));
  }
}

Status AddRmsNorm(OpContext& ctx, const Tensor& x, Tensor& residual, const Tensor& weight,
                  Tensor& out, const NormConfig& config) {
  const int64_t rows = x.Dim(0);
  const int64_t dim = x.Dim(1);
  const float bias = config.plus_one ? 1.0f : 0.0f;
  float* res_f32 = static_cast<float*>(residual.Data());
  const float* x_f32 = static_cast<const float*>(x.Data());
  switch (x.GetDataType()) {
    case DataType::kFloat32:
      for (int64_t r = 0; r < rows; ++r) {
        float* row = res_f32 + r * dim;
        for (int64_t j = 0; j < dim; ++j) row[j] += x_f32[r * dim + j];
        HWY_DYNAMIC_DISPATCH(RmsNormRowF32)
        (row, static_cast<const float*>(weight.Data()),
         static_cast<float*>(out.Data()) + r * dim, static_cast<size_t>(dim), config.eps, bias);
      }
      return OkStatus();
    case DataType::kFloat16:
      return RmsNormConverted(static_cast<const uint16_t*>(x.Data()),
                              static_cast<const uint16_t*>(weight.Data()),
                              static_cast<uint16_t*>(out.Data()), rows, dim, config.eps, bias,
                              F16BitsToFloat, FloatToF16Bits,
                              static_cast<uint16_t*>(residual.Data()),
                              config.round_before_weight);
    case DataType::kBFloat16:
      return RmsNormConverted(static_cast<const uint16_t*>(x.Data()),
                              static_cast<const uint16_t*>(weight.Data()),
                              static_cast<uint16_t*>(out.Data()), rows, dim, config.eps, bias,
                              Bf16BitsToFloat, FloatToBf16Bits,
                              static_cast<uint16_t*>(residual.Data()),
                              config.round_before_weight);
    default:
      return UnimplementedError("AddRmsNorm on CPU does not support dtype ",
                                DataTypeName(x.GetDataType()));
  }
}

Status LayerNorm(OpContext& /*ctx*/, const Tensor& x, const Tensor& weight,
                 const std::optional<Tensor>& bias, Tensor& out, const NormConfig& config) {
  const int64_t rows = x.Dim(0);
  const int64_t dim = x.Dim(1);
  const uint16_t* bias_bits =
      bias ? static_cast<const uint16_t*>(bias->Data()) : nullptr;
  switch (x.GetDataType()) {
    case DataType::kFloat32:
      for (int64_t r = 0; r < rows; ++r) {
        LayerNormRowF32(static_cast<const float*>(x.Data()) + r * dim,
                        static_cast<const float*>(weight.Data()),
                        bias ? static_cast<const float*>(bias->Data()) : nullptr,
                        static_cast<float*>(out.Data()) + r * dim, dim, config.eps);
      }
      return OkStatus();
    case DataType::kFloat16:
      return LayerNormConverted(static_cast<const uint16_t*>(x.Data()),
                                static_cast<const uint16_t*>(weight.Data()), bias_bits,
                                static_cast<uint16_t*>(out.Data()), rows, dim, config.eps,
                                F16BitsToFloat, FloatToF16Bits);
    case DataType::kBFloat16:
      return LayerNormConverted(static_cast<const uint16_t*>(x.Data()),
                                static_cast<const uint16_t*>(weight.Data()), bias_bits,
                                static_cast<uint16_t*>(out.Data()), rows, dim, config.eps,
                                Bf16BitsToFloat, FloatToBf16Bits);
    default:
      return UnimplementedError("LayerNorm on CPU does not support dtype ",
                                DataTypeName(x.GetDataType()));
  }
}

}  // namespace cpu
}  // namespace ops
}  // namespace inferx

#endif  // HWY_ONCE
