// Highway-vectorized CPU RMSNorm. This file is compiled once per SIMD target
// via foreach_target.h; the kernels live in per-target namespaces and the
// HWY_ONCE section below defines the public entry that dispatches to the
// best target at runtime. All of it stays inside the CPU backend.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "ops/cpu/rms_norm.cc"
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

#include "ops/cpu/rms_norm.h"

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

/// \brief Half-precision path: convert each row to fp32, normalize with the
///        vector kernel, convert back.
Status RmsNormConverted(const uint16_t* x, const uint16_t* w, uint16_t* out, int64_t rows,
                        int64_t dim, float eps, float bias, bool round_before_weight,
                        float (*decode)(uint16_t),
                        uint16_t (*encode)(float)) {
  std::vector<float> w_row(dim), x_row(dim), out_row(dim);
  for (int64_t j = 0; j < dim; ++j) w_row[j] = round_before_weight ? 1.0f : decode(w[j]);
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* x_src = x + r * dim;
    for (int64_t j = 0; j < dim; ++j) x_row[j] = decode(x_src[j]);
    HWY_DYNAMIC_DISPATCH(RmsNormRowF32)
    (x_row.data(), w_row.data(), out_row.data(), static_cast<size_t>(dim), eps,
     round_before_weight ? 0.0f : bias);
    uint16_t* out_dst = out + r * dim;
    for (int64_t j = 0; j < dim; ++j) {
      out_dst[j] = round_before_weight
                       ? encode(decode(encode(out_row[j])) * (decode(w[j]) + bias))
                       : encode(out_row[j]);
    }
  }
  return OkStatus();
}

}  // namespace

Status RmsNorm(ExecutionContext& /*ctx*/, const Tensor& x, const Tensor& weight, Tensor& out,
               const RMSNormConfig& config) {
  // Host memory is synchronous: the op completes before returning.
  const int64_t rows = x.Dim(0);
  const int64_t dim = x.Dim(1);
  const float bias = config.plus_one_weight ? 1.0f : 0.0f;
  switch (x.GetDataType()) {
    case DataType::kFloat:
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
                              config.round_before_weight,
                              F16BitsToFloat, FloatToF16Bits);
    case DataType::kBFloat16:
      return RmsNormConverted(static_cast<const uint16_t*>(x.Data()),
                              static_cast<const uint16_t*>(weight.Data()),
                              static_cast<uint16_t*>(out.Data()), rows, dim, config.eps, bias,
                              config.round_before_weight,
                              Bf16BitsToFloat, FloatToBf16Bits);
    default:
      return UnimplementedError("RmsNorm on CPU does not support dtype ",
                                DataTypeName(x.GetDataType()));
  }
}

}  // namespace cpu
}  // namespace ops
}  // namespace inferx

#endif  // HWY_ONCE
