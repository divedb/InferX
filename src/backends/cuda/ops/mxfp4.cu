#include "inferx/ops/mxfp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/backends/cuda/cuda_utils.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kBlock = 256;

// Values per block and bytes per block. Both are properties of MXFP4 rather
// than tuning knobs, which is why they are named rather than written as 32
// and 16 at the four places they appear.
constexpr int kValuesPerBlock = 32;
constexpr int kBytesPerBlock = 16;

// FP4 E2M1, in nibble order: sign in bit 3, then a 2-bit exponent and a 1-bit
// mantissa. Written as a table because that is what it is -- sixteen values --
// and because decoding the fields arithmetically would be slower and no clearer.
__device__ __constant__ float kFp4Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// One block per source row. `dest_row` is computed by the caller's mapping so
// that the de-interleaving variant costs nothing extra: the kernel has to pick
// a destination row regardless, and picking a different one is free.
//
// `deinterleave` splits even/odd source rows into the two halves of the output.
// A branch rather than a template because it is uniform across the whole grid
// and the kernel is bandwidth-bound anyway.
__global__ void DequantizeMxfp4Kernel(const uint8_t* __restrict__ blocks,
                                      const uint8_t* __restrict__ scales,
                                      bf16* __restrict__ out, int64_t rows,
                                      int64_t num_blocks, bool deinterleave) {
  const int64_t src_row = blockIdx.x;

  int64_t dest_row = src_row;
  if (deinterleave) {
    const int64_t half = rows / 2;
    dest_row = (src_row & 1) ? half + (src_row >> 1) : (src_row >> 1);
  }

  const uint8_t* src = blocks + src_row * num_blocks * kBytesPerBlock;
  const uint8_t* row_scales = scales + src_row * num_blocks;
  bf16* dst = out + dest_row * num_blocks * kValuesPerBlock;

  // One thread per *byte*: each decodes the two values packed into it, which
  // keeps the two writes adjacent and the read coalesced.
  const int64_t total_bytes = num_blocks * kBytesPerBlock;

  for (int64_t i = threadIdx.x; i < total_bytes; i += blockDim.x) {
    const uint8_t byte = src[i];

    // E8M0: the stored byte is an exponent with bias 127, so the multiplier is
    // 2^(scale - 127). ldexpf rather than a powf or a float multiply -- it is
    // an exponent add, and it keeps the result exactly representable.
    const int exponent = static_cast<int>(row_scales[i / kBytesPerBlock]) - 127;

    // Low nibble first. This is the line the golden test exists to pin.
    const float lo = ldexpf(kFp4Lut[byte & 0x0F], exponent);
    const float hi = ldexpf(kFp4Lut[byte >> 4], exponent);

    dst[2 * i] = __float2bfloat16(lo);
    dst[2 * i + 1] = __float2bfloat16(hi);
  }
}

Status CheckPacked(const TensorView& blocks, const TensorView& scales,
                   const TensorView& out, int64_t* rows, int64_t* num_blocks) {
  for (const auto& [t, name, dtype, rank] :
       {std::tuple{&blocks, "blocks", DataType::kUInt8, 3},
        std::tuple{&scales, "scales", DataType::kUInt8, 2},
        std::tuple{&out, "out", DataType::kBFloat16, 2}}) {
    if (!t->IsDefined()) return InvalidArgumentError(name, " is undefined");

    if (!t->IsCuda()) {
      return InvalidArgumentError(name, " is on ", t->Device().ToString(),
                                  ", not a CUDA device");
    }

    if (t->GetDataType() != dtype) {
      return InvalidArgumentError(name, " is ", DataTypeName(t->GetDataType()),
                                  ", expected ", DataTypeName(dtype));
    }

    if (t->Rank() != rank) {
      return InvalidArgumentError(name, " has rank ", t->Rank(), ", expected ",
                                  rank);
    }
  }

  *rows = blocks.Dim(0);
  *num_blocks = blocks.Dim(1);

  if (blocks.Dim(2) != kBytesPerBlock) {
    return InvalidArgumentError("blocks' last dimension is ", blocks.Dim(2),
                                ", expected ", kBytesPerBlock,
                                " bytes per MXFP4 block");
  }

  if (scales.Dim(0) != *rows || scales.Dim(1) != *num_blocks) {
    return InvalidArgumentError("scales is ", scales.GetShape().ToString(),
                                " but blocks needs one scale per block: [",
                                *rows, ", ", *num_blocks, "]");
  }

  if (out.Dim(0) != *rows || out.Dim(1) != *num_blocks * kValuesPerBlock) {
    return InvalidArgumentError("out is ", out.GetShape().ToString(),
                                ", expected [", *rows, ", ",
                                *num_blocks * kValuesPerBlock, "]");
  }

  return OkStatus();
}

Status Launch(const TensorView& blocks, const TensorView& scales,
              const TensorView& out, bool deinterleave, Stream stream) {
  int64_t rows = 0;
  int64_t num_blocks = 0;
  INFERX_RETURN_IF_ERROR(CheckPacked(blocks, scales, out, &rows, &num_blocks));

  if (deinterleave && rows % 2 != 0) {
    return InvalidArgumentError(
        "a gate_up weight has ", rows,
        " rows, which is odd; gate and up interleave in pairs");
  }

  if (rows == 0 || num_blocks == 0) return OkStatus();

  DequantizeMxfp4Kernel<<<static_cast<int>(rows), kBlock, 0, stream>>>(
      static_cast<const uint8_t*>(blocks.Data()),
      static_cast<const uint8_t*>(scales.Data()),
      static_cast<bf16*>(out.Data()), rows, num_blocks, deinterleave);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace

Status DequantizeMxfp4ToBf16(const TensorView& blocks, const TensorView& scales,
                             const TensorView& out, Stream stream) {
  return Launch(blocks, scales, out, /*deinterleave=*/false, stream);
}

Status DequantizeMxfp4GateUpToBf16(const TensorView& blocks,
                                   const TensorView& scales,
                                   const TensorView& out,
                                   Stream stream) {
  return Launch(blocks, scales, out, /*deinterleave=*/true, stream);
}

}  // namespace inferx::kernels
