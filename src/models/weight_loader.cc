#include "inferx/models/weight_loader.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "inferx/core/device_runtime.h"

namespace inferx {
namespace models {

WeightBinding WeightBinding::Replicate(Tensor* dest) {
  WeightBinding binding;
  binding.dest = dest;
  binding.spans.push_back({0, 0, dest->Rank() == 2 ? dest->Dim(0) : dest->Numel()});
  return binding;
}

WeightBinding WeightBinding::ShardRows(Tensor* dest, int64_t shard_rows, int64_t shard_index,
                                        int64_t dest_row) {
  WeightBinding binding;
  binding.dest = dest;
  binding.spans.push_back({shard_index * shard_rows, dest_row, shard_rows});
  return binding;
}

WeightBinding WeightBinding::ShardCols(Tensor* dest, int64_t shard_cols, int64_t shard_index) {
  WeightBinding binding;
  binding.dest = dest;
  binding.src_col = shard_index * shard_cols;
  binding.spans.push_back({0, 0, dest->Rank() == 2 ? dest->Dim(0) : dest->Numel()});
  return binding;
}

Status WeightMap::Add(const std::string& checkpoint_name, WeightBinding binding) {
  if (binding.dest == nullptr) {
    return InternalError("weight binding for ", checkpoint_name, " has no destination");
  }
  if (!bindings_.emplace(checkpoint_name, std::move(binding)).second) {
    return InternalError("duplicate weight binding for ", checkpoint_name);
  }
  return OkStatus();
}

WeightBinding* WeightMap::Find(const std::string& checkpoint_name) {
  auto it = bindings_.find(checkpoint_name);
  return it == bindings_.end() ? nullptr : &it->second;
}


namespace {

/// Host rounding between the float formats checkpoints store and the dtype a
/// model executes in. Both low-precision formats are handled as bit patterns
/// because neither has a host arithmetic type here.
float BitsToFloat(uint16_t bits, DataType dtype) {
  if (dtype == DataType::kFloat16) {
    // IEEE 754 half: sign, 5 exponent bits, 10 mantissa bits.
    const uint32_t sign = (bits >> 15) & 0x1;
    const uint32_t exponent = (bits >> 10) & 0x1f;
    const uint32_t mantissa = bits & 0x3ff;
    uint32_t raw;
    if (exponent == 0) {
      raw = (sign << 31) | (mantissa << 13);  // Subnormal rounds to zero-ish.
    } else if (exponent == 0x1f) {
      raw = (sign << 31) | 0x7f800000u | (mantissa << 13);
    } else {
      raw = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
  }
  // bfloat16: the top half of an IEEE 754 single.
  const uint32_t raw = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

uint16_t FloatToBits(float value, DataType dtype) {
  uint32_t raw;
  std::memcpy(&raw, &value, sizeof(raw));
  if (dtype == DataType::kFloat16) {
    const uint32_t sign = (raw >> 16) & 0x8000;
    const int32_t exponent = static_cast<int32_t>((raw >> 23) & 0xff) - 127 + 15;
    uint16_t half;
    if (exponent <= 0) {
      half = static_cast<uint16_t>(sign);  // Flush subnormals to signed zero.
    } else if (exponent >= 0x1f) {
      half = static_cast<uint16_t>(sign | 0x7c00);
    } else {
      half = static_cast<uint16_t>(sign | (exponent << 10) | ((raw >> 13) & 0x3ff));
    }
    return half;
  }
  // Round-to-nearest-even into bfloat16 by adding the rounding bit pattern
  // around the truncated half, the same trick NumPy/PyTorch use.
  const uint32_t rounding = 0x7fff + ((raw >> 16) & 1);
  return static_cast<uint16_t>((raw + rounding) >> 16);
}

/// True when `src` can be byte-copied into `dst` with no conversion.
bool SameLayout(DataType src, DataType dst) { return src == dst; }

/// Converts one source row of `src_bytes` into `dst_bytes` (element count
/// `width`), byte-copying when the dtypes agree.
void ConvertRow(const std::byte* src, DataType src_dtype, std::byte* dst,
                DataType dst_dtype, int64_t width) {
  const size_t src_size = DataTypeSize(src_dtype);
  const size_t dst_size = DataTypeSize(dst_dtype);
  if (SameLayout(src_dtype, dst_dtype)) {
    std::memcpy(dst, src, static_cast<size_t>(width) * dst_size);
    return;
  }
  const bool src_is_float =
      src_dtype == DataType::kFloat32 || src_dtype == DataType::kFloat16 ||
      src_dtype == DataType::kBFloat16;
  const bool dst_is_float =
      dst_dtype == DataType::kFloat32 || dst_dtype == DataType::kFloat16 ||
      dst_dtype == DataType::kBFloat16;
  if (!src_is_float || !dst_is_float) {
    return;  // Non-float conversions are rejected before staging.
  }
  for (int64_t i = 0; i < width; ++i) {
    if (dst_dtype == DataType::kFloat32 && src_dtype != DataType::kFloat32) {
      const uint16_t bits =
          *reinterpret_cast<const uint16_t*>(src + i * src_size);
      const float value = BitsToFloat(bits, src_dtype);
      std::memcpy(dst + i * dst_size, &value, sizeof(value));
    } else if (src_dtype == DataType::kFloat32 && dst_dtype != DataType::kFloat32) {
      const float value = *reinterpret_cast<const float*>(src + i * src_size);
      const uint16_t bits = FloatToBits(value, dst_dtype);
      std::memcpy(dst + i * dst_size, &bits, sizeof(bits));
    } else {
      const uint16_t bits =
          *reinterpret_cast<const uint16_t*>(src + i * src_size);
      const uint16_t out = FloatToBits(BitsToFloat(bits, src_dtype), dst_dtype);
      std::memcpy(dst + i * dst_size, &out, sizeof(out));
    }
  }
}

/// The [rows, width] view of a source tensor; rank-1 tensors are a column.
Status DescribeSource(const Tensor& src, int64_t* rows, int64_t* width) {
  if (src.Rank() == 1) {
    // A column of single-element rows, so span math treats every element
    // as its own row.
    *rows = src.Dim(0);
    *width = 1;
    return OkStatus();
  }
  if (src.Rank() == 2) {
    *rows = src.Dim(0);
    *width = src.Dim(1);
    return OkStatus();
  }
  return InvalidArgumentError("expected a rank-1 or rank-2 weight, got rank ",
                              src.Rank());
}

}  // namespace

Status LoadWeightsInto(const CheckpointReader& reader, WeightMap& weights,
                       DeviceRuntime& runtime, DeviceStream stream) {
  const Device device = runtime.device();
  absl::flat_hash_set<std::string> consumed;

  for (const std::string& name : reader.Names()) {
    WeightBinding* binding = weights.Find(name);
    if (binding == nullptr) {
      if (weights.Skipped().contains(name)) continue;
      return InvalidArgumentError("checkpoint parameter ", name,
                                  " has no destination in this model");
    }
    consumed.insert(name);

    INFERX_ASSIGN_OR_RETURN(const Tensor src, reader.GetTensor(name));
    const Tensor& dest = *binding->dest;
    int64_t src_rows = 0, src_width = 0;
    INFERX_RETURN_IF_ERROR(DescribeSource(src, &src_rows, &src_width));

    const int64_t dst_width = dest.Rank() == 2 ? dest.Dim(1) : 1;
    if (binding->src_col + dst_width > src_width) {
      return InvalidArgumentError("parameter ", name, " provides columns [",
                                  binding->src_col, ", ", binding->src_col + dst_width,
                                  ") but has only ", src_width, " columns");
    }
    for (const WeightBinding::Span& span : binding->spans) {
      if (span.src_row + span.rows > src_rows || span.rows < 0) {
        return InvalidArgumentError("parameter ", name, " binding reads rows [",
                                    span.src_row, ", ", span.src_row + span.rows,
                                    ") of ", src_rows, " rows");
      }
      if (span.dst_row + span.rows > dest.Dim(0)) {
        return InvalidArgumentError("parameter ", name, " binding writes rows [",
                                    span.dst_row, ", ", span.dst_row + span.rows,
                                    ") of a ", dest.Dim(0), "-row destination");
      }
      if (span.rows == 0) continue;

      const DataType src_dtype = src.GetDataType();
      const DataType dst_dtype = dest.GetDataType();
      const bool float_pair =
          (src_dtype == DataType::kFloat32 || src_dtype == DataType::kFloat16 ||
           src_dtype == DataType::kBFloat16) &&
          (dst_dtype == DataType::kFloat32 || dst_dtype == DataType::kFloat16 ||
           dst_dtype == DataType::kBFloat16);
      if (!SameLayout(src_dtype, dst_dtype) && !float_pair) {
        return UnimplementedError("parameter ", name, " would need a ",
                                  DataTypeName(src_dtype), " -> ",
                                  DataTypeName(dst_dtype), " conversion");
      }

      // Stage the span on the host, converting element-wise as needed, then
      // enqueue one contiguous device copy for it.
      const size_t dst_row_bytes =
          static_cast<size_t>(dst_width) * DataTypeSize(dst_dtype);
      std::vector<std::byte> staged(static_cast<size_t>(span.rows) * dst_row_bytes);
      const std::byte* src_base = src.Bytes() + binding->src_col * DataTypeSize(src_dtype);
      const size_t src_row_bytes = static_cast<size_t>(src_width) * DataTypeSize(src_dtype);
      for (int64_t row = 0; row < span.rows; ++row) {
        ConvertRow(src_base + (span.src_row + row) * src_row_bytes, src_dtype,
                   staged.data() + row * dst_row_bytes, dst_dtype, dst_width);
      }
      INFERX_RETURN_IF_ERROR(runtime.CopyAsync(
          dest.Bytes() + span.dst_row * static_cast<int64_t>(dst_row_bytes),
          staged.data(), staged.size(), CopyKind::kHostToDevice, stream));
    }
  }

  std::vector<std::string> missing;
  for (const auto& [name, binding] : weights.Bindings()) {
    if (binding.required && !consumed.contains(name)) missing.push_back(name);
  }
  if (!missing.empty()) {
    std::sort(missing.begin(), missing.end());
    return InvalidArgumentError("checkpoint is missing parameters: ",
                                absl::StrJoin(missing, ", "));
  }
  return runtime.SynchronizeStream(stream);
}

}  // namespace models
}  // namespace inferx
