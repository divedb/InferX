#include <cmath>
#include <cstring>

#include "inferx/models/checkpoint.h"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <sstream>
#include <system_error>
#include <thread>
#include <vector>

#include "inferx/core/status.h"
#include "nlohmann/json.hpp"

namespace inferx::models {
namespace {

/// \brief Converts an IEEE 754 binary16 value to float32.
///
/// \param h Binary16 bit pattern.
/// \return The equivalent float32 value.
float HalfToFloat(uint16_t h) {
  const uint32_t sign = (h >> 15) & 0x1u;
  uint32_t exponent = (h >> 10) & 0x1fu;
  uint32_t mantissa = h & 0x3ffu;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign << 31;
    } else {
      // Subnormal: normalize the mantissa.
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3ffu;
      bits = (sign << 31) | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    bits = (sign << 31) | 0x7f800000u | (mantissa << 13);
  } else {
    bits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/// \brief Rounds a float32 to the nearest bfloat16 bit pattern.
///
/// Round-to-nearest-even, matching torch's float->bfloat16 conversion; a
/// bf16->fp32->bf16 round trip through the loader is exact.
uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  const uint32_t rounding = 0x7fffu + lsb;
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

/// \brief Opens one shard, translating SafeTensorReader's error type into a
///        Status.
StatusOr<SafeTensorReader> OpenShard(const std::filesystem::path& path) {
  auto reader = SafeTensorReader::Open(path);
  if (!reader.has_value()) {
    const SafeTensorError error = std::move(reader.error());
    if (error.code == SafeTensorErrorCode::kFileNotFound) {
      return NotFoundError(error.message);
    }
    return InvalidArgumentError(error.message);
  }
  return std::move(*reader);
}

/// \brief Decodes a host f16/bf16/f32 scale tensor into float values.
StatusOr<std::vector<float>> ScaleValues(const Tensor& scales) {
  std::vector<float> values(scales.Numel());
  const int64_t n = scales.Numel();
  switch (scales.GetDataType()) {
    case DataType::kFloat32: {
      const auto* p = static_cast<const float*>(scales.Data());
      std::memcpy(values.data(), p, n * sizeof(float));
      break;
    }
    case DataType::kFloat16: {
      const auto* p = static_cast<const uint16_t*>(scales.Data());
      for (int64_t i = 0; i < n; ++i) values[i] = HalfToFloat(p[i]);
      break;
    }
    case DataType::kBFloat16: {
      const auto* p = static_cast<const uint16_t*>(scales.Data());
      for (int64_t i = 0; i < n; ++i) {
        const uint32_t bits = static_cast<uint32_t>(p[i]) << 16;
        std::memcpy(&values[i], &bits, sizeof(float));
      }
      break;
    }
    default:
      return InvalidArgumentError("quantization scales have unsupported dtype ",
                                  DataTypeName(scales.GetDataType()));
  }
  return values;
}

/// \brief Runs `body(begin, end)` over [0, count) on the host worker pool.
///
/// Load-time dequantization over tens of millions of values: the chunks are
/// disjoint writes into one allocation, so no synchronization beyond join.
void ParallelFor(int64_t count, const std::function<void(int64_t, int64_t)>& body) {
  const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
  const int64_t wanted = std::max<int64_t>(1, count / 4096);
  const unsigned threads = static_cast<unsigned>(std::min<int64_t>(hardware, wanted));
  if (threads <= 1) {
    body(0, count);
    return;
  }
  std::vector<std::thread> pool;
  std::mutex mu;
  std::exception_ptr failure;
  const int64_t chunk = (count + threads - 1) / threads;
  for (unsigned t = 0; t < threads && t * chunk < count; ++t) {
    const int64_t begin = static_cast<int64_t>(t) * chunk;
    const int64_t end = std::min(count, begin + chunk);
    pool.emplace_back([begin, end, &body, &mu, &failure] {
      try {
        body(begin, end);
      } catch (...) {
        const std::lock_guard<std::mutex> lock(mu);
        if (!failure) failure = std::current_exception();
      }
    });
  }
  for (auto& thread : pool) thread.join();
  if (failure) std::rethrow_exception(failure);
}

}  // namespace

StatusOr<std::optional<QuantPlan>> QuantPlan::FromConfigJson(const std::string& json) {
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(json);
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError("invalid config.json: ", e.what());
  }
  if (!parsed.is_object() || !parsed.contains("quantization_config")) {
    return std::nullopt;
  }
  const auto& section = parsed.at("quantization_config");
  if (!section.is_object()) return std::nullopt;
  const std::string method = section.value("quant_method", std::string());
  if (method.empty()) return std::nullopt;

  QuantPlan plan;
  if (method == "gptq") {
    plan.format = QuantPlan::Format::kGptq;
    const std::string pack_dtype = section.value("pack_dtype", std::string("int32"));
    const std::string checkpoint_format =
        section.value("checkpoint_format", std::string("gptq"));
    if (pack_dtype != "int32" || checkpoint_format != "gptq") {
      return UnimplementedError("unsupported GPTQ packing: pack_dtype=", pack_dtype,
                                ", checkpoint_format=", checkpoint_format);
    }
  } else if (method == "awq") {
    plan.format = QuantPlan::Format::kAwq;
    const std::string version = section.value("version", std::string("gemm"));
    if (version != "gemm") {
      return UnimplementedError("unsupported AWQ packing version: ", version);
    }
  } else if (method == "mxfp4") {
    // gpt-oss: experts are fused MXFP4 tensors the loader unpacks by name.
    return std::nullopt;
  } else {
    return UnimplementedError("unsupported checkpoint quantization method: ", method);
  }

  plan.bits = section.value("bits", 4);
  plan.group_size = section.value("group_size", int64_t{128});
  if (plan.bits != 4 && plan.bits != 8) {
    return UnimplementedError("unsupported ", method, " bit width: ", plan.bits);
  }
  if (plan.group_size <= 0) {
    return InvalidArgumentError("quantization group_size must be positive");
  }
  return plan;
}

/// E2M1 nibble magnitudes (OCP MX FP4): bit 3 is the sign.
constexpr float kMxNibbleMagnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

StatusOr<Tensor> Checkpoint::AsHostBf16(const Tensor& host) {
  if (host.GetDataType() == DataType::kBFloat16) return host;

  const int64_t numel = host.Numel();
  INFERX_ASSIGN_OR_RETURN(
      auto converted, Tensor::Empty(DataType::kBFloat16, host.GetShape(), DeviceId::Cpu()));
  auto* out = static_cast<uint16_t*>(converted.Data());
  switch (host.GetDataType()) {
    case DataType::kFloat32: {
      const auto* p = static_cast<const float*>(host.Data());
      for (int64_t i = 0; i < numel; ++i) out[i] = FloatToBf16Bits(p[i]);
      break;
    }
    case DataType::kFloat16: {
      const auto* p = static_cast<const uint16_t*>(host.Data());
      for (int64_t i = 0; i < numel; ++i) out[i] = FloatToBf16Bits(HalfToFloat(p[i]));
      break;
    }
    default:
      return UnimplementedError("cannot load ",
                                DataTypeName(host.GetDataType()), " tensor as bfloat16");
  }
  return converted;
}

StatusOr<Tensor> Checkpoint::FindHostBf16(std::string_view name,
                                          const Shape& expected_full) const {
  std::optional<Tensor> host = Find(name);
  if (!host.has_value()) {
    return FindQuantizedBf16(name, expected_full);
  }
  if (host->GetShape() != expected_full) {
    return InvalidArgumentError("unexpected shape for ", name, ": got ",
                                host->GetShape().ToString(), ", want ",
                                expected_full.ToString());
  }
  return AsHostBf16(*host);
}

StatusOr<Checkpoint> Checkpoint::Open(const std::string& dir) {
  const std::filesystem::path root(dir);
  std::error_code ec;
  if (!std::filesystem::is_directory(root, ec)) {
    return NotFoundError("not a directory: ", dir);
  }

  Checkpoint checkpoint;
  {
    std::ifstream config(root / "config.json");
    if (config) {
      std::ostringstream text;
      text << config.rdbuf();
      INFERX_ASSIGN_OR_RETURN(checkpoint.quant_, QuantPlan::FromConfigJson(text.str()));
    }
  }
  const std::filesystem::path single = root / "model.safetensors";
  if (std::filesystem::exists(single)) {
    INFERX_ASSIGN_OR_RETURN(SafeTensorReader reader, OpenShard(single));
    checkpoint.shards_.push_back(std::move(reader));
    return checkpoint;
  }

  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".safetensors") {
      continue;
    }
    // Skip common auxiliary checkpoints.
    const std::string filename = entry.path().filename().string();
    if (filename.find("optim") != std::string::npos ||
        filename.find("adapter") != std::string::npos) {
      continue;
    }
    INFERX_ASSIGN_OR_RETURN(SafeTensorReader reader, OpenShard(entry.path()));
    checkpoint.shards_.push_back(std::move(reader));
  }
  if (checkpoint.shards_.empty()) {
    return NotFoundError("no .safetensors files found in ", dir);
  }
  return checkpoint;
}

bool Checkpoint::Contains(std::string_view name) const {
  for (const auto& shard : shards_) {
    if (shard.HasTensor(name)) return true;
  }
  return false;
}

std::optional<Tensor> Checkpoint::Find(std::string_view name) const {
  for (const auto& shard : shards_) {
    if (auto tensor = shard.GetTensor(name)) {
      return tensor;
    }
  }
  return std::nullopt;
}

StatusOr<Tensor> Checkpoint::UploadF32(std::string_view name, const Shape& expected,
                                     DeviceId device) const {
  auto host = Find(name);
  if (!host.has_value()) {
    return NotFoundError("checkpoint tensor not found: ", name);
  }
  if (host->GetShape() != expected) {
    return InvalidArgumentError("tensor ", name, " has shape ", host->GetShape().ToString(),
                                ", expected ", expected.ToString());
  }
  if (host->GetDataType() == DataType::kFloat32) return host->To(device);
  if (host->GetDataType() != DataType::kBFloat16) {
    return UnimplementedError("cannot load ", DataTypeName(host->GetDataType()),
                              " tensor as float32: ", name);
  }
  INFERX_ASSIGN_OR_RETURN(auto converted,
                          Tensor::Empty(DataType::kFloat32, expected, DeviceId::Cpu()));
  const auto* in = static_cast<const uint16_t*>(host->Data());
  auto* out = static_cast<float*>(converted.Data());
  for (int64_t i = 0; i < host->Numel(); ++i) {
    const uint32_t bits = static_cast<uint32_t>(in[i]) << 16;
    std::memcpy(&out[i], &bits, sizeof(float));
  }
  return converted.To(device);
}

StatusOr<Tensor> Checkpoint::DequantMxToBf16(const Tensor& blocks, const Tensor& scales,
                                             const Shape& logical) const {
  if (blocks.GetDataType() != DataType::kUInt8 || scales.GetDataType() != DataType::kUInt8 ||
      blocks.Numel() * 2 != logical.Numel() || scales.Numel() * 32 != logical.Numel()) {
    return InvalidArgumentError("MXFP4 blocks and scales disagree with ", logical.ToString());
  }
  const auto* packed = static_cast<const uint8_t*>(blocks.Data());
  const auto* e8m0 = static_cast<const uint8_t*>(scales.Data());
  INFERX_ASSIGN_OR_RETURN(auto out, Tensor::Empty(DataType::kBFloat16, logical, DeviceId::Cpu()));
  auto* values = static_cast<uint16_t*>(out.Data());
  const int64_t n = logical.Numel();
  for (int64_t i = 0; i < n; ++i) {
    // Element 0 of a byte is the high nibble; one E8M0 byte scales 32 values.
    const uint8_t byte = packed[i / 2];
    const uint8_t nibble = (i % 2 == 0) ? (byte >> 4) : (byte & 0xF);
    const float magnitude = kMxNibbleMagnitude[nibble & 0x7];
    const float sign = (nibble & 0x8) ? -1.0f : 1.0f;
    const float value = sign * magnitude * exp2f(static_cast<float>(e8m0[i / 32]) - 127.0f);
    values[i] = FloatToBf16Bits(value);
  }
  return out;
}

StatusOr<Tensor> Checkpoint::DequantGptq(const Tensor& qweight, const Tensor& qzeros,
                                         const Tensor& scales, const std::optional<Tensor>& g_idx,
                                         const QuantPlan& plan, const Shape& logical) const {
  const int64_t out = logical.Dim(0), in = logical.Dim(1);
  const int64_t groups = in / plan.group_size;
  const int pack = 32 / plan.bits;
  const uint32_t mask = (1u << plan.bits) - 1;
  if (qweight.GetDataType() != DataType::kInt32 || qzeros.GetDataType() != DataType::kInt32 ||
      qweight.Rank() != 2 || qzeros.Rank() != 2 || scales.Rank() != 2 ||
      in % pack != 0 || out % pack != 0 || in % plan.group_size != 0 ||
      qweight.Dim(0) != in / pack || qweight.Dim(1) != out ||
      qzeros.Dim(0) != groups || qzeros.Dim(1) != out / pack ||
      scales.Dim(0) != groups || scales.Dim(1) != out) {
    return InvalidArgumentError("GPTQ tensors disagree with logical ", logical.ToString());
  }
  std::vector<int32_t> group_of;
  if (g_idx.has_value()) {
    if (g_idx->Rank() != 1 || g_idx->Dim(0) != in || g_idx->GetDataType() != DataType::kInt32) {
      return InvalidArgumentError("GPTQ g_idx disagrees with logical ", logical.ToString());
    }
    const auto* idx = static_cast<const int32_t*>(g_idx->Data());
    group_of.assign(idx, idx + in);
    for (int32_t g : group_of) {
      if (g < 0 || g >= groups) {
        return InvalidArgumentError("GPTQ g_idx names group ", g, " outside 0..", groups - 1);
      }
    }
  }
  INFERX_ASSIGN_OR_RETURN(const std::vector<float> scale_values, ScaleValues(scales));

  INFERX_ASSIGN_OR_RETURN(auto result,
                          Tensor::Empty(DataType::kBFloat16, logical, DeviceId::Cpu()));
  auto* values = static_cast<uint16_t*>(result.Data());
  const auto* packed_weights = static_cast<const int32_t*>(qweight.Data());
  const auto* packed_zeros = static_cast<const int32_t*>(qzeros.Data());
  const int bits = plan.bits;
  const int64_t zero_words = out / pack;
  ParallelFor(out, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; ++n) {
      uint16_t* row = values + n * in;
      for (int64_t i = 0; i < in / pack; ++i) {
        const uint32_t word = static_cast<uint32_t>(packed_weights[i * out + n]);
        for (int j = 0; j < pack; ++j) {
          const int64_t k = i * pack + j;
          const int64_t g = group_of.empty() ? k / plan.group_size : group_of[k];
          const uint32_t zero_word =
              static_cast<uint32_t>(packed_zeros[g * zero_words + n / pack]);
          const int zero = static_cast<int>((zero_word >> (bits * (n % pack))) & mask);
          const int value = static_cast<int>((word >> (bits * j)) & mask);
          row[k] = FloatToBf16Bits(scale_values[g * out + n] *
                                   static_cast<float>(value - zero - 1));
        }
      }
    }
  });
  return result;
}

StatusOr<Tensor> Checkpoint::DequantAwq(const Tensor& qweight, const Tensor& qzeros,
                                        const Tensor& scales, const QuantPlan& plan,
                                        const Shape& logical) const {
  const int64_t out = logical.Dim(0), in = logical.Dim(1);
  const int64_t groups = in / plan.group_size;
  const int pack = 32 / plan.bits;
  const uint32_t mask = (1u << plan.bits) - 1;
  if (qweight.GetDataType() != DataType::kInt32 || qzeros.GetDataType() != DataType::kInt32 ||
      qweight.Rank() != 2 || qzeros.Rank() != 2 || scales.Rank() != 2 ||
      in % plan.group_size != 0 || out % pack != 0 ||
      qweight.Dim(0) != in || qweight.Dim(1) != out / pack ||
      qzeros.Dim(0) != groups || qzeros.Dim(1) != out / pack ||
      scales.Dim(0) != groups || scales.Dim(1) != out) {
    return InvalidArgumentError("AWQ tensors disagree with logical ", logical.ToString());
  }
  INFERX_ASSIGN_OR_RETURN(const std::vector<float> scale_values, ScaleValues(scales));

  INFERX_ASSIGN_OR_RETURN(auto result,
                          Tensor::Empty(DataType::kBFloat16, logical, DeviceId::Cpu()));
  auto* values = static_cast<uint16_t*>(result.Data());
  const auto* packed_weights = static_cast<const int32_t*>(qweight.Data());
  const auto* packed_zeros = static_cast<const int32_t*>(qzeros.Data());
  const int bits = plan.bits;
  const int64_t words = out / pack;
  ParallelFor(out, [&](int64_t begin, int64_t end) {
    for (int64_t n = begin; n < end; ++n) {
      uint16_t* row = values + n * in;
      const int64_t word = n / pack;
      const int shift = bits * static_cast<int>(n % pack);
      for (int64_t k = 0; k < in; ++k) {
        const int64_t g = k / plan.group_size;
        const int value =
            static_cast<int>((static_cast<uint32_t>(packed_weights[k * words + word]) >>
                              shift) &
                             mask);
        const int zero =
            static_cast<int>((static_cast<uint32_t>(packed_zeros[g * words + word]) >>
                              shift) &
                             mask);
        row[k] = FloatToBf16Bits(scale_values[g * out + n] *
                                 static_cast<float>(value - zero + 1));
      }
    }
  });
  return result;
}

StatusOr<Tensor> Checkpoint::FindQuantizedBf16(std::string_view name,
                                               const Shape& logical) const {
  constexpr std::string_view kWeightSuffix = ".weight";
  if (!quant_.has_value() || logical.Rank() != 2 || name.size() <= kWeightSuffix.size() ||
      name.substr(name.size() - kWeightSuffix.size()) != kWeightSuffix) {
    return NotFoundError("checkpoint is missing tensor ", name);
  }
  const std::string base(name.substr(0, name.size() - kWeightSuffix.size()));
  const auto qweight = Find(base + ".qweight");
  if (!qweight.has_value()) {
    return NotFoundError("checkpoint is missing tensor ", name);
  }
  const auto qzeros = Find(base + ".qzeros");
  const auto scales = Find(base + ".scales");
  if (!qzeros.has_value() || !scales.has_value()) {
    return NotFoundError("quantized ", base, " is missing qzeros or scales");
  }
  const auto g_idx = Find(base + ".g_idx");
  if (quant_->format == QuantPlan::Format::kGptq) {
    return DequantGptq(*qweight, *qzeros, *scales, g_idx, *quant_, logical);
  }
  return DequantAwq(*qweight, *qzeros, *scales, *quant_, logical);
}

StatusOr<Tensor> Checkpoint::UploadBf16(std::string_view name, const Shape& expected,
                                        DeviceId device) const {
  INFERX_ASSIGN_OR_RETURN(auto host, FindHostBf16(name, expected));
  return host.To(device);
}

StatusOr<LoadedCheckpoint> LoadCheckpointConfig(const std::string& dir) {
  LoadedCheckpoint checkpoint;
  std::ifstream file(dir + "/config.json");
  if (!file) {
    return NotFoundError("could not open config: ", dir);
  }
  std::ostringstream text;
  text << file.rdbuf();
  checkpoint.config_json = text.str();
  INFERX_ASSIGN_OR_RETURN(checkpoint.config,
                          CheckpointConfig::FromJson(checkpoint.config_json));
  return checkpoint;
}

Status OpenCheckpointWeights(LoadedCheckpoint& checkpoint, const std::string& dir) {
  INFERX_ASSIGN_OR_RETURN(checkpoint.weights, Checkpoint::Open(dir));
  return OkStatus();
}

StatusOr<LoadedCheckpoint> LoadCheckpoint(const std::string& dir) {
  INFERX_ASSIGN_OR_RETURN(auto checkpoint, LoadCheckpointConfig(dir));
  INFERX_RETURN_IF_ERROR(OpenCheckpointWeights(checkpoint, dir));
  return checkpoint;
}

}  // namespace inferx::models
