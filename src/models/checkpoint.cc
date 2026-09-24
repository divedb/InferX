#include "inferx/models/checkpoint.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>
#include <vector>

#include "inferx/core/status.h"

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

}  // namespace

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
    return NotFoundError("checkpoint is missing tensor ", name);
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
