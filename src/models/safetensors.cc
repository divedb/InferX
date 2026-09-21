#include "inferx/models/safetensors.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "inferx/core/tensor_io.h"
#include "nlohmann/json.hpp"

namespace inferx {
namespace {

/// \brief Reads an entire file into a byte string.
///
/// \param path Path to the file.
/// \return The file contents, or an error status.
absl::StatusOr<std::string> ReadFileBytes(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return absl::NotFoundError("could not open file: " + path);
  }
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  input.seekg(0, std::ios::beg);
  std::string data(static_cast<size_t>(size), '\0');
  input.read(data.data(), size);
  return data;
}

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

/// \brief Converts a bfloat16 value to float32.
///
/// \param b Bfloat16 bit pattern.
/// \return The equivalent float32 value.
float Bf16ToFloat(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/// \brief Returns the total number of elements implied by a shape.
///
/// \param shape Tensor dimensions.
/// \return The product of all dimensions.
int64_t NumElements(const std::vector<int64_t>& shape) {
  int64_t total = 1;
  for (const int64_t dim : shape) {
    total *= dim;
  }
  return total;
}

/// \brief Converts a raw tensor buffer of `dtype` into float32.
///
/// \param dtype Safetensors dtype name (e.g. "F16", "BF16", "F32").
/// \param src Raw buffer of `count` elements in `dtype`.
/// \param count Number of elements in the buffer.
/// \return The values converted to float32, or an error status for
/// unsupported dtypes.
absl::StatusOr<std::vector<float>> ConvertToFloat(const std::string& dtype,
                                                  const char* src,
                                                  int64_t count) {
  std::vector<float> out(static_cast<size_t>(count));
  if (dtype == "F32") {
    std::memcpy(out.data(), src, static_cast<size_t>(count) * sizeof(float));
  } else if (dtype == "F16") {
    const auto* p = reinterpret_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < count; ++i) {
      out[static_cast<size_t>(i)] = HalfToFloat(p[i]);
    }
  } else if (dtype == "BF16") {
    const auto* p = reinterpret_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < count; ++i) {
      out[static_cast<size_t>(i)] = Bf16ToFloat(p[i]);
    }
  } else if (dtype == "F64") {
    const auto* p = reinterpret_cast<const double*>(src);
    for (int64_t i = 0; i < count; ++i) {
      out[static_cast<size_t>(i)] = static_cast<float>(p[i]);
    }
  } else {
    return absl::UnimplementedError("unsupported safetensors dtype: " + dtype);
  }
  return out;
}

}  // namespace

absl::StatusOr<SafeTensors> SafeTensors::FromFile(const std::string& path) {
  absl::StatusOr<std::string> bytes = ReadFileBytes(path);
  if (!bytes.ok()) {
    return bytes.status();
  }
  const std::string& data = *bytes;
  if (data.size() < 8) {
    return absl::InvalidArgumentError("safetensors file too small: " + path);
  }

  uint64_t header_size = 0;
  std::memcpy(&header_size, data.data(), sizeof(header_size));
  const size_t header_end = 8 + static_cast<size_t>(header_size);
  if (header_end > data.size()) {
    return absl::InvalidArgumentError("invalid safetensors header size: " +
                                      path);
  }

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(data.substr(8, header_size));
  } catch (const nlohmann::json::exception& e) {
    return absl::InvalidArgumentError(
        std::string("failed to parse safetensors header: ") + e.what());
  }

  SafeTensors tensors;
  for (const auto& [name, info] : header.items()) {
    if (name == "__metadata__") {
      continue;
    }
    if (!info.is_object() || !info.contains("dtype") ||
        !info.contains("shape") || !info.contains("data_offsets")) {
      return absl::InvalidArgumentError("invalid safetensors header entry: " +
                                        name);
    }
    const std::string dtype = info.at("dtype").get<std::string>();
    const std::vector<int64_t> shape =
        info.at("shape").get<std::vector<int64_t>>();
    const std::vector<int64_t> offsets =
        info.at("data_offsets").get<std::vector<int64_t>>();
    if (offsets.size() != 2) {
      return absl::InvalidArgumentError("invalid data_offsets for " + name);
    }
    const int64_t count = NumElements(shape);
    const size_t begin = header_end + static_cast<size_t>(offsets[0]);
    const size_t end = header_end + static_cast<size_t>(offsets[1]);
    if (end > data.size() || begin > end) {
      return absl::InvalidArgumentError("tensor out of bounds: " + name);
    }
    absl::StatusOr<std::vector<float>> values =
        ConvertToFloat(dtype, data.data() + begin, count);
    if (!values.ok()) {
      return values.status();
    }
    tensors.tensors_.emplace(name, SafeTensor{shape, std::move(*values)});
  }
  return tensors;
}

absl::StatusOr<SafeTensors> SafeTensors::FromDirectory(const std::string& dir) {
  const std::filesystem::path root(dir);
  if (!std::filesystem::is_directory(root)) {
    return absl::NotFoundError("not a directory: " + dir);
  }

  const std::filesystem::path single = root / "model.safetensors";
  if (std::filesystem::exists(single)) {
    return FromFile(single.string());
  }

  SafeTensors merged;
  bool found_any = false;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (entry.path().extension() != ".safetensors") {
      continue;
    }
    // Skip common auxiliary checkpoints.
    if (filename.find("optim") != std::string::npos ||
        filename.find("adapter") != std::string::npos) {
      continue;
    }
    absl::StatusOr<SafeTensors> shard = FromFile(entry.path().string());
    if (!shard.ok()) {
      return shard.status();
    }
    merged.Merge(std::move(*shard));
    found_any = true;
  }
  if (!found_any) {
    return absl::NotFoundError("no .safetensors files found in " + dir);
  }
  return merged;
}

void SafeTensors::Merge(SafeTensors&& other) {
  for (auto& [name, tensor] : other.tensors_) {
    tensors_.insert_or_assign(name, std::move(tensor));
  }
}

bool SafeTensors::Contains(absl::string_view name) const {
  return tensors_.contains(std::string(name));
}

const SafeTensor* SafeTensors::Find(absl::string_view name) const {
  auto it = tensors_.find(std::string(name));
  return it == tensors_.end() ? nullptr : &it->second;
}

StatusOr<Tensor> SafeTensors::GetTensor(absl::string_view name, DeviceId device,
                                        bool transpose) const {
  const auto* tensor = Find(name);
  if (!tensor) return NotFoundError("missing tensor: ", name);
  if (!transpose) {
    return TensorFromHost(tensor->data,
                          Shape(absl::MakeConstSpan(tensor->shape)), device);
  }
  if (tensor->shape.size() != 2)
    return InvalidArgumentError("transpose requires matrix: ", name);
  const auto rows = tensor->shape[0], cols = tensor->shape[1];
  std::vector<float> values(tensor->data.size());
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j)
      values[j * rows + i] = tensor->data[i * cols + j];
  }
  return TensorFromHost(values, Shape{cols, rows}, device);
}

StatusOr<Tensor> SafeTensors::GetTensorRows(absl::string_view name,
                                            DeviceId device, int64_t begin,
                                            int64_t end) const {
  const auto* tensor = Find(name);
  if (!tensor) return NotFoundError("missing tensor: ", name);
  if (tensor->shape.size() != 2 || begin < 0 || end <= begin ||
      end > tensor->shape[0]) {
    return InvalidArgumentError("invalid row slice for ", name);
  }
  const int64_t cols = tensor->shape[1];
  std::vector<float> values(
      tensor->data.begin() + static_cast<ptrdiff_t>(begin * cols),
      tensor->data.begin() + static_cast<ptrdiff_t>(end * cols));
  return TensorFromHost(values, Shape{end - begin, cols}, device);
}

StatusOr<Tensor> SafeTensors::GetTransposedOutputSlice(absl::string_view name,
                                                       DeviceId device,
                                                       int64_t out_begin,
                                                       int64_t out_end) const {
  const auto* tensor = Find(name);
  if (!tensor) return NotFoundError("missing tensor: ", name);
  if (tensor->shape.size() != 2 || out_begin < 0 || out_end <= out_begin ||
      out_end > tensor->shape[0]) {
    return InvalidArgumentError("invalid output slice for ", name);
  }
  const int64_t cols = tensor->shape[1];
  const int64_t width = out_end - out_begin;
  std::vector<float> values(static_cast<size_t>(cols * width));
  for (int64_t i = 0; i < width; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      values[j * width + i] = tensor->data[(out_begin + i) * cols + j];
    }
  }
  return TensorFromHost(values, Shape{cols, width}, device);
}

StatusOr<Tensor> SafeTensors::GetTransposedInputSlice(absl::string_view name,
                                                      DeviceId device,
                                                      int64_t in_begin,
                                                      int64_t in_end) const {
  const auto* tensor = Find(name);
  if (!tensor) return NotFoundError("missing tensor: ", name);
  if (tensor->shape.size() != 2 || in_begin < 0 || in_end <= in_begin ||
      in_end > tensor->shape[1]) {
    return InvalidArgumentError("invalid input slice for ", name);
  }
  const int64_t rows = tensor->shape[0];
  const int64_t width = in_end - in_begin;
  std::vector<float> values(static_cast<size_t>(width * rows));
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < width; ++j) {
      values[j * rows + i] = tensor->data[i * tensor->shape[1] + in_begin + j];
    }
  }
  return TensorFromHost(values, Shape{width, rows}, device);
}

}  // namespace inferx
