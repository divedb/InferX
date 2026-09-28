// Weight-only quantization loading: GPTQ and AWQ packing reconstruct exactly
// the weights their plan describes, dense tensors win over the plan, and
// unsupported methods fail at open with the method named. Host-only.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/models/checkpoint.h"

namespace inferx::models {
namespace {

using DataType = ::inferx::DataType;

struct TempDir {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
                               ("inferx_quant_" + std::to_string(::getpid()));
  TempDir() { std::filesystem::create_directories(path); }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

struct Spec {
  std::string name;
  const char* dtype;
  std::vector<int64_t> shape;
  std::vector<std::byte> data;
};

void PushU16(std::vector<std::byte>& blob, uint16_t v) {
  for (int i = 0; i < 2; ++i) blob.push_back(static_cast<std::byte>(v >> (8 * i)));
}

void PushU32(std::vector<std::byte>& blob, uint32_t v) {
  for (int i = 0; i < 4; ++i) blob.push_back(static_cast<std::byte>(v >> (8 * i)));
}

void WriteSafetensors(const std::filesystem::path& file, const std::vector<Spec>& specs) {
  std::string header = "{";
  std::vector<std::byte> blob;
  bool first = true;
  for (const auto& spec : specs) {
    const int64_t begin = static_cast<int64_t>(blob.size());
    blob.insert(blob.end(), spec.data.begin(), spec.data.end());
    const int64_t end = static_cast<int64_t>(blob.size());
    if (!first) header += ",";
    first = false;
    header += "\"" + spec.name + "\":{\"dtype\":\"" + std::string(spec.dtype) + "\",\"shape\":[";
    header += std::to_string(spec.shape[0]);
    for (size_t d = 1; d < spec.shape.size(); ++d) header += "," + std::to_string(spec.shape[d]);
    header += "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
  }
  header += "}";
  const uint64_t hlen = header.size();
  std::vector<std::byte> bytes(8 + hlen + blob.size(), std::byte{0});
  std::memcpy(bytes.data(), &hlen, 8);
  std::memcpy(bytes.data() + 8, header.data(), hlen);
  std::memcpy(bytes.data() + 8 + hlen, blob.data(), blob.size());
  std::ofstream out(file.string(), std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

/// \brief Encodes an f32 that is exactly representable as binary16.
uint16_t FloatToHalfExact(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  const uint32_t exponent = ((bits >> 23) & 0xff) - 127 + 15;
  return static_cast<uint16_t>((exponent << 10) | ((bits >> 13) & 0x3ff));
}

/// \brief Encodes an f32 as bf16 bits with round-to-nearest-even.
uint16_t FloatToBf16(float v) {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

float Bf16BitsToFloat(uint16_t bits) {
  const uint32_t wide = static_cast<uint32_t>(bits) << 16;
  float v = 0;
  std::memcpy(&v, &wide, sizeof(v));
  return v;
}

/// Deterministic byte stream; every draw is a fresh value.
struct Lcg {
  uint32_t state = 0x12345678u;
  uint32_t Next() { return state = state * 1664525u + 1013904223u; }
  uint32_t Below(uint32_t bound) { return Next() % bound; }
};

void WriteConfig(const TempDir& dir, const std::string& quantization_config) {
  std::ofstream(dir.path / "config.json")
      << "{\"model_type\":\"qwen3\",\"quantization_config\":" << quantization_config << "}";
}

const std::string kGptqInt8 = "{\"bits\":8,\"quant_method\":\"gptq\",\"group_size\":8,"
                               "\"sym\":true,\"pack_dtype\":\"int32\","
                               "\"checkpoint_format\":\"gptq\"}";
const std::string kGptqInt4 = "{\"bits\":4,\"quant_method\":\"gptq\",\"group_size\":4,"
                               "\"sym\":true,\"pack_dtype\":\"int32\","
                               "\"checkpoint_format\":\"gptq\"}";
const std::string kAwqInt4 = "{\"bits\":4,\"quant_method\":\"awq\",\"group_size\":8,"
                              "\"zero_point\":true,\"version\":\"gemm\"}";

std::vector<float> Bf16Row(const Tensor& host) {
  const auto* bits = static_cast<const uint16_t*>(host.Data());
  std::vector<float> values(host.Numel());
  for (int64_t i = 0; i < host.Numel(); ++i) values[i] = Bf16BitsToFloat(bits[i]);
  return values;
}

/// \brief GPTQ int8 with an explicit g_idx that permutes the static groups.
///
/// Scale values are dyadic (n/1024), exact in f16 and bf16, so the only
/// rounding anywhere is the final f32->bf16 store.
TEST(QuantCheckpointTest, GptqInt8ReconstructsWeightsThroughGIdx) {
  constexpr int64_t kOut = 8, kIn = 16, kGroup = 8;
  constexpr int kPack = 4;  // 32 bits / 8.
  TempDir dir;
  WriteConfig(dir, kGptqInt8);
  Lcg rng;

  // Interleaved groups: even inputs take group 1's scale, odd take group 0's.
  std::vector<int32_t> g_idx(kIn);
  for (int64_t k = 0; k < kIn; ++k) g_idx[k] = static_cast<int32_t>(k % 2);
  std::vector<float> scales;  // [groups, out], f16-exact.
  for (int64_t g = 0; g < kIn / kGroup; ++g)
    for (int64_t n = 0; n < kOut; ++n)
      scales.push_back(static_cast<float>(8 + 4 * g + n) / 1024.0f);
  std::vector<uint8_t> q(kIn * kOut);
  std::vector<float> expected(kOut * kIn);
  for (int64_t n = 0; n < kOut; ++n) {
    for (int64_t k = 0; k < kIn; ++k) {
      q[k * kOut + n] = static_cast<uint8_t>(rng.Below(256));
      expected[n * kIn + k] =
          scales[g_idx[k] * kOut + n] * (static_cast<int>(q[k * kOut + n]) - 128);
    }
  }

  std::vector<Spec> specs;
  {
    std::vector<std::byte> blob;
    for (int64_t i = 0; i < kIn / kPack; ++i)
      for (int64_t n = 0; n < kOut; ++n) {
        uint32_t word = 0;
        for (int j = 0; j < kPack; ++j) {
          word |= static_cast<uint32_t>(q[(i * kPack + j) * kOut + n]) << (8 * j);
        }
        PushU32(blob, word);
      }
    specs.push_back({"w.qweight", "I32", {kIn / kPack, kOut}, blob});
  }
  {
    std::vector<std::byte> blob(kIn / kGroup * (kOut / kPack) * 4);
    for (auto& b : blob) b = static_cast<std::byte>(0x7F);  // Symmetric zero point.
    specs.push_back({"w.qzeros", "I32", {kIn / kGroup, kOut / kPack}, blob});
  }
  {
    std::vector<std::byte> blob;
    for (float s : scales) PushU16(blob, FloatToHalfExact(s));
    specs.push_back({"w.scales", "F16", {kIn / kGroup, kOut}, blob});
  }
  {
    std::vector<std::byte> blob;
    for (int32_t g : g_idx) PushU32(blob, static_cast<uint32_t>(g));
    specs.push_back({"w.g_idx", "I32", {kIn}, blob});
  }
  WriteSafetensors(dir.path / "model.safetensors", specs);

  auto checkpoint = Checkpoint::Open(dir.path.string());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  ASSERT_TRUE(checkpoint->quant_plan().has_value());
  auto host = checkpoint->FindHostBf16("w.weight", Shape({kOut, kIn}));
  ASSERT_TRUE(host.ok()) << host.status();
  ASSERT_EQ(host->GetDataType(), DataType::kBFloat16);
  const std::vector<float> values = Bf16Row(*host);
  for (int64_t i = 0; i < kOut * kIn; ++i) {
    EXPECT_NEAR(values[i], expected[i], std::abs(expected[i]) * 0.01f + 1e-6f) << i;
  }
}

/// \brief GPTQ int4 without g_idx: static grouping and the nibble zero point
///        (7, giving the net offset of 8) reconstruct through the same path.
TEST(QuantCheckpointTest, GptqInt4ReconstructsWithStaticGroups) {
  constexpr int64_t kOut = 8, kIn = 16, kGroup = 4;
  constexpr int kPack = 8;  // 32 bits / 4.
  TempDir dir;
  WriteConfig(dir, kGptqInt4);
  Lcg rng;

  const int64_t groups = kIn / kGroup;
  std::vector<float> scales(groups * kOut);
  for (int64_t g = 0; g < groups; ++g)
    for (int64_t n = 0; n < kOut; ++n)
      scales[g * kOut + n] = static_cast<float>(16 + 2 * g + n) / 2048.0f;
  std::vector<uint32_t> q(kIn * kOut);
  std::vector<float> expected(kOut * kIn);
  for (int64_t n = 0; n < kOut; ++n) {
    for (int64_t k = 0; k < kIn; ++k) {
      q[k * kOut + n] = rng.Below(16);
      expected[n * kIn + k] = scales[(k / kGroup) * kOut + n] *
                               (static_cast<int>(q[k * kOut + n]) - 7 - 1);
    }
  }

  std::vector<Spec> specs;
  {
    std::vector<std::byte> blob;
    for (int64_t i = 0; i < kIn / kPack; ++i)
      for (int64_t n = 0; n < kOut; ++n) {
        uint32_t word = 0;
        for (int j = 0; j < kPack; ++j) {
          word |= q[(i * kPack + j) * kOut + n] << (4 * j);
        }
        PushU32(blob, word);
      }
    specs.push_back({"w.qweight", "I32", {kIn / kPack, kOut}, blob});
  }
  {
    std::vector<std::byte> blob(groups * (kOut / kPack) * 4);
    for (auto& b : blob) b = static_cast<std::byte>(0x77);
    specs.push_back({"w.qzeros", "I32", {groups, kOut / kPack}, blob});
  }
  {
    std::vector<std::byte> blob;
    for (float s : scales) PushU16(blob, FloatToHalfExact(s));
    specs.push_back({"w.scales", "F16", {groups, kOut}, blob});
  }
  WriteSafetensors(dir.path / "model.safetensors", specs);

  auto checkpoint = Checkpoint::Open(dir.path.string());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  auto host = checkpoint->FindHostBf16("w.weight", Shape({kOut, kIn}));
  ASSERT_TRUE(host.ok()) << host.status();
  const std::vector<float> values = Bf16Row(*host);
  for (int64_t i = 0; i < kOut * kIn; ++i) {
    EXPECT_NEAR(values[i], expected[i], std::abs(expected[i]) * 0.01f + 1e-6f) << i;
  }
}

/// \brief AWQ gemm packing: values and zero points pack along the output
///        dimension, zeros carry the +1 bias, scales arrive in bf16.
TEST(QuantCheckpointTest, AwqInt4ReconstructsWeights) {
  constexpr int64_t kOut = 8, kIn = 16, kGroup = 8;
  constexpr int kPack = 8;
  TempDir dir;
  WriteConfig(dir, kAwqInt4);
  Lcg rng;

  const int64_t groups = kIn / kGroup;
  std::vector<float> scales(groups * kOut);
  for (int64_t g = 0; g < groups; ++g)
    for (int64_t n = 0; n < kOut; ++n)
      scales[g * kOut + n] = static_cast<float>(32 + 4 * g + n) / 4096.0f;
  std::vector<uint32_t> q(kIn * kOut), z(groups * kOut);
  std::vector<float> expected(kOut * kIn);
  for (int64_t k = 0; k < kIn; ++k) {
    for (int64_t n = 0; n < kOut; ++n) {
      q[k * kOut + n] = rng.Below(16);
      z[(k / kGroup) * kOut + n] = rng.Below(16);
      expected[n * kIn + k] = scales[(k / kGroup) * kOut + n] *
                              (static_cast<int>(q[k * kOut + n]) -
                               static_cast<int>(z[(k / kGroup) * kOut + n]) + 1);
    }
  }

  std::vector<Spec> specs;
  {
    std::vector<std::byte> blob;
    for (int64_t k = 0; k < kIn; ++k)
      for (int64_t m = 0; m < kOut / kPack; ++m) {
        uint32_t word = 0;
        for (int j = 0; j < kPack; ++j) {
          word |= q[k * kOut + m * kPack + j] << (4 * j);
        }
        PushU32(blob, word);
      }
    specs.push_back({"w.qweight", "I32", {kIn, kOut / kPack}, blob});
  }
  {
    std::vector<std::byte> blob;
    for (int64_t g = 0; g < groups; ++g)
      for (int64_t m = 0; m < kOut / kPack; ++m) {
        uint32_t word = 0;
        for (int j = 0; j < kPack; ++j) {
          word |= z[g * kOut + m * kPack + j] << (4 * j);
        }
        PushU32(blob, word);
      }
    specs.push_back({"w.qzeros", "I32", {groups, kOut / kPack}, blob});
  }
  {
    std::vector<std::byte> blob;
    for (float s : scales) PushU16(blob, FloatToBf16(s));
    specs.push_back({"w.scales", "BF16", {groups, kOut}, blob});
  }
  WriteSafetensors(dir.path / "model.safetensors", specs);

  auto checkpoint = Checkpoint::Open(dir.path.string());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  ASSERT_TRUE(checkpoint->quant_plan().has_value());
  auto host = checkpoint->FindHostBf16("w.weight", Shape({kOut, kIn}));
  ASSERT_TRUE(host.ok()) << host.status();
  const std::vector<float> values = Bf16Row(*host);
  for (int64_t i = 0; i < kOut * kIn; ++i) {
    EXPECT_NEAR(values[i], expected[i], std::abs(expected[i]) * 0.01f + 1e-6f) << i;
  }
}

/// \brief A dense tensor with the requested name always wins over the plan;
///        gpt-oss-style mxfp4 configs open dense with no plan at all.
TEST(QuantCheckpointTest, DenseTensorWinsAndMxfp4StaysDense) {
  TempDir dir;
  WriteConfig(dir, "{\"quant_method\":\"mxfp4\"}");
  std::vector<std::byte> blob;
  for (int i = 0; i < 16; ++i) PushU16(blob, FloatToBf16(0.25f * (i + 1)));
  WriteSafetensors(dir.path / "model.safetensors",
                   {{"w.weight", "BF16", {4, 4}, blob}});
  auto checkpoint = Checkpoint::Open(dir.path.string());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  EXPECT_FALSE(checkpoint->quant_plan().has_value());
  auto host = checkpoint->FindHostBf16("w.weight", Shape({4, 4}));
  ASSERT_TRUE(host.ok()) << host.status();
  const std::vector<float> values = Bf16Row(*host);
  EXPECT_FLOAT_EQ(values[5], 1.5f);
}

/// \brief Unknown quantization methods fail at open, naming the method.
TEST(QuantCheckpointTest, RejectsUnsupportedQuantMethod) {
  TempDir dir;
  WriteConfig(dir, "{\"quant_method\":\"compressed-tensors\",\"bits\":4}");
  auto checkpoint = Checkpoint::Open(dir.path.string());
  ASSERT_FALSE(checkpoint.ok());
  EXPECT_EQ(checkpoint.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(std::string(checkpoint.status().message()).find("compressed-tensors"),
            std::string::npos);
}

/// \brief Packed tensors whose shapes contradict the logical weight report
///        an error instead of dequantizing garbage.
TEST(QuantCheckpointTest, ReportsPackedShapeMismatch) {
  constexpr int64_t kOut = 8, kIn = 16;
  TempDir dir;
  WriteConfig(dir, kGptqInt8);
  std::vector<std::byte> blob(16);  // [1, 4] words: wrong for in=16, pack=4.
  WriteSafetensors(dir.path / "model.safetensors",
                   {{"w.qweight", "I32", {1, 4}, blob},
                    {"w.qzeros", "I32", {2, 2}, blob},
                    {"w.scales", "F16", {2, 8}, std::vector<std::byte>(32)}});
  auto checkpoint = Checkpoint::Open(dir.path.string());
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  auto host = checkpoint->FindHostBf16("w.weight", Shape({kOut, kIn}));
  ASSERT_FALSE(host.ok());
  EXPECT_EQ(host.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(std::string(host.status().message()).find("GPTQ"), std::string::npos);
}

}  // namespace
}  // namespace inferx::models
