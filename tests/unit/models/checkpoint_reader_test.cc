// CheckpointReader directory behavior: single-file checkpoints, sharded
// checkpoints enumerated by glob, auxiliary-file skipping, and error cases.
// The safetensors format itself (header parsing, offsets, dtypes, malformed
// buffers) is covered by safe_tensors_view_test and safe_tensors_reader_test.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/tensor.h"
#include "inferx/models/safetensors.h"

namespace inferx {
namespace models {
namespace {

void WriteFile(const std::string& path, const std::string& bytes) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream(path, std::ios::binary) << bytes;
}

std::string Frame(const std::string& json, const std::string& body) {
  uint64_t length = json.size();
  std::string bytes(8, '\0');
  std::memcpy(bytes.data(), &length, 8);
  return bytes + json + body;
}

/// One float32 tensor named `name` holding `values`.
std::string ShardWith(const std::string& name, std::vector<float> values) {
  std::string offsets = std::to_string(0) + ", " + std::to_string(values.size() * 4);
  const std::string json =
      "{\"" + name + "\": {\"dtype\": \"F32\", \"shape\": [" +
      std::to_string(values.size()) + "], \"data_offsets\": [" + offsets + "]}}";
  std::string body(values.size() * 4, '\0');
  std::memcpy(body.data(), values.data(), body.size());
  return Frame(json, body);
}

TEST(CheckpointReaderTest, OpensSingleFileCheckpoint) {
  const std::string dir = ::testing::TempDir() + "/inferx_cr_single";
  std::filesystem::remove_all(dir);
  WriteFile(dir + "/model.safetensors", ShardWith("w", {1.f, 2.f, 3.f}));

  auto reader = CheckpointReader::Open(dir);
  ASSERT_TRUE(reader.ok()) << reader.status();
  EXPECT_EQ(reader->Names(), (std::vector<std::string>{"w"}));

  auto tensor = reader->GetTensor("w");
  ASSERT_TRUE(tensor.ok()) << tensor.status();
  EXPECT_EQ(tensor->GetDataType(), DataType::kFloat32);
  std::vector<float> out(3);
  std::memcpy(out.data(), tensor->Data(), 12);
  EXPECT_EQ(out, (std::vector<float>{1.f, 2.f, 3.f}));
  EXPECT_FALSE(reader->GetTensor("absent").ok());
}

TEST(CheckpointReaderTest, GlobsShardsAndSkipsAuxiliaryFiles) {
  const std::string dir = ::testing::TempDir() + "/inferx_cr_shards";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  WriteFile(dir + "/shard0.safetensors", ShardWith("a", {7.f}));
  WriteFile(dir + "/shard1.safetensors", ShardWith("c", {8.f, 9.f}));
  WriteFile(dir + "/optimizer.safetensors", ShardWith("ignored", {0.f}));
  WriteFile(dir + "/adapter.safetensors", ShardWith("also_ignored", {0.f}));
  WriteFile(dir + "/config.json", "{}");

  auto reader = CheckpointReader::Open(dir);
  ASSERT_TRUE(reader.ok()) << reader.status();
  // Shard order follows the filesystem's directory order; only membership
  // is meaningful.
  std::vector<std::string> names = reader->Names();
  std::sort(names.begin(), names.end());
  EXPECT_EQ(names, (std::vector<std::string>{"a", "c"}));

  auto tensor = reader->GetTensor("c");
  ASSERT_TRUE(tensor.ok()) << tensor.status();
  std::vector<float> out(2);
  std::memcpy(out.data(), tensor->Data(), 8);
  EXPECT_EQ(out, (std::vector<float>{8.f, 9.f}));
  EXPECT_FALSE(reader->GetTensor("ignored").ok());
}

TEST(CheckpointReaderTest, EmptyDirectoryIsNotFound) {
  const std::string dir = ::testing::TempDir() + "/inferx_cr_empty";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  EXPECT_FALSE(CheckpointReader::Open(dir).ok());
  EXPECT_FALSE(CheckpointReader::Open(dir + "/does_not_exist").ok());
}

}  // namespace
}  // namespace models
}  // namespace inferx
