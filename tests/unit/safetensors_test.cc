#include "inferx/model/safetensors.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/tensor.h"

namespace inferx::model {
namespace {

// Builds a safetensors file in a temp directory so the format handling is
// tested without depending on a multi-gigabyte download. The real checkpoint is
// exercised separately, and skipped when absent.
class TempCheckpoint {
 public:
  TempCheckpoint() {
    char tmpl[] = "/tmp/inferx_st_XXXXXX";
    dir_ = ::mkdtemp(tmpl) ? std::string(tmpl) : std::string();
  }

  ~TempCheckpoint() {
    for (const std::string& f : files_) ::remove(f.c_str());
    if (!dir_.empty()) ::rmdir(dir_.c_str());
  }

  const std::string& dir() const { return dir_; }

  // Writes `header` as the JSON header and `data` as the tensor section.
  std::string Write(const std::string& name, const std::string& header,
                    const std::vector<std::byte>& data) {
    const std::string path = dir_ + "/" + name;

    std::ofstream out(path, std::ios::binary);
    const uint64_t len = header.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!data.empty()) {
      out.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
    }
    out.close();

    files_.push_back(path);
    return path;
  }

  void WriteText(const std::string& name, const std::string& text) {
    const std::string path = dir_ + "/" + name;
    std::ofstream out(path);
    out << text;
    out.close();
    files_.push_back(path);
  }

 private:
  std::string dir_;
  std::vector<std::string> files_;
};

std::vector<std::byte> Bytes(std::initializer_list<int> vals) {
  std::vector<std::byte> out;
  for (int v : vals) out.push_back(static_cast<std::byte>(v));
  return out;
}

TEST(Safetensors, ReadsASingleFile) {
  TempCheckpoint tmp;
  ASSERT_FALSE(tmp.dir().empty());

  // Two f32 tensors: a [2,2] and a [2].
  std::vector<std::byte> data(6 * sizeof(float));
  const float values[6] = {1, 2, 3, 4, 5, 6};
  std::memcpy(data.data(), values, data.size());

  const std::string path = tmp.Write(
      "model.safetensors",
      R"({"__metadata__":{"format":"pt"},)"
      R"("w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},)"
      R"("b":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}})",
      data);

  auto ckpt = Checkpoint::OpenFile(path);
  ASSERT_TRUE(ckpt.ok()) << ckpt.status();

  EXPECT_EQ(ckpt->size(), 2u);  // __metadata__ is not a tensor
  EXPECT_EQ(ckpt->TotalBytes(), 24u);
  EXPECT_EQ(ckpt->Names(), (std::vector<std::string>{"b", "w"}));

  auto w = ckpt->Get("w");
  ASSERT_TRUE(w.ok()) << w.status();
  EXPECT_EQ(w->dtype(), DataType::kFloat);
  EXPECT_EQ(w->shape(), Shape({2, 2}));

  const float* wd = static_cast<const float*>(w->data());
  EXPECT_FLOAT_EQ(wd[0], 1.0f);
  EXPECT_FLOAT_EQ(wd[3], 4.0f);

  auto b = ckpt->Get("b");
  ASSERT_TRUE(b.ok()) << b.status();
  const float* bd = static_cast<const float*>(b->data());
  EXPECT_FLOAT_EQ(bd[0], 5.0f);
  EXPECT_FLOAT_EQ(bd[1], 6.0f);
}

// The property that makes loading a 3B model viable: no copy, no allocation.
// Every tensor points into the mapping, so two tensors from one file are
// pointer-adjacent and their Storage is borrowed rather than owned.
TEST(Safetensors, TensorsBorrowTheMappingRatherThanCopying) {
  TempCheckpoint tmp;
  ASSERT_FALSE(tmp.dir().empty());

  std::vector<std::byte> data(6 * sizeof(float), std::byte{0});
  const std::string path = tmp.Write(
      "model.safetensors",
      R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},)"
      R"("z":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}})",
      data);

  auto ckpt = Checkpoint::OpenFile(path);
  ASSERT_TRUE(ckpt.ok()) << ckpt.status();

  auto a = ckpt->Get("a");
  auto z = ckpt->Get("z");
  ASSERT_TRUE(a.ok() && z.ok());

  EXPECT_TRUE(a->storage()->IsBorrowed());
  EXPECT_TRUE(z->storage()->IsBorrowed());

  // 'z' begins exactly 16 bytes after 'a' -- same mapping, no copy.
  const auto* ap = static_cast<const std::byte*>(a->data());
  const auto* zp = static_cast<const std::byte*>(z->data());
  EXPECT_EQ(zp - ap, 16);
}

TEST(Safetensors, ReadsAShardedCheckpointThroughItsIndex) {
  TempCheckpoint tmp;
  ASSERT_FALSE(tmp.dir().empty());

  tmp.Write("model-00001-of-00002.safetensors",
            R"({"first":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}})",
            Bytes({1, 2, 3, 4}));
  tmp.Write("model-00002-of-00002.safetensors",
            R"({"second":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}})",
            Bytes({9, 8}));

  tmp.WriteText("model.safetensors.index.json",
                R"({"metadata":{"total_size":6},"weight_map":{)"
                R"("first":"model-00001-of-00002.safetensors",)"
                R"("second":"model-00002-of-00002.safetensors"}})");

  auto ckpt = Checkpoint::Open(tmp.dir());
  ASSERT_TRUE(ckpt.ok()) << ckpt.status();

  // Both shards collapse into one flat namespace -- callers never learn how the
  // publisher chose to split the file.
  EXPECT_EQ(ckpt->size(), 2u);
  EXPECT_EQ(ckpt->shard_paths().size(), 2u);

  auto first = ckpt->Get("first");
  auto second = ckpt->Get("second");
  ASSERT_TRUE(first.ok() && second.ok());

  EXPECT_EQ(static_cast<const uint8_t*>(first->data())[3], 4);
  EXPECT_EQ(static_cast<const uint8_t*>(second->data())[0], 9);
}

TEST(Safetensors, RejectsOffsetsPastTheEndOfTheFile) {
  TempCheckpoint tmp;
  ASSERT_FALSE(tmp.dir().empty());

  // Claims 400 bytes of data; the file has 4. A truncated download looks
  // exactly like this, and must fail at load rather than segfault later.
  const std::string path = tmp.Write(
      "model.safetensors",
      R"({"t":{"dtype":"U8","shape":[400],"data_offsets":[0,400]}})",
      Bytes({1, 2, 3, 4}));

  auto ckpt = Checkpoint::OpenFile(path);
  EXPECT_FALSE(ckpt.ok());
  EXPECT_EQ(ckpt.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(Safetensors, RejectsAShapeThatDisagreesWithTheByteRange) {
  TempCheckpoint tmp;
  ASSERT_FALSE(tmp.dir().empty());

  // [4] f32 needs 16 bytes; the range says 8. One of the two is being misread.
  const std::string path = tmp.Write(
      "model.safetensors",
      R"({"t":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}})",
      std::vector<std::byte>(8, std::byte{0}));

  auto ckpt = Checkpoint::OpenFile(path);
  ASSERT_FALSE(ckpt.ok());
  EXPECT_NE(ckpt.status().message().find("needs"), std::string_view::npos)
      << ckpt.status();
}

TEST(Safetensors, RejectsUnknownDtype) {
  TempCheckpoint tmp;
  ASSERT_FALSE(tmp.dir().empty());

  const std::string path = tmp.Write(
      "model.safetensors",
      R"({"t":{"dtype":"WAT","shape":[1],"data_offsets":[0,1]}})",
      Bytes({0}));

  EXPECT_FALSE(Checkpoint::OpenFile(path).ok());
}

TEST(Safetensors, MissingTensorIsNotFound) {
  TempCheckpoint tmp;
  ASSERT_FALSE(tmp.dir().empty());

  const std::string path = tmp.Write(
      "model.safetensors",
      R"({"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})", Bytes({7}));

  auto ckpt = Checkpoint::OpenFile(path);
  ASSERT_TRUE(ckpt.ok()) << ckpt.status();

  EXPECT_TRUE(ckpt->Contains("t"));
  EXPECT_FALSE(ckpt->Contains("nope"));
  EXPECT_EQ(ckpt->Get("nope").status().code(), absl::StatusCode::kNotFound);
}

TEST(Safetensors, GetCheckedNamesBothShapes) {
  TempCheckpoint tmp;
  ASSERT_FALSE(tmp.dir().empty());

  const std::string path = tmp.Write(
      "model.safetensors",
      R"({"t":{"dtype":"U8","shape":[2,3],"data_offsets":[0,6]}})",
      Bytes({1, 2, 3, 4, 5, 6}));

  auto ckpt = Checkpoint::OpenFile(path);
  ASSERT_TRUE(ckpt.ok()) << ckpt.status();

  EXPECT_TRUE(ckpt->GetChecked("t", Shape({2, 3})).ok());

  const auto bad = ckpt->GetChecked("t", Shape({3, 2}));
  ASSERT_FALSE(bad.ok());
  EXPECT_NE(bad.status().message().find("[2, 3]"), std::string_view::npos)
      << bad.status();
}

TEST(Safetensors, EmptyDirectoryIsNotFound) {
  TempCheckpoint tmp;
  ASSERT_FALSE(tmp.dir().empty());

  EXPECT_EQ(Checkpoint::Open(tmp.dir()).status().code(),
            absl::StatusCode::kNotFound);
}

TEST(Safetensors, DtypeSpellings) {
  EXPECT_EQ(*SafetensorsDataType("BF16"), DataType::kBFloat16);
  EXPECT_EQ(*SafetensorsDataType("F16"), DataType::kFloat16);
  EXPECT_EQ(*SafetensorsDataType("F32"), DataType::kFloat);
  EXPECT_EQ(*SafetensorsDataType("I64"), DataType::kInt64);
  EXPECT_FALSE(SafetensorsDataType("float16").ok());  // not the format's name
}

}  // namespace
}  // namespace inferx::model
