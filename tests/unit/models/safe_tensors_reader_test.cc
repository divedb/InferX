#include "inferx/models/safe_tensors_reader.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/tensor.h"

namespace inferx {
namespace {

/// \brief One fresh directory per test, removed in TearDown.
class SafeTensorReaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    dir_ = std::filesystem::temp_directory_path() /
           ("inferx_safe_tensors_reader_" + std::to_string(::getpid()) + "_" + test_name);
    std::filesystem::create_directories(dir_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  /// \brief Writes a safetensors file: an 8-byte little-endian header length,
  ///        the JSON header itself, then the data blob.
  ///
  /// The length prefix is written byte by byte so the test does not depend on
  /// host byte order. Returns the path of the file written.
  std::filesystem::path WriteSafetensors(std::string_view name, std::string_view header_json,
                                         std::span<const std::byte> blob) {
    std::vector<std::byte> bytes(8 + header_json.size() + blob.size(), std::byte{0});
    for (unsigned i = 0; i < 8; ++i) {
      bytes[i] = static_cast<std::byte>(header_json.size() >> (8 * i));
    }
    std::memcpy(bytes.data() + 8, header_json.data(), header_json.size());
    std::memcpy(bytes.data() + 8 + header_json.size(), blob.data(), blob.size());

    const auto path = dir_ / name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      ADD_FAILURE() << "cannot open " << path;
      return {};
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return path;
  }

  /// \brief Appends `value`'s object representation to `blob`.
  template <typename T>
  void AppendPod(std::vector<std::byte>& blob, const T& value) {
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    blob.insert(blob.end(), bytes, bytes + sizeof(T));
  }

  /// \brief Asserts that `result` failed with exactly `code`.
  void ExpectError(const std::expected<SafeTensorReader, SafeTensorError>& result,
                   SafeTensorErrorCode code) {
    ASSERT_FALSE(result.has_value());

    if (!result.has_value()) {
      EXPECT_EQ(result.error().code, code) << result.error().message;
    }
  }

  std::filesystem::path dir_;
};

TEST_F(SafeTensorReaderTest, OpensFileAndReadsTensorData) {
  const float weights[] = {1.5f, -2.0f, 0.25f, 100.0f};
  const std::int64_t count = 42;

  std::vector<std::byte> blob;
  for (const float w : weights) {
    AppendPod(blob, w);
  }
  AppendPod(blob, count);

  const std::string header =
      R"({"__metadata__":{"format":"pt"},)"
      R"("weights":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},)"
      R"("count":{"dtype":"I64","shape":[1],"data_offsets":[16,24]}})";
  const auto path = WriteSafetensors("model.safetensors", header, blob);
  const auto reader = SafeTensorReader::Open(path);

  ASSERT_TRUE(reader.has_value());
  EXPECT_EQ(reader->FileSize(), 8 + header.size() + blob.size());

  const auto names = reader->TensorNames();
  ASSERT_EQ(names.size(), 2u);
  EXPECT_EQ(names[0], "weights");
  EXPECT_EQ(names[1], "count");

  const auto w = reader->GetTensor("weights");
  ASSERT_TRUE(w.has_value());
  EXPECT_EQ(w->GetDataType(), DataType::kFloat32);
  EXPECT_EQ(w->Rank(), 2);
  EXPECT_EQ(w->Dim(0), 2);
  EXPECT_EQ(w->Dim(1), 2);
  EXPECT_EQ(w->Numel(), 4);
  EXPECT_EQ(w->NBytes(), 16);
  EXPECT_TRUE(w->IsCpu());
  const float* w_data = w->DataAs<float>();
  ASSERT_NE(w_data, nullptr);
  for (std::size_t i = 0; i < std::size(weights); ++i) {
    EXPECT_FLOAT_EQ(w_data[i], weights[i]) << "element " << i;
  }

  const auto c = reader->GetTensor("count");
  ASSERT_TRUE(c.has_value());
  EXPECT_EQ(c->GetDataType(), DataType::kInt64);
  const std::int64_t* c_data = c->DataAs<std::int64_t>();
  ASSERT_NE(c_data, nullptr);
  EXPECT_EQ(c_data[0], count);

  EXPECT_EQ(reader->Metadata().at("format"), "pt");
}

TEST_F(SafeTensorReaderTest, GetTensorForMissingNameReturnsNullopt) {
  const auto path = WriteSafetensors(
      "one.safetensors", R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
      std::vector<std::byte>(4));
  const auto reader = SafeTensorReader::Open(path);

  ASSERT_TRUE(reader.has_value());
  EXPECT_FALSE(reader->HasTensor("missing"));
  EXPECT_EQ(reader->GetTensor("missing"), std::nullopt);
}

TEST_F(SafeTensorReaderTest, OpensFileWithEmptyDataBlob) {
  const std::string header = R"({"a":{"dtype":"F32","shape":[0],"data_offsets":[0,0]}})";
  const auto path = WriteSafetensors("empty_blob.safetensors", header, {});
  const auto reader = SafeTensorReader::Open(path);

  ASSERT_TRUE(reader.has_value());
  EXPECT_EQ(reader->FileSize(), 8 + header.size());
  const auto tensor = reader->GetTensor("a");
  ASSERT_TRUE(tensor.has_value());
  EXPECT_EQ(tensor->Numel(), 0);
  EXPECT_EQ(tensor->NBytes(), 0);
}

TEST_F(SafeTensorReaderTest, MovedReaderKeepsMappingUsable) {
  std::vector<std::byte> blob;
  AppendPod(blob, 3.14f);

  const auto path = WriteSafetensors(
      "move.safetensors", R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", blob);
  auto opened = SafeTensorReader::Open(path);
  ASSERT_TRUE(opened.has_value());

  SafeTensorReader reader = std::move(*opened);
  const auto tensor = reader.GetTensor("a");
  ASSERT_TRUE(tensor.has_value());
  ASSERT_NE(tensor->DataAs<float>(), nullptr);
  EXPECT_FLOAT_EQ(tensor->DataAs<float>()[0], 3.14f);
}

TEST_F(SafeTensorReaderTest, OpenMissingFileReturnsFileNotFound) {
  const auto result = SafeTensorReader::Open(dir_ / "absent.safetensors");
  ExpectError(result, SafeTensorErrorCode::kFileNotFound);
}

TEST_F(SafeTensorReaderTest, OpenEmptyFileReturnsIoError) {
  // A zero-byte file fails in the mapping layer -- it never reaches
  // SafeTensorsView::Parse(), which would report kHeaderTooSmall for a
  // merely-too-small (but nonzero) file instead.
  std::ofstream(dir_ / "empty.safetensors", std::ios::binary | std::ios::trunc);
  const auto result = SafeTensorReader::Open(dir_ / "empty.safetensors");
  ExpectError(result, SafeTensorErrorCode::kIoError);
}

TEST_F(SafeTensorReaderTest, OpenDirectoryReturnsIoError) {
  const auto result = SafeTensorReader::Open(dir_);
  ExpectError(result, SafeTensorErrorCode::kIoError);
}

TEST_F(SafeTensorReaderTest, PropagatesJsonErrorFromView) {
  const auto path = WriteSafetensors("garbage.safetensors", "{", {});
  ExpectError(SafeTensorReader::Open(path), SafeTensorErrorCode::kJsonError);
}

TEST_F(SafeTensorReaderTest, PropagatesIncompleteBufferErrorFromView) {
  // The header covers only [0,4] of an 8-byte blob; Parse() rejects the
  // uncovered tail and Open() must forward that verdict unchanged.
  const auto path = WriteSafetensors(
      "tail.safetensors", R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})",
      std::vector<std::byte>(8));
  ExpectError(SafeTensorReader::Open(path), SafeTensorErrorCode::kMetadataIncompleteBuffer);
}

TEST_F(SafeTensorReaderTest, RejectsTruncatedFile) {
  // The length prefix claims a header too large for the bytes that follow:
  // the classic truncated-download shape.
  std::vector<std::byte> bytes(8, std::byte{0});
  const std::uint64_t claimed = 64;
  for (unsigned i = 0; i < 8; ++i) {
    bytes[i] = static_cast<std::byte>(claimed >> (8 * i));
  }

  const auto path = dir_ / "truncated.safetensors";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  out.close();

  ExpectError(SafeTensorReader::Open(path), SafeTensorErrorCode::kInvalidHeaderLength);
}

}  // namespace
}  // namespace inferx
