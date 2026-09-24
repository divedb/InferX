#include "inferx/models/safe_tensors_view.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/tensor.h"

namespace inferx {
namespace {

/// \brief Builds a safetensors buffer: an 8-byte little-endian header length,
///        the JSON header itself, then `data_bytes` of zeroed data blob.
///
/// The length prefix is written byte by byte so the test does not depend on
/// host byte order.
std::vector<std::byte> MakeBuffer(std::string_view header_json, std::size_t data_bytes) {
  std::vector<std::byte> bytes(8 + header_json.size() + data_bytes, std::byte{0});

  for (unsigned i = 0; i < 8; ++i) {
    bytes[i] = static_cast<std::byte>(header_json.size() >> (8 * i));
  }

  std::memcpy(bytes.data() + 8, header_json.data(), header_json.size());

  return bytes;
}

/// \brief Asserts that `result` failed with exactly `code`.
void ExpectError(const std::expected<SafeTensorsView, SafeTensorError>& result,
                 SafeTensorErrorCode code) {
  ASSERT_FALSE(result.has_value());

  if (!result.has_value()) {
    EXPECT_EQ(result.error().code, code) << result.error().message;
  }
}

/// \brief One-tensor header with the given dtype string, shape, and offsets.
std::string OneTensor(std::string_view dtype, std::string_view shape,
                      std::string_view offsets) {
  return std::string("{\"a\":{\"dtype\":\"") + dtype.data() + "\",\"shape\":" + shape.data() +
         ",\"data_offsets\":" + offsets.data() + "}}";
}

TEST(SafeTensorsViewTest, ParsesSingleTensor) {
  const std::string header = OneTensor("F32", "[2]", "[0,8]");
  const auto buffer = MakeBuffer(header, 8);
  const auto view = SafeTensorsView::Parse(buffer);

  ASSERT_TRUE(view.has_value());
  ASSERT_TRUE(view->HasTensor("a"));
  EXPECT_FALSE(view->HasTensor("b"));

  const auto tensor = view->GetTensor("a");
  ASSERT_TRUE(tensor.has_value());
  EXPECT_EQ(tensor->GetDataType(), DataType::kFloat32);
  EXPECT_EQ(tensor->Rank(), 1);
  EXPECT_EQ(tensor->Dim(0), 2);
  EXPECT_EQ(tensor->Numel(), 2);
  EXPECT_EQ(tensor->NBytes(), 8);
  EXPECT_TRUE(tensor->IsCpu());
  // The tensor borrows the caller's buffer, starting right after the header.
  EXPECT_EQ(tensor->Data(), static_cast<const void*>(buffer.data() + 8 + header.size()));

  const auto names = view->TensorNames();
  ASSERT_EQ(names.size(), 1u);
  EXPECT_EQ(names[0], "a");
}

TEST(SafeTensorsViewTest, GetTensorForMissingNameReturnsNullopt) {
  const auto view = SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[1]", "[0,4]"), 4));
  ASSERT_TRUE(view.has_value());
  EXPECT_FALSE(view->HasTensor("missing"));
  EXPECT_EQ(view->GetTensor("missing"), std::nullopt);
}

TEST(SafeTensorsViewTest, TensorNamesPreserveHeaderOrder) {
  const std::string header =
      "{\"zeta\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
      "\"alpha\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]},"
      "\"mid\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}";
  const auto view = SafeTensorsView::Parse(MakeBuffer(header, 12));

  ASSERT_TRUE(view.has_value());
  const auto names = view->TensorNames();
  ASSERT_EQ(names.size(), 3u);
  EXPECT_EQ(names[0], "zeta");
  EXPECT_EQ(names[1], "alpha");
  EXPECT_EQ(names[2], "mid");
}

TEST(SafeTensorsViewTest, AcceptsEverySpecDtypeName) {
  const struct {
    const char* name;
    DataType dtype;
  } kSpec[] = {
      {"BOOL", DataType::kBool},
      {"F4", DataType::kFloat4E2M1},
      {"F6_E2M3", DataType::kFloat6E2M3},
      {"F6_E3M2", DataType::kFloat6E3M2},
      {"U8", DataType::kUInt8},
      {"I8", DataType::kInt8},
      {"F8_E5M2", DataType::kFloat8E5M2},
      {"F8_E4M3", DataType::kFloat8E4M3FN},
      {"F8_E8M0", DataType::kFloat8E8M0},
      {"F8_E4M3FNUZ", DataType::kFloat8E4M3FNUZ},
      {"F8_E5M2FNUZ", DataType::kFloat8E5M2FNUZ},
      {"I16", DataType::kInt16},
      {"U16", DataType::kUInt16},
      {"F16", DataType::kFloat16},
      {"BF16", DataType::kBFloat16},
      {"I32", DataType::kInt32},
      {"U32", DataType::kUInt32},
      {"F32", DataType::kFloat32},
      {"C64", DataType::kComplex64},
      {"F64", DataType::kFloat64},
      {"I64", DataType::kInt64},
      {"U64", DataType::kUInt64},
  };

  for (const auto& [name, dtype] : kSpec) {
    // Eight elements of every dtype land on a whole byte count, including
    // the packed sub-byte formats (8 x F4 = 4 bytes, 8 x F6 = 6 bytes).
    const int64_t span = DataTypeByteSize(dtype, 8);
    const auto view = SafeTensorsView::Parse(
        MakeBuffer(OneTensor(name, "[8]", "[0," + std::to_string(span) + "]"),
                   static_cast<std::size_t>(span)));
    const std::string context = std::string(name) + ": " +
                                (view.has_value() ? std::string("ok") : view.error().message);
    ASSERT_TRUE(view.has_value()) << context;
    ASSERT_TRUE(view->HasTensor("a")) << context;
    EXPECT_EQ(view->GetTensor("a")->GetDataType(), dtype) << context;
  }
}

TEST(SafeTensorsViewTest, PacksSubByteTensorsWholeBytes) {
  const auto view = SafeTensorsView::Parse(MakeBuffer(OneTensor("F4", "[1,2]", "[0,1]"), 1));

  ASSERT_TRUE(view.has_value());
  const auto tensor = view->GetTensor("a");
  ASSERT_TRUE(tensor.has_value());
  EXPECT_EQ(tensor->GetDataType(), DataType::kFloat4E2M1);
  EXPECT_EQ(tensor->Dim(1), 2);
  EXPECT_EQ(tensor->NBytes(), 1);
}

TEST(SafeTensorsViewTest, ParsesScalarTensorWithEmptyShape) {
  const auto view = SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[]", "[0,4]"), 4));

  ASSERT_TRUE(view.has_value());
  const auto tensor = view->GetTensor("a");
  ASSERT_TRUE(tensor.has_value());
  EXPECT_EQ(tensor->Rank(), 0);
  EXPECT_EQ(tensor->Numel(), 1);
  EXPECT_EQ(tensor->NBytes(), 4);
}

TEST(SafeTensorsViewTest, ParsesZeroElementTensorCoveringEmptyBlob) {
  const auto view = SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[0]", "[0,0]"), 0));

  ASSERT_TRUE(view.has_value());
  const auto tensor = view->GetTensor("a");
  ASSERT_TRUE(tensor.has_value());
  EXPECT_EQ(tensor->Numel(), 0);
  EXPECT_EQ(tensor->NBytes(), 0);
}

TEST(SafeTensorsViewTest, ParsesMetadataObject) {
  const std::string header =
      "{\"__metadata__\":{\"format\":\"pt\",\"source\":\"test\"},"
      "\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
  const auto view = SafeTensorsView::Parse(MakeBuffer(header, 4));

  ASSERT_TRUE(view.has_value());
  const auto& metadata = view->Metadata();
  EXPECT_EQ(metadata.at("format"), "pt");
  EXPECT_EQ(metadata.at("source"), "test");
  // __metadata__ is not a tensor.
  EXPECT_EQ(view->TensorNames().size(), 1u);
  EXPECT_FALSE(view->HasTensor("__metadata__"));
}

TEST(SafeTensorsViewTest, RejectsUnknownDtype) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(OneTensor("F16X", "[2]", "[0,8]"), 8)),
              SafeTensorErrorCode::kInvalidHeaderDeserialization);
}

TEST(SafeTensorsViewTest, RejectsEntryMissingFields) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(R"({"a":{"dtype":"F32"}})", 0)),
              SafeTensorErrorCode::kJsonError);
}

TEST(SafeTensorsViewTest, RejectsNonStringDtype) {
  ExpectError(SafeTensorsView::Parse(
                  MakeBuffer(R"({"a":{"dtype":1,"shape":[1],"data_offsets":[0,4]}})", 4)),
              SafeTensorErrorCode::kJsonError);
}

TEST(SafeTensorsViewTest, RejectsNonIntegerShapeExtent) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[1.5]", "[0,4]"), 4)),
              SafeTensorErrorCode::kJsonError);
}

TEST(SafeTensorsViewTest, RejectsNegativeShapeExtent) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[-2]", "[0,8]"), 8)),
              SafeTensorErrorCode::kTensorInvalidInfo);
}

TEST(SafeTensorsViewTest, RejectsMalformedDataOffsets) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[1]", "[0,4,8]"), 8)),
              SafeTensorErrorCode::kInvalidOffset);
}

TEST(SafeTensorsViewTest, RejectsOffsetsBeyondDataBlob) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[2]", "[0,16]"), 8)),
              SafeTensorErrorCode::kInvalidOffset);
}

TEST(SafeTensorsViewTest, RejectsBackwardsOffsets) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[1]", "[4,0]"), 8)),
              SafeTensorErrorCode::kInvalidOffset);
}

TEST(SafeTensorsViewTest, RejectsSpanNotMatchingShapeAndDtype) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[3]", "[0,8]"), 8)),
              SafeTensorErrorCode::kTensorInvalidInfo);
}

TEST(SafeTensorsViewTest, RejectsSubByteTensorStraddlingByteBoundary) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(OneTensor("F4", "[1,3]", "[0,2]"), 2)),
              SafeTensorErrorCode::kMisalignedSlice);
}

TEST(SafeTensorsViewTest, RejectsSubByteTensorWithMisalignedInnermostExtent) {
  // 6 F4 elements span whole bytes, but rows of one nibble cannot be
  // addressed as Tensors: TensorSpec requires the innermost extent to fill
  // whole bytes, so the header is rejected at parse time.
  ExpectError(SafeTensorsView::Parse(MakeBuffer(OneTensor("F4", "[6,1]", "[0,3]"), 3)),
              SafeTensorErrorCode::kMisalignedSlice);
}

TEST(SafeTensorsViewTest, RejectsShapeWhoseByteCountOverflows) {
  // 2^59 elements x 32 bits overflows int64 before any span could match.
  ExpectError(
      SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[576460752303423488]", "[0,8]"), 8)),
      SafeTensorErrorCode::kValidationOverflow);
}

TEST(SafeTensorsViewTest, RejectsGapBetweenTensors) {
  const std::string header =
      "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
      "\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[5,9]}}";
  ExpectError(SafeTensorsView::Parse(MakeBuffer(header, 9)),
              SafeTensorErrorCode::kInvalidOffset);
}

TEST(SafeTensorsViewTest, RejectsUncoveredBufferTail) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(OneTensor("F32", "[1]", "[0,4]"), 8)),
              SafeTensorErrorCode::kMetadataIncompleteBuffer);
}

TEST(SafeTensorsViewTest, RejectsNonObjectMetadata) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(R"({"__metadata__":[1]})", 0)),
              SafeTensorErrorCode::kJsonError);
}

TEST(SafeTensorsViewTest, RejectsNonStringMetadataValue) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer(R"({"__metadata__":{"k":1}})", 0)),
              SafeTensorErrorCode::kJsonError);
}

TEST(SafeTensorsViewTest, RejectsInvalidJsonHeader) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer("{", 0)), SafeTensorErrorCode::kJsonError);
}

TEST(SafeTensorsViewTest, RejectsNonObjectHeader) {
  ExpectError(SafeTensorsView::Parse(MakeBuffer("[1,2]", 0)), SafeTensorErrorCode::kJsonError);
}

TEST(SafeTensorsViewTest, RejectsHeaderLengthBeyondBufferSize) {
  // Ten bytes whose length prefix claims a maximal header: it cannot fit.
  const std::vector<std::byte> buffer(10, std::byte{0xFF});
  ExpectError(SafeTensorsView::Parse(buffer), SafeTensorErrorCode::kInvalidHeaderLength);
}

TEST(SafeTensorsViewTest, RejectsBufferSmallerThanLengthPrefix) {
  const std::vector<std::byte> buffer(4, std::byte{0});
  ExpectError(SafeTensorsView::Parse(buffer), SafeTensorErrorCode::kHeaderTooSmall);
}

TEST(SafeTensorsViewTest, RejectsHeaderOverSanityCap) {
  // A real-sized buffer whose header length claims just over the 100 MiB cap;
  // the claim is checked before the JSON parser ever runs.
  constexpr std::uint64_t kClaimed = 100ull * 1024 * 1024 + 1;
  std::vector<std::byte> buffer(8 + static_cast<std::size_t>(kClaimed), std::byte{0});
  for (unsigned i = 0; i < 8; ++i) {
    buffer[i] = static_cast<std::byte>(kClaimed >> (8 * i));
  }
  ExpectError(SafeTensorsView::Parse(buffer), SafeTensorErrorCode::kHeaderTooLarge);
}

TEST(SafeTensorsViewTest, DefaultViewIsEmpty) {
  const SafeTensorsView view;
  EXPECT_TRUE(view.TensorNames().empty());
  EXPECT_FALSE(view.HasTensor("a"));
  EXPECT_EQ(view.GetTensor("a"), std::nullopt);
  EXPECT_TRUE(view.Metadata().empty());
}

}  // namespace
}  // namespace inferx
