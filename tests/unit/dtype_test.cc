#include "inferx/core/dtype.h"

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <string_view>

namespace inferx {
namespace {

TEST(DataType, BitWidth) {
  EXPECT_EQ(DataTypeBitWidth<DataType::kBool>(), 1);
  EXPECT_EQ(DataTypeBitWidth<DataType::kInt4>(), 4);
  EXPECT_EQ(DataTypeBitWidth<DataType::kInt8>(), 8);
  EXPECT_EQ(DataTypeBitWidth<DataType::kFloat16>(), 16);
  EXPECT_EQ(DataTypeBitWidth<DataType::kFloat>(), 32);
  EXPECT_EQ(DataTypeBitWidth<DataType::kInt64>(), 64);
  EXPECT_EQ(DataTypeBitWidth<DataType::kComplex128>(), 128);
}

TEST(DataType, Name) {
  EXPECT_STREQ(DataTypeName<DataType::kFloat>(), "f32");
  EXPECT_STREQ(DataTypeName<DataType::kBFloat16>(), "bf16");
  EXPECT_STREQ(DataTypeName<DataType::kFloat8E4M3FN>(), "f8e4m3fn");
  EXPECT_STREQ(DataTypeName<DataType::kFloat4E2M1>(), "f4e2m1");
  EXPECT_STREQ(DataTypeName<DataType::kInt4>(), "i4");
  EXPECT_STREQ(DataTypeName<DataType::kComplex64>(), "complex<f32>");
}

// Host storage size is sizeof the backing C++ type, not bit_width / 8.
TEST(DataType, HostStorageSize) {
  // Whole-byte types: size tracks the width.
  EXPECT_EQ(DataTypeSize<DataType::kBool>(), 1);
  EXPECT_EQ(DataTypeSize<DataType::kFloat16>(), 2);
  EXPECT_EQ(DataTypeSize<DataType::kFloat>(), 4);
  EXPECT_EQ(DataTypeSize<DataType::kDouble>(), 8);
  EXPECT_EQ(DataTypeSize<DataType::kInt64>(), 8);
  EXPECT_EQ(DataTypeSize<DataType::kComplex64>(),
            8);  // 64-bit logical, 8 bytes
  EXPECT_EQ(DataTypeSize<DataType::kComplex128>(), 16);

  // Sub-byte types (< 8 bits) still occupy a whole byte on host -- there is no
  // 4-bit C++ type. A packed byte count must round bit widths up separately;
  // DataTypeSize is per-element host storage, not a packed-bytes helper.
  EXPECT_EQ(DataTypeSize<DataType::kInt2>(), 1);
  EXPECT_EQ(DataTypeSize<DataType::kInt4>(), 1);
  EXPECT_EQ(DataTypeSize<DataType::kUInt4>(), 1);
  EXPECT_EQ(DataTypeSize<DataType::kFloat4E2M1>(), 1);
}

TEST(DataType, Alignment) {
  EXPECT_EQ(DataTypeAlign<DataType::kBool>(), 1);
  EXPECT_EQ(DataTypeAlign<DataType::kFloat16>(), 2);
  EXPECT_EQ(DataTypeAlign<DataType::kFloat>(), 4);
  EXPECT_EQ(DataTypeAlign<DataType::kInt64>(), 8);
}

}  // namespace
}  // namespace inferx
