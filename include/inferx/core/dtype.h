#pragma once

#include <complex>
#include <cstdint>
#include <string_view>

namespace inferx {

template <unsigned int NBits>
struct DummyUInt {};

template <unsigned int NBits>
struct DummyInt {};

/// \brief Runtime data type for tensor elements.
enum class DataType : std::uint8_t {
  kUndefined = 0,
  kBool,
  kFloat,
  kUInt2,
  kInt2,
  kUInt4,
  kInt4,
  kUInt8,
  kInt8,
  kUInt16,
  kInt16,
  kUInt32,
  kInt32,
  kUInt64,
  kInt64,
  kFloat16,
  kDouble,
  kComplex64,
  kComplex128,
  kBFloat16,
  kFloat8E4M3FN,
  kFloat8E4M3FNUZ,
  kFloat8E5M2,
  kFloat8E5M2FNUZ,
  kFloat8E8M0,
  kFloat4E2M1,
};

template <DataType>
struct DummyFloat {};

#define INFERX_FOR_EACH_DATA_TYPE(X)                                         \
  X(kBool, bool, "i1", 1)                                                    \
  X(kFloat, float, "f32", 32)                                                \
  X(kUInt2, DummyUInt<2>, "u2", 2)                                           \
  X(kInt2, DummyInt<2>, "i2", 2)                                             \
  X(kUInt4, DummyUInt<4>, "u4", 4)                                           \
  X(kInt4, DummyInt<4>, "i4", 4)                                             \
  X(kUInt8, std::uint8_t, "u8", 8)                                           \
  X(kInt8, std::int8_t, "i8", 8)                                             \
  X(kUInt16, std::uint16_t, "u16", 16)                                       \
  X(kInt16, std::int16_t, "i16", 16)                                         \
  X(kUInt32, std::uint32_t, "u32", 32)                                       \
  X(kInt32, std::int32_t, "i32", 32)                                         \
  X(kUInt64, std::uint64_t, "u64", 64)                                       \
  X(kInt64, std::int64_t, "i64", 64)                                         \
  X(kFloat16, std::uint16_t, "f16", 16)                                      \
  X(kDouble, double, "f64", 64)                                              \
  X(kComplex64, std::complex<float>, "complex<f32>", 64)                     \
  X(kComplex128, std::complex<double>, "complex<f64>", 128)                  \
  X(kBFloat16, std::uint16_t, "bf16", 16)                                    \
  X(kFloat8E4M3FN, DummyFloat<DataType::kFloat8E4M3FN>, "f8e4m3fn", 8)       \
  X(kFloat8E4M3FNUZ, DummyFloat<DataType::kFloat8E4M3FNUZ>, "f8e4m3fnuz", 8) \
  X(kFloat8E5M2, DummyFloat<DataType::kFloat8E5M2>, "f8e5m2", 8)             \
  X(kFloat8E5M2FNUZ, DummyFloat<DataType::kFloat8E5M2FNUZ>, "f8e5m2fnuz", 8) \
  X(kFloat8E8M0, DummyFloat<DataType::kFloat8E8M0>, "f8e8m0", 8)             \
  X(kFloat4E2M1, DummyFloat<DataType::kFloat4E2M1>, "f4e2m1", 4)

template <DataType>
struct DataTypeTrait;

#define X(kind, cpp_type, name, bit_width)               \
  template <>                                            \
  struct DataTypeTrait<DataType::kind> {                 \
    using type = cpp_type;                               \
    static constexpr const char* kName = name;           \
    static constexpr std::size_t kBitWidth = bit_width;  \
    static constexpr std::size_t kSize = sizeof(type);   \
    static constexpr std::size_t kAlign = alignof(type); \
  };
INFERX_FOR_EACH_DATA_TYPE(X)
#undef X

/// \brief Get the logical width in bits of a fixed-width data type.
template <DataType dtype>
constexpr std::size_t DataTypeBitWidth() noexcept {
  static_assert(dtype != DataType::kUndefined,
                "the undefined data type has no bit width");

  return DataTypeTrait<dtype>::kBitWidth;
}

/// \brief Get the host storage size in bytes of a fixed-width data type.
template <DataType dtype>
constexpr std::size_t DataTypeSize() noexcept {
  static_assert(dtype != DataType::kUndefined,
                "the undefined data type has no size");

  return DataTypeTrait<dtype>::kSize;
}

/// \brief Get the host storage alignment in bytes of a fixed-width data type.
template <DataType dtype>
constexpr std::size_t DataTypeAlign() noexcept {
  static_assert(dtype != DataType::kUndefined,
                "the undefined data type has no alignment");

  return DataTypeTrait<dtype>::kAlign;
}

/// \brief Get the human-readable name of a fixed-width data type.
template <DataType dtype>
constexpr const char* DataTypeName() noexcept {
  return DataTypeTrait<dtype>::kName;
}

// The runtime counterparts of the queries above.
//
// The templated forms are for code that knows its dtype at compile time; these
// are for the tensor layer, where the dtype arrives from a model file or a
// request and is only ever a value. Both are generated from the same table, so
// they cannot drift. `kUndefined` has no layout and answers 0 in every width
// query rather than aborting -- callers gate on DataTypeIsValid().

/// \brief Get the human-readable name of `dtype`.
constexpr std::string_view DataTypeName(DataType dtype) noexcept {
  switch (dtype) {
#define X(kind, cpp_type, name, bit_width) \
  case DataType::kind:                     \
    return name;
    INFERX_FOR_EACH_DATA_TYPE(X)
#undef X
    case DataType::kUndefined:
      return "undefined";
  }

  return "undefined";
}

/// \brief Get the logical width of `dtype` in bits, or 0 if it has none.
constexpr std::size_t DataTypeBitWidth(DataType dtype) noexcept {
  switch (dtype) {
#define X(kind, cpp_type, name, bit_width) \
  case DataType::kind:                     \
    return bit_width;
    INFERX_FOR_EACH_DATA_TYPE(X)
#undef X
    case DataType::kUndefined:
      return 0;
  }

  return 0;
}

/// \brief Get the host storage size of one `dtype` element in bytes, or 0 if it
///        has none.
constexpr std::size_t DataTypeSize(DataType dtype) noexcept {
  switch (dtype) {
#define X(kind, cpp_type, name, bit_width) \
  case DataType::kind:                     \
    return sizeof(cpp_type);
    INFERX_FOR_EACH_DATA_TYPE(X)
#undef X
    case DataType::kUndefined:
      return 0;
  }

  return 0;
}

/// \brief True if `dtype` names a type at all.
///
/// `kUndefined` is the zero value a default-constructed view or spec carries;
/// every other DataType has a row in the table above and therefore a width, a
/// size, and a byte count for a given element count.
constexpr bool DataTypeIsValid(DataType dtype) noexcept {
  return dtype != DataType::kUndefined;
}

/// \brief True for types packed several elements to a byte.
///
/// These are the quantized weight formats, which have no addressable C++ type
/// and so cannot be indexed without unpacking. `kBool` is deliberately excluded
/// even though it is one logical bit: `sizeof(bool)` is a whole byte, so a bool
/// tensor is byte-addressable like any other.
constexpr bool DataTypeIsSubByte(DataType dtype) noexcept {
  return DataTypeBitWidth(dtype) < 8 && dtype != DataType::kBool;
}

/// \brief Get the number of bits one `dtype` element occupies in a buffer.
///
/// This is the single definition of layout, and it is not always the logical
/// bit width: a kBool element is one logical bit but a whole byte of storage.
/// Only the packed sub-byte types lay out at their logical width. Anything
/// computing an offset or a byte count must go through this or
/// `DataTypeByteSize()`, never multiply by `DataTypeBitWidth()`.
constexpr std::size_t DataTypeStorageBits(DataType dtype) noexcept {
  return DataTypeIsSubByte(dtype) ? DataTypeBitWidth(dtype)
                                  : DataTypeSize(dtype) * 8;
}

/// \brief Get the number of bytes `count` elements of `dtype` occupy.
///
/// Sub-byte types are packed, so a partial trailing byte is rounded up:
/// DataTypeByteSize(kInt4, 3) is 2. TensorSpec::Verify() rejects sub-byte
/// shapes whose innermost extent would straddle a byte, so that rounding never
/// silently misaligns a row.
constexpr int64_t DataTypeByteSize(DataType dtype, int64_t count) noexcept {
  const int64_t bits = static_cast<int64_t>(DataTypeStorageBits(dtype)) * count;

  return (bits + 7) / 8;
}

/// \brief Maps a C++ type to the DataType it stores.
///
/// Only the types the host actually manipulates are listed. f16, bf16 and the
/// f8/f4 formats are deliberately absent: they have no host arithmetic type
/// here, so there is nothing to map them to that would not be a lie.
template <typename T>
struct DataTypeOf;

#define INFERX_DATA_TYPE_OF(cpp_type, kind)           \
  template <>                                         \
  struct DataTypeOf<cpp_type> {                       \
    static constexpr DataType value = DataType::kind; \
  };

INFERX_DATA_TYPE_OF(bool, kBool)
INFERX_DATA_TYPE_OF(float, kFloat)
INFERX_DATA_TYPE_OF(double, kDouble)
INFERX_DATA_TYPE_OF(std::uint8_t, kUInt8)
INFERX_DATA_TYPE_OF(std::int8_t, kInt8)
INFERX_DATA_TYPE_OF(std::uint16_t, kUInt16)
INFERX_DATA_TYPE_OF(std::int16_t, kInt16)
INFERX_DATA_TYPE_OF(std::uint32_t, kUInt32)
INFERX_DATA_TYPE_OF(std::int32_t, kInt32)
INFERX_DATA_TYPE_OF(std::uint64_t, kUInt64)
INFERX_DATA_TYPE_OF(std::int64_t, kInt64)
INFERX_DATA_TYPE_OF(std::complex<float>, kComplex64)
INFERX_DATA_TYPE_OF(std::complex<double>, kComplex128)

#undef INFERX_DATA_TYPE_OF

template <typename T>
inline constexpr DataType kDataTypeOf = DataTypeOf<T>::value;

}  // namespace inferx
