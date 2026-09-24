#include "inferx/models/safe_tensors_view.h"

#include <boost/endian/conversion.hpp>
#include <limits>
#include <nlohmann/json.hpp>

namespace inferx {

namespace {

/// nlohmann::json's default object type sorts keys alphabetically, which
/// would silently reorder TensorNames() away from the buffer's own header
/// order. ordered_json preserves insertion order (i.e. header order) instead.
using OrderedJson = nlohmann::ordered_json;

constexpr std::size_t kHeaderLengthPrefixBytes = 8;

/// Sanity cap on the JSON header's own size, independent of the file's total size. A
/// safetensors file can legitimately be tens of GB (model weights), so bounding header_len by
/// "buffer size" alone doesn't stop a corrupted or adversarial length field from claiming most
/// of that buffer is "header," which would make the JSON parser attempt to parse gigabytes of
/// garbage before failing. This catches that cheaply, before invoking the parser.
constexpr std::size_t kMaxHeaderBytes = 100 * 1024 * 1024;  // 100 MiB

/// \brief The fields SafeTensorsView::Entry needs, extracted and validated
///        from one tensor's JSON entry.
///
/// A plain struct rather than SafeTensorsView::Entry itself: this lets
/// ParseTensorEntry() live as a free function usable (and unit-testable) on
/// its own, without needing access to SafeTensorsView's private nested
/// Entry type. Parse() converts this into an Entry once parsing succeeds.
struct ParsedTensor {
  DataType dtype;
  Shape shape;
  std::size_t begin;
  std::size_t end;
};

/// \brief Map a safetensors header dtype string to an inferx::DataType.
///
/// This is deliberately separate from DataTypeName(DataType) in dtype.h: the
/// safetensors spec's strings ("F32", "BF16", "F8_E4M3", ...) are a different,
/// fixed vocabulary from inferx's own short names ("f32", "bf16", ...), and
/// the two must not be confused with each other.
///
/// Entries cover every Dtype the safetensors Rust crate defines
/// (huggingface/safetensors, safetensors/src/tensor.rs), in the crate's own
/// order: increasing alignment. Two inferx types are deliberately absent
/// because the spec has no name for them: the packed sub-byte ints
/// (kUInt2..kInt4) and kComplex128 (the spec's only complex dtype, C64, is
/// 64-bit).
std::optional<DataType> DataTypeFromSafetensorsName(std::string_view name) {
  static const std::unordered_map<std::string_view, DataType> kNameToType = {
      {"BOOL", DataType::kBool},

      // MX microscaling formats (OCP MX spec). F4 is E2M1.
      {"F4", DataType::kFloat4E2M1},
      {"F6_E2M3", DataType::kFloat6E2M3},
      {"F6_E3M2", DataType::kFloat6E3M2},

      {"U8", DataType::kUInt8},
      {"I8", DataType::kInt8},

      // Bare E4M3 means the OCP FN variant; the FNUZ names say so explicitly.
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

  const auto it = kNameToType.find(name);

  if (it == kNameToType.end()) return std::nullopt;

  return it->second;
}

/// \brief Product of `shape`'s extents, or nullopt on a negative extent or
///        on overflow.
///
/// An empty `shape` (a scalar) has a count of 1, matching the safetensors
/// convention.
std::optional<int64_t> ElementCount(const std::vector<int64_t>& shape) {
  int64_t count = 1;

  for (const int64_t extent : shape) {
    if (extent < 0) return std::nullopt;

    if (extent != 0 &&
        count > std::numeric_limits<int64_t>::max() / (extent == 0 ? 1 : extent)) {
      return std::nullopt;  // would overflow
    }

    count *= extent;
  }

  return count;
}

std::unexpected<SafeTensorError> Fail(SafeTensorErrorCode code, std::string message) {
  return std::unexpected(SafeTensorError{.code = code, .message = std::move(message)});
}

/// \brief Read and sanity-check the 8-byte little-endian header-length
///        prefix at the start of `bytes`.
std::expected<std::uint64_t, SafeTensorError> ParseHeaderLength(
    std::span<const std::byte> bytes) {
  if (bytes.size() < kHeaderLengthPrefixBytes) {
    return Fail(SafeTensorErrorCode::kHeaderTooSmall,
                "buffer is smaller than the 8-byte header-length prefix");
  }

  const std::byte* base = bytes.data();

  // TODO(gc): Do we need to verify the UTF-8 validity of the header? The safetensors spec says
  // the header is a UTF-8 string, but nlohmann::json happily parses non-UTF-8 bytes as long as
  // they are valid JSON. If we don't verify UTF-8, then a corrupted or adversarial file could
  // have a non-UTF-8 header that still parses as JSON, and then the tensor names would be
  // invalid UTF-8 strings. That might be fine, but it might also break code that assumes tensor
  // names are valid UTF-8.

  // Header length is an explicit little-endian u64, independent of host
  // byte order.
  const std::uint64_t header_len =
      boost::endian::load_little_u64(reinterpret_cast<const unsigned char*>(base));

  if (header_len > bytes.size() - kHeaderLengthPrefixBytes) {
    return Fail(SafeTensorErrorCode::kInvalidHeaderLength, "header length exceeds buffer size");
  }

  if (header_len > kMaxHeaderBytes) {
    return Fail(SafeTensorErrorCode::kHeaderTooLarge,
                "header length " + std::to_string(header_len) + " exceeds the " +
                    std::to_string(kMaxHeaderBytes) + "-byte sanity cap");
  }

  return header_len;
}

/// \brief Parse `header_bytes` -- just the header portion, already sliced
///        out of the file by the caller -- as a JSON object.
///
/// Takes no part in locating the header within a larger buffer; that offset
/// arithmetic belongs to the caller (Parse()), not here.
std::expected<OrderedJson, SafeTensorError> ParseHeaderJson(
    std::span<const std::byte> header_bytes) {
  const std::string_view header_json(reinterpret_cast<const char*>(header_bytes.data()),
                                     header_bytes.size());

  OrderedJson header;

  try {
    header = OrderedJson::parse(header_json);
  } catch (const OrderedJson::parse_error& e) {
    return Fail(SafeTensorErrorCode::kJsonError,
                std::string("invalid JSON header: ") + e.what());
  }

  if (!header.is_object()) {
    return Fail(SafeTensorErrorCode::kJsonError, "header is not a JSON object");
  }

  return header;
}

/// \brief Validate the "__metadata__" entry and merge its string fields into
///        `metadata`.
std::expected<void, SafeTensorError> ParseMetadataObject(
    const OrderedJson& value, std::unordered_map<std::string, std::string>& metadata) {
  if (!value.is_object()) {
    return Fail(SafeTensorErrorCode::kJsonError, "__metadata__ is not a JSON object");
  }

  for (const auto& [meta_key, meta_value] : value.items()) {
    if (!meta_value.is_string()) {
      return Fail(SafeTensorErrorCode::kJsonError,
                  "__metadata__[\"" + meta_key + "\"] is not a string");
    }

    metadata.emplace(meta_key, meta_value.get<std::string>());
  }

  return {};
}

/// \brief Validate one non-"__metadata__" header entry and extract its
///        dtype, shape, and data_offsets.
///
/// Checks that dtype is a recognized safetensors type string, that
/// data_offsets fall within [0, data_size], that a sub-byte dtype's total
/// bit width lands on a byte boundary, and that the span the offsets cover
/// equals DataTypeByteSize(dtype, product(shape)) -- i.e. that the declared
/// shape and dtype actually account for every byte the entry claims.
/// Contiguity between entries and full coverage of the data blob are
/// whole-header properties checked by Parse(), not per-entry ones.
std::expected<ParsedTensor, SafeTensorError> ParseTensorEntry(const std::string& name,
                                                              const OrderedJson& value,
                                                              std::size_t data_size) {
  if (!value.is_object() || !value.contains("dtype") || !value.contains("shape") ||
      !value.contains("data_offsets")) {
    return Fail(SafeTensorErrorCode::kJsonError,
                "tensor \"" + name + "\" is missing dtype/shape/data_offsets");
  }

  const auto& dtype_field = value["dtype"];

  if (!dtype_field.is_string()) {
    return Fail(SafeTensorErrorCode::kJsonError,
                "tensor \"" + name + "\" has a non-string dtype");
  }

  const std::string dtype_name = dtype_field.get<std::string>();
  const std::optional<DataType> dtype = DataTypeFromSafetensorsName(dtype_name);

  if (!dtype.has_value()) {
    return Fail(SafeTensorErrorCode::kInvalidHeaderDeserialization,
                "tensor \"" + name + "\" has unknown dtype \"" + dtype_name + "\"");
  }

  const auto& shape_field = value["shape"];

  if (!shape_field.is_array()) {
    return Fail(SafeTensorErrorCode::kJsonError,
                "tensor \"" + name + "\" has a non-array shape");
  }

  std::vector<int64_t> shape;
  shape.reserve(shape_field.size());

  for (const auto& extent : shape_field) {
    if (!extent.is_number_integer()) {
      return Fail(SafeTensorErrorCode::kJsonError,
                  "tensor \"" + name + "\" has a non-integer shape extent");
    }

    shape.push_back(extent.get<int64_t>());

    if (extent.get<int64_t>() < 0) {
      return Fail(SafeTensorErrorCode::kTensorInvalidInfo,
                  "tensor \"" + name + "\" has a negative shape extent");
    }
  }

  const auto& offsets_field = value["data_offsets"];

  if (!offsets_field.is_array() || offsets_field.size() != 2 ||
      !offsets_field[0].is_number_integer() || !offsets_field[1].is_number_integer()) {
    return Fail(SafeTensorErrorCode::kInvalidOffset,
                "tensor \"" + name + "\" has malformed data_offsets");
  }

  const int64_t raw_begin = offsets_field[0].get<int64_t>();
  const int64_t raw_end = offsets_field[1].get<int64_t>();

  if (raw_begin < 0 || raw_end < raw_begin || static_cast<std::uint64_t>(raw_end) > data_size) {
    return Fail(SafeTensorErrorCode::kInvalidOffset,
                "tensor \"" + name + "\" data_offsets [" + std::to_string(raw_begin) + ", " +
                    std::to_string(raw_end) + "] are outside the data blob (size " +
                    std::to_string(data_size) + ")");
  }

  const std::size_t begin = static_cast<std::size_t>(raw_begin);
  const std::size_t end = static_cast<std::size_t>(raw_end);
  const std::optional<int64_t> count = ElementCount(shape);

  if (!count.has_value()) {
    return Fail(SafeTensorErrorCode::kValidationOverflow,
                "tensor \"" + name + "\" has a shape whose element count overflows");
  }

  // Guard the bit arithmetic before it runs: DataTypeByteSize multiplies in
  // int64_t, so a count this large would overflow (UB) before any later check
  // could see a negative byte count.
  const int64_t bits = static_cast<int64_t>(DataTypeStorageBits(*dtype));

  if (*count > (std::numeric_limits<int64_t>::max() - 7) / bits) {
    return Fail(SafeTensorErrorCode::kValidationOverflow,
                "tensor \"" + name + "\": element count times dtype width overflows");
  }

  // Sub-byte dtypes pack whole elements per byte: a count whose total bit
  // width straddles a byte boundary cannot be laid out at all, and an
  // innermost extent that does not fill whole bytes would make row
  // addressing ill-defined. The latter also mirrors TensorSpec::Verify, so a
  // header that parses can never fail GetTensor's FromBlob.
  if (DataTypeIsSubByte(*dtype)) {
    if ((*count * bits) % 8 != 0) {
      return Fail(SafeTensorErrorCode::kMisalignedSlice,
                  "tensor \"" + name + "\": sub-byte dtype spans " + std::to_string(*count) +
                      " elements x " + std::to_string(bits) +
                      " bits, which does not land on a byte boundary");
    }

    const int64_t per_byte = static_cast<int64_t>(8) / static_cast<int64_t>(bits);

    if (!shape.empty() && shape.back() % per_byte != 0) {
      return Fail(SafeTensorErrorCode::kMisalignedSlice,
                  "tensor \"" + name + "\": sub-byte dtype requires the innermost extent (" +
                      std::to_string(shape.back()) + ") to be a multiple of " +
                      std::to_string(per_byte));
    }
  }

  const int64_t expected_bytes = DataTypeByteSize(*dtype, *count);

  if (static_cast<std::uint64_t>(expected_bytes) != (end - begin)) {
    return Fail(SafeTensorErrorCode::kTensorInvalidInfo,
                "tensor \"" + name + "\" data_offsets span " + std::to_string(end - begin) +
                    " bytes but shape * dtype implies " + std::to_string(expected_bytes));
  }

  return ParsedTensor{.dtype = *dtype, .shape = Shape(shape), .begin = begin, .end = end};
}

}  // namespace

std::expected<SafeTensorsView, SafeTensorError> SafeTensorsView::Parse(
    std::span<const std::byte> bytes) {
  const auto header_len = ParseHeaderLength(bytes);

  if (!header_len.has_value()) return std::unexpected(header_len.error());

  const auto header = ParseHeaderJson(bytes.subspan(kHeaderLengthPrefixBytes, *header_len));

  if (!header.has_value()) return std::unexpected(header.error());

  const std::size_t data_size = bytes.size() - kHeaderLengthPrefixBytes - *header_len;
  const std::byte* data_begin = bytes.data() + kHeaderLengthPrefixBytes + *header_len;

  std::unordered_map<std::string, Entry, StringHash, std::equal_to<>> entries;
  std::vector<std::string> name_order;
  std::unordered_map<std::string, std::string> metadata;

  // The spec requires tensor data to be laid out contiguously, in header
  // order, covering the data blob exactly: each entry begins where the
  // previous one ended, and the last one ends at the end of the file.
  std::size_t expected_begin = 0;

  for (const auto& [name, value] : header->items()) {
    if (name == "__metadata__") {
      const auto status = ParseMetadataObject(value, metadata);

      if (!status.has_value()) return std::unexpected(status.error());

      continue;
    }

    auto parsed = ParseTensorEntry(name, value, data_size);

    if (!parsed.has_value()) return std::unexpected(parsed.error());

    if (parsed->begin != expected_begin) {
      return Fail(SafeTensorErrorCode::kInvalidOffset,
                  "tensor \"" + name + "\" begins at byte " + std::to_string(parsed->begin) +
                      " but the previous tensor ends at " + std::to_string(expected_begin) +
                      ": offsets must be contiguous and in header order");
    }

    expected_begin = parsed->end;

    name_order.push_back(name);
    entries.emplace(name, Entry{.dtype = parsed->dtype,
                                .shape = std::move(parsed->shape),
                                .begin = parsed->begin,
                                .end = parsed->end});
  }

  if (expected_begin != data_size) {
    return Fail(SafeTensorErrorCode::kMetadataIncompleteBuffer,
                "tensor data covers " + std::to_string(expected_begin) + " of " +
                    std::to_string(data_size) +
                    " bytes: the last offset must be the end of the file");
  }

  SafeTensorsView view;
  view.data_begin_ = data_begin;
  view.data_size_ = data_size;
  view.entries_ = std::move(entries);
  view.name_order_ = std::move(name_order);
  view.metadata_ = std::move(metadata);

  return view;
}

std::span<const std::string> SafeTensorsView::TensorNames() const noexcept {
  return name_order_;
}

bool SafeTensorsView::HasTensor(std::string_view name) const noexcept {
  return entries_.contains(name);
}

std::optional<Tensor> SafeTensorsView::GetTensor(std::string_view name) const {
  const auto it = entries_.find(name);

  if (it == entries_.end()) return std::nullopt;

  const Entry& entry = it->second;

  // Infallible in practice: Parse() already ran the same dtype/shape
  // validation FromBlob repeats, including the sub-byte innermost rule.
  // The const_cast is FromBlob's borrowing contract -- the tensor writes
  // through this pointer only if the caller asks it to.
  auto tensor = Tensor::FromBlob(const_cast<std::byte*>(data_begin_ + entry.begin), entry.dtype,
                                 entry.shape, DeviceId::Cpu());

  if (!tensor.ok()) return std::nullopt;

  return std::move(tensor).value();
}

const std::unordered_map<std::string, std::string>& SafeTensorsView::Metadata() const noexcept {
  return metadata_;
}

}  // namespace inferx
