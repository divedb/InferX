#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "inferx/core/tensor.h"

namespace inferx {

/// \brief Error codes for SafeTensorReader::Open().
///
/// Adapted from the Rust safetensors crate's Error enum.
///
/// https://github.com/safetensors/safetensors/blob/main/safetensors/src/tensor.rs
enum class SafeTensorErrorCode {
  /// The header is an invalid UTF-8 string and cannot be read.
  kInvalidHeader,

  /// The header is a valid string, but does not deserialize into the header
  /// schema: an unrecognized dtype string, for instance. (In the Rust crate
  /// this is where serde's deserialization failures land.)
  kInvalidHeaderDeserialization,

  /// The header is large than 100Mo which is considered too large (Might evolve in the future).
  kHeaderTooLarge,

  /// The header is smaller than 8 bytes.
  kHeaderTooSmall,

  /// The header length is invalid. Suppose the header length is N, then the file size must be
  /// at least 8 + N bytes. If not, this error is returned.
  kInvalidHeaderLength,

  /// The tensor name was not found in the archive
  kTensorNotFound,

  /// Invalid information between shape, dtype and the proposed offsets in the file
  kTensorInvalidInfo,

  /// The offsets declared for tensor with name `String` in the header are invalid
  kInvalidOffset,

  /// The file does not exist. And this differs from kIoError in that it is not a permission
  /// issue, but the file is simply not present.
  kFileNotFound,

  /// IoError.
  /// For example permission denied, or file is a directory.
  kIoError,

  /// JSON error.
  /// For example the header is not valid JSON, or the header is not a JSON object.
  kJsonError,

  /// The follow tensor cannot be created because the buffer size doesn't match shape + dtype
  kInvalidTensorView,

  /// The metadata is invalid because the data offsets of the tensor does not
  /// fully cover the buffer part of the file. The last offset **must** be
  /// the end of the file.
  kMetadataIncompleteBuffer,

  /// The metadata contains information (shape or shape * dtype size) which lead to an
  /// arithmetic overflow. This is most likely an error in the file.
  kValidationOverflow,

  /// For smaller than 1 byte dtypes, some slices will happen outside of the byte boundary, some
  /// special care has to be taken and standard functions will fail.
  kMisalignedSlice,
};

struct SafeTensorError {
  SafeTensorErrorCode code;
  std::string message;
};

/// \brief Parses and validates the safetensors format over a byte buffer it
///        does not own, and provides zero-copy tensor lookups into it.
///
/// Buffer layout: an 8-byte little-endian header length N, N bytes of JSON
/// header, then the raw data blob. Each header entry other than
/// "__metadata__" maps a tensor name to {"dtype", "shape", "data_offsets":
/// [begin, end]}, with begin/end byte offsets relative to the start of the
/// data blob.
///
/// This class does no I/O: it only interprets bytes it is handed, so it can
/// be used equally over an mmap'd file (see SafeTensorReader), a buffer read
/// from shared memory, or a plain std::vector<std::byte> in a unit test.
/// Parse() is where all validation happens -- every entry's data_offsets are
/// checked against the buffer's actual size and against
/// DataTypeByteSize(dtype, product(shape)) before the view is considered
/// valid, so a malformed buffer is rejected there rather than causing an
/// out-of-bounds read later from GetTensor().
///
/// \note Lifetime: every Tensor returned by GetTensor(), and the buffer this
///       view was parsed from, must outlive this SafeTensorsView.
///
/// \note Thread-safety: after a successful Parse(), a SafeTensorsView is
///       immutable. Concurrent calls to GetTensor()/TensorNames()/Metadata() from
///       multiple threads require no locking.
class SafeTensorsView {
 public:
  /// \brief An empty view with no tensors. Useful as a default/placeholder
  ///        value; GetTensor() on it always returns std::nullopt.
  SafeTensorsView() = default;

  /// \brief Parse and validate a safetensors buffer.
  ///
  /// Checks that the header is well-formed JSON, that every dtype string is
  /// recognized, and that every tensor's data_offsets are in range and equal
  /// to DataTypeByteSize(dtype, product(shape)). Does not validate tensor
  /// *contents*, and does not copy `bytes` -- the returned view borrows it.
  ///
  /// \param bytes The full safetensors buffer (length prefix + header +
  ///              data blob).
  /// \return      The view, or the first SafeTensorError found.
  static std::expected<SafeTensorsView, SafeTensorError> Parse(
      std::span<const std::byte> bytes);

  /// \brief Tensor names, in the order they appeared in the header.
  std::span<const std::string> TensorNames() const noexcept;

  /// \brief True if `name` names a tensor in this view.
  bool HasTensor(std::string_view name) const noexcept;

  /// \brief Zero-copy Tensor over `name`'s bytes, on the CPU device.
  ///
  /// The tensor borrows the buffer this view was parsed from (FromBlob
  /// semantics): the buffer must outlive the tensor. Parse() validated every
  /// entry's dtype/shape pair -- including the sub-byte innermost-extent rule
  /// TensorSpec::Verify enforces -- so construction cannot fail once a name
  /// resolves.
  ///
  /// \return The tensor, or std::nullopt if `name` is not present.
  std::optional<Tensor> GetTensor(std::string_view name) const;

  /// \brief The "__metadata__" string map from the header, if present.
  const std::unordered_map<std::string, std::string>& Metadata() const noexcept;

 private:
  /// \brief Transparent hash so entries_.find()/.contains() can take a
  ///        std::string_view directly instead of allocating a std::string
  ///        for every lookup.
  struct StringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view s) const noexcept {
      return std::hash<std::string_view>{}(s);
    }
  };

  struct Entry {
    DataType dtype{};
    Shape shape;
    std::size_t begin = 0;  // offset into the data blob
    std::size_t end = 0;
  };

  const std::byte* data_begin_ = nullptr;  // start of the data blob
  std::size_t data_size_ = 0;

  std::unordered_map<std::string, Entry, StringHash, std::equal_to<>> entries_;
  std::vector<std::string> name_order_;
  std::unordered_map<std::string, std::string> metadata_;
};

}  // namespace inferx
