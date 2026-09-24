#pragma once

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

#include "inferx/models/safe_tensors_view.h"

namespace inferx {

/// \brief Owns a memory-mapped .safetensors file and exposes it through a
///        SafeTensorsView.
///
/// All format parsing and validation is delegated to
/// SafeTensorsView::Parse(); this class is responsible only for the file
/// I/O -- stat-ing the file, mapping it read-only via boost::interprocess,
/// and keeping that mapping alive for as long as the view's pointers need
/// it.
///
/// Lifetime: every Tensor returned by GetTensor() borrows this reader's
/// mapping. Do not outlive the SafeTensorReader that produced it, and do not
/// call GetTensor() on a moved-from reader.
///
/// Thread-safety: after a successful Open(), a SafeTensorReader is
/// immutable. Concurrent calls to GetTensor()/TensorNames()/Metadata() from
/// multiple threads require no locking.
///
/// Caveat inherent to mmap: if the underlying file is truncated or
/// unlinked-and-replaced by another process while mapped, reads past the new
/// end of file raise SIGBUS rather than returning an error. SafeTensorReader
/// does not protect against this.
class SafeTensorReader {
 public:
  SafeTensorReader(const SafeTensorReader&) = delete;
  SafeTensorReader& operator=(const SafeTensorReader&) = delete;
  SafeTensorReader(SafeTensorReader&&) noexcept = default;
  SafeTensorReader& operator=(SafeTensorReader&&) noexcept = default;
  ~SafeTensorReader() = default;

  /// \brief Map and validate a .safetensors file.
  ///
  /// \param path Path to the .safetensors file.
  /// \return     The reader, or the first SafeTensorError found -- either
  ///             from the file I/O itself, or from SafeTensorsView::Parse()
  ///             on the mapped bytes.
  static std::expected<SafeTensorReader, SafeTensorError> Open(
      const std::filesystem::path& path);

  std::span<const std::string> TensorNames() const noexcept { return view_.TensorNames(); }

  bool HasTensor(std::string_view name) const noexcept { return view_.HasTensor(name); }

  std::optional<Tensor> GetTensor(std::string_view name) const noexcept {
    return view_.GetTensor(name);
  }

  const std::unordered_map<std::string, std::string>& Metadata() const noexcept {
    return view_.Metadata();
  }

  /// \brief Size in bytes of the mapped file.
  std::size_t FileSize() const noexcept { return mapped_region_.get_size(); }

 private:
  SafeTensorReader() = default;

  boost::interprocess::file_mapping file_mapping_;
  boost::interprocess::mapped_region mapped_region_;
  SafeTensorsView view_;
};

}  // namespace inferx
