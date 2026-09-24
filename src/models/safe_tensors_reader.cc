#include "inferx/models/safe_tensors_reader.h"

#include <boost/interprocess/exceptions.hpp>
#include <system_error>

namespace inferx {

std::expected<SafeTensorReader, SafeTensorError> SafeTensorReader::Open(
    const std::filesystem::path& path) {
  std::error_code ec;
  const std::uintmax_t file_size = std::filesystem::file_size(path, ec);

  if (ec) {
    return std::unexpected(SafeTensorError{
        .code = ec == std::errc::no_such_file_or_directory ? SafeTensorErrorCode::kFileNotFound
                                                           : SafeTensorErrorCode::kIoError,
        .message = "stat(" + path.string() + "): " + ec.message()});
  }

  boost::interprocess::file_mapping file_mapping;
  boost::interprocess::mapped_region mapped_region;

  try {
    const auto mode = boost::interprocess::read_only;
    file_mapping = boost::interprocess::file_mapping(path.c_str(), mode);
    mapped_region = boost::interprocess::mapped_region(file_mapping, mode);
  } catch (const boost::interprocess::interprocess_exception& e) {
    // Covers, among other things, a zero-byte file: mapping an empty region
    // fails at this layer rather than reaching SafeTensorsView::Parse(), so
    // it surfaces as kIoError here instead of the more specific
    // kHeaderTooSmall Parse() would have reported for a merely-too-small
    // (but nonzero) file.
    return std::unexpected(SafeTensorError{.code = SafeTensorErrorCode::kIoError,
                                           .message = std::string("mmap failed: ") + e.what()});
  }

  const auto* base = static_cast<const std::byte*>(mapped_region.get_address());
  const std::span<const std::byte> bytes(base, mapped_region.get_size());
  (void)file_size;  // already reflected in mapped_region.get_size()

  std::expected<SafeTensorsView, SafeTensorError> parsed = SafeTensorsView::Parse(bytes);

  if (!parsed.has_value()) {
    return std::unexpected(std::move(parsed.error()));
  }

  SafeTensorReader reader;
  reader.file_mapping_ = std::move(file_mapping);
  reader.mapped_region_ = std::move(mapped_region);
  reader.view_ = std::move(*parsed);

  return reader;
}

}  // namespace inferx
