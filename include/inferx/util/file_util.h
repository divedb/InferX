#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <string>

#include "inferx/core/status.h"

namespace inferx {

StatusOr<std::string> ReadFileToString(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);

  if (!input) {
    return InternalError("could not open file: ", path.string());
  }

  std::string contents;
  std::array<char, 64 * 1024> chunk;

  // A short final read sets failbit|eofbit but still delivers bytes, so
  // gcount() has to be checked as well as the stream state.
  while (input.read(chunk.data(), chunk.size()) || input.gcount() > 0) {
    contents.append(chunk.data(), static_cast<size_t>(input.gcount()));
  }

  // A clean read ends at EOF. Anything else is an I/O error (for example,
  // the path is a directory, which opens fine on Linux but fails on read).
  if (input.bad() || !input.eof()) {
    return InternalError("failed to read file: ", path.string());
  }

  return contents;
}

}  // namespace inferx