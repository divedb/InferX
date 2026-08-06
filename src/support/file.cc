#include "inferx/support/file.h"

#include <fstream>
#include <sstream>

namespace inferx {

/// \brief Reads an entire file as binary data.
StatusOr<std::string> ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return NotFoundError("cannot open ", path);

  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input.good() && !input.eof()) {
    return InternalError("error reading ", path);
  }
  return contents.str();
}

}  // namespace inferx
