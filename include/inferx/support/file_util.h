#pragma once

#include <string>

#include "inferx/core/status.h"

namespace inferx {

/// \brief Reads an entire file as binary data.
///
/// \param path The path to the file to read.
/// \return     A StatusOr containing the file contents as a string, or an error
///             status if the file could not be read.
StatusOr<std::string> ReadFile(const std::string& path);

}  // namespace inferx
