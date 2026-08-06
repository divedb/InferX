#pragma once

#include <string>

#include "inferx/core/status.h"

namespace inferx {

/// Reads an entire file as binary data.

/// @brief
/// @param path
/// @return
StatusOr<std::string> ReadFile(const std::string& path);

}  // namespace inferx
