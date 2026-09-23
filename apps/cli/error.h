#pragma once

#include <exception>
#include <string>

#include "inferx/core/status.h"

namespace inferx::cli {

class CommandError : public std::exception {
 public:
  explicit CommandError(const Status& status)
      : what_(status.ok() ? std::string("command failed") : std::string(status.message())) {}

  const char* what() const noexcept override { return what_.c_str(); }

 private:
  std::string what_;
};

/// \brief Throws CommandError when `status` is not OK.
inline void ThrowIfError(Status status) {
  if (!status.ok()) throw CommandError(status);
}

}  // namespace inferx::cli
