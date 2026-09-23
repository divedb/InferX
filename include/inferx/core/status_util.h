// Throwing conveniences over Status/StatusOr for implementation bodies that
// prefer exception style internally (file I/O, JSON walking). Library entry
// points wrap their bodies in Guarded so exceptions never cross the boundary;
// the CLI re-applies the error code when converting back (see cli/error.h).
#ifndef INFERX_CORE_STATUS_UTIL_H_
#define INFERX_CORE_STATUS_UTIL_H_

#include <stdexcept>
#include <string>
#include <utility>

#include "inferx/core/status.h"

namespace inferx {

/// \brief Throws std::runtime_error carrying the bare message when `status`
///        is not OK.
inline void Check(const Status& status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}

/// \brief Returns the value of an OK `result`, throwing on error like Check.
template <class T>
T Take(StatusOr<T> result) {
  Check(result.status());
  return *std::move(result);
}

/// \brief Runs `body`, converting any exception into an Internal Status.
template <class Body>
Status Guarded(Body&& body) try {
  body();
  return OkStatus();
} catch (const std::exception& e) {
  return InternalError(e.what());
} catch (...) {
  return InternalError("unknown exception");
}

}  // namespace inferx

#endif  // INFERX_CORE_STATUS_UTIL_H_
