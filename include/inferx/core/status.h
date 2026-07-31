#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace inferx {

using Status = absl::Status;

template <typename T>
using StatusOr = absl::StatusOr<T>;

inline Status OkStatus() { return absl::OkStatus(); }

/// \brief Returns a Status with code `absl::StatusCode::kInvalidArgument` and a
///        message constructed from the given arguments.
///
/// \param ...args The arguments to be concatenated into the error message.
/// \return        A Status with code `absl::StatusCode::kInvalidArgument` and
///                the constructed message.
template <typename... Args>
Status InvalidArgumentError(Args&&... args) {
  return absl::InvalidArgumentError(absl::StrCat(std::forward<Args>(args)...));
}

/// \brief Returns a Status with code `absl::StatusCode::kResourceExhausted` and
///        a message constructed from the given arguments.
///
/// \param ...args The arguments to be concatenated into the error message.
/// \return        A Status with code `absl::StatusCode::kResourceExhausted` and
///                the constructed message.
template <typename... Args>
Status ResourceExhaustedError(Args&&... args) {
  return absl::ResourceExhaustedError(
      absl::StrCat(std::forward<Args>(args)...));
}

/// \brief Returns a Status with code `absl::StatusCode::kFailedPrecondition`
///        and a message constructed from the given arguments.
///
/// \param ...args The arguments to be concatenated into the error message.
/// \return        A Status with code `absl::StatusCode::kFailedPrecondition`
///                and the constructed message.
template <typename... Args>
Status FailedPreconditionError(Args&&... args) {
  return absl::FailedPreconditionError(
      absl::StrCat(std::forward<Args>(args)...));
}

/// \brief Returns a Status with code `absl::StatusCode::kInternal`
///        and a message constructed from the given arguments.
///
/// \param ...args The arguments to be concatenated into the error message.
/// \return        A Status with code `absl::StatusCode::kInternal` and
///                the constructed message.
template <typename... Args>
Status InternalError(Args&&... args) {
  return absl::InternalError(absl::StrCat(std::forward<Args>(args)...));
}

/// \brief Returns a Status with code `absl::StatusCode::kUnimplemented`
///        and a message constructed from the given arguments.
///
/// \param ...args The arguments to be concatenated into the error message.
/// \return        A Status with code `absl::StatusCode::kUnimplemented` and
///                the constructed message.
template <typename... Args>
Status UnimplementedError(Args&&... args) {
  return absl::UnimplementedError(absl::StrCat(std::forward<Args>(args)...));
}

/// \brief Returns a Status with code `absl::StatusCode::kNotFound` and a
///        message constructed from the given arguments.
///
/// \param ...args The arguments to be concatenated into the error message.
/// \return        A Status with code `absl::StatusCode::kNotFound` and the
///                constructed message.
template <typename... Args>
Status NotFoundError(Args&&... args) {
  return absl::NotFoundError(absl::StrCat(std::forward<Args>(args)...));
}

/// \brief Returns a Status with code `absl::StatusCode::kOutOfRange`
///        and a message constructed from the given arguments.
///
/// \param ...args The arguments to be concatenated into the error message.
/// \return        A Status with code `absl::StatusCode::kOutOfRange` and
///                the constructed message.
template <typename... Args>
Status OutOfRangeError(Args&&... args) {
  return absl::OutOfRangeError(absl::StrCat(std::forward<Args>(args)...));
}

}  // namespace inferx

#define INFERX_STATUS_CONCAT_INNER_(a, b) a##b
#define INFERX_STATUS_CONCAT_(a, b) INFERX_STATUS_CONCAT_INNER_(a, b)

#define INFERX_RETURN_IF_ERROR(expr)          \
  do {                                        \
    ::inferx::Status _inferx_status = (expr); \
    if (!_inferx_status.ok()) [[unlikely]] {  \
      return _inferx_status;                  \
    }                                         \
  } while (0)

#define INFERX_ASSIGN_OR_RETURN_IMPL_(tmp, lhs, expr) \
  auto tmp = (expr);                                  \
  if (!tmp.ok()) [[unlikely]] {                       \
    return std::move(tmp).status();                   \
  }                                                   \
  lhs = *std::move(tmp)

/// \brief Evaluates `expr`, which must return a `StatusOr<T>`. If the result is
///        OK, assigns the contained value to `lhs`. Otherwise, returns the
///        error status from the current function.
///
/// EXAMPLE:
/// \code
/// StatusOr<int> ComputeValue();
///
/// StatusOr<int> MyFunction() {
///   INFERX_ASSIGN_OR_RETURN(int value, ComputeValue());
///   return value * 2;
/// }
/// \endcode
///
/// \param lhs  The variable to assign the value to if `expr` is OK.
/// \param expr An expression that returns a `StatusOr<T>`.
/// \return     If `expr` is not OK, returns the error status from the current
///             function.
#define INFERX_ASSIGN_OR_RETURN(lhs, expr) \
  INFERX_ASSIGN_OR_RETURN_IMPL_(           \
      INFERX_STATUS_CONCAT_(_inferx_statusor_, __LINE__), lhs, expr)
