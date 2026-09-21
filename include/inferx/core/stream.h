#pragma once

#include <cstddef>
#include <type_traits>

namespace inferx {

/// \brief Runtime-owned execution handles.
///
/// Core and model code may pass these values around, but only a DeviceRuntime
/// implementation interprets `handle`.
struct Stream {
  /// The runtime's native handle value.
  void* handle = nullptr;

  /// \brief Constructs a null stream.
  constexpr Stream() = default;
  /// \brief Constructs a null stream from nullptr.
  constexpr Stream(std::nullptr_t) {}
  /// \brief Constructs a stream from a raw handle value.
  ///
  /// \param value The native handle.
  constexpr explicit Stream(void* value) : handle(value) {}

  /// \brief Constructs a stream from a native pointer handle.
  ///
  /// Device runtime source files may pass native pointer handles at the boundary
  /// without exposing their types here. Restricted to pointer types so an
  /// integer or unrelated value cannot accidentally become a stream.
  ///
  /// \param value The native handle as its concrete pointer type.
  template <typename T>
    requires std::is_pointer_v<T>
  constexpr Stream(T value) : handle(static_cast<void*>(value)) {}

  /// \brief Converts the stream back to its native pointer type.
  ///
  /// \return The native handle as `T`.
  template <typename T>
    requires std::is_pointer_v<T>
  constexpr operator T() const {
    return static_cast<T>(handle);
  }

  /// \brief Compares two streams for equality (same handle).
  friend constexpr bool operator==(Stream, Stream) = default;
};

/// \brief Runtime-owned event handle, for timing and synchronization.
struct DeviceEvent {
  /// The runtime's native handle value.
  void* handle = nullptr;
  /// \brief Compares two events for equality (same handle).
  friend constexpr bool operator==(DeviceEvent, DeviceEvent) = default;
};

/// \brief Runtime-owned instantiated-graph handle.
struct GraphExec {
  /// The runtime's native handle value.
  void* handle = nullptr;
  /// \brief Compares two graph handles for equality.
  friend constexpr bool operator==(GraphExec, GraphExec) = default;
};

}  // namespace inferx
