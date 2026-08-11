#pragma once

#include <cstddef>
#include <type_traits>

namespace inferx {

// Backend-owned execution handles.  Core and model code may pass these values
// around, but only a DeviceRuntime implementation interprets `handle`.
struct Stream {
  void* handle = nullptr;

  constexpr Stream() = default;
  constexpr Stream(std::nullptr_t) {}
  constexpr explicit Stream(void* value) : handle(value) {}

  // Backend source files may pass native pointer handles at the boundary
  // without exposing their types here. Restrict this to pointer types so an
  // integer or unrelated value cannot accidentally become a stream.
  template <typename T>
    requires std::is_pointer_v<T>
  constexpr Stream(T value) : handle(static_cast<void*>(value)) {}

  template <typename T>
    requires std::is_pointer_v<T>
  constexpr operator T() const {
    return static_cast<T>(handle);
  }

  friend constexpr bool operator==(Stream, Stream) = default;
};

struct DeviceEvent {
  void* handle = nullptr;
  friend constexpr bool operator==(DeviceEvent, DeviceEvent) = default;
};

struct GraphExec {
  void* handle = nullptr;
  friend constexpr bool operator==(GraphExec, GraphExec) = default;
};

}  // namespace inferx
