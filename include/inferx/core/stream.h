#pragma once

#include <cstddef>
#include <type_traits>

namespace inferx {

/// \brief A non-owning, type-erased handle to a runtime-native object.
///
/// Core and model code may store, copy and compare handles, but only a DeviceRuntime
/// implementation interprets `handle`. Copying a handle neither copies the object nor extends
/// its lifetime; the runtime that created it destroys it. The handle does not record which
/// device it belongs to.
///
/// \tparam Tag Distinguishes handle kinds, so a stream cannot be passed
///             where an event is expected.
template <typename Tag>
struct DeviceHandle {
  /// The runtime's native handle value. Null's meaning is runtime-defined
  /// (e.g. the default stream on CUDA).
  void* handle = nullptr;

  /// \brief Constructs a null handle.
  constexpr DeviceHandle() = default;

  /// \brief Constructs a null handle from nullptr.
  constexpr DeviceHandle(std::nullptr_t) {}

  /// \brief Wraps a native pointer handle (e.g. `cudaStream_t`).
  ///
  /// Explicit so that unrelated pointers never become handles by accident.
  /// Const and function pointers are rejected.
  ///
  /// \param native The native handle in its concrete pointer type.
  template <typename T>
    requires(std::is_pointer_v<T> && std::is_convertible_v<T, void*>)
  constexpr explicit DeviceHandle(T native) : handle(native) {}

  /// \brief Returns the handle as its native pointer type.
  ///
  /// \tparam T The concrete native pointer type, e.g. `cudaStream_t`.
  template <typename T>
    requires(std::is_pointer_v<T> && !std::is_function_v<std::remove_pointer_t<T>>)
  constexpr T As() const {
    return static_cast<T>(handle);
  }

  /// \brief Converts the handle to a boolean.
  ///
  /// \return True if the handle is non-null; otherwise false.
  constexpr explicit operator bool() const { return handle != nullptr; }

  /// \brief Compares two handles for equality.
  ///
  /// \param a The first handle.
  /// \param b The second handle.
  /// \return  True if the handles are equal; otherwise false.
  friend constexpr bool operator==(DeviceHandle, DeviceHandle) = default;
};

struct StreamTag;
struct EventTag;
struct GraphExecTag;

/// \brief An execution stream (queue of ordered work).
using DeviceStream = DeviceHandle<StreamTag>;

/// \brief An event, for timing and synchronization.
using DeviceEvent = DeviceHandle<EventTag>;

/// \brief An instantiated graph, ready to launch.
using DeviceGraphExec = DeviceHandle<GraphExecTag>;

static_assert(std::is_trivially_copyable_v<DeviceStream>);
static_assert(std::is_standard_layout_v<DeviceStream>);
static_assert(sizeof(DeviceStream) == sizeof(void*));

}  // namespace inferx
