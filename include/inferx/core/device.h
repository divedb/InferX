#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"

namespace inferx {

/// \brief Enumeration of supported device types.
enum class DeviceType : uint8_t {
  kCpu = 0,   ///< CPU device.
  kCuda = 1,  ///< NVIDIA CUDA GPU.
  kHip = 2,   ///< AMD HIP GPU.
};

/// \brief Returns a string representation of the device type.
///
/// \param kind The device type to convert to a string.
/// \return     A string representing the device type.
constexpr const char* DeviceTypeToString(DeviceType kind) {
  switch (kind) {
    case DeviceType::kCpu:
      return "cpu";
    case DeviceType::kCuda:
      return "cuda";
    case DeviceType::kHip:
      return "hip";
    default:
      break;
  }

  return "unknown";
}

/// \brief Identifies one device as a (kind, ordinal) pair.
///
/// `Device` is the currency of every placement decision in the engine: which
/// allocator backs a Storage, which stream a kernel launches on, which copy
/// path moves bytes between two buffers.
///
///   CPU    the single host address space. There is no notion of NUMA affinity,
///          so the ordinal is always 0 and `Cpu()` is a constant.
///   CUDA   one GPU, addressed by its runtime ordinal (0..N-1). The ordinal
///          pins a primary context and therefore a device-side allocator;
///          cross-device copies are explicit, never implicit.
class Device {
 public:
  /// \brief The default device, which is the host. Equal to `Cpu()`.
  ///
  /// Public because a Device is a member of nearly every buffer and view in
  /// the engine, and a private default constructor would delete theirs.
  constexpr Device() = default;

  /// \brief The canonical CPU device. Always (kCpu, 0).
  static constexpr Device Cpu() { return Device(DeviceType::kCpu, 0); }

  /// \brief The CUDA device with the given runtime `ordinal` (0..N-1).
  ///
  /// \param ordinal The CUDA runtime device ordinal.
  /// \return        The corresponding Device.
  static constexpr Device Cuda(int8_t ordinal) { return Device(DeviceType::kCuda, ordinal); }

  /// \brief The ROCm device with the given runtime `ordinal` (0..N-1).
  ///
  /// \param ordinal The ROCm runtime device ordinal.
  /// \return        The corresponding Device.
  static constexpr Device Hip(int8_t ordinal) { return Device(DeviceType::kHip, ordinal); }

  /// \brief Names a device of `kind` at `ordinal`, normalizing CPU to `Cpu()`.
  ///
  /// \param kind    The device class.
  /// \param ordinal The device ordinal; ignored for CPU.
  /// \return        The corresponding Device.
  static constexpr Device For(DeviceType kind, int8_t ordinal = 0) {
    return kind == DeviceType::kCpu ? Cpu() : Device(kind, ordinal);
  }

  /// \brief Check if this device is a CPU.
  ///
  /// \return True if this device is a CPU; otherwise false.
  constexpr bool IsCpu() const noexcept { return device_type_ == DeviceType::kCpu; }

  /// \brief Check if this device is a CUDA GPU.
  ///
  /// \return True if this device is a CUDA GPU; otherwise false.
  constexpr bool IsCuda() const noexcept { return device_type_ == DeviceType::kCuda; }

  /// \brief Check if this device is a HIP GPU.
  ///
  /// \return True if this device is a HIP GPU; otherwise false.
  constexpr bool IsHip() const noexcept { return device_type_ == DeviceType::kHip; }

  /// \brief The device class. See `DeviceType`.
  constexpr DeviceType kind() const noexcept { return device_type_; }

  /// \brief The runtime device ordinal (0..N-1); always 0 for CPU.
  constexpr int8_t index() const noexcept { return index_; }

  /// \brief Compares two device ids for equality (same kind and ordinal).
  friend constexpr bool operator==(const Device& a, const Device& b) {
    return a.device_type_ == b.device_type_ && a.index_ == b.index_;
  }

  /// \brief Compares two device ids for inequality.
  friend constexpr bool operator!=(const Device& a, const Device& b) { return !(a == b); }

  /// \brief Returns a string representation of the device ID.
  ///
  /// \return A string representing the device ID.
  std::string ToString() const {
    return absl::StrCat(DeviceTypeToString(device_type_), ":", static_cast<int>(index_));
  }

  /// \brief Extends an absl hash to cover Device, for use as a map key.
  template <typename H>
  friend H AbslHashValue(H h, const Device& d) {
    return H::combine(std::move(h), d.device_type_, d.index_);
  }

 private:
  /// \brief Constructs a (kind, ordinal) pair.
  ///
  /// Private so that a device is always named through Cpu() or Cuda(): a bare
  /// (kind, ordinal) pair at a call site reads as two unrelated numbers.
  constexpr Device(DeviceType dtype, int8_t index) : device_type_(dtype), index_(index) {}

  /// The device class. See `DeviceType`.
  DeviceType device_type_ = DeviceType::kCpu;

  /// For CUDA, the runtime device ordinal (0..N-1). Always 0 for CPU.
  int8_t index_ = 0;
};

}  // namespace inferx
