#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"

namespace inferx {

/// \brief The alignment of anything a kernel reads.
///
/// 128 B is the L1/L2 cache line, so a row start maps a warp's 128 B request
/// onto exactly four 32 B sectors. The hard floor is 16 B, the widest
/// vectorized load.
///
/// This is a default for callers, not a property of any allocator: alignment is
/// a per-call parameter, since pinned memory destined for cudaHostRegister
/// wants page alignment instead.
inline constexpr size_t kTensorAlignment = 128;

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
///   CPU       The single host address space. There is no notion of NUMA affinity,
///             so the ordinal is always 0 and `Cpu()` is a constant.
///   CUDA/HIP  One GPU, addressed by its runtime ordinal (0..N-1). The ordinal
///             pins a primary context and therefore a device-side allocator;
///             cross-device copies are explicit, never implicit.
class Device {
 public:
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

  /// \brief Returns a Device for the given type and ordinal.
  ///
  /// \param type    The device type (CPU, CUDA, HIP).
  /// \param ordinal The device ordinal (0..N-1) for CUDA/HIP, or 0 for CPU.
  /// \return        The corresponding Device.
  static constexpr Device For(DeviceType type, int8_t ordinal = 0) {
    return type == DeviceType::kCpu ? Cpu() : Device(type, ordinal);
  }

  /// \brief Default constructor. Constructs a Device with type kCpu and index 0.
  constexpr Device() = default;

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

  /// \brief Returns the device type (CPU, CUDA, HIP).
  ///
  /// \return The device type.
  constexpr DeviceType Type() const noexcept { return device_type_; }

  /// \brief Returns the device ordinal (0..N-1) for CUDA/HIP, or 0 for CPU.
  ///
  /// \return The device ordinal.
  constexpr int8_t Index() const noexcept { return index_; }

  /// \brief Compares two devices for equality. Two devices are equal if they have the same type
  ///        and ordinal.
  ///
  /// \param a The first device.
  /// \param b The second device.
  /// \return  True if the devices are equal; otherwise false.
  friend constexpr bool operator==(const Device& a, const Device& b) {
    return a.device_type_ == b.device_type_ && a.index_ == b.index_;
  }

  /// \brief Compares two devices for inequality.
  ///
  /// \param a The first device.
  /// \param b The second device.
  /// \return  True if the devices are not equal; otherwise false.
  friend constexpr bool operator!=(const Device& a, const Device& b) { return !(a == b); }

  /// \brief Returns a string representation of the device ID.
  ///
  /// \return A string representing the device ID.
  std::string ToString() const {
    return absl::StrCat(DeviceTypeToString(device_type_), ":", static_cast<int>(index_));
  }

  /// \brief Hashes a Device for use in Abseil hash containers.
  ///
  /// \tparam H The hash state type.
  /// \param h  The hash state.
  /// \param d  The device to hash.
  /// \return   The updated hash state.
  template <typename H>
  friend H AbslHashValue(H h, const Device& d) {
    return H::combine(std::move(h), d.device_type_, d.index_);
  }

 private:
  /// \brief Constructs a Device with the given type and index.
  ///
  /// \param dtype The device type (CPU, CUDA, HIP).
  /// \param index The device ordinal (0..N-1) for CUDA/HIP, or 0 for CPU.
  constexpr Device(DeviceType dtype, int8_t index) : device_type_(dtype), index_(index) {}

  /// The device type (CPU, CUDA, HIP).
  DeviceType device_type_ = DeviceType::kCpu;

  /// The device ordinal (0..N-1) for CUDA/HIP, or 0 for CPU.
  int8_t index_ = 0;
};

}  // namespace inferx
