#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"

namespace inferx {

/// \brief The alignment of anything a kernel reads.
///
/// 128 B is the L1/L2 cache line, so a row start maps a warp's 128 B request
/// onto exactly four 32 B sectors. Nothing on sm_89 asks for more -- 256 B is
/// the legacy texture-binding number. The hard floor is 16 B, the widest
/// vectorized load, below which CUTLASS drops to slower tiles.
///
/// This is a default for callers, not a property of any allocator: alignment is
/// a per-call parameter, since pinned memory destined for cudaHostRegister
/// wants page alignment instead. See docs/ARCHITECTURE.md section 6.2.
inline constexpr size_t kTensorAlignment = 128;

/// \brief The alignment of host memory destined for cudaHostRegister.
///
/// Pinning operates on whole pages, so a registration whose range does not
/// start on a page boundary pins the neighbouring allocation along with it.
inline constexpr size_t kPageAlignment = 4096;

enum class DeviceKind : uint8_t {
  kCpu = 0,
  kCuda = 1,
  kRocm = 2,
  kAscend = 3,
};

/// \brief Identifies one device as a (kind, ordinal) pair.
///
/// `DeviceId` is the currency of every placement decision in the engine: which
/// allocator backs a Storage, which stream a kernel launches on, which copy
/// path moves bytes between two buffers.
///
///   CPU    the single host address space. There is no notion of NUMA affinity,
///          so the ordinal is always 0 and `Cpu()` is a constant.
///   CUDA   one GPU, addressed by its runtime ordinal (0..N-1). The ordinal
///          pins a primary context and therefore a device-side allocator;
///          cross-device copies are explicit, never implicit.
struct DeviceId {
  /// \brief The default device, which is the host. Equal to `Cpu()`.
  ///
  /// Public because a DeviceId is a member of nearly every buffer and view in
  /// the engine, and a private default constructor would delete theirs.
  constexpr DeviceId() = default;

  /// The device class. See `DeviceKind`.
  DeviceKind kind = DeviceKind::kCpu;

  /// For CUDA, the runtime device ordinal (0..N-1). Always 0 for CPU.
  int8_t index = 0;

  /// \brief The canonical CPU device. Always (kCpu, 0).
  static constexpr DeviceId Cpu() { return DeviceId(DeviceKind::kCpu, 0); }

  /// \brief The CUDA device with the given runtime `ordinal` (0..N-1).
  static constexpr DeviceId Cuda(int8_t ordinal) {
    return DeviceId(DeviceKind::kCuda, ordinal);
  }

  static constexpr DeviceId Rocm(int8_t ordinal) {
    return DeviceId(DeviceKind::kRocm, ordinal);
  }

  static constexpr DeviceId Ascend(int8_t ordinal) {
    return DeviceId(DeviceKind::kAscend, ordinal);
  }

  /// \brief True if this device is the host (CPU).
  constexpr bool IsCpu() const { return kind == DeviceKind::kCpu; }

  /// \brief True if this device is a CUDA GPU.
  constexpr bool IsCuda() const { return kind == DeviceKind::kCuda; }

  constexpr bool IsAccelerator() const { return !IsCpu(); }

  friend constexpr bool operator==(const DeviceId& a, const DeviceId& b) {
    return a.kind == b.kind && a.index == b.index;
  }

  friend constexpr bool operator!=(const DeviceId& a, const DeviceId& b) {
    return !(a == b);
  }

  /// \brief Returns a string representation of the device ID.
  ///
  /// For CPU devices, this will return "cpu".
  /// For CUDA devices, this will return "cuda:<ordinal>", where <ordinal> is
  /// the CUDA device ordinal.
  ///
  /// \return A string representing the device ID.
  std::string ToString() const {
    if (IsCpu()) return "cpu";
    const char* name = "unknown";
    switch (kind) {
      case DeviceKind::kCpu:
        name = "cpu";
        break;
      case DeviceKind::kCuda:
        name = "cuda";
        break;
      case DeviceKind::kRocm:
        name = "rocm";
        break;
      case DeviceKind::kAscend:
        name = "ascend";
        break;
    }
    return absl::StrCat(name, ":", static_cast<int>(index));
  }

  template <typename H>
  friend H AbslHashValue(H h, const DeviceId& d) {
    return H::combine(std::move(h), d.kind, d.index);
  }

 private:
  // Private so that a device is always named through Cpu() or Cuda(): a bare
  // (kind, ordinal) pair at a call site reads as two unrelated numbers.
  constexpr DeviceId(DeviceKind k, int8_t i) : kind(k), index(i) {}
};

}  // namespace inferx
