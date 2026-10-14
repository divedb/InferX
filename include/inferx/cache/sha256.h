#ifndef INFERX_CACHE_SHA256_H_
#define INFERX_CACHE_SHA256_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace inferx {

/// \brief SHA-256 (FIPS 180-4) over concatenated byte ranges.
///
/// Self-contained so the cache layer — and anything else that needs stable
/// content digests — never grows a crypto dependency. The prefix-cache hash
/// chain uses it because block hashes must be deterministic across processes
/// and builds; per-process seeded hashes cannot provide that.
class Sha256 {
 public:
  Sha256() { Reset(); }

  /// \brief Resets to a fresh state, as if just constructed.
  void Reset();

  /// \brief Absorbs `bytes`.
  void Update(std::span<const std::byte> bytes);

  /// \brief Absorbs a byte range.
  void Update(const void* data, size_t bytes) {
    Update(std::span<const std::byte>(static_cast<const std::byte*>(data), bytes));
  }

  /// \brief Finishes the digest and resets the state for reuse.
  ///
  /// \return The 32-byte digest.
  std::array<uint8_t, 32> Finish();

  /// \brief One-shot convenience.
  static std::array<uint8_t, 32> Digest(std::span<const std::byte> bytes) {
    Sha256 sha;
    sha.Update(bytes);
    return sha.Finish();
  }

 private:
  void Compress(const uint8_t* chunk);

  uint32_t state_[8];
  uint64_t total_bytes_ = 0;
  uint8_t buffer_[64];
  size_t buffered_ = 0;
};

}  // namespace inferx

#endif  // INFERX_CACHE_SHA256_H_
