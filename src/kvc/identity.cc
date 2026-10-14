// SHA-256 and the canonical record encoding behind kvc portable identities.
//
// The hash implementation is deliberately self-contained: the SDK installs
// into arbitrary toolchains and must not grow an OpenSSL, Abseil, or engine
// dependency for the one primitive identity needs.

#include "kvc/identity.h"

#include <array>
#include <cstring>

namespace kvc {
namespace {

constexpr uint32_t kRoundConstants[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t RotateRight(uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
 public:
  Sha256() { Reset(); }

  void Reset() {
    state_[0] = 0x6a09e667;
    state_[1] = 0xbb67ae85;
    state_[2] = 0x3c6ef372;
    state_[3] = 0xa54ff53a;
    state_[4] = 0x510e527f;
    state_[5] = 0x9b05688c;
    state_[6] = 0x1f83d9ab;
    state_[7] = 0x5be0cd19;
    total_bytes_ = 0;
    buffered_ = 0;
  }

  void Update(const void* data, size_t bytes) {
    total_bytes_ += bytes;
    const uint8_t* cursor = static_cast<const uint8_t*>(data);
    size_t remaining = bytes;
    if (buffered_ > 0) {
      const size_t fill = sizeof(buffer_) - buffered_ < remaining
                              ? sizeof(buffer_) - buffered_
                              : remaining;
      std::memcpy(buffer_ + buffered_, cursor, fill);
      buffered_ += fill;
      cursor += fill;
      remaining -= fill;
      if (buffered_ == sizeof(buffer_)) {
        Compress(buffer_);
        buffered_ = 0;
      }
    }
    while (remaining >= sizeof(buffer_)) {
      Compress(cursor);
      cursor += sizeof(buffer_);
      remaining -= sizeof(buffer_);
    }
    if (remaining > 0) {
      std::memcpy(buffer_, cursor, remaining);
      buffered_ = remaining;
    }
  }

  KvcDigest Finish() {
    const uint64_t bits = total_bytes_ * 8;
    const uint8_t one = 0x80;
    Update(&one, 1);
    const uint8_t zero = 0;
    while (buffered_ != 56) Update(&zero, 1);
    uint8_t length[8];
    for (int i = 0; i < 8; ++i) length[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    Update(length, sizeof(length));

    KvcDigest digest{};
    for (int i = 0; i < 8; ++i) {
      digest.bytes[4 * i + 0] = static_cast<unsigned char>(state_[i] >> 24);
      digest.bytes[4 * i + 1] = static_cast<unsigned char>(state_[i] >> 16);
      digest.bytes[4 * i + 2] = static_cast<unsigned char>(state_[i] >> 8);
      digest.bytes[4 * i + 3] = static_cast<unsigned char>(state_[i]);
    }
    Reset();
    return digest;
  }

 private:
  void Compress(const uint8_t* chunk) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(chunk[4 * i]) << 24) |
             (static_cast<uint32_t>(chunk[4 * i + 1]) << 16) |
             (static_cast<uint32_t>(chunk[4 * i + 2]) << 8) |
             static_cast<uint32_t>(chunk[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 =
          RotateRight(w[i - 15], 7) ^ RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 =
          RotateRight(w[i - 2], 17) ^ RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + w[i];
      const uint32_t s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  uint32_t state_[8];
  uint64_t total_bytes_ = 0;
  uint8_t buffer_[64];
  size_t buffered_ = 0;
};

/// Canonical record builder. Every field -- including the domain tag --
/// is encoded as a u64 little-endian length prefix followed by the raw
/// bytes; integers travel as their fixed-width little-endian
/// representations. No field can bleed into the next, and no ambiguity
/// survives a future reordering: profiles change only through new domain
/// tags.
class Record {
 public:
  explicit Record(const char* domain) { Tag(domain); }

  void Tag(const char* tag) { Bytes(tag, std::strlen(tag)); }

  void Bytes(const void* data, size_t size) {
    uint8_t length[8];
    const uint64_t bytes = static_cast<uint64_t>(size);
    for (int i = 0; i < 8; ++i) length[i] = static_cast<uint8_t>(bytes >> (8 * i));
    sha_.Update(length, sizeof(length));
    sha_.Update(data, size);
  }

  void Bytes(std::span<const std::byte> value) {
    Bytes(value.data(), value.size());
  }

  void Digest(const KvcDigest& digest) {
    Bytes(digest.bytes, sizeof(digest.bytes));
  }

  void U32(uint32_t value) {
    uint8_t encoded[4];
    for (int i = 0; i < 4; ++i) encoded[i] = static_cast<uint8_t>(value >> (8 * i));
    Bytes(encoded, sizeof(encoded));
  }

  void U64(uint64_t value) {
    uint8_t encoded[8];
    for (int i = 0; i < 8; ++i) encoded[i] = static_cast<uint8_t>(value >> (8 * i));
    Bytes(encoded, sizeof(encoded));
  }

  void I64(int64_t value) { U64(static_cast<uint64_t>(value)); }

  KvcDigest Finish() { return sha_.Finish(); }

 private:
  Sha256 sha_;
};

}  // namespace

KvcDigest Identity::PrefixSeed(const KvcDigest& cache_namespace,
                               const KvcDigest& model_identity,
                               std::span<const std::byte> invariant_context) {
  Record record("kvc.prefix.v1");
  record.Digest(cache_namespace);
  record.Digest(model_identity);
  record.Bytes(invariant_context);
  return record.Finish();
}

KvcDigest Identity::ExtendPrefix(const KvcDigest& previous, const TokenIdentity& step) {
  Record record("kvc.token.v1");
  record.Digest(previous);
  record.I64(step.token);
  record.I64(step.position);
  record.Bytes(step.auxiliary);
  return record.Finish();
}

KvcDigest Identity::BlockKey(const KvcDigest& cache_namespace, const KvcDigest& model_identity,
                             std::span<const std::byte> logical_partition, uint32_t group,
                             uint64_t logical_index, uint64_t first_token,
                             uint64_t token_count, const KvcDigest& dependency) {
  Record record("kvc.block.v1");
  record.Digest(cache_namespace);
  record.Digest(model_identity);
  record.Bytes(logical_partition);
  record.U32(group);
  record.U64(logical_index);
  record.U64(first_token);
  record.U64(token_count);
  record.Digest(dependency);
  return record.Finish();
}

std::vector<KvcDigest> ComputePrefixes(const KvcDigest& cache_namespace,
                                       const KvcDigest& model_identity,
                                       std::span<const std::byte> invariant_context,
                                       std::span<const TokenIdentity> tokens) {
  std::vector<KvcDigest> prefixes;
  prefixes.reserve(tokens.size() + 1);
  prefixes.push_back(Identity::PrefixSeed(cache_namespace, model_identity, invariant_context));
  for (const TokenIdentity& step : tokens) {
    prefixes.push_back(Identity::ExtendPrefix(prefixes.back(), step));
  }
  return prefixes;
}

}  // namespace kvc
