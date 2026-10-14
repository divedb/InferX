#include "inferx/cache/sha256.h"

#include <cstring>

namespace inferx {
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

void StoreBigEndian(uint8_t* out, uint32_t value) {
  out[0] = static_cast<uint8_t>(value >> 24);
  out[1] = static_cast<uint8_t>(value >> 16);
  out[2] = static_cast<uint8_t>(value >> 8);
  out[3] = static_cast<uint8_t>(value);
}

}  // namespace

void Sha256::Reset() {
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

void Sha256::Compress(const uint8_t* chunk) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(chunk[4 * i]) << 24) |
           (static_cast<uint32_t>(chunk[4 * i + 1]) << 16) |
           (static_cast<uint32_t>(chunk[4 * i + 2]) << 8) |
           static_cast<uint32_t>(chunk[4 * i + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = RotateRight(w[i - 15], 7) ^ RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const uint32_t s1 = RotateRight(w[i - 2], 17) ^ RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
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

void Sha256::Update(std::span<const std::byte> bytes) {
  total_bytes_ += bytes.size();
  const uint8_t* data = reinterpret_cast<const uint8_t*>(bytes.data());
  size_t remaining = bytes.size();

  if (buffered_ > 0) {
    const size_t fill = sizeof(buffer_) - buffered_ < remaining ? sizeof(buffer_) - buffered_
                                                                : remaining;
    std::memcpy(buffer_ + buffered_, data, fill);
    buffered_ += fill;
    data += fill;
    remaining -= fill;
    if (buffered_ == sizeof(buffer_)) {
      Compress(buffer_);
      buffered_ = 0;
    }
  }
  while (remaining >= sizeof(buffer_)) {
    Compress(data);
    data += sizeof(buffer_);
    remaining -= sizeof(buffer_);
  }
  if (remaining > 0) {
    std::memcpy(buffer_, data, remaining);
    buffered_ = remaining;
  }
}

std::array<uint8_t, 32> Sha256::Finish() {
  const uint64_t bits = total_bytes_ * 8;
  const uint8_t one = 0x80;
  Update(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&one), 1));
  const uint8_t zero = 0;
  while (buffered_ != 56) {
    Update(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&zero), 1));
  }
  uint8_t length[8];
  for (int i = 0; i < 8; ++i) {
    length[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
  }
  Update(length, sizeof(length));

  std::array<uint8_t, 32> digest{};
  for (int i = 0; i < 8; ++i) StoreBigEndian(digest.data() + 4 * i, state_[i]);
  Reset();
  return digest;
}

}  // namespace inferx
