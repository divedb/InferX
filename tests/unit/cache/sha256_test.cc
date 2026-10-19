// Unit tests for the SHA-256 primitive the cache layer digests with.

#include "inferx/cache/sha256.h"

#include <gtest/gtest.h>

#include <array>
#include <string>

namespace inferx {
namespace {

std::string Hex(const std::array<uint8_t, 32>& digest) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(digest.size() * 2);
  for (const uint8_t byte : digest) {
    out.push_back(kDigits[byte >> 4]);
    out.push_back(kDigits[byte & 0xf]);
  }
  return out;
}

TEST(Sha256Test, MatchesFipsVectors) {
  // FIPS 180-4 / NIST example vectors.
  EXPECT_EQ(Hex(Sha256::Digest(std::span<const std::byte>())),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  const std::string abc = "abc";
  EXPECT_EQ(Hex(Sha256::Digest(
                std::span<const std::byte>(reinterpret_cast<const std::byte*>(abc.data()),
                                           abc.size()))),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const std::string long_input(1000 * 1000, 'a');
  EXPECT_EQ(Hex(Sha256::Digest(std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(long_input.data()),
                    long_input.size()))),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256Test, IncrementalMatchesOneShot) {
  const std::string data = "streamed update alignment test payload";
  const auto span = std::span<const std::byte>(
      reinterpret_cast<const std::byte*>(data.data()), data.size());
  Sha256 one_shot;
  one_shot.Update(span);
  Sha256 streamed;
  for (size_t i = 0; i < data.size(); ++i) {
    streamed.Update(span.subspan(i, 1));
  }
  EXPECT_EQ(one_shot.Finish(), streamed.Finish());
  // Finish() resets: a reused instance digests only new input.
  Sha256 reused;
  reused.Update(span);
  reused.Finish();
  EXPECT_EQ(reused.Finish(), Sha256::Digest(std::span<const std::byte>()));
}

}  // namespace
}  // namespace inferx
