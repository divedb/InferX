// Portable-identity tests: chain semantics, field sensitivity, and golden
// vectors that pin the encoding profile (any accidental change to the
// digest construction breaks these loudly). The kvc suite does not link
// googletest; it uses the shared CHECK framework.

#include <cstring>
#include <string>
#include <vector>

#include "kvc/identity.h"
#include "test_util.h"

namespace {

using KvcDigest = ::KvcDigest;
using kvc::TokenIdentity;

KvcDigest DigestOf(char fill) {
  KvcDigest digest{};
  for (size_t i = 0; i < sizeof(digest.bytes); ++i) {
    digest.bytes[i] = static_cast<unsigned char>(fill + i);
  }
  return digest;
}

std::string Hex(const KvcDigest& digest) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(sizeof(digest.bytes) * 2);
  for (const unsigned char byte : digest.bytes) {
    out.push_back(kDigits[byte >> 4]);
    out.push_back(kDigits[byte & 0xf]);
  }
  return out;
}

const std::byte* Bytes(const char* text) {
  return reinterpret_cast<const std::byte*>(text);
}

std::vector<TokenIdentity> Sequence(int64_t count) {
  std::vector<TokenIdentity> tokens;
  for (int64_t i = 0; i < count; ++i) {
    tokens.push_back({1000 + i, i, {}});
  }
  return tokens;
}

void SeedBindsNamespaceModelAndContext() {
  const KvcDigest ns = DigestOf(1);
  const KvcDigest model = DigestOf(2);
  const auto base = kvc::Identity::PrefixSeed(ns, model, {});
  CHECK(kvc_test::DigestEq(base, kvc::Identity::PrefixSeed(ns, model, {})));
  CHECK(!kvc_test::DigestEq(base, kvc::Identity::PrefixSeed(DigestOf(9), model, {})));
  CHECK(!kvc_test::DigestEq(base, kvc::Identity::PrefixSeed(ns, DigestOf(9), {})));
  CHECK(!kvc_test::DigestEq(
      base, kvc::Identity::PrefixSeed(ns, model, std::span(Bytes("salt"), 4))));
  // Equal bytes of different lengths stay distinct (length prefix).
  const auto ctx4 = kvc::Identity::PrefixSeed(ns, model, std::span(Bytes("salt"), 4));
  const auto ctx3 = kvc::Identity::PrefixSeed(ns, model, std::span(Bytes("sal"), 3));
  CHECK(!kvc_test::DigestEq(ctx4, ctx3));
}

void ChainIsPrefixStableAndMutationSensitive() {
  const KvcDigest ns = DigestOf(1);
  const KvcDigest model = DigestOf(2);
  const KvcDigest seed = kvc::Identity::PrefixSeed(ns, model, {});

  auto tokens = Sequence(4);
  const auto full = kvc::ComputePrefixes(ns, model, {}, tokens);
  CHECK_EQ(full.size(), 5u);
  CHECK(kvc_test::DigestEq(full[0], seed));

  // A shorter sequence's digests are a prefix of the longer one's.
  tokens.resize(2);
  const auto shorter = kvc::ComputePrefixes(ns, model, {}, tokens);
  CHECK_EQ(shorter.size(), 3u);
  CHECK(kvc_test::DigestEq(shorter[0], full[0]));
  CHECK(kvc_test::DigestEq(shorter[1], full[1]));
  CHECK(kvc_test::DigestEq(shorter[2], full[2]));

  // Manual chaining agrees with the batch helper.
  CHECK(kvc_test::DigestEq(full[3], kvc::Identity::ExtendPrefix(full[2], {1002, 2, {}})));

  // Mutating any later token leaves earlier boundaries unchanged.
  std::vector<TokenIdentity> mutated = Sequence(4);
  mutated[3].token = 4242;
  const auto other = kvc::ComputePrefixes(ns, model, {}, mutated);
  CHECK(kvc_test::DigestEq(other[3], full[3]));
  CHECK(!kvc_test::DigestEq(other[4], full[4]));

  // Position and auxiliary content participate, not just the token id.
  CHECK(!kvc_test::DigestEq(kvc::Identity::ExtendPrefix(seed, {5, 0, {}}),
                            kvc::Identity::ExtendPrefix(seed, {5, 1, {}})));
  const std::byte mm[2] = {std::byte{1}, std::byte{2}};
  CHECK(!kvc_test::DigestEq(kvc::Identity::ExtendPrefix(seed, {5, 0, {}}),
                            kvc::Identity::ExtendPrefix(seed, {5, 0, mm})));
}

void BlockKeyBindsEveryField() {
  const KvcDigest ns = DigestOf(1);
  const KvcDigest model = DigestOf(2);
  const std::byte partition[3] = {std::byte{9}, std::byte{8}, std::byte{7}};
  const KvcDigest dependency = DigestOf(4);
  const KvcDigest base =
      kvc::Identity::BlockKey(ns, model, partition, 0, 0, 0, 16, dependency);

  CHECK(!kvc_test::DigestEq(base, kvc::Identity::BlockKey(DigestOf(9), model, partition, 0,
                                                           0, 0, 16, dependency)));
  CHECK(!kvc_test::DigestEq(base, kvc::Identity::BlockKey(ns, DigestOf(9), partition, 0, 0,
                                                           0, 16, dependency)));
  const std::byte other_partition[3] = {std::byte{1}, std::byte{2}, std::byte{3}};
  CHECK(!kvc_test::DigestEq(
      base, kvc::Identity::BlockKey(ns, model, other_partition, 0, 0, 0, 16, dependency)));
  CHECK(!kvc_test::DigestEq(base,
                            kvc::Identity::BlockKey(ns, model, partition, 1, 0, 0, 16,
                                                    dependency)));
  CHECK(!kvc_test::DigestEq(base,
                            kvc::Identity::BlockKey(ns, model, partition, 0, 1, 0, 16,
                                                    dependency)));
  CHECK(!kvc_test::DigestEq(base,
                            kvc::Identity::BlockKey(ns, model, partition, 0, 0, 16, 16,
                                                    dependency)));
  CHECK(!kvc_test::DigestEq(base,
                            kvc::Identity::BlockKey(ns, model, partition, 0, 0, 0, 15,
                                                    dependency)));
  CHECK(!kvc_test::DigestEq(base,
                            kvc::Identity::BlockKey(ns, model, partition, 0, 0, 0, 16,
                                                    DigestOf(9))));
  CHECK(kvc_test::DigestEq(base,
                           kvc::Identity::BlockKey(ns, model, partition, 0, 0, 0, 16,
                                                   dependency)));
}

void GoldenVectorsPinTheEncodingProfile() {
  // Regenerate deliberately (never silently) by re-running the construction
  // and re-embedding; every consumer of stored identities must agree.
  const KvcDigest ns = DigestOf(1);
  const KvcDigest model = DigestOf(2);
  CHECK_EQ(Hex(kvc::Identity::PrefixSeed(ns, model, {})),
           "c498bb73ff3eca1bb8d9b8042be115cb4430c70c680dafff3a95b94fb0d277cb");
  const auto chain = kvc::ComputePrefixes(ns, model, std::span(Bytes("sys"), 3), Sequence(3));
  CHECK_EQ(Hex(chain.back()),
           "2477e8751758a5cd0eb62c27816cc0e184a2a073de8da3eed7c14b2d375e93d5");
  CHECK_EQ(Hex(kvc::Identity::BlockKey(ns, model, {}, 7, 2, 32, 16, DigestOf(4))),
           "bc11075f80decd13e0865e60d6b085c0d652ce6b99fefa48e854c8ad45316333");
}

}  // namespace

int main() {
  TEST(SeedBindsNamespaceModelAndContext);
  TEST(ChainIsPrefixStableAndMutationSensitive);
  TEST(BlockKeyBindsEveryField);
  TEST(GoldenVectorsPinTheEncodingProfile);
  return kvc_test::ExitCode();
}
