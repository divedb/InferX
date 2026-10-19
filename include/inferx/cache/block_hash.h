#pragma once

#include <array>
#include <cstdint>
#include <cstring>

#include "absl/types/span.h"
#include "inferx/tokenizer/id.h"

namespace inferx {

struct BlockHash {
  std::array<uint8_t, 32> bytes{};  // SHA-256, or truncated to 16 if you prefer
  bool operator==(const BlockHash&) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const BlockHash& k) {
    uint64_t w;  // the digest is already uniform; hash its first 8 bytes
    std::memcpy(&w, k.bytes.data(), sizeof(w));

    return H::combine(std::move(h), w);
  }
};

/// \brief One link of the facade-owned hash chain: the digest of `parent`
///        chained with one block of tokens, with optional `salt` bytes mixed
///        into the first link only, so a salt namespaces every later digest.
///
/// Domain-separated and length-framed; deterministic across processes and
/// builds. The facade stores whole chains as plain digest vectors
/// (CacheManager::RequestState::hashes) and passes the parent explicitly,
/// which is why this is a free function rather than a tip-holding object.
BlockHash ChainHash(const BlockHash& parent, absl::Span<const TokenId> tokens,
                    absl::Span<const uint8_t> salt);

}  // namespace inferx
