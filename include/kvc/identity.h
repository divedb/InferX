// Portable cache identities: the canonical digest construction of
// docs/kvcache_plan.md section 8, SDK-side.
//
// Semantic identity precedes storage: matching tensor shapes never
// establish cache compatibility, and these digests are the only sanctioned
// way to produce the keys and dependency values the control plane carries.
// Ad-hoc hashing (pointer values, shapes, per-process seeds) cannot share
// state across processes or restarts.
//
// Encoding profile (v1): SHA-256 over a domain tag followed by
// length-prefixed fields; integers little-endian, fixed width; byte runs
// prefixed by their u64 length so no field can bleed into the next. The
// profile is fixed and versioned through the domain tags — a future
// encoding change must take a new tag, never silently mix.

#ifndef KVC_IDENTITY_H_
#define KVC_IDENTITY_H_

#include <cstdint>
#include <span>
#include <vector>

#include "kvc/abi/basics.h"

namespace kvc {

/// \brief One token's contribution to a prefix chain.
struct TokenIdentity {
  int64_t token = 0;      ///< Model input identity (token id).
  int64_t position = 0;   ///< Logical position; distinguishes repeats.
  /// Request-conditioning bytes at this position (multimodal placeholder
  /// digest, prompt-embedding digest); empty when absent.
  std::span<const std::byte> auxiliary;
};

/// \brief Domain-separated digest construction for portable identities.
///
/// The three levels follow plan section 7: model compatibility is caller
/// material (namespace + model digests agree on every call), the prefix
/// chain is token-level so identity never depends on any block size, and
/// block keys bind the chain to a group, extent, and partition.
class Identity {
 public:
  /// \brief Prefix seed P0: identifies the empty prefix.
  ///
  /// \param cache_namespace  Tenant/deployment namespace.
  /// \param model_identity   Complete model fingerprint digest.
  /// \param invariant_context Request-invariant conditioning (cache salt,
  ///                          negotiated adapter set, system context).
  /// \return H("kvc.prefix.v1" || ns || model || invariant_context).
  static KvcDigest PrefixSeed(const KvcDigest& cache_namespace,
                              const KvcDigest& model_identity,
                              std::span<const std::byte> invariant_context);

  /// \brief Chain step P(i+1) = H("kvc.token.v1" || P(i) || token ||
  ///        position || auxiliary).
  ///
  /// Position keeps the same token at different offsets distinct; the
  /// auxiliary digest keeps multimodal placeholders and prompt embeddings
  /// position- and content-sensitive.
  static KvcDigest ExtendPrefix(const KvcDigest& previous, const TokenIdentity& step);

  /// \brief Block key: the portable identity of one publication unit.
  ///
  /// \param dependency Prefix digest at the block's valid end (for causal
  ///                   attention; schema-defined for other semantics).
  /// \param logical_partition Canonical partition descriptor bytes.
  /// \return H("kvc.block.v1" || ns || model || partition || group ||
  ///           logical_index || first_token || token_count || dependency).
  static KvcDigest BlockKey(const KvcDigest& cache_namespace, const KvcDigest& model_identity,
                            std::span<const std::byte> logical_partition, uint32_t group,
                            uint64_t logical_index, uint64_t first_token,
                            uint64_t token_count, const KvcDigest& dependency);
};

/// \brief Computes prefix digests for boundaries [0, tokens.size()],
///        including the seed.
///
/// result[i] is the identity of the first `i` tokens. Pure: no storage
/// lookup, no schema-specific resume inference — callers compose candidates
/// from these digests.
std::vector<KvcDigest> ComputePrefixes(const KvcDigest& cache_namespace,
                                       const KvcDigest& model_identity,
                                       std::span<const std::byte> invariant_context,
                                       std::span<const TokenIdentity> tokens);

}  // namespace kvc

#endif  // KVC_IDENTITY_H_
