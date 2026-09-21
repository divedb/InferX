// ABI control-plane records: block identity, reuse candidates, and the
// lookup and write-transaction requests served by the control table.

#ifndef KVC_ABI_CONTROL_H_
#define KVC_ABI_CONTROL_H_

#include "kvc/abi/basics.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Caller-chosen 16-byte transaction identifier. Must be unique
/// within its session for the transaction's lifetime; reuse is Conflict.
typedef struct KvcTransactionId {
  unsigned char bytes[16];
} KvcTransactionId;

/// \brief Lifecycle state of a write transaction, observed through commit
/// and query.
enum KvcWriteState {
  KVC_ALLOCATED = 1,  ///< Reserved; no components stored yet.
  KVC_WRITING,        ///< At least one component stored.
  KVC_COMMITTED,      ///< Published atomically; manifest is the receipt.
  KVC_WRITE_ABORTED   ///< Rolled back; the handle is dead.
};

/// \brief Immutable publication unit. Blocks become visible to lookup only
/// through commit; identity is (key, group, logical extent, dependency).
typedef struct KvcBlock {
  KvcDigest key;           ///< Content digest of this block's state.
  uint32_t group;          ///< Owning group identifier.
  uint64_t logical_index;  ///< Block position in the logical sequence.
  uint64_t first_token;    ///< Must equal logical_index * tokens_per_block.
  uint64_t token_count;    ///< Valid tokens; 1..tokens_per_block.
  KvcDigest dependency;    ///< Prefix digest at the block's valid end.
} KvcBlock;

/// \brief A reusable prefix: its digest and the token count it covers.
typedef struct KvcPrefix {
  KvcDigest digest;
  uint64_t token_count;
} KvcPrefix;

/// \brief Half-open range of indices into a request-owned array.
typedef struct KvcRange {
  uint64_t first;
  uint64_t count;
} KvcRange;

/// \brief One reuse plan: a prefix plus the catalog ranges it requires.
/// Candidates are evaluated independently; the longest fully available one
/// wins, so availability need not be monotonic.
typedef struct KvcCandidate {
  KvcPrefix prefix;
  const KvcRange* required;  ///< Ranges into the request's catalog.
  uint64_t required_count;
} KvcCandidate;

/// \brief Lookup request: a catalog of validated blocks plus candidates
/// composed from it.
typedef struct KvcLookupRequest {
  uint32_t struct_size;
  const KvcBlock* catalog;  ///< Unique keys, provider-validated extents.
  uint64_t catalog_count;
  const KvcCandidate* candidates;
  uint64_t candidate_count;
} KvcLookupRequest;

/// \brief Lookup outcome. A NULL read handle is a normal miss, not an
/// error; a hit pins the returned generations against eviction.
typedef struct KvcLookupResult {
  uint32_t struct_size;
  KvcPrefix prefix;  ///< The matched candidate's prefix.
  KvcObject* read;   ///< One owned reference; NULL is a normal miss.
} KvcLookupResult;

/// \brief Begin-publish request: reserves every block privately under one
/// transaction id. No content is visible until commit.
typedef struct KvcWriteRequest {
  uint32_t struct_size;
  KvcTransactionId transaction;
  const KvcBlock* blocks;  ///< Manifest; keys unique, unpublished,
                           ///< unreserved.
  uint64_t block_count;
} KvcWriteRequest;

/// \brief Transaction state plus, once committed, its receipt. blocks is
/// borrowed until session release.
typedef struct KvcWriteStatus {
  uint32_t struct_size;
  uint32_t state;  ///< A KvcWriteState value.
  // Borrowed until session release; committed records form the receipt.
  const KvcBlock* blocks;
  uint64_t block_count;
} KvcWriteStatus;

#ifdef __cplusplus
}
#endif
#endif  // KVC_ABI_CONTROL_H_
