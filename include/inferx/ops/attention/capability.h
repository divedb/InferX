#ifndef INFERX_ATTN_CAPABILITY_H_
#define INFERX_ATTN_CAPABILITY_H_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "inferx/core/dtype.h"
#include "inferx/ops/attention/spec.h"
#include "inferx/ops/attention/types.h"

namespace inferx::attn {

/// \brief Tiny constexpr bitset keyed by an enum. 64 bits: `DataType` has
///        more than 32 enumerators, so a 32-bit set would silently truncate.
template <class E>
struct EnumSet {
  std::uint64_t bits = 0;

  constexpr EnumSet() = default;
  constexpr EnumSet(std::initializer_list<E> list) {
    for (E e : list) bits |= 1ull << static_cast<unsigned long long>(e);
  }
  constexpr bool Has(E e) const noexcept {
    return bits & (1ull << static_cast<unsigned long long>(e));
  }
};

/// \brief Tokens-per-page constraint a kernel imposes (cf. vLLM's
///        MultipleOf-style kernel block sizes [V]).
struct BlockSizeRule {
  /// plan.block_size must be a multiple of this.
  std::int32_t multiple_of = 1;
  /// 0 = unbounded.
  std::int32_t max_tokens = 0;
};

/// \brief Which batch shapes a kernel handles in one Run() call.
struct PhaseSupport {
  bool decode_only = true;
  bool prefill_only = true;
  /// True: one launch handles decode+prefill together.
  bool mixed = false;
};

/// \brief A backend's abilities as a value, constructible at compile time
///        and matched by the pure CheckCapability() below.
struct Capability {
  std::string_view name;
  /// Mechanism / custom ids handled.
  std::span<const std::uint32_t> mechanisms;
  EnumSet<Role> roles;
  EnumSet<MaskKind> masks;
  FeatureSet features;
  EnumSet<DataType> q_dtypes;
  EnumSet<DataType> kv_dtypes;
  /// Empty = any.
  std::span<const std::int16_t> head_dims;
  /// Spec-declared layouts this backend reads.
  EnumSet<KvPageLayout> layouts;
  BlockSizeRule block;
  PhaseSupport phases;
  KvUpdateMode kv_update = KvUpdateMode::kSeparate;
  GraphSupport graph = GraphSupport::kNever;
  /// 1 = plain decode; >1 admits speculative verify steps.
  std::int32_t decode_threshold = 1;
  /// Compute capability x10 (e.g. 90); max 0 = open.
  std::int32_t min_sm = 0, max_sm = 0;
  bool needs_unit_stride_head_dim = true;
  /// E.g. reference backend.
  bool selectable_only_if_forced = false;
};

/// \brief What the selector knows about the machine and config. After cache
///        planning, `block_size` carries the resolved plan value so backends
///        are checked against reality, not a wish.
struct Env {
  /// E.g. 90; 0 = unknown (arch check skipped).
  int sm = 0;
  /// Resolved plan.block_size; 0 = pre-planning (block check skipped).
  std::int32_t block_size = 0;
  bool cuda_graphs_enabled = false;
  /// Minimum graph support the runner config requires.
  GraphSupport min_graph_support = GraphSupport::kNever;
  /// "" = auto.
  std::string_view forced_backend;
};

/// \brief Why CheckCapability rejected a spec. The order is fixed and
///        documented so rejection messages are stable across runs.
enum class Reject : std::uint8_t {
  kNone,
  kMechanism,
  kRole,
  kMask,
  kFeature,
  kQDtype,
  kKvDtype,
  kHeadDim,
  kLayout,
  kBlockSize,
  kArch,
  kGraph,
  kGqa,
};

/// \brief One rejection with a human-readable detail. Init-time only;
///        allocation in `detail` is acceptable.
struct Verdict {
  Reject why = Reject::kNone;
  std::string detail;
  explicit operator bool() const { return why == Reject::kNone; }
};

/// \brief Pure; init time only. Returns the first failing criterion in the
///        documented order: mechanism, role, mask, features, dtypes, head
///        dim, layout, block size, architecture, graph support, GQA
///        divisibility.
[[nodiscard]] Verdict CheckCapability(const Capability& capability, const AttentionSpec& spec,
                                      const Env& env);

}  // namespace inferx::attn

#endif  // INFERX_ATTN_CAPABILITY_H_
