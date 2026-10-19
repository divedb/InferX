#ifndef INFERX_ATTN_SPEC_H_
#define INFERX_ATTN_SPEC_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <variant>

#include "inferx/core/dtype.h"
#include "inferx/ops/attention/types.h"

namespace inferx::attn {

/// \brief Who attends to whom.
enum class Role : std::uint8_t { kDecoderSelf, kEncoderSelf, kCross };

/// \brief Which positions each query attends to.
enum class MaskKind : std::uint8_t {
  kCausal,
  kBidirectional,
  /// Causal + trailing `window` tokens, inclusive of the current one.
  kSlidingWindow,
  /// Causal within fixed-size chunks; `window` = chunk length.
  kChunked,
};

/// \brief The compute-side mask description. Window and chunk lengths here
///        drive the cache-side retention through BuildCacheSpec.
struct MaskSpec {
  MaskKind kind = MaskKind::kCausal;
  /// Tokens; meaningful for kSlidingWindow / kChunked.
  std::int32_t window = 0;
  /// Permanently retained prefix tokens (sliding window with sinks,
  /// gpt-oss style); must be a multiple of the block size — the cache
  /// planner validates.
  std::int32_t sink_tokens = 0;

  friend bool operator==(const MaskSpec&, const MaskSpec&) = default;
};

/// \brief Optional behaviors, orthogonal to the mechanism.
enum class Feature : std::uint8_t { kSoftCap, kSinks, kAlibi, kReturnLse, kQkNormInKernel, kCount };

/// \brief Set of optional behaviors a layer uses.
struct FeatureSet {
  std::uint32_t bits = 0;

  constexpr bool Has(Feature f) const noexcept { return bits & (1u << static_cast<unsigned>(f)); }
  constexpr FeatureSet With(Feature f) const noexcept {
    return {bits | (1u << static_cast<unsigned>(f))};
  }
  friend bool operator==(const FeatureSet&, const FeatureSet&) = default;
};

/// \brief MHA (kv == q), MQA (kv == 1), GQA (1 < kv < q): one block.
///        They are a number, not a type.
struct DenseParams {
  std::int32_t num_q_heads = 0;
  /// Validated: num_q_heads % num_kv_heads == 0.
  std::int32_t num_kv_heads = 0;
  /// Q/K head dimension.
  std::int32_t head_dim = 0;
  /// 0 => same as head_dim.
  std::int32_t v_head_dim = 0;

  friend bool operator==(const DenseParams&, const DenseParams&) = default;
};

/// \brief Multi-head Latent Attention. The cached state is the latent
///        (+ rope part), not per-head K/V.
struct MlaParams {
  std::int32_t num_heads = 0;
  /// Latent width cached per token.
  std::int32_t kv_lora_rank = 0;
  std::int32_t qk_nope_dim = 0;
  /// Cached alongside the latent.
  std::int32_t qk_rope_dim = 0;
  std::int32_t v_head_dim = 0;

  friend bool operator==(const MlaParams&, const MlaParams&) = default;
};

/// \brief Linear / recurrent attention (gated delta-rule, SSM-style, ...).
///        State is O(1) per request.
struct LinearParams {
  /// Identifies the update rule; kernels key on it.
  std::uint16_t recurrence_id = 0;
  std::int32_t num_k_heads = 0;
  std::int32_t num_v_heads = 0;
  std::int32_t k_dim = 0;
  std::int32_t v_dim = 0;
  /// 0 => no short-conv state component.
  std::int32_t conv_width = 0;
  /// Tokens between state-validity boundaries; must be a positive multiple
  /// of the block size (the cache planner validates). Drives
  /// RecurrentRetention's boundary.
  std::int32_t boundary_tokens = 0;

  friend bool operator==(const LinearParams&, const LinearParams&) = default;
};

/// \brief Sparse attention: a dense base + a rule choosing which KV
///        positions participate.
enum class SparseKind : std::uint8_t {
  /// Fixed pattern computable from (position, params): strided/local+global
  /// blocks.
  kStaticBlock,
  /// An indexer produces per-token KV indices each step (an extra input).
  kIndexedTopK,
};

struct SparseParams {
  DenseParams base;
  SparseKind kind = SparseKind::kStaticBlock;
  std::int32_t block_size = 0;
  std::int32_t top_k = 0;

  friend bool operator==(const SparseParams&, const SparseParams&) = default;
};

/// \brief Escape hatch for mechanisms not yet known. Opaque to the core;
///        only matching backends read it.
struct CustomParams {
  /// From a mechanism registry; >= 0x1000.
  std::uint32_t mechanism_id = 0;
  /// Trivially-copyable payload (bit_cast target).
  std::array<std::byte, 64> blob{};

  friend bool operator==(const CustomParams&, const CustomParams&) = default;
};

using MechanismParams =
    std::variant<DenseParams, MlaParams, LinearParams, SparseParams, CustomParams>;

/// \brief Stable mechanism ids used by Capability matching.
enum class Mechanism : std::uint32_t {
  kDense = 1,
  kMla = 2,
  kLinear = 3,
  kSparse = 4,
  kCustomBase = 0x1000,
};

/// \brief How the family physically lays out one token's cached bytes.
///        Declared by the spec (the family writes it), filtered by backends.
enum class KvPageLayout : std::uint8_t {
  // Within one page (page = `block_tokens` tokens for one layer):
  /// [2, block_tokens, Hkv, D] — K plane then V plane, token-major.
  kKvPlanarNhd,
  /// [2, Hkv, block_tokens, D] — head-major.
  kKvPlanarHnd,
  /// [block_tokens, 2, Hkv, D].
  kKvInterleaved,
  /// [block_tokens, rank + rope] (MLA).
  kLatent,
  /// Opaque kernel-defined layout for one recurrent-state component.
  kRecurrentFlat,
};

/// \brief Full static description of one layer's attention: the *what*,
///        never the *how*. Hashable and comparable so it works as a key
///        for selection caching and grouping.
struct AttentionSpec {
  Role role = Role::kDecoderSelf;
  MaskSpec mask{};
  MechanismParams params{};
  /// Activation dtype.
  DataType q_dtype = DataType::kBFloat16;
  /// Cache dtype (may be fp8).
  DataType kv_dtype = DataType::kBFloat16;
  KvPageLayout kv_layout = KvPageLayout::kKvPlanarNhd;
  /// 0 => 1/sqrt(head_dim).
  float softmax_scale = 0.0f;
  /// Valid if features.Has(Feature::kSoftCap).
  float softcap = 0.0f;
  FeatureSet features{};

  /// \brief Maps the variant alternative to a Mechanism id (or, for
  ///        CustomParams, the custom id it carries).
  [[nodiscard]] std::uint32_t MechanismId() const noexcept {
    return std::visit(
        [](const auto& p) -> std::uint32_t {
          using T = std::decay_t<decltype(p)>;
          if constexpr (std::is_same_v<T, DenseParams>) {
            return static_cast<std::uint32_t>(Mechanism::kDense);
          } else if constexpr (std::is_same_v<T, MlaParams>) {
            return static_cast<std::uint32_t>(Mechanism::kMla);
          } else if constexpr (std::is_same_v<T, LinearParams>) {
            return static_cast<std::uint32_t>(Mechanism::kLinear);
          } else if constexpr (std::is_same_v<T, SparseParams>) {
            return static_cast<std::uint32_t>(Mechanism::kSparse);
          } else {
            return p.mechanism_id;
          }
        },
        params);
  }

  /// \brief FNV-1a over every field; stable within a process, which is all
  ///        selection caching needs (specs never persist).
  [[nodiscard]] std::size_t Hash() const noexcept {
    std::size_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t value) {
      h ^= value;
      h *= 1099511628211ull;
    };
    mix(static_cast<std::uint8_t>(role));
    mix(static_cast<std::uint8_t>(mask.kind));
    mix(static_cast<std::uint32_t>(mask.window));
    mix(static_cast<std::uint32_t>(mask.sink_tokens));
    mix(static_cast<std::uint8_t>(q_dtype));
    mix(static_cast<std::uint8_t>(kv_dtype));
    mix(static_cast<std::uint8_t>(kv_layout));
    // Floats hash through their bit patterns so +0.0/-0.0 stay distinct
    // only when they are distinct values.
    std::uint32_t scale_bits = 0, cap_bits = 0;
    std::memcpy(&scale_bits, &softmax_scale, sizeof(scale_bits));
    std::memcpy(&cap_bits, &softcap, sizeof(cap_bits));
    mix(scale_bits);
    mix(cap_bits);
    mix(features.bits);
    mix(params.index());
    std::visit(
        [&](const auto& p) {
          using T = std::decay_t<decltype(p)>;
          if constexpr (std::is_same_v<T, CustomParams>) {
            mix(p.mechanism_id);
            for (std::size_t i = 0; i < p.blob.size(); i += sizeof(std::uint64_t)) {
              std::uint64_t word = 0;
              std::memcpy(&word, p.blob.data() + i, sizeof(word));
              mix(word);
            }
          } else {
            // Every parameter block is a sequence of int32/uint16 fields;
            // fold them as raw words.
            const auto* bytes = reinterpret_cast<const std::byte*>(&p);
            for (std::size_t i = 0; i < sizeof(p); i += sizeof(std::uint32_t)) {
              std::uint32_t word = 0;
              const std::size_t n = std::min(sizeof(std::uint32_t), sizeof(p) - i);
              std::memcpy(&word, bytes + i, n);
              mix(word);
            }
          }
        },
        params);
    return h;
  }

  friend bool operator==(const AttentionSpec&, const AttentionSpec&) = default;
};

}  // namespace inferx::attn

#endif  // INFERX_ATTN_SPEC_H_
