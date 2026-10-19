#include "inferx/ops/attention/cache_spec_bridge.h"

#include <variant>

#include "inferx/core/dtype.h"

namespace inferx::attn {
namespace {

/// Bytes of one GQA-style K/V pair per token, matching GqaKvComponent so
/// the bridge and the factory can never disagree.
StatusOr<int64_t> DenseBytesPerToken(int64_t kv_heads, int64_t head_dim, DataType dtype) {
  if (kv_heads <= 0 || head_dim <= 0) {
    return InvalidArgumentError("dense attention needs positive kv_heads and head_dim, got kv=",
                                kv_heads, " d=", head_dim);
  }
  return DataTypeByteSize(dtype, 2 * kv_heads * head_dim);
}

/// Retention + components shared by every token-wise mechanism: the mask
/// picks the retention, the component comes precomputed by the caller.
StatusOr<CacheSpec> TokenWiseSpec(const AttentionSpec& spec, CacheComponent component) {
  switch (spec.mask.kind) {
    case MaskKind::kCausal:
    case MaskKind::kBidirectional:
      if (spec.role == Role::kCross) {
        return CacheSpec::Encoder({std::move(component)});
      }
      return CacheSpec::FullAttention({std::move(component)});
    case MaskKind::kSlidingWindow:
      if (spec.mask.window <= 0) {
        return InvalidArgumentError("kSlidingWindow requires window > 0, got ",
                                    spec.mask.window);
      }
      if (spec.mask.sink_tokens < 0) {
        return InvalidArgumentError("sink_tokens must be >= 0, got ", spec.mask.sink_tokens);
      }
      if (spec.role == Role::kCross) {
        return InvalidArgumentError("cross-attention does not take a sliding window");
      }
      return CacheSpec::SlidingWindow({std::move(component)}, spec.mask.window,
                                      spec.mask.sink_tokens);
    case MaskKind::kChunked:
      if (spec.mask.window <= 0) {
        return InvalidArgumentError("kChunked requires a positive chunk size, got ",
                                    spec.mask.window);
      }
      return CacheSpec::ChunkedLocal({std::move(component)}, spec.mask.window);
  }
  return InvalidArgumentError("unreachable mask kind");
}

StatusOr<CacheSpec> BuildDense(const DenseParams& p, const AttentionSpec& spec) {
  if (p.num_q_heads <= 0 || p.num_kv_heads <= 0) {
    return InvalidArgumentError("dense attention needs positive head counts, got q=",
                                p.num_q_heads, " kv=", p.num_kv_heads);
  }
  if (p.num_q_heads % p.num_kv_heads != 0) {
    return InvalidArgumentError("GQA requires num_q_heads % num_kv_heads == 0, got q=",
                                p.num_q_heads, " kv=", p.num_kv_heads);
  }
  if (p.v_head_dim != 0 && p.v_head_dim != p.head_dim) {
    // The GQA component sizes K and V symmetrically; asymmetric V heads
    // need a custom component the bridge cannot infer.
    return InvalidArgumentError(
        "asymmetric v_head_dim (", p.v_head_dim, " vs head_dim ", p.head_dim,
        ") is not representable by GqaKvComponent; declare the cache spec by hand");
  }
  INFERX_ASSIGN_OR_RETURN(const int64_t bytes,
                          DenseBytesPerToken(p.num_kv_heads, p.head_dim, spec.kv_dtype));
  return TokenWiseSpec(spec, CacheComponent::PerToken(bytes, spec.kv_dtype));
}

StatusOr<CacheSpec> BuildMla(const MlaParams& p, const AttentionSpec& spec) {
  if (p.kv_lora_rank <= 0 || p.qk_rope_dim <= 0 || p.num_heads <= 0) {
    return InvalidArgumentError("MLA needs positive ranks and head count");
  }
  // One latent + rope part per token; the latent is not sharded across
  // tensor-parallel ranks (see MlaLatentComponent).
  return TokenWiseSpec(spec,
                       MlaLatentComponent(p.kv_lora_rank + p.qk_rope_dim, spec.kv_dtype));
}

StatusOr<CacheSpec> BuildLinear(const LinearParams& p, const AttentionSpec& spec) {
  if (p.num_k_heads <= 0 || p.num_v_heads <= 0 || p.k_dim <= 0 || p.v_dim <= 0) {
    return InvalidArgumentError("linear attention needs positive head counts and dims");
  }
  if (p.boundary_tokens <= 0) {
    return InvalidArgumentError(
        "linear mechanisms must declare state boundary_tokens > 0 (a multiple of the "
        "block size), got ",
        p.boundary_tokens);
  }
  if (p.conv_width < 0) {
    return InvalidArgumentError("conv_width must be >= 0, got ", p.conv_width);
  }

  // The delta-rule state is [v_heads, k_dim, v_dim]; the conv ring holds
  // the last (conv_width - 1) mixed inputs per head, [v_heads, v_dim] wide.
  // Families whose real geometry differs declare their cache spec by hand.
  const int64_t state_bytes =
      DataTypeByteSize(spec.kv_dtype, int64_t{p.num_v_heads} * p.k_dim * p.v_dim);
  std::vector<CacheComponent> components;
  components.push_back(CacheComponent::FixedSlot(state_bytes, spec.kv_dtype));
  if (p.conv_width > 1) {
    components.push_back(CacheComponent::FixedSlot(
        DataTypeByteSize(spec.kv_dtype,
                         int64_t{p.num_v_heads} * p.v_dim * (p.conv_width - 1)),
        spec.kv_dtype));
  } else if (p.conv_width == 1) {
    return InvalidArgumentError("conv_width == 1 means no convolution state; use 0");
  }
  return CacheSpec::Recurrent(std::move(components), p.boundary_tokens);
}

StatusOr<CacheSpec> BuildSparse(const SparseParams& p, const AttentionSpec& spec) {
  // The cached state is the dense base's K/V; the sparsity rule reads it,
  // it does not change it. IndexedTopK's per-token indices are an input
  // tensor, not cache storage.
  const DenseParams& base = p.base;
  if (base.num_q_heads <= 0 || base.num_kv_heads <= 0 || base.head_dim <= 0) {
    return InvalidArgumentError("sparse attention needs a valid dense base");
  }
  INFERX_ASSIGN_OR_RETURN(const int64_t bytes,
                          DenseBytesPerToken(base.num_kv_heads, base.head_dim, spec.kv_dtype));
  return TokenWiseSpec(spec, CacheComponent::PerToken(bytes, spec.kv_dtype));
}

}  // namespace

bool NeedsCache(const AttentionSpec& spec) noexcept {
  // Encoder-self attention is computed and consumed within one forward:
  // it never persists K/V.
  return spec.role != Role::kEncoderSelf;
}

StatusOr<CacheSpec> BuildCacheSpec(const AttentionSpec& spec) {
  if (!NeedsCache(spec)) {
    return InvalidArgumentError(
        "encoder-self attention is stateless: emit NoCache in the CacheRequirements "
        "instead of calling BuildCacheSpec");
  }
  return std::visit(
      [&](const auto& p) -> StatusOr<CacheSpec> {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, DenseParams>) {
          return BuildDense(p, spec);
        } else if constexpr (std::is_same_v<T, MlaParams>) {
          return BuildMla(p, spec);
        } else if constexpr (std::is_same_v<T, LinearParams>) {
          return BuildLinear(p, spec);
        } else if constexpr (std::is_same_v<T, SparseParams>) {
          return BuildSparse(p, spec);
        } else {
          // Custom mechanisms own their storage story; the bridge does not
          // guess bytes for mechanisms it cannot interpret.
          return InvalidArgumentError(
              "custom mechanism ", p.mechanism_id,
              " must build its CacheSpec in its backend; the bridge does not guess");
        }
      },
      spec.params);
}

StatusOr<CacheRequirements> BuildCacheRequirements(std::span<const AttentionSpec> specs) {
  CacheRequirements requirements;
  requirements.layers.reserve(specs.size());
  for (const AttentionSpec& spec : specs) {
    if (!NeedsCache(spec)) {
      requirements.layers.push_back(NoCache{});
      continue;
    }
    INFERX_ASSIGN_OR_RETURN(CacheSpec cache_spec, BuildCacheSpec(spec));
    requirements.layers.push_back(std::move(cache_spec));
  }
  return requirements;
}

StatusOr<int64_t> CacheBytesPerToken(const AttentionSpec& spec) {
  if (!NeedsCache(spec)) {
    return InvalidArgumentError("encoder-self attention caches nothing");
  }
  return std::visit(
      [&](const auto& p) -> StatusOr<int64_t> {
        using T = std::decay_t<decltype(p)>;
        if constexpr (std::is_same_v<T, DenseParams>) {
          return DenseBytesPerToken(p.num_kv_heads, p.head_dim, spec.kv_dtype);
        } else if constexpr (std::is_same_v<T, MlaParams>) {
          if (p.kv_lora_rank <= 0 || p.qk_rope_dim <= 0) {
            return InvalidArgumentError("MLA needs positive ranks");
          }
          return DataTypeByteSize(spec.kv_dtype, int64_t{p.kv_lora_rank} + p.qk_rope_dim);
        } else if constexpr (std::is_same_v<T, SparseParams>) {
          return DenseBytesPerToken(p.base.num_kv_heads, p.base.head_dim, spec.kv_dtype);
        } else {
          return InvalidArgumentError(
              "per-token bytes are undefined for per-sequence or custom state");
        }
      },
      spec.params);
}

}  // namespace inferx::attn
