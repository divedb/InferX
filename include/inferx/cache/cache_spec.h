#ifndef INFERX_CACHE_CACHE_SPEC_H_
#define INFERX_CACHE_CACHE_SPEC_H_

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"

namespace inferx {

/// \brief Dependent always-false trait for std::visit exhaustiveness checks.
template <typename T>
struct always_false : std::false_type {};

/// \brief Describes the storage requirements of a single cache component
///        within a model layer.
///
/// A cache component represents one independently defined tensor or state
/// buffer, such as a key/value tensor, an MLA latent cache, a sparse-attention
/// indexer, or a recurrent state.
///
/// Storage requirements are expressed in bytes rather than inferred from
/// attention dimensions. This allows the cache manager to support arbitrary
/// layouts, quantization metadata, padding, and model-specific state without
/// understanding their internal representations.
///
/// Components are either token-dependent (allocated in pages) or fixed-size
/// (allocated per sequence). Model-specific code constructs these descriptors;
/// the cache manager only handles their storage requirements and lifetimes.
struct CacheComponent {
  /// \brief Determines how storage scales with sequence length.
  enum class Growth : std::uint8_t {
    kPerToken = 0,   ///< Storage grows with tokens, allocated in pages.
    kFixedSlot = 1,  ///< Fixed-size storage allocated once per sequence.
  };

  Growth growth = Growth::kPerToken;

  /// \brief Storage required per token, in bytes.
  ///
  /// Only applicable when growth == kPerToken. Includes any layout-specific
  /// padding, quantization scales, and auxiliary metadata stored per token.
  /// Must be zero for kFixedSlot components.
  int64_t bytes_per_token = 0;

  /// \brief Storage required per sequence, in bytes.
  ///
  /// Only applicable when growth == kFixedSlot. Includes all state and
  /// metadata belonging to this component's sequence slot.
  /// Must be zero for kPerToken components.
  int64_t bytes_per_slot = 0;

  /// \brief Logical element type of the component.
  ///
  /// Informational only. The cache manager treats storage as opaque bytes;
  /// model-specific kernels determine how the data is interpreted.
  DataType dtype = DataType::kBFloat16;

  bool operator==(const CacheComponent&) const = default;

  /// \brief Creates a token-dependent cache component.
  ///
  /// \param bytes_per_token Storage required per token, in bytes.
  /// \param dtype           Logical element type.
  /// \return                The token-dependent component.
  constexpr static CacheComponent PerToken(int64_t bytes_per_token, DataType dtype) {
    return CacheComponent{Growth::kPerToken, bytes_per_token, 0, dtype};
  }

  /// \brief Creates a fixed-size cache component.
  ///
  /// \param bytes_per_slot Storage required per sequence, in bytes.
  /// \param dtype          Logical element type.
  /// \return               The fixed-size component.
  constexpr static CacheComponent FixedSlot(int64_t bytes_per_slot, DataType dtype) {
    return CacheComponent{Growth::kFixedSlot, 0, bytes_per_slot, dtype};
  }
};

/// \brief Creates a combined K/V cache component for a GQA, MHA, or MQA layer.
///
/// Each token stores both key and value vectors, requiring
/// `2 * kv_heads_per_rank * head_dim` elements.
///
/// \param kv_heads_per_rank Number of KV heads local to this tensor-parallel rank, after
///                          tensor-parallel partitioning.
/// \param head_dim          Dimension of each key/value head.
/// \param dtype             Element type of the cached keys and values.
/// \return                  A per-token cache component sized for both K and V.
constexpr CacheComponent GqaKvComponent(int64_t kv_heads_per_rank, int64_t head_dim,
                                        DataType dtype) {
  return CacheComponent::PerToken(DataTypeByteSize(dtype, 2 * kv_heads_per_rank * head_dim),
                                  dtype);
}

/// \brief Creates a cache component for an MLA compressed latent.
///
/// Each token stores one compressed latent vector of `latent_dim` elements.
/// The latent cache is not partitioned across tensor-parallel ranks, so
/// `latent_dim` must be the full, unsharded dimension.
///
/// \param latent_dim Dimension of the compressed latent vector.
/// \param dtype      Element type of the cached latent.
/// \return           A per-token cache component sized for the compressed latent.
constexpr CacheComponent MlaLatentComponent(int64_t latent_dim, DataType dtype) {
  return CacheComponent::PerToken(DataTypeByteSize(dtype, latent_dim), dtype);
}

/// \brief Retains KV cache entries for the entire sequence lifetime.
///
/// All computed tokens remain resident until the request releases its cache.
/// Used by full-attention layers, including MHA, MQA, GQA, and MLA.
struct FullRetention {
  bool operator==(const FullRetention&) const = default;
};

/// \brief Retains a sliding window of recent tokens and optional attention sinks.
///
/// The trailing `window` tokens remain eligible for attention, while the first
/// `sink_tokens` tokens are retained regardless of their distance from the
/// current position. Tokens outside these regions may be evicted once they
/// are no longer needed by in-flight computations.
///
/// Used by sliding-window attention, including sink-based variants such as
/// those in gpt-oss.
struct WindowRetention {
  /// Number of tokens in the trailing window. Must be positive.
  int64_t window = 0;

  /// Number of permanently retained prefix tokens.
  /// Zero disables sinks; otherwise must be a multiple of block_size.
  int64_t sink_tokens = 0;

  bool operator==(const WindowRetention&) const = default;
};

/// \brief Retains KV entries only within the current attention chunk.
///
/// Tokens are partitioned into non-overlapping chunks of `chunk_tokens`.
/// Attention is restricted to tokens within the same chunk, allowing the
/// previous chunk's cache to be released at a chunk boundary.
///
/// Unlike sliding-window retention, the retained region advances in discrete
/// chunks rather than moving forward with every token.
///
/// Used by chunked local attention, such as in Llama 4.
struct ChunkRetention {
  /// Number of tokens per attention chunk. Must be positive.
  int64_t chunk_tokens = 0;

  bool operator==(const ChunkRetention&) const = default;
};

/// \brief Retains a fixed encoder-side cache throughout decoder execution.
///
/// Cache capacity is determined by the encoder sequence length rather than
/// the number of generated decoder tokens. Once populated, the cache does
/// not grow during decoding.
///
/// Used by encoder-decoder cross-attention, such as in Whisper and T5.
struct EncoderRetention {
  bool operator==(const EncoderRetention&) const = default;
};

/// \brief Retains fixed-size recurrent state instead of token-indexed KV entries.
///
/// Each sequence maintains state that is updated as tokens are processed.
/// State validity is tracked at boundaries separated by `boundary_tokens`.
///
/// Used by recurrent architectures such as Mamba, gated delta networks,
/// and short convolutional mixers.
///
/// Cross-request sharing of boundary states is not currently supported.
struct RecurrentRetention {
  /// Number of tokens between state-validity boundaries.
  /// Must be positive and a multiple of block_size.
  int64_t boundary_tokens = 0;

  bool operator==(const RecurrentRetention&) const = default;
};

/// \brief Defines the retention and eviction policy of a cache component.
///
/// The selected policy determines which cached data must remain resident as
/// a request progresses. It is independent of the component's physical
/// storage layout and element type.
using Retention = std::variant<FullRetention, WindowRetention, ChunkRetention, EncoderRetention,
                               RecurrentRetention>;

/// \brief Describes the storage requirements and retention policy of one
///        model layer's cache.
///
/// A CacheSpec is a declarative description of cache storage. It specifies
/// which components are stored and how long their contents must be retained,
/// without implementing allocation, eviction, or state updates.
///
/// The cache manager selected by `retention` is responsible for enforcing
/// the corresponding lifecycle and memory-management policy.
///
/// Token-based caches use per-token components, while recurrent caches use
/// fixed-size per-sequence state components.
struct CacheSpec {
  /// \name Construction helpers
  /// Preferred over aggregate initialization to ensure consistent specs.
  ///\{

  /// \brief Creates a cache that retains all computed tokens.
  ///
  /// \param components Per-token cache components.
  /// \return           The full-attention cache spec.
  static CacheSpec FullAttention(std::vector<CacheComponent> components);

  /// \brief Creates a sliding-window cache with optional attention sinks.
  ///
  /// \param components  Per-token cache components.
  /// \param window      Number of tokens in the trailing attention window.
  /// \param sink_tokens Number of permanently retained prefix tokens.
  /// \return            The sliding-window cache spec.
  static CacheSpec SlidingWindow(std::vector<CacheComponent> components, int64_t window,
                                 int64_t sink_tokens = 0);

  /// \brief Creates a cache retained within non-overlapping attention chunks.
  ///
  /// \param components   Per-token cache components.
  /// \param chunk_tokens Number of tokens per attention chunk.
  /// \return             The chunked-local cache spec.
  static CacheSpec ChunkedLocal(std::vector<CacheComponent> components, int64_t chunk_tokens);

  /// \brief Creates an encoder-side cache for cross-attention.
  ///
  /// Cache capacity is determined by the encoder sequence length and remains
  /// unchanged throughout decoder execution.
  ///
  /// \param components Per-token cache components.
  /// \return           The encoder-retained cache spec.
  static CacheSpec Encoder(std::vector<CacheComponent> components);

  /// \brief Creates a cache for cross-attention.
  ///
  /// \param components Per-token cache components.
  /// \return           The cross-attention cache spec.
  static CacheSpec CrossAttention(std::vector<CacheComponent> components);

  /// \brief Creates a cache of fixed-size recurrent state components.
  ///
  /// \param slots           Fixed-size per-sequence state components.
  /// \param boundary_tokens Number of tokens between state-validity boundaries.
  ///                        Must be a multiple of block_size.
  /// \return                The recurrent cache spec.
  static CacheSpec Recurrent(std::vector<CacheComponent> slots, int64_t boundary_tokens);

  ///\}

  /// \brief Storage components belonging to one model layer.
  ///
  /// Examples include key/value tensors, compressed latents, sparse-attention
  /// indexers, convolution states, and SSM states.
  ///
  /// All components must use kPerToken growth for token-based retention
  /// policies, or kFixedSlot growth for RecurrentRetention.
  /// The collection must not be empty.
  std::vector<CacheComponent> components;

  /// \brief Policy governing cache retention and eviction.
  ///
  /// Determines which cached data must remain resident as a sequence
  /// progresses. The corresponding cache manager implements the policy.
  Retention retention = FullRetention{};

  bool operator==(const CacheSpec&) const = default;

  /// \brief Returns whether the cache uses token-indexed storage.
  ///
  /// \return True for full, sliding-window, chunked-local, and encoder retention. False for
  ///         recurrent retention, which uses fixed-size sequence slots.
  bool IsTokenWise() const { return !std::holds_alternative<RecurrentRetention>(retention); }

  /// \brief Returns whether completed blocks are eligible for prefix caching.
  ///
  /// True for full, sliding-window, and chunked-local retention.
  /// False for encoder and recurrent retention.
  ///
  /// Eligibility does not guarantee that a block can be shared; actual
  /// reuse depends on cache validity and the selected cache manager.
  ///
  /// \return True if completed blocks are eligible for prefix caching; otherwise false.
  bool PrefixCacheable() const {
    return std::holds_alternative<FullRetention>(retention) ||
           std::holds_alternative<WindowRetention>(retention) ||
           std::holds_alternative<ChunkRetention>(retention);
  }

  /// \brief Returns the total storage required per token for this layer.
  ///
  /// Computed as the sum of bytes_per_token across all components.
  /// Returns zero for recurrent caches.
  ///
  /// Does not include block-level allocation overhead or alignment.
  ///
  /// \return Total storage required per token, in bytes.
  int64_t BytesPerTokenPerLayer() const {
    return std::accumulate(components.begin(), components.end(), int64_t{0},
                           [](int64_t sum, const CacheComponent& component) {
                             return sum + (component.growth == CacheComponent::Growth::kPerToken
                                               ? component.bytes_per_token
                                               : 0);
                           });
  }

  /// \brief Returns the total fixed-state storage required per sequence.
  ///
  /// Computed as the sum of bytes_per_slot across all components.
  /// Returns zero for token-based caches.
  ///
  /// \return Total fixed-state storage required per sequence, in bytes.
  int64_t StateBytesPerLayer() const {
    return std::accumulate(components.begin(), components.end(), int64_t{0},
                           [](int64_t sum, const CacheComponent& component) {
                             return sum +
                                    (component.growth == CacheComponent::Growth::kFixedSlot
                                         ? component.bytes_per_slot
                                         : 0);
                           });
  }

  /// \brief Returns an upper bound on resident cache blocks per sequence.
  ///
  /// \param num_tokens Number of tokens processed by the sequence, or the encoder sequence
  ///                   length for EncoderRetention.
  /// \param block_size Number of tokens represented by one cache block.
  ///
  /// Accounts for the retention policy and includes up to one additional
  /// block for window or chunk boundary misalignment.
  /// Returns zero for recurrent caches.
  ///
  /// \return An upper bound on resident cache blocks per sequence.
  ///         The result is a capacity bound, not the exact number of blocks
  ///         currently allocated.
  int64_t MaxBlocks(int64_t num_tokens, int64_t block_size) const {
    return std::visit(
        [num_tokens, block_size](const auto& retention) -> int64_t {
          using T = std::decay_t<decltype(retention)>;

          if constexpr (std::is_same_v<T, FullRetention>) {
            return (num_tokens + block_size - 1) / block_size;
          } else if constexpr (std::is_same_v<T, WindowRetention>) {
            return (retention.window + retention.sink_tokens + block_size - 1) / block_size;
          } else if constexpr (std::is_same_v<T, ChunkRetention>) {
            return (retention.chunk_tokens + block_size - 1) / block_size;
          } else if constexpr (std::is_same_v<T, EncoderRetention>) {
            return (num_tokens + block_size - 1) / block_size;
          } else if constexpr (std::is_same_v<T, RecurrentRetention>) {
            return 0;
          } else {
            static_assert(always_false<T>::value, "non-exhaustive visitor!");
          }
        },
        retention);
  }

  /// \brief Validates the spec against the configured cache block size.
  ///
  /// Checks that:
  /// - At least one component is present.
  /// - Component growth modes are compatible with the retention policy.
  /// - Window and chunk lengths are positive.
  /// - Sink-token and recurrent-state boundaries are block-aligned.
  /// - The block size is valid.
  ///
  /// \return OK if the spec is valid; otherwise an error describing the
  ///         violated constraint.
  Status Validate(int64_t block_size) const {
    if (components.empty()) {
      return InvalidArgumentError("CacheSpec must have at least one component");
    }

    for (const auto& component : components) {
      if (std::holds_alternative<RecurrentRetention>(retention) &&
          component.growth != CacheComponent::Growth::kFixedSlot) {
        return InvalidArgumentError(
            "RecurrentRetention requires all components to use kFixedSlot growth");
      }

      if (!std::holds_alternative<RecurrentRetention>(retention) &&
          component.growth != CacheComponent::Growth::kPerToken) {
        return InvalidArgumentError(
            "Token-based retention requires all components to use kPerToken growth");
      }
    }

    return std::visit(
        [block_size](const auto& retention) -> Status {
          using T = std::decay_t<decltype(retention)>;

          if constexpr (std::is_same_v<T, FullRetention>) {
            return OkStatus();
          } else if constexpr (std::is_same_v<T, WindowRetention>) {
            if (retention.window <= 0) {
              return InvalidArgumentError("WindowRetention requires window > 0");
            }

            if (retention.sink_tokens < 0 || retention.sink_tokens % block_size != 0) {
              return InvalidArgumentError(
                  "WindowRetention requires sink_tokens >= 0 and a multiple of block_size");
            }

            return OkStatus();
          } else if constexpr (std::is_same_v<T, ChunkRetention>) {
            if (retention.chunk_tokens <= 0 || retention.chunk_tokens % block_size != 0) {
              return InvalidArgumentError(
                  "ChunkRetention requires chunk_tokens > 0 and a multiple of block_size");
            }

            return OkStatus();
          } else if constexpr (std::is_same_v<T, EncoderRetention>) {
            return OkStatus();
          } else if constexpr (std::is_same_v<T, RecurrentRetention>) {
            if (retention.boundary_tokens <= 0 || retention.boundary_tokens % block_size != 0) {
              return InvalidArgumentError(
                  "RecurrentRetention requires boundary_tokens > 0 and a multiple of "
                  "block_size");
            }

            return OkStatus();
          } else {
            static_assert(always_false<T>::value, "non-exhaustive visitor!");
          }
        },
        retention);
  }

  /// \brief Returns a stable, human-readable description of the spec.
  ///
  /// Includes the retention policy and component storage requirements.
  /// Intended for diagnostics, logging, and error messages.
  ///
  /// \return A string describing the cache spec.
  std::string ToString() const {
    std::string result;
    result += "CacheSpec(retention=";

    result += std::visit(
        [](const auto& retention) -> std::string {
          using T = std::decay_t<decltype(retention)>;

          if constexpr (std::is_same_v<T, FullRetention>) {
            return "FullRetention";
          } else if constexpr (std::is_same_v<T, WindowRetention>) {
            return "WindowRetention(window=" + std::to_string(retention.window) +
                   ", sink_tokens=" + std::to_string(retention.sink_tokens) + ")";
          } else if constexpr (std::is_same_v<T, ChunkRetention>) {
            return "ChunkRetention(chunk_tokens=" + std::to_string(retention.chunk_tokens) +
                   ")";
          } else if constexpr (std::is_same_v<T, EncoderRetention>) {
            return "EncoderRetention";
          } else if constexpr (std::is_same_v<T, RecurrentRetention>) {
            return "RecurrentRetention(boundary_tokens=" +
                   std::to_string(retention.boundary_tokens) + ")";
          } else {
            static_assert(always_false<T>::value, "non-exhaustive visitor!");
          }
        },
        retention);

    result += ", components=[";

    for (size_t i = 0; i < components.size(); ++i) {
      const auto& component = components[i];
      result +=
          "{growth=" +
          std::string(component.growth == CacheComponent::Growth::kPerToken ? "PerToken"
                                                                            : "FixedSlot") +
          ", bytes_per_token=" + std::to_string(component.bytes_per_token) +
          ", bytes_per_slot=" + std::to_string(component.bytes_per_slot) +
          ", dtype=" + std::to_string(static_cast<int>(component.dtype)) + "}";
      if (i < components.size() - 1) {
        result += ", ";
      }
    }

    result += "])";

    return result;
  }
};

/// \brief Indicates that a layer requires no persistent cache storage.
///
/// The layer neither owns nor references cache managed by the runtime.
/// This is distinct from an empty CacheSpec, which is invalid.
struct NoCache {
  bool operator==(const NoCache&) const = default;
};

/// \brief Indicates that a layer consumes cache owned by another layer (Gemma 3n, YOCO).
///
/// The consumer does not allocate or own separate cache storage. Instead,
/// it accesses the cache produced by `source_layer`, as explicitly required
/// by the model architecture.
///
/// This represents cross-layer cache sharing, not prefix-cache sharing or
/// an optimization inferred from compatible cache layouts.
///
/// The source must precede the consumer in model order and must own a
/// cache described by CacheSpec.
struct SharedCache {
  /// Index of the cache-owning source layer.
  int64_t source_layer = -1;

  bool operator==(const SharedCache&) const = default;
};

/// \brief Declares the persistent cache requirements of a model.
struct CacheRequirements {
  using LayerCache = std::variant<NoCache, CacheSpec, SharedCache>;

  /// Cache requirements for each decoder layer, in execution order.
  std::vector<LayerCache> layers;
};

/// \brief Describes a group of cache-owning layers managed as one unit.
///
/// All layers in the group have identical CacheSpec values and therefore
/// share the same storage requirements and retention policy. A single cache
/// manager can coordinate their allocation and eviction.
///
/// Each layer owns distinct cache contents. Grouping layers does not imply
/// that they share cached keys, values, or recurrent states.
struct CacheGroupSpec {
  /// Zero-based index of this group in CachePlan::groups.
  ///
  /// Token-wise groups come first, ordered by their first layer index.
  /// The recurrent group, if present, is last.
  int64_t group_index = 0;

  /// Model-layer indices that own cache storage in this group.
  ///
  /// Indices are in model order but need not be consecutive, since hybrid
  /// models may interleave layers with different cache requirements.
  ///
  /// Storage for these layers is laid out contiguously in the group's pool,
  /// even when their model-layer indices are noncontiguous.
  ///
  /// Layers that consume another layer's cache are excluded; their
  /// relationships are recorded in CachePlan::aliases.
  std::vector<int64_t> layers;

  /// Cache specification shared by every layer in `layers`.
  CacheSpec spec;

  /// \brief Returns the number of cache-owning layers in this group.
  int64_t NumLayers() const { return static_cast<int64_t>(layers.size()); }

  /// \brief Returns the storage required for one group-wide cache block.
  ///
  /// Includes the per-token storage of every layer in the group:
  ///
  ///   NumLayers() * spec.BytesPerTokenPerLayer() * block_size
  ///
  /// Returns zero for recurrent groups, which use fixed-size state slots
  /// rather than token-indexed blocks.
  ///
  /// \param block_size Number of tokens per cache block.
  int64_t BytesPerBlock(int64_t block_size) const;

  /// \brief Returns the fixed-state storage required for one sequence.
  ///
  /// Includes the recurrent state of every layer in the group:
  ///
  ///   NumLayers() * spec.StateBytesPerLayer()
  ///
  /// \return The fixed-state storage required for one sequence, in bytes.
  int64_t StateBytesPerSequence() const;

  /// \brief Estimates the maximum resident cache storage for one sequence.
  ///
  /// For token-wise groups, multiplies the retention policy's block-count
  /// bound by BytesPerBlock(block_size). For recurrent groups, returns
  /// StateBytesPerSequence().
  ///
  /// Used during startup to estimate whether a sequence of the configured
  /// maximum length can fit within the cache budget, independently of the
  /// runtime cache managers.
  ///
  /// This bound describes retained cache storage, not necessarily peak
  /// execution memory. In-flight allocations, temporary buffers, and
  /// allocator overhead may require additional capacity.
  ///
  /// \param num_tokens Sequence length used for capacity planning.
  ///        For encoder caches, this is the encoder sequence length.
  /// \param block_size Number of tokens per cache block.
  int64_t MaxBytesPerSequence(int64_t num_tokens, int64_t block_size) const;
};

/// \brief Maps a cache-consuming layer to the layer that owns its storage.
///
/// Represents an explicit cross-layer cache-sharing relationship defined
/// by the model architecture. The consuming layer does not allocate its
/// own cache; it reads the cache produced by `source_layer`.
///
/// Unlike prefix-cache sharing, this relationship exists between layers
/// of the same model rather than between requests.
struct LayerAlias {
  /// Model-layer index of the cache consumer.
  int64_t layer = 0;

  /// Model-layer index of the cache owner.
  int64_t source_layer = 0;

  bool operator==(const LayerAlias&) const = default;
};

/// \brief Describes the cache allocation plan for a model.
///
/// Produced by PlanCacheGroups from the model's CacheRequirements.
/// Groups cache-owning layers by identical specifications, records
/// cross-layer cache aliases, and establishes the block size used by
/// token-wise cache managers.
///
/// PlanCacheGroups guarantees that token-wise groups form a contiguous
/// prefix of `groups`, ordered by their first layer index. The recurrent
/// group, if present, occupies the final position.
///
/// This ordering is required by TokenWiseGroups() and RecurrentGroup().
///
/// A CachePlan describes storage organization and ownership relationships;
/// it does not contain allocated cache tensors or per-request cache state.
struct CachePlan {
  /// Cache groups in planner-defined order.
  std::vector<CacheGroupSpec> groups;

  /// Resolved mappings from cache-consuming layers to cache-owning layers.
  ///
  /// Alias consumers are not included in CacheGroupSpec::layers and do
  /// not require additional cache storage.
  std::vector<LayerAlias> aliases;

  /// Number of tokens per block for token-wise cache groups.
  int64_t block_size = 0;

  /// \brief Returns the total number of cache groups.
  int64_t NumGroups() const { return static_cast<int64_t>(groups.size()); }

  /// \brief Returns the number of token-wise cache groups.
  ///
  /// Counts groups whose CacheSpec::IsTokenWise() returns true.
  ///
  /// \return The number of token-wise cache groups.
  int64_t NumTokenWiseGroups() const noexcept {
    return std::count_if(groups.begin(), groups.end(),
                         [](const CacheGroupSpec& group) { return group.spec.IsTokenWise(); });
  }

  /// \brief Returns the token-wise groups used to initialize KV block pools.
  ///
  /// The returned span references the contiguous token-wise prefix of
  /// `groups`. It does not own the underlying storage and must not outlive
  /// the CachePlan or a modification that invalidates its vector elements.
  absl::Span<const CacheGroupSpec> TokenWiseGroups() const;

  /// \brief Returns whether the plan has zero or one token-wise group.
  ///
  /// Used to select the single-group allocation fast path.
  bool HasAtMostOneTokenWiseGroup() const { return NumTokenWiseGroups() <= 1; }

  /// \brief Returns the recurrent cache group, if present.
  ///
  /// The recurrent group is always the final entry in `groups`.
  ///
  /// \return Pointer to the recurrent group, or nullptr if none exists.
  ///         The pointer is invalidated if `groups` is reallocated.
  const CacheGroupSpec* RecurrentGroup() const;

  /// \brief Finds the cache group that provides storage for a model layer.
  ///
  /// For a cache-owning layer, returns its containing group.
  /// For an alias layer, resolves the source layer and returns the
  /// source's group.
  ///
  /// \param layer Model-layer index to look up.
  /// \return      Pointer to the corresponding group, or nullptr if the layer requires no
  ///              persistent cache.
  ///
  /// The returned group identifies the allocation and retention policy,
  /// not the specific layer's tensor within the group's storage.
  const CacheGroupSpec* GroupOfLayer(int64_t layer) const;
};

/// \brief Plans cache groups for a model.
///
/// \param requirements  Model cache requirements.
/// \param block_size    Number of tokens per cache block.
/// \param max_model_len Optional maximum model sequence length for promotion of
///                      sliding-window or chunked retention to full-attention.
/// \return              A CachePlan describing the grouped cache layout and
///                      cross-layer aliases, or an error if the requirements are
///                      invalid or cannot be satisfied.
StatusOr<CachePlan> PlanCacheGroups(const CacheRequirements& requirements, int64_t block_size,
                                    std::optional<int64_t> max_model_len = std::nullopt);

}  // namespace inferx

#endif  // INFERX_CACHE_CACHE_SPEC_H_