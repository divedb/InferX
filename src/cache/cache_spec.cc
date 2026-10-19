#include "inferx/cache/cache_spec.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "inferx/core/status.h"

namespace inferx {

CacheSpec CacheSpec::FullAttention(std::vector<CacheComponent> components) {
  CacheSpec spec;
  spec.components = std::move(components);
  spec.retention = FullRetention{};

  return spec;
}

CacheSpec CacheSpec::SlidingWindow(std::vector<CacheComponent> components, int64_t window,
                                   int64_t sink_tokens) {
  CacheSpec spec;
  spec.components = std::move(components);
  spec.retention = WindowRetention{window, sink_tokens};

  return spec;
}

CacheSpec CacheSpec::ChunkedLocal(std::vector<CacheComponent> components,
                                  int64_t chunk_tokens) {
  CacheSpec spec;
  spec.components = std::move(components);
  spec.retention = ChunkRetention{chunk_tokens};

  return spec;
}

CacheSpec CacheSpec::Encoder(std::vector<CacheComponent> components) {
  CacheSpec spec;
  spec.components = std::move(components);
  spec.retention = EncoderRetention{};

  return spec;
}

CacheSpec CacheSpec::CrossAttention(std::vector<CacheComponent> components) {
  // Cross-attention K/V are computed once from the encoder output and stay
  // fixed during decoding, which is exactly EncoderRetention.
  return Encoder(std::move(components));
}

CacheSpec CacheSpec::Recurrent(std::vector<CacheComponent> slots, int64_t boundary_tokens) {
  CacheSpec spec;
  spec.components = std::move(slots);
  spec.retention = RecurrentRetention{boundary_tokens};

  return spec;
}

int64_t CacheGroupSpec::BytesPerBlock(int64_t block_size) const {
  if (!spec.IsTokenWise()) {
    return 0;
  }

  return NumLayers() * spec.BytesPerTokenPerLayer() * block_size;
}

int64_t CacheGroupSpec::StateBytesPerSequence() const {
  if (spec.IsTokenWise()) {
    return 0;
  }

  return NumLayers() * spec.StateBytesPerLayer();
}

int64_t CacheGroupSpec::MaxBytesPerSequence(int64_t num_tokens, int64_t block_size) const {
  if (!spec.IsTokenWise()) {
    return StateBytesPerSequence();
  }

  return spec.MaxBlocks(num_tokens, block_size) * BytesPerBlock(block_size);
}

absl::Span<const CacheGroupSpec> CachePlan::TokenWiseGroups() const {
  // Token-wise groups form a contiguous prefix of `groups`.
  return absl::MakeConstSpan(groups).subspan(0, static_cast<size_t>(NumTokenWiseGroups()));
}

const CacheGroupSpec* CachePlan::RecurrentGroup() const {
  if (!groups.empty() && !groups.back().spec.IsTokenWise()) {
    return &groups.back();
  }

  return nullptr;
}

const CacheGroupSpec* CachePlan::GroupOfLayer(int64_t layer) const {
  // Resolve aliases first. Aliases are already resolved to owners by the
  // planner, but follow chains defensively (bounded to avoid cycles).
  int64_t owner = layer;

  for (size_t hops = 0; hops <= aliases.size(); ++hops) {
    auto it = std::find_if(aliases.begin(), aliases.end(),
                           [owner](const LayerAlias& a) { return a.layer == owner; });
    if (it == aliases.end()) {
      break;
    }

    owner = it->source_layer;
  }

  for (const CacheGroupSpec& group : groups) {
    // `layers` is in ascending model order.
    if (std::binary_search(group.layers.begin(), group.layers.end(), owner)) {
      return &group;
    }
  }

  return nullptr;
}

namespace {

// Storage of one layer's page, used for the phase-1 uniform page size check.
// Token-wise: bytes for one block of one layer. Recurrent: one state slot.
int64_t PerLayerPageBytes(const CacheSpec& spec, int64_t block_size) {
  return spec.IsTokenWise() ? spec.BytesPerTokenPerLayer() * block_size
                            : spec.StateBytesPerLayer();
}

// Window/chunk retention that can never clip a sequence is full attention.
CacheSpec MaybePromoteToFull(CacheSpec spec, std::optional<int64_t> max_model_len) {
  if (!max_model_len.has_value()) {
    return spec;
  }
  if (const auto* w = std::get_if<WindowRetention>(&spec.retention)) {
    if (w->window >= *max_model_len) {
      spec.retention = FullRetention{};
    }
  } else if (const auto* c = std::get_if<ChunkRetention>(&spec.retention)) {
    if (c->chunk_tokens >= *max_model_len) {
      spec.retention = FullRetention{};
    }
  }
  return spec;
}

}  // namespace

StatusOr<CachePlan> PlanCacheGroups(const CacheRequirements& requirements, int64_t block_size,
                                    std::optional<int64_t> max_model_len) {
  if (block_size <= 0) {
    return InvalidArgumentError("block_size must be positive");
  }

  if (max_model_len.has_value() && *max_model_len <= 0) {
    return InvalidArgumentError("max_model_len must be positive when provided");
  }

  CachePlan plan;
  plan.block_size = block_size;

  std::vector<CacheGroupSpec> token_groups;
  CacheGroupSpec recurrent_group;
  bool has_recurrent = false;

  const int64_t num_layers = static_cast<int64_t>(requirements.layers.size());
  for (int64_t i = 0; i < num_layers; ++i) {
    const auto& layer_cache = requirements.layers[static_cast<size_t>(i)];

    if (std::holds_alternative<NoCache>(layer_cache)) {
      continue;
    }

    if (const auto* shared = std::get_if<SharedCache>(&layer_cache)) {
      const int64_t src = shared->source_layer;
      if (src < 0 || src >= i) {
        return InvalidArgumentError("Layer " + std::to_string(i) +
                                    ": SharedCache source_layer " + std::to_string(src) +
                                    " must be an earlier layer");
      }
      if (!std::holds_alternative<CacheSpec>(requirements.layers[static_cast<size_t>(src)])) {
        return InvalidArgumentError("Layer " + std::to_string(i) +
                                    ": SharedCache source_layer " + std::to_string(src) +
                                    " does not own a cache");
      }
      plan.aliases.push_back(LayerAlias{i, src});
      continue;
    }

    const CacheSpec& declared = std::get<CacheSpec>(layer_cache);
    if (Status status = declared.Validate(block_size); !status.ok()) {
      return status;
    }
    CacheSpec spec = MaybePromoteToFull(declared, max_model_len);

    if (!spec.IsTokenWise()) {
      if (!has_recurrent) {
        has_recurrent = true;
        recurrent_group.spec = std::move(spec);
        recurrent_group.layers.push_back(i);
      } else if (recurrent_group.spec == spec) {
        recurrent_group.layers.push_back(i);
      } else {
        return InvalidArgumentError("Layer " + std::to_string(i) +
                                    ": all recurrent layers must share one CacheSpec, got " +
                                    spec.ToString() + " vs " + recurrent_group.spec.ToString());
      }
      continue;
    }

    auto it = std::find_if(token_groups.begin(), token_groups.end(),
                           [&spec](const CacheGroupSpec& g) { return g.spec == spec; });
    if (it == token_groups.end()) {
      CacheGroupSpec group;
      group.spec = std::move(spec);
      group.layers.push_back(i);
      token_groups.push_back(std::move(group));
    } else {
      it->layers.push_back(i);
    }
  }

  if (token_groups.empty() && !has_recurrent) {
    return InvalidArgumentError("Model has no layer with a cache");
  }

  // Token-wise groups were created in order of their first layer; the
  // recurrent group goes last.
  plan.groups = std::move(token_groups);
  if (has_recurrent) {
    plan.groups.push_back(std::move(recurrent_group));
  }
  for (size_t g = 0; g < plan.groups.size(); ++g) {
    plan.groups[g].group_index = static_cast<int64_t>(g);
  }

  // Phase-1: every group must demand the same per-layer page size.
  const int64_t page_bytes = PerLayerPageBytes(plan.groups.front().spec, block_size);
  for (const CacheGroupSpec& group : plan.groups) {
    const int64_t group_page_bytes = PerLayerPageBytes(group.spec, block_size);
    if (group_page_bytes != page_bytes) {
      return InvalidArgumentError(
          "Cache groups require different page sizes: group 0 needs " +
          std::to_string(page_bytes) + " bytes per layer page but group " +
          std::to_string(group.group_index) + " needs " + std::to_string(group_page_bytes));
    }
  }

  return plan;
}

}  // namespace inferx