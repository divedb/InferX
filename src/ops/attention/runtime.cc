#include "inferx/ops/attention/runtime.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"

#include "inferx/core/device_buffer.h"
#include "inferx/ops/attention/cache_spec_bridge.h"
#include "inferx/ops/attention/cache_view.h"

namespace inferx::attn {
namespace {

/// Human-readable rejection trail for init errors.
std::string FormatSelection(std::int64_t layer, const AttentionSpec& spec,
                            const Selection& selection) {
  std::string text = "no attention backend for layer " + std::to_string(layer) + " (mechanism " +
                     std::to_string(spec.MechanismId()) + ")";
  for (const auto& [name, verdict] : selection.rejected) {
    text += "; ";
    text.append(name);
    text += ": ";
    text += verdict.detail.empty() ? "unsupported" : verdict.detail;
  }
  if (selection.rejected.empty()) text += "; no backends registered";
  return text;
}

StatusOr<DeviceBuffer> AllocateBuffer(std::int64_t count, Device device) {
  return DeviceBuffer::Allocate(static_cast<size_t>(count) * sizeof(std::int32_t), device);
}

}  // namespace

StatusOr<std::unique_ptr<AttentionRuntime>> AttentionRuntime::Create(
    std::unique_ptr<BackendRegistry> registry, const AttentionConfig& config,
    std::span<const AttentionSpec> specs) {
  if (specs.empty()) return InvalidArgumentError("an attention runtime needs at least one spec");
  if (registry == nullptr) return InvalidArgumentError("an attention runtime needs a registry");

  // 1. Specs -> cache requirements -> plan (docs/attention.md §8.5).
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
  INFERX_ASSIGN_OR_RETURN(CachePlan plan,
                          PlanCacheGroups(requirements, config.block_size, config.max_model_len));

  const std::int64_t block_size = plan.block_size;
  EngineLimits limits;
  limits.max_reqs = static_cast<std::int32_t>(config.max_seqs);
  limits.max_tokens = static_cast<std::int32_t>(config.max_batch_tokens);
  limits.max_seq_len = static_cast<std::int32_t>(config.max_model_len);
  limits.max_blocks_per_req =
      static_cast<std::int32_t>((config.max_model_len + block_size - 1) / block_size);
  limits.max_kv_splits = 16;
  limits.device = config.device;

  // 2. Storage: the recurrent state pool first, the remainder to blocks.
  std::unique_ptr<RecurrentStatePool> state_pool;
  std::int64_t state_bytes = 0;
  if (const CacheGroupSpec* recurrent = plan.RecurrentGroup()) {
    state_bytes = recurrent->StateBytesPerSequence() * config.max_seqs;
    if (state_bytes > config.cache_bytes_budget) {
      return ResourceExhaustedError("recurrent state needs ", state_bytes,
                                    " bytes but the cache budget is ", config.cache_bytes_budget);
    }
    INFERX_ASSIGN_OR_RETURN(auto created_state,
                            RecurrentStatePool::Create(*recurrent, config.max_seqs, config.device));
    state_pool = std::make_unique<RecurrentStatePool>(std::move(created_state));
  }

  // Per-layer components in pool order: token-wise groups in plan order,
  // each group's layers ascending.
  std::vector<CacheComponent> components;
  std::vector<std::int64_t> pool_layer_of;  // Model layer -> pool layer index.
  pool_layer_of.resize(specs.size(), -1);
  std::int64_t pool_layer = 0;
  std::int64_t bytes_per_block = 0;
  for (const CacheGroupSpec& group : plan.groups) {
    if (!group.spec.IsTokenWise()) continue;
    INFERX_ASSIGN_OR_RETURN(CacheSpec member,
                            BuildCacheSpec(specs[static_cast<size_t>(group.layers.front())]));
    for (const std::int64_t model_layer : group.layers) {
      pool_layer_of[static_cast<size_t>(model_layer)] = pool_layer++;
      for (const CacheComponent& component : member.components) {
        components.push_back(component);
        bytes_per_block += component.bytes_per_token * block_size;
      }
    }
  }
  std::unique_ptr<KvBlockPool> pool;
  if (!components.empty()) {
    const std::int64_t remaining = config.cache_bytes_budget - state_bytes;
    const std::int64_t num_blocks = remaining / bytes_per_block;
    if (num_blocks <= 0) {
      return ResourceExhaustedError("the cache budget leaves no blocks: need ",
                                    bytes_per_block, " bytes per block, have ", remaining);
    }
    INFERX_ASSIGN_OR_RETURN(auto created_pool, KvBlockPool::Create(num_blocks, block_size,
                                                                    components, config.device));
    pool = std::make_unique<KvBlockPool>(std::move(created_pool));
  }

  // 3. The facade over plan + pools.
  INFERX_ASSIGN_OR_RETURN(
      std::unique_ptr<inferx::CacheManager> cache,
      inferx::CacheManager::Create(pool.get(), state_pool.get(), plan,
                                   config.enable_prefix_caching));

  // 4. Selection and kernels per layer.
  auto runtime = std::unique_ptr<AttentionRuntime>(new AttentionRuntime());
  runtime->registry_ = std::move(registry);
  runtime->config_ = config;
  runtime->pool_ = std::move(pool);
  runtime->state_pool_ = std::move(state_pool);
  runtime->cache_ = std::move(cache);
  runtime->plan_ = std::move(plan);

  Env env;
  env.block_size = block_size;
  env.forced_backend = config.forced_backend;
  runtime->selector_ = std::make_unique<BackendSelector>(*runtime->registry_, env);

  runtime->num_groups_ = static_cast<std::uint16_t>(runtime->plan_.groups.size());
  runtime->builders_.resize(runtime->num_groups_);
  runtime->group_buffers_.resize(runtime->num_groups_);

  runtime->layers_.resize(specs.size());
  absl::flat_hash_set<std::uint16_t> groups_seen;
  for (size_t i = 0; i < specs.size(); ++i) {
    const AttentionSpec& spec = specs[i];
    BoundLayer& bound = runtime->layers_[i];
    bound.spec = spec;

    const Selection selection = runtime->selector_->Select(spec);
    if (!selection.Ok()) {
      return InvalidArgumentError(FormatSelection(static_cast<std::int64_t>(i), spec, selection));
    }

    if (pool_layer_of[i] >= 0) {
      INFERX_ASSIGN_OR_RETURN(
          bound.view, BindTokenWiseView(*runtime->pool_, spec, pool_layer_of[i], block_size));
    }
    INFERX_ASSIGN_OR_RETURN(bound.kernel,
                            selection.backend->CreateKernel(spec, bound.view, limits));

    const CacheGroupSpec* group = runtime->plan_.GroupOfLayer(static_cast<std::int64_t>(i));
    bound.group = group != nullptr ? static_cast<std::uint16_t>(group->group_index) : 0;

    if (group != nullptr) {
      const std::uint16_t g = bound.group;
      if (groups_seen.insert(g).second) {
        INFERX_ASSIGN_OR_RETURN(runtime->builders_[g],
                                selection.backend->CreateBuilder(spec, limits));
      }
    }
  }

  // 5. Persistent metadata buffers per group.
  for (std::uint16_t g = 0; g < runtime->num_groups_; ++g) {
    DeviceMetaBuffers& buffers = runtime->group_buffers_[g];
    INFERX_ASSIGN_OR_RETURN(buffers.qo_indptr,
                            AllocateBuffer(limits.max_reqs + 1, config.device));
    INFERX_ASSIGN_OR_RETURN(buffers.batch_indices, AllocateBuffer(limits.max_tokens, config.device));
    INFERX_ASSIGN_OR_RETURN(buffers.positions, AllocateBuffer(limits.max_tokens, config.device));
    INFERX_ASSIGN_OR_RETURN(buffers.kv_indptr,
                            AllocateBuffer(limits.max_reqs + 1, config.device));
    INFERX_ASSIGN_OR_RETURN(
        buffers.kv_indices,
        AllocateBuffer(std::int64_t{limits.max_reqs} * limits.max_blocks_per_req, config.device));
    INFERX_ASSIGN_OR_RETURN(buffers.last_page_len, AllocateBuffer(limits.max_reqs, config.device));
  }

  return runtime;
}

StatusOr<AttentionOp> AttentionRuntime::MakeOp(std::int64_t layer) {
  if (layer < 0 || static_cast<size_t>(layer) >= layers_.size()) {
    return InvalidArgumentError("no attention layer ", layer);
  }
  BoundLayer& bound = layers_[static_cast<size_t>(layer)];
  AttentionOp op;
  op.spec_ = bound.spec;
  op.write_kv_ = nullptr;  // Writers run inside the kernels (v1).
  op.cache_ = bound.view;
  op.group_ = bound.group;
  op.kernel_ = std::move(bound.kernel);
  if (op.kernel_ == nullptr) {
    return FailedPreconditionError("layer ", layer, " already handed out by MakeOp");
  }
  return op;
}

StatusOr<StepContext> AttentionRuntime::BeginStep(const BatchDesc& desc,
                                                  ops::OpContext& ctx) {
  StepContext step;
  step.ctx = &ctx;
  step.desc = &desc;
  step.num_groups = num_groups_;
  for (std::uint16_t g = 0; g < num_groups_; ++g) {
    if (builders_[g] == nullptr) {
      return InternalError("cache group ", g, " has no metadata builder");
    }
    BuildContext build{&ctx, &group_buffers_[g]};
    INFERX_RETURN_IF_ERROR(builders_[g]->Build(desc, g, build, step.groups[g]));

    // The retention floor, for kernels that skip below it: the safest
    // scalar is the minimum over the batch (skip only what every request
    // skips).
    std::int32_t window_left = 0;
    for (std::int32_t r = 0; r < desc.num_reqs; ++r) {
      const std::int32_t computed =
          desc.seq_lens[static_cast<size_t>(r)] - desc.query_lens[static_cast<size_t>(r)];
      window_left = r == 0 ? cache_->GroupSkippedTokens(g, computed)
                           : std::min(window_left,
                                      static_cast<std::int32_t>(cache_->GroupSkippedTokens(g, computed)));
    }
    step.groups[g].window_left = window_left;
  }
  return step;
}

}  // namespace inferx::attn
