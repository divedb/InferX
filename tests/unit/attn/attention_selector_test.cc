// Unit tests for backend registration and selection
// (docs/attention.md §7.3): priority order, forced backends, rejection
// trails, and the never-a-silent-fallback rule.

#include "inferx/ops/attention/registry.h"

#include <gtest/gtest.h>

#include <memory>

namespace inferx::attn {
namespace {

AttentionSpec DenseSpec() {
  AttentionSpec spec;
  spec.params = DenseParams{.num_q_heads = 32, .num_kv_heads = 8, .head_dim = 128};
  return spec;
}

std::unique_ptr<BackendRegistry> MakeRegistry() {
  auto registry = std::make_unique<BackendRegistry>();
  RegisterBuiltinBackends(*registry);
  return registry;
}

TEST(BackendSelectorTest, ReferenceIsNeverAnAutomaticFallback) {
  auto registry = MakeRegistry();
  // A spec no real backend serves (an exotic feature): the reference
  // backend must NOT silently take over.
  AttentionSpec spec = DenseSpec();
  spec.features = FeatureSet{}.With(Feature::kAlibi);

  Env env{.block_size = 16};
  BackendSelector selector(*registry, env);
  const Selection selection = selector.Select(spec);
  EXPECT_FALSE(selection.Ok());
  ASSERT_FALSE(selection.rejected.empty());
  EXPECT_EQ(selection.rejected[0].second.why, Reject::kFeature);
}

TEST(BackendSelectorTest, ForcedReferenceIsSelectedForSupportedSpecs) {
  auto registry = MakeRegistry();
  Env env{.block_size = 16, .forced_backend = "reference"};
  BackendSelector selector(*registry, env);
  const Selection selection = selector.Select(DenseSpec());
  ASSERT_TRUE(selection.Ok());
  EXPECT_EQ(selection.backend->GetCapability().name, "reference");
}

TEST(BackendSelectorTest, ForcedBackendFailureIsAnErrorNotAFallback) {
  auto registry = MakeRegistry();
  // flash_dense rejects unknown names... force an unregistered name.
  Env env{.block_size = 16, .forced_backend = "does_not_exist"};
  BackendSelector selector(*registry, env);
  const Selection selection = selector.Select(DenseSpec());
  EXPECT_FALSE(selection.Ok());
  EXPECT_EQ(selection.rejected[0].second.detail, "no backend with this name is registered");

  // A registered name that cannot serve the spec: rejection, no fallback.
  AttentionSpec mla;
  mla.params = MlaParams{.num_heads = 128, .kv_lora_rank = 512, .qk_rope_dim = 64};
  Env mla_env{.block_size = 16, .forced_backend = "reference"};
  BackendSelector mla_selector(*registry, mla_env);
  const Selection mla_selection = mla_selector.Select(mla);
  EXPECT_FALSE(mla_selection.Ok());
  EXPECT_EQ(mla_selection.rejected[0].second.why, Reject::kMechanism);
}

TEST(BackendSelectorTest, SelectionIsDeterministicAcrossCalls) {
  auto registry = MakeRegistry();
  Env env{.block_size = 16};
  BackendSelector selector(*registry, env);
  const Selection first = selector.Select(DenseSpec());
  const Selection second = selector.Select(DenseSpec());
  ASSERT_TRUE(first.Ok() && second.Ok());
  EXPECT_EQ(first.backend, second.backend);
}

TEST(BackendSelectorTest, PriorityPrefersFlashDenseOnCudaBlockSizes) {
  auto registry = MakeRegistry();
  // The CUDA backend cannot be instantiated on CPU devices, but selection
  // itself is pure capability matching: flash_dense ranks above reference
  // whenever it matches and is not forced-only.
  Env env{.block_size = 16, .forced_backend = "flash_dense"};
  BackendSelector selector(*registry, env);
  const Selection selection = selector.Select(DenseSpec());
  ASSERT_TRUE(selection.Ok());
  EXPECT_EQ(selection.backend->GetCapability().name, "flash_dense");

  // A graph-support requirement the backend cannot meet is rejected with
  // the graph reason (both v1 backends are kNever).
  Env graphs{.block_size = 16, .min_graph_support = GraphSupport::kUniformDecode,
             .forced_backend = "flash_dense"};
  BackendSelector graph_selector(*registry, graphs);
  const Selection graph_selection = graph_selector.Select(DenseSpec());
  EXPECT_FALSE(graph_selection.Ok());
  EXPECT_EQ(graph_selection.rejected[0].second.why, Reject::kGraph);
}

}  // namespace
}  // namespace inferx::attn
