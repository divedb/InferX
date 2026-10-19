// Unit tests for the attention-spec -> CacheSpec bridge
// (docs/attention.md §5.3): mechanism/role/mask mapping and validation.

#include "inferx/ops/attention/cache_spec_bridge.h"

#include <gtest/gtest.h>

#include "inferx/cache/cache_spec.h"

namespace inferx::attn {
namespace {

constexpr int64_t kBlockSize = 16;

AttentionSpec DenseGqa() {
  AttentionSpec spec;
  spec.params = DenseParams{.num_q_heads = 32, .num_kv_heads = 8, .head_dim = 128};
  return spec;
}

AttentionSpec Linear(int32_t boundary, int32_t conv_width = 0) {
  AttentionSpec spec;
  spec.params = LinearParams{.recurrence_id = 1,
                             .num_k_heads = 2,
                             .num_v_heads = 2,
                             .k_dim = 64,
                             .v_dim = 64,
                             .conv_width = conv_width,
                             .boundary_tokens = boundary};
  spec.kv_dtype = DataType::kFloat32;
  return spec;
}

const Retention& RetentionOf(const CacheSpec& spec) { return spec.retention; }

TEST(CacheSpecBridgeTest, NeedsCacheOnlyForEncoderSelf) {
  EXPECT_TRUE(NeedsCache(DenseGqa()));
  AttentionSpec cross = DenseGqa();
  cross.role = Role::kCross;
  EXPECT_TRUE(NeedsCache(cross));
  AttentionSpec encoder = DenseGqa();
  encoder.role = Role::kEncoderSelf;
  EXPECT_FALSE(NeedsCache(encoder));
}

TEST(CacheSpecBridgeTest, DenseCausalBecomesFullAttentionGqa) {
  auto built = BuildCacheSpec(DenseGqa());
  ASSERT_TRUE(built.ok()) << built.status();
  EXPECT_EQ(RetentionOf(*built), Retention{FullRetention{}});
  ASSERT_EQ(built->components.size(), 1u);
  // Identical to the factory the cache side exports: 2 * kv * d bf16.
  EXPECT_EQ(built->components[0], GqaKvComponent(8, 128, DataType::kBFloat16));
  EXPECT_TRUE(built->PrefixCacheable());
}

TEST(CacheSpecBridgeTest, SlidingWindowBridgesWindowAndSinks) {
  AttentionSpec spec = DenseGqa();
  spec.mask = MaskSpec{MaskKind::kSlidingWindow, 4096, 32};
  auto built = BuildCacheSpec(spec);
  ASSERT_TRUE(built.ok()) << built.status();
  EXPECT_EQ(RetentionOf(*built), (Retention{WindowRetention{4096, 32}}));

  spec.mask.sink_tokens = 0;
  built = BuildCacheSpec(spec);
  ASSERT_TRUE(built.ok()) << built.status();
  EXPECT_EQ(RetentionOf(*built), (Retention{WindowRetention{4096, 0}}));
}

TEST(CacheSpecBridgeTest, ChunkedMaskBecomesChunkRetention) {
  AttentionSpec spec = DenseGqa();
  spec.mask = MaskSpec{MaskKind::kChunked, 8192, 0};
  auto built = BuildCacheSpec(spec);
  ASSERT_TRUE(built.ok()) << built.status();
  EXPECT_EQ(RetentionOf(*built), (Retention{ChunkRetention{8192}}));
}

TEST(CacheSpecBridgeTest, MlaBecomesLatentComponent) {
  AttentionSpec spec;
  spec.params = MlaParams{.num_heads = 128, .kv_lora_rank = 512, .qk_rope_dim = 64};
  auto built = BuildCacheSpec(spec);
  ASSERT_TRUE(built.ok()) << built.status();
  EXPECT_EQ(RetentionOf(*built), Retention{FullRetention{}});
  ASSERT_EQ(built->components.size(), 1u);
  EXPECT_EQ(built->components[0], MlaLatentComponent(512 + 64, DataType::kBFloat16));
  EXPECT_EQ(built->BytesPerTokenPerLayer(), (512 + 64) * 2);
}

TEST(CacheSpecBridgeTest, LinearBecomesRecurrentFixedSlots) {
  auto built = BuildCacheSpec(Linear(64));
  ASSERT_TRUE(built.ok()) << built.status();
  EXPECT_EQ(RetentionOf(*built), (Retention{RecurrentRetention{64}}));
  // State only: [v_heads, k_dim, v_dim] float32.
  ASSERT_EQ(built->components.size(), 1u);
  EXPECT_EQ(built->StateBytesPerLayer(), 2 * 64 * 64 * 4);
  EXPECT_EQ(built->BytesPerTokenPerLayer(), 0);

  // A short-conv mixer adds a second fixed-slot component.
  auto conv = BuildCacheSpec(Linear(64, /*conv_width=*/4));
  ASSERT_TRUE(conv.ok()) << conv.status();
  ASSERT_EQ(conv->components.size(), 2u);
  EXPECT_EQ(conv->StateBytesPerLayer(), 2 * 64 * 64 * 4 + 2 * 64 * 3 * 4);
}

TEST(CacheSpecBridgeTest, CrossBecomesEncoderRetention) {
  AttentionSpec spec = DenseGqa();
  spec.role = Role::kCross;
  auto built = BuildCacheSpec(spec);
  ASSERT_TRUE(built.ok()) << built.status();
  EXPECT_EQ(RetentionOf(*built), Retention{EncoderRetention{}});
  EXPECT_FALSE(built->PrefixCacheable());
}

TEST(CacheSpecBridgeTest, SparseUsesTheDenseBaseGeometry) {
  AttentionSpec spec;
  SparseParams sparse;
  sparse.base = DenseParams{.num_q_heads = 32, .num_kv_heads = 8, .head_dim = 128};
  spec.params = sparse;
  auto built = BuildCacheSpec(spec);
  ASSERT_TRUE(built.ok()) << built.status();
  ASSERT_EQ(built->components.size(), 1u);
  EXPECT_EQ(built->components[0], GqaKvComponent(8, 128, DataType::kBFloat16));
}

TEST(CacheSpecBridgeTest, BridgeOutputSatisfiesThePlanner) {
  // The bridged specs must be directly plannable: this is the integration
  // contract (docs/attention.md §8.5 steps 2-3).
  CacheRequirements requirements;
  // The planner requires uniform per-layer page bytes across groups, so the
  // recurrent state is sized to match the dense group's page (4096 x 16).
  AttentionSpec linear;
  linear.params = LinearParams{.recurrence_id = 1,
                               .num_k_heads = 4,
                               .num_v_heads = 4,
                               .k_dim = 64,
                               .v_dim = 64,
                               .boundary_tokens = 64};
  linear.kv_dtype = DataType::kFloat32;  // 4 * 64 * 64 * 4 = 65536 = 4096 * 16.
  for (const AttentionSpec& spec :
       {DenseGqa(), linear, [] {
          AttentionSpec swa = DenseGqa();
          swa.mask = MaskSpec{MaskKind::kSlidingWindow, 4096, 32};
          return swa;
        }()}) {
    auto built = BuildCacheSpec(spec);
    ASSERT_TRUE(built.ok()) << built.status();
    requirements.layers.push_back(std::move(*built));
  }
  auto plan = PlanCacheGroups(requirements, kBlockSize);
  ASSERT_TRUE(plan.ok()) << plan.status();
  EXPECT_EQ(plan->NumTokenWiseGroups(), 2);  // full and window are distinct specs
  EXPECT_NE(plan->RecurrentGroup(), nullptr);
}

TEST(CacheSpecBridgeTest, InvalidSpecsAreRejected) {
  const auto invalid_argument = [](const StatusOr<CacheSpec>& built) {
    return built.status().code() == absl::StatusCode::kInvalidArgument;
  };

  // Encoder-self is stateless: NoCache, not a CacheSpec.
  AttentionSpec encoder = DenseGqa();
  encoder.role = Role::kEncoderSelf;
  EXPECT_TRUE(invalid_argument(BuildCacheSpec(encoder)));

  // Custom mechanisms own their storage story.
  AttentionSpec custom;
  custom.params = CustomParams{.mechanism_id = 0x1001};
  EXPECT_TRUE(invalid_argument(BuildCacheSpec(custom)));

  // Non-positive window / chunk.
  AttentionSpec bad_window = DenseGqa();
  bad_window.mask = MaskSpec{MaskKind::kSlidingWindow, 0, 0};
  EXPECT_TRUE(invalid_argument(BuildCacheSpec(bad_window)));
  AttentionSpec bad_chunk = DenseGqa();
  bad_chunk.mask = MaskSpec{MaskKind::kChunked, 0, 0};
  EXPECT_TRUE(invalid_argument(BuildCacheSpec(bad_chunk)));

  // GQA divisibility.
  AttentionSpec bad_gqa;
  bad_gqa.params = DenseParams{.num_q_heads = 33, .num_kv_heads = 8, .head_dim = 128};
  EXPECT_TRUE(invalid_argument(BuildCacheSpec(bad_gqa)));

  // Linear without a state boundary.
  EXPECT_TRUE(invalid_argument(BuildCacheSpec(Linear(0))));

  // Asymmetric V heads are not representable by the GQA component.
  AttentionSpec asym = DenseGqa();
  asym.params = DenseParams{.num_q_heads = 32, .num_kv_heads = 8, .head_dim = 128,
                            .v_head_dim = 64};
  EXPECT_TRUE(invalid_argument(BuildCacheSpec(asym)));

  // Cross-attention with a window is contradictory.
  AttentionSpec cross_window = DenseGqa();
  cross_window.role = Role::kCross;
  cross_window.mask = MaskSpec{MaskKind::kSlidingWindow, 128, 0};
  EXPECT_TRUE(invalid_argument(BuildCacheSpec(cross_window)));
}

TEST(CacheSpecBridgeTest, BytesPerTokenMatchesTheComponents) {
  auto dense = CacheBytesPerToken(DenseGqa());
  ASSERT_TRUE(dense.ok()) << dense.status();
  EXPECT_EQ(*dense, 2 * 8 * 128 * 2);

  AttentionSpec fp8 = DenseGqa();
  fp8.kv_dtype = DataType::kFloat8E4M3FN;
  auto quantized = CacheBytesPerToken(fp8);
  ASSERT_TRUE(quantized.ok()) << quantized.status();
  EXPECT_EQ(*quantized, 2 * 8 * 128);  // 1 byte per element

  // Per-sequence state has no per-token answer.
  EXPECT_EQ(CacheBytesPerToken(Linear(64)).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx::attn
