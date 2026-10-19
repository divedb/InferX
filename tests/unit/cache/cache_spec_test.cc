#include "inferx/cache/cache_spec.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace inferx {
namespace {

constexpr int64_t kBlockSize = 16;

CacheComponent TokenComponent(int64_t bytes_per_token) {
  return CacheComponent::PerToken(bytes_per_token, DataType::kBFloat16);
}

CacheComponent StateComponent(int64_t bytes_per_slot) {
  return CacheComponent::FixedSlot(bytes_per_slot, DataType::kBFloat16);
}

TEST(CacheComponentTest, PerTokenFactorySetsGrowthAndZeroSlotBytes) {
  const CacheComponent component = CacheComponent::PerToken(16, DataType::kFloat16);
  EXPECT_EQ(component.growth, CacheComponent::Growth::kPerToken);
  EXPECT_EQ(component.bytes_per_token, 16);
  EXPECT_EQ(component.bytes_per_slot, 0);
  EXPECT_EQ(component.dtype, DataType::kFloat16);
}

TEST(CacheComponentTest, FixedSlotFactorySetsGrowthAndZeroTokenBytes) {
  const CacheComponent component = CacheComponent::FixedSlot(512, DataType::kFloat32);
  EXPECT_EQ(component.growth, CacheComponent::Growth::kFixedSlot);
  EXPECT_EQ(component.bytes_per_token, 0);
  EXPECT_EQ(component.bytes_per_slot, 512);
  EXPECT_EQ(component.dtype, DataType::kFloat32);
}

TEST(CacheComponentTest, EqualityComparesAllFields) {
  EXPECT_EQ(CacheComponent::PerToken(16, DataType::kBFloat16),
            CacheComponent::PerToken(16, DataType::kBFloat16));
  EXPECT_NE(CacheComponent::PerToken(16, DataType::kBFloat16),
            CacheComponent::PerToken(32, DataType::kBFloat16));
  EXPECT_NE(CacheComponent::PerToken(16, DataType::kBFloat16),
            CacheComponent::PerToken(16, DataType::kFloat16));
  EXPECT_NE(CacheComponent::PerToken(16, DataType::kBFloat16),
            CacheComponent::FixedSlot(16, DataType::kBFloat16));
}

TEST(CacheComponentTest, GqaKvComponentSizesBothKeyAndValue) {
  // Each token stores 2 * kv_heads * head_dim elements.
  EXPECT_EQ(GqaKvComponent(8, 128, DataType::kBFloat16).bytes_per_token, 4096);
  EXPECT_EQ(GqaKvComponent(1, 64, DataType::kInt8).bytes_per_token, 128);
  // Sub-byte dtypes pack: 2 * 2 * 64 = 256 int4 elements = 128 bytes.
  EXPECT_EQ(GqaKvComponent(2, 64, DataType::kInt4).bytes_per_token, 128);
}

TEST(CacheComponentTest, MlaLatentComponentSizesSingleLatent) {
  EXPECT_EQ(MlaLatentComponent(576, DataType::kFloat8E4M3FN).bytes_per_token, 576);
  // 12 int4 elements pack into 6 bytes.
  EXPECT_EQ(MlaLatentComponent(12, DataType::kInt4).bytes_per_token, 6);
}

TEST(CacheSpecTest, FactoriesSelectMatchingRetention) {
  const std::vector<CacheComponent> components = {TokenComponent(64)};

  EXPECT_EQ(CacheSpec::FullAttention(components).retention, Retention{FullRetention{}});
  EXPECT_EQ(CacheSpec::SlidingWindow(components, 1024).retention,
            (Retention{WindowRetention{1024, 0}}));
  EXPECT_EQ(CacheSpec::SlidingWindow(components, 1024, 64).retention,
            (Retention{WindowRetention{1024, 64}}));
  EXPECT_EQ(CacheSpec::ChunkedLocal(components, 8192).retention,
            Retention{ChunkRetention{8192}});
  EXPECT_EQ(CacheSpec::Encoder(components).retention, Retention{EncoderRetention{}});
  // Cross-attention K/V are fixed during decoding, i.e. encoder retention.
  EXPECT_EQ(CacheSpec::CrossAttention(components), CacheSpec::Encoder(components));
  EXPECT_EQ(CacheSpec::Recurrent({StateComponent(1024)}, 64).retention,
            Retention{RecurrentRetention{64}});
}

TEST(CacheSpecTest, RetentionClassifiesTokenWiseAndPrefixCacheable) {
  EXPECT_TRUE(CacheSpec::FullAttention({TokenComponent(64)}).IsTokenWise());
  EXPECT_TRUE(CacheSpec::SlidingWindow({TokenComponent(64)}, 1024).IsTokenWise());
  EXPECT_TRUE(CacheSpec::ChunkedLocal({TokenComponent(64)}, 8192).IsTokenWise());
  EXPECT_TRUE(CacheSpec::Encoder({TokenComponent(64)}).IsTokenWise());
  EXPECT_FALSE(CacheSpec::Recurrent({StateComponent(1024)}, 64).IsTokenWise());

  EXPECT_TRUE(CacheSpec::FullAttention({TokenComponent(64)}).PrefixCacheable());
  EXPECT_TRUE(CacheSpec::SlidingWindow({TokenComponent(64)}, 1024).PrefixCacheable());
  EXPECT_TRUE(CacheSpec::ChunkedLocal({TokenComponent(64)}, 8192).PrefixCacheable());
  EXPECT_FALSE(CacheSpec::Encoder({TokenComponent(64)}).PrefixCacheable());
  EXPECT_FALSE(CacheSpec::Recurrent({StateComponent(1024)}, 64).PrefixCacheable());
}

TEST(CacheSpecTest, ByteAccountingSumsOnlyMatchingGrowthKind) {
  CacheSpec spec;
  spec.components = {TokenComponent(16), StateComponent(96), TokenComponent(48)};
  spec.retention = FullRetention{};
  EXPECT_EQ(spec.BytesPerTokenPerLayer(), 64);
  EXPECT_EQ(spec.StateBytesPerLayer(), 96);
}

TEST(CacheSpecTest, MaxBlocksCeilDividesTokenCount) {
  const CacheSpec full = CacheSpec::FullAttention({TokenComponent(64)});
  EXPECT_EQ(full.MaxBlocks(128, kBlockSize), 8);
  EXPECT_EQ(full.MaxBlocks(129, kBlockSize), 9);
  EXPECT_EQ(CacheSpec::Encoder({TokenComponent(64)}).MaxBlocks(130, kBlockSize), 9);
}

TEST(CacheSpecTest, MaxBlocksForWindowCoversWindowPlusSinks) {
  // ceil((1024 + 64) / 16); independent of the token count.
  const CacheSpec with_sinks = CacheSpec::SlidingWindow({TokenComponent(64)}, 1024, 64);
  EXPECT_EQ(with_sinks.MaxBlocks(1 << 20, kBlockSize), 68);
  // A window misaligned with the block size still fits in ceil(window/16).
  const CacheSpec unaligned = CacheSpec::SlidingWindow({TokenComponent(64)}, 100);
  EXPECT_EQ(unaligned.MaxBlocks(1000, kBlockSize), 7);
  EXPECT_EQ(CacheSpec::ChunkedLocal({TokenComponent(64)}, 8192).MaxBlocks(1 << 20, kBlockSize),
            512);
  EXPECT_EQ(CacheSpec::Recurrent({StateComponent(1024)}, 64).MaxBlocks(1 << 20, kBlockSize),
            0);
}

TEST(CacheSpecTest, ValidateAcceptsWellFormedSpecs) {
  EXPECT_TRUE(CacheSpec::FullAttention({TokenComponent(64)}).Validate(kBlockSize).ok());
  EXPECT_TRUE(CacheSpec::SlidingWindow({TokenComponent(64)}, 1024).Validate(kBlockSize).ok());
  EXPECT_TRUE(
      CacheSpec::SlidingWindow({TokenComponent(64)}, 1024, 64).Validate(kBlockSize).ok());
  EXPECT_TRUE(CacheSpec::ChunkedLocal({TokenComponent(64)}, 8192).Validate(kBlockSize).ok());
  EXPECT_TRUE(CacheSpec::Encoder({TokenComponent(64)}).Validate(kBlockSize).ok());
  EXPECT_TRUE(CacheSpec::Recurrent({StateComponent(1024)}, 64).Validate(kBlockSize).ok());
}

TEST(CacheSpecTest, ValidateRejectsSpecWithoutComponents) {
  CacheSpec spec;
  EXPECT_EQ(spec.Validate(kBlockSize).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(CacheSpec::FullAttention({}).Validate(kBlockSize).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CacheSpecTest, ValidateRejectsGrowthMismatchedToRetention) {
  // Token-based retention requires per-token components.
  EXPECT_EQ(CacheSpec::FullAttention({StateComponent(1024)}).Validate(kBlockSize).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      CacheSpec::SlidingWindow({StateComponent(1024)}, 1024).Validate(kBlockSize).code(),
      absl::StatusCode::kInvalidArgument);
  // Recurrent retention requires fixed-slot components.
  EXPECT_EQ(CacheSpec::Recurrent({TokenComponent(64)}, 64).Validate(kBlockSize).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(CacheSpecTest, ValidateRejectsMalformedRetentions) {
  const auto invalid_argument = [](const CacheSpec& spec) {
    return spec.Validate(kBlockSize).code() == absl::StatusCode::kInvalidArgument;
  };

  EXPECT_TRUE(invalid_argument(CacheSpec::SlidingWindow({TokenComponent(64)}, 0)));
  EXPECT_TRUE(invalid_argument(CacheSpec::SlidingWindow({TokenComponent(64)}, -8)));
  // Sinks must be non-negative and block-aligned.
  EXPECT_TRUE(invalid_argument(CacheSpec::SlidingWindow({TokenComponent(64)}, 1024, 8)));
  EXPECT_TRUE(invalid_argument(CacheSpec::SlidingWindow({TokenComponent(64)}, 1024, -16)));

  EXPECT_TRUE(invalid_argument(CacheSpec::ChunkedLocal({TokenComponent(64)}, 0)));
  EXPECT_TRUE(invalid_argument(CacheSpec::ChunkedLocal({TokenComponent(64)}, 40)));

  EXPECT_TRUE(invalid_argument(CacheSpec::Recurrent({StateComponent(1024)}, 0)));
  EXPECT_TRUE(invalid_argument(CacheSpec::Recurrent({StateComponent(1024)}, 24)));
}

TEST(CacheSpecTest, ToStringDescribesRetentionAndComponents) {
  const CacheSpec spec =
      CacheSpec::SlidingWindow({TokenComponent(64), StateComponent(96)}, 1024, 64);
  const std::string text = spec.ToString();
  EXPECT_NE(text.find("WindowRetention(window=1024, sink_tokens=64)"), std::string::npos);
  EXPECT_NE(text.find("growth=PerToken, bytes_per_token=64, bytes_per_slot=0"),
            std::string::npos);
  EXPECT_NE(text.find("growth=FixedSlot, bytes_per_token=0, bytes_per_slot=96"),
            std::string::npos);

  EXPECT_NE(CacheSpec::FullAttention({TokenComponent(64)}).ToString().find("FullRetention"),
            std::string::npos);
  EXPECT_NE(CacheSpec::ChunkedLocal({TokenComponent(64)}, 8192).ToString().find(
                "ChunkRetention(chunk_tokens=8192)"),
            std::string::npos);
  EXPECT_NE(CacheSpec::Encoder({TokenComponent(64)}).ToString().find("EncoderRetention"),
            std::string::npos);
  EXPECT_NE(CacheSpec::Recurrent({StateComponent(1024)}, 64).ToString().find(
                "RecurrentRetention(boundary_tokens=64)"),
            std::string::npos);
}

TEST(CacheGroupSpecTest, TokenWiseSizingScalesWithLayersAndBlocks) {
  CacheGroupSpec group;
  group.layers = {0, 1, 2};
  group.spec = CacheSpec::FullAttention({TokenComponent(16), TokenComponent(48)});
  EXPECT_EQ(group.NumLayers(), 3);
  EXPECT_EQ(group.BytesPerBlock(kBlockSize), 3 * 64 * kBlockSize);
  EXPECT_EQ(group.StateBytesPerSequence(), 0);
  // ceil(130/16) = 9 blocks of 3 * 64 * 16 bytes.
  EXPECT_EQ(group.MaxBytesPerSequence(130, kBlockSize), 9 * 3 * 64 * kBlockSize);
}

TEST(CacheGroupSpecTest, WindowSizingIsBoundedByRetention) {
  CacheGroupSpec group;
  group.layers = {0};
  group.spec = CacheSpec::SlidingWindow({TokenComponent(64)}, 1024, 64);
  EXPECT_EQ(group.MaxBytesPerSequence(1 << 20, kBlockSize), 68 * 64 * kBlockSize);
}

TEST(CacheGroupSpecTest, RecurrentSizingUsesStateSlotsOnly) {
  CacheGroupSpec group;
  group.layers = {0, 1, 2, 3};
  group.spec = CacheSpec::Recurrent({StateComponent(512)}, 64);
  EXPECT_EQ(group.BytesPerBlock(kBlockSize), 0);
  EXPECT_EQ(group.StateBytesPerSequence(), 4 * 512);
  EXPECT_EQ(group.MaxBytesPerSequence(4096, kBlockSize), 4 * 512);
}

TEST(PlanCacheGroupsTest, GroupsLayersWithIdenticalSpecs) {
  CacheRequirements requirements;
  requirements.layers = {CacheSpec::FullAttention({TokenComponent(64)}), NoCache{},
                         CacheSpec::FullAttention({TokenComponent(64)})};
  auto plan = PlanCacheGroups(requirements, kBlockSize);
  ASSERT_TRUE(plan.ok()) << plan.status();
  ASSERT_EQ(plan->NumGroups(), 1);
  EXPECT_EQ(plan->groups[0].layers, std::vector<int64_t>({0, 2}));
  EXPECT_EQ(plan->groups[0].group_index, 0);
  EXPECT_TRUE(plan->aliases.empty());
  EXPECT_EQ(plan->block_size, kBlockSize);
  EXPECT_EQ(plan->GroupOfLayer(2), &plan->groups[0]);
  // Layers without a cache, and unknown layers, map to no group.
  EXPECT_EQ(plan->GroupOfLayer(1), nullptr);
  EXPECT_EQ(plan->GroupOfLayer(42), nullptr);
}

TEST(PlanCacheGroupsTest, DistinctSpecsFormSeparateGroupsOrderedByFirstLayer) {
  CacheRequirements requirements;
  requirements.layers = {CacheSpec::SlidingWindow({TokenComponent(64)}, 1024, 64),
                         CacheSpec::FullAttention({TokenComponent(64)}),
                         CacheSpec::SlidingWindow({TokenComponent(64)}, 1024, 64),
                         CacheSpec::ChunkedLocal({TokenComponent(64)}, 8192)};
  auto plan = PlanCacheGroups(requirements, kBlockSize);
  ASSERT_TRUE(plan.ok()) << plan.status();
  ASSERT_EQ(plan->NumGroups(), 3);
  EXPECT_EQ(plan->groups[0].layers, std::vector<int64_t>({0, 2}));
  EXPECT_EQ(plan->groups[1].layers, std::vector<int64_t>({1}));
  EXPECT_EQ(plan->groups[2].layers, std::vector<int64_t>({3}));
  for (int64_t g = 0; g < plan->NumGroups(); ++g) {
    EXPECT_EQ(plan->groups[g].group_index, g);
  }
  EXPECT_EQ(plan->NumTokenWiseGroups(), 3);
  EXPECT_FALSE(plan->HasAtMostOneTokenWiseGroup());
  EXPECT_EQ(plan->TokenWiseGroups().size(), 3u);
  EXPECT_EQ(plan->RecurrentGroup(), nullptr);
}

TEST(PlanCacheGroupsTest, RecurrentGroupOccupiesFinalPosition) {
  CacheRequirements requirements;
  // State slots of 64 * 16 bytes match the token group's per-layer page size.
  requirements.layers = {CacheSpec::FullAttention({TokenComponent(64)}),
                         CacheSpec::Recurrent({StateComponent(kBlockSize * 64)}, 64),
                         CacheSpec::FullAttention({TokenComponent(64)})};
  auto plan = PlanCacheGroups(requirements, kBlockSize);
  ASSERT_TRUE(plan.ok()) << plan.status();
  ASSERT_EQ(plan->NumGroups(), 2);
  EXPECT_EQ(plan->groups[0].layers, std::vector<int64_t>({0, 2}));
  EXPECT_EQ(plan->groups[1].layers, std::vector<int64_t>({1}));
  EXPECT_FALSE(plan->groups[1].spec.IsTokenWise());
  EXPECT_EQ(plan->NumTokenWiseGroups(), 1);
  EXPECT_TRUE(plan->HasAtMostOneTokenWiseGroup());
  EXPECT_EQ(plan->TokenWiseGroups().size(), 1u);
  EXPECT_EQ(plan->RecurrentGroup(), &plan->groups[1]);
  EXPECT_EQ(plan->GroupOfLayer(1), &plan->groups[1]);
  EXPECT_EQ(plan->GroupOfLayer(0), &plan->groups[0]);
}

TEST(PlanCacheGroupsTest, DivergentRecurrentSpecsAreRejected) {
  CacheRequirements requirements;
  requirements.layers = {CacheSpec::Recurrent({StateComponent(1024)}, 64),
                         CacheSpec::Recurrent({StateComponent(1024)}, 128)};
  auto plan = PlanCacheGroups(requirements, kBlockSize);
  EXPECT_EQ(plan.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(PlanCacheGroupsTest, SharedCacheLayersResolveToSourceGroup) {
  CacheRequirements requirements;
  requirements.layers = {CacheSpec::FullAttention({TokenComponent(64)}), SharedCache{0},
                         CacheSpec::FullAttention({TokenComponent(64)})};
  auto plan = PlanCacheGroups(requirements, kBlockSize);
  ASSERT_TRUE(plan.ok()) << plan.status();
  ASSERT_EQ(plan->NumGroups(), 1);
  // The consumer is aliased, not counted as a cache-owning layer.
  EXPECT_EQ(plan->groups[0].layers, std::vector<int64_t>({0, 2}));
  ASSERT_EQ(plan->aliases.size(), 1u);
  EXPECT_EQ(plan->aliases[0], (LayerAlias{1, 0}));
  EXPECT_EQ(plan->GroupOfLayer(1), plan->GroupOfLayer(0));
}

TEST(PlanCacheGroupsTest, InvalidSharedCacheSourcesAreRejected) {
  CacheRequirements self_reference;
  self_reference.layers = {CacheSpec::FullAttention({TokenComponent(64)}), SharedCache{1}};
  EXPECT_EQ(PlanCacheGroups(self_reference, kBlockSize).status().code(),
            absl::StatusCode::kInvalidArgument);

  CacheRequirements negative;
  negative.layers = {CacheSpec::FullAttention({TokenComponent(64)}), SharedCache{-1}};
  EXPECT_EQ(PlanCacheGroups(negative, kBlockSize).status().code(),
            absl::StatusCode::kInvalidArgument);

  CacheRequirements unowned;
  unowned.layers = {NoCache{}, SharedCache{0}};
  EXPECT_EQ(PlanCacheGroups(unowned, kBlockSize).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(PlanCacheGroupsTest, WindowAndChunkPromoteToFullWhenNeverClipping) {
  CacheRequirements requirements;
  requirements.layers = {CacheSpec::SlidingWindow({TokenComponent(64)}, 8192),
                         CacheSpec::ChunkedLocal({TokenComponent(64)}, 16384),
                         CacheSpec::SlidingWindow({TokenComponent(64)}, 1024)};
  auto plan = PlanCacheGroups(requirements, kBlockSize, 8192);
  ASSERT_TRUE(plan.ok()) << plan.status();
  // The first two layers promote to full retention and merge into one group;
  // the layer that can still clip keeps its window retention.
  ASSERT_EQ(plan->NumGroups(), 2);
  EXPECT_EQ(plan->groups[0].layers, std::vector<int64_t>({0, 1}));
  EXPECT_EQ(plan->groups[0].spec, CacheSpec::FullAttention({TokenComponent(64)}));
  EXPECT_EQ(plan->groups[1].layers, std::vector<int64_t>({2}));
  EXPECT_EQ(plan->groups[1].spec.retention, (Retention{WindowRetention{1024, 0}}));

  // Without a maximum length nothing is promoted.
  auto unplanned = PlanCacheGroups(requirements, kBlockSize);
  ASSERT_TRUE(unplanned.ok()) << unplanned.status();
  ASSERT_EQ(unplanned->NumGroups(), 3);
  EXPECT_EQ(unplanned->groups[0].spec.retention, (Retention{WindowRetention{8192, 0}}));
}

TEST(PlanCacheGroupsTest, DivergentPageSizesAreRejected) {
  CacheRequirements requirements;
  requirements.layers = {CacheSpec::FullAttention({TokenComponent(64)}),
                         CacheSpec::FullAttention({TokenComponent(128)})};
  auto plan = PlanCacheGroups(requirements, kBlockSize);
  EXPECT_EQ(plan.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(PlanCacheGroupsTest, InvalidArgumentsAreRejected) {
  CacheRequirements requirements;
  requirements.layers = {CacheSpec::FullAttention({TokenComponent(64)})};
  EXPECT_EQ(PlanCacheGroups(requirements, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(PlanCacheGroups(requirements, -kBlockSize).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(PlanCacheGroups(requirements, kBlockSize, 0).status().code(),
            absl::StatusCode::kInvalidArgument);

  // A model whose layers all opt out of caching cannot be planned.
  CacheRequirements uncached;
  uncached.layers = {NoCache{}, NoCache{}};
  EXPECT_EQ(PlanCacheGroups(uncached, kBlockSize).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(PlanCacheGroupsTest, InvalidSpecsAreRejected) {
  CacheRequirements requirements;
  requirements.layers = {CacheSpec::SlidingWindow({TokenComponent(64)}, 1024, /*sink_tokens=*/8)};
  auto plan = PlanCacheGroups(requirements, kBlockSize);
  EXPECT_EQ(plan.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx
