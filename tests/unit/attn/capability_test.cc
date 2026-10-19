// Unit tests for capability matching (docs/attention.md §7.1): one
// generous backend capability probed rejection-reason by rejection-reason.

#include "inferx/ops/attention/capability.h"

#include <gtest/gtest.h>

#include <array>

namespace inferx::attn {
namespace {

constexpr std::uint32_t kDenseId = static_cast<std::uint32_t>(Mechanism::kDense);
constexpr std::uint32_t kMlaId = static_cast<std::uint32_t>(Mechanism::kMla);
constexpr std::array<std::uint32_t, 1> kDenseMech{kDenseId};
constexpr std::array<std::int16_t, 3> kHeadDims{64, 128, 256};

/// Generous dense backend: everything the probe specs need by default.
Capability MakeCapability() {
  Capability cap;
  cap.name = "probe";
  cap.mechanisms = kDenseMech;
  cap.roles = {Role::kDecoderSelf, Role::kEncoderSelf, Role::kCross};
  cap.masks = {MaskKind::kCausal, MaskKind::kBidirectional, MaskKind::kSlidingWindow,
               MaskKind::kChunked};
  cap.q_dtypes = {DataType::kFloat16, DataType::kBFloat16};
  cap.kv_dtypes = {DataType::kFloat16, DataType::kBFloat16, DataType::kFloat8E4M3FN};
  cap.head_dims = kHeadDims;
  cap.layouts = {KvPageLayout::kKvPlanarNhd, KvPageLayout::kKvPlanarHnd};
  cap.block = {.multiple_of = 16, .max_tokens = 0};
  cap.kv_update = KvUpdateMode::kSeparate;
  cap.graph = GraphSupport::kUniformDecode;
  cap.min_sm = 80;
  cap.max_sm = 0;
  return cap;
}

AttentionSpec MatchingSpec() {
  AttentionSpec spec;
  spec.params = DenseParams{.num_q_heads = 32, .num_kv_heads = 8, .head_dim = 128};
  spec.kv_layout = KvPageLayout::kKvPlanarNhd;
  return spec;
}

Env MatchingEnv() { return Env{.sm = 90, .block_size = 16}; }

TEST(CheckCapabilityTest, AcceptsAMatchingSpec) {
  const Verdict verdict = CheckCapability(MakeCapability(), MatchingSpec(), MatchingEnv());
  EXPECT_TRUE(verdict) << verdict.detail;
  EXPECT_EQ(verdict.why, Reject::kNone);
}

TEST(CheckCapabilityTest, RejectsInDocumentedOrder) {
  // Mechanism.
  AttentionSpec mla;
  mla.params = MlaParams{.num_heads = 128, .kv_lora_rank = 512, .qk_rope_dim = 64};
  EXPECT_EQ(CheckCapability(MakeCapability(), mla, MatchingEnv()).why, Reject::kMechanism);

  // Role.
  AttentionSpec role = MatchingSpec();
  role.role = Role::kCross;
  {
    Capability cap = MakeCapability();
    cap.roles = {Role::kDecoderSelf};
    EXPECT_EQ(CheckCapability(cap, role, MatchingEnv()).why, Reject::kRole);
  }

  // Mask.
  {
    AttentionSpec spec = MatchingSpec();
    spec.mask = MaskSpec{MaskKind::kSlidingWindow, 4096, 0};
    Capability cap = MakeCapability();
    cap.masks = {MaskKind::kCausal};
    EXPECT_EQ(CheckCapability(cap, spec, MatchingEnv()).why, Reject::kMask);
  }

  // Feature.
  {
    AttentionSpec spec = MatchingSpec();
    spec.features = FeatureSet{}.With(Feature::kSinks);
    EXPECT_EQ(CheckCapability(MakeCapability(), spec, MatchingEnv()).why, Reject::kFeature);
  }

  // Dtypes.
  {
    AttentionSpec spec = MatchingSpec();
    spec.q_dtype = DataType::kFloat64;
    EXPECT_EQ(CheckCapability(MakeCapability(), spec, MatchingEnv()).why, Reject::kQDtype);
    spec = MatchingSpec();
    spec.kv_dtype = DataType::kInt32;
    EXPECT_EQ(CheckCapability(MakeCapability(), spec, MatchingEnv()).why, Reject::kKvDtype);
  }

  // Head dim.
  {
    AttentionSpec spec = MatchingSpec();
    spec.params = DenseParams{.num_q_heads = 32, .num_kv_heads = 8, .head_dim = 192};
    EXPECT_EQ(CheckCapability(MakeCapability(), spec, MatchingEnv()).why, Reject::kHeadDim);
  }

  // Layout.
  {
    AttentionSpec spec = MatchingSpec();
    spec.kv_layout = KvPageLayout::kLatent;
    EXPECT_EQ(CheckCapability(MakeCapability(), spec, MatchingEnv()).why, Reject::kLayout);
  }

  // Block size: not a multiple / above the cap.
  {
    Capability cap = MakeCapability();
    cap.block = {.multiple_of = 32, .max_tokens = 0};
    EXPECT_EQ(CheckCapability(cap, MatchingSpec(), MatchingEnv()).why, Reject::kBlockSize);
    cap.block = {.multiple_of = 1, .max_tokens = 8};
    EXPECT_EQ(CheckCapability(cap, MatchingSpec(), MatchingEnv()).why, Reject::kBlockSize);
  }

  // Architecture.
  {
    Capability cap = MakeCapability();
    cap.max_sm = 89;
    EXPECT_EQ(CheckCapability(cap, MatchingSpec(), MatchingEnv()).why, Reject::kArch);
  }

  // Graph support requirement.
  {
    Env env = MatchingEnv();
    env.min_graph_support = GraphSupport::kAny;
    EXPECT_EQ(CheckCapability(MakeCapability(), MatchingSpec(), env).why, Reject::kGraph);
  }

  // GQA divisibility.
  {
    AttentionSpec spec = MatchingSpec();
    spec.params = DenseParams{.num_q_heads = 33, .num_kv_heads = 8, .head_dim = 128};
    EXPECT_EQ(CheckCapability(MakeCapability(), spec, MatchingEnv()).why, Reject::kGqa);
  }
}

TEST(CheckCapabilityTest, UnknownMachineAndUnplannedBlocksSkipTheirChecks) {
  // sm == 0 (unknown) and block_size == 0 (pre-planning) must not reject.
  const Env env{};
  const Verdict verdict = CheckCapability(MakeCapability(), MatchingSpec(), env);
  EXPECT_TRUE(verdict) << verdict.detail;
}

TEST(CheckCapabilityTest, SparseRejectsThroughItsDenseBase) {
  // The mechanism gate must pass first, so the probe admits Sparse too;
  // the head-dim check then looks through the sparse wrapper at its base.
  static constexpr std::array<std::uint32_t, 2> kMechs{
      static_cast<std::uint32_t>(Mechanism::kDense),
      static_cast<std::uint32_t>(Mechanism::kSparse)};
  Capability cap = MakeCapability();
  cap.mechanisms = kMechs;

  AttentionSpec spec = MatchingSpec();
  SparseParams sparse;
  sparse.base = DenseParams{.num_q_heads = 32, .num_kv_heads = 8, .head_dim = 512};
  spec.params = sparse;
  EXPECT_EQ(CheckCapability(cap, spec, MatchingEnv()).why, Reject::kHeadDim);

  // A valid base passes; the sparse wrapper adds no constraint of its own.
  sparse.base.head_dim = 128;
  spec.params = sparse;
  const Verdict verdict = CheckCapability(cap, spec, MatchingEnv());
  EXPECT_TRUE(verdict) << verdict.detail;
}

TEST(CheckCapabilityTest, EnumSetHoldsEveryDataTypeWithoutTruncation) {
  // `DataType` has more than 32 enumerators; the set must hold the last
  // one alongside the first.
  EnumSet<DataType> set;
  set.bits = 0;
  const auto first = static_cast<DataType>(0);
  const auto last = static_cast<DataType>(static_cast<int>(DataType::kComplex128));
  EXPECT_FALSE(set.Has(first));
  set = EnumSet<DataType>{first, last};
  EXPECT_TRUE(set.Has(first));
  EXPECT_TRUE(set.Has(last));
}

}  // namespace
}  // namespace inferx::attn
