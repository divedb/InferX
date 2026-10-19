// Unit tests for the kernel-facing CacheView bindings over the implemented
// pools (docs/attention.md §8.2): strides, base pointers, cross-checks.

#include "inferx/ops/attention/cache_view.h"

#include <gtest/gtest.h>

#include <memory>

#include "inferx/cache/kv_block_pool.h"
#include "inferx/cache/recurrent_state_pool.h"
#include "inferx/ops/attention/cache_spec_bridge.h"

namespace inferx::attn {
namespace {

constexpr int64_t kBlockSize = 16;

CacheComponent PerToken(int64_t bytes) {
  return CacheComponent::PerToken(bytes, DataType::kBFloat16);
}

AttentionSpec DenseGqa(int32_t kv_heads = 8, int32_t head_dim = 128) {
  AttentionSpec spec;
  spec.params = DenseParams{.num_q_heads = 32, .num_kv_heads = kv_heads, .head_dim = head_dim};
  return spec;
}

StatusOr<KvBlockPool> MakePool(int64_t bytes_per_token, int64_t num_blocks = 4) {
  return KvBlockPool::Create(1, num_blocks, kBlockSize, PerToken(bytes_per_token),
                             Device::Cpu());
}

std::unique_ptr<RecurrentStatePool> MakeStatePool(int64_t bytes_per_slot) {
  CacheGroupSpec group;
  group.layers = {3};
  group.spec = CacheSpec::Recurrent(
      {CacheComponent::FixedSlot(bytes_per_slot, DataType::kFloat32)}, kBlockSize);
  auto pool = RecurrentStatePool::Create(group, 2, Device::Cpu());
  return std::make_unique<RecurrentStatePool>(std::move(*pool));
}

TEST(CacheViewTest, TokenWiseBindingMirrorsThePoolRegion) {
  // 2 * 8 * 128 bf16 = 4096 bytes per token.
  auto pool = MakePool(4096);
  ASSERT_TRUE(pool.ok()) << pool.status();

  auto bound = BindTokenWiseView(*pool, DenseGqa(), /*layer=*/0, kBlockSize);
  ASSERT_TRUE(bound.ok()) << bound.status();
  auto region = pool->LayerCache(0);
  ASSERT_TRUE(region.ok()) << region.status();

  EXPECT_EQ(bound->base, static_cast<std::byte*>(region->Data()));
  EXPECT_EQ(bound->block_tokens, kBlockSize);
  EXPECT_EQ(bound->page_bytes, kBlockSize * 4096);
  EXPECT_EQ(bound->page_stride_bytes, bound->page_bytes);
  EXPECT_EQ(bound->layout, KvPageLayout::kKvPlanarNhd);
  EXPECT_EQ(bound->dtype, DataType::kBFloat16);
  EXPECT_EQ(bound->Page(2) - bound->Page(1), static_cast<std::ptrdiff_t>(16 * 4096));
  // Page 0 is the null block's reserved row: addressable, never written.
  EXPECT_EQ(bound->Page(0), bound->base);
}

TEST(CacheViewTest, QuantizedSpecsShrinkTheExpectedBytes) {
  // fp8 kv: 2 * 8 * 128 = 2048 bytes per token.
  auto pool = MakePool(2048);
  ASSERT_TRUE(pool.ok()) << pool.status();
  AttentionSpec spec = DenseGqa();
  spec.kv_dtype = DataType::kFloat8E4M3FN;

  auto bound = BindTokenWiseView(*pool, spec, 0, kBlockSize);
  ASSERT_TRUE(bound.ok()) << bound.status();
  EXPECT_EQ(bound->page_bytes, kBlockSize * 2048);
  EXPECT_EQ(bound->dtype, DataType::kFloat8E4M3FN);
}

TEST(CacheViewTest, PoolSpecMismatchFailsLoudly) {
  // Pool built for 512 bytes/token, spec needs 4096.
  auto pool = MakePool(512);
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto bound = BindTokenWiseView(*pool, DenseGqa(), 0, kBlockSize);
  EXPECT_EQ(bound.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(CacheViewTest, BindingValidatesItsArguments) {
  auto pool = MakePool(4096);
  ASSERT_TRUE(pool.ok()) << pool.status();

  // Block size must match the pool's.
  auto mismatched = BindTokenWiseView(*pool, DenseGqa(), 0, kBlockSize * 2);
  EXPECT_EQ(mismatched.status().code(), absl::StatusCode::kInvalidArgument);

  // Linear mechanisms bind through the recurrent path instead.
  AttentionSpec linear;
  linear.params = LinearParams{.recurrence_id = 1,
                               .num_k_heads = 2,
                               .num_v_heads = 2,
                               .k_dim = 64,
                               .v_dim = 64,
                               .boundary_tokens = kBlockSize};
  auto wrong_kind = BindTokenWiseView(*pool, linear, 0, kBlockSize);
  EXPECT_EQ(wrong_kind.status().code(), absl::StatusCode::kInvalidArgument);

  // kRecurrentFlat is not a token-wise layout.
  AttentionSpec flat = DenseGqa();
  flat.kv_layout = KvPageLayout::kRecurrentFlat;
  auto flat_view = BindTokenWiseView(*pool, flat, 0, kBlockSize);
  EXPECT_EQ(flat_view.status().code(), absl::StatusCode::kInvalidArgument);

  // Unknown layer.
  auto bad_layer = BindTokenWiseView(*pool, DenseGqa(), 7, kBlockSize);
  EXPECT_EQ(bad_layer.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(CacheViewTest, RecurrentBindingMirrorsTheComponentRegion) {
  auto pool = MakeStatePool(512);
  ASSERT_NE(pool, nullptr);

  AttentionSpec linear;
  linear.params = LinearParams{.recurrence_id = 1,
                               .num_k_heads = 2,
                               .num_v_heads = 2,
                               .k_dim = 64,
                               .v_dim = 64,
                               .boundary_tokens = kBlockSize};
  linear.kv_dtype = DataType::kFloat32;

  auto bound = BindRecurrentView(*pool, linear, /*layer=*/3, /*component=*/0);
  ASSERT_TRUE(bound.ok()) << bound.status();
  auto region = pool->ComponentView(3, 0);
  ASSERT_TRUE(region.ok()) << region.status();

  EXPECT_EQ(bound->base, static_cast<std::byte*>(region->Data()));
  EXPECT_EQ(bound->slot_stride_bytes, 512);
  EXPECT_EQ(bound->Slot(1) - bound->Slot(0), static_cast<std::ptrdiff_t>(512));
  EXPECT_EQ(bound->layout, KvPageLayout::kRecurrentFlat);
  // Token-wise fields are inert for recurrent views.
  EXPECT_EQ(bound->block_tokens, 0);
  EXPECT_EQ(bound->page_stride_bytes, 0);

  // Only Linear mechanisms bind recurrent views.
  auto wrong_kind = BindRecurrentView(*pool, DenseGqa(), 3, 0);
  EXPECT_EQ(wrong_kind.status().code(), absl::StatusCode::kInvalidArgument);
  // Layers outside the group / components outside the spec fail.
  EXPECT_EQ(BindRecurrentView(*pool, linear, 4, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(BindRecurrentView(*pool, linear, 3, 1).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx::attn
