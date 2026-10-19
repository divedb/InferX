// Unit tests for the attention spec vocabulary: mechanism mapping,
// equality, and hashing (docs/attention.md §5.2).

#include "inferx/ops/attention/spec.h"

#include <gtest/gtest.h>

#include "inferx/core/device.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"

namespace inferx::attn {
namespace {

AttentionSpec DenseSpec() {
  AttentionSpec spec;
  spec.params = DenseParams{.num_q_heads = 32, .num_kv_heads = 8, .head_dim = 128};
  return spec;
}

TEST(AttentionSpecTest, MechanismIdMapsEachAlternative) {
  AttentionSpec spec;

  spec.params = DenseParams{.num_q_heads = 32, .num_kv_heads = 8, .head_dim = 128};
  EXPECT_EQ(spec.MechanismId(), static_cast<std::uint32_t>(Mechanism::kDense));

  spec.params = MlaParams{.num_heads = 128, .kv_lora_rank = 512};
  EXPECT_EQ(spec.MechanismId(), static_cast<std::uint32_t>(Mechanism::kMla));

  spec.params = LinearParams{.recurrence_id = 1, .boundary_tokens = 64};
  EXPECT_EQ(spec.MechanismId(), static_cast<std::uint32_t>(Mechanism::kLinear));

  spec.params = SparseParams{};
  EXPECT_EQ(spec.MechanismId(), static_cast<std::uint32_t>(Mechanism::kSparse));

  CustomParams custom;
  custom.mechanism_id = 0x1001;
  spec.params = custom;
  EXPECT_EQ(spec.MechanismId(), 0x1001u);
}

TEST(AttentionSpecTest, EqualityCoversEveryField) {
  const AttentionSpec base = DenseSpec();
  EXPECT_EQ(base, DenseSpec());

  AttentionSpec other = base;
  other.role = Role::kCross;
  EXPECT_NE(base, other);
  other = base;
  other.mask = MaskSpec{MaskKind::kSlidingWindow, 4096, 0};
  EXPECT_NE(base, other);
  other = base;
  other.kv_layout = KvPageLayout::kLatent;
  EXPECT_NE(base, other);
  other = base;
  other.features = FeatureSet{}.With(Feature::kSinks);
  EXPECT_NE(base, other);
}

TEST(AttentionSpecTest, HashMatchesEquality) {
  const AttentionSpec a = DenseSpec();
  AttentionSpec b = DenseSpec();
  EXPECT_EQ(a.Hash(), b.Hash());

  b.mask = MaskSpec{MaskKind::kSlidingWindow, 4096, 0};
  EXPECT_NE(a.Hash(), b.Hash());
  EXPECT_EQ(a.Hash(), DenseSpec().Hash());

  // Custom payloads hash through their bytes.
  AttentionSpec c1, c2;
  CustomParams custom;
  custom.mechanism_id = 0x1001;
  c1.params = custom;
  c2.params = custom;
  EXPECT_EQ(c1.Hash(), c2.Hash());
  custom.blob[7] = std::byte{1};
  c2.params = custom;
  EXPECT_NE(c1.Hash(), c2.Hash());
}

TEST(AttentionSpecTest, FeatureSetBuildsAndQueries) {
  const FeatureSet none;
  const FeatureSet caps = none.With(Feature::kSoftCap).With(Feature::kReturnLse);
  EXPECT_FALSE(none.Has(Feature::kSoftCap));
  EXPECT_TRUE(caps.Has(Feature::kSoftCap));
  EXPECT_TRUE(caps.Has(Feature::kReturnLse));
  EXPECT_FALSE(caps.Has(Feature::kSinks));
  EXPECT_EQ(none, FeatureSet{});
  EXPECT_NE(none, caps);
}

TEST(TensorViewTest, ViewOfFlattensContiguousTensors) {
  float storage[6] = {};
  auto tensor = Tensor::FromBlob(storage, DataType::kFloat32, Shape({2, 3}), Device::Cpu());
  ASSERT_TRUE(tensor.ok()) << tensor.status();
  const TensorView view = ViewOf(*tensor);
  EXPECT_TRUE(view.Defined());
  EXPECT_EQ(view.data, static_cast<void*>(storage));
  EXPECT_EQ(view.dtype, DataType::kFloat32);
  EXPECT_EQ(view.rank, 2);
  EXPECT_EQ(view.Dim(0), 2);
  EXPECT_EQ(view.Dim(1), 3);
  // Contiguous row-major: stride(i) is the product of the trailing dims.
  EXPECT_EQ(view.Stride(0), 3);
  EXPECT_EQ(view.Stride(1), 1);
}

}  // namespace
}  // namespace inferx::attn
