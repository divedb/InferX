// Tests for the non-owning POD view. Shape itself is covered in shape_test.cc;
// what is tested here is the layering between Shape, TensorSpec and TensorView.

#include "inferx/core/tensor_view.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"
#include "inferx/core/tensor_spec.h"

namespace inferx {
namespace {

// True when a Shape's extents live inside the object itself rather than in a
// heap block. Checking the data pointer against the object's own extent is
// exact and needs no allocator instrumentation.
bool DimsAreInline(const Shape& s) {
  const auto* base = reinterpret_cast<const std::byte*>(&s);
  const auto* data = reinterpret_cast<const std::byte*>(s.Dims().data());

  return data >= base && data < base + sizeof(Shape);
}

std::vector<std::byte>& Storage() {
  static std::vector<std::byte> buf(1 << 20);
  return buf;
}

void* Buf() { return Storage().data(); }

// ---------------------------------------------------------------------------
// Where the rank limit lives
// ---------------------------------------------------------------------------

// The rank limit belongs to TensorView, whose extents must be a POD array, not
// to Shape or TensorSpec. Both happily represent the shape; the view rejects
// it.
TEST(TensorView, RankLimitIsEnforcedByTheViewLayer) {
  Shape s{1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_GT(s.Rank(), TensorView::kMaxRank);  // Shape itself has no rank cap

  // TensorSpec checks dtype/shape invariants but has no rank limit either.
  EXPECT_TRUE(TensorSpec(DataType::kFloat, s).Verify().ok());

  const Status v =
      TensorView::Create(Buf(), DataType::kFloat, s, DeviceId::Cpu()).status();
  EXPECT_EQ(v.code(), absl::StatusCode::kInvalidArgument);
  // The message must name the real rank, not a truncated one.
  EXPECT_NE(v.message().find("9"), std::string::npos);
}

// The header static_asserts kMaxRank <= Shape::kDefaultRank, so the comparison
// itself needs no test. What a compile-time check cannot cover is the property
// that inequality exists for: a shape at the rank limit really does keep its
// extents inside the object, so GetShape() on the step path never reaches the
// allocator.
TEST(TensorView, MaxRankShapesAreAllocationFreeOnTheHost) {
  const int64_t dims[] = {2, 3, 4, 5, 6, 7, 8, 9};
  ASSERT_GE(std::size(dims), static_cast<size_t>(TensorView::kMaxRank));

  Shape s{absl::MakeConstSpan(dims, TensorView::kMaxRank)};
  ASSERT_EQ(s.Rank(), TensorView::kMaxRank);
  EXPECT_TRUE(DimsAreInline(s)) << "a shape at kMaxRank spilled to the heap";

  auto t = TensorView::Create(Buf(), DataType::kFloat, s, DeviceId::Cpu());
  ASSERT_TRUE(t.ok()) << t.status();
  EXPECT_EQ(t->Rank(), TensorView::kMaxRank);
  EXPECT_EQ(t->GetShape(), s);  // every extent survives the POD round trip
  EXPECT_TRUE(DimsAreInline(t->GetShape()));
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(TensorView, DefaultIsUndefined) {
  TensorView t;
  EXPECT_FALSE(t.IsDefined());
  EXPECT_EQ(t.GetDataType(), DataType::kUndefined);
  EXPECT_EQ(t.Rank(), 0);
  EXPECT_EQ(t.ToString(), "TensorView(<undefined>)");
}

TEST(TensorView, Create) {
  auto t = TensorView::Create(Buf(), DataType::kBFloat16, Shape{4, 128},
                              DeviceId::Cpu());
  ASSERT_TRUE(t.ok()) << t.status();
  EXPECT_TRUE(t->IsDefined());
  EXPECT_EQ(t->Numel(), 512);
  EXPECT_EQ(t->NBytes(), 1024);
  EXPECT_EQ(t->Rank(), 2);
  EXPECT_TRUE(t->IsCpu());
  EXPECT_FALSE(t->IsCuda());
}

TEST(TensorView, CreateRejectsInvalidInput) {
  EXPECT_EQ(
      TensorView::Create(Buf(), DataType::kUndefined, Shape{4}, DeviceId::Cpu())
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);

  EXPECT_EQ(
      TensorView::Create(Buf(), DataType::kFloat, Shape{4, -1}, DeviceId::Cpu())
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);

  EXPECT_EQ(
      TensorView::Create(Buf(), DataType::kFloat,
                         Shape{1, 2, 3, 4, 5, 6, 7, 8, 9}, DeviceId::Cpu())
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);

  EXPECT_EQ(
      TensorView::Create(nullptr, DataType::kFloat, Shape{4}, DeviceId::Cpu())
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
}

// A null pointer is legitimate when there is nothing to point at.
TEST(TensorView, CreateAllowsNullForEmpty) {
  auto t = TensorView::Create(nullptr, DataType::kFloat, Shape{0, 8},
                              DeviceId::Cpu());
  ASSERT_TRUE(t.ok()) << t.status();
  EXPECT_TRUE(t->IsEmpty());
  EXPECT_EQ(t->NBytes(), 0);
}

TEST(TensorView, SubByteRequiresEvenInnermostExtent) {
  auto ok =
      TensorView::Create(Buf(), DataType::kInt4, Shape{8, 64}, DeviceId::Cpu());
  ASSERT_TRUE(ok.ok()) << ok.status();
  EXPECT_EQ(ok->NBytes(), 8 * 64 / 2);

  auto bad =
      TensorView::Create(Buf(), DataType::kInt4, Shape{8, 65}, DeviceId::Cpu());
  EXPECT_EQ(bad.status().code(), absl::StatusCode::kInvalidArgument);
}

// Bool is one logical bit but a whole byte of storage, so it must not be
// packed the way the quantized formats are.
TEST(TensorView, BoolIsByteAddressableNotPacked) {
  auto t =
      TensorView::Create(Buf(), DataType::kBool, Shape{3}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok()) << t.status();  // an odd extent is fine for bool
  EXPECT_EQ(t->NBytes(), 3);
}

// ---------------------------------------------------------------------------
// Access
// ---------------------------------------------------------------------------

TEST(TensorView, TypedAccessChecksDataType) {
  auto t =
      TensorView::Create(Buf(), DataType::kInt32, Shape{16}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok());
  EXPECT_NE(t->DataAs<int32_t>(), nullptr);
  EXPECT_EQ(t->DataAs<float>(), nullptr);
  EXPECT_EQ(t->DataAs<int64_t>(), nullptr);
}

TEST(TensorView, ExtentsReadBothWays) {
  auto t = TensorView::Create(Buf(), DataType::kFloat, Shape{2, 3, 4},
                              DeviceId::Cpu());
  ASSERT_TRUE(t.ok());

  EXPECT_EQ(t->GetShape(), (Shape{2, 3, 4}));
  EXPECT_EQ(t->Dims(), absl::MakeConstSpan(t->GetShape().Dims()));
  EXPECT_EQ(t->Dim(0), 2);
  EXPECT_EQ(t->Dim(-1), 4);

  // Out-of-range reads clamp rather than being undefined.
  EXPECT_EQ(t->Dim(99), 4);
  EXPECT_EQ(t->Dim(-99), 2);

  TensorView undefined;
  EXPECT_EQ(undefined.Dim(0), 0);
}

// ---------------------------------------------------------------------------
// Slice / Reshape / Bitcast
// ---------------------------------------------------------------------------

TEST(TensorView, SliceIsContiguousAndOffsetCorrectly) {
  auto t = TensorView::Create(Buf(), DataType::kFloat, Shape{10, 8},
                              DeviceId::Cpu());
  ASSERT_TRUE(t.ok());

  auto s = t->Slice(2, 5);
  ASSERT_TRUE(s.ok()) << s.status();
  EXPECT_EQ(s->GetShape(), (Shape{3, 8}));
  EXPECT_EQ(s->Bytes(), t->Bytes() + 2 * 8 * sizeof(float));
  EXPECT_EQ(s->NBytes(), 3 * 8 * 4);
}

TEST(TensorView, SliceEmptyRange) {
  auto t = TensorView::Create(Buf(), DataType::kFloat, Shape{10, 8},
                              DeviceId::Cpu());
  ASSERT_TRUE(t.ok());
  auto s = t->Slice(4, 4);
  ASSERT_TRUE(s.ok()) << s.status();
  EXPECT_EQ(s->Numel(), 0);
}

TEST(TensorView, SliceBoundsAreChecked) {
  auto t = TensorView::Create(Buf(), DataType::kFloat, Shape{10, 8},
                              DeviceId::Cpu());
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t->Slice(0, 11).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(t->Slice(-1, 4).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(t->Slice(6, 3).status().code(), absl::StatusCode::kOutOfRange);
}

TEST(TensorView, SliceRejectsRank0) {
  auto t =
      TensorView::Create(Buf(), DataType::kFloat, Shape{}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t->Slice(0, 0).status().code(), absl::StatusCode::kInvalidArgument);
}

// Packed int4 rows are addressable only when the row stride is a whole number
// of bytes. Shape{8, 64} qualifies; a rank-1 odd offset does not.
TEST(TensorView, SubByteSliceStaysByteAligned) {
  auto t =
      TensorView::Create(Buf(), DataType::kInt4, Shape{8, 64}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok());
  auto s = t->Slice(1, 3);
  ASSERT_TRUE(s.ok()) << s.status();
  EXPECT_EQ(s->Bytes(), t->Bytes() + 32);

  auto flat =
      TensorView::Create(Buf(), DataType::kInt4, Shape{16}, DeviceId::Cpu());
  ASSERT_TRUE(flat.ok());
  EXPECT_EQ(flat->Slice(1, 4).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(TensorView, Reshape) {
  auto t = TensorView::Create(Buf(), DataType::kFloat16, Shape{4, 8, 16},
                              DeviceId::Cpu());
  ASSERT_TRUE(t.ok());

  auto r = t->Reshape(Shape{32, 16});
  ASSERT_TRUE(r.ok()) << r.status();
  EXPECT_EQ(r->Numel(), t->Numel());
  EXPECT_EQ(r->Data(), t->Data());

  EXPECT_EQ(t->Reshape(Shape{4, 8}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(TensorView, BitcastMatchesTotalBits) {
  auto t =
      TensorView::Create(Buf(), DataType::kFloat, Shape{4, 8}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok());

  auto as_u8 = t->Bitcast(DataType::kUInt8);
  ASSERT_TRUE(as_u8.ok()) << as_u8.status();
  EXPECT_EQ(as_u8->Numel(), 128);
  EXPECT_EQ(as_u8->NBytes(), t->NBytes());

  // 3 bytes is 24 bits, which does not divide into 32-bit elements.
  auto odd =
      TensorView::Create(Buf(), DataType::kUInt8, Shape{3}, DeviceId::Cpu());
  ASSERT_TRUE(odd.ok());
  EXPECT_EQ(odd->Bitcast(DataType::kInt32).status().code(),
            absl::StatusCode::kInvalidArgument);
}

// The packed-weight case this exists for: view an int4 block as bytes to memcpy
// it, then view it back.
TEST(TensorView, BitcastPackedWeights) {
  auto w = TensorView::Create(Buf(), DataType::kInt4, Shape{128, 64},
                              DeviceId::Cpu());
  ASSERT_TRUE(w.ok());
  auto bytes = w->Bitcast(DataType::kUInt8);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  EXPECT_EQ(bytes->Numel(), 128 * 64 / 2);
  EXPECT_EQ(bytes->NBytes(), w->NBytes());
}

// ---------------------------------------------------------------------------
// The kernel-boundary contract
// ---------------------------------------------------------------------------

TEST(TensorView, IsSmallAndTriviallyCopyable) {
  static_assert(std::is_trivially_copyable_v<TensorView>);
  // Passed by value into kernel launches, so it must stay small. The bound is
  // deliberately tight: exceeding it means kMaxRank grew and the cost at the
  // launch boundary should be reconsidered rather than absorbed silently.
  EXPECT_LE(sizeof(TensorView), 96u);
}

TEST(Device, Identity) {
  EXPECT_EQ(DeviceId::Cpu().ToString(), "cpu");
  EXPECT_EQ(DeviceId::Cuda(3).ToString(), "cuda:3");
  EXPECT_EQ(DeviceId::Cuda(0), DeviceId::Cuda(0));
  EXPECT_NE(DeviceId::Cuda(0), DeviceId::Cuda(1));
  EXPECT_NE(DeviceId::Cuda(0), DeviceId::Cpu());
  EXPECT_EQ(DeviceId(), DeviceId::Cpu());
}

}  // namespace
}  // namespace inferx
