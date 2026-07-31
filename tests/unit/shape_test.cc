#include "inferx/core/shape.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/types/span.h"

namespace inferx {
namespace {

TEST(ShapeTest, DefaultConstructIsScalar) {
  Shape s;
  EXPECT_EQ(s.Rank(), 0);
  EXPECT_TRUE(s.IsScalar());
  EXPECT_TRUE(s.Dims().empty());
}

TEST(ShapeTest, InitializerListConstruct) {
  Shape s{2, 3, 4};
  EXPECT_EQ(s.Rank(), 3);
  EXPECT_FALSE(s.IsScalar());
  EXPECT_EQ(s.Dims().size(), 3u);
}

TEST(ShapeTest, SpanConstructCopiesDims) {
  const std::vector<int64_t> v = {5, 6, 7, 8};
  Shape s(absl::MakeConstSpan(v));
  EXPECT_EQ(s.Rank(), 4);
  EXPECT_EQ(s.Dims().size(), 4u);
  EXPECT_EQ(s.Dim(0), 5);
  EXPECT_EQ(s.Dim(-1), 8);
}

TEST(ShapeTest, SpanConstructorIsExplicit) {
  static_assert(!std::is_convertible_v<absl::Span<const int64_t>, Shape>);
  SUCCEED();
}

TEST(ShapeTest, DimReadsPositiveIndices) {
  Shape s{2, 3, 4};
  EXPECT_EQ(s.Dim(0), 2);
  EXPECT_EQ(s.Dim(1), 3);
  EXPECT_EQ(s.Dim(2), 4);
}

TEST(ShapeTest, DimNegativeIndicesWrapFromTheBack) {
  Shape s{2, 3, 4};
  EXPECT_EQ(s.Dim(-1), 4);  // innermost
  EXPECT_EQ(s.Dim(-2), 3);
  EXPECT_EQ(s.Dim(-3), 2);  // outermost
}

TEST(ShapeTest, SetDimWritesPositiveAndNegativeIndices) {
  Shape s{2, 3, 4};
  s.SetDim(1, 30);
  EXPECT_EQ(s.Dim(1), 30);
  s.SetDim(-1, 40);  // innermost
  EXPECT_EQ(s.Dim(2), 40);
  s.SetDim(-3, 20);  // outermost
  EXPECT_EQ(s.Dim(0), 20);
}

TEST(ShapeTest, PushBackAppendsToScalar) {
  Shape s;
  s.PushBack(2);
  s.PushBack(3);
  ASSERT_EQ(s.Rank(), 2);
  EXPECT_EQ(s.Dim(0), 2);
  EXPECT_EQ(s.Dim(1), 3);
}

TEST(ShapeTest, ClearEmpties) {
  Shape s{1, 2, 3};
  ASSERT_GT(s.Rank(), 0);
  s.Clear();
  EXPECT_EQ(s.Rank(), 0);
  EXPECT_TRUE(s.IsScalar());
}

TEST(ShapeTest, NumelIsProductOfDims) {
  EXPECT_EQ(Shape{}.Numel(), 1);  // scalar: empty product is 1
  EXPECT_EQ((Shape{1}).Numel(), 1);
  EXPECT_EQ((Shape{2, 3, 4}).Numel(), 24);
}

TEST(ShapeTest, NumelWithZeroExtentIsZero) {
  EXPECT_EQ((Shape{2, 0, 3}).Numel(), 0);
}

TEST(ShapeTest, InnerNumelIsProductFromDimensionToEnd) {
  Shape s{2, 3, 4, 5};
  EXPECT_EQ(s.InnerNumel(0), 120);  // 2*3*4*5
  EXPECT_EQ(s.InnerNumel(1), 60);   //   3*4*5
  EXPECT_EQ(s.InnerNumel(2), 20);   //     4*5
  EXPECT_EQ(s.InnerNumel(3), 5);    //       5
}

TEST(ShapeTest, EqualityComparesDims) {
  EXPECT_EQ((Shape{2, 3}), (Shape{2, 3}));
  EXPECT_NE((Shape{2, 3}), (Shape{3, 2}));     // same extents, different order
  EXPECT_NE((Shape{2, 3}), (Shape{2, 3, 1}));  // different Rank
  EXPECT_NE((Shape{2, 3}), Shape{});           // Rank 2 vs scalar
  EXPECT_EQ(Shape{}, Shape{});                 // two scalars equal
}

TEST(ShapeTest, EqualShapesHashEqual) {
  const absl::Hash<Shape> h;
  EXPECT_EQ(h(Shape{2, 3, 4}), h(Shape{2, 3, 4}));
  EXPECT_NE(h(Shape{2, 3, 4}), h(Shape{4, 3, 2}));
}

TEST(ShapeTest, UsableInHashSet) {
  absl::flat_hash_set<Shape> set;
  set.insert(Shape{2, 3});
  set.insert(Shape{2, 3});  // duplicate, ignored
  set.insert(Shape{4, 5});
  EXPECT_EQ(set.size(), 2u);
  EXPECT_TRUE(set.contains(Shape{2, 3}));
}

TEST(ShapeTest, IterationYieldsDims) {
  Shape s{2, 3, 4};
  const std::vector<int64_t> got(s.begin(), s.end());
  EXPECT_EQ(got, (std::vector<int64_t>{2, 3, 4}));
}

TEST(ShapeTest, ToString) {
  EXPECT_EQ(Shape{}.ToString(), "[]");
  EXPECT_EQ((Shape{7}).ToString(), "[7]");
  EXPECT_EQ((Shape{2, 3, 4}).ToString(), "[2, 3, 4]");
}

// A Rank above kDefaultRank (8) spills to heap; it must still behave correctly.
TEST(ShapeTest, SupportsRankBeyondInlineCapacity) {
  Shape s{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_EQ(s.Rank(), 10);
  EXPECT_EQ(s.Numel(), 3628800);  // 10!
  EXPECT_EQ(s.Dim(0), 1);
  EXPECT_EQ(s.Dim(-1), 10);
}

#if !defined(NDEBUG)
// Preconditions are enforced with assert(), which only fires in debug builds.
// In a release/NDEBUG build assert is compiled out, so these cases are
// excluded rather than silently passing.
TEST(ShapeDeathTest, DimAssertsOutOfRange) {
  Shape s{2, 3};
  EXPECT_DEATH(
      (void)s.Dim(2),
      "Dimension index must be in the range \\[-Rank\\(\\), Rank\\(\\)\\)");
  EXPECT_DEATH(
      (void)s.Dim(99),
      "Dimension index must be in the range \\[-Rank\\(\\), Rank\\(\\)\\)");
  // just past the back
  EXPECT_DEATH(
      (void)s.Dim(-3),
      "Dimension index must be in the range \\[-Rank\\(\\), Rank\\(\\)\\)");
}

TEST(ShapeDeathTest, ScalarIndexingAsserts) {
  Shape scalar;  // Rank 0: no index is valid
  EXPECT_DEATH(
      (void)scalar.Dim(0),
      "Dimension index must be in the range \\[-Rank\\(\\), Rank\\(\\)\\)");
  EXPECT_DEATH(
      (void)scalar.Dim(-1),
      "Dimension index must be in the range \\[-Rank\\(\\), Rank\\(\\)\\)");
}

TEST(ShapeDeathTest, SetDimAssertsOutOfRange) {
  Shape s{2, 3};
  EXPECT_DEATH(
      (s.SetDim(5, 0)),
      "Dimension index must be in the range \\[-Rank\\(\\), Rank\\(\\)\\)");
  EXPECT_DEATH(
      (s.SetDim(-3, 0)),
      "Dimension index must be in the range \\[-Rank\\(\\), Rank\\(\\)\\)");
}

TEST(ShapeDeathTest, PushBackRequiresPositiveExtent) {
  Shape s;
  EXPECT_DEATH((s.PushBack(0)), "Dimension extent must be positive");
  EXPECT_DEATH((s.PushBack(-1)), "Dimension extent must be positive");
}

TEST(ShapeDeathTest, InnerNumelAssertsOutOfRange) {
  Shape s{2, 3, 4};
  // negative not accepted
  EXPECT_DEATH((void)s.InnerNumel(-1),
               "Dimension index must be in the range \\[0, Rank\\(\\)\\)");
  EXPECT_DEATH((void)s.InnerNumel(3),
               "Dimension index must be in the range \\[0, Rank\\(\\)\\)");
}
#endif  // !defined(NDEBUG)

}  // namespace
}  // namespace inferx
