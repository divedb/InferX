#include "inferx/engine/input_batch.h"

#include "gtest/gtest.h"

namespace inferx {
namespace {
InputBatch::NewRequest RequestRow(RequestId id) {
  InputBatch::NewRequest row;
  row.id = id;
  row.prompt_tokens = {static_cast<TokenId>(id), 10};
  row.num_computed = 1;
  row.params.temperature = 0.7f;
  return row;
}

TEST(InputBatchTest, CompactionPreservesHistoryProgressAndSamplingState) {
  InputBatch batch(4, 8);
  for (RequestId id = 1; id <= 4; ++id) ASSERT_TRUE(batch.Add(RequestRow(id)).ok());
  ASSERT_TRUE(batch.AppendToken(4, 99).ok());
  ASSERT_TRUE(batch.SetNumComputed(4, 2).ok());
  batch.AdvanceRngOffset(*batch.RowOf(4), 3);
  batch.Remove(1);
  batch.Remove(3);
  batch.Condense();
  EXPECT_EQ(batch.NumReqs(), 2);
  EXPECT_EQ(batch.NumRows(), 2);
  const int row = *batch.RowOf(4);
  EXPECT_EQ(row, 0);
  EXPECT_EQ(batch.TokensAt(row)[2], 99);
  EXPECT_EQ(batch.NumComputed(row), 2);
  EXPECT_EQ(batch.NumPromptTokens(row), 2);
  EXPECT_EQ(batch.RngOffset(row), 3);
  EXPECT_FLOAT_EQ(batch.ParamsAt(row).temperature, 0.7f);
  ASSERT_TRUE(batch.Add(RequestRow(5)).ok());
  EXPECT_EQ(*batch.RowOf(5), 2);
}

TEST(InputBatchTest, ReusedRowResetsStateAndEnforcesBounds) {
  InputBatch batch(1, 3);
  ASSERT_TRUE(batch.Add(RequestRow(1)).ok());
  EXPECT_FALSE(batch.Add(RequestRow(1)).ok());
  EXPECT_FALSE(batch.Add(RequestRow(2)).ok());
  EXPECT_FALSE(batch.SetNumComputed(1, -1).ok());
  EXPECT_FALSE(batch.SetNumComputed(1, 3).ok());
  ASSERT_TRUE(batch.AppendToken(1, 3).ok());
  EXPECT_FALSE(batch.AppendToken(1, 4).ok());
  batch.AdvanceRngOffset(0, 5);
  batch.Remove(1);
  batch.Remove(1);
  ASSERT_TRUE(batch.Add(RequestRow(2)).ok());
  EXPECT_EQ(batch.NumTokens(0), 2);
  EXPECT_EQ(batch.RngOffset(0), 0);
  EXPECT_EQ(batch.TokensAt(0)[0], 2);
  EXPECT_FALSE(batch.AppendToken(1, 4).ok());
  batch.Remove(2);
  batch.Condense();
  EXPECT_EQ(batch.NumRows(), 0);
  auto invalid = RequestRow(3);
  invalid.num_computed = 3;
  EXPECT_FALSE(batch.Add(invalid).ok());
  EXPECT_EQ(batch.NumReqs(), 0);
}
}  // namespace
}  // namespace inferx
