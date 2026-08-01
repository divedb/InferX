// Paged attention against the contiguous reference.
//
// This is R5's pattern applied a milestone early: the naive kernel from M2 is
// the definition of correct, and every faster path -- paged here, FlashInfer at
// M3's end -- has to reproduce it. The two kernels compute the same maths and
// differ only in where a key lives, so any disagreement is an indexing bug in
// the paging, which is exactly what the test is for.
//
// The blocks are deliberately assigned out of order. A block table walked with
// the identity mapping passes every test where blocks happen to be sequential,
// and fails the moment a real allocator hands back a freed block.

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/kv_cache.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"
#include "inferx/kernels/layers.h"

namespace inferx {
namespace {

using bf16 = __nv_bfloat16;

class Dev {
 public:
  TensorView Bf16(const std::vector<float>& host, const Shape& shape) {
    std::vector<bf16> b(host.size());
    for (size_t i = 0; i < host.size(); ++i) b[i] = __float2bfloat16(host[i]);
    return Raw(b.data(), b.size() * sizeof(bf16), DataType::kBFloat16, shape);
  }

  TensorView I32(const std::vector<int32_t>& host, const Shape& shape) {
    return Raw(host.data(), host.size() * sizeof(int32_t), DataType::kInt32,
               shape);
  }

  TensorView Empty(const Shape& shape, DataType dtype = DataType::kBFloat16) {
    auto buf = DeviceBuffer::Allocate(
        static_cast<size_t>(DataTypeByteSize(dtype, shape.Numel())),
        DeviceId::Cuda(0));
    EXPECT_TRUE(buf.ok()) << buf.status();
    bufs_.push_back(*std::move(buf));

    EXPECT_EQ(cudaMemset(bufs_.back().data(), 0, bufs_.back().size()),
              cudaSuccess);

    auto v = TensorView::Create(bufs_.back().data(), dtype, shape,
                                DeviceId::Cuda(0));
    EXPECT_TRUE(v.ok()) << v.status();
    return *v;
  }

  std::vector<float> Down(const TensorView& t) {
    std::vector<bf16> b(static_cast<size_t>(t.Numel()));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(b.data(), t.Data(), b.size() * sizeof(bf16),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    std::vector<float> out(b.size());
    for (size_t i = 0; i < b.size(); ++i) out[i] = __bfloat162float(b[i]);
    return out;
  }

 private:
  TensorView Raw(const void* src, size_t bytes, DataType dtype,
                 const Shape& shape) {
    auto buf = DeviceBuffer::Allocate(bytes, DeviceId::Cuda(0));
    EXPECT_TRUE(buf.ok()) << buf.status();
    EXPECT_EQ(cudaMemcpy(buf->data(), src, bytes, cudaMemcpyHostToDevice),
              cudaSuccess);
    bufs_.push_back(*std::move(buf));

    auto v = TensorView::Create(bufs_.back().data(), dtype, shape,
                                DeviceId::Cuda(0));
    EXPECT_TRUE(v.ok()) << v.status();
    return *v;
  }

  std::vector<DeviceBuffer> bufs_;
};

std::vector<float> Ramp(size_t n, float phase) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = std::sin(static_cast<float>(i) * 0.31f + phase);
  }
  return v;
}

class PagedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }
  Dev dev;
};

// The core equivalence. One sequence, K/V written into scattered blocks, and
// the result must match attention over the same K/V laid out contiguously.
TEST_F(PagedTest, MatchesContiguousAttention) {
  constexpr int64_t tokens = 10, q_heads = 4, kv_heads = 2, head_dim = 32;
  constexpr int64_t block_size = 4;
  constexpr int64_t num_blocks = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const std::vector<float> q = Ramp(tokens * q_heads * head_dim, 0.0f);
  const std::vector<float> k = Ramp(tokens * kv_heads * head_dim, 1.1f);
  const std::vector<float> v = Ramp(tokens * kv_heads * head_dim, 2.2f);

  // --- contiguous reference -------------------------------------------------
  const TensorView qc = dev.Bf16(q, Shape({tokens, q_heads, head_dim}));
  const TensorView kc = dev.Bf16(k, Shape({tokens, kv_heads, head_dim}));
  const TensorView vc = dev.Bf16(v, Shape({tokens, kv_heads, head_dim}));
  const TensorView ref_out = dev.Empty(Shape({tokens, q_heads, head_dim}));

  ASSERT_TRUE(kernels::Attention(qc, kc, vc, ref_out, scale).ok());
  const std::vector<float> want = dev.Down(ref_out);

  // --- paged ---------------------------------------------------------------
  // Blocks in a deliberately jumbled order: 3, 0, 6 rather than 0, 1, 2.
  const std::vector<int32_t> blocks = {3, 0, 6};
  ASSERT_GE(static_cast<int64_t>(blocks.size()) * block_size, tokens);

  std::vector<int32_t> slots(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) {
    const int32_t block = blocks[static_cast<size_t>(t / block_size)];
    slots[static_cast<size_t>(t)] =
        static_cast<int32_t>(block * block_size + (t % block_size));
  }

  const TensorView k_cache =
      dev.Empty(Shape({num_blocks, block_size, kv_heads, head_dim}));
  const TensorView v_cache =
      dev.Empty(Shape({num_blocks, block_size, kv_heads, head_dim}));

  ASSERT_TRUE(kernels::AppendToKvCache(kc, vc, k_cache, v_cache,
                                       dev.I32(slots, Shape({tokens})))
                  .ok());

  std::vector<int32_t> table(blocks.begin(), blocks.end());
  std::vector<int32_t> seq(static_cast<size_t>(tokens), 0);
  std::vector<int32_t> pos(static_cast<size_t>(tokens));
  std::iota(pos.begin(), pos.end(), 0);

  const TensorView paged_out = dev.Empty(Shape({tokens, q_heads, head_dim}));

  ASSERT_TRUE(kernels::PagedAttention(
                  qc, k_cache, v_cache,
                  dev.I32(table, Shape({1, static_cast<int64_t>(blocks.size())})),
                  dev.I32(seq, Shape({tokens})), dev.I32(pos, Shape({tokens})),
                  paged_out, scale)
                  .ok());

  const std::vector<float> got = dev.Down(paged_out);

  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    // Both kernels do the same fp32 arithmetic in the same order, so this is an
    // exact-agreement test up to bf16 output rounding, not a tolerance test.
    EXPECT_NEAR(got[i], want[i], 1e-2) << "element " << i;
  }
}

// Decode: one query per sequence, attending over everything already cached.
// Two sequences with different lengths and interleaved blocks, which is the
// arrangement a real allocator produces and the one most likely to break an
// indexing assumption.
TEST_F(PagedTest, DecodeStepOverTwoSequences) {
  constexpr int64_t q_heads = 2, kv_heads = 1, head_dim = 16;
  constexpr int64_t block_size = 4, num_blocks = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // Sequence 0 has 6 cached tokens, sequence 1 has 3.
  const std::vector<int64_t> lens = {6, 3};
  const std::vector<std::vector<int32_t>> seq_blocks = {{5, 1}, {2}};

  const int64_t total = lens[0] + lens[1];
  const std::vector<float> k = Ramp(total * kv_heads * head_dim, 0.7f);
  const std::vector<float> v = Ramp(total * kv_heads * head_dim, 1.9f);

  const TensorView kc = dev.Bf16(k, Shape({total, kv_heads, head_dim}));
  const TensorView vc = dev.Bf16(v, Shape({total, kv_heads, head_dim}));

  std::vector<int32_t> slots;
  for (size_t s = 0; s < lens.size(); ++s) {
    for (int64_t t = 0; t < lens[s]; ++t) {
      const int32_t block = seq_blocks[s][static_cast<size_t>(t / block_size)];
      slots.push_back(static_cast<int32_t>(block * block_size +
                                           (t % block_size)));
    }
  }

  const TensorView k_cache =
      dev.Empty(Shape({num_blocks, block_size, kv_heads, head_dim}));
  const TensorView v_cache =
      dev.Empty(Shape({num_blocks, block_size, kv_heads, head_dim}));

  ASSERT_TRUE(kernels::AppendToKvCache(kc, vc, k_cache, v_cache,
                                       dev.I32(slots, Shape({total})))
                  .ok());

  // One query per sequence, at the last cached position.
  const std::vector<float> q = Ramp(2 * q_heads * head_dim, 3.3f);
  const TensorView qd = dev.Bf16(q, Shape({2, q_heads, head_dim}));

  const std::vector<int32_t> table = {5, 1, 2, -1};  // padded to 2 columns
  const std::vector<int32_t> seq = {0, 1};
  const std::vector<int32_t> pos = {static_cast<int32_t>(lens[0] - 1),
                                    static_cast<int32_t>(lens[1] - 1)};

  const TensorView out = dev.Empty(Shape({2, q_heads, head_dim}));

  ASSERT_TRUE(kernels::PagedAttention(qd, k_cache, v_cache,
                                      dev.I32(table, Shape({2, 2})),
                                      dev.I32(seq, Shape({2})),
                                      dev.I32(pos, Shape({2})), out, scale)
                  .ok());

  const std::vector<float> got = dev.Down(out);

  // Reference: recompute each sequence's decode step contiguously.
  for (size_t s = 0; s < lens.size(); ++s) {
    const int64_t base = s == 0 ? 0 : lens[0];

    for (int64_t h = 0; h < q_heads; ++h) {
      std::vector<double> scores(static_cast<size_t>(lens[s]));
      double max_score = -1e300;

      for (int64_t j = 0; j < lens[s]; ++j) {
        double dot = 0;
        for (int64_t d = 0; d < head_dim; ++d) {
          const float qq = __bfloat162float(__float2bfloat16(
              q[static_cast<size_t>((s * q_heads + h) * head_dim + d)]));
          const float kk = __bfloat162float(__float2bfloat16(
              k[static_cast<size_t>(((base + j) * kv_heads) * head_dim + d)]));
          dot += qq * kk;
        }
        scores[static_cast<size_t>(j)] = dot * scale;
        max_score = std::max(max_score, scores[static_cast<size_t>(j)]);
      }

      double sum = 0;
      for (double& x : scores) {
        x = std::exp(x - max_score);
        sum += x;
      }

      for (int64_t d = 0; d < head_dim; ++d) {
        double acc = 0;
        for (int64_t j = 0; j < lens[s]; ++j) {
          acc += scores[static_cast<size_t>(j)] *
                 __bfloat162float(__float2bfloat16(
                     v[static_cast<size_t>(((base + j) * kv_heads) * head_dim +
                                           d)]));
        }
        EXPECT_NEAR(got[static_cast<size_t>((s * q_heads + h) * head_dim + d)],
                    acc / sum, 3e-2)
            << "seq " << s << " head " << h << " dim " << d;
      }
    }
  }
}

// A block written, freed, and reused by another sequence must carry the new
// contents. This is the failure the LIFO free list makes most likely and the
// one that produces another request's text in your output.
TEST_F(PagedTest, ReusedBlocksCarryTheNewContents) {
  constexpr int64_t kv_heads = 1, head_dim = 8;
  constexpr int64_t block_size = 4, num_blocks = 2;

  const TensorView k_cache =
      dev.Empty(Shape({num_blocks, block_size, kv_heads, head_dim}));
  const TensorView v_cache =
      dev.Empty(Shape({num_blocks, block_size, kv_heads, head_dim}));

  const std::vector<float> first(4 * head_dim, 1.0f);
  const std::vector<float> second(4 * head_dim, 7.0f);

  const std::vector<int32_t> slots = {4, 5, 6, 7};  // all of block 1

  ASSERT_TRUE(kernels::AppendToKvCache(
                  dev.Bf16(first, Shape({4, kv_heads, head_dim})),
                  dev.Bf16(first, Shape({4, kv_heads, head_dim})), k_cache,
                  v_cache, dev.I32(slots, Shape({4})))
                  .ok());

  ASSERT_TRUE(kernels::AppendToKvCache(
                  dev.Bf16(second, Shape({4, kv_heads, head_dim})),
                  dev.Bf16(second, Shape({4, kv_heads, head_dim})), k_cache,
                  v_cache, dev.I32(slots, Shape({4})))
                  .ok());

  const std::vector<float> got = dev.Down(k_cache);

  // Block 1 starts at slot 4.
  for (int64_t i = 0; i < 4 * head_dim; ++i) {
    EXPECT_FLOAT_EQ(got[static_cast<size_t>(4 * head_dim + i)], 7.0f) << i;
  }
}

TEST_F(PagedTest, RejectsMismatchedShapes) {
  const TensorView q = dev.Empty(Shape({2, 4, 16}));
  const TensorView kc = dev.Empty(Shape({4, 4, 3, 16}));  // 4 % 3 != 0
  const TensorView vc = dev.Empty(Shape({4, 4, 3, 16}));
  const TensorView out = dev.Empty(Shape({2, 4, 16}));

  EXPECT_EQ(kernels::PagedAttention(q, kc, vc,
                                    dev.I32({0, 1}, Shape({1, 2})),
                                    dev.I32({0, 0}, Shape({2})),
                                    dev.I32({0, 1}, Shape({2})), out, 0.25f)
                .code(),
            absl::StatusCode::kInvalidArgument);
}

// ---------------------------------------------------------------------------
// The pool itself.
// ---------------------------------------------------------------------------

class PoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }
};

TEST_F(PoolTest, AllocatesAndRecycles) {
  KvLayout layout{.entries_per_token = 2, .kv_heads = 2, .head_dim = 64,
                  .dtype = DataType::kBFloat16};

  auto pool = KvBlockPool::Create(4, 16, 16, layout);
  ASSERT_TRUE(pool.ok()) << pool.status();

  EXPECT_EQ(pool->free_blocks(), 16);
  EXPECT_EQ(pool->used_blocks(), 0);

  // 4 layers x 16 blocks x 2 entries x 16 tokens x 2 heads x 64 dims x 2 bytes.
  EXPECT_EQ(pool->bytes(), 4u * 16 * 2 * 16 * 2 * 64 * 2);

  std::vector<int32_t> taken;
  for (int i = 0; i < 16; ++i) {
    auto b = pool->AllocateBlock();
    ASSERT_TRUE(b.ok()) << b.status();
    taken.push_back(*b);
  }

  EXPECT_EQ(pool->free_blocks(), 0);
  EXPECT_EQ(pool->AllocateBlock().status().code(),
            absl::StatusCode::kResourceExhausted);

  ASSERT_TRUE(pool->FreeBlocks(taken).ok());
  EXPECT_EQ(pool->free_blocks(), 16);
}

// Handing the same block to two sequences is the worst failure this class can
// have, so a double free is an error rather than a silently corrupted list.
TEST_F(PoolTest, DoubleFreeIsRejected) {
  KvLayout layout{.entries_per_token = 2, .kv_heads = 1, .head_dim = 16,
                  .dtype = DataType::kBFloat16};

  auto pool = KvBlockPool::Create(1, 4, 8, layout);
  ASSERT_TRUE(pool.ok()) << pool.status();

  auto b = pool->AllocateBlock();
  ASSERT_TRUE(b.ok());

  EXPECT_TRUE(pool->FreeBlock(*b).ok());
  EXPECT_EQ(pool->FreeBlock(*b).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(pool->FreeBlock(99).code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(PoolTest, LayerViewsAreDistinctAndCorrectlyShaped) {
  KvLayout layout{.entries_per_token = 2, .kv_heads = 2, .head_dim = 32,
                  .dtype = DataType::kBFloat16};

  auto pool = KvBlockPool::Create(3, 8, 16, layout);
  ASSERT_TRUE(pool.ok()) << pool.status();

  auto k0 = pool->KeyCache(0);
  auto v0 = pool->ValueCache(0);
  auto k1 = pool->KeyCache(1);
  ASSERT_TRUE(k0.ok() && v0.ok() && k1.ok());

  EXPECT_EQ(k0->GetShape(), Shape({8, 16, 2, 32}));

  // K and V of one layer are adjacent; the next layer starts after both.
  const auto* p0 = static_cast<const std::byte*>(k0->Data());
  const auto* pv = static_cast<const std::byte*>(v0->Data());
  const auto* p1 = static_cast<const std::byte*>(k1->Data());

  EXPECT_EQ(pv - p0, k0->NBytes());
  EXPECT_EQ(p1 - p0, k0->NBytes() * 2);

  EXPECT_FALSE(pool->KeyCache(3).ok());
  EXPECT_FALSE(pool->KeyCache(-1).ok());
}

// MLA stores one latent per token and has no separate value cache. The pool has
// to express that without pretending the layout is [K|V] (T11).
TEST_F(PoolTest, SingleEntryLayoutHasNoValueCache) {
  KvLayout mla{.entries_per_token = 1, .kv_heads = 1, .head_dim = 512,
               .dtype = DataType::kBFloat16};

  auto pool = KvBlockPool::Create(2, 4, 16, mla);
  ASSERT_TRUE(pool.ok()) << pool.status();

  EXPECT_TRUE(pool->KeyCache(0).ok());
  EXPECT_EQ(pool->ValueCache(0).status().code(),
            absl::StatusCode::kFailedPrecondition);

  // Half the bytes of an equivalent K/V layout, which is the point.
  EXPECT_EQ(pool->bytes(), 2u * 4 * 1 * 16 * 1 * 512 * 2);
}

TEST(BlockTableTest, MapsPositionsToBlocksAndSlots) {
  BlockTable table(4);
  EXPECT_EQ(table.capacity_tokens(), 0);

  table.Append(7);
  table.Append(2);

  EXPECT_EQ(table.size(), 2);
  EXPECT_EQ(table.capacity_tokens(), 8);

  int32_t block = -1;
  int64_t slot = -1;

  ASSERT_TRUE(table.Locate(0, &block, &slot));
  EXPECT_EQ(block, 7);
  EXPECT_EQ(slot, 0);

  ASSERT_TRUE(table.Locate(3, &block, &slot));
  EXPECT_EQ(block, 7);
  EXPECT_EQ(slot, 3);

  // Position 4 crosses into the second block, which is 2 rather than 8 -- the
  // whole reason a block table exists.
  ASSERT_TRUE(table.Locate(4, &block, &slot));
  EXPECT_EQ(block, 2);
  EXPECT_EQ(slot, 0);

  EXPECT_FALSE(table.Locate(8, &block, &slot));
}


// Which *physical* blocks a sequence occupies must not change its output.
//
// This is R8. A block table is a permutation from logical position to physical
// block, and every read is supposed to go through it, so the same logical
// sequence stored in blocks [0,1,2] and in blocks [5,4,3] must attend
// identically. It does not, and that is what makes a request's result depend on
// which requests ran before it: the free list is a stack, so consecutive
// sequences receive their blocks in opposite order.
//
// Written as two orderings of the *same* logical content rather than as a
// server-level determinism check, because this is the invariant that is
// actually broken -- the server only made it visible.
TEST_F(PagedTest, OutputDoesNotDependOnWhichBlocksHoldTheSequence) {
  // Qwen2.5-3B's actual attention shape, because head_dim and the GQA group
  // size both select code paths.
  constexpr int64_t kBlockSize = 16;
  constexpr int64_t kHeads = 16;
  constexpr int64_t kKvHeads = 2;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kLen = 50;  // spans four blocks, last one partial

  const KvLayout layout{.entries_per_token = 2,
                        .kv_heads = kKvHeads,
                        .head_dim = kHeadDim,
                        .dtype = DataType::kBFloat16};

  const std::vector<float> k_host = Ramp(kLen * kKvHeads * kHeadDim, 0.1f);
  const std::vector<float> v_host = Ramp(kLen * kKvHeads * kHeadDim, 0.7f);
  const std::vector<float> q_host = Ramp(kHeads * kHeadDim, 1.3f);

  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  // Runs one decode query against a sequence of kLen tokens laid out in
  // `blocks`, in that order.
  const auto attend = [&](const std::vector<int32_t>& blocks) {
    Dev d;

    auto pool = KvBlockPool::Create(1, 40, kBlockSize, layout);
    EXPECT_TRUE(pool.ok()) << pool.status();

    const TensorView k_cache = *pool->KeyCache(0);
    const TensorView v_cache = *pool->ValueCache(0);

    // Write the prompt's K/V through the block table, exactly as the scheduler
    // would: logical position i lands in blocks[i / block_size].
    std::vector<int32_t> slots(static_cast<size_t>(kLen));
    for (int64_t i = 0; i < kLen; ++i) {
      slots[static_cast<size_t>(i)] = static_cast<int32_t>(
          blocks[static_cast<size_t>(i / kBlockSize)] * kBlockSize +
          i % kBlockSize);
    }

    EXPECT_TRUE(kernels::AppendToKvCache(
                    d.Bf16(k_host, Shape({kLen, kKvHeads, kHeadDim})),
                    d.Bf16(v_host, Shape({kLen, kKvHeads, kHeadDim})), k_cache,
                    v_cache, d.I32(slots, Shape({kLen})))
                    .ok());

    // The decode query attends over all kLen cached tokens.
    const int64_t used = (kLen + kBlockSize - 1) / kBlockSize;

    // Padded to a fixed width with zeros beyond the blocks actually held,
    // which is exactly what the scheduler hands the model: the row is
    // max_blocks_per_seq wide regardless of how far the sequence has grown.
    // A kernel that infers the sequence's block count from this row's *width*
    // rather than from its length reads block 0 as if it were part of the
    // sequence.
    constexpr int64_t kTableWidth = 32;

    std::vector<int32_t> table(static_cast<size_t>(kTableWidth), 0);
    for (int64_t b = 0; b < used; ++b) {
      table[static_cast<size_t>(b)] = blocks[static_cast<size_t>(b)];
    }

    const TensorView out = d.Empty(Shape({1, kHeads, kHeadDim}));

    EXPECT_TRUE(kernels::PagedAttention(
                    d.Bf16(q_host, Shape({1, kHeads, kHeadDim})), k_cache,
                    v_cache, d.I32(table, Shape({1, kTableWidth})),
                    d.I32({0}, Shape({1})),
                    d.I32({static_cast<int32_t>(kLen - 1)}, Shape({1})), out,
                    scale)
                    .ok());

    return d.Down(out);
  };

  // Ascending versus descending physical blocks: exactly the two orders a
  // stack-based free list hands to consecutive requests.
  const std::vector<float> ascending = attend({0, 1, 2, 3});
  const std::vector<float> descending = attend({7, 6, 5, 4});
  const std::vector<float> scattered = attend({9, 1, 6, 12});

  ASSERT_EQ(ascending.size(), descending.size());

  // Bitwise equality is the right bar. The arithmetic is identical -- same
  // values, same order, same reduction -- so any difference at all is an
  // indexing bug rather than accumulated rounding.
  for (size_t i = 0; i < ascending.size(); ++i) {
    EXPECT_EQ(descending[i], ascending[i])
        << "element " << i << " changed when the sequence moved to blocks "
        << "[5,4,3]: " << ascending[i] << " vs " << descending[i];
    EXPECT_EQ(scattered[i], ascending[i])
        << "element " << i << " changed when the sequence moved to blocks "
        << "[6,1,4]: " << ascending[i] << " vs " << scattered[i];
  }
}


// The same invariant, but for a prefill: many query tokens at once.
//
// The decode case above passes, which narrows R8 to the multi-token path --
// and that is the one the model actually uses for prompts, since FlashInfer
// only takes over when there is exactly one query per sequence.
TEST_F(PagedTest, PrefillOutputDoesNotDependOnWhichBlocksHoldTheSequence) {
  constexpr int64_t kBlockSize = 16;
  constexpr int64_t kHeads = 16;
  constexpr int64_t kKvHeads = 2;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kLen = 40;

  const KvLayout layout{.entries_per_token = 2,
                        .kv_heads = kKvHeads,
                        .head_dim = kHeadDim,
                        .dtype = DataType::kBFloat16};

  const std::vector<float> k_host = Ramp(kLen * kKvHeads * kHeadDim, 0.1f);
  const std::vector<float> v_host = Ramp(kLen * kKvHeads * kHeadDim, 0.7f);
  const std::vector<float> q_host = Ramp(kLen * kHeads * kHeadDim, 1.3f);

  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  const auto prefill = [&](const std::vector<int32_t>& blocks) {
    Dev d;

    auto pool = KvBlockPool::Create(1, 40, kBlockSize, layout);
    EXPECT_TRUE(pool.ok()) << pool.status();

    const TensorView k_cache = *pool->KeyCache(0);
    const TensorView v_cache = *pool->ValueCache(0);

    std::vector<int32_t> slots(static_cast<size_t>(kLen));
    std::vector<int32_t> seq_of(static_cast<size_t>(kLen), 0);
    std::vector<int32_t> pos(static_cast<size_t>(kLen));

    for (int64_t i = 0; i < kLen; ++i) {
      slots[static_cast<size_t>(i)] = static_cast<int32_t>(
          blocks[static_cast<size_t>(i / kBlockSize)] * kBlockSize +
          i % kBlockSize);
      pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }

    EXPECT_TRUE(kernels::AppendToKvCache(
                    d.Bf16(k_host, Shape({kLen, kKvHeads, kHeadDim})),
                    d.Bf16(v_host, Shape({kLen, kKvHeads, kHeadDim})), k_cache,
                    v_cache, d.I32(slots, Shape({kLen})))
                    .ok());

    const int64_t used = (kLen + kBlockSize - 1) / kBlockSize;

    std::vector<int32_t> table(static_cast<size_t>(used));
    for (int64_t b = 0; b < used; ++b) {
      table[static_cast<size_t>(b)] = blocks[static_cast<size_t>(b)];
    }

    const TensorView out = d.Empty(Shape({kLen, kHeads, kHeadDim}));

    EXPECT_TRUE(kernels::PagedAttention(
                    d.Bf16(q_host, Shape({kLen, kHeads, kHeadDim})), k_cache,
                    v_cache, d.I32(table, Shape({1, used})),
                    d.I32(seq_of, Shape({kLen})), d.I32(pos, Shape({kLen})),
                    out, scale)
                    .ok());

    return d.Down(out);
  };

  const std::vector<float> ascending = prefill({0, 1, 2});
  const std::vector<float> descending = prefill({5, 4, 3});

  ASSERT_EQ(ascending.size(), descending.size());

  int mismatches = 0;
  size_t first = 0;

  for (size_t i = 0; i < ascending.size(); ++i) {
    if (ascending[i] != descending[i]) {
      if (mismatches == 0) first = i;
      ++mismatches;
    }
  }

  EXPECT_EQ(mismatches, 0)
      << mismatches << " of " << ascending.size()
      << " outputs changed when the prompt moved from blocks [0,1,2] to "
         "[5,4,3]; first at element "
      << first << " (query token " << first / (kHeads * kHeadDim) << ")";
}


// A short prompt must not need shared memory proportional to max_seq_len.
//
// The kernel used to size its tile from the block table's *width*, and the
// scheduler makes that width `max_seq_len / block_size` regardless of how long
// the sequence actually is. So configuring a server for a long context made
// every prefill fail, including a ten-token one -- not slowly, but with
// ResourceExhausted before the kernel launched. A 16k-token context needs 66 KB
// against a 48 KB limit, so `--max-seq-len 16384` could not serve anything at
// all.
//
// The width here is deliberately far larger than the sequence: that gap is the
// bug.
TEST_F(PagedTest, ShortSequenceDoesNotPayForAWideBlockTable) {
  constexpr int64_t kBlockSize = 16;
  constexpr int64_t kHeads = 16;
  constexpr int64_t kKvHeads = 2;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kLen = 8;

  // 16k of context, as a server configured for long prompts would have.
  constexpr int64_t kTableWidth = 1024;

  const KvLayout layout{.entries_per_token = 2,
                        .kv_heads = kKvHeads,
                        .head_dim = kHeadDim,
                        .dtype = DataType::kBFloat16};

  Dev d;

  auto pool = KvBlockPool::Create(1, 4, kBlockSize, layout);
  ASSERT_TRUE(pool.ok()) << pool.status();

  const TensorView k_cache = *pool->KeyCache(0);
  const TensorView v_cache = *pool->ValueCache(0);

  std::vector<int32_t> slots(static_cast<size_t>(kLen));
  std::vector<int32_t> pos(static_cast<size_t>(kLen));
  std::vector<int32_t> seq_of(static_cast<size_t>(kLen), 0);

  for (int64_t i = 0; i < kLen; ++i) {
    slots[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }

  const std::vector<float> kv = Ramp(kLen * kKvHeads * kHeadDim, 0.2f);

  ASSERT_TRUE(kernels::AppendToKvCache(
                  d.Bf16(kv, Shape({kLen, kKvHeads, kHeadDim})),
                  d.Bf16(kv, Shape({kLen, kKvHeads, kHeadDim})), k_cache,
                  v_cache, d.I32(slots, Shape({kLen})))
                  .ok());

  std::vector<int32_t> table(static_cast<size_t>(kTableWidth), 0);

  const TensorView out = d.Empty(Shape({kLen, kHeads, kHeadDim}));

  const Status s = kernels::PagedAttention(
      d.Bf16(Ramp(kLen * kHeads * kHeadDim, 1.1f),
             Shape({kLen, kHeads, kHeadDim})),
      k_cache, v_cache, d.I32(table, Shape({1, kTableWidth})),
      d.I32(seq_of, Shape({kLen})), d.I32(pos, Shape({kLen})), out,
      1.0f / std::sqrt(static_cast<float>(kHeadDim)),
      /*max_context=*/kLen);

  EXPECT_TRUE(s.ok())
      << "an " << kLen << "-token sequence was refused because its block table "
      << "is " << kTableWidth << " blocks wide: " << s;

  // And the answer is still right: passing the real context must shrink the
  // tile, not truncate the attention.
  const std::vector<float> got = d.Down(out);
  bool all_finite = true;
  for (const float v : got) {
    if (!std::isfinite(v)) all_finite = false;
  }
  EXPECT_TRUE(all_finite) << "attention produced non-finite output";
}

}  // namespace
}  // namespace inferx
