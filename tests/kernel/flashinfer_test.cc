// FlashInfer's decode kernel against our own.
//
// R5's whole point: FlashInfer is pinned to a commit and upgraded by manual
// merge, so the only thing standing between a bad merge and silently wrong
// output is a test that runs both kernels and compares. Ours is written to be
// obviously correct and theirs to be fast; agreement is what lets the fast one
// be trusted.
//
// The two do not compute identically -- FlashInfer splits long sequences across
// blocks and reduces afterwards, so its accumulation order differs from our
// single-block walk. The tolerance below is bf16 rounding over a different
// reduction order, which is the same budget every other numerical test in this
// tree is stated against.

#include "inferx/kernels/flashinfer_attention.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
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

  TensorView Empty(const Shape& shape) {
    auto buf = DeviceBuffer::Allocate(
        static_cast<size_t>(shape.Numel()) * sizeof(bf16), DeviceId::Cuda(0));
    EXPECT_TRUE(buf.ok()) << buf.status();
    bufs_.push_back(*std::move(buf));
    EXPECT_EQ(cudaMemset(bufs_.back().data(), 0, bufs_.back().size()),
              cudaSuccess);

    auto v = TensorView::Create(bufs_.back().data(), DataType::kBFloat16, shape,
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
    v[i] = 0.5f * std::sin(static_cast<float>(i) * 0.17f + phase);
  }
  return v;
}

class FlashInferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }

  /// Runs both kernels over the same cache and compares.
  ///
  /// `lens` gives each sequence's cached length; the blocks are assigned in a
  /// jumbled order so a table walked as the identity fails.
  void CompareForLengths(const std::vector<int64_t>& lens, int64_t q_heads,
                         int64_t kv_heads, int64_t page_size,
                         int64_t num_blocks) {
    constexpr int64_t head_dim = 128;  // the only instantiated size
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int64_t batch = static_cast<int64_t>(lens.size());

    Dev dev;

    // Assign blocks back to front, so sequence 0 gets high indices.
    std::vector<std::vector<int32_t>> seq_blocks(lens.size());
    int32_t next = static_cast<int32_t>(num_blocks) - 1;

    for (size_t s = 0; s < lens.size(); ++s) {
      const int64_t need = (lens[s] + page_size - 1) / page_size;
      for (int64_t b = 0; b < need; ++b) seq_blocks[s].push_back(next--);
      EXPECT_GE(next, -1);
    }

    // Fill the cache directly through the append kernel, one sequence at a
    // time, so both kernels read identical bytes.
    const TensorView k_cache =
        dev.Empty(Shape({num_blocks, page_size, kv_heads, head_dim}));
    const TensorView v_cache =
        dev.Empty(Shape({num_blocks, page_size, kv_heads, head_dim}));

    for (size_t s = 0; s < lens.size(); ++s) {
      const int64_t n = lens[s];
      const std::vector<float> k =
          Ramp(static_cast<size_t>(n * kv_heads * head_dim),
               1.0f + static_cast<float>(s));
      const std::vector<float> v =
          Ramp(static_cast<size_t>(n * kv_heads * head_dim),
               2.0f + static_cast<float>(s));

      std::vector<int32_t> slots;
      for (int64_t t = 0; t < n; ++t) {
        const int32_t block = seq_blocks[s][static_cast<size_t>(t / page_size)];
        slots.push_back(
            static_cast<int32_t>(block * page_size + (t % page_size)));
      }

      ASSERT_TRUE(kernels::AppendToKvCache(
                      dev.Bf16(k, Shape({n, kv_heads, head_dim})),
                      dev.Bf16(v, Shape({n, kv_heads, head_dim})), k_cache,
                      v_cache, dev.I32(slots, Shape({n})))
                      .ok());
    }

    // One query per sequence: the decode shape.
    const std::vector<float> q =
        Ramp(static_cast<size_t>(batch * q_heads * head_dim), 0.0f);
    const TensorView qd = dev.Bf16(q, Shape({batch, q_heads, head_dim}));

    // --- ours: dense block table -------------------------------------------
    int64_t max_blocks = 0;
    for (const auto& b : seq_blocks) {
      max_blocks = std::max<int64_t>(max_blocks, static_cast<int64_t>(b.size()));
    }

    std::vector<int32_t> dense(
        static_cast<size_t>(batch * max_blocks), 0);
    std::vector<int32_t> blocks_used;
    std::vector<int32_t> q_pos;
    std::vector<int32_t> seq_of_token;

    for (size_t s = 0; s < lens.size(); ++s) {
      for (size_t b = 0; b < seq_blocks[s].size(); ++b) {
        dense[s * static_cast<size_t>(max_blocks) + b] = seq_blocks[s][b];
      }
      blocks_used.push_back(static_cast<int32_t>(seq_blocks[s].size()));
      q_pos.push_back(static_cast<int32_t>(lens[s] - 1));
      seq_of_token.push_back(static_cast<int32_t>(s));
    }

    const TensorView ours = dev.Empty(Shape({batch, q_heads, head_dim}));

    ASSERT_TRUE(kernels::PagedAttention(
                    qd, k_cache, v_cache,
                    dev.I32(dense, Shape({batch, max_blocks})),
                    dev.I32(seq_of_token, Shape({batch})),
                    dev.I32(q_pos, Shape({batch})), ours, scale)
                    .ok());

    // --- FlashInfer: CSR block table ---------------------------------------
    std::vector<int32_t> indices, indptr;
    ASSERT_TRUE(kernels::BuildCsrBlockTable(dense, batch, max_blocks,
                                            blocks_used, &indices, &indptr)
                    .ok());

    // Tokens used in each sequence's last page. Never 0: a sequence whose
    // length is an exact multiple of the page size fills its final page.
    std::vector<int32_t> last_page;
    for (const int64_t n : lens) {
      const int64_t rem = n % page_size;
      last_page.push_back(static_cast<int32_t>(rem == 0 ? page_size : rem));
    }

    auto fi = kernels::FlashInferDecode::Create();
    ASSERT_TRUE(fi.ok()) << fi.status();

    const TensorView theirs = dev.Empty(Shape({batch, q_heads, head_dim}));

    const Status s = fi->Decode(
        qd, k_cache, v_cache,
        dev.I32(indices, Shape({static_cast<int64_t>(indices.size())})),
        dev.I32(indptr, Shape({batch + 1})), absl::MakeConstSpan(indptr),
        dev.I32(last_page, Shape({batch})), theirs, scale);
    ASSERT_TRUE(s.ok()) << s;

    // --- compare -----------------------------------------------------------
    const std::vector<float> a = dev.Down(ours);
    const std::vector<float> b = dev.Down(theirs);
    ASSERT_EQ(a.size(), b.size());

    double worst = 0;
    double sumsq = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      worst = std::max<double>(worst, std::abs(a[i] - b[i]));
      sumsq += a[i] * a[i];
    }

    const double rms = std::sqrt(sumsq / a.size());

    // Relative to the signal's own magnitude, which is what makes this bound
    // meaningful across the different sequence lengths below rather than tuned
    // to one of them.
    EXPECT_LT(worst, rms * 0.15)
        << "worst |difference| " << worst << " against rms " << rms
        << " -- FlashInfer and the reference kernel disagree by more than bf16 "
        << "and a different reduction order explain";
  }
};

// One sequence, several pages. The simplest case where paging matters at all.
TEST_F(FlashInferTest, AgreesWithTheReferenceOnASingleSequence) {
  CompareForLengths({40}, /*q_heads=*/16, /*kv_heads=*/2, /*page_size=*/16,
                    /*num_blocks=*/32);
}

// Ragged batch: different lengths, none a multiple of the page size, so every
// sequence has a partly-filled last page. That is what `last_page_len` is for
// and the most likely thing to be off by one.
TEST_F(FlashInferTest, AgreesOnARaggedBatch) {
  CompareForLengths({7, 33, 1, 60, 18}, 16, 2, 16, 64);
}

// Lengths that are exact multiples of the page size. The boundary case where
// `last_page_len` must be page_size rather than 0.
TEST_F(FlashInferTest, AgreesWhenLengthsFillTheirLastPageExactly) {
  CompareForLengths({16, 32, 48}, 16, 2, 16, 32);
}

// Long enough that FlashInfer splits the sequence across blocks and reduces --
// a different code path inside it, and the one our single-block kernel is least
// like.
TEST_F(FlashInferTest, AgreesOnASequenceLongEnoughToSplit) {
  CompareForLengths({1000}, 16, 2, 16, 128);
}

// MHA rather than GQA: group size 1 is a separate template instantiation.
TEST_F(FlashInferTest, AgreesWithoutGroupedQueryAttention) {
  CompareForLengths({25, 9}, /*q_heads=*/4, /*kv_heads=*/4, 16, 32);
}

// Qwen2.5-3B's own geometry: 16 q heads over 2 kv heads, group size 8.
TEST_F(FlashInferTest, AgreesAtQwenGeometry) {
  CompareForLengths({5, 128, 64}, 16, 2, 16, 64);
}

// The decode path must not synchronize. §5.2 is explicit that a device sync in
// the decode loop destroys the overlap pipeline, and an earlier version of this
// wrapper had one: it copied the KV indptr back from the device for
// FlashInfer's planner and waited on it. The fix was to take the host copy the
// scheduler already has.
//
// Asserted rather than assumed, because "I removed the sync" is exactly the
// kind of claim that rots.
//
// The first version of this test polled cudaStreamQuery after the call and was
// useless: a sync at the *start* of Decode still leaves that call's own work
// queued on return, so the stream is legitimately busy either way. It passed
// with the bug reintroduced, which is the only reason it was caught.
//
// What a sync actually costs is host time. So the measurement is: enqueue N
// launches and time the *host* loop, then drain and time the whole thing. An
// asynchronous path returns in launch overhead -- microseconds -- while the GPU
// works for milliseconds. A synchronizing one makes the two nearly equal,
// because the host waits for each launch before issuing the next.
TEST_F(FlashInferTest, DecodeDoesNotBlockTheHost) {
  constexpr int64_t head_dim = 128, q_heads = 16, kv_heads = 2;
  constexpr int64_t page_size = 16, blocks_per_seq = 128;
  constexpr int64_t batch = 128;
  constexpr int64_t num_blocks = batch * blocks_per_seq;
  constexpr int kLaunches = 10;

  Dev dev;

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  const TensorView k_cache =
      dev.Empty(Shape({num_blocks, page_size, kv_heads, head_dim}));
  const TensorView v_cache =
      dev.Empty(Shape({num_blocks, page_size, kv_heads, head_dim}));
  const TensorView q = dev.Empty(Shape({batch, q_heads, head_dim}));
  const TensorView out = dev.Empty(Shape({batch, q_heads, head_dim}));

  std::vector<int32_t> indices, indptr, last_page;
  for (int64_t s = 0; s < batch; ++s) {
    indptr.push_back(static_cast<int32_t>(indices.size()));
    for (int64_t b = 0; b < blocks_per_seq; ++b) {
      indices.push_back(static_cast<int32_t>(s * blocks_per_seq + b));
    }
    last_page.push_back(static_cast<int32_t>(page_size));
  }
  indptr.push_back(static_cast<int32_t>(indices.size()));

  const TensorView indices_v =
      dev.I32(indices, Shape({static_cast<int64_t>(indices.size())}));
  const TensorView indptr_v = dev.I32(indptr, Shape({batch + 1}));
  const TensorView last_page_v = dev.I32(last_page, Shape({batch}));

  auto fi = kernels::FlashInferDecode::Create();
  ASSERT_TRUE(fi.ok()) << fi.status();

  const auto decode = [&] {
    return fi->Decode(q, k_cache, v_cache, indices_v, indptr_v,
                      absl::MakeConstSpan(indptr), last_page_v, out, 0.088f,
                      stream);
  };

  // Warm, so first-call costs are not what is measured.
  ASSERT_TRUE(decode().ok());
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kLaunches; ++i) ASSERT_TRUE(decode().ok());
  const auto t1 = std::chrono::steady_clock::now();

  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const auto t2 = std::chrono::steady_clock::now();

  const double enqueue_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double total_ms =
      std::chrono::duration<double, std::milli>(t2 - t0).count();

  std::printf("  enqueued %d launches in %.3f ms of host time; total %.3f ms\n",
              kLaunches, enqueue_ms, total_ms);

  // The GPU has to actually be busy for the comparison to mean anything -- if
  // the work were trivial both numbers would be small and equal.
  ASSERT_GT(total_ms, 1.0)
      << "the configuration is too small to distinguish sync from async";

  EXPECT_LT(enqueue_ms, total_ms * 0.5)
      << "enqueuing took " << enqueue_ms << " ms of the " << total_ms
      << " ms total, so the host was waiting on the device: Decode is "
         "synchronizing";
}

TEST_F(FlashInferTest, CsrConversionMatchesTheDenseTable) {
  // Two sequences: the first holds 3 blocks, the second 1. The rest of each row
  // is padding that must not appear in the output.
  const std::vector<int32_t> dense = {7, 3, 9, 0,   // seq 0, 3 used
                                      2, 0, 0, 0};  // seq 1, 1 used
  std::vector<int32_t> indices, indptr;

  ASSERT_TRUE(kernels::BuildCsrBlockTable(dense, 2, 4, {3, 1}, &indices,
                                          &indptr)
                  .ok());

  EXPECT_EQ(indices, (std::vector<int32_t>{7, 3, 9, 2}));
  EXPECT_EQ(indptr, (std::vector<int32_t>{0, 3, 4}));
}

TEST_F(FlashInferTest, RejectsAnUninstantiatedHeadDim) {
  if (!CudaAvailable()) GTEST_SKIP();

  Dev dev;
  auto fi = kernels::FlashInferDecode::Create();
  ASSERT_TRUE(fi.ok()) << fi.status();

  const TensorView q = dev.Empty(Shape({1, 4, 64}));  // head_dim 64
  const TensorView kc = dev.Empty(Shape({4, 16, 4, 64}));
  const TensorView vc = dev.Empty(Shape({4, 16, 4, 64}));
  const TensorView out = dev.Empty(Shape({1, 4, 64}));

  const std::vector<int32_t> indptr = {0, 1};
  const Status s = fi->Decode(q, kc, vc, dev.I32({0}, Shape({1})),
                              dev.I32(indptr, Shape({2})),
                              absl::MakeConstSpan(indptr),
                              dev.I32({16}, Shape({1})), out, 0.125f);

  EXPECT_EQ(s.code(), absl::StatusCode::kUnimplemented) << s;
}


// Which *physical* pages a sequence occupies must not change its output.
//
// This is R8. The block table is a permutation from logical position to
// physical page, so the same logical sequence stored in pages [0,1,2] and in
// pages [5,4,3] has to attend identically. Our own paged kernel does; this
// checks that FlashInfer does too, which matters because the free list is a
// stack and consecutive sequences therefore receive their pages in opposite
// order -- making a request's output depend on what ran before it.
TEST_F(FlashInferTest, OutputDoesNotDependOnWhichPagesHoldTheSequence) {
  constexpr int64_t head_dim = 128, q_heads = 16, kv_heads = 2;
  constexpr int64_t page_size = 16;
  constexpr int64_t num_blocks = 8;
  constexpr int64_t kLen = 40;  // three pages, the last one partial

  Dev dev;

  const std::vector<float> q_host = Ramp(q_heads * head_dim, 1.3f);
  const std::vector<float> page_host =
      Ramp(page_size * kv_heads * head_dim, 0.1f);

  auto fi = kernels::FlashInferDecode::Create();
  ASSERT_TRUE(fi.ok()) << fi.status();

  // Lays the same three logical pages of K/V into `pages`, then attends.
  const auto attend = [&](const std::vector<int32_t>& pages) {
    Dev d;

    const TensorView k_cache =
        d.Empty(Shape({num_blocks, page_size, kv_heads, head_dim}));
    const TensorView v_cache =
        d.Empty(Shape({num_blocks, page_size, kv_heads, head_dim}));

    // Distinct content per logical page, written to whichever physical page
    // holds it, so a mis-ordered read is visible rather than cancelling out.
    for (size_t logical = 0; logical < pages.size(); ++logical) {
      std::vector<float> k = Ramp(page_size * kv_heads * head_dim,
                                  0.1f + 0.9f * static_cast<float>(logical));
      std::vector<float> v = Ramp(page_size * kv_heads * head_dim,
                                  0.5f + 0.9f * static_cast<float>(logical));

      const size_t bytes = k.size() * sizeof(__nv_bfloat16);
      std::vector<__nv_bfloat16> kb(k.size()), vb(v.size());
      for (size_t i = 0; i < k.size(); ++i) {
        kb[i] = __float2bfloat16(k[i]);
        vb[i] = __float2bfloat16(v[i]);
      }

      const size_t offset =
          static_cast<size_t>(pages[logical]) * bytes;

      EXPECT_EQ(cudaMemcpy(static_cast<std::byte*>(k_cache.Data()) + offset,
                           kb.data(), bytes, cudaMemcpyHostToDevice),
                cudaSuccess);
      EXPECT_EQ(cudaMemcpy(static_cast<std::byte*>(v_cache.Data()) + offset,
                           vb.data(), bytes, cudaMemcpyHostToDevice),
                cudaSuccess);
    }

    const std::vector<int32_t> indptr = {0,
                                         static_cast<int32_t>(pages.size())};
    const TensorView out = d.Empty(Shape({1, q_heads, head_dim}));

    const Status s = fi->Decode(
        d.Bf16(q_host, Shape({1, q_heads, head_dim})), k_cache, v_cache,
        d.I32(pages, Shape({static_cast<int64_t>(pages.size())})),
        d.I32(indptr, Shape({2})), absl::MakeConstSpan(indptr),
        d.I32({static_cast<int32_t>(kLen - 2 * page_size)}, Shape({1})), out,
        0.088f);

    EXPECT_TRUE(s.ok()) << s;

    return d.Down(out);
  };

  const std::vector<float> ascending = attend({0, 1, 2});
  const std::vector<float> descending = attend({5, 4, 3});

  ASSERT_EQ(ascending.size(), descending.size());

  int mismatches = 0;
  for (size_t i = 0; i < ascending.size(); ++i) {
    if (ascending[i] != descending[i]) ++mismatches;
  }

  EXPECT_EQ(mismatches, 0)
      << mismatches << " of " << ascending.size()
      << " outputs changed when the sequence moved from pages [0,1,2] to "
         "[5,4,3]";
}

}  // namespace
}  // namespace inferx
