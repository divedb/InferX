// FP8 KV cache decode against the bf16 KV reference.
//
// The correctness gate for the DecodeFp8 path: the same K/V content, quantized
// to fp8 e4m3, must produce attention output within fp8 quantization error of
// the bf16 path. This is what lets KV memory halve (§6.4) on sm_89, where the
// fa2 decode upcasts fp8 to fp32 on load and uses no fp8 tensor cores.

#include "inferx/core/kv_cache.h"
#include "inferx/kernels/flashinfer_attention.h"
#include "inferx/kernels/flashinfer_prefill.h"
#include "inferx/kernels/quantize.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "absl/types/span.h"
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
    return Raw(/*zeroed=*/true, shape, DataType::kBFloat16,
               shape.Numel() * sizeof(bf16));
  }

  // An fp8 buffer of the given shape, zeroed. fp8 is one byte per element.
  TensorView Fp8(const Shape& shape) {
    return Raw(/*zeroed=*/true, shape, DataType::kFloat8E4M3FN, shape.Numel());
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

  TensorView Raw(bool zeroed, const Shape& shape, DataType dtype, size_t bytes) {
    auto buf = DeviceBuffer::Allocate(bytes, DeviceId::Cuda(0));
    EXPECT_TRUE(buf.ok()) << buf.status();
    if (zeroed) {
      EXPECT_EQ(cudaMemset(buf->data(), 0, buf->size()), cudaSuccess);
    }
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

// A device float the quantize kernel can write a scale into, read back to host.
struct DeviceScalar {
  DeviceBuffer buf;
  explicit DeviceScalar() {
    auto b = DeviceBuffer::Allocate(sizeof(float), DeviceId::Cuda(0));
    EXPECT_TRUE(b.ok()) << b.status();
    EXPECT_EQ(cudaMemset(b->data(), 0, b->size()), cudaSuccess);
    buf = *std::move(b);
  }
  float Read() {
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    float v = 0.0f;
    EXPECT_EQ(cudaMemcpy(&v, buf.data(), sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return v;
  }
};

class Fp8KvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }
};

// The headline case: fp8 KV decode agrees with bf16 KV decode within fp8
// quantization error. Runs a ragged batch (mix of partly-filled and exactly
// filled last pages) at Qwen2.5-3B's GQA geometry.
TEST_F(Fp8KvTest, DecodeWithFp8KvMatchesBf16WithinQuantError) {
  constexpr int64_t q_heads = 16, kv_heads = 2, head_dim = 128;
  constexpr int64_t page_size = 16, num_blocks = 64;
  const std::vector<int64_t> lens = {33, 64, 18};
  const int64_t batch = static_cast<int64_t>(lens.size());
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  Dev dev;

  // Assign blocks back to front so a table walked as the identity fails.
  std::vector<std::vector<int32_t>> seq_blocks(lens.size());
  int32_t next = static_cast<int32_t>(num_blocks) - 1;
  for (size_t s = 0; s < lens.size(); ++s) {
    const int64_t need = (lens[s] + page_size - 1) / page_size;
    for (int64_t b = 0; b < need; ++b) seq_blocks[s].push_back(next--);
    ASSERT_GE(next, -1);
  }

  const Shape cache_shape({num_blocks, page_size, kv_heads, head_dim});
  const TensorView k_cache = dev.Empty(cache_shape);
  const TensorView v_cache = dev.Empty(cache_shape);

  for (size_t s = 0; s < lens.size(); ++s) {
    const int64_t n = lens[s];
    const std::vector<float> k =
        Ramp(static_cast<size_t>(n * kv_heads * head_dim), 1.0f + s);
    const std::vector<float> v =
        Ramp(static_cast<size_t>(n * kv_heads * head_dim), 2.0f + s);

    std::vector<int32_t> slots;
    for (int64_t t = 0; t < n; ++t) {
      const int32_t block = seq_blocks[s][static_cast<size_t>(t / page_size)];
      slots.push_back(static_cast<int32_t>(block * page_size + (t % page_size)));
    }

    ASSERT_TRUE(kernels::AppendToKvCache(
                    dev.Bf16(k, Shape({n, kv_heads, head_dim})),
                    dev.Bf16(v, Shape({n, kv_heads, head_dim})), k_cache,
                    v_cache, dev.I32(slots, Shape({n})))
                    .ok());
  }

  const std::vector<float> q =
      Ramp(static_cast<size_t>(batch * q_heads * head_dim), 0.0f);
  const TensorView qd = dev.Bf16(q, Shape({batch, q_heads, head_dim}));

  // Dense -> CSR block table, and the per-sequence last-page lengths.
  int64_t max_blocks = 0;
  for (const auto& b : seq_blocks) {
    max_blocks = std::max<int64_t>(max_blocks, static_cast<int64_t>(b.size()));
  }
  std::vector<int32_t> dense(static_cast<size_t>(batch * max_blocks), 0);
  std::vector<int32_t> blocks_used;
  std::vector<int32_t> last_page;
  for (size_t s = 0; s < lens.size(); ++s) {
    for (size_t b = 0; b < seq_blocks[s].size(); ++b) {
      dense[s * static_cast<size_t>(max_blocks) + b] = seq_blocks[s][b];
    }
    blocks_used.push_back(static_cast<int32_t>(seq_blocks[s].size()));
    const int64_t rem = lens[s] % page_size;
    last_page.push_back(static_cast<int32_t>(rem == 0 ? page_size : rem));
  }

  std::vector<int32_t> indices, indptr;
  ASSERT_TRUE(kernels::BuildCsrBlockTable(dense, batch, max_blocks, blocks_used,
                                          &indices, &indptr)
                  .ok());

  auto fi = kernels::FlashInferDecode::Create();
  ASSERT_TRUE(fi.ok()) << fi.status();

  // --- reference: bf16 KV ------------------------------------------------
  const TensorView ref = dev.Empty(Shape({batch, q_heads, head_dim}));
  ASSERT_TRUE(fi->Decode(
                  qd, k_cache, v_cache,
                  dev.I32(indices, Shape({static_cast<int64_t>(indices.size())})),
                  dev.I32(indptr, Shape({batch + 1})),
                  absl::MakeConstSpan(indptr),
                  dev.I32(last_page, Shape({batch})), ref, scale)
                  .ok());

  // --- fp8 KV: quantize the same caches, per-tensor scale ----------------
  // QuantizeToF8E4M3Dynamic handles bf16 input and writes the scale it used.
  // The cache's amax is the live data's amax -- unused slots are zero from
  // Empty() -- so the per-tensor scale is the right dequant factor for every
  // page attention will actually read.
  const TensorView k_fp8 = dev.Fp8(cache_shape);
  const TensorView v_fp8 = dev.Fp8(cache_shape);
  DeviceScalar k_scale_dev, v_scale_dev;
  ASSERT_TRUE(kernels::QuantizeToF8E4M3Dynamic(
                  k_cache, k_fp8,
                  reinterpret_cast<float*>(k_scale_dev.buf.data()))
                  .ok());
  ASSERT_TRUE(kernels::QuantizeToF8E4M3Dynamic(
                  v_cache, v_fp8,
                  reinterpret_cast<float*>(v_scale_dev.buf.data()))
                  .ok());
  const float k_scale = k_scale_dev.Read();
  const float v_scale = v_scale_dev.Read();
  ASSERT_GT(k_scale, 0.0f);
  ASSERT_GT(v_scale, 0.0f);

  const TensorView got = dev.Empty(Shape({batch, q_heads, head_dim}));
  const Status s = fi->DecodeFp8(
      qd, k_fp8, v_fp8,
      dev.I32(indices, Shape({static_cast<int64_t>(indices.size())})),
      dev.I32(indptr, Shape({batch + 1})), absl::MakeConstSpan(indptr),
      dev.I32(last_page, Shape({batch})), got, scale, k_scale, v_scale);
  ASSERT_TRUE(s.ok()) << s;

  // --- compare -----------------------------------------------------------
  // fp8 e4m3 keeps 3 mantissa bits (~6% per element) on both K and V, and that
  // error routes through softmax, so the bound is looser than the bf16-vs-bf16
  // reference test's 0.15*rms. Stated against the output's own RMS so it holds
  // across sequence lengths without tuning to one.
  const std::vector<float> a = dev.Down(ref);
  const std::vector<float> b = dev.Down(got);
  ASSERT_EQ(a.size(), b.size());

  double worst = 0, sumsq = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    worst = std::max<double>(worst, std::abs(a[i] - b[i]));
    sumsq += static_cast<double>(a[i]) * a[i];
  }
  const double rms = std::sqrt(sumsq / a.size());

  EXPECT_LT(worst, rms * 0.25)
      << "worst |bf16-fp8| " << worst << " against rms " << rms;
}

// Rejects a bf16 cache handed to the fp8 path. A dtype mismatch is a silent
// reinterpretation of bytes otherwise, which is exactly the kind of bug the
// explicit check exists to catch.
TEST_F(Fp8KvTest, RejectsBf16CacheOnTheFp8Path) {
  auto fi = kernels::FlashInferDecode::Create();
  ASSERT_TRUE(fi.ok()) << fi.status();

  Dev dev;
  const Shape cache_shape({2, 16, 2, 128});
  const TensorView k_bf16 = dev.Empty(cache_shape);
  const TensorView v_bf16 = dev.Empty(cache_shape);
  const TensorView q = dev.Empty(Shape({1, 16, 128}));
  const TensorView out = dev.Empty(Shape({1, 16, 128}));

  std::vector<int32_t> indptr = {0, 1};
  const std::vector<int32_t> last_page = {1};
  const Status s =
      fi->DecodeFp8(q, k_bf16, v_bf16,
                    dev.I32({0}, Shape({1})), dev.I32(indptr, Shape({2})),
                    absl::MakeConstSpan(indptr), dev.I32(last_page, Shape({1})),
                    out, 0.1f, 1.0f, 1.0f);
  EXPECT_EQ(s.code(), absl::StatusCode::kInvalidArgument);
}

// The prefill writes the prompt's K/V and attends over them, so for a cache the
// fp8 decode path will read, prefill must speak fp8 too. The gate: the same K/V
// content, quantized to fp8 e4m3, must attend within fp8 quantization error of
// the bf16 path.
//
// History: this previously failed with output uniformly ~0.82x the bf16
// reference, which traced to k_scale being folded into sm_scale (sm_scale =
// scale*k_scale ~ 1e-4). fa2 prefill suppresses masked/padding K positions
// inside exp2(logit*sm_scale_log2); at sm_scale_log2 ~ 1e-4 those positions stop
// being zeroed and leak into the softmax denominator. The bf16 path collapses
// the same way at an equally tiny sm_scale, so it is not fp8-specific. The fix
// folds k_scale into the query (q*k_scale) and keeps sm_scale = scale.
TEST_F(Fp8KvTest, PrefillWithFp8KvMatchesBf16WithinQuantError) {
  constexpr int64_t kPageSize = 16, kHeads = 16, kKvHeads = 2, kHeadDim = 128;
  constexpr int64_t kLen = 40;  // three pages, last partial
  const int64_t pages = (kLen + kPageSize - 1) / kPageSize;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

  Dev dev;
  const Shape cache_shape({8, kPageSize, kKvHeads, kHeadDim});
  const TensorView k_cache = dev.Empty(cache_shape);
  const TensorView v_cache = dev.Empty(cache_shape);

  const std::vector<float> kv =
      Ramp(static_cast<size_t>(kLen * kKvHeads * kHeadDim), 0.2f);
  const std::vector<float> qv =
      Ramp(static_cast<size_t>(kLen * kHeads * kHeadDim), 1.1f);

  std::vector<int32_t> slots(static_cast<size_t>(kLen));
  for (int64_t i = 0; i < kLen; ++i) slots[i] = static_cast<int32_t>(i);

  ASSERT_TRUE(kernels::AppendToKvCache(
                  dev.Bf16(kv, Shape({kLen, kKvHeads, kHeadDim})),
                  dev.Bf16(kv, Shape({kLen, kKvHeads, kHeadDim})), k_cache,
                  v_cache, dev.I32(slots, Shape({kLen})))
                  .ok());

  const TensorView q = dev.Bf16(qv, Shape({kLen, kHeads, kHeadDim}));
  std::vector<int32_t> table(static_cast<size_t>(pages));
  for (int64_t b = 0; b < pages; ++b) table[b] = static_cast<int32_t>(b);
  const std::vector<int32_t> qo_indptr = {0, static_cast<int32_t>(kLen)};
  const std::vector<int32_t> kv_indptr = {0, static_cast<int32_t>(pages)};
  const int32_t last_page =
      static_cast<int32_t>(kLen - (pages - 1) * kPageSize);

  auto fi = kernels::FlashInferPrefill::Create();
  ASSERT_TRUE(fi.ok()) << fi.status();

  // Reference: bf16 KV.
  const TensorView ref = dev.Empty(Shape({kLen, kHeads, kHeadDim}));
  ASSERT_TRUE((*fi)->Prefill(q, k_cache, v_cache,
                          dev.I32(qo_indptr, Shape({2})),
                          absl::MakeConstSpan(qo_indptr),
                          dev.I32(table, Shape({pages})),
                          dev.I32(kv_indptr, Shape({2})),
                          absl::MakeConstSpan(kv_indptr),
                          dev.I32({last_page}, Shape({1})), ref, scale)
                  .ok());

  // fp8 KV: quantize the same caches, per-tensor scale.
  const TensorView k_fp8 = dev.Fp8(cache_shape);
  const TensorView v_fp8 = dev.Fp8(cache_shape);
  DeviceScalar k_scale_dev, v_scale_dev;
  ASSERT_TRUE(kernels::QuantizeToF8E4M3Dynamic(
                  k_cache, k_fp8,
                  reinterpret_cast<float*>(k_scale_dev.buf.data()))
                  .ok());
  ASSERT_TRUE(kernels::QuantizeToF8E4M3Dynamic(
                  v_cache, v_fp8,
                  reinterpret_cast<float*>(v_scale_dev.buf.data()))
                  .ok());
  const float k_scale = k_scale_dev.Read();
  const float v_scale = v_scale_dev.Read();
  ASSERT_GT(k_scale, 0.0f);
  ASSERT_GT(v_scale, 0.0f);

  const TensorView got = dev.Empty(Shape({kLen, kHeads, kHeadDim}));
  const Status s = (*fi)->PrefillFp8(
      q, k_fp8, v_fp8, dev.I32(qo_indptr, Shape({2})),
      absl::MakeConstSpan(qo_indptr), dev.I32(table, Shape({pages})),
      dev.I32(kv_indptr, Shape({2})), absl::MakeConstSpan(kv_indptr),
      dev.I32({last_page}, Shape({1})), got, scale, k_scale, v_scale);
  ASSERT_TRUE(s.ok()) << s;

  const std::vector<float> a = dev.Down(ref);
  const std::vector<float> b = dev.Down(got);
  ASSERT_EQ(a.size(), b.size());
  double worst = 0, sumsq = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    worst = std::max<double>(worst, std::abs(a[i] - b[i]));
    sumsq += static_cast<double>(a[i]) * a[i];
  }
  const double rms = std::sqrt(sumsq / a.size());
  EXPECT_LT(worst, rms * 0.25)
      << "fp8 KV prefill drifted from bf16 by " << worst << " (rms " << rms << ")";
}

// The reason PlanFp8/RunFp8 exist at all: the engine captures the decode step
// into a CUDA graph (M6), and a one-shot DecodeFp8 cannot be recorded. This
// captures RunFp8 into a graph and replays it, then checks the replay matches a
// direct RunFp8 on the same plan. Same plan + same inputs is the same
// computation, so the bound is tight -- the test is here to catch anything
// uncapturable creeping into RunFp8 (a host sync, an allocation), which would
// fail the capture or diverge the replay.
TEST_F(Fp8KvTest, GraphCapturedRunFp8ReplaysTheDirectRun) {
  constexpr int64_t q_heads = 16, kv_heads = 2, head_dim = 128;
  constexpr int64_t page_size = 16, num_blocks = 64;
  const std::vector<int64_t> lens = {33, 64, 18};
  const int64_t batch = static_cast<int64_t>(lens.size());
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  Dev dev;

  std::vector<std::vector<int32_t>> seq_blocks(lens.size());
  int32_t next = static_cast<int32_t>(num_blocks) - 1;
  for (size_t s = 0; s < lens.size(); ++s) {
    const int64_t need = (lens[s] + page_size - 1) / page_size;
    for (int64_t b = 0; b < need; ++b) seq_blocks[s].push_back(next--);
  }

  const Shape cache_shape({num_blocks, page_size, kv_heads, head_dim});
  const TensorView k_cache = dev.Empty(cache_shape);
  const TensorView v_cache = dev.Empty(cache_shape);

  for (size_t s = 0; s < lens.size(); ++s) {
    const int64_t n = lens[s];
    const std::vector<float> k =
        Ramp(static_cast<size_t>(n * kv_heads * head_dim), 1.0f + s);
    const std::vector<float> v =
        Ramp(static_cast<size_t>(n * kv_heads * head_dim), 2.0f + s);
    std::vector<int32_t> slots;
    for (int64_t t = 0; t < n; ++t) {
      const int32_t block = seq_blocks[s][static_cast<size_t>(t / page_size)];
      slots.push_back(static_cast<int32_t>(block * page_size + (t % page_size)));
    }
    ASSERT_TRUE(kernels::AppendToKvCache(
                    dev.Bf16(k, Shape({n, kv_heads, head_dim})),
                    dev.Bf16(v, Shape({n, kv_heads, head_dim})), k_cache,
                    v_cache, dev.I32(slots, Shape({n})))
                    .ok());
  }

  const std::vector<float> q =
      Ramp(static_cast<size_t>(batch * q_heads * head_dim), 0.0f);
  const TensorView qd = dev.Bf16(q, Shape({batch, q_heads, head_dim}));

  int64_t max_blocks = 0;
  for (const auto& b : seq_blocks) {
    max_blocks = std::max<int64_t>(max_blocks, static_cast<int64_t>(b.size()));
  }
  std::vector<int32_t> dense(static_cast<size_t>(batch * max_blocks), 0);
  std::vector<int32_t> blocks_used;
  std::vector<int32_t> last_page;
  for (size_t s = 0; s < lens.size(); ++s) {
    for (size_t b = 0; b < seq_blocks[s].size(); ++b) {
      dense[s * static_cast<size_t>(max_blocks) + b] = seq_blocks[s][b];
    }
    blocks_used.push_back(static_cast<int32_t>(seq_blocks[s].size()));
    const int64_t rem = lens[s] % page_size;
    last_page.push_back(static_cast<int32_t>(rem == 0 ? page_size : rem));
  }
  std::vector<int32_t> indices, indptr;
  ASSERT_TRUE(kernels::BuildCsrBlockTable(dense, batch, max_blocks, blocks_used,
                                          &indices, &indptr)
                  .ok());

  const TensorView k_fp8 = dev.Fp8(cache_shape);
  const TensorView v_fp8 = dev.Fp8(cache_shape);
  DeviceScalar k_scale_dev, v_scale_dev;
  ASSERT_TRUE(kernels::QuantizeToF8E4M3Dynamic(
                  k_cache, k_fp8,
                  reinterpret_cast<float*>(k_scale_dev.buf.data()))
                  .ok());
  ASSERT_TRUE(kernels::QuantizeToF8E4M3Dynamic(
                  v_cache, v_fp8,
                  reinterpret_cast<float*>(v_scale_dev.buf.data()))
                  .ok());
  const float k_scale = k_scale_dev.Read();
  const float v_scale = v_scale_dev.Read();

  auto fi = kernels::FlashInferDecode::Create();
  ASSERT_TRUE(fi.ok()) << fi.status();

  // The index/indptr tensors are allocated once, before capture, and reused
  // for both runs -- cudaMalloc inside a captured region is forbidden, and the
  // engine does the same (stable buffers across steps is what makes the graph
  // replayable).
  const TensorView indices_tv =
      dev.I32(indices, Shape({static_cast<int64_t>(indices.size())}));
  const TensorView indptr_tv = dev.I32(indptr, Shape({batch + 1}));
  const TensorView last_page_tv = dev.I32(last_page, Shape({batch}));

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  // graph_safe=true: the fixed-shape mode whose workspace offsets stop moving,
  // which is the property a captured graph needs to replay as the batch grows.
  ASSERT_TRUE(fi->PlanFp8(batch, q_heads, kv_heads, head_dim, page_size,
                          absl::MakeConstSpan(indptr),
                          /*graph_safe=*/true, stream)
                  .ok());

  // Direct run on that plan -> the reference replay should match.
  const TensorView ref = dev.Empty(Shape({batch, q_heads, head_dim}));
  ASSERT_TRUE(fi->RunFp8(qd, k_fp8, v_fp8, indices_tv, indptr_tv, last_page_tv,
                         ref, scale, k_scale, v_scale, stream)
                  .ok());
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  // Capture a second run into a graph and launch it.
  const TensorView cap = dev.Empty(Shape({batch, q_heads, head_dim}));
  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  ASSERT_TRUE(fi->RunFp8(qd, k_fp8, v_fp8, indices_tv, indptr_tv, last_page_tv,
                         cap, scale, k_scale, v_scale, stream)
                  .ok());
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess) << "RunFp8 not capturable";
  cudaGraphExec_t exec = nullptr;
  ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(exec, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);

  const std::vector<float> a = dev.Down(ref);
  const std::vector<float> b = dev.Down(cap);
  ASSERT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    worst = std::max<double>(worst, std::abs(a[i] - b[i]));
  }
  // Same plan, same inputs, same kernels -> identical computation. The bound is
  // tight because it is not measuring numerical error, only that the captured
  // replay does the same work the direct launch did.
  EXPECT_LT(worst, 1e-4) << "captured RunFp8 diverged from direct by " << worst;
}

}  // namespace
}  // namespace inferx
