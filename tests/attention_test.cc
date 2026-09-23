#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device_runtime.h"
#include "inferx/ops/flash_attention.h"

namespace inferx {
namespace {
float Bf16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  bits += 0x7fff + ((bits >> 16) & 1);
  bits &= 0xffff0000;
  std::memcpy(&v, &bits, 4);
  return v;
}
class AttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto rt = RuntimeFor(DeviceId::Cuda(0));
    ASSERT_TRUE(rt.ok());
    runtime_ = *rt;
    ASSERT_TRUE(runtime_->Activate().ok());
    auto stream = runtime_->CreateStream();
    ASSERT_TRUE(stream.ok());
    stream_ = *stream;
  }
  void TearDown() override {
    ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
    ASSERT_TRUE(runtime_->DestroyStream(stream_).ok());
  }
  Tensor Ints(const std::vector<int32_t>& x) {
    auto t =
        Tensor::Empty(DataType::kInt32, Shape({int64_t(x.size())}), DeviceId::Cuda(0)).value();
    EXPECT_TRUE(runtime_->Copy(t.Data(), x.data(), x.size() * 4, CopyKind::kHostToDevice).ok());
    return t;
  }
  Tensor Upload(const std::vector<float>& x, Shape shape) {
    auto t = Tensor::Empty(DataType::kBFloat16, shape, DeviceId::Cuda(0)).value();
    std::vector<uint16_t> bits(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      float rounded = Bf16(x[i]);
      uint32_t b;
      std::memcpy(&b, &rounded, 4);
      bits[i] = b >> 16;
    }
    EXPECT_TRUE(
        runtime_->Copy(t.Data(), bits.data(), bits.size() * 2, CopyKind::kHostToDevice).ok());
    return t;
  }
  std::vector<float> Read(const Tensor& t) {
    EXPECT_TRUE(runtime_->SynchronizeStream(stream_).ok());
    std::vector<uint16_t> bits(t.Numel());
    EXPECT_TRUE(
        runtime_->Copy(bits.data(), t.Data(), bits.size() * 2, CopyKind::kDeviceToHost).ok());
    std::vector<float> x(bits.size());
    for (size_t i = 0; i < x.size(); ++i) {
      uint32_t b = uint32_t(bits[i]) << 16;
      std::memcpy(&x[i], &b, 4);
    }
    return x;
  }
  void Check(int dim, int group, int page, std::vector<int> lengths, std::vector<int> queries,
             bool split = false, bool graph = false, float scale_multiplier = 1.0f,
             int kvheads = 2) {
    SCOPED_TRACE(::testing::Message() << "dim=" << dim << " group=" << group << " page=" << page
                                      << "split=" << split << " graph=" << graph);
    const int heads = kvheads * group, batch = lengths.size();
    std::vector<int> qo{0}, kv{0}, last;
    for (int s = 0; s < batch; ++s) {
      qo.push_back(qo.back() + queries[s]);
      kv.push_back(kv.back() + (lengths[s] + page - 1) / page);
      last.push_back((lengths[s] - 1) % page + 1);
    }
    const int tokens = qo.back(), blocks = kv.back();
    std::vector<int> ids(blocks);
    for (int i = 0; i < blocks; ++i) ids[i] = blocks - 1 - i;
    std::vector<float> keys(blocks * page * kvheads * dim), values(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = Bf16(std::sin(float(i % 1009) * 0.13f) * 0.7f);
      values[i] = Bf16(std::cos(float(i % 997) * 0.19f) * 0.8f);
    }
    std::vector<float> qv(tokens * heads * dim);
    for (size_t i = 0; i < qv.size(); ++i)
      qv[i] = Bf16(std::sin(float(i % 991) * 0.17f) * 0.9f);
    auto q = Upload(qv, Shape({tokens, heads * dim}));
    auto key = Upload(keys, Shape({blocks, page, kvheads, dim}));
    auto value = Upload(values, Shape({blocks, page, kvheads, dim}));
    auto qo_d = Ints(qo), kv_d = Ints(kv), ids_d = Ints(ids), last_d = Ints(last);
    auto out = Tensor::Empty(DataType::kBFloat16, q.GetShape(), DeviceId::Cuda(0)).value();
    int tiles = 0;
    for (int n : queries) tiles += (n * group + 63) / 64;
    auto plan = Ints(std::vector<int>(3 * tiles + 1));
    ops::FlashDecodeWorkspace workspace;
    if (split) {
      const int n = batch * ops::FlashDecodeWorkspace::kPartitions;
      workspace.plan = Ints(std::vector<int>(3 * n + batch + 2));
      workspace.values =
          Tensor::Empty(DataType::kBFloat16, Shape({n * heads * dim}), DeviceId::Cuda(0))
              .value();
      workspace.scores =
          Tensor::Empty(DataType::kFloat, Shape({n * heads}), DeviceId::Cuda(0)).value();
    }
    ops::ExecutionContext ctx(*runtime_, stream_);
    ops::AttentionParams p{heads, kvheads, dim, scale_multiplier / std::sqrt(float(dim))};
    auto run = [&]() -> Status {
      INFERX_RETURN_IF_ERROR(ops::PrepareFlashAttention(ctx, qo_d, plan, group, tiles));
      if (split)
        INFERX_RETURN_IF_ERROR(ops::PrepareFlashDecode(ctx, kv_d, last_d, page, workspace));
      return ops::FlashPagedAttention(ctx, q, qo_d, kv_d, ids_d, last_d, key, value, page, p,
                                      plan, tiles, out, split ? &workspace : nullptr);
    };
    ASSERT_TRUE(run().ok());
    if (graph) {
      ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
      ASSERT_TRUE(runtime_->BeginCapture(stream_).ok());
      ASSERT_TRUE(run().ok());
      auto executable = runtime_->EndCaptureAndInstantiate(stream_);
      ASSERT_TRUE(executable.ok());
      // Change page tables and last-page lengths at stable addresses: replay must
      // rebuild its plan and read new metadata, not reuse capture-time contents.
      std::reverse(ids.begin(), ids.end());
      for (int s = 0; s < batch; ++s) {
        if (last[s] > 1 && lengths[s] > queries[s]) {
          --last[s];
          --lengths[s];
        }
      }
      ASSERT_TRUE(
          runtime_->Copy(ids_d.Data(), ids.data(), ids.size() * 4, CopyKind::kHostToDevice)
              .ok());
      ASSERT_TRUE(
          runtime_->Copy(last_d.Data(), last.data(), last.size() * 4, CopyKind::kHostToDevice)
              .ok());
      ASSERT_TRUE(runtime_->LaunchGraph(*executable, stream_).ok());
      ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
      ASSERT_TRUE(runtime_->DestroyGraph(*executable).ok());
    }
    const auto got = Read(out);
    double max_error = 0;
    for (int s = 0; s < batch; ++s)
      for (int t = 0; t < queries[s]; ++t) {
        const int causal_length = lengths[s] - queries[s] + t + 1;
        for (int h = 0; h < heads; ++h) {
          std::vector<double> scores(causal_length);
          const int qbase = ((qo[s] + t) * heads + h) * dim;
          auto address = [&](int pos) {
            return ((ids[kv[s] + pos / page] * page + pos % page) * kvheads + h / group) * dim;
          };
          for (int pos = 0; pos < causal_length; ++pos) {
            double dot = 0;
            const int base = address(pos);
            for (int d = 0; d < dim; ++d) dot += double(qv[qbase + d]) * keys[base + d];
            scores[pos] = dot * p.scale;
          }
          const double max_score = *std::max_element(scores.begin(), scores.end());
          double sum = 0;
          for (auto& v : scores) {
            v = std::exp(v - max_score);
            sum += v;
          }
          for (int d = 0; d < dim; ++d) {
            double expected = 0;
            for (int pos = 0; pos < causal_length; ++pos)
              expected += scores[pos] / sum * values[address(pos) + d];
            ASSERT_TRUE(std::isfinite(got[qbase + d]));
            max_error = std::max(max_error, std::abs(double(got[qbase + d]) - expected));
          }
        }
      }
    EXPECT_LE(max_error, 0.005);  // Existing BF16 attention tolerance, unchanged.
  }
  DeviceRuntime* runtime_ = nullptr;
  Stream stream_;
};

TEST_F(AttentionTest, HeadDimensionsAndGroupedQueryLayouts) {
  for (int dim : {64, 128, 256})
    for (int group : {1, 2, 3, 4, 5, 6, 8, 16, 32}) {
      Check(dim, group, 16, {33, 17}, {33, 17});  // Full causal prefill.
      Check(dim, group, 32, {65, 33}, {1, 3});    // Mixed decode / cached-prefix prefill.
      Check(dim, group, 64, {129, 66}, {1, 1});   // Decode, including prefill-kernel dispatch.
    }
}
TEST_F(AttentionTest, EverySupportedGroupRatio) {
  for (int group = 1; group <= 32; ++group) Check(128, group, 16, {17, 9}, {1, 1});
}
TEST_F(AttentionTest, LongPromptsAndDecodeContexts) {
  Check(128, 2, 16, {1024, 128}, {1024, 128});
  Check(128, 2, 16, {1152, 1025, 513}, {1, 1, 1});
  Check(128, 2, 16, {1024, 513, 129}, {1, 129, 17});
  Check(128, 2, 16, {32768, 4097}, {1, 1});
  Check(128, 8, 16, {33, 17}, {33, 17}, false, false, 1.0f, 1);
  Check(128, 8, 16, {33, 17}, {1, 1}, false, false, 1.0f, 1);
  Check(128, 2, 16, std::vector<int>(32, 129), std::vector<int>(32, 1));
  Check(128, 2, 16, {129, 33}, {129, 33}, false, false, 16.0f);
}
TEST_F(AttentionTest, GraphReplayAndSplitDecodeReadChangedMetadata) {
  for (int dim : {64, 128, 256}) {
    Check(dim, 2, 16, {129, 33}, {1, 1}, false, true);
    Check(dim, 2, 16, {129, 33}, {1, 1}, true, true);
  }
  Check(128, 16, 16, {129, 33}, {1, 1}, false, true);
}
TEST(AttentionBackendTest, ResolvesDefaultAndExplicitBackend) {
  for (const char* name : {"flashinfer", "default", "flash"}) {
    auto result = ops::ParseAttentionBackend(name);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(*result, ops::AttentionBackend::kFlashInfer);
  }
  EXPECT_FALSE(ops::ParseAttentionBackend("cutlass").ok());
  EXPECT_FALSE(ops::ParseAttentionBackend("typo").ok());
  for (auto p : {ops::AttentionParams{4, 0, 128}, ops::AttentionParams{3, 2, 128},
                 ops::AttentionParams{4, 2, 128, std::numeric_limits<float>::quiet_NaN()}}) {
    EXPECT_EQ(ops::ValidateAttentionGeometry(ops::AttentionBackend::kFlashInfer, p).code(),
              absl::StatusCode::kInvalidArgument);
  }
  for (auto p : {ops::AttentionParams{4, 2, 96}, ops::AttentionParams{66, 2, 128},
                 ops::AttentionParams{4, 2, 128, 1.0f, 64}}) {
    EXPECT_EQ(ops::ValidateAttentionGeometry(ops::AttentionBackend::kFlashInfer, p).code(),
              absl::StatusCode::kUnimplemented);
  }
}
TEST_F(AttentionTest, RejectsMalformedMetadataAndWorkspacesBeforeLaunching) {
  ops::ExecutionContext ctx(*runtime_, stream_);
  auto qo = Ints({0, 1}), kv = Ints({0, 1}), ids = Ints({0}), last = Ints({1});
  auto plan = Ints(std::vector<int>(4));
  auto q = Upload(std::vector<float>(128), Shape({1, 128}));
  auto key = Upload(std::vector<float>(16 * 128), Shape({1, 16, 1, 128}));
  auto out = Upload(std::vector<float>(128), Shape({1, 128}));
  ops::AttentionParams p{1, 1, 128};
  auto bad = ops::FlashPagedAttention(ctx, q, qo, kv, ids, last, key, key, 16,
                                      ops::AttentionParams{1, 0, 128}, plan, 1, out);
  EXPECT_EQ(bad.code(), absl::StatusCode::kInvalidArgument);
  auto short_plan = Ints({0});
  EXPECT_FALSE(ops::PrepareFlashAttention(ctx, qo, short_plan, 1, 1).ok());
  auto short_kv = Ints({0});
  EXPECT_FALSE(
      ops::FlashPagedAttention(ctx, q, qo, short_kv, ids, last, key, key, 16, p, plan, 1, out)
          .ok());
  auto wrong_last = Ints({1, 1});
  EXPECT_FALSE(
      ops::FlashPagedAttention(ctx, q, qo, kv, ids, wrong_last, key, key, 16, p, plan, 1, out)
          .ok());
  ops::FlashDecodeWorkspace empty;
  EXPECT_FALSE(ops::PrepareFlashDecode(ctx, kv, last, 16, empty).ok());
  EXPECT_FALSE(
      ops::FlashPagedAttention(ctx, q, qo, kv, ids, last, key, key, 16, p, plan, 1, out, &empty)
          .ok());
}
}  // namespace
}  // namespace inferx
