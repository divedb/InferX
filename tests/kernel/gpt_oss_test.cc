// The three things gpt-oss does that no other model in this engine does.
//
// Phase 2 of docs/gpt-oss-20b-llm.md. Each of these was read out of
// `transformers.models.gpt_oss` rather than derived, because each is a
// silent-wrongness bug if guessed -- a model that generates fluent text and the
// wrong text. So each is checked here against the reference's own numbers.
//
// The attention-sink test is the one that carries a design decision rather than
// just an implementation. gpt-oss's sink participates in the softmax
// denominator, which looked like it needed a change inside FlashInfer's decode
// kernel -- and that kernel has no hook for it, where the prefill one does. The
// way out is that the sink factors out of the softmax exactly, so it becomes a
// scalar rescale applied afterwards using the log-sum-exp the kernel already
// computes. This test is what says that identity is real.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"
#include "inferx/kernels/gpt_oss.h"

namespace inferx {
namespace {

using bf16 = __nv_bfloat16;

float Fill(int64_t a, int64_t b, float salt) {
  return std::sin(0.53f * static_cast<float>(a) + salt) *
         std::cos(0.29f * static_cast<float>(b) - salt);
}

std::vector<bf16> ToBf16(const std::vector<float>& v) {
  std::vector<bf16> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = __float2bfloat16(v[i]);
  return out;
}

std::vector<float> RoundTrip(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out[i] = __bfloat162float(__float2bfloat16(v[i]));
  }
  return out;
}

template <typename T>
TensorView Upload(std::vector<DeviceBuffer>& keep, const std::vector<T>& host,
                  DataType dtype, const Shape& shape) {
  auto buf = DeviceBuffer::Allocate(host.size() * sizeof(T), DeviceId::Cuda(0));
  EXPECT_TRUE(buf.ok()) << buf.status();
  EXPECT_EQ(cudaMemcpy(buf->data(), host.data(), host.size() * sizeof(T),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  keep.push_back(*std::move(buf));

  auto v = TensorView::Create(keep.back().data(), dtype, shape,
                              DeviceId::Cuda(0));
  EXPECT_TRUE(v.ok()) << v.status();
  return *v;
}

template <typename T>
std::vector<T> Download(const TensorView& v, int64_t count) {
  std::vector<T> out(static_cast<size_t>(count));
  EXPECT_EQ(cudaMemcpy(out.data(), v.Data(), out.size() * sizeof(T),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return out;
}

class GptOssTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }
};

// --- Attention sinks ---------------------------------------------------------

// HF's formulation, on the host: concatenate the sink to the row's scores,
// softmax over the lot, then drop the sink column. This is the definition the
// rescale has to reproduce.
std::vector<float> ReferenceSinkedAttention(const std::vector<float>& scores,
                                            const std::vector<float>& values,
                                            float sink, int64_t context,
                                            int64_t head_dim) {
  float max_v = sink;
  for (float s : scores) max_v = std::max(max_v, s);

  double denom = std::exp(static_cast<double>(sink - max_v));
  std::vector<double> weights(static_cast<size_t>(context));
  for (int64_t j = 0; j < context; ++j) {
    weights[static_cast<size_t>(j)] =
        std::exp(static_cast<double>(scores[static_cast<size_t>(j)] - max_v));
    denom += weights[static_cast<size_t>(j)];
  }

  std::vector<float> out(static_cast<size_t>(head_dim), 0.0f);
  for (int64_t d = 0; d < head_dim; ++d) {
    double acc = 0.0;
    for (int64_t j = 0; j < context; ++j) {
      acc += weights[static_cast<size_t>(j)] *
             values[static_cast<size_t>(j * head_dim + d)];
    }
    out[static_cast<size_t>(d)] = static_cast<float>(acc / denom);
  }
  return out;
}

TEST_F(GptOssTest, SinkRescaleReproducesTheConcatenatedSoftmax) {
  constexpr int64_t tokens = 3, heads = 4, head_dim = 16, context = 29;

  // Sinks spanning the range where the correction goes from negligible to
  // total. A sink of +6 against these scores holds most of the softmax mass,
  // so the rescale has to be doing real work in at least one head.
  const std::vector<float> sinks_f = {-4.0f, 0.0f, 2.5f, 6.0f};

  std::vector<float> plain(static_cast<size_t>(tokens * heads * head_dim));
  std::vector<float> lse(static_cast<size_t>(tokens * heads));
  std::vector<std::vector<float>> want(static_cast<size_t>(tokens * heads));

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      std::vector<float> scores(static_cast<size_t>(context));
      for (int64_t j = 0; j < context; ++j) {
        scores[static_cast<size_t>(j)] = 2.0f * Fill(t * heads + h, j, 0.3f);
      }

      std::vector<float> values(static_cast<size_t>(context * head_dim));
      for (size_t i = 0; i < values.size(); ++i) {
        values[i] = Fill(static_cast<int64_t>(i), h, 1.1f);
      }

      // What a sink-unaware kernel produces: ordinary attention, plus the
      // log-sum-exp in FlashInfer's base-2 convention.
      float max_v = -INFINITY;
      for (float s : scores) max_v = std::max(max_v, s);

      double d = 0.0;
      for (float s : scores) d += std::exp(static_cast<double>(s - max_v));

      const int64_t row = t * heads + h;
      lse[static_cast<size_t>(row)] = static_cast<float>(
          (static_cast<double>(max_v) + std::log(d)) / std::log(2.0));

      for (int64_t dim = 0; dim < head_dim; ++dim) {
        double acc = 0.0;
        for (int64_t j = 0; j < context; ++j) {
          acc += std::exp(static_cast<double>(scores[static_cast<size_t>(j)] - max_v)) *
                 values[static_cast<size_t>(j * head_dim + dim)];
        }
        plain[static_cast<size_t>(row * head_dim + dim)] =
            static_cast<float>(acc / d);
      }

      want[static_cast<size_t>(row)] = ReferenceSinkedAttention(
          scores, values, sinks_f[static_cast<size_t>(h)], context, head_dim);
    }
  }

  plain = RoundTrip(plain);

  std::vector<DeviceBuffer> keep;
  const TensorView out = Upload(keep, ToBf16(plain), DataType::kBFloat16,
                                Shape({tokens, heads, head_dim}));
  const TensorView lse_v =
      Upload(keep, lse, DataType::kFloat, Shape({tokens, heads}));
  const TensorView sinks =
      Upload(keep, ToBf16(sinks_f), DataType::kBFloat16, Shape({heads}));

  ASSERT_TRUE(
      kernels::ApplyAttentionSinks(out, lse_v, sinks, /*lse_is_log2=*/true).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got = Download<bf16>(out, tokens * heads * head_dim);

  double worst = 0.0;
  for (int64_t row = 0; row < tokens * heads; ++row) {
    for (int64_t d = 0; d < head_dim; ++d) {
      const float g =
          __bfloat162float(got[static_cast<size_t>(row * head_dim + d)]);
      worst = std::max<double>(
          worst, std::abs(g - want[static_cast<size_t>(row)][static_cast<size_t>(d)]));
    }
  }

  // bf16 in, bf16 out, one multiply: the only error is representation.
  EXPECT_LT(worst, 5e-3) << "worst deviation from the concatenated softmax: "
                         << worst;
}

TEST_F(GptOssTest, ALargeSinkDrivesTheOutputTowardsZero) {
  // The behaviour the sink exists for: a head that prefers to attend nowhere
  // emits nearly nothing. If the rescale were inverted or the log base wrong,
  // this would come out near 1 instead of near 0.
  constexpr int64_t tokens = 1, heads = 1, head_dim = 8;

  const std::vector<float> ones(static_cast<size_t>(head_dim), 1.0f);
  const std::vector<float> lse = {0.0f};   // ln D = 0, so D = 1
  const std::vector<float> sink = {20.0f};

  std::vector<DeviceBuffer> keep;
  const TensorView out = Upload(keep, ToBf16(ones), DataType::kBFloat16,
                                Shape({tokens, heads, head_dim}));
  const TensorView lse_v =
      Upload(keep, lse, DataType::kFloat, Shape({tokens, heads}));
  const TensorView sinks =
      Upload(keep, ToBf16(sink), DataType::kBFloat16, Shape({heads}));

  ASSERT_TRUE(kernels::ApplyAttentionSinks(out, lse_v, sinks, true).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got = Download<bf16>(out, head_dim);
  for (int64_t d = 0; d < head_dim; ++d) {
    EXPECT_LT(std::abs(__bfloat162float(got[static_cast<size_t>(d)])), 1e-6f)
        << "a sink of 20 against lse 0 should annihilate the output";
  }
}

TEST_F(GptOssTest, ANegligibleSinkLeavesTheOutputAlone) {
  constexpr int64_t head_dim = 8;

  const std::vector<float> ones(static_cast<size_t>(head_dim), 1.0f);
  const std::vector<float> lse = {30.0f};   // enormous mass on real keys
  const std::vector<float> sink = {-20.0f};

  std::vector<DeviceBuffer> keep;
  const TensorView out = Upload(keep, ToBf16(ones), DataType::kBFloat16,
                                Shape({1, 1, head_dim}));
  const TensorView lse_v = Upload(keep, lse, DataType::kFloat, Shape({1, 1}));
  const TensorView sinks =
      Upload(keep, ToBf16(sink), DataType::kBFloat16, Shape({1}));

  ASSERT_TRUE(kernels::ApplyAttentionSinks(out, lse_v, sinks, true).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got = Download<bf16>(out, head_dim);
  for (int64_t d = 0; d < head_dim; ++d) {
    EXPECT_NEAR(__bfloat162float(got[static_cast<size_t>(d)]), 1.0f, 1e-3f);
  }
}

// --- The activation ----------------------------------------------------------

TEST_F(GptOssTest, SwiGluMatchesTheReferenceDefinition) {
  constexpr int64_t tokens = 7, inter = 40;
  constexpr float kLimit = 7.0f;
  constexpr float kAlpha = 1.702f;

  // Deliberately wide enough to cross the clamp on both sides -- the clamp is
  // asymmetric (gate one-sided, up two-sided) and an implementation that
  // clamped both the same way would agree everywhere inside ±7 and differ only
  // here.
  std::vector<float> gate_up(static_cast<size_t>(tokens * 2 * inter));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t j = 0; j < 2 * inter; ++j) {
      gate_up[static_cast<size_t>(t * 2 * inter + j)] = 14.0f * Fill(t, j, 0.7f);
    }
  }
  gate_up = RoundTrip(gate_up);

  std::vector<DeviceBuffer> keep;
  const TensorView gu = Upload(keep, ToBf16(gate_up), DataType::kBFloat16,
                               Shape({tokens, 2 * inter}));
  auto out_buf = DeviceBuffer::Allocate(
      DataTypeByteSize(DataType::kBFloat16, tokens * inter), DeviceId::Cuda(0));
  ASSERT_TRUE(out_buf.ok());
  auto out = TensorView::Create(out_buf->data(), DataType::kBFloat16,
                                Shape({tokens, inter}), DeviceId::Cuda(0));
  ASSERT_TRUE(out.ok());

  ASSERT_TRUE(kernels::GptOssSwiGlu(gu, *out, kLimit, kAlpha).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got = Download<bf16>(*out, tokens * inter);

  double worst = 0.0;
  int clamped_gate = 0;
  int clamped_up = 0;

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t j = 0; j < inter; ++j) {
      const float raw_gate = gate_up[static_cast<size_t>(t * 2 * inter + j)];
      const float raw_up = gate_up[static_cast<size_t>(t * 2 * inter + inter + j)];

      if (raw_gate > kLimit) ++clamped_gate;
      if (raw_up > kLimit || raw_up < -kLimit) ++clamped_up;

      const float gate = std::min(raw_gate, kLimit);
      const float up = std::max(std::min(raw_up, kLimit), -kLimit);
      const float glu = gate / (1.0f + std::exp(-gate * kAlpha));
      const float want = (up + 1.0f) * glu;

      worst = std::max<double>(
          worst, std::abs(__bfloat162float(got[static_cast<size_t>(t * inter + j)]) -
                          want));
    }
  }

  EXPECT_GT(clamped_gate, 0) << "the test data never exercises the gate clamp";
  EXPECT_GT(clamped_up, 0) << "the test data never exercises the up clamp";

  // Values reach ~8·(1+7) = 64 here, and bf16's step at 64 is 0.5.
  EXPECT_LT(worst, 0.35) << "worst deviation " << worst;
}

TEST_F(GptOssTest, SwiGluIsNotPlainSiluMul) {
  // A guard against the plausible mistake: reusing SiluMulFused and calling it
  // close enough. On the same input the two must visibly disagree, or this
  // whole kernel is dead code.
  constexpr int64_t inter = 8;

  std::vector<float> gate_up(static_cast<size_t>(2 * inter));
  for (int64_t j = 0; j < 2 * inter; ++j) {
    gate_up[static_cast<size_t>(j)] = 0.5f + 0.25f * static_cast<float>(j);
  }
  gate_up = RoundTrip(gate_up);

  std::vector<DeviceBuffer> keep;
  const TensorView gu = Upload(keep, ToBf16(gate_up), DataType::kBFloat16,
                               Shape({1, 2 * inter}));
  auto out_buf = DeviceBuffer::Allocate(
      DataTypeByteSize(DataType::kBFloat16, inter), DeviceId::Cuda(0));
  ASSERT_TRUE(out_buf.ok());
  auto out = TensorView::Create(out_buf->data(), DataType::kBFloat16,
                                Shape({1, inter}), DeviceId::Cuda(0));
  ASSERT_TRUE(out.ok());

  ASSERT_TRUE(kernels::GptOssSwiGlu(gu, *out, 7.0f, 1.702f).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got = Download<bf16>(*out, inter);

  for (int64_t j = 0; j < inter; ++j) {
    const float gate = gate_up[static_cast<size_t>(j)];
    const float up = gate_up[static_cast<size_t>(inter + j)];
    const float plain_silu = (gate / (1.0f + std::exp(-gate))) * up;

    EXPECT_GT(std::abs(__bfloat162float(got[static_cast<size_t>(j)]) - plain_silu),
              1e-2f)
        << "element " << j << " agrees with plain silu(gate)*up, which gpt-oss "
           "is not";
  }
}

// --- YaRN --------------------------------------------------------------------

TEST(GptOssYarn, MatchesTheReferenceInverseFrequencies) {
  // gpt-oss-20b's own rope_scaling, and the 32 frequencies transformers'
  // _compute_yarn_parameters produces for it. Embedded rather than generated,
  // because they are a fixed property of a released checkpoint and a test that
  // needs a 13 GB download to run is a test that does not run.
  constexpr int64_t kHeadDim = 64;
  constexpr double kBase = 150000.0;
  constexpr double kFactor = 32.0;
  constexpr double kBetaFast = 32.0;
  constexpr double kBetaSlow = 1.0;
  constexpr int64_t kOriginalMax = 4096;

  static constexpr float kExpected[32] = {
      1.000000000e+00f, 6.890442967e-01f, 4.747820497e-01f, 3.271458745e-01f,
      2.254180014e-01f, 1.553229839e-01f, 1.070244238e-01f, 7.374456525e-02f,
      5.081327260e-02f, 3.170569614e-02f, 1.933499984e-02f, 1.159204915e-02f,
      6.794959307e-03f, 3.860359080e-03f, 2.093792660e-03f, 1.052602194e-03f,
      4.564839182e-04f, 1.293186942e-04f, 3.830881178e-05f, 2.639646846e-05f,
      1.818833698e-05f, 1.253256960e-05f, 8.635495760e-06f, 5.950239483e-06f,
      4.099978469e-06f, 2.825066758e-06f, 1.946596285e-06f, 1.341290954e-06f,
      9.242089618e-07f, 6.368209142e-07f, 4.387978549e-07f, 3.023511397e-07f,
  };

  std::vector<float> got(kHeadDim / 2);
  const float attn_factor = kernels::ComputeYarnInvFreq(
      kHeadDim, kBase, kFactor, kBetaFast, kBetaSlow, kOriginalMax,
      /*truncate=*/false, got.data());

  // 0.1·ln(32) + 1. Reached by falling through a default rather than by being
  // configured, which is exactly why it is easy to miss.
  EXPECT_NEAR(attn_factor, 1.3465735903f, 1e-6f);

  for (size_t i = 0; i < got.size(); ++i) {
    // Relative, because these span seven orders of magnitude and an absolute
    // bound would be vacuous at the small end.
    EXPECT_NEAR(got[i] / kExpected[i], 1.0f, 1e-5f)
        << "inv_freq[" << i << "]: got " << got[i] << ", want " << kExpected[i];
  }
}

TEST(GptOssYarn, DegeneratesToPlainRopeWhenNotScaling) {
  // factor 1 with a window that covers everything: YaRN should reduce to
  // theta^(-2j/d), the closed form RotaryEmbedding computes inline. If it does
  // not, the blend is wrong in a way the gpt-oss numbers above might not show.
  constexpr int64_t kHeadDim = 16;
  constexpr double kBase = 10000.0;

  std::vector<float> got(kHeadDim / 2);
  kernels::ComputeYarnInvFreq(kHeadDim, kBase, /*factor=*/1.0,
                              /*beta_fast=*/32.0, /*beta_slow=*/1.0,
                              /*original_max=*/4096, /*truncate=*/false,
                              got.data());

  for (int64_t j = 0; j < kHeadDim / 2; ++j) {
    const double want = std::pow(
        kBase, -2.0 * static_cast<double>(j) / static_cast<double>(kHeadDim));
    EXPECT_NEAR(got[static_cast<size_t>(j)] / want, 1.0, 1e-6)
        << "j = " << j;
  }
}

TEST_F(GptOssTest, RopeFromTableWithFactorOneMatchesTheInlineFormula) {
  // The table-driven kernel has to be the same rotation as RotaryEmbedding,
  // just with the frequencies handed in. Checked at attention factor 1 against
  // the closed form, so gpt-oss's YaRN table is the only thing left to trust.
  constexpr int64_t tokens = 4, q_heads = 3, kv_heads = 1, head_dim = 8;
  constexpr float kTheta = 10000.0f;

  std::vector<float> q(static_cast<size_t>(tokens * q_heads * head_dim));
  std::vector<float> k(static_cast<size_t>(tokens * kv_heads * head_dim));
  for (size_t i = 0; i < q.size(); ++i) q[i] = Fill(static_cast<int64_t>(i), 1, 0.2f);
  for (size_t i = 0; i < k.size(); ++i) k[i] = Fill(static_cast<int64_t>(i), 2, 1.4f);
  q = RoundTrip(q);
  k = RoundTrip(k);

  const std::vector<int32_t> positions = {0, 1, 7, 13};

  std::vector<float> inv_freq(head_dim / 2);
  for (int64_t j = 0; j < head_dim / 2; ++j) {
    inv_freq[static_cast<size_t>(j)] = static_cast<float>(std::pow(
        static_cast<double>(kTheta),
        -2.0 * static_cast<double>(j) / static_cast<double>(head_dim)));
  }

  std::vector<DeviceBuffer> keep;
  const TensorView q_v = Upload(keep, ToBf16(q), DataType::kBFloat16,
                                Shape({tokens, q_heads, head_dim}));
  const TensorView k_v = Upload(keep, ToBf16(k), DataType::kBFloat16,
                                Shape({tokens, kv_heads, head_dim}));
  const TensorView pos =
      Upload(keep, positions, DataType::kInt32, Shape({tokens}));
  const TensorView freq =
      Upload(keep, inv_freq, DataType::kFloat, Shape({head_dim / 2}));

  ASSERT_TRUE(kernels::RotaryEmbeddingFromTable(q_v, k_v, pos, freq,
                                                /*attn_factor=*/1.0f)
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got = Download<bf16>(q_v, tokens * q_heads * head_dim);

  const int64_t half = head_dim / 2;
  double worst = 0.0;

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < q_heads; ++h) {
      for (int64_t j = 0; j < half; ++j) {
        const double angle = positions[static_cast<size_t>(t)] *
                             inv_freq[static_cast<size_t>(j)];
        const double s = std::sin(angle);
        const double c = std::cos(angle);

        const size_t base = static_cast<size_t>((t * q_heads + h) * head_dim);
        const double lo = q[base + static_cast<size_t>(j)];
        const double hi = q[base + static_cast<size_t>(j + half)];

        worst = std::max(
            worst, std::abs(__bfloat162float(got[base + static_cast<size_t>(j)]) -
                            (lo * c - hi * s)));
        worst = std::max(
            worst,
            std::abs(__bfloat162float(got[base + static_cast<size_t>(j + half)]) -
                     (hi * c + lo * s)));
      }
    }
  }

  EXPECT_LT(worst, 5e-3) << "worst deviation " << worst;
}

}  // namespace
}  // namespace inferx
