// Every kernel here is checked against an independent fp64 CPU implementation
// written from the formula rather than from the kernel. That is the point: a
// device kernel and a host reference that share code share bugs.

#include "inferx/kernels/layers.h"

#include <cmath>
#include <set>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"

namespace inferx {
namespace {

using bf16 = __nv_bfloat16;

// bf16 keeps 8 mantissa bits, so a value round-trips to ~0.4% relative. The
// tolerances below are stated per test against that, not chosen until green.
constexpr double kBf16Eps = 1.0 / 256.0;

class Device {
 public:
  TensorView Upload(const std::vector<float>& host, const Shape& shape) {
    std::vector<bf16> b(host.size());
    for (size_t i = 0; i < host.size(); ++i) b[i] = __float2bfloat16(host[i]);

    auto buf = DeviceBuffer::Allocate(b.size() * sizeof(bf16),
                                      DeviceId::Cuda(0));
    EXPECT_TRUE(buf.ok()) << buf.status();
    EXPECT_EQ(cudaMemcpy(buf->data(), b.data(), b.size() * sizeof(bf16),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    bufs_.push_back(*std::move(buf));
    auto v = TensorView::Create(bufs_.back().data(), DataType::kBFloat16, shape,
                                DeviceId::Cuda(0));
    EXPECT_TRUE(v.ok()) << v.status();
    return *v;
  }

  TensorView UploadInt32(const std::vector<int32_t>& host, const Shape& shape) {
    auto buf = DeviceBuffer::Allocate(host.size() * sizeof(int32_t),
                                      DeviceId::Cuda(0));
    EXPECT_TRUE(buf.ok()) << buf.status();
    EXPECT_EQ(cudaMemcpy(buf->data(), host.data(),
                         host.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
              cudaSuccess);

    bufs_.push_back(*std::move(buf));
    auto v = TensorView::Create(bufs_.back().data(), DataType::kInt32, shape,
                                DeviceId::Cuda(0));
    EXPECT_TRUE(v.ok()) << v.status();
    return *v;
  }

  // Typed uploads for the sampler, whose inputs are not all bf16: per-row
  // temperature and top_p are float, and seeds are 64-bit.
  template <typename T>
  TensorView Typed(const std::vector<T>& host, const Shape& shape,
                   DataType dtype) {
    auto buf = DeviceBuffer::Allocate(host.size() * sizeof(T),
                                      DeviceId::Cuda(0));
    EXPECT_TRUE(buf.ok()) << buf.status();
    EXPECT_EQ(cudaMemcpy(buf->data(), host.data(), host.size() * sizeof(T),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    bufs_.push_back(*std::move(buf));
    auto v = TensorView::Create(bufs_.back().data(), dtype, shape,
                                DeviceId::Cuda(0));
    EXPECT_TRUE(v.ok()) << v.status();
    return *v;
  }

  TensorView I32(const std::vector<int32_t>& h, const Shape& s) {
    return Typed(h, s, DataType::kInt32);
  }
  TensorView F32(const std::vector<float>& h, const Shape& s) {
    return Typed(h, s, DataType::kFloat);
  }
  TensorView U64(const std::vector<uint64_t>& h, const Shape& s) {
    return Typed(h, s, DataType::kUInt64);
  }

  TensorView EmptyI32(const Shape& shape) {
    return Typed(std::vector<int32_t>(static_cast<size_t>(shape.Numel()), 0),
                 shape, DataType::kInt32);
  }

  std::vector<int32_t> DownloadI32(const TensorView& t) {
    std::vector<int32_t> out(static_cast<size_t>(t.Numel()));
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(out.data(), t.Data(), out.size() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return out;
  }

  TensorView Empty(const Shape& shape) {
    auto buf = DeviceBuffer::Allocate(
        static_cast<size_t>(shape.Numel()) * sizeof(bf16), DeviceId::Cuda(0));
    EXPECT_TRUE(buf.ok()) << buf.status();

    bufs_.push_back(*std::move(buf));
    auto v = TensorView::Create(bufs_.back().data(), DataType::kBFloat16, shape,
                                DeviceId::Cuda(0));
    EXPECT_TRUE(v.ok()) << v.status();
    return *v;
  }

  std::vector<float> Download(const TensorView& t) {
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
  std::vector<DeviceBuffer> bufs_;
};

// Rounds through bf16 so references compare against what the kernel could
// possibly have read, not against fp32 inputs it never saw.
float Bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

std::vector<float> Ramp(size_t n, float scale, float phase) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = scale * std::sin(static_cast<float>(i) * 0.37f + phase);
  }
  return v;
}

class LayersTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }
  Device dev;
};

TEST_F(LayersTest, RmsNormMatchesReference) {
  constexpr int64_t tokens = 5, hidden = 256;

  const std::vector<float> x = Ramp(tokens * hidden, 1.5f, 0.0f);
  const std::vector<float> w = Ramp(hidden, 0.8f, 2.1f);

  const TensorView xd = dev.Upload(x, Shape({tokens, hidden}));
  const TensorView wd = dev.Upload(w, Shape({hidden}));
  const TensorView od = dev.Empty(Shape({tokens, hidden}));

  const float eps = 1e-6f;
  ASSERT_TRUE(kernels::RmsNorm(xd, wd, od, eps).ok());

  const std::vector<float> got = dev.Download(od);

  for (int64_t t = 0; t < tokens; ++t) {
    double sumsq = 0.0;
    for (int64_t i = 0; i < hidden; ++i) {
      const double v = Bf16(x[t * hidden + i]);
      sumsq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sumsq / hidden + eps);

    for (int64_t i = 0; i < hidden; ++i) {
      const double want = Bf16(x[t * hidden + i]) * inv * Bf16(w[i]);
      EXPECT_NEAR(got[t * hidden + i], want, std::abs(want) * 3 * kBf16Eps + 1e-3)
          << "t=" << t << " i=" << i;
    }
  }
}

// The formula, spelled out independently of the kernel. If both were written
// from the same mental model this test would be worthless; the half-split
// indexing is exactly the thing being checked.
TEST_F(LayersTest, RopeMatchesTheHalfSplitFormula) {
  constexpr int64_t tokens = 4, q_heads = 2, kv_heads = 1, head_dim = 16;
  constexpr float theta = 1000000.0f;

  const std::vector<float> q = Ramp(tokens * q_heads * head_dim, 1.0f, 0.0f);
  const std::vector<float> k = Ramp(tokens * kv_heads * head_dim, 1.0f, 1.3f);
  const std::vector<int32_t> pos = {0, 1, 2, 7};

  const TensorView qd = dev.Upload(q, Shape({tokens, q_heads, head_dim}));
  const TensorView kd = dev.Upload(k, Shape({tokens, kv_heads, head_dim}));
  const TensorView pd = dev.UploadInt32(pos, Shape({tokens}));

  ASSERT_TRUE(kernels::RotaryEmbedding(qd, kd, pd, theta).ok());

  const std::vector<float> got_q = dev.Download(qd);
  const int64_t half = head_dim / 2;

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < q_heads; ++h) {
      const int64_t base = (t * q_heads + h) * head_dim;

      for (int64_t j = 0; j < half; ++j) {
        const double inv_freq =
            std::pow(static_cast<double>(theta),
                     -2.0 * static_cast<double>(j) / head_dim);
        const double angle = pos[t] * inv_freq;
        const double c = std::cos(angle), s = std::sin(angle);

        const double lo = Bf16(q[base + j]);
        const double hi = Bf16(q[base + j + half]);

        EXPECT_NEAR(got_q[base + j], lo * c - hi * s, 0.02)
            << "t=" << t << " h=" << h << " j=" << j;
        EXPECT_NEAR(got_q[base + j + half], hi * c + lo * s, 0.02)
            << "t=" << t << " h=" << h << " j=" << j;
      }
    }
  }
}

// Position 0 must be the identity: cos(0)=1, sin(0)=0. A kernel that has the
// rotation direction backwards still passes many tests but fails this one only
// if it also has an offset bug -- so this is the cheap sanity check, not the
// real one above.
TEST_F(LayersTest, RopeAtPositionZeroIsIdentity) {
  constexpr int64_t tokens = 1, heads = 1, head_dim = 8;

  const std::vector<float> q = {0.5f, -1.0f, 2.0f, 0.25f,
                                -0.5f, 1.5f, -2.0f, 0.75f};
  const TensorView qd = dev.Upload(q, Shape({tokens, heads, head_dim}));
  const TensorView kd = dev.Upload(q, Shape({tokens, heads, head_dim}));
  const TensorView pd = dev.UploadInt32({0}, Shape({tokens}));

  ASSERT_TRUE(kernels::RotaryEmbedding(qd, kd, pd, 10000.0f).ok());

  const std::vector<float> got = dev.Download(qd);
  for (size_t i = 0; i < q.size(); ++i) EXPECT_NEAR(got[i], q[i], 1e-3) << i;
}

TEST_F(LayersTest, SiluMulMatchesReference) {
  constexpr int64_t tokens = 3, inter = 128;

  const std::vector<float> g = Ramp(tokens * inter, 3.0f, 0.0f);
  const std::vector<float> u = Ramp(tokens * inter, 2.0f, 1.7f);

  const TensorView gd = dev.Upload(g, Shape({tokens, inter}));
  const TensorView ud = dev.Upload(u, Shape({tokens, inter}));
  const TensorView od = dev.Empty(Shape({tokens, inter}));

  ASSERT_TRUE(kernels::SiluMul(gd, ud, od).ok());

  const std::vector<float> got = dev.Download(od);

  for (size_t i = 0; i < got.size(); ++i) {
    const double gv = Bf16(g[i]);
    const double want = (gv / (1.0 + std::exp(-gv))) * Bf16(u[i]);
    EXPECT_NEAR(got[i], want, std::abs(want) * 3 * kBf16Eps + 2e-2) << i;
  }
}

// Causal masking, softmax and the GQA head mapping, all against a reference
// that recomputes them from scratch in fp64.
TEST_F(LayersTest, AttentionMatchesReferenceWithGqa) {
  constexpr int64_t tokens = 6, q_heads = 4, kv_heads = 2, head_dim = 32;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  const std::vector<float> q = Ramp(tokens * q_heads * head_dim, 1.0f, 0.0f);
  const std::vector<float> k = Ramp(tokens * kv_heads * head_dim, 1.0f, 0.9f);
  const std::vector<float> v = Ramp(tokens * kv_heads * head_dim, 1.0f, 2.4f);

  const TensorView qd = dev.Upload(q, Shape({tokens, q_heads, head_dim}));
  const TensorView kd = dev.Upload(k, Shape({tokens, kv_heads, head_dim}));
  const TensorView vd = dev.Upload(v, Shape({tokens, kv_heads, head_dim}));
  const TensorView od = dev.Empty(Shape({tokens, q_heads, head_dim}));

  ASSERT_TRUE(kernels::Attention(qd, kd, vd, od, scale).ok());

  const std::vector<float> got = dev.Download(od);
  const int64_t group = q_heads / kv_heads;

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < q_heads; ++h) {
      const int64_t kvh = h / group;

      // Causal: keys 0..t inclusive.
      std::vector<double> scores(t + 1);
      double max_score = -1e300;

      for (int64_t j = 0; j <= t; ++j) {
        double dot = 0.0;
        for (int64_t d = 0; d < head_dim; ++d) {
          dot += Bf16(q[(t * q_heads + h) * head_dim + d]) *
                 Bf16(k[(j * kv_heads + kvh) * head_dim + d]);
        }
        scores[j] = dot * scale;
        max_score = std::max(max_score, scores[j]);
      }

      double sum = 0.0;
      for (double& s : scores) {
        s = std::exp(s - max_score);
        sum += s;
      }

      for (int64_t d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int64_t j = 0; j <= t; ++j) {
          acc += scores[j] * Bf16(v[(j * kv_heads + kvh) * head_dim + d]);
        }
        const double want = acc / sum;

        EXPECT_NEAR(got[(t * q_heads + h) * head_dim + d], want, 3e-2)
            << "t=" << t << " h=" << h << " d=" << d;
      }
    }
  }
}

// The first token can only attend to itself, so its output is exactly V[0]
// regardless of the scores. A masking bug that lets token 0 see the future
// fails here immediately.
TEST_F(LayersTest, AttentionIsCausal) {
  constexpr int64_t tokens = 4, heads = 1, head_dim = 16;

  const std::vector<float> q = Ramp(tokens * heads * head_dim, 1.0f, 0.0f);
  const std::vector<float> k = Ramp(tokens * heads * head_dim, 1.0f, 0.5f);

  std::vector<float> v(tokens * heads * head_dim, 0.0f);
  for (int64_t d = 0; d < head_dim; ++d) v[d] = 1.0f;  // token 0's V is all 1s
  for (int64_t t = 1; t < tokens; ++t) {
    for (int64_t d = 0; d < head_dim; ++d) v[t * head_dim + d] = 99.0f;
  }

  const TensorView qd = dev.Upload(q, Shape({tokens, heads, head_dim}));
  const TensorView kd = dev.Upload(k, Shape({tokens, heads, head_dim}));
  const TensorView vd = dev.Upload(v, Shape({tokens, heads, head_dim}));
  const TensorView od = dev.Empty(Shape({tokens, heads, head_dim}));

  ASSERT_TRUE(kernels::Attention(qd, kd, vd, od, 0.125f).ok());

  const std::vector<float> got = dev.Download(od);

  // If the future leaked in, these would be pulled toward 99.
  for (int64_t d = 0; d < head_dim; ++d) EXPECT_NEAR(got[d], 1.0f, 1e-2) << d;
}

TEST_F(LayersTest, EmbeddingLookupGathersRows) {
  constexpr int64_t vocab = 32, hidden = 64;

  const std::vector<float> table = Ramp(vocab * hidden, 1.0f, 0.0f);
  const std::vector<int32_t> ids = {5, 0, 31, 17};

  const TensorView td = dev.Upload(table, Shape({vocab, hidden}));
  const TensorView idd = dev.UploadInt32(ids, Shape({4}));
  const TensorView od = dev.Empty(Shape({4, hidden}));

  ASSERT_TRUE(kernels::EmbeddingLookup(td, idd, od).ok());

  const std::vector<float> got = dev.Download(od);

  for (size_t t = 0; t < ids.size(); ++t) {
    for (int64_t i = 0; i < hidden; ++i) {
      EXPECT_FLOAT_EQ(got[t * hidden + i], Bf16(table[ids[t] * hidden + i]))
          << "t=" << t << " i=" << i;
    }
  }
}

TEST_F(LayersTest, AddInPlaceAccumulates) {
  constexpr int64_t tokens = 2, hidden = 32;

  const std::vector<float> a = Ramp(tokens * hidden, 1.0f, 0.0f);
  const std::vector<float> b = Ramp(tokens * hidden, 1.0f, 1.1f);

  const TensorView ad = dev.Upload(a, Shape({tokens, hidden}));
  const TensorView bd = dev.Upload(b, Shape({tokens, hidden}));

  ASSERT_TRUE(kernels::AddInPlace(ad, bd).ok());

  const std::vector<float> got = dev.Download(ad);
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], Bf16(a[i]) + Bf16(b[i]), 1e-2) << i;
  }
}

TEST_F(LayersTest, ShapeMismatchesAreRejected) {
  const TensorView x = dev.Empty(Shape({2, 16}));
  const TensorView w = dev.Empty(Shape({8}));  // wrong width
  const TensorView o = dev.Empty(Shape({2, 16}));

  EXPECT_EQ(kernels::RmsNorm(x, w, o, 1e-6f).code(),
            absl::StatusCode::kInvalidArgument);

  const TensorView q = dev.Empty(Shape({4, 6, 16}));
  const TensorView k = dev.Empty(Shape({4, 4, 16}));  // 6 % 4 != 0
  const TensorView v = dev.Empty(Shape({4, 4, 16}));
  const TensorView out = dev.Empty(Shape({4, 6, 16}));

  EXPECT_EQ(kernels::Attention(q, k, v, out, 0.25f).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace

// Temperature and nucleus sampling.
//
// A sampler is the one kernel whose output is *supposed* to vary, which makes
// it easy to ship broken -- "different every time" is both the correct
// behaviour and the symptom of a bug. So every property here is one that must
// hold despite the randomness: the greedy limit, reproducibility from a seed,
// the nucleus actually truncating, and the empirical distribution matching the
// softmax it claims to draw from.
class SamplingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }

  // Draws `n` samples from one logits row, one per call, varying only the seed.
  std::vector<int32_t> Draw(const std::vector<float>& row, float temperature,
                            float top_p, int n, uint64_t seed0 = 1) {
    Device d;
    const int64_t vocab = static_cast<int64_t>(row.size());

    std::vector<float> flat(row);
    const TensorView logits = d.Upload(flat, Shape({1, vocab}));

    std::vector<int32_t> rows(static_cast<size_t>(n), 0);
    std::vector<float> temps(static_cast<size_t>(n), temperature);
    std::vector<float> ps(static_cast<size_t>(n), top_p);
    std::vector<uint64_t> seeds(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) seeds[static_cast<size_t>(i)] = seed0 + i;

    const TensorView out = d.EmptyI32(Shape({n}));

    EXPECT_TRUE(kernels::SampleTokens(
                    logits, d.I32(rows, Shape({n})),
                    d.F32(temps, Shape({n})), d.F32(ps, Shape({n})),
                    d.U64(seeds, Shape({n})), out)
                    .ok());

    return d.DownloadI32(out);
  }
};

TEST_F(SamplingTest, TemperatureZeroIsExactlyArgmax) {
  // Greedy is the temperature -> 0 limit, and the path most requests take, so
  // it has to be exact rather than merely likely.
  std::vector<float> row(512, 0.0f);
  for (size_t i = 0; i < row.size(); ++i) {
    row[i] = std::sin(static_cast<float>(i) * 0.37f);
  }
  row[311] = 5.0f;  // a clear winner

  for (const int32_t id : Draw(row, /*temperature=*/0.0f, /*top_p=*/1.0f, 8)) {
    EXPECT_EQ(id, 311);
  }
}

TEST_F(SamplingTest, SameSeedGivesSameToken) {
  // Reproducibility is what makes a sampled server debuggable at all: a bug
  // report with a seed has to be replayable.
  std::vector<float> row(1024);
  for (size_t i = 0; i < row.size(); ++i) {
    row[i] = std::sin(static_cast<float>(i) * 0.11f) * 2.0f;
  }

  const std::vector<int32_t> a = Draw(row, 1.0f, 1.0f, 16, /*seed0=*/12345);
  const std::vector<int32_t> b = Draw(row, 1.0f, 1.0f, 16, /*seed0=*/12345);

  EXPECT_EQ(a, b);
}

TEST_F(SamplingTest, DifferentSeedsExploreTheDistribution) {
  // The other half of the same property: a sampler that ignores its seed and
  // returns the argmax would pass the test above and be useless.
  std::vector<float> row(1024);
  for (size_t i = 0; i < row.size(); ++i) {
    row[i] = std::sin(static_cast<float>(i) * 0.11f) * 2.0f;
  }

  const std::vector<int32_t> draws = Draw(row, 1.0f, 1.0f, 64, /*seed0=*/900);
  const std::set<int32_t> distinct(draws.begin(), draws.end());

  EXPECT_GT(distinct.size(), 3u)
      << "64 draws produced only " << distinct.size()
      << " distinct tokens; the seed is not reaching the draw";
}

TEST_F(SamplingTest, NucleusExcludesTheTail) {
  // A distribution with two dominant tokens holding ~99% of the mass. top_p
  // 0.9 must never return anything else, however many times it is asked.
  std::vector<float> row(4096, -8.0f);
  row[7] = 4.0f;
  row[9] = 4.0f;

  for (const int32_t id : Draw(row, 1.0f, /*top_p=*/0.9f, 128)) {
    EXPECT_TRUE(id == 7 || id == 9)
        << "top_p 0.9 sampled token " << id << " from the tail";
  }
}

TEST_F(SamplingTest, TopPOneKeepsTheWholeDistribution) {
  // The complement: with truncation disabled, low-probability tokens must
  // still be reachable, or top_p is silently always on.
  std::vector<float> row(64, 0.0f);
  row[0] = 1.0f;

  const std::vector<int32_t> draws = Draw(row, 1.0f, 1.0f, 256, /*seed0=*/5);
  const std::set<int32_t> distinct(draws.begin(), draws.end());

  EXPECT_GT(distinct.size(), 20u)
      << "a near-uniform 64-way distribution produced only " << distinct.size()
      << " distinct tokens with top_p = 1";
}

TEST_F(SamplingTest, EmpiricalFrequenciesMatchTheSoftmax) {
  // The property that says this is sampling rather than just varying: over
  // many draws the frequencies must track the softmax it claims to draw from.
  // Four tokens with known probabilities, and a tolerance derived from the
  // binomial standard error rather than picked -- 4 sigma at n = 4000 is about
  // 3 percentage points for p near 0.25.
  constexpr int kDraws = 4000;

  std::vector<float> row(4, 0.0f);
  row[0] = std::log(4.0f);
  row[1] = std::log(3.0f);
  row[2] = std::log(2.0f);
  row[3] = std::log(1.0f);

  const std::vector<float> expected = {0.4f, 0.3f, 0.2f, 0.1f};

  const std::vector<int32_t> draws = Draw(row, 1.0f, 1.0f, kDraws, 777);

  std::vector<int> counts(4, 0);
  for (const int32_t id : draws) {
    ASSERT_GE(id, 0);
    ASSERT_LT(id, 4);
    ++counts[static_cast<size_t>(id)];
  }

  for (size_t i = 0; i < expected.size(); ++i) {
    const double got = static_cast<double>(counts[i]) / kDraws;
    const double se = std::sqrt(expected[i] * (1.0 - expected[i]) / kDraws);

    std::fprintf(stderr, "  token %zu: expected %.3f got %.3f (4 sigma %.3f)\n",
                 i, expected[i], got, 4.0 * se);

    EXPECT_NEAR(got, expected[i], 4.0 * se);
  }
}

}  // namespace inferx
