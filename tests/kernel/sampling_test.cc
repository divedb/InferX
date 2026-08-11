/// The device sampler, against references computed on the host.
///
/// Sampling is the one kernel whose output is deliberately random, so
/// "correct" needs care: where a parameter makes the draw deterministic
/// (temperature 0, top_k 1, min_p 1, a nucleus of one) the sampled id must
/// equal the host argmax exactly; where it stays random, the sampled id must
/// land inside the host-computed support set for every seed tried, and the
/// same seed must reproduce the same draw. Penalties and logprobs are
/// deterministic arithmetic and get compared value by value.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"
#include "inferx/ops/layers.h"

namespace inferx {
namespace {

using bf16 = __nv_bfloat16;

std::vector<bf16> ToBf16(const std::vector<float>& host) {
  std::vector<bf16> out(host.size());
  for (size_t i = 0; i < host.size(); ++i) out[i] = __float2bfloat16(host[i]);
  return out;
}

// Round-trips through bf16 so the reference sees the same values the device
// does. Without this every comparison would also be measuring the input's
// quantization error, which is not what is under test.
std::vector<float> RoundTrip(const std::vector<float>& host) {
  std::vector<float> out(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    out[i] = __bfloat162float(__float2bfloat16(host[i]));
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

TensorView Empty(std::vector<DeviceBuffer>& keep, DataType dtype,
                 const Shape& shape) {
  const size_t bytes = DataTypeByteSize(dtype, shape.Numel());

  auto buf = DeviceBuffer::Allocate(bytes, DeviceId::Cuda(0));
  EXPECT_TRUE(buf.ok()) << buf.status();
  EXPECT_EQ(cudaMemset(buf->data(), 0, bytes), cudaSuccess);

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

// --- Host references ---------------------------------------------------------

// The argmax the greedy path must reproduce: largest value, lowest index on a
// tie. Inputs are the bf16-round-tripped floats the device reads.
int32_t ReferenceArgmax(const std::vector<float>& row) {
  int32_t best = 0;
  for (size_t i = 1; i < row.size(); ++i) {
    if (row[i] > row[static_cast<size_t>(best)]) best = static_cast<int32_t>(i);
  }
  return best;
}

// Temperature-scaled softmax in double, off the round-tripped logits.
std::vector<double> ReferenceSoftmax(const std::vector<float>& row, float t) {
  std::vector<double> probs(row.size());
  double max_v = -INFINITY;
  for (const float v : row) {
    max_v = std::max(max_v, static_cast<double>(v) / t);
  }
  double sum = 0.0;
  for (size_t i = 0; i < row.size(); ++i) {
    probs[i] = std::exp(static_cast<double>(row[i]) / t - max_v);
    sum += probs[i];
  }
  for (double& p : probs) p /= sum;
  return probs;
}

// The top-p support set the way the kernel defines it: every token whose
// probability is at least that of the last member of the minimal descending
// prefix summing to p. A hair of slack on the cutoff, since the device
// resolves it by bisection over __expf-computed probabilities -- slack only
// grows the allowed set, so membership assertions stay valid.
std::set<int32_t> ReferenceNucleus(const std::vector<double>& probs, double p) {
  std::vector<int32_t> order(probs.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    if (probs[static_cast<size_t>(a)] != probs[static_cast<size_t>(b)]) {
      return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)];
    }
    return a < b;
  });

  double mass = 0.0;
  double cutoff = 0.0;
  for (const int32_t i : order) {
    mass += probs[static_cast<size_t>(i)];
    cutoff = probs[static_cast<size_t>(i)];
    if (mass >= p) break;
  }

  std::set<int32_t> nucleus;
  for (size_t i = 0; i < probs.size(); ++i) {
    if (probs[i] >= cutoff * (1.0 - 1e-3)) {
      nucleus.insert(static_cast<int32_t>(i));
    }
  }
  return nucleus;
}

// The k most probable ids. Probabilities are monotone in the logits, so this
// is a plain sort of the round-tripped row.
std::set<int32_t> ReferenceTopK(const std::vector<float>& row, int k) {
  std::vector<int32_t> order(row.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    if (row[static_cast<size_t>(a)] != row[static_cast<size_t>(b)]) {
      return row[static_cast<size_t>(a)] > row[static_cast<size_t>(b)];
    }
    return a < b;
  });
  return std::set<int32_t>(order.begin(), order.begin() + k);
}

// One sampling launch where every entry draws from `logits_row` of the logits
// tensor, one seed per entry. This is how "over many seeds" is spelled: n
// blocks against the same row in a single launch.
std::vector<int32_t> SampleManySeeds(const TensorView& logits,
                                     int32_t logits_row, int n, float t,
                                     float p, int32_t k, float mp,
                                     uint64_t seed_base) {
  std::vector<DeviceBuffer> keep;

  const std::vector<int32_t> rows_h(static_cast<size_t>(n), logits_row);
  const std::vector<float> temp_h(static_cast<size_t>(n), t);
  const std::vector<float> top_p_h(static_cast<size_t>(n), p);
  const std::vector<int32_t> top_k_h(static_cast<size_t>(n), k);
  const std::vector<float> min_p_h(static_cast<size_t>(n), mp);
  std::vector<uint64_t> seeds_h(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    seeds_h[static_cast<size_t>(i)] =
        seed_base + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(i);
  }

  const TensorView rows = Upload(keep, rows_h, DataType::kInt32, Shape({n}));
  const TensorView temp = Upload(keep, temp_h, DataType::kFloat, Shape({n}));
  const TensorView top_p = Upload(keep, top_p_h, DataType::kFloat, Shape({n}));
  const TensorView top_k = Upload(keep, top_k_h, DataType::kInt32, Shape({n}));
  const TensorView min_p = Upload(keep, min_p_h, DataType::kFloat, Shape({n}));
  const TensorView seeds =
      Upload(keep, seeds_h, DataType::kUInt64, Shape({n}));
  const TensorView out = Empty(keep, DataType::kInt32, Shape({n}));

  EXPECT_TRUE(
      ops::SampleTokens(logits, rows, temp, top_p, top_k, min_p, seeds, out)
          .ok());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  return Download<int32_t>(out, n);
}

class SamplingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!cuda::Available()) GTEST_SKIP() << "no CUDA device available";
  }
};

// --- SampleTokens ------------------------------------------------------------

TEST_F(SamplingTest, GreedyMatchesHostArgmax) {
  constexpr int64_t vocab = 128;

  // Three rows, each with its peak somewhere different; the middle row is
  // skipped, so `rows` indirection is exercised too.
  std::vector<float> logits(static_cast<size_t>(3 * vocab));
  for (int64_t r = 0; r < 3; ++r) {
    for (int64_t i = 0; i < vocab; ++i) {
      logits[static_cast<size_t>(r * vocab + i)] =
          std::sin(0.37f * static_cast<float>(i) + static_cast<float>(r));
    }
    logits[static_cast<size_t>(r * vocab + 17 + 31 * r)] = 5.0f;
  }
  logits = RoundTrip(logits);

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({3, vocab}));

  const std::vector<int32_t> rows_h = {0, 2};
  // Zero and negative both mean greedy; the doc calls it the limit, not a
  // special value.
  const std::vector<float> temp_h = {0.0f, -1.0f};
  const std::vector<float> top_p_h = {1.0f, 1.0f};
  const std::vector<int32_t> top_k_h = {0, 0};
  const std::vector<float> min_p_h = {0.0f, 0.0f};
  const std::vector<uint64_t> seeds_h = {1, 2};

  const TensorView rows = Upload(keep, rows_h, DataType::kInt32, Shape({2}));
  const TensorView temp = Upload(keep, temp_h, DataType::kFloat, Shape({2}));
  const TensorView top_p = Upload(keep, top_p_h, DataType::kFloat, Shape({2}));
  const TensorView top_k = Upload(keep, top_k_h, DataType::kInt32, Shape({2}));
  const TensorView min_p = Upload(keep, min_p_h, DataType::kFloat, Shape({2}));
  const TensorView seeds =
      Upload(keep, seeds_h, DataType::kUInt64, Shape({2}));
  const TensorView out = Empty(keep, DataType::kInt32, Shape({2}));

  ASSERT_TRUE(ops::SampleTokens(logits_v, rows, temp, top_p, top_k, min_p,
                                    seeds, out)
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got = Download<int32_t>(out, 2);
  for (int i = 0; i < 2; ++i) {
    const int64_t r = rows_h[static_cast<size_t>(i)];
    const std::vector<float> row(
        logits.begin() + r * vocab, logits.begin() + (r + 1) * vocab);
    EXPECT_EQ(got[static_cast<size_t>(i)], ReferenceArgmax(row))
        << "entry " << i;
  }
}

TEST_F(SamplingTest, SameSeedReproducesTheSameDraw) {
  constexpr int64_t vocab = 128;

  std::vector<float> logits(static_cast<size_t>(vocab));
  for (int64_t i = 0; i < vocab; ++i) {
    logits[static_cast<size_t>(i)] =
        2.0f * std::sin(0.53f * static_cast<float>(i));
  }
  logits = RoundTrip(logits);

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({1, vocab}));

  const auto first = SampleManySeeds(logits_v, 0, 16, /*t=*/0.8f, /*p=*/1.0f,
                                     /*k=*/0, /*mp=*/0.0f, /*seed_base=*/7);
  const auto second = SampleManySeeds(logits_v, 0, 16, 0.8f, 1.0f, 0, 0.0f, 7);

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(first[static_cast<size_t>(i)], second[static_cast<size_t>(i)])
        << "seed slot " << i << " did not reproduce";
  }

  // A degenerate distribution would make reproducibility vacuous -- 16 seeds
  // over a flat-ish 128-way softmax must not all land on one token.
  EXPECT_GT(std::set<int32_t>(first.begin(), first.end()).size(), 1u);
}

TEST_F(SamplingTest, TopPWithDominantTokenIsDeterministic) {
  constexpr int64_t vocab = 128;

  // Token 42 holds ~0.96 of the mass at temperature 1, so a nucleus of 0.5 is
  // exactly {42} and every seed must return it.
  std::vector<float> logits(static_cast<size_t>(vocab));
  for (int64_t i = 0; i < vocab; ++i) {
    logits[static_cast<size_t>(i)] =
        0.2f * std::sin(0.91f * static_cast<float>(i));
  }
  logits[42] = 8.0f;
  logits = RoundTrip(logits);

  const std::vector<double> probs = ReferenceSoftmax(logits, 1.0f);
  ASSERT_GT(probs[42], 0.9) << "the test premise requires a dominant token";

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({1, vocab}));

  const auto got = SampleManySeeds(logits_v, 0, 64, /*t=*/1.0f, /*p=*/0.5f,
                                   /*k=*/0, /*mp=*/0.0f, /*seed_base=*/11);
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(got[static_cast<size_t>(i)], 42) << "seed slot " << i;
  }
}

TEST_F(SamplingTest, TopKOneIsArgmaxAtAnyTemperature) {
  constexpr int64_t vocab = 128;

  std::vector<float> logits(static_cast<size_t>(vocab));
  for (int64_t i = 0; i < vocab; ++i) {
    logits[static_cast<size_t>(i)] =
        std::sin(0.71f * static_cast<float>(i) + 0.3f);
  }
  logits[93] = 3.0f;
  logits = RoundTrip(logits);

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({1, vocab}));

  const int32_t want = ReferenceArgmax(logits);
  const auto got = SampleManySeeds(logits_v, 0, 64, /*t=*/2.0f, /*p=*/1.0f,
                                   /*k=*/1, /*mp=*/0.0f, /*seed_base=*/13);
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(got[static_cast<size_t>(i)], want) << "seed slot " << i;
  }
}

TEST_F(SamplingTest, MinPOfOneIsArgmax) {
  constexpr int64_t vocab = 128;

  std::vector<float> logits(static_cast<size_t>(vocab));
  for (int64_t i = 0; i < vocab; ++i) {
    logits[static_cast<size_t>(i)] =
        std::cos(0.43f * static_cast<float>(i));
  }
  logits[7] = 2.5f;
  logits = RoundTrip(logits);

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({1, vocab}));

  const int32_t want = ReferenceArgmax(logits);
  const auto got = SampleManySeeds(logits_v, 0, 64, /*t=*/0.9f, /*p=*/1.0f,
                                   /*k=*/0, /*mp=*/1.0f, /*seed_base=*/17);
  for (int i = 0; i < 64; ++i) {
    EXPECT_EQ(got[static_cast<size_t>(i)], want) << "seed slot " << i;
  }
}

TEST_F(SamplingTest, TopKThreeStaysInsideTheHostTopThree) {
  constexpr int64_t vocab = 128;
  constexpr float kTemp = 1.5f;

  // Three clear leaders over a warm background. At this temperature the
  // background holds most of the mass, so an ignored top_k would leave the
  // top-3 set almost every draw -- which is what makes this a test.
  std::vector<float> logits(static_cast<size_t>(vocab));
  for (int64_t i = 0; i < vocab; ++i) {
    logits[static_cast<size_t>(i)] =
        1.0f + 0.3f * std::sin(0.7f * static_cast<float>(i));
  }
  logits[3] = 2.0f;
  logits[7] = 1.8f;
  logits[11] = 1.6f;
  logits = RoundTrip(logits);

  const std::set<int32_t> want = ReferenceTopK(logits, 3);
  ASSERT_EQ(want, (std::set<int32_t>{3, 7, 11}));

  const std::vector<double> probs = ReferenceSoftmax(logits, kTemp);
  double top3_mass = 0.0;
  for (const int32_t i : want) top3_mass += probs[static_cast<size_t>(i)];
  ASSERT_LT(top3_mass, 0.5) << "the background must dominate for the test to "
                               "have teeth";

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({1, vocab}));

  const auto got = SampleManySeeds(logits_v, 0, 64, kTemp, /*p=*/1.0f,
                                   /*k=*/3, /*mp=*/0.0f, /*seed_base=*/19);
  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(want.count(got[static_cast<size_t>(i)]))
        << "seed slot " << i << " sampled " << got[static_cast<size_t>(i)]
        << " outside the top-3";
  }
}

TEST_F(SamplingTest, NewKnobsOffLeavesTopPSamplingIntact) {
  constexpr int64_t vocab = 128;
  constexpr float kTemp = 0.7f;
  constexpr float kTopP = 0.9f;

  // The pre-extension configuration: temperature and top_p only, top_k 0 and
  // min_p 0. Every draw must fall inside the host-computed nucleus.
  std::vector<float> logits(static_cast<size_t>(vocab));
  for (int64_t i = 0; i < vocab; ++i) {
    logits[static_cast<size_t>(i)] =
        2.0f * std::sin(0.29f * static_cast<float>(i) + 1.0f);
  }
  logits = RoundTrip(logits);

  const std::set<int32_t> nucleus =
      ReferenceNucleus(ReferenceSoftmax(logits, kTemp), kTopP);
  ASSERT_GT(nucleus.size(), 1u);
  ASSERT_LT(nucleus.size(), static_cast<size_t>(vocab))
      << "a nucleus of everything would test nothing";

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({1, vocab}));

  const auto got = SampleManySeeds(logits_v, 0, 64, kTemp, kTopP,
                                   /*k=*/0, /*mp=*/0.0f, /*seed_base=*/23);
  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(nucleus.count(got[static_cast<size_t>(i)]))
        << "seed slot " << i << " sampled " << got[static_cast<size_t>(i)]
        << " outside the nucleus";
  }
}

// --- ApplyPenalties ----------------------------------------------------------

TEST_F(SamplingTest, PenaltiesMatchTheHostFormulaAndMaskForcesNegInf) {
  constexpr int64_t vocab = 64;
  constexpr int64_t hist_cap = 4;
  constexpr int64_t mask_cap = 2;
  constexpr float kPresence = 0.5f;
  constexpr float kFrequency = 0.25f;
  constexpr float kRepetition = 1.2f;

  std::vector<float> logits(static_cast<size_t>(vocab));
  for (int64_t i = 0; i < vocab; ++i) {
    logits[static_cast<size_t>(i)] =
        1.5f * std::sin(0.61f * static_cast<float>(i));
  }
  // One positive and one negative history logit, so both branches of the HF
  // repetition rule (divide when positive, multiply when negative) run.
  logits[5] = 2.5f;
  logits[9] = -1.25f;
  logits = RoundTrip(logits);

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({1, vocab}));

  const std::vector<int32_t> rows_h = {0};
  const std::vector<float> presence_h = {kPresence};
  const std::vector<float> frequency_h = {kFrequency};
  const std::vector<float> repetition_h = {kRepetition};
  const std::vector<int32_t> history_ids_h = {5, 9, -1, -1};
  const std::vector<int32_t> history_counts_h = {2, 1, 0, 0};
  const std::vector<int32_t> mask_ids_h = {3, -1};

  const TensorView rows = Upload(keep, rows_h, DataType::kInt32, Shape({1}));
  const TensorView presence =
      Upload(keep, presence_h, DataType::kFloat, Shape({1}));
  const TensorView frequency =
      Upload(keep, frequency_h, DataType::kFloat, Shape({1}));
  const TensorView repetition =
      Upload(keep, repetition_h, DataType::kFloat, Shape({1}));
  const TensorView history_ids =
      Upload(keep, history_ids_h, DataType::kInt32, Shape({1, hist_cap}));
  const TensorView history_counts =
      Upload(keep, history_counts_h, DataType::kInt32, Shape({1, hist_cap}));
  const TensorView mask_ids =
      Upload(keep, mask_ids_h, DataType::kInt32, Shape({1, mask_cap}));

  ASSERT_TRUE(ops::ApplyPenalties(logits_v, rows, presence, frequency,
                                      repetition, history_ids, history_counts,
                                      mask_ids)
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // The reference: the same per-id update, in float, off the round-tripped
  // inputs, then through bf16 like the kernel's store.
  std::vector<float> want = logits;
  for (size_t h = 0; h < history_ids_h.size(); ++h) {
    const int32_t id = history_ids_h[h];
    if (id < 0) continue;
    float v = want[static_cast<size_t>(id)];
    v = v > 0.0f ? v / kRepetition : v * kRepetition;
    v -= kPresence;
    v -= kFrequency * static_cast<float>(history_counts_h[h]);
    want[static_cast<size_t>(id)] = v;
  }
  want = RoundTrip(want);

  const auto got = Download<bf16>(logits_v, vocab);
  for (int64_t i = 0; i < vocab; ++i) {
    const float g = __bfloat162float(got[static_cast<size_t>(i)]);
    if (i == 3) {
      EXPECT_TRUE(std::isinf(g) && g < 0.0f) << "masked logit 3 is " << g;
      continue;
    }
    if (i == 5 || i == 9) {
      // One bf16 ulp of headroom: the kernel may fuse the multiply-subtract.
      EXPECT_NEAR(g, want[static_cast<size_t>(i)],
                  0.01f * std::max(1.0f, std::abs(want[static_cast<size_t>(i)])))
          << "penalized logit " << i;
      continue;
    }
    // Everything else must be untouched, bit for bit.
    EXPECT_EQ(g, logits[static_cast<size_t>(i)]) << "logit " << i;
  }
}

// --- ComputeLogprobs ---------------------------------------------------------

TEST_F(SamplingTest, LogprobsMatchTheHostAndSkippedRowsStayUntouched) {
  constexpr int64_t vocab = 128;
  constexpr int64_t max_k = 8;
  constexpr int32_t kWanted = 5;
  constexpr int32_t kChosen = 17;
  constexpr float kSentinelLp = 12345.0f;
  constexpr int32_t kSentinelId = 7777;

  std::vector<float> logits(static_cast<size_t>(2 * vocab));
  for (int64_t r = 0; r < 2; ++r) {
    for (int64_t i = 0; i < vocab; ++i) {
      logits[static_cast<size_t>(r * vocab + i)] =
          std::sin(0.47f * static_cast<float>(i) + static_cast<float>(r));
    }
  }
  // A clear top-5 in row 0, with a deliberate exact tie between 10 and 20:
  // descending value, ascending index must place 10 before 20.
  logits[4] = 4.0f;
  logits[10] = 3.5f;
  logits[20] = 3.5f;
  logits[30] = 3.0f;
  logits[40] = 2.75f;
  logits = RoundTrip(logits);

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({2, vocab}));

  const std::vector<int32_t> rows_h = {0, 1};
  const std::vector<int32_t> chosen_h = {kChosen, 0};
  const std::vector<int32_t> k_wanted_h = {kWanted, -1};

  const TensorView rows = Upload(keep, rows_h, DataType::kInt32, Shape({2}));
  const TensorView chosen =
      Upload(keep, chosen_h, DataType::kInt32, Shape({2}));
  const TensorView k_wanted =
      Upload(keep, k_wanted_h, DataType::kInt32, Shape({2}));

  // Outputs pre-filled with sentinels, so "row 1 was skipped" is observable
  // rather than assumed.
  const std::vector<float> lp_sentinel(2, kSentinelLp);
  const std::vector<int32_t> id_sentinel(static_cast<size_t>(2 * max_k),
                                         kSentinelId);
  const std::vector<float> lps_sentinel(static_cast<size_t>(2 * max_k),
                                        kSentinelLp);
  const TensorView chosen_lp =
      Upload(keep, lp_sentinel, DataType::kFloat, Shape({2}));
  const TensorView top_ids =
      Upload(keep, id_sentinel, DataType::kInt32, Shape({2, max_k}));
  const TensorView top_lps =
      Upload(keep, lps_sentinel, DataType::kFloat, Shape({2, max_k}));

  ASSERT_TRUE(ops::ComputeLogprobs(logits_v, rows, chosen, k_wanted,
                                       chosen_lp, top_ids, top_lps)
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got_lp = Download<float>(chosen_lp, 2);
  const auto got_ids = Download<int32_t>(top_ids, 2 * max_k);
  const auto got_lps = Download<float>(top_lps, 2 * max_k);

  // Row 0: logsumexp in double against the device's __expf/__logf, so the
  // tolerance is loose-ish rather than ulp-tight.
  const std::vector<float> row0(logits.begin(), logits.begin() + vocab);
  double max_v = -INFINITY;
  for (const float v : row0) max_v = std::max(max_v, static_cast<double>(v));
  double sum = 0.0;
  for (const float v : row0) sum += std::exp(static_cast<double>(v) - max_v);
  const double log_z = std::log(sum) + max_v;

  const double want_chosen = static_cast<double>(row0[kChosen]) - log_z;
  EXPECT_NEAR(got_lp[0], want_chosen,
              2e-2 * std::max(1.0, std::abs(want_chosen)));

  const std::vector<int32_t> want_top = {4, 10, 20, 30, 40};
  for (int j = 0; j < kWanted; ++j) {
    const int32_t id = got_ids[static_cast<size_t>(j)];
    EXPECT_EQ(id, want_top[static_cast<size_t>(j)]) << "rank " << j;

    const double want_lp =
        static_cast<double>(row0[static_cast<size_t>(
            want_top[static_cast<size_t>(j)])]) - log_z;
    EXPECT_NEAR(got_lps[static_cast<size_t>(j)], want_lp,
                2e-2 * std::max(1.0, std::abs(want_lp)))
        << "rank " << j;
  }

  // Padding past k: -1 ids, -inf logprobs, never leftovers.
  for (int64_t j = kWanted; j < max_k; ++j) {
    EXPECT_EQ(got_ids[static_cast<size_t>(j)], -1) << "pad slot " << j;
    const float lp = got_lps[static_cast<size_t>(j)];
    EXPECT_TRUE(std::isinf(lp) && lp < 0.0f) << "pad slot " << j;
  }

  // Row 1 asked for nothing (-1) and must have written nothing.
  EXPECT_EQ(got_lp[1], kSentinelLp);
  for (int64_t j = 0; j < max_k; ++j) {
    EXPECT_EQ(got_ids[static_cast<size_t>(max_k + j)], kSentinelId)
        << "skipped-row slot " << j;
    EXPECT_EQ(got_lps[static_cast<size_t>(max_k + j)], kSentinelLp)
        << "skipped-row slot " << j;
  }
}

}  // namespace
}  // namespace inferx
