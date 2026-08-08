/// The mixture-of-experts FFN, against a reference computed on the host.
///
/// There is no MoE checkpoint on this box, so "correct" cannot mean "matches
/// HF's logits" the way M2's reference test does. It means something narrower
/// and still worth having: for weights and activations we choose, the routed
/// mixture equals what the definition of the mixture says it should, and the
/// pieces that make it fast -- the grouping, the permutation, the per-expert
/// GEMM loop -- do not change the answer relative to computing every expert for
/// every token and throwing the unrouted ones away.
///
/// That is the property that actually breaks in an MoE implementation. A
/// misrouted token, an off-by-one in the group offsets, or a combine that adds
/// a token's contributions in the wrong place all produce output that is
/// numerically plausible and wrong.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"
#include "inferx/kernels/gemm.h"
#include "inferx/kernels/moe.h"
#include "inferx/model/moe_ffn.h"

namespace inferx {
namespace {

using bf16 = __nv_bfloat16;
using model::MoeFfn;
using model::MoeWeights;

float Fill(int64_t a, int64_t b, float salt) {
  return 0.5f * std::sin(0.7f * static_cast<float>(a) + salt) *
         std::cos(0.31f * static_cast<float>(b) - salt);
}

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

class MoeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }
};

// --- The router --------------------------------------------------------------

// Softmax then top-k, computed on the host the way the definition reads.
struct Routing {
  std::vector<float> weights;   // [tokens, k]
  std::vector<int32_t> experts; // [tokens, k]
};

Routing ReferenceRoute(const std::vector<float>& logits, int64_t tokens,
                       int64_t num_experts, int k, bool renormalize) {
  Routing out;
  out.weights.resize(static_cast<size_t>(tokens * k));
  out.experts.resize(static_cast<size_t>(tokens * k));

  for (int64_t t = 0; t < tokens; ++t) {
    std::vector<float> row(static_cast<size_t>(num_experts));
    float max_v = -INFINITY;
    for (int64_t e = 0; e < num_experts; ++e) {
      row[static_cast<size_t>(e)] = logits[static_cast<size_t>(t * num_experts + e)];
      max_v = std::max(max_v, row[static_cast<size_t>(e)]);
    }

    float sum = 0.0f;
    for (float& v : row) {
      v = std::exp(v - max_v);
      sum += v;
    }
    for (float& v : row) v /= sum;

    float total = 0.0f;
    for (int slot = 0; slot < k; ++slot) {
      int best = -1;
      float best_v = -INFINITY;
      for (int64_t e = 0; e < num_experts; ++e) {
        if (row[static_cast<size_t>(e)] > best_v) {
          best_v = row[static_cast<size_t>(e)];
          best = static_cast<int>(e);
        }
      }
      out.experts[static_cast<size_t>(t * k + slot)] = best;
      out.weights[static_cast<size_t>(t * k + slot)] = best_v;
      total += best_v;
      row[static_cast<size_t>(best)] = -INFINITY;
    }

    if (renormalize && total > 0.0f) {
      for (int slot = 0; slot < k; ++slot) {
        out.weights[static_cast<size_t>(t * k + slot)] /= total;
      }
    }
  }

  return out;
}

TEST_F(MoeTest, RouterMatchesSoftmaxTopK) {
  constexpr int64_t tokens = 37, num_experts = 60;
  constexpr int k = 4;

  std::vector<float> logits(static_cast<size_t>(tokens * num_experts));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t e = 0; e < num_experts; ++e) {
      logits[static_cast<size_t>(t * num_experts + e)] = 3.0f * Fill(t, e, 0.2f);
    }
  }
  logits = RoundTrip(logits);

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({tokens, num_experts}));
  const TensorView w = Empty(keep, DataType::kFloat, Shape({tokens, k}));
  const TensorView e = Empty(keep, DataType::kInt32, Shape({tokens, k}));

  ASSERT_TRUE(kernels::MoeRouteTopK(logits_v, w, e, /*renormalize=*/true).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const Routing want = ReferenceRoute(logits, tokens, num_experts, k, true);
  const auto got_w = Download<float>(w, tokens * k);
  const auto got_e = Download<int32_t>(e, tokens * k);

  for (size_t i = 0; i < got_e.size(); ++i) {
    EXPECT_EQ(got_e[i], want.experts[i]) << "assignment " << i;
    EXPECT_NEAR(got_w[i], want.weights[i], 1e-5f) << "assignment " << i;
  }

  // Renormalized weights sum to 1 per token, which is the property the flag
  // exists for and the one a downstream mixture depends on.
  for (int64_t t = 0; t < tokens; ++t) {
    float sum = 0.0f;
    for (int slot = 0; slot < k; ++slot) {
      sum += got_w[static_cast<size_t>(t * k + slot)];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "token " << t;
  }
}

TEST_F(MoeTest, RouterWithoutRenormalizationKeepsSoftmaxMass) {
  constexpr int64_t tokens = 8, num_experts = 16;
  constexpr int k = 2;

  std::vector<float> logits(static_cast<size_t>(tokens * num_experts));
  for (size_t i = 0; i < logits.size(); ++i) {
    logits[i] = Fill(static_cast<int64_t>(i), 3, 1.1f);
  }
  logits = RoundTrip(logits);

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({tokens, num_experts}));
  const TensorView w = Empty(keep, DataType::kFloat, Shape({tokens, k}));
  const TensorView e = Empty(keep, DataType::kInt32, Shape({tokens, k}));

  ASSERT_TRUE(kernels::MoeRouteTopK(logits_v, w, e, /*renormalize=*/false).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const Routing want = ReferenceRoute(logits, tokens, num_experts, k, false);
  const auto got_w = Download<float>(w, tokens * k);

  for (size_t i = 0; i < got_w.size(); ++i) {
    EXPECT_NEAR(got_w[i], want.weights[i], 1e-5f) << "assignment " << i;
  }

  // Two of sixteen experts hold well under all the probability mass, so an
  // un-normalized top-2 must sum to less than 1. If this ever reads 1.0 the
  // flag is being ignored.
  for (int64_t t = 0; t < tokens; ++t) {
    const float sum = got_w[static_cast<size_t>(t * k)] +
                      got_w[static_cast<size_t>(t * k + 1)];
    EXPECT_LT(sum, 0.999f) << "token " << t;
  }
}

TEST_F(MoeTest, RouterBreaksTiesTowardsTheLowerIndex) {
  // Every logit identical, so every expert is tied. The rule has to pick 0 and
  // 1 -- if it picked by scan order or by whichever thread got there first,
  // routing would not be a function of the logits and identical requests could
  // take different experts.
  constexpr int64_t tokens = 4, num_experts = 32;
  constexpr int k = 2;

  const std::vector<float> logits(
      static_cast<size_t>(tokens * num_experts), 0.25f);

  std::vector<DeviceBuffer> keep;
  const TensorView logits_v = Upload(keep, ToBf16(logits), DataType::kBFloat16,
                                     Shape({tokens, num_experts}));
  const TensorView w = Empty(keep, DataType::kFloat, Shape({tokens, k}));
  const TensorView e = Empty(keep, DataType::kInt32, Shape({tokens, k}));

  ASSERT_TRUE(kernels::MoeRouteTopK(logits_v, w, e, true).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got_e = Download<int32_t>(e, tokens * k);
  for (int64_t t = 0; t < tokens; ++t) {
    EXPECT_EQ(got_e[static_cast<size_t>(t * k)], 0);
    EXPECT_EQ(got_e[static_cast<size_t>(t * k + 1)], 1);
  }
}

// --- The dispatch ------------------------------------------------------------

TEST_F(MoeTest, DispatchGroupsStablyAndInvertsExactly) {
  constexpr int64_t tokens = 129, num_experts = 8;
  constexpr int k = 3;
  constexpr int64_t assignments = tokens * k;

  // A deliberately lumpy assignment: expert 0 gets a lot, expert 7 gets none
  // at all. An expert with no rows is the case the GEMM loop has to skip and
  // the offsets have to represent as an empty range rather than as garbage.
  std::vector<int32_t> experts(static_cast<size_t>(assignments));
  for (int64_t a = 0; a < assignments; ++a) {
    const int64_t bucket = (a * a + 3 * a) % 11;
    experts[static_cast<size_t>(a)] =
        static_cast<int32_t>(bucket < 4 ? 0 : bucket % 7);
  }

  std::vector<DeviceBuffer> keep;
  const TensorView experts_v =
      Upload(keep, experts, DataType::kInt32, Shape({tokens, k}));
  const TensorView offsets =
      Empty(keep, DataType::kInt32, Shape({num_experts + 1}));
  const TensorView rows = Empty(keep, DataType::kInt32, Shape({assignments}));
  const TensorView dest = Empty(keep, DataType::kInt32, Shape({assignments}));

  ASSERT_TRUE(kernels::MoeBuildDispatch(experts_v, num_experts, offsets, rows,
                                        dest)
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got_offsets = Download<int32_t>(offsets, num_experts + 1);
  const auto got_rows = Download<int32_t>(rows, assignments);
  const auto got_dest = Download<int32_t>(dest, assignments);

  EXPECT_EQ(got_offsets.front(), 0);
  EXPECT_EQ(got_offsets.back(), assignments)
      << "the offsets do not cover every assignment";

  // Expert 7 was never chosen, so its range must be empty rather than absent.
  EXPECT_EQ(got_offsets[7], got_offsets[8]);

  // Every assignment appears exactly once, in its own expert's range, and the
  // two directions agree.
  std::vector<int> seen(static_cast<size_t>(assignments), 0);
  for (int64_t a = 0; a < assignments; ++a) {
    const int32_t slot = got_dest[static_cast<size_t>(a)];
    ASSERT_GE(slot, 0);
    ASSERT_LT(slot, assignments);
    ++seen[static_cast<size_t>(slot)];

    const int32_t expert = experts[static_cast<size_t>(a)];
    EXPECT_GE(slot, got_offsets[static_cast<size_t>(expert)]);
    EXPECT_LT(slot, got_offsets[static_cast<size_t>(expert) + 1]);

    EXPECT_EQ(got_rows[static_cast<size_t>(slot)],
              static_cast<int32_t>(a / k))
        << "dest and rows disagree about assignment " << a;
  }

  for (size_t i = 0; i < seen.size(); ++i) {
    EXPECT_EQ(seen[i], 1) << "grouped row " << i << " claimed " << seen[i]
                          << " times";
  }

  // Stability: within an expert, assignments come out in ascending order. This
  // is what makes the whole layer reproducible -- an unstable grouping would
  // reorder a GEMM's rows between runs and change the low bits of nothing at
  // all, right up until an expert's rows get summed somewhere.
  std::vector<int32_t> last_seen(static_cast<size_t>(num_experts), -1);
  for (int64_t a = 0; a < assignments; ++a) {
    const int32_t expert = experts[static_cast<size_t>(a)];
    const int32_t slot = got_dest[static_cast<size_t>(a)];
    EXPECT_GT(slot, last_seen[static_cast<size_t>(expert)])
        << "expert " << expert << " grouped assignment " << a
        << " out of order";
    last_seen[static_cast<size_t>(expert)] = slot;
  }
}

TEST_F(MoeTest, GatherAndCombineRoundTripThroughThePermutation) {
  constexpr int64_t tokens = 40, width = 48, num_experts = 6;
  constexpr int k = 2;
  constexpr int64_t assignments = tokens * k;

  std::vector<float> x(static_cast<size_t>(tokens * width));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t c = 0; c < width; ++c) {
      x[static_cast<size_t>(t * width + c)] = Fill(t, c, 0.9f);
    }
  }
  x = RoundTrip(x);

  std::vector<int32_t> experts(static_cast<size_t>(assignments));
  for (int64_t a = 0; a < assignments; ++a) {
    experts[static_cast<size_t>(a)] = static_cast<int32_t>((a * 5 + 1) % num_experts);
  }

  // Weights that sum to 1 per token, so a correct gather-then-combine with no
  // expert in between is the identity and any permutation bug shows up as a
  // difference from x rather than as a scale factor.
  std::vector<float> weights(static_cast<size_t>(assignments));
  for (int64_t t = 0; t < tokens; ++t) {
    weights[static_cast<size_t>(t * k)] = 0.25f;
    weights[static_cast<size_t>(t * k + 1)] = 0.75f;
  }

  std::vector<DeviceBuffer> keep;
  const TensorView x_v =
      Upload(keep, ToBf16(x), DataType::kBFloat16, Shape({tokens, width}));
  const TensorView experts_v =
      Upload(keep, experts, DataType::kInt32, Shape({tokens, k}));
  const TensorView weights_v =
      Upload(keep, weights, DataType::kFloat, Shape({tokens, k}));

  const TensorView offsets =
      Empty(keep, DataType::kInt32, Shape({num_experts + 1}));
  const TensorView rows = Empty(keep, DataType::kInt32, Shape({assignments}));
  const TensorView dest = Empty(keep, DataType::kInt32, Shape({assignments}));
  const TensorView gathered =
      Empty(keep, DataType::kBFloat16, Shape({assignments, width}));
  const TensorView out =
      Empty(keep, DataType::kBFloat16, Shape({tokens, width}));

  ASSERT_TRUE(kernels::MoeBuildDispatch(experts_v, num_experts, offsets, rows,
                                        dest).ok());
  ASSERT_TRUE(kernels::MoeGatherRows(x_v, rows, gathered).ok());
  ASSERT_TRUE(kernels::MoeCombineRows(gathered, dest, weights_v, out).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got = Download<bf16>(out, tokens * width);
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t c = 0; c < width; ++c) {
      const size_t i = static_cast<size_t>(t * width + c);
      EXPECT_NEAR(__bfloat162float(got[i]), x[i], 8e-3f)
          << "token " << t << " column " << c;
    }
  }
}

// --- The whole layer ---------------------------------------------------------

// Every expert applied to every token, mixed by the routing weights. This is
// the definition of a mixture of experts and it is O(E) times more work than
// the implementation, which is exactly why the implementation exists and why
// the reference is written this way.
std::vector<float> ReferenceMoe(const std::vector<float>& x,
                                const std::vector<float>& router,
                                const std::vector<float>& gate_up,
                                const std::vector<float>& down,
                                const MoeFfn::Config& c, int64_t tokens) {
  const int64_t h = c.hidden;
  const int64_t inter = c.moe_intermediate;
  const int k = static_cast<int>(c.top_k);

  std::vector<float> logits(static_cast<size_t>(tokens * c.num_experts));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t e = 0; e < c.num_experts; ++e) {
      float acc = 0.0f;
      for (int64_t i = 0; i < h; ++i) {
        acc += x[static_cast<size_t>(t * h + i)] *
               router[static_cast<size_t>(e * h + i)];
      }
      logits[static_cast<size_t>(t * c.num_experts + e)] = acc;
    }
  }

  const Routing routing =
      ReferenceRoute(logits, tokens, c.num_experts, k, c.norm_topk_prob);

  std::vector<float> out(static_cast<size_t>(tokens * h), 0.0f);

  for (int64_t t = 0; t < tokens; ++t) {
    for (int slot = 0; slot < k; ++slot) {
      const int64_t e = routing.experts[static_cast<size_t>(t * k + slot)];
      const float w = routing.weights[static_cast<size_t>(t * k + slot)];

      const float* gu = gate_up.data() + static_cast<size_t>(e * 2 * inter * h);
      const float* dw = down.data() + static_cast<size_t>(e * h * inter);

      std::vector<float> act(static_cast<size_t>(inter));
      for (int64_t j = 0; j < inter; ++j) {
        float g = 0.0f;
        float u = 0.0f;
        for (int64_t i = 0; i < h; ++i) {
          const float xv = x[static_cast<size_t>(t * h + i)];
          g += xv * gu[static_cast<size_t>(j * h + i)];
          u += xv * gu[static_cast<size_t>((inter + j) * h + i)];
        }
        act[static_cast<size_t>(j)] = (g / (1.0f + std::exp(-g))) * u;
      }

      for (int64_t i = 0; i < h; ++i) {
        float acc = 0.0f;
        for (int64_t j = 0; j < inter; ++j) {
          acc += act[static_cast<size_t>(j)] * dw[static_cast<size_t>(i * inter + j)];
        }
        out[static_cast<size_t>(t * h + i)] += w * acc;
      }
    }
  }

  return out;
}

TEST_F(MoeTest, LayerMatchesTheDenseReferenceMixture) {
  MoeFfn::Config c;
  c.hidden = 64;
  c.num_experts = 8;
  c.top_k = 2;
  c.moe_intermediate = 32;
  c.norm_topk_prob = true;

  constexpr int64_t tokens = 24;

  std::vector<float> x(static_cast<size_t>(tokens * c.hidden));
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = Fill(static_cast<int64_t>(i), 2, 0.4f);
  }

  std::vector<float> router(static_cast<size_t>(c.num_experts * c.hidden));
  for (size_t i = 0; i < router.size(); ++i) {
    router[i] = Fill(static_cast<int64_t>(i), 7, 1.3f);
  }

  std::vector<float> gate_up(
      static_cast<size_t>(c.num_experts * 2 * c.moe_intermediate * c.hidden));
  for (size_t i = 0; i < gate_up.size(); ++i) {
    gate_up[i] = 0.4f * Fill(static_cast<int64_t>(i), 11, 0.05f);
  }

  std::vector<float> down(
      static_cast<size_t>(c.num_experts * c.hidden * c.moe_intermediate));
  for (size_t i = 0; i < down.size(); ++i) {
    down[i] = 0.4f * Fill(static_cast<int64_t>(i), 13, 2.2f);
  }

  x = RoundTrip(x);
  router = RoundTrip(router);
  gate_up = RoundTrip(gate_up);
  down = RoundTrip(down);

  std::vector<DeviceBuffer> keep;
  MoeWeights w;
  const TensorView x_v = Upload(keep, ToBf16(x), DataType::kBFloat16,
                                Shape({tokens, c.hidden}));
  w.router = Upload(keep, ToBf16(router), DataType::kBFloat16,
                    Shape({c.num_experts, c.hidden}));
  w.gate_up = Upload(keep, ToBf16(gate_up), DataType::kBFloat16,
                     Shape({c.num_experts, 2 * c.moe_intermediate, c.hidden}));
  w.down = Upload(keep, ToBf16(down), DataType::kBFloat16,
                  Shape({c.num_experts, c.hidden, c.moe_intermediate}));

  const TensorView out =
      Empty(keep, DataType::kBFloat16, Shape({tokens, c.hidden}));

  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  auto ffn = MoeFfn::Create(c, tokens);
  ASSERT_TRUE(ffn.ok()) << ffn.status();

  ASSERT_TRUE(ffn->Forward(x_v, w, out, &*gemm).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::vector<float> want = ReferenceMoe(x, router, gate_up, down, c,
                                               tokens);
  const auto got = Download<bf16>(out, tokens * c.hidden);

  double worst = 0.0;
  double scale = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    scale = std::max<double>(scale, std::abs(want[i]));
  }
  ASSERT_GT(scale, 0.0) << "degenerate reference";

  for (size_t i = 0; i < want.size(); ++i) {
    worst = std::max<double>(
        worst, std::abs(__bfloat162float(got[i]) - want[i]));
  }

  // bf16 keeps 8 mantissa bits, and the value has been through two GEMMs of
  // depth 64 and 32 plus a mixture. Relative to the largest output, 2% bounds
  // that comfortably while still being orders of magnitude tighter than a
  // misrouted token would be.
  EXPECT_LT(worst / scale, 0.02) << "worst absolute error " << worst
                                 << " against scale " << scale;

  // Every assignment landed somewhere: the counts have to sum to tokens · k.
  auto counts = ffn->LastExpertCounts();
  ASSERT_TRUE(counts.ok()) << counts.status();
  EXPECT_EQ(std::accumulate(counts->begin(), counts->end(), 0),
            tokens * c.top_k);
}

TEST_F(MoeTest, SharedExpertIsAddedToTheRoutedMixture) {
  // The shared expert runs for every token regardless of routing, so turning
  // it on must change every row. A shared expert wired into the top-k mixture
  // by mistake would leave some rows untouched.
  MoeFfn::Config c;
  c.hidden = 32;
  c.num_experts = 4;
  c.top_k = 1;
  c.moe_intermediate = 16;
  c.shared_intermediate = 16;

  constexpr int64_t tokens = 12;

  // Both indices vary: Fill's second argument held constant would make the
  // whole tensor share one cosine factor, and at some salts that factor is
  // near zero -- which produces a "correct" all-but-zero tensor and a test
  // that passes for the wrong reason.
  auto rand_vec = [](size_t n, float salt) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
      v[i] = 0.4f * Fill(static_cast<int64_t>(i),
                         static_cast<int64_t>(i % 13), salt);
    }
    return RoundTrip(v);
  };

  const std::vector<float> x = rand_vec(
      static_cast<size_t>(tokens * c.hidden), 0.1f);
  const std::vector<float> router = rand_vec(
      static_cast<size_t>(c.num_experts * c.hidden), 0.6f);
  const std::vector<float> gate_up = rand_vec(
      static_cast<size_t>(c.num_experts * 2 * c.moe_intermediate * c.hidden),
      1.4f);
  const std::vector<float> down = rand_vec(
      static_cast<size_t>(c.num_experts * c.hidden * c.moe_intermediate), 2.7f);
  const std::vector<float> sgu = rand_vec(
      static_cast<size_t>(2 * c.shared_intermediate * c.hidden), 3.1f);
  const std::vector<float> sdown = rand_vec(
      static_cast<size_t>(c.hidden * c.shared_intermediate), 3.9f);
  const std::vector<float> sgate = rand_vec(
      static_cast<size_t>(c.hidden), 4.3f);

  std::vector<DeviceBuffer> keep;
  const TensorView x_v = Upload(keep, ToBf16(x), DataType::kBFloat16,
                                Shape({tokens, c.hidden}));

  MoeWeights w;
  w.router = Upload(keep, ToBf16(router), DataType::kBFloat16,
                    Shape({c.num_experts, c.hidden}));
  w.gate_up = Upload(keep, ToBf16(gate_up), DataType::kBFloat16,
                     Shape({c.num_experts, 2 * c.moe_intermediate, c.hidden}));
  w.down = Upload(keep, ToBf16(down), DataType::kBFloat16,
                  Shape({c.num_experts, c.hidden, c.moe_intermediate}));

  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  // Routed only.
  MoeFfn::Config routed_only = c;
  routed_only.shared_intermediate = 0;

  const TensorView out_routed =
      Empty(keep, DataType::kBFloat16, Shape({tokens, c.hidden}));
  auto routed_ffn = MoeFfn::Create(routed_only, tokens);
  ASSERT_TRUE(routed_ffn.ok()) << routed_ffn.status();
  ASSERT_TRUE(routed_ffn->Forward(x_v, w, out_routed, &*gemm).ok());

  // Routed plus shared.
  w.shared_gate_up = Upload(keep, ToBf16(sgu), DataType::kBFloat16,
                            Shape({2 * c.shared_intermediate, c.hidden}));
  w.shared_down = Upload(keep, ToBf16(sdown), DataType::kBFloat16,
                         Shape({c.hidden, c.shared_intermediate}));
  w.shared_gate = Upload(keep, ToBf16(sgate), DataType::kBFloat16,
                         Shape({1, c.hidden}));

  const TensorView out_both =
      Empty(keep, DataType::kBFloat16, Shape({tokens, c.hidden}));
  auto both_ffn = MoeFfn::Create(c, tokens);
  ASSERT_TRUE(both_ffn.ok()) << both_ffn.status();
  ASSERT_TRUE(both_ffn->Forward(x_v, w, out_both, &*gemm).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto routed = Download<bf16>(out_routed, tokens * c.hidden);
  const auto both = Download<bf16>(out_both, tokens * c.hidden);

  for (int64_t t = 0; t < tokens; ++t) {
    double row_delta = 0.0;
    for (int64_t i = 0; i < c.hidden; ++i) {
      const size_t idx = static_cast<size_t>(t * c.hidden + i);
      row_delta += std::abs(__bfloat162float(both[idx]) -
                            __bfloat162float(routed[idx]));
    }
    EXPECT_GT(row_delta, 1e-3) << "token " << t
                               << " was untouched by the shared expert";
  }
}

TEST_F(MoeTest, IdenticalInputProducesIdenticalOutput) {
  // The determinism claim in moe.h, checked rather than asserted. The grouping
  // is the part at risk: an atomic cursor would reorder rows between runs, and
  // reordered rows change a GEMM's accumulation order.
  MoeFfn::Config c;
  c.hidden = 64;
  c.num_experts = 16;
  c.top_k = 4;
  c.moe_intermediate = 32;

  constexpr int64_t tokens = 64;

  auto rand_vec = [](size_t n, float salt) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
      v[i] = 0.4f * Fill(static_cast<int64_t>(i),
                         static_cast<int64_t>(i % 13), salt);
    }
    return RoundTrip(v);
  };

  std::vector<DeviceBuffer> keep;
  const TensorView x_v =
      Upload(keep, ToBf16(rand_vec(static_cast<size_t>(tokens * c.hidden), 0.2f)),
             DataType::kBFloat16, Shape({tokens, c.hidden}));

  MoeWeights w;
  w.router = Upload(
      keep, ToBf16(rand_vec(static_cast<size_t>(c.num_experts * c.hidden), 0.8f)),
      DataType::kBFloat16, Shape({c.num_experts, c.hidden}));
  w.gate_up = Upload(
      keep,
      ToBf16(rand_vec(static_cast<size_t>(c.num_experts * 2 * c.moe_intermediate *
                                          c.hidden), 1.9f)),
      DataType::kBFloat16,
      Shape({c.num_experts, 2 * c.moe_intermediate, c.hidden}));
  w.down = Upload(
      keep,
      ToBf16(rand_vec(static_cast<size_t>(c.num_experts * c.hidden *
                                          c.moe_intermediate), 2.5f)),
      DataType::kBFloat16, Shape({c.num_experts, c.hidden, c.moe_intermediate}));

  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  auto ffn = MoeFfn::Create(c, tokens);
  ASSERT_TRUE(ffn.ok()) << ffn.status();

  const TensorView out =
      Empty(keep, DataType::kBFloat16, Shape({tokens, c.hidden}));

  std::vector<bf16> first;
  for (int run = 0; run < 4; ++run) {
    ASSERT_EQ(cudaMemset(out.Data(), 0,
                         DataTypeByteSize(DataType::kBFloat16,
                                          tokens * c.hidden)),
              cudaSuccess);
    ASSERT_TRUE(ffn->Forward(x_v, w, out, &*gemm).ok());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto got = Download<bf16>(out, tokens * c.hidden);
    if (run == 0) {
      first = got;
      continue;
    }

    for (size_t i = 0; i < first.size(); ++i) {
      EXPECT_EQ(__bfloat162float(got[i]), __bfloat162float(first[i]))
          << "run " << run << " differed at element " << i;
    }
  }
}

}  // namespace
}  // namespace inferx
