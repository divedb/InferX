/// Multi-head latent attention, against a reference computed on the host.
///
/// Like the MoE suite, this cannot mean "matches DeepSeek's logits" — there is
/// no DeepSeek checkpoint on this box. It means the two things that are
/// actually at risk in an MLA implementation and that no amount of staring at
/// the code settles:
///
///   * The latent round-trips. What comes out of the paged cache and through
///     the up-projection is what the definition says K and V are, and the
///     decoupled RoPE key really is shared across heads rather than
///     accidentally per-head.
///   * **Decoding one token at a time equals prefilling the whole prompt.**
///     That is the property the cache exists to provide, and it is the one that
///     breaks when a slot mapping, a block table walk, or the causal bound is
///     off by one — silently, because both paths still produce plausible
///     numbers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/kv_cache.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"
#include "inferx/kernels/gemm.h"
#include "inferx/kernels/gpt_oss.h"
#include "inferx/kernels/mla.h"
#include "inferx/model/config.h"
#include "inferx/model/mla.h"

namespace inferx {
namespace {

using bf16 = __nv_bfloat16;
using model::MlaAttentionLayer;
using model::MlaWeights;
using model::ModelConfig;

float Fill(int64_t a, int64_t b, float salt) {
  return 0.4f * std::sin(0.61f * static_cast<float>(a) + salt) *
         std::cos(0.23f * static_cast<float>(b) - salt);
}

std::vector<float> RoundTrip(const std::vector<float>& host) {
  std::vector<float> out(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    out[i] = __bfloat162float(__float2bfloat16(host[i]));
  }
  return out;
}

std::vector<bf16> ToBf16(const std::vector<float>& host) {
  std::vector<bf16> out(host.size());
  for (size_t i = 0; i < host.size(); ++i) out[i] = __float2bfloat16(host[i]);
  return out;
}

std::vector<float> RandomTensor(size_t n, float salt) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = Fill(static_cast<int64_t>(i), static_cast<int64_t>(i % 17), salt);
  }
  return RoundTrip(v);
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

class MlaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }
};

// A small MLA shape. Real ones are ~30x wider; this is sized so a host
// reference over the whole thing runs in milliseconds, which is what makes an
// exhaustive comparison affordable.
ModelConfig SmallMlaConfig() {
  ModelConfig c;
  c.architecture = model::Architecture::kLlama;
  c.hidden_size = 64;
  c.intermediate_size = 128;
  c.num_hidden_layers = 1;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 4;
  c.head_dim = 16;
  c.vocab_size = 128;
  c.rope_theta = 10000.0;
  c.rms_norm_eps = 1e-6;

  c.kv_lora_rank = 32;
  c.q_lora_rank = 24;
  c.qk_nope_head_dim = 16;
  c.qk_rope_head_dim = 8;
  c.v_head_dim = 16;

  return c;
}

struct Weights {
  std::vector<float> q_a, q_a_norm, q_b, kv_a, kv_a_norm, kv_b, o;
};

Weights MakeWeights(const ModelConfig& c) {
  const int64_t h = c.hidden_size;
  const int64_t heads = c.num_attention_heads;
  const int64_t qk = c.qk_nope_head_dim + c.qk_rope_head_dim;

  Weights w;
  w.q_a = RandomTensor(static_cast<size_t>(c.q_lora_rank * h), 0.11f);
  w.q_a_norm = RandomTensor(static_cast<size_t>(c.q_lora_rank), 0.71f);
  w.q_b = RandomTensor(static_cast<size_t>(heads * qk * c.q_lora_rank), 1.31f);
  w.kv_a = RandomTensor(
      static_cast<size_t>((c.kv_lora_rank + c.qk_rope_head_dim) * h), 2.02f);
  w.kv_a_norm = RandomTensor(static_cast<size_t>(c.kv_lora_rank), 2.53f);
  w.kv_b = RandomTensor(
      static_cast<size_t>(heads * (c.qk_nope_head_dim + c.v_head_dim) *
                          c.kv_lora_rank),
      3.17f);
  w.o = RandomTensor(static_cast<size_t>(h * heads * c.v_head_dim), 3.91f);

  // Norm weights near 1: a norm scale drawn from the same distribution as a
  // projection would shrink every activation by an order of magnitude and make
  // the comparison below insensitive to everything else.
  for (float& v : w.q_a_norm) v = 1.0f + 0.1f * v;
  for (float& v : w.kv_a_norm) v = 1.0f + 0.1f * v;
  w.q_a_norm = RoundTrip(w.q_a_norm);
  w.kv_a_norm = RoundTrip(w.kv_a_norm);

  return w;
}

MlaWeights UploadWeights(std::vector<DeviceBuffer>& keep, const ModelConfig& c,
                         const Weights& w) {
  const int64_t h = c.hidden_size;
  const int64_t heads = c.num_attention_heads;
  const int64_t qk = c.qk_nope_head_dim + c.qk_rope_head_dim;

  MlaWeights mw;
  mw.q_a = Upload(keep, ToBf16(w.q_a), DataType::kBFloat16,
                  Shape({c.q_lora_rank, h}));
  mw.q_a_norm = Upload(keep, ToBf16(w.q_a_norm), DataType::kBFloat16,
                       Shape({c.q_lora_rank}));
  mw.q_b = Upload(keep, ToBf16(w.q_b), DataType::kBFloat16,
                  Shape({heads * qk, c.q_lora_rank}));
  mw.kv_a = Upload(keep, ToBf16(w.kv_a), DataType::kBFloat16,
                   Shape({c.kv_lora_rank + c.qk_rope_head_dim, h}));
  mw.kv_a_norm = Upload(keep, ToBf16(w.kv_a_norm), DataType::kBFloat16,
                        Shape({c.kv_lora_rank}));
  mw.kv_b = Upload(keep, ToBf16(w.kv_b), DataType::kBFloat16,
                   Shape({heads * (c.qk_nope_head_dim + c.v_head_dim),
                          c.kv_lora_rank}));
  mw.o = Upload(keep, ToBf16(w.o), DataType::kBFloat16,
                Shape({h, heads * c.v_head_dim}));
  return mw;
}

// y[m, n] = x[m, k] · w[n, k]ᵀ, the same mapping LinearBF16 implements.
std::vector<float> MatMulT(const std::vector<float>& x,
                           const std::vector<float>& w, int64_t m, int64_t k,
                           int64_t n) {
  std::vector<float> y(static_cast<size_t>(m * n), 0.0f);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        acc += x[static_cast<size_t>(i * k + p)] *
               w[static_cast<size_t>(j * k + p)];
      }
      y[static_cast<size_t>(i * n + j)] = acc;
    }
  }
  return y;
}

void RmsNormRows(std::vector<float>& x, const std::vector<float>& weight,
                 int64_t rows, int64_t width, float eps) {
  for (int64_t r = 0; r < rows; ++r) {
    float sq = 0.0f;
    for (int64_t c = 0; c < width; ++c) {
      const float v = x[static_cast<size_t>(r * width + c)];
      sq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sq / static_cast<float>(width) + eps);
    for (int64_t c = 0; c < width; ++c) {
      x[static_cast<size_t>(r * width + c)] *=
          inv * weight[static_cast<size_t>(c)];
    }
  }
}

// The half-split rotation, applied to the trailing `rope_dim` of each row of
// width `width`.
void RopeTail(std::vector<float>& x, int64_t rows, int64_t width,
              int64_t rope_dim, const std::vector<int32_t>& positions,
              int64_t rows_per_position, float theta) {
  const int64_t half = rope_dim / 2;

  for (int64_t r = 0; r < rows; ++r) {
    const float pos = static_cast<float>(
        positions[static_cast<size_t>(r / rows_per_position)]);
    float* row = x.data() + r * width + (width - rope_dim);

    for (int64_t j = 0; j < half; ++j) {
      const float inv_freq = std::pow(
          theta, -2.0f * static_cast<float>(j) / static_cast<float>(rope_dim));
      const float angle = pos * inv_freq;
      const float s = std::sin(angle);
      const float cs = std::cos(angle);

      const float lo = row[j];
      const float hi = row[j + half];
      row[j] = lo * cs - hi * s;
      row[j + half] = hi * cs + lo * s;
    }
  }
}

// The whole layer, from the definition, with no cache: every token's latent is
// recomputed and every query attends over every earlier token.
std::vector<float> ReferenceMla(const ModelConfig& c, const Weights& w,
                                const std::vector<float>& x, int64_t tokens) {
  const int64_t h = c.hidden_size;
  const int64_t heads = c.num_attention_heads;
  const int64_t nope = c.qk_nope_head_dim;
  const int64_t rope = c.qk_rope_head_dim;
  const int64_t vd = c.v_head_dim;
  const int64_t qk = nope + rope;
  const int64_t latent_dim = c.kv_lora_rank;

  std::vector<int32_t> positions(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) positions[static_cast<size_t>(t)] = t;

  // Q: down, norm, up, rotate the tail of each head.
  std::vector<float> q_a = MatMulT(x, w.q_a, tokens, h, c.q_lora_rank);
  RmsNormRows(q_a, w.q_a_norm, tokens, c.q_lora_rank,
              static_cast<float>(c.rms_norm_eps));
  std::vector<float> q =
      MatMulT(q_a, w.q_b, tokens, c.q_lora_rank, heads * qk);
  RopeTail(q, tokens * heads, qk, rope, positions, heads,
           static_cast<float>(c.rope_theta));

  // KV: down to [latent | rope_key], rotate the key, norm the latent.
  std::vector<float> kv_a = MatMulT(x, w.kv_a, tokens, h, latent_dim + rope);
  RopeTail(kv_a, tokens, latent_dim + rope, rope, positions, 1,
           static_cast<float>(c.rope_theta));

  std::vector<float> latent(static_cast<size_t>(tokens * latent_dim));
  std::vector<float> k_rope(static_cast<size_t>(tokens * rope));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t d = 0; d < latent_dim; ++d) {
      latent[static_cast<size_t>(t * latent_dim + d)] =
          kv_a[static_cast<size_t>(t * (latent_dim + rope) + d)];
    }
    for (int64_t d = 0; d < rope; ++d) {
      k_rope[static_cast<size_t>(t * rope + d)] =
          kv_a[static_cast<size_t>(t * (latent_dim + rope) + latent_dim + d)];
    }
  }
  RmsNormRows(latent, w.kv_a_norm, tokens, latent_dim,
              static_cast<float>(c.rms_norm_eps));

  // Up-project the latent into each head's K_nope and V.
  const std::vector<float> kv =
      MatMulT(latent, w.kv_b, tokens, latent_dim, heads * (nope + vd));

  // Attention, per head, causal.
  std::vector<float> attn(static_cast<size_t>(tokens * heads * vd), 0.0f);

  for (int64_t i = 0; i < tokens; ++i) {
    for (int64_t hd = 0; hd < heads; ++hd) {
      std::vector<float> scores(static_cast<size_t>(i + 1));
      const float scale = 1.0f / std::sqrt(static_cast<float>(qk));

      for (int64_t j = 0; j <= i; ++j) {
        float dot = 0.0f;
        for (int64_t d = 0; d < nope; ++d) {
          dot += q[static_cast<size_t>((i * heads + hd) * qk + d)] *
                 kv[static_cast<size_t>((j * heads + hd) * (nope + vd) + d)];
        }
        // The shared RoPE key: no head index on the right-hand side.
        for (int64_t d = 0; d < rope; ++d) {
          dot += q[static_cast<size_t>((i * heads + hd) * qk + nope + d)] *
                 k_rope[static_cast<size_t>(j * rope + d)];
        }
        scores[static_cast<size_t>(j)] = dot * scale;
      }

      float max_v = -INFINITY;
      for (float s : scores) max_v = std::max(max_v, s);

      float sum = 0.0f;
      for (float& s : scores) {
        s = std::exp(s - max_v);
        sum += s;
      }

      for (int64_t d = 0; d < vd; ++d) {
        float acc = 0.0f;
        for (int64_t j = 0; j <= i; ++j) {
          acc += scores[static_cast<size_t>(j)] *
                 kv[static_cast<size_t>((j * heads + hd) * (nope + vd) + nope + d)];
        }
        attn[static_cast<size_t>((i * heads + hd) * vd + d)] = acc / sum;
      }
    }
  }

  return MatMulT(attn, w.o, tokens, heads * vd, h);
}

// --- The layout --------------------------------------------------------------

TEST(MlaLayout, CachesOneLatentPerTokenRatherThanKAndV) {
  const ModelConfig c = SmallMlaConfig();
  const KvLayout layout = MlaAttentionLayer::LayoutFor(c);

  EXPECT_EQ(layout.entries_per_token, 1) << "MLA has no separate value cache";
  EXPECT_EQ(layout.kv_heads, 1) << "the latent is not per-head";
  EXPECT_EQ(layout.head_dim, c.kv_lora_rank + c.qk_rope_head_dim);
  EXPECT_EQ(layout.ElementsPerToken(), c.KvElementsPerTokenPerLayer());
}

TEST(MlaLayout, IsMuchSmallerThanTheGqaEquivalent) {
  // DeepSeek-V2's real numbers, which is the whole argument for MLA: 576
  // elements per token per layer against 8192 for the GQA shape it replaces.
  ModelConfig mla;
  mla.kv_lora_rank = 512;
  mla.qk_rope_head_dim = 64;
  EXPECT_EQ(mla.KvElementsPerTokenPerLayer(), 576);

  ModelConfig gqa;
  gqa.num_key_value_heads = 32;
  gqa.head_dim = 128;
  EXPECT_EQ(gqa.KvElementsPerTokenPerLayer(), 8192);
}

TEST(MlaLayout, PoolRefusesToHandOutAValueCache) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  const ModelConfig c = SmallMlaConfig();
  auto pool = KvBlockPool::Create(/*num_layers=*/1, /*num_blocks=*/4,
                                  /*block_size=*/8, MlaAttentionLayer::LayoutFor(c));
  ASSERT_TRUE(pool.ok()) << pool.status();

  EXPECT_TRUE(pool->KeyCache(0).ok());

  // Returning half a latent would be worse than failing: the caller would get
  // a tensor of the right shape holding the wrong thing.
  const auto value = pool->ValueCache(0);
  EXPECT_FALSE(value.ok());
  EXPECT_EQ(value.status().code(), absl::StatusCode::kFailedPrecondition);
}

// --- The layer ---------------------------------------------------------------

class MlaLayerTest : public MlaTest {};

TEST_F(MlaLayerTest, PrefillMatchesTheHostReference) {
  const ModelConfig c = SmallMlaConfig();
  const Weights w = MakeWeights(c);

  constexpr int64_t tokens = 12;
  constexpr int64_t block_size = 8;

  const std::vector<float> x =
      RandomTensor(static_cast<size_t>(tokens * c.hidden_size), 0.33f);

  auto pool = KvBlockPool::Create(1, 8, block_size,
                                  MlaAttentionLayer::LayoutFor(c));
  ASSERT_TRUE(pool.ok()) << pool.status();

  std::vector<int32_t> blocks;
  for (int64_t b = 0; b < (tokens + block_size - 1) / block_size; ++b) {
    auto got = pool->AllocateBlock();
    ASSERT_TRUE(got.ok()) << got.status();
    blocks.push_back(*got);
  }

  std::vector<int32_t> positions(static_cast<size_t>(tokens));
  std::vector<int32_t> slots(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) {
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
    slots[static_cast<size_t>(t)] =
        blocks[static_cast<size_t>(t / block_size)] * block_size +
        static_cast<int32_t>(t % block_size);
  }

  std::vector<DeviceBuffer> keep;
  const TensorView x_v = Upload(keep, ToBf16(x), DataType::kBFloat16,
                                Shape({tokens, c.hidden_size}));
  const TensorView pos_v =
      Upload(keep, positions, DataType::kInt32, Shape({tokens}));
  const TensorView slots_v =
      Upload(keep, slots, DataType::kInt32, Shape({tokens}));
  const TensorView table_v = Upload(
      keep, blocks, DataType::kInt32,
      Shape({static_cast<int64_t>(blocks.size())}));
  const TensorView out_v =
      Empty(keep, DataType::kBFloat16, Shape({tokens, c.hidden_size}));

  const MlaWeights mw = UploadWeights(keep, c, w);

  auto cache = pool->KeyCache(0);
  ASSERT_TRUE(cache.ok()) << cache.status();

  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  auto layer = MlaAttentionLayer::Create(c, tokens, tokens);
  ASSERT_TRUE(layer.ok()) << layer.status();

  ASSERT_TRUE(layer->Forward(x_v, pos_v, slots_v, table_v, tokens, *cache, mw,
                             out_v, &*gemm)
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::vector<float> want = ReferenceMla(c, w, x, tokens);
  const auto got = Download<bf16>(out_v, tokens * c.hidden_size);

  double scale = 0.0;
  for (float v : want) scale = std::max<double>(scale, std::abs(v));
  ASSERT_GT(scale, 0.0) << "degenerate reference";

  double worst = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    worst = std::max<double>(worst,
                             std::abs(__bfloat162float(got[i]) - want[i]));
  }

  // Four GEMMs, two norms, a softmax and a rotation, all in bf16 with 8
  // mantissa bits. 3% of the largest output bounds that; a swapped nope/rope
  // split or a per-head RoPE key misses it by a factor of ten or more.
  EXPECT_LT(worst / scale, 0.03)
      << "worst absolute error " << worst << " against scale " << scale;
}

// The property the cache exists for, as a function of the config so the YaRN
// and non-YaRN paths assert it identically. Prefill 9 tokens in one call;
// then, in a second pool, feed the same 9 tokens one at a time. The last row
// must agree — if the slot mapping, the block-table walk, or the causal bound
// is off by one, it will not, and neither path will look obviously wrong on
// its own.
void ExpectDecodeMatchesPrefill(const ModelConfig& c) {
  const Weights w = MakeWeights(c);

  constexpr int64_t tokens = 9;
  constexpr int64_t block_size = 4;
  const int64_t num_blocks = (tokens + block_size - 1) / block_size;

  const std::vector<float> x =
      RandomTensor(static_cast<size_t>(tokens * c.hidden_size), 1.77f);

  std::vector<DeviceBuffer> keep;
  const MlaWeights mw = UploadWeights(keep, c, w);

  auto gemm = kernels::CublasLtGemm::Create();
  ASSERT_TRUE(gemm.ok()) << gemm.status();

  auto run = [&](bool one_at_a_time) {
    auto pool = KvBlockPool::Create(1, 8, block_size,
                                    MlaAttentionLayer::LayoutFor(c));
    EXPECT_TRUE(pool.ok()) << pool.status();

    std::vector<int32_t> blocks;
    for (int64_t b = 0; b < num_blocks; ++b) {
      auto got = pool->AllocateBlock();
      EXPECT_TRUE(got.ok()) << got.status();
      blocks.push_back(*got);
    }

    auto cache = pool->KeyCache(0);
    EXPECT_TRUE(cache.ok()) << cache.status();

    std::vector<DeviceBuffer> local;
    const TensorView table_v = Upload(
        local, blocks, DataType::kInt32,
        Shape({static_cast<int64_t>(blocks.size())}));

    auto layer = MlaAttentionLayer::Create(c, tokens, tokens);
    EXPECT_TRUE(layer.ok()) << layer.status();

    std::vector<float> last_row;

    const int64_t step = one_at_a_time ? 1 : tokens;
    for (int64_t begin = 0; begin < tokens; begin += step) {
      const int64_t n = std::min(step, tokens - begin);

      std::vector<float> chunk(
          x.begin() + static_cast<ptrdiff_t>(begin * c.hidden_size),
          x.begin() + static_cast<ptrdiff_t>((begin + n) * c.hidden_size));

      std::vector<int32_t> positions(static_cast<size_t>(n));
      std::vector<int32_t> slots(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i) {
        const int64_t t = begin + i;
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(t);
        slots[static_cast<size_t>(i)] =
            blocks[static_cast<size_t>(t / block_size)] * block_size +
            static_cast<int32_t>(t % block_size);
      }

      const TensorView x_v = Upload(local, ToBf16(chunk), DataType::kBFloat16,
                                    Shape({n, c.hidden_size}));
      const TensorView pos_v =
          Upload(local, positions, DataType::kInt32, Shape({n}));
      const TensorView slots_v =
          Upload(local, slots, DataType::kInt32, Shape({n}));
      const TensorView out_v =
          Empty(local, DataType::kBFloat16, Shape({n, c.hidden_size}));

      EXPECT_TRUE(layer->Forward(x_v, pos_v, slots_v, table_v, begin + n,
                                 *cache, mw, out_v, &*gemm)
                      .ok());
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

      if (begin + n == tokens) {
        const auto rows = Download<bf16>(out_v, n * c.hidden_size);
        last_row.assign(static_cast<size_t>(c.hidden_size), 0.0f);
        for (int64_t d = 0; d < c.hidden_size; ++d) {
          last_row[static_cast<size_t>(d)] = __bfloat162float(
              rows[static_cast<size_t>((n - 1) * c.hidden_size + d)]);
        }
      }
    }

    return last_row;
  };

  const std::vector<float> prefilled = run(/*one_at_a_time=*/false);
  const std::vector<float> decoded = run(/*one_at_a_time=*/true);

  ASSERT_EQ(prefilled.size(), decoded.size());

  double scale = 0.0;
  for (float v : prefilled) scale = std::max<double>(scale, std::abs(v));
  ASSERT_GT(scale, 0.0);

  for (size_t i = 0; i < prefilled.size(); ++i) {
    EXPECT_LT(std::abs(prefilled[i] - decoded[i]) / scale, 0.02)
        << "element " << i << ": prefill " << prefilled[i] << " vs decode "
        << decoded[i];
  }
}

TEST_F(MlaLayerTest, DecodingTokenByTokenEqualsPrefillingThemAtOnce) {
  ExpectDecodeMatchesPrefill(SmallMlaConfig());
}

TEST_F(MlaLayerTest, DecodingEqualsPrefillingUnderYarn) {
  // The same cache-consistency property through the YaRN table path: the
  // shared key is rotated by the blended frequencies before caching, so a
  // query rotated differently — or a table that differs between the two call
  // sites — breaks the agreement immediately.
  ModelConfig c = SmallMlaConfig();
  c.yarn_factor = 4.0;
  c.yarn_beta_fast = 32.0;
  c.yarn_beta_slow = 1.0;
  c.yarn_original_max_position = 8;
  c.yarn_truncate = true;
  c.yarn_mscale = 0.707;
  c.yarn_mscale_all_dim = 0.707;

  ExpectDecodeMatchesPrefill(c);
}

TEST_F(MlaLayerTest, DeepSeekV2LiteSoftmaxScaleIsPinned) {
  // The real V2-Lite attention shape and YaRN parameters. The scale must be
  // mscale(0.707)² / sqrt(192) with mscale = 0.1·0.707·ln(40) + 1 — not the
  // plain 1/sqrt(192), and not gpt-oss's 0.1·ln(40)+1 temperature.
  ModelConfig c;
  c.architecture = model::Architecture::kLlama;
  c.hidden_size = 64;
  c.intermediate_size = 128;
  c.num_hidden_layers = 1;
  c.num_attention_heads = 16;
  c.num_key_value_heads = 16;
  c.head_dim = 4;
  c.vocab_size = 128;
  c.rope_theta = 10000.0;
  c.kv_lora_rank = 512;
  c.q_lora_rank = 0;
  c.qk_nope_head_dim = 128;
  c.qk_rope_head_dim = 64;
  c.v_head_dim = 128;
  c.yarn_factor = 40.0;
  c.yarn_beta_fast = 32.0;
  c.yarn_beta_slow = 1.0;
  c.yarn_original_max_position = 4096;
  c.yarn_mscale = 0.707;
  c.yarn_mscale_all_dim = 0.707;

  auto layer = MlaAttentionLayer::Create(c, 1, 1);
  ASSERT_TRUE(layer.ok()) << layer.status();
  EXPECT_NEAR(layer->softmax_scale(), 0.114721f, 1e-4f);

  // Without YaRN the same shape falls back to the plain inverse square root.
  c.yarn_factor = 0.0;
  auto plain = MlaAttentionLayer::Create(c, 1, 1);
  ASSERT_TRUE(plain.ok()) << plain.status();
  EXPECT_NEAR(plain->softmax_scale(), 1.0f / std::sqrt(192.0f), 1e-6f);
}

// --- The kernels the layer is built from ------------------------------------

TEST_F(MlaTest, RopeInPlaceRotatesOnlyTheTail) {
  constexpr int64_t tokens = 3, heads = 2, head_dim = 12, rope_dim = 4;

  std::vector<float> x(
      static_cast<size_t>(tokens * heads * head_dim));
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = Fill(static_cast<int64_t>(i), 3, 0.5f) + 0.7f;
  }
  x = RoundTrip(x);

  // Position 0 makes the rotation the identity (angle 0), so any change at all
  // is a bug; the leading columns must be untouched at every position.
  std::vector<int32_t> positions{0, 5, 11};

  std::vector<DeviceBuffer> keep;
  const TensorView x_v = Upload(keep, ToBf16(x), DataType::kBFloat16,
                                Shape({tokens, heads, head_dim}));
  const TensorView pos_v =
      Upload(keep, positions, DataType::kInt32, Shape({tokens}));

  ASSERT_TRUE(kernels::MlaRopeInPlace(x_v, rope_dim, pos_v, 10000.0f).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got = Download<bf16>(x_v, tokens * heads * head_dim);

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t d = 0; d < head_dim - rope_dim; ++d) {
        const size_t i = static_cast<size_t>((t * heads + h) * head_dim + d);
        EXPECT_FLOAT_EQ(__bfloat162float(got[i]), x[i])
            << "the nope part was rotated at token " << t << " head " << h
            << " column " << d;
      }
    }
  }

  // Token 0 sits at position 0, so even its tail is unchanged.
  for (int64_t h = 0; h < heads; ++h) {
    for (int64_t d = head_dim - rope_dim; d < head_dim; ++d) {
      const size_t i = static_cast<size_t>(h * head_dim + d);
      EXPECT_NEAR(__bfloat162float(got[i]), x[i], 1e-2f)
          << "position 0 must be the identity rotation";
    }
  }

  // Token 1 sits at position 5, so its tail must have moved.
  double delta = 0.0;
  for (int64_t d = head_dim - rope_dim; d < head_dim; ++d) {
    const size_t i = static_cast<size_t>(heads * head_dim + d);
    delta += std::abs(__bfloat162float(got[i]) - x[i]);
  }
  EXPECT_GT(delta, 1e-2) << "a non-zero position left the tail untouched";
}

TEST_F(MlaTest, RopeFromTableMatchesTheClosedFormAtFactorOne) {
  // With inv_freq[j] = theta^(-2j/rope_dim) and attention factor 1, the table
  // kernel is the in-place kernel by definition. This is the equivalence that
  // lets the YaRN path share every downstream test with the closed form.
  constexpr int64_t tokens = 4, heads = 3, head_dim = 16, rope_dim = 8;
  constexpr float theta = 10000.0f;

  std::vector<float> x(static_cast<size_t>(tokens * heads * head_dim));
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = Fill(static_cast<int64_t>(i), 5, 0.9f) + 0.3f;
  }
  x = RoundTrip(x);

  const std::vector<int32_t> positions{0, 3, 17, 40};

  std::vector<float> inv_freq(rope_dim / 2);
  for (int64_t j = 0; j < rope_dim / 2; ++j) {
    inv_freq[static_cast<size_t>(j)] = std::pow(
        theta, -2.0f * static_cast<float>(j) / static_cast<float>(rope_dim));
  }

  std::vector<DeviceBuffer> keep;
  const TensorView a_v = Upload(keep, ToBf16(x), DataType::kBFloat16,
                                Shape({tokens, heads, head_dim}));
  const TensorView b_v = Upload(keep, ToBf16(x), DataType::kBFloat16,
                                Shape({tokens, heads, head_dim}));
  const TensorView pos_v =
      Upload(keep, positions, DataType::kInt32, Shape({tokens}));
  const TensorView freq_v =
      Upload(keep, inv_freq, DataType::kFloat, Shape({rope_dim / 2}));

  ASSERT_TRUE(kernels::MlaRopeInPlace(a_v, rope_dim, pos_v, theta).ok());
  ASSERT_TRUE(
      kernels::MlaRopeFromTable(b_v, rope_dim, pos_v, freq_v, 1.0f).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto a = Download<bf16>(a_v, tokens * heads * head_dim);
  const auto b = Download<bf16>(b_v, tokens * heads * head_dim);

  // Not bitwise: the in-place kernel raises theta with device fast math while
  // the table came from host std::pow. One bf16 step of slack absorbs that.
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(__bfloat162float(a[i]), __bfloat162float(b[i]), 1.5e-2f)
        << "element " << i;
  }
}

TEST_F(MlaTest, YarnFrequenciesAreExactAtTheLadderEnds) {
  // DeepSeek-V2-Lite's YaRN over its 64-wide rope sub-vector. The correction
  // window computes to [10.47, 22.51], truncated to [10, 23] — so dimensions
  // at or below 10 must be *pure extrapolation* (the original frequencies,
  // keeping local detail) and dimensions at or past 23 pure interpolation
  // (the original divided by the factor, reaching the longer context). The
  // ladder ends are exact, not blended, and pinning them catches a ramp with
  // its endpoints off by one.
  constexpr int64_t rope_dim = 64;
  constexpr double theta = 10000.0, factor = 40.0;

  std::vector<float> got(rope_dim / 2);
  const float attn = kernels::ComputeYarnInvFreq(
      rope_dim, theta, factor, /*beta_fast=*/32.0, /*beta_slow=*/1.0,
      /*original_max=*/4096, /*truncate=*/true, got.data());

  for (int64_t j = 0; j < rope_dim / 2; ++j) {
    const double original = std::pow(theta, -2.0 * static_cast<double>(j) /
                                                static_cast<double>(rope_dim));
    if (j <= 10) {
      EXPECT_NEAR(got[static_cast<size_t>(j)], original, original * 1e-6)
          << "dimension " << j << " should extrapolate";
    } else if (j >= 23) {
      EXPECT_NEAR(got[static_cast<size_t>(j)], original / factor,
                  original / factor * 1e-6)
          << "dimension " << j << " should interpolate";
    }
    if (j > 0) {
      EXPECT_LT(got[static_cast<size_t>(j)], got[static_cast<size_t>(j) - 1])
          << "frequencies must descend";
    }
  }

  // The returned temperature is HF's default form, YarnMscale at coefficient
  // 1 — and DeepSeek's parameterized form must agree with it there.
  EXPECT_NEAR(attn, kernels::YarnMscale(factor, 1.0), 1e-6f);
  EXPECT_NEAR(kernels::YarnMscale(factor, 0.707), 1.260804f, 1e-5f);
  EXPECT_FLOAT_EQ(kernels::YarnMscale(0.5, 0.707), 1.0f);  // factor <= 1
  EXPECT_FLOAT_EQ(kernels::YarnMscale(factor, 0.0), 1.0f);  // coefficient 0
}

TEST_F(MlaTest, AppendAndGatherRoundTripThroughTheBlockTable) {
  constexpr int64_t latent_dim = 12, rope_dim = 4, block_size = 4, tokens = 10;
  const int64_t width = latent_dim + rope_dim;

  ModelConfig c;
  c.kv_lora_rank = latent_dim;
  c.qk_rope_head_dim = rope_dim;

  auto pool = KvBlockPool::Create(1, 6, block_size,
                                  MlaAttentionLayer::LayoutFor(c));
  ASSERT_TRUE(pool.ok()) << pool.status();

  // Blocks allocated out of order, which is the case a block-table walk that
  // assumed contiguity would get wrong.
  std::vector<int32_t> blocks;
  for (int64_t b = 0; b < 3; ++b) {
    auto got = pool->AllocateBlock();
    ASSERT_TRUE(got.ok());
    blocks.push_back(*got);
  }
  std::swap(blocks[0], blocks[2]);

  const std::vector<float> latent =
      RandomTensor(static_cast<size_t>(tokens * latent_dim), 0.9f);
  const std::vector<float> rope =
      RandomTensor(static_cast<size_t>(tokens * rope_dim), 1.6f);

  std::vector<int32_t> slots(static_cast<size_t>(tokens));
  for (int64_t t = 0; t < tokens; ++t) {
    slots[static_cast<size_t>(t)] =
        blocks[static_cast<size_t>(t / block_size)] * block_size +
        static_cast<int32_t>(t % block_size);
  }

  std::vector<DeviceBuffer> keep;
  const TensorView latent_v = Upload(keep, ToBf16(latent), DataType::kBFloat16,
                                     Shape({tokens, latent_dim}));
  const TensorView rope_v = Upload(keep, ToBf16(rope), DataType::kBFloat16,
                                   Shape({tokens, rope_dim}));
  const TensorView slots_v =
      Upload(keep, slots, DataType::kInt32, Shape({tokens}));
  const TensorView table_v = Upload(
      keep, blocks, DataType::kInt32,
      Shape({static_cast<int64_t>(blocks.size())}));
  const TensorView out_v =
      Empty(keep, DataType::kBFloat16, Shape({tokens, width}));

  auto cache = pool->KeyCache(0);
  ASSERT_TRUE(cache.ok()) << cache.status();

  ASSERT_TRUE(
      kernels::MlaAppendLatent(latent_v, rope_v, *cache, slots_v).ok());
  ASSERT_TRUE(
      kernels::MlaGatherLatents(*cache, table_v, tokens, out_v).ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto got = Download<bf16>(out_v, tokens * width);

  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t d = 0; d < latent_dim; ++d) {
      EXPECT_FLOAT_EQ(
          __bfloat162float(got[static_cast<size_t>(t * width + d)]),
          latent[static_cast<size_t>(t * latent_dim + d)])
          << "latent, token " << t << " column " << d;
    }
    for (int64_t d = 0; d < rope_dim; ++d) {
      EXPECT_FLOAT_EQ(
          __bfloat162float(got[static_cast<size_t>(t * width + latent_dim + d)]),
          rope[static_cast<size_t>(t * rope_dim + d)])
          << "rope key, token " << t << " column " << d;
    }
  }
}

}  // namespace
}  // namespace inferx
