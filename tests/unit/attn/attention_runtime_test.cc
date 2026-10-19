// End-to-end tests for the attention runtime on CPU devices with the
// forced reference backend: prefill/decode/mixed batches, MHA/MQA/GQA,
// sliding-window roll-off, and prefix-cache reuse through the implemented
// cache facade. Outputs are checked against a float oracle computed from
// the source activations (docs/attention.md §15.2 differential strategy).

#include "inferx/ops/attention/runtime.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "inferx/core/device_runtime.h"
#include "inferx/ops/op_context.h"

namespace inferx::attn {
namespace {

constexpr float kEps = 1e-4f;  // f32 activations and cache end to end.

/// Deterministic activations: token t, head h, dim d -> sin mixture.
float Activation(int token, int head, int dim) {
  return std::sin((token * 7 + head * 13 + dim * 3) * 0.37f);
}

/// The float oracle: one query against a request's K/V history under the
/// mask, straight from the source activations.
std::vector<float> OracleAttention(const std::vector<float>& query,
                                   const std::vector<std::vector<float>>& keys,
                                   const std::vector<std::vector<float>>& values, int kv_head,
                                   int head_dim, float scale, const MaskSpec& mask, int pos) {
  std::int64_t lo = 0;
  switch (mask.kind) {
    case MaskKind::kSlidingWindow: lo = std::max<std::int64_t>(0, pos - mask.window + 1); break;
    case MaskKind::kChunked: lo = (pos / mask.window) * mask.window; break;
    default: break;
  }
  std::vector<float> scores;
  float max_score = -1e30f;
  for (int j = static_cast<int>(lo); j <= pos; ++j) {
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += query[static_cast<size_t>(d)] *
             keys[static_cast<size_t>(j)][static_cast<size_t>(kv_head * head_dim + d)];
    }
    const float s = dot * scale;
    scores.push_back(s);
    max_score = std::max(max_score, s);
  }
  float denom = 0.0f;
  for (float& s : scores) {
    s = std::exp(s - max_score);
    denom += s;
  }
  std::vector<float> out(static_cast<size_t>(head_dim), 0.0f);
  size_t idx = 0;
  for (int j = static_cast<int>(lo); j <= pos; ++j, ++idx) {
    const float w = scores[idx] / denom;
    for (int d = 0; d < head_dim; ++d) {
      out[static_cast<size_t>(d)] += w * values[static_cast<size_t>(j)][static_cast<size_t>(kv_head * head_dim + d)];
    }
  }
  return out;
}

/// One request's evolving history for the oracle and activation building.
struct OracleRequest {
  std::vector<std::vector<float>> keys;    // Per token [Hkv * D].
  std::vector<std::vector<float>> values;  // Per token [Hkv * D].
  void Append(int token, int kv_heads, int head_dim) {
    std::vector<float> k(static_cast<size_t>(kv_heads * head_dim));
    std::vector<float> v(static_cast<size_t>(kv_heads * head_dim));
    for (int h = 0; h < kv_heads; ++h) {
      for (int d = 0; d < head_dim; ++d) {
        k[static_cast<size_t>(h * head_dim + d)] = Activation(token, h + 100, d);
        v[static_cast<size_t>(h * head_dim + d)] = Activation(token, h + 200, d);
      }
    }
    keys.push_back(std::move(k));
    values.push_back(std::move(v));
  }
};

AttentionSpec MakeSpec(int q_heads, int kv_heads, MaskKind kind = MaskKind::kCausal,
                       int window = 0) {
  AttentionSpec spec;
  spec.params = DenseParams{.num_q_heads = q_heads, .num_kv_heads = kv_heads, .head_dim = 8};
  spec.q_dtype = DataType::kFloat32;
  spec.kv_dtype = DataType::kFloat32;
  spec.mask = MaskSpec{kind, window, 0};
  return spec;
}

/// Drives one step: reserves through the facade, builds the batch, runs
/// the op, and checks every output token against the oracle.
class StepDriver {
 public:
  StepDriver(AttentionRuntime& runtime, AttentionOp& op, DeviceRuntime& device_runtime)
      : runtime_(runtime), op_(op), device_runtime_(device_runtime) {}

  void Run(const std::vector<uint64_t>& reqs, const std::vector<int>& computed,
           const std::vector<int>& chunks, const std::vector<std::vector<TokenId>>& histories,
           std::vector<OracleRequest>& oracles) {
    const AttentionSpec& spec = op_.GetSpec();
    const auto& dense = std::get<DenseParams>(spec.params);
    const int q_heads = dense.num_q_heads;
    const int kv_heads = dense.num_kv_heads;
    const int head_dim = dense.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const size_t num_reqs = reqs.size();
    int total_tokens = 0;
    for (int c : chunks) total_tokens += c;

    std::vector<std::int32_t> seq_lens(num_reqs), query_lens(num_reqs);
    for (size_t r = 0; r < num_reqs; ++r) {
      query_lens[r] = chunks[r];
      seq_lens[r] = computed[r] + chunks[r];
      ASSERT_TRUE(runtime_.Cache().Reserve(reqs[r], GrowthTarget{computed[r], seq_lens[r]}).ok());
    }

    BatchDesc desc;
    desc.num_reqs = static_cast<std::int32_t>(num_reqs);
    desc.num_tokens = total_tokens;
    desc.query_lens = query_lens;
    desc.seq_lens = seq_lens;
    desc.max_query_len = *std::max_element(query_lens.begin(), query_lens.end());
    desc.max_seq_len = *std::max_element(seq_lens.begin(), seq_lens.end());
    desc.block_size = static_cast<std::int32_t>(runtime_.GetPlan().block_size);
    desc.num_decodes = static_cast<std::int32_t>(
        std::count_if(chunks.begin(), chunks.end(), [](int c) { return c == 1; }));
    desc.num_decode_tokens = desc.num_decodes;
    desc.num_prefills = static_cast<std::int32_t>(num_reqs) - desc.num_decodes;
    desc.num_prefill_tokens = total_tokens - desc.num_decode_tokens;
    std::vector<BlockId> flat_blocks;
    for (size_t r = 0; r < num_reqs; ++r) {
      const auto blocks = runtime_.Cache().GroupBlocks(reqs[r], 0);
      flat_blocks.insert(flat_blocks.end(), blocks.begin(), blocks.end());
    }
    desc.block_ids[0] = flat_blocks;
    desc.block_ids_num_pages[0] = static_cast<std::int32_t>(flat_blocks.size());

    // Activations in batch order (decodes and prefills interleaved as
    // given; the builder computes positions from seq/query lens).
    std::vector<float> q(static_cast<size_t>(total_tokens * q_heads * head_dim));
    std::vector<float> k(static_cast<size_t>(total_tokens * kv_heads * head_dim));
    std::vector<float> v(static_cast<size_t>(total_tokens * kv_heads * head_dim));
    std::vector<float> out(static_cast<size_t>(total_tokens * q_heads * head_dim));
    int t = 0;
    for (size_t r = 0; r < num_reqs; ++r) {
      for (int j = 0; j < chunks[r]; ++j, ++t) {
        const int token =
            static_cast<int>(histories[r][static_cast<size_t>(computed[r] + j)]);
        for (int h = 0; h < q_heads; ++h) {
          for (int d = 0; d < head_dim; ++d) {
            q[static_cast<size_t>((t * q_heads + h) * head_dim + d)] =
                Activation(token, h, d);
          }
        }
        oracles[r].Append(token, kv_heads, head_dim);
        for (int h = 0; h < kv_heads; ++h) {
          for (int d = 0; d < head_dim; ++d) {
            k[static_cast<size_t>((t * kv_heads + h) * head_dim + d)] =
                oracles[r].keys.back()[static_cast<size_t>(h * head_dim + d)];
            v[static_cast<size_t>((t * kv_heads + h) * head_dim + d)] =
                oracles[r].values.back()[static_cast<size_t>(h * head_dim + d)];
          }
        }
      }
    }

    ops::OpContext ctx(device_runtime_, DeviceStream{});
    auto step = runtime_.BeginStep(desc, ctx);
    ASSERT_TRUE(step.ok()) << step.status();

    Inputs in;
    in.q = Flat(q, total_tokens, q_heads, head_dim);
    in.k = Flat(k, total_tokens, kv_heads, head_dim);
    in.v = Flat(v, total_tokens, kv_heads, head_dim);
    const TensorView out_view = Flat(out, total_tokens, q_heads, head_dim);
    ASSERT_TRUE(op_.Forward(*step, in, out_view).ok());

    const int group = q_heads / kv_heads;
    t = 0;
    for (size_t r = 0; r < num_reqs; ++r) {
      for (int j = 0; j < chunks[r]; ++j, ++t) {
        const int pos = computed[r] + j;
        for (int h = 0; h < q_heads; ++h) {
          std::vector<float> query(static_cast<size_t>(head_dim));
          for (int d = 0; d < head_dim; ++d) {
            query[static_cast<size_t>(d)] = q[static_cast<size_t>((t * q_heads + h) * head_dim + d)];
          }
          const std::vector<float> expected =
              OracleAttention(query, oracles[r].keys, oracles[r].values, h / group, head_dim,
                              scale, spec.mask, pos);
          for (int d = 0; d < head_dim; ++d) {
            EXPECT_NEAR(out[static_cast<size_t>((t * q_heads + h) * head_dim + d)],
                        expected[static_cast<size_t>(d)], kEps)
                << "req " << r << " pos " << pos << " head " << h << " dim " << d;
          }
        }
      }
    }

    for (size_t r = 0; r < num_reqs; ++r) {
      const std::vector<TokenId> prompt(
          histories[r].begin(), histories[r].begin() + seq_lens[static_cast<size_t>(r)]);
      ASSERT_TRUE(
          runtime_.Cache().PublishComputed(reqs[r], prompt, {}, seq_lens[static_cast<size_t>(r)])
              .ok());
    }
  }

 private:
  static TensorView Flat(const std::vector<float>& data, int tokens, int heads, int dim) {
    const std::int64_t row = static_cast<std::int64_t>(heads) * dim;
    return TensorView{const_cast<float*>(data.data()), DataType::kFloat32, 3,
                      {tokens, heads, dim}, {row, static_cast<std::int64_t>(dim), 1}};
  }

  AttentionRuntime& runtime_;
  AttentionOp& op_;
  DeviceRuntime& device_runtime_;
};

std::vector<TokenId> History(int count, int base) {
  std::vector<TokenId> tokens(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    tokens[static_cast<size_t>(i)] = static_cast<TokenId>(base + i);
  }
  return tokens;
}

class AttentionRuntimeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto runtime = RuntimeFor(Device::Cpu());
    ASSERT_TRUE(runtime.ok()) << runtime.status();
    device_runtime_ = *runtime;
  }

  StatusOr<std::unique_ptr<AttentionRuntime>> MakeRuntime(const AttentionSpec& spec) {
    auto registry = std::make_unique<BackendRegistry>();
    RegisterBuiltinBackends(*registry);
    AttentionConfig config;
    config.block_size = 4;
    config.max_model_len = 64;
    config.max_seqs = 4;
    config.max_batch_tokens = 256;
    config.cache_bytes_budget = 1 << 20;
    config.device = Device::Cpu();
    config.enable_prefix_caching = true;
    config.forced_backend = "reference";
    return AttentionRuntime::Create(std::move(registry), config,
                                     std::span<const AttentionSpec>(&spec, 1));
  }

  DeviceRuntime* device_runtime_ = nullptr;
};

TEST_F(AttentionRuntimeTest, PrefillThenDecodeAcrossHeadShapes) {
  // MHA (kv == q), MQA (kv == 1), GQA (1 < kv < q): one capability row.
  for (const auto [q_heads, kv_heads] : std::vector<std::pair<int, int>>{{4, 4}, {4, 1}, {4, 2}}) {
    const AttentionSpec spec = MakeSpec(q_heads, kv_heads);
    auto made = MakeRuntime(spec);
    ASSERT_TRUE(made.ok()) << made.status();
    auto op = (*made)->MakeOp(0);
    ASSERT_TRUE(op.ok()) << op.status();
    StepDriver driver(**made, *op, *device_runtime_);

    std::vector<OracleRequest> oracles(1);
    const std::vector<std::vector<TokenId>> histories = {History(30, 100)};
    driver.Run({1}, {0}, {7}, histories, oracles);    // Prefill.
    driver.Run({1}, {7}, {1}, histories, oracles);    // Decode.
    driver.Run({1}, {8}, {5}, histories, oracles);    // Chunked verify-like step.
  }
}

TEST_F(AttentionRuntimeTest, MixedPrefillAndDecodeShareAStep) {
  const AttentionSpec spec = MakeSpec(4, 2);
  auto made = MakeRuntime(spec);
  ASSERT_TRUE(made.ok()) << made.status();
  auto op = (*made)->MakeOp(0);
  ASSERT_TRUE(op.ok()) << op.status();
  StepDriver driver(**made, *op, *device_runtime_);

  std::vector<OracleRequest> oracles(2);
  const std::vector<std::vector<TokenId>> histories = {History(20, 50), History(20, 700)};
  // Request 1 prefills first, so its history is really in the cache.
  driver.Run({1}, {0}, {4}, histories, oracles);
  // Then request 1 decodes (1 new token) while request 2 prefills
  // (6 tokens): a mixed batch in one launch.
  driver.Run({1, 2}, {4, 0}, {1, 6}, histories, oracles);
  EXPECT_EQ(oracles[0].keys.size(), 5u);
  EXPECT_EQ(oracles[1].keys.size(), 6u);
}

TEST_F(AttentionRuntimeTest, SlidingWindowRollsOffAndStaysCorrect) {
  const AttentionSpec spec = MakeSpec(4, 2, MaskKind::kSlidingWindow, /*window=*/4);
  auto made = MakeRuntime(spec);
  ASSERT_TRUE(made.ok()) << made.status();
  auto op = (*made)->MakeOp(0);
  ASSERT_TRUE(op.ok()) << op.status();
  StepDriver driver(**made, *op, *device_runtime_);

  std::vector<OracleRequest> oracles(1);
  const std::vector<std::vector<TokenId>> histories = {History(30, 100)};
  driver.Run({1}, {0}, {8}, histories, oracles);   // Two blocks at bs=4.
  // Rolling decode: each step crosses the floor; entries become null.
  for (int step = 0; step < 4; ++step) {
    const int computed = 8 + step;
    driver.Run({1}, {computed}, {1}, histories, oracles);
    const auto blocks = (*made)->Cache().GroupBlocks(1, 0);
    ASSERT_FALSE(blocks.empty());
    // The first block falls below the window floor once computed >= 9.
    if (computed >= 9) {
      EXPECT_EQ(blocks[0], kNullBlockId) << "first block should have rolled off at " << computed;
    }
  }
}

TEST_F(AttentionRuntimeTest, PrefixCachingAdoptsAcrossRequests) {
  const AttentionSpec spec = MakeSpec(4, 2);
  auto made = MakeRuntime(spec);
  ASSERT_TRUE(made.ok()) << made.status();
  auto op = (*made)->MakeOp(0);
  ASSERT_TRUE(op.ok()) << op.status();
  StepDriver driver(**made, *op, *device_runtime_);

  // Producer: compute 8 tokens (2 blocks at bs=4), publish, release.
  std::vector<OracleRequest> producer_oracles(1);
  const std::vector<std::vector<TokenId>> prompt = {History(8, 100)};
  driver.Run({1}, {0}, {8}, prompt, producer_oracles);
  ASSERT_TRUE((*made)->Cache().Release(1).ok());

  // Consumer, same prompt and salt: the n-1 cap (7) truncates the search
  // to whole blocks under it -> one block, 4 tokens.
  auto capped = (*made)->Cache().AdoptPrefix(2, prompt[0], /*salt=*/"", /*max=*/7);
  ASSERT_TRUE(capped.ok()) << capped.status();
  EXPECT_EQ(capped->cached_tokens, 4);
  ASSERT_EQ(capped->group_block_ids[0].size(), 1u);
  ASSERT_TRUE((*made)->Cache().Release(2).ok());

  // Uncapped: both blocks adopt (8 tokens), and decoding on top of the
  // adopted prefix produces oracle-correct output -- the adopted blocks
  // carry exactly the producer's bytes.
  auto adopted = (*made)->Cache().AdoptPrefix(3, prompt[0], "", /*max=*/8);
  ASSERT_TRUE(adopted.ok()) << adopted.status();
  EXPECT_EQ(adopted->cached_tokens, 8);
  std::vector<OracleRequest> consumer_oracles(1);
  for (int i = 0; i < 8; ++i) {
    consumer_oracles[0].Append(100 + i, /*kv_heads=*/2, /*head_dim=*/8);
  }
  const std::vector<std::vector<TokenId>> history = {History(30, 100)};
  driver.Run({3}, {8}, {1}, history, consumer_oracles);

  // A different salt never adopts.
  auto tenant = (*made)->Cache().AdoptPrefix(4, prompt[0], "tenant-b", 8);
  ASSERT_TRUE(tenant.ok()) << tenant.status();
  EXPECT_EQ(tenant->cached_tokens, 0);
}

}  // namespace
}  // namespace inferx::attn
