// GPU differential test: the FlashInfer-backed dense backend vs a host
// float oracle (docs/attention.md §15.2). Causal GQA prefill + decode
// through the full runtime stack (cache facade, metadata builder, kernel)
// on a CUDA device.

#include "inferx/ops/attention/runtime.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "inferx/core/device_runtime.h"
#include "inferx/core/device_buffer.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/op_context.h"

namespace inferx::attn {
namespace {

constexpr int kHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kDim = 64;
constexpr float kTolerance = 8e-2f;  // bf16 cache, accumulation, softmax.

float Activation(int token, int head, int dim) {
  return std::sin((token * 7 + head * 13 + dim * 3) * 0.37f);
}

std::vector<float> OracleAttention(const std::vector<float>& query,
                                   const std::vector<std::vector<float>>& keys,
                                   const std::vector<std::vector<float>>& values, int kv_head,
                                   float scale, int pos) {
  std::vector<float> scores;
  float max_score = -1e30f;
  for (int j = 0; j <= pos; ++j) {
    float dot = 0.0f;
    for (int d = 0; d < kDim; ++d) {
      dot += query[static_cast<size_t>(d)] *
             keys[static_cast<size_t>(j)][static_cast<size_t>(kv_head * kDim + d)];
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
  std::vector<float> out(static_cast<size_t>(kDim), 0.0f);
  for (int j = 0; j <= pos; ++j) {
    const float w = scores[static_cast<size_t>(j)] / denom;
    for (int d = 0; d < kDim; ++d) {
      out[static_cast<size_t>(d)] +=
          w * values[static_cast<size_t>(j)][static_cast<size_t>(kv_head * kDim + d)];
    }
  }
  return out;
}

AttentionSpec GqaSpec() {
  AttentionSpec spec;
  spec.params = DenseParams{.num_q_heads = kHeads, .num_kv_heads = kKvHeads, .head_dim = kDim};
  spec.q_dtype = DataType::kBFloat16;
  spec.kv_dtype = DataType::kBFloat16;
  return spec;
}

/// bf16 host encoding (round to nearest even) for the H2D activations.
std::vector<uint16_t> ToBf16(const std::vector<float>& in) {
  std::vector<uint16_t> out(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    uint32_t bits;
    std::memcpy(&bits, &in[i], sizeof(bits));
    out[i] = static_cast<uint16_t>((bits + ((bits >> 16) & 1) + 0x7fff) >> 16);
  }
  return out;
}

float FromBf16(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

TEST(FlashDenseGpuProbe, DirectWritePagedKv) {
  auto device_result = RuntimeFor(Device::Cuda(0));
  if (!device_result.ok()) GTEST_SKIP() << "no CUDA runtime";
  DeviceRuntime* rt = *device_result;
  const Device device = rt->device();
  ops::OpContext ctx(*rt, DeviceStream{});

  // One token, one kv head, head_dim 8, one block of 4.
  const int block_size = 4;
  auto alloc_i32 = [&](std::vector<int32_t> host) -> Tensor {
    auto buf = DeviceBuffer::Allocate(host.size() * 4, device);
    EXPECT_TRUE(buf.ok()) << buf.status();
    EXPECT_TRUE(rt->Memory().CopyAsync(buf->data(), host.data(), host.size() * 4,
                                       CopyKind::kHostToDevice, DeviceStream{}).ok());
    auto t = Tensor::FromBlob(buf->data(), DataType::kInt32, Shape({int64_t(host.size())}),
                              device);
    EXPECT_TRUE(t.ok()) << t.status();
    static std::vector<DeviceBuffer> keep;
    keep.push_back(std::move(*buf));
    return std::move(*t);
  };
  auto alloc_bf16 = [&](size_t floats) -> Tensor {
    auto buf = DeviceBuffer::Allocate(floats * 2, device);
    EXPECT_TRUE(buf.ok()) << buf.status();
    auto t = Tensor::FromBlob(buf->data(), DataType::kBFloat16, Shape({int64_t(floats)}),
                              device);
    EXPECT_TRUE(t.ok()) << t.status();
    static std::vector<DeviceBuffer> keep2;
    keep2.push_back(std::move(*buf));
    return std::move(*t);
  };
  // Build a full runtime first: if anything in its creation poisons the
  // CUDA error state, the direct probe below will surface it.
  {
    const AttentionSpec spec = GqaSpec();
    auto registry = std::make_unique<BackendRegistry>();
    RegisterBuiltinBackends(*registry);
    AttentionConfig config;
    config.block_size = 16;
    config.max_model_len = 64;
    config.max_seqs = 4;
    config.max_batch_tokens = 256;
    config.cache_bytes_budget = 32 << 20;
    config.device = device;
    auto made = AttentionRuntime::Create(std::move(registry), config,
                                         std::span<const AttentionSpec>(&spec, 1));
    ASSERT_TRUE(made.ok()) << made.status();
    ops::OpContext begin_ctx(*rt, DeviceStream{});
    const std::vector<std::int32_t> ql{20}, sl{20};
    std::vector<BlockId> blocks{1, 2};
    BatchDesc desc;
    desc.num_reqs = 1;
    desc.num_tokens = 20;
    desc.query_lens = ql;
    desc.seq_lens = sl;
    desc.block_size = 16;
    desc.block_ids[0] = blocks;
    desc.block_ids_num_pages[0] = 2;
    auto step = (*made)->BeginStep(desc, begin_ctx);
    ASSERT_TRUE(step.ok()) << step.status();
    auto op = (*made)->MakeOp(0);
    ASSERT_TRUE(op.ok()) << op.status();
    // A one-token decode through the full stack (cache write included).
    auto qbuf = DeviceBuffer::Allocate(20 * 4 * 64 * 2, device);
    ASSERT_TRUE(qbuf.ok()) << qbuf.status();
    const TensorView qv{qbuf->data(), DataType::kBFloat16, 3, {20, 4, 64}, {256, 64, 1}};
    const TensorView kv{qbuf->data(), DataType::kBFloat16, 3, {20, 2, 64}, {128, 64, 1}};
    auto obuf = DeviceBuffer::Allocate(20 * 4 * 64 * 2, device);
    const TensorView ov{obuf->data(), DataType::kBFloat16, 3, {20, 4, 64}, {256, 64, 1}};
    Inputs in;
    in.q = qv;
    in.k = kv;
    in.v = kv;
    const Status forward = op->Forward(*step, in, ov);
    ASSERT_TRUE(forward.ok()) << forward;
  }

  Tensor k = alloc_bf16(8), v = alloc_bf16(8);
  Tensor positions = alloc_i32({0});
  Tensor batch_indices = alloc_i32({0});
  Tensor kv_indptr = alloc_i32({0, 1});
  Tensor kv_indices = alloc_i32({1});  // block id 1 (0 is the null block)
  Tensor key_cache = alloc_bf16(2 * 4 * 1 * 8);
  Tensor value_cache = alloc_bf16(2 * 4 * 1 * 8);
  auto k2 = Tensor::FromBlob(k.Data(), DataType::kBFloat16, Shape({1, 8}), device);
  auto v2 = Tensor::FromBlob(v.Data(), DataType::kBFloat16, Shape({1, 8}), device);
  auto kc = Tensor::FromBlob(key_cache.Data(), DataType::kBFloat16, Shape({2, 4, 1, 8}), device);
  auto vc = Tensor::FromBlob(value_cache.Data(), DataType::kBFloat16, Shape({2, 4, 1, 8}),
                             device);
  ASSERT_TRUE(k2.ok() && v2.ok() && kc.ok() && vc.ok());
  const Status written =
      ops::WritePagedKv(ctx, *k2, *v2, positions, batch_indices, kv_indptr, kv_indices, *kc,
                        *vc, block_size);
  ASSERT_TRUE(written.ok()) << written;
  SUCCEED();
}

class FlashDenseGpuTest : public testing::TestWithParam<int> {};

TEST_P(FlashDenseGpuTest, PrefillAndDecodeMatchTheOracle) {
  auto device_result = RuntimeFor(Device::Cuda(0));
  if (!device_result.ok()) {
    GTEST_SKIP() << "no CUDA runtime available: " << device_result.status();
  }
  DeviceRuntime* device_runtime = *device_result;
  const Device device = device_runtime->device();

  const AttentionSpec spec = GqaSpec();
  auto registry = std::make_unique<BackendRegistry>();
  RegisterBuiltinBackends(*registry);
  AttentionConfig config;
  config.block_size = 16;
  config.max_model_len = 64;
  config.max_seqs = 4;
  config.max_batch_tokens = 256;
  config.cache_bytes_budget = 32 << 20;
  config.device = device;
  config.forced_backend = "flash_dense";
  config.enable_prefix_caching = false;  // Isolate the kernel path.
  auto made = AttentionRuntime::Create(std::move(registry), config,
                                       std::span<const AttentionSpec>(&spec, 1));
  ASSERT_TRUE(made.ok()) << made.status();
  auto op = (*made)->MakeOp(0);
  ASSERT_TRUE(op.ok()) << op.status();
  AttentionRuntime& runtime = **made;

  const float scale = 1.0f / std::sqrt(static_cast<float>(kDim));
  std::vector<std::vector<float>> keys, values;  // Oracle history (f32).

  auto run_step = [&](int computed, int chunk) {
    SCOPED_TRACE(testing::Message() << "computed=" << computed << " chunk=" << chunk);
    const std::int32_t seq_len = computed + chunk;
    ASSERT_TRUE(runtime.Cache().Reserve(1, GrowthTarget{computed, seq_len}).ok());

    BatchDesc desc;
    desc.num_reqs = 1;
    desc.num_tokens = chunk;
    const std::vector<std::int32_t> query_lens{chunk};
    const std::vector<std::int32_t> seq_lens{seq_len};
    desc.query_lens = query_lens;
    desc.seq_lens = seq_lens;
    desc.max_query_len = chunk;
    desc.max_seq_len = seq_len;
    desc.block_size = static_cast<std::int32_t>(runtime.GetPlan().block_size);
    desc.num_prefills = chunk > 1 ? 1 : 0;
    desc.num_prefill_tokens = chunk;
    const auto blocks = runtime.Cache().GroupBlocks(1, 0);
    desc.block_ids[0] = blocks;
    desc.block_ids_num_pages[0] = static_cast<std::int32_t>(blocks.size());

    // Host activations, then one H2D per tensor.
    std::vector<float> q(static_cast<size_t>(chunk * kHeads * kDim));
    std::vector<float> k(static_cast<size_t>(chunk * kKvHeads * kDim));
    std::vector<float> v(static_cast<size_t>(chunk * kKvHeads * kDim));
    for (int j = 0; j < chunk; ++j) {
      const int token = 10 + computed + j;
      for (int h = 0; h < kHeads; ++h) {
        for (int d = 0; d < kDim; ++d) {
          q[static_cast<size_t>((j * kHeads + h) * kDim + d)] = Activation(token, h, d);
        }
      }
      keys.emplace_back(static_cast<size_t>(kKvHeads * kDim));
      values.emplace_back(static_cast<size_t>(kKvHeads * kDim));
      for (int h = 0; h < kKvHeads; ++h) {
        for (int d = 0; d < kDim; ++d) {
          keys.back()[static_cast<size_t>(h * kDim + d)] = Activation(token, h + 100, d);
          values.back()[static_cast<size_t>(h * kDim + d)] = Activation(token, h + 200, d);
          k[static_cast<size_t>((j * kKvHeads + h) * kDim + d)] = keys.back()[static_cast<size_t>(h * kDim + d)];
          v[static_cast<size_t>((j * kKvHeads + h) * kDim + d)] = values.back()[static_cast<size_t>(h * kDim + d)];
        }
      }
    }

    // Host activations -> device bf16 buffers; kept alive for the step.
    std::vector<DeviceBuffer> keep;
    std::vector<std::vector<uint16_t>> host_buffers;
    auto upload_flat = [&](const std::vector<float>& host, int heads) -> TensorView {
      host_buffers.push_back(ToBf16(host));
      const auto& bf16 = host_buffers.back();
      auto buffer = DeviceBuffer::Allocate(bf16.size() * 2, device);
      EXPECT_TRUE(buffer.ok()) << buffer.status();
      if (!buffer.ok()) return {};
      keep.push_back(std::move(*buffer));
      EXPECT_TRUE(device_runtime->Memory()
                      .CopyAsync(keep.back().data(), bf16.data(), bf16.size() * 2,
                                 CopyKind::kHostToDevice, DeviceStream{})
                      .ok());
      const int64_t row = static_cast<int64_t>(heads) * kDim;
      return TensorView{keep.back().data(), DataType::kBFloat16, 3,
                        {chunk, heads, kDim}, {row, static_cast<int64_t>(kDim), 1}};
    };

    const TensorView qv = upload_flat(q, kHeads);
    const TensorView kv = upload_flat(k, kKvHeads);
    const TensorView vv = upload_flat(v, kKvHeads);
    ASSERT_FALSE(HasFailure());
    // Derive storage from the full shape: allocating one row here corrupts
    // device memory during prefill and can fail a later decode launch.
    auto output = Tensor::Empty(DataType::kBFloat16, Shape({chunk, kHeads, kDim}), device);
    ASSERT_TRUE(output.ok()) << output.status();
    TensorView out_view{output->Data(), DataType::kBFloat16, 3,
                        {chunk, kHeads, kDim},
                        {int64_t(kHeads) * kDim, int64_t(kDim), 1}};

    ops::OpContext ctx(*device_runtime, DeviceStream{});
    auto step = runtime.BeginStep(desc, ctx);
    ASSERT_TRUE(step.ok()) << step.status();

    Inputs in;
    in.q = qv;
    in.k = kv;
    in.v = vv;
    const Status forward_status = op->Forward(*step, in, out_view);
    ASSERT_TRUE(forward_status.ok()) << forward_status;
    ASSERT_TRUE(device_runtime->Streams().SynchronizeStream(DeviceStream{}).ok());

    // D2H the output and compare against the oracle.
    std::vector<uint16_t> out_bf16(output->Numel());
    ASSERT_TRUE(device_runtime->Memory().CopyAsync(out_bf16.data(), output->Data(),
                                                   out_bf16.size() * 2, CopyKind::kDeviceToHost,
                                                   DeviceStream{}).ok());
    ASSERT_TRUE(device_runtime->Streams().SynchronizeStream(DeviceStream{}).ok());
    for (int j = 0; j < chunk; ++j) {
      const int pos = computed + j;
      for (int h = 0; h < kHeads; ++h) {
        std::vector<float> query(static_cast<size_t>(kDim));
        for (int d = 0; d < kDim; ++d) {
          query[static_cast<size_t>(d)] =
              FromBf16(ToBf16({Activation(10 + pos, h, d)})[0]);
        }
        const std::vector<float> expected =
            OracleAttention(query, keys, values, h / (kHeads / kKvHeads), scale, pos);
        for (int d = 0; d < kDim; ++d) {
          const float actual =
              FromBf16(out_bf16[static_cast<size_t>((j * kHeads + h) * kDim + d)]);
          EXPECT_NEAR(actual, expected[static_cast<size_t>(d)], kTolerance)
              << "pos " << pos << " head " << h << " dim " << d;
        }
      }
    }
  };

  const int prefill = GetParam();
  ASSERT_NO_FATAL_FAILURE(run_step(0, prefill));
  ASSERT_NO_FATAL_FAILURE(run_step(prefill, 1));
  ASSERT_NO_FATAL_FAILURE(run_step(prefill + 1, 1));
  ASSERT_NO_FATAL_FAILURE(run_step(prefill + 2, 3));  // Back to chunked prefill.
  ASSERT_NO_FATAL_FAILURE(run_step(prefill + 5, 1));
}

// Below/at the vectorized KV-write threshold, and a longer multipage prefill.
INSTANTIATE_TEST_SUITE_P(PrefillSizes, FlashDenseGpuTest, testing::Values(5, 20, 32, 48));

TEST(FlashDenseGpuTest, QwenGeometryLongDecodeMatchesOracle) {
  auto device_result = RuntimeFor(Device::Cuda(0));
  if (!device_result.ok()) GTEST_SKIP() << "no CUDA runtime";
  // Same machinery as PrefillAndDecodeMatchTheOracle but qwen3-0.6B's
  // geometry: 16 q heads over 8 kv heads, head_dim 128, 5-token prompt,
  // and 24 decode steps crossing one block boundary.
  constexpr int kQ = 16, kKV = 8, kD = 128;
  AttentionSpec spec;
  spec.params = DenseParams{.num_q_heads = kQ, .num_kv_heads = kKV, .head_dim = kD};
  spec.q_dtype = DataType::kBFloat16;
  spec.kv_dtype = DataType::kBFloat16;

  auto registry = std::make_unique<BackendRegistry>();
  RegisterBuiltinBackends(*registry);
  AttentionConfig config;
  config.block_size = 16;
  config.max_model_len = 64;
  config.max_seqs = 1;
  config.max_batch_tokens = 256;
  config.cache_bytes_budget = 32 << 20;
  config.device = (*device_result)->device();
  config.enable_prefix_caching = false;
  auto made = AttentionRuntime::Create(std::move(registry), config,
                                       std::span<const AttentionSpec>(&spec, 1));
  ASSERT_TRUE(made.ok()) << made.status();
  auto op = (*made)->MakeOp(0);
  ASSERT_TRUE(op.ok()) << op.status();
  AttentionRuntime& runtime = **made;

  const float scale = 1.0f / std::sqrt(static_cast<float>(kD));
  std::vector<std::vector<float>> keys, values;
  int step = 0;
  auto run_one = [&](int computed, int chunk) {
    const std::int32_t seq_len = computed + chunk;
    ASSERT_TRUE(runtime.Cache().Reserve(1, GrowthTarget{computed, seq_len}).ok());
    BatchDesc desc;
    desc.num_reqs = 1;
    desc.num_tokens = chunk;
    const std::vector<std::int32_t> query_lens{chunk};
    const std::vector<std::int32_t> seq_lens{seq_len};
    desc.query_lens = query_lens;
    desc.seq_lens = seq_lens;
    desc.max_query_len = chunk;
    desc.max_seq_len = seq_len;
    desc.block_size = 16;
    desc.num_prefills = chunk > 1 ? 1 : 0;
    desc.num_prefill_tokens = chunk;
    const auto blocks = runtime.Cache().GroupBlocks(1, 0);
    desc.block_ids[0] = blocks;
    desc.block_ids_num_pages[0] = static_cast<std::int32_t>(blocks.size());

    std::vector<float> q(static_cast<size_t>(chunk * kQ * kD));
    std::vector<float> k(static_cast<size_t>(chunk * kKV * kD));
    std::vector<float> v(static_cast<size_t>(chunk * kKV * kD));
    for (int j = 0; j < chunk; ++j) {
      const int token = 300 + computed + j;
      for (int h = 0; h < kQ; ++h) {
        for (int d = 0; d < kD; ++d) {
          q[static_cast<size_t>((j * kQ + h) * kD + d)] = Activation(token, h, d);
        }
      }
      keys.emplace_back(static_cast<size_t>(kKV * kD));
      values.emplace_back(static_cast<size_t>(kKV * kD));
      for (int h = 0; h < kKV; ++h) {
        for (int d = 0; d < kD; ++d) {
          keys.back()[static_cast<size_t>(h * kD + d)] = Activation(token, h + 100, d);
          values.back()[static_cast<size_t>(h * kD + d)] = Activation(token, h + 200, d);
          k[static_cast<size_t>((j * kKV + h) * kD + d)] = keys.back()[static_cast<size_t>(h * kD + d)];
          v[static_cast<size_t>((j * kKV + h) * kD + d)] = values.back()[static_cast<size_t>(h * kD + d)];
        }
      }
    }

    ops::OpContext ctx(**device_result, DeviceStream{});
    auto step_ctx = runtime.BeginStep(desc, ctx);
    ASSERT_TRUE(step_ctx.ok()) << step_ctx.status();

    std::vector<DeviceBuffer> keep;
    auto upload = [&](const std::vector<float>& host, int heads) {
      const std::vector<uint16_t> bf16 = ToBf16(host);
      keep.push_back(std::move(*DeviceBuffer::Allocate(bf16.size() * 2, config.device)));
      EXPECT_TRUE((*device_result)
                      ->Memory()
                      .CopyAsync(keep.back().data(), bf16.data(), bf16.size() * 2,
                                 CopyKind::kHostToDevice, DeviceStream{})
                      .ok());
      return TensorView{keep.back().data(), DataType::kBFloat16, 3, {chunk, heads, kD},
                        {int64_t(heads) * kD, int64_t(kD), 1}};
    };
    Inputs in;
    in.q = upload(q, kQ);
    in.k = upload(k, kKV);
    in.v = upload(v, kKV);
    keep.push_back(std::move(*DeviceBuffer::Allocate(size_t(chunk) * kQ * kD * 2, config.device)));
    TensorView out_view{keep.back().data(), DataType::kBFloat16, 3, {chunk, kQ, kD},
                        {int64_t(kQ) * kD, int64_t(kD), 1}};
    const Status forward = op->Forward(*step_ctx, in, out_view);
    ASSERT_TRUE(forward.ok()) << forward << " (step " << step << ", computed " << computed
                              << ", chunk " << chunk << ")";
    ++step;
    ASSERT_TRUE((*device_result)->Streams().SynchronizeStream(DeviceStream{}).ok());
    std::vector<uint16_t> out_bf16(size_t(chunk) * kQ * kD);
    EXPECT_TRUE((*device_result)
                    ->Memory()
                    .Copy(out_bf16.data(), keep.back().data(), out_bf16.size() * 2,
                          CopyKind::kDeviceToHost)
                    .ok());
    ASSERT_TRUE((*device_result)->Streams().SynchronizeStream(DeviceStream{}).ok());

    for (int j = 0; j < chunk; ++j) {
      const int pos = computed + j;
      for (int h = 0; h < kQ; ++h) {
        std::vector<float> query(static_cast<size_t>(kD));
        for (int d = 0; d < kD; ++d) {
          query[static_cast<size_t>(d)] = FromBf16(ToBf16({Activation(300 + pos, h, d)})[0]);
        }
        // Softmax over bf16-rounded scores to mirror the kernel's precision.
        std::vector<float> scores;
        float max_score = -1e30f;
        for (int j2 = 0; j2 <= pos; ++j2) {
          float dot = 0.0f;
          for (int d = 0; d < kD; ++d) {
            dot += query[static_cast<size_t>(d)] *
                   FromBf16(ToBf16({keys[static_cast<size_t>(j2)][static_cast<size_t>((h / (kQ / kKV)) * kD + d)]})[0]);
          }
          const float s = FromBf16(ToBf16({dot * scale})[0]);
          scores.push_back(s);
          max_score = std::max(max_score, s);
        }
        float denom = 0.0f;
        for (float& s : scores) {
          s = std::exp(s - max_score);
          denom += s;
        }
        for (int d = 0; d < kD; ++d) {
          const float actual = FromBf16(out_bf16[static_cast<size_t>((j * kQ + h) * kD + d)]);
          const float expected_value = [&] {
            float acc = 0.0f;
            for (int j2 = 0; j2 <= pos; ++j2) {
              acc += scores[static_cast<size_t>(j2)] / denom *
                     FromBf16(ToBf16({values[static_cast<size_t>(j2)][static_cast<size_t>((h / (kQ / kKV)) * kD + d)]})[0]);
            }
            return acc;
          }();
          EXPECT_NEAR(actual, expected_value, 8e-2f)
              << "pos " << pos << " head " << h << " dim " << d;
        }
      }
    }
  };

  run_one(0, 5);
  for (int decode = 0; decode < 24; ++decode) run_one(5 + decode, 1);
}


}  // namespace
}  // namespace inferx::attn
