// Numerical gate for the model layer: a tiny synthetic Qwen3 checkpoint is
// written to disk (BF16 safetensors), loaded through the registry and weight
// loader, executed on CUDA, and compared against an FP32 CPU reference
// computed in this file. A second pass runs the same checkpoint sharded
// across two LoopbackComm ranks and checks the logits agree with the
// single-rank execution -- that is the tensor-parallel sharding proof.
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/device_runtime.h"
#include "inferx/dist/loopback_comm.h"
#include "inferx/models/model.h"
#include "inferx/ops/attention/registry.h"
#include "inferx/ops/attention/runtime.h"
#include "inferx/models/load_options.h"
#include "inferx/models/registry.h"
#include "inferx/models/scratch_arena.h"
#include "inferx/util/hf_config.h"

namespace inferx {
namespace models {
namespace {

// ---- Tiny architecture ----------------------------------------------------

constexpr int64_t kVocab = 96, kHidden = 256, kLayers = 2, kHeads = 4,
                  kKvHeads = 2, kHeadDim = 64, kInter = 512, kSeq = 20;
constexpr int64_t kBlockSize = 16;
constexpr float kEps = 1e-6f;

struct HostWeight {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

using HostWeights = std::map<std::string, HostWeight>;

// ---- Synthetic checkpoint -------------------------------------------------

HostWeights MakeWeights(unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-1.f, 1.f);
  auto fill = [&](HostWeight& w, int64_t fan_in) {
    const float scale = 1.f / std::sqrt(static_cast<float>(fan_in));
    for (float& x : w.data) x = uniform(rng) * scale;
  };

  HostWeights weights;
  auto add = [&](const std::string& name, std::vector<int64_t> shape) -> HostWeight& {
    int64_t numel = 1;
    for (int64_t d : shape) numel *= d;
    return weights[name] = HostWeight{std::move(shape), std::vector<float>(numel)};
  };

  fill(add("model.embed_tokens.weight", {kVocab, kHidden}), kHidden);
  for (int64_t layer = 0; layer < kLayers; ++layer) {
    const std::string p = "model.layers." + std::to_string(layer);
    fill(add(p + ".self_attn.q_proj.weight", {kHeads * kHeadDim, kHidden}), kHidden);
    fill(add(p + ".self_attn.k_proj.weight", {kKvHeads * kHeadDim, kHidden}), kHidden);
    fill(add(p + ".self_attn.v_proj.weight", {kKvHeads * kHeadDim, kHidden}), kHidden);
    fill(add(p + ".self_attn.o_proj.weight", {kHidden, kHeads * kHeadDim}),
         kHeads * kHeadDim);
    fill(add(p + ".self_attn.q_norm.weight", {kHeadDim}), kHeadDim);
    fill(add(p + ".self_attn.k_norm.weight", {kHeadDim}), kHeadDim);
    fill(add(p + ".mlp.gate_proj.weight", {kInter, kHidden}), kHidden);
    fill(add(p + ".mlp.up_proj.weight", {kInter, kHidden}), kHidden);
    fill(add(p + ".mlp.down_proj.weight", {kHidden, kInter}), kInter);
    fill(add(p + ".input_layernorm.weight", {kHidden}), kHidden);
    fill(add(p + ".post_attention_layernorm.weight", {kHidden}), kHidden);
  }
  fill(add("model.norm.weight", {kHidden}), kHidden);
  return weights;
}

void WriteCheckpoint(const std::string& dir, const HostWeights& weights) {
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string config = R"({
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
    "vocab_size": )" + std::to_string(kVocab) + R"(,
    "hidden_size": )" + std::to_string(kHidden) + R"(,
    "num_hidden_layers": )" + std::to_string(kLayers) + R"(,
    "num_attention_heads": )" + std::to_string(kHeads) + R"(,
    "num_key_value_heads": )" + std::to_string(kKvHeads) + R"(,
    "head_dim": )" + std::to_string(kHeadDim) + R"(,
    "intermediate_size": )" + std::to_string(kInter) + R"(,
    "max_position_embeddings": 128, "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0, "tie_word_embeddings": true,
    "eos_token_id": 95, "torch_dtype": "bfloat16"
  })";
  { std::ofstream(dir + "/config.json") << config; }

  // Every tensor stored BF16; the loader byte-copies into the model.
  std::string body, entries;
  int64_t offset = 0;
  for (const auto& [name, weight] : weights) {
    std::vector<uint16_t> bits(weight.data.size());
    for (size_t i = 0; i < weight.data.size(); ++i) {
      const uint32_t raw = std::bit_cast<uint32_t>(weight.data[i]);
      bits[i] = static_cast<uint16_t>((raw + 0x7fff + ((raw >> 16) & 1)) >> 16);
    }
    const int64_t bytes = static_cast<int64_t>(bits.size()) * 2;
    if (!entries.empty()) entries += ",";
    entries += "\"" + name + "\": {\"dtype\": \"BF16\", \"shape\": [";
    for (size_t d = 0; d < weight.shape.size(); ++d) {
      entries += (d ? "," : "") + std::to_string(weight.shape[d]);
    }
    entries += "], \"data_offsets\": [" + std::to_string(offset) + ", " +
               std::to_string(offset + bytes) + "]}";
    body.append(reinterpret_cast<const char*>(bits.data()), bits.size() * 2);
    offset += bytes;
  }
  const std::string header = "{" + entries + "}";
  const uint64_t header_size = header.size();
  std::string framed(8, '\0');
  std::memcpy(framed.data(), &header_size, 8);
  framed += header + body;
  std::ofstream(dir + "/model.safetensors", std::ios::binary) << framed;
}

// ---- FP32 CPU reference ---------------------------------------------------

class Reference {
 public:
  Reference(const HostWeights& w, std::vector<int32_t> ids)
      : w_(w), ids_(std::move(ids)) {}

  std::vector<float> LogitsAtLastToken() {
    const int64_t T = static_cast<int64_t>(ids_.size());
    Matrix x(T, Row(kHidden));
    for (int64_t t = 0; t < T; ++t) {
      const HostWeight& e = w_.at("model.embed_tokens.weight");
      for (int64_t c = 0; c < kHidden; ++c) x[t][c] = e.data[ids_[t] * kHidden + c];
    }

    for (int64_t layer = 0; layer < kLayers; ++layer) {
      const std::string p = "model.layers." + std::to_string(layer);
      Matrix attn = AttentionBlock(p, x);
      for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < kHidden; ++c) x[t][c] += attn[t][c];
      }
      Matrix mlp = MlpBlock(p + ".mlp", w_.at(p + ".post_attention_layernorm.weight"), x);
      for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < kHidden; ++c) x[t][c] += mlp[t][c];
      }
    }

    std::vector<float> final_row = RmsRow(x[T - 1], w_.at("model.norm.weight"));
    const HostWeight& head = w_.at("model.embed_tokens.weight");  // Tied.
    std::vector<float> logits(kVocab, 0.f);
    for (int64_t v = 0; v < kVocab; ++v) {
      float sum = 0.f;
      for (int64_t c = 0; c < kHidden; ++c) sum += head.data[v * kHidden + c] * final_row[c];
      logits[v] = sum;
    }
    return logits;
  }

 private:
  using Row = std::vector<float>;
  using Matrix = std::vector<Row>;

  Row RmsRow(const Row& in, const HostWeight& gamma) {
    float sq = 0.f;
    for (float v : in) sq += v * v;
    const float inv = 1.f / std::sqrt(sq / static_cast<float>(in.size()) + kEps);
    Row out(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = in[i] * inv * gamma.data[i];
    return out;
  }

  // out[t] = norm(x[t]) @ W^T for W stored [rows, cols].
  Matrix Project(const Matrix& x, const HostWeight& w) {
    const int64_t rows = w.shape[0], cols = w.shape[1];
    Matrix out(x.size(), Row(rows, 0.f));
    for (size_t t = 0; t < x.size(); ++t) {
      for (int64_t r = 0; r < rows; ++r) {
        float sum = 0.f;
        for (int64_t c = 0; c < cols; ++c) sum += w.data[r * cols + c] * x[t][c];
        out[t][r] = sum;
      }
    }
    return out;
  }

  void HeadNorm(Row& per_token, int64_t head, const HostWeight& gamma) {
    const int64_t base = head * kHeadDim;
    float sq = 0.f;
    for (int64_t i = 0; i < kHeadDim; ++i) sq += per_token[base + i] * per_token[base + i];
    const float inv = 1.f / std::sqrt(sq / static_cast<float>(kHeadDim) + kEps);
    for (int64_t i = 0; i < kHeadDim; ++i) per_token[base + i] *= inv * gamma.data[i];
  }

  void Rope(Row& per_token, int64_t head, int64_t pos) {
    const int64_t base = head * kHeadDim;
    for (int64_t i = 0; i < kHeadDim / 2; ++i) {
      const float freq = std::pow(10000.f, -2.f * static_cast<float>(i) / kHeadDim);
      const float angle = static_cast<float>(pos) * freq;
      const float c = std::cos(angle), s = std::sin(angle);
      const float a = per_token[base + i], b = per_token[base + i + kHeadDim / 2];
      per_token[base + i] = a * c - b * s;
      per_token[base + i + kHeadDim / 2] = a * s + b * c;
    }
  }

  Matrix AttentionBlock(const std::string& p, const Matrix& x) {
    const int64_t T = static_cast<int64_t>(x.size());
    const float scale = 1.f / std::sqrt(static_cast<float>(kHeadDim));
    const int64_t group = kHeads / kKvHeads;

    Matrix normed(T);
    for (int64_t t = 0; t < T; ++t) {
      normed[t] = RmsRow(x[t], w_.at(p + ".input_layernorm.weight"));
    }
    Matrix q = Project(normed, w_.at(p + ".self_attn.q_proj.weight"));
    Matrix k = Project(normed, w_.at(p + ".self_attn.k_proj.weight"));
    Matrix v = Project(normed, w_.at(p + ".self_attn.v_proj.weight"));
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t h = 0; h < kHeads; ++h) HeadNorm(q[t], h, w_.at(p + ".self_attn.q_norm.weight"));
      for (int64_t h = 0; h < kKvHeads; ++h) HeadNorm(k[t], h, w_.at(p + ".self_attn.k_norm.weight"));
      for (int64_t h = 0; h < kHeads; ++h) Rope(q[t], h, t);
      for (int64_t h = 0; h < kKvHeads; ++h) Rope(k[t], h, t);
    }

    Matrix pre_out(T, Row(kHeads * kHeadDim, 0.f));
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t h = 0; h < kHeads; ++h) {
        const int64_t kv = h / group;
        std::vector<float> scores(t + 1);
        float max_score = -1e30f;
        for (int64_t s = 0; s <= t; ++s) {
          float dot = 0.f;
          for (int64_t i = 0; i < kHeadDim; ++i) {
            dot += q[t][h * kHeadDim + i] * k[s][kv * kHeadDim + i];
          }
          scores[s] = dot * scale;
          max_score = std::max(max_score, scores[s]);
        }
        float total = 0.f;
        for (float& sc : scores) {
          sc = std::exp(sc - max_score);
          total += sc;
        }
        Row acc(kHeadDim, 0.f);
        for (int64_t s = 0; s <= t; ++s) {
          const float weight = scores[s] / total;
          for (int64_t i = 0; i < kHeadDim; ++i) {
            acc[i] += weight * v[s][kv * kHeadDim + i];
          }
        }
        for (int64_t i = 0; i < kHeadDim; ++i) pre_out[t][h * kHeadDim + i] = acc[i];
      }
    }
    return Project(pre_out, w_.at(p + ".self_attn.o_proj.weight"));
  }

  Matrix MlpBlock(const std::string& p, const HostWeight& post_norm, const Matrix& x) {
    const int64_t T = static_cast<int64_t>(x.size());
    Matrix normed(T);
    for (int64_t t = 0; t < T; ++t) normed[t] = RmsRow(x[t], post_norm);

    Matrix gate = Project(normed, w_.at(p + ".gate_proj.weight"));
    Matrix up = Project(normed, w_.at(p + ".up_proj.weight"));
    Matrix act(T, Row(kInter, 0.f));
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t i = 0; i < kInter; ++i) {
        const float g = gate[t][i];
        act[t][i] = g / (1.f + std::exp(-g)) * up[t][i];
      }
    }
    return Project(act, w_.at(p + ".down_proj.weight"));
  }

  const HostWeights& w_;
  std::vector<int32_t> ids_;
};

// ---- Device execution -----------------------------------------------------

struct DeviceStep {
  DeviceRuntime* runtime = nullptr;
  OwnedStream stream = OwnedStream(DeviceStream{}, nullptr);
  std::unique_ptr<ops::OpContext> op;
  std::unique_ptr<ScratchArena> scratch;
  std::unique_ptr<attn::AttentionRuntime> attention;
  std::vector<attn::AttentionOp> ops;

  DeviceStream raw_stream() const { return stream.Get(); }
};

StatusOr<Tensor> Upload(DeviceRuntime& runtime, DeviceStream stream,
                        const std::vector<int32_t>& values) {
  Tensor tensor = *Tensor::Empty(
      DataType::kInt32, Shape({static_cast<int64_t>(values.size())}), runtime.device());
  if (!values.empty()) {
    EXPECT_TRUE(runtime.Memory().CopyAsync(tensor.Data(), values.data(),
                                  values.size() * sizeof(int32_t),
                                  CopyKind::kHostToDevice, stream)
                    .ok());
  }
  return tensor;
}

std::vector<float> RunSingleRank(const std::string& dir, const std::vector<int32_t>& ids,
                                 DeviceStep& step) {
  EXPECT_TRUE(RuntimeFor(Device::Cuda(0)).ok());
  step.runtime = *RuntimeFor(Device::Cuda(0));
  EXPECT_TRUE(step.runtime->Activate().ok());
  step.stream = *step.runtime->Streams().CreateStream(StreamPriority::kNormal);
  ModelLoadOptions options;
  options.directory = dir;
  options.device = Device::Cuda(0);
  std::unique_ptr<Model> model = *LoadModel(options);
  step.op = std::make_unique<ops::OpContext>(*step.runtime, step.raw_stream());

  const ModelSpec& spec = model->GetModelSpec();
  const int64_t T = static_cast<int64_t>(ids.size());

  step.scratch = std::make_unique<ScratchArena>(
      ScratchArena::Measuring(step.runtime->device()));

  // The attention runtime owns pool + metadata + kernels; the facade
  // registers this single request and the builder shapes its step.
  auto registry = std::make_unique<attn::BackendRegistry>();
  attn::RegisterBuiltinBackends(*registry);
  attn::AttentionConfig attention_config;
  attention_config.block_size = kBlockSize;
  attention_config.max_model_len = std::max<int64_t>(spec.max_position_embeddings, T);
  attention_config.max_seqs = 1;
  attention_config.max_batch_tokens = T;
  attention_config.cache_bytes_budget = 1LL << 32;
  attention_config.device = step.runtime->device();
  attention_config.enable_prefix_caching = false;
  attention_config.forced_backend = "flash_dense";
  step.attention = std::move(
      *attn::AttentionRuntime::Create(std::move(registry), attention_config,
                                      model->GetAttentionSpecs()));
  step.ops.clear();
  for (size_t l = 0; l < model->GetAttentionSpecs().size(); ++l) {
    step.ops.push_back(std::move(*step.attention->MakeOp(static_cast<int64_t>(l))));
  }
  EXPECT_TRUE(step.attention->Cache().Reserve(1, GrowthTarget{0, T}).ok());

  std::vector<int32_t> positions(T);
  for (int64_t i = 0; i < T; ++i) positions[i] = static_cast<int32_t>(i);
  auto uploaded = [&](const std::vector<int32_t>& values) -> Tensor {
    auto tensor = Upload(*step.runtime, step.raw_stream(), values);
    if (!tensor.ok()) {
      ADD_FAILURE() << tensor.status();
      std::abort();
    }
    return *std::move(tensor);
  };
  StepInputs inputs;
  inputs.token_ids = uploaded(ids);
  inputs.positions = uploaded(positions);

  const std::vector<std::int32_t> query_lens{static_cast<std::int32_t>(T)};
  const std::vector<std::int32_t> seq_lens{static_cast<std::int32_t>(T)};
  attn::BatchDesc desc;
  desc.num_reqs = 1;
  desc.num_tokens = static_cast<std::int32_t>(T);
  desc.query_lens = query_lens;
  desc.seq_lens = seq_lens;
  desc.max_query_len = static_cast<std::int32_t>(T);
  desc.max_seq_len = static_cast<std::int32_t>(T);
  desc.block_size = static_cast<std::int32_t>(kBlockSize);
  desc.num_prefills = 1;
  desc.num_prefill_tokens = static_cast<std::int32_t>(T);
  const auto blocks = step.attention->Cache().GroupBlocks(1, 0);
  desc.block_ids[0] = blocks;
  desc.block_ids_num_pages[0] = static_cast<std::int32_t>(blocks.size());
  attn::StepContext step_ctx = std::move(*step.attention->BeginStep(desc, *step.op));

  SingleRankComm comm;
  ForwardContext ctx(*step.op, step_ctx, comm, *step.scratch,
                     absl::MakeConstSpan(step.ops));
  Tensor hidden = *Tensor::Empty(spec.dtype, Shape({T, spec.hidden_size}), step.scratch.get());
  const Tensor sample_rows = uploaded({static_cast<int32_t>(T) - 1});
  { const auto fwd = model->Forward(ctx, inputs, hidden);
    EXPECT_TRUE(fwd.ok()) << fwd; }
  Tensor logits = *Tensor::Empty(spec.dtype, Shape({1, spec.vocab_size}), step.scratch.get());
  { const auto got = model->ComputeLogits(ctx, hidden, sample_rows, logits);
    EXPECT_TRUE(got.ok()) << got; }
  EXPECT_TRUE(step.runtime->Streams().SynchronizeStream(step.raw_stream()).ok());

  std::vector<uint16_t> bits(spec.vocab_size);
  EXPECT_TRUE(step.runtime
                  ->Memory()
                  .Copy(bits.data(), logits.Data(), bits.size() * 2,
                        CopyKind::kDeviceToHost)
                  .ok());
  std::vector<float> out(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) {
    out[i] = std::bit_cast<float>(static_cast<uint32_t>(bits[i]) << 16);
  }
  return out;
}

class Qwen3ForwardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = ::testing::TempDir() + "/inferx_qwen3_forward";
    weights_ = MakeWeights(/*seed=*/20261009);
    WriteCheckpoint(dir_, weights_);
    std::mt19937 rng(7);
    for (int i = 0; i < kSeq; ++i) ids_.push_back(static_cast<int32_t>(rng() % kVocab));
  }

  std::string dir_;
  HostWeights weights_;
  std::vector<int32_t> ids_;
};

TEST_F(Qwen3ForwardTest, CudaLogitsMatchFp32Reference) {
  DeviceStep step;
  std::vector<float> logits = RunSingleRank(dir_, ids_, step);
  ASSERT_EQ(logits.size(), static_cast<size_t>(kVocab));

  Reference reference(weights_, ids_);
  std::vector<float> expected = reference.LogitsAtLastToken();

  float max_abs = 0.f;
  size_t arg_cuda = 0, arg_ref = 0;
  for (size_t i = 0; i < logits.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(logits[i] - expected[i]));
    if (logits[i] > logits[arg_cuda]) arg_cuda = i;
    if (expected[i] > expected[arg_ref]) arg_ref = i;
  }
  // BF16 execution against an FP32 reference: same argmax and bounded drift.
  EXPECT_EQ(arg_cuda, arg_ref);
  EXPECT_LT(max_abs, 0.25f) << "max |cuda - fp32 reference| = " << max_abs;
}

/// The golden-trace shape: short prefill, then many one-token decode
/// steps through the full model stack, each compared against the FP32
/// reference recomputed over the growing sequence.
TEST_F(Qwen3ForwardTest, DecodeStepsMatchFp32Reference) {
  DeviceStep step;
  std::vector<int32_t> ids(ids_.begin(), ids_.begin() + 5);
  std::mt19937 rng(99);
  std::vector<float> logits = RunSingleRank(dir_, ids, step);
  ASSERT_EQ(logits.size(), static_cast<size_t>(kVocab));

  for (int decode = 0; decode < 12; ++decode) {
    // Greedy continuation: take the engine's argmax as the next token.
    size_t argmax = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
      if (logits[i] > logits[argmax]) argmax = i;
    }
    ids.push_back(static_cast<int32_t>(argmax));

    // One decode step through the same runtime.
    DeviceStep step2;
    logits = RunSingleRank(dir_, ids, step2);
    ASSERT_EQ(logits.size(), static_cast<size_t>(kVocab));

    Reference reference(weights_, ids);
    std::vector<float> expected = reference.LogitsAtLastToken();
    float max_abs = 0.f;
    for (size_t i = 0; i < logits.size(); ++i) {
      max_abs = std::max(max_abs, std::abs(logits[i] - expected[i]));
    }
    // Same tolerance band as the prefill gate.
    EXPECT_LT(max_abs, 0.5f) << "decode step " << decode << " (seq len " << ids.size()
                             << ") drifted to " << max_abs;
  }
}


namespace {

/// One rank's sharded execution: own model, weights, stream, KV shard, and
/// scratch; collectives go through the shared LoopbackWorld.
Status RunShardedRank(const std::string& dir, const std::vector<int32_t>& ids,
                      CommBackend& comm, int rank, std::vector<float>* out) {
  const Device device = Device::Cuda(0);
  INFERX_ASSIGN_OR_RETURN(DeviceRuntime * runtime, RuntimeFor(device));
  INFERX_RETURN_IF_ERROR(runtime->Activate());
  const OwnedStream owned_stream = *runtime->Streams().CreateStream(StreamPriority::kNormal);
  const DeviceStream stream = owned_stream.Get();

  ParallelConfig parallel;
  parallel.tensor_parallel_size = 2;
  parallel.tensor_parallel_rank = rank;
  ModelLoadOptions options;
  options.directory = dir;
  options.device = device;
  options.parallel = parallel;
  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<Model> model, LoadModel(options));

  const ModelSpec& spec = model->GetModelSpec();
  const int64_t T = static_cast<int64_t>(ids.size());

  ScratchArena scratch = ScratchArena::Measuring(device);

  const auto upload_filled =
      [&](const std::vector<int32_t>& values) -> StatusOr<Tensor> {
    INFERX_ASSIGN_OR_RETURN(Tensor tensor,
                            Tensor::Empty(DataType::kInt32,
                                          Shape({static_cast<int64_t>(values.size())}),
                                          device));
    if (!values.empty()) {
      INFERX_RETURN_IF_ERROR(runtime->Memory().CopyAsync(
          tensor.Data(), values.data(), values.size() * sizeof(int32_t),
          CopyKind::kHostToDevice, stream));
    }
    return tensor;
  };

  INFERX_ASSIGN_OR_RETURN(Tensor ids_tensor, upload_filled(ids));
  std::vector<int32_t> positions(T);
  for (int64_t i = 0; i < T; ++i) positions[i] = static_cast<int32_t>(i);
  INFERX_ASSIGN_OR_RETURN(Tensor positions_tensor, upload_filled(positions));

  ops::OpContext op(*runtime, stream);
  auto registry = std::make_unique<attn::BackendRegistry>();
  attn::RegisterBuiltinBackends(*registry);
  attn::AttentionConfig attention_config;
  attention_config.block_size = kBlockSize;
  attention_config.max_model_len = std::max<int64_t>(spec.max_position_embeddings, T);
  attention_config.max_seqs = 1;
  attention_config.max_batch_tokens = T;
  attention_config.cache_bytes_budget = 1LL << 32;
  attention_config.device = device;
  attention_config.enable_prefix_caching = false;
  attention_config.forced_backend = "flash_dense";
  INFERX_ASSIGN_OR_RETURN(
      std::unique_ptr<attn::AttentionRuntime> attention,
      attn::AttentionRuntime::Create(std::move(registry), attention_config,
                                     model->GetAttentionSpecs()));
  std::vector<attn::AttentionOp> ops;
  for (size_t l = 0; l < model->GetAttentionSpecs().size(); ++l) {
    INFERX_ASSIGN_OR_RETURN(attn::AttentionOp layer_op,
                            attention->MakeOp(static_cast<int64_t>(l)));
    ops.push_back(std::move(layer_op));
  }
  INFERX_RETURN_IF_ERROR(attention->Cache().Reserve(1, GrowthTarget{0, T}));

  const std::vector<std::int32_t> query_lens{static_cast<std::int32_t>(T)};
  const std::vector<std::int32_t> seq_lens{static_cast<std::int32_t>(T)};
  attn::BatchDesc desc;
  desc.num_reqs = 1;
  desc.num_tokens = static_cast<std::int32_t>(T);
  desc.query_lens = query_lens;
  desc.seq_lens = seq_lens;
  desc.max_query_len = static_cast<std::int32_t>(T);
  desc.max_seq_len = static_cast<std::int32_t>(T);
  desc.block_size = static_cast<std::int32_t>(kBlockSize);
  desc.num_prefills = 1;
  desc.num_prefill_tokens = static_cast<std::int32_t>(T);
  const auto blocks = attention->Cache().GroupBlocks(1, 0);
  desc.block_ids[0] = blocks;
  desc.block_ids_num_pages[0] = static_cast<std::int32_t>(blocks.size());
  INFERX_ASSIGN_OR_RETURN(attn::StepContext step_ctx, attention->BeginStep(desc, op));

  ForwardContext ctx(op, step_ctx, comm, scratch, absl::MakeConstSpan(ops));
  StepInputs inputs;
  inputs.token_ids = ids_tensor;
  inputs.positions = positions_tensor;
  INFERX_ASSIGN_OR_RETURN(
      Tensor hidden, Tensor::Empty(spec.dtype, Shape({T, spec.hidden_size}), &scratch));
  INFERX_RETURN_IF_ERROR(model->Forward(ctx, inputs, hidden));
  INFERX_ASSIGN_OR_RETURN(
      Tensor logits, Tensor::Empty(spec.dtype, Shape({1, spec.vocab_size}), &scratch));
  const std::vector<int32_t> last_row{static_cast<int32_t>(T) - 1};
  INFERX_ASSIGN_OR_RETURN(Tensor sample_rows, upload_filled(last_row));
  INFERX_RETURN_IF_ERROR(model->ComputeLogits(ctx, hidden, sample_rows, logits));
  INFERX_RETURN_IF_ERROR(runtime->Streams().SynchronizeStream(stream));

  std::vector<uint16_t> bits(spec.vocab_size);
  INFERX_RETURN_IF_ERROR(
      runtime->Memory().Copy(bits.data(), logits.Data(), bits.size() * 2,
                             CopyKind::kDeviceToHost));
  out->resize(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) {
    (*out)[i] = std::bit_cast<float>(static_cast<uint32_t>(bits[i]) << 16);
  }
  return OkStatus();
}

}  // namespace

TEST_F(Qwen3ForwardTest, TwoRankLoopbackMatchesSingleRank) {
  DeviceStep step;
  std::vector<float> single = RunSingleRank(dir_, ids_, step);

  auto world = *LoopbackWorld::Create(2);
  std::vector<float> rank0, rank1;
  auto future0 = std::async(std::launch::async, [&] {
    return RunShardedRank(dir_, ids_, world->rank(0), 0, &rank0);
  });
  auto future1 = std::async(std::launch::async, [&] {
    return RunShardedRank(dir_, ids_, world->rank(1), 1, &rank1);
  });
  ASSERT_TRUE(future0.get().ok());
  ASSERT_TRUE(future1.get().ok());

  float max_gap = 0.f;
  for (size_t i = 0; i < single.size(); ++i) {
    max_gap = std::max(max_gap, std::abs(single[i] - rank0[i]));
    max_gap = std::max(max_gap, std::abs(rank0[i] - rank1[i]));
  }
  // Same kernels, same weights; only the collective split differs, so the
  // ranks agree to within a couple of BF16 roundings.
  EXPECT_LT(max_gap, 0.1f) << "max |tp1 - tp2| = " << max_gap;
}

}  // namespace
}  // namespace models
}  // namespace inferx
