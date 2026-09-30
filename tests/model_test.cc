#include "inferx/models/model.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/ops/op_context.h"

namespace inferx {
namespace {

class ModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto runtime = RuntimeFor(DeviceId::Cuda(0));
    ASSERT_TRUE(runtime.ok());
    runtime_ = *runtime;
    ASSERT_TRUE(runtime_->Activate().ok());
    auto stream = runtime_->CreateStream();
    ASSERT_TRUE(stream.ok());
    stream_ = *stream;
  }

  /// \brief A minimal valid ModelInput: one single-token sequence.
  StatusOr<ModelInput> MakeInput(int num_tokens) {
    const DeviceId device = DeviceId::Cuda(0);
    auto i32 = [&](Shape shape) { return Tensor::Empty(DataType::kInt32, shape, device); };
    INFERX_ASSIGN_OR_RETURN(Tensor token_ids, i32(Shape({num_tokens})));
    INFERX_ASSIGN_OR_RETURN(Tensor positions, i32(Shape({num_tokens})));
    INFERX_ASSIGN_OR_RETURN(Tensor batch_indices, i32(Shape({num_tokens})));
    INFERX_ASSIGN_OR_RETURN(Tensor qo_indptr, i32(Shape({2})));
    INFERX_ASSIGN_OR_RETURN(Tensor kv_indptr, i32(Shape({2})));
    INFERX_ASSIGN_OR_RETURN(Tensor kv_indices, i32(Shape({1})));
    INFERX_ASSIGN_OR_RETURN(Tensor last_page_len, i32(Shape({1})));
    INFERX_ASSIGN_OR_RETURN(Tensor logit_rows, i32(Shape({1})));
    std::vector<int32_t> pos(num_tokens);
    for (int i = 0; i < num_tokens; ++i) pos[i] = i;
    const std::vector<int32_t> zeros(num_tokens, 0);
    INFERX_RETURN_IF_ERROR(runtime_->Copy(
        token_ids.Data(), pos.data(), num_tokens * sizeof(int32_t), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(runtime_->Copy(
        positions.Data(), pos.data(), num_tokens * sizeof(int32_t), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(runtime_->Copy(batch_indices.Data(), zeros.data(),
                                          num_tokens * sizeof(int32_t),
                                          CopyKind::kHostToDevice));
    host_qo_ = {0, num_tokens};
    const int32_t qo[] = {0, num_tokens};
    const int32_t kv[] = {0, 1};
    const int32_t block[] = {0};
    const int32_t page[] = {num_tokens};
    const int32_t row[] = {num_tokens - 1};
    INFERX_RETURN_IF_ERROR(
        runtime_->Copy(qo_indptr.Data(), qo, sizeof(qo), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(
        runtime_->Copy(kv_indptr.Data(), kv, sizeof(kv), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(
        runtime_->Copy(kv_indices.Data(), block, sizeof(block), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(
        runtime_->Copy(last_page_len.Data(), page, sizeof(page), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(
        runtime_->Copy(logit_rows.Data(), row, sizeof(row), CopyKind::kHostToDevice));

    return ModelInput{std::move(token_ids),
                      AttentionBatch{std::move(positions),
                                     std::move(batch_indices),
                                     std::move(qo_indptr),
                                     std::move(kv_indptr),
                                     std::move(kv_indices),
                                     std::move(last_page_len),
                                     host_qo_,
                                     {},
                                     num_tokens,
                                     1},
                      std::move(logit_rows)};
  }

  std::vector<int32_t> host_qo_;
  DeviceRuntime* runtime_ = nullptr;
  Stream stream_;
};

TEST_F(ModelTest, LoadsQwen3CheckpointAndConfig) {
  auto model = Model::Load("models/Qwen3-0.6B", DeviceId::Cuda(0), /*max_tokens=*/8,
                           /*max_seqs=*/2);
  ASSERT_TRUE(model.ok()) << model.status();
  const CheckpointConfig& config = (*model)->config();
  EXPECT_EQ(config.model_type, "qwen3");
  EXPECT_EQ(config.architectures, "Qwen3ForCausalLM");
  EXPECT_EQ(config.hidden_size, 1024);
  EXPECT_EQ(config.intermediate_size, 3072);
  EXPECT_EQ(config.num_hidden_layers, 28);
  EXPECT_EQ(config.num_attention_heads, 16);
  EXPECT_EQ(config.num_key_value_heads, 8);
  EXPECT_EQ(config.head_dim, 128);
  EXPECT_EQ(config.vocab_size, 151936);
  EXPECT_FLOAT_EQ(config.rms_norm_eps, 1e-6f);
  EXPECT_FLOAT_EQ(config.rope_theta, 1e6f);
  EXPECT_TRUE(config.tie_word_embeddings);
}

TEST_F(ModelTest, ForwardReturnsLogits) {
  auto model = Model::Load("models/Qwen3-0.6B", DeviceId::Cuda(0), /*max_tokens=*/8,
                           /*max_seqs=*/2);
  ASSERT_TRUE(model.ok()) << model.status();
  const CheckpointConfig& config = (*model)->config();
  KvLayout layout;
  layout.entries_per_token = 2;
  layout.kv_heads = config.num_key_value_heads;
  layout.head_dim = config.head_dim;
  layout.dtype = DataType::kBFloat16;
  auto pool = KvBlockPool::Create(config.num_hidden_layers, /*num_kv_blocks=*/4,
                                  /*block_size=*/2, layout, DeviceId::Cuda(0));
  ASSERT_TRUE(pool.ok());
  auto input = MakeInput(2);
  ASSERT_TRUE(input.ok());
  ops::OpContext ctx(*runtime_, stream_);
  dist::SingleRankComm comm;
  ModelState state;
  state.paged_kv = &*pool;
  for (int64_t i = 0; i < config.num_hidden_layers; ++i) state.layers.push_back(PagedKvState{i});
  const StatusOr<Tensor> logits = (*model)->Forward(*input, state, ctx, comm);
  ASSERT_TRUE(logits.ok()) << logits.status();
  EXPECT_EQ(logits->Rank(), 2);
  EXPECT_EQ(logits->Dim(0), 1);
  EXPECT_EQ(logits->Dim(1), config.vocab_size);
  EXPECT_EQ(logits->GetDataType(), DataType::kBFloat16);
}

namespace {

/// Runs one fixed two-token prompt through a loaded model and returns the
/// last-row logits on the host.
StatusOr<std::vector<float>> ForwardLogits(Model& model, const CheckpointConfig& config,
                                           DeviceRuntime& runtime, Stream stream,
                                           const ModelInput& input, KvBlockPool& pool) {
  ops::OpContext ctx(runtime, stream);
  dist::SingleRankComm comm;
  ModelState state;
  state.paged_kv = &pool;
  for (int64_t i = 0; i < config.num_hidden_layers; ++i) state.layers.push_back(PagedKvState{i});
  INFERX_ASSIGN_OR_RETURN(auto logits, model.Forward(input, state, ctx, comm));
  INFERX_RETURN_IF_ERROR(runtime.SynchronizeStream(stream));
  std::vector<uint16_t> raw(logits.Numel());
  INFERX_RETURN_IF_ERROR(runtime.Copy(raw.data(), logits.Data(), raw.size() * 2,
                                      CopyKind::kDeviceToHost));
  std::vector<float> values(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    const uint32_t wide = static_cast<uint32_t>(raw[i]) << 16;
    std::memcpy(&values[i], &wide, sizeof(float));
  }
  return values;
}

}  // namespace

/// \brief The local GPTQ int8 checkpoint must dequantize into weights close
///        enough to the bf16 original that the logits agree.
///
/// Int8/G=128 quantization noise measured at the weights is ~0.8%; through
/// 28 layers the logits may drift an order of magnitude more, but the argmax
/// token has to survive.
TEST_F(ModelTest, GptqInt8CheckpointMatchesBf16Reference) {
  if (!std::filesystem::exists("models/Qwen3-0.6B-GPTQ-Int8/model.safetensors") ||
      !std::filesystem::exists("models/Qwen3-0.6B/model.safetensors")) {
    GTEST_SKIP() << "local Qwen3-0.6B checkpoints not present";
  }
  auto reference = Model::Load("models/Qwen3-0.6B", DeviceId::Cuda(0), 8, 2);
  ASSERT_TRUE(reference.ok()) << reference.status();
  auto quantized = Model::Load("models/Qwen3-0.6B-GPTQ-Int8", DeviceId::Cuda(0), 8, 2);
  ASSERT_TRUE(quantized.ok()) << quantized.status();

  const CheckpointConfig& config = (*quantized)->config();
  ASSERT_EQ(config.num_hidden_layers, (*reference)->config().num_hidden_layers);
  KvLayout layout;
  layout.entries_per_token = 2;
  layout.kv_heads = config.num_key_value_heads;
  layout.head_dim = config.head_dim;
  layout.dtype = DataType::kBFloat16;
  const DeviceId device = DeviceId::Cuda(0);
  auto ref_pool = KvBlockPool::Create(config.num_hidden_layers, 4, 2, layout, device);
  ASSERT_TRUE(ref_pool.ok());
  auto quant_pool = KvBlockPool::Create(config.num_hidden_layers, 4, 2, layout, device);
  ASSERT_TRUE(quant_pool.ok());
  auto input = MakeInput(2);
  ASSERT_TRUE(input.ok());

  auto ref_logits = ForwardLogits(**reference, (*reference)->config(), *runtime_, stream_,
                                  *input, *ref_pool);
  ASSERT_TRUE(ref_logits.ok()) << ref_logits.status();
  auto quant_logits = ForwardLogits(**quantized, config, *runtime_, stream_, *input,
                                    *quant_pool);
  ASSERT_TRUE(quant_logits.ok()) << quant_logits.status();
  ASSERT_EQ(ref_logits->size(), quant_logits->size());

  double dot = 0, ref_norm = 0;
  size_t ref_argmax = 0, quant_argmax = 0;
  for (size_t i = 0; i < ref_logits->size(); ++i) {
    dot += static_cast<double>((*ref_logits)[i] - (*quant_logits)[i]) *
           static_cast<double>((*ref_logits)[i] - (*quant_logits)[i]);
    ref_norm += static_cast<double>((*ref_logits)[i]) * (*ref_logits)[i];
    if ((*ref_logits)[i] > (*ref_logits)[ref_argmax]) ref_argmax = i;
    if ((*quant_logits)[i] > (*quant_logits)[quant_argmax]) quant_argmax = i;
  }
  const double relative = std::sqrt(dot / ref_norm);
  EXPECT_LT(relative, 0.08) << "logit relative error " << relative;
  EXPECT_EQ(ref_argmax, quant_argmax);
}

namespace {

/// Writes a tiny synthetic Qwen2.5 checkpoint and loads it end to end: the
/// biased QKV projections, the packed-bias execution path, and the family
/// registry selection all have to agree for forward to produce logits.
struct TempDir {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
                               ("inferx_qwen2_smoke_" + std::to_string(::getpid()));
  TempDir() { std::filesystem::create_directories(path); }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

void WriteF32(const std::filesystem::path& file,
              const std::vector<std::pair<std::string, std::pair<std::vector<int64_t>, float>>>&
                  tensors) {
  std::string header = "{";
  std::vector<std::byte> blob;
  bool first = true;
  for (const auto& [name, spec] : tensors) {
    int64_t numel = 1;
    for (auto d : spec.first) numel *= d;
    const int64_t begin = static_cast<int64_t>(blob.size());
    for (int64_t i = 0; i < numel; ++i) {
      const float value = spec.second + static_cast<float>(i % 97) * 0.01f;
      std::byte raw[4];
      std::memcpy(raw, &value, sizeof(raw));
      for (auto b : raw) blob.push_back(b);
    }
    const int64_t end = static_cast<int64_t>(blob.size());
    if (!first) header += ",";
    first = false;
    header += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(spec.first[0]);
    for (size_t d = 1; d < spec.first.size(); ++d) header += "," + std::to_string(spec.first[d]);
    header += "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
  }
  header += "}";
  uint64_t hlen = header.size();
  std::vector<std::byte> bytes(8 + hlen + blob.size(), std::byte{0});
  std::memcpy(bytes.data(), &hlen, 8);
  std::memcpy(bytes.data() + 8, header.data(), hlen);
  std::memcpy(bytes.data() + 8 + hlen, blob.data(), blob.size());
  std::ofstream out(file.string(), std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

}  // namespace

TEST_F(ModelTest, LoadsSyntheticQwen2AndRunsBiasedAttention) {
  constexpr int64_t kHidden = 16, kHeads = 4, kKv = 2, kDim = 8, kInter = 32, kVocab = 50;
  constexpr int64_t kQ = kHeads * kDim, kKV = kKv * kDim;
  TempDir dir;
  const std::string config =
      "{\"model_type\":\"qwen2\",\"architectures\":[\"Qwen2ForCausalLM\"],"
      "\"hidden_size\":16,\"intermediate_size\":32,\"num_hidden_layers\":2,"
      "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":8,"
      "\"vocab_size\":50,\"max_position_embeddings\":128,\"rms_norm_eps\":1e-5,"
      "\"rope_theta\":10000,\"attention_bias\":true,\"tie_word_embeddings\":false}";
  std::ofstream(dir.path / "config.json") << config;
  std::vector<std::pair<std::string, std::pair<std::vector<int64_t>, float>>> tensors{
      {"model.embed_tokens.weight", {{kVocab, kHidden}, 0.02f}},
      {"model.norm.weight", {{kHidden}, 0.5f}},
      {"lm_head.weight", {{kVocab, kHidden}, 0.01f}}};
  for (int layer = 0; layer < 2; ++layer) {
    const auto p = "model.layers." + std::to_string(layer) + ".";
    tensors.push_back({p + "input_layernorm.weight", {{kHidden}, 0.5f}});
    tensors.push_back({p + "post_attention_layernorm.weight", {{kHidden}, 0.5f}});
    tensors.push_back({p + "self_attn.q_proj.weight", {{kQ, kHidden}, 0.03f}});
    tensors.push_back({p + "self_attn.q_proj.bias", {{kQ}, 0.01f}});
    tensors.push_back({p + "self_attn.k_proj.weight", {{kKV, kHidden}, 0.03f}});
    tensors.push_back({p + "self_attn.k_proj.bias", {{kKV}, 0.01f}});
    tensors.push_back({p + "self_attn.v_proj.weight", {{kKV, kHidden}, 0.03f}});
    tensors.push_back({p + "self_attn.v_proj.bias", {{kKV}, 0.01f}});
    tensors.push_back({p + "self_attn.o_proj.weight", {{kHidden, kQ}, 0.03f}});
    tensors.push_back({p + "mlp.gate_proj.weight", {{kInter, kHidden}, 0.03f}});
    tensors.push_back({p + "mlp.up_proj.weight", {{kInter, kHidden}, 0.03f}});
    tensors.push_back({p + "mlp.down_proj.weight", {{kHidden, kInter}, 0.03f}});
  }
  WriteF32(dir.path / "model.safetensors", tensors);

  auto model = Model::Load(dir.path.string(), DeviceId::Cuda(0), /*max_tokens=*/8,
                           /*max_seqs=*/2);
  ASSERT_TRUE(model.ok()) << model.status();
  const CheckpointConfig& cfg = (*model)->config();
  KvLayout layout{2, cfg.num_key_value_heads, cfg.head_dim, DataType::kBFloat16};
  auto pool = KvBlockPool::Create(cfg.num_hidden_layers, 4, 2, layout, DeviceId::Cuda(0));
  ASSERT_TRUE(pool.ok());
  auto input = MakeInput(2);
  ASSERT_TRUE(input.ok());
  ops::OpContext ctx(*runtime_, stream_);
  dist::SingleRankComm comm;
  ModelState state;
  state.paged_kv = &*pool;
  for (int64_t i = 0; i < cfg.num_hidden_layers; ++i) state.layers.push_back(PagedKvState{i});
  const StatusOr<Tensor> logits = (*model)->Forward(*input, state, ctx, comm);
  ASSERT_TRUE(logits.ok()) << logits.status();
  ASSERT_EQ(logits->Dim(1), kVocab);
  // F32 weights round to small nonzero values; every logit must be finite.
  std::vector<uint16_t> raw(logits->Numel());
  ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
  ASSERT_TRUE(runtime_->Copy(raw.data(), logits->Data(), raw.size() * 2,
                             CopyKind::kDeviceToHost).ok());
  for (uint16_t bits : raw) {
    const uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0;
    std::memcpy(&value, &wide, 4);
    EXPECT_TRUE(std::isfinite(value));
  }
}

TEST_F(ModelTest, RejectsUnsupportedArchitecture) {
  const std::string dir = ::testing::TempDir() + "inferx-unsupported-arch";
  std::filesystem::create_directories(dir);
  std::ofstream(dir + "/config.json")
      << R"({"model_type": "gpt_neox", "architectures": ["GPTNeoXForCausalLM"],)"
      << R"( "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1,)"
      << R"( "vocab_size": 16})";
  auto model = Model::Load(dir, DeviceId::Cuda(0), 8, 2);
  EXPECT_EQ(model.status().code(), absl::StatusCode::kUnimplemented);
}


TEST_F(ModelTest, RejectsMissingConfig) {
  auto model = Model::Load("/nonexistent/model-dir", DeviceId::Cuda(0), 8, 2);
  EXPECT_FALSE(model.ok());
}

}  // namespace
}  // namespace inferx
