#include "inferx/models/model.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/engine/model_executor.h"
#include "inferx/engine/scheduler.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/models/architectures/qwen3/config.h"
#include "inferx/models/load_options.h"
#include "inferx/ops/op_context.h"
#include "inferx/util/hf_config.h"

namespace inferx {
namespace {

models::ModelLoadOptions LoadOptions(const std::string& directory) {
  models::ModelLoadOptions options;
  options.directory = directory;
  options.device = DeviceId::Cuda(0);


  return options;
}

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

  /// \brief A minimal valid batch: one single-token sequence.
  StatusOr<models::ModelInputs> MakeInput(int num_tokens) {
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
    logit_rows_ = std::move(logit_rows);

    return models::ModelInputs{
        std::move(token_ids),
        models::AttentionBatch{std::move(positions),
                               std::move(batch_indices),
                               std::move(qo_indptr),
                               std::move(kv_indptr),
                               std::move(kv_indices),
                               std::move(last_page_len),
                               host_qo_,
                               {},
                               num_tokens,
                               1}};
  }

  /// Binds runner-style state: one paged-KV pool layer per declared layer.
  StatusOr<models::ModelState> MakeState(const models::CausalLanguageModel& model,
                                         int num_kv_blocks, int block_size) {
    const auto& requirements = model.GetRequirements().layers;
    std::vector<KvLayout> layouts;
    for (const auto& spec : requirements) {
      layouts.push_back(std::get<models::PagedKvStateSpec>(spec).layout);
    }
    INFERX_ASSIGN_OR_RETURN(auto pool, KvBlockPool::Create(num_kv_blocks, block_size, layouts,
                                                           DeviceId::Cuda(0)));
    pools_.push_back(std::make_unique<KvBlockPool>(std::move(pool)));
    models::ModelState state;
    state.paged_kv = pools_.back().get();
    for (int64_t i = 0; i < static_cast<int64_t>(requirements.size()); ++i) {
      state.layers.push_back(models::PagedKvState{i});
    }
    return state;
  }

  StatusOr<Tensor> ForwardLogits(models::CausalLanguageModel& model,
                                 const models::ModelInputs& input,
                                 models::ModelState& state) {
    ops::OpContext ctx(*runtime_, stream_);
    dist::SingleRankComm comm;
    models::ForwardContext fctx{ctx, state, comm};
    INFERX_RETURN_IF_ERROR(model.Prepare({64, 2, 64}, fctx));
    models::OutputRequest outputs{models::LogitSelection{*logit_rows_}};
    INFERX_ASSIGN_OR_RETURN(auto preparation, model.PrepareStep(input, outputs, fctx));
    (void)preparation;
    INFERX_ASSIGN_OR_RETURN(auto result, model.Forward(input, outputs, fctx));
    return std::move(result.logits->values);
  }

  std::vector<int32_t> host_qo_;
  std::optional<Tensor> logit_rows_;
  std::vector<std::unique_ptr<KvBlockPool>> pools_;
  DeviceRuntime* runtime_ = nullptr;
  Stream stream_;
};

TEST_F(ModelTest, LoadsQwen3CheckpointAndSpec) {
  auto model = models::LoadModel(LoadOptions("models/Qwen3-0.6B"));
  ASSERT_TRUE(model.ok()) << model.status();
  const models::ModelSpec& spec = (*model)->GetModelSpec();
  EXPECT_EQ(spec.architecture, "Qwen3ForCausalLM");
  EXPECT_EQ(spec.vocab_size, 151936);
  EXPECT_EQ(spec.max_position_embeddings, 40960);
  ASSERT_FALSE((*model)->GetRequirements().layers.empty());
  EXPECT_EQ(spec.hidden_size, 1024);
  EXPECT_EQ(spec.intermediate_size, 3072);
  EXPECT_EQ(spec.num_hidden_layers, 28);
  EXPECT_EQ(spec.num_attention_heads, 16);
  EXPECT_EQ(spec.num_key_value_heads, 8);
  EXPECT_EQ(spec.head_dim, 128);
  EXPECT_EQ(spec.eos_token_ids, (std::vector<TokenId>{151645}));
  EXPECT_FLOAT_EQ(spec.rms_norm_eps, 1e-6);
}

TEST_F(ModelTest, CompleteForwardSupportsSelectedRowsAndPreservesHiddenStates) {
  auto loaded = models::LoadModel(LoadOptions("models/Qwen3-0.6B"));
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  auto& model = **loaded;
  auto input = MakeInput(2);
  ASSERT_TRUE(input.ok());
  auto state = MakeState(model, 2, 4);
  ASSERT_TRUE(state.ok());
  ops::OpContext ctx(*runtime_, stream_);
  dist::SingleRankComm comm;
  models::ForwardContext fctx{ctx, *state, comm};
  EXPECT_EQ(model.Forward(*input, {}, fctx).status().code(),
            absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(model.Prepare({2, 1, 2}, fctx).ok());
  const auto workspace = model.GetMemoryUsage().workspace_bytes;

  auto read = [&](const Tensor& tensor) {
    std::vector<uint16_t> values(tensor.Numel());
    EXPECT_TRUE(runtime_->SynchronizeStream(stream_).ok());
    EXPECT_TRUE(runtime_->Copy(values.data(), tensor.Data(), tensor.NBytes(),
                               CopyKind::kDeviceToHost).ok());
    return values;
  };
  models::OutputRequest hidden_only{std::nullopt, true};
  auto hidden = model.Forward(*input, hidden_only, fctx);
  ASSERT_TRUE(hidden.ok()) << hidden.status();
  ASSERT_TRUE(hidden->hidden_states.has_value());
  EXPECT_FALSE(hidden->logits.has_value());
  const auto expected_hidden = read(hidden->hidden_states->hidden);

  auto rows = Tensor::Empty(DataType::kInt32, Shape{2}, DeviceId::Cuda(0));
  ASSERT_TRUE(rows.ok());
  const int32_t selected[] = {1, 0};
  ASSERT_TRUE(runtime_->Copy(rows->Data(), selected, sizeof(selected),
                             CopyKind::kHostToDevice).ok());
  models::OutputRequest both{models::LogitSelection{*rows}, true};
  ASSERT_TRUE(model.PrepareStep(*input, both, fctx).ok());
  auto combined = model.Forward(*input, both, fctx);
  ASSERT_TRUE(combined.ok()) << combined.status();
  ASSERT_TRUE(combined->logits.has_value());
  ASSERT_TRUE(combined->hidden_states.has_value());
  EXPECT_EQ(combined->logits->values.Dim(0), 2);  // More rows than sequences.
  EXPECT_EQ(read(combined->hidden_states->hidden), expected_hidden);
  const auto combined_logits = read(combined->logits->values);

  auto single = model.Forward(*input,
      models::OutputRequest{models::LogitSelection{*logit_rows_}}, fctx);
  ASSERT_TRUE(single.ok());
  ASSERT_TRUE(single->logits.has_value());
  EXPECT_FALSE(single->hidden_states.has_value());
  const auto single_logits = read(single->logits->values);
  EXPECT_EQ(single_logits, std::vector<uint16_t>(combined_logits.begin(),
      combined_logits.begin() + single_logits.size()));

  auto state_only = model.Forward(*input, {}, fctx);
  ASSERT_TRUE(state_only.ok());
  EXPECT_FALSE(state_only->logits.has_value());
  EXPECT_FALSE(state_only->hidden_states.has_value());
  EXPECT_TRUE(runtime_->SynchronizeStream(stream_).ok());
  EXPECT_EQ(model.GetMemoryUsage().workspace_bytes, workspace);
  EXPECT_FALSE(model.Prepare({3, 1, 2}, fctx).ok());
}

/// The architecture's typed config owns the full dimension set; the
/// checkpoint's config.json parses directly into it.
TEST(Qwen3ConfigTest, ParsesCheckpointConfig) {
  auto hf = HfConfig::FromFile("models/Qwen3-0.6B/config.json");
  if (!hf.ok()) GTEST_SKIP() << "local Qwen3-0.6B checkpoint not present";
  auto config = models::Qwen3Config::Parse(hf->Text());
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->hidden_size, 1024);
  EXPECT_EQ(config->intermediate_size, 3072);
  EXPECT_EQ(config->num_hidden_layers, 28);
  EXPECT_EQ(config->num_attention_heads, 16);
  EXPECT_EQ(config->num_key_value_heads, 8);
  EXPECT_EQ(config->head_dim, 128);
  EXPECT_EQ(config->vocab_size, 151936);
  EXPECT_FLOAT_EQ(config->rms_norm_eps, 1e-6);
  EXPECT_FLOAT_EQ(config->rope.theta, 1e6f);
  EXPECT_TRUE(config->tie_word_embeddings);
  EXPECT_TRUE(config->Validate().ok());
}

TEST_F(ModelTest, ForwardReturnsLogits) {
  auto model = models::LoadModel(LoadOptions("models/Qwen3-0.6B"));
  ASSERT_TRUE(model.ok()) << model.status();
  const auto& requirements = (*model)->GetRequirements().layers;
  ASSERT_EQ(requirements.size(), 28u);
  const KvLayout& layout = std::get<models::PagedKvStateSpec>(requirements[0]).layout;
  EXPECT_EQ(layout.kv_heads, 8);
  EXPECT_EQ(layout.head_dim, 128);
  auto input = MakeInput(2);
  ASSERT_TRUE(input.ok());
  auto state = MakeState(**model, /*num_kv_blocks=*/4, /*block_size=*/2);
  ASSERT_TRUE(state.ok());
  const StatusOr<Tensor> logits = ForwardLogits(**model, *input, *state);
  ASSERT_TRUE(logits.ok()) << logits.status();
  EXPECT_EQ(logits->Rank(), 2);
  EXPECT_EQ(logits->Dim(0), 1);
  EXPECT_EQ(logits->Dim(1), (*model)->GetModelSpec().vocab_size);
  EXPECT_EQ(logits->GetDataType(), DataType::kBFloat16);
}

namespace {

/// Runs one fixed two-token prompt through a loaded model and returns the
/// last-row logits on the host.
StatusOr<std::vector<float>> ForwardLogitsOnHost(models::CausalLanguageModel& model,
                                           const models::ModelSpec& info, DeviceRuntime& runtime,
                                           Stream stream, const models::ModelInputs& input,
                                           models::ModelState& state, Tensor logit_rows) {
  ops::OpContext ctx(runtime, stream);
  dist::SingleRankComm comm;
  models::ForwardContext fctx{ctx, state, comm};
  INFERX_RETURN_IF_ERROR(model.Prepare({64, 2, 64}, fctx));
  models::OutputRequest outputs{models::LogitSelection{logit_rows}};
  INFERX_ASSIGN_OR_RETURN(auto preparation, model.PrepareStep(input, outputs, fctx));
  (void)preparation;
  INFERX_ASSIGN_OR_RETURN(auto result, model.Forward(input, outputs, fctx));
  const auto& out = *result.logits;
  INFERX_RETURN_IF_ERROR(runtime.SynchronizeStream(stream));
  std::vector<uint16_t> raw(out.values.Numel());
  INFERX_RETURN_IF_ERROR(runtime.Copy(raw.data(), out.values.Data(), raw.size() * 2,
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
  auto reference = models::LoadModel(LoadOptions("models/Qwen3-0.6B"));
  ASSERT_TRUE(reference.ok()) << reference.status();
  auto quantized = models::LoadModel(LoadOptions("models/Qwen3-0.6B-GPTQ-Int8"));
  ASSERT_TRUE(quantized.ok()) << quantized.status();

  ASSERT_EQ((*quantized)->GetRequirements().layers.size(),
            (*reference)->GetRequirements().layers.size());
  const DeviceId device = DeviceId::Cuda(0);
  auto pool_for = [&](const models::CausalLanguageModel& model) {
    std::vector<KvLayout> layouts;
    for (const auto& spec : model.GetRequirements().layers) {
      layouts.push_back(std::get<models::PagedKvStateSpec>(spec).layout);
    }
    return KvBlockPool::Create(4, 2, layouts, device);
  };
  auto ref_pool = pool_for(**reference);
  ASSERT_TRUE(ref_pool.ok());
  auto quant_pool = pool_for(**quantized);
  ASSERT_TRUE(quant_pool.ok());
  auto input = MakeInput(2);
  ASSERT_TRUE(input.ok());

  models::ModelState ref_state;
  ref_state.paged_kv = &*ref_pool;
  for (size_t i = 0; i < (*reference)->GetRequirements().layers.size(); ++i) {
    ref_state.layers.push_back(models::PagedKvState{static_cast<int64_t>(i)});
  }
  models::ModelState quant_state;
  quant_state.paged_kv = &*quant_pool;
  for (size_t i = 0; i < (*quantized)->GetRequirements().layers.size(); ++i) {
    quant_state.layers.push_back(models::PagedKvState{static_cast<int64_t>(i)});
  }
  auto ref_logits = ForwardLogitsOnHost(**reference, (*reference)->GetModelSpec(), *runtime_, stream_, *input,
                                  ref_state, *logit_rows_);
  ASSERT_TRUE(ref_logits.ok()) << ref_logits.status();
  auto quant_logits = ForwardLogitsOnHost(**quantized, (*quantized)->GetModelSpec(), *runtime_, stream_,
                                    *input, quant_state, *logit_rows_);
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
/// biased QKV projections, the packed-bias execution path, and the registry
/// selection all have to agree for forward to produce logits.
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

  auto options = LoadOptions(dir.path.string());
  auto model = models::LoadModel(options);
  ASSERT_TRUE(model.ok()) << model.status();
  auto state = MakeState(**model, 4, 2);
  ASSERT_TRUE(state.ok());
  auto input = MakeInput(2);
  ASSERT_TRUE(input.ok());
  const StatusOr<Tensor> logits = ForwardLogits(**model, *input, *state);
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
  auto model = models::LoadModel(LoadOptions(dir));
  EXPECT_EQ(model.status().code(), absl::StatusCode::kUnimplemented);
}


/// Graph replay must reproduce eager sampling exactly: sampling stays
/// outside the captured model path, so per-step parameters and RNG offsets
/// are applied fresh on every replay.
TEST_F(ModelTest, GraphReplaySamplingMatchesEager) {
  if (!std::filesystem::exists("models/Qwen3-0.6B/model.safetensors")) {
    GTEST_SKIP() << "local Qwen3-0.6B checkpoint not present";
  }
  const auto generate = [&](bool graphs) -> std::vector<int> {
    ModelConfig mc;
    mc.device.device_type = "cuda";
    mc.model_dir = "models/Qwen3-0.6B";
    SchedulerConfig sc;
    sc.max_num_batched_tokens = 4096;
    sc.max_num_seqs = 2;
    CacheConfig cc;
    cc.num_kv_blocks = 512;
    cc.block_size = 16;
    ExecutionConfig ec;
    ec.enable_cuda_graphs = graphs;
    auto built = ModelExecutor::Create(mc, cc, sc, ec);
    if (!built.ok()) {
      // No CUDA device in this environment.
      return std::vector<int>{};
    }
    Scheduler scheduler(sc, (*built)->kv_pool(), (*built)->GetModelSpec().eos_token_ids);
    sampling::SamplingParams params;
    params.temperature = 0.8f;
    params.top_p = 0.95f;
    params.seed = 42;
    params.ignore_eos = true;
    params.max_tokens = 8;
    Request request(1, {785, 6722, 315, 9625, 374}, params);
    EXPECT_TRUE(scheduler.AddRequest(std::move(request)).ok());
    std::vector<int> tokens;
    while (scheduler.HasRequests()) {
      auto plan = scheduler.Schedule();
      if (!plan.ok()) {
        ADD_FAILURE() << plan.status();
        return {};
      }
      auto result = (*built)->Execute(*plan);
      if (!result.ok()) {
        ADD_FAILURE() << result.status();
        return {};
      }
      if (!scheduler.UpdateFromOutput(*plan, *result).ok()) {
        ADD_FAILURE() << "update failed";
        return {};
      }
    }
    while (auto finished = scheduler.PopFinished()) {
      tokens = finished->output();
    }
    return tokens;
  };
  const auto eager = generate(false);
  ASSERT_EQ(eager.size(), 8u) << "eager generation incomplete";
  const auto replayed = generate(true);
  ASSERT_EQ(replayed.size(), 8u) << "graph generation incomplete";
  EXPECT_EQ(eager, replayed);
}

TEST_F(ModelTest, RejectsMissingConfig) {
  auto model = models::LoadModel(LoadOptions("/nonexistent/model-dir"));
  EXPECT_FALSE(model.ok());
}

}  // namespace
}  // namespace inferx
