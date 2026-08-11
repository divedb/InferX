#include "inferx/engine/model_runner_factory.h"

#include <utility>

#include "inferx/engine/engine.h"
#include "inferx/engine/kv_autosize.h"
#include "inferx/engine/model_runner_validation.h"
#include "inferx/engine/sync_model_runner.h"
#include "inferx/engine/tensor_parallel_runner.h"
#include "inferx/model/deepseek_v2.h"
#include "inferx/model/gpt_oss.h"
#include "inferx/model/qwen2.h"

namespace inferx::engine {
namespace {

class Qwen2RankModel final : public RankModel {
 public:
  Qwen2RankModel(model::Qwen2Model model, bool fp8_weights, bool int4_weights,
                 bool fp8_kv_cache)
      : model_(std::move(model)),
        fp8_weights_(fp8_weights),
        int4_weights_(int4_weights),
        fp8_kv_cache_(fp8_kv_cache) {}

  Status PrepareWeights() override {
    if (fp8_weights_) INFERX_RETURN_IF_ERROR(model_.QuantizeWeightsToF8());
    if (int4_weights_) INFERX_RETURN_IF_ERROR(model_.QuantizeWeightsToInt4());
    if (fp8_kv_cache_) INFERX_RETURN_IF_ERROR(model_.EnableFp8KvCache());
    return OkStatus();
  }
  int64_t KvBlockBytes(int64_t block_size) const override {
    return model_.KvBlockBytes(block_size);
  }
  Status AttachKvCache(int64_t blocks, int64_t block_size) override {
    return model_.AttachKvCache(blocks, block_size);
  }
  Status EnableSampling(int64_t max_rows) override {
    return model_.EnableDeviceSampling(max_rows);
  }
  KvBlockPool* kv_pool() override { return model_.kv_pool(); }
  Status ReserveActivations(int64_t max_tokens) override {
    return model_.ReserveActivations(max_tokens);
  }
  Status StepAsync(const model::ForwardBatch& batch) override {
    return model_.StepAsync(batch);
  }
  Status AwaitStep(std::vector<int32_t>* sampled) override {
    return model_.AwaitStep(sampled);
  }
  Status ReadSampledLogprobs(std::vector<SampledLogprob>* logprobs) override {
    std::vector<model::Qwen2Model::SampledLogprob> raw;
    INFERX_RETURN_IF_ERROR(model_.ReadSampledLogprobs(&raw));
    logprobs->clear();
    logprobs->reserve(raw.size());
    for (auto& item : raw) {
      logprobs->push_back({.present = item.present,
                           .logprob = item.logprob,
                           .top = std::move(item.top)});
    }
    return OkStatus();
  }
  Status CaptureDecodeGraph(int64_t num_seqs,
                            int64_t max_blocks_per_seq) override {
    return model_.CaptureDecodeGraph(num_seqs, max_blocks_per_seq);
  }
  Status AbortCommunication() override { return model_.AbortCommunicator(); }
  float last_step_device_ms() const override {
    return model_.last_step_device_ms();
  }

 private:
  model::Qwen2Model model_;
  bool fp8_weights_ = false;
  bool int4_weights_ = false;
  bool fp8_kv_cache_ = false;
};

class GptOssRankModel final : public RankModel {
 public:
  GptOssRankModel(model::GptOssModel model, HostSampler sampler, int rank)
      : model_(std::move(model)), sampler_(std::move(sampler)), rank_(rank) {}

  Status PrepareWeights() override { return OkStatus(); }
  int64_t KvBlockBytes(int64_t block_size) const override {
    return model_.KvBlockBytes(block_size);
  }
  Status AttachKvCache(int64_t blocks, int64_t block_size) override {
    return model_.AttachKvCache(blocks, block_size);
  }
  Status EnableSampling(int64_t) override { return OkStatus(); }
  KvBlockPool* kv_pool() override { return model_.kv_pool(); }
  Status ReserveActivations(int64_t max_tokens) override {
    return model_.ReserveActivations(max_tokens);
  }
  Status StepAsync(const model::ForwardBatch& batch) override {
    batch_ = batch;
    sampled_.clear();
    logprobs_.clear();
    return model_.Step(batch, &logits_);
  }
  Status AwaitStep(std::vector<int32_t>* sampled) override {
    sampled->clear();
    if (rank_ != 0) return OkStatus();

    const int64_t vocab = model_.config().vocab_size;
    sampled_.reserve(batch_.logits_indices.size());
    logprobs_.resize(batch_.logits_indices.size());
    for (size_t i = 0; i < batch_.logits_indices.size(); ++i) {
      float* row = logits_.data() + i * static_cast<size_t>(vocab);
      sampled_.push_back(sampler_(row, vocab, batch_, i, &logprobs_[i]));
    }
    *sampled = sampled_;
    return OkStatus();
  }
  Status ReadSampledLogprobs(std::vector<SampledLogprob>* logprobs) override {
    *logprobs = logprobs_;
    return OkStatus();
  }
  Status CaptureDecodeGraph(int64_t num_seqs,
                            int64_t max_blocks_per_seq) override {
    return model_.CaptureDecodeGraph(num_seqs, max_blocks_per_seq);
  }
  Status AbortCommunication() override { return model_.AbortCommunicator(); }
  float last_step_device_ms() const override { return 0.0f; }

 private:
  model::GptOssModel model_;
  HostSampler sampler_;
  int rank_ = 0;
  model::ForwardBatch batch_;
  std::vector<float> logits_;
  std::vector<int32_t> sampled_;
  std::vector<SampledLogprob> logprobs_;
};

template <typename Model>
Status AttachSizedKvCache(Model* model, const EngineConfig& config,
                          DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(
      const int64_t kv_blocks,
      AutosizeKvBlocksOnDevice(
          KvSizingSpec{.explicit_blocks = config.kv_blocks,
                       .explicit_bytes = config.kv_cache_memory_bytes,
                       .gpu_memory_utilization = config.gpu_memory_utilization,
                       .block_bytes = model->KvBlockBytes(config.block_size),
                       .min_blocks = (config.scheduler.max_seq_len +
                                      config.block_size - 1) /
                                     config.block_size},
          device));
  return model->AttachKvCache(kv_blocks, config.block_size);
}

StatusOr<std::unique_ptr<ModelRunner>> BuildDeepseekV2Runner(
    const EngineConfig& config, DeviceId device, HostSampler sampler) {
  INFERX_ASSIGN_OR_RETURN(
      model::DeepseekV2Model model,
      model::DeepseekV2Model::Load(config.model_dir, device));
  INFERX_RETURN_IF_ERROR(AttachSizedKvCache(&model, config, device));
  return MakeSyncModelRunner(std::move(model), std::move(sampler));
}

StatusOr<std::unique_ptr<ModelRunner>> BuildGptOssRunner(
    const EngineConfig& config, HostSampler sampler) {
  TensorParallelRunnerConfig runner_config;
  runner_config.device_kind = config.device_kind;
  runner_config.devices = config.devices;
  runner_config.use_nccl = config.comm_backend == "nccl";
  runner_config.collective_timing_sample_every =
      config.collective_timing_sample_every;
  runner_config.kv_blocks = config.kv_blocks;
  runner_config.kv_cache_memory_bytes = config.kv_cache_memory_bytes;
  runner_config.gpu_memory_utilization = config.gpu_memory_utilization;
  runner_config.block_size = config.block_size;
  runner_config.max_seq_len = config.scheduler.max_seq_len;
  runner_config.max_sampling_rows = config.scheduler.max_running;
  runner_config.supports_graph_capture = config.tensor_parallel_size == 1;
  runner_config.requires_graph_warmup = false;

  const std::string model_dir = config.model_dir;
  RankModelFactory factory =
      [model_dir, sampler = std::move(sampler)](
          ParallelContext context) -> StatusOr<std::unique_ptr<RankModel>> {
    const int rank = context.tp_rank();
    INFERX_ASSIGN_OR_RETURN(
        model::GptOssModel model,
        model::GptOssModel::Load(model_dir, context.TakeTpCommunicator()));
    return std::unique_ptr<RankModel>(
        std::make_unique<GptOssRankModel>(std::move(model), sampler, rank));
  };

  INFERX_ASSIGN_OR_RETURN(
      std::unique_ptr<TensorParallelRunner> runner,
      TensorParallelRunner::Create(runner_config, std::move(factory)));
  return std::unique_ptr<ModelRunner>(std::move(runner));
}

StatusOr<std::unique_ptr<ModelRunner>> BuildQwenFamilyRunner(
    const EngineConfig& config) {
  TensorParallelRunnerConfig runner_config;
  runner_config.device_kind = config.device_kind;
  runner_config.devices = config.devices;
  runner_config.use_nccl = config.comm_backend == "nccl";
  runner_config.collective_timing_sample_every =
      config.collective_timing_sample_every;
  runner_config.kv_blocks = config.kv_blocks;
  runner_config.kv_cache_memory_bytes = config.kv_cache_memory_bytes;
  runner_config.gpu_memory_utilization = config.gpu_memory_utilization;
  runner_config.block_size = config.block_size;
  runner_config.max_seq_len = config.scheduler.max_seq_len;
  runner_config.max_sampling_rows = config.scheduler.max_running;
  runner_config.supports_graph_capture = true;
  runner_config.requires_graph_warmup = true;

  const std::string model_dir = config.model_dir;
  const bool fp8_weights = config.fp8_weights;
  const bool int4_weights = config.int4_weights;
  const bool fp8_kv_cache = config.fp8_kv_cache;
  RankModelFactory factory =
      [model_dir, fp8_weights, int4_weights, fp8_kv_cache](
          ParallelContext context) -> StatusOr<std::unique_ptr<RankModel>> {
    INFERX_ASSIGN_OR_RETURN(model::Qwen2Model model,
                            model::Qwen2Model::LoadFromDirectory(
                                model_dir, context.TakeTpCommunicator()));
    return std::unique_ptr<RankModel>(std::make_unique<Qwen2RankModel>(
        std::move(model), fp8_weights, int4_weights, fp8_kv_cache));
  };

  INFERX_ASSIGN_OR_RETURN(
      std::unique_ptr<TensorParallelRunner> runner,
      TensorParallelRunner::Create(runner_config, std::move(factory)));
  return std::unique_ptr<ModelRunner>(std::move(runner));
}

}  // namespace

StatusOr<std::unique_ptr<ModelRunner>> BuildModelRunner(
    const EngineConfig& config, model::Architecture architecture,
    DeviceId primary_device, HostSampler host_sampler) {
  INFERX_RETURN_IF_ERROR(ValidateModelRunnerFeatures(
      architecture, {.tensor_parallel_size = config.tensor_parallel_size,
                     .fp8_weights = config.fp8_weights,
                     .int4_weights = config.int4_weights,
                     .fp8_kv_cache = config.fp8_kv_cache}));

  switch (architecture) {
    case model::Architecture::kDeepSeekV2:
      return BuildDeepseekV2Runner(config, primary_device,
                                   std::move(host_sampler));
    case model::Architecture::kGptOss:
      return BuildGptOssRunner(config, std::move(host_sampler));
    case model::Architecture::kQwen2:
    case model::Architecture::kQwen2Moe:
    case model::Architecture::kLlama:
      return BuildQwenFamilyRunner(config);
  }
  return InvalidArgumentError("unsupported model architecture ",
                              static_cast<int>(architecture));
}

}  // namespace inferx::engine
