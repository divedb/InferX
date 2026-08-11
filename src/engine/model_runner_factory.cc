#include "model_runner_factory.h"

#include <utility>

#include "inferx/engine/engine.h"
#include "inferx/model/deepseek_v2.h"
#include "inferx/model/gpt_oss.h"
#include "kv_autosize.h"
#include "qwen2_runner.h"

namespace inferx::engine {

StatusOr<std::unique_ptr<ModelRunner>> BuildModelRunner(
    const EngineConfig& config, model::Architecture architecture,
    DeviceId primary_device, HostSampler host_sampler) {
  if (architecture == model::Architecture::kDeepSeekV2) {
    if (config.tensor_parallel_size != 1) {
      return UnimplementedError(
          "tensor parallel serving is currently implemented for Qwen2 only");
    }
    if (config.fp8_weights || config.int4_weights || config.fp8_kv_cache) {
      return InvalidArgumentError(
          "--quantization / --kv-cache-dtype are not supported for the "
          "DeepSeek-V2 architecture");
    }
    INFERX_ASSIGN_OR_RETURN(
        model::DeepseekV2Model model,
        model::DeepseekV2Model::Load(config.model_dir, primary_device));
    INFERX_ASSIGN_OR_RETURN(
        const int64_t kv_blocks,
        AutosizeKvBlocksOnDevice(
            KvSizingSpec{
                .explicit_blocks = config.kv_blocks,
                .explicit_bytes = config.kv_cache_memory_bytes,
                .gpu_memory_utilization = config.gpu_memory_utilization,
                .block_bytes = model.KvBlockBytes(config.block_size),
                .min_blocks =
                    (config.scheduler.max_seq_len + config.block_size - 1) /
                    config.block_size},
            primary_device));
    INFERX_RETURN_IF_ERROR(model.AttachKvCache(kv_blocks, config.block_size));
    return MakeSyncModelRunner(std::move(model), std::move(host_sampler));
  }

  if (architecture == model::Architecture::kGptOss) {
    if (config.tensor_parallel_size != 1) {
      return UnimplementedError(
          "tensor parallel serving is currently implemented for Qwen2 only");
    }
    if (config.fp8_weights || config.int4_weights || config.fp8_kv_cache) {
      return InvalidArgumentError(
          "--quantization / --kv-cache-dtype are not supported for the "
          "gpt-oss architecture");
    }
    INFERX_ASSIGN_OR_RETURN(
        model::GptOssModel model,
        model::GptOssModel::Load(config.model_dir, primary_device));
    INFERX_ASSIGN_OR_RETURN(
        const int64_t kv_blocks,
        AutosizeKvBlocksOnDevice(
            KvSizingSpec{
                .explicit_blocks = config.kv_blocks,
                .explicit_bytes = config.kv_cache_memory_bytes,
                .gpu_memory_utilization = config.gpu_memory_utilization,
                .block_bytes = model.KvBlockBytes(config.block_size),
                .min_blocks =
                    (config.scheduler.max_seq_len + config.block_size - 1) /
                    config.block_size},
            primary_device));
    INFERX_RETURN_IF_ERROR(model.AttachKvCache(kv_blocks, config.block_size));
    return MakeSyncModelRunner(std::move(model), std::move(host_sampler));
  }

  Qwen2RunnerConfig runner_config;
  runner_config.model_dir = config.model_dir;
  runner_config.device_kind = config.device_kind;
  runner_config.devices = config.devices;
  runner_config.use_nccl = config.comm_backend == "nccl";
  runner_config.collective_timing_sample_every =
      config.collective_timing_sample_every;
  runner_config.fp8_weights = config.fp8_weights;
  runner_config.int4_weights = config.int4_weights;
  runner_config.fp8_kv_cache = config.fp8_kv_cache;
  runner_config.kv_blocks = config.kv_blocks;
  runner_config.kv_cache_memory_bytes = config.kv_cache_memory_bytes;
  runner_config.gpu_memory_utilization = config.gpu_memory_utilization;
  runner_config.block_size = config.block_size;
  runner_config.max_seq_len = config.scheduler.max_seq_len;
  runner_config.max_sampling_rows = config.scheduler.max_running;

  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<Qwen2Runner> runner,
                          Qwen2Runner::Create(runner_config));
  return std::unique_ptr<ModelRunner>(std::move(runner));
}

}  // namespace inferx::engine
