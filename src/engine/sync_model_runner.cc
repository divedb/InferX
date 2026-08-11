#include "sync_model_runner.h"

#include <utility>

#include "inferx/model/deepseek_v2.h"
#include "inferx/model/gpt_oss.h"

namespace inferx::engine {
namespace {

template <typename Model>
class SyncModelAdapter final : public ModelRunner {
 public:
  SyncModelAdapter(Model model, HostSampler sampler)
      : model_(std::move(model)), sampler_(std::move(sampler)) {}

  KvBlockPool* kv_pool() override { return model_.kv_pool(); }
  int64_t vocab_size() const { return model_.config().vocab_size; }
  Status ReserveActivations(int64_t max_tokens) override {
    return model_.ReserveActivations(max_tokens);
  }
  Status Step(const model::ForwardBatch& batch, std::vector<int32_t>* sampled,
              std::vector<SampledLogprob>* logprobs) override {
    INFERX_RETURN_IF_ERROR(model_.Step(batch, &logits_));
    const int64_t vocab = vocab_size();
    sampled->clear();
    sampled->reserve(batch.logits_indices.size());
    logprobs->clear();
    logprobs->resize(batch.logits_indices.size());
    for (size_t i = 0; i < batch.logits_indices.size(); ++i) {
      float* row = logits_.data() + i * static_cast<size_t>(vocab);
      sampled->push_back(sampler_(row, vocab, batch, i, &(*logprobs)[i]));
    }
    return OkStatus();
  }
  Status CaptureDecodeGraph(int64_t num_seqs,
                            int64_t max_blocks_per_seq) override {
    return model_.CaptureDecodeGraph(num_seqs, max_blocks_per_seq);
  }
  float last_step_device_ms() const override { return 0.0f; }
  std::vector<RankTelemetry> telemetry() const override { return {}; }

 private:
  Model model_;
  HostSampler sampler_;
  std::vector<float> logits_;
};

}  // namespace

std::unique_ptr<ModelRunner> MakeSyncModelRunner(model::GptOssModel model,
                                                 HostSampler sampler) {
  return std::make_unique<SyncModelAdapter<model::GptOssModel>>(
      std::move(model), std::move(sampler));
}

std::unique_ptr<ModelRunner> MakeSyncModelRunner(model::DeepseekV2Model model,
                                                 HostSampler sampler) {
  return std::make_unique<SyncModelAdapter<model::DeepseekV2Model>>(
      std::move(model), std::move(sampler));
}

}  // namespace inferx::engine
