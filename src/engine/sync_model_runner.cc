#include "sync_model_runner.h"

#include <utility>

#include "inferx/model/deepseek_v2.h"
#include "inferx/model/gpt_oss.h"

namespace inferx::engine {
namespace {

template <typename Model>
class SyncModelAdapter final : public SyncModelRunner {
 public:
  explicit SyncModelAdapter(Model model) : model_(std::move(model)) {}

  KvBlockPool* kv_pool() override { return model_.kv_pool(); }
  int64_t vocab_size() const override { return model_.config().vocab_size; }
  Status ReserveActivations(int64_t max_tokens) override {
    return model_.ReserveActivations(max_tokens);
  }
  Status Step(const model::ForwardBatch& batch,
              std::vector<float>* logits) override {
    return model_.Step(batch, logits);
  }
  Status CaptureDecodeGraph(int64_t num_seqs,
                            int64_t max_blocks_per_seq) override {
    return model_.CaptureDecodeGraph(num_seqs, max_blocks_per_seq);
  }

 private:
  Model model_;
};

}  // namespace

std::unique_ptr<SyncModelRunner> MakeSyncModelRunner(model::GptOssModel model) {
  return std::make_unique<SyncModelAdapter<model::GptOssModel>>(
      std::move(model));
}

std::unique_ptr<SyncModelRunner> MakeSyncModelRunner(
    model::DeepseekV2Model model) {
  return std::make_unique<SyncModelAdapter<model::DeepseekV2Model>>(
      std::move(model));
}

}  // namespace inferx::engine
