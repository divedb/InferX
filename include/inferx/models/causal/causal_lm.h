/// \file
/// \brief CausalLM: a DecoderStack backbone plus a language-model head; the
/// Model implementation served for token generation.

#ifndef INFERX_MODELS_CAUSAL_CAUSAL_LM_H_
#define INFERX_MODELS_CAUSAL_CAUSAL_LM_H_

#include <memory>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/config/parallel_config.h"
#include "inferx/models/causal/decoder_stack.h"
#include "inferx/models/causal/weight_mapping.h"
#include "inferx/models/checkpoint.h"
#include "inferx/models/model.h"
#include "inferx/ops/execution_context.h"

namespace inferx::causal {

/// \brief Vocabulary projection over selected hidden-state rows.
struct LanguageModelHead {
  /// \brief Projects with `w`, a [vocab, hidden] matrix that may alias the
  ///        token embedding.
  explicit LanguageModelHead(Tensor w) : weight(std::move(w)) {}

  int capacity = 0;  ///< Stable row capacity for graph replay.
  Tensor weight;  ///< [vocab, hidden]; may alias the token embedding.

  /// \brief Gathers `rows`, then projects them to [rows, vocab] logits.
  /// The row buffer and logits borrow workspace allocated on first use.
  StatusOr<Tensor> Forward(const Tensor& hidden, const Tensor& rows,
                           ops::ExecutionContext& ctx);

 private:
  bool workspace_ready_ = false;
  std::optional<Tensor> rows_, logits_;
};

/// \brief Causal language model over the shared dense decoder stack.
class CausalLM final : public Model {
 public:
  CausalLM(DecoderStack decoder, LanguageModelHead head, int max_seqs);

  bool SupportsCudaGraphs() const override { return true; }
  const CheckpointConfig& config() const override { return decoder_.config(); }
  std::vector<LayerStateSpec> StateRequirements() const override {
    return decoder_.StateRequirements();
  }
  StatusOr<Tensor> Forward(const ModelInput& input, ModelState& state,
                           ops::ExecutionContext& ctx) override;

 private:
  DecoderStack decoder_;
  LanguageModelHead head_;
  int max_seqs_;
};

/// \brief Validates, maps, and assembles a CausalLM from an opened
///        checkpoint: the single build entry the families call.
///
/// Rejects configurations with no executable implementation (MoE, recurrent
/// mixers, gated or biased projections) before loading any weight. `config`
/// carries TOTAL head counts; `parallel` both shards the executed stack to
/// this rank's heads and selects its row slices of the packed QKV weights.
StatusOr<std::unique_ptr<Model>> BuildCausalLM(models::LoadedCheckpoint& checkpoint,
                                               DecoderConfig config,
                                               const CheckpointLayout& layout, DeviceId device,
                                               int max_tokens, int max_seqs,
                                               const ParallelConfig& parallel = {});

}  // namespace inferx::causal

#endif  // INFERX_MODELS_CAUSAL_CAUSAL_LM_H_
