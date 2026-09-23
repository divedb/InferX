/// \file
/// \brief CausalLM: a Decoder backbone plus a language-model head; the Model
/// implementation served for token generation.

#ifndef INFERX_MODELS_LM_CAUSAL_LM_H_
#define INFERX_MODELS_LM_CAUSAL_LM_H_

#include <memory>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/lm/stack.h"
#include "inferx/models/model.h"
#include "inferx/ops/execution_context.h"

namespace inferx::lm {

/// \brief Vocabulary projection over selected hidden-state rows.
struct LanguageModelHead {
  int capacity = 0;  ///< Stable row capacity for graph replay.
  Tensor weight;  ///< [vocab, hidden]; may alias the token embedding.

  /// \brief Gathers `rows`, then projects them to [rows, vocab] logits.
  /// The row buffer and logits borrow workspace allocated on first use.
  StatusOr<Tensor> Forward(const Tensor& hidden, const Tensor& rows,
                           ops::ExecutionContext& ctx);

 private:
  bool workspace_ready_ = false;
  Tensor rows_, logits_;
};

/// \brief Causal language model over any Decoder backbone.
class CausalLM final : public Model {
 public:
  CausalLM(std::unique_ptr<Decoder> decoder, LanguageModelHead head, int max_seqs);

  bool SupportsCudaGraphs() const override { return true; }
  const CheckpointConfig& config() const override { return decoder_->config(); }
  std::vector<LayerStateSpec> StateRequirements() const override {
    return decoder_->StateRequirements();
  }
  StatusOr<Tensor> Forward(const ModelInput& input, ModelState& state,
                           ops::ExecutionContext& ctx) override;

 private:
  std::unique_ptr<Decoder> decoder_;
  LanguageModelHead head_;
  int max_seqs_;
};

}  // namespace inferx::lm

#endif  // INFERX_MODELS_LM_CAUSAL_LM_H_
