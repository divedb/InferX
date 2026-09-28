/// \file
/// \brief CausalLM: a DecoderStack backbone plus a language-model head; the
/// Model implementation served for token generation.

#ifndef INFERX_MODELS_CAUSAL_CAUSAL_LM_H_
#define INFERX_MODELS_CAUSAL_CAUSAL_LM_H_

#include <memory>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/causal/decoder_stack.h"
#include "inferx/models/checkpoint.h"
#include "inferx/models/loading/weight_loader.h"
#include "inferx/models/model.h"
#include "inferx/ops/execution_context.h"

namespace inferx::causal {

/// \brief Vocabulary projection over selected hidden-state rows.
struct LanguageModelHead {
  /// \brief Projects with `w`, a [vocab, hidden] matrix that may alias the
  ///        token embedding.
  explicit LanguageModelHead(Tensor w) : weight(std::move(w)) {}

  int capacity = 0;  ///< Stable row capacity for graph replay.
  Tensor weight;     ///< [vocab, hidden]; may alias the token embedding.

  /// \brief Gathers `rows`, then projects them to [rows, vocab] logits.
  /// The row buffer and logits borrow workspace allocated on first use.
  StatusOr<Tensor> Forward(const Tensor& hidden, const Tensor& rows,
                           ops::ExecutionContext& ctx);

 private:
  bool workspace_ready_ = false;
  std::optional<Tensor> rows_, logits_;
};

/// Validated and loaded common state, before selecting a typed executor.
struct PreparedCausalLM {
  DecoderConfig config;
  DecoderWeights weights;
  Tensor head;
};

StatusOr<PreparedCausalLM> PrepareCausalLM(models::LoadedCheckpoint& checkpoint,
                                           DecoderConfig config,
                                           const models::WeightNames& names,
                                           const models::WeightLayout& layout, DeviceId device,
                                           int max_tokens, int max_seqs,
                                           const ParallelConfig& parallel);

/// The only virtual execution boundary is Model. Layers and components are
/// concrete types selected by Traits.
template <ModelTraits Traits>
class CausalLM final : public Model {
 public:
  CausalLM(PreparedCausalLM prepared, int max_tokens, int max_seqs)
      : decoder_(std::move(prepared.config), std::move(prepared.weights), max_tokens),
        head_(std::move(prepared.head)),
        max_seqs_(max_seqs) {
    head_.capacity = max_seqs;
  }

  bool SupportsCudaGraphs() const override { return true; }
  const CheckpointConfig& config() const override { return decoder_.config(); }
  std::vector<LayerStateSpec> StateRequirements() const override {
    return decoder_.StateRequirements();
  }
  StatusOr<Tensor> Forward(const ModelInput& input, ModelState& state,
                           ops::ExecutionContext& ctx) override {
    if (input.attention.num_seqs <= 0 || input.attention.num_seqs > max_seqs_ ||
        input.logit_rows.Rank() != 1 || input.logit_rows.Numel() != input.attention.num_seqs ||
        input.logit_rows.GetDataType() != DataType::kInt32 ||
        input.logit_rows.Device() != ctx.device()) {
      return InvalidArgumentError("invalid requested language-model output rows");
    }
    DecoderInput decoder_input{input.token_ids, {}, input.attention};
    INFERX_ASSIGN_OR_RETURN(Tensor hidden, decoder_.Forward(decoder_input, state, ctx));
    return head_.Forward(hidden, input.logit_rows, ctx);
  }

 private:
  DecoderStack<Traits> decoder_;
  LanguageModelHead head_;
  int max_seqs_;
};

}  // namespace inferx::causal

#endif  // INFERX_MODELS_CAUSAL_CAUSAL_LM_H_
