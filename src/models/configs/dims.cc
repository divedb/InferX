#include "inferx/models/configs/dims.h"

namespace inferx::models {

StatusOr<ops::Activation> ParseActivation(const ConfigView& view) {
  // gpt-oss states "silu" but clamps and shifts via swiglu_limit.
  INFERX_ASSIGN_OR_RETURN(const double limit, view.Float("swiglu_limit", 0.0));
  if (limit > 0.0) return ops::Activation::kSiluOai;
  INFERX_ASSIGN_OR_RETURN(const std::string act, view.Str("hidden_act", "silu"));
  if (act == "silu" || act == "swish") return ops::Activation::kSilu;
  if (act == "gelu_pytorch_tanh" || act == "geglu") return ops::Activation::kGeluTanh;
  return UnimplementedError("unsupported feed-forward activation: ", act);
}

StatusOr<ModelDims> ModelDims::Parse(const ConfigView& view) {
  ModelDims dims;
  INFERX_ASSIGN_OR_RETURN(dims.hidden_size, view.RequireInt("hidden_size"));
  INFERX_ASSIGN_OR_RETURN(dims.num_hidden_layers, view.RequireInt("num_hidden_layers"));
  INFERX_ASSIGN_OR_RETURN(dims.num_attention_heads, view.RequireInt("num_attention_heads"));
  INFERX_ASSIGN_OR_RETURN(dims.vocab_size, view.RequireInt("vocab_size"));
  INFERX_ASSIGN_OR_RETURN(dims.intermediate_size, view.Int("intermediate_size", 0));
  INFERX_ASSIGN_OR_RETURN(dims.num_key_value_heads,
                          view.Int("num_key_value_heads", dims.num_attention_heads));
  INFERX_ASSIGN_OR_RETURN(dims.head_dim, view.Int("head_dim", 0));
  if (dims.head_dim == 0 && dims.num_attention_heads > 0) {
    dims.head_dim = dims.hidden_size / dims.num_attention_heads;
  }
  INFERX_ASSIGN_OR_RETURN(dims.max_position_embeddings,
                          view.Int("max_position_embeddings", 2048));
  INFERX_ASSIGN_OR_RETURN(dims.rms_norm_eps, view.Float("rms_norm_eps", 1e-5));
  INFERX_ASSIGN_OR_RETURN(dims.activation, ParseActivation(view));
  INFERX_ASSIGN_OR_RETURN(const double limit, view.Float("swiglu_limit", 0.0));
  dims.swiglu_limit = static_cast<float>(limit);
  INFERX_ASSIGN_OR_RETURN(dims.attention_bias, view.Bool("attention_bias", false));
  INFERX_ASSIGN_OR_RETURN(dims.tie_word_embeddings, view.Bool("tie_word_embeddings", false));
  INFERX_ASSIGN_OR_RETURN(dims.bos_token_id, view.Int("bos_token_id", -1));
  // eos_token_id may be a scalar or a list; generation stops on the first.
  INFERX_ASSIGN_OR_RETURN(const auto eos_ids, view.IntList("eos_token_id"));

  if (!eos_ids.empty()) dims.eos_token_id = eos_ids.front();

  INFERX_RETURN_IF_ERROR(dims.Validate());

  return dims;
}

Status ModelDims::Validate() const {
  if (hidden_size <= 0 || num_hidden_layers <= 0 || num_attention_heads <= 0 ||
      vocab_size <= 0) {
    return InvalidArgumentError("config.json is missing required model dimensions");
  }
  if (num_key_value_heads <= 0 || head_dim <= 0) {
    return InvalidArgumentError("attention dimensions must be positive");
  }
  if (num_attention_heads % num_key_value_heads != 0) {
    return InvalidArgumentError("num_attention_heads must be divisible by num_key_value_heads");
  }
  return OkStatus();
}

}  // namespace inferx::models
