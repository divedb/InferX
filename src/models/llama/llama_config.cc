#include "inferx/models/llama/llama_config.h"

#include "inferx/models/config_utils.h"

namespace inferx {
namespace models {
namespace llama {

StatusOr<LlamaConfig> LlamaConfig::FromHf(const HfConfig& config) {
  const ConfigView& text = config.Text();
  LlamaConfig parsed;

  INFERX_ASSIGN_OR_RETURN(parsed.vocab_size, text.RequireInt("vocab_size"));
  INFERX_ASSIGN_OR_RETURN(parsed.hidden_size, text.RequireInt("hidden_size"));
  INFERX_ASSIGN_OR_RETURN(parsed.num_hidden_layers, text.RequireInt("num_hidden_layers"));
  INFERX_ASSIGN_OR_RETURN(parsed.num_attention_heads, text.RequireInt("num_attention_heads"));
  INFERX_ASSIGN_OR_RETURN(
      parsed.num_key_value_heads,
      text.Int("num_key_value_heads", parsed.num_attention_heads));
  INFERX_ASSIGN_OR_RETURN(parsed.intermediate_size, text.RequireInt("intermediate_size"));
  INFERX_ASSIGN_OR_RETURN(parsed.max_position_embeddings,
                          text.RequireInt("max_position_embeddings"));

  if (parsed.vocab_size <= 0 || parsed.hidden_size <= 0 || parsed.num_hidden_layers <= 0 ||
      parsed.num_attention_heads <= 0 || parsed.num_key_value_heads <= 0 ||
      parsed.intermediate_size <= 0 || parsed.max_position_embeddings <= 0) {
    return InvalidArgumentError("llama config has non-positive dimensions");
  }
  if (parsed.num_attention_heads % parsed.num_key_value_heads != 0) {
    return InvalidArgumentError("llama kv heads (", parsed.num_key_value_heads,
                                ") must divide attention heads (",
                                parsed.num_attention_heads, ")");
  }

  // Most Llama checkpoints derive head_dim; some declare it (3.1 70B).
  INFERX_ASSIGN_OR_RETURN(
      parsed.head_dim,
      text.Int("head_dim", parsed.hidden_size / parsed.num_attention_heads));
  if (parsed.head_dim <= 0 || parsed.head_dim % 2 != 0) {
    return InvalidArgumentError("llama head_dim must be positive and even, got ",
                                parsed.head_dim);
  }

  INFERX_ASSIGN_OR_RETURN(const double eps, text.Float("rms_norm_eps", 1e-5));
  parsed.rms_norm_eps = static_cast<float>(eps);
  INFERX_ASSIGN_OR_RETURN(parsed.tie_word_embeddings,
                          text.Bool("tie_word_embeddings", false));

  INFERX_ASSIGN_OR_RETURN(auto eos, text.IntList("eos_token_id"));
  for (const int64_t id : eos) parsed.eos_token_ids.push_back(static_cast<TokenId>(id));

  INFERX_ASSIGN_OR_RETURN(const bool use_window, text.Bool("use_sliding_window", false));
  if (use_window) {
    INFERX_ASSIGN_OR_RETURN(parsed.sliding_window, text.RequireInt("sliding_window"));
    return UnimplementedError("llama sliding-window attention is not supported");
  }

  INFERX_ASSIGN_OR_RETURN(parsed.rope, ParseRope(text, parsed.head_dim, 500'000.0));
  return parsed;
}

}  // namespace llama
}  // namespace models
}  // namespace inferx
