#include "inferx/models/qwen3/qwen3_config.h"

#include "inferx/models/config_utils.h"

namespace inferx {
namespace models {
namespace qwen3 {

namespace {

/// The view that should hold `key`: text configs keep model fields, while
/// multimodal wrappers move token ids and tie flags to the root.
const ConfigView& Pick(const HfConfig& hf, std::string_view key) {
  return hf.Text().Has(key) ? hf.Text() : hf.Root();
}

StatusOr<DataType> ParseDTypeName(const std::string& name) {
  if (name.empty() || name == "bfloat16") return DataType::kBFloat16;
  if (name == "float16") return DataType::kFloat16;
  if (name == "float32") return DataType::kFloat32;
  return InvalidArgumentError("qwen3 torch_dtype '", name, "' is not supported");
}

/// Builds the rope parameters. Newer transformers nest `rope_theta` and
/// scaling under `rope_parameters`; older ones use top-level `rope_theta`
/// plus a `rope_scaling` object.
StatusOr<ops::RotaryParams> BuildRope(const ConfigView& text, int64_t head_dim) {
  double theta = 1'000'000.0;
  INFERX_ASSIGN_OR_RETURN(std::optional<ConfigView> nested, text.Sub("rope_parameters"));
  if (nested.has_value() && nested->Has("rope_theta")) {
    INFERX_ASSIGN_OR_RETURN(theta, nested->RequireFloat("rope_theta"));
  } else {
    INFERX_ASSIGN_OR_RETURN(theta, text.Float("rope_theta", theta));
  }
  if (!(theta > 0.0)) {
    return InvalidArgumentError(text.PathOf("rope_theta"), " must be positive");
  }

  std::optional<ConfigView> scaling;
  if (nested.has_value() && (nested->Has("rope_type") || nested->Has("type") ||
                             nested->Has("factor"))) {
    scaling = nested;
  } else {
    INFERX_ASSIGN_OR_RETURN(scaling, text.Sub("rope_scaling"));
  }

  ops::RotaryParams params;
  params.rotary_dim = head_dim;
  params.theta = static_cast<float>(theta);
  INFERX_ASSIGN_OR_RETURN(params.scaling, ParseRopeScaling(scaling, head_dim, theta));
  return params;
}

}  // namespace

StatusOr<Qwen3Config> Qwen3Config::FromHf(const HfConfig& hf) {
  Qwen3Config cfg;
  const ConfigView& text = hf.Text();

  INFERX_ASSIGN_OR_RETURN(cfg.model_type, text.RequireStr("model_type"));
  // "qwen3_moe" parses through this base too: Qwen3MoeConfig::FromHf layers
  // its expert fields on top. Other families are rejected.
  if (cfg.model_type != "qwen3" && cfg.model_type != "qwen3_moe") {
    return InvalidArgumentError(text.PathOf("model_type"), " is '", cfg.model_type,
                                "', expected 'qwen3'");
  }

  // `dtype` is the newer key; `torch_dtype` the older one. Either may sit at
  // the root of a multimodal wrapper's config.
  {
    const ConfigView& dv = Pick(hf, "torch_dtype");
    std::string legacy;
    INFERX_ASSIGN_OR_RETURN(legacy, dv.Str("torch_dtype", ""));
    std::string name;
    INFERX_ASSIGN_OR_RETURN(name, Pick(hf, "dtype").Str("dtype", legacy));
    INFERX_ASSIGN_OR_RETURN(cfg.dtype, ParseDTypeName(name));
  }

  // ---- Dimensions ----
  INFERX_ASSIGN_OR_RETURN(cfg.hidden_size, text.RequireInt("hidden_size"));
  INFERX_ASSIGN_OR_RETURN(cfg.vocab_size, text.RequireInt("vocab_size"));
  INFERX_ASSIGN_OR_RETURN(cfg.num_hidden_layers, text.RequireInt("num_hidden_layers"));
  INFERX_ASSIGN_OR_RETURN(cfg.intermediate_size, text.RequireInt("intermediate_size"));
  INFERX_ASSIGN_OR_RETURN(cfg.hidden_act, text.Str("hidden_act", "silu"));
  if (cfg.hidden_act != "silu") {
    return UnimplementedError(text.PathOf("hidden_act"), " '", cfg.hidden_act,
                              "' is not supported (only 'silu')");
  }

  // ---- Attention ----
  INFERX_ASSIGN_OR_RETURN(cfg.num_attention_heads, text.RequireInt("num_attention_heads"));
  INFERX_ASSIGN_OR_RETURN(cfg.num_key_value_heads,
                          text.Int("num_key_value_heads", cfg.num_attention_heads));
  INFERX_ASSIGN_OR_RETURN(cfg.head_dim, text.Int("head_dim", 128));
  INFERX_ASSIGN_OR_RETURN(cfg.attention_bias, text.Bool("attention_bias", false));
  double eps = 0.0;
  INFERX_ASSIGN_OR_RETURN(eps, text.Float("rms_norm_eps", cfg.rms_norm_eps));
  cfg.rms_norm_eps = static_cast<float>(eps);

  // ---- Positions ----
  INFERX_ASSIGN_OR_RETURN(cfg.max_position_embeddings,
                          text.RequireInt("max_position_embeddings"));
  INFERX_ASSIGN_OR_RETURN(cfg.rotary_params, BuildRope(text, cfg.head_dim));

  // ---- Sliding window (unused by released dense Qwen3) ----
  INFERX_ASSIGN_OR_RETURN(cfg.use_sliding_window, text.Bool("use_sliding_window", false));
  if (text.Has("sliding_window")) {
    int64_t window = 0;
    INFERX_ASSIGN_OR_RETURN(window, text.Int("sliding_window", 0));
    if (window > 0) cfg.sliding_window = window;
  }
  INFERX_ASSIGN_OR_RETURN(cfg.max_window_layers, text.Int("max_window_layers", 0));
  if (cfg.use_sliding_window) {
    return UnimplementedError(text.PathOf("use_sliding_window"),
                              " = true: sliding-window attention is not supported");
  }

  // ---- Embeddings & tokens ----
  INFERX_ASSIGN_OR_RETURN(
      cfg.tie_word_embeddings,
      Pick(hf, "tie_word_embeddings").Bool("tie_word_embeddings", cfg.tie_word_embeddings));

  int64_t id = 0;
  INFERX_ASSIGN_OR_RETURN(id, Pick(hf, "bos_token_id").Int("bos_token_id", -1));
  cfg.bos_token_id = static_cast<TokenId>(id);

  std::vector<int64_t> eos;
  INFERX_ASSIGN_OR_RETURN(eos, Pick(hf, "eos_token_id").IntList("eos_token_id"));
  cfg.eos_token_ids.reserve(eos.size());
  for (int64_t e : eos) cfg.eos_token_ids.push_back(static_cast<TokenId>(e));

  // ---- Training-only ----
  double dropout = 0.0, init_range = 0.0;
  INFERX_ASSIGN_OR_RETURN(dropout, text.Float("attention_dropout", 0.0));
  INFERX_ASSIGN_OR_RETURN(init_range, text.Float("initializer_range", cfg.initializer_range));
  cfg.attention_dropout = static_cast<float>(dropout);
  cfg.initializer_range = static_cast<float>(init_range);
  INFERX_ASSIGN_OR_RETURN(cfg.use_cache, text.Bool("use_cache", true));

  // `layer_types` and other unknown keys are intentionally ignored.

  // ---- Validation ----
  const auto positive = [&](int64_t v, const char* key) -> Status {
    if (v <= 0) {
      return InvalidArgumentError(text.PathOf(key), " must be positive, got ", v);
    }
    return OkStatus();
  };
  INFERX_RETURN_IF_ERROR(positive(cfg.hidden_size, "hidden_size"));
  INFERX_RETURN_IF_ERROR(positive(cfg.vocab_size, "vocab_size"));
  INFERX_RETURN_IF_ERROR(positive(cfg.num_hidden_layers, "num_hidden_layers"));
  INFERX_RETURN_IF_ERROR(positive(cfg.intermediate_size, "intermediate_size"));
  INFERX_RETURN_IF_ERROR(positive(cfg.num_attention_heads, "num_attention_heads"));
  INFERX_RETURN_IF_ERROR(positive(cfg.num_key_value_heads, "num_key_value_heads"));
  INFERX_RETURN_IF_ERROR(positive(cfg.head_dim, "head_dim"));
  INFERX_RETURN_IF_ERROR(positive(cfg.max_position_embeddings, "max_position_embeddings"));

  if (cfg.head_dim % 2 != 0) {
    return InvalidArgumentError(text.PathOf("head_dim"), " must be even for RoPE, got ",
                                cfg.head_dim);
  }
  if (cfg.num_attention_heads % cfg.num_key_value_heads != 0) {
    return InvalidArgumentError(text.Path(), ": num_attention_heads (",
                                cfg.num_attention_heads,
                                ") must be a multiple of num_key_value_heads (",
                                cfg.num_key_value_heads, ")");
  }
  if (cfg.eos_token_ids.empty()) {
    return InvalidArgumentError(text.PathOf("eos_token_id"), " is missing or empty");
  }

  return cfg;
}

}  // namespace qwen3
}  // namespace models
}  // namespace inferx
