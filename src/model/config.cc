#include "inferx/model/config.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "absl/strings/str_cat.h"
#include "inferx/support/json.h"

namespace inferx::model {
namespace {

StatusOr<std::string> ReadFile(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return NotFoundError("cannot open '", path, "': ", std::strerror(errno));
  }

  std::string out;
  char buf[65536];

  for (;;) {
    const ssize_t n = ::read(fd, buf, sizeof(buf));
    if (n < 0) {
      const Status s =
          InternalError("cannot read '", path, "': ", std::strerror(errno));
      ::close(fd);
      return s;
    }
    if (n == 0) break;
    out.append(buf, static_cast<size_t>(n));
  }

  ::close(fd);
  return out;
}

StatusOr<Architecture> ParseArchitecture(std::string_view name) {
  if (name == "Qwen2ForCausalLM") return Architecture::kQwen2;
  if (name == "LlamaForCausalLM") return Architecture::kLlama;
  if (name == "Qwen2MoeForCausalLM") return Architecture::kQwen2Moe;
  if (name == "GptOssForCausalLM") return Architecture::kGptOss;

  // Rejected rather than defaulted. Running an unread architecture through the
  // Llama path produces output that looks like text and is wrong, which is the
  // most expensive kind of failure to notice.
  return UnimplementedError(
      "unsupported architecture '", name,
      "'; known: Qwen2ForCausalLM, LlamaForCausalLM, "
      "Qwen2MoeForCausalLM, GptOssForCausalLM");
}

StatusOr<DataType> ParseTorchDtype(std::string_view name) {
  if (name == "bfloat16") return DataType::kBFloat16;
  if (name == "float16") return DataType::kFloat16;
  if (name == "float32") return DataType::kFloat;

  return InvalidArgumentError("unsupported torch_dtype '", name, "'");
}

}  // namespace

const char* ArchitectureName(Architecture arch) {
  switch (arch) {
    case Architecture::kQwen2: return "Qwen2ForCausalLM";
    case Architecture::kLlama: return "LlamaForCausalLM";
    case Architecture::kQwen2Moe: return "Qwen2MoeForCausalLM";
    case Architecture::kGptOss: return "GptOssForCausalLM";
  }
  return "?";
}

Status ModelConfig::Validate() const {
  if (hidden_size <= 0 || intermediate_size <= 0 || num_hidden_layers <= 0 ||
      num_attention_heads <= 0 || num_key_value_heads <= 0 || vocab_size <= 0 ||
      head_dim <= 0) {
    return InvalidArgumentError("config has a non-positive dimension: ",
                                ToString());
  }

  if (num_attention_heads % num_key_value_heads != 0) {
    return InvalidArgumentError(
        "num_attention_heads (", num_attention_heads,
        ") is not a multiple of num_key_value_heads (", num_key_value_heads,
        "), so the GQA group size is not an integer");
  }

  if (num_key_value_heads > num_attention_heads) {
    return InvalidArgumentError("num_key_value_heads (", num_key_value_heads,
                                ") exceeds num_attention_heads (",
                                num_attention_heads, ")");
  }

  // head_dim need not divide hidden_size in general -- some models decouple
  // them -- but the Q projection width must be consistent with what it is.
  if (q_dim() <= 0 || kv_dim() <= 0) {
    return InvalidArgumentError("degenerate projection widths: q_dim=", q_dim(),
                                " kv_dim=", kv_dim());
  }

  if (rms_norm_eps <= 0.0) {
    return InvalidArgumentError("rms_norm_eps must be positive, got ",
                                rms_norm_eps);
  }

  if (rope_theta <= 0.0) {
    return InvalidArgumentError("rope_theta must be positive, got ",
                                rope_theta);
  }

  // RoPE rotates pairs across the two halves of a head, so an odd head_dim has
  // no meaning for it.
  if (head_dim % 2 != 0) {
    return InvalidArgumentError("head_dim must be even for RoPE, got ",
                                head_dim);
  }

  if (num_experts < 0 || num_experts_per_tok < 0) {
    return InvalidArgumentError("negative expert counts: num_experts=",
                                num_experts, " num_experts_per_tok=",
                                num_experts_per_tok);
  }

  // A router that has to pick more experts than exist cannot; and a model
  // declaring experts nobody routes to would silently run as dense, which is
  // the failure that produces plausible nonsense rather than an error.
  if (num_experts > 0) {
    if (num_experts_per_tok <= 0) {
      return InvalidArgumentError("num_experts is ", num_experts,
                                  " but num_experts_per_tok is ",
                                  num_experts_per_tok);
    }
    if (num_experts_per_tok > num_experts) {
      return InvalidArgumentError("num_experts_per_tok (", num_experts_per_tok,
                                  ") exceeds num_experts (", num_experts, ")");
    }
    if (moe_intermediate_size <= 0) {
      return InvalidArgumentError("moe_intermediate_size must be positive for "
                                  "a MoE model, got ", moe_intermediate_size);
    }
  } else if (num_experts_per_tok > 0) {
    return InvalidArgumentError("num_experts_per_tok is ", num_experts_per_tok,
                                " but the model declares no experts");
  }

  if (architecture == Architecture::kGptOss) {
    if (!is_moe()) {
      return InvalidArgumentError(
          "architecture is GptOssForCausalLM but num_local_experts is 0");
    }
    if (!layer_is_sliding.empty() &&
        static_cast<int64_t>(layer_is_sliding.size()) != num_hidden_layers) {
      return InvalidArgumentError(
          "layer_types lists ", layer_is_sliding.size(),
          " layers but the model has ", num_hidden_layers);
    }
  }

  if (architecture == Architecture::kQwen2Moe && !is_moe()) {
    return InvalidArgumentError(
        "architecture is Qwen2MoeForCausalLM but num_experts is 0");
  }

  if (is_mla()) {
    if (qk_nope_head_dim <= 0 || qk_rope_head_dim <= 0 || v_head_dim <= 0) {
      return InvalidArgumentError(
          "MLA needs positive qk_nope_head_dim, qk_rope_head_dim and "
          "v_head_dim; got ", qk_nope_head_dim, ", ", qk_rope_head_dim, ", ",
          v_head_dim);
    }

    // The decoupled RoPE key is rotated in half-split pairs like any other.
    if (qk_rope_head_dim % 2 != 0) {
      return InvalidArgumentError("qk_rope_head_dim must be even for RoPE, got ",
                                  qk_rope_head_dim);
    }

    // MLA reconstructs K and V from the latent, so a latent narrower than one
    // head's worth of either would be lossy by construction rather than by
    // design. Every published MLA config has it far wider than both.
    if (kv_lora_rank < qk_nope_head_dim || kv_lora_rank < v_head_dim) {
      return InvalidArgumentError("kv_lora_rank (", kv_lora_rank,
                                  ") is narrower than a single reconstructed "
                                  "head (nope ", qk_nope_head_dim, ", v ",
                                  v_head_dim, ")");
    }
  } else if (qk_nope_head_dim > 0 || qk_rope_head_dim > 0 || v_head_dim > 0) {
    return InvalidArgumentError(
        "MLA head dimensions are set but kv_lora_rank is 0, so there is no "
        "latent to reconstruct them from");
  }

  return OkStatus();
}

std::string ModelConfig::ToString() const {
  std::string moe;
  if (is_moe()) {
    moe = absl::StrCat(" experts=", num_experts, "/", num_experts_per_tok,
                       " moe_inter=", moe_intermediate_size,
                       shared_expert_intermediate_size > 0
                           ? absl::StrCat(" shared=",
                                          shared_expert_intermediate_size)
                           : "");
  }

  return absl::StrCat(
      ArchitectureName(architecture), "{hidden=", hidden_size,
      " inter=", intermediate_size, " layers=", num_hidden_layers,
      " heads=", num_attention_heads, " kv_heads=", num_key_value_heads,
      " head_dim=", head_dim, " vocab=", vocab_size,
      " rope_theta=", rope_theta, " eps=", rms_norm_eps,
      " tied=", tie_word_embeddings ? "yes" : "no",
      " attn_bias=", attention_bias ? "yes" : "no",
      " dtype=", DataTypeName(weight_dtype), moe, "}");
}

StatusOr<ModelConfig> ModelConfig::FromJson(std::string_view json_text) {
  INFERX_ASSIGN_OR_RETURN(const JsonValue root, ParseJson(json_text));

  if (!root.IsObject()) {
    return InvalidArgumentError("config.json is not a JSON object");
  }

  ModelConfig c;

  // "architectures" is an array with one entry in every checkpoint we have
  // seen. More than one would mean something we do not understand.
  const JsonValue* archs = root.Find("architectures");
  if (archs == nullptr) {
    return InvalidArgumentError("config.json has no 'architectures'");
  }
  INFERX_ASSIGN_OR_RETURN(const auto* arch_list, archs->AsArray());
  if (arch_list->size() != 1) {
    return InvalidArgumentError("expected exactly one architecture, got ",
                                arch_list->size());
  }
  INFERX_ASSIGN_OR_RETURN(const std::string_view arch_name,
                          (*arch_list)[0].AsString());
  INFERX_ASSIGN_OR_RETURN(c.architecture, ParseArchitecture(arch_name));

  INFERX_ASSIGN_OR_RETURN(c.hidden_size, root.RequiredInt("hidden_size"));
  INFERX_ASSIGN_OR_RETURN(c.intermediate_size,
                          root.RequiredInt("intermediate_size"));
  INFERX_ASSIGN_OR_RETURN(c.num_hidden_layers,
                          root.RequiredInt("num_hidden_layers"));
  INFERX_ASSIGN_OR_RETURN(c.num_attention_heads,
                          root.RequiredInt("num_attention_heads"));
  INFERX_ASSIGN_OR_RETURN(c.vocab_size, root.RequiredInt("vocab_size"));

  // MHA when absent: a config that does not mention KV heads has one per Q
  // head, which is what the field's absence meant before GQA existed.
  INFERX_ASSIGN_OR_RETURN(
      c.num_key_value_heads,
      root.OptionalInt("num_key_value_heads", c.num_attention_heads));

  // Stated only when it is not hidden_size / heads. Deriving it unconditionally
  // would be wrong for models that decouple the two.
  INFERX_ASSIGN_OR_RETURN(
      c.head_dim,
      root.OptionalInt("head_dim", c.num_attention_heads > 0
                                       ? c.hidden_size / c.num_attention_heads
                                       : 0));

  INFERX_ASSIGN_OR_RETURN(c.max_position_embeddings,
                          root.OptionalInt("max_position_embeddings", 0));

  const JsonValue* eps = root.Find("rms_norm_eps");
  if (eps != nullptr) {
    INFERX_ASSIGN_OR_RETURN(c.rms_norm_eps, eps->AsDouble());
  }

  const JsonValue* theta = root.Find("rope_theta");
  if (theta != nullptr) {
    INFERX_ASSIGN_OR_RETURN(c.rope_theta, theta->AsDouble());
  }

  INFERX_ASSIGN_OR_RETURN(c.tie_word_embeddings,
                          root.OptionalBool("tie_word_embeddings", false));

  // Qwen2 biases Q/K/V and does not say so in config.json -- it is a property
  // of the architecture, not a field. Llama has an explicit attention_bias that
  // is false in every checkpoint we know of. Reading the flag when present and
  // falling back to the architecture's own convention covers both.
  INFERX_ASSIGN_OR_RETURN(
      c.attention_bias,
      root.OptionalBool("attention_bias",
                        c.architecture == Architecture::kQwen2));

  const JsonValue* dtype = root.Find("torch_dtype");
  if (dtype != nullptr && !dtype->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const std::string_view dtype_name,
                            dtype->AsString());
    INFERX_ASSIGN_OR_RETURN(c.weight_dtype, ParseTorchDtype(dtype_name));
  }

  // MoE. Read for every architecture rather than only the MoE ones: a dense
  // checkpoint simply has none of these keys and `num_experts` stays 0, which
  // is the dense case. Reading them unconditionally means a new MoE
  // architecture needs a line in ParseArchitecture and nothing here.
  INFERX_ASSIGN_OR_RETURN(c.num_experts, root.OptionalInt("num_experts", 0));
  INFERX_ASSIGN_OR_RETURN(c.num_experts_per_tok,
                          root.OptionalInt("num_experts_per_tok", 0));
  INFERX_ASSIGN_OR_RETURN(
      c.moe_intermediate_size,
      root.OptionalInt("moe_intermediate_size", c.intermediate_size));
  INFERX_ASSIGN_OR_RETURN(c.norm_topk_prob,
                          root.OptionalBool("norm_topk_prob", true));
  INFERX_ASSIGN_OR_RETURN(
      c.shared_expert_intermediate_size,
      root.OptionalInt("shared_expert_intermediate_size", 0));

  // gpt-oss. `num_local_experts` is its spelling of `num_experts`, and the
  // rest describe things no other architecture here has.
  if (c.num_experts == 0) {
    INFERX_ASSIGN_OR_RETURN(c.num_experts,
                            root.OptionalInt("num_local_experts", 0));
    if (c.num_experts > 0 && c.moe_intermediate_size == c.intermediate_size) {
      // gpt-oss gives every expert the model's `intermediate_size` and has no
      // separate `moe_intermediate_size`, which the fallback above already
      // produced. Nothing to do -- but the case is worth naming, because a
      // silent fallback to the *dense* width is exactly the sort of thing that
      // would make each expert 4x too wide on a checkpoint that did differ.
    }
  }

  INFERX_ASSIGN_OR_RETURN(c.sliding_window,
                          root.OptionalInt("sliding_window", 0));

  const JsonValue* swiglu_limit = root.Find("swiglu_limit");
  if (swiglu_limit != nullptr && !swiglu_limit->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(c.swiglu_limit, swiglu_limit->AsDouble());
  }

  // Per-layer attention types. Read as a list rather than as "every other one"
  // because the alternation is this checkpoint's, not the architecture's.
  if (const JsonValue* types = root.Find("layer_types");
      types != nullptr && !types->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const auto* list, types->AsArray());

    for (const JsonValue& entry : *list) {
      INFERX_ASSIGN_OR_RETURN(const std::string_view name, entry.AsString());

      if (name == "sliding_attention") {
        c.layer_is_sliding.push_back(true);
      } else if (name == "full_attention") {
        c.layer_is_sliding.push_back(false);
      } else {
        return UnimplementedError("unknown layer_type '", name, "'");
      }
    }
  }

  // YaRN. Only the `yarn` rope_type is implemented; the others (linear,
  // dynamic, longrope, llama3) each change the frequencies differently, and
  // running one as another produces a model with no long-range coherence.
  if (const JsonValue* rope = root.Find("rope_scaling");
      rope != nullptr && !rope->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const std::string_view rope_type,
                            rope->RequiredString("rope_type"));

    if (rope_type != "yarn") {
      return UnimplementedError("rope_scaling.rope_type '", rope_type,
                                "' is not implemented; only 'yarn' is");
    }

    INFERX_ASSIGN_OR_RETURN(const double factor, rope->RequiredDouble("factor"));
    c.yarn_factor = factor;

    if (const JsonValue* v = rope->Find("beta_fast"); v != nullptr) {
      INFERX_ASSIGN_OR_RETURN(c.yarn_beta_fast, v->AsDouble());
    }
    if (const JsonValue* v = rope->Find("beta_slow"); v != nullptr) {
      INFERX_ASSIGN_OR_RETURN(c.yarn_beta_slow, v->AsDouble());
    }
    INFERX_ASSIGN_OR_RETURN(
        c.yarn_original_max_position,
        rope->OptionalInt("original_max_position_embeddings",
                          c.max_position_embeddings));
    INFERX_ASSIGN_OR_RETURN(c.yarn_truncate,
                            rope->OptionalBool("truncate", true));
  }

  // MLA, read the same way and for the same reason: absent keys mean GQA.
  INFERX_ASSIGN_OR_RETURN(c.kv_lora_rank, root.OptionalInt("kv_lora_rank", 0));
  INFERX_ASSIGN_OR_RETURN(c.q_lora_rank, root.OptionalInt("q_lora_rank", 0));
  INFERX_ASSIGN_OR_RETURN(c.qk_nope_head_dim,
                          root.OptionalInt("qk_nope_head_dim", 0));
  INFERX_ASSIGN_OR_RETURN(c.qk_rope_head_dim,
                          root.OptionalInt("qk_rope_head_dim", 0));
  INFERX_ASSIGN_OR_RETURN(c.v_head_dim, root.OptionalInt("v_head_dim", 0));

  INFERX_RETURN_IF_ERROR(c.Validate());

  return c;
}

StatusOr<ModelConfig> ModelConfig::FromDirectory(std::string_view dir) {
  std::string path(dir);
  if (!path.empty() && path.back() != '/') path.push_back('/');
  path.append("config.json");

  INFERX_ASSIGN_OR_RETURN(const std::string text, ReadFile(path));

  auto config = FromJson(text);
  if (!config.ok()) {
    return InvalidArgumentError("'", path, "': ", config.status().message());
  }

  return config;
}

}  // namespace inferx::model
