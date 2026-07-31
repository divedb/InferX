#include "inferx/model/config.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "absl/strings/str_cat.h"
#include "inferx/common/json.h"

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

  // Rejected rather than defaulted. Running an unread architecture through the
  // Llama path produces output that looks like text and is wrong, which is the
  // most expensive kind of failure to notice.
  return UnimplementedError("unsupported architecture '", name,
                            "'; known: Qwen2ForCausalLM, LlamaForCausalLM");
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

  return OkStatus();
}

std::string ModelConfig::ToString() const {
  return absl::StrCat(
      ArchitectureName(architecture), "{hidden=", hidden_size,
      " inter=", intermediate_size, " layers=", num_hidden_layers,
      " heads=", num_attention_heads, " kv_heads=", num_key_value_heads,
      " head_dim=", head_dim, " vocab=", vocab_size,
      " rope_theta=", rope_theta, " eps=", rms_norm_eps,
      " tied=", tie_word_embeddings ? "yes" : "no",
      " attn_bias=", attention_bias ? "yes" : "no",
      " dtype=", DataTypeName(weight_dtype), "}");
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
