#ifndef INFERX_CONFIG_SPECULATIVE_CONFIG_H_
#define INFERX_CONFIG_SPECULATIVE_CONFIG_H_

#include <string>

#include "inferx/core/status.h"

namespace inferx {

/// \brief Speculative decoding method (vLLM: speculative_config.method).
enum class SpeculativeMethod {
  kNone,        ///< Speculative decoding disabled.
  kNgram,       ///< Prompt-lookup drafting; needs no draft model.
  kDraftModel,  ///< Separate smaller model proposes tokens.
  kMedusa,
  kEagle,
  kEagle3,
  kMtp,  ///< Multi-token-prediction heads built into the target model.
};

/// \brief Returns the CLI spelling ("none", "ngram", "draft_model", ...).
const char* ToString(SpeculativeMethod method);

/// \brief Parses a CLI spelling into `*out`; InvalidArgument if unknown.
Status ParseSpeculativeMethod(const std::string& text, SpeculativeMethod* out);

/// \brief Speculative decoding settings.
struct SpeculativeConfig {
  SpeculativeMethod method = SpeculativeMethod::kNone;
  /// Draft checkpoint path or id. Required for kDraftModel, kMedusa,
  /// kEagle and kEagle3; unused for kNgram and kMtp.
  std::string model;
  /// Tokens proposed per step.
  int num_speculative_tokens = 0;
  /// Tensor-parallel size of the draft model; 0 means "same as target".
  int draft_tensor_parallel_size = 0;
  /// Ngram window bounds (kNgram only).
  int prompt_lookup_max = 5;
  int prompt_lookup_min = 1;

  bool Enabled() const { return method != SpeculativeMethod::kNone; }

  /// True for methods that load a separate draft checkpoint.
  bool RequiresDraftModel() const;

  /// Checks the fields only when speculative decoding is enabled.
  Status Validate() const;
};

}  // namespace inferx

#endif  // INFERX_CONFIG_SPECULATIVE_CONFIG_H_