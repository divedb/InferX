#pragma once

#include <memory>
#include <string>

#include "inferx/config/model_config.h"
#include "inferx/core/status.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx {

/// \brief Loads the tokenizer associated with a model configuration.
///
/// Mirrors vLLM's tokenizer resolution: the tokenizer_dir is used when
/// provided, otherwise the model_dir is searched. The directory must hold a
/// HuggingFace fast-tokenizer `tokenizer.json`; `tokenizer_config.json` and
/// `special_tokens_map.json` are merged in when present (special tokens,
/// chat template, model_max_length, truncation side).
///
/// \param config Model configuration containing tokenizer paths.
/// \return       The loaded tokenizer, or an error status.
StatusOr<std::shared_ptr<Tokenizer>> ResolveTokenizer(const ModelConfig& config);

}  // namespace inferx
