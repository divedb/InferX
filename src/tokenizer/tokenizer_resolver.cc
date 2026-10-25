#include "inferx/tokenizer/tokenizer_resolver.h"

#include "inferx/tokenizer/hf_tokenizer.h"

namespace inferx {

StatusOr<std::shared_ptr<Tokenizer>> ResolveTokenizer(const ModelConfig& config) {
  // vLLM's tokenizer resolution: tokenizer_dir when provided, otherwise the
  // model directory; both must hold a fast-tokenizer tokenizer.json.
  return HFTokenizer::Load(config.ResolvedTokenizerDir());
}

}  // namespace inferx
