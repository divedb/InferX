#pragma once

#include "inferx/models/lm/stack.h"

namespace inferx::qwen3 {

// Translate family configuration into reusable decoder components.
StatusOr<lm::DecoderConfig> ParseConfig(const std::string& json);
StatusOr<std::unique_ptr<Model>> Load(const std::string& directory, DeviceId device,
                                      int max_tokens, int max_seqs,
                                      ops::AttentionBackend backend = ops::AttentionBackend::kFlashInfer);

}  // namespace inferx::qwen3
