#pragma once

#include "inferx/models/model_registry.h"

namespace inferx::families {

// Explicit references keep factories reachable in static-library builds.
Family Llama();
Family Qwen2();
Family Qwen3();
Family Mistral();
Family Gemma3();
Family Mixtral();
Family Qwen3Moe();
Family Qwen3Next();
Family GptOss();
Family DeepseekV3();

}  // namespace inferx::families
