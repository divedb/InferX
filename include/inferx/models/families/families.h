#pragma once

#include "inferx/models/model_registry.h"

namespace inferx::families {

// Explicit references keep factories reachable in static-library builds.
Family Llama();
Family Qwen3();
Family Qwen3Moe();
Family Qwen3Next();

}  // namespace inferx::families
