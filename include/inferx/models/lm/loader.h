/// \file
/// \brief Checkpoint loading for the generic stack: maps safetensors names
/// to components through a per-family name layout, then assembles a CausalLM.

#ifndef INFERX_MODELS_LM_LOADER_H_
#define INFERX_MODELS_LM_LOADER_H_

#include <memory>
#include <string>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/models/lm/stack.h"

namespace inferx::lm {

/// \brief Checkpoint names belong to the architecture adapter, not the
///        components. This is the Llama-style convention shared by llama,
///        qwen, mistral, phi, internlm, and friends.
struct CheckpointLayout {
  std::string backbone_prefix = "model.";
  std::string head_name = "lm_head.weight";
  std::string attention_name = "self_attn.";
  std::string feed_forward_name = "mlp.";
};

/// \brief Loads supported component layouts with shape checks before device
///        upload, then assembles a CausalLM. Recurrent projection unpacking
///        remains architecture-specific and unsupported.
StatusOr<std::unique_ptr<Model>> LoadCausalLM(const std::string& directory,
                                              const DecoderConfig& config,
                                              const CheckpointLayout& names,
                                              DeviceId device, int max_tokens,
                                              int max_seqs);

}  // namespace inferx::lm

#endif  // INFERX_MODELS_LM_LOADER_H_
