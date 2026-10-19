# inferx_models -- the model layer: family-specific architectures (Qwen3,
# Qwen3-MoE, Llama) built from the reusable modules under models/layers,
# plus the architecture registry, checkpoint loading, and the rank-local
# ModelRunner that binds a scheduler step to a forward pass.
inferx_add_library(
  NAME models
  SRCS
    model.cc
    scratch_arena.cc
    layers/embedding.cc
    layers/fused_moe.cc
    layers/linear.cc
    layers/mlp.cc
    layers/rms_norm.cc
    safetensors.cc
    loading/safe_tensors_view.cc
    loading/safe_tensors_reader.cc
    weight_loader.cc
    config_utils.cc
    registry.cc
    model_runner.cc
    qwen3/qwen3_config.cc
    qwen3/qwen3_attention.cc
    qwen3/qwen3_mlp.cc
    qwen3/qwen3_decoder_layer.cc
    qwen3/qwen3_model.cc
    qwen3/qwen3_causal_lm.cc
    qwen3_moe/qwen3_moe_config.cc
    qwen3_moe/qwen3_moe.cc
    qwen3_moe/qwen3_moe_decoder_layer.cc
    qwen3_moe/qwen3_moe_model.cc
    qwen3_moe/qwen3_moe_causal_lm.cc
    llama/llama_config.cc
    llama/llama_attention.cc
    llama/llama_mlp.cc
    llama/llama_decoder_layer.cc
    llama/llama_model.cc
    llama/llama_causal_lm.cc
  DEPS
    inferx::core
    inferx::config
    inferx::ops
    inferx::attn
    inferx::cache_manager
    inferx::comm
    inferx::util
    inferx::sampling
    inferx::boost
    nlohmann_json::nlohmann_json
    absl::flat_hash_map
    absl::span
)
