# inferx_models -- reusable transformer components under models/components
# (config + weights + execution), typed decoder composition under models/causal,
# checkpoint mapping under models/loading, and family factories selected by
# checkpoint identity. Explicit factory references retain families in static links.
inferx_add_library(
  NAME models
  SRCS
    model.cc
    model_registry.cc
    families/llama.cc
    families/qwen3.cc
    families/dense_families.cc
    families/moe_families.cc
    causal/config_parser.cc
    checkpoint_config.cc
    checkpoint.cc
    safe_tensors_reader.cc
    safe_tensors_view.cc
    components/attention.cc
    components/decoder_layer.cc
    components/mlp.cc
    components/moe.cc
    components/mla.cc
    components/gdn.cc
    components/parallel_linear.cc
    components/qkv_linear.cc
    causal/causal_lm.cc
    causal/decoder_stack.cc
    causal/decoder_config.cc
    loading/weight_loader.cc
  DEPS
    inferx::core
    inferx::cache
    inferx::ops
    inferx::boost
    nlohmann_json::nlohmann_json
  PRIVATE_DEPS
    absl::flat_hash_map
    absl::span
    absl::status
    absl::statusor
    absl::strings
)
