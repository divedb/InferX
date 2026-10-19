# inferx_attn -- the unified attention architecture (docs/attention.md):
# spec vocabulary, cache-spec bridge, capability selection, metadata
# builders, kernel-facing cache views, the runtime composition root, and
# the reference + FlashInfer-backed dense backends.
inferx_add_library(
  NAME attn
  SRCS
    cache_spec_bridge.cc
    capability.cc
    cache_view.cc
    fill_common.cc
    selector.cc
    runtime.cc
    backends/reference.cc
    backends/flash_dense.cc
  DEPS
    inferx::core
    inferx::cache_manager
    inferx::ops
)
