# Bug fixes

## 2026-10-19: flash_dense prefill-to-decode memory corruption

**Status: fixed.** The GPU differential test now runs without
`INFERX_ATTN_GPU_DIFF`. CUDA-unavailable hosts still skip it.

### Symptoms and reproduction

The causal GQA differential test reported intermittent launch errors on decode
following a 20-token prefill, including `write paged kv launch: invalid argument`,
in the WSL2/CUDA-13 environment. Standalone shapes appeared to work. Running the
original gated test also reproduced a failed prefill device-to-host output copy
and numerical failures. This was not evidence of a WSL2 or CUDA driver defect.

### Root causes

1. **Undersized test output.** The test allocated `heads * head_dim * 2` bytes
   (512 bytes), but described and wrote a 20-token output requiring 10,240 bytes.
   Both prefill writes and the subsequent output download exceeded the allocation.
   Device memory corruption can surface at a later API call, making decode's
   launch appear responsible.
2. **Incorrect physical KV page stride.** The new cache view stores each page as
   `[K plane, V plane]`. `flash_dense` correctly offset the V base by one plane,
   but the legacy CUDA wrappers advanced both bases by only one plane per page.
   Consequently V for page i overlapped K for page i+1. Fixing the test allocation
   alone exposed numerical failures; the cache stride also needed correction.

### Fix

- Allocate the test output with `Tensor::Empty` using its complete token/head/dim
  shape, and derive the download length from that tensor. Check allocation/copy
  failures before comparing outputs and retain host upload buffers until stream
  completion.
- Pass the cache's physical page stride, in elements, through the KV writer and
  attention parameters. Both scalar and vectorized KV writes honor it, as do
  FlashInfer's K/V page strides and the generic attention fallback. Validate
  explicit strides; the default zero retains tightly packed separate-cache
  behavior for existing callers.
- Force `flash_dense` in the regression, remove the opt-in gate, and parameterize
  prefills at 20, 32, and 48 tokens. Each case runs prefill → decode → decode →
  three-token chunked prefill → decode against the host oracle. These sizes cover
  scalar/vectorized writes, multiple pages, and decode allocation of a new page.
- Replace the stale investigation notes in `docs/attention.md` with this record.

### Validation

Validated on the available RTX 4080-series GPU in WSL2 with the CUDA 13.0 toolkit.
The following commands passed after rebuilding the seven attention test targets:

```sh
cmake --build build-rel --target \
  inferx_attention_spec_test inferx_attention_cache_spec_bridge_test \
  inferx_attention_capability_test inferx_attention_cache_view_test \
  inferx_attention_selector_test inferx_attention_runtime_test \
  inferx_attention_flash_gpu_test -j 4
ctest --test-dir build-rel -R '^attention_' --output-on-failure
build-rel/tests/unit/attn/inferx_attention_flash_gpu_test \
  --gtest_filter='PrefillSizes/*' --gtest_repeat=100 --gtest_break_on_failure
/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  build-rel/tests/unit/attn/inferx_attention_flash_gpu_test
```

Results: all seven CTest targets passed; all three differential cases passed
100 repetitions (300 cases, 1,500 attention steps); the GPU binary's four tests
passed under Compute Sanitizer with `ERROR SUMMARY: 0 errors`. Use the toolkit's
sanitizer path: this environment's older `/usr/bin/compute-sanitizer` failed to
locate its injection library. The validation covers the exercised causal GQA
paths, not every attention mechanism or parallel configuration.
