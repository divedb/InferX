# Test layout

- `unit/<component>/`: component contracts and numerical kernel tests. GPU
  requirements do not by themselves make a test an integration test. Model
  runner tests use fake models to isolate execution behavior.
- `integration/models/`: checkpoint loading, complete-model numerical parity,
  and tensor-parallel execution across model and runtime components.
- `integration/server/`: tokenizer-pool behavior with the tokenizer backend.
- `kvc/`: independently built KV-cache provider ABI, loader, conformance, and
  adapter tests. This suite also supports builds with the inference engine off.
- `testdata/`: shared fixtures; tests resolve these from the project root.

Register engine tests in `tests/CMakeLists.txt` with explicit source paths.
Keep target names, CTest names, working directories, timeouts, and device
environment settings stable when moving files. Tests requiring local model
checkpoints or multiple GPUs may skip when those resources are unavailable.

Run the engine unit or integration suites after building:

```sh
ctest --test-dir build-cuda13 -L '^unit$' --output-on-failure
ctest --test-dir build-cuda13 -L '^integration$' --output-on-failure
```

Run without a label filter to include the independent KVC suite as well.
