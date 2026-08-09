# Symbolized crash-stack example

This example installs InferX's fatal-signal handler and deliberately causes
`SIGSEGV` through a named call chain. Build it with debug information, then run
it directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target crash_stack_example
./build/examples/unit/support/crash_stack_example
```

The process intentionally exits unsuccessfully after writing a stack trace to
stderr. The trace identifies the signal and symbolizes the application call
chain:

```text
*** SIGSEGV received ...
PC: @ ... inferx::example::TriggerSegmentationFault()
    @ ... absl::AbslFailureSignalHandler()
    @ ... inferx::example::RunWorker()
    @ ... inferx::example::RunRequest()
    @ ... main
```

Exact addresses and system-library frames vary by platform. Do not use this
program as a health check: receiving `SIGSEGV` and terminating is its expected
behavior.
