# Bug fixes

## 2026-10-24: HTTP lifecycle test hangs after drain completes

`HttpServerLifecycleTest.InFlightRequestFinishesDuringDrain` could finish its
request/drain checks and hang in `ServerInstance` destruction waiting for the
`Stop()` future. Reproduced before the fix with a 10-second process timeout
(exit 124). All I/O worker threads could already have exited.

Root cause: the test fixture did not hold an Asio executor work guard.
Once drain closed the listener and sessions, `io_context::run()` could exhaust
its work and stop. `EnsureIoRunning()` checked `stopped()` before queuing
`Stop()`, leaving a check/submission race. Its recovery path also launched
`run()` threads on an empty restarted context before submitting `Stop()`;
those threads could immediately exit again. Additionally, it called
`restart()` before joining the previous threads, contrary to Asio's requirement
that no run/poll calls remain unfinished. The coroutine had not vanished:
it was queued on an executor with no running workers. An un-awaited timer
object does not keep an io_context running.

Fix: `ServerInstance` owns an `executor_work_guard` from before worker startup
until the shutdown future completes. `StopAndJoin()` submits and awaits
`Stop()`, releases the guard, then joins workers after their work drains
naturally. Teardown is idempotent; lifecycle tests use the same shutdown path.
No production server change is needed for this particular fixture bug.
Applications owning the executor must likewise keep it running while awaiting
server lifecycle operations.

Regression coverage: `StopAfterDrainHasCompleted` deliberately waits for the
server to reach stopped state before submitting `Stop()`, checks future
completion within two seconds, and verifies natural executor exhaustion and
repeated teardown. The original in-flight drain test remains covered.

The full run also exposed an invalid-config test that cleared `model_id` but
left valid served aliases. It now clears both. Startup-failure cleanup releases
the work guard and joins worker threads before throwing, avoiding termination
from destructing joinable threads during constructor unwinding.

Validation:

```sh
cmake --build build-dbg --target inferx_server_http_test -j 4
timeout 10s build-dbg/tests/unit/server/inferx_server_http_test --gtest_brief=1
timeout 10s build-dbg/tests/unit/server/inferx_server_http_test --gtest_filter=HttpServerLifecycleTest.InFlightRequestFinishesDuringDrain:HttpServerLifecycleTest.StopAfterDrainHasCompleted --gtest_repeat=100 --gtest_brief=1
```

Result: all 27 HTTP tests passed; both drain regressions passed 100 repetitions
(200 test executions). Socket tests ran outside the sandbox for loopback access.

## 2026-09-22: EventQueue notification channel data race

`EventQueue` used `boost::asio::experimental::channel` for its one-slot
notifier. `Notify()` posted `try_send()` to the server's `io_context`, while
consumer coroutines called `async_receive()` on that channel. The server
runs the context on up to four threads: posting work selects an executor,
but does not serialize handlers. Sends could overlap each other and receive
initiation, accessing the channel's unsynchronized internal state. The
payload deque's mutex did not protect the channel. This was undefined
behavior, with possible lost wakeups, stalled requests, or memory corruption.

The notifier now uses `boost::asio::experimental::concurrent_channel`, which
synchronizes shared access internally. The mutex-protected payload deque,
one-slot wakeup coalescing, and posted notification lifetime remain intact.

`tests/unit/server/event_queue_test.cc` covers a slow consumer draining 8,192 queued
deltas and a terminal event after notifications coalesce, plus 16 queues
receiving 2,048 deltas each from an engine thread while four I/O threads
process notifications and coroutine receives. Both finish and error terminal
events are checked, along with exact event ordering and counts. The
concurrency case is a stress regression, not a deterministic race detector.

Validation command:

```sh
cmake --build build-cuda13 --target inferx_event_queue_test inferx_server_api_test
ctest --test-dir build-cuda13 -R '^(event_queue_test|server_api_test)$' --output-on-failure
```

Result: both CTest targets passed, including all four EventQueue test cases.
A standalone GCC ThreadSanitizer build of the queue tests compiled, but its
runtime exited before running tests with `FATAL: ThreadSanitizer: unexpected
memory mapping`. No sanitizer-clean result is claimed for this environment.

## 2026-09-22: Streaming detokenizer crash on split multi-byte characters

The first InferX pass of the serving sweep (15/18 valid) aborted in the
`i128-o512-c16` cell: the engine thread terminated with
`std::out_of_range: basic_string_view::substr: __pos (which is 1107) >
__size (which is 1106)`, killing every in-flight request.

Root cause: the gateway re-decodes the whole generated token list every
step and emits the suffix past a monotonic byte offset. A full re-decode is
not prefix-stable. A byte-level decoder (`ByteLevel::decode_chain` ends in
`String::from_utf8_lossy`) renders a trailing partial UTF-8 character as
U+FFFD (three bytes); when later tokens complete the character, the real
character can be shorter (two bytes for a 2-byte character), so the decoded
text shrinks below the emitted offset and `substr` throws on the engine
thread, where nothing catches it. The existing `Utf8Boundary` hold-back
cannot prevent this: the replacement character is itself complete UTF-8, so
the boundary check let the offset advance over bytes the next decode
rewrote.

Fix: the offset logic moved into `inferx/server/text_delta.h`
(`TextDelta`). `NextDelta` emits only settled prefixes — everything up to
trailing replacement characters, which the next decode may rewrite;
settled replacements (more text follows) are permanent and emitted.
`Flush` returns held-back bytes for a final delta emitted before `kFinish`,
so a finished request's deltas concatenate to exactly its final decode. A
defensive re-sync replaces the throw if a decode ever shrinks anyway.

`tests/unit/server/text_delta_test.cc` covers ASCII growth, 2/3/4-byte characters split
across steps (the field signature reproduced as
`__pos (which is 4) > __size (which is 3)` before the fix), multiple
trailing replacements, settled mid-text replacements, the finish flush,
and end-to-end concatenation equality over a mixed stream. All cases
failed or aborted on the extracted old logic and pass on the new one.

## 2026-09-22: EventQueue lost wake-up stalled requests at high concurrency

The same sweep's `i1024-o128-c16` and `i1024-o512-c16` cells timed out with
the client stuck at 62/64 and 61/64 requests; the server survived until
teardown, where it aborted with glibc allocator errors
(`double free or corruption`, `corrupted size vs. prev_size`). The
`concurrent_channel` fix above was in that build: the allocator errors
were a symptom, not the defect. gdb showed the engine thread asleep in
`EngineLoop` with the scheduler fully drained while consumer coroutines
waited forever.

Root cause: `Emit` pushed each event of a step and then posted one
notification per queue, coalescing by queue. The notification itself was
delivered asynchronously (posted `try_send`). A consumer could consume the
token, drain a prefix of the queue's batch, and re-park while the engine
thread was still pushing the rest — typically the `kFinish` from
`PopFinished` — which then arrived with no further notification. The
terminal event sat in the deque forever; the client hung; and the graceful
shutdown later destroyed the `io_context` with those consumers still
suspended, which is undefined behavior in asio and produced the allocator
aborts. The shutdown UB remains a latent defect of its own (any parked
coroutine at exit can trigger it); with the stall gone it no longer
manifests in these workloads.

Fix: delivery is armed by the push itself. `EventQueue::Push` calls
`notify_.try_send` directly — safe from the engine thread because the
channel is a `concurrent_channel`; `try_send` completes a parked receiver
or leaves one buffered token. A `try_send` that finds the one-slot channel
full is safe because `TakeAll` is level-triggered on the deque. The
separate `Notify()` step a producer could forget or coalesce away is gone,
along with `Emit`'s dedup vector. `AbortSlowConsumers` also moved ahead of
the `PopFinished` loop so slow-consumer aborts deliver their terminal
events in the same step instead of stranding them in the finished list
when the scheduler drains.

`tests/unit/server/event_queue_test.cc` adds two deterministic regressions: events
pushed with no separate notification must reach a parked consumer, and an
event pushed after the consumer drains and re-parks (the stall window)
must still be delivered. Both time out against the old queue; the existing
backlog and four-thread stress cases pass unchanged against the new one.

Validation command:

```sh
cmake --build build-cuda13 --target inferx_event_queue_test inferx_text_delta_test inferx_server_api_test
ctest --test-dir build-cuda13 -R '^(event_queue_test|text_delta_test|server_api_test)$' --output-on-failure
```

Result: all suites pass. An end-to-end check with the sweep's failing
workload shape (1024-token random prompts, 128 output tokens, concurrency
16, 64 requests per trial, `vllm bench`-equivalent streaming client)
completed 8/8 trials with zero failures, zero stalls, and a clean SIGINT
exit; the pre-fix binary aborted or stalled within three trials of the
same driver.

## 2026-09-23: Open concurrency findings from the benchmark rerun

The [new investigation](../benchmarks/qwen3/serve_results/rerun_20260923/CONCURRENCY.md)
records a fresh matched InferX/vLLM length sweep plus separate concurrency probes.
It preserves the existing fixes above and makes no production code changes.
Findings include KV-pressure progress failure under concurrent admission,
a shutdown grace timer armed only after a blocking engine join, HTTP executors
without Beast's required strand serialization, missing encode/decode
serialization, admission racing shutdown, and coroutine cleanup outliving the
gateway. The report distinguishes runtime reproductions from source-inspection
findings and retains commands, outputs, and validation limits.

One correction to the earlier shutdown explanation: destroying an `io_context`
with suspended operations is supported by Asio; it destroys pending handlers.
The concrete application lifetime risk is that a suspended `CancelGuard` can
call an already-destroyed gateway. Neither that inspection finding nor an
executor-contract violation proves the cause of the historical allocator errors.

## 2026-10-09: InferX vs vLLM greedy agreement — 4 of 8 completions differ

The first post-refactor serving comparison
(`benchmarks/qwen3/serve_results/refactor_fix_20261009`; 8 fixed prompts,
`temperature: 0`, 64 completion tokens, two repetitions per stack)
completed all 80 InferX requests successfully. Four of the eight greedy
completions match vLLM exactly; four differ, so exact output parity was
not established.

**Root cause: BF16 last-ulp divergence at near-tied decisions, not a
semantic or serving defect.** The two stacks compute the same model with
independent kernel orderings (cuBLAS GEMM tilings, FlashInfer
attention/norm variants, rope roundings), so their logits differ in the
final one or two BF16 bits. Whenever a step's top-2 candidates sit that
close, each stack's argmax lands on its own rounding, and greedy decoding
amplifies the single flipped token into a wholly different — still fluent
— continuation. Both stacks are internally deterministic (each reproduced
its own outputs across both repetitions, and today's build reproduces the
recorded InferX outputs byte-for-byte on all 8 prompts), which rules out
sampling configuration, mid-refactor build drift, and nondeterminism.

Evidence, from the teacher-forced probe kept as
`tests/integration/models/vllm_parity_probe_test.cc`: for each differing
completion, tokenize both full texts, locate the first differing token,
teacher-force the shared prefix through the model, and read the logits at
exactly that decision position. InferX's own logits at the four decision
points:

| prompt | first diff token | logit(InferX) | logit(vLLM) | gap |
|---|---|---|---|---|
| 0 "The capital of France is" | 17 (`French` vs `Republic`) | 14.9375 | 14.8750 | 1 ulp |
| 4 "The largest planet…" | 43 (`\n` vs `' '` after "Mars**") | 23.5000 | 23.5000 | exact tie |
| 5 "A classic example of a prime…" | 16 (`Let's` vs `What`) | 17.3750 | 17.2500 | 1 ulp |
| 7 "Cross-entropy loss…" | 16 (`incorrect` vs `correct`) | 19.5000 | 19.2500 | 2 ulps |

Prompt 4 is a literal BF16 tie (both candidates 23.5000, bits 0x41BC):
the engine's fused argmax reduction keeps one maximal token and vLLM's
keeps the other — a tie-break with no numerically meaningful winner. The
same mechanism explains the golden-trace tie documented in
`model_test.cc` (15344 vs 9625 at exactly 16.6250).

What this is not: not sampling (requests carried `temperature: 0` and
both servers were deterministic across repetitions); not a regression
(today's binary regenerates all 8 recorded outputs exactly, including the
4 divergent ones); not KV/state corruption (the probe reproduces every
InferX choice from the shared prefix up to the near-tie).

Consequence for parity claims: exact text equality across independently
implemented BF16 stacks is attainable only where every decision has a
margin larger than rounding noise. The meaningful criterion is
agreement-where-decided: the probe asserts each divergence occurs at a
≤2-ulp decision point, and 4/8 prompts matching exactly is consistent
with the remainder hitting near-ties within 64 sampled decisions each.

### 2026-10-09 follow-up: tie semantics verified against vLLM; no engine defect

Referred to vLLM's implementation to check the one engine behavior the
tie explanation depends on: `vllm/v1/sample/sampler.py::greedy_sample` is
`logits.argmax(dim=-1)`, and `torch.argmax` returns the FIRST maximal
index on ties (verified empirically on CUDA, torch 2.12). InferX's three
sampling paths -- the fused `GreedyArgmax` op, the mixed-batch pipeline's
greedy rows, and the CPU reference -- all use the same comparator
(maximum value, ties to the lowest index) and therefore already implement
vLLM's convention; nothing to fix. What was missing was proof, so:

- `tests/regression/argmax_tie_test.cc` pins the semantics at full
  vocabulary scale (151,936 columns) using the investigation's actual
  ties: 334/56177 at 23.5, 9625/15344 at 16.625, and a three-way tie
  spanning the kernel's partial reductions. All resolve to the lowest
  index, including across the 38 partial blocks.
- `tests/regression/vllm_parity_probe_test.cc` (moved from
  tests/integration) adds the serving-path check: walking each differing
  prompt as the runner executed it -- prompt prefill, then one-token
  decode steps against a persistent KV pool -- reproduces the engine's
  recorded choice as the strict argmax of that path's logits. Notably,
  for the exact-tie prompt (334/56177, both 23.5 in the single-chunk
  view), the serving path's own logits rank 56177 strictly first: the
  decode-kernel roundings differ from the single-chunk-prefill roundings
  by an ulp, each view is self-consistent, and the engine picks its own
  path's true argmax either way.

Conclusion unchanged and now proven end to end: the divergences are
strict-argmax differences over 0-2-ulp logit gaps produced by independent
kernel paths (InferX's cuBLAS/FlashInfer invocations vs vLLM's), not a
tie-break, sampling, or state defect. Exact text parity between the
stacks is bounded by that; matching it further would require
kernel-for-kernel bitwise equality, which is not a goal.

Validation command:

```sh
cmake --build build-cuda13 --target inferx_argmax_tie_test inferx_vllm_parity_probe_test
ctest --test-dir build-cuda13 -R '^(argmax_tie_test|vllm_parity_probe_test)$' --output-on-failure
```

Result: both pass. The probe reports the 0-, 1-, 1-, and 2-ulp decision
gaps, reproduces InferX's own choice at every decision point under both
the single-chunk and the serving execution path, and the argmax suite
confirms lowest-index ties at vocabulary scale.
