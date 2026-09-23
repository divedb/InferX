# Bug fixes

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

`tests/event_queue_test.cc` covers a slow consumer draining 8,192 queued
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

`tests/text_delta_test.cc` covers ASCII growth, 2/3/4-byte characters split
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

`tests/event_queue_test.cc` adds two deterministic regressions: events
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
