# CacheManager

`inferx::CacheManager` owns logical request-to-block
mappings over a borrowed `KvBlockPool`. The scheduler owns the manager; the
rank-local runner's `StateStorage` continues to own device memory. The pool must
outlive the manager. For tensor parallel execution, the coordinating rank's
manager assigns logical block IDs that all ranks use in their local storage.

## Operations

- `Reserve(request_id, target_tokens)` grows capacity and returns only newly
  granted block IDs in logical order. Failure preserves both the existing mapping
  and pool availability. Smaller targets do not shrink the allocation.
- `Blocks(request_id)` borrows the full logical mapping; an unknown request
  returns an empty span. Growth or release invalidates that request's span.
- `Release(request_id)` frees all blocks for the request. Repeated release is
  harmless. Destruction releases any outstanding allocations.
- `usage()`, `free_blocks()`, and `total_blocks()` expose pool occupancy.

The manager operates on request identifiers without depending on engine request
classes. `Request` no longer owns a block table. `Scheduler::Schedule` reserves
capacity and emits the same new-request/block-delta protocol consumed by runners.
Completion and cancellation release allocations through the manager.

## Execution contract

This first implementation is single-threaded and synchronous. The engine must
finish GPU work before releasing storage or destroying the scheduler. A successful
reservation grants capacity; it does not assert that KV contents have been
computed. The existing scheduler/runner progress protocol is unchanged.

Prefix lookup, shared blocks, copy-on-write, asynchronous leases, and recurrent
state snapshots are not implemented here. Those policies can be introduced at
this boundary without teaching the scheduler about device tensors. Prefix reuse
for hybrid models will also require compatible recurrent state.

Tests cover failed admission/growth, mapping order, repeated release, ID reuse,
large capacity validation, and scheduler/manager teardown.
