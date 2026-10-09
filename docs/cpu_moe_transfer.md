# CPU MoE with concurrent expert weight transfer

Set `expert_offload_config.cpu_moe.transfer_one_expert` to `true` to enable.
The default is `false`, retaining the original CPU/NPU split and join order.
This is for the existing single-card CPU hybrid decode path with replicated
CPU weights; exclusive dynamic ownership is rejected.

When at least one **distinct** routed expert is absent from NPU, move exactly
one expert to a physical slot unused by every current route. Zero misses
do not trigger. Duplicate routes count once. If no safe slot exists, preserve
CPU ownership. The promoted expert is excluded from CPU computation, and its
weight stays resident on NPU for later steps. CPU storage stays intact.
With a single miss, promotion leaves no CPU expert routes for this step.
The >=1 threshold increases transfer frequency for profiling; transfer still
requires a safe unused slot and the explicit enable switch.

## Host timing diagnostics

Reuse the existing `VLLM_ASCEND_CPU_MOE_TRACE` JSONL path switch before starting
the service, pointing to local `/tmp`. Paths under `/mnt/share/` are redirected
to `/tmp/cpu_moe_trace_<pid>.jsonl` to avoid NFS writes in runtime callbacks.
`cpu_moe.submit_end` now includes callback entry wall/monotonic
timestamps, submission end, and expert metadata collection duration.
`cpu_moe.transfer_weights` includes callback entry, weight load/enqueue begin
and end, H2D synchronization end, layer/expert/slot and phase durations.
`cpu_moe.transfer_skipped` identifies a callback with no selected expert.
Load duration includes host preparation and runtime enqueue, not device copy
duration; use the device trace for actual memcpy durations. Entry timestamps
help separate dispatch delays from callback execution after clock alignment.

One diagnostic record is written after each weight callback's measured phases.
The existing trace writer uses synchronous JSONL output, so its write time
and runtime completion notification are outside the measured callback work
and can still extend `NOTIFY_WAIT`. Disable diagnostics for throughput runs.

## Execution order

1. Stage routing/mapping and select promotion. Publish the updated mapping
   before splitting CPU/NPU routes, including all occurrences of the expert.
2. Stage CPU inputs and submit the remaining CPU work asynchronously.
3. In Graph mode, record the CPU-submitted event after the submission callback.
   The main stream waits for that already-recorded event before its weight
   loading callback. This waits for submission, never for CPU completion.
4. Load the full expert weights and quantization attributes through the
   manager's existing H2D transport on `load_stream`, then synchronize that
   transport before the NPU expert kernels consume the new physical slot.
   CPU work continues in the background during loading and NPU computation.
5. When consuming CPU results, record NPU completion, let the CPU communication
   stream wait for it, then register the blocking CPU join and result H2D.
   Record output readiness after result H2D; the main stream joins it before
   addition. Eager execution similarly submits CPU before loading weights
   and waits for CPU results only after issuing NPU computation.

Deferring the blocking Graph join avoids holding a shared host-callback
dispatcher while the weight-loading callback still needs to run. All events
are recorded before their waits, including first capture/replay. The tradeoff
is that CPU result H2D starts after the NPU phase in transfer mode, even on a
step without promotion. The default mode retains immediate CPU result H2D.

The unquantized, W4A8, MXFP4 and dynamic W8A8 hybrid entry points pass the
manager to the executor. Scheduling streams are shared by manager/device;
pinned buffers and events remain private to each captured layer/call. The
existing prefetch guard prevents paging into a layer owned by CPU hybrid.

## Validation

The focused CPU mock suite covers the >=1 threshold, safe
slot selection, duplicate route ownership, a real background thread active
during simulated weight loading, and Graph record/wait/callback ordering.
These tests allocate no NPU memory and do not exercise the real kt-kernel.

`tests/e2e/pull_request/one_card/test_cpu_moe_hybrid_transfer.py` adds an actual
NPU H2D/ACL Graph regression with one/two layers and changing route replays,
using a small synthetic CPU worker. Both variants pass on idle NPU 0 after the
original benchmark has written its summary and the serving process has exited.
Each variant checks 24 changing-route replays, route ownership and an active
CPU background task during H2D. This is scheduling evidence; the synthetic
worker intentionally remains active until the deferred join and does not
measure actual CPU kernel performance.

Real kt-kernel/model correctness, performance and CPU memory-bandwidth
contention remain unverified. The serving process was not stopped, restarted
or recaptured by this change. Source edits do not hot-update already loaded
Python modules; a future launch can opt into the new mode.
