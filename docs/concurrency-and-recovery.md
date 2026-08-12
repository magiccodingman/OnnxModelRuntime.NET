# Concurrency, queueing, and recovery

## Model copies are not request concurrency

`ModelInstanceCount`, `ThreadsPerModel`, and `ConcurrentRequestsPerModel` tune different resources.

A model instance is an independently created executable instance and, with the standard factory, an independent ONNX `InferenceSession`. `ConcurrentRequestsPerModel` is the number of requests permitted to call that instance concurrently. A second concurrent request therefore does not imply a second model copy.

Automatic concurrent requests/model are resolved as:

```text
max(1, min(ThreadsPerModel / 2, 8))
```

Examples:

| Threads/model | Automatic concurrent requests/model |
|---:|---:|
| 1 | 1 |
| 2 | 1 |
| 4 | 2 |
| 8 | 4 |
| 12 | 6 |
| 16 | 8 |
| 24 | 8 |
| 32 | 8 |

The cap is a package default. `ConcurrentRequestsPerModel = 24`, for example, is accepted and honored.

## Bounded global queue

The runtime uses one `BoundedChannel<T>` with `FullMode = Wait`.

That design provides two boundaries:

1. the channel limits requests waiting to be scheduled;
2. each model instance limits active requests already executing against it.

When the queue is full, a producer awaits `WriteAsync`. No extra unbounded overflow list is created.

Cancellation can stop a producer waiting to enter the channel. Once the scheduler has read a request, cancellation is also linked into waits for instance capacity, so a canceled request does not need an unrelated future capacity event just to be noticed.

## Least-loaded scheduler

For every dispatch the scheduler first finds the smallest active-request count among healthy, non-full instances. It then searches from a rotating cursor for an instance at that minimum.

For two instances with maximum 4 requests each:

```text
request 1 -> A 1/4, B 0/4
request 2 -> A 1/4, B 1/4
request 3 -> A 2/4, B 1/4
request 4 -> A 2/4, B 2/4
```

This spreads work as capacity is added instead of filling A to 4/4 before B receives anything.

## Reservation correctness

A reservation increments `ActiveRequests` before execution begins. Every execution path releases it in `finally`, including:

- successful model execution;
- request cancellation;
- ordinary application exceptions;
- recoverable infrastructure exceptions;
- OOM/memory-pressure exceptions;
- fatal failures.

A canceled or failed call therefore cannot leak a slot.

## Recoverable instance failure

When a failure is classified `RecoverableInstance`:

```text
Healthy
  |
  | failure: remove from scheduler immediately
  v
Draining
  |
  | existing ActiveRequests -> 0
  v
Recovering
  |
  | dispose failed model/session
  | create replacement
  |
  +-- failure --> Faulted --backoff--> Recovering
  |
  +-- success --> Healthy, generation + 1
```

The instance remains unavailable during all recreation failures. Merely waiting for a backoff timer does not make it healthy.

The first recreation attempt after an ordinary recoverable failure has no artificial delay. Subsequent failures use bounded exponential backoff beginning at 250 ms and capped at 10 seconds.

## Request retry

A request that encountered `RecoverableInstance` is re-enqueued through the global scheduler at most once.

This matters because re-enqueueing, rather than directly invoking a different session, preserves all normal behavior:

- least-loaded routing;
- queue bounds;
- cancellation;
- use of another healthy instance when available;
- waiting for a replacement when the failed instance was the only copy.

If the retried request hits another recoverable instance failure, it fails to the caller rather than recursively retrying.

## Memory pressure

`MemoryPressure` still quarantines and rebuilds the failing instance, but the failed request is completed with an error instead of being sent to another loaded copy.

The first memory-pressure rebuild attempt waits one second. This creates a small pressure-release window before allocating another full session/model copy.

This policy is intentionally conservative: if one instance cannot allocate memory, immediately replaying the request against another loaded instance can cause cascading allocation failures.

The runtime can only recover if the process remains alive. An operating-system, container, or cgroup OOM kill is outside an in-process recovery library's control.

## Application failures

`Application` errors are returned to the caller unchanged. They do not mark the model unhealthy and do not recreate a session.

This is important for a generic runtime: invalid model-specific input should not be conflated with broken ONNX infrastructure.

## Fatal failures

`Fatal` removes the affected model instance from service after its active work drains and disposes it without a rebuild loop. Other healthy instances remain eligible.

## Disposal

Disposal follows a deterministic shutdown order:

1. mark the runtime disposed so new work is rejected;
2. complete the channel writer;
3. cancel scheduler/recovery waits;
4. complete pending queued work deterministically;
5. await scheduler termination;
6. await in-flight request tasks;
7. await recovery/fault lifecycle tasks;
8. dispose every remaining model instance;
9. mark every slot `Disposed`;
10. dispose scheduler synchronization resources.

Synchronous model execution that ignores cancellation is still awaited. This prevents the runtime from disposing an `InferenceSession` while a native `Run()` call is still using it.
