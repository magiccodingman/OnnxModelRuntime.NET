# Architecture

## Purpose

`OnnxModelRuntime.NET` is an execution-orchestration library. Its job is to keep executable ONNX model instances healthy and feed them work safely under concurrency. It deliberately stops at the boundary where model semantics begin.

The runtime answers questions such as:

- How many independently loaded model copies exist?
- How many concurrent requests may each copy execute?
- Which healthy copy should receive the next request?
- What happens when the queue is full?
- What happens when one ONNX session becomes unhealthy?
- Should the failed request be retried?
- When is a replacement instance safe to put back into rotation?
- How are all native sessions disposed during shutdown?

It does not answer:

- What are the input tensor names?
- Is the model an embedder, classifier, reranker, decoder, or something else?
- How is text tokenized?
- How are outputs pooled, normalized, decoded, scored, or serialized?

## Managed layers

```text
consumer application
       |
       | TRequest / TResponse
       v
model-specific adapter
IOnnxModelExecutor<TRequest,TResponse>
       |
       | constructs inputs / interprets outputs
       v
InferenceSession
       ^
       | ownership, creation, disposal
OnnxSessionModelInstanceFactory<TRequest,TResponse>
       ^
       |
OnnxModelRuntime<TRequest,TResponse>
       |
       +-- bounded global queue
       +-- least-loaded scheduling
       +-- per-instance concurrency accounting
       +-- health / draining / recovery
       +-- failure classification / retry policy
       +-- diagnostics / async disposal
```

### `IOnnxModelExecutor<TRequest,TResponse>`

This is the normal model adapter boundary when the package owns an ONNX `InferenceSession`. It receives the actual session and a strongly typed request. It can inspect model metadata, create any tensors it needs, invoke `Run()`, and return any managed response type.

The runtime never examines the request or response.

### `IOnnxModelInstanceFactory<TRequest,TResponse>`

This is the lower-level boundary. Implementing it bypasses the standard session factory and lets a consumer create any independently executable model-instance object. The runtime still owns its lifecycle and scheduling.

The factory receives:

- `InstanceIndex` — stable slot index;
- `Generation` — `1` for startup, incremented after each successful rebuild;
- `ThreadsPerModel` — the resolved package-level thread policy.

It is heavily used by unit tests so scheduler behavior can be tested without downloading a model.

## Session ownership

The standard `OnnxSessionModelInstanceFactory` owns `InferenceSession` creation and disposal. Its baseline `SessionOptions` are:

```text
ExecutionMode = ORT_SEQUENTIAL
GraphOptimizationLevel = ORT_ENABLE_ALL
IntraOpNumThreads = resolved ThreadsPerModel
```

A callback can modify `SessionOptions` before session creation. This is the extension point for execution providers and future model/provider-specific tuning.

The baseline intentionally does not configure tensor names, output names, sequence lengths, embedding dimensions, or any other model contract.

## Instance model

Each runtime slot has stable identity and mutable generation:

```text
slot 0, generation 1  --failure-->  slot 0, generation 2
```

A successful replacement keeps the same slot index and increments generation. Diagnostics therefore make repeated failures/rebuilds visible without treating a replacement as a brand-new logical slot.

An instance is schedulable only when all are true:

```text
Health == Healthy
Model != null
ActiveRequests < MaxConcurrentRequests
```

## Scheduling model

There is one bounded multi-writer/single-reader channel. A single scheduler is responsible for reservations so the least-loaded decision and tie rotation are consistent.

Reservation is short and protected by an internal lock. Actual model execution never occurs under that lock.

Once reserved, synchronous ONNX/native execution is launched independently so the central scheduler can immediately continue dispatching other queued work.

## Failure classification

The orchestration layer cannot infer every model-specific failure from arbitrary exception classes, so failure policy is explicit:

```text
Exception -> IInferenceFailureClassifier -> InferenceFailureKind
```

The default understands `OnnxRuntimeException` and common memory-allocation signals. Consumers can replace it when their model adapter exposes additional categories.

Application failures intentionally do not reconstruct a session. This prevents a malformed request, a tensor-shape validation error owned by the adapter, or other model-specific failure from accidentally causing an expensive model reload.

## Native architecture

Managed generics cannot be projected directly into a stable C ABI. Native ABI v1 therefore exposes only the piece that *is* truly generic: opaque byte payloads scheduled through callbacks.

```text
foreign caller
  |
  | omr_executor_v1 callbacks
  v
NativeCallbackFactory<byte[],byte[]>
  |
  v
OnnxModelRuntime<byte[],byte[]>
```

Higher-level native packages are expected to define useful model-specific contracts and encode/decode their request types around this generic callback boundary.

## AOT boundary

No reflection-based serialization or universal tensor object is required by the core runtime. Managed APIs use normal generic types and interfaces. The native project uses `UnmanagedCallersOnly`, opaque handles, function pointers, explicit buffers, and stable numeric status codes.
