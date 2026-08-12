# OnnxModelRuntime.NET

**OnnxModelRuntime.NET** is a reusable .NET 10 runtime for hosting ONNX model sessions behind a bounded request queue with explicit concurrency control, least-loaded scheduling, per-instance failure isolation, automatic recovery, diagnostics, and Native AOT support.

It solves the infrastructure problem around inference without deciding what a model *means*.

The runtime owns:

- one or more independently hosted model instances / ONNX `InferenceSession`s;
- a global bounded request queue and backpressure;
- per-instance concurrent-request limits;
- least-loaded routing with fair tie rotation;
- health state, draining, recovery, generation tracking, and diagnostics;
- one-time retry of recoverable runtime/session failures;
- special handling for memory-pressure failures;
- clean asynchronous disposal;
- a small Native AOT C ABI for the generic scheduling runtime.

The consumer owns:

- model-specific input and output types;
- ONNX tensor names, shapes, and element types;
- tokenization and prompt construction;
- pooling, normalization, classification, reranking, decoding, or other output interpretation;
- model downloading, caching, updates, and application-specific lifecycle.

**This package does not interpret model-specific tensor contracts.** It does not assume `input_ids`, `attention_mask`, `token_type_ids`, embeddings, vectors, or any particular output shape.

## Install

```bash
dotnet add package OnnxModelRuntime.NET
```

The managed package targets **.NET 10**, enables nullable reference types and deterministic builds, and declares `IsAotCompatible=true`.

## The three knobs are different

An important design point is that **concurrent inference calls do not inherently require multiple copies of a model**. ONNX Runtime permits multiple callers to invoke `InferenceSession.Run()` concurrently.

```csharp
var options = new OnnxModelRuntimeOptions
{
    ModelInstanceCount = 1,
    ThreadsPerModel = 16,
    ConcurrentRequestsPerModel = 0, // automatic
    QueueCapacity = 256
};
```

- `ModelInstanceCount` is the number of independent model instances / sessions loaded into memory.
- `ThreadsPerModel` is the ONNX Runtime intra-op thread count configured for each standard session.
- `ConcurrentRequestsPerModel` is how many requests the scheduler may execute simultaneously against one model instance.

Those controls are deliberately independent. A single loaded `InferenceSession` can, for example, allow four concurrent calls without loading four copies of the model.

When `ConcurrentRequestsPerModel` is zero, the runtime uses:

```text
max(1, min(ThreadsPerModel / 2, 8))
```

The cap of eight is a package policy, **not** an ONNX Runtime limitation. Explicit positive values are honored even when they exceed eight.

`ThreadsPerModel = 0` enables hardware-based automatic resolution. The resolved thread count is bounded by `MaximumAutoThreadsPerModel` and divided across configured model instances in the same familiar way as OnnxTextEmbeddings.NET.

## Minimal strongly typed ONNX example

The most common managed API lets this package own the actual `InferenceSession` while your adapter owns the model-specific tensor contract.

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxModelRuntime;

public sealed record ClassifierInput(float[] Features);
public sealed record ClassificationResult(float[] Scores);

public sealed class ClassifierExecutor
    : IOnnxModelExecutor<ClassifierInput, ClassificationResult>
{
    public ClassificationResult Execute(
        InferenceSession session,
        ClassifierInput request,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // These names/shapes belong to THIS model adapter, not the runtime.
        var input = NamedOnnxValue.CreateFromTensor(
            "features",
            new DenseTensor<float>(request.Features, new[] { 1, request.Features.Length }));

        using var results = session.Run([input]);
        return new ClassificationResult(results.Single().AsEnumerable<float>().ToArray());
    }
}

await using var runtime = new OnnxModelRuntime<ClassifierInput, ClassificationResult>(
    "/models/classifier.onnx",
    new ClassifierExecutor(),
    new OnnxModelRuntimeOptions
    {
        ModelInstanceCount = 1,
        ThreadsPerModel = 16,
        ConcurrentRequestsPerModel = 4,
        QueueCapacity = 256
    });

var result = await runtime.RunAsync(new ClassifierInput(features), cancellationToken);
```

The exact same runtime can host an embedding adapter returning `float[]`, a reranker returning a score record, a classifier returning labels, or a generative/token-related model with completely different request and response types. The scheduler never inspects them.

## Fully custom model instances

For advanced cases, implement `IOnnxModelInstanceFactory<TRequest,TResponse>` directly. This is useful for deterministic tests, custom session ownership, provider-specific initialization, or another executable ONNX abstraction.

```csharp
public interface IOnnxModelInstanceFactory<TRequest, TResponse>
{
    IOnnxModelInstance<TRequest, TResponse> Create(
        OnnxModelInstanceCreationContext context,
        CancellationToken cancellationToken = default);
}
```

Each created instance is independently owned and disposed by the runtime. The creation context includes the instance index, generation, and resolved threads/model.

## ONNX session defaults and customization

`OnnxSessionModelInstanceFactory<TRequest,TResponse>` creates each standard `InferenceSession` with the baseline used by the reference implementation:

```text
ExecutionMode              = ORT_SEQUENTIAL
GraphOptimizationLevel     = ORT_ENABLE_ALL
IntraOpNumThreads          = resolved ThreadsPerModel
```

Consumers can supply a `SessionOptions` configuration callback for execution providers or other model-specific/runtime-specific changes:

```csharp
await using var runtime = new OnnxModelRuntime<MyRequest, MyResponse>(
    modelPath,
    executor,
    options,
    failureClassifier: null,
    configureSessionOptions: (sessionOptions, context) =>
    {
        // Add an execution provider or tune an option here.
    });
```

The runtime applies its sensible baseline first and then invokes the callback.

## Queueing and backpressure

Every request enters one **global bounded channel**. `QueueCapacity` is a real backpressure boundary: once the channel is full, producers asynchronously wait for room rather than allowing request objects and associated tensors to accumulate without bound.

Cancellation is honored while writing to a full queue, while queued, while waiting for model capacity, and during model execution when the model adapter observes the supplied token.

A concurrency slot is always released from a `finally` path, so cancellation and execution failures cannot permanently consume instance capacity.

## Least-loaded scheduling

The scheduler routes each request to the **healthy instance with the fewest active requests**. It does not fill instance 0 to capacity before considering instance 1.

```text
global bounded queue
        |
        v
least-loaded healthy scheduler
        |
        +--> instance 0   3/8
        +--> instance 1   2/8  <-- next request
        +--> instance 2   recovering (not eligible)
```

Equal loads use a rotating tie-break cursor. If two idle instances receive two requests, each receives one before either receives its second.

## Health, failure isolation, and recovery

Every model instance has explicit state:

```text
Starting -> Healthy -> Draining -> Recovering -> Healthy
                                      |
                                      +-> Faulted -> retry recreation

Disposed is terminal.
```

A recoverable runtime/session failure quarantines only the affected instance:

1. it is immediately removed from scheduling;
2. already active work is allowed to drain;
3. the failed model/session is disposed;
4. a fresh instance is created;
5. recreation failures leave the instance unavailable and retry with bounded exponential backoff;
6. only a successfully created replacement becomes healthy again.

A successful rebuild increments `Generation` and `TotalRecoveries`.

With multiple model instances, healthy copies continue serving traffic while one recovers. With one instance, queued work remains bounded and waits for a healthy replacement.

### Request retry policy

Failures are classified as one of:

- `Application` — model/input/application failure; the session is not reconstructed.
- `RecoverableInstance` — runtime/session infrastructure failure; quarantine the instance and transparently retry the affected request **at most once** through the global scheduler.
- `MemoryPressure` — quarantine/rebuild the instance, but **do not immediately retry that request against another loaded copy**.
- `Fatal` — permanently remove that instance from service rather than recreating it.

The default `OnnxRuntimeFailureClassifier` treats ordinary `OnnxRuntimeException` failures as recoverable and recognizes common ONNX allocation/OOM messages plus `OutOfMemoryException` as memory pressure.

Consumers can replace the classifier when a particular model adapter has additional failure semantics.

The OOM rule is intentional: if one loaded copy has just failed an allocation, immediately throwing the same request at another loaded copy can turn local memory pressure into cascading failures.

## Diagnostics

```csharp
var info = runtime.GetRuntimeInfo();

Console.WriteLine($"active: {info.ActiveRequests}");
Console.WriteLine($"queued: {info.QueuedRequests}/{info.QueueCapacity}");
Console.WriteLine($"healthy instances: {info.HealthyModelInstanceCount}");

foreach (var instance in info.Instances)
{
    Console.WriteLine(
        $"#{instance.Index} {instance.Health} " +
        $"{instance.ActiveRequests}/{instance.MaxConcurrentRequests} " +
        $"generation={instance.Generation} recoveries={instance.TotalRecoveries}");

    if (instance.LastFailure is not null)
        Console.WriteLine(instance.LastFailure);
}
```

Per-instance diagnostics expose:

- index;
- health;
- active requests;
- maximum concurrent requests;
- generation;
- total successful recoveries;
- current recovery attempt;
- last failure.

## Disposal

`OnnxModelRuntime<TRequest,TResponse>` implements `IAsyncDisposable`. Disposal stops accepting requests, closes/cancels scheduler activity, deterministically completes queued requests, waits for in-flight executions, stops recovery loops, and disposes every model instance/session.

```csharp
await runtime.DisposeAsync();
```

## Native AOT

The managed NuGet is designed for Native AOT consumers. The repository also contains `OnnxModelRuntime.Native`, which publishes as a Native AOT **shared library**.

The native project intentionally does **not** invent a model-specific universal inference schema. ABI v1 exposes the generic orchestration boundary through opaque runtime handles and caller-provided executor callbacks:

- create an instance;
- execute opaque request bytes;
- destroy an instance;
- optionally release callback-owned response memory.

The generic runtime copies callback responses before returning, and library-owned returned buffers are released with `omr_buffer_free`.

See [`docs/native-abi.md`](docs/native-abi.md) and [`native/include/onnx_model_runtime.h`](native/include/onnx_model_runtime.h).

## What the native ABI intentionally does not mirror

ABI v1 does not attempt to mirror managed generic types, `InferenceSession`, `SessionOptions`, arbitrary tensor objects, tokenizers, embedding vectors, reranker records, classifiers, or streaming/generative model contracts. Those belong in higher-level native packages where a real stable application-specific C contract can be defined.

For example, a future `OnnxTextEmbeddings.Native` can expose an embedding-specific C API while using this runtime underneath it.

## Intended OnnxTextEmbeddings.NET migration

This library was extracted so that a later `OnnxTextEmbeddings.NET` refactor can keep its tokenizer, chunker, model cache, vectors, semantic search, database integration, and public embedding APIs unchanged.

That project should eventually own only an embedding-specific ONNX adapter responsible for validating expected embedding inputs, constructing tensors, pooling model output, and normalizing vectors. `OnnxModelRuntime.NET` then owns sessions, model copies, queueing, concurrency, scheduling, health, recovery, retries, disposal, and diagnostics.

The option names deliberately map cleanly: `ModelInstanceCount`, `ThreadsPerModel`, `MaximumAutoThreadsPerModel`, `ConcurrentRequestsPerModel`, and `QueueCapacity`.

## Repository layout

```text
src/OnnxModelRuntime/          managed reusable runtime / NuGet package
src/OnnxModelRuntime.Native/   Native AOT shared-library exports
native/include/                public C header
native/smoke/c/                standalone C ABI smoke consumer
tests/                         deterministic unit tests + real ONNX smoke
docs/                          architecture and detailed behavior
```

## CI and releases

Pull requests build and test on Linux, Windows, and macOS. Native AOT CI also publishes and executes a managed AOT smoke, builds the shared library, compiles a standalone C consumer against the checked-in header, and runs it against the produced library. The smoke path includes a real ONNX Runtime model rather than validating mocks only.

Merges to `release` use `.github/workflows/publish-nuget.yml`. Package-affecting changes are detected before publication; versions are resolved against both Git tags and Nuget.org; trusted publishing uses `NuGet/login@v1`; successful publication is tagged and receives a GitHub Release. README/icon-only package changes are release-affecting.

## More documentation

- [`docs/architecture.md`](docs/architecture.md) — ownership boundaries and extension points.
- [`docs/concurrency-and-recovery.md`](docs/concurrency-and-recovery.md) — queueing, load balancing, failure recovery, and OOM semantics.
- [`docs/native-abi.md`](docs/native-abi.md) — Native AOT and C ABI v1.
- [`docs/onnx-text-embeddings-migration.md`](docs/onnx-text-embeddings-migration.md) — intended future extraction path for OnnxTextEmbeddings.NET.

## License

Apache-2.0.
