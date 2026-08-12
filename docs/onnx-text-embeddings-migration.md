# Intended OnnxTextEmbeddings.NET migration

This repository does **not** refactor `OnnxTextEmbeddings.NET`. It is designed so that a later change can remove the embedding project's custom worker pool without disturbing the rest of its architecture.

## Before

The current embedding worker/session layer combines:

```text
InferenceWorkerPool
  + queue / scheduling
  + model-instance concurrency
  + health and recovery
  + retry policy
  + ONNX session creation
  + embedding input validation
  + embedding tensor construction
  + output pooling
  + vector normalization
```

The first six responsibilities are reusable. The final four are model-specific.

## After

The embedding project can provide an adapter conceptually like:

```csharp
internal sealed class EmbeddingOnnxModelExecutor
    : IOnnxModelExecutor<TokenizedModelInput, float[]>
{
    public float[] Execute(
        InferenceSession session,
        TokenizedModelInput input,
        CancellationToken cancellationToken = default)
    {
        // Validate the embedding model contract.
        // Build input_ids / attention_mask / token_type_ids as appropriate.
        // Run the session.
        // Pool output if required.
        // Normalize the embedding.
    }
}
```

Then it can construct:

```csharp
new OnnxModelRuntime<TokenizedModelInput, float[]>(
    snapshot.ModelPath,
    embeddingExecutor,
    mappedRuntimeOptions);
```

## Option mapping

The existing embedding inference controls map directly:

| OnnxTextEmbeddings.NET | OnnxModelRuntime.NET |
|---|---|
| `ModelInstanceCount` | `ModelInstanceCount` |
| `ThreadsPerModel` | `ThreadsPerModel` |
| `MaximumAutoThreadsPerModel` | `MaximumAutoThreadsPerModel` |
| `ConcurrentRequestsPerModel` | `ConcurrentRequestsPerModel` |
| `QueueCapacity` | `QueueCapacity` |

Automatic thread and concurrency behavior intentionally stays familiar.

## What should not move

The migration should **not** move any of these into this package:

- `TokenizedModelInput` as a shared runtime type;
- `EmbeddingDimensions` as a runtime concept;
- embedding model input-name validation;
- mean pooling;
- normalization;
- vector quantization/storage formats;
- tokenizer or chunker code;
- Jasper presets;
- Hugging Face download/cache/update code;
- semantic-search/database code.

If embedding dimensions are needed by the embedding service, its adapter can inspect the session/model metadata and expose that metadata through its own embedding-specific layer. It is not a generic runtime diagnostic.

## Diagnostics mapping

The generic runtime already exposes the per-instance fields used by the current embedding service:

```text
Index
Health
ActiveRequests
MaxConcurrentRequests
Generation
TotalRecoveries
RecoveryAttempts
LastFailure
```

The embedding project's public `ModelInfo` can continue projecting these into its own service diagnostics without coupling this package to embedding concepts.

## Failure mapping

The default classifier already handles ordinary `OnnxRuntimeException` and memory pressure. If the embedding adapter has a model-validation exception that should remain an application error, no special handling is needed: non-ONNX exceptions default to `Application` and therefore do not reconstruct the session.
