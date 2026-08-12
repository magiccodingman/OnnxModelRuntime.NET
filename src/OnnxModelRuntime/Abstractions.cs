using Microsoft.ML.OnnxRuntime;

namespace OnnxModelRuntime;

/// <summary>Creates independently hosted executable model instances for the runtime.</summary>
public interface IOnnxModelInstanceFactory<TRequest, TResponse>
{
    IOnnxModelInstance<TRequest, TResponse> Create(
        OnnxModelInstanceCreationContext context,
        CancellationToken cancellationToken = default);
}

/// <summary>An executable model instance owned and disposed by <see cref="OnnxModelRuntime{TRequest,TResponse}"/>.</summary>
public interface IOnnxModelInstance<in TRequest, out TResponse> : IDisposable
{
    TResponse Execute(TRequest request, CancellationToken cancellationToken = default);
}

/// <summary>Model-specific ONNX tensor construction and output interpretation.</summary>
public interface IOnnxModelExecutor<in TRequest, out TResponse>
{
    TResponse Execute(InferenceSession session, TRequest request, CancellationToken cancellationToken = default);
}

/// <summary>Context supplied whenever the runtime creates or rebuilds an instance.</summary>
public sealed record OnnxModelInstanceCreationContext(
    int InstanceIndex,
    int Generation,
    int ThreadsPerModel);
