using Microsoft.ML.OnnxRuntime;

namespace OnnxModelRuntime;

/// <summary>
/// Standard factory that lets the runtime own <see cref="InferenceSession"/> creation/disposal while a strongly typed
/// executor owns model-specific tensor construction and output interpretation.
/// </summary>
public sealed class OnnxSessionModelInstanceFactory<TRequest, TResponse> : IOnnxModelInstanceFactory<TRequest, TResponse>
{
    private readonly string _modelPath;
    private readonly IOnnxModelExecutor<TRequest, TResponse> _executor;
    private readonly Action<SessionOptions, OnnxModelInstanceCreationContext>? _configureSessionOptions;

    public OnnxSessionModelInstanceFactory(
        string modelPath,
        IOnnxModelExecutor<TRequest, TResponse> executor,
        Action<SessionOptions, OnnxModelInstanceCreationContext>? configureSessionOptions = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentNullException.ThrowIfNull(executor);
        _modelPath = modelPath;
        _executor = executor;
        _configureSessionOptions = configureSessionOptions;
    }

    public IOnnxModelInstance<TRequest, TResponse> Create(
        OnnxModelInstanceCreationContext context,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var sessionOptions = new SessionOptions
        {
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            IntraOpNumThreads = context.ThreadsPerModel
        };
        _configureSessionOptions?.Invoke(sessionOptions, context);
        cancellationToken.ThrowIfCancellationRequested();
        var session = new InferenceSession(_modelPath, sessionOptions);
        return new SessionModelInstance(session, _executor);
    }

    private sealed class SessionModelInstance(
        InferenceSession session,
        IOnnxModelExecutor<TRequest, TResponse> executor) : IOnnxModelInstance<TRequest, TResponse>
    {
        public TResponse Execute(TRequest request, CancellationToken cancellationToken = default) =>
            executor.Execute(session, request, cancellationToken);

        public void Dispose() => session.Dispose();
    }
}
