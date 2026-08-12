using System.Collections.Concurrent;
using System.Threading.Channels;

namespace OnnxModelRuntime;

/// <summary>
/// Generic bounded, least-loaded runtime for independently hosted model instances. It intentionally knows nothing
/// about model tensor names, tokenization, pooling, vector shapes, prompts, or response semantics.
/// </summary>
public sealed partial class OnnxModelRuntime<TRequest, TResponse> : IAsyncDisposable
{
    private sealed record WorkItem(
        TRequest Request,
        TaskCompletionSource<TResponse> Completion,
        CancellationToken CancellationToken,
        int InfrastructureRetries = 0);

    private sealed class ModelInstance(
        int index,
        IOnnxModelInstance<TRequest, TResponse> model,
        int maxConcurrentRequests)
    {
        public int Index { get; } = index;
        public IOnnxModelInstance<TRequest, TResponse>? Model { get; set; } = model;
        public int MaxConcurrentRequests { get; } = maxConcurrentRequests;
        public int ActiveRequests { get; set; }
        public ModelInstanceHealth Health { get; set; } = ModelInstanceHealth.Healthy;
        public int Generation { get; set; } = 1;
        public int TotalRecoveries { get; set; }
        public int RecoveryAttempts { get; set; }
        public string? LastFailure { get; set; }
        public TaskCompletionSource<bool>? Drained { get; set; }
        public Task? RecoveryTask { get; set; }
        public bool PermanentlyFaulted { get; set; }
    }

    private readonly Channel<WorkItem> _channel;
    private readonly ModelInstance[] _instances;
    private readonly IOnnxModelInstanceFactory<TRequest, TResponse> _factory;
    private readonly IInferenceFailureClassifier _failureClassifier;
    private readonly object _gate = new();
    private readonly SemaphoreSlim _capacitySignal = new(0, 1);
    private readonly ConcurrentDictionary<long, Task> _inflight = new();
    private readonly CancellationTokenSource _shutdown = new();
    private readonly Task _scheduler;
    private long _nextWorkId;
    private int _tieBreakerCursor;
    private int _queuedRequests;
    private int _disposeStarted;

    public OnnxModelRuntime(
        IOnnxModelInstanceFactory<TRequest, TResponse> factory,
        OnnxModelRuntimeOptions? options = null,
        IInferenceFailureClassifier? failureClassifier = null)
    {
        ArgumentNullException.ThrowIfNull(factory);
        _factory = factory;
        _failureClassifier = failureClassifier ?? OnnxRuntimeFailureClassifier.Default;
        var resolved = (options ?? new OnnxModelRuntimeOptions()).Resolve();

        ModelInstanceCount = resolved.ModelInstanceCount;
        ThreadsPerModel = resolved.ThreadsPerModel;
        ConcurrentRequestsPerModel = resolved.ConcurrentRequestsPerModel;
        QueueCapacity = resolved.QueueCapacity;

        _channel = Channel.CreateBounded<WorkItem>(new BoundedChannelOptions(resolved.QueueCapacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false,
            AllowSynchronousContinuations = false
        });

        _instances = new ModelInstance[resolved.ModelInstanceCount];
        try
        {
            for (var i = 0; i < _instances.Length; i++)
            {
                var model = _factory.Create(
                    new OnnxModelInstanceCreationContext(i, 1, resolved.ThreadsPerModel),
                    CancellationToken.None);
                _instances[i] = new ModelInstance(i, model, resolved.ConcurrentRequestsPerModel);
            }
        }
        catch
        {
            foreach (var instance in _instances)
            {
                try { instance?.Model?.Dispose(); }
                catch { }
            }
            throw;
        }

        _scheduler = Task.Run(() => SchedulerLoopAsync(_shutdown.Token));
    }

    /// <summary>Convenience constructor for the common case where this runtime directly owns ONNX sessions.</summary>
    public OnnxModelRuntime(
        string modelPath,
        IOnnxModelExecutor<TRequest, TResponse> executor,
        OnnxModelRuntimeOptions? options = null,
        IInferenceFailureClassifier? failureClassifier = null,
        Action<Microsoft.ML.OnnxRuntime.SessionOptions, OnnxModelInstanceCreationContext>? configureSessionOptions = null)
        : this(
            new OnnxSessionModelInstanceFactory<TRequest, TResponse>(modelPath, executor, configureSessionOptions),
            options,
            failureClassifier)
    {
    }

    public int ModelInstanceCount { get; }
    public int ThreadsPerModel { get; }
    public int ConcurrentRequestsPerModel { get; }
    public int QueueCapacity { get; }
    public int TotalConcurrentRequests => ModelInstanceCount * ConcurrentRequestsPerModel;

    public OnnxModelRuntimeInfo GetRuntimeInfo()
    {
        lock (_gate)
        {
            var instances = _instances.Select(instance => new ModelInstanceRuntimeInfo(
                instance.Index,
                instance.Health,
                instance.ActiveRequests,
                instance.MaxConcurrentRequests,
                instance.Generation,
                instance.TotalRecoveries,
                instance.RecoveryAttempts,
                instance.LastFailure)).ToArray();

            return new OnnxModelRuntimeInfo(
                ModelInstanceCount,
                ThreadsPerModel,
                ConcurrentRequestsPerModel,
                QueueCapacity,
                Math.Max(0, Volatile.Read(ref _queuedRequests)),
                instances.Sum(instance => instance.ActiveRequests),
                instances.Count(instance => instance.Health == ModelInstanceHealth.Healthy),
                instances.Count(instance => instance.Health is ModelInstanceHealth.Draining or ModelInstanceHealth.Recovering or ModelInstanceHealth.Faulted),
                instances);
        }
    }

    public async Task<TResponse> RunAsync(TRequest request, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        var completion = new TaskCompletionSource<TResponse>(TaskCreationOptions.RunContinuationsAsynchronously);
        var work = new WorkItem(request, completion, cancellationToken);
        await EnqueueAsync(work, cancellationToken).ConfigureAwait(false);
        return await completion.Task.WaitAsync(cancellationToken).ConfigureAwait(false);
    }

    private async ValueTask EnqueueAsync(WorkItem work, CancellationToken cancellationToken)
    {
        try
        {
            await _channel.Writer.WriteAsync(work, cancellationToken).ConfigureAwait(false);
            Interlocked.Increment(ref _queuedRequests);
        }
        catch (ChannelClosedException ex)
        {
            throw new ObjectDisposedException(nameof(OnnxModelRuntime<TRequest, TResponse>), ex.Message);
        }
    }
}
