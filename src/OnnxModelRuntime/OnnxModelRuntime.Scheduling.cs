using System.Threading.Channels;

namespace OnnxModelRuntime;

public sealed partial class OnnxModelRuntime<TRequest, TResponse>
{
    private async Task SchedulerLoopAsync(CancellationToken shutdownToken)
    {
        WorkItem? pending = null;
        try
        {
            await foreach (var work in _channel.Reader.ReadAllAsync(shutdownToken).ConfigureAwait(false))
            {
                Interlocked.Decrement(ref _queuedRequests);
                pending = work;
                if (work.CancellationToken.IsCancellationRequested)
                {
                    work.Completion.TrySetCanceled(work.CancellationToken);
                    pending = null;
                    continue;
                }

                ModelInstance? instance;
                while ((instance = TryReserveLeastLoaded()) is null)
                {
                    if (AllInstancesPermanentlyUnavailable())
                    {
                        work.Completion.TrySetException(new OnnxModelExecutionException(
                            "No model instance can accept work because every instance is permanently faulted.",
                            InferenceFailureKind.Fatal,
                            new InvalidOperationException("All model instances are permanently faulted.")));
                        pending = null;
                        break;
                    }

                    using var wait = CancellationTokenSource.CreateLinkedTokenSource(shutdownToken, work.CancellationToken);
                    try
                    {
                        await _capacitySignal.WaitAsync(wait.Token).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (work.CancellationToken.IsCancellationRequested && !shutdownToken.IsCancellationRequested)
                    {
                        work.Completion.TrySetCanceled(work.CancellationToken);
                        pending = null;
                        break;
                    }

                    if (work.CancellationToken.IsCancellationRequested)
                    {
                        work.Completion.TrySetCanceled(work.CancellationToken);
                        pending = null;
                        break;
                    }
                }

                if (pending is null || instance is null)
                    continue;

                StartExecution(instance, work);
                pending = null;
            }
        }
        catch (OperationCanceledException) when (shutdownToken.IsCancellationRequested)
        {
            if (pending is not null)
                pending.Completion.TrySetException(new ObjectDisposedException(nameof(OnnxModelRuntime<TRequest, TResponse>)));
        }
        finally
        {
            while (_channel.Reader.TryRead(out var queued))
            {
                Interlocked.Decrement(ref _queuedRequests);
                queued.Completion.TrySetException(new ObjectDisposedException(nameof(OnnxModelRuntime<TRequest, TResponse>)));
            }
        }
    }

    private bool AllInstancesPermanentlyUnavailable()
    {
        lock (_gate)
        {
            return _instances.All(instance =>
                instance.Health == ModelInstanceHealth.Disposed ||
                (instance.Health == ModelInstanceHealth.Faulted && instance.PermanentlyFaulted));
        }
    }

    private ModelInstance? TryReserveLeastLoaded()
    {
        lock (_gate)
        {
            var minimum = int.MaxValue;
            foreach (var instance in _instances)
            {
                if (instance.Health != ModelInstanceHealth.Healthy ||
                    instance.Model is null ||
                    instance.ActiveRequests >= instance.MaxConcurrentRequests)
                    continue;
                minimum = Math.Min(minimum, instance.ActiveRequests);
            }

            if (minimum == int.MaxValue)
                return null;

            for (var offset = 0; offset < _instances.Length; offset++)
            {
                var index = (_tieBreakerCursor + offset) % _instances.Length;
                var instance = _instances[index];
                if (instance.Health != ModelInstanceHealth.Healthy ||
                    instance.Model is null ||
                    instance.ActiveRequests >= instance.MaxConcurrentRequests ||
                    instance.ActiveRequests != minimum)
                    continue;

                instance.ActiveRequests++;
                _tieBreakerCursor = (index + 1) % _instances.Length;
                return instance;
            }

            return null;
        }
    }

    private void StartExecution(ModelInstance instance, WorkItem work)
    {
        var id = Interlocked.Increment(ref _nextWorkId);
        var task = Task.Run(() => ExecuteWorkAsync(instance, work));
        _inflight[id] = task;
        _ = ObserveCompletionAsync(id, task);
    }

    private async Task ObserveCompletionAsync(long id, Task task)
    {
        try { await task.ConfigureAwait(false); }
        finally { _inflight.TryRemove(id, out _); }
    }

    private async Task ExecuteWorkAsync(ModelInstance instance, WorkItem work)
    {
        var retry = false;
        try
        {
            if (work.CancellationToken.IsCancellationRequested)
            {
                work.Completion.TrySetCanceled(work.CancellationToken);
                return;
            }

            IOnnxModelInstance<TRequest, TResponse> model;
            lock (_gate)
                model = instance.Model ?? throw new InvalidOperationException("The selected model instance no longer has an executable model.");

            var response = model.Execute(work.Request, work.CancellationToken);
            work.Completion.TrySetResult(response);
        }
        catch (OperationCanceledException) when (work.CancellationToken.IsCancellationRequested)
        {
            work.Completion.TrySetCanceled(work.CancellationToken);
        }
        catch (Exception ex)
        {
            var kind = _failureClassifier.Classify(ex);
            switch (kind)
            {
                case InferenceFailureKind.Application:
                    work.Completion.TrySetException(ex);
                    break;

                case InferenceFailureKind.RecoverableInstance:
                    BeginRecovery(instance, ex, memoryPressure: false);
                    if (work.InfrastructureRetries < 1 && !work.CancellationToken.IsCancellationRequested && !IsDisposed)
                        retry = true;
                    else
                        work.Completion.TrySetException(new OnnxModelExecutionException(
                            "ONNX model execution failed after a recoverable model-instance failure.",
                            kind,
                            ex));
                    break;

                case InferenceFailureKind.MemoryPressure:
                    BeginRecovery(instance, ex, memoryPressure: true);
                    work.Completion.TrySetException(new OnnxModelExecutionException(
                        "ONNX model execution encountered memory pressure. The instance is being rebuilt and this request will not be immediately retried on another loaded copy.",
                        kind,
                        ex));
                    break;

                case InferenceFailureKind.Fatal:
                    BeginPermanentFault(instance, ex);
                    work.Completion.TrySetException(new OnnxModelExecutionException(
                        "ONNX model execution encountered a fatal nonrecoverable instance failure.",
                        kind,
                        ex));
                    break;

                default:
                    work.Completion.TrySetException(ex);
                    break;
            }
        }
        finally
        {
            ReleaseReservation(instance);
        }

        if (retry)
        {
            var retryWork = work with { InfrastructureRetries = work.InfrastructureRetries + 1 };
            try
            {
                await EnqueueAsync(retryWork, work.CancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (work.CancellationToken.IsCancellationRequested)
            {
                work.Completion.TrySetCanceled(work.CancellationToken);
            }
            catch (Exception ex) when (ex is ObjectDisposedException or ChannelClosedException)
            {
                work.Completion.TrySetException(new ObjectDisposedException(
                    nameof(OnnxModelRuntime<TRequest, TResponse>),
                    $"The recoverable inference retry could not be queued: {ex.Message}"));
            }
        }
    }

    private void ReleaseReservation(ModelInstance instance)
    {
        var signalCapacity = false;
        TaskCompletionSource<bool>? drained = null;
        lock (_gate)
        {
            instance.ActiveRequests = Math.Max(0, instance.ActiveRequests - 1);
            if (instance.Health == ModelInstanceHealth.Healthy)
                signalCapacity = true;
            else if (instance.ActiveRequests == 0)
                drained = instance.Drained;
        }

        drained?.TrySetResult(true);
        if (signalCapacity)
            SignalCapacityChanged();
    }
}
