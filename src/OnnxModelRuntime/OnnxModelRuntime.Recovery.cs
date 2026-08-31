namespace OnnxModelRuntime;

public sealed partial class OnnxModelRuntime<TRequest, TResponse>
{
    private void BeginRecovery(ModelInstance instance, Exception exception, bool memoryPressure)
    {
        var start = false;
        lock (_gate)
        {
            instance.LastFailure = exception.GetBaseException().Message;
            if (instance.Health == ModelInstanceHealth.Healthy && !IsDisposed)
            {
                instance.Health = ModelInstanceHealth.Draining;
                instance.PermanentlyFaulted = false;
                instance.MemoryPressureRecovery = memoryPressure;
                instance.Drained = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                if (instance.ActiveRequests == 0)
                    instance.Drained.TrySetResult(true);
                start = true;
            }
            else if (instance.Health == ModelInstanceHealth.Draining && memoryPressure && !IsDisposed)
            {
                // Multiple requests can fail against the same still-active generation.
                // Memory-pressure recovery must dominate an earlier ordinary rebuild.
                instance.MemoryPressureRecovery = true;
            }
        }

        if (!start)
            return;

        var task = RecoverInstanceAsync(instance, _shutdown.Token);
        lock (_gate)
            instance.RecoveryTask = task;
        _ = ObserveLifecycleTaskAsync(task);
    }

    private void BeginPermanentFault(ModelInstance instance, Exception exception)
    {
        var start = false;
        lock (_gate)
        {
            instance.LastFailure = exception.GetBaseException().Message;
            if (instance.Health == ModelInstanceHealth.Healthy && !IsDisposed)
            {
                instance.Health = ModelInstanceHealth.Draining;
                instance.PermanentlyFaulted = true;
                instance.Drained = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                if (instance.ActiveRequests == 0)
                    instance.Drained.TrySetResult(true);
                start = true;
            }
            else if (instance.Health == ModelInstanceHealth.Draining && !IsDisposed)
            {
                // A later fatal failure from the same generation must dominate an
                // earlier recoverable or memory-pressure transition.
                instance.PermanentlyFaulted = true;
            }
        }

        if (!start)
            return;

        var task = PermanentlyFaultInstanceAsync(instance, _shutdown.Token);
        lock (_gate)
            instance.RecoveryTask = task;
        _ = ObserveLifecycleTaskAsync(task);
    }

    private static async Task ObserveLifecycleTaskAsync(Task task)
    {
        try { await task.ConfigureAwait(false); }
        catch (OperationCanceledException) { }
        catch { }
    }

    private async Task RecoverInstanceAsync(ModelInstance instance, CancellationToken cancellationToken)
    {
        Task drainedTask;
        lock (_gate)
            drainedTask = instance.Drained?.Task ?? Task.CompletedTask;
        await drainedTask.WaitAsync(cancellationToken).ConfigureAwait(false);

        IOnnxModelInstance<TRequest, TResponse>? oldModel;
        bool permanentlyFaulted;
        bool memoryPressure;
        lock (_gate)
        {
            permanentlyFaulted = instance.PermanentlyFaulted;
            memoryPressure = instance.MemoryPressureRecovery;
            instance.Health = permanentlyFaulted
                ? ModelInstanceHealth.Faulted
                : ModelInstanceHealth.Recovering;
            oldModel = instance.Model;
            instance.Model = null;
            if (permanentlyFaulted)
                instance.RecoveryAttempts = 0;
        }
        TryDispose(oldModel);

        if (permanentlyFaulted)
        {
            SignalCapacityChanged();
            return;
        }

        var attempt = 0;
        while (!cancellationToken.IsCancellationRequested)
        {
            attempt++;
            lock (_gate)
            {
                instance.Health = ModelInstanceHealth.Recovering;
                instance.RecoveryAttempts = attempt;
            }

            var delay = GetRecoveryDelay(attempt, memoryPressure);
            if (delay > TimeSpan.Zero)
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);

            try
            {
                int generation;
                lock (_gate)
                    generation = instance.Generation + 1;

                var fresh = await Task.Run(
                    () => _factory.Create(
                        new OnnxModelInstanceCreationContext(instance.Index, generation, ThreadsPerModel),
                        cancellationToken),
                    cancellationToken).ConfigureAwait(false);

                lock (_gate)
                {
                    if (IsDisposed)
                    {
                        TryDispose(fresh);
                        return;
                    }
                    instance.Model = fresh;
                    instance.Health = ModelInstanceHealth.Healthy;
                    instance.PermanentlyFaulted = false;
                    instance.MemoryPressureRecovery = false;
                    instance.Generation = generation;
                    instance.TotalRecoveries++;
                    instance.RecoveryAttempts = 0;
                    instance.LastFailure = null;
                    instance.Drained = null;
                }

                SignalCapacityChanged();
                return;
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex)
            {
                lock (_gate)
                {
                    instance.Health = ModelInstanceHealth.Faulted;
                    instance.LastFailure = ex.GetBaseException().Message;
                }
            }
        }
    }

    private async Task PermanentlyFaultInstanceAsync(ModelInstance instance, CancellationToken cancellationToken)
    {
        Task drainedTask;
        lock (_gate)
            drainedTask = instance.Drained?.Task ?? Task.CompletedTask;
        await drainedTask.WaitAsync(cancellationToken).ConfigureAwait(false);

        IOnnxModelInstance<TRequest, TResponse>? oldModel;
        lock (_gate)
        {
            oldModel = instance.Model;
            instance.Model = null;
            instance.Health = ModelInstanceHealth.Faulted;
            instance.RecoveryAttempts = 0;
        }
        TryDispose(oldModel);
        SignalCapacityChanged();
    }

    private void SignalCapacityChanged()
    {
        try { _capacitySignal.Release(); }
        catch (SemaphoreFullException) { }
        catch (ObjectDisposedException) { }
    }

    private static TimeSpan GetRecoveryDelay(int attempt, bool memoryPressure)
    {
        if (attempt <= 1)
            return memoryPressure ? TimeSpan.FromSeconds(1) : TimeSpan.Zero;
        var milliseconds = Math.Min(10_000, 250 * Math.Pow(2, Math.Min(attempt - 2, 6)));
        return TimeSpan.FromMilliseconds(milliseconds);
    }

    private static void TryDispose(IDisposable? value)
    {
        if (value is null) return;
        try { value.Dispose(); }
        catch { }
    }

    private bool IsDisposed => Volatile.Read(ref _disposeStarted) != 0;

    private void ThrowIfDisposed()
    {
        if (IsDisposed)
            throw new ObjectDisposedException(nameof(OnnxModelRuntime<TRequest, TResponse>));
    }

    public async ValueTask DisposeAsync()
    {
        lock (_gate)
        {
            if (_disposeStarted != 0)
                return;
            Volatile.Write(ref _disposeStarted, 1);
            _channel.Writer.TryComplete();
        }

        _shutdown.Cancel();
        SignalCapacityChanged();

        try { await _scheduler.ConfigureAwait(false); }
        catch (OperationCanceledException) { }

        var inflight = _inflight.Values.ToArray();
        if (inflight.Length > 0)
        {
            try { await Task.WhenAll(inflight).ConfigureAwait(false); }
            catch { }
        }

        Task[] lifecycleTasks;
        lock (_gate)
            lifecycleTasks = _instances.Select(instance => instance.RecoveryTask).OfType<Task>().ToArray();
        if (lifecycleTasks.Length > 0)
        {
            try { await Task.WhenAll(lifecycleTasks).ConfigureAwait(false); }
            catch (OperationCanceledException) { }
            catch { }
        }

        lock (_gate)
        {
            foreach (var instance in _instances)
            {
                TryDispose(instance.Model);
                instance.Model = null;
                instance.Health = ModelInstanceHealth.Disposed;
                instance.ActiveRequests = 0;
            }
        }

        _capacitySignal.Dispose();
        _shutdown.Dispose();
    }
}
