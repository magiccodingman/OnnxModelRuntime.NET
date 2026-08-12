using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxModelRuntime.Tests;


public sealed partial class RuntimeTests
{
    [Fact]
    public async Task SuccessfulRecovery_IncrementsGeneration()
    {
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (request, _) =>
        {
            if (context.Generation == 1) throw new RecoverableTestException("once");
            return request;
        }));
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1, classifier: TestFailureClassifier.Instance);

        Assert.Equal(4, await runtime.RunAsync(4));
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().Instances[0].Generation == 2);
        var info = runtime.GetRuntimeInfo().Instances[0];
        Assert.Equal(2, info.Generation);
        Assert.Equal(1, info.TotalRecoveries);
        Assert.Equal(0, info.RecoveryAttempts);
    }

    [Fact]
    public async Task FailedRecoveryAttempt_LeavesInstanceUnavailableAndRetries()
    {
        var failedRecoveryCreates = 0;
        var factory = new FakeFactory<int, int>((context, _) =>
        {
            if (context.Generation == 1)
                return new FakeModel<int, int>(0, (_, _) => throw new RecoverableTestException("initial"));
            if (Interlocked.Increment(ref failedRecoveryCreates) == 1)
                throw new InvalidOperationException("recreation failed");
            return new FakeModel<int, int>(0, (request, _) => request);
        });
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1, classifier: TestFailureClassifier.Instance);

        var request = runtime.RunAsync(9);
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().Instances[0] is { Health: ModelInstanceHealth.Faulted, RecoveryAttempts: >= 1 });
        Assert.Equal(0, runtime.GetRuntimeInfo().HealthyModelInstanceCount);
        Assert.Equal(9, await request);
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().Instances[0] is { Health: ModelInstanceHealth.Healthy, Generation: 2 });
        Assert.True(Volatile.Read(ref failedRecoveryCreates) >= 2);
    }

    [Fact]
    public async Task MemoryPressure_QuarantinesButDoesNotImmediatelyRetryElsewhere()
    {
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (request, _) =>
        {
            if (context.InstanceIndex == 0) throw new MemoryPressureTestException("oom");
            return request;
        }));
        await using var runtime = CreateRuntime(factory, modelCount: 2, concurrency: 1, classifier: TestFailureClassifier.Instance);

        var error = await Assert.ThrowsAsync<OnnxModelExecutionException>(() => runtime.RunAsync(1));
        Assert.Equal(InferenceFailureKind.MemoryPressure, error.FailureKind);
        Assert.Equal(0, factory.Models.First(model => model.Index == 1).RunCount);
    }

    [Fact]
    public void PartialStartupFailure_DisposesAlreadyCreatedInstances()
    {
        FakeModel<int, int>? first = null;
        var factory = new FakeFactory<int, int>((context, _) =>
        {
            if (context.InstanceIndex == 0)
                return first = new FakeModel<int, int>(0, (request, _) => request);
            throw new InvalidOperationException("second instance failed to start");
        });

        Assert.Throws<InvalidOperationException>(() => new global::OnnxModelRuntime.OnnxModelRuntime<int, int>(factory, new OnnxModelRuntimeOptions { ModelInstanceCount = 2 }));
        Assert.NotNull(first);
        Assert.Equal(1, first.DisposeCount);
    }

    [Fact]
    public async Task Disposal_CleansQueuedInflightRecoverySchedulerAndInstances()
    {
        using var executionGate = new ManualResetEventSlim(false);
        var recoveryCreateCanceled = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        var factory = new FakeFactory<int, int>((context, token) =>
        {
            if (context.InstanceIndex == 0 && context.Generation == 1)
                return new FakeModel<int, int>(0, (_, _) => throw new RecoverableTestException("recover me"));
            if (context.InstanceIndex == 0)
            {
                try
                {
                    token.WaitHandle.WaitOne();
                    token.ThrowIfCancellationRequested();
                }
                catch (OperationCanceledException)
                {
                    recoveryCreateCanceled.TrySetResult(true);
                    throw;
                }
            }
            return new FakeModel<int, int>(context.InstanceIndex, (request, _) =>
            {
                executionGate.Wait();
                return request;
            });
        });
        var runtime = CreateRuntime(factory, modelCount: 2, concurrency: 1, queueCapacity: 1, classifier: TestFailureClassifier.Instance);

        var first = runtime.RunAsync(1);
        await WaitUntilAsync(() =>
        {
            var info = runtime.GetRuntimeInfo();
            return info.Instances[0].Health != ModelInstanceHealth.Healthy && info.Instances[1].ActiveRequests == 1;
        });
        var queued = runtime.RunAsync(2);
        await Task.Delay(50);
        Assert.False(queued.IsCompleted);

        var dispose = runtime.DisposeAsync().AsTask();
        executionGate.Set();
        await dispose;
        await recoveryCreateCanceled.Task.WaitAsync(TimeSpan.FromSeconds(5));

        Assert.All(factory.Models, model => Assert.Equal(1, model.DisposeCount));
        Assert.All(runtime.GetRuntimeInfo().Instances, instance => Assert.Equal(ModelInstanceHealth.Disposed, instance.Health));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => queued);
        Assert.Equal(1, await first);
    }

}
