using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxModelRuntime.Tests;

public sealed partial class RuntimeTests
{
    [Fact]
    public async Task RecoverableFailure_QuarantinesInstance()
    {
        using var recoveryGate = new ManualResetEventSlim(false);
        var factory = new FakeFactory<int, int>((context, token) =>
        {
            if (context.InstanceIndex == 0 && context.Generation == 1) return new FakeModel<int, int>(0, (_, _) => throw new RecoverableTestException("runtime failed"));
            if (context.InstanceIndex == 0) { recoveryGate.Wait(token); return new FakeModel<int, int>(0, (request, _) => request); }
            return new FakeModel<int, int>(1, (request, _) => request);
        });
        await using var runtime = CreateRuntime(factory, modelCount: 2, concurrency: 1, classifier: TestFailureClassifier.Instance);
        Assert.Equal(7, await runtime.RunAsync(7));
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().Instances[0].Health != ModelInstanceHealth.Healthy);
        Assert.Contains(runtime.GetRuntimeInfo().Instances[0].Health, new[] { ModelInstanceHealth.Draining, ModelInstanceHealth.Recovering, ModelInstanceHealth.Faulted });
        recoveryGate.Set();
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().Instances[0].Health == ModelInstanceHealth.Healthy);
    }

    [Fact]
    public async Task RecoverableRequest_RetriesThroughGlobalSchedulerAtMostOnce()
    {
        var executeCount = 0;
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (_, _) =>
        {
            Interlocked.Increment(ref executeCount);
            if (context.Generation <= 2) throw new RecoverableTestException($"failure generation {context.Generation}");
            return 1;
        }));
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1, classifier: TestFailureClassifier.Instance);
        var error = await Assert.ThrowsAsync<OnnxModelExecutionException>(() => runtime.RunAsync(1));
        Assert.Equal(InferenceFailureKind.RecoverableInstance, error.FailureKind);
        Assert.Equal(2, Volatile.Read(ref executeCount));
    }

    [Fact]
    public async Task HealthyInstance_ContinuesServingWhileAnotherRecovers()
    {
        using var recoveryGate = new ManualResetEventSlim(false);
        var factory = new FakeFactory<int, int>((context, token) =>
        {
            if (context.InstanceIndex == 0 && context.Generation == 1) return new FakeModel<int, int>(0, (_, _) => throw new RecoverableTestException("boom"));
            if (context.InstanceIndex == 0) { recoveryGate.Wait(token); return new FakeModel<int, int>(0, (request, _) => request); }
            return new FakeModel<int, int>(1, (request, _) => request * 10);
        });
        await using var runtime = CreateRuntime(factory, modelCount: 2, concurrency: 1, classifier: TestFailureClassifier.Instance);
        Assert.Equal(20, await runtime.RunAsync(2));
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().Instances[0].Health != ModelInstanceHealth.Healthy);
        Assert.Equal(30, await runtime.RunAsync(3));
        recoveryGate.Set();
    }

    [Fact]
    public async Task OneInstanceConfiguration_PausesQueuedWorkUntilRecoverySucceeds()
    {
        using var recoveryGate = new ManualResetEventSlim(false);
        var factory = new FakeFactory<int, int>((context, token) =>
        {
            if (context.Generation == 1) return new FakeModel<int, int>(0, (_, _) => throw new RecoverableTestException("boom"));
            recoveryGate.Wait(token);
            return new FakeModel<int, int>(0, (request, _) => request);
        });
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1, classifier: TestFailureClassifier.Instance);
        var first = runtime.RunAsync(1);
        var second = runtime.RunAsync(2);
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().Instances[0].Health != ModelInstanceHealth.Healthy);
        Assert.False(first.IsCompleted);
        Assert.False(second.IsCompleted);
        recoveryGate.Set();
        Assert.Equal(new[] { 1, 2 }, await Task.WhenAll(first, second));
    }
}
