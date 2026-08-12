using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxModelRuntime.Tests;

public sealed partial class RuntimeTests
{
    [Fact]
    public async Task TwoIdleInstances_ReceiveOneRequestEachBeforeEitherGetsASecond()
    {
        using var gate = new ManualResetEventSlim(false);
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (request, _) =>
        {
            gate.Wait();
            return request;
        }));
        await using var runtime = CreateRuntime(factory, modelCount: 2, concurrency: 2);
        var first = runtime.RunAsync(1);
        var second = runtime.RunAsync(2);
        await WaitUntilAsync(() => factory.Models.Count >= 2 && factory.Models.Take(2).All(model => model.RunCount >= 1));
        Assert.Equal(1, factory.Models[0].RunCount);
        Assert.Equal(1, factory.Models[1].RunCount);
        gate.Set();
        await Task.WhenAll(first, second);
    }

    [Fact]
    public async Task SeveralInstances_UseLeastLoadedRouting()
    {
        using var gate = new ManualResetEventSlim(false);
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (request, _) => { gate.Wait(); return request; }));
        await using var runtime = CreateRuntime(factory, modelCount: 3, concurrency: 3);
        var tasks = Enumerable.Range(0, 5).Select(runtime.RunAsync).ToArray();
        await WaitUntilAsync(() => factory.Models.Take(3).Sum(model => model.RunCount) == 5);
        Assert.Equal(new[] { 1, 2, 2 }, factory.Models.Take(3).Select(model => model.RunCount).Order().ToArray());
        gate.Set();
        await Task.WhenAll(tasks);
    }

    [Fact]
    public async Task SingleInstance_PermitsConcurrentRequestsUpToConfiguredMaximum()
    {
        using var gate = new ManualResetEventSlim(false);
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (request, _) => { gate.Wait(); return request; }));
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 3);
        var tasks = Enumerable.Range(0, 4).Select(runtime.RunAsync).ToArray();
        await WaitUntilAsync(() => factory.Models[0].RunCount == 3);
        Assert.Equal(3, runtime.GetRuntimeInfo().ActiveRequests);
        Assert.Equal(3, factory.Models[0].MaxObservedConcurrentExecutions);
        gate.Set();
        await Task.WhenAll(tasks);
    }

    [Fact]
    public async Task RequestsBeyondInstanceCapacity_RemainQueued()
    {
        using var gate = new ManualResetEventSlim(false);
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (request, _) => { gate.Wait(); return request; }));
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1);
        var first = runtime.RunAsync(1);
        await WaitUntilAsync(() => factory.Models[0].RunCount == 1);
        var second = runtime.RunAsync(2);
        await Task.Delay(75);
        Assert.False(second.IsCompleted);
        Assert.Equal(1, factory.Models[0].RunCount);
        Assert.Equal(1, runtime.GetRuntimeInfo().ActiveRequests);
        gate.Set();
        await Task.WhenAll(first, second);
    }

    [Fact]
    public async Task QueueCapacity_ProducesBackpressure()
    {
        using var gate = new ManualResetEventSlim(false);
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (request, _) => { gate.Wait(); return request; }));
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1, queueCapacity: 1);
        var first = runtime.RunAsync(1);
        await WaitUntilAsync(() => factory.Models[0].RunCount == 1);
        var second = runtime.RunAsync(2);
        var third = runtime.RunAsync(3);
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().QueuedRequests == 1);
        using var cts = new CancellationTokenSource();
        var fourth = runtime.RunAsync(4, cts.Token);
        await Task.Delay(75);
        Assert.Equal(1, runtime.GetRuntimeInfo().QueuedRequests);
        Assert.False(fourth.IsCompleted);
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => fourth);
        gate.Set();
        await Task.WhenAll(first, second, third);
    }

    [Fact]
    public async Task CancellationWhileQueued_IsObservedWithoutWaitingForCapacityChange()
    {
        using var gate = new ManualResetEventSlim(false);
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (request, _) => { gate.Wait(); return request; }));
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1);
        var first = runtime.RunAsync(1);
        await WaitUntilAsync(() => factory.Models[0].RunCount == 1);
        using var cts = new CancellationTokenSource();
        var queued = runtime.RunAsync(2, cts.Token);
        await Task.Delay(50);
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => queued);
        Assert.Equal(1, runtime.GetRuntimeInfo().ActiveRequests);
        gate.Set();
        await first;
    }

    [Fact]
    public async Task Cancellation_DoesNotLeakActiveRequestSlot()
    {
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (request, token) =>
        {
            if (request == 1) { token.WaitHandle.WaitOne(); token.ThrowIfCancellationRequested(); }
            return request;
        }));
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1);
        using var cts = new CancellationTokenSource();
        var first = runtime.RunAsync(1, cts.Token);
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().ActiveRequests == 1);
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => first);
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().ActiveRequests == 0);
        Assert.Equal(2, await runtime.RunAsync(2));
        Assert.Equal(0, runtime.GetRuntimeInfo().ActiveRequests);
    }

    [Fact]
    public async Task OrdinaryExecutionException_DoesNotLeakSlot()
    {
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (request, _) => { if (request == 1) throw new InvalidOperationException("application failure"); return request; }));
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1);
        await Assert.ThrowsAsync<InvalidOperationException>(() => runtime.RunAsync(1));
        Assert.Equal(2, await runtime.RunAsync(2));
        Assert.Equal(0, runtime.GetRuntimeInfo().ActiveRequests);
    }
}
