namespace OnnxModelRuntime.Tests;

public sealed partial class RuntimeTests
{
    [Fact]
    public void AutomaticThreadAndConcurrencyResolution_MatchesDocumentedBehavior()
    {
        var options = new OnnxModelRuntimeOptions
        {
            ModelInstanceCount = 2,
            ThreadsPerModel = 0,
            MaximumAutoThreadsPerModel = 12,
            ConcurrentRequestsPerModel = 0
        };

        var resolved = options.Resolve();
        var expectedThreads = Math.Max(1, Math.Min(12, Environment.ProcessorCount / 2));
        var expectedConcurrency = Math.Clamp(expectedThreads / 2, 1, 8);
        Assert.Equal(expectedThreads, resolved.ThreadsPerModel);
        Assert.Equal(expectedConcurrency, resolved.ConcurrentRequestsPerModel);
    }

    [Fact]
    public void DefaultFailureClassifier_TreatsOutOfMemoryAsMemoryPressure()
    {
        Assert.Equal(InferenceFailureKind.MemoryPressure, OnnxRuntimeFailureClassifier.Default.Classify(new OutOfMemoryException("simulated")));
        Assert.Equal(InferenceFailureKind.Application, OnnxRuntimeFailureClassifier.Default.Classify(new InvalidOperationException("model-specific")));
    }

    [Fact]
    public void ExplicitConcurrency_IsHonoredAboveAutomaticCap()
    {
        var resolved = new OnnxModelRuntimeOptions { ThreadsPerModel = 16, ConcurrentRequestsPerModel = 24 }.Resolve();
        Assert.Equal(24, resolved.ConcurrentRequestsPerModel);
    }

    [Fact]
    public async Task FatalFailure_PermanentlyRemovesInstanceAndDoesNotLeaveQueuedWorkHanging()
    {
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (_, _) => throw new FatalTestException("fatal")));
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1, classifier: TestFailureClassifier.Instance);

        var first = await Assert.ThrowsAsync<OnnxModelExecutionException>(() => runtime.RunAsync(1));
        Assert.Equal(InferenceFailureKind.Fatal, first.FailureKind);
        await WaitUntilAsync(() => runtime.GetRuntimeInfo().Instances[0].Health == ModelInstanceHealth.Faulted);

        var second = await Assert.ThrowsAsync<OnnxModelExecutionException>(() => runtime.RunAsync(2));
        Assert.Equal(InferenceFailureKind.Fatal, second.FailureKind);
        Assert.Equal(1, factory.CreateCount);
    }

    [Fact]
    public async Task ApplicationError_DoesNotTriggerSessionReconstruction()
    {
        var factory = new FakeFactory<int, int>((context, _) => new FakeModel<int, int>(context.InstanceIndex, (_, _) => throw new InvalidOperationException("bad input")));
        await using var runtime = CreateRuntime(factory, modelCount: 1, concurrency: 1, classifier: TestFailureClassifier.Instance);

        await Assert.ThrowsAsync<InvalidOperationException>(() => runtime.RunAsync(1));
        await Task.Delay(100);
        Assert.Equal(1, factory.CreateCount);
        Assert.Equal(ModelInstanceHealth.Healthy, runtime.GetRuntimeInfo().Instances[0].Health);
    }

    [Fact]
    public async Task RuntimeWorksWithNonEmbeddingRequestAndResponseTypes()
    {
        var factory = new FakeFactory<ClassificationInput, ClassificationResult>((context, _) =>
            new FakeModel<ClassificationInput, ClassificationResult>(context.InstanceIndex, (request, _) =>
                new ClassificationResult(request.Text.Length % 2 == 0 ? "even" : "odd", request.Text.Length)));
        await using var runtime = new global::OnnxModelRuntime.OnnxModelRuntime<ClassificationInput, ClassificationResult>(factory, new OnnxModelRuntimeOptions());

        var result = await runtime.RunAsync(new ClassificationInput("hello"));
        Assert.Equal("odd", result.Label);
        Assert.Equal(5, result.Score);
    }

    [Fact]
    public async Task ActualOnnxRuntimeSession_CanBeHostedAndExecuted()
    {
        var modelPath = Path.Combine(AppContext.BaseDirectory, "assets", "mul_1.onnx");
        var executor = new MulModelExecutor();
        await using var runtime = new global::OnnxModelRuntime.OnnxModelRuntime<float[], float[]>(
            modelPath,
            executor,
            new OnnxModelRuntimeOptions { ModelInstanceCount = 1, ThreadsPerModel = 1, ConcurrentRequestsPerModel = 2, QueueCapacity = 8 });

        var result = await runtime.RunAsync([1f, 1f, 1f, 1f, 1f, 1f]);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, result);
    }
}
