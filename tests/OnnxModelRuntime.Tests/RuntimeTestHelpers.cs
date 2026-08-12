using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxModelRuntime.Tests;

public sealed partial class RuntimeTests
{
    private static global::OnnxModelRuntime.OnnxModelRuntime<TRequest, TResponse> CreateRuntime<TRequest, TResponse>(
        IOnnxModelInstanceFactory<TRequest, TResponse> factory,
        int modelCount,
        int concurrency,
        int queueCapacity = 16,
        IInferenceFailureClassifier? classifier = null) => new(
            factory,
            new OnnxModelRuntimeOptions
            {
                ModelInstanceCount = modelCount,
                ThreadsPerModel = 2,
                ConcurrentRequestsPerModel = concurrency,
                QueueCapacity = queueCapacity
            },
            classifier);

    private static async Task WaitUntilAsync(Func<bool> predicate)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(8));
        while (!predicate())
            await Task.Delay(10, timeout.Token);
    }

    private sealed record ClassificationInput(string Text);
    private sealed record ClassificationResult(string Label, int Score);

    private sealed class MulModelExecutor : IOnnxModelExecutor<float[], float[]>
    {
        public float[] Execute(InferenceSession session, float[] request, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var input = NamedOnnxValue.CreateFromTensor("X", new DenseTensor<float>(request, new[] { 3, 2 }));
            using var results = session.Run([input]);
            return results.Single().AsEnumerable<float>().ToArray();
        }
    }

    private sealed class InvalidShapeMulModelExecutor : IOnnxModelExecutor<float[], float[]>
    {
        public float[] Execute(InferenceSession session, float[] request, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var input = NamedOnnxValue.CreateFromTensor("X", new DenseTensor<float>(request, new[] { 1, 6 }));
            using var results = session.Run([input]);
            return results.Single().AsEnumerable<float>().ToArray();
        }
    }

    private sealed class FakeFactory<TRequest, TResponse>(
        Func<OnnxModelInstanceCreationContext, CancellationToken, FakeModel<TRequest, TResponse>> create)
        : IOnnxModelInstanceFactory<TRequest, TResponse>
    {
        private readonly ConcurrentQueue<FakeModel<TRequest, TResponse>> _models = new();
        private int _createCount;

        public int CreateCount => Volatile.Read(ref _createCount);
        public IReadOnlyList<FakeModel<TRequest, TResponse>> Models => _models.ToArray();

        public IOnnxModelInstance<TRequest, TResponse> Create(OnnxModelInstanceCreationContext context, CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref _createCount);
            var model = create(context, cancellationToken);
            _models.Enqueue(model);
            return model;
        }
    }

    private sealed class FakeModel<TRequest, TResponse>(
        int index,
        Func<TRequest, CancellationToken, TResponse> execute) : IOnnxModelInstance<TRequest, TResponse>
    {
        private int _runCount;
        private int _disposeCount;
        private int _active;
        private int _maxObserved;

        public int Index { get; } = index;
        public int RunCount => Volatile.Read(ref _runCount);
        public int DisposeCount => Volatile.Read(ref _disposeCount);
        public int MaxObservedConcurrentExecutions => Volatile.Read(ref _maxObserved);

        public TResponse Execute(TRequest request, CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref _runCount);
            var active = Interlocked.Increment(ref _active);
            UpdateMaximum(active);
            try { return execute(request, cancellationToken); }
            finally { Interlocked.Decrement(ref _active); }
        }

        private void UpdateMaximum(int value)
        {
            while (true)
            {
                var current = Volatile.Read(ref _maxObserved);
                if (value <= current) return;
                if (Interlocked.CompareExchange(ref _maxObserved, value, current) == current) return;
            }
        }

        public void Dispose() => Interlocked.Increment(ref _disposeCount);
    }

    private sealed class RecoverableTestException(string message) : Exception(message);
    private sealed class MemoryPressureTestException(string message) : Exception(message);
    private sealed class FatalTestException(string message) : Exception(message);

    private sealed class TestFailureClassifier : IInferenceFailureClassifier
    {
        public static TestFailureClassifier Instance { get; } = new();
        public InferenceFailureKind Classify(Exception exception) => exception switch
        {
            RecoverableTestException => InferenceFailureKind.RecoverableInstance,
            MemoryPressureTestException => InferenceFailureKind.MemoryPressure,
            FatalTestException => InferenceFailureKind.Fatal,
            _ => InferenceFailureKind.Application
        };
    }
}
