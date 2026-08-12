namespace OnnxModelRuntime;

public static class OnnxModelRuntimeDefaults
{
    public const int ThreadsPerModel = 16;
    public const int MaximumAutoThreadsPerModel = 16;
    public const int AutomaticConcurrentRequestsPerModelCap = 8;
    public const int QueueCapacity = 256;
}

public sealed record ResolvedOnnxModelRuntimeOptions(
    int ModelInstanceCount,
    int ThreadsPerModel,
    int ConcurrentRequestsPerModel,
    int QueueCapacity)
{
    public int TotalConcurrentRequests => ModelInstanceCount * ConcurrentRequestsPerModel;
}

public sealed class OnnxModelRuntimeOptions
{
    /// <summary>Number of independent model instances/sessions held in memory.</summary>
    public int ModelInstanceCount { get; set; } = 1;

    /// <summary>ONNX Runtime intra-op threads per model instance. Zero enables hardware-based automatic resolution.</summary>
    public int ThreadsPerModel { get; set; } = OnnxModelRuntimeDefaults.ThreadsPerModel;

    /// <summary>Maximum threads/model used only when <see cref="ThreadsPerModel"/> is zero.</summary>
    public int MaximumAutoThreadsPerModel { get; set; } = OnnxModelRuntimeDefaults.MaximumAutoThreadsPerModel;

    /// <summary>
    /// Simultaneous inference calls allowed per model instance. Zero means automatic:
    /// max(1, min(ThreadsPerModel / 2, 8)). Explicit positive values are honored as-is.
    /// </summary>
    public int ConcurrentRequestsPerModel { get; set; }

    /// <summary>Maximum number of requests held by the global bounded channel before producers asynchronously wait.</summary>
    public int QueueCapacity { get; set; } = OnnxModelRuntimeDefaults.QueueCapacity;

    public ResolvedOnnxModelRuntimeOptions Resolve()
    {
        Validate();
        var threads = ThreadsPerModel > 0
            ? ThreadsPerModel
            : Math.Max(1, Math.Min(MaximumAutoThreadsPerModel, Environment.ProcessorCount / ModelInstanceCount));

        var concurrency = ConcurrentRequestsPerModel > 0
            ? ConcurrentRequestsPerModel
            : Math.Clamp(threads / 2, 1, OnnxModelRuntimeDefaults.AutomaticConcurrentRequestsPerModelCap);

        return new ResolvedOnnxModelRuntimeOptions(ModelInstanceCount, threads, concurrency, QueueCapacity);
    }

    private void Validate()
    {
        if (ModelInstanceCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(ModelInstanceCount));
        if (ThreadsPerModel < 0)
            throw new ArgumentOutOfRangeException(nameof(ThreadsPerModel));
        if (MaximumAutoThreadsPerModel <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaximumAutoThreadsPerModel));
        if (ConcurrentRequestsPerModel < 0)
            throw new ArgumentOutOfRangeException(nameof(ConcurrentRequestsPerModel));
        if (QueueCapacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(QueueCapacity));
    }
}
