namespace OnnxModelRuntime;

public enum ModelInstanceHealth
{
    Starting = 0,
    Healthy = 1,
    Draining = 2,
    Recovering = 3,
    Faulted = 4,
    Disposed = 5
}

public sealed record ModelInstanceRuntimeInfo(
    int Index,
    ModelInstanceHealth Health,
    int ActiveRequests,
    int MaxConcurrentRequests,
    int Generation,
    int TotalRecoveries,
    int RecoveryAttempts,
    string? LastFailure);

public sealed record OnnxModelRuntimeInfo(
    int ModelInstanceCount,
    int ThreadsPerModel,
    int ConcurrentRequestsPerModel,
    int QueueCapacity,
    int QueuedRequests,
    int ActiveRequests,
    int HealthyModelInstanceCount,
    int RecoveringModelInstanceCount,
    IReadOnlyList<ModelInstanceRuntimeInfo> Instances)
{
    public int TotalConcurrentRequests => ModelInstanceCount * ConcurrentRequestsPerModel;
}
