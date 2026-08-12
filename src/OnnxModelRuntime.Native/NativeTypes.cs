using System.Runtime.InteropServices;

namespace OnnxModelRuntime.Native;

internal enum OmrStatus
{
    Ok = 0,
    InvalidArgument = 1,
    BufferTooSmall = 2,
    InvalidHandle = 3,
    ApplicationError = 4,
    RecoverableRuntimeError = 5,
    MemoryPressure = 6,
    FatalRuntimeError = 7,
    Disposed = 8,
    InternalError = 255
}

internal enum OmrExecutorStatus
{
    Ok = 0,
    ApplicationError = 1,
    RecoverableRuntimeError = 2,
    MemoryPressure = 3,
    FatalRuntimeError = 4
}

[StructLayout(LayoutKind.Sequential)]
internal struct OmrRuntimeOptions
{
    public uint StructSize;
    public uint AbiVersion;
    public int ModelInstanceCount;
    public int ThreadsPerModel;
    public int MaximumAutoThreadsPerModel;
    public int ConcurrentRequestsPerModel;
    public int QueueCapacity;
}

[StructLayout(LayoutKind.Sequential)]
internal struct OmrExecutor
{
    public uint StructSize;
    public uint AbiVersion;
    public nint UserData;
    public nint CreateInstance;
    public nint Execute;
    public nint DestroyInstance;
    public nint ReleaseResponse;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct OmrBufferView
{
    public byte* Data;
    public nuint Length;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct OmrBuffer
{
    public byte* Data;
    public nuint Length;
}

[StructLayout(LayoutKind.Sequential)]
internal struct OmrInstanceInfo
{
    public uint StructSize;
    public uint AbiVersion;
    public int Index;
    public int Health;
    public int ActiveRequests;
    public int MaxConcurrentRequests;
    public int Generation;
    public int TotalRecoveries;
    public int RecoveryAttempts;
}
