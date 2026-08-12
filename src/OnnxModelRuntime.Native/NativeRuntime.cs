using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Text;
using OnnxModelRuntime;

namespace OnnxModelRuntime.Native;

internal static unsafe class NativeRuntime
{
    internal const uint AbiVersion = 1;
    private static readonly ConcurrentDictionary<nint, NativeRuntimeHost> Hosts = new();
    private static long _nextHandle;

    [ThreadStatic]
    private static string? _lastError;

    internal static string LastError => _lastError ?? string.Empty;
    internal static void ClearError() => _lastError = null;
    internal static void SetError(Exception exception) => _lastError = exception.GetBaseException().Message;
    internal static void SetError(string message) => _lastError = message;

    internal static nint AddHost(NativeRuntimeHost host)
    {
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        if (!Hosts.TryAdd(handle, host))
            throw new InvalidOperationException("Unable to allocate a native runtime handle.");
        return handle;
    }

    internal static NativeRuntimeHost GetHost(nint handle)
    {
        if (handle == 0 || !Hosts.TryGetValue(handle, out var host))
            throw new KeyNotFoundException("The native runtime handle is invalid or has already been destroyed.");
        return host;
    }

    internal static NativeRuntimeHost RemoveHost(nint handle)
    {
        if (handle == 0 || !Hosts.TryRemove(handle, out var host))
            throw new KeyNotFoundException("The native runtime handle is invalid or has already been destroyed.");
        return host;
    }

    internal static OmrStatus MapException(Exception exception) => exception switch
    {
        ArgumentException => OmrStatus.InvalidArgument,
        ObjectDisposedException => OmrStatus.Disposed,
        KeyNotFoundException => OmrStatus.InvalidHandle,
        NativeExecutorFailureException failure => MapFailure(failure.Kind),
        OnnxModelExecutionException execution => MapFailure(execution.FailureKind),
        OutOfMemoryException => OmrStatus.MemoryPressure,
        _ => OmrStatus.InternalError
    };

    internal static OmrStatus MapFailure(InferenceFailureKind kind) => kind switch
    {
        InferenceFailureKind.Application => OmrStatus.ApplicationError,
        InferenceFailureKind.RecoverableInstance => OmrStatus.RecoverableRuntimeError,
        InferenceFailureKind.MemoryPressure => OmrStatus.MemoryPressure,
        InferenceFailureKind.Fatal => OmrStatus.FatalRuntimeError,
        _ => OmrStatus.InternalError
    };

    internal static byte[] ReadBytes(byte* data, nuint length)
    {
        if (length == 0) return [];
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (length > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(length));
        return new ReadOnlySpan<byte>(data, checked((int)length)).ToArray();
    }

    internal static void WriteBuffer(ReadOnlySpan<byte> bytes, OmrBuffer* output)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        output->Data = null;
        output->Length = 0;
        if (bytes.IsEmpty) return;
        var memory = (byte*)Marshal.AllocHGlobal(bytes.Length);
        bytes.CopyTo(new Span<byte>(memory, bytes.Length));
        output->Data = memory;
        output->Length = (nuint)bytes.Length;
    }

    internal static void FreeBuffer(OmrBuffer* buffer)
    {
        if (buffer is null) return;
        if (buffer->Data is not null)
            Marshal.FreeHGlobal((nint)buffer->Data);
        buffer->Data = null;
        buffer->Length = 0;
    }
}

internal sealed unsafe class NativeRuntimeHost : IAsyncDisposable
{
    private readonly global::OnnxModelRuntime.OnnxModelRuntime<byte[], byte[]> _runtime;

    public NativeRuntimeHost(OmrRuntimeOptions options, OmrExecutor executor)
    {
        var managed = new OnnxModelRuntimeOptions
        {
            ModelInstanceCount = options.ModelInstanceCount > 0 ? options.ModelInstanceCount : 1,
            ThreadsPerModel = options.ThreadsPerModel,
            MaximumAutoThreadsPerModel = options.MaximumAutoThreadsPerModel > 0
                ? options.MaximumAutoThreadsPerModel
                : OnnxModelRuntimeDefaults.MaximumAutoThreadsPerModel,
            ConcurrentRequestsPerModel = options.ConcurrentRequestsPerModel,
            QueueCapacity = options.QueueCapacity > 0 ? options.QueueCapacity : OnnxModelRuntimeDefaults.QueueCapacity
        };
        _runtime = new global::OnnxModelRuntime.OnnxModelRuntime<byte[], byte[]>(
            new NativeCallbackFactory(executor),
            managed,
            NativeFailureClassifier.Instance);
    }

    public byte[] Execute(byte[] request) => _runtime.RunAsync(request).GetAwaiter().GetResult();
    public OnnxModelRuntimeInfo GetRuntimeInfo() => _runtime.GetRuntimeInfo();
    public ValueTask DisposeAsync() => _runtime.DisposeAsync();
}

internal sealed class NativeFailureClassifier : IInferenceFailureClassifier
{
    public static NativeFailureClassifier Instance { get; } = new();

    public InferenceFailureKind Classify(Exception exception) => exception is NativeExecutorFailureException native
        ? native.Kind
        : OnnxRuntimeFailureClassifier.Default.Classify(exception);
}

internal sealed class NativeExecutorFailureException(
    InferenceFailureKind kind,
    string message) : Exception(message)
{
    public InferenceFailureKind Kind { get; } = kind;
}

internal sealed unsafe class NativeCallbackFactory(OmrExecutor executor) : IOnnxModelInstanceFactory<byte[], byte[]>
{
    private readonly OmrExecutor _executor = executor;

    public IOnnxModelInstance<byte[], byte[]> Create(
        OnnxModelInstanceCreationContext context,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        nint instanceContext = _executor.UserData;
        if (_executor.CreateInstance != 0)
        {
            var create = (delegate* unmanaged[Cdecl]<nint, int, int, int, nint*, int>)_executor.CreateInstance;
            var status = create(
                _executor.UserData,
                context.InstanceIndex,
                context.Generation,
                context.ThreadsPerModel,
                &instanceContext);
            if (status != (int)OmrExecutorStatus.Ok)
                throw new NativeExecutorFailureException(MapExecutorStatus(status), $"Native create-instance callback returned status {status}.");
        }
        return new NativeCallbackInstance(_executor, instanceContext);
    }

    private static InferenceFailureKind MapExecutorStatus(int status) => status switch
    {
        (int)OmrExecutorStatus.ApplicationError => InferenceFailureKind.Application,
        (int)OmrExecutorStatus.RecoverableRuntimeError => InferenceFailureKind.RecoverableInstance,
        (int)OmrExecutorStatus.MemoryPressure => InferenceFailureKind.MemoryPressure,
        (int)OmrExecutorStatus.FatalRuntimeError => InferenceFailureKind.Fatal,
        _ => InferenceFailureKind.Fatal
    };

    private sealed unsafe class NativeCallbackInstance(OmrExecutor executor, nint instanceContext) : IOnnxModelInstance<byte[], byte[]>
    {
        private int _disposed;

        public byte[] Execute(byte[] request, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(NativeCallbackInstance));
            if (executor.Execute == 0)
                throw new InvalidOperationException("The native executor does not provide an execute callback.");

            OmrBufferView response = default;
            int status;
            fixed (byte* requestPtr = request)
            {
                var execute = (delegate* unmanaged[Cdecl]<nint, byte*, nuint, OmrBufferView*, int>)executor.Execute;
                status = execute(instanceContext, requestPtr, (nuint)request.Length, &response);
            }

            byte[] copied;
            try
            {
                if (response.Length > int.MaxValue)
                    throw new InvalidOperationException("The native executor returned a response larger than the managed ABI can copy.");
                if (response.Length > 0 && response.Data is null)
                    throw new InvalidOperationException("The native executor returned a null response pointer with a non-zero length.");
                copied = response.Length == 0
                    ? []
                    : new ReadOnlySpan<byte>(response.Data, checked((int)response.Length)).ToArray();
            }
            finally
            {
                if (response.Data is not null && executor.ReleaseResponse != 0)
                {
                    var release = (delegate* unmanaged[Cdecl]<nint, byte*, nuint, void>)executor.ReleaseResponse;
                    release(instanceContext, response.Data, response.Length);
                }
            }

            if (status == (int)OmrExecutorStatus.Ok)
                return copied;

            var message = copied.Length == 0
                ? $"Native execute callback returned status {status}."
                : Encoding.UTF8.GetString(copied);
            throw new NativeExecutorFailureException(MapExecutorStatus(status), message);
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            if (executor.DestroyInstance == 0)
                return;
            var destroy = (delegate* unmanaged[Cdecl]<nint, void>)executor.DestroyInstance;
            destroy(instanceContext);
        }
    }
}
