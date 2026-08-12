using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using OnnxModelRuntime;

namespace OnnxModelRuntime.Native;

internal static unsafe class NativeExports
{
    [UnmanagedCallersOnly(EntryPoint = "omr_abi_version", CallConvs = [typeof(CallConvCdecl)])]
    public static uint AbiVersion() => NativeRuntime.AbiVersion;

    [UnmanagedCallersOnly(EntryPoint = "omr_get_last_error", CallConvs = [typeof(CallConvCdecl)])]
    public static int GetLastError(byte* buffer, nuint bufferLength, nuint* requiredLength)
    {
        try
        {
            var bytes = Encoding.UTF8.GetBytes(NativeRuntime.LastError);
            if (requiredLength is not null)
                *requiredLength = (nuint)bytes.Length;
            if (bytes.Length == 0)
                return (int)OmrStatus.Ok;
            if (buffer is null || bufferLength < (nuint)bytes.Length)
                return (int)OmrStatus.BufferTooSmall;
            bytes.CopyTo(new Span<byte>(buffer, bytes.Length));
            return (int)OmrStatus.Ok;
        }
        catch
        {
            return (int)OmrStatus.InternalError;
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "omr_runtime_create", CallConvs = [typeof(CallConvCdecl)])]
    public static int RuntimeCreate(OmrRuntimeOptions* options, OmrExecutor* executor, nint* outputHandle)
    {
        NativeRuntime.ClearError();
        NativeRuntimeHost? host = null;
        try
        {
            if (outputHandle is null)
                throw new ArgumentNullException(nameof(outputHandle));
            *outputHandle = 0;
            if (executor is null)
                throw new ArgumentNullException(nameof(executor));

            ValidateVersionedStruct(executor->StructSize, executor->AbiVersion, (uint)sizeof(OmrExecutor), "omr_executor_v1");
            if (executor->Execute == 0)
                throw new ArgumentException("omr_executor_v1.execute must be provided.");

            OmrRuntimeOptions resolved;
            if (options is null)
            {
                resolved = new OmrRuntimeOptions
                {
                    StructSize = (uint)sizeof(OmrRuntimeOptions),
                    AbiVersion = NativeRuntime.AbiVersion,
                    ModelInstanceCount = 1,
                    ThreadsPerModel = OnnxModelRuntimeDefaults.ThreadsPerModel,
                    MaximumAutoThreadsPerModel = OnnxModelRuntimeDefaults.MaximumAutoThreadsPerModel,
                    ConcurrentRequestsPerModel = 0,
                    QueueCapacity = OnnxModelRuntimeDefaults.QueueCapacity
                };
            }
            else
            {
                ValidateVersionedStruct(options->StructSize, options->AbiVersion, (uint)sizeof(OmrRuntimeOptions), "omr_runtime_options_v1");
                resolved = *options;
            }

            host = new NativeRuntimeHost(resolved, *executor);
            *outputHandle = NativeRuntime.AddHost(host);
            host = null;
            return (int)OmrStatus.Ok;
        }
        catch (Exception ex)
        {
            if (host is not null)
            {
                try { host.DisposeAsync().AsTask().GetAwaiter().GetResult(); }
                catch { }
            }
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "omr_runtime_destroy", CallConvs = [typeof(CallConvCdecl)])]
    public static int RuntimeDestroy(nint handle)
    {
        NativeRuntime.ClearError();
        try
        {
            if (handle == 0)
                return (int)OmrStatus.Ok;
            NativeRuntime.RemoveHost(handle).DisposeAsync().AsTask().GetAwaiter().GetResult();
            return (int)OmrStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "omr_runtime_execute", CallConvs = [typeof(CallConvCdecl)])]
    public static int RuntimeExecute(nint handle, byte* request, nuint requestLength, OmrBuffer* output)
    {
        NativeRuntime.ClearError();
        try
        {
            if (output is null)
                throw new ArgumentNullException(nameof(output));
            output->Data = null;
            output->Length = 0;
            var result = NativeRuntime.GetHost(handle).Execute(NativeRuntime.ReadBytes(request, requestLength));
            NativeRuntime.WriteBuffer(result, output);
            return (int)OmrStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "omr_runtime_get_instance_count", CallConvs = [typeof(CallConvCdecl)])]
    public static int RuntimeGetInstanceCount(nint handle, int* outputCount)
    {
        NativeRuntime.ClearError();
        try
        {
            if (outputCount is null)
                throw new ArgumentNullException(nameof(outputCount));
            *outputCount = NativeRuntime.GetHost(handle).GetRuntimeInfo().ModelInstanceCount;
            return (int)OmrStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "omr_runtime_get_instance_info", CallConvs = [typeof(CallConvCdecl)])]
    public static int RuntimeGetInstanceInfo(nint handle, int instanceIndex, OmrInstanceInfo* output)
    {
        NativeRuntime.ClearError();
        try
        {
            if (output is null)
                throw new ArgumentNullException(nameof(output));
            ValidateVersionedStruct(output->StructSize, output->AbiVersion, (uint)sizeof(OmrInstanceInfo), "omr_instance_info_v1");
            var runtime = NativeRuntime.GetHost(handle).GetRuntimeInfo();
            if ((uint)instanceIndex >= (uint)runtime.Instances.Count)
                throw new ArgumentOutOfRangeException(nameof(instanceIndex));
            var info = runtime.Instances[instanceIndex];
            *output = new OmrInstanceInfo
            {
                StructSize = (uint)sizeof(OmrInstanceInfo),
                AbiVersion = NativeRuntime.AbiVersion,
                Index = info.Index,
                Health = (int)info.Health,
                ActiveRequests = info.ActiveRequests,
                MaxConcurrentRequests = info.MaxConcurrentRequests,
                Generation = info.Generation,
                TotalRecoveries = info.TotalRecoveries,
                RecoveryAttempts = info.RecoveryAttempts
            };
            return (int)OmrStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "omr_runtime_get_instance_last_failure", CallConvs = [typeof(CallConvCdecl)])]
    public static int RuntimeGetInstanceLastFailure(nint handle, int instanceIndex, OmrBuffer* output)
    {
        NativeRuntime.ClearError();
        try
        {
            if (output is null)
                throw new ArgumentNullException(nameof(output));
            var runtime = NativeRuntime.GetHost(handle).GetRuntimeInfo();
            if ((uint)instanceIndex >= (uint)runtime.Instances.Count)
                throw new ArgumentOutOfRangeException(nameof(instanceIndex));
            var text = runtime.Instances[instanceIndex].LastFailure ?? string.Empty;
            NativeRuntime.WriteBuffer(Encoding.UTF8.GetBytes(text), output);
            return (int)OmrStatus.Ok;
        }
        catch (Exception ex)
        {
            NativeRuntime.SetError(ex);
            return (int)NativeRuntime.MapException(ex);
        }
    }

    [UnmanagedCallersOnly(EntryPoint = "omr_buffer_free", CallConvs = [typeof(CallConvCdecl)])]
    public static void BufferFree(OmrBuffer* buffer)
    {
        try { NativeRuntime.FreeBuffer(buffer); }
        catch { }
    }

    private static void ValidateVersionedStruct(uint structSize, uint abiVersion, uint requiredSize, string name)
    {
        if (abiVersion != NativeRuntime.AbiVersion)
            throw new ArgumentException($"{name}.abi_version must be {NativeRuntime.AbiVersion}.");
        if (structSize < requiredSize)
            throw new ArgumentException($"{name}.struct_size is smaller than the ABI v1 structure.");
    }
}
