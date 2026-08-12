using Microsoft.ML.OnnxRuntime;

namespace OnnxModelRuntime;

public enum InferenceFailureKind
{
    Application = 0,
    RecoverableInstance = 1,
    MemoryPressure = 2,
    Fatal = 3
}

public interface IInferenceFailureClassifier
{
    InferenceFailureKind Classify(Exception exception);
}

/// <summary>Default classifier for ONNX Runtime failures. Consumers may replace it for model-specific behavior.</summary>
public sealed class OnnxRuntimeFailureClassifier : IInferenceFailureClassifier
{
    public static OnnxRuntimeFailureClassifier Default { get; } = new();

    public InferenceFailureKind Classify(Exception exception)
    {
        ArgumentNullException.ThrowIfNull(exception);
        if (exception is OutOfMemoryException)
            return InferenceFailureKind.MemoryPressure;
        if (exception is OnnxRuntimeException onnx)
            return LooksLikeMemoryPressure(onnx)
                ? InferenceFailureKind.MemoryPressure
                : InferenceFailureKind.RecoverableInstance;
        return InferenceFailureKind.Application;
    }

    private static bool LooksLikeMemoryPressure(Exception exception)
    {
        var message = exception.Message;
        return message.Contains("out of memory", StringComparison.OrdinalIgnoreCase) ||
               message.Contains("failed to allocate", StringComparison.OrdinalIgnoreCase) ||
               message.Contains("memory allocation", StringComparison.OrdinalIgnoreCase) ||
               message.Contains("not enough memory", StringComparison.OrdinalIgnoreCase);
    }
}

/// <summary>Failure reported by the runtime after applying its retry/recovery policy.</summary>
public sealed class OnnxModelExecutionException : Exception
{
    public OnnxModelExecutionException(string message, InferenceFailureKind failureKind, Exception innerException)
        : base(message, innerException) => FailureKind = failureKind;

    public InferenceFailureKind FailureKind { get; }
}
