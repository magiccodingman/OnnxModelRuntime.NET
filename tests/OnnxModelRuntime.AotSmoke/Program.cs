using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxModelRuntime;

var modelPath = Path.Combine(AppContext.BaseDirectory, "mul_1.onnx");
await using var runtime = new global::OnnxModelRuntime.OnnxModelRuntime<float[], float[]>(
    modelPath,
    new MulExecutor(),
    new OnnxModelRuntimeOptions
    {
        ModelInstanceCount = 1,
        ThreadsPerModel = 1,
        ConcurrentRequestsPerModel = 2,
        QueueCapacity = 8
    });

var input = new[] { 1f, 1f, 1f, 1f, 1f, 1f };
var results = await Task.WhenAll(runtime.RunAsync(input), runtime.RunAsync(input));
var expected = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
if (results.Any(result => !result.SequenceEqual(expected)))
    throw new InvalidOperationException("Native AOT ONNX Runtime smoke produced an unexpected result.");

var info = runtime.GetRuntimeInfo();
if (info.ModelInstanceCount != 1 || info.ConcurrentRequestsPerModel != 2)
    throw new InvalidOperationException("Native AOT runtime diagnostics were unexpected.");

Console.WriteLine("Native AOT ONNX Runtime smoke passed.");

file sealed class MulExecutor : IOnnxModelExecutor<float[], float[]>
{
    public float[] Execute(InferenceSession session, float[] request, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var input = NamedOnnxValue.CreateFromTensor("X", new DenseTensor<float>(request, new[] { 3, 2 }));
        using var results = session.Run([input]);
        return results.Single().AsEnumerable<float>().ToArray();
    }
}
