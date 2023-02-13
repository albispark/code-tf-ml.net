using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;

class Program
{
    static string ONNX_MODEL_PATH = "C:\\Users\\Albi\\source\\repos\\VS_ML\\AbaloneONNX\\assets\\model_from_keras.onnx";

    public class OnnxInput
    {
        [VectorType(10)]
        [ColumnName("dense_input")]
        public float[] Input { get; set; }
    }

    public class OnnxOutput
    {
        [VectorType(30)]
        [ColumnName("dense_3")]
        public float[] Prediction { get; set; }
    }

    static ITransformer GetPredictionPipeline(MLContext mlContext)
    {
        var inputColumns = new [] { "dense_input" };

        var outputColumns = new [] { "dense_3" };

        var onnxPredictionPipeline =
            mlContext
                .Transforms
                .ApplyOnnxModel(
                    outputColumnNames: outputColumns,
                    inputColumnNames: inputColumns,
                    ONNX_MODEL_PATH);

        var emptyDv = mlContext.Data.LoadFromEnumerable(new OnnxInput[] { });

        return onnxPredictionPipeline.Fit(emptyDv);
    }

    static void Main(string[] args)
    {
        MLContext mlContext = new MLContext();

        var onnxPredictionPipeline = GetPredictionPipeline(mlContext);

        var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(onnxPredictionPipeline);

        var testInput = new OnnxInput
        {
            Input = new float[10] {
                0f,
                0f,
                1f,
                0.65705925F,
                0.45693547F,
                0.46309394F,
                0.5386008F,
                0.2557273F,
                1.0862371F,
                0.5909706F
            }
        };

        var prediction = onnxPredictionEngine.Predict(testInput);

        if (prediction.Prediction != null)
        {
            float m = prediction.Prediction.Max();
            Console.WriteLine($"Predicted age: {Array.IndexOf(prediction.Prediction, m)}");
        }
        else { Console.WriteLine("Predicion is null"); };

    }
} 