using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Tensorflow;

namespace AbaloneTensorFlow
{

    class Program
    {
        static readonly string _modelPath = Path.Combine("C:\\Users\\Albi\\source\\repos\\VS_ML\\AbaloneTensorFlow\\AbaloneTensorFlow\\", "abalone_model");

        public class TFInput
        {
            [VectorType(10)]
            [ColumnName("serving_default_dense_input")]
            public float[] Input { get; set; }
        }

        public class TFOutput
        {
            [VectorType(30)]
            [ColumnName("StatefulPartitionedCall")]
            public float[] Prediction { get; set; }
        }

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            TensorFlowModel tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);

            DataViewSchema schema = tensorFlowModel.GetModelSchema();
            Console.WriteLine(" =============== TensorFlow Model Schema =============== ");
            var featuresType = (VectorDataViewType)schema["serving_default_dense_input"].Type;
            Console.WriteLine($"Name: serving_default_dense_input, Type: {featuresType.ItemType.RawType}, Size: ({featuresType.Dimensions[0]})");
            var predictionType = (VectorDataViewType)schema["StatefulPartitionedCall"].Type;
            Console.WriteLine($"Name: StatefulPartitionedCall, Type: {predictionType.ItemType.RawType}, Size: ({predictionType.Dimensions[0]})");


            var inputColumns = new[] { "serving_default_dense_input" };
            var outputColumns = new[] { "StatefulPartitionedCall" };

            var testInput = mlContext.Data.LoadFromEnumerable(new TFInput[] { new TFInput { Input = new float[10] {
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
                } } });


            var model = mlContext.Model.LoadTensorFlowModel(_modelPath);
            var pipeline = model.ScoreTensorFlowModel(outputColumns, inputColumns);

            var estimator = pipeline.Fit(testInput);
            var transformedValues = estimator.Transform(testInput);

            var outScores = mlContext.Data.CreateEnumerable<TFOutput>(
                transformedValues, reuseRowObject: false);

            foreach (var pred in outScores)
            {
                if (pred.Prediction != null)
                {
                    int numClasses = 0;
                    foreach (var classScore in pred.Prediction.Take(30))
                    {
                        Console.WriteLine(
                            $"Class #{numClasses++} score = {classScore}");
                    }
                    Console.WriteLine(new string('-', 10));

                    
                    float m = pred.Prediction.Max();
                    Console.WriteLine($"Predicted age: {Array.IndexOf(pred.Prediction, m)}");
                }
                else { Console.WriteLine("Predicion is null"); };
            }

        }
    }
}