using System;
using System.Data;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers;

class Program
{
    public class AbaloneData
    {
        [ColumnName("Sex"), LoadColumn(0)]
        public string? Sex { get; set; }

        [ColumnName("Length"), LoadColumn(1)]
        public float Length { get; set; }

        [ColumnName("Diameter"), LoadColumn(2)]
        public float Diameter { get; set; }

        [ColumnName("Height"), LoadColumn(3)]
        public float Height { get; set; }

        [ColumnName("Whole weight"), LoadColumn(4)]
        public float Wholeweight { get; set; }

        [ColumnName("Shucked weight"), LoadColumn(5)]
        public float Shuckedweight { get; set; }

        [ColumnName("Viscera weight"), LoadColumn(6)]
        public float Visceraweight { get; set; }

        [ColumnName("Shell weight"), LoadColumn(7)]
        public float Shellweight { get; set; }

        [ColumnName("Rings"), LoadColumn(8)]
        public int Rings { get; set; }

    }


    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public int Rings { get; set; }
    }


    static void Main(string[] args)
    {
        MLContext mlContext = new MLContext(seed: 1);


        // 1. Import or create training data
        var data_path = "C:\\Users\\Albi\\Progetti\\Università\\Laurea\\data\\abalone.csv";
        IDataView data = mlContext.Data.LoadFromTextFile<AbaloneData>(
            path: data_path, 
            hasHeader: true, 
            separatorChar: ',', 
            allowQuoting: true, 
            allowSparse: false);

        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.3);


        // 2. Specify data preparation and model training pipeline
        // Perceptron trainer
        var averagedPerceptronOptions = new AveragedPerceptronTrainer.Options
        {
            LossFunction = new SmoothedHingeLoss(),
            LearningRate = 0.1f,
            LazyUpdate = false,
            RecencyGain = 0.1f,
            NumberOfIterations = 10,
            LabelColumnName = @"Rings",
            FeatureColumnName = @"Features"
        };
        var averagedPerceptron = mlContext.BinaryClassification.Trainers.AveragedPerceptron(averagedPerceptronOptions);

        // Model Builder's best trainer
        var fastTreeOptions = new FastTreeBinaryTrainer.Options()
        {
            NumberOfLeaves = 40,
            MinimumExampleCountPerLeaf = 62,
            NumberOfTrees = 5,
            MaximumBinCountPerFeature = 843,
            FeatureFraction = 0.84016003053467,
            LearningRate = 0.0135145459422405,
            LabelColumnName = @"Rings",
            FeatureColumnName = @"Features"
        };
        var fastTree = mlContext.BinaryClassification.Trainers.FastTree(fastTreeOptions);

        var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(@"Sex", @"Sex", outputKind: OneHotEncodingEstimator.OutputKind.Indicator)
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"Sex", @"Length", @"Diameter", @"Height", @"Whole weight", 
                                        @"Shucked weight", @"Viscera weight", @"Shell weight" }))
                                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"Rings", inputColumnName: @"Rings"))
                                    .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
                                        binaryEstimator: averagedPerceptron,
                                        labelColumnName: @"Rings"))
                                    //.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(
                                    //    binaryEstimator: fastTree,
                                    //    labelColumnName: @"Rings"))
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));
        

        Console.WriteLine($"=============== Training the model {DateTime.Now.ToString()} ===============");
        // 3. Train model
        var model = pipeline.Fit(split.TrainSet);
        Console.WriteLine($"=============== Finished Training the model Ending time: {DateTime.Now.ToString()} ===============");


        //Evaluate the model on a test dataset and calculate metrics of the model on the test data.
        var testMetrics = mlContext.MulticlassClassification.Evaluate(model.Transform(split.TestSet), labelColumnName: @"Rings");
        Console.WriteLine($"=============== Evaluating to get model's accuracy metrics ===============");
        PrintMetrics(testMetrics);


        // 4. Make a prediction
        var abalone = new AbaloneData() { 
            Sex = "M",
            Length = 0.35F,
            Diameter = 0.265F,
            Height = 0.9F,
            Wholeweight = 0.2255F,
            Shuckedweight = 0.995F,
            Visceraweight = 0.485F,
            Shellweight = 7F,
        };
        var age = mlContext.Model.CreatePredictionEngine<AbaloneData, Prediction>(model).Predict(abalone);
        Console.WriteLine($"Predicted sample age = {age.Rings}");
    }

    private static void PrintMetrics(MulticlassClassificationMetrics metrics)
    {
        Console.WriteLine($"*************************************************************************************************************");
        Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
        Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
        Console.WriteLine($"*       MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
        Console.WriteLine($"*       MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
        Console.WriteLine($"*       LogLoss:          {metrics.LogLoss:#.###}");
        Console.WriteLine($"*       LogLossReduction: {metrics.LogLossReduction:#.###}");
        Console.WriteLine($"*************************************************************************************************************");
    }
}