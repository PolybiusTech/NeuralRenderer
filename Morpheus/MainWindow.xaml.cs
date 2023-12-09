using System;
using System.Linq;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Collections.Immutable;

using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

using OnnxStack.Core;
using OnnxStack.Core.Config;
using OnnxStack.Core.Services;

using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;

namespace Morpheus
{
    public partial class MainWindow : Window
    {
        WriteableBitmap display;
        VideoCapture camera;
        ModelOptions model;
        IOnnxModelService onnx;
        StableDiffusionConfig config;

        Image<Bgr, byte> canny;

        bool running = false;
        bool loaded = false;
        byte[] buffer = new byte[512 * 512 * 3];
        int seed;

        DenseTensor<float> promptembeds;
        DenseTensor<float> timeTensor = new DenseTensor<float>(new int[] { 1 });
        DenseTensor<double> condScale = new DenseTensor<double>(new int[] { 1 });

        Dictionary<string, OrtValue> cnetI = new Dictionary<string, OrtValue>();
        Dictionary<string, OrtValue> cnetO = new Dictionary<string, OrtValue>();
        Dictionary<string, OrtValue> unetI = new Dictionary<string, OrtValue>();
        Dictionary<string, OrtValue> unetO = new Dictionary<string, OrtValue>();
        Dictionary<string, OrtValue> dnetI = new Dictionary<string, OrtValue>();
        Dictionary<string, OrtValue> dnetO = new Dictionary<string, OrtValue>();

        public MainWindow()
        {
            InitializeComponent();
            display = new WriteableBitmap(512, 512, 96, 96, PixelFormats.Bgr24, null);
            Canvas.Source = display;
            camera = new VideoCapture(0);
            camera.ImageGrabbed += OnCameraFrame;
            camera.Start();
            onnx = App.GetService<IOnnxModelService>();
            config = App.GetService<StableDiffusionConfig>();
            model = config.OnnxModelSets[0];
        }

        private void OnCameraFrame(object? sender, EventArgs e)
        {
            if (!Dispatcher.CheckAccess())
            {
                Dispatcher.Invoke(() => OnCameraFrame(sender, e));
                return;
            }

            Image<Bgr, byte> img = camera.QueryFrame().ToImage<Bgr, byte>();

            int size = Math.Min(img.Width, img.Height);
            int x = (img.Width - size) / 2;
            int y = (img.Height - size) / 2;
            img.ROI = new System.Drawing.Rectangle(x, y, size, size);
            img = img.Resize(512, 512, Inter.Linear);
            img.ROI = System.Drawing.Rectangle.Empty;

            canny = img.Canny(90, 100).Convert<Bgr, byte>();

            if (NeuralCheck.IsChecked ?? true)
            {
                display.WritePixels(new Int32Rect(0, 0, 512, 512), buffer, 512 * 3, 0);
            } 
            else if (CameraCheck.IsChecked ?? true)
            {
                display.WritePixels(new Int32Rect(0, 0, img.Width, img.Height), img.MIplImage.ImageData, img.MIplImage.ImageSize, img.MIplImage.WidthStep);
            }
            else
            {
                display.WritePixels(new Int32Rect(0, 0, canny.Width, canny.Height), canny.MIplImage.ImageData, canny.MIplImage.ImageSize, canny.MIplImage.WidthStep);
            }
        }

        private async Task RenderLoop()
        {
            DenseTensor<float>? latent = null;

            while (running)
            {
                SchedulerOptions scheduler = new SchedulerOptions
                {
                    InferenceSteps = 1,
                    OriginalInferenceSteps = 1,
                    BetaStart = 0.00001f, // 0.00085f
                    BetaEnd = 0.011f, // 0.012f
                    TrainTimesteps = 1040,
                    BetaSchedule = BetaScheduleType.Linear,
                    Seed = seed
                };

                var sw = Stopwatch.StartNew();

                latent = await SchedulerStepAsync(scheduler, ImageToBinaryTensor(canny), latent);

                Debug.WriteLine("Generate: " + sw.ElapsedMilliseconds);
                sw.Restart();

                var shit = await DecodeLatentsAsync(latent);

                Debug.WriteLine("Decode: " + sw.ElapsedMilliseconds);
                sw.Restart();

                TensorToBuffer(shit, buffer);

                Debug.WriteLine("ToBuffer: " + sw.ElapsedMilliseconds);
                sw.Restart();
            }
        }

        public async Task<DenseTensor<float>> SchedulerStepAsync(SchedulerOptions s, DenseTensor<float> controlImage, DenseTensor<float>? latent = null)
        {
            var sw = Stopwatch.StartNew();
            // Get Scheduler
            using (var scheduler = new LCMScheduler(s))
            {
                // Get timesteps
                var timesteps = scheduler.Timesteps;

                // Create latent sample
                var noise = scheduler.CreateRandomSample(new int[] { 1, 4, 64, 64 }, 1f);
                var latents = latent != null ? scheduler.AddNoise(latent, noise, timesteps) : noise;

                Debug.WriteLine("Sample: " + sw.ElapsedMilliseconds);
                sw.Restart();

                // Get Guidance Scale Embedding
                var guidanceEmbeddings = GetGuidanceScaleEmbedding(12f);

                TensorToSpan(promptembeds).CopyTo(cnetI["encoder_hidden_states"].GetTensorMutableDataAsSpan<Float16>());
                TensorToSpan(controlImage).CopyTo(cnetI["controlnet_cond"].GetTensorMutableDataAsSpan<Float16>());
                TensorToSpan(guidanceEmbeddings).CopyTo(unetI["timestep_cond"].GetTensorMutableDataAsSpan<Float16>());

                // Denoised result
                DenseTensor<float> denoised = null;

                // Loop though the timesteps
                var step = 0;
                foreach (var timestep in timesteps)
                {
                    step++;

                    condScale[0] = 0.5f;
                    timeTensor[0] = timestep;

                    TensorToSpan(latents).CopyTo(cnetI["sample"].GetTensorMutableDataAsSpan<Float16>());
                    TensorToSpan(timeTensor).CopyTo(cnetI["timestep"].GetTensorMutableDataAsSpan<Float16>());
                    
                    condScale.Buffer.Span.CopyTo(cnetI["conditioning_scale"].GetTensorMutableDataAsSpan<double>());

                    Debug.WriteLine("Copy: " + sw.ElapsedMilliseconds);
                    sw.Restart();

                    if (true) { await onnx.RunInferenceAsync(model, OnnxModelType.Controlnet, cnetI, cnetO); }
                    Debug.WriteLine("Cnet: " + sw.ElapsedMilliseconds);
                    sw.Restart();

                    await onnx.RunInferenceAsync(model, OnnxModelType.Unet, unetI, unetO);
                    Debug.WriteLine("Unet: " + sw.ElapsedMilliseconds);
                    sw.Restart();

                    var noisePred = unetO["out_sample"].ToDenseTensor();

                    // Scheduler Step
                    var schedulerResult = scheduler.Step(noisePred, timestep, latents);
                    Debug.WriteLine("Sched: " + sw.ElapsedMilliseconds);
                    sw.Restart();

                    latents = schedulerResult.Result;
                    denoised = schedulerResult.SampleData;
                }

                return denoised;
            }
        }

        static Span<Float16> TensorToSpan(DenseTensor<float> sourceTensor)
        {
            var source = sourceTensor.Buffer.Span.ToArray();
            var target = new Float16[source.Length];

            Parallel.For(0, source.Length, i =>
            {
                target[i] = (Float16) source[i];
            });

            return target;
        }

        private static void TensorToBuffer(DenseTensor<float> image, byte[] buffer)
        {
            Parallel.For(0, buffer.Length / 3, i =>
            {
                int pos = i * 3;
                int x = i % 512;
                int y = (i - x) / 512;
                buffer[pos]   = (byte)(Math.Clamp((image[0, 2, y, x] * 0.5) + 0.5, 0, 1) * 255); // Blue
                buffer[pos+1] = (byte)(Math.Clamp((image[0, 1, y, x] * 0.5) + 0.5, 0, 1) * 255); // Green
                buffer[pos+2] = (byte)(Math.Clamp((image[0, 0, y, x] * 0.5) + 0.5, 0, 1) * 255); // Red
            });
        }

        private static DenseTensor<float> ImageToBinaryTensor(Image<Bgr, byte> inputImage)
        {
            var height = inputImage.Height;
            var width = inputImage.Width;
            var result = new DenseTensor<float>(new int[] { 1, 3, width, height });
            Parallel.For(0, width * height, i =>
            {
                int x = i % 512;
                int y = (i - x) / 512;
                float v = inputImage[x, y].Red > 1.0 ? 1.0f : 0.0f;
                result[0, 0, x, y] = v;
                result[0, 1, x, y] = v;
                result[0, 2, x, y] = v;
            });
            return result;
        }

        private DenseTensor<float> GetGuidanceScaleEmbedding(float guidance, int embeddingDim = 256)
        {
            float scale = guidance - 1f;
            int halfDim = embeddingDim / 2;
            float log = MathF.Log(10000.0f) / (halfDim - 1);
            var emb = Enumerable.Range(0, halfDim)
                .Select(x => MathF.Exp(x * -log))
                .ToArray();
            var embSin = emb.Select(MathF.Sin).ToArray();
            var embCos = emb.Select(MathF.Cos).ToArray();
            var result = new DenseTensor<float>(new[] { 1, 2 * halfDim });
            for (int i = 0; i < halfDim; i++)
            {
                result[0, i] = embSin[i];
                result[0, i + halfDim] = embCos[i];
            }
            return result;
        }

        public async Task<DenseTensor<float>> DecodeLatentsAsync(DenseTensor<float> latents)
        {
            var sw = Stopwatch.StartNew();

            TensorToSpan(latents).CopyTo(dnetI["latent_sample"].GetTensorMutableDataAsSpan<Float16>());

            await onnx.RunInferenceAsync(model, OnnxModelType.VaeDecoder, dnetI, dnetO);
            Debug.WriteLine("VAERun: " + sw.ElapsedMilliseconds);
            sw.Restart();

            var fuck = dnetO["sample"].ToDenseTensor();
            Debug.WriteLine("F**K: " + sw.ElapsedMilliseconds);
            sw.Restart();

            return fuck;
        }

        public async Task<DenseTensor<float>> CreatePromptAsync(string prompt)
        {
            var promptTokens = await DecodeTextAsync(model, prompt);
            var promptEmbeddings = await GenerateEmbedsAsync(model, promptTokens, promptTokens.Length);
            return promptEmbeddings;
        }

        public Task<int[]> DecodeTextAsync(IModelOptions model, string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return Task.FromResult(Array.Empty<int>());

            var inputNames = onnx.GetInputNames(model, OnnxModelType.Tokenizer);
            var outputNames = onnx.GetOutputNames(model, OnnxModelType.Tokenizer);
            var inputTensor = new DenseTensor<string>(new string[] { inputText }, new int[] { 1 });
            using (var inputTensorValue = OrtValue.CreateFromStringTensor(inputTensor))
            {
                var outputs = new string[] { outputNames[0] };
                var inputs = new Dictionary<string, OrtValue> { { inputNames[0], inputTensorValue } };
                var results = onnx.RunInference(model, OnnxModelType.Tokenizer, inputs, outputs);
                using (var result = results.First())
                {
                    var resultData = result.GetTensorDataAsSpan<long>().ToArray();
                    return Task.FromResult(Array.ConvertAll(resultData, Convert.ToInt32));
                }
            }
        }

        public async Task<float[]> EncodeTokensAsync(IModelOptions model, int[] tokenizedInput)
        {
            var inputNames = onnx.GetInputNames(model, OnnxModelType.TextEncoder);
            var outputNames = onnx.GetOutputNames(model, OnnxModelType.TextEncoder);
            var outputMetaData = onnx.GetOutputMetadata(model, OnnxModelType.TextEncoder);
            var outputTensorMetaData = outputMetaData.Values.First();

            var inputDim = new[] { 1L, tokenizedInput.Length };
            var outputDim = new[] { 1L, tokenizedInput.Length, model.EmbeddingsLength };
            using (var outputTensorValue = outputTensorMetaData.CreateOutputBuffer(outputDim.ToInt()))
            using (var inputTensorValue = OrtValue.CreateTensorValueFromMemory(tokenizedInput, inputDim))
            {
                var inputs = new Dictionary<string, OrtValue> { { inputNames[0], inputTensorValue } };
                var outputs = new Dictionary<string, OrtValue> { { outputNames[0], outputTensorValue } };
                var results = await onnx.RunInferenceAsync(model, OnnxModelType.TextEncoder, inputs, outputs);
                using (var result = results.First())
                {
                    return outputTensorValue.ToArray();
                }
            }
        }

        private async Task<DenseTensor<float>> GenerateEmbedsAsync(IModelOptions model, int[] inputTokens, int minimumLength)
        {
            // If less than minimumLength pad with blank tokens
            if (inputTokens.Length < minimumLength)
                inputTokens = PadWithBlankTokens(inputTokens, minimumLength, model.BlankTokenValueArray).ToArray();

            // The CLIP tokenizer only supports 77 tokens, batch process in groups of 77 and concatenate
            var embeddings = new List<float>();
            foreach (var tokenBatch in inputTokens.Batch(model.TokenizerLimit))
            {
                var tokens = PadWithBlankTokens(tokenBatch, model.TokenizerLimit, model.BlankTokenValueArray);
                embeddings.AddRange(await EncodeTokensAsync(model, tokens.ToArray()));
            }

            var dim = new[] { 1, embeddings.Count / model.EmbeddingsLength, model.EmbeddingsLength };
            return TensorHelper.CreateTensor(embeddings.ToArray(), dim);
        }

        private IEnumerable<int> PadWithBlankTokens(IEnumerable<int> inputs, int requiredLength, ImmutableArray<int> blankTokens)
        {
            var count = inputs.Count();
            if (requiredLength > count)
                return inputs.Concat(blankTokens[..(requiredLength - count)]);
            return inputs;
        }

        private async void LoadModelClicked(object sender, RoutedEventArgs e)
        {
            LoadModel.IsEnabled = false;

            if (onnx.IsModelLoaded(model))
            {   // Unload the model
                running = false;
                await onnx.UnloadModelAsync(model);

                foreach (string name in cnetI.Keys) { cnetI[name].Dispose(); cnetI.Remove(name); }
                foreach (string name in cnetO.Keys) { cnetO[name].Dispose(); cnetO.Remove(name); }
                foreach (string name in unetI.Keys) { unetI[name].Dispose(); unetI.Remove(name); }
                foreach (string name in unetO.Keys) { unetO[name].Dispose(); unetO.Remove(name); }
                foreach (string name in dnetI.Keys) { unetI[name].Dispose(); unetI.Remove(name); }
                foreach (string name in dnetO.Keys) { unetO[name].Dispose(); unetO.Remove(name); }

                LoadModel.Content = "Load Model";
                StartRender.IsEnabled = false;
            }
            else
            {   // Load the Model
                await onnx.LoadModelAsync(model);

                cnetI["sample"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 4, 64, 64 });
                cnetI["timestep"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1 });
                cnetI["encoder_hidden_states"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 77, 512 + 256 });
                cnetI["controlnet_cond"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 3, 512, 512 });
                cnetI["conditioning_scale"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Double, new long[] { 1 });

                cnetO["down_block0"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 320, 64, 64 });
                cnetO["down_block1"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 320, 64, 64 });
                cnetO["down_block2"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 320, 64, 64 });
                cnetO["down_block3"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 320, 32, 32 });
                cnetO["down_block4"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 640, 32, 32 });
                cnetO["down_block5"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 640, 32, 32 });
                cnetO["down_block6"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 640, 16, 16 });
                cnetO["down_block7"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 1280, 16, 16 });
                cnetO["down_block8"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 1280, 16, 16 });
                cnetO["down_block9"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 1280, 8, 8 });
                cnetO["down_block10"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 1280, 8, 8 });
                cnetO["down_block11"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 1280, 8, 8 });
                cnetO["mid_block"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1280, 8, 8 });

                unetI["sample"] = cnetI["sample"];
                unetI["timestep"] = cnetI["timestep"];
                unetI["encoder_hidden_states"] = cnetI["encoder_hidden_states"];
                unetI["timestep_cond"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 256 });
                unetI["down_block0"] = cnetO["down_block0"];
                unetI["down_block1"] = cnetO["down_block1"];
                unetI["down_block2"] = cnetO["down_block2"];
                unetI["down_block3"] = cnetO["down_block3"];
                unetI["down_block4"] = cnetO["down_block4"];
                unetI["down_block5"] = cnetO["down_block5"];
                unetI["down_block6"] = cnetO["down_block6"];
                unetI["down_block7"] = cnetO["down_block7"];
                unetI["down_block8"] = cnetO["down_block8"];
                unetI["down_block9"] = cnetO["down_block9"];
                unetI["down_block10"] = cnetO["down_block10"];
                unetI["down_block11"] = cnetO["down_block11"];
                unetI["mid_block"] = cnetO["mid_block"];

                unetO["out_sample"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 4, 64, 64 });

                dnetI["latent_sample"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 4, 64, 64 });
                dnetO["sample"] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, TensorElementType.Float16, new long[] { 1, 3, 512, 512 });

                LoadModel.Content = "Unload Model";
                StartRender.IsEnabled = true;
            }

            LoadModel.IsEnabled = true;
        }

        private async void StartRenderClicked(object sender, RoutedEventArgs e)
        {
            if (running)
            {
                StartRender.Content = "Start Render";
                running = false;
            }
            else
            {
                StartRender.Content = "Stop Render";
                running = true;
                seed = Random.Shared.Next();
                promptembeds = await CreatePromptAsync(PromptBox.Text);
                await RenderLoop();
            }
        }

        private async void PromptBoxChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
        {
            if (running)
            {
                StartRender.Content = "Start Render";
                running = false;
            }
        }
    }
}
