using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;

using OnnxStack.Core;

using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Helpers;
using OnnxStack.StableDiffusion.Schedulers;

using System;
using System.Collections.Generic;
using System.Linq;

namespace Morpheus
{
    public class LCMScheduler : IDisposable
    {
        private readonly Random _random;
        private readonly List<int> _timesteps;
        private readonly SchedulerOptions _options;
        private float _initNoiseSigma;

        private float[]? _alphasCumProd;
        private float _finalAlphaCumprod;

        public LCMScheduler(SchedulerOptions schedulerOptions)
        {
            _options = schedulerOptions;
            _random = new Random(_options.Seed);
            Initialize();
            _timesteps = new List<int>(SetTimesteps());
        }

        public SchedulerOptions Options => _options;
        public Random Random => _random;
        public float InitNoiseSigma => _initNoiseSigma;
        public IReadOnlyList<int> Timesteps => _timesteps;

        public DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            return sample;
        }

        public SchedulerStepResult Step(DenseTensor<float> modelOutput, int timestep, DenseTensor<float> sample, int order = 4)
        {
            //# Latent Consistency Models paper https://arxiv.org/abs/2310.04378

            int currentTimestep = timestep;

            // 1. get previous step value
            int prevIndex = Timesteps.IndexOf(currentTimestep) + 1;
            int previousTimestep = prevIndex < Timesteps.Count
                ? Timesteps[prevIndex]
                : currentTimestep;

            //# 2. compute alphas, betas
            float alphaProdT = _alphasCumProd[currentTimestep];
            float alphaProdTPrev = previousTimestep >= 0
                ? _alphasCumProd[previousTimestep]
                : _finalAlphaCumprod;
            float betaProdT = 1f - alphaProdT;
            float betaProdTPrev = 1f - alphaProdTPrev;
            float alphaSqrt = MathF.Sqrt(alphaProdT);
            float betaSqrt = MathF.Sqrt(betaProdT);
            float betaProdTPrevSqrt = MathF.Sqrt(betaProdTPrev);
            float alphaProdTPrevSqrt = MathF.Sqrt(alphaProdTPrev);


            // 3.Get scalings for boundary conditions
            (float cSkip, float cOut) = GetBoundaryConditionScalings(currentTimestep);


            //# 4. compute predicted original sample from predicted noise also called "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            DenseTensor<float> predOriginalSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                predOriginalSample = sample
                    .SubtractTensors(modelOutput.MultiplyTensorByFloat(betaSqrt))
                    .DivideTensorByFloat(alphaSqrt);
            }
            else if (Options.PredictionType == PredictionType.Sample)
            {
                predOriginalSample = modelOutput;
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                predOriginalSample = sample
                    .MultiplyTensorByFloat(alphaSqrt)
                    .SubtractTensors(modelOutput.MultiplyTensorByFloat(betaSqrt));
            }


            //# 5. Clip or threshold "predicted x_0"
            // TODO: OnnxStack does not yet support Threshold and Clipping


            //# 6. Denoise model output using boundary conditions
            var denoised = sample
                .MultiplyTensorByFloat(cSkip)
                .AddTensors(predOriginalSample.MultiplyTensorByFloat(cOut));


            //# 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
            var prevSample = Timesteps.Count > 1
                ? CreateRandomSample(modelOutput.Dimensions)
                    .MultiplyTensorByFloat(betaProdTPrevSqrt)
                    .AddTensors(denoised.MultiplyTensorByFloat(alphaProdTPrevSqrt))
                : denoised;

            return new SchedulerStepResult(prevSample, denoised);
        }

        public DenseTensor<float> AddNoise(DenseTensor<float> originalSamples, DenseTensor<float> noise, IReadOnlyList<int> timesteps)
        {
            // Ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L456
            int timestep = timesteps[0];
            float alphaProd = _alphasCumProd[timestep];
            float sqrtAlpha = MathF.Sqrt(alphaProd);
            float sqrtOneMinusAlpha = MathF.Sqrt(1.0f - alphaProd);

            return noise
                .MultiplyTensorByFloat(sqrtOneMinusAlpha)
                .AddTensors(originalSamples.MultiplyTensorByFloat(sqrtAlpha));
        }

        public void Initialize()
        {
            _alphasCumProd = null;

            var betas = GetBetaSchedule();
            var alphas = betas.Select(beta => 1.0f - beta);
            _alphasCumProd = alphas
                .Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b))
                .ToArray();

            bool setAlphaToOne = true;
            _finalAlphaCumprod = setAlphaToOne
                ? 1.0f
            : _alphasCumProd.First();

            SetInitNoiseSigma(1.0f);
        }

        public int[] SetTimesteps()
        {
            // LCM Timesteps Setting
            // Currently, only linear spacing is supported.
            var timeIncrement = Options.TrainTimesteps / Options.OriginalInferenceSteps;

            //# LCM Training Steps Schedule
            var lcmOriginTimesteps = Enumerable.Range(1, Options.OriginalInferenceSteps)
                .Select(x => x * timeIncrement - 1f)
                .ToArray();

            var skippingStep = lcmOriginTimesteps.Length / Options.InferenceSteps;

            // LCM Inference Steps Schedule
            return lcmOriginTimesteps
                .Where((t, index) => index % skippingStep == 0)
                .Take(Options.InferenceSteps)
                .Select(x => (int)x)
                .OrderByDescending(x => x)
                .ToArray();
        }

        public virtual DenseTensor<float> CreateRandomSample(ReadOnlySpan<int> dimensions, float initialNoiseSigma = 1f)
        {
            return TensorHelper.GetRandomTensor(Random, dimensions, initialNoiseSigma);
        }

        protected virtual float[] GetBetaSchedule()
        {
            var betas = Enumerable.Empty<float>();
            if (Options.TrainedBetas != null)
            {
                betas = Options.TrainedBetas;
            }
            else if (Options.BetaSchedule == BetaScheduleType.Linear)
            {
                betas = np.linspace(Options.BetaStart, Options.BetaEnd, Options.TrainTimesteps).ToArray<float>();
            }
            else if (Options.BetaSchedule == BetaScheduleType.ScaledLinear)
            {
                var start = (float)Math.Sqrt(Options.BetaStart);
                var end = (float)Math.Sqrt(Options.BetaEnd);
                betas = np.linspace(start, end, Options.TrainTimesteps)
                    .ToArray<float>()
                    .Select(x => x * x);
            }
            else if (Options.BetaSchedule == BetaScheduleType.SquaredCosCapV2)
            {
                betas = GetBetasForAlphaBar();
            }
            else if (Options.BetaSchedule == BetaScheduleType.Sigmoid)
            {
                var mul = Options.BetaEnd - Options.BetaStart;
                var betaSig = np.linspace(-6f, 6f, Options.TrainTimesteps).ToArray<float>();
                var sigmoidBetas = betaSig
                    .Select(beta => 1.0f / (1.0f + (float)Math.Exp(-beta)))
                    .ToArray();
                betas = sigmoidBetas
                    .Select(x => (x * mul) + Options.BetaStart)
                    .ToArray();
            }
            return betas.ToArray();
        }

        public (float cSkip, float cOut) GetBoundaryConditionScalings(float timestep)
        {
            //self.sigma_data = 0.5  # Default: 0.5
            var sigmaData = 0.5f;

            float c = MathF.Pow(timestep / 0.1f, 2f) + MathF.Pow(sigmaData, 2f);
            float cSkip = MathF.Pow(sigmaData, 2f) / c;
            float cOut = timestep / 0.1f / MathF.Pow(c, 0.5f);
            return (cSkip, cOut);
        }

        protected virtual float GetInitNoiseSigma(float[] sigmas)
        {
            var maxSigma = sigmas.Max();
            return Options.TimestepSpacing == TimestepSpacingType.Linspace
                || Options.TimestepSpacing == TimestepSpacingType.Trailing
                ? maxSigma : (float)Math.Sqrt(maxSigma * maxSigma + 1);
        }

        protected virtual float[] GetTimesteps()
        {
            NDArray timestepsArray = null;
            if (Options.TimestepSpacing == TimestepSpacingType.Linspace)
            {
                timestepsArray = np.linspace(0, Options.TrainTimesteps - 1, Options.InferenceSteps);
                timestepsArray = np.around(timestepsArray)["::1"];
            }
            else if (Options.TimestepSpacing == TimestepSpacingType.Leading)
            {
                var stepRatio = Options.TrainTimesteps / Options.InferenceSteps;
                timestepsArray = np.arange(0, (float)Options.InferenceSteps) * stepRatio;
                timestepsArray = np.around(timestepsArray)["::1"];
                timestepsArray += Options.StepsOffset;
            }
            else if (Options.TimestepSpacing == TimestepSpacingType.Trailing)
            {
                var stepRatio = Options.TrainTimesteps / (Options.InferenceSteps - 1);
                timestepsArray = np.arange((float)Options.TrainTimesteps, 0, -stepRatio)["::-1"];
                timestepsArray = np.around(timestepsArray);
                timestepsArray -= 1;
            }

            return timestepsArray.ToArray<float>();
        }

        protected virtual DenseTensor<float> GetPredictedSample(DenseTensor<float> modelOutput, DenseTensor<float> sample, float sigma)
        {
            DenseTensor<float> predOriginalSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                predOriginalSample = sample.SubtractTensors(modelOutput.MultiplyTensorByFloat(sigma));
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                var sigmaSqrt = (float)Math.Sqrt(sigma * sigma + 1);
                predOriginalSample = sample.DivideTensorByFloat(sigmaSqrt)
                    .AddTensors(modelOutput.MultiplyTensorByFloat(-sigma / sigmaSqrt));
            }
            else if (Options.PredictionType == PredictionType.Sample)
            {
                //prediction_type not implemented yet: sample
                predOriginalSample = sample.ToDenseTensor();
            }
            return predOriginalSample;
        }

        protected void SetInitNoiseSigma(float initNoiseSigma)
        {
            _initNoiseSigma = initNoiseSigma;
        }

        protected int GetPreviousTimestep(int timestep)
        {
            return timestep - _options.TrainTimesteps / _options.InferenceSteps;
        }

        protected float[] GetBetasForAlphaBar()
        {
            Func<float, float> alphaBarFn = null;
            if (_options.AlphaTransformType == AlphaTransformType.Cosine)
            {
                alphaBarFn = t => (float)Math.Pow(Math.Cos((t + 0.008f) / 1.008f * Math.PI / 2.0f), 2.0f);
            }
            else if (_options.AlphaTransformType == AlphaTransformType.Exponential)
            {
                alphaBarFn = t => (float)Math.Exp(t * -12.0f);
            }

            return Enumerable
                .Range(0, _options.TrainTimesteps)
                .Select(i =>
                {
                    var t1 = (float)i / _options.TrainTimesteps;
                    var t2 = (float)(i + 1) / _options.TrainTimesteps;
                    return Math.Min(1f - alphaBarFn(t2) / alphaBarFn(t1), _options.MaximumBeta);
                }).ToArray();
        }

        protected float[] Interpolate(float[] timesteps, float[] range, float[] sigmas)
        {
            // Create an output array with the same shape as timesteps
            var result = new float[timesteps.Length];

            // Loop over each element of timesteps
            for (int i = 0; i < timesteps.Length; i++)
            {
                // Find the index of the first element in range that is greater than or equal to timesteps[i]
                int index = Array.BinarySearch(range, timesteps[i]);

                // If timesteps[i] is exactly equal to an element in range, use the corresponding value in sigma
                if (index >= 0)
                {
                    result[i] = sigmas[(sigmas.Length - 1) - index];
                }

                // If timesteps[i] is less than the first element in range, use the first value in sigmas
                else if (index == -1)
                {
                    result[i] = sigmas[sigmas.Length - 1];
                }

                // If timesteps[i] is greater than the last element in range, use the last value in sigmas
                else if (index == -range.Length - 1)
                {
                    result[i] = sigmas[0];
                }

                // Otherwise, interpolate linearly between two adjacent values in sigmas
                else
                {
                    index = ~index; // bitwise complement of j gives the insertion point of x[i]
                    var startIndex = (sigmas.Length - 1) - index;
                    float t = (timesteps[i] - range[index - 1]) / (range[index] - range[index - 1]); // fractional distance between two points
                    result[i] = sigmas[startIndex - 1] + t * (sigmas[startIndex] - sigmas[startIndex - 1]); // linear interpolation formula
                }
            }
            return result;
        }

        protected float[] ConvertToKarras(float[] inSigmas)
        {
            // Get the minimum and maximum values from the input sigmas
            float sigmaMin = inSigmas[inSigmas.Length - 1];
            float sigmaMax = inSigmas[0];

            // Set the value of rho, which is used in the calculation
            float rho = 7.0f; // 7.0 is the value used in the paper

            // Create a linear ramp from 0 to 1
            float[] ramp = Enumerable.Range(0, _options.InferenceSteps)
                .Select(i => (float)i / (_options.InferenceSteps - 1))
                .ToArray();

            // Calculate the inverse of sigmaMin and sigmaMax raised to the power of 1/rho
            float minInvRho = (float)Math.Pow(sigmaMin, 1.0 / rho);
            float maxInvRho = (float)Math.Pow(sigmaMax, 1.0 / rho);

            // Calculate the Karras noise schedule using the formula from the paper
            float[] sigmas = new float[_options.InferenceSteps];
            for (int i = 0; i < _options.InferenceSteps; i++)
            {
                sigmas[i] = (float)Math.Pow(maxInvRho + ramp[i] * (minInvRho - maxInvRho), rho);
            }

            // Return the resulting noise schedule
            return sigmas;
        }

        protected float[] SigmaToTimestep(float[] sigmas, float[] logSigmas)
        {
            var timesteps = new float[sigmas.Length];
            for (int i = 0; i < sigmas.Length; i++)
            {
                float logSigma = (float)Math.Log(sigmas[i]);
                float[] dists = new float[logSigmas.Length];

                for (int j = 0; j < logSigmas.Length; j++)
                {
                    dists[j] = logSigma - logSigmas[j];
                }

                int lowIdx = 0;
                int highIdx = 1;

                for (int j = 0; j < logSigmas.Length - 1; j++)
                {
                    if (dists[j] >= 0)
                    {
                        lowIdx = j;
                        highIdx = j + 1;
                    }
                }

                float low = logSigmas[lowIdx];
                float high = logSigmas[highIdx];

                float w = (low - logSigma) / (low - high);
                w = Math.Clamp(w, 0, 1);

                float ti = (1 - w) * lowIdx + w * highIdx;
                timesteps[i] = ti;
            }

            return timesteps;
        }

        #region IDisposable

        private bool disposed = false;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (disposing)
            {
                // Dispose managed resources here.
                _alphasCumProd = null;
                _timesteps?.Clear();
            }

            // Dispose unmanaged resources here (if any).
            disposed = true;
        }

        #endregion
    }
}