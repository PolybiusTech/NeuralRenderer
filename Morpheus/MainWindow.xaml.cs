using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;
using SixLabors.ImageSharp.Formats;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Enums;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp;
using OnnxStack.StableDiffusion.Diffusers.LatentConsistency;

namespace Morpheus
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        WriteableBitmap display;
        VideoCapture camera;
        ModelOptions model;
        IStableDiffusionService renderer;
        bool running = false;
        byte[] buffer;

        public MainWindow()
        {
            InitializeComponent();
            display = new WriteableBitmap(512, 512, 96, 96, PixelFormats.Bgr24, null);
            Canvas.Source = display;
            camera = new VideoCapture();
            camera.ImageGrabbed += OnCameraFrame;
            camera.Start();
            buffer = new byte[512 * 512 * 3];
            renderer = App.GetService<IStableDiffusionService>();
            model = renderer.Models[0];
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

            // display.WritePixels(new Int32Rect(0, 0, img.Width, img.Height), img.MIplImage.ImageData, img.MIplImage.ImageSize, img.MIplImage.WidthStep);
            display.WritePixels(new Int32Rect(0, 0, 512, 512), buffer, 512 * 3, 0);
        }

        private async Task RenderLoop()
        {
            while (running)
            {
                PromptOptions prompt = new PromptOptions { Prompt = PromptBox.Text };
                SchedulerOptions scheduler = new SchedulerOptions
                {
                    InferenceSteps = 1,
                    SchedulerType = SchedulerType.LCM,
                    Seed = 0
                };
                (await renderer.GenerateAsBufferAsync(model, prompt, scheduler)).CopyPixelDataTo(buffer);
            }
        }

        private async void LoadModelClicked(object sender, RoutedEventArgs e)
        {
            LoadModel.IsEnabled = false;

            if (renderer.IsModelLoaded(model))
            {
                running = false;
                await renderer.UnloadModelAsync(model);
                LoadModel.Content = "Load Model";
                StartRender.IsEnabled = false;
            }
            else
            {
                await renderer.LoadModelAsync(model);
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
                await RenderLoop();
            }
        }
    }
}
