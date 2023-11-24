using System;
using System.Windows;
using System.Windows.Threading;
using OnnxStack.Core;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;

namespace Morpheus
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private static IHost _applicationHost;
        public App()
        {
            var builder = Host.CreateApplicationBuilder();
            builder.Services.AddOnnxStackStableDiffusion();
            builder.Services.AddSingleton<MainWindow>();
            _applicationHost = builder.Build();
        }


        public static T GetService<T>() => _applicationHost.Services.GetService<T>();

        public static void UIInvoke(Action action, DispatcherPriority priority = DispatcherPriority.Render) => Current.Dispatcher.BeginInvoke(priority, action);


        /// <summary>
        /// Raises the <see cref="E:Startup" /> event.
        /// </summary>
        /// <param name="e">The <see cref="StartupEventArgs"/> instance containing the event data.</param>
        protected override async void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);
            await _applicationHost.StartAsync();
            GetService<MainWindow>().Show();
        }


        /// <summary>
        /// Raises the <see cref="E:Exit" /> event.
        /// </summary>
        /// <param name="e">The <see cref="ExitEventArgs"/> instance containing the event data.</param>
        protected override async void OnExit(ExitEventArgs e)
        {
            await _applicationHost.StopAsync();
            base.OnExit(e);
        }

    }
}
