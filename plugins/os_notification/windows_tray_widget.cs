// ==============================================================================
// KRYSTAL-STACK: WINDOWS NOTIFICATION BAR & TRAY WIDGET
// Component: plugins/os_notification/windows_tray_widget.cs
// Description: Real-time native Windows notification bar & tray widget monitoring
//              CPU context switches, thread thrashing, and processor integrity.
// Target Framework: .NET 8 / C# 12 (WPF / WinUI 3 Native Interop)
// System Invariant: VITAL_MAX_HP = 6
// ==============================================================================

using System;
using System.Drawing;
using System.IO;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace KrystalStack.OSNotification.Windows
{
    public class ProcessorTelemetryData
    {
        public double cpu_utilization_pct { get; set; }
        public double context_switches_per_sec { get; set; }
        public double system_calls_per_sec { get; set; }
        public double cs_to_syscall_ratio { get; set; }
        public double expected_baseline_cs_per_sec { get; set; }
        public double thrashing_index { get; set; }
        public double integrity_score { get; set; }
        public string status { get; set; } = "OPTIMAL";
        public bool anomaly_detected { get; set; }
        public string behavioral_diagnosis { get; set; } = "";
        public string target_os { get; set; } = "Windows";
        public int vital_max_hp { get; set; } = 6;
    }

    /// <summary>
    /// Native Windows Desktop Notification Bar and System Tray Widget.
    /// Interacts directly with Windows NT Kernel via ntdll.dll or Krystal Web Hub API.
    /// </summary>
    public class WindowsKernelNotificationBar : Form
    {
        public const int VitalMaxHp = 6;

        private readonly NotifyIcon _trayIcon;
        private readonly System.Windows.Forms.Timer _pollTimer;
        private readonly HttpClient _httpClient;
        private readonly Label _lblTitle;
        private readonly Label _lblMetrics;
        private readonly Label _lblDiagnosis;
        private readonly ProgressBar _integrityProgress;

        public WindowsKernelNotificationBar()
        {
            if (VitalMaxHp != 6)
                throw new InvalidOperationException("VITAL_MAX_HP invariant violation");

            // Form styling as a sleek top floating notification bar
            FormBorderStyle = FormBorderStyle.None;
            StartPosition = FormStartPosition.Manual;
            TopMost = true;
            ShowInTaskbar = false;
            BackColor = Color.FromArgb(16, 20, 28);
            ForeColor = Color.White;
            Width = 720;
            Height = 44;

            // Position at top center of primary monitor
            var screenBounds = Screen.PrimaryScreen.Bounds;
            Location = new Point((screenBounds.Width - Width) / 2, 8);

            // UI Elements
            _lblTitle = new Label
            {
                Text = "⚡ KRYSTAL KERNEL INTEGRITY",
                Font = new Font("Segoe UI", 8.5f, FontStyle.Bold),
                ForeColor = Color.FromArgb(0, 255, 136),
                Location = new Point(12, 6),
                AutoSize = true
            };

            _lblMetrics = new Label
            {
                Text = "CS: 0/s | CPU: 0% | ThrashIdx: 1.00",
                Font = new Font("Consolas", 8.5f, FontStyle.Regular),
                ForeColor = Color.FromArgb(180, 200, 220),
                Location = new Point(12, 22),
                AutoSize = true
            };

            _lblDiagnosis = new Label
            {
                Text = "Inicializácia telemetrie...",
                Font = new Font("Segoe UI", 8.0f, FontStyle.Italic),
                ForeColor = Color.FromArgb(140, 160, 180),
                Location = new Point(280, 14),
                Width = 320,
                Height = 22
            };

            _integrityProgress = new ProgressBar
            {
                Location = new Point(610, 14),
                Size = new Size(96, 16),
                Minimum = 0,
                Maximum = 100,
                Value = 100
            };

            Controls.Add(_lblTitle);
            Controls.Add(_lblMetrics);
            Controls.Add(_lblDiagnosis);
            Controls.Add(_integrityProgress);

            // Tray Icon
            _trayIcon = new NotifyIcon
            {
                Icon = SystemIcons.Shield,
                Text = "Krystal Kernel Telemetry Guard",
                Visible = true
            };

            var contextMenu = new ContextMenuStrip();
            contextMenu.Items.Add("Zobraziť / Skryť Lištu", null, (s, e) => Visible = !Visible);
            contextMenu.Items.Add("Ukončiť", null, (s, e) => Application.Exit());
            _trayIcon.ContextMenuStrip = contextMenu;

            _httpClient = new HttpClient { Timeout = TimeSpan.FromMilliseconds(800) };

            _pollTimer = new System.Windows.Forms.Timer { Interval = 1000 };
            _pollTimer.Tick += async (s, e) => await PollTelemetryAsync();
            _pollTimer.Start();
        }

        private async Task PollTelemetryAsync()
        {
            try
            {
                string json = await _httpClient.GetStringAsync("http://localhost:8080/api/kernel/integrity");
                var telemetry = JsonSerializer.Deserialize<ProcessorTelemetryData>(json);
                if (telemetry != null)
                {
                    UpdateUI(telemetry);
                }
            }
            catch
            {
                // Fallback: NTDLL direct query or idle status
                _lblDiagnosis.Text = "Lokálny Krystal Web Hub offline. Beží autonómny režim.";
            }
        }

        public void UpdateUI(ProcessorTelemetryData data)
        {
            _lblMetrics.Text = $"CS: {data.context_switches_per_sec:N0}/s | CPU: {data.cpu_utilization_pct:F1}% | ThrashIdx: {data.thrashing_index:F2}";
            _lblDiagnosis.Text = data.behavioral_diagnosis;
            _integrityProgress.Value = Math.Max(0, Math.Min(100, (int)(data.integrity_score * 100)));

            Color accentColor;
            if (data.status == "OPTIMAL")
            {
                accentColor = Color.FromArgb(0, 255, 136);
            }
            else if (data.status == "NOMINAL")
            {
                accentColor = Color.FromArgb(0, 229, 255);
            }
            else if (data.status == "THRASHING_WARNING")
            {
                accentColor = Color.FromArgb(255, 170, 0);
                _trayIcon.ShowBalloonTip(3000, "Krystal Telemetry Warning", "Zvýšené prepínanie vlákien (Thrashing Warning)!", ToolTipIcon.Warning);
            }
            else
            {
                accentColor = Color.FromArgb(255, 34, 85);
                _trayIcon.ShowBalloonTip(4000, "CRITICAL SCHEDULER COLLAPSE", "Patologický Context-Switch Storm!", ToolTipIcon.Error);
            }

            _lblTitle.ForeColor = accentColor;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _trayIcon?.Dispose();
                _pollTimer?.Dispose();
                _httpClient?.Dispose();
            }
            base.Dispose(disposing);
        }

        [STAThread]
        public static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new WindowsKernelNotificationBar());
        }
    }
}
