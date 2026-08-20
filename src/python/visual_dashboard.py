"""
GAMESA Visual Dashboard - Real-time Monitoring Web Interface

Web-based dashboard for monitoring GAMESA optimization in real-time.
Uses Flask for backend, WebSockets for live updates.
"""

import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
from datetime import datetime

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-socketio")


@dataclass
class DashboardMetrics:
    """Metrics snapshot for dashboard."""
    timestamp: float
    temperature: float
    thermal_headroom: float
    cpu_util: float
    gpu_util: float
    memory_util: float
    power_draw: float
    fps: float
    latency: float
    brain_decision: str
    brain_source: str
    game_state: str
    power_state: str
    thermal_action: str
    anomaly_count: int
    active_apps: int


class MetricsBuffer:
    """Thread-safe metrics buffer."""

    def __init__(self, maxlen: int = 300):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, metrics: DashboardMetrics):
        with self.lock:
            self.buffer.append(metrics)

    def get_recent(self, n: int = 60) -> List[Dict]:
        with self.lock:
            recent = list(self.buffer)[-n:]
            return [asdict(m) for m in recent]

    def get_latest(self) -> Optional[Dict]:
        with self.lock:
            if self.buffer:
                return asdict(self.buffer[-1])
            return None


class VisualDashboard:
    """
    Real-time web dashboard for GAMESA.
    """

    def __init__(self, port: int = 8080):
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask not installed")

        self.port = port
        self.metrics_buffer = MetricsBuffer()

        # Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'gamesa-dashboard-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Setup routes
        self._setup_routes()

        # Background thread
        self.running = False
        self.update_thread = None

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            return self._render_dashboard()

        @self.app.route('/api/metrics/latest')
        def get_latest():
            metrics = self.metrics_buffer.get_latest()
            return jsonify(metrics or {})

        @self.app.route('/api/metrics/history')
        def get_history():
            n = request.args.get('n', 60, type=int)
            history = self.metrics_buffer.get_recent(n)
            return jsonify(history)

        @self.socketio.on('connect')
        def handle_connect():
            print(f"Client connected from {request.remote_addr}")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Client disconnected")

    def _render_dashboard(self) -> str:
        """Render HTML dashboard."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>GAMESA Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            color: white;
            font-size: 2.5em;
            margin-bottom: 5px;
        }
        .header p {
            color: rgba(255,255,255,0.9);
            font-size: 1.1em;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            background: #333;
            border-radius: 5px;
        }
        .metric-label {
            color: #aaa;
        }
        .metric-value {
            font-weight: bold;
            font-size: 1.2em;
        }
        .temp { color: #ff6b6b; }
        .cpu { color: #51cf66; }
        .gpu { color: #339af0; }
        .power { color: #ffd43b; }
        .fps { color: #ff6b6b; }
        .chart-container {
            position: relative;
            height: 250px;
            margin-top: 10px;
        }
        .status-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .status-good { background: #51cf66; color: #1a1a1a; }
        .status-warn { background: #ffd43b; color: #1a1a1a; }
        .status-critical { background: #ff6b6b; color: white; }
        .connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 10px 20px;
            border-radius: 5px;
            background: #51cf66;
            color: white;
            font-weight: bold;
        }
        .connection-status.disconnected {
            background: #ff6b6b;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connection">Connected</div>

    <div class="header">
        <h1>⚡ GAMESA Dashboard</h1>
        <p>Real-time System Optimization Monitor</p>
    </div>

    <div class="grid">
        <div class="card">
            <h3>🌡️ Thermal Status</h3>
            <div class="metric">
                <span class="metric-label">Temperature</span>
                <span class="metric-value temp" id="temp">--°C</span>
            </div>
            <div class="metric">
                <span class="metric-label">Headroom</span>
                <span class="metric-value" id="headroom">--°C</span>
            </div>
            <div class="metric">
                <span class="metric-label">Action</span>
                <span class="status-badge status-good" id="thermal-action">normal</span>
            </div>
        </div>

        <div class="card">
            <h3>💻 Resource Usage</h3>
            <div class="metric">
                <span class="metric-label">CPU</span>
                <span class="metric-value cpu" id="cpu">--%</span>
            </div>
            <div class="metric">
                <span class="metric-label">GPU</span>
                <span class="metric-value gpu" id="gpu">--%</span>
            </div>
            <div class="metric">
                <span class="metric-label">Memory</span>
                <span class="metric-value" id="memory">--%</span>
            </div>
        </div>

        <div class="card">
            <h3>⚡ Power & Performance</h3>
            <div class="metric">
                <span class="metric-label">Power Draw</span>
                <span class="metric-value power" id="power">--W</span>
            </div>
            <div class="metric">
                <span class="metric-label">FPS</span>
                <span class="metric-value fps" id="fps">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Latency</span>
                <span class="metric-value" id="latency">--ms</span>
            </div>
        </div>

        <div class="card">
            <h3>🧠 Brain Decisions</h3>
            <div class="metric">
                <span class="metric-label">Decision</span>
                <span class="metric-value" id="decision">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Source</span>
                <span class="metric-value" id="source">--</span>
            </div>
            <div class="metric">
                <span class="metric-label">Game State</span>
                <span class="status-badge status-good" id="game-state">unknown</span>
            </div>
        </div>
    </div>

    <div class="card">
        <h3>📊 Temperature History</h3>
        <div class="chart-container">
            <canvas id="tempChart"></canvas>
        </div>
    </div>

    <div class="card">
        <h3>📈 Resource History</h3>
        <div class="chart-container">
            <canvas id="resourceChart"></canvas>
        </div>
    </div>

    <script>
        const socket = io();

        // Connection status
        const connectionEl = document.getElementById('connection');
        socket.on('connect', () => {
            connectionEl.textContent = 'Connected';
            connectionEl.classList.remove('disconnected');
        });
        socket.on('disconnect', () => {
            connectionEl.textContent = 'Disconnected';
            connectionEl.classList.add('disconnected');
        });

        // Charts
        const tempCtx = document.getElementById('tempChart').getContext('2d');
        const tempChart = new Chart(tempCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Temperature (°C)',
                    data: [],
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Headroom (°C)',
                    data: [],
                    borderColor: '#51cf66',
                    backgroundColor: 'rgba(81, 207, 102, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#e0e0e0' } } },
                scales: {
                    x: { ticks: { color: '#aaa' }, grid: { color: '#333' } },
                    y: { ticks: { color: '#aaa' }, grid: { color: '#333' } }
                }
            }
        });

        const resourceCtx = document.getElementById('resourceChart').getContext('2d');
        const resourceChart = new Chart(resourceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU %',
                    data: [],
                    borderColor: '#51cf66',
                    tension: 0.4
                }, {
                    label: 'GPU %',
                    data: [],
                    borderColor: '#339af0',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#e0e0e0' } } },
                scales: {
                    x: { ticks: { color: '#aaa' }, grid: { color: '#333' } },
                    y: { ticks: { color: '#aaa' }, grid: { color: '#333' }, min: 0, max: 100 }
                }
            }
        });

        // Update metrics
        socket.on('metrics', (data) => {
            document.getElementById('temp').textContent = data.temperature.toFixed(1) + '°C';
            document.getElementById('headroom').textContent = data.thermal_headroom.toFixed(1) + '°C';
            document.getElementById('cpu').textContent = (data.cpu_util * 100).toFixed(0) + '%';
            document.getElementById('gpu').textContent = (data.gpu_util * 100).toFixed(0) + '%';
            document.getElementById('memory').textContent = (data.memory_util * 100).toFixed(0) + '%';
            document.getElementById('power').textContent = data.power_draw.toFixed(1) + 'W';
            document.getElementById('fps').textContent = data.fps.toFixed(0);
            document.getElementById('latency').textContent = data.latency.toFixed(1) + 'ms';
            document.getElementById('decision').textContent = data.brain_decision;
            document.getElementById('source').textContent = data.brain_source;
            document.getElementById('game-state').textContent = data.game_state;
            document.getElementById('thermal-action').textContent = data.thermal_action;

            // Update charts
            const now = new Date().toLocaleTimeString();
            if (tempChart.data.labels.length > 60) {
                tempChart.data.labels.shift();
                tempChart.data.datasets[0].data.shift();
                tempChart.data.datasets[1].data.shift();
            }
            tempChart.data.labels.push(now);
            tempChart.data.datasets[0].data.push(data.temperature);
            tempChart.data.datasets[1].data.push(data.thermal_headroom);
            tempChart.update('none');

            if (resourceChart.data.labels.length > 60) {
                resourceChart.data.labels.shift();
                resourceChart.data.datasets[0].data.shift();
                resourceChart.data.datasets[1].data.shift();
            }
            resourceChart.data.labels.push(now);
            resourceChart.data.datasets[0].data.push(data.cpu_util * 100);
            resourceChart.data.datasets[1].data.push(data.gpu_util * 100);
            resourceChart.update('none');
        });

        // Fetch initial data
        fetch('/api/metrics/latest')
            .then(r => r.json())
            .then(data => {
                if (data.timestamp) socket.emit('metrics', data);
            });
    </script>
</body>
</html>
        """

    def update_metrics(self, metrics: DashboardMetrics):
        """Update metrics (called by breakingscript)."""
        self.metrics_buffer.append(metrics)

        # Emit to connected clients
        if self.running:
            self.socketio.emit('metrics', asdict(metrics))

    def _background_update(self):
        """Background thread for periodic updates."""
        while self.running:
            time.sleep(1.0)
            latest = self.metrics_buffer.get_latest()
            if latest:
                self.socketio.emit('metrics', latest)

    def run(self, host: str = '0.0.0.0', debug: bool = False):
        """Run dashboard server."""
        self.running = True

        # Start background thread
        self.update_thread = threading.Thread(target=self._background_update, daemon=True)
        self.update_thread.start()

        print(f"🌐 GAMESA Dashboard starting on http://{host}:{self.port}")
        print(f"   Open in browser: http://localhost:{self.port}")

        self.socketio.run(self.app, host=host, port=self.port, debug=debug, allow_unsafe_werkzeug=True)

    def stop(self):
        """Stop dashboard."""
        self.running = False


def create_dashboard(port: int = 8080) -> VisualDashboard:
    """Factory function."""
    return VisualDashboard(port=port)


if __name__ == "__main__":
    # Test dashboard
    import random

    dashboard = create_dashboard()

    # Simulate metrics
    def simulate_metrics():
        while True:
            metrics = DashboardMetrics(
                timestamp=time.time(),
                temperature=random.uniform(60, 80),
                thermal_headroom=random.uniform(10, 25),
                cpu_util=random.uniform(0.3, 0.8),
                gpu_util=random.uniform(0.4, 0.9),
                memory_util=random.uniform(0.5, 0.7),
                power_draw=random.uniform(18, 28),
                fps=random.uniform(55, 70),
                latency=random.uniform(8, 15),
                brain_decision="boost",
                brain_source="cognitive",
                game_state="exploration",
                power_state="balanced",
                thermal_action="normal",
                anomaly_count=0,
                active_apps=3,
            )
            dashboard.update_metrics(metrics)
            time.sleep(1.0)

    # Start simulation thread
    sim_thread = threading.Thread(target=simulate_metrics, daemon=True)
    sim_thread.start()

    # Run dashboard
    dashboard.run()
