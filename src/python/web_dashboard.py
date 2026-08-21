"""
Web Dashboard - FastAPI/Flask interface for KrystalSDK

Provides:
- REST API for SDK operations
- Admin panel
- Real-time metrics
- Agent management
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json
import time
import threading

# Try FastAPI first, fall back to minimal server
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


# ============================================================
# Minimal HTTP Server (no dependencies)
# ============================================================

class DashboardState:
    """Shared state for dashboard."""

    def __init__(self):
        self.krystal = None
        self.platform = None
        self.metrics_history: List[Dict] = []
        self.start_time = time.time()

    def init_krystal(self):
        try:
            from .krystal_sdk import Krystal
            self.krystal = Krystal()
        except ImportError:
            pass

    def init_platform(self):
        try:
            from .generative_platform import create_generative_platform
            self.platform = create_generative_platform()
        except ImportError:
            pass

    def record_metrics(self):
        if self.krystal:
            self.metrics_history.append({
                "timestamp": time.time(),
                "metrics": self.krystal.get_metrics()
            })
            if len(self.metrics_history) > 1000:
                self.metrics_history.pop(0)


STATE = DashboardState()


# HTML Templates
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>KrystalSDK Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #0d1117; color: #c9d1d9; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
        .card h2 { color: #8b949e; font-size: 14px; margin-bottom: 10px; text-transform: uppercase; }
        .metric { font-size: 32px; font-weight: bold; color: #58a6ff; }
        .metric-small { font-size: 14px; color: #8b949e; }
        .status { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                  margin-right: 8px; }
        .status.online { background: #3fb950; }
        .status.offline { background: #f85149; }
        .btn { background: #238636; color: white; border: none; padding: 8px 16px;
               border-radius: 6px; cursor: pointer; margin: 4px; }
        .btn:hover { background: #2ea043; }
        .btn.secondary { background: #30363d; }
        .log { background: #0d1117; border: 1px solid #30363d; padding: 10px;
               font-family: monospace; font-size: 12px; max-height: 200px;
               overflow-y: auto; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #30363d; }
        th { color: #8b949e; }
        .phase { padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        .phase.SOLID { background: #1f6feb; }
        .phase.LIQUID { background: #238636; }
        .phase.GAS { background: #9e6a03; }
        .phase.PLASMA { background: #f85149; }
    </style>
</head>
<body>
    <div class="container">
        <h1>KrystalSDK Dashboard</h1>
        <div class="grid">
            <div class="card">
                <h2>System Status</h2>
                <p><span class="status online"></span> <span id="status">Running</span></p>
                <p class="metric-small">Uptime: <span id="uptime">0s</span></p>
            </div>
            <div class="card">
                <h2>Current Phase</h2>
                <p><span id="phase" class="phase LIQUID">LIQUID</span></p>
                <p class="metric-small">Exploration: <span id="explore">10%</span></p>
            </div>
            <div class="card">
                <h2>Total Cycles</h2>
                <p class="metric"><span id="cycles">0</span></p>
            </div>
            <div class="card">
                <h2>Total Reward</h2>
                <p class="metric"><span id="reward">0.00</span></p>
            </div>
        </div>

        <div class="grid" style="margin-top: 20px;">
            <div class="card">
                <h2>Controls</h2>
                <button class="btn" onclick="observe()">Observe</button>
                <button class="btn" onclick="decide()">Decide</button>
                <button class="btn" onclick="reward()">Reward</button>
                <button class="btn secondary" onclick="reset()">Reset</button>
            </div>
            <div class="card">
                <h2>Agents</h2>
                <table>
                    <tr><th>Agent</th><th>Calls</th><th>Status</th></tr>
                    <tbody id="agents"></tbody>
                </table>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h2>Activity Log</h2>
            <div class="log" id="log"></div>
        </div>
    </div>

    <script>
        function log(msg) {
            const el = document.getElementById('log');
            el.innerHTML = `[${new Date().toLocaleTimeString()}] ${msg}<br>` + el.innerHTML;
        }

        async function api(endpoint, method='GET', body=null) {
            const opts = { method };
            if (body) { opts.body = JSON.stringify(body); opts.headers = {'Content-Type': 'application/json'}; }
            const resp = await fetch('/api/' + endpoint, opts);
            return resp.json();
        }

        async function observe() {
            const data = await api('observe', 'POST', {cpu: Math.random(), mem: Math.random()});
            log('Observed: ' + JSON.stringify(data));
            refresh();
        }

        async function decide() {
            const data = await api('decide');
            log('Decision: ' + JSON.stringify(data.action));
            refresh();
        }

        async function reward() {
            const r = Math.random();
            const data = await api('reward', 'POST', {value: r});
            log(`Reward ${r.toFixed(2)}: TD error = ${data.td_error?.toFixed(4)}`);
            refresh();
        }

        async function reset() {
            await api('reset', 'POST');
            log('System reset');
            refresh();
        }

        async function refresh() {
            const data = await api('metrics');
            document.getElementById('cycles').textContent = data.cycles || 0;
            document.getElementById('reward').textContent = (data.total_reward || 0).toFixed(2);
            document.getElementById('phase').textContent = data.phase || 'LIQUID';
            document.getElementById('phase').className = 'phase ' + (data.phase || 'LIQUID');
            document.getElementById('explore').textContent = ((data.exploration_rate || 0.1) * 100).toFixed(0) + '%';

            const agents = data.agents || {};
            let html = '';
            for (const [name, m] of Object.entries(agents)) {
                html += `<tr><td>${name}</td><td>${m.calls || 0}</td><td><span class="status online"></span></td></tr>`;
            }
            document.getElementById('agents').innerHTML = html;
        }

        function updateUptime() {
            fetch('/api/uptime').then(r => r.json()).then(d => {
                document.getElementById('uptime').textContent = d.uptime + 's';
            });
        }

        setInterval(refresh, 2000);
        setInterval(updateUptime, 1000);
        refresh();
    </script>
</body>
</html>
"""


class MinimalHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler without dependencies."""

    def log_message(self, format, *args):
        pass  # Suppress logs

    def send_json(self, data: Dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_html(self, html: str):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/' or path == '/dashboard':
            self.send_html(DASHBOARD_HTML)

        elif path == '/api/metrics':
            if STATE.krystal:
                self.send_json(STATE.krystal.get_metrics())
            elif STATE.platform:
                self.send_json(STATE.platform.get_metrics())
            else:
                self.send_json({"cycles": 0, "phase": "INIT"})

        elif path == '/api/uptime':
            self.send_json({"uptime": int(time.time() - STATE.start_time)})

        elif path == '/api/decide':
            if STATE.krystal:
                action = STATE.krystal.decide()
                self.send_json({"action": action})
            else:
                self.send_json({"action": [0.5, 0.5, 0.5, 0.5]})

        elif path == '/api/agents':
            if STATE.platform:
                self.send_json({
                    "agents": list(STATE.platform.orchestrator.agents.keys())
                })
            else:
                self.send_json({"agents": []})

        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        content_len = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_len)) if content_len else {}

        if path == '/api/observe':
            if STATE.krystal:
                STATE.krystal.observe(body)
                self.send_json({"status": "ok", "state": body})
            else:
                self.send_json({"status": "no krystal"})

        elif path == '/api/reward':
            if STATE.krystal:
                td = STATE.krystal.reward(body.get('value', 0))
                self.send_json({"td_error": td})
            else:
                self.send_json({"td_error": 0})

        elif path == '/api/reset':
            STATE.init_krystal()
            self.send_json({"status": "reset"})

        elif path == '/api/generate':
            if STATE.platform:
                artifact = STATE.platform.generate(body, actor="web")
                if artifact:
                    self.send_json({
                        "content": artifact.content,
                        "quality": artifact.quality_score,
                        "approved": artifact.approved
                    })
                else:
                    self.send_json({"error": "generation failed"})
            else:
                self.send_json({"error": "no platform"})

        else:
            self.send_json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# ============================================================
# FastAPI App (if available)
# ============================================================

def create_fastapi_app() -> "FastAPI":
    """Create FastAPI application."""
    app = FastAPI(title="KrystalSDK Dashboard")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    @app.get("/api/metrics")
    async def metrics():
        if STATE.krystal:
            return STATE.krystal.get_metrics()
        return {"cycles": 0, "phase": "INIT"}

    @app.get("/api/uptime")
    async def uptime():
        return {"uptime": int(time.time() - STATE.start_time)}

    @app.get("/api/decide")
    async def decide():
        if STATE.krystal:
            return {"action": STATE.krystal.decide()}
        return {"action": [0.5] * 4}

    @app.post("/api/observe")
    async def observe(request: Request):
        body = await request.json()
        if STATE.krystal:
            STATE.krystal.observe(body)
        return {"status": "ok"}

    @app.post("/api/reward")
    async def reward(request: Request):
        body = await request.json()
        if STATE.krystal:
            td = STATE.krystal.reward(body.get("value", 0))
            return {"td_error": td}
        return {"td_error": 0}

    @app.post("/api/generate")
    async def generate(request: Request):
        body = await request.json()
        if STATE.platform:
            artifact = STATE.platform.generate(body, actor="api")
            if artifact:
                return {
                    "content": artifact.content,
                    "quality": artifact.quality_score
                }
        raise HTTPException(400, "Generation failed")

    return app


# ============================================================
# Server Runner
# ============================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run dashboard server."""
    STATE.init_krystal()
    STATE.init_platform()

    print(f"Starting KrystalSDK Dashboard at http://{host}:{port}")

    if HAS_FASTAPI:
        app = create_fastapi_app()
        uvicorn.run(app, host=host, port=port, log_level="warning")
    else:
        server = HTTPServer((host, port), MinimalHandler)
        print("(Using minimal server - install fastapi+uvicorn for better performance)")
        server.serve_forever()


def run_background(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Run server in background thread."""
    thread = threading.Thread(target=run_server, args=(host, port), daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    run_server()
