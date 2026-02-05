/**
 * FANUC RISE DASHBOARD ENGINE
 * Connects to Python Backend via REST API.
 */

const DEFAULT_API_URL = "http://localhost:8000";

class DashboardApp {
    constructor() {
        this.machineId = this.getMachineIdFromQuery();
        this.apiUrl = this.getApiUrl();
        this.wsBaseUrl = this.apiUrl.replace(/^http/i, "ws");
        this.wsRetryDelayMs = 5000;
        this.wsAttemptOrder = [
            `/ws/telemetry/${encodeURIComponent(this.machineId)}`,
            "/ws/telemetry"
        ];

        this.statusEl = document.getElementById("system-status");
        this.consoleEl = document.getElementById("console-output");
        this.pollingInterval = null;
        this.brainState = { dopamine: 50, cortisol: 0, serotonin: 50 };

        this.init();
    }

    getMachineIdFromQuery() {
        const params = new URLSearchParams(window.location.search);
        const rawMachineId = (params.get("machine_id") || "CNC-001").trim();
        return this.sanitizeMachineId(rawMachineId);
    }

    sanitizeMachineId(machineId) {
        const safeMachineId = machineId.replace(/[^a-zA-Z0-9_-]/g, "");
        return safeMachineId || "CNC-001";
    }

    getApiUrl() {
        const params = new URLSearchParams(window.location.search);
        const apiBase = params.get("api_base");

        if (apiBase) {
            try {
                return new URL(apiBase, window.location.origin).origin;
            } catch (err) {
                console.warn("Invalid api_base query parameter, falling back to origin/default.", err);
            }
        }

        if (window.location.protocol.startsWith("http")) {
            return window.location.origin;
        }

        return DEFAULT_API_URL;
    }

    init() {
        this.log(`Initializing Dashboard for ${this.machineId}...`);
        this.log(`API Base: ${this.apiUrl}`);
        this.statusEl.innerText = `CONNECTING ${this.machineId}`;
        this.connectWebSocket(0);
    }

    log(msg) {
        const line = document.createElement("div");
        const ts = new Date().toLocaleTimeString();
        line.innerText = `[${ts}] ${msg}`;
        this.consoleEl.appendChild(line);
        this.consoleEl.scrollTop = this.consoleEl.scrollHeight;
    }

    connectWebSocket(attemptIndex = 0) {
        const wsPath = this.wsAttemptOrder[Math.min(attemptIndex, this.wsAttemptOrder.length - 1)];
        const wsUrl = `${this.wsBaseUrl}${wsPath}`;
        const usingFallback = wsPath === "/ws/telemetry";
        let opened = false;

        this.log(`Connecting to WebSocket: ${wsUrl}`);

        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            opened = true;
            this.log(usingFallback
                ? "Connected to fallback real-time stream"
                : "Connected to machine-scoped real-time stream");
            this.statusEl.innerText = `SYSTEM ONLINE (LIVE) • ${this.machineId}`;
            this.statusEl.style.borderColor = "var(--neon-green)";
        };

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateUIFromSocket(data);
        };

        this.socket.onclose = () => {
            if (!opened && attemptIndex < this.wsAttemptOrder.length - 1) {
                this.log("Machine stream unavailable. Switching to fallback websocket endpoint...");
                this.connectWebSocket(attemptIndex + 1);
                return;
            }

            this.log("WebSocket disconnected. Retrying in 5s...");
            this.statusEl.innerText = `RECONNECTING ${this.machineId}...`;
            this.statusEl.style.borderColor = "var(--neon-red)";
            setTimeout(() => this.connectWebSocket(0), this.wsRetryDelayMs);
        };

        this.socket.onerror = (err) => {
            console.error("WS Error:", err);
            this.socket.close();
        };
    }

    updateUIFromSocket(data) {
        const neuroState = data.neuro_state || { dopamine: 50, cortisol: 10, serotonin: 70 };
        const action = data.action || "MONITORING_LIVE";

        if (data.spindle_rpm || data.rpm) {
            const rpm = data.spindle_rpm || data.rpm;
            this.setBar("bar-rpm", "val-rpm", rpm.toFixed(0), rpm / 20000 * 100);
        }
        if (data.spindle_load || data.load) {
            const load = data.spindle_load || data.load;
            this.setBar("bar-load", "val-load", load.toFixed(1) + "%", load);
        }
        if (data.vibration) {
            this.setBar("bar-vib", "val-vib", data.vibration.toFixed(2) + "g", data.vibration * 100);
        }

        this.setBar("neuro-dopamine", null, null, neuroState.dopamine);
        this.setBar("neuro-cortisol", null, null, neuroState.cortisol);
        this.setBar("neuro-serotonin", null, null, neuroState.serotonin);

        document.getElementById("ai-action").innerText = action;

        const nodeSensory = document.getElementById("node-sensory");
        if (nodeSensory) nodeSensory.querySelector(".node-val").innerText = "LIVE";

        const nodeDopamine = document.getElementById("node-dopamine");
        if (nodeDopamine) nodeDopamine.querySelector(".node-val").innerText = neuroState.dopamine.toFixed(0);
    }

    generateMockTelemetry() {
        return {
            fanuc_data: {
                load: Math.random() * 80,
                rpm: 5000 + Math.random() * 2000,
                vibration: Math.random() * 0.3
            },
            sw_data: {
                curvature: Math.random() * 0.2
            }
        };
    }

    updateUI(apiResponse, inputs) {
        const telemetry = inputs.fanuc_data;
        const brain = apiResponse.neuro_state;
        const action = apiResponse.recommended_action;

        this.setBar("bar-rpm", "val-rpm", telemetry.rpm.toFixed(0), telemetry.rpm / 20000 * 100);
        this.setBar("bar-load", "val-load", telemetry.load.toFixed(1) + "%", telemetry.load);
        this.setBar("bar-vib", "val-vib", telemetry.vibration.toFixed(2) + "g", telemetry.vibration * 100);

        this.setBar("neuro-dopamine", null, null, brain.dopamine);
        this.setBar("neuro-cortisol", null, null, brain.cortisol);
        this.setBar("neuro-serotonin", null, null, brain.serotonin);

        const curve = inputs.sw_data.curvature;
        const curveEl = document.getElementById("val-curve");
        if (curveEl) curveEl.innerText = curve.toFixed(3);

        const stressEl = document.getElementById("val-stress");
        if (stressEl) {
            if (curve > 0.15) {
                stressEl.innerText = "HIGH (Refining Ghost Path)";
                stressEl.style.color = "var(--neon-red)";
            } else {
                stressEl.innerText = "OPTIMAL";
                stressEl.style.color = "var(--neon-green)";
            }
        }

        document.getElementById("ai-action").innerText = action;

        if (brain.cortisol > 50) this.log("WARNING: High Cortisol!");
    }

    setBar(barId, valId, valText, percent) {
        const bar = document.getElementById(barId);
        if (bar) bar.style.width = `${Math.min(100, Math.max(0, percent))}%`;

        if (valId) {
            document.getElementById(valId).innerText = valText;
        }
    }

    async conductProtocol() {
        const prompt = document.getElementById("protocol-prompt").value;
        const resultEl = document.getElementById("protocol-result");

        if (!prompt) return;

        this.log(`CONDUCTING PROTOCOL: ${prompt}`);
        resultEl.style.display = "block";
        resultEl.innerText = "Analyzing Topology & Intent...";

        try {
            const response = await fetch(`${this.apiUrl}/conduct`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: "User_Intent", prompt: prompt })
            });

            const res = await response.json();
            resultEl.innerText = `[AI RESPONSE]: ${res.scenario_text || JSON.stringify(res)}`;
        } catch (e) {
            this.log("Protocol Failed: " + e.message);
            resultEl.innerText = "Error conducting protocol.";
        }
    }

    async triggerOptimization() {
        const mat = document.getElementById("input-material").value;
        const mode = document.querySelector('input[name="mode"]:checked').value;

        this.log(`OPTIMIZING FOR: ${mat} [${mode}]`);

        const gcode = ["G01 X100 F1000", "G01 X200 F2000"];

        try {
            const response = await fetch(`${this.apiUrl}/optimize?material=${mat}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(gcode)
            });

            const res = await response.json();
            this.log("OPTIMIZATION COMPLETE. IR length: " + (res.optimized_ir ? res.optimized_ir.length : "N/A"));
        } catch (e) {
            this.log("Optimization Service Offline");
        }
    }
}

window.app = new DashboardApp();
