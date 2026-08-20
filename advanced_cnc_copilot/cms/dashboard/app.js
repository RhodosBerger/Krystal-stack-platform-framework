/**
 * FANUC RISE DASHBOARD ENGINE
 * Connects to Python Backend via REST API.
 */

const API_URL = "http://localhost:8000";

class DashboardApp {
    constructor() {
        this.statusEl = document.getElementById("system-status");
        this.consoleEl = document.getElementById("console-output");
        this.pollingInterval = null;
        this.brainState = { dopamine: 50, cortisol: 0, serotonin: 50 };

        this.init();
    }

    init() {
        this.log("Initializing Dashboard...");
        this.connectWebSocket();
    }

    log(msg) {
        const line = document.createElement("div");
        const ts = new Date().toLocaleTimeString();
        line.innerText = `[${ts}] ${msg}`;
        this.consoleEl.appendChild(line);
        this.consoleEl.scrollTop = this.consoleEl.scrollHeight;
    }

    connectWebSocket() {
        const wsUrl = API_URL.replace("http", "ws") + "/ws/telemetry";
        this.log(`Connecting to WebSocket: ${wsUrl}`);

        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            this.log("Connected to Real-time Stream");
            this.statusEl.innerText = "SYSTEM ONLINE (LIVE)";
            this.statusEl.style.borderColor = "var(--neon-green)";
        };

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateUIFromSocket(data);
        };

        this.socket.onclose = () => {
            this.log("WebSocket Disconnected. Retrying in 5s...");
            this.statusEl.innerText = "RECONNECTING...";
            this.statusEl.style.borderColor = "var(--neon-red)";
            setTimeout(() => this.connectWebSocket(), 5000);
        };

        this.socket.onerror = (err) => {
            console.error("WS Error:", err);
            this.socket.close();
        };
    }

    updateUIFromSocket(data) {
        // Map backend WebSocket data to existing UI update logic
        // We simulate the neuro_state and action if not present in socket yet
        // In the next step we will update the backend to send these
        const neuro_state = data.neuro_state || { dopamine: 50, cortisol: 10, serotonin: 70 };
        const action = data.action || "MONITORING_LIVE";

        // 1. Update HAL Panel
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

        // 2. Update Neuro Panel
        this.setBar("neuro-dopamine", null, null, neuro_state.dopamine);
        this.setBar("neuro-cortisol", null, null, neuro_state.cortisol);
        this.setBar("neuro-serotonin", null, null, neuro_state.serotonin);

        document.getElementById("ai-action").innerText = action;

        // 3. Logic Node Updates
        const nodeSensory = document.getElementById("node-sensory");
        if (nodeSensory) nodeSensory.querySelector(".node-val").innerText = "LIVE";

        const nodeDopamine = document.getElementById("node-dopamine");
        if (nodeDopamine) nodeDopamine.querySelector(".node-val").innerText = neuro_state.dopamine.toFixed(0);
    }

    generateMockTelemetry() {
        // Simulate a running machine for the visualizer
        return {
            fanuc_data: {
                load: Math.random() * 80, // 0-80%
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

        // 1. Update HAL Panel
        this.setBar("bar-rpm", "val-rpm", telemetry.rpm.toFixed(0), telemetry.rpm / 20000 * 100);
        this.setBar("bar-load", "val-load", telemetry.load.toFixed(1) + "%", telemetry.load);
        this.setBar("bar-vib", "val-vib", telemetry.vibration.toFixed(2) + "g", telemetry.vibration * 100);

        // 2. Update Neuro Panel
        this.setBar("neuro-dopamine", null, null, brain.dopamine);
        this.setBar("neuro-cortisol", null, null, brain.cortisol);
        this.setBar("neuro-serotonin", null, null, brain.serotonin);

        // 3. Update Digital Twin (Creative Canvas)
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

        // Log critical events
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
            const response = await fetch(`${API_URL}/conduct`, {
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

        // Mock GCode from UI
        const gcode = ["G01 X100 F1000", "G01 X200 F2000"];

        try {
            const response = await fetch(`${API_URL}/optimize?material=${mat}`, {
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

// Start
window.app = new DashboardApp();
