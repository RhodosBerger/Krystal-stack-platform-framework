// Arch UI - System Administration Portal Logic
// Interacts with Krystal-Stack Localhost Hub APIs

document.addEventListener("DOMContentLoaded", () => {
    console.log("Arch UI Initialized. Establishing telemetry connection...");
    initTelemetry();
    initChart("ow-chart", 20);
});

function initTelemetry() {
    // SSE Stream for Real-time telemetry
    const evtSource = new EventSource("/api/stream");
    
    evtSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            
            // Update Cyclic Organism
            if (data.type === 'state_update') {
                updateCyclic(data);
            }
            
            // Update Hardware Governor / Vulkan
            if (data.type === 'vulkan_telemetry') {
                updateHardware(data);
            }

            // Update OpenWorld Engine
            if (data.type === 'engine_telemetry' || data.type === 'state_update') {
                updateOpenWorld(data);
            }
            
        } catch (e) {
            console.error("Telemetry parse error:", e);
        }
    };
    
    evtSource.onerror = function() {
        document.getElementById("global-status").innerText = "CONNECTION LOST";
        document.getElementById("global-status").style.borderColor = "var(--alert-red)";
        document.getElementById("global-status").style.color = "var(--alert-red)";
    };
    
    // Fallback polling for specific endpoints if stream is insufficient
    setInterval(pollHealth, 5000);
}

function updateCyclic(data) {
    const phaseEl = document.getElementById("cyc-phase");
    const symbolEl = document.getElementById("cyc-symbol");
    const energyEl = document.getElementById("cyc-energy");
    const lyapEl = document.getElementById("cyc-lyap");
    const entropyEl = document.getElementById("cyc-entropy");
    const entropyBar = document.getElementById("entropy-bar");

    if (data.brainwave_phase) {
        phaseEl.innerText = data.brainwave_phase;
        
        // Map Phase to Symbol and Color
        let color = "var(--accent-cyan)";
        let symbol = "≈";
        
        if(data.brainwave_phase === 'BETA') { color = "var(--alert-orange)"; symbol = "::"; }
        else if(data.brainwave_phase === 'GAMMA') { color = "var(--alert-red)"; symbol = "⚡"; }
        else if(data.brainwave_phase === 'OMEGA') { color = "#888"; symbol = "Ω"; }
        
        phaseEl.style.color = color;
        symbolEl.innerText = symbol;
        symbolEl.style.color = color;
    }
    
    if (data.H !== undefined) {
        energyEl.innerText = data.H.toFixed(4);
    }
    
    if (data.lyapunov !== undefined) {
        lyapEl.innerText = data.lyapunov.toFixed(4);
    }

    if (data.entropy !== undefined) {
        const pct = Math.min(100, Math.max(0, data.entropy * 100));
        entropyEl.innerText = pct.toFixed(1) + "%";
        entropyBar.style.width = pct + "%";
        
        if (pct > 70) entropyBar.style.background = "var(--alert-red)";
        else if (pct > 40) entropyBar.style.background = "var(--alert-orange)";
        else entropyBar.style.background = "var(--accent-cyan)";
    }
}

function updateHardware(data) {
    if (data.compute_backend) document.getElementById("hw-backend").innerText = data.compute_backend;
    if (data.render_strategy) document.getElementById("hw-strategy").innerText = data.render_strategy;
    
    if (data.readback_mb_s !== undefined) {
        document.getElementById("hw-readback").innerText = data.readback_mb_s.toFixed(2);
        const maxExpected = 1000; // Expected PCIe bandwidth max for UI mapping
        const pct = Math.min(100, (data.readback_mb_s / maxExpected) * 100);
        document.getElementById("readback-bar").style.width = pct + "%";
    }
    
    if (data.dispatch_us !== undefined) {
        document.getElementById("hw-dispatch").innerText = data.dispatch_us.toFixed(1) + " µs";
    }
}

function updateOpenWorld(data) {
    if (data.preset) document.getElementById("ow-preset").innerText = data.preset;
    if (data.ray_steps) document.getElementById("ow-steps").innerText = data.ray_steps;
    
    // Simulate evals/sec if not provided
    const evals = data.evals_per_sec || Math.floor(Math.random() * 50000 + 900000); 
    document.getElementById("ow-evals").innerText = evals.toLocaleString();
    
    // Add to chart
    appendChartVal("ow-chart", evals, 1500000);
}

function initChart(containerId, bars) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    for(let i=0; i<bars; i++) {
        const bar = document.createElement("div");
        bar.className = "bar";
        bar.style.height = "5%";
        container.appendChild(bar);
    }
}

function appendChartVal(containerId, val, maxVal) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const bars = container.children;
    
    // Shift left
    for(let i=0; i<bars.length - 1; i++) {
        bars[i].style.height = bars[i+1].style.height;
        bars[i].style.background = bars[i+1].style.background;
    }
    
    // New val
    const pct = Math.min(100, (val / maxVal) * 100);
    const lastBar = bars[bars.length - 1];
    lastBar.style.height = pct + "%";
    
    if (pct > 80) lastBar.style.background = "var(--alert-red)";
    else if (pct > 50) lastBar.style.background = "var(--alert-orange)";
    else lastBar.style.background = "var(--accent-dark)";
}

function pollHealth() {
    fetch('/api/health')
        .then(res => res.json())
        .then(data => {
            const badge = document.getElementById("global-status");
            if(data.status === "HEALTHY" || data.status === "OK") {
                badge.innerText = "SYSTEM ONLINE";
                badge.style.borderColor = "var(--accent-cyan)";
                badge.style.color = "var(--accent-cyan)";
            }
        })
        .catch(err => console.error(err));
}

// Actions
function toggleVulkan() {
    fetch('/api/vulkan/toggle', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            console.log("Vulkan toggle response:", data);
            // Will update via SSE stream
        })
        .catch(err => console.error("Error toggling Vulkan:", err));
}

function triggerEasterEgg() {
    // Generate a pseudo-UUID for visual effect on the portal
    // In reality this would trigger an endpoint on the Synthesizer
    const uuid = 'krystal-' + Math.random().toString(36).substring(2, 10) + '-' + Date.now().toString(36);
    document.getElementById("synth-uuid").innerText = uuid;
    
    // Visual flash
    document.getElementById("synth-uuid").style.color = "var(--text-bright)";
    setTimeout(() => {
        document.getElementById("synth-uuid").style.color = "var(--accent-cyan)";
    }, 500);
}
