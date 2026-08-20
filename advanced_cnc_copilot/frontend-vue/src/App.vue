<template>
  <div class="vue-nexus">
    <!-- Header -->
    <header class="app-header">
      <div class="brand">
        <Activity class="icon-pulse" />
        <div class="brand-text">
          <h1>FANUC RISE</h1>
          <span class="subtitle">NEURO-ADAPTIVE CORE // V2.0</span>
        </div>
      </div>
      
      <div class="header-stats">
        <div class="h-stat">
          <label>SWARM</label>
          <span :class="{ active: swarmNodes.length > 0 }">{{ swarmNodes.length }} NODES</span>
        </div>
        <div class="h-stat">
          <label>UPLINK</label>
          <span :class="{ online: connected }">{{ connected ? 'ESTABLISHED' : 'SEARCHING...' }}</span>
        </div>
        <div class="h-stat">
          <label>TIME</label>
          <span>{{ timestamp }}</span>
        </div>
      </div>
    </header>

    <!-- Main Grid Layout -->
    <main class="main-grid">
      <!-- COLUMN 1: SENSORY INPUT (Telemetry) -->
      <section class="grid-col col-left">
        <div class="panel-box telemetry-panel">
          <div class="panel-title">SENSORY FEED</div>
          
          <div class="telemetry-display">
            <div class="big-metric rpm-metric">
              <svg class="rpm-dial" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="45" stroke="#222" stroke-width="5" fill="none" />
                <path d="M 50 50 L 50 10" stroke="#00d4ff" stroke-width="2" :transform="`rotate(${telemetry.rpm * 0.03}, 50, 50)`" />
              </svg>
              <div class="metric-val">{{ telemetry.rpm.toFixed(0) }}</div>
              <div class="metric-label">RPM</div>
            </div>
            
            <div class="sub-metrics">
              <div class="sm-row">
                <label>LOAD</label>
                <div class="bar-bg"><div class="bar-fill red" :style="{ width: telemetry.load + '%' }"></div></div>
                <span>{{ telemetry.load.toFixed(1) }}%</span>
              </div>
              <div class="sm-row">
                <label>VIB</label>
                <div class="bar-bg"><div class="bar-fill orange" :style="{ width: (telemetry.vibration * 100) + '%' }"></div></div>
                <span>{{ telemetry.vibration.toFixed(3) }}g</span>
              </div>
            </div>
          </div>
        </div>

        <div class="panel-box swarm-panel">
          <div class="panel-title">SWARM TOPOLOGY</div>
          <div class="swarm-grid">
            <div v-for="node in swarmNodes" :key="node.id" class="swarm-node" :class="{ active: node.load > 0 }">
              <div class="status-dot" :style="{ background: node.load > 80 ? '#f40' : '#0f8' }"></div>
              <span>{{ node.id }}</span>
            </div>
            <div v-if="swarmNodes.length === 0" class="no-swarm">NO NODES DETECTED</div>
          </div>
        </div>
      </section>

      <!-- COLUMN 2: COGNITION (Brain & Stats) -->
      <section class="grid-col col-center">
        <!-- The Mind -->
        <NeuroState 
          :dopamine="neuro.dopamine" 
          :cortisol="neuro.cortisol" 
          :serotonin="neuro.serotonin" 
        />
        
        <!-- The Benchmarking Inspector -->
        <NormInspector />
        
        <!-- The Shadow Council -->
        <CouncilLog :logs="councilLogs" style="margin-top: 20px; flex-grow: 1;" />
      </section>

      <!-- COLUMN 3: EXECUTIVE FUNCTION (Action) -->
      <section class="grid-col col-right">
        <div class="panel-box action-panel">
          <div class="panel-title">EXECUTIVE DECISION</div>
          <div class="action-display">
            <h2 class="action-text glitch-text">{{ action }}</h2>
            <div class="confidence-meter">
               <label>CONFIDENCE</label>
               <div class="conf-bar"><div class="conf-fill" :style="{ width: neuro.serotonin + '%' }"></div></div>
            </div>
          </div>
        </div>

        <div class="panel-box controls-panel">
           <button class="control-btn danger">EMERGENCY STOP</button>
           <button class="control-btn warning">RESET ALARMS</button>
           <button class="control-btn primary">OPTIMIZE FLOW</button>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { Activity } from 'lucide-vue-next';

// Components
import NeuroState from './components/NeuroState.vue';
import CouncilLog from './components/CouncilLog.vue';
import NormInspector from './components/NormInspector.vue';

// State
const connected = ref(false);
const telemetry = ref({ rpm: 0, load: 0, vibration: 0 });
const neuro = ref({ dopamine: 50, cortisol: 10, serotonin: 70 });
const action = ref("INITIALIZING...");
const timestamp = ref("");
const swarmNodes = ref([]);
const councilLogs = ref([]);

let socket = null;

const connect = () => {
  socket = new WebSocket('ws://localhost:8000/ws/telemetry/VUE_MASTER');
  
  socket.onopen = () => {
    connected.value = true;
    addLog("SYSTEM", "Uplink established.");
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    telemetry.value = {
      rpm: data.rpm || 0,
      load: data.load || 0,
      vibration: data.vibration || 0
    };
    if (data.neuro_state) neuro.value = data.neuro_state;
    if (data.action) action.value = data.action;
    
    // Simulate discrete logs
    if (Math.random() > 0.98) {
       const agents = ["AUDITOR", "VISION", "REGEDIT"];
       const msgs = ["Optimizing feed rate...", "Scanning surface finish...", "Vibration drift detected."];
       addLog(agents[Math.floor(Math.random()*agents.length)], msgs[Math.floor(Math.random()*msgs.length)]);
    }
  };

  socket.onclose = () => {
    connected.value = false;
    setTimeout(connect, 3000);
  };
};

const addLog = (agent, message) => {
  councilLogs.value.push({
    time: new Date().toLocaleTimeString(),
    agent,
    message,
    type: agent
  });
  if (councilLogs.value.length > 50) councilLogs.value.shift();
};

const fetchSwarm = async () => {
  try {
    const res = await fetch('http://localhost:8000/api/swarm/status');
    const data = await res.json();
    if (data.machines) {
        swarmNodes.value = Object.entries(data.machines).map(([id, info]) => ({
          id, ...info
        }));
    }
  } catch (e) { }
};

onMounted(() => {
  connect();
  fetchSwarm();
  setInterval(() => { timestamp.value = new Date().toLocaleTimeString(); }, 1000);
  setInterval(fetchSwarm, 5000);
});

onUnmounted(() => { if (socket) socket.close(); });
</script>

<style>
/* GLOBAL RESETS & VARS */
:root {
  --bg-dark: #050508;
  --panel-bg: rgba(20, 20, 30, 0.6);
  --primary: #00ff88;
  --secondary: #00d4ff;
  --accent: #ff00ff;
  --danger: #ff4400;
  --font-ui: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

body {
  margin: 0;
  background: var(--bg-dark);
  color: #fff;
  font-family: var(--font-ui);
  overflow: hidden;
}

.vue-nexus {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: radial-gradient(circle at 50% -20%, #1a1a2e, var(--bg-dark));
}

/* HEADER */
.app-header {
  height: 70px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 30px;
  background: rgba(0,0,0,0.3);
  backdrop-filter: blur(5px);
}

.brand {
  display: flex;
  align-items: center;
  gap: 15px;
  color: var(--primary);
}

.icon-pulse { animation: pulse 2s infinite; }

.brand-text h1 {
  font-size: 1.2rem;
  margin: 0;
  letter-spacing: 2px;
  font-weight: 900;
  text-transform: uppercase;
}

.subtitle {
  font-size: 0.6rem;
  color: #666;
  letter-spacing: 3px;
}

.header-stats {
  display: flex;
  gap: 30px;
  font-family: var(--font-mono);
}

.h-stat {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.h-stat label {
  font-size: 0.55rem;
  color: #555;
  font-weight: bold;
}

.h-stat span {
  font-size: 0.8rem;
  color: #888;
}

.h-stat span.active { color: #fff; text-shadow: 0 0 5px #fff; }
.h-stat span.online { color: var(--primary); text-shadow: 0 0 5px var(--primary); }

/* MAIN GRID */
.main-grid {
  flex-grow: 1;
  display: grid;
  grid-template-columns: 350px 1fr 350px;
  gap: 20px;
  padding: 20px;
  overflow: hidden;
}

/* PANEL STYLES */
.panel-box {
  background: var(--panel-bg);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 8px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  margin-bottom: 20px;
}

.panel-title {
  font-size: 0.6rem;
  color: #555;
  font-weight: 900;
  letter-spacing: 1px;
  margin-bottom: 15px;
}

/* LEFT COLUMN */
.telemetry-display {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.rpm-metric {
  position: relative;
  width: 150px;
  height: 150px;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
}

.rpm-dial {
  position: absolute;
  width: 100%;
  height: 100%;
}

.metric-val {
  font-size: 2rem;
  font-family: var(--font-mono);
  font-weight: 800;
  z-index: 2;
}

.metric-label {
  font-size: 0.7rem;
  color: var(--secondary);
}

.sub-metrics {
  width: 100%;
  margin-top: 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.sm-row {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
}

.bar-bg {
  flex-grow: 1;
  height: 6px;
  background: rgba(255,255,255,0.1);
  border-radius: 3px;
}

.bar-fill { height: 100%; border-radius: 3px; }
.red { background: var(--danger); box-shadow: 0 0 5px var(--danger); }
.orange { background: #ffaa00; box-shadow: 0 0 5px #ffaa00; }

.swarm-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.swarm-node {
  background: rgba(255,255,255,0.05);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.6rem;
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
}

.status-dot { width: 6px; height: 6px; border-radius: 50%; }
.no-swarm { font-size: 0.6rem; opacity: 0.5; padding: 10px; }

/* CENTER COLUMN */
.col-center {
  display: flex;
  flex-direction: column;
  gap: 20px;
  overflow-y: auto; /* Allow scroll if needed */
  padding-right: 5px;
}

/* RIGHT COLUMN */
.action-display {
  text-align: center;
  padding: 20px 0;
}

.action-text {
  font-size: 1.5rem;
  font-weight: 800;
  color: #fff;
  text-shadow: 0 0 10px rgba(255,255,255,0.5);
  margin: 0 0 20px 0;
}

.confidence-meter label { display: block; font-size: 0.6rem; color: #555; margin-bottom: 5px; }
.conf-bar { height: 4px; background: rgba(255,255,255,0.1); width: 100%; }
.conf-fill { height: 100%; background: var(--secondary); box-shadow: 0 0 5px var(--secondary); }

.controls-panel { gap: 10px; }
.control-btn {
  padding: 12px;
  border: none;
  font-weight: 800;
  font-size: 0.7rem;
  cursor: pointer;
  border-radius: 4px;
  transition: 0.2s;
}

.control-btn.danger { background: rgba(255, 68, 0, 0.2); color: var(--danger); border: 1px solid var(--danger); }
.control-btn.danger:hover { background: var(--danger); color: #000; }

.control-btn.warning { background: rgba(255, 170, 0, 0.2); color: #ffaa00; border: 1px solid #ffaa00; }
.control-btn.primary { background: rgba(0, 255, 136, 0.1); color: var(--primary); border: 1px solid var(--primary); }
.control-btn.primary:hover { background: var(--primary); color: #000; }

/* SCROLLBARS */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
</style>
