<template>
  <div class="norm-inspector glass-panel">
    <div class="panel-header">
      <div class="header-left">
        <span class="icon">🌐</span>
        <span class="title">GLOBAL BENCHMARKING</span>
      </div>
      <button class="scan-btn" @click="fetchNorms" :disabled="loading">
        {{ loading ? 'SCANNING...' : 'INIT SCAN' }}
      </button>
    </div>
    
    <div class="data-grid">
      <div class="grid-header">
        <span>MATERIAL</span>
        <span>GLOBAL STARTARD</span>
        <span>LOCAL PERFORMANCE</span>
        <span>RATING</span>
      </div>
      
      <div class="scroller">
        <div v-for="item in report" :key="item.material" class="grid-row" :class="item.status">
          <span class="mat-name">{{ item.material.replace('6Al4V', '').replace('6061', '') }}</span>
          <span class="val-global">{{ item.global_sfm }} <small>SFM</small></span>
          <span class="val-local">{{ item.local_sfm }} <small>SFM</small></span>
          
          <div class="status-cell">
             <div class="badge">{{ item.status }}</div>
             <div class="efficiency-bar">
               <div class="fill" :style="{ width: Math.min(item.efficiency, 150) + '%' }"></div>
             </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const report = ref([]);
const loading = ref(false);

const fetchNorms = async () => {
  loading.value = true;
  try {
    const res = await fetch('http://localhost:8000/api/benchmarking/compare');
    if (!res.ok) throw new Error("API Error");
    const data = await res.json();
    report.value = data.report;
  } catch (e) {
    // Mock for demo if backend offline
    setTimeout(() => {
        report.value = [
           { material: "Aluminum6061", global_sfm: 600, local_sfm: 750, efficiency: 125, status: "ELITE" },
           { material: "Titanium", global_sfm: 150, local_sfm: 140, efficiency: 93, status: "OPTIMAL" },
           { material: "Steel4140", global_sfm: 300, local_sfm: 0, efficiency: 0, status: "UNKNOWN" }
        ];
    }, 500);
  } finally {
    setTimeout(() => loading.value = false, 800);
  }
};

onMounted(() => {
  fetchNorms();
});
</script>

<style scoped>
.glass-panel {
  background: rgba(16, 20, 28, 0.7);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(0, 212, 255, 0.1);
  border-radius: 6px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  height: 250px;
  font-family: 'JetBrains Mono', monospace;
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
}

.panel-header {
  background: rgba(0, 212, 255, 0.05);
  padding: 10px 15px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(0, 212, 255, 0.1);
}

.header-left { display: flex; gap: 10px; align-items: center; }
.title { font-size: 0.7rem; font-weight: 800; color: #00d4ff; letter-spacing: 1px; }

.scan-btn {
  background: rgba(0, 212, 255, 0.1);
  border: 1px solid #00d4ff;
  color: #00d4ff;
  font-family: inherit;
  font-size: 0.6rem;
  padding: 4px 10px;
  cursor: pointer;
  transition: 0.2s;
}
.scan-btn:hover { background: #00d4ff; color: #000; }

.data-grid {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  padding: 10px;
}

.grid-header {
  display: grid;
  grid-template-columns: 1.5fr 1fr 1fr 1.5fr;
  font-size: 0.6rem;
  color: #555;
  padding-bottom: 5px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  margin-bottom: 5px;
}

.scroller {
  overflow-y: auto;
  flex-grow: 1;
}

.grid-row {
  display: grid;
  grid-template-columns: 1.5fr 1fr 1fr 1.5fr;
  font-size: 0.75rem;
  color: #bbb;
  padding: 8px 5px;
  border-bottom: 1px solid rgba(255,255,255,0.02);
  align-items: center;
}

.grid-row:hover { background: rgba(255,255,255,0.03); }

.mat-name { color: #fff; font-weight: bold; }
.val-global { color: #666; }
.val-local { color: #ccc; }
small { font-size: 0.5rem; opacity: 0.5; }

.status-cell {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.badge {
  font-size: 0.6rem;
  font-weight: bold;
}

.efficiency-bar {
  height: 2px;
  background: rgba(255,255,255,0.1);
  width: 100%;
}
.fill { height: 100%; box-shadow: 0 0 5px currentColor; }

/* Status Styles */
.ELITE .badge { color: #00ff88; }
.ELITE .fill { background: #00ff88; width: 100%; }

.OPTIMAL .badge { color: #00d4ff; }
.OPTIMAL .fill { background: #00d4ff; }

.SAFE .badge { color: #aaa; }
.SAFE .fill { background: #aaa; }

.CONSERVATIVE .badge { color: #ff8800; }
.CONSERVATIVE .fill { background: #ff8800; }

.UNKNOWN .badge { color: #444; }
.UNKNOWN .fill { display: none; }

</style>
