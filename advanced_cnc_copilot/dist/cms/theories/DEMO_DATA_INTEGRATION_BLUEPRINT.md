# DEMO DATA INTEGRATION BLUEPRINT
## Complete Implementation Plan for Live Dashboard

---

## 🎯 OBJECTIVE

Transform static `index.html` into **live working demo** with realistic data flowing through all dashboard panels.

---

## STEP 1: BACKEND API ENDPOINTS (Python/FastAPI)

### A. Update `cms/fanuc_api.py`

Add these endpoints to serve demo data:

```python
from cms.demo_data_generator import DemoDataGenerator

# Initialize generator
demo_gen = DemoDataGenerator()

@app.get("/api/telemetry/current")
async def get_current_telemetry():
    """Real-time telemetry snapshot"""
    return demo_gen.generate_telemetry()

@app.get("/api/telemetry/stream")
async def telemetry_stream():
    """Server-Sent Events for live updates"""
    async def event_generator():
        while True:
            data = demo_gen.generate_telemetry()
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)  # 1Hz update
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/telemetry/history")
async def get_history(minutes: int = 60):
    """Historical telemetry for charts"""
    return demo_gen.generate_historical_batch(minutes)

@app.get("/api/projects")
async def get_projects():
    """Projects dataset for LLM context"""
    return demo_gen.generate_projects_dataset(20)
```

---

## STEP 2: FRONTEND JAVASCRIPT (Dashboard Integration)

### B. Create `cms/dashboard/app.js`

```javascript
// ====================
// CONFIGURATION
// ====================
const API_BASE = 'http://localhost:8000';
const UPDATE_INTERVAL = 1000; // 1 second

// ====================
// DATA FETCHER
// ====================
class DashboardDataFetcher {
    constructor() {
        this.currentData = null;
        this.history = [];
    }
    
    async fetchCurrent() {
        try {
            const response = await fetch(`${API_BASE}/api/telemetry/current`);
            this.currentData = await response.json();
            return this.currentData;
        } catch (error) {
            console.error('Failed to fetch telemetry:', error);
            return this.getMockFallback();
        }
    }
    
    async fetchHistory() {
        try {
            const response = await fetch(`${API_BASE}/api/telemetry/history?minutes=60`);
            this.history = await response.json();
            return this.history;
        } catch (error) {
            console.error('Failed to fetch history:', error);
            return [];
        }
    }
    
    getMockFallback() {
        // Fallback if API down
        return {
            rpm: 8000,
            load: 65,
            vibration: {z: 0.03},
            dopamine: 75,
            cortisol: 25,
            signal: 'GREEN'
        };
    }
}

// ====================
// UI UPDATER
// ====================
class DashboardUI {
    constructor(dataFetcher) {
        this.fetcher = dataFetcher;
        this.charts = {};
    }
    
    // Update Sensory Panel (#1)
    updateSensoryPanel(data) {
        document.getElementById('rpm-value').textContent = data.rpm;
        document.getElementById('load-value').textContent = data.load + '%';
        document.getElementById('vibration-value').textContent = data.vibration.z.toFixed(3) + 'g';
        
        // Update progress bars
        this.updateBar('rpm-bar', (data.rpm / 12000) * 100);
        this.updateBar('load-bar', data.load);
        this.updateBar('vibration-bar', (data.vibration.z / 0.2) * 100);
    }
    
    // Update Neuro-Engine Panel (#2)
    updateNeuroPanel(data) {
        this.updateBar('dopamine-bar', data.dopamine);
        this.updateBar('cortisol-bar', data.cortisol);
        this.updateBar('serotonin-bar', data.serotonin || 70);
    }
    
    // Update Logic Map Panel (#3)
    updateLogicMap(data) {
        const flow = data.logic_flow;
        document.getElementById('logic-sensory').textContent = flow.sensory;
        document.getElementById('logic-reward').textContent = flow.reward;
        document.getElementById('logic-signal').textContent = flow.signal;
        document.getElementById('logic-economics').textContent = '€' + flow.economics;
        
        // Color code signal
        const signalEl = document.getElementById('logic-signal');
        signalEl.className = `logic-node-val signal-${flow.signal.toLowerCase()}`;
    }
    
    // Update helper
    updateBar(barId, percentage) {
        const bar = document.getElementById(barId);
        if (bar) {
            bar.style.width = Math.min(100, Math.max(0, percentage)) + '%';
            
            // Color based on value
            if (percentage > 90) bar.classList.add('danger');
            else if (percentage > 75) bar.classList.add('warning');
            else bar.classList.remove('danger', 'warning');
        }
    }
    
    // Initialize Chart.js graphs
    initCharts() {
        const ctx = document.getElementById('load-chart');
        if (ctx && typeof Chart !== 'undefined') {
            this.charts.load = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Load %',
                        data: [],
                        borderColor: 'rgb(56, 189, 248)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
    }
    
    // Update chart with new data
    updateChart(history) {
        if (this.charts.load && history.length > 0) {
            this.charts.load.data.labels = history.map(h => new Date(h.timestamp).toLocaleTimeString());
            this.charts.load.data.datasets[0].data = history.map(h => h.load);
            this.charts.load.update();
        }
    }
}

// ====================
// MAIN APPLICATION
// ====================
class FanucRiseApp {
    constructor() {
        this.fetcher = new DashboardDataFetcher();
        this.ui = new DashboardUI(this.fetcher);
        this.isRunning = false;
    }
    
    async init() {
        console.log('🚀 Fanuc Rise Dashboard initializing...');
        
        // Fetch initial data
        await this.fetcher.fetchCurrent();
        const history = await this.fetcher.fetchHistory();
        
        // Setup UI
        this.ui.initCharts();
        this.ui.updateChart(history);
        
        // Start live updates
        this.startLiveUpdates();
        
        console.log('✅ Dashboard ready!');
    }
    
    startLiveUpdates() {
        this.isRunning = true;
        
        setInterval(async () => {
            if (!this.isRunning) return;
            
            const data = await this.fetcher.fetchCurrent();
            if (data) {
                this.ui.updateSensoryPanel(data);
                this.ui.updateNeuroPanel(data);
                this.ui.updateLogicMap(data);
                
                // Update timestamp
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            }
        }, UPDATE_INTERVAL);
    }
    
    stop() {
        this.isRunning = false;
    }
}

// Auto-start when DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.fanucRiseApp = new FanucRiseApp();
    window.fanucRiseApp.init();
});
```

---

## STEP 3: UPDATE INDEX.HTML

### C. Modify `cms/dashboard/index.html`

Add before closing `</body>` tag:

```html
<!-- Load Chart.js for graphs -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

<!-- Load our app -->
<script src="app.js"></script>

<!-- Add IDs to elements for JavaScript to target -->
<script>
// Quick fix: Add IDs to existing HTML elements if missing
document.addEventListener('DOMContentLoaded', () => {
    // Find all .value elements and add IDs
    const values = document.querySelectorAll('.metric-card .value');
    if (values[0]) values[0].id = 'rpm-value';
    if (values[1]) values[1].id = 'load-value';
    if (values[2]) values[2].id = 'vibration-value';
    
    // Add IDs to bars
    const bars = document.querySelectorAll('.bar');
    if (bars[0]) bars[0].id = 'rpm-bar';
    if (bars[1]) bars[1].id = 'load-bar';
    if (bars[2]) bars[2].id = 'vibration-bar';
});
</script>
```

---

## STEP 4: CORS FIX (If needed)

### D. Update `cms/fanuc_api.py` CORS settings

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## STEP 5: TESTING CHECKLIST

### E. Verification Steps

```bash
# 1. Backend running?
curl http://localhost:8000/api/telemetry/current
# Should return JSON

# 2. Generate demo data
python cms/demo_data_generator.py
# Should print sample data

# 3. Open dashboard
start http://localhost:8000/dashboard/index.html

# 4. Check browser console (F12)
# Should see: "✅ Dashboard ready!"
```

---

## USER-LEVEL VIEWS (All Aspects)

### Panel #1: Sensory (Real-time Gauges)
- **RPM**: 8234 (updates every second)
- **Load**: 67% (animated bar)
- **Vibration**: 0.023g (color changes green/amber/red)

### Panel #2: Neuro-Engine (Emotional State)
- **Dopamine**: 75% (reward score)
- **Cortisol**: 25% (stress indicator)
- **Serotonin**: 70% (stability)

### Panel #3: Logic Map (Decision Flow)
- **Sensory**: 67 (input)
- **Reward**: 75 (dopamine)
- **Signal**: GREEN (traffic light)
- **Economics**: €25.7 (cost impact)

### Panel #4: Economics (Business Metrics)
- **Cost/Part**: €25.73
- **Throughput**: 8.3 parts/hour
- **Cycle Time**: 7.2 min
- **Parts Done**: 42

### Panel #5: Chart (Historical Trends)
- **Load Chart**: Last 60 minutes (line graph)
- **Vibration Chart**: Frequency analysis
- **Dopamine Chart**: Mood over time

---

## EXPECTED RESULT

When you open `http://localhost:8000/dashboard/index.html`:

1. ✅ All gauges show **live numbers** (updating every 1s)
2. ✅ Progress bars **animate**
3. ✅ Signal lights **change color** (Green/Amber/Red)
4. ✅ Charts **scroll** with new data
5. ✅ Browser console shows no errors

---

## NEXT STEPS AFTER BLUEPRINT

1. **Week 1**: Implement `app.js` (JavaScript dashboard controller)
2. **Week 2**: Add Chart.js visualizations
3. **Week 3**: WebSocket upgrade (replace polling with push)
4. **Week 4**: Add user interactions (buttons, sliders for manual override)

---

*Blueprint by Dusan Berger, ready for implementation, January 2026*
