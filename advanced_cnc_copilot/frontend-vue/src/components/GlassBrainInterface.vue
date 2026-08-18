<template>
  <div class="glass-brain-interface">
    <div class="neuro-safety-display">
      <!-- Visual representation of dopamine/cortisol gradients -->
      <div class="gradient-indicators">
        <div class="dopamine-indicator" 
             :style="{ 
               'border-color': dopamineColor, 
               'animation': `pulse ${dopaminePulseSpeed}s infinite alternate` 
             }">
          <div class="indicator-label">DOPAMINE (Reward)</div>
          <div class="indicator-value">{{ dopamineLevel.toFixed(3) }}</div>
          <div class="indicator-bar">
            <div class="fill" :style="{ width: `${dopamineLevel * 100}%`, 'background-color': dopamineColor }"></div>
          </div>
        </div>
        
        <div class="cortisol-indicator"
             :style="{ 
               'border-color': cortisolColor, 
               'animation': `pulse ${cortisolPulseSpeed}s infinite alternate` 
             }">
          <div class="indicator-label">CORTISOL (Stress)</div>
          <div class="indicator-value">{{ cortisolLevel.toFixed(3) }}</div>
          <div class="indicator-bar">
            <div class="fill" :style="{ width: `${cortisolLevel * 100}%`, 'background-color': cortisolColor }"></div>
          </div>
        </div>
      </div>
      
      <!-- Machine status visualization -->
      <div class="machine-status" :class="statusClass">
        <div class="status-header">
          <h3>Machine Status: {{ currentMode }}</h3>
          <div class="status-indicator" :style="{ 'background-color': statusColor }"></div>
        </div>
        
        <div class="telemetry-data">
          <div class="telemetry-item">
            <span class="label">Spindle Load:</span>
            <span class="value">{{ telemetry.spindle_load }}%</span>
          </div>
          <div class="telemetry-item">
            <span class="label">Temperature:</span>
            <span class="value">{{ telemetry.temperature }}°C</span>
          </div>
          <div class="telemetry-item">
            <span class="label">Vibration X:</span>
            <span class="value">{{ telemetry.vibration_x }}G</span>
          </div>
          <div class="telemetry-item">
            <span class="label">Vibration Y:</span>
            <span class="value">{{ telemetry.vibration_y }}G</span>
          </div>
          <div class="telemetry-item">
            <span class="label">Feed Rate:</span>
            <span class="value">{{ telemetry.feed_rate }} mm/min</span>
          </div>
          <div class="telemetry-item">
            <span class="label">RPM:</span>
            <span class="value">{{ telemetry.rpm }}</span>
          </div>
        </div>
      </div>
      
      <!-- Shadow Council Decision Visualization -->
      <div class="shadow-council-decision">
        <h4>Shadow Council Decision Trace</h4>
        <div class="decision-path">
          <div class="agent-decision creator-agent" :class="{ active: decisionTrace.creator_active }">
            <div class="agent-name">CREATOR AGENT</div>
            <div class="decision-output">{{ decisionTrace.creator_output }}</div>
            <div class="confidence-score">Confidence: {{ decisionTrace.creator_confidence }}</div>
          </div>
          
          <div class="agent-decision auditor-agent" :class="{ active: decisionTrace.auditor_active, rejected: decisionTrace.auditor_rejected }">
            <div class="agent-name">AUDITOR AGENT</div>
            <div class="decision-output">{{ decisionTrace.auditor_output }}</div>
            <div class="reasoning-trace">Reasoning: {{ decisionTrace.auditor_reasoning }}</div>
          </div>
          
          <div class="agent-decision accountant-agent" :class="{ active: decisionTrace.accountant_active }">
            <div class="agent-name">ACCOUNTANT AGENT</div>
            <div class="decision-output">{{ decisionTrace.accountant_output }}</div>
            <div class="economic-impact">Economic Impact: ${{ decisionTrace.economic_impact }}/hr</div>
          </div>
        </div>
        
        <div class="final-decision" :class="{ approved: decisionTrace.final_approval, rejected: !decisionTrace.final_approval }">
          <div class="decision-label">COUNCIL DECISION:</div>
          <div class="decision-value">{{ decisionTrace.final_approval ? 'APPROVED' : 'REJECTED' }}</div>
          <div class="fitness-score">Fitness: {{ decisionTrace.fitness_score }}</div>
        </div>
      </div>
      
      <!-- Genetic Lineage Visualization -->
      <div class="genetic-lineage">
        <h4>Strategy Genetic Lineage</h4>
        <div class="lineage-tree">
          <div v-for="generation in lineageHistory" :key="generation.generation_id" class="generation-node">
            <div class="node-info">
              <div class="generation-label">Generation {{ generation.number }}</div>
              <div class="mutation-type">{{ generation.mutation_type }}</div>
              <div class="fitness-change">ΔFitness: {{ generation.fitness_change }}</div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Economic Dashboard -->
      <div class="economic-dashboard">
        <h4>Economic Metrics</h4>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-label">Profit Rate</div>
            <div class="metric-value">${{ economicMetrics.profit_rate }}/hr</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Churn Risk</div>
            <div class="metric-value">{{ economicMetrics.churn_risk }}%</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">ROI</div>
            <div class="metric-value">{{ economicMetrics.roi }}%</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Savings Today</div>
            <div class="metric-value">${{ economicMetrics.daily_savings }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, onUnmounted } from 'vue';

export default {
  name: 'GlassBrainInterface',
  setup() {
    // Reactive data
    const dopamineLevel = ref(0.65);
    const cortisolLevel = ref(0.23);
    const telemetry = ref({
      spindle_load: 65.0,
      temperature: 38.5,
      vibration_x: 0.3,
      vibration_y: 0.2,
      feed_rate: 2000,
      rpm: 4000
    });
    const currentMode = ref('BALANCED_MODE');
    const decisionTrace = ref({
      creator_active: true,
      creator_output: 'Proposed aggressive feed optimization',
      creator_confidence: 0.85,
      auditor_active: true,
      auditor_output: 'Validated against physics constraints',
      auditor_rejected: false,
      auditor_reasoning: 'All constraints satisfied, no death penalty applied',
      accountant_active: true,
      accountant_output: 'Economic impact: $125/hr improvement',
      economic_impact: 125.00,
      final_approval: true,
      fitness_score: 0.87
    });
    const lineageHistory = ref([
      { generation_id: 'G001', number: 1, mutation_type: 'Initial Strategy', fitness_change: 0.0 },
      { generation_id: 'G002', number: 2, mutation_type: 'Parameter Optimization', fitness_change: 0.12 },
      { generation_id: 'G003', number: 3, mutation_type: 'Feed Rate Adjustment', fitness_change: 0.08 },
      { generation_id: 'G004', number: 4, mutation_type: 'Current Generation', fitness_change: 0.03 }
    ]);
    const economicMetrics = ref({
      profit_rate: 285.50,
      churn_risk: 12.5,
      roi: 24.3,
      daily_savings: 1245.75
    });
    
    // Calculate derived properties
    const dopamineColor = computed(() => {
      // Color based on dopamine level: green (high reward) to yellow (low reward)
      const intensity = Math.floor(dopamineLevel.value * 255);
      return `rgb(0, ${intensity}, ${255 - intensity})`;
    });
    
    const cortisolColor = computed(() => {
      // Color based on cortisol level: orange (low stress) to red (high stress)
      const greenVal = Math.floor((1 - cortisolLevel.value) * 200 + 55);
      return `rgb(255, ${greenVal}, 0)`;
    });
    
    const statusClass = computed(() => {
      if (cortisolLevel.value > 0.8) return 'critical';
      if (cortisolLevel.value > 0.6) return 'warning';
      if (cortisolLevel.value > 0.4) return 'caution';
      return 'normal';
    });
    
    const statusColor = computed(() => {
      if (cortisolLevel.value > 0.8) return '#EF4444';  // Red for critical
      if (cortisolLevel.value > 0.6) return '#F59E0B'; // Amber for warning
      if (cortisolLevel.value > 0.4) return '#3B82F6'; // Blue for caution
      return '#10B981';  // Green for normal
    });
    
    const dopaminePulseSpeed = computed(() => {
      // Faster pulse when dopamine is high (excited state)
      return 2.0 - (dopamineLevel.value * 1.5);  // 0.5s to 2.0s pulse
    });
    
    const cortisolPulseSpeed = computed(() => {
      // Faster pulse when cortisol is high (stress response)
      return 1.5 - (cortisolLevel.value * 0.8);  // 0.7s to 1.5s pulse
    });
    
    // Simulate real-time data updates
    let updateInterval;
    
    const updateData = () => {
      // Simulate real-time telemetry updates
      telemetry.value.spindle_load = Math.max(30, Math.min(95, telemetry.value.spindle_load + (Math.random() - 0.5) * 2));
      telemetry.value.temperature = Math.max(25, Math.min(75, telemetry.value.temperature + (Math.random() - 0.5) * 1.5));
      telemetry.value.vibration_x = Math.max(0.1, Math.min(3.0, telemetry.value.vibration_x + (Math.random() - 0.5) * 0.3));
      telemetry.value.vibration_y = Math.max(0.1, Math.min(3.0, telemetry.value.vibration_y + (Math.random() - 0.5) * 0.3));
      telemetry.value.feed_rate = Math.max(1000, Math.min(5000, telemetry.value.feed_rate + (Math.random() - 0.5) * 50));
      telemetry.value.rpm = Math.max(2000, Math.min(12000, telemetry.value.rpm + (Math.random() - 0.5) * 100));
      
      // Update neuro-safety gradients based on telemetry
      const avg_vibration = (telemetry.value.vibration_x + telemetry.value.vibration_y) / 2;
      const stress_factor = 
        (telemetry.value.spindle_load / 100 * 0.3) +
        (telemetry.value.temperature / 80 * 0.3) +
        (avg_vibration / 3 * 0.4);
      
      // Apply exponential smoothing to cortisol (stress accumulates and decays slowly)
      cortisolLevel.value = cortisolLevel.value * 0.9 + stress_factor * 0.1;
      
      // Update dopamine based on efficiency (higher feed/rpm = higher reward, but penalized by stress)
      const efficiency_factor = 
        (telemetry.value.feed_rate / 5000 * 0.5) +
        (telemetry.value.rpm / 12000 * 0.3) +
        ((100 - telemetry.value.spindle_load) / 100 * 0.2);
      
      // Dopamine decreases with stress
      dopamineLevel.value = Math.max(0, dopamineLevel.value * 0.95 + (efficiency_factor * (1 - cortisolLevel.value * 0.5)) * 0.05);
      
      // Determine operational mode based on neuro-safety levels
      if (cortisolLevel.value > 0.7 && dopamineLevel.value < 0.3) {
        currentMode.value = 'ECONOMY_MODE';
      } else if (dopamineLevel.value > 0.7 && cortisolLevel.value < 0.4) {
        currentMode.value = 'RUSH_MODE';
      } else {
        currentMode.value = 'BALANCED_MODE';
      }
      
      // Simulate occasional decision updates
      if (Math.random() < 0.1) {  // 10% chance of decision update per tick
        updateDecisionTrace();
      }
    };
    
    const updateDecisionTrace = () => {
      // Simulate a new decision trace from the Shadow Council
      const decisions = [
        {
          creator_output: 'Proposed feed rate increase to 2400mm/min',
          creator_confidence: 0.82,
          auditor_output: 'Validated - within Quadratic Mantinel limits',
          auditor_rejected: false,
          auditor_reasoning: 'Feed rate safe for current curvature radius',
          accountant_output: 'Economic impact: +$45/hr profit',
          economic_impact: 45.00,
          final_approval: true,
          fitness_score: 0.89
        },
        {
          creator_output: 'Proposed RPM increase to 8500',
          creator_confidence: 0.78,
          auditor_output: 'REJECTED - Death Penalty applied',
          auditor_rejected: true,
          auditor_reasoning: 'Thermal limits would be exceeded with current coolant flow',
          accountant_output: 'Economic impact: -$0/hr (prevented thermal damage)',
          economic_impact: 0.00,
          final_approval: false,
          fitness_score: 0.0
        },
        {
          creator_output: 'Proposed conservative parameters for Inconel',
          creator_confidence: 0.91,
          auditor_output: 'Validated - conservative approach approved',
          auditor_rejected: false,
          auditor_reasoning: 'All constraints satisfied with safety margin',
          accountant_output: 'Economic impact: -$15/hr but zero risk',
          economic_impact: -15.00,
          final_approval: true,
          fitness_score: 0.72
        }
      ];
      
      const randomDecision = decisions[Math.floor(Math.random() * decisions.length)];
      
      decisionTrace.value = {
        ...decisionTrace.value,
        ...randomDecision,
        creator_active: true,
        auditor_active: true,
        accountant_active: true
      };
    };
    
    onMounted(() => {
      // Start real-time updates
      updateInterval = setInterval(updateData, 1000); // Update every second
    });
    
    onUnmounted(() => {
      // Clean up interval
      if (updateInterval) {
        clearInterval(updateInterval);
      }
    });
    
    return {
      dopamineLevel,
      cortisolLevel,
      telemetry,
      currentMode,
      decisionTrace,
      lineageHistory,
      economicMetrics,
      dopamineColor,
      cortisolColor,
      statusClass,
      statusColor,
      dopaminePulseSpeed,
      cortisolPulseSpeed
    };
  }
};
</script>

<style scoped>
.glass-brain-interface {
  font-family: 'Inter', sans-serif;
  padding: 20px;
  background: linear-gradient(135deg, #0f172a, #1e293b);
  color: #e2e8f0;
  min-height: 100vh;
}

.neuro-safety-display {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.gradient-indicators {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.dopamine-indicator, .cortisol-indicator {
  border: 2px solid;
  border-radius: 8px;
  padding: 15px;
  transition: all 0.3s ease;
}

.indicator-label {
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 5px;
  color: #94a3b8;
}

.indicator-value {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 10px;
}

.indicator-bar {
  height: 20px;
  background-color: #334155;
  border-radius: 10px;
  overflow: hidden;
}

.indicator-bar .fill {
  height: 100%;
  transition: width 0.5s ease;
}

.machine-status {
  border-radius: 8px;
  padding: 20px;
  margin-top: 20px;
  border: 2px solid;
  transition: all 0.3s ease;
}

.machine-status.normal {
  border-color: #10b981;
  background-color: rgba(16, 185, 129, 0.1);
}

.machine-status.caution {
  border-color: #3b82f6;
  background-color: rgba(59, 130, 246, 0.1);
}

.machine-status.warning {
  border-color: #f59e0b;
  background-color: rgba(245, 158, 11, 0.1);
}

.machine-status.critical {
  border-color: #ef4444;
  background-color: rgba(239, 68, 68, 0.1);
}

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.status-header h3 {
  margin: 0;
  font-size: 1.2rem;
}

.status-indicator {
  width: 20px;
  height: 20px;
  border-radius: 50%;
}

.telemetry-data {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}

.telemetry-item {
  display: flex;
  justify-content: space-between;
  padding: 8px;
  border-bottom: 1px solid #334155;
}

.label {
  color: #94a3b8;
}

.value {
  font-weight: 600;
}

.shadow-council-decision {
  grid-column: span 2;
  margin-top: 30px;
  padding: 20px;
  border: 1px solid #475569;
  border-radius: 8px;
  background-color: rgba(30, 41, 59, 0.5);
}

.decision-path {
  display: flex;
  justify-content: space-around;
  margin: 20px 0;
  position: relative;
}

.decision-path::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 2px;
  background: #64748b;
  z-index: 1;
}

.agent-decision {
  background: #1e293b;
  border: 1px solid #64748b;
  border-radius: 8px;
  padding: 15px;
  width: 30%;
  position: relative;
  z-index: 2;
  text-align: center;
}

.agent-decision.active {
  border-width: 2px;
  box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
}

.agent-decision.creator-agent {
  border-color: #8b5cf6;
}

.agent-decision.auditor-agent.rejected {
  border-color: #ef4444;
  background-color: rgba(239, 68, 68, 0.1);
}

.agent-decision.accountant-agent {
  border-color: #10b981;
}

.agent-name {
  font-weight: 700;
  margin-bottom: 10px;
  font-size: 1.1rem;
}

.reasoning-trace {
  font-size: 0.8rem;
  color: #cbd5e1;
  margin-top: 8px;
  font-style: italic;
}

.confidence-score, .economic-impact {
  font-size: 0.9rem;
  color: #94a3b8;
  margin-top: 5px;
}

.final-decision {
  text-align: center;
  padding: 15px;
  margin-top: 15px;
  border-radius: 8px;
  font-weight: 700;
  font-size: 1.2rem;
}

.final-decision.approved {
  background-color: rgba(16, 185, 129, 0.2);
  border: 1px solid #10b981;
}

.final-decision.rejected {
  background-color: rgba(239, 68, 68, 0.2);
  border: 1px solid #ef4444;
}

.genetic-lineage {
  grid-column: span 2;
  margin-top: 20px;
  padding: 20px;
  border: 1px solid #475569;
  border-radius: 8px;
  background-color: rgba(30, 41, 59, 0.5);
}

.lineage-tree {
  display: flex;
  overflow-x: auto;
  padding: 10px 0;
  gap: 10px;
}

.generation-node {
  min-width: 150px;
  background: #1e293b;
  border: 1px solid #64748b;
  border-radius: 8px;
  padding: 10px;
  text-align: center;
}

.generation-label {
  font-weight: 700;
  color: #60a5fa;
}

.mutation-type {
  font-size: 0.9rem;
  margin: 5px 0;
}

.fitness-change {
  font-size: 0.8rem;
  color: #94a3b8;
}

.economic-dashboard {
  grid-column: span 2;
  margin-top: 20px;
  padding: 20px;
  border: 1px solid #475569;
  border-radius: 8px;
  background-color: rgba(30, 41, 59, 0.5);
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 15px;
  margin-top: 15px;
}

.metric-card {
  background: #1e293b;
  border: 1px solid #64748b;
  border-radius: 8px;
  padding: 15px;
  text-align: center;
}

.metric-label {
  font-size: 0.9rem;
  color: #94a3b8;
  margin-bottom: 5px;
}

.metric-value {
  font-size: 1.3rem;
  font-weight: 700;
}

/* Animation for pulsing borders */
@keyframes pulse {
  0% {
    border-width: 2px;
    box-shadow: 0 0 5px currentColor;
  }
  100% {
    border-width: 4px;
    box-shadow: 0 0 15px currentColor;
  }
}

/* Responsive design */
@media (max-width: 768px) {
  .neuro-safety-display {
    grid-template-columns: 1fr;
  }
  
  .decision-path {
    flex-direction: column;
    gap: 15px;
  }
  
  .decision-path::before {
    display: none;
  }
  
  .agent-decision {
    width: 100%;
  }
  
  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .telemetry-data {
    grid-template-columns: 1fr;
  }
}
</style>