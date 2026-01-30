<template>
  <div class="neuro-hud">
    <div class="hud-header">
      <span class="hud-title">NEURO-CHEMISTRY</span>
      <div class="live-indicator">LIVE</div>
    </div>

    <div class="gauges-container">
      <!-- DOPAMINE -->
      <div class="gauge-wrapper">
        <div class="circular-progress" :style="getCircleStyle(dopamine, '#d0f')">
          <div class="inner-circle">
            <span class="value" style="color: #d0f">{{ dopamine.toFixed(0) }}</span>
          </div>
        </div>
        <label>DOPAMINE</label>
      </div>

      <!-- CORTISOL -->
      <div class="gauge-wrapper">
        <div class="circular-progress" :style="getCircleStyle(cortisol, '#f40')">
          <div class="inner-circle">
            <span class="value" style="color: #f40">{{ cortisol.toFixed(0) }}</span>
          </div>
        </div>
        <label>CORTISOL</label>
        <span v-if="cortisol > 80" class="warning-blink">KRITICKÁ ÚROVEŇ</span>
      </div>

      <!-- SEROTONIN -->
      <div class="gauge-wrapper">
        <div class="circular-progress" :style="getCircleStyle(serotonin, '#0df')">
          <div class="inner-circle">
            <span class="value" style="color: #0df">{{ serotonin.toFixed(0) }}</span>
          </div>
        </div>
        <label>SEROTONIN</label>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  dopamine: { type: Number, default: 50 },
  cortisol: { type: Number, default: 20 },
  serotonin: { type: Number, default: 80 }
});

const getCircleStyle = (val, color) => {
  const deg = (val / 100) * 360;
  return {
    background: `conic-gradient(${color} ${deg}deg, rgba(255,255,255,0.05) 0deg)`
  };
};
</script>

<style scoped>
.neuro-hud {
  background: rgba(10, 10, 16, 0.85);
  border: 1px solid rgba(0, 255, 136, 0.2);
  border-left: 4px solid rgba(0, 255, 136, 0.6);
  padding: 20px;
  border-radius: 4px;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
  font-family: 'JetBrains Mono', monospace;
}

.hud-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  padding-bottom: 10px;
}

.hud-title {
  color: #fff;
  font-size: 0.8rem;
  letter-spacing: 2px;
  font-weight: 800;
}

.live-indicator {
  font-size: 0.6rem;
  color: #00ff88;
  animation: blink 2s infinite;
}

.gauges-container {
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.gauge-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
}

.circular-progress {
  width: 70px;
  height: 70px;
  border-radius: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  transition: background 0.5s ease;
  box-shadow: 0 0 10px rgba(0,0,0,0.5);
}

.circular-progress::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    padding: 2px;
    background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
}

.inner-circle {
  width: 55px;
  height: 55px;
  background: #0a0a10;
  border-radius: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
  box-shadow: inset 0 0 10px rgba(0,0,0,0.8);
}

.value {
  font-size: 1.1rem;
  font-weight: bold;
}

label {
  margin-top: 10px;
  font-size: 0.6rem;
  color: #888;
  letter-spacing: 1px;
}

.warning-blink {
  position: absolute;
  bottom: -15px;
  font-size: 0.5rem;
  color: #f40;
  animation: blink 0.5s infinite;
  white-space: nowrap;
}

@keyframes blink { 50% { opacity: 0; } }
</style>
