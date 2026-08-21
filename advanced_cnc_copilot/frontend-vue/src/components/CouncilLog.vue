<template>
  <div class="council-log">
    <div class="header">
      <span class="title">SHADOW COUNCIL LOG</span>
    </div>
    <div class="log-window" ref="logWindow">
      <div v-for="(entry, index) in logs" :key="index" class="log-entry" :class="entry.type">
        <span class="ts">[{{ entry.time }}]</span>
        <span class="agent">{{ entry.agent }}</span>
        <span class="msg">{{ entry.message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue';

const props = defineProps({
  logs: { type: Array, default: () => [] }
});

const logWindow = ref(null);

watch(() => props.logs.length, async () => {
  await nextTick();
  if (logWindow.value) {
    logWindow.value.scrollTop = logWindow.value.scrollHeight;
  }
});
</script>

<style scoped>
.council-log {
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  height: 200px;
  font-family: 'JetBrains Mono', monospace;
}

.header {
  padding: 8px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}

.title {
  font-size: 0.6rem;
  color: #555;
  font-weight: bold;
}

.log-window {
  flex-grow: 1;
  overflow-y: auto;
  padding: 10px;
  font-size: 0.7rem;
}

.log-entry {
  margin-bottom: 5px;
  display: flex;
  gap: 8px;
  opacity: 0.8;
}

.ts { color: #555; }
.agent { font-weight: bold; color: #fff; }
.msg { color: #aaa; }

.log-entry.AUDITOR .agent { color: #ffcc00; }
.log-entry.REGEDIT .agent { color: #ff00ff; }
.log-entry.VISION .agent { color: #00d4ff; }
.log-entry.ERROR .msg { color: #ff0033; }

/* Scrollbar */
.log-window::-webkit-scrollbar { width: 4px; }
.log-window::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
</style>
