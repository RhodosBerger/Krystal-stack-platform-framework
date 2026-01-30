# Frontend Architecture Guide
## FANUC RISE v2.1 Glass Brain Interface

### Date: January 26, 2026

---

## Executive Summary

This document outlines the frontend architecture for the FANUC RISE v2.1 system, focusing on the "Glass Brain" concept that visualizes the cognitive state of the manufacturing system. The interface implements biological metaphors and "Neuro-Safety" principles to create an "alive UI" that breathes with the machine's operational state.

---

## 1. Project Context & Philosophy

### The "Glass Brain" Concept
The frontend is not a standard dashboard but a "Glass Brain" that visualizes the cognitive state of a machine using biological metaphors. The interface must "breathe" - borders and indicators pulse based on the machine's stress level (Cortisol) and reward level (Dopamine).

### Tech Stack
- **Framework**: React 18+ (Vite)
- **Language**: TypeScript
- **Styling**: TailwindCSS
- **Animation**: Framer Motion (critical for the "Neuro-Safety" pulsing effects)
- **Icons**: Lucide-React or Heroicons
- **Backend**: Python/FastAPI streaming data at 1kHz via WebSockets

### Design Philosophy (The "Neuro-Safety" Paradigm)
1. **Alive UI**: The interface must "breathe." Borders and indicators pulse based on the machine's stress level (Cortisol) and reward level (Dopamine).
2. **Cognitive Load Shedding**: Operators should see "Safety Orange" alerts; Managers see "Emerald" economic data. Financial data is hidden from the Operator view to prevent cognitive clutter.
3. **Synesthesia**: Visuals should represent physical forces (e.g., vibration = visual entropy/blur).

---

## 2. Design System & Tailwind Configuration

### Color Palette
- **Safety Orange**: `#ff5722` for High Stress/Cortisol
- **Emerald Green**: `#10b981` for High OEE/Dopamine
- **Cyber Blue**: `#3b82f6` for the Creator/AI mode
- **Dark Mode Backgrounds**: Slate/Zinc 900 for the industrial environment

### Typography
- **Headings**: Inter (Weights 600/700)
- **Data/Telemetry**: JetBrains Mono (Monospace for RPM/Load)

### Animation Tokens
- **Heartbeat Pulse**: Animation curve that speeds up as `stress_level` (0.0 to 1.0) increases

### Tailwind Configuration

```javascript
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'sans': ['Inter', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace'],
      },
      colors: {
        'safety-orange': '#ff5722',
        'emerald-green': '#10b981',
        'cyber-blue': '#3b82f6',
        'neuro-coral': '#FF6B6B',
        'neuro-amber': '#FBBF24',
        'neuro-indigo': '#8B5CF6',
      },
      animation: {
        'pulse-stress': 'pulseStress 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-emerald': 'pulseEmerald 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pulse-danger': 'pulseDanger 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'heartbeat': 'heartbeat 1.5s ease-in-out infinite',
      },
      keyframes: {
        pulseStress: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(255, 87, 34, 0.7)' },
          '50%': { boxShadow: '0 0 0 10px rgba(255, 87, 34, 0)' },
        },
        pulseEmerald: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(16, 185, 129, 0.7)' },
          '50%': { boxShadow: '0 0 0 6px rgba(16, 185, 129, 0)' },
        },
        pulseDanger: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(239, 68, 68, 0.7)' },
          '50%': { boxShadow: '0 0 0 8px rgba(239, 68, 68, 0)' },
        },
        heartbeat: {
          '0%': { transform: 'scale(1)' },
          '25%': { transform: 'scale(1.05)' },
          '50%': { transform: 'scale(1)' },
          '75%': { transform: 'scale(1.03)' },
          '100%': { transform: 'scale(1)' },
        }
      }
    },
  },
  plugins: [],
}
```

### Global CSS for Industrial/Neuro Aesthetic

```css
/* src/index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

body {
  font-family: 'Inter', sans-serif;
  background-color: #0f172a; /* slate-900 */
  color: #e2e8f0; /* slate-200 */
  overflow-x: hidden;
}

/* Industrial/Neuro Styling */
.neuro-card {
  @apply bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl shadow-lg transition-all duration-300;
}

.neuro-card:hover {
  @apply border-slate-600/70 shadow-xl shadow-slate-900/30;
}

.neuro-card.stress-high {
  @apply border-safety-orange/80 shadow-lg shadow-safety-orange/20 animate-pulse-danger;
}

.neuro-card.stress-medium {
  @apply border-neuro-amber/60 shadow-lg shadow-neuro-amber/20 animate-pulse-stress;
}

.neuro-card.stress-low {
  @apply border-emerald-green/60 shadow-lg shadow-emerald-green/20 animate-pulse-emerald;
}

.neuro-meter {
  @apply h-2 rounded-full bg-slate-700 overflow-hidden;
}

.neuro-meter-fill {
  @apply h-full rounded-full transition-all duration-500 ease-out;
}

.neuro-meter-fill.dopamine {
  @apply bg-gradient-to-r from-emerald-green to-emerald-400;
}

.neuro-meter-fill.cortisol {
  @apply bg-gradient-to-r from-safety-orange to-orange-500;
}

.neuro-meter-fill.serotonin {
  @apply bg-gradient-to-r from-cyan-400 to-blue-500;
}

.monospace-data {
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.05em;
}

.terminal-style {
  @apply font-mono text-green-400 bg-black/80 p-4 rounded-lg border border-green-800/50;
}

.glass-panel {
  @apply bg-white/5 backdrop-blur-md border border-white/10 rounded-xl;
}
```

---

## 3. Component Architecture

### 3.1 The NeuroCard Component (Sensory Cortex)

```tsx
// components/NeuroCard.tsx
import React from 'react';
import { motion } from 'framer-motion';

interface NeuroCardProps {
  title: string;
  children: React.ReactNode;
  stressLevel: number; // Float 0-1
  dopamineLevel?: number;
  cortisolLevel?: number;
}

const NeuroCard: React.FC<NeuroCardProps> = ({ 
  title, 
  children, 
  stressLevel, 
  dopamineLevel = 0.5,
  cortisolLevel = 0.1
}) => {
  // Determine animation based on stress level
  const getAnimationState = () => {
    if (stressLevel > 0.8) {
      return {
        borderColor: 'border-safety-orange',
        backgroundColor: 'bg-slate-800/70',
        animation: 'animate-pulse-danger',
        pulseFrequency: 3, // 3Hz for high stress
      };
    } else if (stressLevel > 0.5) {
      return {
        borderColor: 'border-neuro-amber',
        backgroundColor: 'bg-slate-800/60',
        animation: 'animate-pulse-stress',
        pulseFrequency: 1.5, // 1.5Hz for medium stress
      };
    } else {
      return {
        borderColor: 'border-emerald-green',
        backgroundColor: 'bg-slate-800/50',
        animation: 'animate-pulse-emerald',
        pulseFrequency: 0.5, // 0.5Hz for low stress
      };
    }
  };

  const animationState = getAnimationState();

  return (
    <motion.div
      className={`neuro-card p-6 ${animationState.borderColor} ${animationState.backgroundColor} ${animationState.animation}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ 
        opacity: 1, 
        y: 0,
        scale: stressLevel > 0.8 ? [1, 1.02, 1] : [1, 1, 1]
      }}
      transition={{
        duration: 0.5,
        scale: {
          repeat: stressLevel > 0.8 ? Infinity : 0,
          repeatType: "loop",
          duration: 1 / animationState.pulseFrequency,
        }
      }}
    >
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-slate-200">{title}</h3>
        <div className="flex space-x-2">
          {dopamineLevel !== undefined && (
            <span className={`px-2 py-1 rounded text-xs ${
              dopamineLevel > 0.7 ? 'bg-emerald-500/20 text-emerald-300' : 
              dopamineLevel > 0.4 ? 'bg-emerald-500/10 text-emerald-400' : 
              'bg-slate-700/50 text-slate-400'
            }`}>
              D: {dopamineLevel.toFixed(2)}
            </span>
          )}
          {cortisolLevel !== undefined && (
            <span className={`px-2 py-1 rounded text-xs ${
              cortisolLevel > 0.7 ? 'bg-safety-orange/20 text-safety-orange' : 
              cortisolLevel > 0.4 ? 'bg-safety-orange/10 text-safety-orange' : 
              'bg-slate-700/50 text-slate-400'
            }`}>
              C: {cortisolLevel.toFixed(2)}
            </span>
          )}
        </div>
      </div>
      <div className="mt-4">
        {children}
      </div>
    </motion.div>
  );
};

export default NeuroCard;
```

### 3.2 The Dopamine Engine Widget

```tsx
// components/DopamineEngineWidget.tsx
import React from 'react';
import { motion } from 'framer-motion';

interface DopamineEngineWidgetProps {
  dopamineScore: number; // 0-1
  cortisolScore: number; // 0-1
  serotoninScore: number; // 0-1
}

const DopamineEngineWidget: React.FC<DopamineEngineWidgetProps> = ({
  dopamineScore,
  cortisolScore,
  serotoninScore
}) => {
  // Calculate reward ratio: R = (Speed * Quality) / Stress
  const rewardRatio = (dopamineScore * serotoninScore) / (cortisolScore + 0.1); // +0.1 to prevent division by zero

  return (
    <div className="neuro-card p-6">
      <h3 className="text-lg font-semibold text-slate-200 mb-4">Dopamine Engine</h3>
      
      <div className="space-y-4">
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-sm text-slate-300">Dopamine (Reward)</span>
            <span className="text-sm font-mono">{dopamineScore.toFixed(2)}</span>
          </div>
          <div className="neuro-meter">
            <motion.div 
              className="neuro-meter-fill dopamine"
              initial={{ width: 0 }}
              animate={{ width: `${dopamineScore * 100}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between mb-1">
            <span className="text-sm text-slate-300">Cortisol (Stress)</span>
            <span className="text-sm font-mono">{cortisolScore.toFixed(2)}</span>
          </div>
          <div className="neuro-meter">
            <motion.div 
              className="neuro-meter-fill cortisol"
              initial={{ width: 0 }}
              animate={{ width: `${cortisolScore * 100}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between mb-1">
            <span className="text-sm text-slate-300">Serotonin (Stability)</span>
            <span className="text-sm font-mono">{serotoninScore.toFixed(2)}</span>
          </div>
          <div className="neuro-meter">
            <motion.div 
              className="neuro-meter-fill serotonin"
              initial={{ width: 0 }}
              animate={{ width: `${serotoninScore * 100}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
            />
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-slate-700/50">
        <div className="flex justify-between items-center">
          <span className="text-sm text-slate-300">Reward Ratio:</span>
          <span className={`text-lg font-bold font-mono ${
            rewardRatio > 1.0 ? 'text-emerald-400' :
            rewardRatio > 0.5 ? 'text-yellow-400' : 
            'text-safety-orange'
          }`}>
            {rewardRatio.toFixed(2)}
          </span>
        </div>
        <p className="text-xs mt-2 text-slate-400">
          {cortisolScore > 0.7 ? '⚠️ Defense Mode Active' : '✅ Normal Operation'}
        </p>
      </div>
    </div>
  );
};

export default DopamineEngineWidget;
```

### 3.3 The "Invisible Church" (Reasoning Trace)

```tsx
// components/ReasoningTrace.tsx
import React, { useEffect, useRef } from 'react';

interface ReasoningTraceItem {
  id: string;
  timestamp: string;
  source: 'creator' | 'auditor' | 'accountant' | 'operator';
  message: string;
  status: 'info' | 'warning' | 'error' | 'success';
}

interface ReasoningTraceProps {
  traces: ReasoningTraceItem[];
}

const ReasoningTrace: React.FC<ReasoningTraceProps> = ({ traces }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [traces]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'text-emerald-400';
      case 'warning': return 'text-neuro-amber';
      case 'error': return 'text-safety-orange';
      default: return 'text-slate-300';
    }
  };

  const getSourceIcon = (source: string) => {
    switch (source) {
      case 'creator': return '🤖';
      case 'auditor': return '🛡️';
      case 'accountant': return '📊';
      case 'operator': return '👷';
      default: return '❓';
    }
  };

  return (
    <div className="neuro-card p-4 h-64 overflow-y-auto" ref={scrollRef}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-slate-200">Invisible Church (Reasoning Trace)</h3>
        <span className="text-xs px-2 py-1 bg-slate-700/50 rounded">Live</span>
      </div>
      
      <div className="space-y-2 font-mono text-sm">
        {traces.map((trace) => (
          <div 
            key={trace.id} 
            className={`p-2 rounded border-l-4 ${
              trace.status === 'error' ? 'border-safety-orange bg-red-900/10' :
              trace.status === 'warning' ? 'border-neuro-amber bg-amber-900/10' :
              trace.status === 'success' ? 'border-emerald-green bg-emerald-900/10' :
              'border-slate-500 bg-slate-800/30'
            }`}
          >
            <div className="flex items-start">
              <span className="mr-2">{getSourceIcon(trace.source)}</span>
              <div className="flex-1">
                <div className="flex justify-between">
                  <span className={getStatusColor(trace.status)}>
                    {trace.source.toUpperCase()}
                  </span>
                  <span className="text-slate-500 text-xs">{trace.timestamp}</span>
                </div>
                <div className="mt-1 text-slate-300">{trace.message}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ReasoningTrace;
```

### 3.4 The Quadratic Mantinel Visualizer

```tsx
// components/QuadraticMantinelVisualizer.tsx
import React from 'react';

interface QuadraticMantinelVisualizerProps {
  currentSpeed: number;
  currentCurvature: number;
  materialHardness: number;
  isSafe: boolean;
}

const QuadraticMantinelVisualizer: React.FC<QuadraticMantinelVisualizerProps> = ({
  currentSpeed,
  currentCurvature,
  materialHardness,
  isSafe
}) => {
  // Generate parabolic curve (the mantinel)
  const mantinelPoints = [];
  for (let x = 0; x <= 100; x += 5) {
    // Quadratic mantinel: Speed = sqrt(Limit / Curvature)
    const y = Math.sqrt(10000 / (x / 10 + 0.1)); // Add 0.1 to prevent division by zero
    mantinelPoints.push({ x, y });
  }

  // Normalize values for display
  const maxDisplaySpeed = 15000; // RPM
  const maxDisplayCurvature = 100; // arbitrary units
  
  const normalizedSpeed = Math.min(100, (currentSpeed / maxDisplaySpeed) * 100);
  const normalizedCurvature = Math.min(100, (currentCurvature / maxDisplayCurvature) * 100);

  // Calculate if current point is safe (below the curve)
  const theoreticalMaxSpeed = Math.sqrt(10000 / (normalizedCurvature / 10 + 0.1));
  const isCurrentPointSafe = currentSpeed <= theoreticalMaxSpeed;

  return (
    <div className="neuro-card p-6">
      <h3 className="text-lg font-semibold text-slate-200 mb-4">Quadratic Mantinel</h3>
      
      <div className="relative h-64 w-full bg-slate-900/50 rounded-lg border border-slate-700 overflow-hidden">
        {/* Grid lines */}
        <div className="absolute inset-0 grid grid-cols-10 grid-rows-10 opacity-20">
          {[...Array(11)].map((_, i) => (
            <React.Fragment key={i}>
              <div className="border-r border-slate-600" style={{ gridColumn: i + 1 }}></div>
              <div className="border-b border-slate-600" style={{ gridRow: i + 1 }}></div>
            </React.Fragment>
          ))}
        </div>
        
        {/* The Mantinel Curve */}
        <svg className="absolute inset-0 w-full h-full">
          <path
            d={`M 0,${500 - (Math.sqrt(10000 / 0.1) * 5)} ${mantinelPoints.map(p => 
              `L ${(p.x / 100) * 500},${500 - (p.y * 5)}`
            ).join(' ')}`}
            stroke={isCurrentPointSafe ? "#10b981" : "#ff5722"}
            strokeWidth="2"
            fill="none"
            opacity="0.7"
          />
          
          {/* Current operation point */}
          <circle
            cx={(normalizedCurvature / 100) * 500}
            cy={500 - (normalizedSpeed / 100) * 500}
            r="6"
            fill={isCurrentPointSafe ? "#10b981" : "#ff5722"}
            stroke="#ffffff"
            strokeWidth="2"
          />
        </svg>
        
        <div className="absolute bottom-2 left-2 text-xs text-slate-400">
          Speed vs. Curvature² Constraint
        </div>
      </div>
      
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div>
          <div className="text-sm text-slate-400">Current Speed</div>
          <div className="font-mono text-lg">{currentSpeed.toLocaleString()} RPM</div>
        </div>
        <div>
          <div className="text-sm text-slate-400">Current Curvature</div>
          <div className="font-mono text-lg">{currentCurvature.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-sm text-slate-400">Material Hardness</div>
          <div className="font-mono text-lg">{materialHardness} HRc</div>
        </div>
        <div>
          <div className="text-sm text-slate-400">Safety Status</div>
          <div className={`font-mono text-lg ${
            isCurrentPointSafe ? 'text-emerald-400' : 'text-safety-orange'
          }`}>
            {isCurrentPointSafe ? 'SAFE' : 'UNSAFE'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuadraticMantinelVisualizer;
```

---

## 4. Operator Dashboard (The "Glass Brain")

```tsx
// OperatorDashboard.tsx
import React, { useState, useEffect } from 'react';
import NeuroCard from './components/NeuroCard';
import DopamineEngineWidget from './components/DopamineEngineWidget';
import ReasoningTrace from './components/ReasoningTrace';
import QuadraticMantinelVisualizer from './components/QuadraticMantinelVisualizer';

const OperatorDashboard = () => {
  const [telemetry, setTelemetry] = useState({
    spindleRPM: 0,
    spindleLoad: 0,
    vibrationX: 0,
    vibrationY: 0,
    vibrationZ: 0,
    temperature: 0,
    feedRate: 0,
    toolWear: 0,
    dopamineScore: 0.5,
    cortisolLevel: 0.1,
    serotoninLevel: 0.7,
    stressLevel: 0.2,
    machineStatus: 'RUNNING',
    activeProgram: 'N/A',
    remainingTime: 'N/A'
  });

  const [reasoningTraces, setReasoningTraces] = useState<any[]>([]);

  // Mock WebSocket data simulation
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate realistic telemetry data
      const newRPM = 8000 + Math.random() * 2000;
      const newLoad = 65 + Math.random() * 15;
      const newVibration = 0.1 + Math.random() * 0.3;
      const newTemp = 35 + Math.random() * 5;
      
      // Calculate stress level based on vibration and temperature
      const stress = Math.min(1.0, (newVibration * 2) + ((newTemp - 30) * 0.05));
      
      // Calculate dopamine based on efficiency
      const dopamine = Math.max(0.1, 0.8 - (stress * 0.5));
      
      // Calculate cortisol based on stress with persistence
      const cortisol = Math.min(1.0, telemetry.cortisolLevel + (stress * 0.1));
      
      setTelemetry(prev => ({
        ...prev,
        spindleRPM: Math.round(newRPM),
        spindleLoad: parseFloat(newLoad.toFixed(1)),
        vibrationX: parseFloat(newVibration.toFixed(3)),
        vibrationY: parseFloat(newVibration.toFixed(3)),
        vibrationZ: parseFloat(newVibration.toFixed(3)),
        temperature: parseFloat(newTemp.toFixed(1)),
        dopamineScore: parseFloat(dopamine.toFixed(2)),
        cortisolLevel: parseFloat(cortisol.toFixed(2)),
        stressLevel: parseFloat(stress.toFixed(2))
      }));

      // Occasionally add reasoning traces
      if (Math.random() > 0.7) {
        const newTrace = {
          id: Date.now().toString(),
          timestamp: new Date().toLocaleTimeString(),
          source: ['creator', 'auditor'][Math.floor(Math.random() * 2)],
          message: [
            'Creator proposed RPM 12,000. Auditor rejected: Violates Inconel thermal limit. Reverting to 9,000.',
            'Optimizer suggested feedrate increase. Physics check passed. Applying new parameters.',
            'High vibration detected. Entering defense mode to protect tool.',
            'Quality sensor detected anomaly. Initiating inspection protocol.',
            'Thermal model predicts safe operation. Increasing aggression level.'
          ][Math.floor(Math.random() * 5)],
          status: ['info', 'warning', 'error', 'success'][Math.floor(Math.random() * 4)]
        };
        
        setReasoningTraces(prev => [...prev.slice(-9), newTrace]);
      }
    }, 100); // Update every 100ms to simulate 1kHz data reduced for UI

    return () => clearInterval(interval);
  }, [telemetry.cortisolLevel]);

  return (
    <div className="min-h-screen bg-slate-900 p-6">
      <header className="mb-8">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-green to-cyber-blue bg-clip-text text-transparent">
            FANUC RISE v2.1 - Operator Dashboard
          </h1>
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-semibold ${
              telemetry.machineStatus === 'RUNNING' ? 'bg-emerald-500/20 text-emerald-300' :
              telemetry.machineStatus === 'STOPPED' ? 'bg-slate-600/20 text-slate-300' :
              'bg-safety-orange/20 text-safety-orange'
            }`}>
              {telemetry.machineStatus}
            </div>
            <div className="text-sm text-slate-400">
              {telemetry.activeProgram} | ETA: {telemetry.remainingTime}
            </div>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          <NeuroCard 
            title="Sensory Cortex" 
            stressLevel={telemetry.stressLevel}
            dopamineLevel={telemetry.dopamineScore}
            cortisolLevel={telemetry.cortisolLevel}
          >
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-sm text-slate-400">Spindle RPM</div>
                <div className="font-mono text-2xl">{telemetry.spindleRPM.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-sm text-slate-400">Spindle Load</div>
                <div className="font-mono text-2xl">{telemetry.spindleLoad}%</div>
              </div>
              <div>
                <div className="text-sm text-slate-400">Vibration</div>
                <div className="font-mono text-xl">
                  X:{telemetry.vibrationX} Y:{telemetry.vibrationY} Z:{telemetry.vibrationZ}
                </div>
              </div>
              <div>
                <div className="text-sm text-slate-400">Temperature</div>
                <div className="font-mono text-xl">{telemetry.temperature}°C</div>
              </div>
            </div>
          </NeuroCard>

          <DopamineEngineWidget
            dopamineScore={telemetry.dopamineScore}
            cortisolScore={telemetry.cortisolLevel}
            serotoninScore={telemetry.serotoninLevel}
          />
        </div>

        {/* Center Column */}
        <div className="space-y-6">
          <NeuroCard 
            title="Quadratic Mantinel" 
            stressLevel={telemetry.stressLevel}
          >
            <QuadraticMantinelVisualizer
              currentSpeed={telemetry.spindleRPM}
              currentCurvature={telemetry.vibrationX + telemetry.vibrationY + telemetry.vibrationZ}
              materialHardness={35}
              isSafe={telemetry.stressLevel < 0.7}
            />
          </NeuroCard>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          <NeuroCard 
            title="Reasoning Trace" 
            stressLevel={telemetry.stressLevel}
          >
            <ReasoningTrace traces={reasoningTraces} />
          </NeuroCard>

          <NeuroCard 
            title="Tool Status" 
            stressLevel={telemetry.toolWear * 0.1}
          >
            <div className="space-y-3">
              <div>
                <div className="text-sm text-slate-400">Tool Wear</div>
                <div className="font-mono text-xl">{(telemetry.toolWear * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div className="text-sm text-slate-400">Feed Rate</div>
                <div className="font-mono text-xl">{telemetry.feedRate} mm/min</div>
              </div>
              <div className="pt-2">
                <button className={`w-full py-2 rounded-lg font-semibold ${
                  telemetry.stressLevel < 0.5 ? 
                    'bg-emerald-600 hover:bg-emerald-700 text-white' : 
                    'bg-slate-700 text-slate-300 cursor-not-allowed'
                }`}>
                  {telemetry.stressLevel < 0.5 ? 'Enable Rush Mode' : 'Defense Mode Active'}
                </button>
              </div>
            </div>
          </NeuroCard>
        </div>
      </div>
    </div>
  );
};

export default OperatorDashboard;
```

---

## 5. Manager Dashboard (Swarm & Economics)

```tsx
// ManagerDashboard.tsx
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface MachineData {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'maintenance' | 'alarm';
  oee: number;
  load: number;
  temperature: number;
  vibration: number;
  toolWear: number;
  productionValue: number;
  downtimeHours: number;
  efficiencyRating: number;
}

const ManagerDashboard = () => {
  const [machines, setMachines] = useState<MachineData[]>([]);
  const [economicMetrics, setEconomicMetrics] = useState({
    churnRate: 0,
    cac: 0,
    profitRate: 0,
    totalRevenue: 0,
    costPerPart: 0
  });

  useEffect(() => {
    // Simulate machine data
    const mockMachines: MachineData[] = [
      { id: 'M001', name: 'Fanuc ROBOSHOT S-2000', status: 'running', oee: 0.85, load: 78, temperature: 38.2, vibration: 0.12, toolWear: 0.15, productionValue: 1250, downtimeHours: 0.5, efficiencyRating: 0.92 },
      { id: 'M002', name: 'Fanuc ROBOSHOT S-1500', status: 'idle', oee: 0.72, load: 15, temperature: 25.0, vibration: 0.05, toolWear: 0.08, productionValue: 800, downtimeHours: 0, efficiencyRating: 0.75 },
      { id: 'M003', name: 'Fanuc ROBOSHOT S-3000', status: 'maintenance', oee: 0.0, load: 0, temperature: 22.1, vibration: 0.03, toolWear: 0.95, productionValue: 0, downtimeHours: 2.3, efficiencyRating: 0.1 },
      { id: 'M004', name: 'Fanuc ROBOSHOT S-2500', status: 'running', oee: 0.91, load: 85, temperature: 41.5, vibration: 0.18, toolWear: 0.22, productionValue: 1400, downtimeHours: 0.2, efficiencyRating: 0.95 },
      { id: 'M005', name: 'Fanuc ROBOSHOT S-1800', status: 'alarm', oee: 0.0, load: 0, temperature: 48.7, vibration: 0.85, toolWear: 0.65, productionValue: 0, downtimeHours: 0.8, efficiencyRating: 0.05 },
    ];
    
    setMachines(mockMachines);
    
    // Calculate economic metrics
    const churnRate = mockMachines.reduce((acc, m) => acc + m.toolWear, 0) / mockMachines.length * 100;
    const totalRevenue = mockMachines.reduce((acc, m) => acc + m.productionValue, 0);
    const totalDowntime = mockMachines.reduce((acc, m) => acc + m.downtimeHours, 0);
    const profitRate = totalRevenue > 0 ? (totalRevenue - (totalDowntime * 50)) / 8 : 0; // Simplified calculation
    
    setEconomicMetrics({
      churnRate: parseFloat(churnRate.toFixed(2)),
      cac: 45, // Fixed value for this example
      profitRate: parseFloat(profitRate.toFixed(2)),
      totalRevenue,
      costPerPart: 12.50
    });
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'bg-emerald-500';
      case 'idle': return 'bg-slate-500';
      case 'maintenance': return 'bg-cyber-blue';
      case 'alarm': return 'bg-safety-orange';
      default: return 'bg-slate-700';
    }
  };

  const getGravityWellSize = (oee: number) => {
    // Higher OEE = larger "gravity well" effect
    const baseSize = 80;
    return baseSize + (oee * 40);
  };

  return (
    <div className="min-h-screen bg-slate-900 p-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-green to-cyber-blue bg-clip-text text-transparent">
          FANUC RISE v2.1 - Manager Dashboard
        </h1>
        <p className="text-slate-400">Fleet Command & Swarm Intelligence</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Economic Metrics Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="neuro-card p-6">
            <h3 className="text-lg font-semibold text-slate-200 mb-4">Economic Engine</h3>
            
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-400">Churn Rate (Tool Wear)</span>
                  <span className="text-sm font-mono">{economicMetrics.churnRate}%</span>
                </div>
                <div className="neuro-meter">
                  <div 
                    className={`neuro-meter-fill ${economicMetrics.churnRate > 50 ? 'bg-safety-orange' : 'bg-emerald-green'}`} 
                    style={{ width: `${Math.min(100, economicMetrics.churnRate)}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-400">CAC (Setup Time)</span>
                  <span className="text-sm font-mono">{economicMetrics.cac} min</span>
                </div>
                <div className="neuro-meter">
                  <div 
                    className="neuro-meter-fill serotonin" 
                    style={{ width: `${Math.min(100, economicMetrics.cac / 2)}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-400">Profit Rate (Pr)</span>
                  <span className="text-sm font-mono">${economicMetrics.profitRate}/hr</span>
                </div>
                <div className="neuro-meter">
                  <div 
                    className="neuro-meter-fill dopamine" 
                    style={{ width: `${Math.min(100, economicMetrics.profitRate * 10)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          <div className="neuro-card p-6">
            <h3 className="text-lg font-semibold text-slate-200 mb-4">Anti-Fragile Marketplace</h3>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg">
                <span className="text-slate-300">Turbo Milling Script</span>
                <span className="text-xs bg-emerald-500/20 text-emerald-300 px-2 py-1 rounded">Survivor Badge</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg">
                <span className="text-slate-300">Precision Boring Routine</span>
                <span className="text-xs bg-emerald-500/20 text-emerald-300 px-2 py-1 rounded">Survivor Badge</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg">
                <span className="text-slate-300">Surface Finish Optimizer</span>
                <span className="text-xs bg-slate-700/50 text-slate-400 px-2 py-1 rounded">Standard</span>
              </div>
            </div>
          </div>
        </div>

        {/* Swarm Map */}
        <div className="lg:col-span-3">
          <div className="neuro-card p-6">
            <h3 className="text-lg font-semibold text-slate-200 mb-4">Swarm Map (Fleet Command)</h3>
            
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              {machines.map((machine) => (
                <motion.div
                  key={machine.id}
                  className={`rounded-lg p-4 border ${
                    machine.status === 'alarm' ? 'border-safety-orange bg-red-900/10' :
                    machine.oee > 0.85 ? 'border-emerald-500/50 bg-emerald-900/10' :
                    machine.oee > 0.7 ? 'border-cyber-blue/50 bg-blue-900/10' :
                    'border-slate-600 bg-slate-800/30'
                  }`}
                  style={{ 
                    width: `${getGravityWellSize(machine.oee)}px`,
                    height: `${getGravityWellSize(machine.oee)}px`
                  }}
                  whileHover={{ scale: 1.05 }}
                  animate={{
                    boxShadow: machine.status === 'alarm' 
                      ? '0 0 20px rgba(255, 87, 34, 0.7)' 
                      : machine.oee > 0.85 
                        ? '0 0 10px rgba(16, 185, 129, 0.4)' 
                        : '0 0 5px rgba(59, 130, 246, 0.3)'
                  }}
                >
                  <div className="flex items-center mb-2">
                    <div className={`w-3 h-3 rounded-full mr-2 ${getStatusColor(machine.status)}`}></div>
                    <span className="text-sm font-semibold">{machine.name}</span>
                  </div>
                  
                  <div className="text-xs space-y-1">
                    <div>ID: {machine.id}</div>
                    <div>OEE: {(machine.oee * 100).toFixed(1)}%</div>
                    <div>Load: {machine.load}%</div>
                    <div className={`${machine.toolWear > 0.8 ? 'text-safety-orange' : 'text-slate-400'}`}>
                      Tool: {(machine.toolWear * 100).toFixed(1)}%
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
            
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="neuro-card p-4">
                <div className="text-sm text-slate-400">Total Revenue</div>
                <div className="font-mono text-xl text-emerald-400">${economicMetrics.totalRevenue}</div>
              </div>
              <div className="neuro-card p-4">
                <div className="text-sm text-slate-400">Avg. Cost/Part</div>
                <div className="font-mono text-xl">${economicMetrics.costPerPart}</div>
              </div>
              <div className="neuro-card p-4">
                <div className="text-sm text-slate-400">Fleet OEE</div>
                <div className="font-mono text-xl">
                  {((machines.reduce((acc, m) => acc + m.oee, 0) / machines.filter(m => m.status !== 'maintenance').length) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ManagerDashboard;
```

---

## 6. Creator Dashboard (Generative Interface)

```tsx
// CreatorStudio.tsx
import React, { useReducer } from 'react';
import { motion } from 'framer-motion';

interface CouncilVote {
  id: string;
  agent: 'creator' | 'auditor' | 'accountant';
  message: string;
  status: 'pending' | 'approved' | 'rejected';
  timestamp: string;
}

interface CreatorState {
  aggressionLevel: number;
  creativityLevel: number;
  proposal: string;
  votes: CouncilVote[];
  canExecute: boolean;
  lastApprovedProposal: string | null;
}

type CreatorAction = 
  | { type: 'SET_AGGRESSION'; payload: number }
  | { type: 'SET_CREATIVITY'; payload: number }
  | { type: 'SUBMIT_PROPOSAL'; payload: string }
  | { type: 'ADD_VOTE'; payload: CouncilVote }
  | { type: 'RESET_VOTES' };

const creatorReducer = (state: CreatorState, action: CreatorAction): CreatorState => {
  switch (action.type) {
    case 'SET_AGGRESSION':
      return { ...state, aggressionLevel: action.payload };
    case 'SET_CREATIVITY':
      return { ...state, creativityLevel: action.payload };
    case 'SUBMIT_PROPOSAL':
      return { 
        ...state, 
        proposal: action.payload,
        votes: [],
        canExecute: false
      };
    case 'ADD_VOTE':
      const newVotes = [...state.votes, action.payload];
      const allApproved = newVotes.every(vote => vote.status === 'approved');
      return { 
        ...state, 
        votes: newVotes,
        canExecute: allApproved && newVotes.length > 0
      };
    case 'RESET_VOTES':
      return { 
        ...state, 
        votes: [],
        canExecute: false
      };
    default:
      return state;
  }
};

const CreatorStudio = () => {
  const [state, dispatch] = useReducer(creatorReducer, {
    aggressionLevel: 0.5,
    creativityLevel: 0.3,
    proposal: '',
    votes: [],
    canExecute: false,
    lastApprovedProposal: null
  });

  const [generatedCode, setGeneratedCode] = React.useState('');
  const [isLoading, setIsLoading] = React.useState(false);

  const handleSubmit = () => {
    if (!state.proposal.trim()) return;
    
    setIsLoading(true);
    
    // Simulate AI processing
    setTimeout(() => {
      // Generate mock code based on intent
      const mockCode = `// Generated from intent: "${state.proposal}"
// Aggression: ${state.aggressionLevel.toFixed(2)}, Creativity: ${state.creativityLevel.toFixed(2)}

def optimize_machining_path(toolpath):
    # Apply thermal-biased mutations based on aggression level
    if ${state.aggressionLevel} > 0.7:
        boost_feed_rate(toolpath, 1.2)
    else:
        conservative_feed_rate(toolpath)
    
    # Apply creative path optimization based on creativity level
    if ${state.creativityLevel} > 0.5:
        apply_trochoidal_milling(toolpath)
    
    return toolpath`;

      setGeneratedCode(mockCode);
      
      // Simulate Shadow Council voting
      const creatorVote: CouncilVote = {
        id: Date.now().toString(),
        agent: 'creator',
        message: `Generated optimization for: ${state.proposal}`,
        status: 'approved',
        timestamp: new Date().toLocaleTimeString()
      };
      
      const auditorVote: CouncilVote = {
        id: (Date.now() + 1).toString(),
        agent: 'auditor',
        message: `Physics check passed: Thermal limits safe, no collision detected`,
        status: Math.random() > 0.2 ? 'approved' : 'rejected', // 80% approval rate
        timestamp: new Date().toLocaleTimeString()
      };
      
      const accountantVote: CouncilVote = {
        id: (Date.now() + 2).toString(),
        agent: 'accountant',
        message: `Economic check: Expected 15% time saving, 8% cost reduction`,
        status: 'approved',
        timestamp: new Date().toLocaleTimeString()
      };
      
      dispatch({ type: 'ADD_VOTE', payload: creatorVote });
      dispatch({ type: 'ADD_VOTE', payload: auditorVote });
      dispatch({ type: 'ADD_VOTE', payload: accountantVote });
      
      setIsLoading(false);
    }, 2000);
  };

  const handleExecute = () => {
    if (state.canExecute) {
      alert('Executing optimized program on CNC...');
      dispatch({ type: 'RESET_VOTES' });
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 p-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-cyber-blue to-purple-500 bg-clip-text text-transparent">
          FANUC RISE v2.1 - Creative Studio
        </h1>
        <p className="text-slate-400">Generative Design & AI-Powered Optimization</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column - Input & Controls */}
        <div className="space-y-6">
          <div className="neuro-card p-6">
            <h3 className="text-lg font-semibold text-slate-200 mb-4">Emotional Nexus (Intent Input)</h3>
            
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Machining Intent
                </label>
                <textarea
                  value={state.proposal}
                  onChange={(e) => dispatch({ type: 'SUBMIT_PROPOSAL', payload: e.target.value })}
                  placeholder="Describe your machining intent (e.g., 'Make this part faster without compromising quality')"
                  className="w-full p-3 bg-slate-800/50 border border-slate-700 rounded-lg text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyber-blue"
                  rows={4}
                />
              </div>
              
              <div>
                <div className="flex justify-between mb-2">
                  <label className="text-sm font-medium text-slate-300">Aggression Level</label>
                  <span className="text-sm font-mono">{(state.aggressionLevel * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={state.aggressionLevel}
                  onChange={(e) => dispatch({ type: 'SET_AGGRESSION', payload: parseFloat(e.target.value) })}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, 
                      ${state.aggressionLevel < 0.3 ? '#3b82f6' : '#8b5cf6'}, 
                      ${state.aggressionLevel > 0.7 ? '#ff5722' : '#fbbf24'} 
                      ${state.aggressionLevel * 100}%, 
                      #334155 0%)`
                  }}
                />
                <div className="flex justify-between text-xs text-slate-400 mt-1">
                  <span>Economy</span>
                  <span>Rush Mode</span>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between mb-2">
                  <label className="text-sm font-medium text-slate-300">Creativity Level</label>
                  <span className="text-sm font-mono">{(state.creativityLevel * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={state.creativityLevel}
                  onChange={(e) => dispatch({ type: 'SET_CREATIVITY', payload: parseFloat(e.target.value) })}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, 
                      #3b82f6, 
                      #8b5cf6 ${state.creativityLevel * 100}%, 
                      #334155 0%)`
                  }}
                />
                <div className="flex justify-between text-xs text-slate-400 mt-1">
                  <span>Standard</span>
                  <span>Generative</span>
                </div>
              </div>
              
              <button
                onClick={handleSubmit}
                disabled={!state.proposal.trim() || isLoading}
                className={`w-full py-3 rounded-lg font-semibold ${
                  !state.proposal.trim() || isLoading
                    ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                    : 'bg-cyber-blue hover:bg-blue-600 text-white'
                }`}
              >
                {isLoading ? 'Generating...' : 'Generate with AI'}
              </button>
            </div>
          </div>
          
          <div className="neuro-card p-6">
            <h3 className="text-lg font-semibold text-slate-200 mb-4">Shadow Council Voting</h3>
            
            <div className="space-y-3">
              {state.votes.map((vote) => (
                <div 
                  key={vote.id}
                  className={`p-3 rounded-lg border-l-4 ${
                    vote.status === 'approved' 
                      ? 'border-emerald-500 bg-emerald-900/10' 
                      : vote.status === 'rejected'
                        ? 'border-safety-orange bg-red-900/10'
                        : 'border-slate-500 bg-slate-800/30'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex items-center">
                      <span className="mr-2">
                        {vote.agent === 'creator' ? '🤖' : 
                         vote.agent === 'auditor' ? '🛡️' : '📊'}
                      </span>
                      <span className="font-medium capitalize">{vote.agent}</span>
                    </div>
                    <span className="text-xs text-slate-400">{vote.timestamp}</span>
                  </div>
                  <div className="mt-2 text-sm text-slate-300">{vote.message}</div>
                  <div className={`mt-2 text-xs font-semibold ${
                    vote.status === 'approved' ? 'text-emerald-400' :
                    vote.status === 'rejected' ? 'text-safety-orange' : 'text-slate-400'
                  }`}>
                    {vote.status.toUpperCase()}
                  </div>
                </div>
              ))}
              
              {state.votes.length === 0 && !isLoading && (
                <div className="text-center py-8 text-slate-500">
                  Submit an intent to generate a proposal and see Shadow Council validation
                </div>
              )}
            </div>
            
            <button
              onClick={handleExecute}
              disabled={!state.canExecute || isLoading}
              className={`w-full mt-4 py-3 rounded-lg font-semibold ${
                state.canExecute && !isLoading
                  ? 'bg-emerald-600 hover:bg-emerald-700 text-white'
                  : 'bg-slate-700 text-slate-400 cursor-not-allowed'
              }`}
            >
              {state.canExecute ? 'Execute on CNC' : 'Waiting for Council Approval'}
            </button>
          </div>
        </div>

        {/* Right Column - Output & Visualization */}
        <div className="space-y-6">
          <div className="neuro-card p-6">
            <h3 className="text-lg font-semibold text-slate-200 mb-4">Generated Code</h3>
            
            <div className="bg-slate-800/50 rounded-lg p-4 font-mono text-sm overflow-auto max-h-64">
              {generatedCode ? (
                <pre className="text-slate-300">{generatedCode}</pre>
              ) : (
                <div className="text-slate-500 italic">Generated code will appear here...</div>
              )}
            </div>
          </div>
          
          <div className="neuro-card p-6">
            <h3 className="text-lg font-semibold text-slate-200 mb-4">Ghost Pass Preview</h3>
            
            <div className="aspect-video bg-slate-800/50 rounded-lg flex items-center justify-center">
              {state.proposal ? (
                <div className="text-center">
                  <div className="text-4xl mb-2">👻</div>
                  <p className="text-slate-300">Simulated toolpath for: "{state.proposal.substring(0, 30)}..."</p>
                  <p className="text-sm text-slate-500 mt-2">Real-time physics simulation in progress...</p>
                </div>
              ) : (
                <p className="text-slate-500 italic">Submit an intent to preview the ghost pass</p>
              )}
            </div>
            
            <button className="w-full mt-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-slate-300">
              Run Ghost Pass Simulation
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreatorStudio;
```

---

## 7. Telemetry Hook Implementation

```ts
// hooks/useTelemetry.ts
import { useState, useEffect, useRef } from 'react';

interface TelemetryData {
  spindle_load: number;
  vibration_x: number;
  vibration_y: number;
  vibration_z: number;
  temperature: number;
  dopamine_score: number;
  cortisol_level: number;
  spindle_rpm: number;
  feed_rate: number;
  tool_wear: number;
  oee: number;
}

interface WebSocketMessage {
  type: string;
  data: TelemetryData;
  timestamp: string;
}

/**
 * Telemetry Hook for High-Frequency Data Streaming
 * Implements a Kalman Filter for smoothing noisy sensor data
 */
const useTelemetry = (url: string = 'ws://localhost:8000/ws/telemetry') => {
  const [data, setData] = useState<TelemetryData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  
  // Kalman filter parameters for smoothing sensor data
  const kalmanState = useRef({
    x: 0, // state
    P: 1, // error covariance
    Q: 0.1, // process noise
    R: 0.5, // measurement noise
  });

  /**
   * Apply Kalman filter to smooth incoming sensor data
   * This helps prevent "Phantom Trauma" from sensor noise
   */
  const applyKalmanFilter = (measurement: number): number => {
    const state = kalmanState.current;
    
    // Prediction step
    const x_pred = state.x;
    const P_pred = state.P + state.Q;
    
    // Update step
    const K = P_pred / (P_pred + state.R); // Kalman gain
    state.x = x_pred + K * (measurement - x_pred);
    state.P = (1 - K) * P_pred;
    
    return state.x;
  };

  /**
   * Get neuro color based on dopamine and cortisol levels
   * High cortisol overrides high dopamine (safety first)
   */
  const getNeuroColor = (dopamine: number, cortisol: number): string => {
    if (cortisol > 0.7) {
      return 'border-safety-orange'; // High stress = danger
    } else if (dopamine > 0.7 && cortisol < 0.3) {
      return 'border-emerald-green'; // High reward + low stress = safe
    } else if (cortisol > 0.5) {
      return 'border-neuro-amber'; // Medium stress
    } else {
      return 'border-slate-600'; // Normal
    }
  };

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        
        // Apply Kalman filtering to vibration data to prevent Phantom Trauma
        const filteredData: TelemetryData = {
          ...message.data,
          vibration_x: applyKalmanFilter(message.data.vibration_x),
          vibration_y: applyKalmanFilter(message.data.vibration_y),
          vibration_z: applyKalmanFilter(message.data.vibration_z),
        };
        
        setData(filteredData);
      } catch (err) {
        setError('Failed to parse telemetry data');
      }
    };

    ws.onerror = (err) => {
      setError('WebSocket error occurred');
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [url]);

  return {
    data,
    isConnected,
    error,
    getNeuroColor,
    applyKalmanFilter
  };
};

export default useTelemetry;
```

---

## 8. Machine State Context Provider

```tsx
// context/MachineStateContext.tsx
import React, { createContext, useContext, useReducer } from 'react';

type UserRole = 'operator' | 'manager' | 'creator' | 'admin';

interface MachineState {
  role: UserRole;
  machineId: string | null;
  machineStatus: 'idle' | 'running' | 'maintenance' | 'alarm';
  currentTelemetry: any | null;
  isEmergencyStop: boolean;
  cortisolLevel: number;
  dopamineScore: number;
}

type MachineStateAction = 
  | { type: 'SET_ROLE'; payload: UserRole }
  | { type: 'SET_MACHINE_ID'; payload: string }
  | { type: 'UPDATE_TELEMETRY'; payload: any }
  | { type: 'TRIGGER_ESTOP' }
  | { type: 'CLEAR_ESTOP' }
  | { type: 'UPDATE_NEUROCHEMICALS'; payload: { cortisol?: number; dopamine?: number } };

const MachineStateContext = createContext<{
  state: MachineState;
  dispatch: React.Dispatch<MachineStateAction>;
} | undefined>(undefined);

const machineStateReducer = (state: MachineState, action: MachineStateAction): MachineState => {
  switch (action.type) {
    case 'SET_ROLE':
      return { ...state, role: action.payload };
    case 'SET_MACHINE_ID':
      return { ...state, machineId: action.payload };
    case 'UPDATE_TELEMETRY':
      return { ...state, currentTelemetry: action.payload };
    case 'TRIGGER_ESTOP':
      return { ...state, isEmergencyStop: true };
    case 'CLEAR_ESTOP':
      return { ...state, isEmergencyStop: false };
    case 'UPDATE_NEUROCHEMICALS':
      return {
        ...state,
        cortisolLevel: action.payload.cortisol ?? state.cortisolLevel,
        dopamineScore: action.payload.dopamine ?? state.dopamineScore
      };
    default:
      return state;
  }
};

interface MachineStateProviderProps {
  children: React.ReactNode;
}

export const MachineStateProvider: React.FC<MachineStateProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(machineStateReducer, {
    role: 'operator',
    machineId: null,
    machineStatus: 'idle',
    currentTelemetry: null,
    isEmergencyStop: false,
    cortisolLevel: 0.1,
    dopamineScore: 0.5
  });

  return (
    <MachineStateContext.Provider value={{ state, dispatch }}>
      {children}
    </MachineStateContext.Provider>
  );
};

export const useMachineState = () => {
  const context = useContext(MachineStateContext);
  if (!context) {
    throw new Error('useMachineState must be used within a MachineStateProvider');
  }
  return context;
};
```

---

## 9. API Connection Discovery Methodics

### The Interface Topology Approach for Production Systems

To figure out connections between disparate endpoints (e.g., a CAD kernel vs. a Real-time CNC controller), do not view them as simple data pipes. View them as a Translation Layer between two different domains of physics and time.

#### Step 1: Define the "Domain Mismatch"
Before coding, map the fundamental differences between the two endpoints to identify the necessary "Middleware Logic."

- **Time Domain**: Does Endpoint A run in microseconds (CNC/FOCAS) while Endpoint B runs in event-loops (SolidWorks/COM)?
  - Rule: If Latency Delta > 100ms, you need an Async Event Buffer (Redis/RabbitMQ).

- **Data Integrity**: Is the data Deterministic (Coordinates) or Probabilistic (AI Suggestions)?
  - Rule: Deterministic data requires strict validation; Probabilistic data requires a "Shadow Council" audit.

#### Step 2: The "Great Translation" Mapping
Create a dictionary that maps Source Metrics to Target Behaviors, following the "Great Translation" theory.

- **Example Mapping**:
  - Source (SolidWorks API): `PartDoc.FeatureByName("Hole1").GetHoleData().Diameter`
  - Translation Logic: Apply material-specific feed rate formula
  - Target (Fanuc API): `cnc_wrparam(tool_feed_override, calculated_value)`

#### Step 3: Architecture Layering (The Builder Pattern)
Use the Application Layers Builder pattern to segregate connection logic:
1. Presentation Layer: The human interface (Dashboard/Plugin)
2. Service Layer: The "Business Logic" (e.g., calculating stress based on geometry)
3. Data Access (Repository): The raw API wrappers (ctypes for FOCAS, pywin32 for SolidWorks)

### The SolidWorks ↔ CNC Bridge Knowledge Base

#### Connection Interfaces (Raw Protocols)

**Node A: The Visual Cortex (SolidWorks)**
- **Protocol**: COM Automation (Component Object Model)
- **Access Method**: Python pywin32 library to dispatch `SldWorks.Application`
- **Latency**: Slow (>500ms). Blocks on UI events (Dialogs)
- **Key Objects**: `ModelDoc2` (Active Document), `FeatureManager` (Design Tree), `EquationMgr` (Global Variables)

**Node B: The Spinal Cord (Fanuc CNC)**
- **Protocol**: FOCAS 2 (Ethernet/HSSB)
- **Access Method**: Python ctypes wrapper for `Fwlib32.dll`
- **Latency**: Fast (<1ms via HSSB, ~10ms via Ethernet)
- **Key Functions**: `cnc_rdload` (Read Load), `cnc_wrparam` (Write Parameter)

#### Data Mapping Strategy (Physics-Match Check)

| SolidWorks Endpoint | Fanuc Endpoint | Bridge Logic |
|-------------------|----------------|--------------|
| `Face2.GetCurvature(radius)` | `cnc_rdspeed(actual_feed_rate)` | **Quadratic Mantinel**: If curvature radius is small, cap Max Feed Rate to prevent servo jerk |
| `MassProperty.CenterOfMass` | `odm_svdiff(servoval_lag)` | **Inertia Compensation**: If CoG is offset, expect higher Servo Lag on rotary axes |
| `Simulation.FactorOfSafety` | `cnc_rdload(spindle_load%)` | **Physics Match**: If Actual Load >> Simulated Load, tool is dull or material differs |
| `Dimension.SystemValue` | `cnc_wrmacro(macro_variable_500)` | **Adaptive Resize**: Update CNC macros based on CAD dimensions for probing cycles |

#### Scaling Architectures (Implementation Patterns)

**Pattern A: "The Ghost" (Reality → Digital)**
- Goal: Visualization of the physical machine inside the CAD environment
- Data Flow: Fanuc API reads coordinates → Bridge normalizes → SolidWorks API updates ghost model → Semi-transparent overlay for collision checking
- Result: Real-time visualization of physical machine in CAD space

**Pattern B: "The Optimizer" (Digital → Reality)**
- Goal: Using simulation to drive physical parameters
- Data Flow: SolidWorks API runs headless FEA → Bridge checks stress limits → Fanuc API adjusts feed rates if safe
- Result: AI-driven optimization with safety validation

### Troubleshooting Theories for API Connections

**Theory of "Phantom Trauma" (Sensor Drift vs. Stress)**
- Problem: System incorrectly flags operations as dangerous due to sensor noise or API timing issues.
- Derivative Logic: In the "Neuro-Safety" model, stress responses linger. However, if API response timing is inconsistent, the system may interpret normal fluctuations as dangerous events.
- Troubleshooting Strategy: Implement Kalman Filter for API response smoothing

**Theory of "The Spinal Reflex" (Latency Gap Resolution)**
- Problem: Cloud-based decision making has insufficient response time for immediate hardware control.
- Solution: Implement Neuro-C architecture principles in the API bridge with integer-only operations and edge processing.

---

## 10. Summary of Frontend Architecture

This frontend architecture implements the "Glass Brain" concept that visualizes the cognitive state of the manufacturing system. The interface breathes with the machine's operational state using biological metaphors:

1. **Neuro-Safety**: Visual indicators that pulse based on stress (cortisol) and reward (dopamine) levels
2. **Alive UI**: Components that animate and respond to the machine's emotional state
3. **Cognitive Load Shedding**: Role-based interfaces that show only relevant information
4. **Shadow Council**: Distributed governance with Creator, Auditor, and Accountant agents
5. **Quadratic Mantinel**: Physics-informed geometric constraints visualized in real-time
6. **Reasoning Trace**: Chain-of-thought visualization for transparency
7. **The Great Translation**: Mapping of SaaS metrics to manufacturing physics

The architecture provides three distinct personas with specialized interfaces:
- **Operator**: Execution-focused with real-time safety indicators
- **Manager**: Fleet command with economic metrics and swarm intelligence
- **Creator**: Generative design with AI-powered optimization and safety validation

This design bridges the gap between theoretical concepts and practical implementation, creating an interface that is both scientifically rigorous and practically applicable to manufacturing excellence.