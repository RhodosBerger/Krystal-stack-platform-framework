
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { NeuroCard } from './NeuroCard';
import { QuadraticMantinel } from './QuadraticMantinel';
import { VoiceInterface } from './VoiceInterface';
import { Telemetry } from '../types';
import { THEME } from '../design-tokens';

interface Props {
  telemetry: Telemetry;
  setTelemetry: React.Dispatch<React.SetStateAction<Telemetry>>;
  stats: { cortisol: number; serotonin: number };
  reward: number;
}

export const OperatorDashboard: React.FC<Props> = ({ telemetry, setTelemetry, stats, reward }) => {
  const [logs, setLogs] = useState<string[]>([]);
  
  useEffect(() => {
    const messages = [
      "Auditor: Toolpath scan complete. No deflection detected.",
      "Council: High-vibration event flagged. Damping adjusted.",
      "Sensory: Spindle torque nominal for Al-6061.",
      "Shadow: Strategy optimization locked. Feed 120%.",
      "Auditor: Rejected RPM increase. Exceeds Mantinel boundary."
    ];
    if (Math.random() > 0.95) {
      setLogs(prev => [messages[Math.floor(Math.random() * messages.length)], ...prev].slice(0, 10));
    }
  }, [telemetry]);

  const handleSpeedChange = (val: number) => {
    setTelemetry(prev => ({ ...prev, speed: val }));
  };

  const handleQualityChange = (val: number) => {
    setTelemetry(prev => ({ ...prev, quality: val }));
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left: Bio-Monitoring */}
      <div className="lg:col-span-4 space-y-6">
        <NeuroCard 
          title="Execution Brain" 
          stress_level={stats.cortisol / 5} 
          efficiency={stats.serotonin}
          vibration={telemetry.vibration}
          speed={telemetry.speed}
          quality={telemetry.quality}
          onSpeedChange={handleSpeedChange}
          onQualityChange={handleQualityChange}
        >
          <div className="space-y-4">
             <div className="bg-black/40 p-3 rounded border border-white/5">
                <div className="flex justify-between items-center mb-1">
                   <span className="text-[8px] text-zinc-600 font-bold uppercase">Reward ($R$)</span>
                   <span className="text-[8px] text-emerald-500 mono">{(reward).toFixed(2)}</span>
                </div>
                <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                   <motion.div className="h-full bg-emerald-500" animate={{ width: `${Math.min(100, reward * 10)}%` }} />
                </div>
             </div>
             <div className="bg-black/40 p-3 rounded border border-white/5">
                <div className="flex justify-between items-center mb-1">
                   <span className="text-[8px] text-zinc-600 font-bold uppercase">Stress (Cortisol)</span>
                   <span className="text-[8px] text-safety-orange mono">{stats.cortisol.toFixed(2)}</span>
                </div>
                <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                   <motion.div className="h-full bg-safety-orange" animate={{ width: `${Math.min(100, stats.cortisol * 20)}%` }} />
                </div>
             </div>
          </div>
        </NeuroCard>

        <div className="bg-zinc-950 border border-zinc-800 rounded-xl p-5 h-[300px] flex flex-col">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <h3 className="text-zinc-500 text-[9px] font-bold uppercase tracking-widest">Reasoning Trace // Auditor</h3>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 font-mono text-[9px] text-zinc-400">
             {logs.map((log, i) => (
               <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} key={i} className="border-l border-zinc-800 pl-3 py-1">
                 <span className="text-zinc-600 mr-2">[{new Date().toLocaleTimeString()}]</span>
                 {log}
               </motion.div>
             ))}
             {logs.length === 0 && <span className="text-zinc-800 italic">Waiting for reasoning data...</span>}
          </div>
        </div>
      </div>

      {/* Right: Physics & Voice */}
      <div className="lg:col-span-8 flex flex-col gap-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-[400px]">
          <QuadraticMantinel currentSpeed={telemetry.speed} curvature={telemetry.vibration * 100} />
          <VoiceInterface />
        </div>
        
        <div className="grid grid-cols-3 gap-6">
           {[
             { label: 'RPM', val: telemetry.rpm, max: 15000, unit: '' },
             { label: 'Load', val: telemetry.spindleLoad, max: 100, unit: '%' },
             { label: 'Temp', val: telemetry.temp, max: 120, unit: '°C' }
           ].map(t => (
             <div key={t.label} className="bg-zinc-900 border border-zinc-800 p-4 rounded-xl">
               <span className="text-[8px] text-zinc-600 font-bold uppercase">{t.label}</span>
               <div className="text-xl font-mono font-bold text-white mt-1">{t.val.toFixed(0)}{t.unit}</div>
               <div className="h-0.5 bg-zinc-800 mt-2">
                 <motion.div className="h-full bg-white" animate={{ width: `${(t.val/t.max)*100}%` }} />
               </div>
             </div>
           ))}
        </div>
      </div>
    </div>
  );
};
