
import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Machine, MarketplaceScript } from '../types';
import { THEME } from '../design-tokens';

const MOCK_SCRIPTS: MarketplaceScript[] = [
  { id: 's1', name: 'Al-Turbo_Core_v4', survivorScore: 98, runs: 1400, avgVibration: 0.01 },
  { id: 's2', name: 'Ti-Endurance_X', survivorScore: 85, runs: 420, avgVibration: 0.04 },
  { id: 's3', name: 'Inconel_Shadow_Path', survivorScore: 92, runs: 120, avgVibration: 0.02 },
  { id: 's4', name: 'Steel_Rapid_v2', survivorScore: 78, runs: 850, avgVibration: 0.05 },
];

interface ManagerDashboardProps {
  onMachineSelect: (id: string) => void;
}

export const ManagerDashboard: React.FC<ManagerDashboardProps> = ({ onMachineSelect }) => {
  const [machines] = useState<Machine[]>(
    Array.from({ length: 12 }, (_, i) => ({
      id: `m${i}`,
      name: `R-${(i + 1).toString().padStart(2, '0')}`,
      gravity: 30 + Math.random() * 70, // 30-100
      status: Math.random() > 0.85 ? 'warning' : 'running',
      toolWear: Math.random() * 100
    }))
  );

  // Sorting logic: Survivor Score Descending
  const sortedScripts = useMemo(() => 
    [...MOCK_SCRIPTS].sort((a, b) => b.survivorScore - a.survivorScore), 
  []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
      {/* Swarm Map */}
      <div className="lg:col-span-8 space-y-6">
        <div className="bg-zinc-950 border border-zinc-900 rounded-2xl p-6 relative overflow-hidden">
          <div className="flex justify-between items-center mb-10">
            <div>
              <h3 className="text-zinc-500 text-[10px] font-bold uppercase tracking-[0.3em]">Fleet Swarm Map // Emerald Logic</h3>
              <p className="text-[9px] text-zinc-700 uppercase mt-1">Gravitational Scale tied to OEE Stability</p>
            </div>
            <div className="flex gap-4 text-[8px] uppercase font-bold text-zinc-600">
               <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-emerald-green" /> Stable Node</div>
               <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-safety-orange" /> High Churn</div>
            </div>
          </div>
          
          <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6">
            {machines.map(m => {
              const oeeScale = 0.8 + (m.gravity / 100) * 0.4; // High OEE machines are larger
              const isWarning = m.status === 'warning' || m.toolWear > 85;

              return (
                <motion.div
                  key={m.id}
                  onClick={() => onMachineSelect(m.id)}
                  whileHover={{ scale: oeeScale * 1.1, zIndex: 20 }}
                  className="aspect-square bg-zinc-900 rounded-xl border border-white/5 flex flex-col items-center justify-center relative cursor-pointer group"
                  style={{
                    scale: oeeScale,
                    boxShadow: m.gravity > 85 ? `0 0 30px rgba(16, 185, 129, 0.2)` : 'none'
                  }}
                >
                  {/* Gravitational Glow */}
                  {m.gravity > 85 && (
                    <motion.div 
                      animate={{ scale: [1, 1.3, 1], opacity: [0.1, 0.4, 0.1] }}
                      transition={{ repeat: Infinity, duration: 3 }}
                      className="absolute inset-[-4px] border border-emerald-green/20 rounded-2xl pointer-events-none" 
                    />
                  )}

                  <div className={`w-2.5 h-2.5 rounded-full mb-3 shadow-lg ${isWarning ? 'bg-safety-orange animate-pulse shadow-safety-orange/50' : 'bg-emerald-green'}`} />
                  <span className="text-[12px] font-mono text-white font-black">{m.name}</span>
                  <div className="flex flex-col items-center mt-1">
                    <span className="text-[8px] text-zinc-600 font-bold uppercase tracking-tighter">{m.gravity.toFixed(0)}% OEE</span>
                    {isWarning && <span className="text-[7px] text-safety-orange font-black mt-1">CHURN ALERT</span>}
                  </div>

                  {/* Tool Wear Progress Bar */}
                  <div className="absolute bottom-2 left-2 right-2 h-[2px] bg-zinc-800 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${m.toolWear > 80 ? 'bg-safety-orange' : 'bg-zinc-600'}`} 
                      style={{ width: `${m.toolWear}%` }} 
                    />
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

        <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-zinc-500 text-[10px] font-bold uppercase tracking-widest">Anti-Fragile Marketplace // Survivor Sorting</h3>
            <span className="text-[8px] text-zinc-600 font-mono">RANKED BY VIBRATION INTEGRITY</span>
          </div>
          <div className="space-y-3">
             {sortedScripts.map(s => (
               <div key={s.id} className="flex items-center justify-between p-4 bg-black/40 rounded-xl border border-white/5 hover:border-emerald-green/30 transition-all cursor-pointer group">
                  <div className="flex items-center gap-4">
                     <div className="w-10 h-10 rounded-lg bg-zinc-800 flex items-center justify-center border border-white/5 group-hover:bg-zinc-700 transition-colors">
                        <span className="text-lg">🛡️</span>
                     </div>
                     <div>
                        <div className="text-[11px] font-black text-white uppercase tracking-tight flex items-center gap-2">
                          {s.name}
                          {s.survivorScore > 90 && <span className="text-[7px] px-1.5 py-0.5 bg-emerald-green/20 text-emerald-green border border-emerald-green/30 rounded">ELITE SURVIVOR</span>}
                        </div>
                        <div className="text-[8px] text-zinc-600 font-bold uppercase mt-1">{s.runs} Verified Successful Cycles</div>
                     </div>
                  </div>
                  <div className="text-right">
                     <div className="text-[12px] font-mono text-emerald-green font-black">{s.survivorScore}% SURVIVAL</div>
                     <div className="text-[8px] text-zinc-500 uppercase tracking-tighter mt-1">Stress Tolerance: {s.avgVibration}g</div>
                  </div>
               </div>
             ))}
          </div>
        </div>
      </div>

      {/* Economic Panel */}
      <div className="lg:col-span-4 space-y-6">
        <div className="bg-zinc-900/50 backdrop-blur-xl border border-emerald-green/20 rounded-2xl p-8 relative overflow-hidden">
           <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-green/5 blur-3xl pointer-events-none" />
           <h3 className="text-emerald-green text-[10px] font-black uppercase tracking-[0.2em] mb-8">The Great Translation</h3>
           
           <div className="space-y-8">
              <div>
                 <div className="flex justify-between items-end mb-2">
                    <span className="text-zinc-500 text-[9px] uppercase font-black tracking-tight">Fleet Profit Rate ($Pr$)</span>
                    <span className="text-[9px] text-emerald-green font-mono">+2.4% Δ</span>
                 </div>
                 <div className="text-5xl font-mono text-white font-black tracking-tighter">$512.80<span className="text-emerald-green/50 text-base ml-2">/hr</span></div>
              </div>

              <div className="h-[1px] bg-gradient-to-r from-emerald-green/20 to-transparent" />

              <div className="grid grid-cols-1 gap-5">
                 <div className="bg-black/40 p-4 rounded-xl border border-white/5 hover:border-safety-orange/20 transition-colors">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-[9px] text-zinc-600 font-bold uppercase">Tool Churn Cost</span>
                      <span className="text-[9px] text-safety-orange mono">HIGH RISK</span>
                    </div>
                    <div className="text-2xl font-mono text-white font-black">$142.00<span className="text-[10px] text-zinc-600 ml-1">/shift</span></div>
                    <p className="text-[8px] text-zinc-600 mt-2 italic leading-tight">Mapped from R-02 vibration spikes.</p>
                 </div>

                 <div className="bg-black/40 p-4 rounded-xl border border-white/5 hover:border-cyber-blue/20 transition-colors">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-[9px] text-zinc-600 font-bold uppercase">Setup CAC</span>
                      <span className="text-[9px] text-cyber-blue mono">OPTIMAL</span>
                    </div>
                    <div className="text-2xl font-mono text-white font-black">$8.45<span className="text-[10px] text-zinc-600 ml-1">/batch</span></div>
                    <p className="text-[8px] text-zinc-600 mt-2 italic leading-tight">Setup time (Ts) reduction via Gravitational Scheduling.</p>
                 </div>
              </div>

              <div className="p-4 bg-emerald-green/5 border border-emerald-green/10 rounded-xl">
                 <div className="flex gap-3">
                    <span className="text-lg">📈</span>
                    <p className="text-[10px] text-emerald-green/80 leading-snug font-medium italic">
                      "Fleet OEE Translation: Current machine stability is yielding a 14% improvement in tool lifespan compared to baseline industrial benchmarks."
                    </p>
                 </div>
              </div>
           </div>
        </div>
        
        <div className="bg-zinc-950 border border-zinc-900 rounded-2xl p-6">
           <h3 className="text-zinc-500 text-[10px] font-bold uppercase tracking-widest mb-6">Swarm Intelligence Alerts</h3>
           <div className="space-y-4">
              <div className="flex gap-4 p-3 bg-safety-orange/5 border border-safety-orange/10 rounded-lg">
                 <div className="w-1.5 h-full bg-safety-orange rounded-full flex-shrink-0" />
                 <div>
                    <span className="text-[9px] font-black text-safety-orange uppercase">Vibration Cascade</span>
                    <p className="text-[10px] text-zinc-400 mt-0.5">R-11 reporting harmonic feedback. Potential tool fracture in 14 minutes if feed rate isn't modulated.</p>
                 </div>
              </div>
              <div className="flex gap-4 p-3 bg-emerald-green/5 border border-emerald-green/10 rounded-lg">
                 <div className="w-1.5 h-full bg-emerald-green rounded-full flex-shrink-0" />
                 <div>
                    <span className="text-[9px] font-black text-emerald-green uppercase">Peak Profit Cycle</span>
                    <p className="text-[10px] text-zinc-400 mt-0.5">Fleet-wide OEE optimized. System has entered 'Rush Mode' to capitalize on current throughput velocity.</p>
                 </div>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
};
