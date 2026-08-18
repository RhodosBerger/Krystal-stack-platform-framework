
import React from 'react';
import { motion } from 'framer-motion';
import { THEME } from '../design-tokens';

interface NeuroCardProps {
  title: string;
  stress_level: number;
  efficiency: number;
  vibration?: number;
  speed?: number;
  quality?: number;
  onSpeedChange?: (val: number) => void;
  onQualityChange?: (val: number) => void;
  children?: React.ReactNode;
}

export const NeuroCard: React.FC<NeuroCardProps> = ({ 
  title, 
  stress_level, 
  efficiency, 
  vibration = 0, 
  speed,
  quality,
  onSpeedChange,
  onQualityChange,
  children 
}) => {
  const isHighStress = stress_level > 0.7;
  const borderColor = isHighStress ? THEME.colors.safety : THEME.colors.emerald;
  
  // Synesthesia: Apply visual entropy based on physical vibration
  const entropyStyles = THEME.physics.getEntropy(vibration);
  const heartbeat = THEME.physics.getHeartbeat(stress_level);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative bg-zinc-900 border border-zinc-800/50 rounded-2xl p-6 overflow-hidden glass-panel"
      style={{
        boxShadow: `0 0 40px rgba(${isHighStress ? '255, 87, 34' : '16, 185, 129'}, ${stress_level * 0.15})`
      }}
    >
      {/* Biological Heartbeat Border */}
      <motion.div
        className="absolute inset-0 pointer-events-none rounded-2xl"
        // Fix: Cast the heartbeat object to any to avoid complex type intersection issues in animate prop
        animate={{
          border: `1.5px solid ${borderColor}`,
          ...(heartbeat as any)
        }}
      />

      <motion.div 
        // Fix: Cast entropyStyles to any to resolve TargetAndTransition type mismatch
        animate={vibration > 0.05 ? (entropyStyles as any) : {}}
        className="relative z-10"
      >
        <div className="flex justify-between items-start mb-6">
          <div>
            <h3 className="text-zinc-500 text-[10px] font-bold uppercase tracking-[0.2em]">{title}</h3>
            <div className="flex items-baseline gap-3 mt-1">
              <span className="text-3xl font-bold font-mono tracking-tighter text-white">
                {(efficiency * 100).toFixed(1)}%
              </span>
              <span className="text-[9px] text-zinc-600 font-bold uppercase tracking-widest">Efficiency</span>
            </div>
          </div>
          
          <div className={`px-2.5 py-1 rounded-md text-[9px] font-black tracking-tighter border ${
            isHighStress 
              ? 'border-safety-orange text-safety-orange bg-safety-orange/10' 
              : 'border-emerald-green text-emerald-green bg-emerald-green/10'
          }`}>
            {isHighStress ? 'CORTISOL SPIKE' : 'STABLE STATE'}
          </div>
        </div>

        <div className="space-y-5">
          {children}

          {/* Manual Overrides for Reward Tuning */}
          {(onSpeedChange || onQualityChange) && (
            <div className="pt-4 mt-4 border-t border-zinc-800/50 space-y-4">
              <span className="text-[8px] text-zinc-500 font-black uppercase tracking-widest">Inference Overrides</span>
              
              {onSpeedChange && speed !== undefined && (
                <div className="space-y-1">
                  <div className="flex justify-between items-center text-[9px] font-bold uppercase">
                    <span className="text-zinc-600">Intentional Speed</span>
                    <span className="text-cyber-blue font-mono">{speed.toFixed(0)}</span>
                  </div>
                  <input 
                    type="range" min="0" max="250" step="1"
                    value={speed}
                    onChange={(e) => onSpeedChange(Number(e.target.value))}
                    className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyber-blue"
                  />
                </div>
              )}

              {onQualityChange && quality !== undefined && (
                <div className="space-y-1">
                  <div className="flex justify-between items-center text-[9px] font-bold uppercase">
                    <span className="text-zinc-600">Quality Precision</span>
                    <span className="text-emerald-green font-mono">{(quality * 100).toFixed(1)}%</span>
                  </div>
                  <input 
                    type="range" min="0.5" max="1" step="0.001"
                    value={quality}
                    onChange={(e) => onQualityChange(Number(e.target.value))}
                    className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-green"
                  />
                </div>
              )}
            </div>
          )}
          
          <div className="pt-2">
            <div className="flex justify-between text-[9px] font-bold text-zinc-600 uppercase mb-2">
              <span>Neuro-Biological Gradient</span>
              <span className={isHighStress ? 'text-safety-orange' : 'text-emerald-green'}>
                {Math.round(stress_level * 100)}%
              </span>
            </div>
            <div className="h-1.5 bg-zinc-800/50 rounded-full overflow-hidden p-[1px]">
              <motion.div 
                className={`h-full rounded-full ${isHighStress ? 'bg-safety-orange' : 'bg-emerald-green'}`}
                animate={{ width: `${stress_level * 100}%` }}
                transition={{ type: 'spring', damping: 15 }}
              />
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};
