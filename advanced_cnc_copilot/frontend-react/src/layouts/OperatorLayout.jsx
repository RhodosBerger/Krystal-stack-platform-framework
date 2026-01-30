import React, { useState } from 'react';
import GCodePreview from '../components/GCodePreview';
import LLMChatPanel from '../components/LLMChatPanel';
import AppleCard from '../components/AppleCard';
import DopamineGauge from '../components/DopamineGauge';
import { useTelemetry } from '../hooks/useTelemetry';
import { usePersona } from '../context/PersonaContext';
import { motion } from 'framer-motion';
import {
   Play,
   Pause,
   Power,
   Activity,
   Zap,
   Terminal,
   Settings,
   Cpu
} from 'lucide-react';

const OperatorLayout = () => {
   const telemetry = useTelemetry();
   const { config } = usePersona();
   const [activeGCode] = useState("G90 (Absolute Distance Mode)\nG21 (Metric Units)\nM03 S8000 (Spindle On CW)\nG00 X10.5 Y20.0 (Rapid to Start)\nG01 Z-5.0 F250 (Linear Cut)\nG01 X50.0 Y20.0 F500\nG01 X50.0 Y50.0\nG01 X10.5 Y50.0\nG01 X10.5 Y20.0\nG00 Z10.0 (Retract)\nM05 (Spindle Off)\nM30 (End of Program)");

   return (
      <div className="h-screen w-full bg-[var(--apple-bg)] flex flex-col font-sans overflow-hidden">

         {/* 1. Control Center Header */}
         <header className="flex-none h-16 px-6 flex items-center justify-between z-20 bg-white/70 backdrop-blur-md border-b border-white/40">
            <div className="flex items-center gap-4">
               <div className="w-10 h-10 rounded-xl bg-black text-white flex items-center justify-center shadow-lg">
                  <Terminal size={20} />
               </div>
               <div>
                  <h1 className="text-lg font-bold text-[var(--apple-text-primary)] tracking-tight">Active Operation</h1>
                  <div className="flex items-center gap-2">
                     <span className={`w-2 h-2 rounded-full ${telemetry.is_simulated ? 'bg-yellow-500' : 'bg-green-500'} animate-pulse`} />
                     <span className="text-[11px] font-medium text-[var(--apple-text-secondary)]">
                        {telemetry.is_simulated ? 'SIMULATION MODE' : 'LIVE CONNECTION'}
                     </span>
                  </div>
               </div>
            </div>

            <div className="flex items-center gap-3">
               <div className="px-4 py-2 bg-gray-100 rounded-full text-xs font-semibold text-gray-600 font-mono">
                  T04: END MILL 6MM
               </div>
               <button className="p-2 rounded-full hover:bg-gray-100 text-red-500 transition-colors">
                  <Power size={22} />
               </button>
            </div>
         </header>

         {/* 2. Main Grid */}
         <div className="flex-1 dashboard-grid pb-24">

            {/* COLUMN 1: CONTROLS & GCODE */}
            <div className="col-span-12 lg:col-span-3 flex flex-col gap-6 h-full">
               <AppleCard className="flex-none bg-black text-white shadow-2xl border-none" title="Machine Control" icon={Zap}>
                  <div className="grid grid-cols-2 gap-3 mt-2">
                     <button className="h-24 rounded-2xl bg-green-500 hover:bg-green-400 text-white flex flex-col items-center justify-center gap-2 transition-all shadow-lg active:scale-95">
                        <Play fill="currentColor" size={28} />
                        <span className="text-xs font-bold tracking-wider">START</span>
                     </button>
                     <button className="h-24 rounded-2xl bg-gray-800 hover:bg-gray-700 text-white flex flex-col items-center justify-center gap-2 transition-all active:scale-95">
                        <Pause fill="currentColor" size={28} />
                        <span className="text-xs font-bold tracking-wider">HOLD</span>
                     </button>
                  </div>
               </AppleCard>

               <AppleCard className="flex-1 min-h-0" title="Execution Buffer">
                  <div className="h-full rounded-xl overflow-hidden bg-gray-50 border border-gray-100">
                     <GCodePreview
                        gcode={activeGCode}
                        title=""
                        validation={{ is_valid: true, is_absolute: true }}
                     />
                  </div>
               </AppleCard>
            </div>

            {/* COLUMN 2: VISUALIZATION (Main Stage) */}
            <div className="col-span-12 lg:col-span-6 flex flex-col gap-6 h-full">
               <AppleCard className="flex-1 relative overflow-hidden bg-gray-50" title="Digital Twin" icon={Activity}>
                  <div className="absolute inset-0 flex items-center justify-center">
                     <div className="w-[400px] h-[400px] rounded-full border border-gray-200 flex items-center justify-center opacity-30 animate-spin-slow">
                        <div className="w-[300px] h-[300px] rounded-full border border-gray-300 border-dashed" />
                     </div>
                  </div>

                  {/* Central Hud Element */}
                  <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
                     <motion.div
                        animate={{ scale: [1, 1.05, 1] }}
                        transition={{ duration: 4, repeat: Infinity }}
                        className="w-48 h-48 bg-white rounded-full shadow-[0_20px_60px_rgba(0,0,0,0.1)] flex flex-col items-center justify-center"
                     >
                        <span className="text-4xl font-bold text-gray-900 tracking-tighter">{telemetry.rpm}</span>
                        <span className="text-xs text-gray-400 font-medium uppercase tracking-widest mt-1">RPM</span>
                     </motion.div>

                     <div className="mt-12 grid grid-cols-3 gap-12 text-center">
                        <div>
                           <div className="text-2xl font-bold text-gray-800">{(telemetry.feed || 0).toFixed(0)}</div>
                           <div className="text-[10px] text-gray-400 font-bold uppercase">Feed Rate</div>
                        </div>
                        <div>
                           <div className="text-2xl font-bold text-gray-800">{(telemetry?.scalpel?.fro || 1) * 100}%</div>
                           <div className="text-[10px] text-gray-400 font-bold uppercase">Override</div>
                        </div>
                        <div>
                           <div className="text-2xl font-bold text-gray-800">{(telemetry.load || 12).toFixed(1)}%</div>
                           <div className="text-[10px] text-gray-400 font-bold uppercase">Load</div>
                        </div>
                     </div>
                  </div>
               </AppleCard>

               <div className="h-48 grid grid-cols-2 gap-6">
                  <AppleCard title="Dopamine" className="bg-gradient-to-br from-blue-50 to-white">
                     <div className="h-24">
                        <DopamineGauge value={telemetry.neuro_state.dopamine} />
                     </div>
                  </AppleCard>
                  <AppleCard title="Thermals">
                     <div className="flex items-end justify-between mt-4">
                        <div>
                           <div className="text-3xl font-bold text-gray-900">{telemetry.temperature_c}°C</div>
                           <div className="text-xs text-green-600 font-medium bg-green-50 px-2 py-1 rounded-full inline-block mt-2">Optimal Range</div>
                        </div>
                        <Activity className="text-gray-300 mb-2" size={48} />
                     </div>
                  </AppleCard>
               </div>
            </div>

            {/* COLUMN 3: HIVE MIND */}
            <div className="col-span-12 lg:col-span-3 h-full">
               <AppleCard className="h-full flex flex-col" title="Co-Pilot" icon={Cpu}>
                  <div className="flex-1 overflow-hidden -mx-2">
                     <LLMChatPanel title="" />
                  </div>
               </AppleCard>
            </div>

         </div>
      </div>
   );
};

export default OperatorLayout;