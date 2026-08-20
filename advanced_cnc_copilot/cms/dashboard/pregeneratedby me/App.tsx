
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { DopamineEngine, ProfitOptimizer, SecurityService } from './services/cms';
import { Telemetry, UserRole, User } from './types';
import { OperatorDashboard } from './components/OperatorDashboard';
import { ManagerDashboard } from './components/ManagerDashboard';
import { CreativeStudio } from './components/CreativeStudio';
import { AIChatbot } from './components/AIChatbot';
import { useTelemetry } from './hooks/useTelemetry';

const App: React.FC = () => {
  const [telemetry, setTelemetry] = useTelemetry({
    speed: 120, quality: 0.99, vibration: 0.015, temp: 42, rpm: 12000, spindleLoad: 25
  });

  // Services
  const [dopamine] = useState(() => new DopamineEngine());
  const [profit] = useState(() => new ProfitOptimizer());
  const [auth] = useState(() => new SecurityService());
  
  // App State
  const [role, setRole] = useState<UserRole>('OPERATOR');
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [reward, setReward] = useState(0);
  const [stats, setStats] = useState({ cortisol: 0, serotonin: 1.0 });

  // Handle Role Change / Auth
  useEffect(() => {
    const user = auth.login(role);
    setCurrentUser(user);
  }, [role, auth]);

  useEffect(() => {
    setReward(dopamine.calculateReward(telemetry));
    setStats(dopamine.getStats());
  }, [telemetry, dopamine]);

  const handleMachineSelect = (id: string) => {
    setSelectedMachineId(id);
    setRole('OPERATOR'); 
  };

  const handleBackToFleet = () => {
    setSelectedMachineId(null);
    setRole('MANAGER');
  };

  return (
    <div className={`min-h-screen p-6 md:p-10 transition-colors duration-1000 ${role === 'MANAGER' ? 'bg-industrial-dark' : 'bg-black'}`}>
      
      {/* Global AI Chatbot Layer */}
      <AIChatbot telemetry={telemetry} stats={stats} role={role} />

      {/* Universal Header */}
      <header className="flex flex-col md:flex-row justify-between items-end gap-6 mb-12">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
             <div className="w-8 h-8 rounded-lg bg-cyber-blue flex items-center justify-center font-black text-white italic">F</div>
             <h1 className="text-4xl font-black italic tracking-tighter text-white">FANUC RISE <span className="text-zinc-700 not-italic font-light">v2.1</span></h1>
          </div>
          <div className="flex items-center gap-2">
            <p className="text-zinc-500 text-[10px] uppercase font-bold tracking-[0.4em]">Glass Brain Machine Copilot</p>
            {selectedMachineId && (
              <motion.div 
                initial={{ opacity: 0, x: -10 }} 
                animate={{ opacity: 1, x: 0 }}
                className="text-cyber-blue text-[10px] font-black uppercase tracking-widest bg-cyber-blue/10 px-2 py-0.5 rounded border border-cyber-blue/20"
              >
                // DRILL_DOWN: {selectedMachineId}
              </motion.div>
            )}
            <div className="text-[9px] text-zinc-700 font-mono uppercase ml-4">
              Authenticated: {currentUser?.username} ({currentUser?.role})
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6">
           <div className="flex gap-2 bg-zinc-900 px-3 py-2 rounded-lg border border-white/5">
              <div className={`w-3 h-3 rounded-full border border-black/20 ${telemetry.vibration > 0.05 ? 'bg-safety-orange animate-pulse shadow-[0_0_10px_#ff5722]' : 'bg-zinc-800'}`} />
              <div className={`w-3 h-3 rounded-full border border-black/20 ${telemetry.vibration > 0.02 && telemetry.vibration <= 0.05 ? 'bg-yellow-500' : 'bg-zinc-800'}`} />
              <div className={`w-3 h-3 rounded-full border border-black/20 ${telemetry.vibration <= 0.02 ? 'bg-emerald-green' : 'bg-zinc-800'}`} />
           </div>

           <div className="flex items-center gap-3 bg-zinc-900/50 p-1.5 rounded-xl border border-zinc-800">
             {(['OPERATOR', 'MANAGER', 'CREATOR', 'ADMIN'] as UserRole[]).map(r => (
               <button
                 key={r}
                 onClick={() => {
                   setRole(r);
                   if (r === 'MANAGER' || r === 'CREATOR' || r === 'ADMIN') setSelectedMachineId(null);
                 }}
                 className={`px-5 py-2 rounded-lg text-[10px] font-black transition-all ${
                   role === r 
                     ? 'bg-zinc-800 text-white shadow-xl border border-zinc-700' 
                     : 'text-zinc-600 hover:text-zinc-400'
                 }`}
               >
                 {r}
               </button>
             ))}
           </div>
        </div>
      </header>

      {selectedMachineId && (
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={handleBackToFleet}
          className="mb-8 flex items-center gap-2 text-[10px] font-black uppercase text-zinc-500 hover:text-white transition-colors tracking-widest"
        >
          <span>←</span> Back to Fleet Overview
        </motion.button>
      )}

      {/* Persona Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={role + (selectedMachineId || '')}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.3 }}
        >
          {role === 'OPERATOR' && <OperatorDashboard telemetry={telemetry} setTelemetry={setTelemetry} stats={stats} reward={reward} />}
          {role === 'MANAGER' && <ManagerDashboard onMachineSelect={handleMachineSelect} />}
          {(role === 'CREATOR' || role === 'ADMIN') && <CreativeStudio />}
        </motion.div>
      </AnimatePresence>

      <footer className="mt-12 pt-8 border-t border-zinc-900 flex justify-between items-center opacity-40 hover:opacity-100 transition-opacity">
         <div className="text-[9px] text-zinc-600 uppercase font-black mono tracking-[0.2em]">
            System Architecture: FANUC-RISE // Unified Cognition // 1kHz Link
         </div>
         <div className="flex gap-6">
            <span className="text-[9px] text-zinc-700 uppercase font-bold">Reflex Link: Nominal</span>
            <span className="text-[9px] text-emerald-green uppercase font-bold tracking-widest">Active Neural Cortex</span>
         </div>
      </footer>
    </div>
  );
};

export default App;
