
import React, { useReducer, useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Message, CouncilState, CouncilAction, Material } from '../types';
import { AuditorAgent } from '../services/cms';
import { GCodeHighlighter } from './GCodeHighlighter';

const auditor = new AuditorAgent();

const MATERIALS: Material[] = [
  { name: 'Al-6061', maxPower: 4500, thermalLimit: 120, ductility: 0.9 },
  { name: 'Ti-6Al-4V', maxPower: 2200, thermalLimit: 450, ductility: 0.4 },
  { name: 'Inconel 718', maxPower: 1800, thermalLimit: 900, ductility: 0.2 },
];

const councilReducer = (state: CouncilState, action: CouncilAction): CouncilState => {
  switch (action.type) {
    case 'ADD_MESSAGE':
      const canRun = action.message.status === 'approved';
      return { ...state, messages: [...state.messages, action.message], canRun };
    case 'SET_LOCK':
      return { ...state, isLocked: action.locked };
    case 'RESET':
      return { messages: [], isLocked: false, canRun: false };
    default:
      return state;
  }
};

export const CreativeStudio: React.FC = () => {
  const [aggression, setAggression] = useState(30);
  const [creativity, setCreativity] = useState(50);
  const [prompt, setPrompt] = useState("");
  const [selectedMaterial, setSelectedMaterial] = useState<Material>(MATERIALS[0]);
  const [state, dispatch] = useReducer(councilReducer, { messages: [], isLocked: false, canRun: false });
  
  // Simulation State
  const [simLine, setSimLine] = useState(-1);
  const [isSimulating, setIsSimulating] = useState(false);

  const lastGCode = useMemo(() => {
    return state.messages
      .filter(m => m.role === 'creator')
      .map(m => m.text)
      .pop() || null;
  }, [state.messages]);

  // Simulate Ghost Pass Dry Run
  useEffect(() => {
    if (lastGCode && !isSimulating && state.canRun) {
      const lines = lastGCode.split('\n').filter(l => l.trim().length > 0);
      setIsSimulating(true);
      let current = 0;
      
      const interval = setInterval(() => {
        if (current < lines.length) {
          setSimLine(current);
          current++;
        } else {
          clearInterval(interval);
          setSimLine(-1);
          setIsSimulating(false);
        }
      }, 400); // Simulated machine speed

      return () => clearInterval(interval);
    }
  }, [lastGCode, state.canRun]);

  const handleGenerate = async () => {
    if (!prompt) return;
    dispatch({ type: 'SET_LOCK', locked: true });
    setSimLine(-1);
    setIsSimulating(false);
    
    // User Intent
    dispatch({ type: 'ADD_MESSAGE', message: { id: Date.now().toString(), role: 'user', text: `${prompt} (${selectedMaterial.name})` } });
    setPrompt("");

    // Phase 1: Creator Persona Propose Strategy
    await new Promise(r => setTimeout(r, 800));
    const calculatedRpm = 8000 + (aggression * 100);
    const calculatedFeed = 0.5 + (aggression / 200);
    
    const proposal = `(RISE GENERATIVE STRATEGY: ${selectedMaterial.name.toUpperCase()})\n(TIMESTAMP: ${new Date().toLocaleTimeString()})\n\nN10 G00 G90 G54 Z10.0;\nN20 M03 S${calculatedRpm};\nN30 G01 X100.0 Y50.0 F${calculatedFeed.toFixed(2)};\nN40 Z-2.0 F50.0;\nN50 G02 X150.0 Y50.0 R25.0 F${calculatedFeed.toFixed(2)};\nN60 G01 Z10.0;\nN70 M05;\nN80 M30; (Strategy: ${aggression > 70 ? 'High-Torque Rush' : 'Thermal Optimization'})`;
    dispatch({ type: 'ADD_MESSAGE', message: { id: Date.now().toString() + '_c', role: 'creator', text: proposal } });

    // Phase 2: Deterministic Auditor Logic
    await new Promise(r => setTimeout(r, 1200));
    const auditResult = auditor.validatePlan(proposal, selectedMaterial, { rpm: calculatedRpm, feed: calculatedFeed });
    
    dispatch({ 
      type: 'ADD_MESSAGE', 
      message: { 
        id: Date.now().toString() + '_a', 
        role: 'auditor', 
        text: auditResult.reasoningTrace,
        status: auditResult.approved ? 'approved' : 'veto'
      } 
    });
    
    dispatch({ type: 'SET_LOCK', locked: false });
  };

  const bgGradient = `linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(255, 87, 34, ${aggression/200}) 100%)`;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 min-h-[600px] pb-20" style={{ background: bgGradient }}>
      
      {/* Left: Emotional Nexus */}
      <div className="lg:col-span-5 space-y-8 bg-zinc-900/40 p-8 rounded-3xl border border-white/5 backdrop-blur-xl h-fit">
        <div>
          <h3 className="text-white text-[10px] font-black uppercase tracking-[0.4em] mb-8">Emotional Nexus // Input</h3>
          
          <div className="space-y-10">
            <div>
               <span className="text-[10px] font-bold text-zinc-500 uppercase block mb-4">Target Material</span>
               <div className="flex gap-2">
                 {MATERIALS.map(m => (
                   <button 
                     key={m.name}
                     onClick={() => setSelectedMaterial(m)}
                     className={`flex-1 py-2 rounded text-[9px] font-black border transition-all ${
                       selectedMaterial.name === m.name ? 'bg-cyber-blue border-cyber-blue text-white' : 'bg-black/20 border-zinc-800 text-zinc-600 hover:text-white'
                     }`}
                   >
                     {m.name}
                   </button>
                 ))}
               </div>
            </div>

            <div>
               <div className="flex justify-between items-baseline mb-4">
                  <span className="text-[10px] font-bold text-zinc-500 uppercase">Aggression</span>
                  <span className={`text-xl font-mono font-bold ${aggression > 70 ? 'text-safety-orange' : 'text-cyber-blue'}`}>{aggression}%</span>
               </div>
               <input 
                 type="range" value={aggression} onChange={(e) => setAggression(Number(e.target.value))}
                 className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyber-blue"
               />
            </div>

            <div>
               <div className="flex justify-between items-baseline mb-4">
                  <span className="text-[10px] font-bold text-zinc-500 uppercase">Creativity</span>
                  <span className="text-xl font-mono font-bold text-emerald-green">{creativity}%</span>
               </div>
               <input 
                 type="range" value={creativity} onChange={(e) => setCreativity(Number(e.target.value))}
                 className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-emerald-green"
               />
            </div>
          </div>
        </div>

        <div className="pt-8 border-t border-white/5">
           <textarea 
             placeholder="Natural language intent (e.g., 'Make the roughing cycle faster without melting the tool')..."
             className="w-full bg-black/40 border border-zinc-800 rounded-xl p-4 text-xs font-mono text-zinc-300 focus:outline-none focus:border-cyber-blue/50 transition-colors h-24 resize-none"
             value={prompt}
             onChange={(e) => setPrompt(e.target.value)}
           />
           <button 
             onClick={handleGenerate}
             disabled={state.isLocked}
             className="w-full mt-4 py-4 bg-cyber-blue hover:bg-blue-600 disabled:opacity-50 text-white font-black text-[10px] uppercase tracking-[0.2em] rounded-xl transition-all shadow-[0_10px_30px_rgba(59,130,246,0.3)]"
           >
             {state.isLocked ? "Consulting Council..." : "Generate Proposed Strategy"}
           </button>
        </div>
      </div>

      {/* Right: Shadow Council & Ghost Pass */}
      <div className="lg:col-span-7 flex flex-col gap-6">
        {/* Deliberation Chat */}
        <div className="h-[400px] bg-zinc-950/80 border border-zinc-900 rounded-3xl p-6 flex flex-col overflow-hidden shadow-2xl backdrop-blur-md">
          <div className="flex justify-between items-center mb-6">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-cyber-blue animate-pulse shadow-[0_0_10px_rgba(59,130,246,0.8)]" />
              <h3 className="text-zinc-500 text-[10px] font-bold uppercase tracking-widest">Shadow Council DELIBERATION</h3>
            </div>
            {isSimulating && (
              <span className="text-[9px] text-cyber-blue font-mono animate-pulse uppercase">Dry Run In Progress...</span>
            )}
          </div>

          <div className="flex-1 overflow-y-auto space-y-6 px-2 scroll-smooth">
            {state.messages.map((m) => (
              <motion.div 
                key={m.id} 
                initial={{ opacity: 0, x: m.role === 'user' ? 20 : -20 }} 
                animate={{ opacity: 1, x: 0 }}
                className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div className={`max-w-[90%] p-4 rounded-2xl text-[11px] leading-relaxed relative ${
                  m.role === 'user' ? 'bg-zinc-800 text-zinc-300 shadow-inner' :
                  m.role === 'creator' ? 'bg-cyber-blue/5 border border-cyber-blue/20 text-cyber-blue font-mono' :
                  `bg-zinc-900 border ${m.status === 'veto' ? 'border-safety-orange/40 bg-safety-orange/5' : 'border-emerald-green/40 bg-emerald-green/5'} text-zinc-400`
                }`}>
                  {m.role === 'auditor' && (
                    <div className={`absolute -top-2 -left-2 w-6 h-6 rounded-full bg-zinc-950 flex items-center justify-center border shadow-xl ${m.status === 'approved' ? 'border-emerald-green' : 'border-safety-orange'}`}>
                      {m.status === 'approved' ? '🛡️' : '⚠️'}
                    </div>
                  )}
                  {m.role === 'creator' ? (
                    <GCodeHighlighter code={m.text} className="bg-transparent" activeLine={m.id === state.messages.filter(msg => msg.role === 'creator').pop()?.id ? simLine : -1} />
                  ) : m.text}
                </div>
                <span className="text-[8px] text-zinc-700 font-bold uppercase mt-2 px-1 tracking-[0.1em]">{m.role}</span>
              </motion.div>
            ))}
            {state.isLocked && (
               <div className="flex gap-2 p-3 bg-zinc-900/50 rounded-xl border border-white/5">
                  <div className="flex gap-1 items-center">
                    {[1,2,3].map(i => <div key={i} className="w-1 h-1 bg-cyber-blue rounded-full animate-pulse" style={{ animationDelay: `${i*0.2}s` }} />)}
                  </div>
                  <span className="text-[8px] text-zinc-600 font-mono uppercase">Analyzing kinematic feasibility...</span>
               </div>
            )}
          </div>
        </div>

        {/* Ghost Pass Preview */}
        <AnimatePresence>
          {lastGCode && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-black/90 border border-zinc-800 rounded-3xl p-6 shadow-2xl relative overflow-hidden"
            >
              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full shadow-[0_0_10px_rgba(255,87,34,0.8)] ${isSimulating ? 'bg-emerald-green' : 'bg-safety-orange'}`} />
                  <h3 className="text-zinc-500 text-[10px] font-black uppercase tracking-[0.3em]">Ghost Pass // Dry Run Matrix</h3>
                </div>
                <div className="flex gap-4 items-center">
                  <div className="text-[8px] text-zinc-700 font-mono bg-zinc-900 px-2 py-0.5 rounded uppercase">
                    Status: {isSimulating ? 'Active Simulation' : 'Standby'}
                  </div>
                  <div className="text-[8px] text-zinc-700 font-mono bg-zinc-900 px-2 py-0.5 rounded uppercase">
                    Protocol: FANUC-31i-B
                  </div>
                </div>
              </div>
              
              <div className="bg-zinc-950/50 rounded-2xl border border-white/5 p-2 max-h-[350px] overflow-y-auto custom-scrollbar">
                <GCodeHighlighter code={lastGCode} activeLine={simLine} />
              </div>

              <div className="mt-6 flex gap-4">
                 <button 
                   className="px-6 py-3 border border-zinc-800 rounded-xl text-[9px] font-black uppercase tracking-widest text-zinc-600 hover:text-white hover:bg-zinc-800 transition-all"
                   onClick={() => {
                     dispatch({ type: 'RESET' });
                     setSimLine(-1);
                     setIsSimulating(false);
                   }}
                 >
                   Purge Session
                 </button>
                 <button 
                   disabled={!state.canRun || isSimulating}
                   className={`flex-1 py-3 rounded-xl text-[9px] font-black uppercase tracking-widest transition-all ${
                     state.canRun 
                     ? 'bg-emerald-green text-white shadow-[0_0_30px_rgba(16,185,129,0.5)] active:scale-95' 
                     : 'bg-zinc-900 text-zinc-700 cursor-not-allowed border border-zinc-800 grayscale'
                   }`}
                 >
                   {isSimulating ? 'Simulation Processing...' : `Commit Plan to HAL // ${selectedMaterial.name}`}
                 </button>
              </div>
              
              {/* Simulation Progress Overlay */}
              {isSimulating && (
                <div className="absolute bottom-0 left-0 h-1 bg-cyber-blue shadow-[0_0_10px_#3b82f6]" style={{ width: `${(simLine / lastGCode.split('\n').filter(l=>l.trim()).length) * 100}%`, transition: 'width 0.4s linear' }} />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
