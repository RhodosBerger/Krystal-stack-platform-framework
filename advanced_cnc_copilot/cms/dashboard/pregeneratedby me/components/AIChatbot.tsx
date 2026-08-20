
import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GeminiChatService } from '../services/chat';
import { Telemetry, UserRole, FutureScenario } from '../types';
import { GCodeHighlighter } from './GCodeHighlighter';

interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: number;
  scenarios?: FutureScenario[];
}

interface AIChatbotProps {
  telemetry: Telemetry;
  stats: { cortisol: number; serotonin: number };
  role: UserRole;
}

const PRESETS = {
  CREATOR: "Analyze Voxel History for Inconel. Generate Thermal-Biased Mutation. Prioritize Cooling.",
  AUDITOR: "Review G-Code. Apply Death Penalty to any vertex where Curvature < 0.5mm and Feed > 1000.",
  DREAM_STATE: "Initiate Nightmare Training. Replay logs and inject random Spindle Stall at 14:00."
};

export const AIChatbot: React.FC<AIChatbotProps> = ({ telemetry, stats, role }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const [mode, setMode] = useState<'CREATOR' | 'AUDITOR' | 'DREAM_STATE'>('CREATOR');
  const [knowledgeBase, setKnowledgeBase] = useState<string>("Standard G-Code Protocols & Neuro-C Theoretical Core.");
  
  const chatService = useRef(new GeminiChatService());
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isThinking]);

  const handleSend = async (customPrompt?: string) => {
    const promptToSend = customPrompt || input;
    if (!promptToSend.trim() || isThinking) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      text: promptToSend,
      timestamp: Date.now()
    };

    setMessages(prev => [...prev, userMsg]);
    if (!customPrompt) setInput("");
    setIsThinking(true);

    const result = await chatService.current.sendMessage(promptToSend, { 
      telemetry, stats, role, mode, knowledgeBase 
    });

    const modelMsg: ChatMessage = {
      id: (Date.now() + 1).toString(),
      role: 'model',
      text: result.text,
      timestamp: Date.now(),
      scenarios: result.scenarios
    };

    setMessages(prev => [...prev, modelMsg]);
    setIsThinking(false);
  };

  const renderContent = (text: string) => {
    // Detect [CODE] blocks or generic G-code blocks (G/M sequences)
    const codeRegex = /\[CODE\]([\s\S]*?)\[\/CODE\]/g;
    const parts = text.split(codeRegex);
    
    if (parts.length > 1) {
      return parts.map((part, i) => (
        i % 2 === 1 ? (
          <div key={i} className="my-4 bg-black p-4 rounded-xl border border-white/5">
            <GCodeHighlighter code={part.trim()} />
          </div>
        ) : <p key={i} className="mb-2 last:mb-0">{part}</p>
      ));
    }

    // Fallback detection for generic blocks starting with G/M/N
    if (/^[GMN]\d+/m.test(text)) {
      return (
        <div className="bg-black p-4 rounded-xl border border-white/5 my-2">
          <GCodeHighlighter code={text} />
        </div>
      );
    }

    return text;
  };

  return (
    <>
      <motion.button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-8 right-8 w-16 h-16 rounded-2xl bg-cyber-blue shadow-[0_0_30px_rgba(59,130,246,0.4)] flex items-center justify-center z-50 text-white border border-blue-400/20"
        whileHover={{ scale: 1.05, rotate: [0, -5, 5, 0] }}
        whileTap={{ scale: 0.95 }}
      >
        <span className="text-2xl">🧠</span>
        <div className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-green rounded-full border-2 border-black" />
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 100 }}
            className="fixed top-10 right-10 bottom-10 w-[480px] bg-industrial-dark border-l border-white/5 shadow-[-50px_0_100px_rgba(0,0,0,0.8)] z-50 flex flex-col overflow-hidden glass-panel"
          >
            {/* Header: Persona & Book Selection */}
            <div className="p-6 border-b border-zinc-900 bg-zinc-900/30">
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="text-white text-xs font-black uppercase tracking-[0.3em]">Cortex Neuralis</h3>
                  <p className="text-[9px] text-zinc-600 font-bold uppercase tracking-widest mt-1">Unified Probabilistic Canvas</p>
                </div>
                <button onClick={() => setIsOpen(false)} className="text-zinc-700 hover:text-white transition-colors p-2">✕</button>
              </div>
              
              <div className="flex gap-1 bg-black/50 p-1 rounded-xl border border-white/5">
                {(['CREATOR', 'AUDITOR', 'DREAM_STATE'] as const).map(m => (
                  <button 
                    key={m}
                    onClick={() => setMode(m)}
                    className={`flex-1 py-2 rounded-lg text-[8px] font-black tracking-widest transition-all ${
                      mode === m ? 'bg-cyber-blue text-white shadow-lg' : 'text-zinc-600 hover:text-zinc-400'
                    }`}
                  >
                    {m.replace('_', ' ')}
                  </button>
                ))}
              </div>
            </div>

            {/* Knowledge Base Field */}
            <div className="px-6 py-3 border-b border-zinc-900 bg-black/20 flex items-center gap-4">
               <div className="flex items-center gap-2">
                 <div className="w-1.5 h-1.5 rounded-full bg-cyber-blue" />
                 <span className="text-[8px] text-zinc-600 font-black uppercase">Source:</span>
               </div>
               <input 
                 className="flex-1 bg-transparent border-none text-[10px] text-zinc-400 font-mono focus:outline-none"
                 value={knowledgeBase}
                 onChange={(e) => setKnowledgeBase(e.target.value)}
                 placeholder="Inject theoretical core data..."
               />
               <span className="text-[8px] text-zinc-800 font-mono italic">CRC_OK</span>
            </div>

            {/* Chat Body */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-8 scroll-smooth">
              {messages.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-center px-10">
                  <motion.div 
                    animate={{ rotate: 360 }}
                    transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                    className="w-24 h-24 border border-dashed border-zinc-800 rounded-full mb-6 flex items-center justify-center"
                  >
                    <div className="w-16 h-16 border border-zinc-700 rounded-full flex items-center justify-center">
                       <div className="w-4 h-4 bg-cyber-blue/20 rounded-full" />
                    </div>
                  </motion.div>
                  <p className="text-[10px] font-black uppercase tracking-[0.4em] text-zinc-500 mb-8">Awaiting Input Intent</p>
                  
                  <div className="grid grid-cols-1 gap-3 w-full">
                    {Object.entries(PRESETS).map(([key, val]) => (
                      <button 
                        key={key}
                        onClick={() => handleSend(val)}
                        className="p-4 bg-zinc-900/50 border border-white/5 rounded-2xl text-left hover:bg-zinc-800 transition-all group"
                      >
                        <div className="text-[8px] font-black text-cyber-blue mb-1 uppercase tracking-widest">Chapter: {key}</div>
                        <p className="text-[10px] text-zinc-500 group-hover:text-zinc-300 leading-snug">{val}</p>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((m) => (
                <div key={m.id} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className={`max-w-[95%] p-5 rounded-2xl text-[11px] leading-relaxed relative ${
                    m.role === 'user' 
                      ? 'bg-zinc-900 text-zinc-300 border border-white/5' 
                      : 'bg-black border border-zinc-800 text-zinc-300 font-mono shadow-2xl'
                  }`}>
                    {renderContent(m.text)}

                    {/* Probability Canvas for Scenarios */}
                    {m.scenarios && m.scenarios.length > 0 && (
                      <div className="mt-8 pt-6 border-t border-zinc-900 space-y-4">
                        <div className="flex justify-between items-center mb-4">
                           <span className="text-[8px] font-black uppercase text-zinc-600 tracking-[0.2em]">Scenario Probability Array</span>
                           <div className="flex gap-4 text-[7px] font-bold text-zinc-700 uppercase">
                              <span>D: Dopamine</span>
                              <span>C: Cortisol</span>
                           </div>
                        </div>
                        <div className="grid grid-cols-1 gap-4">
                          {m.scenarios.map(s => (
                            <motion.div 
                              key={s.id}
                              whileHover={{ scale: 1.02, x: 5 }}
                              className={`p-4 rounded-xl border relative overflow-hidden transition-colors ${
                                s.is_viable ? 'bg-emerald-green/5 border-emerald-green/20' : 'bg-safety-orange/5 border-safety-orange/20'
                              }`}
                            >
                              <div className="flex justify-between items-start relative z-10">
                                <div className="space-y-1">
                                  <div className={`text-[10px] font-black uppercase flex items-center gap-2 ${s.is_viable ? 'text-emerald-green' : 'text-safety-orange'}`}>
                                    {s.is_viable ? '✓' : '⚠'} {s.name}
                                  </div>
                                  <p className="text-[9px] text-zinc-500 leading-tight italic">{s.reasoning}</p>
                                </div>
                                <div className="text-right">
                                  <div className="text-[14px] font-black font-mono text-white">{(s.predicted_dopamine * 100).toFixed(0)}<span className="text-[8px] text-zinc-700 ml-0.5">%</span></div>
                                  <div className="text-[7px] font-bold text-zinc-600 uppercase">Probability</div>
                                </div>
                              </div>
                              
                              {/* Stress Visualization Bar */}
                              <div className="mt-3 h-1 bg-zinc-900 rounded-full overflow-hidden">
                                <motion.div 
                                  initial={{ width: 0 }}
                                  animate={{ width: `${s.predicted_cortisol * 100}%` }}
                                  className={`h-full ${s.predicted_cortisol > 0.6 ? 'bg-safety-orange' : 'bg-cyber-blue'}`} 
                                />
                              </div>
                            </motion.div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isThinking && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-4 p-5 bg-zinc-950 rounded-2xl border border-zinc-900">
                  <div className="relative w-8 h-8">
                     <motion.div 
                       animate={{ rotate: 360 }}
                       transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                       className="absolute inset-0 border-2 border-cyber-blue/20 border-t-cyber-blue rounded-full"
                     />
                  </div>
                  <div className="space-y-1">
                    <span className="text-[10px] text-zinc-400 font-mono uppercase tracking-widest block">Simulating Probability Matrix</span>
                    <span className="text-[8px] text-zinc-600 uppercase font-black">Scanning G-Code for Mantinel Violations...</span>
                  </div>
                </motion.div>
              )}
            </div>

            {/* Footer Input */}
            <div className="p-6 bg-zinc-950 border-t border-zinc-900">
              <div className="flex gap-3">
                <input
                  type="text"
                  placeholder={`[${mode}] Input G-Code or intent...`}
                  className="flex-1 bg-zinc-900 border border-zinc-800 rounded-2xl px-6 py-4 text-xs text-white focus:outline-none focus:border-cyber-blue/50 transition-colors shadow-inner"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                />
                <button
                  onClick={() => handleSend()}
                  disabled={isThinking || !input.trim()}
                  className="w-14 h-14 bg-cyber-blue rounded-2xl flex items-center justify-center text-white shadow-lg disabled:opacity-20 hover:bg-blue-600 transition-all"
                >
                  <span className="text-xl">↑</span>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};
