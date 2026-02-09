
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { GeminiVoiceService } from '../services/gemini';
import { TranscriptionPart } from '../types';

export const VoiceInterface: React.FC = () => {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'open' | 'closed' | 'error'>('idle');
  const [transcriptions, setTranscriptions] = useState<TranscriptionPart[]>([]);
  const voiceService = useRef<GeminiVoiceService | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcriptions]);

  const handleStart = async () => {
    if (!voiceService.current) {
      voiceService.current = new GeminiVoiceService({
        onStatusChange: (s) => setStatus(s),
        onTranscription: (text, role) => {
          setTranscriptions(prev => {
            const last = prev[prev.length - 1];
            if (last && last.role === role && (Date.now() - last.timestamp) < 2000) {
               return [...prev.slice(0, -1), { ...last, text: last.text + text }];
            }
            return [...prev, { text, role, timestamp: Date.now() }];
          });
        }
      });
    }
    await voiceService.current.connect();
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden flex flex-col h-full">
      <div className="p-4 border-b border-zinc-800 flex justify-between items-center bg-zinc-900/50">
        <div>
          <h3 className="text-zinc-400 text-xs font-bold uppercase tracking-widest">Neural Link</h3>
          <p className="text-[10px] text-zinc-500">Gemini 2.5 Native Audio Bridge</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-2 py-1 rounded text-[10px] font-bold ${
            status === 'open' ? 'text-emerald-500 bg-emerald-500/10 border border-emerald-500/20' : 
            status === 'connecting' ? 'text-yellow-500 animate-pulse' : 'text-zinc-600'
          }`}>
            <div className={`w-1.5 h-1.5 rounded-full ${status === 'open' ? 'bg-emerald-500' : 'bg-zinc-600'}`} />
            {status.toUpperCase()}
          </div>
          
          {status !== 'open' && (
            <button 
              onClick={handleStart}
              className="bg-emerald-600 hover:bg-emerald-500 text-white text-[10px] font-bold px-4 py-1.5 rounded transition-colors uppercase tracking-tight"
            >
              Initialize Link
            </button>
          )}
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 p-4 space-y-4 overflow-y-auto max-h-[300px] scroll-smooth">
        {transcriptions.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-zinc-600 text-center py-10">
            <svg className="w-12 h-12 mb-3 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-10-8a3 3 0 013-3h.01M17 11a3 3 0 01-3-3h.01M7 21a10 10 0 01-10-10h10a10 10 0 0110 10h-10z" />
            </svg>
            <p className="text-[11px] uppercase tracking-widest font-bold">Waiting for interaction...</p>
            <p className="text-[9px] max-w-[200px] mt-1 opacity-60 italic">"Say: Computer, status report on the Dopamine Engine."</p>
          </div>
        )}
        
        <AnimatePresence initial={false}>
          {transcriptions.map((t, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex flex-col ${t.role === 'user' ? 'items-end' : 'items-start'}`}
            >
              <div className={`max-w-[85%] rounded-lg p-3 ${
                t.role === 'user' ? 'bg-zinc-800 text-emerald-400' : 'bg-zinc-800/50 text-zinc-300'
              }`}>
                <p className="text-[11px] mono leading-relaxed">{t.text}</p>
                <div className="text-[8px] text-zinc-500 mt-1 uppercase">
                  {t.role === 'user' ? 'Operator' : 'Cognitive Core'} • {new Date(t.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
      
      {status === 'open' && (
        <div className="p-4 bg-zinc-950 border-t border-zinc-800">
           <div className="flex items-center gap-3">
              <div className="flex-1 h-8 bg-zinc-900 rounded-full border border-zinc-800 flex items-center px-4">
                 <div className="flex gap-1 h-3 items-center">
                    {[1,2,3,4,5,6,7,8].map(i => (
                      <motion.div 
                        key={i}
                        className="w-1 bg-emerald-500 rounded-full"
                        animate={{ height: ['20%', '100%', '20%'] }}
                        transition={{ duration: 0.5, repeat: Infinity, delay: i * 0.1 }}
                      />
                    ))}
                 </div>
                 <span className="ml-3 text-[10px] text-zinc-500 mono">LINK ACTIVE: STREAMING PCM 16kHz</span>
              </div>
           </div>
        </div>
      )}
    </div>
  );
};
