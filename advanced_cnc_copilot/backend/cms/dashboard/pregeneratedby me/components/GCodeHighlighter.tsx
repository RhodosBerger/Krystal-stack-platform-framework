
import React from 'react';
import { motion } from 'framer-motion';

interface Props {
  code: string;
  className?: string;
  activeLine?: number;
  highlightCritical?: boolean;
}

export const GCodeHighlighter: React.FC<Props> = ({ 
  code, 
  className = "", 
  activeLine = -1,
  highlightCritical = true 
}) => {
  const highlightLine = (line: string) => {
    // Advanced Regex for FANUC/ISO G-Code Tokenization
    // Group 1: Comments ( ;... or (...) )
    // Group 2: Macro Variables and Expressions ( #100, #[#500+1] )
    // Group 3: G-Commands ( G00, G01, etc. )
    // Group 4: M-Commands ( M03, M05, etc. )
    // Group 5: Axis/Coordinates ( X10.5, Z-2.0, etc. )
    // Group 6: Feed/Spindle/Tool ( F100, S2000, T01 )
    const tokens = line.split(/(\s+|;.*|\(.*\)|#\d+|#\[.*?\]|G\d+|M\d+|[XYZIJKRPQLHD][-]?\d*\.?\d*|[FST]\d*\.?\d*)/i);
    
    return tokens.map((token, i) => {
      const trimmed = token.trim();
      if (!trimmed) return <span key={i}>{token}</span>;

      // 1. Comments: Subdued Zinc
      if (trimmed.startsWith(';') || trimmed.startsWith('(')) {
        return <span key={i} className="text-zinc-600 italic select-none">{token}</span>;
      }

      // 2. Macro Variables: Electric Purple
      if (trimmed.startsWith('#')) {
        return <span key={i} className="text-purple-400 font-bold">{token}</span>;
      }

      // 3. G-Commands: Logic-based Coloration
      if (/^G\d+/i.test(trimmed)) {
        const gNum = parseInt(trimmed.substring(1));
        // Rapid (G00) is often marked as dangerous/fast in industrial UIs
        const isRapid = gNum === 0;
        return (
          <span key={i} className={`${isRapid ? 'text-safety-orange font-black' : 'text-cyber-blue font-bold'}`}>
            {token}
          </span>
        );
      }

      // 4. M-Commands: Action/Stop Coloration
      if (/^M\d+/i.test(trimmed)) {
        const mNum = parseInt(trimmed.substring(1));
        const isCritical = [0, 1, 6, 30, 99].includes(mNum);
        return (
          <motion.span 
            key={i} 
            className={`${isCritical && highlightCritical ? 'text-red-400 underline decoration-dotted' : 'text-blue-300'} font-bold`}
            animate={isCritical && highlightCritical ? { opacity: [1, 0.7, 1] } : {}}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            {token}
          </motion.span>
        );
      }

      // 5. Axis/Coordinates: High-Visibility Emerald
      if (/^[XYZIJKRPQLHD]/i.test(trimmed)) {
        return <span key={i} className="text-emerald-green font-mono">{token}</span>;
      }

      // 6. F/S/T: Process Parameters in Amber
      if (/^[FST]/i.test(trimmed)) {
        return <span key={i} className="text-amber-500 font-bold">{token}</span>;
      }

      // Default Numbers or miscellaneous tokens
      return <span key={i} className="text-zinc-400">{token}</span>;
    });
  };

  return (
    <div className={`font-mono text-[11px] leading-relaxed select-text bg-black/20 p-1 rounded-lg ${className}`}>
      {code.split('\n').filter(l => l.trim().length > 0).map((line, i) => {
        const isActive = activeLine === i;
        return (
          <div 
            key={i} 
            className={`flex gap-4 group transition-colors px-3 py-1 rounded border-l-2 ${
              isActive 
                ? 'bg-cyber-blue/10 border-cyber-blue' 
                : 'border-transparent hover:border-zinc-700 hover:bg-white/5'
            }`}
          >
            <span className={`text-right w-10 flex-shrink-0 select-none font-mono text-[9px] border-r border-zinc-800 pr-3 ${
              isActive ? 'text-cyber-blue font-black' : 'text-zinc-700'
            }`}>
              {(i + 1).toString().padStart(4, '0')}
            </span>
            <div className={`flex-1 whitespace-pre-wrap ${isActive ? 'text-white' : ''}`}>
              {isActive && (
                <motion.span 
                  initial={{ opacity: 0, x: -5 }} 
                  animate={{ opacity: 1, x: 0 }} 
                  className="inline-block mr-2 text-cyber-blue font-black"
                >
                  ▶
                </motion.span>
              )}
              {highlightLine(line)}
            </div>
          </div>
        );
      })}
    </div>
  );
};
