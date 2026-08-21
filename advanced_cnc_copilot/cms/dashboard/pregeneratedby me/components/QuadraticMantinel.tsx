
import React from 'react';
import { motion } from 'framer-motion';
import { THEME } from '../design-tokens';

interface MantinelProps {
  currentSpeed: number; // 0-200
  curvature: number; // 0-10
}

export const QuadraticMantinel: React.FC<MantinelProps> = ({ currentSpeed, curvature }) => {
  // Parabolic Curve: V_max = 200 / sqrt(curvature + 1)
  const points = Array.from({ length: 20 }, (_, i) => {
    const x = i * 0.5;
    const y = 200 / Math.sqrt(x + 1);
    return `${x * 40},${200 - y}`;
  }).join(' ');

  const limitAtCurrentX = 200 / Math.sqrt(curvature + 1);
  const isViolated = currentSpeed > limitAtCurrentX;

  return (
    <div className="bg-black/60 rounded-xl p-4 border border-zinc-800 h-full flex flex-col">
      <h3 className="text-zinc-500 text-[9px] font-bold uppercase tracking-widest mb-4">Quadratic Mantinel v1.2</h3>
      <div className="flex-1 relative">
        <svg viewBox="0 0 400 200" className="w-full h-full">
          {/* Grid Lines */}
          <line x1="0" y1="190" x2="400" y2="190" stroke="#333" strokeWidth="0.5" />
          <line x1="10" y1="0" x2="10" y2="200" stroke="#333" strokeWidth="0.5" />
          
          {/* The Mantinel Curve */}
          <polyline
            points={points}
            fill="none"
            stroke={THEME.colors.safety}
            strokeWidth="2"
            strokeDasharray="4 2"
            opacity="0.6"
          />

          {/* Current Operation Dot */}
          <motion.circle
            cx={curvature * 40 + 10}
            cy={200 - currentSpeed}
            r="4"
            fill={isViolated ? THEME.colors.safety : THEME.colors.emerald}
            animate={{ scale: isViolated ? [1, 1.5, 1] : 1 }}
            transition={{ repeat: Infinity, duration: 0.5 }}
          />

          {isViolated && (
            <text x="20" y="30" fill={THEME.colors.safety} fontSize="10" fontWeight="bold" className="animate-pulse">
              E-STOP THRESHOLD CROSSED
            </text>
          )}
        </svg>
      </div>
      <div className="flex justify-between mt-2 text-[8px] mono text-zinc-500">
        <span>Curvature (ρ)</span>
        <span>Velocity (V)</span>
      </div>
    </div>
  );
};
