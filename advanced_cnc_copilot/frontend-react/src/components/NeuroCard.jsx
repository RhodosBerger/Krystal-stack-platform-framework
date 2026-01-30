import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { usePersona } from '../context/PersonaContext';

/**
 * NeuroCard component that "breathes" based on the volatility of the metric.
 * Uses persona-based design tokens.
 */
export const NeuroCard = ({ title, metric, status, volatility = 0, unit = "" }) => {
  const { config } = usePersona();

  const isOk = status === 'OK' || status === 'OPTIMAL' || status === 'NOMINAL';
  const statusColor = isOk ? 'text-neuro-success' : 'text-neuro-danger';

  // Calculate pulse duration based on volatility (0 to 1)
  // Higher volatility = faster pulse (0.5s to 3s)
  const pulseDuration = useMemo(() => {
    const v = Math.min(Math.max(volatility, 0), 1);
    return 3 - (v * 2.5);
  }, [volatility]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.01 }}
      className="relative overflow-hidden p-4 bg-black/40 backdrop-blur-md border border-white/5 group hover:border-white/20 transition-all duration-300"
      style={{ borderRadius: config.borderRadius }}
    >
      {/* Subtle Synapse Glow (Only visible on hover or volatility) */}
      <motion.div
        className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-10 transition-opacity"
        style={{
          background: `radial-gradient(circle at 70% 30%, ${config.primary}33 0%, transparent 70%)`
        }}
      />

      <div className="relative z-10">
        <h3 className="text-[10px] font-mono tracking-widest text-gray-500 uppercase mb-4">
          {title}
        </h3>

        <div className="flex items-baseline gap-2">
          <span className="text-4xl font-bold font-mono tracking-tighter" style={{ color: config.primary }}>
            {metric}
          </span>
          <span className="text-sm font-mono text-gray-400 opacity-50">{unit}</span>
        </div>

        <div className="mt-6 flex items-center justify-between">
          <div className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${isOk ? 'border-neuro-success/30 bg-neuro-success/5' : 'border-neuro-danger/30 bg-neuro-danger/5'} ${statusColor}`}>
            {status}
          </div>

          {/* Minimal volatility sparkline simulation */}
          <div className="flex gap-1 items-end h-4">
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                className="w-1 bg-white/10 rounded-full"
                animate={{ height: [4, Math.random() * 16 + 4, 4] }}
                transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.1 }}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Accent Corner Pipe */}
      <div
        className="absolute bottom-0 right-0 w-8 h-8 opacity-20"
        style={{
          background: `linear-gradient(135deg, transparent 50%, ${config.primary} 50%)`
        }}
      />
    </motion.div>
  );
};

export default NeuroCard;
