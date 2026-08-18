
import React from 'react';
import { motion } from 'framer-motion';
import { Machine, Job } from '../types';

interface GravitationalViewProps {
  machines: Machine[];
  jobs: Job[];
}

export const GravitationalView: React.FC<GravitationalViewProps> = ({ machines, jobs }) => {
  return (
    <div className="bg-black/40 rounded-xl p-6 border border-zinc-800 h-[400px] relative overflow-hidden flex items-center justify-center">
      <div className="absolute top-4 left-4">
        <h3 className="text-zinc-400 text-xs font-bold uppercase tracking-widest">Process Gravitator v1.0</h3>
        <p className="text-[10px] text-zinc-600">Dynamic Multi-Tier Scheduling Orbits</p>
      </div>

      {machines.map((m, i) => {
        const x = (i % 3) * 200 - 200;
        const y = Math.floor(i / 3) * 150 - 75;
        const orbitSize = 60 + m.gravity * 20;

        return (
          <div key={m.id} className="absolute" style={{ transform: `translate(${x}px, ${y}px)` }}>
            {/* Machine Core */}
            <div className="relative flex items-center justify-center">
              <motion.div
                className="w-12 h-12 rounded-full border border-emerald-500/50 flex items-center justify-center bg-zinc-900"
                animate={{ scale: [0.95, 1.05, 0.95] }}
                transition={{ duration: 4, repeat: Infinity }}
              >
                <span className="mono text-[8px] text-emerald-400 font-bold">{m.name}</span>
              </motion.div>
              
              {/* Gravitational Field */}
              <div 
                className="absolute border border-zinc-800 rounded-full border-dashed"
                style={{ width: orbitSize * 2, height: orbitSize * 2 }}
              />

              {/* Satellites (Jobs) */}
              {jobs.filter(j => j.id.length % machines.length === i).map((job, ji) => (
                <motion.div
                  key={job.id}
                  className="absolute w-4 h-4 rounded-full bg-zinc-700 border border-zinc-600 flex items-center justify-center"
                  animate={{ rotate: 360 }}
                  transition={{ 
                    duration: (100 - job.velocity) / 5 + 2, 
                    repeat: Infinity, 
                    ease: "linear",
                    delay: ji * 0.5
                  }}
                  style={{ originX: `${orbitSize / 2}px`, left: -orbitSize }}
                >
                  <div className="mono text-[6px] text-zinc-300 font-bold">J{ji+1}</div>
                </motion.div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
};
