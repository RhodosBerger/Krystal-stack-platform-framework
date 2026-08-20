import React from 'react';
import { motion } from 'framer-motion';
import { usePersona } from '../context/PersonaContext';

/**
 * DopamineGauge component showing a liquid SVG animation.
 * The level represents the Reinforcement Learning "Reward Score".
 */
export const DopamineGauge = ({ value = 50, label = "DOPAMINE" }) => {
    const { config } = usePersona();

    // Constrain value 0-100
    const level = Math.min(Math.max(value, 0), 100);
    const fillHeight = 100 - level;

    return (
        <div className="flex flex-col items-center gap-4 p-4 bg-black/40 backdrop-blur-md border border-white/5 rounded-2xl h-full justify-center">
            <span className="text-[9px] font-mono tracking-[0.2em] text-gray-500 uppercase">{label}</span>

            <div className="relative w-16 h-48 bg-gray-900 border border-white/10 overflow-hidden" style={{ borderRadius: '32px' }}>
                {/* Liquid SVG */}
                <motion.svg
                    viewBox="0 0 100 100"
                    preserveAspectRatio="none"
                    className="absolute inset-0 w-full h-[200%]"
                    animate={{
                        y: [`${fillHeight}%`, `${fillHeight - 2}%`, `${fillHeight}%`]
                    }}
                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                >
                    <motion.path
                        d="M 0 50 Q 25 45 50 50 T 100 50 V 100 H 0 Z"
                        fill={config.primary}
                        animate={{
                            d: [
                                "M 0 50 Q 25 45 50 50 T 100 50 V 100 H 0 Z",
                                "M 0 50 Q 25 55 50 50 T 100 50 V 100 H 0 Z",
                                "M 0 50 Q 25 45 50 50 T 100 50 V 100 H 0 Z"
                            ]
                        }}
                        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                        style={{ opacity: 0.8 }}
                    />
                </motion.svg>

                {/* Surface Reflection */}
                <div className="absolute inset-0 w-full h-full bg-gradient-to-r from-white/10 to-transparent pointer-events-none" />

                {/* Level Text Overlay */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                    <span className="text-xs font-mono font-bold text-white drop-shadow-md">
                        {Math.round(level)}%
                    </span>
                </div>
            </div>

            <div className="flex gap-1">
                {[...Array(3)].map((_, i) => (
                    <div
                        key={i}
                        className="w-1 h-1 rounded-full animate-pulse"
                        style={{ backgroundColor: config.primary, animationDelay: `${i * 0.3}s` }}
                    />
                ))}
            </div>
        </div>
    );
};

export default DopamineGauge;
