import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Cpu, Zap, Radio, Target, Activity } from 'lucide-react';

const RoboticsSim = () => {
    const [robotState, setRobotState] = useState({
        joint1: 0,
        joint2: 45,
        joint3: -30,
        temp: 38.2,
        power: 1.2, // kW
        mode: "AUTO"
    });

    useEffect(() => {
        const interval = setInterval(() => {
            setRobotState(prev => ({
                ...prev,
                joint1: prev.joint1 + (Math.random() * 2 - 1),
                joint2: 45 + Math.sin(Date.now() / 1000) * 10,
                power: 1.1 + Math.random() * 0.4
            }));
        }, 100);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="flex flex-col h-full gap-6">
            <header className="flex justify-between items-center border-b border-white/5 pb-4">
                <div className="flex items-center gap-3">
                    <Cpu className="text-industrial-primary" />
                    <h2 className="neuro-text text-white font-bold">ROBOTIC CELL SIMULATOR</h2>
                </div>
                <div className="px-2 py-0.5 bg-industrial-primary text-black text-[8px] font-bold rounded">KUKA-KR60-V2</div>
            </header>

            <div className="flex-1 flex flex-col gap-6">
                {/* 2D Robot Schematic / Animation Placeholder */}
                <div className="flex-1 glass-panel rounded-2xl relative overflow-hidden flex items-center justify-center p-8">
                    <div className="absolute inset-0 bg-grid-white/5 opacity-20" />
                    
                    <svg viewBox="0 0 200 200" className="w-full h-full max-w-[200px] drop-shadow-[0_0_20px_rgba(255,215,0,0.2)]">
                        {/* Base */}
                        <rect x="80" y="170" width="40" height="10" fill="#1E293B" />
                        {/* Arm Link 1 */}
                        <motion.g animate={{ rotate: robotState.joint1 }} style={{ originX: "100px", originY: "170px" }}>
                            <rect x="95" y="100" width="10" height="70" fill="#FFD700" />
                            {/* Joint 2 */}
                            <motion.g animate={{ rotate: robotState.joint2 }} style={{ originX: "100px", originY: "100px" }}>
                                <rect x="95" y="40" width="10" height="60" fill="#FFD700" />
                                {/* Gripper */}
                                <motion.g animate={{ rotate: robotState.joint3 }} style={{ originX: "100px", originY: "40px" }}>
                                    <path d="M90,20 L110,20 L105,40 L95,40 Z" fill="#475569" />
                                    <circle cx="100" cy="15" r="5" fill="#00FFC8" className="animate-pulse" />
                                </motion.g>
                            </motion.g>
                        </motion.g>
                    </svg>

                    {/* Telemetry Tags */}
                    <div className="absolute top-4 left-4 space-y-2">
                        <div className="text-[8px] neuro-text text-industrial-primary font-bold">J1: {robotState.joint1.toFixed(1)}°</div>
                        <div className="text-[8px] neuro-text text-industrial-primary font-bold">J2: {robotState.joint2.toFixed(1)}°</div>
                    </div>
                </div>

                {/* Real-time Load Cards */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-black/30 p-4 rounded-xl border border-white/5">
                        <div className="flex justify-between items-center mb-2">
                            <Zap size={14} className="text-yellow-500" />
                            <span className="text-[8px] neuro-text text-gray-500">CONSUMPTION</span>
                        </div>
                        <div className="text-lg font-mono font-bold text-white">{robotState.power.toFixed(2)} <span className="text-[10px] text-gray-600">kW</span></div>
                    </div>
                    <div className="bg-black/30 p-4 rounded-xl border border-white/5">
                        <div className="flex justify-between items-center mb-2">
                            <Activity size={14} className="text-neuro-pulse" />
                            <span className="text-[8px] neuro-text text-gray-500">EFFICIENCY</span>
                        </div>
                        <div className="text-lg font-mono font-bold text-white">99.4 <span className="text-[10px] text-gray-600">%</span></div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default RoboticsSim;
