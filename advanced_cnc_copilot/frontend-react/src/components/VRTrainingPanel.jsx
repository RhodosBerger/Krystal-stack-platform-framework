import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Box, Glasses, Play, CheckCircle, AlertCircle, Info } from 'lucide-react';

const VRTrainingPanel = () => {
    const [step, setStep] = useState(0);
    const [trainingActive, setTrainingActive] = useState(false);

    const scenarioSteps = [
        { title: "Tool Offset Calibration", desc: "Align the touch probe with the workpiece zero point.", action: "ZERO_PROBE" },
        { title: "Spindle Warm-up", desc: "Run spindle at 2000 RPM for 2 minutes to distribute lubricant.", action: "WARMUP" },
        { title: "Material Check", desc: "Verify that Titanium Stock is securely clamped in the 4th axis.", action: "VERIFY_CLAMP" },
        { title: "Emergency Stop Test", desc: "Simulate a manual E-Stop to verify logic relay status.", action: "TEST_ESTOP" }
    ];

    const startTraining = () => {
        setTrainingActive(true);
        setStep(0);
    };

    return (
        <div className="flex flex-col h-full gap-6">
            <header className="flex justify-between items-center border-b border-white/5 pb-4">
                <div className="flex items-center gap-3">
                    <Glasses className="text-neuro-pulse" />
                    <h2 className="neuro-text text-white font-bold">IMMERSIVE VR TRAINING</h2>
                </div>
                {!trainingActive ? (
                    <button 
                        onClick={startTraining}
                        className="bg-neuro-pulse text-black px-4 py-1.5 rounded-lg text-[10px] neuro-text font-bold hover:scale-105 transition-all"
                    >
                        ENTER WEBXR
                    </button>
                ) : (
                    <div className="flex items-center gap-2 text-neuro-success text-[10px] neuro-text">
                        <div className="w-2 h-2 rounded-full bg-neuro-success animate-ping" />
                        UPLINK ACTIVE
                    </div>
                )}
            </header>

            {!trainingActive ? (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-8 bg-black/40 rounded-2xl border border-dashed border-white/10">
                    <div className="w-20 h-20 bg-neuro-pulse/10 rounded-full flex items-center justify-center mb-6">
                        <Glasses size={40} className="text-neuro-pulse" />
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2 font-mono uppercase">Prepare for Immersion</h3>
                    <p className="text-gray-500 text-sm max-w-sm mb-8">
                        The VR Studio allows you to practice complex machining protocols in a risk-free digital twin environment.
                    </p>
                    <div className="grid grid-cols-2 gap-4 w-full">
                        <div className="bg-white/5 p-4 rounded-xl border border-white/5 text-left">
                            <Info size={16} className="text-industrial-primary mb-2" />
                            <div className="text-[10px] font-bold text-gray-300 uppercase">Latency</div>
                            <div className="text-lg font-mono text-white">4.2ms</div>
                        </div>
                        <div className="bg-white/5 p-4 rounded-xl border border-white/5 text-left">
                            <CheckCircle size={16} className="text-neuro-success mb-2" />
                            <div className="text-[10px] font-bold text-gray-300 uppercase">Physics</div>
                            <div className="text-lg font-mono text-white">Real-Time</div>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="flex-1 flex flex-col gap-6 relative">
                    {/* Simulated VR Viewport */}
                    <div className="flex-1 bg-gradient-to-br from-gray-900 to-black rounded-2xl border border-neuro-pulse/30 relative overflow-hidden group">
                        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] opacity-20" />
                        
                        {/* 3D Placeholder - Machine Head */}
                        <motion.div 
                            animate={{ 
                                y: step * -20,
                                rotate: [0, 1, -1, 0]
                            }}
                            transition={{ duration: 2, repeat: Infinity }}
                            className="absolute inset-0 flex items-center justify-center pointer-events-none"
                        >
                            <div className="w-48 h-48 border-2 border-neuro-pulse/20 rounded-full flex items-center justify-center">
                                <div className="w-32 h-32 border-4 border-neuro-pulse animate-spin-slow rounded-full opacity-40 shadow-[0_0_30px_#00FFC8]" />
                                <div className="absolute w-1 h-24 bg-neuro-pulse top-0" />
                            </div>
                        </motion.div>

                        {/* HUD Overlay */}
                        <div className="absolute top-6 left-6 flex flex-col gap-2">
                            <div className="bg-black/60 backdrop-blur px-3 py-1.5 rounded-lg border border-white/10">
                                <div className="text-[8px] neuro-text text-gray-500">OBJECTIVE</div>
                                <div className="text-xs font-bold text-neuro-pulse">{scenarioSteps[step].title}</div>
                            </div>
                        </div>

                        <div className="absolute bottom-6 right-6">
                            <div className="bg-neuro-pulse/10 backdrop-blur px-4 py-2 rounded-xl border border-neuro-pulse/40 flex items-center gap-3">
                                <span className="text-[10px] neuro-text text-neuro-pulse font-bold">READY TO COMMIT</span>
                                <button 
                                    onClick={() => step < scenarioSteps.length - 1 ? setStep(s => s + 1) : setTrainingActive(false)}
                                    className="bg-neuro-pulse text-black p-2 rounded-full hover:scale-110 active:scale-95 transition-all"
                                >
                                    <Play size={16} fill="currentColor" />
                                </button>
                            </div>
                        </div>

                        {/* Safety Warning */}
                        <div className="absolute top-6 right-6">
                            <div className="flex items-center gap-2 bg-neuro-danger/20 px-3 py-1 rounded text-[8px] neuro-text text-neuro-danger border border-neuro-danger/30">
                                <AlertCircle size={10} /> BOUNDARY WARNING
                            </div>
                        </div>
                    </div>

                    {/* Step List */}
                    <div className="flex gap-4">
                        {scenarioSteps.map((s, i) => (
                            <div key={i} className={`flex-1 h-1 rounded-full transition-all duration-500 ${i <= step ? 'bg-neuro-pulse shadow-[0_0_10px_#00FFC8]' : 'bg-white/10'}`} />
                        ))}
                    </div>
                    <div className="p-4 glass-panel rounded-xl">
                        <p className="text-gray-400 text-xs leading-relaxed italic">
                            "{scenarioSteps[step].desc}"
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default VRTrainingPanel;
