import React, { useState } from 'react';
import { ToggleLeft, ToggleRight, Sliders, Zap, Shield, Eye } from 'lucide-react';

const MasterPreferences = () => {
    const [prefs, setPrefs] = useState({
        autoSeal: true,
        highPerformance: false,
        debugOverlay: true,
        safetyOverride: false,
        aiSensitivity: 75
    });

    const toggle = (key) => {
        setPrefs(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const handleSlider = (key, val) => {
        setPrefs(prev => ({ ...prev, [key]: val }));
    };

    return (
        <div className="flex flex-col gap-6">
            <div className="space-y-4">
                <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                    <div className="flex items-center gap-3">
                        <Shield size={16} className={prefs.autoSeal ? 'text-neuro-success' : 'text-gray-500'} />
                        <div>
                            <div className="text-[10px] font-bold text-gray-200">AUTO-SEAL PROTOCOL</div>
                            <div className="text-[8px] neuro-text text-gray-500">Lock down system on threat detection</div>
                        </div>
                    </div>
                    <button onClick={() => toggle('autoSeal')} className="text-gray-400 hover:text-white transition-colors">
                        {prefs.autoSeal ? <ToggleRight size={24} className="text-neuro-success" /> : <ToggleLeft size={24} />}
                    </button>
                </div>

                <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                    <div className="flex items-center gap-3">
                        <Zap size={16} className={prefs.highPerformance ? 'text-neuro-danger' : 'text-gray-500'} />
                        <div>
                            <div className="text-[10px] font-bold text-gray-200">OVERCLOCK MODE</div>
                            <div className="text-[8px] neuro-text text-gray-500">Bypass safety limits for speed</div>
                        </div>
                    </div>
                    <button onClick={() => toggle('highPerformance')} className="text-gray-400 hover:text-white transition-colors">
                        {prefs.highPerformance ? <ToggleRight size={24} className="text-neuro-danger" /> : <ToggleLeft size={24} />}
                    </button>
                </div>

                <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                    <div className="flex items-center gap-3">
                        <Eye size={16} className={prefs.debugOverlay ? 'text-industrial-primary' : 'text-gray-500'} />
                        <div>
                            <div className="text-[10px] font-bold text-gray-200">AR DEBUG OVERLAY</div>
                            <div className="text-[8px] neuro-text text-gray-500">Show telemetry on video feed</div>
                        </div>
                    </div>
                    <button onClick={() => toggle('debugOverlay')} className="text-gray-400 hover:text-white transition-colors">
                        {prefs.debugOverlay ? <ToggleRight size={24} className="text-industrial-primary" /> : <ToggleLeft size={24} />}
                    </button>
                </div>
            </div>

            <div className="p-4 bg-black/20 rounded-lg border border-white/5">
                <div className="flex justify-between items-center mb-3">
                    <span className="text-[10px] font-bold text-gray-300 flex items-center gap-2"><Sliders size={12} /> AI SENSITIVITY</span>
                    <span className="text-[10px] font-mono text-neuro-pulse">{prefs.aiSensitivity}%</span>
                </div>
                <input 
                    type="range" 
                    min="0" 
                    max="100" 
                    value={prefs.aiSensitivity} 
                    onChange={(e) => handleSlider('aiSensitivity', e.target.value)}
                    className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-neuro-pulse"
                />
                <div className="flex justify-between mt-2 text-[8px] neuro-text text-gray-600">
                    <span>PASSIVE</span>
                    <span>AGGRESSIVE</span>
                </div>
            </div>
        </div>
    );
};

export default MasterPreferences;