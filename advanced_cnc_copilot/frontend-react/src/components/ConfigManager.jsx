import React, { useState } from 'react';
import { Key, Lock, FileJson, Save, RotateCcw } from 'lucide-react';

const ConfigManager = () => {
    const [params, setParams] = useState([
        { key: 'CORE_LATENCY_BIAS', value: 'LOW_LATENCY', locked: true },
        { key: 'MAX_SPINDLE_RPM', value: '12000', locked: false },
        { key: 'DEFAULT_MATERIAL', value: 'ALU_6061', locked: false },
        { key: 'LOG_RETENTION_DAYS', value: '30', locked: false },
        { key: 'REMOTE_ACCESS_PORT', value: '8080', locked: true },
        { key: 'NEURAL_WEIGHT_PATH', value: '/models/v4.2.bin', locked: true },
    ]);

    const handleEdit = (idx, newValue) => {
        if (params[idx].locked) return;
        const newParams = [...params];
        newParams[idx].value = newValue;
        setParams(newParams);
    };

    return (
        <div className="flex flex-col h-full">
            <div className="flex gap-2 mb-4">
                <div className="relative flex-1">
                    <input 
                        type="text" 
                        placeholder="Search parameters..." 
                        className="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-xs text-gray-300 focus:border-industrial-primary outline-none"
                    />
                </div>
                <button className="p-2 bg-white/5 hover:bg-white/10 rounded-lg border border-white/5 text-gray-400 hover:text-white transition-colors">
                    <FileJson size={14} />
                </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2 pr-2 scrollbar-thin scrollbar-thumb-white/10">
                {params.map((param, idx) => (
                    <div key={idx} className="group flex items-center justify-between p-3 bg-white/5 hover:bg-white/10 rounded border border-white/5 hover:border-white/10 transition-colors">
                        <div className="flex items-center gap-3">
                            {param.locked ? <Lock size={12} className="text-gray-600" /> : <Key size={12} className="text-industrial-primary" />}
                            <div>
                                <div className="text-[9px] neuro-text text-gray-500">{param.key}</div>
                                {param.locked ? (
                                    <div className="text-xs font-mono text-gray-400 mt-0.5">{param.value}</div>
                                ) : (
                                    <input 
                                        type="text" 
                                        value={param.value}
                                        onChange={(e) => handleEdit(idx, e.target.value)}
                                        className="bg-transparent border-b border-transparent hover:border-gray-500 focus:border-neuro-pulse outline-none text-xs font-mono text-white mt-0.5 w-full"
                                    />
                                )}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            <div className="mt-4 pt-4 border-t border-white/5 flex justify-end gap-3">
                <button className="px-4 py-2 rounded text-[10px] font-bold text-gray-400 hover:text-white flex items-center gap-2 transition-colors">
                    <RotateCcw size={12} /> REVERT
                </button>
                <button className="px-6 py-2 bg-neuro-pulse text-black rounded text-[10px] font-bold hover:bg-neuro-success transition-colors flex items-center gap-2 shadow-[0_0_15px_rgba(0,255,200,0.2)]">
                    <Save size={12} /> COMMIT CHANGES
                </button>
            </div>
        </div>
    );
};

export default ConfigManager;