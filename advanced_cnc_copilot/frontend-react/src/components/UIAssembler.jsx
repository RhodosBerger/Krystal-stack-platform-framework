import React, { useState } from 'react';
import { Layers, MousePointer2, Box, PenTool, Maximize2, Palette, Save, Plus } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const UIAssembler = () => {
    const [assembly, setAssembly] = useState({
        name: "MASTER_DASHBOARD_V1",
        parts: [],
        mates: []
    });
    const [selectedPart, setSelectedPart] = useState(null);

    const addPart = (type) => {
        const newPart = {
            id: `PART-${Math.random().toString(36).substr(2, 4).toUpperCase()}`,
            type: type,
            material: 'Aluminum',
            dimensions: { w: 300, h: 200 },
            pos: { x: 50, y: 50 }
        };
        setAssembly({ ...assembly, parts: [...assembly.parts, newPart] });
        setSelectedPart(newPart.id);
    };

    const updateMaterial = (material) => {
        setAssembly({
            ...assembly,
            parts: assembly.parts.map(p => p.id === selectedPart ? { ...p, material } : p)
        });
    };

    return (
        <div className="flex flex-col h-full gap-4 text-gray-300 font-mono text-[11px]">
            {/* Header / Command Bar */}
            <div className="flex justify-between items-center bg-white/5 p-3 rounded-lg border border-white/5">
                <div className="flex items-center gap-3">
                    <Layers size={16} className="text-neuro-pulse" />
                    <span className="font-bold tracking-widest text-white uppercase">{assembly.name} <span className="text-gray-600">[ASSEMBLY]</span></span>
                </div>
                <div className="flex gap-2">
                    <button className="flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 rounded transition-all border border-white/5">
                        <Save size={12} /> REBUILD
                    </button>
                    <button className="flex items-center gap-2 px-3 py-1.5 bg-neuro-pulse text-black font-bold rounded hover:scale-105 transition-all">
                        <Save size={12} /> EXPORT SOURCE
                    </button>
                </div>
            </div>

            <div className="flex-1 grid grid-cols-12 gap-4 min-h-0">
                {/* 1. Feature Tree (Left Sidebar) */}
                <div className="col-span-3 bg-black/20 rounded-xl border border-white/5 p-4 flex flex-col overflow-hidden">
                    <div className="flex items-center gap-2 mb-4 text-gray-500 uppercase border-b border-white/5 pb-2">
                        <PenTool size={12} /> Feature Tree
                    </div>
                    <div className="flex-1 overflow-y-auto space-y-1 pr-2 scrollbar-thin scrollbar-thumb-white/5">
                        {assembly.parts.map(part => (
                            <div
                                key={part.id}
                                onClick={() => setSelectedPart(part.id)}
                                className={`flex items-center justify-between p-2 rounded cursor-pointer transition-all ${selectedPart === part.id ? 'bg-neuro-pulse/20 border-l-2 border-neuro-pulse text-white' : 'hover:bg-white/5 text-gray-500'}`}
                            >
                                <div className="flex items-center gap-2">
                                    <Box size={10} />
                                    <span>{part.id} ({part.type})</span>
                                </div>
                            </div>
                        ))}
                        <button
                            onClick={() => addPart('GAUGE')}
                            className="w-full mt-4 p-2 border border-dashed border-white/10 rounded text-gray-600 hover:text-white hover:border-white/20 transition-all flex items-center justify-center gap-2"
                        >
                            <Plus size={12} /> INSERT COMPONENT
                        </button>
                    </div>
                </div>

                {/* 2. Work Plane (Center Canvas) */}
                <div className="col-span-6 bg-[#0a0a0a] rounded-xl border border-white/5 relative overflow-hidden group">
                    <div className="absolute inset-0 opacity-20 pointer-events-none"
                        style={{ backgroundImage: 'radial-gradient(circle, #333 1px, transparent 1px)', backgroundSize: '20px 20px' }} />

                    <AnimatePresence>
                        {assembly.parts.map(part => (
                            <motion.div
                                key={part.id}
                                layoutId={part.id}
                                onClick={() => setSelectedPart(part.id)}
                                className={`absolute rounded-lg border flex flex-col items-center justify-center shadow-2xl cursor-move transition-all ${selectedPart === part.id ? 'border-neuro-pulse ring-2 ring-neuro-pulse/20' : 'border-white/10'}`}
                                style={{
                                    width: part.dimensions.w / 2,
                                    height: part.dimensions.h / 2,
                                    left: part.pos.x,
                                    top: part.pos.y,
                                    background: part.material === 'Aluminum' ? 'linear-gradient(145deg, #222, #111)' : 'rgba(0,255,136,0.05)',
                                    backdropFilter: part.material === 'Neuro-Glass' ? 'blur(10px)' : 'none'
                                }}
                            >
                                <div className="text-[8px] text-gray-600 mb-1">{part.id}</div>
                                <div className={`font-bold ${selectedPart === part.id ? 'text-neuro-pulse' : 'text-gray-500'}`}>{part.type}</div>
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {assembly.parts.length === 0 && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-700 italic">
                            <MousePointer2 size={32} className="mb-4 opacity-20" />
                            <span>Select a Plane (Sketch) to begin extrusion...</span>
                        </div>
                    )}
                </div>

                {/* 3. Property Manager (Right Sidebar) */}
                <div className="col-span-3 bg-black/20 rounded-xl border border-white/5 p-4 flex flex-col">
                    <div className="flex items-center gap-2 mb-4 text-gray-500 uppercase border-b border-white/5 pb-2">
                        <Maximize2 size={12} /> Property Manager
                    </div>
                    {selectedPart ? (
                        <div className="space-y-4">
                            <div>
                                <label className="text-gray-600 block mb-2">MATERIAL / THEME</label>
                                <div className="grid grid-cols-2 gap-2">
                                    {['Aluminum', 'Neuro-Glass', 'Carbon', 'Emerald'].map(mat => (
                                        <button
                                            key={mat}
                                            onClick={() => updateMaterial(mat)}
                                            className={`p-2 rounded border transition-all ${assembly.parts.find(p => p.id === selectedPart)?.material === mat ? 'bg-neuro-pulse text-black font-bold' : 'bg-white/5 border-white/5 hover:bg-white/10'}`}
                                        >
                                            {mat}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div>
                                <label className="text-gray-600 block mb-2">DIMENSIONS (SKETCH)</label>
                                <div className="flex gap-2">
                                    <div className="flex-1 p-2 bg-white/5 rounded border border-white/5 text-gray-400">W: 300px</div>
                                    <div className="flex-1 p-2 bg-white/5 rounded border border-white/5 text-gray-400">H: 200px</div>
                                </div>
                            </div>

                            <div className="p-3 bg-neuro-pulse/5 rounded-lg border border-neuro-pulse/20 mt-6">
                                <div className="flex items-center gap-2 text-neuro-pulse font-bold mb-1">
                                    <Palette size={12} /> CAD ADVICE
                                </div>
                                <p className="text-[10px] text-gray-500 italic leading-relaxed">
                                    Theory 6 suggests applying a 'Neuro-Glass' material to high-volatility metrics for better cognitive grouping.
                                </p>
                            </div>
                        </div>
                    ) : (
                        <div className="flex-1 flex items-center justify-center text-gray-700 text-center">
                            Select a "Part" from the tree or canvas to edit its features.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default UIAssembler;
