import React, { useState, useEffect } from 'react';
import { Palette, Box, Share2, Award, Download, Signal, Zap } from 'lucide-react';
import axios from 'axios';

const CreativeTwinPanel = () => {
    const [assets, setAssets] = useState([]);
    const [hallOfFame, setHallOfFame] = useState([]);
    const [loading, setLoading] = useState(true);
    const [liveData, setLiveData] = useState({
        viewport_fps: 60,
        active_mesh: "Engine_Bracket_V2_Top",
        tris: 42082,
        render_engine: "Cycles (OptiX)",
        thermal_flux: "2140 W",
        voxel_risk: "0.12"
    });

    useEffect(() => {
        const handleNewPayload = (e) => {
            // Update metrics when a new protocol is generated
            const emotion = e.detail.detected_emotion;
            setLiveData(prev => ({
                ...prev,
                thermal_flux: emotion === "AGGRESSIVE" ? "4500 W" : "1200 W", // Mocking based to emotion
                voxel_risk: emotion === "AGGRESSIVE" ? "0.78" : "0.05"
            }));
        };
        window.addEventListener('new-synaptic-payload', handleNewPayload);
        return () => window.removeEventListener('new-synaptic-payload', handleNewPayload);
    }, []);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [assetRes, hofRes] = await Promise.all([
                    axios.get('/api/blender/assets'),
                    axios.get('/api/blender/hall-of-fame')
                ]);
                setAssets(assetRes.data.assets || []);
                setHallOfFame(hofRes.data.creations || []);
            } catch (e) {
                console.error("Resource Sync Failed", e);
            } finally {
                setLoading(false);
            }
        };
        fetchData();

        const interval = setInterval(() => {
            setLiveData(prev => ({
                ...prev,
                viewport_fps: 58 + Math.floor(Math.random() * 5)
            }));
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="flex flex-col gap-8 h-full">
            <div className="flex items-center gap-3 border-b border-white/5 pb-4 neuro-text text-gray-400">
                <Palette size={20} className="text-industrial-primary" />
                <span>CREATIVE TWIN // RESOURCE & DATA HUB</span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
                {/* Live Link Viewport Monitor */}
                <div className="lg:col-span-2 glass-panel p-6 rounded-xl border border-white/5 bg-gradient-to-br from-industrial-surface to-black/40">
                    <div className="flex items-center gap-2 mb-6 neuro-text text-[10px] text-neuro-success">
                        <Signal size={16} className="animate-pulse" />
                        <span>LIVE VIEWPORT TELEMETRY</span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                        {[
                            { label: 'FPS', value: liveData.viewport_fps, color: 'text-neuro-success' },
                            { label: 'ACTIVE MESH', value: liveData.active_mesh, color: 'text-white' },
                            { label: 'GEOMETRY', value: `${liveData.tris.toLocaleString()} Tris`, color: 'text-white' },
                            { label: 'ENGINE', value: liveData.render_engine, color: 'text-neuro-pulse' },
                            // New Deep Thought Metrics
                            { label: 'THERMAL FLUX', value: liveData.thermal_flux || "0 W", color: 'text-orange-400' },
                            { label: 'VOXEL RISK', value: liveData.voxel_risk || "0.00", color: 'text-red-400' }
                        ].map((item, i) => (
                            <div key={i} className="flex flex-col">
                                <span className="text-[9px] neuro-text text-gray-500 mb-1">{item.label}</span>
                                <span className={`text-sm font-bold font-mono ${item.color}`}>{item.value}</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Asset Gallery */}
                <div className="glass-panel p-6 rounded-xl border border-white/5 flex flex-col min-h-0">
                    <div className="flex items-center gap-2 mb-4 neuro-text text-[10px] text-gray-400">
                        <Box size={16} />
                        <span>PROCEDURAL ASSET LIBRARY</span>
                    </div>
                    <div className="grid grid-cols-2 gap-4 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-white/10">
                        {assets.map(asset => (
                            <div key={asset.id} className="group bg-black/20 rounded-lg overflow-hidden border border-white/5 hover:border-neuro-pulse transition-all hover:-translate-y-1">
                                <div className="h-20 bg-black flex items-center justify-center">
                                    <Box size={32} className="text-gray-800 group-hover:text-neuro-pulse transition-colors" />
                                </div>
                                <div className="p-3">
                                    <div className="text-[10px] font-bold text-gray-200 mb-1 truncate">{asset.name}</div>
                                    <div className="text-[8px] neuro-text text-gray-600 mb-2">{asset.type} • {asset.complexity}</div>
                                    <button className="w-full flex items-center justify-center gap-2 py-1.5 bg-white/5 hover:bg-neuro-pulse hover:text-black rounded text-[9px] neuro-text transition-all">
                                        <Download size={10} /> INJECT
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Hall of Fame */}
                <div className="glass-panel p-6 rounded-xl border border-white/5 flex flex-col min-h-0">
                    <div className="flex items-center gap-2 mb-4 neuro-text text-[10px] text-yellow-500">
                        <Award size={16} />
                        <span>BEST CREATIONS (HALL OF FAME)</span>
                    </div>
                    <div className="flex-1 overflow-y-auto pr-2 space-y-3 scrollbar-thin scrollbar-thumb-white/10">
                        {hallOfFame.length === 0 ? (
                            <div className="text-center py-12 border border-dashed border-white/5 rounded-lg text-[10px] text-gray-600 italic">
                                No shared masterpieces yet.
                            </div>
                        ) : (
                            hallOfFame.map(creation => (
                                <div key={creation.id} className="bg-black/20 p-3 rounded-lg border border-white/5 flex items-center gap-4 hover:bg-white/5 transition-colors">
                                    <Zap size={14} className="text-yellow-500" />
                                    <div className="flex-1 min-w-0">
                                        <div className="text-[10px] font-bold text-gray-200 truncate">{creation.name}</div>
                                        <div className="text-[8px] neuro-text text-gray-600">ROI: {creation.roi_score}% • {creation.author}</div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                    <button className="mt-6 w-full flex items-center justify-center gap-2 py-3 border border-yellow-500/50 text-yellow-500 hover:bg-yellow-500 hover:text-black rounded-lg text-[10px] neuro-text font-bold transition-all shadow-[0_0_15px_rgba(234,179,8,0.1)] hover:shadow-[0_0_25px_rgba(234,179,8,0.3)]">
                        <Share2 size={14} /> SHARE CURRENT WORKSPACE
                    </button>
                </div>
            </div>
        </div>
    );
};

export default CreativeTwinPanel;
