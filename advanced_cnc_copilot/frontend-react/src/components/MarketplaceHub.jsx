import React, { useState, useEffect } from 'react';
import { ShoppingBag, Search, Download, Star, Filter, Share2, Package, ShieldCheck, Trophy } from 'lucide-react';
import axios from 'axios';

const MarketplaceHub = () => {
    const [components, setComponents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filter, setCategoryFilter] = useState('ALL');

    useEffect(() => {
        const fetchComponents = async () => {
            setLoading(true);
            try {
                const categoryParam = filter === 'ALL' ? '' : `?category=${filter}`;
                const res = await axios.get(`/api/marketplace/components${categoryParam}`);
                setComponents(res.data.components || []);
            } catch (e) {
                console.error("Marketplace sync failed", e);
            } finally {
                setLoading(false);
            }
        };
        fetchComponents();
    }, [filter]);

    const handleDownload = async (id) => {
        try {
            const res = await axios.post(`/api/marketplace/download/${id}`);
            alert(`Downloaded payload for ${id}. Logic would now inject this into your local library.`);
        } catch (e) {
            alert("Download failed.");
        }
    };

    return (
        <div className="flex flex-col h-full gap-6">
            <header className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                    <ShoppingBag className="text-neuro-pulse" />
                    <h2 className="neuro-text text-white">ECOSYSTEM MARKETPLACE</h2>
                </div>
                <div className="flex gap-2">
                    {['ALL', 'GCODE', 'MATERIAL', 'CONFIG'].map(cat => (
                        <button
                            key={cat}
                            onClick={() => setCategoryFilter(cat)}
                            className={`px-3 py-1 rounded text-[10px] neuro-text transition-all ${filter === cat ? 'bg-neuro-pulse text-black font-bold' : 'bg-white/5 text-gray-500 hover:text-white'}`}
                        >
                            {cat}
                        </button>
                    ))}
                </div>
            </header>

            <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-600" size={14} />
                <input
                    type="text"
                    placeholder="Search the hive community..."
                    className="w-full bg-black/20 border border-white/5 rounded-lg pl-10 pr-4 py-2 text-xs text-gray-300 focus:border-neuro-pulse outline-none"
                />
            </div>

            <div className="flex-1 overflow-y-auto pr-2 space-y-4 scrollbar-thin scrollbar-thumb-white/10">
                {loading ? (
                    <div className="text-center py-20 neuro-text text-gray-600 animate-pulse italic">SYNCING WITH HIVE NODES...</div>
                ) : components.length === 0 ? (
                    <div className="text-center py-20 border border-dashed border-white/5 rounded-xl text-gray-600 italic text-xs">
                        No components found in this sector.
                    </div>
                ) : (
                    components.map(comp => (
                        <div key={comp.id} className="glass-panel p-4 rounded-xl border border-white/5 hover:border-white/10 transition-all group">
                            <div className="flex justify-between items-start mb-3">
                                <div>
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="text-[8px] bg-white/5 text-industrial-primary px-1.5 py-0.5 rounded border border-industrial-primary/20">{comp.category}</span>
                                        <h3 className="text-sm font-bold text-white group-hover:text-neuro-pulse transition-colors">{comp.name}</h3>
                                        {comp.stress_tested && (
                                            <span className="flex items-center gap-1 text-[8px] bg-neuro-success/10 text-neuro-success px-1.5 py-0.5 rounded border border-neuro-success/20 font-bold uppercase tracking-tighter shadow-[0_0_10px_rgba(34,197,94,0.1)]">
                                                <ShieldCheck size={10} /> Survivor
                                            </span>
                                        )}
                                    </div>
                                    <p className="text-[10px] text-gray-500 line-clamp-1">{comp.description}</p>
                                </div>
                                <div className="text-right">
                                    <div className="flex items-center justify-end gap-1 text-yellow-500 text-[10px] font-mono">
                                        <Star size={10} fill="currentColor" /> {comp.rating.toFixed(1)}
                                    </div>
                                    <div className="flex items-center justify-end gap-1 text-neuro-pulse text-[10px] font-mono mt-0.5">
                                        <Trophy size={10} /> {Math.round(comp.success_score)} pts
                                    </div>
                                    <div className="text-[8px] text-gray-600 mt-1 uppercase">v{comp.version}</div>
                                </div>
                            </div>

                            <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/5">
                                <div className="text-[9px] neuro-text text-gray-600 italic">
                                    Author: <span className="text-gray-400">{comp.author}</span>
                                </div>
                                <div className="flex gap-3">
                                    <div className="flex items-center gap-1 text-[9px] text-gray-500">
                                        <Download size={10} /> {comp.downloads}
                                    </div>
                                    <button
                                        onClick={() => handleDownload(comp.id)}
                                        className="bg-neuro-pulse/10 hover:bg-neuro-pulse text-neuro-pulse hover:text-black p-1.5 rounded transition-all"
                                    >
                                        <Download size={14} />
                                    </button>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>

            <button className="w-full py-3 bg-industrial-primary text-black font-bold rounded-xl text-[10px] neuro-text flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-[0.98] transition-all shadow-[0_0_20px_rgba(255,215,0,0.1)]">
                <Share2 size={14} /> SHARE YOUR CONFIGURATION
            </button>
        </div>
    );
};

export default MarketplaceHub;
