import React, { useState, useEffect } from 'react';
import { Share2, Server, Terminal, Zap, ShieldCheck, AlertCircle } from 'lucide-react';

const SwarmMap = () => {
    const [swarm, setSwarm] = useState({ nodes: [], nodeCount: 0 });
    const [selectedNode, setSelectedNode] = useState(null);

    useEffect(() => {
        const fetchSwarm = async () => {
            try {
                const res = await fetch('/api/swarm/status');
                const data = await res.json();
                const nodes = Object.entries(data.machines).map(([id, info]) => ({
                    id, ...info
                }));
                setSwarm({ nodes, nodeCount: data.node_count });
            } catch (e) {
                console.error("Swarm Fetch Failed", e);
            }
        };

        fetchSwarm();
        const interval = setInterval(fetchSwarm, 2000);
        return () => clearInterval(interval);
    }, []);

    const getStatusColor = (load) => {
        if (load > 85) return 'bg-neuro-danger shadow-[0_0_10px_#FF453A]'; 
        if (load > 60) return 'bg-yellow-500 shadow-[0_0_10px_#EAB308]'; 
        return 'bg-neuro-success shadow-[0_0_10px_#10B981]'; 
    };

    return (
        <div className="glass-panel p-6 rounded-xl flex flex-col h-full border border-white/5">
            <div className="flex justify-between items-center mb-6">
                <div className="flex items-center gap-2 neuro-text text-gray-400">
                    <Share2 size={18} className="text-neuro-pulse" />
                    <span>GLOBAL SWARM INTELLIGENCE</span>
                </div>
                <div className="text-[10px] neuro-text text-gray-600">
                    ACTIVE NODES: <span className="text-neuro-success">{swarm.nodeCount}</span>
                </div>
            </div>

            <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-3 flex-1 overflow-y-auto p-4 bg-black/20 rounded-lg border border-white/5 scrollbar-thin scrollbar-thumb-white/10">
                {swarm.nodes.map(node => (
                    <button
                        key={node.id}
                        onClick={() => setSelectedNode(node)}
                        className={`group relative w-full aspect-square bg-industrial-surface rounded-lg border flex flex-col items-center justify-center transition-all hover:scale-105 active:scale-95 ${
                            selectedNode?.id === node.id ? 'border-neuro-pulse' : 'border-white/5'
                        }`}
                    >
                        <div className={`w-2.5 h-2.5 rounded-full mb-1 transition-all ${getStatusColor(node.load)}`} />
                        <span className="text-[8px] font-mono text-gray-500 group-hover:text-gray-300">{node.id.split('-').pop()}</span>
                        {node.load > 90 && (
                            <div className="absolute -top-1 -right-1 bg-neuro-danger w-3 h-3 rounded-full border border-industrial-bg flex items-center justify-center">
                                <AlertCircle size={8} className="text-white" />
                            </div>
                        )}
                    </button>
                ))}
            </div>

            {selectedNode && (
                <div className="mt-6 p-4 glass-panel bg-neuro-pulse/5 border-neuro-pulse/30 rounded-xl animate-in fade-in slide-in-from-bottom-2">
                    <div className="flex justify-between items-center mb-3">
                        <strong className="text-sm font-mono text-white">{selectedNode.id}</strong>
                        <span className="neuro-text text-[10px] text-neuro-success flex items-center gap-1">
                            <ShieldCheck size={12} /> {selectedNode.status}
                        </span>
                    </div>
                    <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-[10px] neuro-text text-gray-400">
                        <div className="flex justify-between">LOAD: <span className={selectedNode.load > 85 ? 'text-neuro-danger' : 'text-neuro-success'}>{selectedNode.load.toFixed(1)}%</span></div>
                        <div className="flex justify-between">RPM: <span className="text-gray-200">{selectedNode.rpm}</span></div>
                        <div className="flex justify-between">VIB: <span className="text-gray-200">{selectedNode.vibration.toFixed(2)}g</span></div>
                        <div className="flex justify-between">ACTION: <span className="text-gray-200">{selectedNode.activity}</span></div>
                    </div>
                    {selectedNode.load > 80 && (
                        <div className="mt-3 p-2 bg-neuro-danger/10 border border-neuro-danger/30 rounded text-[9px] text-neuro-danger neuro-text leading-tight">
                            CRITICAL LOAD DETECTED. TASK STEALING INITIATED. OFF-LOADING TO IDLE NODE.
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default SwarmMap;
