import React, { useState, useEffect } from 'react';
import { Cpu, HardDrive, Network, Thermometer, Activity } from 'lucide-react';

const SystemHealth = () => {
    // Mock Data simulating HAL (Hardware Abstraction Layer)
    const [stats, setStats] = useState({
        cpu_load: 12,
        memory_usage: 4.2, // GB
        disk_io: 45, // MB/s
        net_latency: 14, // ms
        temp: 42 // Celsius
    });

    useEffect(() => {
        const interval = setInterval(() => {
            setStats(prev => ({
                cpu_load: Math.min(100, Math.max(0, prev.cpu_load + (Math.random() * 10 - 5))),
                memory_usage: Math.min(16, Math.max(2, prev.memory_usage + (Math.random() * 0.2 - 0.1))),
                disk_io: Math.max(0, prev.disk_io + (Math.random() * 20 - 10)),
                net_latency: Math.max(1, prev.net_latency + (Math.random() * 4 - 2)),
                temp: Math.min(90, Math.max(30, prev.temp + (Math.random() * 2 - 1)))
            }));
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    const StatusIndicator = ({ val, limit }) => (
        <div className={`w-2 h-2 rounded-full ${val > limit ? 'bg-neuro-danger animate-pulse' : 'bg-neuro-success'}`} />
    );

    return (
        <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-black/30 p-3 rounded-lg border border-white/5 flex items-center gap-3">
                    <Cpu size={18} className="text-industrial-primary" />
                    <div>
                        <div className="text-[9px] neuro-text text-gray-500">CORE LOAD</div>
                        <div className="text-sm font-mono font-bold text-white flex items-center gap-2">
                            {stats.cpu_load.toFixed(1)}% <StatusIndicator val={stats.cpu_load} limit={80} />
                        </div>
                    </div>
                </div>
                <div className="bg-black/30 p-3 rounded-lg border border-white/5 flex items-center gap-3">
                    <HardDrive size={18} className="text-industrial-primary" />
                    <div>
                        <div className="text-[9px] neuro-text text-gray-500">MEMORY</div>
                        <div className="text-sm font-mono font-bold text-white flex items-center gap-2">
                            {stats.memory_usage.toFixed(1)} GB <StatusIndicator val={stats.memory_usage} limit={14} />
                        </div>
                    </div>
                </div>
                <div className="bg-black/30 p-3 rounded-lg border border-white/5 flex items-center gap-3">
                    <Network size={18} className="text-industrial-primary" />
                    <div>
                        <div className="text-[9px] neuro-text text-gray-500">LATENCY</div>
                        <div className="text-sm font-mono font-bold text-white flex items-center gap-2">
                            {stats.net_latency.toFixed(0)} ms <StatusIndicator val={stats.net_latency} limit={50} />
                        </div>
                    </div>
                </div>
                <div className="bg-black/30 p-3 rounded-lg border border-white/5 flex items-center gap-3">
                    <Thermometer size={18} className="text-industrial-primary" />
                    <div>
                        <div className="text-[9px] neuro-text text-gray-500">THERMAL</div>
                        <div className="text-sm font-mono font-bold text-white flex items-center gap-2">
                            {stats.temp.toFixed(1)}°C <StatusIndicator val={stats.temp} limit={75} />
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-black/30 p-4 rounded-lg border border-white/5 h-[150px] relative overflow-hidden flex flex-col">
                <div className="text-[9px] neuro-text text-gray-500 mb-2 flex items-center gap-2">
                    <Activity size={12} /> REAL-TIME BUS ACTIVITY
                </div>
                <div className="flex-1 flex items-end gap-1">
                    {[...Array(40)].map((_, i) => {
                        const h = 20 + Math.random() * 60;
                        return (
                            <div 
                                key={i} 
                                className="flex-1 bg-neuro-pulse/20 hover:bg-neuro-pulse transition-colors" 
                                style={{ height: `${h}%` }}
                            />
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

export default SystemHealth;