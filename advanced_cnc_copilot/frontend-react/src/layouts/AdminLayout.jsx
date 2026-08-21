import React from 'react';
import DebugConsole from '../components/DebugConsole';
import SystemHealth from '../components/SystemHealth';
import MasterPreferences from '../components/MasterPreferences';
import ConfigManager from '../components/ConfigManager';
import { Terminal, Settings, ShieldAlert, Cpu } from 'lucide-react';

const AdminLayout = () => {
  return (
    <div className="p-8 max-w-[1800px] mx-auto min-h-screen bg-black flex flex-col">
      <header className="mb-12 flex justify-between items-center border-b border-neuro-pulse/20 pb-8">
        <div>
          <h1 className="text-2xl font-mono font-bold text-neuro-pulse mb-2 flex items-center gap-3">
            <Terminal className="text-neuro-pulse" />
            SYSTEM_ADMIN <span className="text-gray-700 font-light">::</span> GEEK_MODE
          </h1>
          <p className="font-mono text-[9px] text-gray-500 uppercase tracking-[0.2em]">Root Access Enabled - High-Level System Manipulation</p>
        </div>
        <div className="flex gap-4">
           <div className="px-4 py-2 border border-neuro-pulse/30 rounded font-mono text-[10px] text-neuro-pulse bg-neuro-pulse/5">
              ENCRYPTION: AES-256-GCM
           </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 flex-1">
        {/* Left: System Health & Preferences */}
        <div className="lg:col-span-4 flex flex-col gap-8">
           <div className="glass-panel p-6 rounded border border-white/5 bg-black/40">
              <h3 className="font-mono text-[10px] text-neuro-pulse mb-6 flex items-center gap-2 uppercase"><Cpu size={14} /> Hardware_Abstraction_Layer</h3>
              <SystemHealth />
           </div>
           <div className="glass-panel p-6 rounded border border-white/5 bg-black/40 flex-1">
              <h3 className="font-mono text-[10px] text-gray-500 mb-6 flex items-center gap-2 uppercase"><Settings size={14} /> Global_Override_Prefs</h3>
              <MasterPreferences />
           </div>
        </div>

        {/* Center: Debug Console */}
        <div className="lg:col-span-5 rounded border border-neuro-pulse/20 overflow-hidden flex flex-col">
           <DebugConsole />
        </div>

        {/* Right: Config Manager */}
        <div className="lg:col-span-3 flex flex-col gap-8">
           <div className="glass-panel p-6 rounded border border-white/5 bg-black/40 flex-1">
              <h3 className="font-mono text-[10px] text-gray-500 mb-6 flex items-center gap-2 uppercase"><ShieldAlert size={14} /> Security_Parameter_Matrix</h3>
              <ConfigManager />
           </div>
        </div>
      </div>
    </div>
  );
};

export default AdminLayout;
