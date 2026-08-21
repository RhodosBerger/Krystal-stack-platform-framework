import React, { useState } from 'react';
import {
  Settings,
  Cpu,
  Shield,
  Activity,
  Menu,
  Search,
  Bell,
  Users,
  Database,
  Terminal,
  Layers,
  Zap,
  Fingerprint,
  ShieldAlert,
  LayoutDashboard
} from 'lucide-react';
import { motion } from 'framer-motion';
import { usePersona } from '../context/PersonaContext';

// Import Panels
import SystemHealth from '../components/SystemHealth';
import DebugConsole from '../components/DebugConsole';
import ConfigManager from '../components/ConfigManager';
import AIConfigPanel from '../components/AIConfigPanel';
import IntegrationsPanel from '../components/IntegrationsPanel';
import MasterPreferences from '../components/MasterPreferences';
import DashboardBuilder from '../components/DashboardBuilder';

const ConfigurationLayout = () => {
  const { config } = usePersona();
  const [activeTab, setActiveTab] = useState('DASHBOARD');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const menuItems = [
    { id: 'DASHBOARD', icon: Activity, label: 'Shadow Council' },
    { id: 'DASHBOARD_BUILDER', icon: LayoutDashboard, label: 'UI Builder' },
    { id: 'AI_CONFIG', icon: Cpu, label: 'Intelligence' },
    { id: 'INTEGRATIONS', icon: Layers, label: 'FOCAS Bridge' },
    { id: 'PARAMETERS', icon: Database, label: 'JSON Evidence' },
    { id: 'PROVENANCE', icon: Fingerprint, label: 'Audit Chain' },
    { id: 'DEBUG', icon: Terminal, label: 'Root Access' },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'DASHBOARD_BUILDER':
        return (
          <div className="h-[calc(100vh-160px)]">
            <DashboardBuilder />
          </div>
        );
      case 'DASHBOARD':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-1">
            <div className="bg-black/40 border border-white/5 p-8" style={{ borderRadius: config.borderRadius }}>
              <h3 className="text-[10px] font-mono tracking-widest text-gray-400 mb-8 border-l-2 pl-4 uppercase" style={{ borderColor: config.primary }}>
                Shadow Council Pulse
              </h3>
              <div className="grid grid-cols-3 gap-4 mb-8">
                {['AUDITOR', 'VISION', 'DOPAMINE'].map(agent => (
                  <div key={agent} className="flex flex-col items-center gap-2 p-4 bg-white/5 rounded-lg border border-white/5 overflow-hidden relative group">
                    <motion.div
                      animate={{ opacity: [0.3, 0.6, 0.3], scale: [1, 1.1, 1] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: config.primary }}
                    />
                    <span className="text-[9px] font-mono text-white opacity-60 group-hover:opacity-100 transition-opacity">{agent}</span>
                    <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent pointer-events-none" />
                  </div>
                ))}
              </div>
              <SystemHealth />
            </div>
            <div className="bg-black/40 border border-white/5 p-8" style={{ borderRadius: config.borderRadius }}>
              <h3 className="text-[10px] font-mono tracking-widest text-gray-400 mb-8 border-l-2 pl-4 uppercase" style={{ borderColor: config.primary }}>
                Hardware Abstraction
              </h3>
              <MasterPreferences />
            </div>
          </div>
        );
      case 'AI_CONFIG':
        return <AIConfigPanel />;
      case 'INTEGRATIONS':
        return <IntegrationsPanel />;
      case 'PARAMETERS':
        return (
          <div className="bg-black/40 border border-white/5 p-8 h-full" style={{ borderRadius: config.borderRadius }}>
            <ConfigManager />
          </div>
        );
      case 'PROVENANCE':
        return (
          <div className="bg-black/40 border border-white/5 p-8 h-full" style={{ borderRadius: config.borderRadius }}>
            <h3 className="text-[10px] font-mono tracking-widest text-gray-400 mb-8 border-l-2 pl-4 uppercase" style={{ borderColor: config.primary }}>
              Immutable Audit Chain (SHA-256)
            </h3>
            <div className="space-y-4 max-h-[600px] overflow-y-auto pr-4 scrollbar-hide">
              {[
                { sig: "f7a2...bc10", job: "JOB_882", date: "2026-01-25 16:45", actor: "CORTEX_WORKER", reason: "Reflex FRO Adjustment" },
                { sig: "e291...33a1", job: "JOB_881", date: "2026-01-25 16:40", actor: "CORTEX_WORKER", reason: "Evolutionary Re-optimization" }
              ].map((entry, idx) => (
                <div key={idx} className="p-4 bg-white/5 rounded border border-white/10 flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] font-mono text-neuro-success font-bold tracking-widest">SIG: {entry.sig}</span>
                    <span className="text-[9px] font-mono text-gray-500 uppercase">{entry.date}</span>
                  </div>
                  <div className="text-[11px] font-bold text-white">ID: {entry.job} // {entry.reason}</div>
                  <div className="flex justify-between text-[9px] font-mono text-gray-400">
                    <span>ACTOR: {entry.actor}</span>
                    <span className="text-blue-400">CUSTODY_VERIFIED</span>
                  </div>
                </div>
              ))}
              <div className="text-center py-10 opacity-30 text-[10px] font-mono uppercase tracking-[0.5em]">
                End of Signed Ledger
              </div>
            </div>
          </div>
        );
      case 'DEBUG':
        return (
          <div className="bg-black/40 border border-white/5 overflow-hidden h-[calc(100vh-200px)]" style={{ borderRadius: config.borderRadius }}>
            <DebugConsole />
          </div>
        );
      default:
        return <div className="text-white">Select a module</div>;
    }
  };

  return (
    <div className="min-h-screen bg-[#0a050f] text-gray-200 flex font-sans selection:bg-neuro-pulse selection:text-black">

      {/* SIDEBAR (Neural Purple Theme) */}
      <motion.aside
        initial={false}
        animate={{ width: sidebarOpen ? 260 : 80 }}
        className="h-screen bg-black/60 backdrop-blur-3xl border-r border-white/5 flex flex-col fixed z-20"
      >
        <div className="p-6 flex items-center gap-4 border-b border-white/5 h-[80px]">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0 shadow-2xl transition-all duration-500"
            style={{ backgroundColor: `${config.primary}11`, border: `1px solid ${config.primary}44` }}
          >
            <Settings size={20} style={{ color: config.primary }} className="animate-spin-slow" />
          </div>
          {sidebarOpen && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <h1 className="font-bold text-lg tracking-[0.2em] text-white leading-none">ROOT</h1>
              <span className="text-[9px] font-mono text-gray-500 tracking-widest uppercase mt-1 block opacity-50">CORTEX CONSOLE</span>
            </motion.div>
          )}
        </div>

        <nav className="flex-1 p-3 space-y-1">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-4 px-4 py-4 rounded-xl transition-all group relative overflow-hidden ${activeTab === item.id
                ? 'text-white'
                : 'hover:bg-white/5 text-gray-500 hover:text-gray-300'
                }`}
            >
              {activeTab === item.id && (
                <motion.div
                  layoutId="active-tab"
                  className="absolute inset-0 z-0 bg-white/5"
                  initial={false}
                  transition={{ duration: 0.2 }}
                  style={{ borderLeft: `2px solid ${config.primary}` }}
                />
              )}
              <item.icon size={18} className={`relative z-10 ${activeTab === item.id ? 'opacity-100' : 'opacity-40'}`} style={{ color: activeTab === item.id ? config.primary : 'inherit' }} />
              {sidebarOpen && <span className="text-[11px] font-mono tracking-widest uppercase relative z-10">{item.label}</span>}
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-white/5 flex justify-center">
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-3 rounded-full hover:bg-white/10 text-gray-500 transition-colors">
            <Menu size={18} />
          </button>
        </div>
      </motion.aside>

      {/* MAIN CONTENT */}
      <main
        className={`flex-1 flex flex-col min-h-screen transition-all duration-300 ${sidebarOpen ? 'pl-[260px]' : 'pl-[80px]'}`}
      >
        {/* TOP BAR */}
        <header className="h-[80px] border-b border-white/5 bg-black/40 backdrop-blur-3xl sticky top-0 z-10 flex items-center justify-between px-10">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 border border-red-500/30 rounded text-[9px] font-bold text-red-500 animate-pulse tracking-widest">
              <Fingerprint size={12} /> ROOT UPLINK ACTIVE
            </div>
            <div className="h-4 w-px bg-white/10" />
            <div className="text-[10px] font-mono text-gray-500 uppercase tracking-widest">
              Kernel: v4.12.0-neuro
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="text-right">
              <div className="text-[11px] font-bold text-white tracking-widest">SYS_ADMIN</div>
              <div className="text-[9px] text-gray-500 font-mono">NODE: 127.0.0.1</div>
            </div>
            <div className="w-10 h-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-gray-400">
              <Shield size={18} />
            </div>
          </div>
        </header>

        {/* CONTENT AREA */}
        <div className="p-8 pb-12">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
          >
            {renderContent()}
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default ConfigurationLayout;
