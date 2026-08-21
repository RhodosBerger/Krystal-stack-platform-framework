import React from 'react';
import SwarmMap from '../components/SwarmMap';
import ManufacturingAnalytics from '../components/ManufacturingAnalytics';
import MarketplaceHub from '../components/MarketplaceHub';
import NeuroCard from '../components/NeuroCard';
import { usePersona } from '../context/PersonaContext';
import {
  Globe,
  Shield,
  Activity,
  Zap,
  TrendingUp,
  DollarSign,
  Box,
  Server,
  BarChart3
} from 'lucide-react';
import IntelligenceDashboard from '../components/IntelligenceDashboard';


const BentoCard = ({ title, children, className = "", icon: Icon, accentColor = "text-white" }) => {
  const { config } = usePersona();
  return (
    <div
      className={`glass-panel-pro relative overflow-hidden p-4 lg:p-6 transition-all duration-300 hover:border-white/20 group ${className}`}
      style={{ borderRadius: config.borderRadius }}
    >
      <div className="flex justify-between items-start mb-3">
        {title && (
          <h3 className="text-[10px] font-mono tracking-[0.2em] text-gray-400 border-l-2 pl-3 uppercase flex items-center gap-2" style={{ borderColor: config.primary }}>
            {Icon && <Icon size={12} className={accentColor} />}
            {title}
          </h3>
        )}
        <div className="w-1.5 h-1.5 rounded-full bg-white/10 group-hover:bg-white/30 transition-colors" />
      </div>
      <div className=" relative z-10 h-full">
        {children}
      </div>

      {/* Subtle Glow */}
      <div
        className="absolute -top-24 -right-24 w-48 h-48 opacity-0 group-hover:opacity-5 rounded-full blur-3xl transition-all"
        style={{ background: config.primary }}
      />
    </div>
  );
};

const ManagerLayout = () => {
  const { config } = usePersona();
  const [activeView, setActiveView] = React.useState('FLEET');

  return (
    <div className="p-6 min-h-screen bg-[#050505] text-gray-200 flex flex-col font-sans selection:bg-neuro-success selection:text-black">
      {/* CENTERED CONTAINER */}
      <div className="max-w-[1920px] mx-auto w-full flex flex-col h-full">

        {/* 1. HEADER ROW */}
        <header className="flex-none h-16 mb-8 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center border" style={{ borderColor: `${config.primary}33`, backgroundColor: `${config.primary}11` }}>
              <Globe size={20} style={{ color: config.primary }} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-widest uppercase leading-none">Fleet Command</h1>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-[9px] font-mono text-gray-500 tracking-widest uppercase">Global Operations Registry</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* VIEW SWITCHER */}
            <div className="bg-white/10 p-1 rounded-lg flex gap-1">
              <button
                onClick={() => setActiveView('FLEET')}
                className={`px-3 py-1.5 rounded-md text-[10px] font-bold tracking-wider transition-all ${activeView === 'FLEET' ? 'bg-white text-black shadow-lg' : 'text-gray-400 hover:text-white'
                  }`}
              >
                FLEET VIEW
              </button>
              <button
                onClick={() => setActiveView('INTELLIGENCE')}
                className={`px-3 py-1.5 rounded-md text-[10px] font-bold tracking-wider transition-all ${activeView === 'INTELLIGENCE' ? 'bg-white text-black shadow-lg' : 'text-gray-400 hover:text-white'
                  }`}
              >
                INTELLIGENCE
              </button>
            </div>

            <div className="w-px h-8 bg-white/10" />

            <div className="flex items-center gap-6 bg-white/5 backdrop-blur-xl rounded-xl px-6 py-3 border border-white/5">
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-neuro-success" />
                <span className="text-[10px] font-mono font-bold text-gray-400">STATUS: OPTIMAL</span>
              </div>
              <div className="w-px h-4 bg-white/10" />
              <div className="flex items-center gap-2">
                <Shield size={12} className="text-neuro-success" />
                <span className="text-[10px] font-mono font-bold text-gray-400">RBAC: ACTIVE</span>
              </div>
            </div>
          </div>
        </header>

        {/* 2. MAIN CONTENT AREA */}
        <div className="flex-1 overflow-hidden relative">

          {activeView === 'INTELLIGENCE' ? (
            <div className="h-full animate-in fade-in zoom-in duration-300">
              <IntelligenceDashboard />
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-1 h-full overflow-y-auto lg:overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500">
              {/* ROW 1: ECONOMIC NEUROCARDS */}
              <div className="col-span-12 lg:col-span-8 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-1 min-h-[100px] lg:h-32 mb-1">
                <NeuroCard
                  title="24H Yield (USD)"
                  metric="$42.9k"
                  status="OPTIMAL"
                  volatility={0.2}
                />
                <NeuroCard
                  title="Fleet Efficiency"
                  metric="88.4%"
                  status="NOMINAL"
                  volatility={0.4}
                  unit="OEE"
                />
                <NeuroCard
                  title="Active Nodes"
                  metric="12"
                  status="OK"
                  volatility={0.1}
                  unit="UNITS"
                />
              </div>

              {/* SIDE BAR: ECOSYSTEM & MARKETPLACE */}
              <div className="col-span-12 lg:col-span-4 row-span-3 min-h-[400px]">
                <BentoCard title="Ecosystem Hub" className="h-full" icon={BarChart3}>
                  <MarketplaceHub />
                </BentoCard>
              </div>

              {/* ROW 2: SWARM MAP (The Network) */}
              <div className="col-span-12 lg:col-span-8 min-h-[300px] lg:h-80 mb-1">
                <BentoCard title="Fleet Swarm Map" className="h-full" icon={Activity}>
                  <SwarmMap />
                </BentoCard>
              </div>

              {/* ROW 3: ANALYTICS (Economic Engine) */}
              <div className="col-span-12 lg:col-span-8 min-h-[400px]">
                <BentoCard title="Economic Performance Analytics" className="h-full" icon={TrendingUp}>
                  <ManufacturingAnalytics />
                </BentoCard>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ManagerLayout;