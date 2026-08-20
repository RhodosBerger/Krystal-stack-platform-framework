import React, { useState } from 'react';
import {
  LayoutDashboard,
  Box,
  Users,
  Settings,
  Search,
  Bell,
  Menu,
  MoreVertical,
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  Package,
  Cpu,
  Server,
  Activity
} from 'lucide-react';
import { motion } from 'framer-motion';

// --- ATOMIC COMPONENTS (Material Design Inspired) ---

const MaterialCard = ({ children, className = "", elevation = 1 }) => {
  const shadow = elevation === 1 ? 'shadow-md' : elevation === 2 ? 'shadow-xl' : 'shadow-2xl';
  return (
    <div className={`bg-industrial-surface rounded-xl border border-white/5 ${shadow} p-6 transition-all hover:bg-white/5 ${className}`}>
      {children}
    </div>
  );
};

const StatCard = ({ title, value, unit, trend, icon: Icon, color = "text-white" }) => (
  <MaterialCard elevation={1} className="relative overflow-hidden group">
    <div className="flex justify-between items-start mb-4">
      <div>
        <h3 className="text-[10px] font-mono tracking-widest text-gray-400 uppercase mb-1">{title}</h3>
        <div className="flex items-baseline gap-1">
          <span className={`text-3xl font-bold font-sans ${color}`}>{value}</span>
          {unit && <span className="text-xs text-gray-500 font-mono">{unit}</span>}
        </div>
      </div>
      <div className={`p-3 rounded-lg bg-white/5 group-hover:scale-110 transition-transform ${color}`}>
        <Icon size={20} />
      </div>
    </div>

    {trend && (
      <div className="flex items-center gap-2 text-xs font-mono">
        <span className={trend > 0 ? "text-neuro-success" : "text-neuro-danger"}>
          {trend > 0 ? "▲" : "▼"} {Math.abs(trend)}%
        </span>
        <span className="text-gray-600">vs last shift</span>
      </div>
    )}

    {/* Decorative Tech Elements */}
    <div className="absolute -bottom-4 -right-4 text-[100px] opacity-[0.02] pointer-events-none select-none font-mono">
      {value}
    </div>
  </MaterialCard>
);

const Badge = ({ status }) => {
  const styles = {
    active: "bg-neuro-success/10 text-neuro-success border-neuro-success/20",
    warning: "bg-industrial-primary/10 text-industrial-primary border-industrial-primary/20",
    critical: "bg-neuro-danger/10 text-neuro-danger border-neuro-danger/20",
    offline: "bg-gray-500/10 text-gray-400 border-gray-500/20"
  };

  const config = {
    active: { icon: CheckCircle2, label: "OPTIMAL" },
    warning: { icon: AlertTriangle, label: "WARNING" },
    critical: { icon: Activity, label: "CRITICAL" },
    offline: { icon: Server, label: "OFFLINE" }
  }[status.toLowerCase()] || { icon: Server, label: status };

  const Icon = config.icon;

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-bold border ${styles[status.toLowerCase()] || styles.offline}`}>
      <Icon size={10} />
      {config.label}
    </span>
  );
};

// --- LAYOUT ---

const ResourceLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Mock Data
  const resources = [
    { id: "CNC-001", name: "Fanuc VMC Alpha", type: "Machine", load: 88, status: "Active", maintenance: "4d" },
    { id: "CNC-002", name: "Fanuc Robodrill", type: "Machine", load: 45, status: "Warning", maintenance: "Overdue" },
    { id: "MAT-A7", name: "Aluminum 6061-T6", type: "Inventory", load: 240, status: "Active", unit: "kg" },
    { id: "SRV-MAIN", name: "Neural Core Server", type: "Infrastructure", load: 12, status: "Optimal", unit: "%" },
    { id: "BOT-ARM", name: "Robotic Arm L7", type: "Robotics", load: 0, status: "Offline", maintenance: "N/A" },
  ];

  return (
    <div className="min-h-screen bg-industrial-bg text-gray-200 flex font-sans selection:bg-neuro-pulse selection:text-black">

      {/* 1. SIDEBAR NAVIGATION */}
      <motion.aside
        initial={false}
        animate={{ width: sidebarOpen ? 280 : 80 }}
        className="h-screen bg-industrial-surface border-r border-white/5 flex flex-col fixed z-20"
      >
        <div className="p-6 flex items-center gap-3 border-b border-white/5 h-[80px]">
          <div className="w-8 h-8 rounded bg-gradient-to-br from-industrial-primary to-orange-500 flex items-center justify-center shrink-0">
            <Cpu size={18} className="text-black" />
          </div>
          {sidebarOpen && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <h1 className="font-bold text-lg tracking-tight text-white leading-none">NEXUS</h1>
              <span className="text-[9px] font-mono text-gray-500 tracking-widest">RESOURCE MANAGER</span>
            </motion.div>
          )}
        </div>

        <nav className="flex-1 p-4 space-y-2">
          {[
            { icon: LayoutDashboard, label: "Dashboard", active: true },
            { icon: Box, label: "Inventory" },
            { icon: Users, label: "Personnel" },
            { icon: Activity, label: "Analytics" },
            { icon: Settings, label: "Settings" },
          ].map((item) => (
            <button
              key={item.label}
              className={`w-full flex items-center gap-4 px-3 py-3 rounded-lg transition-all group ${item.active ? 'bg-white/10 text-white' : 'hover:bg-white/5 text-gray-400 hover:text-white'}`}
            >
              <item.icon size={20} className={item.active ? 'text-industrial-primary' : ''} />
              {sidebarOpen && <span className="text-sm font-medium">{item.label}</span>}
              {!sidebarOpen && item.active && (
                <div className="absolute left-16 bg-white text-black text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
                  {item.label}
                </div>
              )}
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-white/5">
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 rounded hover:bg-white/10 text-gray-400">
            <Menu size={20} />
          </button>
        </div>
      </motion.aside>

      {/* 2. MAIN CONTENT AREA */}
      <main
        className={`flex-1 flex flex-col min-h-screen transition-all duration-300 ${sidebarOpen ? 'pl-[280px]' : 'pl-[80px]'}`}
      >
        {/* TOP BAR */}
        <header className="h-[80px] border-b border-white/5 bg-industrial-bg/80 backdrop-blur-md sticky top-0 z-10 flex items-center justify-between px-8">
          <div className="flex items-center gap-4 w-1/3">
            <Search size={18} className="text-gray-500" />
            <input
              type="text"
              placeholder="Search resources, IDs, or maintenance logs..."
              className="bg-transparent border-none outline-none text-sm text-white w-full placeholder:text-gray-600 font-mono"
            />
          </div>

          <div className="flex items-center gap-6">
            <button className="relative text-gray-400 hover:text-white transition-colors">
              <Bell size={20} />
              <span className="absolute top-0 right-0 w-2 h-2 bg-neuro-danger rounded-full animate-pulse"></span>
            </button>
            <div className="flex items-center gap-3 pl-6 border-l border-white/10">
              <div className="text-right hidden md:block">
                <div className="text-sm font-bold text-white">Admin User</div>
                <div className="text-[10px] text-gray-500 font-mono">ROOT_ACCESS</div>
              </div>
              <div className="w-10 h-10 rounded-full bg-white/10 border border-white/20 flex items-center justify-center">
                <Users size={18} className="text-industrial-primary" />
              </div>
            </div>
          </div>
        </header>

        {/* DASHBOARD CONTENT */}
        <div className="p-8 space-y-8">

          {/* Header & Actions */}
          <div className="flex justify-between items-end">
            <div>
              <h2 className="text-3xl font-bold text-white mb-2">Resource Overview</h2>
              <p className="text-gray-500 font-mono text-sm">SYSTEM_STATUS: <span className="text-neuro-success">NOMINAL</span> // LAST_SYNC: 12ms ago</p>
            </div>
            <button className="bg-industrial-primary text-black px-6 py-2.5 rounded-lg font-bold text-sm shadow-[0_0_20px_rgba(255,215,0,0.2)] hover:shadow-[0_0_30px_rgba(255,215,0,0.4)] transition-all flex items-center gap-2">
              <Package size={16} /> ALLOCATE NEW
            </button>
          </div>

          {/* KPI GRID */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatCard
              title="ACTIVE MACHINES"
              value="12"
              unit="/ 15"
              trend={8.5}
              icon={Cpu}
              color="text-industrial-primary"
            />
            <StatCard
              title="TOTAL POWER DRAW"
              value="48.2"
              unit="kW"
              trend={-2.4}
              icon={TrendingUp}
              color="text-neuro-pulse"
            />
            <StatCard
              title="MATERIAL STOCK"
              value="892"
              unit="Tons"
              trend={0.0}
              icon={Package}
              color="text-white"
            />
            <StatCard
              title="SYSTEM ALERTS"
              value="03"
              unit="Active"
              trend={12}
              icon={AlertTriangle}
              color="text-neuro-danger"
            />
          </div>

          {/* MAIN TABLE SECTION */}
          <MaterialCard elevation={2} className="overflow-hidden p-0">
            <div className="p-6 border-b border-white/5 flex justify-between items-center">
              <h3 className="font-bold text-lg text-white flex items-center gap-2">
                <Server size={18} className="text-gray-500" />
                Connected Entities
              </h3>
              <div className="flex gap-2">
                <button className="px-3 py-1 text-xs font-mono border border-white/10 rounded hover:bg-white/5 text-gray-400">FILTER</button>
                <button className="px-3 py-1 text-xs font-mono border border-white/10 rounded hover:bg-white/5 text-gray-400">EXPORT</button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-white/5 text-gray-400 font-mono text-xs uppercase">
                  <tr>
                    <th className="px-6 py-4 font-normal tracking-wider">Entity ID</th>
                    <th className="px-6 py-4 font-normal tracking-wider">Name / Type</th>
                    <th className="px-6 py-4 font-normal tracking-wider">Status</th>
                    <th className="px-6 py-4 font-normal tracking-wider">Load</th>
                    <th className="px-6 py-4 font-normal tracking-wider">Maintenance</th>
                    <th className="px-6 py-4 font-normal tracking-wider text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {resources.map((res) => (
                    <tr key={res.id} className="hover:bg-white/[0.02] transition-colors group">
                      <td className="px-6 py-4 font-mono text-industrial-primary">{res.id}</td>
                      <td className="px-6 py-4">
                        <div className="font-bold text-white">{res.name}</div>
                        <div className="text-xs text-gray-500">{res.type}</div>
                      </td>
                      <td className="px-6 py-4">
                        <Badge status={res.status} />
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="w-24 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${res.load > 80 ? 'bg-neuro-danger' : 'bg-neuro-success'}`}
                              style={{ width: `${Math.min(res.load, 100)}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs">{res.load}%</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 font-mono text-xs text-gray-400">{res.maintenance}</td>
                      <td className="px-6 py-4 text-right">
                        <button className="p-2 hover:bg-white/10 rounded-full text-gray-500 hover:text-white transition-colors">
                          <MoreVertical size={16} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="p-4 border-t border-white/5 flex justify-between items-center text-xs text-gray-500 font-mono">
              <span>SHOWING 5 OF 128 RESOURCES</span>
              <div className="flex gap-2">
                <button className="hover:text-white">PREV</button>
                <button className="text-white">1</button>
                <button className="hover:text-white">2</button>
                <button className="hover:text-white">3</button>
                <button className="hover:text-white">NEXT</button>
              </div>
            </div>
          </MaterialCard>

        </div>
      </main>
    </div>
  );
};

export default ResourceLayout;
