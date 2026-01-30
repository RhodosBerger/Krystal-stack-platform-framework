import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  AlertTriangle,
  Zap,
  Activity,
  Bell,
  Settings,
  ChevronDown,
  RefreshCw,
  Clock,
  CheckCircle2,
  XCircle,
  Database,
  DollarSign
} from 'lucide-react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

// --- VISUALIZATION COMPONENTS ---

const RadialGauge = ({ value, label, subtext, color = "text-neuro-success", strokeColor = "stroke-neuro-success" }) => (
  <div className="flex flex-col items-center justify-center p-4">
    <div className="relative w-24 h-24 mb-2">
      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="40" fill="none" stroke="currentColor" strokeWidth="8" className="text-white/5" />
        <circle
          cx="50"
          cy="50"
          r="40"
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          strokeDasharray={`${(value / 100) * 251} 251`}
          strokeLinecap="round"
          className={`${strokeColor} transition-all duration-1000 ease-out`}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-xl font-bold font-mono tracking-tighter ${color}`}>{value}%</span>
      </div>
    </div>
    <div className="text-xs font-bold text-gray-300">{label}</div>
    <div className="text-[10px] font-mono text-gray-600">{subtext}</div>
  </div>
);

const TrendLine = ({ data, color = "#00FFC8" }) => {
  const points = data.map((val, i) => `${i * (100 / (data.length - 1))},${100 - val}`).join(" ");
  return (
    <div className="w-full h-16 relative overflow-hidden">
      <svg className="w-full h-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id={`grad-${color}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.2" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon points={`0,100 ${points} 100,100`} fill={`url(#grad-${color})`} />
        <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
};

// --- MAIN COMPONENT ---

const ManufacturingAnalytics = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('24H');

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000); // 5s Polling
    return () => clearInterval(interval);
  }, []);

  const fetchMetrics = async () => {
    try {
      const res = await axios.get('/api/analytics/metrics');
      if (res.data.status === 'SUCCESS') {
        setMetrics(res.data.metrics);
      }
    } catch (e) {
      console.error("Analytics fetch failed", e);
    } finally {
      setLoading(false);
    }
  };

  // Safe fallback if metrics are still loading or empty
  if (loading || !metrics) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500 font-mono text-xs animate-pulse">
        <Database size={16} className="mr-2" />
        LOADING NEURAL METRICS...
      </div>
    );
  }

  // --- DERIVED MOCK DATA FOR VISUALIZATION (Since the simple metrics engine only stores scalars) ---
  const productionTrend = [45, 52, 49, 60, 55, 65, 70, 72, 68, 75, 80, 85, 82, 90, 88, 94];
  const efficiencyTrend = [80, 82, 81, 85, 84, 88, 90, 92, 91, 93, 94, 95, 96, 96, 98, 98];

  return (
    <div className="h-full flex flex-col gap-4">
      {/* 1. Header Toolbar */}
      <div className="flex justify-between items-center mb-2">
        <div className="flex gap-1 bg-white/5 rounded-lg p-1">
          {['1H', '24H', '7D', '30D'].map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 rounded text-[10px] font-bold font-mono transition-all ${timeRange === range
                ? 'bg-white/10 text-white shadow-sm'
                : 'text-gray-500 hover:text-white hover:bg-white/5'
                }`}
            >
              {range}
            </button>
          ))}
        </div>

        <button
          onClick={fetchMetrics}
          className="p-2 rounded bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* 2. Primary KPI Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* KPI 1: Production Volume */}
        <div className="bg-white/5 rounded-xl border border-white/5 p-4 flex flex-col justify-between group hover:border-white/10 transition-colors">
          <div className="flex justify-between items-start mb-2">
            <div className="p-2 rounded bg-industrial-primary/10 text-industrial-primary">
              <Settings size={18} />
            </div>
            <span className="text-[10px] font-mono text-neuro-success">+12.5%</span>
          </div>
          <div>
            <div className="text-2xl font-bold text-white font-sans">{metrics.total_products.toLocaleString()}</div>
            <div className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">Total Units</div>
          </div>
          <div className="mt-2 opacity-50 group-hover:opacity-100 transition-opacity">
            <TrendLine data={productionTrend} color="#FFD700" />
          </div>
        </div>

        {/* KPI 2: Success Rate */}
        <div className="bg-white/5 rounded-xl border border-white/5 p-4 flex flex-col justify-between group hover:border-white/10 transition-colors">
          <div className="flex justify-between items-start mb-2">
            <div className="p-2 rounded bg-neuro-success/10 text-neuro-success">
              <CheckCircle2 size={18} />
            </div>
            <span className="text-[10px] font-mono text-neuro-success">NOMINAL</span>
          </div>
          <div>
            <div className="text-2xl font-bold text-white font-sans">{metrics.success_rate.toFixed(1)}%</div>
            <div className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">Quality Yield</div>
          </div>
          <div className="mt-2 opacity-50 group-hover:opacity-100 transition-opacity">
            <TrendLine data={efficiencyTrend} color="#10B981" />
          </div>
        </div>

        {/* KPI 3: Profit Rate (Emerald Logic) */}
        <div className="bg-white/5 rounded-xl border border-white/5 p-4 flex flex-col justify-between group hover:border-white/10 transition-colors">
          <div className="flex justify-between items-start mb-2">
            <div className="p-2 rounded bg-neuro-success/10 text-neuro-success">
              <DollarSign size={18} />
            </div>
            <span className="text-[10px] font-mono text-neuro-success">MAX_PREMIUM</span>
          </div>
          <div>
            <div className="text-2xl font-bold text-white font-sans">${metrics.emerald?.profit_rate_min.toFixed(2)}/m</div>
            <div className="text-[10px] font-mono text-emerald-400 uppercase tracking-wider">Profit Rate (Pr)</div>
          </div>
          <div className="mt-2 text-[9px] font-mono text-gray-500">
            Mode: <span className="text-neuro-pulse">OPTIMIZED_PROFIT</span>
          </div>
        </div>

        {/* KPI 4: Spindle Balance (Emerald Logic) */}
        <div className="bg-white/5 rounded-xl border border-white/5 p-4 flex flex-col justify-between group hover:border-white/10 transition-colors">
          <div className="flex justify-between items-start mb-2">
            <div className="p-2 rounded bg-neuro-synapse/10 text-neuro-synapse">
              <RefreshCw size={18} />
            </div>
            <span className="text-[10px] font-mono text-neuro-pulse">
              {Math.round(metrics.emerald?.spindle_balance.efficiency * 100)}% SYNC
            </span>
          </div>
          <div>
            <div className="text-2xl font-bold text-white font-sans">{metrics.emerald?.spindle_balance.throughput_hourly.toFixed(1)}/h</div>
            <div className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">Spindle Throughput (Rp)</div>
          </div>
          <div className="mt-2 text-[9px] font-mono text-gray-400 italic">
            Delta: {metrics.emerald?.spindle_balance.unbalance_delta_mins.toFixed(1)}% unbalance
          </div>
        </div>
      </div>

      {/* 3. Detailed Breakdown & Utilization */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">

        {/* Utilization Gauges */}
        <div className="lg:col-span-1 bg-white/5 rounded-xl border border-white/5 p-4 flex flex-col items-center justify-center relative">
          <h3 className="absolute top-4 left-4 text-[10px] font-mono text-gray-500 uppercase flex items-center gap-2">
            <Zap size={12} /> System Utilization
          </h3>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
            <RadialGauge value={88} label="CPU Load" subtext="Neural Core" color="text-industrial-primary" strokeColor="stroke-industrial-primary" />
            <RadialGauge value={42} label="Memory" subtext="16GB Alloc" color="text-neuro-pulse" strokeColor="stroke-neuro-pulse" />
            <RadialGauge
              value={Math.round(metrics.emerald?.fleet_gravity * 10) || 75}
              label="Fleet Gravity"
              subtext="Swarm Pull"
              color="text-neuro-success"
              strokeColor="stroke-neuro-success"
            />
          </div>
        </div>

        {/* Event Timeline */}
        <div className="lg:col-span-1 bg-white/5 rounded-xl border border-white/5 p-4 flex flex-col">
          <h3 className="text-[10px] font-mono text-gray-500 uppercase mb-4 flex items-center gap-2">
            <Bell size={12} /> Recent System Events
          </h3>
          <div className="flex-1 overflow-y-auto pr-2 space-y-2 scrollbar-thin scrollbar-thumb-white/10">
            {metrics.timeline && metrics.timeline.length > 0 ? (
              metrics.timeline.slice().reverse().map((event, i) => (
                <div key={i} className="flex items-start gap-3 p-2 rounded bg-white/5 border border-white/5 hover:bg-white/10 transition-colors group">
                  <div className={`mt-1 w-2 h-2 rounded-full shrink-0 ${event.success ? 'bg-neuro-success shadow-[0_0_5px_rgba(16,185,129,0.5)]' : 'bg-neuro-danger'}`} />
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-baseline mb-0.5">
                      <span className="text-xs font-bold text-white truncate">{event.event}</span>
                      <span className="text-[9px] font-mono text-gray-500">{new Date(event.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div className="text-[10px] text-gray-400 font-mono truncate">
                      Payloads: {event.payload_count} | Duration: {event.duration_ms.toFixed(1)}ms
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="h-full flex items-center justify-center text-[10px] text-gray-600 font-mono">
                NO RECENT EVENTS
              </div>
            )}
          </div>
        </div>

        {/* Economic Risks (High Churn) */}
        <div className="lg:col-span-1 bg-white/5 rounded-xl border border-white/5 p-4 flex flex-col border-l-neuro-danger/20">
          <h3 className="text-[10px] font-mono text-neuro-danger uppercase mb-4 flex items-center gap-2">
            <AlertTriangle size={12} /> Economic Risk (High Churn)
          </h3>
          <div className="flex-1 overflow-y-auto pr-2 space-y-2 scrollbar-thin scrollbar-thumb-white/10">
            {metrics.emerald?.high_churn_alerts?.map((alert, i) => (
              <div key={i} className="p-3 rounded bg-neuro-danger/5 border border-neuro-danger/10 hover:bg-neuro-danger/10 transition-colors group">
                <div className="flex justify-between items-start mb-1">
                  <span className="text-xs font-bold text-white uppercase">{alert.name}</span>
                  <span className="text-[10px] font-mono text-neuro-danger">{(alert.churn_rate * 100).toFixed(0)}% CHURN</span>
                </div>
                <div className="text-[9px] text-gray-400 font-mono italic">
                  Recommendation: Deprecate or Refactor (Theory 6)
                </div>
              </div>
            ))}
            {(!metrics.emerald?.high_churn_alerts || metrics.emerald.high_churn_alerts.length === 0) && (
              <div className="h-full flex items-center justify-center text-[10px] text-gray-600 font-mono italic">
                NO ACTIVE ECONOMIC THREATS
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ManufacturingAnalytics;