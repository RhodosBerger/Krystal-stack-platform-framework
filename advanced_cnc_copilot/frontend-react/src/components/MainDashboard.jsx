import React, { useState, useEffect } from 'react';
import { Activity, Zap, AlertTriangle, Clock, CheckCircle2, TrendingUp, Gauge, Thermometer, Loader2 } from 'lucide-react';
import axios from 'axios';
import SwarmMap from './SwarmMap';

/**
 * Main Dashboard - Based on mockup design
 * Matches the FANUC RISE // CNC COPILOT visual design
 */
const MainDashboard = () => {
    const [machineStatus, setMachineStatus] = useState({
        spindleRpm: 8500,
        feedRate: 1200,
        tool: 'T01',
        temp: 45,
        load: 65
    });
    const [gcodePreview, setGcodePreview] = useState([]);
    const [analytics, setAnalytics] = useState({ programs: 88, production: 89, utilization: 92 });
    const [loading, setLoading] = useState(true);

    const [neuroState, setNeuroState] = useState({ dopamine: 50, cortisol: 10, serotonin: 70 });
    const [lastAction, setLastAction] = useState("MONITORING_LIVE");

    useEffect(() => {
        fetchData();

        // Connect to Real-time WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host === 'localhost:3000' ? 'localhost:8000' : window.location.host;
        const wsUrl = `${protocol}//${host}/ws/telemetry`;
        const socket = new WebSocket(wsUrl);

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setMachineStatus(prev => ({
                ...prev,
                spindleRpm: data.rpm || prev.spindleRpm,
                load: data.load || prev.load,
                vibration: data.vibration || 0
            }));

            if (data.neuro_state) {
                setNeuroState(data.neuro_state);
            }
            if (data.action) {
                setLastAction(data.action);
            }
        };

        socket.onclose = () => console.log("WebSocket Disconnected");

        return () => socket.close();
    }, []);

    const fetchData = async () => {
        setGcodePreview([
            { line: 1, code: 'O1234 (CNC MILLING PROGRAM)' },
            { line: 2, code: 'G21 G17 G90 G40 G49 G80' },
            { line: 3, code: 'T01 M06 (TOOL 1 - 10MM END MILL)' },
            { line: 4, code: 'G00 X-50. Y-50. S8500 M03 (RAPID MOVE, SPINDLE ON)' },
            { line: 5, code: 'G43 H01 Z50. (TOOL LENGTH OFFSET)' },
            { line: 6, code: 'M08 (COOLANT ON)' },
            { line: 7, code: 'G01 Z-2. F500. (FEED MOVE Z-AXIS)' },
            { line: 8, code: 'X50. F1200. (FEED MOVE X-AXIS)' },
            { line: 9, code: 'Y50.' },
            { line: 10, code: 'X-50.' },
            { line: 11, code: 'Y-50.' },
            { line: 12, code: 'G00 Z50. M09 (COOLANT OFF)' },
            { line: 13, code: 'M30 (END OF PROGRAM)' }
        ]);
        setLoading(false);
    };

    const highlightCode = (code) => {
        if (code.includes('(')) return 'comment';
        if (code.match(/^[GMT]\d+/)) return 'command';
        if (code.match(/^[XYZIJKFSR]/)) return 'coord';
        return 'default';
    };

    return (
        <div className="main-dashboard">
            {/* Header */}
            <div className="dash-header">
                <div className="logo">
                    <Zap size={20} className="logo-icon" />
                    <span>FANUC RISE // <strong>CNC COPILOT</strong></span>
                </div>
                <div className="status-badge online">
                    <div className="pulse"></div>
                    <span>SYSTEM ONLINE</span>
                </div>
            </div>

            <div className="dash-grid">
                {/* G-Code Preview Panel */}
                <div className="panel gcode-panel">
                    <div className="panel-header">
                        <span>G-Code Preview // PROGRAM: O1234_MILL_OP</span>
                    </div>
                    <div className="gcode-content">
                        {gcodePreview.map(line => (
                            <div key={line.line} className={`gcode-line ${highlightCode(line.code)}`}>
                                <span className="line-num">{line.line}</span>
                                <span className="line-code">{line.code}</span>
                            </div>
                        ))}
                    </div>
                    <div className="progress-bar">
                        <div className="progress-fill" style={{ width: '65%' }}></div>
                    </div>
                </div>

                {/* Machine Status Panel */}
                <div className="panel status-panel">
                    <div className="panel-header">
                        <span>Machine Status // VMC-01</span>
                        <div className="status-dot online"></div>
                    </div>

                    <div className="gauge-row">
                        <div className="gauge-item">
                            <div className="gauge-label">SPINDLE RPM</div>
                            <div className="gauge-value cyan">{machineStatus.spindleRpm}</div>
                            <div className="gauge-unit">RPM</div>
                            <div className="mini-bar">
                                <div className="mini-fill" style={{ width: `${(machineStatus.spindleRpm / 12000) * 100}%` }}></div>
                            </div>
                        </div>
                        <div className="gauge-item">
                            <div className="gauge-label">FEED RATE</div>
                            <div className="gauge-value green">{machineStatus.feedRate}</div>
                            <div className="gauge-unit">MM/MIN</div>
                            <div className="mini-bar green">
                                <div className="mini-fill" style={{ width: `${(machineStatus.feedRate / 1200) * 100}%` }}></div>
                            </div>
                        </div>
                    </div>

                    <div className="status-cards">
                        <div className="status-card">
                            <span className="card-label">TOOL:</span>
                            <span className="card-value">{machineStatus.tool}</span>
                        </div>
                        <div className="status-card">
                            <Thermometer size={14} />
                            <span className="card-label">TEMP:</span>
                            <span className="card-value">{machineStatus.temp}°C</span>
                        </div>
                        <div className="status-card">
                            <Activity size={14} />
                            <span className="card-label">LOAD:</span>
                            <span className="card-value">{machineStatus.load}%</span>
                        </div>
                    </div>
                </div>

                {/* G-Code Programs Chart */}
                <div className="panel chart-panel">
                    <div className="panel-header">
                        <span>G-Code Programs // LAST 24H</span>
                    </div>
                    <div className="chart-area">
                        <svg viewBox="0 0 300 100" className="line-chart">
                            <defs>
                                <linearGradient id="chartGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                                    <stop offset="0%" stopColor="#00ff88" stopOpacity="0.3" />
                                    <stop offset="100%" stopColor="#00ff88" stopOpacity="0" />
                                </linearGradient>
                            </defs>
                            <path d="M0,80 L30,70 L60,75 L90,50 L120,55 L150,40 L180,45 L210,30 L240,25 L270,15 L300,12"
                                fill="url(#chartGradient)" stroke="none" />
                            <path d="M0,80 L30,70 L60,75 L90,50 L120,55 L150,40 L180,45 L210,30 L240,25 L270,15 L300,12"
                                fill="none" stroke="#00ff88" strokeWidth="2" />
                        </svg>
                        <div className="chart-value">{analytics.programs}%</div>
                    </div>
                </div>

                {/* Real-Time Analytics */}
                <div className="panel analytics-panel">
                    <div className="panel-header">
                        <span>Real-Time Analytics // PRODUCTION RATE</span>
                    </div>
                    <div className="chart-area small">
                        <svg viewBox="0 0 200 60" className="line-chart">
                            <path d="M0,50 L25,45 L50,48 L75,35 L100,30 L125,25 L150,20 L175,15 L200,10"
                                fill="none" stroke="#00d4ff" strokeWidth="2" />
                        </svg>
                        <div className="chart-value cyan">{analytics.production}%</div>
                    </div>

                    <div className="util-section">
                        <div className="panel-header small">
                            <span>Machine Utilization // LAST 24H</span>
                        </div>
                        <div className="util-bars">
                            {[85, 90, 75, 95, 88, 92].map((v, i) => (
                                <div key={i} className="util-bar" style={{ height: `${v}%` }}></div>
                            ))}
                        </div>
                        <div className="util-value">{analytics.utilization}%</div>
                    </div>
                </div>

                {/* Neuro-Sync Panel */}
                <div className="panel neuro-panel" style={{ gridColumn: 'span 2' }}>
                    <div className="panel-header">
                        <span>Neuro-Sync // AI COGNITIVE STATE</span>
                        <div className="status-label">{lastAction}</div>
                    </div>
                    <div className="neuro-grid" style={{ display: 'flex', justifyContent: 'space-around', padding: '20px' }}>
                        <div className="neuro-item">
                            <div className="neuro-label">DOPAMINE</div>
                            <div className="neuro-bar-container" style={{ width: '150px', height: '10px', background: '#222', borderRadius: '5px', overflow: 'hidden' }}>
                                <div className="neuro-bar" style={{ height: '100%', width: `${neuroState.dopamine}%`, background: '#ff00ff', boxShadow: '0 0 10px #ff00ff' }}></div>
                            </div>
                            <div className="neuro-val" style={{ textAlign: 'center', marginTop: '5px', color: '#ff00ff' }}>{neuroState.dopamine.toFixed(1)}</div>
                        </div>
                        <div className="neuro-item">
                            <div className="neuro-label">CORTISOL</div>
                            <div className="neuro-bar-container" style={{ width: '150px', height: '10px', background: '#222', borderRadius: '5px', overflow: 'hidden' }}>
                                <div className="neuro-bar" style={{ height: '100%', width: `${neuroState.cortisol}%`, background: '#ff4400', boxShadow: '0 0 10px #ff4400' }}></div>
                            </div>
                            <div className="neuro-val" style={{ textAlign: 'center', marginTop: '5px', color: '#ff4400' }}>{neuroState.cortisol.toFixed(1)}</div>
                        </div>
                        <div className="neuro-item">
                            <div className="neuro-label">SEROTONIN</div>
                            <div className="neuro-bar-container" style={{ width: '150px', height: '10px', background: '#222', borderRadius: '5px', overflow: 'hidden' }}>
                                <div className="neuro-bar" style={{ height: '100%', width: `${neuroState.serotonin}%`, background: '#00d4ff', boxShadow: '0 0 10px #00d4ff' }}></div>
                            </div>
                            <div className="neuro-val" style={{ textAlign: 'center', marginTop: '5px', color: '#00d4ff' }}>{neuroState.serotonin.toFixed(1)}</div>
                        </div>
                    </div>
                </div>

                {/* Global Swarm Intelligence Map */}
                <div className="panel swarm-panel" style={{ gridColumn: 'span 2' }}>
                    <SwarmMap />
                </div>
            </div>

            <style>{`
        .main-dashboard { background: #0a0a0f; min-height: 100vh; padding: 20px; font-family: 'Inter', sans-serif; }
        
        .dash-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding: 15px 20px; background: linear-gradient(90deg, #111 0%, #1a1a1f 100%); border: 1px solid #00ff88; border-radius: 12px; }
        .logo { display: flex; align-items: center; gap: 10px; font-size: 1.1rem; color: #00ff88; }
        .logo-icon { animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .status-badge { display: flex; align-items: center; gap: 8px; padding: 8px 15px; border-radius: 20px; font-size: 0.7rem; font-weight: bold; }
        .status-badge.online { background: rgba(0,255,136,0.15); color: #00ff88; }
        .pulse { width: 8px; height: 8px; background: #00ff88; border-radius: 50%; animation: blink 1s infinite; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        
        .dash-grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto; gap: 15px; }
        
        .panel { background: #111; border: 1px solid #222; border-radius: 12px; overflow: hidden; }
        .panel-header { padding: 12px 15px; background: #0f0f12; border-bottom: 1px solid #222; font-size: 0.7rem; color: #888; font-weight: 600; letter-spacing: 0.5px; display: flex; justify-content: space-between; align-items: center; }
        .panel-header.small { padding: 8px 15px; font-size: 0.65rem; }
        
        .gcode-content { padding: 15px; max-height: 280px; overflow-y: auto; font-family: 'Fira Code', monospace; font-size: 0.75rem; }
        .gcode-line { display: flex; gap: 15px; padding: 3px 0; }
        .line-num { color: #444; min-width: 20px; }
        .line-code.comment { color: #666; font-style: italic; }
        .line-code.command { color: #00d4ff; font-weight: bold; }
        .line-code.coord { color: #00ff88; }
        .line-code.default { color: #eee; }
        .progress-bar { height: 3px; background: #222; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #00ff88, #00d4ff); border-radius: 3px; }
        
        .gauge-row { display: flex; justify-content: space-around; padding: 20px; }
        .gauge-item { text-align: center; }
        .gauge-label { font-size: 0.6rem; color: #666; margin-bottom: 5px; }
        .gauge-value { font-size: 2rem; font-weight: bold; }
        .gauge-value.cyan { color: #00d4ff; }
        .gauge-value.green { color: #00ff88; }
        .gauge-unit { font-size: 0.65rem; color: #555; }
        .mini-bar { height: 4px; width: 100px; background: #222; border-radius: 2px; margin-top: 8px; }
        .mini-bar .mini-fill { height: 100%; background: #00d4ff; border-radius: 2px; }
        .mini-bar.green .mini-fill { background: #00ff88; }
        
        .status-cards { display: flex; justify-content: space-around; padding: 15px; border-top: 1px solid #222; }
        .status-card { display: flex; align-items: center; gap: 6px; padding: 10px 15px; background: #1a1a1f; border-radius: 8px; font-size: 0.75rem; }
        .status-card svg { color: #888; }
        .card-label { color: #666; }
        .card-value { color: #eee; font-weight: bold; }
        
        .chart-area { padding: 20px; position: relative; }
        .chart-area.small { padding: 15px; }
        .line-chart { width: 100%; height: 80px; }
        .chart-value { position: absolute; top: 15px; right: 20px; font-size: 1.5rem; font-weight: bold; color: #00ff88; }
        .chart-value.cyan { color: #00d4ff; }
        
        .util-section { border-top: 1px solid #222; }
        .util-bars { display: flex; justify-content: space-around; align-items: flex-end; height: 60px; padding: 15px; }
        .util-bar { width: 20px; background: linear-gradient(to top, #00ff88, #00d4ff); border-radius: 3px 3px 0 0; }
        .util-value { text-align: right; padding: 0 15px 10px; font-size: 1.2rem; font-weight: bold; color: #00ff88; }
        
        .status-dot { width: 8px; height: 8px; border-radius: 50%; }
        .status-dot.online { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
      `}</style>
        </div>
    );
};

export default MainDashboard;
