import React, { useState, useEffect } from 'react';
import { Home, Activity, Bell, Settings, FolderArchive, Globe, BarChart3, Heart, Layers, Palette, Package, Sliders, ArrowRight, Zap, CheckCircle2 } from 'lucide-react';
import axios from 'axios';

const DashboardOverview = ({ onNavigate }) => {
    const [healthStatus, setHealthStatus] = useState(null);
    const [stats, setStats] = useState({ products: 0, notifications: 0 });

    useEffect(() => {
        fetchHealth();
        fetchStats();
    }, []);

    const fetchHealth = async () => {
        try {
            const res = await axios.get('/api/health');
            setHealthStatus(res.data);
        } catch (e) { setHealthStatus({ status: 'ERROR' }); }
    };

    const fetchStats = async () => {
        try {
            const [prodRes, notifRes] = await Promise.all([
                axios.get('/api/llm/products').catch(() => ({ data: { products: [] } })),
                axios.get('/api/notifications/unread-count').catch(() => ({ data: { unread_count: 0 } }))
            ]);
            setStats({
                products: prodRes.data?.products?.length || 0,
                notifications: notifRes.data?.unread_count || 0
            });
        } catch (e) { console.error(e); }
    };

    const views = [
        { key: 'nexus', name: 'React Nexus', icon: Home, desc: 'Main Assembly Canvas', color: '#00ff88' },
        { key: 'portal', name: 'Pro Portal', icon: Globe, desc: 'Product & G-Code Generation', color: '#00d4ff' },
        { key: 'analytics', name: 'Analytics', icon: Activity, desc: 'Metrics & Workflows', color: '#ff6b6b' },
        { key: 'notifications', name: 'Alerts', icon: Bell, desc: `${stats.notifications} unread`, color: '#ffaa00' },
        { key: 'config', name: 'Config Manager', icon: Sliders, desc: 'LLM-Maintainable Settings', color: '#a855f7' },
        { key: 'data', name: 'Data Hub', icon: FolderArchive, desc: 'Export/Import Projects', color: '#14b8a6' },
        { key: 'business', name: 'Business Overlay', icon: BarChart3, desc: 'ROI & Sustainability', color: '#f97316' },
        { key: 'sentience', name: 'Sentience Layer', icon: Heart, desc: 'Emotional Intelligence', color: '#ec4899' },
        { key: 'creative', name: 'Creative Twin', icon: Palette, desc: 'Blender Integration', color: '#8b5cf6' },
        { key: 'platform', name: 'Platform Hub', icon: Layers, desc: 'Generation Pipeline', color: '#06b6d4' },
        { key: 'prefs', name: 'Preferences', icon: Settings, desc: 'Master Settings', color: '#84cc16' },
    ];

    return (
        <div className="dashboard-overview">
            {/* Hero Section */}
            <div className="hero-section">
                <div className="hero-content">
                    <h1>🏭 FANUC RISE // CNC COPILOT</h1>
                    <p>Advanced Manufacturing Intelligence System</p>
                    <div className="status-bar">
                        <div className={`status-pill ${healthStatus?.status === 'OK' ? 'online' : 'offline'}`}>
                            <Zap size={14} />
                            <span>{healthStatus?.status === 'OK' ? 'SYSTEM ONLINE' : 'CONNECTING...'}</span>
                        </div>
                        <div className="stat-pill">
                            <Package size={14} />
                            <span>{stats.products} Products</span>
                        </div>
                        <div className="stat-pill">
                            <Bell size={14} />
                            <span>{stats.notifications} Alerts</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* View Grid */}
            <div className="views-grid">
                {views.map(v => {
                    const IconComponent = v.icon;
                    return (
                        <div
                            key={v.key}
                            className="view-card"
                            onClick={() => onNavigate && onNavigate(v.key)}
                            style={{ '--accent': v.color }}
                        >
                            <div className="view-icon" style={{ background: v.color + '22', color: v.color }}>
                                <IconComponent size={24} />
                            </div>
                            <div className="view-info">
                                <h3>{v.name}</h3>
                                <p>{v.desc}</p>
                            </div>
                            <ArrowRight size={16} className="arrow" />
                        </div>
                    );
                })}
            </div>

            {/* Quick Stats */}
            <div className="quick-stats">
                <div className="stat-card">
                    <CheckCircle2 size={20} color="#00ff88" />
                    <div>
                        <span className="stat-value">48</span>
                        <span className="stat-label">Phases Complete</span>
                    </div>
                </div>
                <div className="stat-card">
                    <Activity size={20} color="#00d4ff" />
                    <div>
                        <span className="stat-value">11</span>
                        <span className="stat-label">Dashboard Views</span>
                    </div>
                </div>
                <div className="stat-card">
                    <Settings size={20} color="#a855f7" />
                    <div>
                        <span className="stat-value">LLM</span>
                        <span className="stat-label">Config Ready</span>
                    </div>
                </div>
            </div>

            <style>{`
        .dashboard-overview { padding: 30px; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        
        .hero-section { background: linear-gradient(135deg, #111 0%, #1a1a1f 100%); border: 1px solid #222; border-radius: 16px; padding: 40px; margin-bottom: 30px; text-align: center; }
        .hero-content h1 { font-size: 2rem; margin: 0 0 10px; background: linear-gradient(90deg, #00ff88, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .hero-content p { color: #888; margin: 0 0 20px; }
        
        .status-bar { display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; }
        .status-pill, .stat-pill { display: flex; align-items: center; gap: 8px; padding: 8px 16px; border-radius: 20px; font-size: 0.75rem; font-weight: bold; }
        .status-pill.online { background: rgba(0,255,136,0.15); color: #00ff88; }
        .status-pill.offline { background: rgba(255,68,68,0.15); color: #ff4444; }
        .stat-pill { background: #222; color: #888; }
        
        .views-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .view-card { display: flex; align-items: center; gap: 15px; padding: 20px; background: #111; border: 1px solid #222; border-radius: 12px; cursor: pointer; transition: all 0.2s; }
        .view-card:hover { border-color: var(--accent); transform: translateY(-2px); box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
        .view-icon { padding: 12px; border-radius: 10px; }
        .view-info { flex: 1; }
        .view-info h3 { margin: 0 0 4px; font-size: 0.95rem; color: #eee; }
        .view-info p { margin: 0; font-size: 0.7rem; color: #666; }
        .arrow { color: #444; transition: 0.2s; }
        .view-card:hover .arrow { color: var(--accent); transform: translateX(5px); }
        
        .quick-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-card { display: flex; align-items: center; gap: 15px; padding: 20px; background: #111; border: 1px solid #222; border-radius: 12px; }
        .stat-value { display: block; font-size: 1.5rem; font-weight: bold; color: #eee; }
        .stat-label { font-size: 0.7rem; color: #666; }
      `}</style>
        </div>
    );
};

export default DashboardOverview;
