import React, { useState, useEffect } from 'react';
import { Rocket, Star, Zap, RefreshCw, ChevronRight, Sparkles, Target, Clock, Brain, Layers } from 'lucide-react';
import axios from 'axios';

const PossibilitiesExplorer = () => {
    const [possibilities, setPossibilities] = useState([]);
    const [recommended, setRecommended] = useState([]);
    const [quickWins, setQuickWins] = useState([]);
    const [randomIdea, setRandomIdea] = useState(null);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('recommended');

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        try {
            const [allRes, recRes, qwRes] = await Promise.all([
                axios.get('/api/possibilities'),
                axios.get('/api/possibilities/recommended'),
                axios.get('/api/possibilities/quick-wins')
            ]);
            setPossibilities(allRes.data?.possibilities || []);
            setRecommended(recRes.data?.recommended || []);
            setQuickWins(qwRes.data?.quick_wins || []);
        } catch (e) {
            // Demo data fallback
            const demo = [
                { id: 'P001', name: 'Voice Commands', category: 'AI-Native Manufacturing', priority: 'HIGH', effort: 'medium', description: 'Control CNC Copilot with natural voice commands' },
                { id: 'P002', name: 'Achievement System', category: 'Immersive Experience', priority: 'HIGH', effort: 'low', description: 'Gamification with badges and leaderboards' },
                { id: 'P003', name: 'Webhook System', category: 'Quick Win', priority: 'HIGH', effort: 'low', description: 'Event notifications to external services' }
            ];
            setPossibilities(demo);
            setRecommended(demo);
            setQuickWins(demo.filter(d => d.effort === 'low'));
        }
        setLoading(false);
    };

    const generateRandomIdea = async () => {
        try {
            const res = await axios.get('/api/possibilities/random');
            setRandomIdea(res.data?.idea);
        } catch (e) {
            setRandomIdea({ idea: 'AI-Powered Quality Inspection', category: 'AI-Native Manufacturing', generated: true });
        }
    };

    const getCategoryIcon = (cat) => {
        if (cat.includes('AI')) return <Brain size={16} />;
        if (cat.includes('Connected')) return <Layers size={16} />;
        if (cat.includes('Immersive')) return <Star size={16} />;
        if (cat.includes('Quick')) return <Zap size={16} />;
        return <Rocket size={16} />;
    };

    const getPriorityColor = (priority) => {
        if (priority === 'CRITICAL') return '#ff4444';
        if (priority === 'HIGH') return '#00ff88';
        if (priority === 'MEDIUM') return '#ffaa00';
        return '#888';
    };

    const tabs = [
        { id: 'recommended', label: 'Recommended', icon: Target },
        { id: 'quick-wins', label: 'Quick Wins', icon: Zap },
        { id: 'all', label: 'All Ideas', icon: Layers }
    ];

    const currentList = activeTab === 'recommended' ? recommended : activeTab === 'quick-wins' ? quickWins : possibilities;

    return (
        <div className="possibilities-explorer">
            <div className="explorer-header">
                <div className="header-title">
                    <Rocket size={24} className="rocket-icon" />
                    <div>
                        <h1>NEW GENERATION POSSIBILITIES</h1>
                        <p>Explore the future of CNC manufacturing</p>
                    </div>
                </div>
                <button className="inspire-btn" onClick={generateRandomIdea}>
                    <Sparkles size={16} /> Generate Idea
                </button>
            </div>

            {randomIdea && (
                <div className="random-idea">
                    <Sparkles size={18} />
                    <div className="idea-content">
                        <span className="idea-label">Random Inspiration:</span>
                        <span className="idea-text">{randomIdea.idea}</span>
                        <span className="idea-category">{randomIdea.category}</span>
                    </div>
                </div>
            )}

            <div className="tab-bar">
                {tabs.map(tab => {
                    const IconComponent = tab.icon;
                    return (
                        <button
                            key={tab.id}
                            className={activeTab === tab.id ? 'active' : ''}
                            onClick={() => setActiveTab(tab.id)}
                        >
                            <IconComponent size={14} />
                            {tab.label}
                        </button>
                    );
                })}
            </div>

            <div className="possibilities-grid">
                {loading ? (
                    <div className="loading">Loading possibilities...</div>
                ) : (
                    currentList.map(p => (
                        <div key={p.id} className="possibility-card">
                            <div className="card-header">
                                <div className="cat-icon" style={{ color: getPriorityColor(p.priority) }}>
                                    {getCategoryIcon(p.category)}
                                </div>
                                <div className="card-title">
                                    <h3>{p.name}</h3>
                                    <span className="category">{p.category}</span>
                                </div>
                                <div className="priority-badge" style={{ background: getPriorityColor(p.priority) + '22', color: getPriorityColor(p.priority) }}>
                                    {p.priority}
                                </div>
                            </div>
                            <p className="description">{p.description}</p>
                            <div className="card-footer">
                                <span className="effort">Effort: <strong>{p.effort}</strong></span>
                                <ChevronRight size={16} className="arrow" />
                            </div>
                        </div>
                    ))
                )}
            </div>

            <style>{`
        .possibilities-explorer { background: #0a0a0f; min-height: 100vh; padding: 30px; }
        
        .explorer-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; }
        .header-title { display: flex; align-items: center; gap: 15px; }
        .rocket-icon { color: #00ff88; animation: float 3s ease-in-out infinite; }
        @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
        .header-title h1 { margin: 0; font-size: 1.5rem; color: #eee; }
        .header-title p { margin: 0; font-size: 0.8rem; color: #666; }
        
        .inspire-btn { display: flex; align-items: center; gap: 8px; background: linear-gradient(135deg, #a855f7, #00d4ff); color: #fff; border: none; padding: 12px 20px; border-radius: 10px; font-weight: bold; cursor: pointer; }
        
        .random-idea { display: flex; align-items: center; gap: 15px; padding: 15px 20px; background: linear-gradient(90deg, #a855f722, #00d4ff22); border: 1px solid #a855f7; border-radius: 12px; margin-bottom: 25px; color: #a855f7; }
        .idea-content { display: flex; flex-direction: column; }
        .idea-label { font-size: 0.65rem; color: #888; }
        .idea-text { font-size: 1.1rem; font-weight: bold; color: #eee; }
        .idea-category { font-size: 0.7rem; color: #00d4ff; }
        
        .tab-bar { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab-bar button { display: flex; align-items: center; gap: 6px; background: #1a1a1f; border: 1px solid #222; color: #666; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 0.8rem; }
        .tab-bar button.active { background: #00ff8822; color: #00ff88; border-color: #00ff88; }
        
        .possibilities-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 15px; }
        .possibility-card { background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px; transition: 0.3s; }
        .possibility-card:hover { border-color: #00ff88; transform: translateY(-3px); }
        
        .card-header { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px; }
        .cat-icon { padding: 8px; background: #1a1a1f; border-radius: 8px; }
        .card-title { flex: 1; }
        .card-title h3 { margin: 0 0 4px; font-size: 1rem; color: #eee; }
        .category { font-size: 0.65rem; color: #666; }
        .priority-badge { padding: 4px 10px; border-radius: 4px; font-size: 0.65rem; font-weight: bold; }
        
        .description { font-size: 0.8rem; color: #888; margin-bottom: 15px; line-height: 1.5; }
        
        .card-footer { display: flex; justify-content: space-between; align-items: center; padding-top: 12px; border-top: 1px solid #222; }
        .effort { font-size: 0.7rem; color: #555; }
        .effort strong { color: #888; }
        .arrow { color: #333; }
        .possibility-card:hover .arrow { color: #00ff88; transform: translateX(5px); }
        
        .loading { text-align: center; padding: 50px; color: #666; }
      `}</style>
        </div>
    );
};

export default PossibilitiesExplorer;
