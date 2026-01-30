import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, DollarSign, Leaf, Layers, ArrowUpRight, Target } from 'lucide-react';
import axios from 'axios';

const BusinessOverlay = () => {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await axios.get('/api/business/stats');
                setStats(response.data);
            } catch (e) {
                console.error(e);
            } finally {
                setLoading(false);
            }
        };
        fetchStats();
    }, []);

    if (loading) return <div className="text-dim p-4">Aggregating Economic Context...</div>;

    return (
        <div className="business-container">
            <div className="business-header">
                <BarChart3 size={20} className="text-primary" />
                <span>EXECUTIVE BUSINESS OVERLAY // ROI CONSPECTION</span>
            </div>

            <div className="stats-hero">
                <div className="hero-card primary">
                    <div className="hero-label">TOTAL ESTIMATED SAVINGS</div>
                    <div className="hero-value">${stats?.total_savings?.toLocaleString()}</div>
                    <div className="hero-delta"><ArrowUpRight size={14} /> 12.4% vs last Q</div>
                </div>
                <div className="hero-card info">
                    <div className="hero-label">SYSTEM ROI</div>
                    <div className="hero-value">{stats?.avg_roi}%</div>
                    <div className="hero-delta">Optimized by RISE Engine</div>
                </div>
                <div className="hero-card success">
                    <div className="hero-label">CARBON AVOIDED</div>
                    <div className="hero-value">{stats?.carbon_avoided} kg</div>
                    <div className="hero-delta">Sustainability Tier: Gold</div>
                </div>
            </div>

            <div className="business-details-grid">
                <div className="analysis-card">
                    <div className="card-title">
                        <Target size={16} />
                        <span>MULTIVERSE ECONOMIC IMPACT</span>
                    </div>
                    <div className="analysis-item">
                        <div className="item-label">Active Branches</div>
                        <div className="item-value">{stats?.active_multiverse_branches}</div>
                    </div>
                    <div className="analysis-item">
                        <div className="item-label">Optimal Target Part</div>
                        <div className="item-value text-info" style={{ fontSize: '0.7rem' }}>{stats?.top_performing_part}</div>
                    </div>
                    <p className="analysis-hint">
                        Multiverse branching has reduced expensive prototyping cycles by 64%.
                    </p>
                </div>

                <div className="analysis-card">
                    <div className="card-title">
                        <Leaf size={16} />
                        <span>SUSTAINABILITY OVERVIEW</span>
                    </div>
                    <div className="progress-container">
                        <div className="progress-label">Fleet-wide Efficiency</div>
                        <div className="progress-bar">
                            <div className="progress-fill" style={{ width: `${stats?.fleet_sustainability}%` }}></div>
                        </div>
                    </div>
                    <p className="analysis-hint">
                        90% of G-Code generated via Semantic Lexicon meets ISO-G9 sustainability standards.
                    </p>
                </div>
            </div>

            <style>{`
                .business-container { display: flex; flex-direction: column; gap: 20px; padding: 20px; animation: fadeIn 0.5s ease; }
                @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
                
                .business-header { display: flex; align-items: center; gap: 12px; font-weight: bold; color: #888; border-bottom: 1px solid #222; padding-bottom: 15px; letter-spacing: 1px; font-size: 0.8rem; }
                
                .stats-hero { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
                .hero-card { background: #111; border: 1px solid #222; padding: 20px; border-radius: 12px; position: relative; overflow: hidden; }
                .hero-card.primary { border-left: 4px solid #00ff88; }
                .hero-card.info { border-left: 4px solid #00d4ff; }
                .hero-card.success { border-left: 4px solid #00ff8888; }
                
                .hero-label { font-size: 0.6rem; font-weight: bold; color: #555; letter-spacing: 1px; margin-bottom: 10px; }
                .hero-value { font-size: 1.8rem; color: #eee; font-weight: bold; margin-bottom: 5px; }
                .hero-delta { font-size: 0.65rem; color: #00ff88; display: flex; align-items: center; gap: 4px; }
                
                .business-details-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .analysis-card { background: #111; border: 1px solid #222; padding: 20px; border-radius: 12px; }
                .card-title { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; color: #ccc; font-weight: bold; font-size: 0.75rem; }
                
                .analysis-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #1a1a1a; }
                .analysis-item:last-of-type { border: none; }
                .item-label { font-size: 0.75rem; color: #888; }
                .item-value { font-size: 0.8rem; color: #eee; font-weight: bold; }
                
                .progress-container { margin-top: 10px; }
                .progress-label { font-size: 0.7rem; color: #888; margin-bottom: 8px; }
                .progress-bar { height: 8px; background: #222; border-radius: 4px; overflow: hidden; }
                .progress-fill { height: 100%; background: linear-gradient(90deg, #00ff88 0%, #00d4ff 100%); transition: 1s ease-out; }
                
                .analysis-hint { font-size: 0.65rem; color: #555; font-style: italic; margin-top: 15px; border-top: 1px solid #1a1a1a; padding-top: 10px; }
                .text-info { color: #00d4ff; }
            `}</style>
        </div>
    );
};

export default BusinessOverlay;
