import React from 'react';
import { TrendingUp, TrendingDown, Minus, ArrowUp, ArrowDown } from 'lucide-react';

const MetricCard = ({ title, value, unit = '', trend = 0, icon: IconComponent, color = '#00ff88', description = '' }) => {
    const getTrendIcon = () => {
        if (trend > 0) return <TrendingUp size={14} className="trend-up" />;
        if (trend < 0) return <TrendingDown size={14} className="trend-down" />;
        return <Minus size={14} className="trend-neutral" />;
    };

    const getTrendLabel = () => {
        if (trend > 0) return `+${trend}%`;
        if (trend < 0) return `${trend}%`;
        return '0%';
    };

    return (
        <div className="metric-card" style={{ '--accent': color }}>
            <div className="metric-header">
                {IconComponent && (
                    <div className="metric-icon" style={{ background: color + '22', color: color }}>
                        <IconComponent size={20} />
                    </div>
                )}
                <span className="metric-title">{title}</span>
            </div>

            <div className="metric-body">
                <span className="metric-value">{value}</span>
                {unit && <span className="metric-unit">{unit}</span>}
            </div>

            <div className="metric-footer">
                <div className="trend-indicator">
                    {getTrendIcon()}
                    <span className={`trend-label ${trend > 0 ? 'up' : trend < 0 ? 'down' : ''}`}>{getTrendLabel()}</span>
                </div>
                {description && <span className="metric-desc">{description}</span>}
            </div>

            <style>{`
        .metric-card { background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px; transition: 0.2s; }
        .metric-card:hover { border-color: var(--accent); transform: translateY(-2px); }
        
        .metric-header { display: flex; align-items: center; gap: 12px; margin-bottom: 15px; }
        .metric-icon { padding: 10px; border-radius: 10px; }
        .metric-title { font-size: 0.75rem; color: #888; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        
        .metric-body { display: flex; align-items: baseline; gap: 6px; margin-bottom: 10px; }
        .metric-value { font-size: 2rem; font-weight: bold; color: #eee; }
        .metric-unit { font-size: 0.9rem; color: #666; }
        
        .metric-footer { display: flex; align-items: center; justify-content: space-between; }
        .trend-indicator { display: flex; align-items: center; gap: 5px; }
        .trend-up { color: #00ff88; }
        .trend-down { color: #ff4444; }
        .trend-neutral { color: #888; }
        .trend-label { font-size: 0.75rem; font-weight: bold; }
        .trend-label.up { color: #00ff88; }
        .trend-label.down { color: #ff4444; }
        .metric-desc { font-size: 0.65rem; color: #555; }
      `}</style>
        </div>
    );
};

export default MetricCard;
