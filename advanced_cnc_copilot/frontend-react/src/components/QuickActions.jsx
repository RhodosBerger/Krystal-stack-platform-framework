import React from 'react';
import { Zap, FileCode, Upload, Download, RefreshCw, Settings, Bell } from 'lucide-react';

const QuickActions = ({ onAction }) => {
    const actions = [
        { id: 'generate', label: 'Generate G-Code', icon: FileCode, color: '#00ff88', desc: 'Create new machining program' },
        { id: 'export', label: 'Export Project', icon: Download, color: '#00d4ff', desc: 'Download as ZIP' },
        { id: 'import', label: 'Import Package', icon: Upload, color: '#a855f7', desc: 'Load from file' },
        { id: 'refresh', label: 'Refresh All', icon: RefreshCw, color: '#f97316', desc: 'Reload system data' },
        { id: 'settings', label: 'Quick Settings', icon: Settings, color: '#84cc16', desc: 'Adjust preferences' },
        { id: 'alerts', label: 'View Alerts', icon: Bell, color: '#ffaa00', desc: 'Check notifications' },
    ];

    return (
        <div className="quick-actions">
            <div className="actions-header">
                <Zap size={16} className="text-primary" />
                <span>QUICK ACTIONS</span>
            </div>
            <div className="actions-grid">
                {actions.map(a => {
                    const IconComponent = a.icon;
                    return (
                        <button
                            key={a.id}
                            className="action-card"
                            onClick={() => onAction && onAction(a.id)}
                            style={{ '--accent': a.color }}
                        >
                            <div className="action-icon" style={{ background: a.color + '22', color: a.color }}>
                                <IconComponent size={18} />
                            </div>
                            <div className="action-text">
                                <span className="action-label">{a.label}</span>
                                <span className="action-desc">{a.desc}</span>
                            </div>
                        </button>
                    );
                })}
            </div>

            <style>{`
        .quick-actions { background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px; }
        .actions-header { display: flex; align-items: center; gap: 10px; font-weight: bold; font-size: 0.8rem; color: #888; margin-bottom: 15px; letter-spacing: 1px; }
        .actions-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
        .action-card { display: flex; align-items: center; gap: 12px; padding: 15px; background: #1a1a1f; border: 1px solid #222; border-radius: 10px; cursor: pointer; transition: all 0.2s; text-align: left; }
        .action-card:hover { border-color: var(--accent); transform: translateY(-2px); }
        .action-icon { padding: 10px; border-radius: 8px; }
        .action-text { display: flex; flex-direction: column; }
        .action-label { font-size: 0.8rem; font-weight: bold; color: #eee; }
        .action-desc { font-size: 0.6rem; color: #666; }
      `}</style>
        </div>
    );
};

export default QuickActions;
