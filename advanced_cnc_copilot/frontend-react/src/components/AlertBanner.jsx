import React from 'react';
import { AlertTriangle, Info, CheckCircle, XCircle, X } from 'lucide-react';

const AlertBanner = ({ type = 'info', message, onDismiss, action, actionLabel }) => {
    const config = {
        info: { icon: Info, color: '#00d4ff', bg: 'rgba(0,212,255,0.1)' },
        success: { icon: CheckCircle, color: '#00ff88', bg: 'rgba(0,255,136,0.1)' },
        warning: { icon: AlertTriangle, color: '#ffaa00', bg: 'rgba(255,170,0,0.1)' },
        error: { icon: XCircle, color: '#ff4444', bg: 'rgba(255,68,68,0.1)' }
    };

    const { icon: IconComponent, color, bg } = config[type] || config.info;

    return (
        <div className="alert-banner" style={{ '--color': color, '--bg': bg }}>
            <div className="alert-content">
                <IconComponent size={18} className="alert-icon" />
                <span className="alert-message">{message}</span>
            </div>
            <div className="alert-actions">
                {action && actionLabel && (
                    <button className="action-btn" onClick={action}>{actionLabel}</button>
                )}
                {onDismiss && (
                    <button className="dismiss-btn" onClick={onDismiss}><X size={16} /></button>
                )}
            </div>

            <style>{`
        .alert-banner { display: flex; align-items: center; justify-content: space-between; padding: 12px 20px; background: var(--bg); border: 1px solid var(--color); border-radius: 10px; margin-bottom: 15px; }
        .alert-content { display: flex; align-items: center; gap: 12px; }
        .alert-icon { color: var(--color); flex-shrink: 0; }
        .alert-message { font-size: 0.85rem; color: #eee; }
        .alert-actions { display: flex; align-items: center; gap: 10px; }
        .action-btn { background: var(--color); color: #000; border: none; padding: 6px 14px; border-radius: 6px; font-size: 0.75rem; font-weight: bold; cursor: pointer; }
        .dismiss-btn { background: none; border: none; color: #666; cursor: pointer; padding: 4px; }
        .dismiss-btn:hover { color: #eee; }
      `}</style>
        </div>
    );
};

export default AlertBanner;
