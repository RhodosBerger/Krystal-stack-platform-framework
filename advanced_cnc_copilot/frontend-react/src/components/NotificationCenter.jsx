import React, { useState, useEffect } from 'react';
import { Bell, Check, Trash2, AlertTriangle, AlertCircle, Info, CheckCircle, RefreshCw } from 'lucide-react';
import axios from 'axios';

const iconMap = {
    AlertTriangle: AlertTriangle,
    AlertCircle: AlertCircle,
    Info: Info,
    CheckCircle: CheckCircle,
    Bell: Bell
};

const NotificationCenter = () => {
    const [notifications, setNotifications] = useState([]);
    const [unreadCount, setUnreadCount] = useState(0);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchNotifications();
        // Poll every 30 seconds
        const interval = setInterval(fetchNotifications, 30000);
        return () => clearInterval(interval);
    }, []);

    const fetchNotifications = async () => {
        try {
            const res = await axios.get('/api/notifications?limit=50&include_read=true');
            setNotifications(res.data.notifications || []);
            setUnreadCount(res.data.unread_count || 0);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    const markAsRead = async (id) => {
        try {
            await axios.post(`/api/notifications/read/${id}`);
            fetchNotifications();
        } catch (e) { console.error(e); }
    };

    const markAllRead = async () => {
        try {
            await axios.post('/api/notifications/read-all');
            fetchNotifications();
        } catch (e) { console.error(e); }
    };

    const getIcon = (priority) => {
        const icons = {
            CRITICAL: AlertTriangle,
            WARNING: AlertCircle,
            INFO: Info,
            SUCCESS: CheckCircle
        };
        return icons[priority] || Bell;
    };

    const formatTime = (isoString) => {
        const date = new Date(isoString);
        const now = new Date();
        const diff = (now - date) / 1000 / 60; // minutes
        if (diff < 1) return 'Just now';
        if (diff < 60) return `${Math.floor(diff)}m ago`;
        if (diff < 1440) return `${Math.floor(diff / 60)}h ago`;
        return date.toLocaleDateString();
    };

    return (
        <div className="notification-center">
            <div className="section-header">
                <Bell size={20} className="text-primary" />
                <span>NOTIFICATION CENTER</span>
                <div className="header-actions">
                    {unreadCount > 0 && (
                        <span className="unread-badge">{unreadCount} unread</span>
                    )}
                    <button className="action-btn" onClick={fetchNotifications}><RefreshCw size={14} /></button>
                    <button className="action-btn" onClick={markAllRead}><Check size={14} /> Mark All Read</button>
                </div>
            </div>

            <div className="notification-list">
                {loading ? (
                    <div className="loading-state">Loading notifications...</div>
                ) : notifications.length === 0 ? (
                    <div className="empty-state">No notifications. You're all caught up! 🎉</div>
                ) : (
                    notifications.map(n => {
                        const IconComponent = getIcon(n.priority);
                        return (
                            <div key={n.id} className={`notification-item ${n.read ? 'read' : 'unread'}`}>
                                <div className="notif-icon" style={{ color: n.color }}>
                                    <IconComponent size={20} />
                                </div>
                                <div className="notif-content">
                                    <div className="notif-header">
                                        <span className="notif-title">{n.title}</span>
                                        <span className="notif-time">{formatTime(n.created_at)}</span>
                                    </div>
                                    <div className="notif-message">{n.message}</div>
                                    <div className="notif-meta">
                                        <span className="category-badge">{n.category}</span>
                                        <span className="priority-badge" style={{ backgroundColor: n.color + '22', color: n.color }}>{n.priority}</span>
                                    </div>
                                </div>
                                {!n.read && (
                                    <button className="mark-read-btn" onClick={() => markAsRead(n.id)}>
                                        <Check size={14} />
                                    </button>
                                )}
                            </div>
                        );
                    })
                )}
            </div>

            <style>{`
        .notification-center { display: flex; flex-direction: column; gap: 20px; padding: 20px; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .section-header { display: flex; align-items: center; gap: 12px; font-weight: bold; border-bottom: 1px solid #222; padding-bottom: 15px; color: #888; letter-spacing: 1.5px; }
        .header-actions { margin-left: auto; display: flex; align-items: center; gap: 10px; }
        .unread-badge { background: #ff4444; color: #fff; padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: bold; }
        .action-btn { background: #222; border: 1px solid #333; color: #888; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.7rem; display: flex; align-items: center; gap: 5px; }
        .action-btn:hover { color: #00ff88; border-color: #00ff88; }
        .notification-list { display: flex; flex-direction: column; gap: 10px; max-height: calc(100vh - 200px); overflow-y: auto; }
        .notification-item { display: flex; align-items: flex-start; gap: 15px; padding: 15px; background: #111; border: 1px solid #222; border-radius: 10px; transition: 0.2s; }
        .notification-item.unread { background: #1a1a1f; border-left: 3px solid #00ff88; }
        .notification-item.read { opacity: 0.6; }
        .notif-icon { flex-shrink: 0; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 8px; }
        .notif-content { flex: 1; }
        .notif-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
        .notif-title { font-weight: bold; font-size: 0.9rem; color: #eee; }
        .notif-time { font-size: 0.65rem; color: #666; }
        .notif-message { font-size: 0.8rem; color: #aaa; margin-bottom: 8px; }
        .notif-meta { display: flex; gap: 8px; }
        .category-badge { font-size: 0.6rem; background: #333; color: #888; padding: 3px 8px; border-radius: 4px; }
        .priority-badge { font-size: 0.6rem; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
        .mark-read-btn { background: #00ff88; color: #000; border: none; padding: 6px; border-radius: 6px; cursor: pointer; }
        .loading-state, .empty-state { text-align: center; color: #555; padding: 50px; }
      `}</style>
        </div>
    );
};

export default NotificationCenter;
