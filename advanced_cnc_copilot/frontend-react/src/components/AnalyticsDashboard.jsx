import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, Zap, Clock, CheckCircle2, Play, RefreshCw } from 'lucide-react';
import axios from 'axios';

const AnalyticsDashboard = () => {
    const [metrics, setMetrics] = useState(null);
    const [timeline, setTimeline] = useState([]);
    const [workflows, setWorkflows] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 10000); // Auto-refresh every 10s
        return () => clearInterval(interval);
    }, []);

    const fetchData = async () => {
        try {
            const [metricsRes, timelineRes, workflowsRes] = await Promise.all([
                axios.get('/api/analytics/metrics'),
                axios.get('/api/analytics/timeline?limit=10'),
                axios.get('/api/workflows')
            ]);
            setMetrics(metricsRes.data.metrics);
            setTimeline(timelineRes.data.timeline || []);
            setWorkflows(workflowsRes.data.workflows || []);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    const runWorkflow = async (workflowId) => {
        try {
            await axios.post('/api/workflow/run', { workflow_id: workflowId, context: {} });
            fetchData();
        } catch (e) { console.error(e); }
    };

    if (loading) return <div className="loading-state">Loading Analytics...</div>;

    return (
        <div className="analytics-dashboard">
            <div className="section-header">
                <Activity size={20} className="text-primary" />
                <span>REAL-TIME ANALYTICS & WORKFLOW AUTOMATION</span>
                <button className="refresh-btn" onClick={fetchData}><RefreshCw size={14} /></button>
            </div>

            {/* Metrics Hero */}
            <div className="metrics-grid">
                <div className="metric-card">
                    <div className="metric-icon"><Zap size={24} /></div>
                    <div className="metric-value">{metrics?.total_generations || 0}</div>
                    <div className="metric-label">Total Generations</div>
                </div>
                <div className="metric-card">
                    <div className="metric-icon"><TrendingUp size={24} /></div>
                    <div className="metric-value">{metrics?.total_payloads || 0}</div>
                    <div className="metric-label">Payloads Emitted</div>
                </div>
                <div className="metric-card">
                    <div className="metric-icon"><Clock size={24} /></div>
                    <div className="metric-value">{Math.round(metrics?.avg_generation_time_ms || 0)}ms</div>
                    <div className="metric-label">Avg Generation Time</div>
                </div>
                <div className="metric-card success">
                    <div className="metric-icon"><CheckCircle2 size={24} /></div>
                    <div className="metric-value">{metrics?.success_rate?.toFixed(1) || 100}%</div>
                    <div className="metric-label">Success Rate</div>
                </div>
            </div>

            <div className="dashboard-grid">
                {/* Event Timeline */}
                <div className="dashboard-card timeline-card">
                    <div className="card-header"><Clock size={16} /> EVENT TIMELINE</div>
                    <div className="timeline-list">
                        {timeline.length === 0 ? (
                            <div className="empty">No events recorded yet.</div>
                        ) : (
                            timeline.slice().reverse().map((evt, idx) => (
                                <div key={idx} className={`timeline-item ${evt.success ? 'success' : 'fail'}`}>
                                    <div className="event-dot"></div>
                                    <div className="event-info">
                                        <span className="event-type">{evt.event}</span>
                                        <span className="event-meta">{evt.payload_count} payloads • {evt.duration_ms}ms</span>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Workflow Automation */}
                <div className="dashboard-card workflow-card">
                    <div className="card-header"><Play size={16} /> WORKFLOW AUTOMATION</div>
                    <div className="workflow-list">
                        {workflows.map(wf => (
                            <div key={wf.id} className="workflow-item">
                                <div className="wf-info">
                                    <div className="wf-name">{wf.name}</div>
                                    <div className="wf-desc">{wf.description}</div>
                                    <div className="wf-steps">{wf.steps.length} steps • {wf.status}</div>
                                </div>
                                <button className="run-wf-btn" onClick={() => runWorkflow(wf.id)}>
                                    <Play size={14} /> RUN
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <style>{`
        .analytics-dashboard { display: flex; flex-direction: column; gap: 25px; padding: 20px; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .section-header { display: flex; align-items: center; gap: 12px; font-weight: bold; border-bottom: 1px solid #222; padding-bottom: 15px; color: #888; letter-spacing: 1.5px; }
        .refresh-btn { margin-left: auto; background: #222; border: 1px solid #333; color: #888; padding: 8px; border-radius: 6px; cursor: pointer; }
        .refresh-btn:hover { color: #00ff88; border-color: #00ff88; }
        .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
        .metric-card { background: #111; border: 1px solid #222; border-radius: 12px; padding: 25px; text-align: center; }
        .metric-card.success { border-color: #00ff8844; }
        .metric-icon { color: #00d4ff; margin-bottom: 15px; }
        .metric-card.success .metric-icon { color: #00ff88; }
        .metric-value { font-size: 2rem; font-weight: bold; color: #eee; }
        .metric-label { font-size: 0.7rem; color: #666; margin-top: 8px; text-transform: uppercase; }
        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .dashboard-card { background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px; }
        .card-header { display: flex; align-items: center; gap: 10px; font-size: 0.8rem; font-weight: bold; color: #ccc; margin-bottom: 15px; }
        .timeline-list { display: flex; flex-direction: column; gap: 12px; max-height: 250px; overflow-y: auto; }
        .timeline-item { display: flex; align-items: center; gap: 12px; padding: 10px; background: #1a1a1f; border-radius: 8px; }
        .event-dot { width: 10px; height: 10px; border-radius: 50%; background: #00ff88; }
        .timeline-item.fail .event-dot { background: #ff4444; }
        .event-info { display: flex; flex-direction: column; }
        .event-type { font-size: 0.8rem; font-weight: bold; color: #eee; }
        .event-meta { font-size: 0.65rem; color: #666; }
        .workflow-list { display: flex; flex-direction: column; gap: 12px; }
        .workflow-item { display: flex; justify-content: space-between; align-items: center; padding: 15px; background: #1a1a1f; border-radius: 8px; }
        .wf-info { display: flex; flex-direction: column; }
        .wf-name { font-size: 0.9rem; font-weight: bold; color: #eee; }
        .wf-desc { font-size: 0.7rem; color: #888; margin: 4px 0; }
        .wf-steps { font-size: 0.65rem; color: #00d4ff; }
        .run-wf-btn { background: #00ff88; color: #000; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold; display: flex; align-items: center; gap: 6px; font-size: 0.75rem; }
        .run-wf-btn:hover { transform: scale(1.05); }
        .empty { text-align: center; color: #555; padding: 30px; }
        .loading-state { text-align: center; color: #555; padding: 50px; }
      `}</style>
        </div>
    );
};

export default AnalyticsDashboard;
