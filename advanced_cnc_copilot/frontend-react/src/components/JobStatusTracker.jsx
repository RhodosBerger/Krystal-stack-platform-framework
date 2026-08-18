import React, { useState, useEffect } from 'react';
import { Activity, Clock, CheckCircle2, XCircle, Loader2, RefreshCw, Eye } from 'lucide-react';
import axios from 'axios';

const JobStatusTracker = ({ limit = 10 }) => {
    const [jobs, setJobs] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchJobs();
        const interval = setInterval(fetchJobs, 10000);
        return () => clearInterval(interval);
    }, []);

    const fetchJobs = async () => {
        try {
            const res = await axios.get(`/api/jobs?limit=${limit}`);
            setJobs(res.data?.jobs || []);
        } catch (e) {
            // Demo jobs
            setJobs([
                { id: 'JOB-001', type: 'G-Code Generation', status: 'COMPLETED', created_at: new Date().toISOString() },
                { id: 'JOB-002', type: 'Optimization', status: 'RUNNING', created_at: new Date().toISOString() },
                { id: 'JOB-003', type: 'Export', status: 'PENDING', created_at: new Date().toISOString() }
            ]);
        }
        setLoading(false);
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'COMPLETED': return <CheckCircle2 size={14} className="status-icon completed" />;
            case 'FAILED': return <XCircle size={14} className="status-icon failed" />;
            case 'RUNNING': return <Loader2 size={14} className="status-icon running spin" />;
            default: return <Clock size={14} className="status-icon pending" />;
        }
    };

    const formatTime = (isoString) => {
        const date = new Date(isoString);
        const now = new Date();
        const diff = (now - date) / 1000 / 60;
        if (diff < 1) return 'Just now';
        if (diff < 60) return `${Math.floor(diff)}m ago`;
        return date.toLocaleTimeString();
    };

    return (
        <div className="job-tracker">
            <div className="tracker-header">
                <Activity size={16} className="text-primary" />
                <span>JOB QUEUE</span>
                <button className="refresh-btn" onClick={fetchJobs}><RefreshCw size={14} /></button>
            </div>

            <div className="jobs-list">
                {loading ? (
                    <div className="loading-state">Loading jobs...</div>
                ) : jobs.length === 0 ? (
                    <div className="empty-state">No jobs in queue.</div>
                ) : (
                    jobs.map(job => (
                        <div key={job.id} className={`job-item ${job.status.toLowerCase()}`}>
                            <div className="job-info">
                                {getStatusIcon(job.status)}
                                <div className="job-details">
                                    <span className="job-id">{job.id}</span>
                                    <span className="job-type">{job.type}</span>
                                </div>
                            </div>
                            <div className="job-meta">
                                <span className="job-time">{formatTime(job.created_at)}</span>
                                <button className="view-btn"><Eye size={12} /></button>
                            </div>
                        </div>
                    ))
                )}
            </div>

            <style>{`
        .job-tracker { background: #111; border: 1px solid #222; border-radius: 12px; overflow: hidden; }
        .tracker-header { display: flex; align-items: center; gap: 10px; padding: 15px; background: #0a0a0f; border-bottom: 1px solid #222; font-weight: bold; font-size: 0.8rem; color: #888; }
        .refresh-btn { margin-left: auto; background: #222; border: 1px solid #333; color: #888; padding: 4px; border-radius: 4px; cursor: pointer; }
        
        .jobs-list { max-height: 300px; overflow-y: auto; }
        .job-item { display: flex; justify-content: space-between; align-items: center; padding: 12px 15px; border-bottom: 1px solid #1a1a1f; transition: 0.2s; }
        .job-item:hover { background: #1a1a1f; }
        .job-info { display: flex; align-items: center; gap: 12px; }
        .job-details { display: flex; flex-direction: column; }
        .job-id { font-size: 0.75rem; font-weight: bold; color: #eee; }
        .job-type { font-size: 0.65rem; color: #666; }
        .job-meta { display: flex; align-items: center; gap: 10px; }
        .job-time { font-size: 0.65rem; color: #555; }
        .view-btn { background: none; border: 1px solid #333; color: #666; padding: 4px 8px; border-radius: 4px; cursor: pointer; }
        
        .status-icon.completed { color: #00ff88; }
        .status-icon.failed { color: #ff4444; }
        .status-icon.running { color: #00d4ff; }
        .status-icon.pending { color: #888; }
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        
        .loading-state, .empty-state { text-align: center; padding: 30px; color: #555; font-size: 0.8rem; }
      `}</style>
        </div>
    );
};

export default JobStatusTracker;
