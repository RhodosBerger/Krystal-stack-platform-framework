import React, { useState, useEffect } from 'react';
import { Layers, Cpu, GitBranch, Play, PlusCircle, CheckCircle2, Activity } from 'lucide-react';
import axios from 'axios';

const PlatformHub = () => {
    const [entities, setEntities] = useState([]);
    const [newName, setNewName] = useState('');
    const [loading, setLoading] = useState(true);
    const [pipelineResult, setPipelineResult] = useState(null);

    useEffect(() => {
        fetchEntities();
    }, []);

    const fetchEntities = async () => {
        setLoading(true);
        try {
            const res = await axios.get('/api/platform/entities');
            setEntities(res.data.entities || []);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    const createEntity = async () => {
        if (!newName.trim()) return;
        try {
            await axios.post('/api/platform/entity', { name: newName, entity_type: 'GENERATION_ASSET' });
            setNewName('');
            fetchEntities();
        } catch (e) { console.error(e); }
    };

    const runPipeline = async (entityId) => {
        try {
            const res = await axios.post('/api/platform/run-pipeline', {
                entity_id: entityId,
                data: 'G01 X10 Y20 Z5 F500'
            });
            setPipelineResult(res.data.result);
            fetchEntities();
        } catch (e) { console.error(e); }
    };

    return (
        <div className="platform-hub">
            <div className="section-header">
                <Layers size={20} className="text-primary" />
                <span>GENERATION PLATFORM // CORE STRUCTURES</span>
            </div>

            <div className="platform-grid">
                {/* Entity Creator */}
                <div className="platform-card creator-card">
                    <div className="card-header"><PlusCircle size={16} /> CREATE NEW ENTITY</div>
                    <input
                        type="text"
                        placeholder="Entity Name (e.g., Engine Block V5)"
                        value={newName}
                        onChange={(e) => setNewName(e.target.value)}
                    />
                    <button onClick={createEntity}>INITIALIZE ENTITY</button>
                </div>

                {/* Entity Registry */}
                <div className="platform-card registry-card">
                    <div className="card-header"><Cpu size={16} /> PLATFORM REGISTRY</div>
                    <div className="entity-list">
                        {loading ? <div className="loading">Loading...</div> : entities.length === 0 ? (
                            <div className="empty">No entities registered. Create one to begin.</div>
                        ) : (
                            entities.map(e => (
                                <div key={e.id} className="entity-item">
                                    <div className="entity-info">
                                        <span className="name">{e.name}</span>
                                        <span className="status">{e.status}</span>
                                    </div>
                                    <button className="run-btn" onClick={() => runPipeline(e.id)}>
                                        <Play size={14} /> Run Pipeline
                                    </button>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Pipeline Result */}
                {pipelineResult && (
                    <div className="platform-card result-card">
                        <div className="card-header"><CheckCircle2 size={16} className="text-success" /> PIPELINE RESULT</div>
                        <div className="result-data">
                            <pre>{pipelineResult.final_output}</pre>
                        </div>
                    </div>
                )}
            </div>

            <style>{`
        .platform-hub { display: flex; flex-direction: column; gap: 25px; padding: 20px; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .section-header { display: flex; align-items: center; gap: 12px; font-weight: bold; border-bottom: 1px solid #222; padding-bottom: 15px; color: #888; letter-spacing: 1.5px; }
        .platform-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }
        .platform-card { background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px; }
        .card-header { display: flex; align-items: center; gap: 10px; font-size: 0.8rem; font-weight: bold; color: #ccc; margin-bottom: 15px; }
        .creator-card input { width: 100%; background: #1a1a1f; border: 1px solid #333; color: #eee; padding: 10px; border-radius: 6px; margin-bottom: 10px; }
        .creator-card button { width: 100%; background: #00ff88; color: #000; border: none; padding: 12px; border-radius: 6px; font-weight: bold; cursor: pointer; }
        .entity-list { display: flex; flex-direction: column; gap: 10px; max-height: 300px; overflow-y: auto; }
        .entity-item { display: flex; justify-content: space-between; align-items: center; background: #1a1a1f; padding: 12px; border-radius: 8px; }
        .entity-info { display: flex; flex-direction: column; }
        .entity-info .name { font-size: 0.85rem; font-weight: bold; color: #eee; }
        .entity-info .status { font-size: 0.65rem; color: #00ff88; }
        .run-btn { background: #222; border: 1px solid #444; color: #ccc; padding: 6px 12px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 6px; font-size: 0.7rem; }
        .run-btn:hover { background: #00ff88; color: #000; border-color: #00ff88; }
        .result-card { grid-column: 1 / -1; }
        .result-card pre { background: #000; padding: 15px; border-radius: 8px; font-size: 0.75rem; color: #00ff88; overflow-x: auto; }
        .empty, .loading { font-size: 0.75rem; color: #555; text-align: center; padding: 20px; }
        .text-success { color: #00ff88; }
      `}</style>
        </div>
    );
};

export default PlatformHub;
