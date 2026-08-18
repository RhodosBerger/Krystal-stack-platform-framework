import React, { useState, useEffect } from 'react';
import { Download, Upload, Package, FileArchive, CheckCircle2, AlertCircle, History } from 'lucide-react';
import axios from 'axios';

const DataHub = () => {
    const [exportHistory, setExportHistory] = useState([]);
    const [importHistory, setImportHistory] = useState([]);
    const [exporting, setExporting] = useState(false);
    const [importing, setImporting] = useState(false);
    const [lastExport, setLastExport] = useState(null);
    const [lastImport, setLastImport] = useState(null);
    const [projectName, setProjectName] = useState('');

    useEffect(() => {
        fetchHistory();
    }, []);

    const fetchHistory = async () => {
        try {
            const [expRes, impRes] = await Promise.all([
                axios.get('/api/export/history'),
                axios.get('/api/import/history')
            ]);
            setExportHistory(expRes.data.history || []);
            setImportHistory(impRes.data.history || []);
        } catch (e) { console.error(e); }
    };

    const handleExport = async () => {
        if (!projectName.trim()) return;
        setExporting(true);
        try {
            const res = await axios.post('/api/export/project', { project_name: projectName });
            setLastExport(res.data.export);

            // Trigger download
            const link = document.createElement('a');
            link.href = `data:application/zip;base64,${res.data.export.data}`;
            link.download = `${projectName.replace(/\s+/g, '_')}_export.zip`;
            link.click();

            fetchHistory();
        } catch (e) { console.error(e); }
        finally { setExporting(false); }
    };

    const handleImport = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setImporting(true);
        try {
            const reader = new FileReader();
            reader.onload = async (evt) => {
                const base64 = evt.target.result.split(',')[1];
                const res = await axios.post('/api/import/project', { data: base64 });
                setLastImport(res.data.import);
                fetchHistory();
                setImporting(false);
            };
            reader.readAsDataURL(file);
        } catch (e) {
            console.error(e);
            setImporting(false);
        }
    };

    return (
        <div className="data-hub">
            <div className="section-header">
                <Package size={20} className="text-primary" />
                <span>EXPORT / IMPORT HUB // DATA MIGRATION</span>
            </div>

            <div className="hub-grid">
                {/* Export Panel */}
                <div className="hub-card export-card">
                    <div className="card-header"><Download size={16} /> EXPORT PROJECT</div>
                    <input
                        type="text"
                        placeholder="Project Name"
                        value={projectName}
                        onChange={(e) => setProjectName(e.target.value)}
                        className="input-field"
                    />
                    <button className="action-btn export-btn" onClick={handleExport} disabled={exporting || !projectName.trim()}>
                        {exporting ? 'PACKAGING...' : 'EXPORT AS ZIP'}
                    </button>
                    {lastExport && (
                        <div className="result-box success">
                            <CheckCircle2 size={16} />
                            <div>
                                <div className="result-title">Export Complete</div>
                                <div className="result-meta">{lastExport.export_id} • {lastExport.size_bytes} bytes</div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Import Panel */}
                <div className="hub-card import-card">
                    <div className="card-header"><Upload size={16} /> IMPORT PROJECT</div>
                    <label className="file-input-label">
                        <FileArchive size={24} />
                        <span>{importing ? 'Importing...' : 'Select ZIP Package'}</span>
                        <input type="file" accept=".zip" onChange={handleImport} disabled={importing} />
                    </label>
                    {lastImport && (
                        <div className={`result-box ${lastImport.status === 'SUCCESS' ? 'success' : 'error'}`}>
                            {lastImport.status === 'SUCCESS' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
                            <div>
                                <div className="result-title">{lastImport.status}</div>
                                <div className="result-meta">{lastImport.import_id}</div>
                            </div>
                        </div>
                    )}
                </div>

                {/* History */}
                <div className="hub-card history-card">
                    <div className="card-header"><History size={16} /> TRANSFER HISTORY</div>
                    <div className="history-list">
                        {[...exportHistory.map(e => ({ ...e, type: 'EXPORT' })), ...importHistory.map(i => ({ ...i, type: 'IMPORT' }))]
                            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
                            .slice(0, 10)
                            .map((item, idx) => (
                                <div key={idx} className={`history-item ${item.type.toLowerCase()}`}>
                                    <span className="type-badge">{item.type}</span>
                                    <span className="id">{item.export_id || item.import_id}</span>
                                </div>
                            ))}
                    </div>
                </div>
            </div>

            <style>{`
        .data-hub { display: flex; flex-direction: column; gap: 25px; padding: 20px; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .section-header { display: flex; align-items: center; gap: 12px; font-weight: bold; border-bottom: 1px solid #222; padding-bottom: 15px; color: #888; letter-spacing: 1.5px; }
        .hub-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .hub-card { background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px; }
        .history-card { grid-column: 1 / -1; }
        .card-header { display: flex; align-items: center; gap: 10px; font-size: 0.8rem; font-weight: bold; color: #ccc; margin-bottom: 15px; }
        .input-field { width: 100%; background: #1a1a1f; border: 1px solid #333; color: #eee; padding: 12px; border-radius: 8px; margin-bottom: 15px; }
        .action-btn { width: 100%; padding: 15px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 0.85rem; }
        .export-btn { background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%); color: #000; }
        .export-btn:disabled { background: #444; color: #888; cursor: not-allowed; }
        .file-input-label { display: flex; flex-direction: column; align-items: center; gap: 10px; padding: 30px; border: 2px dashed #333; border-radius: 8px; cursor: pointer; color: #888; transition: 0.3s; }
        .file-input-label:hover { border-color: #00ff88; color: #00ff88; }
        .file-input-label input { display: none; }
        .result-box { display: flex; align-items: center; gap: 12px; padding: 15px; border-radius: 8px; margin-top: 15px; }
        .result-box.success { background: rgba(0,255,136,0.1); color: #00ff88; }
        .result-box.error { background: rgba(255,68,68,0.1); color: #ff4444; }
        .result-title { font-weight: bold; font-size: 0.85rem; }
        .result-meta { font-size: 0.7rem; opacity: 0.7; }
        .history-list { display: flex; flex-direction: column; gap: 8px; max-height: 200px; overflow-y: auto; }
        .history-item { display: flex; align-items: center; gap: 12px; padding: 10px; background: #1a1a1f; border-radius: 6px; }
        .type-badge { font-size: 0.6rem; font-weight: bold; padding: 3px 8px; border-radius: 4px; }
        .history-item.export .type-badge { background: #00d4ff22; color: #00d4ff; }
        .history-item.import .type-badge { background: #00ff8822; color: #00ff88; }
        .id { font-size: 0.75rem; color: #888; font-family: monospace; }
      `}</style>
        </div>
    );
};

export default DataHub;
