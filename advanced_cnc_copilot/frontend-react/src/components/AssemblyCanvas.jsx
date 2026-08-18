import React, { useState } from 'react';
import { Layers, MousePointer2 } from 'lucide-react';
import axios from 'axios';

const AssemblyCanvas = ({ assembled, setAssembled }) => {
    const [generating, setGenerating] = useState(false);

    const handleGenerate = async () => {
        if (assembled.length === 0) return;
        setGenerating(true);
        try {
            const response = await axios.post('/api/generate/custom', {
                element_ids: assembled.map(e => e.id),
                format: 'GCODE'
            });
            if (response.data.status === 'SUCCESS') {
                window.dispatchEvent(new CustomEvent('new-payload', { detail: response.data.content }));
            }
        } catch (error) {
            console.error('Assembly Failed:', error);
        } finally {
            setGenerating(false);
        }
    };

    return (
        <div className="canvas-wrapper">
            <div className="canvas-header">
                <Layers size={16} />
                <span>GENERATIVE ASSEMBLY CANVAS</span>
            </div>

            <div className="canvas-area">
                {assembled.length === 0 ? (
                    <div className="canvas-empty">
                        <MousePointer2 size={32} className="text-dim" />
                        <p>Drag elements here to assemble custom generation</p>
                    </div>
                ) : (
                    <div className="assembly-stack">
                        {assembled.map((item, idx) => (
                            <div key={idx} className="assembly-item">
                                <span className="idx">{idx + 1}</span>
                                <span className="type">{item.type}</span>
                                <span className="name">{item.filename}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <div className="canvas-controls">
                <button className="gen-btn">
                    GENERATE FULLY CUSTOM DOCUMENT
                </button>
            </div>

            <style>{`
        .canvas-wrapper { display: flex; flex-direction: column; height: 100%; }
        .canvas-header { padding: 15px 20px; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 10px; color: #888; font-size: 0.8rem; font-weight: bold; }
        .canvas-area { flex: 1; padding: 40px; display: flex; justify-content: center; align-items: flex-start; overflow-y: auto; }
        .canvas-empty { text-align: center; color: #444; margin-top: 100px; display: flex; flex-direction: column; align-items: center; gap: 15px; }
        .assembly-stack { width: 100%; max-width: 600px; display: flex; flex-direction: column; gap: 12px; }
        .assembly-item {
          background: #111;
          border: 1px solid #00ff8844;
          padding: 15px;
          border-radius: 10px;
          display: flex;
          align-items: center;
          gap: 15px;
          box-shadow: 0 4px 15px rgba(0,255,136,0.05);
        }
        .assembly-item .idx { background: #00ff88; color: #000; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: bold; }
        .assembly-item .type { color: #00ff88; font-size: 0.65rem; font-weight: bold; text-transform: uppercase; border: 1px solid #00ff8844; padding: 2px 8px; border-radius: 4px; }
        .assembly-item .name { font-size: 0.9rem; color: #eee; }
        .canvas-controls { padding: 20px; border-top: 1px solid #222; display: flex; justify-content: center; }
        .gen-btn {
          background: transparent;
          color: #00ff88;
          border: 1px solid #00ff88;
          padding: 12px 30px;
          border-radius: 30px;
          font-weight: bold;
          font-size: 0.85rem;
          cursor: pointer;
          transition: 0.3s;
          letter-spacing: 1px;
        }
        .gen-btn:hover { background: #00ff88; color: #000; box-shadow: 0 0 20px rgba(0,255,136,0.3); }
      `}</style>
        </div>
    );
};

export default AssemblyCanvas;
