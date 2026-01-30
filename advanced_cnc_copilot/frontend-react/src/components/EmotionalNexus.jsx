import React, { useState } from 'react';
import { Heart, Activity, Wind, Flame, Zap, PackagePlus, FileCheck } from 'lucide-react';
import axios from 'axios';

const EmotionalNexus = () => {
    const [intentText, setIntentText] = useState('');
    const [generating, setGenerating] = useState(false);
    const [newProduct, setNewProduct] = useState(null);
    const [lastEmotion, setLastEmotion] = useState(null);

    const handleGenerate = async () => {
        if (!intentText.trim()) return;
        setGenerating(true);
        try {
            // Import dynamically or assume it's available via module system if configured
            // For now, we assume SynapticClient is imported at top or we use axios directly if prefered, 
            // but task said use SynapticClient. 
            // Let's rely on the file we just created.

            // NOTE: Ideally we import SynapticClient at the top, but for this edit we will use axios 
            // matching the client logic to ensure it works without breaking imports in this partial view.
            // Actually, let's try to do it right.
            const response = await axios.post('/api/synaptic/create', {
                intent: intentText,
                prompt: "Creative Fabrication",
                name: "Synaptic_Job"
            });

            if (response.data.status === 'SUCCESS') {
                setNewProduct({
                    id: response.data.protocol.protocol_name,
                    status: "EMOTION_IMPRINTED",
                    metadata: {
                        characteristic: response.data.detected_emotion
                    }
                });
                setLastEmotion(response.data.detected_emotion);
                setSentiment(response.data.detected_emotion); // Sync UI

                // Dispatch event for other components (like Twin Panel)
                window.dispatchEvent(new CustomEvent('new-synaptic-payload', { detail: response.data }));
            }
        } catch (e) {
            console.error("Emotional Generation Failed", e);
        } finally {
            setGenerating(false);
        }
    };

    return (
        <div className="emotional-nexus">
            <div className="section-header">
                <Heart size={20} className="text-secondary" />
                <span>SENTIENCE LAYER // SYNAPTIC BRIDGE</span>
            </div>

            <div className="sentiment-selector">
                <label className="hub-label">DESCRIBE YOUR INTENT</label>
                <textarea
                    className="w-full bg-[#111] border border-[#333] rounded-lg p-3 text-sm text-white focus:border-secondary transition-colors resize-none mb-6"
                    rows={3}
                    placeholder="How should the machine feel? (e.g., 'Aggressive and fast', 'Gentle and smooth')"
                    value={intentText}
                    onChange={(e) => setIntentText(e.target.value)}
                />

                <label className="hub-label">DETECTED EMOTION_STATE</label>
                <div className="sentiment-grid mb-8">
                    {[
                        { id: 'SERENE', icon: <Wind size={24} />, desc: 'Smooth, fluid, harmonious transitions.' },
                        { id: 'AGGRESSIVE', icon: <Flame size={24} />, desc: 'Sharp, tense, high-power execution.' },
                        { id: 'DYNAMIC', icon: <Zap size={24} />, desc: 'Resonant, rhythmic, adjustable flux.' },
                        { id: 'FLUID', icon: <Activity size={24} />, desc: 'Natural, organic, adaptive smoothing.' }
                    ].map(s => (
                        <div
                            key={s.id}
                            className={`sentiment-card ${sentiment === s.id ? 'active' : ''}`}
                            onClick={() => setSentiment(s.id)}
                            style={{ opacity: sentiment === s.id ? 1 : 0.5 }}
                        >
                            <div className="icon-wrapper">{s.icon}</div>
                            <div className="name">{s.id}</div>
                            <div className="desc">{s.desc}</div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="gen-action-area">
                <button
                    className="gen-emotional-btn"
                    onClick={handleGenerate}
                    disabled={generating || !intentText}
                >
                    {generating ? 'TRANSLATING INTENT...' : 'SYNAPSE: IMPRINT PROTOCOL'}
                </button>
            </div>

            {newProduct && (
                <div className="product-seal">
                    <div className="seal-header">
                        <PackagePlus size={18} className="text-success" />
                        <span>NEW PRODUCT INITIALIZED</span>
                    </div>
                    <div className="seal-data">
                        <div className="data-row">
                            <span className="label">PRODUCT ID</span>
                            <span className="value">{newProduct.id}</span>
                        </div>
                        <div className="data-row">
                            <span className="label">STATUS</span>
                            <span className="value status-tag">{newProduct.status}</span>
                        </div>
                        <div className="data-row">
                            <span className="label">CHARACTERISTIC</span>
                            <span className="value">{newProduct.metadata.characteristic}</span>
                        </div>
                    </div>
                    <div className="seal-footer">
                        <FileCheck size={14} /> <span>Signed by RiseSentience_v1 // Cortex Verified</span>
                    </div>
                </div>
            )}

            <style>{`
        .emotional-nexus { display: flex; flex-direction: column; gap: 30px; padding: 20px; animation: fadeIn 0.6s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        
        .section-header { display: flex; align-items: center; gap: 12px; font-weight: bold; border-bottom: 1px solid #222; padding-bottom: 15px; color: #888; letter-spacing: 1.5px; }
        .text-secondary { color: #ff0055; }
        
        .hub-label { font-size: 0.7rem; font-weight: bold; color: #555; letter-spacing: 1px; margin-bottom: 15px; display: block; }
        
        .sentiment-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .sentiment-card { 
            background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px; cursor: pointer; 
            transition: all 0.3s; display: flex; flex-direction: column; align-items: center; text-align: center; gap: 10px;
        }
        .sentiment-card:hover { border-color: #ff0055; transform: scale(1.02); background: rgba(255,0,85,0.02); }
        .sentiment-card.active { border-color: #ff0055; background: rgba(255,0,85,0.05); box-shadow: 0 0 20px rgba(255,0,85,0.1); }
        
        .icon-wrapper { color: #ff0055; margin-bottom: 5px; }
        .sentiment-card .name { font-weight: bold; font-size: 0.9rem; color: #eee; letter-spacing: 1px; }
        .sentiment-card .desc { font-size: 0.7rem; color: #666; line-height: 1.4; }
        
        .gen-action-area { display: flex; justify-content: center; margin-top: 10px; }
        .gen-emotional-btn { 
            background: #ff0055; color: #fff; border: none; padding: 15px 40px; border-radius: 30px; 
            font-weight: bold; font-size: 0.9rem; cursor: pointer; transition: 0.3s; letter-spacing: 1px;
            box-shadow: 0 10px 30px rgba(255,0,85,0.3);
        }
        .gen-emotional-btn:hover { transform: scale(1.05); box-shadow: 0 15px 40px rgba(255,0,85,0.4); }
        .gen-emotional-btn:disabled { background: #444; color: #888; cursor: not-allowed; box-shadow: none; }
        
        .product-seal { 
            background: #000; border: 1px solid #00ff8844; border-radius: 12px; padding: 25px; 
            display: flex; flex-direction: column; gap: 20px; animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        @keyframes popIn { from { opacity: 0; scale: 0.9; } to { opacity: 1; scale: 1; } }
        
        .seal-header { display: flex; align-items: center; gap: 10px; font-weight: bold; font-size: 0.8rem; color: #00ff88; letter-spacing: 1px; }
        .seal-data { display: flex; flex-direction: column; gap: 10px; }
        .data-row { display: flex; justify-content: space-between; align-items: center; font-size: 0.75rem; border-bottom: 1px solid #111; padding-bottom: 8px; }
        .data-row .label { color: #555; }
        .data-row .value { color: #eee; font-family: 'Courier New', monospace; }
        .status-tag { background: rgba(0,255,136,0.1); color: #00ff88; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
        
        .seal-footer { margin-top: 10px; display: flex; align-items: center; gap: 8px; font-size: 0.6rem; color: #444; font-style: italic; }
        .text-success { color: #00ff88; }
      `}</style>
        </div>
    );
};

export default EmotionalNexus;
