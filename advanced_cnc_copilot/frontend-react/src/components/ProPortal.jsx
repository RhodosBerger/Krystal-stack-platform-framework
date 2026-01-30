import UIAssembler from './UIAssembler';
import { Search, Package, Zap, FileCode, FileJson, FileText, Sparkles, Download, Layers } from 'lucide-react';

const ProPortal = () => {
    const [view, setView] = useState('PAYLOAD'); // 'PAYLOAD' or 'ASSEMBLY'
    const [products, setProducts] = useState([]);
    // ... (rest of states remain the same)

    return (
        <div className="pro-portal h-full flex flex-col">
            <div className="section-header flex justify-between items-center">
                <div className="flex items-center gap-3">
                    <Sparkles size={20} className="text-neuro-pulse" />
                    <span>PROFESSIONAL CREATOR PORTAL // PROJECT FORGE v2.1</span>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => setView('PAYLOAD')}
                        className={`px-4 py-2 rounded-lg text-[10px] font-bold transition-all ${view === 'PAYLOAD' ? 'bg-neuro-pulse text-black' : 'bg-white/5 text-gray-500'}`}
                    >
                        PAYLOAD GENERATOR
                    </button>
                    <button
                        onClick={() => setView('ASSEMBLY')}
                        className={`px-4 py-2 rounded-lg text-[10px] font-bold transition-all ${view === 'ASSEMBLY' ? 'bg-neuro-pulse text-black' : 'bg-white/5 text-gray-500'}`}
                    >
                        <div className="flex items-center gap-2"><Layers size={14} /> UI ASSEMBLY (CAD)</div>
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-hidden mt-6">
                {view === 'ASSEMBLY' ? (
                    <UIAssembler />
                ) : (
                    <div className="portal-grid">
                        {/* Search Panel */}
                        <div className="portal-card search-panel">
                            <div className="card-header"><Search size={16} /> PRODUCT SEARCH</div>
                            <input
                                type="text"
                                placeholder="Search products..."
                                value={searchQuery}
                                onChange={handleSearch}
                                className="search-input"
                            />
                            <div className="product-list">
                                {products.map(p => (
                                    <div
                                        key={p.id}
                                        className={`product-item ${selectedProduct?.id === p.id ? 'selected' : ''}`}
                                        onClick={() => setSelectedProduct(p)}
                                    >
                                        <Package size={16} />
                                        <div className="product-info">
                                            <span className="name">{p.name}</span>
                                            <span className="cat">{p.category}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Payload Configuration */}
                        <div className="portal-card config-panel">
                            {/* ... (existing logic for payload config) ... */}
                            <div className="card-header"><Zap size={16} /> PAYLOAD CONFIGURATION</div>
                            {selectedProduct ? (
                                <>
                                    <div className="selected-product">
                                        <Package size={20} />
                                        <div>
                                            <div className="name">{selectedProduct.name}</div>
                                            <div className="dims">{selectedProduct.dim_x} x {selectedProduct.dim_y} x {selectedProduct.dim_z} mm</div>
                                        </div>
                                    </div>
                                    <textarea
                                        placeholder="LLM Prompt..."
                                        value={llmPrompt}
                                        onChange={(e) => setLlmPrompt(e.target.value)}
                                        className="w-full bg-black/20 p-3 rounded h-24 text-xs border border-white/5"
                                    />
                                    <button className="generate-btn mt-4" onClick={generatePayloads} disabled={generating}>
                                        {generating ? 'GENERATING...' : 'GENERATE MULTI-PAYLOADS'}
                                    </button>
                                </>
                            ) : (
                                <div className="text-gray-600 italic text-center py-20">Select a product...</div>
                            )}
                        </div>

                        {/* Generated Results */}
                        {generatedBatch && (
                            <div className="portal-card results-panel mt-4 col-span-2">
                                <div className="card-header"><Download size={16} /> RESULTS ({generatedBatch.batch_id})</div>
                                <div className="max-h-60 overflow-y-auto space-y-2">
                                    {generatedBatch.payloads.map((pl, i) => (
                                        <div key={i} className="p-3 bg-black/40 rounded border border-white/5 font-mono text-[9px] text-neuro-success">
                                            <div className="uppercase opacity-50 mb-1">{pl.type}</div>
                                            <pre>{pl.content}</pre>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>

            <style>{`
        .pro-portal { display: flex; flex-direction: column; gap: 25px; padding: 20px; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .section-header { display: flex; align-items: center; gap: 12px; font-weight: bold; border-bottom: 1px solid #222; padding-bottom: 15px; color: #888; letter-spacing: 1.5px; }
        .portal-grid { display: grid; grid-template-columns: 1fr 1.5fr; gap: 20px; }
        .portal-card { background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px; }
        .card-header { display: flex; align-items: center; gap: 10px; font-size: 0.8rem; font-weight: bold; color: #ccc; margin-bottom: 15px; }
        .search-input { width: 100%; background: #1a1a1f; border: 1px solid #333; color: #eee; padding: 12px; border-radius: 8px; margin-bottom: 15px; }
        .product-list { display: flex; flex-direction: column; gap: 8px; max-height: 300px; overflow-y: auto; }
        .product-item { display: flex; align-items: center; gap: 12px; background: #1a1a1f; padding: 12px; border-radius: 8px; cursor: pointer; border: 1px solid transparent; transition: 0.2s; }
        .product-item:hover { border-color: #444; }
        .product-item.selected { border-color: #00ff88; background: rgba(0,255,136,0.05); }
        .product-info { display: flex; flex-direction: column; }
        .product-info .name { font-size: 0.85rem; font-weight: bold; color: #eee; }
        .product-info .cat { font-size: 0.65rem; color: #666; }
        .selected-product { display: flex; align-items: center; gap: 15px; background: rgba(0,255,136,0.05); padding: 15px; border-radius: 8px; border: 1px solid #00ff88; margin-bottom: 15px; }
        .selected-product .name { font-size: 1rem; font-weight: bold; color: #00ff88; }
        .selected-product .dims { font-size: 0.7rem; color: #888; }
        .llm-prompt-section { margin-bottom: 15px; }
        .llm-prompt-section label { font-size: 0.7rem; font-weight: bold; color: #555; display: block; margin-bottom: 8px; }
        .llm-prompt-section textarea { width: 100%; height: 80px; background: #1a1a1f; border: 1px solid #333; color: #eee; padding: 12px; border-radius: 8px; resize: none; }
        .payload-types label { font-size: 0.7rem; font-weight: bold; color: #555; display: block; margin-bottom: 10px; }
        .type-buttons { display: flex; gap: 10px; flex-wrap: wrap; }
        .type-btn { background: #222; border: 1px solid #333; color: #aaa; padding: 8px 12px; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 6px; font-size: 0.75rem; transition: 0.2s; }
        .type-btn:hover { background: #333; }
        .type-btn.active { background: #00ff88; color: #000; border-color: #00ff88; }
        .generate-btn { width: 100%; margin-top: 20px; background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%); color: #000; border: none; padding: 15px; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 0.9rem; }
        .generate-btn:disabled { background: #444; color: #888; cursor: not-allowed; }
        .empty-state { text-align: center; color: #555; padding: 40px; }
        .results-panel { grid-column: 1 / -1; }
        .payload-list { display: flex; flex-direction: column; gap: 15px; }
        .payload-item { background: #000; border: 1px solid #222; border-radius: 8px; overflow: hidden; }
        .payload-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 15px; background: #1a1a1f; }
        .type-badge { background: #00ff88; color: #000; padding: 3px 8px; border-radius: 4px; font-size: 0.65rem; font-weight: bold; }
        .size { font-size: 0.65rem; color: #666; }
        .payload-content { padding: 15px; font-size: 0.7rem; color: #00ff88; overflow-x: auto; margin: 0; max-height: 200px; overflow-y: auto; }
        .text-success { color: #00ff88; }
      `}</style>
        </div>
    );
};

export default ProPortal;
