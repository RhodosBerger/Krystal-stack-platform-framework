import React, { useState } from 'react';
import { MessageSquare, Search, Check, Edit2, X, ChevronDown, ChevronUp, Save } from 'lucide-react';
import axios from 'axios';

/**
 * AI Config Panel - Based on mockup design
 * Features: AI chat interface, parameter cards with edit mode, category filters
 */
const AIConfigPanel = () => {
    const [chatMessages, setChatMessages] = useState([
        { role: 'user', content: 'Set max spindle RPM to 8000' },
        { role: 'assistant', content: 'Okay, updating the configuration for maximum spindle speed.' }
    ]);
    const [chatInput, setChatInput] = useState('');
    const [searchQuery, setSearchQuery] = useState('');
    const [activeCategory, setActiveCategory] = useState('MACHINING');
    const [editingParam, setEditingParam] = useState(null);
    const [editValue, setEditValue] = useState('');

    const categories = ['MACHINING', 'SAFETY', 'LLM', 'SYSTEM'];

    const [params, setParams] = useState([
        { key: 'max_spindle_rpm', value: 8000, unit: 'RPM', type: 'Number', category: 'MACHINING', validated: true },
        { key: 'feed_rate', value: 500, unit: 'mm/min', type: 'Number', category: 'MACHINING', validated: true },
        { key: 'coolant_flow_rate', value: 5.0, unit: 'L/min', type: 'Number', category: 'MACHINING', validated: true },
        { key: 'emergency_stop_threshold', value: 50, unit: 'ms', type: 'Number', category: 'SAFETY', validated: true },
        { key: 'llm_model_version', value: 'v2.1.0', unit: '', type: 'LLM', category: 'LLM', validated: true },
        { key: 'system_log_level', value: 'INFO', unit: '', type: 'INFO', category: 'SYSTEM', validated: true }
    ]);

    const sendChat = async () => {
        if (!chatInput.trim()) return;

        setChatMessages(prev => [...prev, { role: 'user', content: chatInput }]);
        const userMsg = chatInput;
        setChatInput('');

        // Simulate AI response
        setTimeout(() => {
            setChatMessages(prev => [...prev, {
                role: 'assistant',
                content: `Processing: "${userMsg}". Configuration updated successfully.`
            }]);
        }, 800);
    };

    const startEdit = (param) => {
        setEditingParam(param.key);
        setEditValue(param.value);
    };

    const saveEdit = (key) => {
        setParams(prev => prev.map(p =>
            p.key === key ? { ...p, value: editValue } : p
        ));
        setEditingParam(null);
    };

    const cancelEdit = () => {
        setEditingParam(null);
        setEditValue('');
    };

    const filteredParams = params.filter(p =>
        p.category === activeCategory &&
        (searchQuery === '' || p.key.toLowerCase().includes(searchQuery.toLowerCase()))
    );

    return (
        <div className="ai-config-panel">
            {/* AI Chat Section */}
            <div className="chat-section">
                <div className="chat-header">
                    <div className="avatar">🤖</div>
                    <span>AI Assistant</span>
                </div>
                <div className="chat-messages">
                    {chatMessages.map((msg, i) => (
                        <div key={i} className={`chat-bubble ${msg.role}`}>
                            {msg.role === 'user' && <div className="user-avatar">👤</div>}
                            <div className="bubble-content">{msg.content}</div>
                        </div>
                    ))}
                </div>
                <div className="chat-input-row">
                    <input
                        type="text"
                        placeholder="Ask AI to configure..."
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && sendChat()}
                    />
                </div>
            </div>

            {/* Search & Filters */}
            <div className="filter-section">
                <div className="search-box">
                    <Search size={14} />
                    <input
                        type="text"
                        placeholder="Search parameters..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                    />
                </div>
                <div className="category-pills">
                    {categories.map(cat => (
                        <button
                            key={cat}
                            className={activeCategory === cat ? 'active' : ''}
                            onClick={() => setActiveCategory(cat)}
                        >
                            {cat}
                        </button>
                    ))}
                </div>
            </div>

            {/* Parameter List */}
            <div className="params-list">
                {filteredParams.map(param => (
                    <div key={param.key} className={`param-card ${editingParam === param.key ? 'editing' : ''}`}>
                        <div className="param-row">
                            <div className="param-expand">
                                {editingParam === param.key ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                            </div>
                            <div className="param-info">
                                <span className="param-label">parameter</span>
                                <span className="param-key">{param.key}: {param.value} {param.unit}</span>
                            </div>
                            <div className="param-badges">
                                <span className="type-badge">{param.type}</span>
                                {param.validated && <Check size={14} className="valid-icon" />}
                            </div>
                        </div>

                        {editingParam !== param.key && (
                            <button className="edit-btn" onClick={() => startEdit(param)}>
                                <Edit2 size={12} /> Edit Mode
                            </button>
                        )}

                        {editingParam === param.key && (
                            <div className="edit-section">
                                <div className="edit-row">
                                    <input
                                        type="text"
                                        value={editValue}
                                        onChange={(e) => setEditValue(e.target.value)}
                                    />
                                    <span className="unit-label">{param.unit}</span>
                                </div>
                                <div className="edit-actions">
                                    <button className="save-btn" onClick={() => saveEdit(param.key)}>
                                        SAVE CHANGES
                                    </button>
                                    <button className="cancel-btn" onClick={cancelEdit}>
                                        CANCEL
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>

            <div className="version-footer">Config Manager v3.0</div>

            <style>{`
        .ai-config-panel { background: linear-gradient(135deg, #0a0a0f 0%, #111118 100%); min-height: 100vh; padding: 0; font-family: 'Inter', sans-serif; }
        
        .chat-section { padding: 20px; border-bottom: 1px solid #222; }
        .chat-header { display: flex; align-items: center; gap: 10px; margin-bottom: 15px; }
        .avatar { font-size: 1.5rem; }
        .chat-header span { color: #eee; font-weight: 500; }
        
        .chat-messages { display: flex; flex-direction: column; gap: 10px; margin-bottom: 15px; }
        .chat-bubble { display: flex; gap: 10px; align-items: flex-start; max-width: 80%; }
        .chat-bubble.user { align-self: flex-end; flex-direction: row-reverse; }
        .chat-bubble.assistant { align-self: flex-start; }
        .user-avatar { font-size: 1rem; }
        .bubble-content { padding: 12px 16px; border-radius: 12px; font-size: 0.85rem; }
        .chat-bubble.user .bubble-content { background: #1a1a24; color: #eee; border-radius: 12px 12px 0 12px; }
        .chat-bubble.assistant .bubble-content { background: #00ff8822; color: #00ff88; border-radius: 12px 12px 12px 0; }
        
        .chat-input-row input { width: 100%; background: #1a1a24; border: 1px solid #333; color: #eee; padding: 12px 15px; border-radius: 8px; }
        
        .filter-section { padding: 15px 20px; display: flex; gap: 15px; align-items: center; border-bottom: 1px solid #222; }
        .search-box { display: flex; align-items: center; gap: 8px; background: #1a1a24; border: 1px solid #333; padding: 8px 12px; border-radius: 8px; flex: 1; }
        .search-box input { background: none; border: none; color: #eee; outline: none; flex: 1; }
        .search-box svg { color: #666; }
        
        .category-pills { display: flex; gap: 8px; }
        .category-pills button { background: #1a1a24; border: 1px solid #333; color: #666; padding: 8px 16px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; cursor: pointer; }
        .category-pills button.active { background: #00ff8822; color: #00ff88; border-color: #00ff88; }
        
        .params-list { padding: 15px 20px; display: flex; flex-direction: column; gap: 10px; }
        .param-card { background: #1a1a24; border: 1px solid #222; border-radius: 10px; padding: 15px; }
        .param-card.editing { border-color: #00ff88; }
        
        .param-row { display: flex; align-items: center; gap: 12px; }
        .param-expand { color: #555; }
        .param-info { flex: 1; }
        .param-label { display: block; font-size: 0.6rem; color: #555; }
        .param-key { font-size: 0.85rem; color: #eee; font-family: monospace; }
        .param-badges { display: flex; align-items: center; gap: 8px; }
        .type-badge { background: #00d4ff22; color: #00d4ff; padding: 4px 10px; border-radius: 4px; font-size: 0.65rem; }
        .valid-icon { color: #00ff88; }
        
        .edit-btn { margin-top: 10px; background: #222; border: 1px solid #333; color: #888; padding: 6px 12px; border-radius: 6px; font-size: 0.7rem; cursor: pointer; display: flex; align-items: center; gap: 5px; }
        
        .edit-section { margin-top: 15px; padding-top: 15px; border-top: 1px solid #333; }
        .edit-row { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
        .edit-row input { flex: 1; background: #111; border: 1px solid #333; color: #eee; padding: 10px; border-radius: 6px; font-family: monospace; }
        .unit-label { color: #666; font-size: 0.8rem; }
        .edit-actions { display: flex; gap: 10px; }
        .save-btn { background: #00ff88; color: #000; border: none; padding: 8px 16px; border-radius: 6px; font-size: 0.7rem; font-weight: bold; cursor: pointer; }
        .cancel-btn { background: none; border: 1px solid #333; color: #888; padding: 8px 16px; border-radius: 6px; font-size: 0.7rem; cursor: pointer; }
        
        .version-footer { text-align: right; padding: 15px 20px; color: #333; font-size: 0.7rem; }
      `}</style>
        </div>
    );
};

export default AIConfigPanel;
