import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, Bot, User, Sparkles, Loader2 } from 'lucide-react';
import axios from 'axios';

const LLMChatPanel = ({ title = 'NEURAL ASSISTANT', endpoint = '/api/intelligence/ask', placeholder = 'Consult the hive mind...' }) => {
    const [messages, setMessages] = useState([
        { role: 'assistant', content: 'Neural link established. I am the Fanuc Rise Intelligence. How can I optimize your manufacturing process today?' }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const sendMessage = async () => {
        if (!input.trim() || loading) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            // Note: backend expects { question: "..." } based on backend/main.py
            const res = await axios.post(endpoint, { question: input });
            const assistantMessage = { role: 'assistant', content: res.data?.answer || 'I have processed your query but have no verbal response.' };
            setMessages(prev => [...prev, assistantMessage]);
        } catch (e) {
            setMessages(prev => [...prev, { role: 'assistant', content: 'Connectivity error. The Neural Core is currently shielded.' }]);
        }
        setLoading(false);
    };

    return (
        <div className="glass-panel rounded-xl overflow-hidden flex flex-col h-full border border-white/5">
            <div className="flex items-center gap-2 p-4 bg-industrial-bg/50 border-b border-white/5 neuro-text text-gray-400">
                <Sparkles size={16} className="text-industrial-primary animate-pulse" />
                <span>{title}</span>
            </div>

            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-white/10">
                {messages.map((msg, i) => (
                    <div key={i} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 border border-white/10 ${msg.role === 'assistant' ? 'bg-neuro-pulse/10 text-neuro-pulse' : 'bg-industrial-primary/10 text-industrial-primary'}`}>
                            {msg.role === 'assistant' ? <Bot size={16} /> : <User size={16} />}
                        </div>
                        <div className={`max-w-[80%] p-3 rounded-xl text-xs leading-relaxed border ${
                            msg.role === 'user' 
                            ? 'bg-industrial-primary/5 border-industrial-primary/20 text-gray-200' 
                            : 'bg-industrial-surface/50 border-white/5 text-gray-300'
                        }`}>
                            {msg.content}
                        </div>
                    </div>
                ))}
                {loading && (
                    <div className="flex gap-3">
                        <div className="w-8 h-8 rounded-full flex items-center justify-center bg-neuro-pulse/10 text-neuro-pulse border border-neuro-pulse/20">
                            <Loader2 size={16} className="animate-spin" />
                        </div>
                        <div className="p-3 rounded-xl text-xs bg-industrial-surface/50 border border-white/5 text-gray-500 italic">
                            Decrypting neural patterns...
                        </div>
                    </div>
                )}
            </div>

            <div className="p-4 bg-black/20 border-top border-white/5 flex gap-2">
                <input
                    type="text"
                    placeholder={placeholder}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    disabled={loading}
                    className="flex-1 bg-industrial-surface border border-white/10 text-gray-200 px-4 py-2 rounded-lg text-xs focus:border-industrial-primary outline-none transition-colors"
                />
                <button 
                    onClick={sendMessage} 
                    disabled={loading || !input.trim()}
                    className="bg-industrial-primary disabled:bg-gray-700 text-black px-4 py-2 rounded-lg transition-all hover:scale-105 active:scale-95 flex items-center justify-center"
                >
                    <Send size={14} />
                </button>
            </div>
        </div>
    );
};

export default LLMChatPanel;
