import React, { useState, useEffect, useRef } from 'react';
import AppleCard from './AppleCard';
import {
    Sparkles,
    Send,
    Bot,
    User,
    TrendingUp,
    AlertTriangle,
    Lightbulb,
    History
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const IntelligenceDashboard = () => {
    const [messages, setMessages] = useState([
        { id: 1, role: 'assistant', text: "Hello! I'm your AI manufacturing assistant. Ask me anything about your CNC operations!" }
    ]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMsg = { id: Date.now(), role: 'user', text: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsTyping(true);

        // Simulate AI Response
        setTimeout(() => {
            const aiMsg = {
                id: Date.now() + 1,
                role: 'assistant',
                text: "I've analyzed the recent spindle load data. There seems to be a recurring harmonic vibration at 8000 RPM. I recommend checking the tool balancing or reducing speed by 5%."
            };
            setMessages(prev => [...prev, aiMsg]);
            setIsTyping(false);
        }, 1500);
    };

    const QuickQuestion = ({ text }) => (
        <button
            onClick={() => setInput(text)}
            className="px-4 py-2 bg-blue-50 text-blue-600 rounded-full text-xs font-semibold hover:bg-blue-100 transition-colors border border-blue-100"
        >
            {text}
        </button>
    );

    return (
        <div className="h-full grid grid-cols-12 gap-6 p-1">

            {/* LEFT COLUMN: CHAT (iMessage Style) */}
            <div className="col-span-12 lg:col-span-8 h-full flex flex-col">
                <AppleCard title="Neural Interface" icon={Sparkles} className="flex-1 flex flex-col h-full bg-[#f5f5f7]">

                    {/* Chat Area */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-6">
                        {messages.map((msg) => (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                key={msg.id}
                                className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                            >
                                <div className={`w-8 h-8 rounded-full flex-none flex items-center justify-center shadow-sm ${msg.role === 'user' ? 'bg-gray-900 text-white' : 'bg-gradient-to-br from-indigo-500 to-purple-600 text-white'}`}>
                                    {msg.role === 'user' ? <User size={14} /> : <Bot size={14} />}
                                </div>
                                <div className={`max-w-[80%] rounded-2xl px-5 py-3 text-sm leading-relaxed shadow-sm ${msg.role === 'user'
                                        ? 'bg-blue-500 text-white rounded-br-none'
                                        : 'bg-white text-gray-800 border border-gray-100 rounded-bl-none'
                                    }`}>
                                    {msg.text}
                                </div>
                            </motion.div>
                        ))}
                        {isTyping && (
                            <div className="flex gap-3">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 text-white flex items-center justify-center">
                                    <Bot size={14} />
                                </div>
                                <div className="bg-white border border-gray-100 rounded-2xl rounded-bl-none px-4 py-3 flex items-center gap-1 shadow-sm">
                                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" />
                                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce delay-75" />
                                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce delay-150" />
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input Area */}
                    <div className="p-4 bg-white border-t border-gray-100 rounded-b-2xl">
                        <div className="flex gap-2 overflow-x-auto pb-3 scrollbar-hide">
                            <QuickQuestion text="Why did quality drop?" />
                            <QuickQuestion text="Analyze tool wear patterns" />
                            <QuickQuestion text="Predict maintenance needs" />
                        </div>
                        <div className="relative">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                                placeholder="Ask Deep Thought..."
                                className="w-full pl-5 pr-12 py-4 bg-gray-100 rounded-full focus:bg-white focus:ring-2 focus:ring-blue-100 focus:outline-none transition-all text-sm font-medium"
                            />
                            <button
                                onClick={handleSend}
                                className="absolute right-2 top-2 p-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors shadow-lg active:scale-95"
                            >
                                <Send size={18} />
                            </button>
                        </div>
                    </div>
                </AppleCard>
            </div>

            {/* RIGHT COLUMN: INSIGHTS & PREDICTIONS */}
            <div className="col-span-12 lg:col-span-4 flex flex-col gap-6 h-full">

                {/* Predictions */}
                <AppleCard title="Future Predictions" icon={TrendingUp} className="flex-none">
                    <div className="space-y-3">
                        <div className="p-3 bg-red-50 border border-red-100 rounded-xl">
                            <div className="flex items-center gap-2 mb-2 text-red-600 font-bold text-xs uppercase tracking-wider">
                                <AlertTriangle size={12} />
                                <span>Tool Failure Risk</span>
                            </div>
                            <p className="text-sm font-medium text-gray-800">High probability of end-mill fracture in next 40 mins.</p>
                            <div className="mt-2 w-full bg-red-200 h-1.5 rounded-full overflow-hidden">
                                <div className="h-full bg-red-500 w-[82%]" />
                            </div>
                            <div className="text-right text-[10px] text-red-600 font-bold mt-1">82% Confidence</div>
                        </div>

                        <div className="p-3 bg-yellow-50 border border-yellow-100 rounded-xl">
                            <div className="flex items-center gap-2 mb-2 text-yellow-600 font-bold text-xs uppercase tracking-wider">
                                <TrendingUp size={12} />
                                <span>Efficiency Opportunity</span>
                            </div>
                            <p className="text-sm font-medium text-gray-800">Coolant pressure can be optimized for current alloy.</p>
                        </div>
                    </div>
                </AppleCard>

                {/* Insights */}
                <AppleCard title="Key Insights" icon={Lightbulb} className="flex-1 min-h-0">
                    <div className="h-full overflow-y-auto space-y-3 pr-2">
                        {[1, 2, 3].map((i) => (
                            <div key={i} className="flex gap-3 p-3 hover:bg-gray-50 rounded-xl transition-colors cursor-pointer group">
                                <div className="w-8 h-8 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center flex-none group-hover:bg-indigo-100 transition-colors">
                                    <Sparkles size={16} />
                                </div>
                                <div>
                                    <h4 className="text-sm font-bold text-gray-900">Pattern Detected</h4>
                                    <p className="text-xs text-gray-500 mt-1 leading-relaxed">Spindle thermal spikes correlate with afternoon shift changes.</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </AppleCard>

            </div>
        </div>
    );
};

export default IntelligenceDashboard;
