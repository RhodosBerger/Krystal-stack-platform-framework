import React, { useState, useRef, useEffect } from 'react';
import { Terminal, Play, RefreshCw, Trash2, ChevronRight } from 'lucide-react';
import axios from 'axios';

const DebugConsole = () => {
    const [command, setCommand] = useState('');
    const [history, setHistory] = useState([
        { type: 'system', content: '🤓 GEEK MODE // Debug Console v1.0' },
        { type: 'system', content: 'Type "help" for available commands.' }
    ]);
    const [loading, setLoading] = useState(false);
    const outputRef = useRef(null);

    useEffect(() => {
        if (outputRef.current) {
            outputRef.current.scrollTop = outputRef.current.scrollHeight;
        }
    }, [history]);

    const executeCommand = async () => {
        if (!command.trim()) return;

        // Add command to history
        setHistory(prev => [...prev, { type: 'input', content: `> ${command}` }]);
        const currentCmd = command;
        setCommand('');
        setLoading(true);

        try {
            const res = await axios.post('/api/debug/console', { command: currentCmd });
            const result = res.data?.result || {};

            // Format output
            let output = '';
            if (result.error) {
                output = `❌ Error: ${result.error}`;
                if (result.available) {
                    output += `\nAvailable commands: ${result.available.join(', ')}`;
                }
            } else {
                output = JSON.stringify(result, null, 2);
            }

            setHistory(prev => [...prev, { type: 'output', content: output }]);
        } catch (e) {
            setHistory(prev => [...prev, { type: 'error', content: `⚠️ Request failed: ${e.message}` }]);
        }
        setLoading(false);
    };

    const quickCommands = [
        { cmd: 'sysinfo', label: 'System Info' },
        { cmd: 'memory', label: 'Memory' },
        { cmd: 'threads', label: 'Threads' },
        { cmd: 'endpoints', label: 'Endpoints' },
        { cmd: 'gc', label: 'Run GC' }
    ];

    const runQuickCommand = (cmd) => {
        setCommand(cmd);
        setTimeout(() => executeCommand(), 100);
    };

    const clearHistory = () => {
        setHistory([{ type: 'system', content: '🤓 Console cleared.' }]);
    };

    return (
        <div className="debug-console">
            <div className="console-header">
                <Terminal size={16} className="text-primary" />
                <span>DEBUG CONSOLE // GEEK MODE</span>
                <button className="clear-btn" onClick={clearHistory}><Trash2 size={14} /></button>
            </div>

            <div className="quick-commands">
                {quickCommands.map(q => (
                    <button key={q.cmd} onClick={() => runQuickCommand(q.cmd)}>{q.label}</button>
                ))}
            </div>

            <div className="console-output" ref={outputRef}>
                {history.map((entry, i) => (
                    <div key={i} className={`console-line ${entry.type}`}>
                        <pre>{entry.content}</pre>
                    </div>
                ))}
                {loading && <div className="console-line loading">⏳ Executing...</div>}
            </div>

            <div className="console-input">
                <ChevronRight size={16} className="prompt" />
                <input
                    type="text"
                    value={command}
                    onChange={(e) => setCommand(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && executeCommand()}
                    placeholder="Enter command (e.g., help, sysinfo, memory)"
                    disabled={loading}
                />
                <button onClick={executeCommand} disabled={loading}>
                    <Play size={16} />
                </button>
            </div>

            <style>{`
        .debug-console { background: #0a0a0f; border: 1px solid #00ff88; border-radius: 12px; overflow: hidden; height: 100%; display: flex; flex-direction: column; font-family: 'Fira Code', 'Monaco', monospace; }
        .console-header { display: flex; align-items: center; gap: 10px; padding: 15px; background: #111; border-bottom: 1px solid #222; font-weight: bold; font-size: 0.8rem; color: #00ff88; letter-spacing: 1px; }
        .clear-btn { margin-left: auto; background: #222; border: 1px solid #333; color: #888; padding: 6px; border-radius: 4px; cursor: pointer; }
        
        .quick-commands { display: flex; gap: 8px; padding: 10px 15px; background: #0f0f12; border-bottom: 1px solid #1a1a1f; flex-wrap: wrap; }
        .quick-commands button { background: #1a1a1f; border: 1px solid #333; color: #888; padding: 4px 10px; border-radius: 4px; font-size: 0.65rem; cursor: pointer; }
        .quick-commands button:hover { color: #00ff88; border-color: #00ff88; }
        
        .console-output { flex: 1; overflow-y: auto; padding: 15px; font-size: 0.75rem; line-height: 1.6; }
        .console-line { margin-bottom: 8px; }
        .console-line pre { margin: 0; white-space: pre-wrap; word-break: break-all; }
        .console-line.system { color: #666; font-style: italic; }
        .console-line.input { color: #00d4ff; }
        .console-line.output { color: #eee; background: #111; padding: 10px; border-radius: 6px; border-left: 2px solid #00ff88; }
        .console-line.error { color: #ff4444; }
        .console-line.loading { color: #ffaa00; }
        
        .console-input { display: flex; align-items: center; gap: 8px; padding: 12px 15px; background: #111; border-top: 1px solid #222; }
        .prompt { color: #00ff88; flex-shrink: 0; }
        .console-input input { flex: 1; background: #1a1a1f; border: 1px solid #333; color: #eee; padding: 10px 15px; border-radius: 6px; font-family: inherit; font-size: 0.8rem; }
        .console-input input:focus { border-color: #00ff88; outline: none; }
        .console-input button { background: #00ff88; color: #000; border: none; padding: 10px 15px; border-radius: 6px; cursor: pointer; }
        .console-input button:disabled { background: #333; color: #666; }
      `}</style>
        </div>
    );
};

export default DebugConsole;
