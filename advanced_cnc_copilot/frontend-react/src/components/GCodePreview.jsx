import React from 'react';
import { FileCode, Copy, Download, Play, CheckCircle2, AlertTriangle } from 'lucide-react';

const GCodePreview = ({ gcode = '', title = 'G-Code Preview', validation = null }) => {
    const lines = gcode.split('\n').filter(l => l.trim());

    const copyToClipboard = () => {
        navigator.clipboard.writeText(gcode);
    };

    const downloadGCode = () => {
        const blob = new Blob([gcode], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'program.nc';
        a.click();
    };

    const highlightLine = (line) => {
        if (line.startsWith('(') || line.startsWith(';')) return 'text-gray-600 italic';
        if (line.match(/^[GMT]\d+/)) return 'text-neuro-pulse font-bold';
        if (line.match(/^[XYZIJKFSR]/)) return 'text-industrial-primary';
        if (line.match(/^[NOP]/)) return 'text-neuro-synapse';
        return 'text-gray-300';
    };

    return (
        <div className="glass-panel rounded-xl overflow-hidden flex flex-col h-full border border-white/5">
            <div className="flex justify-between items-center p-4 bg-industrial-bg/50 border-b border-white/5">
                <div className="flex items-center gap-2 neuro-text text-gray-400">
                    <FileCode size={16} className="text-industrial-primary" />
                    <span>{title}</span>
                </div>
                <div className="flex gap-2">
                    <button onClick={copyToClipboard} className="p-2 hover:bg-white/5 rounded-md text-gray-500 hover:text-industrial-primary transition-colors" title="Copy to clipboard"><Copy size={14} /></button>
                    <button onClick={downloadGCode} className="p-2 hover:bg-white/5 rounded-md text-gray-500 hover:text-industrial-primary transition-colors" title="Download .nc file"><Download size={14} /></button>
                </div>
            </div>

            {validation && (
                <div className={`flex items-center gap-2 px-4 py-2 neuro-text text-[10px] ${validation.is_valid ? 'bg-neuro-success/10 text-neuro-success' : 'bg-neuro-danger/10 text-neuro-danger'}`}>
                    {validation.is_valid ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
                    <span>{validation.is_valid ? 'PROGRAM VALIDATED' : `${validation.errors?.length || 0} ERRORS DETECTED`}</span>
                    {validation.is_absolute && <span className="ml-auto bg-neuro-success/20 px-2 py-0.5 rounded">G90 ABSOLUTE</span>}
                </div>
            )}

            <div className="flex-1 flex overflow-hidden min-h-0">
                <div className="py-4 px-3 bg-black/20 text-right select-none border-r border-white/5 font-mono text-[10px] text-gray-700 leading-6">
                    {lines.map((_, i) => <div key={i}>{i + 1}</div>)}
                </div>
                <pre className="flex-1 p-4 m-0 font-mono text-xs leading-6 overflow-auto scrollbar-thin scrollbar-thumb-white/10">
                    {lines.map((line, i) => (
                        <div key={i} className={`whitespace-pre ${highlightLine(line)}`}>{line}</div>
                    ))}
                </pre>
            </div>
        </div>
    );
};

export default GCodePreview;
