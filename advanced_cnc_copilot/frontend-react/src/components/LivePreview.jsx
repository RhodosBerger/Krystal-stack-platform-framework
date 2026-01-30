import React from 'react';
import { Eye, FileCode } from 'lucide-react';

const LivePreview = () => {
    const [content, setContent] = useState('');

    useEffect(() => {
        const handleNewPayload = (e) => {
            setContent(e.detail);
        };
        window.addEventListener('new-payload', handleNewPayload);
        return () => window.removeEventListener('new-payload', handleNewPayload);
    }, []);

    return (
        <div className="preview-wrapper">
            <div className="preview-header">
                <Eye size={16} />
                <span>LIVE DOCUMENT PREVIEW</span>
            </div>

            <div className="preview-content">
                <div className="file-header">
                    <FileCode size={20} className="text-info" />
                    <span>{content ? 'GENERATED_PAYLOAD.gcode' : 'Waiting for assembly...'}</span>
                </div>
                <pre className="code-viewer">
                    {content || '(Empty Canvas)'}
                </pre>
            </div>

            <style>{`
        .preview-wrapper { display: flex; flex-direction: column; height: 100%; }
        .preview-header { padding: 15px 0; border-bottom: 1px solid #222; display: flex; align-items: center; gap: 10px; color: #888; font-size: 0.8rem; font-weight: bold; margin-bottom: 20px; }
        .preview-content { flex: 1; background: #000; border-radius: 12px; border: 1px solid #222; padding: 20px; display: flex; flex-direction: column; overflow: hidden; }
        .file-header { display: flex; align-items: center; gap: 10px; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #222; font-size: 0.8rem; color: #eee; }
        .text-info { color: #00d4ff; }
        .code-viewer {
          flex: 1;
          margin: 0;
          font-family: 'Courier New', monospace;
          font-size: 0.75rem;
          color: #00ff88;
          overflow-y: auto;
          line-height: 1.4;
        }
      `}</style>
        </div>
    );
};

export default LivePreview;
