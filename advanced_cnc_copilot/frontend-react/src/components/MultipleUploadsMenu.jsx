import React, { useState } from 'react';
import { Upload, X, CheckCircle, AlertCircle } from 'lucide-react';
import axios from 'axios';

const MultipleUploadsMenu = ({ onUploadSuccess }) => {
    const [files, setFiles] = useState([]);
    const [uploading, setUploading] = useState(false);

    const handleFileChange = (e) => {
        setFiles([...files, ...Array.from(e.target.files)]);
    };

    const removeFile = (index) => {
        setFiles(files.filter((_, i) => i !== index));
    };

    const uploadFiles = async () => {
        if (files.length === 0) return;
        setUploading(true);

        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });

        try {
            const response = await axios.post('/api/upload/batch', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            if (response.data.status === 'SUCCESS') {
                const newElements = response.data.uploaded.map(u => ({
                    id: u.id,
                    filename: u.filename,
                    type: 'CUSTOM_UPLOAD'
                }));
                onUploadSuccess(newElements);
                setFiles([]);
            }
        } catch (error) {
            console.error('Batch Upload Failed:', error);
            alert("Upload failed. Verify backend connection.");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="upload-menu">
            <div className="upload-dropzone" onClick={() => document.getElementById('file-input').click()}>
                <Upload size={24} className="text-dim" />
                <span>Drop elements or click to browse</span>
                <input
                    id="file-input"
                    type="file"
                    multiple
                    hidden
                    onChange={handleFileChange}
                />
            </div>

            {files.length > 0 && (
                <div className="file-staging">
                    {files.map((file, idx) => (
                        <div key={idx} className="file-item">
                            <span>{file.name}</span>
                            <X size={14} className="remove-btn" onClick={() => removeFile(idx)} />
                        </div>
                    ))}
                    <button className="upload-confirm" onClick={uploadFiles} disabled={uploading}>
                        {uploading ? 'UPLOADING...' : `INJECT ${files.length} ELEMENTS`}
                    </button>
                </div>
            )}

            <style>{`
        .upload-menu { width: 100%; }
        .upload-dropzone {
          border: 2px dashed #444;
          border-radius: 12px;
          padding: 20px;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 10px;
          cursor: pointer;
          transition: all 0.3s;
          color: #888;
          font-size: 0.8rem;
        }
        .upload-dropzone:hover { border-color: #00ff88; color: #eee; background: rgba(0,255,136,0.05); }
        .file-staging { margin-top: 15px; display: flex; flex-direction: column; gap: 8px; }
        .file-item {
          background: #1a1a1e;
          padding: 8px 12px;
          border-radius: 6px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 0.75rem;
        }
        .remove-btn { color: #ff4d4d; cursor: pointer; }
        .upload-confirm {
          margin-top: 10px;
          background: #00ff88;
          color: #000;
          border: none;
          padding: 10px;
          border-radius: 6px;
          font-weight: bold;
          font-size: 0.8rem;
          cursor: pointer;
          transition: 0.2s;
        }
        .upload-confirm:hover { background: #00cc6a; transform: scale(1.02); }
        .upload-confirm:disabled { background: #555; cursor: not-allowed; }
      `}</style>
        </div>
    );
};

export default MultipleUploadsMenu;
