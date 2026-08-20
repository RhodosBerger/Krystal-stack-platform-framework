"""
Flask Application - Web Interface for Document Analysis

Provides:
- File upload interface
- Document management
- LLM chat interface
- Analysis results viewer
- Autocomplete API
"""

import os
import json
from typing import Optional
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import logging

from .document_processor import DocumentProcessor, ProcessingStatus
from .llm_bridge import LLMBridge, LLMConfig, create_llm_bridge
from .mongo_store import MongoStore, create_mongo_store

logger = logging.getLogger(__name__)

# HTML Templates
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GAMESA Document Analyzer</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00d9ff; }
        .card { background: #16213e; border-radius: 8px; padding: 20px; margin: 20px 0; }
        .upload-zone { border: 2px dashed #00d9ff; padding: 40px; text-align: center; cursor: pointer; }
        .upload-zone:hover { background: rgba(0,217,255,0.1); }
        input[type="file"] { display: none; }
        button { background: #00d9ff; color: #1a1a2e; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-weight: bold; }
        button:hover { background: #00b8d4; }
        .doc-list { list-style: none; padding: 0; }
        .doc-item { background: #0f3460; padding: 15px; margin: 10px 0; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; }
        .doc-item .status { padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        .status.completed { background: #00c853; }
        .status.processing { background: #ffc107; color: #000; }
        .status.failed { background: #ff5252; }
        .status.pending { background: #607d8b; }
        .chat-box { height: 400px; overflow-y: auto; background: #0f3460; padding: 15px; border-radius: 4px; margin-bottom: 15px; }
        .chat-input { display: flex; gap: 10px; }
        .chat-input input { flex: 1; padding: 10px; border: none; border-radius: 4px; background: #16213e; color: #eee; }
        .message { margin: 10px 0; padding: 10px; border-radius: 4px; }
        .message.user { background: #1e88e5; margin-left: 20%; }
        .message.assistant { background: #0f3460; margin-right: 20%; }
        .analysis { background: #0f3460; padding: 15px; border-radius: 4px; white-space: pre-wrap; font-family: monospace; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; }
        .tab { padding: 10px 20px; background: #16213e; border-radius: 4px 4px 0 0; cursor: pointer; }
        .tab.active { background: #00d9ff; color: #1a1a2e; }
        .progress { height: 4px; background: #0f3460; border-radius: 2px; overflow: hidden; }
        .progress-bar { height: 100%; background: #00d9ff; transition: width 0.3s; }
    </style>
</head>
<body>
    <div class="container">
        <h1>GAMESA Document Analyzer</h1>

        <div class="tabs">
            <div class="tab active" onclick="showTab('upload')">Upload</div>
            <div class="tab" onclick="showTab('documents')">Documents</div>
            <div class="tab" onclick="showTab('chat')">Chat</div>
            <div class="tab" onclick="showTab('analysis')">Analysis</div>
        </div>

        <div id="upload" class="card">
            <h2>Upload Document</h2>
            <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                <p>Drop files here or click to upload</p>
                <p style="color: #888;">Supports: PDF, Images, Text, JSON, CSV</p>
            </div>
            <input type="file" id="fileInput" multiple onchange="uploadFiles(this.files)">
            <div id="uploadProgress" style="margin-top: 15px;"></div>
        </div>

        <div id="documents" class="card" style="display:none;">
            <h2>Documents</h2>
            <ul class="doc-list" id="docList"></ul>
        </div>

        <div id="chat" class="card" style="display:none;">
            <h2>Chat with Documents</h2>
            <div class="chat-box" id="chatBox"></div>
            <div class="chat-input">
                <input type="text" id="chatInput" placeholder="Ask about your documents..." onkeypress="if(event.key==='Enter')sendChat()">
                <button onclick="sendChat()">Send</button>
            </div>
        </div>

        <div id="analysis" class="card" style="display:none;">
            <h2>Analysis Results</h2>
            <select id="docSelect" onchange="loadAnalysis(this.value)" style="padding:10px;margin-bottom:15px;width:100%;">
                <option value="">Select a document...</option>
            </select>
            <div class="analysis" id="analysisResult">Select a document to view analysis</div>
        </div>
    </div>

    <script>
        let docs = [];

        function showTab(name) {
            document.querySelectorAll('.card').forEach(c => c.style.display = 'none');
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(name).style.display = 'block';
            event.target.classList.add('active');
            if (name === 'documents') loadDocuments();
        }

        async function uploadFiles(files) {
            const progress = document.getElementById('uploadProgress');
            for (let file of files) {
                progress.innerHTML = `<div class="progress"><div class="progress-bar" style="width:50%"></div></div><p>Uploading ${file.name}...</p>`;

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
                    const data = await resp.json();
                    progress.innerHTML = `<p style="color:#00c853">Uploaded: ${file.name} (${data.doc_id})</p>`;
                } catch (e) {
                    progress.innerHTML = `<p style="color:#ff5252">Failed: ${file.name}</p>`;
                }
            }
        }

        async function loadDocuments() {
            const resp = await fetch('/api/documents');
            docs = await resp.json();
            const list = document.getElementById('docList');
            const select = document.getElementById('docSelect');

            list.innerHTML = docs.map(d => `
                <li class="doc-item">
                    <div>
                        <strong>${d.filename}</strong>
                        <br><small>${d.doc_type} - ${(d.size_bytes/1024).toFixed(1)}KB</small>
                    </div>
                    <div>
                        <span class="status ${d.status}">${d.status}</span>
                        <button onclick="analyzeDoc('${d.id}')" style="margin-left:10px;">Analyze</button>
                    </div>
                </li>
            `).join('');

            select.innerHTML = '<option value="">Select a document...</option>' +
                docs.map(d => `<option value="${d.id}">${d.filename}</option>`).join('');
        }

        async function analyzeDoc(docId) {
            const resp = await fetch(`/api/analyze/${docId}`, { method: 'POST' });
            const data = await resp.json();
            alert(data.message || 'Analysis started');
            loadDocuments();
        }

        async function loadAnalysis(docId) {
            if (!docId) return;
            const resp = await fetch(`/api/analysis/${docId}`);
            const data = await resp.json();
            document.getElementById('analysisResult').textContent = JSON.stringify(data, null, 2);
        }

        async function sendChat() {
            const input = document.getElementById('chatInput');
            const box = document.getElementById('chatBox');
            const query = input.value.trim();
            if (!query) return;

            box.innerHTML += `<div class="message user">${query}</div>`;
            input.value = '';

            const resp = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ query, doc_ids: docs.map(d => d.id) })
            });
            const data = await resp.json();
            box.innerHTML += `<div class="message assistant">${data.response}</div>`;
            box.scrollTop = box.scrollHeight;
        }

        // Initial load
        loadDocuments();
    </script>
</body>
</html>
"""


def create_app(
    upload_dir: str = "/tmp/gamesa_uploads",
    llm_url: str = "http://localhost:1234/v1",
    mongo_host: str = "localhost",
) -> Flask:
    """Create Flask application."""

    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

    # Initialize services
    processor = DocumentProcessor(upload_dir=upload_dir)
    llm = create_llm_bridge(base_url=llm_url)
    store = create_mongo_store(host=mongo_host)

    @app.route('/')
    def index():
        return render_template_string(INDEX_TEMPLATE)

    @app.route('/api/upload', methods=['POST'])
    def upload():
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        content = file.read()
        tags = request.form.getlist('tags')

        # Upload and process
        record = processor.upload(filename, content, tags)
        processor.process(record.id)

        # Save to MongoDB
        store.save_document({
            'doc_id': record.id,
            'filename': record.filename,
            'doc_type': record.doc_type.value,
            'size_bytes': record.size_bytes,
            'hash': record.hash,
            'status': record.status.value,
            'tags': record.tags,
        })

        store.log_audit('upload', 'document', record.id, details={'filename': filename})

        return jsonify({
            'doc_id': record.id,
            'filename': record.filename,
            'status': record.status.value,
        })

    @app.route('/api/documents')
    def list_documents():
        records = processor.list_documents()
        return jsonify([{
            'id': r.id,
            'filename': r.filename,
            'doc_type': r.doc_type.value,
            'size_bytes': r.size_bytes,
            'status': r.status.value,
            'tags': r.tags,
        } for r in records])

    @app.route('/api/analyze/<doc_id>', methods=['POST'])
    def analyze(doc_id: str):
        analysis_type = request.json.get('type', 'financial') if request.json else 'financial'

        # Set LLM bridge
        processor.llm = llm
        record = processor.analyze_with_llm(doc_id, analysis_type)

        if record.llm_analysis:
            store.save_analysis(
                doc_id=doc_id,
                analysis_type=analysis_type,
                result=record.llm_analysis.get('result', {}),
                tokens_used=record.llm_analysis.get('tokens_used', 0),
            )

        return jsonify({
            'message': 'Analysis complete',
            'analysis': record.llm_analysis,
        })

    @app.route('/api/analysis/<doc_id>')
    def get_analysis(doc_id: str):
        record = processor.get_document(doc_id)
        if not record:
            return jsonify({'error': 'Document not found'}), 404

        return jsonify({
            'doc_id': doc_id,
            'filename': record.filename,
            'extracted_text': record.extracted.text[:2000] if record.extracted else None,
            'llm_analysis': record.llm_analysis,
        })

    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.json
        query = data.get('query', '')
        doc_ids = data.get('doc_ids', [])

        # Build context from documents
        context_docs = []
        for doc_id in doc_ids[:5]:  # Limit to 5 docs
            record = processor.get_document(doc_id)
            if record and record.extracted:
                context_docs.append({
                    'source': record.filename,
                    'content': record.extracted.text[:2000],
                })

        if context_docs:
            response = llm.rag_query(query, context_docs)
        else:
            response = llm.generate(query)

        return jsonify({
            'response': response.content,
            'tokens_used': response.tokens_used,
        })

    @app.route('/api/autocomplete', methods=['POST'])
    def autocomplete():
        data = request.json
        partial = data.get('text', '')
        field_type = data.get('field_type', 'general')

        # Get history
        history = store.get_autocomplete_history(field_type, limit=10)

        # Get LLM suggestions
        response = llm.suggest_autocomplete(partial, field_type, history)

        try:
            suggestions = json.loads(response.content)
        except:
            suggestions = [response.content]

        # Save for future
        if partial:
            store.save_autocomplete(field_type, partial)

        return jsonify({'suggestions': suggestions})

    @app.route('/api/batch-analyze', methods=['POST'])
    def batch_analyze():
        data = request.json
        doc_ids = data.get('doc_ids', [])
        analysis_type = data.get('type', 'financial')

        processor.llm = llm
        result = processor.batch_analyze(doc_ids, analysis_type)

        return jsonify(result)

    @app.route('/api/stats')
    def stats():
        return jsonify({
            'documents': len(processor.documents),
            'db_stats': store.get_stats(),
            'llm_cache': llm.cache.stats(),
            'llm_available': llm.health_check(),
        })

    @app.route('/api/health')
    def health():
        return jsonify({
            'status': 'ok',
            'llm': llm.health_check(),
        })

    return app


def run_app(host: str = '0.0.0.0', port: int = 5000, debug: bool = True):
    """Run the Flask application."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app()
