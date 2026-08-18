"""
Developer Tools API Documentation Generator 📚
Responsibility:
1. Auto-generate API documentation.
2. Provide OpenAPI/Swagger enhancements.
3. Generate markdown docs from code.

DEVELOPERS! DEVELOPERS! DEVELOPERS! 🎉
"""
from typing import Dict, Any, List
import json

class DocsGenerator:
    """Generates developer documentation."""
    
    def generate_endpoint_docs(self, endpoints: List[Dict[str, Any]]) -> str:
        """Generates markdown documentation for endpoints."""
        docs = "# API Reference 📡\n\n"
        docs += "> Auto-generated documentation for all API endpoints.\n\n"
        
        # Group by prefix
        groups = {}
        for ep in endpoints:
            path = ep.get("path", "")
            parts = path.split("/")
            group = parts[2] if len(parts) > 2 else "root"
            if group not in groups:
                groups[group] = []
            groups[group].append(ep)
        
        for group, eps in sorted(groups.items()):
            docs += f"## {group.upper()}\n\n"
            for ep in eps:
                path = ep.get("path", "")
                methods = ep.get("methods", [])
                name = ep.get("name", "")
                
                for method in methods:
                    if method in ["HEAD", "OPTIONS"]:
                        continue
                    docs += f"### `{method} {path}`\n"
                    if name:
                        docs += f"**Function:** `{name}`\n\n"
                    docs += "```bash\n"
                    docs += f'curl -X {method} "http://localhost:8000{path}"\n'
                    docs += "```\n\n"
        
        return docs

    def generate_quickstart(self) -> str:
        """Generates a quickstart guide for developers."""
        return '''# 🚀 Developer Quickstart

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd advanced_cnc_copilot

# Start with Docker
docker-compose up -d

# Or run directly
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## Authentication

```python
import requests

# Login
response = requests.post("http://localhost:8000/api/auth/login", json={
    "username": "admin",
    "password": "your_password"
})
token = response.json()["access_token"]

# Use token in requests
headers = {"Authorization": f"Bearer {token}"}
```

## Quick Examples

### Health Check
```python
requests.get("http://localhost:8000/api/health")
```

### List Products
```python
requests.get("http://localhost:8000/api/llm/products", headers=headers)
```

### Generate G-Code
```python
requests.post("http://localhost:8000/api/generate/PROD-001/gcode", 
    json={"format": "fanuc"}, headers=headers)
```

### Debug Console
```python
requests.post("http://localhost:8000/api/debug/console",
    json={"command": "sysinfo"}, headers=headers)
```

## API Base URL
- Local: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:3000`

## Need Help?
Check the API docs at `/docs` or use the Debug Console!
'''

    def generate_changelog(self, phases: int = 52) -> str:
        """Generates a changelog summary."""
        return f'''# 📋 Changelog

## Version 1.0.0 - CNC Copilot Gold Master

**{phases} Development Phases Completed!** 🏁

### Major Features
- 🔧 G-Code Generation Engine (FANUC, Haas, Mazak, Siemens)
- 🤖 LLM Integration (OpenAI, Ollama)
- ⚛️ React Frontend with 23 Components
- 🔒 Security & Isolation Layer
- 🤓 Geek Mode - Debug Console
- 👨‍💻 Developer SDK & Tools

### API Endpoints
- 50+ REST API endpoints
- Full OpenAPI documentation
- Rate limiting & security

### Developer Tools
- SDK Generator (Python, JS, TypeScript, cURL)
- Code Scaffolding (Components, Endpoints, Models)
- Debug Console with 10+ commands
- Performance Profiler

### Infrastructure
- Docker-ready deployment
- TimescaleDB + Redis
- Celery workers
- Flower monitoring
'''


# Global Instance
docs_generator = DocsGenerator()
