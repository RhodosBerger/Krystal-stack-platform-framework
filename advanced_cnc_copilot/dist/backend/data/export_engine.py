"""
Export Engine 📤
Responsibility:
1. Export projects, products, and payloads in multiple formats.
2. Create portable ZIP packages for migration.
"""
import json
import csv
import io
import zipfile
import uuid
from typing import Dict, Any, List
from datetime import datetime, timezone

class ExportFormat:
    JSON = "JSON"
    CSV = "CSV"
    ZIP = "ZIP"

class ExportEngine:
    def __init__(self):
        self.export_log: List[Dict[str, Any]] = []

    def export_to_json(self, data: Dict[str, Any]) -> str:
        """Exports data as JSON string."""
        return json.dumps(data, indent=2)

    def export_to_csv(self, data: List[Dict[str, Any]]) -> str:
        """Exports list of dicts as CSV string."""
        if not data:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()

    def export_to_zip(self, files: Dict[str, str]) -> bytes:
        """Creates a ZIP archive from filename->content dict."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename, content in files.items():
                zf.writestr(filename, content)
        return buffer.getvalue()

    def export_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Exports a full project package."""
        export_id = f"EXP-{uuid.uuid4().hex[:6].upper()}"
        
        # Create multi-file package
        files = {
            "project.json": self.export_to_json(project_data),
            "manifest.json": self.export_to_json({
                "export_id": export_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "type": "PROJECT_BUNDLE"
            })
        }
        
        # Add products as CSV if present
        if "products" in project_data and project_data["products"]:
            files["products.csv"] = self.export_to_csv(project_data["products"])
        
        zip_data = self.export_to_zip(files)
        
        result = {
            "export_id": export_id,
            "format": ExportFormat.ZIP,
            "size_bytes": len(zip_data),
            "files": list(files.keys()),
            "data": zip_data
        }
        
        self.export_log.append({
            "export_id": export_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return result

    def get_export_history(self) -> List[Dict[str, Any]]:
        return self.export_log

# Global Instance
export_engine = ExportEngine()
