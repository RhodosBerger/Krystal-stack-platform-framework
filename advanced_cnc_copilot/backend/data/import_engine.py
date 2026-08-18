"""
Import Engine 📥
Responsibility:
1. Import projects from JSON or ZIP packages.
2. Validate and merge imported data.
"""
import json
import zipfile
import io
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

class ImportEngine:
    def __init__(self):
        self.import_log: List[Dict[str, Any]] = []

    def import_from_json(self, json_string: str) -> Dict[str, Any]:
        """Imports data from JSON string."""
        return json.loads(json_string)

    def import_from_zip(self, zip_data: bytes) -> Dict[str, Any]:
        """Extracts and parses a ZIP package."""
        buffer = io.BytesIO(zip_data)
        extracted = {}
        
        with zipfile.ZipFile(buffer, 'r') as zf:
            for filename in zf.namelist():
                content = zf.read(filename).decode('utf-8')
                if filename.endswith('.json'):
                    extracted[filename] = json.loads(content)
                else:
                    extracted[filename] = content
        
        return extracted

    def validate_package(self, package: Dict[str, Any]) -> Dict[str, Any]:
        """Validates an imported package structure."""
        errors = []
        warnings = []
        
        if "manifest.json" not in package:
            errors.append("Missing manifest.json")
        if "project.json" not in package:
            errors.append("Missing project.json")
        
        manifest = package.get("manifest.json", {})
        if manifest.get("version") != "1.0.0":
            warnings.append(f"Version mismatch: expected 1.0.0, got {manifest.get('version')}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    def import_project(self, zip_data: bytes) -> Dict[str, Any]:
        """Imports a full project from ZIP package."""
        import_id = f"IMP-{uuid.uuid4().hex[:6].upper()}"
        
        try:
            package = self.import_from_zip(zip_data)
            validation = self.validate_package(package)
            
            if not validation["valid"]:
                return {
                    "import_id": import_id,
                    "status": "FAILED",
                    "errors": validation["errors"]
                }
            
            project_data = package.get("project.json", {})
            manifest = package.get("manifest.json", {})
            
            result = {
                "import_id": import_id,
                "status": "SUCCESS",
                "original_export_id": manifest.get("export_id"),
                "project": project_data,
                "warnings": validation["warnings"]
            }
            
            self.import_log.append({
                "import_id": import_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "SUCCESS"
            })
            
            return result
            
        except Exception as e:
            return {
                "import_id": import_id,
                "status": "ERROR",
                "message": str(e)
            }

    def get_import_history(self) -> List[Dict[str, Any]]:
        return self.import_log

# Global Instance
import_engine = ImportEngine()
