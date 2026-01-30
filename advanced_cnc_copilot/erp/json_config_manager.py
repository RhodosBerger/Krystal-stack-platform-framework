"""
JSON Configuration Manager - Evidence Mechanics
Manages multiple JSON configurations for various system parts
"""

import json
import jsonschema
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib


class JSONConfigRegistry:
    """
    Central registry for managing JSON configurations
    Supports versioning, validation, categorization
    """
    
    # Configuration categories
    CATEGORIES = {
        'dashboard': 'Dashboard configurations',
        'component': 'Component definitions',
        'data_source': 'Data source configurations',
        'form': 'Dynamic form definitions',
        'workflow': 'Workflow and automation rules',
        'theme': 'UI theme configurations',
        'api': 'API endpoint configurations',
        'alert': 'Alert and notification rules',
        'report': 'Report templates',
        'integration': 'External integration configs'
    }
    
    # JSON Schemas for validation
    SCHEMAS = {
        'dashboard': {
            "type": "object",
            "required": ["dashboard_id", "name", "components"],
            "properties": {
                "dashboard_id": {"type": "string"},
                "name": {"type": "string"},
                "version": {"type": "string"},
                "layout": {"type": "object"},
                "data_sources": {"type": "array"},
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "type"],
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string"},
                            "position": {"type": "object"},
                            "data_binding": {"type": "object"}
                        }
                    }
                }
            }
        },
        
        'component': {
            "type": "object",
            "required": ["component_id", "component_type", "config"],
            "properties": {
                "component_id": {"type": "string"},
                "component_type": {"type": "string"},
                "config": {"type": "object"}
            }
        },
        
        'data_source': {
            "type": "object",
            "required": ["source_id", "source_type"],
            "properties": {
                "source_id": {"type": "string"},
                "source_type": {"type": "string", "enum": ["rest_api", "websocket", "database"]},
                "base_url": {"type": "string"},
                "endpoints": {"type": "object"}
            }
        },
        
        'form': {
            "type": "object",
            "required": ["form_id", "title", "sections"],
            "properties": {
                "form_id": {"type": "string"},
                "title": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["section_id", "fields"],
                        "properties": {
                            "section_id": {"type": "string"},
                            "title": {"type": "string"},
                            "fields": {"type": "array"}
                        }
                    }
                }
            }
        }
    }
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path('config_registry')
        self.base_path.mkdir(exist_ok=True, parents=True)
        self.registry_file = self.base_path / 'registry.json'
        self.load_registry()
    
    def load_registry(self):
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'configs': {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_configs': 0
                }
            }
    
    def save_registry(self):
        """Save registry to disk"""
        self.registry['metadata']['updated_at'] = datetime.now().isoformat()
        self.registry['metadata']['total_configs'] = len(self.registry['configs'])
        
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def generate_config_hash(self, config: Dict) -> str:
        """Generate hash for config content"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def validate_config(self, config: Dict, category: str) -> tuple[bool, Optional[str]]:
        """
        Validate config against schema
        Returns (is_valid, error_message)
        """
        if category not in self.SCHEMAS:
            return True, None  # No schema defined, skip validation
        
        try:
            jsonschema.validate(instance=config, schema=self.SCHEMAS[category])
            return True, None
        except jsonschema.exceptions.ValidationError as e:
            return False, str(e)
    
    def register_config(
        self,
        config_id: str,
        config: Dict,
        category: str,
        description: str = '',
        tags: List[str] = None,
        auto_validate: bool = True
    ) -> Dict[str, Any]:
        """
        Register new JSON configuration
        Returns registration result with metadata
        """
        # Validate category
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category. Must be one of: {list(self.CATEGORIES.keys())}")
        
        # Validate config structure
        if auto_validate:
            is_valid, error = self.validate_config(config, category)
            if not is_valid:
                raise ValueError(f"Config validation failed: {error}")
        
        # Generate hash and version
        config_hash = self.generate_config_hash(config)
        
        # Check if config already exists
        if config_id in self.registry['configs']:
            # Update existing config (create new version)
            return self.update_config(config_id, config, f"Updated config")
        
        # Create new config entry
        config_entry = {
            'config_id': config_id,
            'category': category,
            'description': description,
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'hash': config_hash,
            'versions': [
                {
                    'version': '1.0.0',
                    'hash': config_hash,
                    'created_at': datetime.now().isoformat(),
                    'change_log': 'Initial version'
                }
            ],
            'status': 'active'
        }
        
        # Save config file
        config_path = self.get_config_path(config_id, category)
        config_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update registry
        self.registry['configs'][config_id] = config_entry
        self.save_registry()
        
        return {
            'status': 'registered',
            'config_id': config_id,
            'version': '1.0.0',
            'path': str(config_path)
        }
    
    def update_config(
        self,
        config_id: str,
        config: Dict,
        change_log: str = '',
        version_bump: str = 'patch'
    ) -> Dict[str, Any]:
        """
        Update existing configuration (creates new version)
        version_bump: 'major', 'minor', 'patch'
        """
        if config_id not in self.registry['configs']:
            raise ValueError(f"Config '{config_id}' not found")
        
        entry = self.registry['configs'][config_id]
        
        # Validate
        is_valid, error = self.validate_config(config, entry['category'])
        if not is_valid:
            raise ValueError(f"Config validation failed: {error}")
        
        # Generate new hash
        new_hash = self.generate_config_hash(config)
        
        # Check if content actually changed
        if new_hash == entry['hash']:
            return {
                'status': 'unchanged',
                'config_id': config_id,
                'version': entry['version']
            }
        
        # Bump version
        current_version = entry['version']
        new_version = self.bump_version(current_version, version_bump)
        
        # Save new version
        config_path = self.get_config_path(config_id, entry['category'])
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Archive old version
        archive_path = self.get_archive_path(config_id, current_version)
        archive_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Update entry
        entry['version'] = new_version
        entry['hash'] = new_hash
        entry['updated_at'] = datetime.now().isoformat()
        entry['versions'].append({
            'version': new_version,
            'hash': new_hash,
            'created_at': datetime.now().isoformat(),
            'change_log': change_log or f'Version {new_version}'
        })
        
        self.save_registry()
        
        return {
            'status': 'updated',
            'config_id': config_id,
            'old_version': current_version,
            'new_version': new_version,
            'path': str(config_path)
        }
    
    def get_config(self, config_id: str, version: str = None) -> Dict:
        """Get configuration by ID and optional version"""
        if config_id not in self.registry['configs']:
            raise ValueError(f"Config '{config_id}' not found")
        
        entry = self.registry['configs'][config_id]
        
        if version and version != entry['version']:
            # Load from archive
            archive_path = self.get_archive_path(config_id, version)
            if not archive_path.exists():
                raise ValueError(f"Version {version} not found")
            
            with open(archive_path, 'r') as f:
                return json.load(f)
        
        # Load current version
        config_path = self.get_config_path(config_id, entry['category'])
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def delete_config(self, config_id: str, permanent: bool = False):
        """Delete or archive configuration"""
        if config_id not in self.registry['configs']:
            raise ValueError(f"Config '{config_id}' not found")
        
        entry = self.registry['configs'][config_id]
        
        if permanent:
            # Permanently delete
            config_path = self.get_config_path(config_id, entry['category'])
            config_path.unlink(missing_ok=True)
            del self.registry['configs'][config_id]
        else:
            # Soft delete (archive)
            entry['status'] = 'archived'
            entry['archived_at'] = datetime.now().isoformat()
        
        self.save_registry()
    
    def list_configs(
        self,
        category: str = None,
        tags: List[str] = None,
        status: str = 'active'
    ) -> List[Dict]:
        """List configurations with filters"""
        results = []
        
        for config_id, entry in self.registry['configs'].items():
            # Filter by status
            if status and entry['status'] != status:
                continue
            
            # Filter by category
            if category and entry['category'] != category:
                continue
            
            # Filter by tags
            if tags and not any(tag in entry['tags'] for tag in tags):
                continue
            
            results.append({
                'config_id': config_id,
                'category': entry['category'],
                'description': entry['description'],
                'version': entry['version'],
                'tags': entry['tags'],
                'updated_at': entry['updated_at']
            })
        
        return results
    
    def search_configs(self, query: str) -> List[Dict]:
        """Search configurations by text"""
        results = []
        query_lower = query.lower()
        
        for config_id, entry in self.registry['configs'].items():
            if (query_lower in config_id.lower() or
                query_lower in entry['description'].lower() or
                any(query_lower in tag.lower() for tag in entry['tags'])):
                
                results.append({
                    'config_id': config_id,
                    'category': entry['category'],
                    'description': entry['description'],
                    'version': entry['version']
                })
        
        return results
    
    def get_config_path(self, config_id: str, category: str) -> Path:
        """Get file path for configuration"""
        return self.base_path / category / f"{config_id}.json"
    
    def get_archive_path(self, config_id: str, version: str) -> Path:
        """Get archive path for specific version"""
        return self.base_path / 'archive' / config_id / f"{version}.json"
    
    def bump_version(self, current: str, bump_type: str) -> str:
        """Bump semantic version"""
        major, minor, patch = map(int, current.split('.'))
        
        if bump_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif bump_type == 'minor':
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        return f"{major}.{minor}.{patch}"
    
    def export_config(self, config_id: str, output_path: str):
        """Export configuration to file"""
        config = self.get_config(config_id)
        entry = self.registry['configs'][config_id]
        
        export_data = {
            'metadata': {
                'config_id': config_id,
                'category': entry['category'],
                'version': entry['version'],
                'exported_at': datetime.now().isoformat()
            },
            'config': config
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_config(self, import_path: str, override: bool = False):
        """Import configuration from file"""
        with open(import_path, 'r') as f:
            import_data = json.load(f)
        
        metadata = import_data.get('metadata', {})
        config = import_data.get('config', {})
        
        config_id = metadata.get('config_id')
        category = metadata.get('category')
        
        if not config_id or not category:
            raise ValueError("Import file must contain metadata with config_id and category")
        
        if config_id in self.registry['configs'] and not override:
            raise ValueError(f"Config '{config_id}' already exists. Use override=True to replace.")
        
        return self.register_config(
            config_id=config_id,
            config=config,
            category=category,
            description=f"Imported from {import_path}"
        )


# ===== USAGE EXAMPLES =====

if __name__ == '__main__':
    manager = JSONConfigRegistry('config_registry')
    
    # Example 1: Register dashboard config
    dashboard_config = {
        "dashboard_id": "production_monitoring",
        "name": "Production Monitoring Dashboard",
        "version": "1.0.0",
        "components": [
            {
                "id": "machine_1",
                "type": "machine-card",
                "position": {"row": 0, "col": 0, "w": 4, "h": 3}
            }
        ]
    }
    
    result = manager.register_config(
        config_id='production_monitoring',
        config=dashboard_config,
        category='dashboard',
        description='Main production monitoring dashboard',
        tags=['production', 'monitoring', 'cnc']
    )
    print(f"Registered: {result}")
    
    # Example 2: List all dashboard configs
    dashboards = manager.list_configs(category='dashboard')
    print(f"Dashboards: {dashboards}")
    
    # Example 3: Update config
    dashboard_config['components'].append({
        "id": "machine_2",
        "type": "machine-card",
        "position": {"row": 0, "col": 4, "w": 4, "h": 3}
    })
    
    update_result = manager.update_config(
        config_id='production_monitoring',
        config=dashboard_config,
        change_log='Added second machine card'
    )
    print(f"Updated: {update_result}")
