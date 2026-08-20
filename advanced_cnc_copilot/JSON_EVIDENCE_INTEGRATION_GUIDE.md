# JSON Evidence Mechanics - Integration Guide

## 🎯 System Overview

Complete JSON configuration management system for managing dashboards, components, data sources, forms, and all system configurations with versioning, deployments, and audit trails.

---

## 📦 Components Created

### ✅ Core Files

1. **`erp/json_config_manager.py`** - Registry-based file manager with versioning
2. **`erp/json_config_serializers.py`** - DRF serializers for API
3. **`erp/json_config_views.py`** - API ViewSets with CRUD + versioning
4. **`erp/json_config_urls.py`** - URL routing
5. **`erp/json_config_admin.py`** - Django admin registration
6. **`erp/json_config_models.py`** - Models (needs manual integration)
7. **`cms/theories/JSON_DRIVEN_CONFIGURATION.md`** - Configuration spec
8. **`cms/theories/JSON_EVIDENCE_MECHANICS.md`** - Complete documentation

### ✅ URL Integration

Already integrated in `erp/urls.py`:
```python
path('json-configs/', include('erp.json_config_urls'))  ✅ DONE
```

---

## ⚠️ Manual Integration Required

### **Step 1: Add Models to `erp/models.py`**

Copy the models from `erp/json_config_models.py` and **append them to the bottom** of `erp/models.py` (after ConfigurationProfile model).

The models to add:
- `JSONConfigCategory`
- `JSONConfiguration`
- `JSONConfigVersion`
- `JSONConfigDeployment`
- `JSONConfigTemplate`
- `JSONConfigAuditLog`

**Important:** Change all foreign key references from string quotes to direct model references:
```python
# Change from:
organization = models.ForeignKey('Organization', ...)
created_by = models.ForeignKey('RiseUser', ...)

# They are already in the same file, so Organization and RiseUser are defined above
# Keep them as is if they come BEFORE these new models
```

### **Step 2: Import Admin in `erp/admin.py`**

Add to the top of `erp/admin.py`:
```python
# Add to imports
from erp.json_config_models import (
    JSONConfigCategory,
    JSONConfiguration,
    JSONConfigVersion,
    JSONConfigDeployment,
    JSONConfigTemplate,
    JSONConfigAuditLog
)

# Then copy all @admin.register classes from json_config_admin.py
```

### **Step 3: Run Migrations**

```bash
python manage.py makemigrations
python manage.py migrate
```

### **Step 4: Create Initial Categories**

Via Django admin or Django shell:
```python
python manage.py shell

from erp.models import JSONConfigCategory

categories = [
    {'category_id': 'dashboard', 'name': 'Dashboards', 'description': 'Dashboard configurations', 'icon': '📊'},
    {'category_id': 'component', 'name': 'Components', 'description': 'UI component definitions', 'icon': '🧩'},
    {'category_id': 'data_source', 'name': 'Data Sources', 'description': 'Data source configs', 'icon': '🔌'},
    {'category_id': 'form', 'name': 'Forms', 'description': 'Dynamic form definitions', 'icon': '📝'},
    {'category_id': 'workflow', 'name': 'Workflows', 'description': 'Automation workflows', 'icon': '⚙️'},
    {'category_id': 'theme', 'name': 'Themes', 'description': 'UI themes', 'icon': '🎨'},
    {'category_id': 'api', 'name': 'APIs', 'description': 'API configurations', 'icon': '🔗'},
    {'category_id': 'alert', 'name': 'Alerts', 'description': 'Alert rules', 'icon': '🚨'},
]

for cat in categories:
    JSONConfigCategory.objects.get_or_create(**cat)
```

---

## 🚀 Usage Examples

### **1. Register Dashboard Configuration**

```bash
POST http://localhost:8000/api/json-configs/configs/
Content-Type: application/json

{
  "config_id": "production_dashboard",
  "category": "dashboard",
  "name": "Production Monitoring",
  "description": "Main production floor dashboard",
  "config_data": {
    "dashboard_id": "production_dashboard",
    "layout": { "columns": 12, "row_height": 100 },
    "components": [
      {
        "id": "machine_1",
        "type": "machine-card",
        "position": {"row": 0, "col": 0, "w": 4, "h": 3},
        "data_binding": {
          "source": "flask_api",
          "endpoint": "/api/telemetry/current"
        }
      }
    ]
  },
  "tags": ["production", "monitoring"],
  "organization": 1
}
```

### **2. List All Configurations**

```bash
GET http://localhost:8000/api/json-configs/configs/
GET http://localhost:8000/api/json-configs/configs/?category=dashboard
GET http://localhost:8000/api/json-configs/configs/?tags=production
GET http://localhost:8000/api/json-configs/configs/?search=monitoring
```

### **3. Update Configuration (Auto-versioning)**

```bash
PUT http://localhost:8000/api/json-configs/configs/production_dashboard/
{
  "config_data": { /* updated config */ },
  "change_log": "Added second machine widget"
}

# Version automatically bumps: 1.0.0 → 1.0.1
```

### **4. Rollback to Previous Version**

```bash
POST http://localhost:8000/api/json-configs/configs/production_dashboard/rollback/
{
  "version": "1.0.0"
}

# Creates version 1.0.2 with content from 1.0.0
```

### **5. Deploy to Environment**

```bash
POST http://localhost:8000/api/json-configs/configs/production_dashboard/deploy/
{
  "environment": "PRODUCTION",
  "notes": "Production deployment"
}
```

### **6. Export Configuration**

```bash
GET http://localhost:8000/api/json-configs/configs/production_dashboard/export/
```

### **7. Create from Template**

```bash
POST http://localhost:8000/api/json-configs/templates/basic_dashboard/instantiate/
{
  "config_id": "quality_dashboard",
  "name": "Quality Dashboard",
  "variables": {
    "title": "Quality Monitoring",
    "refresh_rate": 5
  }
}
```

### **8. View Audit Trail**

```bash
GET http://localhost:8000/api/json-configs/audit-logs/?config_id=production_dashboard
```

---

## 📋 API Endpoints Summary

### Categories
- `GET /api/json-configs/categories/` - List categories
- `POST /api/json-configs/categories/` - Create category
- `GET /api/json-configs/categories/{id}/configs/` - Get category configs

### Configurations
- `GET /api/json-configs/configs/` - List with filters
- `POST /api/json-configs/configs/` - Create
- `GET /api/json-configs/configs/{id}/` - Get with history
- `PUT /api/json-configs/configs/{id}/` - Update (auto-version)
- `DELETE /api/json-configs/configs/{id}/` - Delete
- `POST /api/json-configs/configs/{id}/archive/` - Archive
- `POST /api/json-configs/configs/{id}/restore/` - Restore
- `POST /api/json-configs/configs/{id}/rollback/` - Rollback
- `GET /api/json-configs/configs/{id}/validate/` - Validate
- `POST /api/json-configs/configs/{id}/deploy/` - Deploy
- `GET /api/json-configs/configs/{id}/export/` - Export
- `POST /api/json-configs/configs/import_config/` - Import

### Templates
- `GET /api/json-configs/templates/` - List
- `POST /api/json-configs/templates/{id}/instantiate/` - Create from template

### Deployments
- `GET /api/json-configs/deployments/` - List
- `POST /api/json-configs/deployments/{id}/rollback_deployment/` - Rollback

### Audit Logs
- `GET /api/json-configs/audit-logs/` - List with filters

---

## 🎨 Features

### ✅ Versioning
- Automatic semantic versioning (1.0.0 → 1.0.1)
- Complete version history
- Rollback capability
- Change logs

### ✅ Validation
- JSON Schema validation per category
- Automatic content hashing
- Dependency checking

### ✅ Deployment Tracking
- Environment-specific (DEV/STAGING/PROD/TEST)
- Deployment history
- Rollback versions

### ✅ Audit Trail
- All CRUD operations logged
- User attribution
- IP address tracking
- Before/after values

### ✅ Organization
- Tagged configurations
- Categories
- Search functionality
- Dependencies

### ✅ Templates
- Reusable configurations
- Variable placeholders
- Instantiation API
- Usage tracking

---

## 🔧 File-based Manager

For batch operations or backups:

```python
from erp.json_config_manager import JSONConfigRegistry

manager = JSONConfigRegistry('config_backup')

# Export all configs
manager.export_config('dashboard_001', 'backup/dashboard.json')

# Import
manager.import_config('backup/dashboard.json')

# List
configs = manager.list_configs(category='dashboard')

# Search
results = manager.search_configs('monitoring')
```

---

## 💡 Benefits

### For Development
- Version control for configs
- Easy rollback
- Template reuse
- Change tracking

### For Operations  
- Environment deployment
- Configuration validation
- Dependency management
- Complete audit trail

### For Compliance
- Full audit logs
- User attribution
- IP tracking
- Change history

---

## 🎯 Next Steps

1. ✅ **Integrate models** into `erp/models.py`
2. ✅ **Run migrations** 
3. ✅ **Create categories** via admin or shell
4. ✅ **Define JSON schemas** for validation
5. ✅ **Create templates** for common configs
6. ✅ **Import existing** dashboard configs

---

## 📝 Complete Architecture

```
JSON Configuration Management System
├── Models (Database)
│   ├── JSONConfigCategory (categories)
│   ├── JSONConfiguration (main configs)
│   ├── JSONConfigVersion (version history)
│   ├── JSONConfigDeployment (deployment tracking)
│   ├── JSONConfigTemplate (templates)
│   └── JSONConfigAuditLog (audit trail)
├── Serializers (API Data)
│   └── json_config_serializers.py
├── Views (API Logic)
│   └── json_config_views.py
├── URLs (Routing)
│   └── json_config_urls.py
├── Admin (Management UI)
│   └── json_config_admin.py
└── Manager (File Operations)
    └── json_config_manager.py
```

*Complete JSON Evidence Mechanics - Ready for Integration* 🎯
