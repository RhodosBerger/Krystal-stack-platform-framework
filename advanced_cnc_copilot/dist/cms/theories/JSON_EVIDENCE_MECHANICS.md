# JSON Evidence Mechanics - Complete System Documentation

## 🎯 Overview

**JSON Evidence Mechanics** is a comprehensive configuration management system that treats JSON configurations as first-class citizens with full lifecycle management, versioning, deployment tracking, and audit trails.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│           JSON Configuration Management              │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Categories  │  │  Templates   │  │  Registry │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │Configurations│  │   Versions   │  │Deployments│ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐                │
│  │  Validation  │  │  Audit Logs  │                │
│  └──────────────┘  └──────────────┘                │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Core Components

### 1. **Configuration Categories**

Organize configurations by type:
- `dashboard` - Dashboard layouts
- `component` - UI component definitions
- `data_source` - Data source configurations
- `form` - Dynamic form definitions
- `workflow` - Automation rules
- `theme` - UI themes
- `api` - API endpoint configs
- `alert` - Alert rules
- `report` - Report templates
- `integration` - External integrations

### 2. **Configurations**

Main configuration storage with:
- ✅ Unique `config_id`
- ✅ JSON data storage
- ✅ Automatic versioning
- ✅ Content hashing
- ✅ Tags for categorization
- ✅ Status lifecycle (DRAFT → ACTIVE → ARCHIVED)
- ✅ Dependencies tracking

### 3. **Version History**

Every change is tracked:
- Semantic versioning (1.0.0 → 1.0.1)
- Change logs
- User attribution
- Diff summaries
- Rollback capability

### 4. **Deployments**

Track where configs are deployed:
- Environment targeting (DEV, STAGING, PROD, TEST)
- Deployment status
- Rollback versions
- Deployment notes

### 5. **Templates**

Reusable configuration blueprints:
- Variable placeholders `{{variable_name}}`
- Public/private templates
- Usage tracking
- Instantiation API

### 6. **Audit Logs**

Complete audit trail:
- All CRUD operations
- Deployments and rollbacks
- User attribution
- IP address tracking
- Before/after values

---

## 🔧 API Endpoints

### **Categories**

```
GET    /api/json-configs/categories/              # List all categories
POST   /api/json-configs/categories/              # Create category
GET    /api/json-configs/categories/{id}/         # Get category
GET    /api/json-configs/categories/{id}/configs/ # Get category configs
```

### **Configurations**

```
GET    /api/json-configs/configs/                      # List configs
POST   /api/json-configs/configs/                      # Create config
GET    /api/json-configs/configs/{id}/                 # Get config with history
PUT    /api/json-configs/configs/{id}/                 # Update config
DELETE /api/json-configs/configs/{id}/                 # Delete config

# Actions
POST   /api/json-configs/configs/{id}/archive/         # Archive config
POST   /api/json-configs/configs/{id}/restore/         # Restore config
POST   /api/json-configs/configs/{id}/rollback/        # Rollback version
GET    /api/json-configs/configs/{id}/validate/        # Validate schema
POST   /api/json-configs/configs/{id}/deploy/          # Deploy to environment
GET    /api/json-configs/configs/{id}/export/          # Export as JSON
POST   /api/json-configs/configs/import_config/        # Import JSON
```

### **Templates**

```
GET    /api/json-configs/templates/                    # List templates
POST   /api/json-configs/templates/                    # Create template
GET    /api/json-configs/templates/{id}/               # Get template
POST   /api/json-configs/templates/{id}/instantiate/   # Create from template
```

### **Deployments**

```
GET    /api/json-configs/deployments/                  # List deployments
POST   /api/json-configs/deployments/                  # Create deployment
POST   /api/json-configs/deployments/{id}/rollback_deployment/  # Rollback
```

### **Audit Logs**

```
GET    /api/json-configs/audit-logs/                   # List audit logs
GET    /api/json-configs/audit-logs/{id}/              # Get audit log
```

---

## 💡 Usage Examples

### **1. Register Dashboard Configuration**

```bash
POST /api/json-configs/configs/
{
  "config_id": "production_monitoring",
  "category": "dashboard",
  "name": "Production Monitoring Dashboard",
  "description": "Main production floor monitoring",
  "config_data": {
    "dashboard_id": "production_monitoring",
    "components": [
      {
        "id": "machine_1",
        "type": "machine-card",
        "position": {"row": 0, "col": 0, "w": 4, "h": 3}
      }
    ]
  },
  "tags": ["production", "monitoring", "cnc"],
  "organization": 1
}
```

### **2. Update Configuration (Auto-versioning)**

```bash
PUT /api/json-configs/configs/production_monitoring/
{
  "config_data": {
    "dashboard_id": "production_monitoring",
    "components": [
      {
        "id": "machine_1",
        "type": "machine-card",
        "position": {"row": 0, "col": 0, "w": 4, "h": 3}
      },
      {
        "id": "machine_2",
        "type": "machine-card",
        "position": {"row": 0, "col": 4, "w": 4, "h": 3}
      }
    ]
  },
  "change_log": "Added second machine card"
}

# Response: version bumped from 1.0.0 → 1.0.1
```

### **3. Deploy to Production**

```bash
POST /api/json-configs/configs/production_monitoring/deploy/
{
  "environment": "PRODUCTION",
  "notes": "Initial production deployment"
}
```

### **4. Rollback to Previous Version**

```bash
POST /api/json-configs/configs/production_monitoring/rollback/
{
  "version": "1.0.0"
}

# Creates new version 1.0.2 with content from 1.0.0
```

### **5. Create from Template**

```bash
POST /api/json-configs/templates/basic_dashboard/instantiate/
{
  "config_id": "quality_dashboard",
  "name": "Quality Monitoring Dashboard",
  "variables": {
    "title": "Quality Dashboard",
    "refresh_rate": 5
  }
}
```

### **6. Search Configurations**

```bash
# By category
GET /api/json-configs/configs/?category=dashboard

# By tags
GET /api/json-configs/configs/?tags=production,monitoring

# By status
GET /api/json-configs/configs/?status=ACTIVE

# Search text
GET /api/json-configs/configs/?search=monitoring
```

### **7. Get Audit Trail**

```bash
GET /api/json-configs/audit-logs/?config_id=production_monitoring
```

---

## 🔒 Validation & Schema

Configurations can be validated against JSON Schema:

```python
# Category with schema
{
  "category_id": "dashboard",
  "schema": {
    "type": "object",
    "required": ["dashboard_id", "components"],
    "properties": {
      "dashboard_id": {"type": "string"},
      "components": {
        "type": "array",
        "items": {
          "required": ["id", "type"],
          "properties": {
            "id": {"type": "string"},
            "type": {"type": "string"}
          }
        }
      }
    }
  }
}

# Validate endpoint
GET /api/json-configs/configs/production_monitoring/validate/
```

---

## 📦 File-based Registry

`JSONConfigRegistry` class provides file-based management:

```python
from erp.json_config_manager import JSONConfigRegistry

manager = JSONConfigRegistry('config_registry')

# Register
result = manager.register_config(
    config_id='my_dashboard',
    config={...},
    category='dashboard',
    tags=['custom']
)

# Update
result = manager.update_config(
    config_id='my_dashboard',
    config={...},
    change_log='Updated layout'
)

# Get
config = manager.get_config('my_dashboard')

# List
configs = manager.list_configs(category='dashboard', status='active')

# Search
results = manager.search_configs('monitoring')

# Export/Import
manager.export_config('my_dashboard', 'backup.json')
manager.import_config('backup.json')
```

---

## 🎨 Benefits

### **For Developers**
✅ Version control for configurations  
✅ Easy rollback on errors  
✅ Template reusability  
✅ Complete audit trail  

### **For Operations**
✅ Environment-specific deployments  
✅ Change tracking  
✅ Configuration validation  
✅ Dependency management  

### **For Compliance**
✅ Full audit logs  
✅ User attribution  
✅ Before/after tracking  
✅ IP address logging  

---

## 🚀 Next Steps

1. **Create initial categories** via Django admin
2. **Define JSON schemas** for validation
3. **Create templates** for common configs
4. **Import existing configs** to centralize management
5. **Set up deployment pipelines** per environment

---

## 📝 Database Models

- `JSONConfigCategory` - Configuration categories
- `JSONConfiguration` - Main configs with versioning
- `JSONConfigVersion` - Version history
- `JSONConfigDeployment` - Deployment tracking
- `JSONConfigTemplate` - Reusable templates
- `JSONConfigAuditLog` - Complete audit trail

*Complete JSON Configuration Management System - Evidence Mechanics v1.0*
