# JSON-DRIVEN CONFIGURATION SYSTEM
## Complete Feature & Data Manipulation via JSON

---

## 🎯 CONCEPT: Configuration as Code

**Everything definable in JSON:**
- Dashboard components
- Data sources
- Database queries
- Data transformations
- UI bindings
- Business logic rules

---

## 1. COMPONENT DEFINITION JSON

### Example: Machine Status Card

```json
{
  "component_id": "machine_status_001",
  "component_type": "machine-card",
  "config": {
    "title": "CNC VMC 01",
    "refresh_rate": 1,
    "position": {
      "row": 1,
      "col": 1,
      "width": 6,
      "height": 4
    },
    "data_source": {
      "type": "rest_api",
      "endpoint": "/api/machines/1/telemetry",
      "method": "GET",
      "headers": {
        "Authorization": "Bearer ${API_TOKEN}"
      }
    },
    "data_mapping": {
      "machine_name": "$.machine.name",
      "status": "$.machine.is_active ? 'RUNNING' : 'STOPPED'",
      "load": "$.telemetry.load",
      "rpm": "$.telemetry.rpm",
      "temperature": "$.telemetry.spindle_temp"
    },
    "conditional_rendering": {
      "warning_threshold": {
        "condition": "data.load > 85",
        "apply_class": "warning-state"
      },
      "critical_threshold": {
        "condition": "data.load > 95",
        "apply_class": "critical-state"
      }
    },
    "actions": [
      {
        "trigger": "click",
        "action_type": "navigate",
        "target": "/machine-detail/{machine_id}"
      },
      {
        "trigger": "long_press",
        "action_type": "show_modal",
        "modal_id": "machine_controls"
      }
    ]
  }
}
```

---

## 2. DATA SOURCE DEFINITIONS

### A. REST API Source

```json
{
  "source_id": "telemetry_api",
  "source_type": "rest_api",
  "base_url": "http://localhost:5000",
  "endpoints": {
    "current_telemetry": {
      "path": "/api/telemetry/current",
      "method": "GET",
      "cache_ttl": 1,
      "response_transform": "$.data"
    },
    "historical_telemetry": {
      "path": "/api/telemetry/history",
      "method": "GET",
      "params": {
        "minutes": 60,
        "machine_id": "${MACHINE_ID}"
      },
      "response_transform": "$.data"
    }
  }
}
```

### B. WebSocket Source

```json
{
  "source_id": "live_telemetry",
  "source_type": "websocket",
  "url": "ws://localhost:5000",
  "events": {
    "telemetry_update": {
      "event_name": "telemetry_update",
      "data_path": "$.payload",
      "trigger_components": ["gauge_001", "chart_002"]
    }
  },
  "reconnect": {
    "enabled": true,
    "max_attempts": 5,
    "backoff": "exponential"
  }
}
```

### C. Database Direct Query

```json
{
  "source_id": "machine_db",
  "source_type": "database",
  "connection": {
    "type": "postgresql",
    "host": "localhost",
    "port": 5432,
    "database": "fanuc_rise",
    "schema": "public"
  },
  "queries": {
    "get_active_machines": {
      "sql": "SELECT * FROM erp_machine WHERE is_active = true",
      "cache_ttl": 60
    },
    "get_telemetry_stats": {
      "sql": "SELECT AVG(load) as avg_load, MAX(load) as max_load FROM erp_telemetry WHERE machine_id = :machine_id AND timestamp > NOW() - INTERVAL ':hours hours'",
      "params": {
        "machine_id": "$CONTEXT.machine_id",
        "hours": 24
      }
    }
  }
}
```

---

## 3. DATA TRANSFORMATION PIPELINE

```json
{
  "transformation_id": "oee_calculator",
  "input_sources": [
    {
      "source": "machine_db",
      "query": "get_machine_jobs"
    }
  ],
  "steps": [
    {
      "step_name": "calculate_availability",
      "operation": "javascript",
      "code": "const operatingTime = data.reduce((sum, job) => sum + job.duration, 0); return (operatingTime / (8 * 60)) * 100;"
    },
    {
      "step_name": "calculate_performance",
      "operation": "aggregate",
      "config": {
        "sum_field": "completed_quantity",
        "divide_by": "target_quantity",
        "multiply": 100
      }
    },
    {
      "step_name": "calculate_oee",
      "operation": "formula",
      "formula": "(availability * performance * quality) / 10000"
    }
  ],
  "output": {
    "format": "json",
    "schema": {
      "oee": "number",
      "availability": "number",
      "performance": "number",
      "quality": "number"
    }
  }
}
```

---

## 4. COMPLETE DASHBOARD JSON

```json
{
  "dashboard_id": "monitoring_001",
  "name": "Machine Monitoring Dashboard",
  "version": "1.0.0",
  "created_at": "2026-01-23T20:00:00Z",
  "layout": {
    "type": "grid",
    "columns": 12,
    "row_height": 80,
    "gap": 16
  },
  "data_sources": [
    {
      "id": "flask_api",
      "type": "rest_api",
      "base_url": "http://localhost:5000"
    },
    {
      "id": "django_api",
      "type": "rest_api",
      "base_url": "http://localhost:8000/api"
    },
    {
      "id": "websocket_stream",
      "type": "websocket",
      "url": "ws://localhost:5000"
    }
  ],
  "components": [
    {
      "id": "machine_card_1",
      "type": "machine-card",
      "position": {"row": 0, "col": 0, "w": 4, "h": 3},
      "data_binding": {
        "source": "flask_api",
        "endpoint": "/api/telemetry/current",
        "mapping": {
          "name": "$.machine.name",
          "status": "$.signal",
          "load": "$.load",
          "rpm": "$.rpm"
        }
      }
    },
    {
      "id": "dopamine_gauge",
      "type": "gauge",
      "position": {"row": 0, "col": 4, "w": 4, "h": 3},
      "data_binding": {
        "source": "flask_api",
        "endpoint": "/api/dopamine/evaluate",
        "method": "POST",
        "body": {
          "load": "${components.machine_card_1.load}",
          "vibration": "${components.machine_card_1.vibration}",
          "temperature": "${components.machine_card_1.temperature}"
        },
        "mapping": {
          "value": "$.dopamine",
          "label": "Dopamine"
        }
      },
      "styling": {
        "ranges": [
          {"min": 0, "max": 40, "color": "#ef4444"},
          {"min": 40, "max": 70, "color": "#f59e0b"},
          {"min": 70, "max": 100, "color": "#10b981"}
        ]
      }
    },
    {
      "id": "telemetry_chart",
      "type": "line-chart",
      "position": {"row": 3, "col": 0, "w": 8, "h": 4},
      "data_binding": {
        "source": "django_api",
        "endpoint": "/analytics/telemetry-history",
        "params": {
          "machine_id": 1,
          "hours": 1
        },
        "mapping": {
          "x_axis": "$.data[*].timestamp",
          "y_axis": "$.data[*].load",
          "series_name": "Load %"
        }
      },
      "chart_config": {
        "x_axis_label": "Time",
        "y_axis_label": "Load %",
        "min_y": 0,
        "max_y": 100,
        "show_grid": true
      }
    },
    {
      "id": "oee_widget",
      "type": "oee-breakdown",
      "position": {"row": 3, "col": 8, "w": 4, "h": 4},
      "data_binding": {
        "source": "django_api",
        "endpoint": "/machines/1/oee",
        "mapping": {
          "oee": "$.oee",
          "availability": "$.availability",
          "performance": "$.performance",
          "quality": "$.quality"
        }
      }
    }
  ],
  "refresh_policies": {
    "global_interval": 1000,
    "component_overrides": {
      "telemetry_chart": 5000,
      "oee_widget": 30000
    }
  },
  "event_handlers": {
    "on_load": [
      {
        "action": "fetch_initial_data",
        "targets": ["all"]
      }
    ],
    "on_error": [
      {
        "action": "show_notification",
        "config": {
          "type": "error",
          "message": "Failed to load data"
        }
      }
    ]
  }
}
```

---

## 5. DATA MANIPULATION RULES

### A. Filtering

```json
{
  "manipulation_id": "filter_high_load",
  "operation": "filter",
  "source_data": "telemetry_stream",
  "conditions": [
    {
      "field": "load",
      "operator": ">",
      "value": 80
    },
    {
      "field": "signal",
      "operator": "in",
      "value": ["AMBER", "RED"]
    }
  ],
  "output_target": "high_load_alerts"
}
```

### B. Aggregation

```json
{
  "manipulation_id": "hourly_avg_load",
  "operation": "aggregate",
  "source_data": "telemetry_history",
  "group_by": {
    "field": "timestamp",
    "interval": "1 hour"
  },
  "aggregations": [
    {
      "field": "load",
      "function": "avg",
      "alias": "avg_load"
    },
    {
      "field": "load",
      "function": "max",
      "alias": "peak_load"
    }
  ]
}
```

### C. Join Multiple Sources

```json
{
  "manipulation_id": "job_with_economics",
  "operation": "join",
  "left_source": {
    "source": "django_api",
    "endpoint": "/jobs",
    "key": "job_id"
  },
  "right_source": {
    "source": "machine_db",
    "query": "get_economic_records",
    "key": "job_id"
  },
  "join_type": "left",
  "output_fields": [
    "left.job_id",
    "left.part_number",
    "left.status",
    "right.total_cost",
    "right.cost_per_part"
  ]
}
```

---

## 6. DYNAMIC FORM GENERATION FROM JSON

```json
{
  "form_id": "machine_registration",
  "title": "Register New Machine",
  "sections": [
    {
      "section_id": "basic_info",
      "title": "Basic Information",
      "fields": [
        {
          "field_id": "machine_name",
          "type": "text",
          "label": "Machine Name",
          "placeholder": "CNC VMC 01",
          "required": true,
          "validation": {
            "min_length": 3,
            "max_length": 50,
            "pattern": "^[A-Z0-9_]+$"
          }
        },
        {
          "field_id": "controller_type",
          "type": "select",
          "label": "Controller Type",
          "options": [
            {"value": "FANUC", "label": "Fanuc"},
            {"value": "SIEMENS", "label": "Siemens"},
            {"value": "HAAS", "label": "Haas"}
          ],
          "required": true
        },
        {
          "field_id": "ip_address",
          "type": "text",
          "label": "IP Address",
          "validation": {
            "pattern": "^(?:[0-9]{1,3}\\.){3}[0-9]{1,3}$"
          },
          "async_validation": {
            "endpoint": "/api/validate/ip",
            "debounce": 500
          }
        }
      ]
    },
    {
      "section_id": "specs",
      "title": "Specifications",
      "depends_on": {
        "field": "controller_type",
        "value": "FANUC"
      },
      "fields": [
        {
          "field_id": "max_rpm",
          "type": "number",
          "label": "Max RPM",
          "default": 12000,
          "min": 1000,
          "max": 30000
        }
      ]
    }
  ],
  "submit": {
    "endpoint": "/api/machines/register",
    "method": "POST",
    "success_action": {
      "type": "navigate",
      "url": "/machines"
    },
    "error_action": {
      "type": "show_errors",
      "display": "inline"
    }
  }
}
```

---

## 7. BUSINESS LOGIC RULES ENGINE

```json
{
  "rule_id": "safety_alert",
  "name": "Safety Alert Generator",
  "trigger": {
    "source": "websocket_stream",
    "event": "telemetry_update"
  },
  "conditions": [
    {
      "field": "load",
      "operator": ">",
      "value": 95,
      "duration": "5 minutes"
    },
    {
      "or": [
        {"field": "vibration_z", "operator": ">", "value": 0.05},
        {"field": "spindle_temp", "operator": ">", "value": 80}
      ]
    }
  ],
  "actions": [
    {
      "action_type": "create_alert",
      "config": {
        "severity": "CRITICAL",
        "message": "Machine ${machine_id} in critical state!",
        "channels": ["database", "websocket", "email"]
      }
    },
    {
      "action_type": "api_call",
      "config": {
        "endpoint": "/api/machines/${machine_id}/emergency_stop",
        "method": "POST"
      }
    }
  ]
}
```

---

## 8. IMPLEMENTATION: JSON Renderer

```python
# erp/json_renderer.py

import json
import jsonpath_ng
from typing import Dict, Any
import requests

class JSONConfigRenderer:
    """Render components from JSON configuration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_cache = {}
    
    def fetch_data(self, source_id: str, endpoint: str, params: Dict = None):
        """Fetch data from configured source"""
        source = next((s for s in self.config['data_sources'] if s['id'] == source_id), None)
        if not source:
            raise ValueError(f"Data source {source_id} not found")
        
        url = f"{source['base_url']}{endpoint}"
        response = requests.get(url, params=params)
        return response.json()
    
    def apply_jspath_mapping(self, data: Dict, mapping: Dict[str, str]):
        """Apply JSONPath mappings to data"""
        result = {}
        for key, jspath in mapping.items():
            jsonpath_expr = jsonpath_ng.parse(jspath)
            matches = jsonpath_expr.find(data)
            result[key] = matches[0].value if matches else None
        return result
    
    def render_component(self, component_config: Dict):
        """Render single component"""
        # Fetch data
        binding = component_config.get('data_binding', {})
        if binding:
            raw_data = self.fetch_data(
                binding['source'],
                binding['endpoint'],
                binding.get('params')
            )
            
            # Apply mapping
            mapped_data = self.apply_jspath_mapping(
                raw_data,
                binding['mapping']
            )
            
            # Render with data
            return self.generate_html(
                component_config['type'],
                component_config['id'],
                mapped_data,
                component_config.get('styling', {})
            )
    
    def generate_html(self, component_type: str, component_id: str, data: Dict, styling: Dict):
        """Generate HTML for component"""
        templates = {
            'gauge': '''
                <div class="gauge-widget" id="{id}">
                    <h3>{label}</h3>
                    <div class="gauge-value">{value}%</div>
                </div>
            ''',
            'machine-card': '''
                <div class="machine-card" id="{id}">
                    <h3>{name}</h3>
                    <span class="status">{status}</span>
                    <div class="metrics">
                        <div>Load: {load}%</div>
                        <div>RPM: {rpm}</div>
                    </div>
                </div>
            '''
        }
        
        template = templates.get(component_type, '<div>Unknown component</div>')
        return template.format(id=component_id, **data)
    
    def render_dashboard(self):
        """Render entire dashboard"""
        html_components = []
        for component in self.config['components']:
            html_components.append(self.render_component(component))
        
        return '\n'.join(html_components)
```

---

## COMPLETE EXAMPLE: Dashboard from JSON

```json
{
  "dashboard": "monitoring",
  "components": [
    {"type": "gauge", "data": "telemetry.dopamine"},
    {"type": "chart", "data": "telemetry.history"}
  ],
  "data_sources": {
    "telemetry": "http://localhost:5000/api/telemetry"
  }
}
```

**Result**: Fully functional dashboard generated from 10 lines of JSON! 🎯

*JSON-Driven Platform Architecture - Complete Configuration System*
