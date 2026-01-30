"""
JSON Configuration Renderer - v1.0
Transform JSON config into live dashboards
"""

import json
import jsonpath_ng
from typing import Dict, Any, List
import requests
from jinja2 import Template

class JSONConfigRenderer:
    """
    Render dashboards from JSON configuration
    Supports:
    - Multiple data sources (REST, WebSocket, DB)
    - Data transformations (filter, aggregate, join)
    - Component rendering
    - Real-time updates
    """
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide config_path or config_dict")
        
        self.data_cache = {}
        self.component_instances = {}
    
    # ===== DATA FETCHING =====
    
    def fetch_data_from_source(self, source_id: str, endpoint: str, params: Dict = None, method: str = 'GET'):
        """Fetch data from configured data source"""
        source = next((s for s in self.config.get('data_sources', []) if s['id'] == source_id), None)
        
        if not source:
            raise ValueError(f"Data source '{source_id}' not found")
        
        url = f"{source.get('base_url', '')}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params)
            elif method == 'POST':
                response = requests.post(url, json=params)
            
            response.raise_for_status()
            data = response.json()
            
            # Apply response transform if specified
            if 'response_transform' in source.get('endpoints', {}).get(endpoint.split('/')[-1], {}):
                transform = source['endpoints'][endpoint.split('/')[-1]]['response_transform']
                data = self.apply_jsonpath(data, transform)
            
            return data
        
        except Exception as e:
            print(f"Error fetching from {url}: {e}")
            return None
    
    # ===== JSON PATH OPERATIONS =====
    
    def apply_jsonpath(self, data: Any, jspath: str):
        """Apply JSONPath expression to data"""
        if jspath.startswith('$.'):
            jsonpath_expr = jsonpath_ng.parse(jspath)
            matches = jsonpath_expr.find(data)
            return matches[0].value if matches else None
        return data
    
    def apply_mapping(self, data: Dict, mapping: Dict[str, str]):
        """Apply JSONPath mappings to extract fields"""
        result = {}
        for key, jspath in mapping.items():
            if jspath.startswith('$.'):
                result[key] = self.apply_jsonpath(data, jspath)
            else:
                # Simple key access
                result[key] = data.get(jspath, None)
        return result
    
    # ===== DATA TRANSFORMATIONS =====
    
    def apply_filter(self, data: List[Dict], conditions: List[Dict]) -> List[Dict]:
        """Filter data based on conditions"""
        filtered = data
        
        for condition in conditions:
            field = condition['field']
            operator = condition['operator']
            value = condition['value']
            
            if operator == '>':
                filtered = [item for item in filtered if item.get(field, 0) > value]
            elif operator == '<':
                filtered = [item for item in filtered if item.get(field, 0) < value]
            elif operator == '==':
                filtered = [item for item in filtered if item.get(field) == value]
            elif operator == 'in':
                filtered = [item for item in filtered if item.get(field) in value]
        
        return filtered
    
    def apply_aggregation(self, data: List[Dict], aggregations: List[Dict]) -> Dict:
        """Apply aggregations (avg, sum, max, min)"""
        results = {}
        
        for agg in aggregations:
            field = agg['field']
            function = agg['function']
            alias = agg.get('alias', f"{function}_{field}")
            
            values = [item.get(field, 0) for item in data if field in item]
            
            if function == 'avg':
                results[alias] = sum(values) / len(values) if values else 0
            elif function == 'sum':
                results[alias] = sum(values)
            elif function == 'max':
                results[alias] = max(values) if values else 0
            elif function == 'min':
                results[alias] = min(values) if values else 0
            elif function == 'count':
                results[alias] = len(values)
        
        return results
    
    # ===== COMPONENT RENDERING =====
    
    def render_component(self, component_config: Dict) -> str:
        """Render single component from config"""
        component_id = component_config['id']
        component_type = component_config['type']
        
        # Fetch and bind data
        data = {}
        if 'data_binding' in component_config:
            binding = component_config['data_binding']
            raw_data = self.fetch_data_from_source(
                binding['source'],
                binding['endpoint'],
                binding.get('params'),
                binding.get('method', 'GET')
            )
            
            if raw_data and 'mapping' in binding:
                data = self.apply_mapping(raw_data, binding['mapping'])
        
        # Render HTML
        return self.generate_component_html(
            component_type,
            component_id,
            data,
            component_config.get('styling', {}),
            component_config.get('position', {})
        )
    
    def generate_component_html(self, comp_type: str, comp_id: str, data: Dict, styling: Dict, position: Dict) -> str:
        """Generate HTML for component"""
        
        templates = {
            'gauge': '''
                <div class="component gauge-widget" id="{{id}}" style="{{style}}">
                    <div class="gauge-header">
                        <h3>{{label}}</h3>
                    </div>
                    <div class="gauge-body">
                        <div class="gauge-value">{{value}}%</div>
                        <div class="gauge-bar">
                            <div class="gauge-fill" style="width: {{value}}%; background: {{color}};"></div>
                        </div>
                    </div>
                </div>
            ''',
            
            'machine-card': '''
                <div class="component machine-card" id="{{id}}" style="{{style}}">
                    <div class="machine-header">
                        <h3>{{name}}</h3>
                        <span class="status-badge status-{{status}}">{{status}}</span>
                    </div>
                    <div class="machine-metrics">
                        <div class="metric">
                            <span class="metric-label">Load</span>
                            <span class="metric-value">{{load}}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">RPM</span>
                            <span class="metric-value">{{rpm}}</span>
                        </div>
                    </div>
                </div>
            ''',
            
            'line-chart': '''
                <div class="component chart-widget" id="{{id}}" style="{{style}}">
                    <div class="chart-header">
                        <h3>{{title}}</h3>
                    </div>
                    <div class="chart-body">
                        <canvas id="chart_{{id}}"></canvas>
                    </div>
                    <script>
                        renderLineChart('chart_{{id}}', {{data}});
                    </script>
                </div>
            ''',
            
            'oee-breakdown': '''
                <div class="component oee-widget" id="{{id}}" style="{{style}}">
                    <div class="oee-header">
                        <h3>OEE Dashboard</h3>
                    </div>
                    <div class="oee-overall">
                        <div class="oee-value">{{oee}}%</div>
                        <div class="oee-label">Overall Equipment Effectiveness</div>
                    </div>
                    <div class="oee-breakdown">
                        <div class="oee-metric">
                            <div class="metric-label">Availability</div>
                            <div class="metric-value">{{availability}}%</div>
                        </div>
                        <div class="oee-metric">
                            <div class="metric-label">Performance</div>
                            <div class="metric-value">{{performance}}%</div>
                        </div>
                        <div class="oee-metric">
                            <div class="metric-label">Quality</div>
                            <div class="metric-value">{{quality}}%</div>
                        </div>
                    </div>
                </div>
            '''
        }
        
        template_str = templates.get(comp_type, '<div>Unknown component: {{type}}</div>')
        template = Template(template_str)
        
        # Generate grid position style
        style = self.generate_grid_style(position)
        
        # Determine gauge color based on value
        color = '#10b981'  # default green
        if comp_type == 'gauge' and 'value' in data:
            val = data['value']
            if val < 40:
                color = '#ef4444'  # red
            elif val < 70:
                color = '#f59e0b'  # amber
        
        return template.render(
            id=comp_id,
            style=style,
            color=color,
            type=comp_type,
            **data
        )
    
    def generate_grid_style(self, position: Dict) -> str:
        """Generate CSS grid position style"""
        if not position:
            return ''
        
        return f'''
            grid-row: {position.get('row', 0) + 1} / span {position.get('h', 1)};
            grid-column: {position.get('col', 0) + 1} / span {position.get('w', 1)};
        '''
    
    # ===== DASHBOARD RENDERING =====
    
    def render_dashboard(self) -> str:
        """Render complete dashboard from config"""
        layout = self.config.get('layout', {})
        components = self.config.get('components', [])
        
        # Generate container
        container_style = f'''
            display: grid;
            grid-template-columns: repeat({layout.get('columns', 12)}, 1fr);
            grid-auto-rows: {layout.get('row_height', 80)}px;
            gap: {layout.get('gap', 16)}px;
            padding: 20px;
        '''
        
        html_parts = [f'<div class="dashboard-container" style="{container_style}">']
        
        # Render each component
        for component in components:
            html_parts.append(self.render_component(component))
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def save_rendered_dashboard(self, output_path: str):
        """Save rendered dashboard to HTML file"""
        dashboard_html = self.render_dashboard()
        
        full_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>{self.config.get('name', 'Dashboard')}</title>
    <link rel="stylesheet" href="dashboard-builder.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    {dashboard_html}
    <script>
        // Auto-refresh based on config
        const refreshRate = {self.config.get('refresh_policies', {}).get('global_interval', 1000)};
        setInterval(() => {{
            location.reload();
        }}, refreshRate);
    </script>
</body>
</html>
        '''
        
        with open(output_path, 'w') as f:
            f.write(full_html)
        
        print(f"Dashboard saved to {output_path}")


# ===== USAGE EXAMPLE =====

if __name__ == '__main__':
    # Example config
    config = {
        "dashboard_id": "test_dashboard",
        "name": "Test Dashboard",
        "layout": {
            "columns": 12,
            "row_height": 100,
            "gap": 20
        },
        "data_sources": [
            {
                "id": "flask_api",
                "base_url": "http://localhost:5000"
            }
        ],
        "components": [
            {
                "id": "gauge_1",
                "type": "gauge",
                "position": {"row": 0, "col": 0, "w": 4, "h": 2},
                "data_binding": {
                    "source": "flask_api",
                    "endpoint": "/api/telemetry/current",
                    "mapping": {
                        "value": "$.dopamine",
                        "label": "Dopamine"
                    }
                }
            }
        ]
    }
    
    renderer = JSONConfigRenderer(config_dict=config)
    renderer.save_rendered_dashboard('output_dashboard.html')
