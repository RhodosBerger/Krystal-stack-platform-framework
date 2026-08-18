"""
Field-of-View Analyzer - Code Inspection Engine
Scans and analyzes component files to extract structural patterns
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib


class FieldOfViewAnalyzer:
    """
    Analyzes code files using 'field-of-view' mechanics
    Extracts structural elements, patterns, and metadata
    """
    
    def __init__(self):
        self.scan_patterns = {
            'props': r'(?:props|properties)[\s\S]*?{([^}]+)}',
            'state': r'(?:state|data)[\s\S]*?{([^}]+)}',
            'methods': r'(?:function|const)\s+(\w+)\s*\(',
            'data_bindings': r'(?:data-\w+|v-bind:[\w-]+|{{\s*\w+\s*}})',
            'event_handlers': r'(?:@click|onclick|addEventListener)\s*=\s*["\']?(\w+)',
            'css_classes': r'class\s*=\s*["\']([^"\']+)["\']',
            'api_calls': r'(?:fetch|axios|ajax)\s*\(["\']([^"\']+)["\']',
        }
    
    def scan_file(self, file_path: str) -> Dict[str, Any]:
        """
        Scan a single file and extract its structural elements
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        file_type = self._detect_file_type(file_path)
        
        return {
            'file_path': file_path,
            'file_type': file_type,
            'file_hash': self._generate_hash(content),
            'size': len(content),
            'lines': content.count('\n') + 1,
            'structure': self._extract_structure(content, file_type),
            'metadata': self._extract_metadata(content, file_type)
        }
    
    def scan_directory(self, directory: str, extensions: List[str] = None) -> List[Dict]:
        """
        Scan entire directory for component files
        """
        if extensions is None:
            extensions = ['.js', '.html', '.css', '.vue', '.jsx']
        
        results = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    try:
                        scan_result = self.scan_file(file_path)
                        results.append(scan_result)
                    except Exception as e:
                        print(f"Error scanning {file_path}: {e}")
        
        return results
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension"""
        ext = Path(file_path).suffix.lower()
        type_map = {
            '.js': 'javascript',
            '.jsx': 'react',
            '.vue': 'vue',
            '.html': 'html',
            '.css': 'css',
            '.py': 'python'
        }
        return type_map.get(ext, 'unknown')
    
    def _generate_hash(self, content: str) -> str:
        """Generate unique hash for file content"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _extract_structure(self, content: str, file_type: str) -> Dict:
        """
        Extract structural elements based on file type
        """
        structure = {
            'props': [],
            'state': [],
            'methods': [],
            'data_bindings': [],
            'event_handlers': [],
            'css_classes': [],
            'api_calls': []
        }
        
        # Extract using regex patterns
        for key, pattern in self.scan_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                structure[key] = matches if isinstance(matches, list) else [matches]
        
        return structure
    
    def _extract_metadata(self, content: str, file_type: str) -> Dict:
        """Extract metadata like imports, dependencies"""
        metadata = {
            'imports': [],
            'exports': [],
            'comments': []
        }
        
        # Extract imports
        import_patterns = [
            r'import\s+.*?from\s+["\']([^"\']+)["\']',
            r'require\(["\']([^"\']+)["\']\)',
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            metadata['imports'].extend(matches)
        
        # Extract exports
        export_matches = re.findall(r'export\s+(?:default\s+)?(\w+)', content)
        metadata['exports'] = export_matches
        
        # Extract comments (for documentation)
        comment_matches = re.findall(r'//\s*(.+)|/\*\s*([\s\S]*?)\s*\*/', content)
        metadata['comments'] = [c[0] or c[1] for c in comment_matches if c[0] or c[1]]
        
        return metadata
    
    def extract_component_signature(self, scan_result: Dict) -> Dict:
        """
        Generate a unique signature for a component
        This signature is used for pattern matching
        """
        structure = scan_result['structure']
        
        signature = {
            'file_hash': scan_result['file_hash'],
            'type': self._infer_component_type(structure),
            'complexity': self._calculate_complexity(structure),
            'data_flow': {
                'inputs': len(structure.get('props', [])),
                'state_vars': len(structure.get('state', [])),
                'api_calls': len(structure.get('api_calls', []))
            },
            'interactions': {
                'event_handlers': len(structure.get('event_handlers', [])),
                'bindings': len(structure.get('data_bindings', []))
            },
            'visual_elements': len(structure.get('css_classes', [])),
            'methods_count': len(structure.get('methods', []))
        }
        
        return signature
    
    def _infer_component_type(self, structure: Dict) -> str:
        """
        Infer component type from its structure
        """
        # Heuristics to determine component type
        
        if len(structure.get('event_handlers', [])) > 3:
            return 'interactive-widget'
        
        if len(structure.get('api_calls', [])) > 0:
            return 'data-card'
        
        if len(structure.get('css_classes', [])) > 5:
            return 'styled-component'
        
        if len(structure.get('methods', [])) > 5:
            return 'complex-component'
        
        return 'basic-component'
    
    def _calculate_complexity(self, structure: Dict) -> int:
        """
        Calculate component complexity score (1-10)
        """
        score = 0
        
        # More props/state = more complex
        score += min(len(structure.get('props', [])), 3)
        score += min(len(structure.get('state', [])), 3)
        
        # More methods = more complex
        score += min(len(structure.get('methods', [])) // 2, 2)
        
        # API calls add complexity
        score += min(len(structure.get('api_calls', [])), 2)
        
        return min(score, 10)
    
    def compare_components(self, sig1: Dict, sig2: Dict) -> float:
        """
        Compare two component signatures
        Returns similarity score (0.0 - 1.0)
        """
        similarity = 0.0
        weights = {
            'type': 0.3,
            'complexity': 0.1,
            'data_flow': 0.3,
            'interactions': 0.2,
            'visual_elements': 0.1
        }
        
        # Type similarity
        if sig1['type'] == sig2['type']:
            similarity += weights['type']
        
        # Complexity similarity (closer = more similar)
        complexity_diff = abs(sig1['complexity'] - sig2['complexity'])
        similarity += weights['complexity'] * (1 - complexity_diff / 10)
        
        # Data flow similarity
        df1 = sum(sig1['data_flow'].values())
        df2 = sum(sig2['data_flow'].values())
        if df1 > 0 and df2 > 0:
            df_similarity = min(df1, df2) / max(df1, df2)
            similarity += weights['data_flow'] * df_similarity
        
        # Interactions similarity
        int1 = sum(sig1['interactions'].values())
        int2 = sum(sig2['interactions'].values())
        if int1 > 0 and int2 > 0:
            int_similarity = min(int1, int2) / max(int1, int2)
            similarity += weights['interactions'] * int_similarity
        
        # Visual elements similarity
        ve1 = sig1['visual_elements']
        ve2 = sig2['visual_elements']
        if ve1 > 0 and ve2 > 0:
            ve_similarity = min(ve1, ve2) / max(ve1, ve2)
            similarity += weights['visual_elements'] * ve_similarity
        
        return similarity
    
    def visualize_scan_results(self, scan_result: Dict) -> str:
        """
        Create ASCII visualization of scan results
        """
        structure = scan_result['structure']
        
        viz = f"""
╔════════════════════════════════════════════════════════╗
║  FIELD-OF-VIEW SCAN RESULTS                            ║
╠════════════════════════════════════════════════════════╣
  File: {os.path.basename(scan_result['file_path'])}
  Type: {scan_result['file_type']}
  Hash: {scan_result['file_hash']}
  
  Structure Breakdown:
  ├─ Props/Inputs:     {len(structure.get('props', []))}
  ├─ State Variables:  {len(structure.get('state', []))}
  ├─ Methods:          {len(structure.get('methods', []))}
  ├─ Data Bindings:    {len(structure.get('data_bindings', []))}
  ├─ Event Handlers:   {len(structure.get('event_handlers', []))}
  ├─ CSS Classes:      {len(structure.get('css_classes', []))}
  └─ API Calls:        {len(structure.get('api_calls', []))}
  
  Signature:
  {json.dumps(self.extract_component_signature(scan_result), indent=2)}
╚════════════════════════════════════════════════════════╝
        """
        
        return viz


# ===== USAGE EXAMPLE =====

if __name__ == '__main__':
    analyzer = FieldOfViewAnalyzer()
    
    # Example: Scan dashboard-builder.js
    dashboard_path = '../dashboard/dashboard-builder.js'
    
    if os.path.exists(dashboard_path):
        result = analyzer.scan_file(dashboard_path)
        print(analyzer.visualize_scan_results(result))
        
        # Save results
        with open('scan_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print("\n✅ Scan completed! Results saved to scan_results.json")
    else:
        print(f"File not found: {dashboard_path}")
        print("Run from cms/component_builder directory")
