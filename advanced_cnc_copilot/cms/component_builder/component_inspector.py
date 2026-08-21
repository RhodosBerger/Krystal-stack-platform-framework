"""
Component Inspector - High-Level Component Analysis
Uses Field-of-View analyzer to extract reusable patterns
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from field_of_view_analyzer import FieldOfViewAnalyzer


class ComponentInspector:
    """
    High-level inspector that analyzes components
    and extracts reusable patterns
    """
    
    def __init__(self):
        self.fov_analyzer = FieldOfViewAnalyzer()
        self.inspected_components = {}
    
    def inspect_component(self, component_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive component inspection
        """
        # Scan file structure
        scan_result = self.fov_analyzer.scan_file(component_path)
        
        # Extract component signature
        signature = self.fov_analyzer.extract_component_signature(scan_result)
        
        # Analyze reusability
        reusability = self._analyze_reusability(scan_result)
        
        # Extract pattern template
        pattern = self._extract_pattern(scan_result)
        
        inspection_result = {
            'component_id': Path(component_path).stem,
            'file_path': component_path,
            'scan_data': scan_result,
            'signature': signature,
            'reusability': reusability,
            'pattern': pattern,
            'applicability': self._infer_applicability(scan_result)
        }
        
        # Cache result
        self.inspected_components[inspection_result['component_id']] = inspection_result
        
        return inspection_result
    
    def inspect_directory(self, directory: str) -> Dict[str, Dict]:
        """
        Inspect all components in a directory
        """
        scan_results = self.fov_analyzer.scan_directory(directory)
        
        inspections = {}
        for scan in scan_results:
            component_id = Path(scan['file_path']).stem
            
            signature = self.fov_analyzer.extract_component_signature(scan)
            reusability = self._analyze_reusability(scan)
            pattern = self._extract_pattern(scan)
            
            inspections[component_id] = {
                'component_id': component_id,
                'file_path': scan['file_path'],
                'signature': signature,
                'reusability': reusability,
                'pattern': pattern,
                'applicability': self._infer_applicability(scan)
            }
        
        self.inspected_components.update(inspections)
        return inspections
    
    def _analyze_reusability(self, scan_result: Dict) -> Dict:
        """
        Analyze how reusable this component is
        """
        structure = scan_result['structure']
        
        # Calculate reusability score
        score = 0
        factors = []
        
        # Few hard-coded values = more reusable
        if len(structure.get('api_calls', [])) <= 1:
            score += 30
            factors.append('Generic API structure')
        
        # Parameterized through props = more reusable
        if len(structure.get('props', [])) > 0:
            score += 40
            factors.append('Props-driven configuration')
        
        # Clean separation of concerns
        if len(structure.get('css_classes', [])) > 0:
            score += 20
            factors.append('Styled with classes')
        
        # Not overly complex
        signature = self.fov_analyzer.extract_component_signature(scan_result)
        if signature['complexity'] <= 5:
            score += 10
            factors.append('Moderate complexity')
        
        return {
            'score': min(score, 100),
            'level': self._score_to_level(score),
            'factors': factors
        }
    
    def _score_to_level(self, score: int) -> str:
        """Convert score to reusability level"""
        if score >= 80:
            return 'HIGHLY_REUSABLE'
        elif score >= 50:
            return 'MODERATELY_REUSABLE'
        else:
            return 'LOW_REUSABILITY'
    
    def _extract_pattern(self, scan_result: Dict) -> Dict:
        """
        Extract reusable pattern from component
        """
        structure = scan_result['structure']
        signature = self.fov_analyzer.extract_component_signature(scan_result)
        
        pattern = {
            'pattern_type': signature['type'],
            'template_structure': {
                'has_header': 'header' in str(structure.get('css_classes', [])).lower(),
                'has_body': 'body' in str(structure.get('css_classes', [])).lower(),
                'has_footer': 'footer' in str(structure.get('css_classes', [])).lower(),
            },
            'data_requirements': {
                'props': structure.get('props', []),
                'api_endpoint': structure.get('api_calls', [None])[0] if structure.get('api_calls') else None
            },
            'interactivity': {
                'events': structure.get('event_handlers', []),
                'bindings': structure.get('data_bindings', [])
            },
            'styling': {
                'classes': structure.get('css_classes', [])
            }
        }
        
        return pattern
    
    def _infer_applicability(self, scan_result: Dict) -> List[str]:
        """
        Infer what types of data this component can work with
        """
        structure = scan_result['structure']
        signature = self.fov_analyzer.extract_component_signature(scan_result)
        
        applicability = []
        
        # Check component type
        comp_type = signature['type']
        
        if comp_type == 'data-card':
            # Can be applied to any entity with status
            applicability.extend(['machine', 'tool', 'job', 'sensor', 'quality'])
        
        if comp_type == 'interactive-widget':
            # Interactive controls
            applicability.extend(['controls', 'forms', 'dialogs'])
        
        if len(structure.get('api_calls', [])) > 0:
            # Has data fetching
            applicability.append('real-time-data')
        
        # Check for specific patterns in code
        content_str = str(structure).lower()
        
        if 'gauge' in content_str or 'meter' in content_str:
            applicability.append('metrics')
        
        if 'chart' in content_str or 'graph' in content_str:
            applicability.append('analytics')
        
        if 'status' in content_str or 'state' in content_str:
            applicability.append('status-monitoring')
        
        return list(set(applicability)) if applicability else ['generic']
    
    def find_similar_components(self, component_id: str, threshold: float = 0.6) -> List[Dict]:
        """
        Find components similar to the given one
        """
        if component_id not in self.inspected_components:
            raise ValueError(f"Component {component_id} not inspected")
        
        target = self.inspected_components[component_id]
        target_sig = target['signature']
        
        similar = []
        
        for comp_id, comp in self.inspected_components.items():
            if comp_id == component_id:
                continue
            
            similarity = self.fov_analyzer.compare_components(target_sig, comp['signature'])
            
            if similarity >= threshold:
                similar.append({
                    'component_id': comp_id,
                    'similarity': similarity,
                    'file_path': comp['file_path'],
                    'type': comp['signature']['type']
                })
        
        # Sort by similarity (highest first)
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar
    
    def generate_pattern_library(self, output_path: str = 'pattern_library.json'):
        """
        Generate a pattern library from all inspected components
        """
        library = {
            'version': '1.0.0',
            'patterns': []
        }
        
        for comp_id, inspection in self.inspected_components.items():
            # Only include moderately or highly reusable components
            if inspection['reusability']['score'] >= 50:
                pattern_entry = {
                    'pattern_id': f"{comp_id}_pattern",
                    'source_component': comp_id,
                    'source_file': inspection['file_path'],
                    'signature': inspection['signature'],
                    'pattern': inspection['pattern'],
                    'reusability': inspection['reusability'],
                    'applicability': inspection['applicability']
                }
                
                library['patterns'].append(pattern_entry)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(library, f, indent=2)
        
        print(f"✅ Pattern library generated with {len(library['patterns'])} patterns")
        print(f"   Saved to: {output_path}")
        
        return library
    
    def visualize_inspection(self, component_id: str) -> str:
        """
        Create visual report of inspection results
        """
        if component_id not in self.inspected_components:
            return f"Component {component_id} not found"
        
        inspection = self.inspected_components[component_id]
        reusability = inspection['reusability']
        signature = inspection['signature']
        
        viz = f"""
╔════════════════════════════════════════════════════════════════╗
║  COMPONENT INSPECTION REPORT: {component_id:<30} ║
╠════════════════════════════════════════════════════════════════╣

  📄 File: {Path(inspection['file_path']).name}
  
  🔍 Signature:
     Type: {signature['type']}
     Complexity: {'█' * signature['complexity']}{'░' * (10 - signature['complexity'])} ({signature['complexity']}/10)
  
  ♻️  Reusability:
     Score: {reusability['score']}/100 ({reusability['level']})
     Factors:
     {chr(10).join(f'     • {f}' for f in reusability['factors'])}
  
  🎯 Applicability:
     Can be used for: {', '.join(inspection['applicability'])}
  
  📊 Pattern Analysis:
     Template Structure:
       • Header: {'✓' if inspection['pattern']['template_structure']['has_header'] else '✗'}
       • Body:   {'✓' if inspection['pattern']['template_structure']['has_body'] else '✗'}
       • Footer: {'✓' if inspection['pattern']['template_structure']['has_footer'] else '✗'}
     
     Data Requirements:
       • Props: {len(inspection['pattern']['data_requirements']['props'])}
       • API: {inspection['pattern']['data_requirements']['api_endpoint'] or 'None'}
     
     Interactivity:
       • Events: {len(inspection['pattern']['interactivity']['events'])}
       • Bindings: {len(inspection['pattern']['interactivity']['bindings'])}

╚════════════════════════════════════════════════════════════════╝
        """
        
        return viz


# ===== USAGE EXAMPLE =====

if __name__ == '__main__':
    inspector = ComponentInspector()
    
    # Example: Inspect dashboard builder components
    dashboard_dir = '../dashboard'
    
    print("🔍 Inspecting components...")
    inspections = inspector.inspect_directory(dashboard_dir)
    
    print(f"\n✅ Inspected {len(inspections)} components\n")
    
    # Show reports for each component
    for comp_id in inspections.keys():
        print(inspector.visualize_inspection(comp_id))
    
    # Generate pattern library
    inspector.generate_pattern_library('pattern_library.json')
    
    print("\n✅ Component inspection complete!")
