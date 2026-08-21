"""
Pattern Matcher - Component Pattern Recognition
Matches components to patterns and suggests similar components
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class Pattern:
    """Pattern data structure"""
    pattern_id: str
    pattern_type: str
    signature: Dict
    template: Dict
    reusability_score: int
    applicability: List[str]
    source_component: str


class PatternMatcher:
    """
    Matches components against known patterns
    Suggests which pattern to use for new components
    """
    
    def __init__(self, pattern_library_path: str = None):
        self.patterns: Dict[str, Pattern] = {}
        
        if pattern_library_path:
            self.load_pattern_library(pattern_library_path)
    
    def load_pattern_library(self, file_path: str):
        """Load patterns from JSON library"""
        with open(file_path, 'r') as f:
            library = json.load(f)
        
        for pattern_data in library.get('patterns', []):
            pattern = Pattern(
                pattern_id=pattern_data['pattern_id'],
                pattern_type=pattern_data['signature']['type'],
                signature=pattern_data['signature'],
                template=pattern_data['pattern'],
                reusability_score=pattern_data['reusability']['score'],
                applicability=pattern_data['applicability'],
                source_component=pattern_data['source_component']
            )
            
            self.patterns[pattern.pattern_id] = pattern
        
        print(f"✅ Loaded {len(self.patterns)} patterns from library")
    
    def save_pattern_library(self, file_path: str):
        """Save current patterns to file"""
        library = {
            'version': '1.0.0',
            'patterns': []
        }
        
        for pattern in self.patterns.values():
            library['patterns'].append({
                'pattern_id': pattern.pattern_id,
                'source_component': pattern.source_component,
                'signature': pattern.signature,
                'pattern': pattern.template,
                'reusability': {'score': pattern.reusability_score},
                'applicability': pattern.applicability
            })
        
        with open(file_path, 'w') as f:
            json.dump(library, f, indent=2)
    
    def add_pattern(self, pattern: Pattern):
        """Add a new pattern to the library"""
        self.patterns[pattern.pattern_id] = pattern
    
    def find_matching_patterns(
        self,
        component_signature: Dict,
        use_case: str = None,
        min_reusability: int = 50
    ) -> List[Dict]:
        """
        Find patterns that match the given requirements
        """
        matches = []
        
        for pattern in self.patterns.values():
            # Filter by reusability
            if pattern.reusability_score < min_reusability:
                continue
            
            # Filter by use case if specified
            if use_case and use_case not in pattern.applicability:
                continue
            
            # Calculate match score
            match_score = self._calculate_match_score(component_signature, pattern.signature)
            
            if match_score > 0.3:  # Threshold for considering a match
                matches.append({
                    'pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type,
                    'match_score': match_score,
                    'reusability': pattern.reusability_score,
                    'applicability': pattern.applicability,
                    'source': pattern.source_component
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return matches
    
    def suggest_pattern_for_use_case(self, use_case: str, data_source: str = None) -> Optional[Dict]:
        """
        Suggest best pattern for a specific use case
        """
        # Filter patterns by applicability
        candidates = [
            p for p in self.patterns.values()
            if use_case in p.applicability
        ]
        
        if not candidates:
            return None
        
        # Sort by reusability
        candidates.sort(key=lambda x: x.reusability_score, reverse=True)
        
        best_pattern = candidates[0]
        
        return {
            'pattern_id': best_pattern.pattern_id,
            'pattern_type': best_pattern.pattern_type,
            'reusability': best_pattern.reusability_score,
            'template': best_pattern.template,
            'applicability': best_pattern.applicability,
            'suggestion': f"Use {best_pattern.source_component} as base for {use_case}"
        }
    
    def _calculate_match_score(self, sig1: Dict, sig2: Dict) -> float:
        """
        Calculate how well two signatures match
        """
        score = 0.0
        
        # Type match
        if sig1.get('type') == sig2.get('type'):
            score += 0.4
        
        # Complexity match (closer is better)
        complexity_diff = abs(sig1.get('complexity', 5) - sig2.get('complexity', 5))
        score += 0.2 * (1 - complexity_diff / 10)
        
        # Data flow similarity
        df1_total = sum(sig1.get('data_flow', {}).values())
        df2_total = sum(sig2.get('data_flow', {}).values())
        
        if df1_total > 0 and df2_total > 0:
            df_sim = min(df1_total, df2_total) / max(df1_total, df2_total)
            score += 0.2 * df_sim
        
        # Interactions similarity
        int1_total = sum(sig1.get('interactions', {}).values())
        int2_total = sum(sig2.get('interactions', {}).values())
        
        if int1_total > 0 and int2_total > 0:
            int_sim = min(int1_total, int2_total) / max(int1_total, int2_total)
            score += 0.2 * int_sim
        
        return min(score, 1.0)
    
    def create_component_signature(
        self,
        component_type: str,
        has_api: bool = False,
        interactive: bool = False,
        complexity: int = 3
    ) -> Dict:
        """
        Create a signature for requirements matching
        """
        return {
            'type': component_type,
            'complexity': complexity,
            'data_flow': {
                'inputs': 1,
                'state_vars': 1 if interactive else 0,
                'api_calls': 1 if has_api else 0
            },
            'interactions': {
                'event_handlers': 3 if interactive else 0,
                'bindings': 2 if has_api else 0
            }
        }
    
    def list_patterns_by_type(self, pattern_type: str = None) -> List[Dict]:
        """List all patterns, optionally filtered by type"""
        patterns = list(self.patterns.values())
        
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        return [
            {
                'pattern_id': p.pattern_id,
                'type': p.pattern_type,
                'reusability': p.reusability_score,
                'applicability': p.applicability,
                'source': p.source_component
            }
            for p in patterns
        ]
    
    def visualize_pattern(self, pattern_id: str) -> str:
        """Create visual representation of pattern"""
        if pattern_id not in self.patterns:
            return f"Pattern {pattern_id} not found"
        
        pattern = self.patterns[pattern_id]
        
        viz = f"""
╔════════════════════════════════════════════════════════╗
║  PATTERN: {pattern.pattern_id:<42} ║
╠════════════════════════════════════════════════════════╣

  📦 Type: {pattern.pattern_type}
  📁 Source: {pattern.source_component}
  
  ♻️  Reusability: {'█' * (pattern.reusability_score // 10)}{'░' * (10 - pattern.reusability_score // 10)} {pattern.reusability_score}%
  
  🎯 Applicable for:
     {chr(10).join(f'     • {app}' for app in pattern.applicability)}
  
  📊 Signature:
     • Complexity: {pattern.signature.get('complexity', 'N/A')}/10
     • Type: {pattern.signature.get('type', 'unknown')}
  
  🔧 Template Structure:
     {json.dumps(pattern.template.get('template_structure', {}), indent=5)}

╚════════════════════════════════════════════════════════╝
        """
        
        return viz


# ===== USAGE EXAMPLE =====

if __name__ == '__main__':
    # Load existing pattern library
    matcher = PatternMatcher('pattern_library.json')
    
    # Example 1: Find pattern for quality inspection card
    print("🔍 Finding pattern for quality inspection...\n")
    
    suggestion = matcher.suggest_pattern_for_use_case('quality')
    
    if suggestion:
        print(f"✅ Suggestion: {suggestion['suggestion']}")
        print(f"   Pattern: {suggestion['pattern_id']}")
        print(f"   Reusability: {suggestion['reusability']}%\n")
    
    # Example 2: Match requirements to patterns
    print("🔍 Matching requirements to patterns...\n")
    
    requirements = matcher.create_component_signature(
        component_type='data-card',
        has_api=True,
        interactive=True,
        complexity=4
    )
    
    matches = matcher.find_matching_patterns(requirements)
    
    print(f"Found {len(matches)} matching patterns:\n")
    for match in matches[:3]:  # Top 3
        print(f"  • {match['pattern_id']}")
        print(f"    Match Score: {match['match_score']:.2%}")
        print(f"    Reusability: {match['reusability']}%")
        print(f"    Applicability: {', '.join(match['applicability'])}\n")
    
    # Example 3: List all patterns
    print("\n📚 Pattern Library:\n")
    all_patterns = matcher.list_patterns_by_type()
    
    for p in all_patterns:
        print(f"  {p['pattern_id']:<30} | {p['type']:<20} | {p['reusability']}%")
