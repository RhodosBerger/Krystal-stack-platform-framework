"""
Visual Component Builder - Demo Runner
Demonstrates the code inspection and pattern extraction system
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from field_of_view_analyzer import FieldOfViewAnalyzer
from component_inspector import ComponentInspector
from pattern_matcher import PatternMatcher


def demo_field_of_view():
    """Demonstrate Field-of-View analyzer"""
    print("\n" + "="*60)
    print("DEMO 1: Field-of-View Code Analyzer")
    print("="*60 + "\n")
    
    analyzer = FieldOfViewAnalyzer()
    
    # Scan dashboard directory
    dashboard_dir = Path(__file__).parent.parent / 'dashboard'
    
    if not dashboard_dir.exists():
        print(f"❌ Dashboard directory not found: {dashboard_dir}")
        return None
    
    print(f"📂 Scanning directory: {dashboard_dir}\n")
    
    scan_results = analyzer.scan_directory(str(dashboard_dir))
    
    print(f"✅ Found {len(scan_results)} components\n")
    
    for result in scan_results[:3]:  # Show first 3
        print(analyzer.visualize_scan_results(result))
    
    return scan_results


def demo_component_inspector(scan_results=None):
    """Demonstrate Component Inspector"""
    print("\n" + "="*60)
    print("DEMO 2: Component Inspector & Pattern Extraction")
    print("="*60 + "\n")
    
    inspector = ComponentInspector()
    
    # Inspect dashboard directory
    dashboard_dir = Path(__file__).parent.parent / 'dashboard'
    
    if not dashboard_dir.exists():
        print(f"❌ Dashboard directory not found")
        return None
    
    print(f"🔍 Inspecting components in: {dashboard_dir}\n")
    
    inspections = inspector.inspect_directory(str(dashboard_dir))
    
    print(f"✅ Inspected {len(inspections)} components\n")
    
    # Show detailed reports
    for comp_id in list(inspections.keys())[:2]:  # Show first 2
        print(inspector.visualize_inspection(comp_id))
    
    # Generate pattern library
    output_path = Path(__file__).parent / 'pattern_library.json'
    library = inspector.generate_pattern_library(str(output_path))
    
    print(f"\n📚 Pattern Library Stats:")
    print(f"   Total Patterns: {len(library['patterns'])}")
    print(f"   Highly Reusable: {sum(1 for p in library['patterns'] if p['reusability']['score'] >= 80)}")
    print(f"   Moderately Reusable: {sum(1 for p in library['patterns'] if 50 <= p['reusability']['score'] < 80)}")
    
    return library


def demo_pattern_matcher():
    """Demonstrate Pattern Matcher"""
    print("\n" + "="*60)
    print("DEMO 3: Pattern Matching & Suggestions")
    print("="*60 + "\n")
    
    pattern_lib_path = Path(__file__).parent / 'pattern_library.json'
    
    if not pattern_lib_path.exists():
        print("❌ Pattern library not found. Run demo_component_inspector first.")
        return
    
    matcher = PatternMatcher(str(pattern_lib_path))
    
    # Example 1: Suggest pattern for a use case
    print("📋 Example 1: Find pattern for 'quality monitoring'\n")
    
    suggestion = matcher.suggest_pattern_for_use_case('quality')
    
    if suggestion:
        print(f"   ✅ {suggestion['suggestion']}")
        print(f"   Pattern ID: {suggestion['pattern_id']}")
        print(f"   Reusability: {suggestion['reusability']}%")
    else:
        # Try generic use case
        suggestion = matcher.suggest_pattern_for_use_case('generic')
        if suggestion:
            print(f"   ✅ {suggestion['suggestion']} (generic pattern)")
    
    # Example 2: Match requirements
    print("\n📋 Example 2: Match requirements to patterns\n")
    
    requirements = matcher.create_component_signature(
        component_type='data-card',
        has_api=True,
        interactive=True,
        complexity=5
    )
    
    matches = matcher.find_matching_patterns(requirements)
    
    if matches:
        print(f"   Found {len(matches)} matching patterns:\n")
        for i, match in enumerate(matches[:3], 1):
            print(f"   {i}. {match['pattern_id']}")
            print(f"      Match Score: {match['match_score']:.1%}")
            print(f"      Reusability: {match['reusability']}%")
            print()
    else:
        print("   No matches found")
    
    # Example 3: List all patterns
    print("📋 Example 3: Pattern Library Contents\n")
    
    all_patterns = matcher.list_patterns_by_type()
    
    if all_patterns:
        print(f"   {'Pattern ID':<35} | {'Type':<20} | Reusability")
        print(f"   {'-'*35}-+-{'-'*20}-+-{'-'*11}")
        
        for p in all_patterns:
            print(f"   {p['pattern_id']:<35} | {p['type']:<20} | {p['reusability']:>3}%")
    else:
        print("   No patterns in library")


def demo_new_component_generation():
    """Demonstrate creating a new component from a pattern"""
    print("\n" + "="*60)
    print("DEMO 4: Generating New Component from Pattern")
    print("="*60 + "\n")
    
    pattern_lib_path = Path(__file__).parent / 'pattern_library.json'
    
    if not pattern_lib_path.exists():
        print("❌ Pattern library not found")
        return
    
    matcher = PatternMatcher(str(pattern_lib_path))
    
    # Simulate user request
    print("👤 User Request: Create a 'Tool Status Card' component")
    print("   Requirements:")
    print("   • Show tool information (name, status, wear)")
    print("   • Real-time data from API")
    print("   • Interactive (responds to clicks)")
    print()
    
    # Find best pattern
    suggestion = matcher.suggest_pattern_for_use_case('tool')
    
    if not suggestion:
        # Fallback to generic
        suggestion = matcher.suggest_pattern_for_use_case('generic')
    
    if suggestion:
        print(f"🔧 System Suggestion:")
        print(f"   {suggestion['suggestion']}")
        print(f"   Pattern: {suggestion['pattern_id']}")
        print(f"   Reusability: {suggestion['reusability']}%")
        print()
        
        print(f"📝 Generated Component Configuration:")
        
        component_config = {
            "component_id": "tool_status_card",
            "type": "tool-card",
            "based_on": suggestion['pattern_id'],
            "data_binding": {
                "source": "django_api",
                "endpoint": "/api/tools/",
                "mapping": {
                    "tool_id": "title",
                    "status": "badge",
                    "remaining_life": "metric",
                    "cycles_used": "secondary_metric"
                }
            },
            "position": {"row": 0, "col": 0, "w": 4, "h": 3},
            "styling": {
                "theme": "glassmorphism",
                "color_scheme": "tool-status"
            }
        }
        
        print(json.dumps(component_config, indent=2))
        
        # Save configuration
        output_path = Path(__file__).parent / 'generated_component_example.json'
        with open(output_path, 'w') as f:
            json.dump(component_config, f, indent=2)
        
        print(f"\n✅ Component configuration saved to: {output_path}")


def run_all_demos():
    """Run all demonstrations"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*10 + "VISUAL COMPONENT BUILDER - DEMO" + " "*17 + "║")
    print("╚" + "="*58 + "╝")
    
    # Demo 1: Field-of-View
    scan_results = demo_field_of_view()
    
    # Demo 2: Component Inspector
    library = demo_component_inspector(scan_results)
    
    # Demo 3: Pattern Matcher
    demo_pattern_matcher()
    
    # Demo 4: New Component Generation
    demo_new_component_generation()
    
    print("\n" + "="*60)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated Files:")
    print("  • pattern_library.json - Extracted patterns")
    print("  • generated_component_example.json - Example new component")
    print("\nNext Steps:")
    print("  1. Review pattern_library.json")
    print("  2. Use patterns to create new components")
    print("  3. Build visual UI for this system")
    print()


if __name__ == '__main__':
    run_all_demos()
