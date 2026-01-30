#!/usr/bin/env python3
"""
Test Suite for GAMESA System Identification and Conceptual Understanding Framework
"""

import unittest
from system_identifier import (
    SystemIdentifier, 
    SystemComponentType, 
    SystemLayer, 
    ConceptualElement,
    ComponentMetadata,
    SystemRelationship
)


class TestSystemIdentifier(unittest.TestCase):
    """Test the SystemIdentifier class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.identifier = SystemIdentifier()
    
    def test_component_registration(self):
        """Test that all components are properly registered."""
        expected_components = [
            'windows_extension',
            'essential_encoder', 
            'openvino_integration',
            'hexadecimal_system',
            'ascii_image_renderer',
            'guardian_framework',
            'grid_memory_controller',
            'safety_monitor'
        ]
        
        for comp_name in expected_components:
            self.assertIn(comp_name, self.identifier.components)
            comp = self.identifier.components[comp_name]
            self.assertIsInstance(comp, ComponentMetadata)
            # Just check that the name is not empty
            self.assertGreater(len(comp.name), 0)
    
    def test_component_metadata(self):
        """Test component metadata properties."""
        comp = self.identifier.components['guardian_framework']
        
        self.assertEqual(comp.type, SystemComponentType.CORE_ENGINE)
        self.assertEqual(comp.layer, SystemLayer.CORE_ENGINE)
        self.assertEqual(comp.version, "1.0.0")
        self.assertEqual(comp.status, "active")
        self.assertIn('windows_extension', comp.dependencies)
        self.assertIn('cpu_governor', comp.interfaces)
        self.assertIn('C/Rust layer integration', comp.description)
    
    def test_relationship_registration(self):
        """Test that relationships are properly registered."""
        self.assertGreater(len(self.identifier.relationships), 0)
        
        # Find a specific relationship
        windows_to_encoder_rel = None
        for rel in self.identifier.relationships:
            if (rel.source_component == 'windows_extension' and 
                rel.target_component == 'essential_encoder' and
                rel.relationship_type == 'depends_on'):
                windows_to_encoder_rel = rel
                break
        
        self.assertIsNotNone(windows_to_encoder_rel)
        self.assertEqual(windows_to_encoder_rel.strength, 0.8)
        self.assertIn('uses essential encoder', windows_to_encoder_rel.description)
    
    def test_concept_registration(self):
        """Test that conceptual elements are properly registered."""
        expected_concepts = [
            'economic_resource_trading',
            'three_d_grid_memory',
            'neural_hardware_fabric',
            'cross_forex_market',
            'metacognitive_framework',
            'safety_validation',
            'trigonometric_optimization',
            'fibonacci_escalation',
            'hexadecimal_trading',
            'signal_processing'
        ]
        
        for concept_name in expected_concepts:
            self.assertIn(concept_name, self.identifier.concepts)
            concept = self.identifier.concepts[concept_name]
            self.assertIsInstance(concept, ConceptualElement)
            self.assertGreater(len(concept.name), 0)
            self.assertGreater(len(concept.definition), 0)
            self.assertGreater(len(concept.implementation_locations), 0)
    
    def test_component_identification(self):
        """Test component identification functionality."""
        comp = self.identifier.identify_component('essential_encoder')
        self.assertIsNotNone(comp)
        self.assertEqual(comp.name, "Essential Encoder System")
        self.assertEqual(comp.type, SystemComponentType.DATA_PROCESSOR)
        
        # Test non-existent component
        none_comp = self.identifier.identify_component('non_existent')
        self.assertIsNone(none_comp)
    
    def test_component_dependencies(self):
        """Test component dependency lookup."""
        deps = self.identifier.get_component_dependencies('guardian_framework')
        self.assertIn('windows_extension', deps)
        
        # Check that dependencies are actually registered components
        for dep in deps:
            self.assertIn(dep, self.identifier.components)
    
    def test_component_interfaces(self):
        """Test component interface lookup."""
        interfaces = self.identifier.get_component_interfaces('grid_memory_controller')
        expected_interfaces = ['grid_controller', 'functional_runtime', 'coherence_protocol']
        
        for expected_interface in expected_interfaces:
            self.assertIn(expected_interface, interfaces)
    
    def test_related_components(self):
        """Test related component lookup."""
        related = self.identifier.get_related_components('essential_encoder')
        
        # Essential encoder should be related to several components
        self.assertGreater(len(related), 0)
        self.assertIn('windows_extension', related)  # Windows extension depends on it
        self.assertIn('openvino_integration', related)  # OpenVINO integration uses it
    
    def test_concept_lookup(self):
        """Test concept lookup functionality."""
        concept = self.identifier.find_concept('economic_resource_trading')
        self.assertIsNotNone(concept)
        self.assertEqual(concept.category, 'concept')
        self.assertIn('tradable assets', concept.definition)
        self.assertIn('windows_extension', concept.implementation_locations)
    
    def test_concept_implementation_locations(self):
        """Test concept implementation location lookup."""
        locations = self.identifier.get_concept_implementation_locations('metacognitive_framework')
        self.assertIn('guardian_framework', locations)
        self.assertIn('hexadecimal_system', locations)
    
    def test_component_search_by_type(self):
        """Test component search by type."""
        resource_managers = self.identifier.search_components_by_type(SystemComponentType.RESOURCE_MANAGER)
        self.assertIn('windows_extension', resource_managers)
        
        core_engines = self.identifier.search_components_by_type(SystemComponentType.CORE_ENGINE)
        self.assertIn('guardian_framework', core_engines)
    
    def test_component_search_by_layer(self):
        """Test component search by layer."""
        core_engine_components = self.identifier.search_components_by_layer(SystemLayer.CORE_ENGINE)
        expected = ['essential_encoder', 'guardian_framework', 'grid_memory_controller']
        
        for expected_comp in expected:
            self.assertIn(expected_comp, core_engine_components)
    
    def test_system_architecture(self):
        """Test system architecture view."""
        architecture = self.identifier.get_system_architecture()
        
        # Check that all layers exist
        for layer in SystemLayer:
            self.assertIn(layer.value, architecture)
            self.assertIsInstance(architecture[layer.value], list)
    
    def test_component_relationships(self):
        """Test component relationship lookup."""
        rels = self.identifier.get_component_relationships('guardian_framework')
        
        # Guardian framework should have several relationships
        self.assertGreater(len(rels), 0)
        
        # Check for specific relationship types
        has_depends_on = any(rel.relationship_type == 'depends_on' for rel in rels)
        self.assertTrue(has_depends_on)
    
    def test_system_complexity_analysis(self):
        """Test system complexity analysis."""
        stats = self.identifier.analyze_system_complexity()
        
        self.assertIn('total_components', stats)
        self.assertIn('total_relationships', stats)
        self.assertIn('average_coupling', stats)
        self.assertIn('complexity_score', stats)
        self.assertIn('layer_cohesion', stats)
        
        self.assertGreaterEqual(stats['total_components'], 8)  # At least our registered components
        self.assertGreaterEqual(stats['total_relationships'], 9)  # At least our registered relationships


class TestSystemView(unittest.TestCase):
    """Test the SystemView class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.identifier = SystemIdentifier()
        self.view = self.identifier.system_view
    
    def test_architectural_view(self):
        """Test architectural view generation."""
        arch_view = self.view.create_architectural_view(self.identifier)
        
        self.assertIsInstance(arch_view, str)
        self.assertIn('GAMESA SYSTEM ARCHITECTURE VIEW', arch_view)
        self.assertIn('APPLICATION', arch_view)
        self.assertIn('CORE_ENGINE', arch_view)
        self.assertIn('BUSINESS_LOGIC', arch_view)
    
    def test_dependency_view(self):
        """Test dependency view generation."""
        dep_view = self.view.create_dependency_view(self.identifier)
        
        self.assertIsInstance(dep_view, str)
        self.assertIn('GAMESA SYSTEM DEPENDENCY VIEW', dep_view)
        self.assertIn('DEPENDS ON:', dep_view)
    
    def test_conceptual_view(self):
        """Test conceptual view generation."""
        concept_view = self.view.create_conceptual_view(self.identifier)
        
        self.assertIsInstance(concept_view, str)
        self.assertIn('GAMESA SYSTEM CONCEPTUAL VIEW', concept_view)
        self.assertIn('CONCEPT:', concept_view)
        self.assertIn('Definition:', concept_view)
    
    def test_relationship_view(self):
        """Test relationship view generation."""
        rel_view = self.view.create_relationship_view(self.identifier)
        
        self.assertIsInstance(rel_view, str)
        self.assertIn('GAMESA SYSTEM RELATIONSHIP VIEW', rel_view)
        self.assertIn('--depends_on-->', rel_view)
        self.assertIn('--uses-->', rel_view)
    
    def test_component_detail_view(self):
        """Test component detail view generation."""
        detail_view = self.view.create_component_detail_view(self.identifier, 'essential_encoder')

        self.assertIsInstance(detail_view, str)
        self.assertIn('GAMESA COMPONENT DETAIL VIEW', detail_view)
        self.assertIn('Essential Encoder System', detail_view)
        self.assertIn('data_processor', detail_view)  # Lowercase as it appears in the output
        self.assertIn('core_engine', detail_view)  # Lowercase as it appears in the output
    
    def test_system_report(self):
        """Test system report generation."""
        report = self.view.generate_system_report(self.identifier)
        
        self.assertIsInstance(report, str)
        self.assertIn('GAMESA SYSTEM COMPREHENSIVE REPORT', report)
        self.assertIn('SYSTEM STATISTICS:', report)
        self.assertIn('SYSTEM ARCHITECTURE:', report)
        self.assertIn('COMPONENT TYPES DISTRIBUTION:', report)
        self.assertIn('KEY CONCEPTS IMPLEMENTED:', report)


def run_comprehensive_tests():
    """Run comprehensive tests for the system identification framework."""
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE: GAMESA SYSTEM IDENTIFICATION")
    print("=" * 80)
    
    # Create test suites
    identifier_suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemIdentifier)
    view_suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemView)
    
    # Combine all tests
    all_tests = unittest.TestSuite([
        identifier_suite,
        view_suite
    ])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n[OK] All tests passed! System identification framework is working correctly.")
        return True
    else:
        print("\n[X] Some tests failed. Please review the output above.")
        return False


def demo_integration():
    """Demonstrate integration between all system identification components."""
    print("\n" + "=" * 80)
    print("INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create system identifier
    identifier = SystemIdentifier()
    print("[OK] System Identifier created")
    
    # Create system view
    view = identifier.system_view
    print("[OK] System View created")
    
    # Demonstrate comprehensive system analysis
    print(f"\n--- System Analysis Demo ---")
    
    # Get system statistics
    stats = identifier.analyze_system_complexity()
    print(f"System Complexity: {stats['total_components']} components, {stats['total_relationships']} relationships")
    print(f"Average Coupling: {stats['average_coupling']:.2f}")
    print(f"Complexity Score: {stats['complexity_score']:.2f}")
    
    # Show architecture
    architecture = identifier.get_system_architecture()
    print(f"System Architecture:")
    for layer, components in architecture.items():
        print(f"  {layer}: {len(components)} components")
    
    # Show component types distribution
    print(f"\nComponent Types:")
    for comp_type in SystemComponentType:
        count = len(identifier.search_components_by_type(comp_type))
        if count > 0:
            print(f"  {comp_type.value}: {count} components")
    
    # Show key concepts
    print(f"\nKey Concepts Implemented: {len(identifier.concepts)}")
    for concept_name in list(identifier.concepts.keys())[:5]:  # Show first 5
        concept = identifier.concepts[concept_name]
        locations = len(concept.implementation_locations)
        print(f"  {concept_name}: implemented in {locations} components")
    
    # Demonstrate search capabilities
    print(f"\n--- Search Capabilities Demo ---")
    
    # Search by type
    data_processors = identifier.search_components_by_type(SystemComponentType.DATA_PROCESSOR)
    print(f"Data Processors: {data_processors}")
    
    # Search by layer
    app_components = identifier.search_components_by_layer(SystemLayer.APPLICATION)
    print(f"Application Layer: {app_components}")
    
    # Demonstrate relationship analysis
    print(f"\n--- Relationship Analysis Demo ---")
    
    # Analyze core components
    core_components = ['guardian_framework', 'essential_encoder', 'grid_memory_controller']
    for comp_name in core_components:
        rels = identifier.get_component_relationships(comp_name)
        print(f"{comp_name} has {len(rels)} relationships")
        
        # Show relationship types
        rel_types = [rel.relationship_type for rel in rels]
        print(f"  Relationship types: {list(set(rel_types))}")
    
    # Demonstrate conceptual understanding
    print(f"\n--- Conceptual Understanding Demo ---")
    
    # Analyze core concepts
    core_concepts = ['economic_resource_trading', 'three_d_grid_memory', 'neural_hardware_fabric']
    for concept_name in core_concepts:
        concept = identifier.find_concept(concept_name)
        if concept:
            print(f"{concept_name}:")
            print(f"  Category: {concept.category}")
            print(f"  Implemented in: {len(concept.implementation_locations)} components")
            print(f"  Examples: {len(concept.examples)} examples")
    
    # Show component details
    print(f"\n--- Component Detail Demo ---")
    
    detail_comp = 'grid_memory_controller'
    comp_detail = identifier.identify_component(detail_comp)
    if comp_detail:
        print(f"{detail_comp}:")
        print(f"  Type: {comp_detail.type.value}")
        print(f"  Layer: {comp_detail.layer.value}")
        print(f"  Interfaces: {comp_detail.interfaces}")
        print(f"  Dependencies: {comp_detail.dependencies}")
    
    # Show related components
    related = identifier.get_related_components('essential_encoder')
    print(f"\nComponents related to Essential Encoder: {related}")
    
    # Show dependency chain
    print(f"\n--- Dependency Chain Analysis ---")
    dependency_chains = {}
    for comp_name in identifier.components:
        deps = identifier.get_component_dependencies(comp_name)
        dependency_chains[comp_name] = len(deps)
    
    # Show components with most dependencies
    sorted_deps = sorted(dependency_chains.items(), key=lambda x: x[1], reverse=True)
    print("Top dependency consumers:")
    for comp_name, dep_count in sorted_deps[:5]:
        if dep_count > 0:
            print(f"  {comp_name}: {dep_count} dependencies")
    
    print(f"\n[OK] Integration demonstration completed successfully")
    print("All system identification components work together seamlessly:")
    print("- Component identification and metadata management")
    print("- Relationship mapping and dependency tracking")
    print("- Conceptual understanding and categorization")
    print("- Architectural and layer-based views")
    print("- Search and navigation capabilities")
    print("- Complexity analysis and metrics")
    print("- Comprehensive system reporting")
    
    print("=" * 80)


if __name__ == "__main__":
    print("Starting comprehensive test suite for GAMESA System Identification Framework...")
    
    # Run comprehensive tests
    tests_passed = run_comprehensive_tests()
    
    if tests_passed:
        # Run integration demonstration
        demo_integration()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED - INTEGRATION IS STABLE")
        print("=" * 80)
        print("The System Identification Framework is fully functional:")
        print("- Component identification and metadata management")
        print("- Relationship mapping and dependency tracking")
        print("- Conceptual understanding and categorization")
        print("- Architectural and layer-based views")
        print("- Search and navigation capabilities")
        print("- Complexity analysis and metrics")
        print("- Comprehensive system reporting")
        print("- Integration with all GAMESA framework components")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("TESTS FAILED - PLEASE REVIEW OUTPUT")
        print("=" * 80)