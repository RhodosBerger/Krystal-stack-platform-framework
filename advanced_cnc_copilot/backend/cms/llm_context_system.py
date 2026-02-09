"""
Real-Time LLM Context System with Semantic Meaning Library
Intelligent information picker that composes contextualized strings for LLM

ARCHITECTURE:
1. Meaning Library - Semantic definitions for all system concepts
2. Information Picker - Intelligently selects relevant data
3. Context Composer - Combines data with meanings
4. Real-Time LLM Accessor - Queries with rich context

PARADIGM: LLM sees the system through a "lens of meaning"
Each piece of data is annotated with its semantic significance
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re


# =============================================================================
# SEMANTIC MEANING LIBRARY
# =============================================================================

class MeaningCategory(Enum):
    """Categories of semantic meanings"""
    TECHNICAL = "technical"  # Engineering/manufacturing concepts
    ECONOMIC = "economic"  # Cost, profit, ROI
    TEMPORAL = "temporal"  # Time, scheduling, deadlines
    QUALITY = "quality"  # Defects, tolerances, standards
    BIOCHEMICAL = "biochemical"  # Cortisol, dopamine, stress
    PROCESS = "process"  # Workflows, procedures
    MATERIAL = "material"  # Material properties
    METAPHOR = "metaphor"  # KrystalStack paradigms


@dataclass
class SemanticMeaning:
    """
    Semantic meaning for a system concept
    
    Connects raw data strings to human/LLM-understandable concepts
    """
    concept_id: str
    category: MeaningCategory
    
    # Core meaning
    definition: str
    context: str
    
    # Related concepts
    synonyms: List[str] = field(default_factory=list)
    antonyms: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    
    # Interpretation rules
    interpretations: Dict[str, str] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Metaphorical mapping (KrystalStack)
    metaphor: Optional[str] = None
    
    # Examples
    examples: List[str] = field(default_factory=list)


class SemanticMeaningLibrary:
    """
    Library of semantic meanings for all system concepts
    
    METAPHOR: This is the "Rosetta Stone" that lets LLM understand
    manufacturing data in human terms
    """
    
    def __init__(self):
        self.meanings: Dict[str, SemanticMeaning] = {}
        self._initialize_library()
    
    def _initialize_library(self):
        """Initialize with comprehensive meanings"""
        
        # =====================================================================
        # TECHNICAL MEANINGS
        # =====================================================================
        
        self.add_meaning(SemanticMeaning(
            concept_id="spindle_load",
            category=MeaningCategory.TECHNICAL,
            definition="Percentage of maximum spindle motor capacity being used",
            context="Indicates how hard the machine is working. High load = difficult cutting.",
            synonyms=["spindle_utilization", "motor_load", "cutting_force"],
            interpretations={
                "low": "Spindle operating with minimal effort - possibly air cutting or light material",
                "normal": "Healthy cutting operation - appropriate for material and tool",
                "high": "Spindle working hard - may indicate tool wear or inappropriate parameters",
                "critical": "Spindle overloaded - risk of motor overheating or tool breakage"
            },
            thresholds={
                "low": 30.0,
                "normal": 60.0,
                "high": 80.0,
                "critical": 95.0
            },
            metaphor="Like a car engine - low RPM cruising vs. high RPM racing",
            examples=[
                "Spindle load at 45% - normal aluminum milling",
                "Spindle load at 92% - titanium roughing with worn tool"
            ]
        ))
        
        self.add_meaning(SemanticMeaning(
            concept_id="vibration",
            category=MeaningCategory.TECHNICAL,
            definition="Oscillation amplitude of machine structure during cutting",
            context="Indicates cutting stability. High vibration = chatter = poor finish.",
            synonyms=["chatter", "oscillation", "instability"],
            related_concepts=["tool_wear", "surface_finish", "cutting_parameters"],
            interpretations={
                "minimal": "Stable cutting - good tool condition and parameters",
                "moderate": "Some vibration present - monitor for increase",
                "severe": "Chatter detected - poor surface finish likely",
                "critical": "Violent vibration - stop immediately to prevent damage"
            },
            thresholds={
                "minimal": 0.5,
                "moderate": 1.5,
                "severe": 3.0,
                "critical": 5.0
            },
            metaphor="Visual entropy in KrystalStack - chaos in the cutting field",
            examples=[
                "Vibration 0.15mm - smooth cutting, good finish expected",
                "Vibration 4.2mm - severe chatter, reduce spindle speed"
            ]
        ))
        
        self.add_meaning(SemanticMeaning(
            concept_id="tool_health",
            category=MeaningCategory.TECHNICAL,
            definition="Remaining useful life of cutting tool as percentage",
            context="Predicts when tool needs replacement. Affects quality and cycle time.",
            synonyms=["tool_wear", "tool_life", "edge_condition"],
            related_concepts=["surface_finish", "dimensional_accuracy", "cycle_time"],
            interpretations={
                "excellent": "Tool like new - optimal cutting performance",
                "good": "Tool still effective - normal wear present",
                "fair": "Tool degrading - consider replacement soon",
                "poor": "Tool worn - replacement needed",
                "critical": "Tool failed - emergency stop required"
            },
            thresholds={
                "excellent": 0.9,
                "good": 0.7,
                "fair": 0.5,
                "poor": 0.3,
                "critical": 0.1
            },
            metaphor="Aging organism - vitality decreases with use",
            examples=[
                "Tool health 95% - brand new insert, 380 parts remaining",
                "Tool health 15% - replace before next part"
            ]
        ))
        
        # =====================================================================
        # BIOCHEMICAL MEANINGS (KrystalStack Paradigm)
        # =====================================================================
        
        self.add_meaning(SemanticMeaning(
            concept_id="cortisol",
            category=MeaningCategory.BIOCHEMICAL,
            definition="System stress level (0-100) responding to operational challenges",
            context="High cortisol = system under stress from vibration, wear, or anomalies",
            synonyms=["stress_level", "system_anxiety", "operational_stress"],
            antonyms=["calm", "stable"],
            related_concepts=["vibration", "tool_health", "quality"],
            interpretations={
                "calm": "System relaxed - smooth operation, no concerns",
                "alert": "Mild stress - system detecting minor issues",
                "stressed": "Moderate stress - multiple concerns present",
                "critical": "Extreme stress - emergency situation"
            },
            thresholds={
                "calm": 20.0,
                "alert": 50.0,
                "stressed": 75.0,
                "critical": 90.0
            },
            metaphor="Stress hormone - rises with system discomfort",
            examples=[
                "Cortisol 12 - machine happy, smooth cutting",
                "Cortisol 88 - machine stressed, high vibration detected"
            ]
        ))
        
        self.add_meaning(SemanticMeaning(
            concept_id="dopamine",
            category=MeaningCategory.BIOCHEMICAL,
            definition="Flow state indicator (0-100) - system operating optimally",
            context="High dopamine = machine in 'flow' - effortless production",
            synonyms=["flow_state", "optimal_operation", "system_happiness"],
            antonyms=["struggle", "difficulty"],
            related_concepts=["efficiency", "quality", "cycle_time"],
            interpretations={
                "struggling": "System not in flow - parameters suboptimal",
                "working": "Moderate flow - acceptable operation",
                "flowing": "Good flow state - efficient production",
                "peak": "Peak performance - everything perfect"
            },
            thresholds={
                "struggling": 30.0,
                "working": 60.0,
                "flowing": 80.0,
                "peak": 95.0
            },
            metaphor="Flow hormone - rises when system operating effortlessly",
            examples=[
                "Dopamine 95 - machine loves this job, perfect parameters",
                "Dopamine 25 - machine struggling, adjust speeds/feeds"
            ]
        ))
        
        # =====================================================================
        # ECONOMIC MEANINGS
        # =====================================================================
        
        self.add_meaning(SemanticMeaning(
            concept_id="cost_per_part",
            category=MeaningCategory.ECONOMIC,
            definition="Total manufacturing cost for single part (material + labor + overhead)",
            context="Key metric for profitability and pricing decisions",
            synonyms=["unit_cost", "piece_cost", "manufacturing_cost"],
            related_concepts=["profit_margin", "price", "volume"],
            interpretations={
                "very_low": "Extremely competitive cost - high profit potential",
                "low": "Good cost position - healthy margins possible",
                "moderate": "Average cost - standard industry pricing",
                "high": "Expensive - may need process optimization",
                "very_high": "Uncompetitive - requires immediate cost reduction"
            },
            thresholds={
                "very_low": 20.0,
                "low": 50.0,
                "moderate": 100.0,
                "high": 200.0,
                "very_high": 500.0
            },
            examples=[
                "Cost $35/part - profitable for aerospace application",
                "Cost $250/part - too high for automotive, optimize material usage"
            ]
        ))
        
        # =====================================================================
        # QUALITY MEANINGS
        # =====================================================================
        
        self.add_meaning(SemanticMeaning(
            concept_id="dimensional_accuracy",
            category=MeaningCategory.QUALITY,
            definition="Deviation from nominal dimension in millimeters",
            context="How close part is to perfect - critical for fit/function",
            synonyms=["tolerance", "precision", "deviation"],
            related_concepts=["tool_health", "machine_capability", "inspection"],
            interpretations={
                "excellent": "Well within tolerance - perfect part",
                "good": "Within tolerance - acceptable quality",
                "marginal": "Near tolerance limit - borderline accept",
                "out_of_spec": "Outside tolerance - reject part"
            },
            thresholds={
                "excellent": 0.01,  # ±0.01mm
                "good": 0.03,
                "marginal": 0.05,
                "out_of_spec": 0.10
            },
            examples=[
                "Accuracy ±0.008mm - excellent precision, tight tolerance met",
                "Accuracy ±0.12mm - reject, investigate tool wear"
            ]
        ))
        
        # =====================================================================
        # MATERIAL MEANINGS
        # =====================================================================
        
        self.add_meaning(SemanticMeaning(
            concept_id="material_hardness",
            category=MeaningCategory.MATERIAL,
            definition="Resistance to deformation measured in Brinell Hardness (HB)",
            context="Harder materials = slower cutting, more tool wear, higher cost",
            synonyms=["hardness", "material_strength", "machinability_factor"],
            related_concepts=["cutting_speed", "tool_life", "surface_finish"],
            interpretations={
                "soft": "Easy to machine - high speeds possible (Aluminum, Brass)",
                "medium": "Moderate difficulty - standard parameters (Low carbon steel)",
                "hard": "Difficult to machine - slow speeds required (Tool steel, Stainless)",
                "very_hard": "Extremely challenging - specialized tooling needed (Titanium, Inconel)"
            },
            thresholds={
                "soft": 100,
                "medium": 200,
                "hard": 300,
                "very_hard": 400
            },
            examples=[
                "Aluminum 6061: 95 HB - soft, machines easily",
                "Inconel 718: 380 HB - very hard, exotic tooling required"
            ]
        ))
    
    def add_meaning(self, meaning: SemanticMeaning):
        """Add semantic meaning to library"""
        self.meanings[meaning.concept_id] = meaning
    
    def get_meaning(self, concept_id: str) -> Optional[SemanticMeaning]:
        """Retrieve meaning for concept"""
        return self.meanings.get(concept_id)
    
    def interpret_value(self, concept_id: str, value: float) -> str:
        """
        Interpret numeric value using semantic thresholds
        
        Args:
            concept_id: Concept to interpret
            value: Numeric value
        
        Returns:
            Interpretation string
        """
        meaning = self.get_meaning(concept_id)
        if not meaning or not meaning.thresholds:
            return f"Value: {value}"
        
        # Find matching threshold
        for level, threshold in sorted(meaning.thresholds.items(), key=lambda x: x[1]):
            if value <= threshold:
                interpretation = meaning.interpretations.get(level, level)
                return f"{interpretation} (value: {value})"
        
        # Above all thresholds
        last_level = list(meaning.thresholds.keys())[-1]
        return meaning.interpretations.get(last_level, last_level)
    
    def get_related_concepts(self, concept_id: str, depth: int = 1) -> List[str]:
        """Get related concepts up to specified depth"""
        meaning = self.get_meaning(concept_id)
        if not meaning:
            return []
        
        related = set(meaning.related_concepts)
        
        if depth > 1:
            for rel_id in meaning.related_concepts:
                related.update(self.get_related_concepts(rel_id, depth - 1))
        
        return list(related)


# =============================================================================
# INTELLIGENT INFORMATION PICKER
# =============================================================================

@dataclass
class SystemDataSource:
    """Data source in the system"""
    source_id: str
    source_type: str  # 'database', 'sensor', 'api', 'file'
    description: str
    available_concepts: List[str]  # Concept IDs available from this source
    query_function: Optional[callable] = None


class IntelligentInformationPicker:
    """
    Intelligently selects relevant information based on context
    
    CAPABILITY: Natural language request → Relevant data selection
    """
    
    def __init__(self, meaning_library: SemanticMeaningLibrary):
        self.meaning_library = meaning_library
        self.data_sources: Dict[str, SystemDataSource] = {}
        self.query_cache: Dict[str, Tuple[datetime, Any]] = {}
        self.cache_ttl_seconds = 60
    
    def register_data_source(self, source: SystemDataSource):
        """Register a data source"""
        self.data_sources[source.source_id] = source
    
    def pick_relevant_concepts(self, query: str, max_concepts: int = 10) -> List[str]:
        """
        Pick relevant concepts based on natural language query
        
        Args:
            query: Natural language query
            max_concepts: Maximum concepts to return
        
        Returns:
            List of relevant concept IDs
        """
        query_lower = query.lower()
        
        # Score concepts by relevance
        scored_concepts = []
        
        for concept_id, meaning in self.meaning_library.meanings.items():
            score = 0
            
            # Direct match in concept ID
            if concept_id.replace('_', ' ') in query_lower:
                score += 10
            
            # Match in definition
            if any(word in query_lower for word in meaning.definition.lower().split()):
                score += 5
            
            # Match in synonyms
            for synonym in meaning.synonyms:
                if synonym.replace('_', ' ') in query_lower:
                    score += 7
            
            # Match in context
            if any(word in query_lower for word in meaning.context.lower().split()):
                score += 3
            
            # Category relevance
            if meaning.category.value in query_lower:
                score += 4
            
            if score > 0:
                scored_concepts.append((concept_id, score))
        
        # Sort by score and return top N
        scored_concepts.sort(key=lambda x: x[1], reverse=True)
        return [concept_id for concept_id, score in scored_concepts[:max_concepts]]
    
    def gather_data_for_concepts(self, concept_ids: List[str]) -> Dict[str, Any]:
        """
        Gather actual data for concept IDs
        
        Args:
            concept_ids: List of concept IDs
        
        Returns:
            Dictionary mapping concept_id to current value
        """
        data = {}
        
        for concept_id in concept_ids:
            # Check cache
            if concept_id in self.query_cache:
                cached_time, cached_value = self.query_cache[concept_id]
                age_seconds = (datetime.now() - cached_time).total_seconds()
                
                if age_seconds < self.cache_ttl_seconds:
                    data[concept_id] = cached_value
                    continue
            
            # Find data source
            for source_id, source in self.data_sources.items():
                if concept_id in source.available_concepts:
                    if source.query_function:
                        value = source.query_function(concept_id)
                        data[concept_id] = value
                        
                        # Cache
                        self.query_cache[concept_id] = (datetime.now(), value)
                    break
        
        return data


# =============================================================================
# CONTEXT COMPOSER
# =============================================================================

class LLMContextComposer:
    """
    Composes rich context for LLM by combining data with meanings
    
    OUTPUT: Semantically rich text that LLM can deeply understand
    """
    
    def __init__(self, 
                 meaning_library: SemanticMeaningLibrary,
                 information_picker: IntelligentInformationPicker):
        self.meaning_library = meaning_library
        self.information_picker = information_picker
    
    def compose_context(self, 
                       query: str,
                       include_definitions: bool = True,
                       include_metaphors: bool = True,
                       include_thresholds: bool = True) -> str:
        """
        Compose rich context for LLM
        
        Args:
            query: User's question/request
            include_definitions: Include semantic definitions
            include_metaphors: Include KrystalStack metaphors
            include_thresholds: Include interpretation thresholds
        
        Returns:
            Rich context string for LLM
        """
        # Pick relevant concepts
        concept_ids = self.information_picker.pick_relevant_concepts(query)
        
        # Gather current data
        current_data = self.information_picker.gather_data_for_concepts(concept_ids)
        
        # Compose context
        context_parts = []
        
        context_parts.append("=== MANUFACTURING SYSTEM CONTEXT ===\n")
        context_parts.append(f"Query: {query}\n")
        context_parts.append(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        context_parts.append("=== CURRENT SYSTEM STATE ===\n")
        
        for concept_id in concept_ids:
            meaning = self.meaning_library.get_meaning(concept_id)
            if not meaning:
                continue
            
            value = current_data.get(concept_id, "N/A")
            
            context_parts.append(f"\n**{concept_id.replace('_', ' ').title()}**")
            
            # Current value with interpretation
            if isinstance(value, (int, float)):
                interpretation = self.meaning_library.interpret_value(concept_id, value)
                context_parts.append(f"  Current: {interpretation}")
            else:
                context_parts.append(f"  Current: {value}")
            
            # Definition
            if include_definitions:
                context_parts.append(f"  Definition: {meaning.definition}")
                context_parts.append(f"  Context: {meaning.context}")
            
            # Metaphor
            if include_metaphors and meaning.metaphor:
                context_parts.append(f"  Metaphor: {meaning.metaphor}")
            
            # Thresholds
            if include_thresholds and meaning.thresholds:
                threshold_str = ", ".join([f"{k}:<{v}" for k, v in meaning.thresholds.items()])
                context_parts.append(f"  Thresholds: {threshold_str}")
            
            # Related concepts
            if meaning.related_concepts:
                context_parts.append(f"  Related: {', '.join(meaning.related_concepts)}")
        
        return "\n".join(context_parts)


# =============================================================================
# REAL-TIME LLM ACCESSOR
# =============================================================================

class RealTimeLLMAccessor:
    """
    Accesses LLM with real-time contextualized information
    
    WORKFLOW:
    1. User asks question
    2. Pick relevant concepts
    3. Gather current data
    4. Compose rich context
    5. Query LLM with context
    6. Return answer
    """
    
    def __init__(self):
        self.meaning_library = SemanticMeaningLibrary()
        self.information_picker = IntelligentInformationPicker(self.meaning_library)
        self.context_composer = LLMContextComposer(self.meaning_library, self.information_picker)
        
        # Register simulated data sources
        self._register_simulated_sources()
    
    def _register_simulated_sources(self):
        """Register simulated data sources for demo"""
        
        # Simulated sensor data
        def get_sensor_data(concept_id: str):
            # Simulate real-time sensor readings
            import random
            simulated_values = {
                'spindle_load': random.uniform(40, 70),
                'vibration': random.uniform(0.1, 0.8),
                'tool_health': random.uniform(0.6, 0.95),
                'cortisol': random.uniform(10, 40),
                'dopamine': random.uniform(60, 95),
                'dimensional_accuracy': random.uniform(0.01, 0.04),
                'cost_per_part': random.uniform(50, 150),
                'material_hardness': random.choice([95, 250, 340, 380])
            }
            return simulated_values.get(concept_id, 0)
        
        self.information_picker.register_data_source(SystemDataSource(
            source_id="realtime_sensors",
            source_type="sensor",
            description="Real-time machine sensor data",
            available_concepts=[
                'spindle_load', 'vibration', 'tool_health',
                'cortisol', 'dopamine', 'dimensional_accuracy'
            ],
            query_function=get_sensor_data
        ))
        
        self.information_picker.register_data_source(SystemDataSource(
            source_id="economic_database",
            source_type="database",
            description="Economic and cost data",
            available_concepts=['cost_per_part'],
            query_function=get_sensor_data
        ))
        
        self.information_picker.register_data_source(SystemDataSource(
            source_id="material_database",
            source_type="database",
            description="Material properties database",
            available_concepts=['material_hardness'],
            query_function=get_sensor_data
        ))
    
    def query_with_context(self, user_question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Answer question with full semantic context
        
        Args:
            user_question: User's question
            verbose: Include full context in response
        
        Returns:
            Dictionary with context and (simulated) LLM response
        """
        # Compose context
        context = self.context_composer.compose_context(user_question)
        
        if verbose:
            print("\n" + "="*70)
            print("COMPOSED CONTEXT FOR LLM:")
            print("="*70)
            print(context)
            print("="*70 + "\n")
        
        # Simulate LLM response (would call actual LLM in production)
        llm_response = self._simulate_llm_response(user_question, context)
        
        return {
            'question': user_question,
            'context': context,
            'llm_response': llm_response,
            'timestamp': datetime.now().isoformat()
        }
    
    def _simulate_llm_response(self, question: str, context: str) -> str:
        """Simulate LLM response (would be real LLM call)"""
        return (
            f"Based on the current system state, I can see that:\n\n"
            f"The manufacturing system is operating within normal parameters. "
            f"The biochemical indicators show healthy operation with low stress (cortisol) "
            f"and good flow state (dopamine). Tool health is good and dimensional accuracy "
            f"is within acceptable tolerances.\n\n"
            f"[This is a simulated response - actual LLM would provide deeper analysis based on context]"
        )


# Example usage
if __name__ == "__main__":
    print("Real-Time LLM Context System with Semantic Meaning Library")
    print("=" * 70)
    
    # Create accessor
    accessor = RealTimeLLMAccessor()
    
    # Example queries
    queries = [
        "How is the machine performing right now?",
        "Is there any chatter or vibration issues?",
        "What's the current system stress level?",
        "Are we making good quality parts?"
    ]
    
    for query in queries:
        print(f"\n🤔 USER QUESTION: {query}")
        result = accessor.query_with_context(query, verbose=False)
        
        # Show picked concepts
        concept_ids = accessor.information_picker.pick_relevant_concepts(query, max_concepts=5)
        print(f"\n📊 Relevant concepts picked: {', '.join(concept_ids)}")
        
        # Show current values with interpretations
        data = accessor.information_picker.gather_data_for_concepts(concept_ids)
        print(f"\n📈 Current values:")
        for concept_id, value in data.items():
            interpretation = accessor.meaning_library.interpret_value(concept_id, value)
            print(f"  - {concept_id}: {interpretation}")
        
        print(f"\n💬 LLM Response: {result['llm_response']}")
        print("\n" + "-" * 70)
