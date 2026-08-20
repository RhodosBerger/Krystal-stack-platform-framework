"""
LLM Conversation Mediator & Prompt Library
Structured communication framework for User ↔ LLM ↔ Backend ↔ LLM ↔ User

FEATURES:
- Standardized prompt templates for all manufacturing scenarios
- Conversation flow orchestration
- Intent detection and topic classification
- Database logging of all interactions
- Topic relationship mapping
- Context preservation across multi-turn conversations

ARCHITECTURE:
User Input → Intent Classifier → Prompt Template → LLM → Backend Action
    ↓                                                          ↓
Database Log                                            Response Generator
    ↓                                                          ↓
Topic Relations                                         LLM → User Output
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re


# =============================================================================
# CONVERSATION TAXONOMY
# =============================================================================

class ConversationIntent(Enum):
    """User intent classification"""
    # Manufacturing Operations
    QUOTE_REQUEST = "quote_request"
    TECHNICAL_QUESTION = "technical_question"
    TROUBLESHOOTING = "troubleshooting"
    OPTIMIZATION = "optimization"
    MATERIAL_SELECTION = "material_selection"
    
    # Programming
    GCODE_GENERATION = "gcode_generation"
    GCODE_DEBUG = "gcode_debug"
    PROGRAM_MODIFICATION = "program_modification"
    
    # Quality & Inspection
    QUALITY_ISSUE = "quality_issue"
    DIMENSIONAL_PROBLEM = "dimensional_problem"
    SURFACE_FINISH = "surface_finish"
    
    # Maintenance
    MACHINE_PROBLEM = "machine_problem"
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    TOOL_SELECTION = "tool_selection"
    
    # Design
    DESIGN_VALIDATION = "design_validation"
    FEASIBILITY_CHECK = "feasibility_check"
    COST_REDUCTION = "cost_reduction"
    
    # General
    INFORMATION_LOOKUP = "information_lookup"
    TRAINING_QUESTION = "training_question"
    GENERAL_CHAT = "general_chat"


class ManufacturingTopic(str, Enum):
    """Manufacturing knowledge topics"""
    SPEEDS_FEEDS = "speeds_feeds"
    TOOLING = "tooling"
    MATERIALS = "materials"
    MACHINING_OPERATIONS = "machining_operations"
    SETUP_WORKHOLDING = "setup_workholding"
    QUALITY_CONTROL = "quality_control"
    GCODE_PROGRAMMING = "gcode_programming"
    MACHINE_MAINTENANCE = "machine_maintenance"
    DESIGN_MANUFACTURING = "design_manufacturing"
    COST_ESTIMATION = "cost_estimation"
    PROCESS_PLANNING = "process_planning"
    SAFETY_PROCEDURES = "safety_procedures"


# =============================================================================
# CONVERSATION DATA STRUCTURES
# =============================================================================

@dataclass
class ConversationContext:
    """Context preserved across conversation turns"""
    conversation_id: str
    user_id: str
    session_start: datetime
    turn_number: int = 0
    
    # Accumulated context
    mentioned_materials: List[str] = field(default_factory=list)
    mentioned_operations: List[str] = field(default_factory=list)
    mentioned_machines: List[str] = field(default_factory=list)
    active_topics: List[ManufacturingTopic] = field(default_factory=list)
    
    # Previous interactions
    conversation_history: List[Dict] = field(default_factory=list)
    
    def add_turn(self, user_input: str, bot_response: str, intent: ConversationIntent, topics: List[ManufacturingTopic]):
        """Add conversation turn"""
        self.turn_number += 1
        self.conversation_history.append({
            'turn': self.turn_number,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': bot_response,
            'intent': intent.value,
            'topics': [t.value for t in topics]
        })
        
        # Update active topics
        for topic in topics:
            if topic not in self.active_topics:
                self.active_topics.append(topic)


@dataclass
class PromptTemplate:
    """Structured prompt template"""
    template_id: str
    intent: ConversationIntent
    topics: List[ManufacturingTopic]
    system_prompt: str
    user_prompt_template: str
    example_inputs: List[str]
    expected_output_format: str
    backend_actions: List[str]  # What backend should do with response


# =============================================================================
# PROMPT LIBRARY
# =============================================================================

PROMPT_LIBRARY = {
    # =========================================================================
    # MANUFACTURING OPERATIONS
    # =========================================================================
    
    ConversationIntent.QUOTE_REQUEST: PromptTemplate(
        template_id="QUOTE_001",
        intent=ConversationIntent.QUOTE_REQUEST,
        topics=[ManufacturingTopic.COST_ESTIMATION, ManufacturingTopic.PROCESS_PLANNING],
        system_prompt="""You are QuoteMaster, an expert manufacturing cost estimator.
Your role: Extract part specifications from customer requests and provide accurate quotes.

RESPONSE FORMAT:
1. Part specification summary
2. Material recommendation
3. Operations required
4. Cost breakdown
5. Lead time estimate
6. Confidence level

Be professional, accurate, and ask clarifying questions if needed.""",
        
        user_prompt_template="""Customer quote request:
{user_input}

Current context:
- Customer: {customer_name}
- Previous orders: {order_history}
- Preferred materials: {material_preferences}

Extract:
1. Part description and dimensions
2. Quantity
3. Material (if specified)
4. Tolerance requirements
5. Delivery deadline

Provide quote estimation.""",
        
        example_inputs=[
            "Need 100 aluminum brackets, 50x50x20mm",
            "Quote for 500 titanium shafts, aerospace grade",
            "How much for stainless steel gears, 30 teeth, 50mm diameter?"
        ],
        
        expected_output_format="""JSON:
{
  "part_spec": {...},
  "material": "...",
  "operations": [...],
  "unit_cost": 45.00,
  "total_cost": 4500.00,
  "lead_time_days": 14,
  "confidence": 0.92
}""",
        
        backend_actions=[
            "create_quote_record",
            "notify_sales_team",
            "check_material_inventory",
            "reserve_machine_capacity"
        ]
    ),
    
    # =========================================================================
    
    ConversationIntent.TECHNICAL_QUESTION: PromptTemplate(
        template_id="TECH_001",
        intent=ConversationIntent.TECHNICAL_QUESTION,
        topics=[ManufacturingTopic.SPEEDS_FEEDS, ManufacturingTopic.TOOLING],
        system_prompt="""You are MachineMaster, expert CNC machinist with 30+ years experience.
Your role: Answer technical questions with specific, actionable advice.

KNOWLEDGE AREAS:
- Speeds & feeds for all materials
- Tool selection and geometry
- Machining strategies
- Best practices

RESPONSE STYLE:
1. Direct answer first
2. Explanation of why
3. Specific recommendations (numbers, not ranges)
4. Safety considerations if relevant
5. Alternative approaches

Always cite best practices from database when available.""",
        
        user_prompt_template="""Technical question:
{user_input}

Context:
- Material: {material}
- Operation: {operation}
- Machine: {machine}
- Operator experience: {operator_level}

Relevant best practices from database:
{best_practices}

Provide expert technical answer.""",
        
        example_inputs=[
            "What cutting speed for aluminum 6061?",
            "Why am I getting chatter when milling steel?",
            "Best tool for titanium machining?"
        ],
        
        expected_output_format="""Markdown:
**Direct Answer:** [Specific recommendation]

**Explanation:** [Why this works]

**Recommended Parameters:**
- Speed: X RPM
- Feed: Y mm/min
- Tool: Z

**Cautions:** [Safety/quality notes]""",
        
        backend_actions=[
            "log_technical_query",
            "retrieve_best_practices",
            "track_topic_popularity",
            "update_knowledge_graph"
        ]
    ),
    
    # =========================================================================
    
    ConversationIntent.GCODE_GENERATION: PromptTemplate(
        template_id="GCODE_001",
        intent=ConversationIntent.GCODE_GENERATION,
        topics=[ManufacturingTopic.GCODE_PROGRAMMING],
        system_prompt="""You are CodeCraft, expert G-Code programming assistant.
Your role: Generate safe, efficient, validated G-Code from natural language.

RULES:
1. Always include safety blocks (coordinate system, cancel modes)
2. Use incremental depths for deep cuts
3. Include tool changes and spindle control
4. Add comments for clarity
5. Validate before sending

RESPONSE:
1. Confirm understanding of request
2. Present program structure
3. Provide complete G-Code
4. List validation checks performed
5. Suggest optimizations if any""",
        
        user_prompt_template="""G-Code generation request:
{user_input}

Part details:
- Material: {material}
- Current setup: {setup_info}
- Available tools: {tool_list}

Best practices for this operation:
{operation_best_practices}

Generate validated G-Code program.""",
        
        example_inputs=[
            "Mill 50mm square pocket, 10mm deep, leave 0.5mm for finishing",
            "Drill 4 holes, 8mm diameter on 100mm bolt circle",
            "Face mill top surface, remove 1mm"
        ],
        
        expected_output_format="""
**Program Summary:**
- Operation: ...
- Tool: ...
- Estimated time: ...

**G-Code:**
```gcode
O0001 (PROGRAM NAME)
...
M30
%
```

**Validation:** ✅ All checks passed
**Optimizations:** [Suggestions]""",
        
        backend_actions=[
            "generate_gcode",
            "validate_program",
            "simulate_toolpath",
            "save_to_program_library",
            "log_generated_program"
        ]
    ),
    
    # =========================================================================
    
    ConversationIntent.TROUBLESHOOTING: PromptTemplate(
        template_id="TROUBLE_001",
        intent=ConversationIntent.TROUBLESHOOTING,
        topics=[ManufacturingTopic.QUALITY_CONTROL, ManufacturingTopic.MACHINING_OPERATIONS],
        system_prompt="""You are PrecisionGuard, quality troubleshooting expert.
Your role: Diagnose manufacturing problems and provide solutions.

DIAGNOSTIC APPROACH:
1. Identify symptoms
2. List possible root causes
3. Recommend diagnostic steps
4. Provide solutions ranked by likelihood
5. Preventive measures

Use systematic problem-solving methodology.
Reference similar issues from database.""",
        
        user_prompt_template="""Problem report:
{user_input}

Current conditions:
- Part: {part_info}
- Material: {material}
- Machine: {machine}
- Recent changes: {recent_changes}

Similar past issues:
{similar_issues}

Telemetry data (if available):
{telemetry_summary}

Diagnose and provide solution.""",
        
        example_inputs=[
            "Getting rough surface finish on aluminum",
            "Parts are out of tolerance by 0.05mm",
            "Tool breaking frequently on this job"
        ],
        
        expected_output_format="""
**Symptoms:** [Summary]

**Likely Root Causes:**
1. [Cause 1] - 70% probability
2. [Cause 2] - 20% probability
3. [Cause 3] - 10% probability

**Diagnostic Steps:**
- [Test 1]
- [Test 2]

**Solutions:**
1. **Primary:** [Action]
2. **Alternative:** [Action]

**Prevention:** [How to avoid in future]""",
        
        backend_actions=[
            "log_issue",
            "retrieve_similar_cases",
            "analyze_telemetry",
            "create_corrective_action",
            "update_troubleshooting_kb"
        ]
    ),
    
    # =========================================================================
    
    ConversationIntent.MATERIAL_SELECTION: PromptTemplate(
        template_id="MATERIAL_001",
        intent=ConversationIntent.MATERIAL_SELECTION,
        topics=[ManufacturingTopic.MATERIALS, ManufacturingTopic.DESIGN_MANUFACTURING],
        system_prompt="""You are MaterialExpert, materials science specialist.
Your role: Recommend optimal materials based on application requirements.

CONSIDER:
- Mechanical properties (strength, hardness, toughness)
- Environmental conditions (corrosion, temperature)
- Manufacturability (machinability, weldability)
- Cost and availability
- Industry standards and certifications

RESPONSE:
1. Rank top 3 materials with pros/cons
2. Explain trade-offs
3. Provide specific grades (e.g., "6061-T6" not just "aluminum")
4. Include machinability notes
5. Reference industry standards""",
        
        user_prompt_template="""Material selection request:
{user_input}

Application requirements:
- Load/stress: {load_requirements}
- Environment: {environment}
- Temperature: {temperature_range}
- Budget: {budget_constraints}

Available materials in database:
{available_materials}

Machining capabilities:
{machine_capabilities}

Recommend material with justification.""",
        
        example_inputs=[
            "What material for gear housing? Needs to be lightweight and strong",
            "Corrosion-resistant material for marine application",
            "Best material for high-temperature aerospace bracket?"
        ],
        
        expected_output_format="""
**Recommendations:**

1. ⭐ **[Material Grade]** (BEST CHOICE)
   - Strength: X MPa
   - Weight: Y g/cm³
   - Cost factor: Z
   - Machinability: Excellent/Good/Fair
   - Pros: [List]
   - Cons: [List]

2. **[Alternative 1]**
   [Same format]

3. **[Alternative 2]**
   [Same format]

**Final Recommendation:** [Material] because [justification]""",
        
        backend_actions=[
            "query_material_database",
            "calculate_properties",
            "check_inventory",
            "log_material_selection",
            "update_material_recommendations"
        ]
    ),
    
    # =========================================================================
    
    ConversationIntent.OPTIMIZATION: PromptTemplate(
        template_id="OPT_001",
        intent=ConversationIntent.OPTIMIZATION,
        topics=[ManufacturingTopic.PROCESS_PLANNING, ManufacturingTopic.COST_ESTIMATION],
        system_prompt="""You are CostOptimizer, process optimization specialist.
Your role: Identify opportunities to reduce cost, time, or improve quality.

OPTIMIZATION TARGETS:
- Cycle time reduction
- Material waste minimization
- Tool life extension
- Setup time reduction
- Quality improvement

METHODOLOGY:
1. Analyze current process
2. Identify bottlenecks/waste
3. Propose specific improvements
4. Quantify expected benefits
5. Consider trade-offs

Provide actionable recommendations with measurable results.""",
        
        user_prompt_template="""Optimization request:
{user_input}

Current process:
- Cycle time: {current_cycle_time}
- Cost per part: {current_cost}
- Quality metrics: {quality_metrics}
- Bottlenecks: {bottlenecks}

Historical data:
{performance_data}

Similar optimized processes:
{similar_optimizations}

Analyze and recommend improvements.""",
        
        example_inputs=[
            "How can I reduce cycle time for this bracket?",
            "Too much scrap on this job, help optimize",
            "Need to cut costs by 20% on shaft production"
        ],
        
        expected_output_format="""
**Current State Analysis:**
- Cycle time: X min
- Cost: $Y
- Issues: [List]

**Optimization Opportunities:**

1. **[Opportunity 1]** (Impact: HIGH)
   - Change: [Specific action]
   - Expected benefit: -Z% time / -$W cost
   - Implementation: [How to do it]
   - Risk: Low/Medium/High

2. **[Opportunity 2]** (Impact: MEDIUM)
   [Same format]

**Total Projected Savings:**
- Time: -X%
- Cost: -$Y
- Quality: +Z%

**Implementation Priority:** [1, 2, 3...]""",
        
        backend_actions=[
            "analyze_process",
            "run_optimization_algorithm",
            "simulate_improvements",
            "calculate_roi",
            "create_implementation_plan",
            "log_optimization"
        ]
    ),
    
    # =========================================================================
    
    ConversationIntent.MACHINE_PROBLEM: PromptTemplate(
        template_id="MAINT_001",
        intent=ConversationIntent.MACHINE_PROBLEM,
        topics=[ManufacturingTopic.MACHINE_MAINTENANCE],
        system_prompt="""You are MaintenanceExpert, CNC maintenance specialist.
Your role: Diagnose machine problems and guide repairs.

EXPERTISE:
- Electrical systems
- Mechanical components
- Hydraulics/pneumatics
- Control systems
- Predictive maintenance

APPROACH:
1. Safety first - ensure machine is safe to inspect
2. Systematic diagnostics
3. Prioritize by likelihood
4. Step-by-step troubleshooting
5. Spare parts identification

Use predictive maintenance data if available.""",
        
        user_prompt_template="""Machine problem:
{user_input}

Machine details:
- Model: {machine_model}
- Age: {machine_age}
- Recent maintenance: {maintenance_history}

Symptoms:
{problem_symptoms}

Sensor data (last 24 hours):
{sensor_readings}

Predictive maintenance alerts:
{predictive_alerts}

Diagnose and provide repair guidance.""",
        
        example_inputs=[
            "Spindle making weird noise",
            "Machine won't home properly on X-axis",
            "Getting alarm 074 on FANUC controller"
        ],
        
        expected_output_format="""
**⚠️ SAFETY:** [Any safety precautions]

**Diagnosis:**
- Primary suspect: [Component]
- Confidence: X%
- Symptoms match: [Known issues]

**Diagnostic Steps:**
1. [Check this]
2. [Test that]
3. [Measure X]

**Likely Solution:**
- Action: [What to do]
- Parts needed: [SKU numbers]
- Time estimate: [Duration]
- Difficulty: Easy/Moderate/Expert

**If problem persists:** [Next steps]""",
        
        backend_actions=[
            "log_machine_issue",
            "check_predictive_alerts",
            "retrieve_maintenance_history",
            "check_parts_inventory",
            "create_work_order",
            "update_maintenance_kb"
        ]
    )
}


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================

class IntentClassifier:
    """Classify user intent from input"""
    
    def __init__(self):
        """Initialize classifier"""
        self.intent_keywords = {
            ConversationIntent.QUOTE_REQUEST: [
                'quote', 'cost', 'price', 'how much', 'estimate', 'need', 'order'
            ],
            ConversationIntent.TECHNICAL_QUESTION: [
                'speed', 'feed', 'rpm', 'cutting', 'tool', 'what', 'how', 'why'
            ],
            ConversationIntent.GCODE_GENERATION: [
                'mill', 'drill', 'gcode', 'g-code', 'program', 'face', 'pocket'
            ],
            ConversationIntent.TROUBLESHOOTING: [
                'problem', 'issue', 'rough', 'chatter', 'vibration', 'breaking', 'scrap'
            ],
            ConversationIntent.MATERIAL_SELECTION: [
                'material', 'aluminum', 'steel', 'titanium', 'best material', 'recommend material'
            ],
            ConversationIntent.OPTIMIZATION: [
                'optimize', 'faster', 'reduce', 'improve', 'better', 'cycle time'
            ],
            ConversationIntent.MACHINE_PROBLEM: [
                'alarm', 'error', 'broken', 'won\'t', 'noise', 'maintenance'
            ]
        }
    
    def classify(self, user_input: str) -> Tuple[ConversationIntent, float]:
        """
        Classify user intent
        
        Args:
            user_input: User's message
        
        Returns:
            (intent, confidence)
        """
        input_lower = user_input.lower()
        scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for kw in keywords if kw in input_lower)
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return (ConversationIntent.GENERAL_CHAT, 0.3)
        
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] / 3.0, 1.0)  # Normalize
        
        return (best_intent, confidence)


# =============================================================================
# TOPIC EXTRACTOR
# =============================================================================

class TopicExtractor:
    """Extract manufacturing topics from conversation"""
    
    def __init__(self):
        """Initialize extractor"""
        self.topic_keywords = {
            ManufacturingTopic.SPEEDS_FEEDS: ['speed', 'feed', 'rpm', 'sfm', 'cutting speed'],
            ManufacturingTopic.TOOLING: ['tool', 'end mill', 'insert', 'cutter', 'drill bit'],
            ManufacturingTopic.MATERIALS: ['aluminum', 'steel', 'titanium', 'material', 'alloy'],
            ManufacturingTopic.MACHINING_OPERATIONS: ['mill', 'drill', 'turn', 'bore', 'tap', 'ream'],
            ManufacturingTopic.SETUP_WORKHOLDING: ['vise', 'fixture', 'clamp', 'setup', 'workholding'],
            ManufacturingTopic.QUALITY_CONTROL: ['tolerance', 'dimension', 'finish', 'quality', 'inspection'],
            ManufacturingTopic.GCODE_PROGRAMMING: ['gcode', 'g-code', 'program', 'cam', 'toolpath'],
            ManufacturingTopic.MACHINE_MAINTENANCE: ['maintenance', 'repair', 'preventive', 'bearing', 'lubrication'],
            ManufacturingTopic.COST_ESTIMATION: ['cost', 'price', 'quote', 'budget', 'expensive']
        }
    
    def extract(self, text: str) -> List[ManufacturingTopic]:
        """Extract topics from text"""
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected_topics.append(topic)
        
        return detected_topics if detected_topics else [ManufacturingTopic.MACHINING_OPERATIONS]


# =============================================================================
# CONVERSATION MEDIATOR
# =============================================================================

class ConversationMediator:
    """
    Orchestrates conversation flow:
    User → Intent Detection → Prompt Selection → LLM → Backend → Response
    
    Logs everything to database and builds topic relationships
    """
    
    def __init__(self):
        """Initialize mediator"""
        self.intent_classifier = IntentClassifier()
        self.topic_extractor = TopicExtractor()
        self.active_conversations: Dict[str, ConversationContext] = {}
    
    def process_user_input(self, user_input: str, user_id: str, conversation_id: Optional[str] = None) -> Dict:
        """
        Process user input through complete pipeline
        
        Args:
            user_input: User's message
            user_id: User identifier
            conversation_id: Optional conversation ID (creates new if None)
        
        Returns:
            Complete response with metadata
        """
        # Get or create conversation context
        if conversation_id and conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
        else:
            conversation_id = f"CONV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            context = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                session_start=datetime.now()
            )
            self.active_conversations[conversation_id] = context
        
        # Step 1: Classify intent
        intent, intent_confidence = self.intent_classifier.classify(user_input)
        
        # Step 2: Extract topics
        topics = self.topic_extractor.extract(user_input)
        
        # Step 3: Get prompt template
        template = PROMPT_LIBRARY.get(intent)
        if not template:
            template = PROMPT_LIBRARY[ConversationIntent.TECHNICAL_QUESTION]  # Default
        
        # Step 4: Build complete prompt
        prompt = self._build_prompt(template, user_input, context)
        
        # Step 5: Simulate LLM response (in production, call actual LLM)
        llm_response = self._simulate_llm_response(intent, user_input)
        
        # Step 6: Log to database (simulated)
        self._log_to_database(context, user_input, intent, topics, llm_response)
        
        # Step 7: Update context
        context.add_turn(user_input, llm_response, intent, topics)
        
        # Step 8: Return structured response
        return {
            'conversation_id': conversation_id,
            'turn_number': context.turn_number,
            'intent': intent.value,
            'intent_confidence': intent_confidence,
            'topics': [t.value for t in topics],
            'response': llm_response,
            'prompt_template_used': template.template_id,
            'backend_actions_triggered': template.backend_actions,
            'context_summary': {
                'active_topics': [t.value for t in context.active_topics],
                'conversation_length': len(context.conversation_history)
            }
        }
    
    def _build_prompt(self, template: PromptTemplate, user_input: str, context: ConversationContext) -> str:
        """Build complete prompt from template"""
        # For production: Fill template with actual context data
        # For now, simplified
        return f"{template.system_prompt}\n\n{user_input}"
    
    def _simulate_llm_response(self, intent: ConversationIntent, user_input: str) -> str:
        """Simulate LLM response (replace with actual LLM call)"""
        responses = {
            ConversationIntent.QUOTE_REQUEST: f"Based on your request for parts: I estimate $45/part for 100 units. Lead time: 14 days. Material: Aluminum 6061. Would you like a detailed breakdown?",
            ConversationIntent.TECHNICAL_QUESTION: f"For this material and operation, I recommend: 3,500 RPM, 1,500 mm/min feed rate, using a 3-flute carbide end mill. This provides optimal balance of speed and tool life.",
            ConversationIntent.GCODE_GENERATION: f"I've generated G-Code for your operation. Program O0001 includes roughing and finishing passes. Estimated cycle time: 12 minutes. Would you like me to explain the program?",
            ConversationIntent.TROUBLESHOOTING: f"Based on your symptoms, this is likely tool wear (70% probability). Recommend replacing the end mill and reducing feed rate by 15%. Let me know if surface finish improves."
        }
        return responses.get(intent, f"I understand your question about: {user_input}. Let me provide a detailed answer...")
    
    def _log_to_database(self, context: ConversationContext, user_input: str, intent: ConversationIntent, topics: List[ManufacturingTopic], response: str):
        """Log conversation to database"""
        # In production: Insert into conversations table
        log_entry = {
            'conversation_id': context.conversation_id,
            'user_id': context.user_id,
            'turn_number': context.turn_number + 1,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'detected_intent': intent.value,
            'detected_topics': [t.value for t in topics],
            'bot_response': response
        }
        # print(f"📝 Logged: {log_entry}")  # Simulated


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("LLM Conversation Mediator - Demo")
    print("=" * 70)
    
    mediator = ConversationMediator()
    
    # Simulate multi-turn conversation
    test_inputs = [
        "I need a quote for 100 aluminum brackets",
        "What cutting speed should I use for aluminum 6061?",
        "Generate G-Code to mill a 50mm pocket",
        "I'm getting rough surface finish, what's wrong?"
    ]
    
    user_id = "USER-001"
    conversation_id = None
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n{'='*70}")
        print(f"Turn {i}")
        print(f"{'='*70}")
        print(f"👤 USER: {user_input}")
        
        response = mediator.process_user_input(user_input, user_id, conversation_id)
        conversation_id = response['conversation_id']
        
        print(f"\n🤖 BOT: {response['response']}")
        print(f"\n📊 Metadata:")
        print(f"   Intent: {response['intent']} ({response['intent_confidence']:.0%} confidence)")
        print(f"   Topics: {', '.join(response['topics'])}")
        print(f"   Template: {response['prompt_template_used']}")
        print(f"   Backend Actions: {len(response['backend_actions_triggered'])}")
    
    print(f"\n{'='*70}")
    print(f"✅ Conversation complete!")
    print(f"   Total turns: {response['turn_number']}")
    print(f"   Active topics: {', '.join(response['context_summary']['active_topics'])}")
