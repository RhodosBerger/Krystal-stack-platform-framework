"""
Multi-Bot Consulting System
Specialized AI bots for different manufacturing domains

BOT ROSTER:
1. CNC Manufacturing Expert - Machining, tooling, parameters
2. Software Engineering Bot - Code, algorithms, integration
3. Quality Control Specialist - Inspection, tolerances, standards
4. Manufacturing Economist - Cost optimization, ROI analysis
5. Materials Science Advisor - Material selection, properties
6. Maintenance Engineer - Preventive maintenance, diagnostics
7. Process Optimization Bot - Efficiency, workflow improvement
8. CAD/CAM Specialist - Design, G-Code generation

ARCHITECTURE:
- Each bot has specialized knowledge domain
- Bot Coordinator routes questions to appropriate expert(s)
- Bots can collaborate on complex questions
- LLM-powered with domain-specific context
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from cms.llm_context_system import RealTimeLLMAccessor, SemanticMeaning, MeaningCategory


# =============================================================================
# BOT EXPERTISE DOMAINS
# =============================================================================

class ExpertiseDomain(Enum):
    """Expertise domains for specialized bots"""
    CNC_MANUFACTURING = "cnc_manufacturing"
    SOFTWARE_ENGINEERING = "software_engineering"
    QUALITY_CONTROL = "quality_control"
    ECONOMICS = "economics"
    MATERIALS_SCIENCE = "materials_science"
    MAINTENANCE = "maintenance"
    PROCESS_OPTIMIZATION = "process_optimization"
    CAD_CAM = "cad_cam"


@dataclass
class BotProfile:
    """Profile for a specialized bot"""
    bot_id: str
    name: str
    expertise: ExpertiseDomain
    specializations: List[str]
    knowledge_base: Dict[str, Any]
    personality: str  # How the bot communicates
    consultation_count: int = 0
    success_rate: float = 1.0


# =============================================================================
# SPECIALIZED BOTS
# =============================================================================

class ManufacturingBot:
    """
    Base class for specialized manufacturing bots
    
    Each bot has:
    - Domain expertise
    - Specialized knowledge
    - Consultation methods
    - LLM integration
    """
    
    def __init__(self, profile: BotProfile):
        """Initialize bot with profile"""
        self.profile = profile
        self.llm_accessor = RealTimeLLMAccessor()
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize bot-specific knowledge - override in subclasses"""
        pass
    
    def can_answer(self, question: str) -> float:
        """
        Determine if bot can answer question
        
        Args:
            question: User's question
        
        Returns:
            Confidence score 0.0-1.0
        """
        question_lower = question.lower()
        score = 0.0
        
        # Check for specialization keywords
        for specialization in self.profile.specializations:
            if specialization.lower() in question_lower:
                score += 0.3
        
        # Check for domain keywords
        domain_keywords = self._get_domain_keywords()
        for keyword in domain_keywords:
            if keyword.lower() in question_lower:
                score += 0.1
        
        return min(1.0, score)
    
    def _get_domain_keywords(self) -> List[str]:
        """Get domain-specific keywords - override in subclasses"""
        return []
    
    def consult(self, question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Provide consultation for question
        
        Args:
            question: User's question
            context: Additional context
        
        Returns:
            Consultation response
        """
        self.profile.consultation_count += 1
        
        # Get domain-specific context
        domain_context = self._build_domain_context(question, context)
        
        # Generate response
        response = self._generate_response(question, domain_context)
        
        return {
            'bot_id': self.profile.bot_id,
            'bot_name': self.profile.name,
            'expertise': self.profile.expertise.value,
            'response': response,
            'confidence': self.can_answer(question),
            'timestamp': datetime.now().isoformat()
        }
    
    def _build_domain_context(self, question: str, context: Optional[Dict]) -> str:
        """Build domain-specific context - override in subclasses"""
        return f"Question about {self.profile.expertise.value}: {question}"
    
    def _generate_response(self, question: str, domain_context: str) -> str:
        """Generate response using LLM with domain context"""
        # Simulated response (would use actual LLM in production)
        return f"[{self.profile.name}]: Based on my expertise in {self.profile.expertise.value}, I recommend..."


# =============================================================================
# 1. CNC MANUFACTURING EXPERT
# =============================================================================

class CNCManufacturingBot(ManufacturingBot):
    """
    Expert in CNC machining operations
    
    SPECIALIZATIONS:
    - Machining parameters (speeds, feeds, DOC)
    - Tool selection and toolpaths
    - Machine capabilities
    - Fixture design
    - Chip control
    """
    
    def _initialize_knowledge_base(self):
        """Initialize CNC knowledge"""
        self.profile.knowledge_base = {
            'materials': {
                'Aluminum6061': {
                    'cutting_speed_range': [200, 400],
                    'feed_per_tooth': [0.1, 0.3],
                    'recommended_tools': ['Carbide end mill', 'HSS'],
                    'coolant': 'Flood or mist'
                },
                'Steel4140': {
                    'cutting_speed_range': [80, 150],
                    'feed_per_tooth': [0.08, 0.15],
                    'recommended_tools': ['Coated carbide', 'CBN'],
                    'coolant': 'Flood required'
                },
                'Titanium6Al4V': {
                    'cutting_speed_range': [40, 80],
                    'feed_per_tooth': [0.05, 0.12],
                    'recommended_tools': ['Solid carbide low helix', 'PCD'],
                    'coolant': 'High pressure flood'
                }
            },
            'operations': {
                'roughing': {
                    'stepover': '40-60% of diameter',
                    'depth_of_cut': '1-3x diameter',
                    'priority': 'Material removal rate'
                },
                'finishing': {
                    'stepover': '5-15% of diameter',
                    'depth_of_cut': '0.1-0.5mm',
                    'priority': 'Surface finish'
                }
            }
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """CNC-specific keywords"""
        return [
            'machining', 'cutting', 'milling', 'turning', 'drilling',
            'speeds', 'feeds', 'rpm', 'tool', 'toolpath', 'fixture',
            'spindle', 'coolant', 'chip', 'surface finish', 'roughing',
            'finishing', 'depth of cut', 'stepover'
        ]
    
    def _build_domain_context(self, question: str, context: Optional[Dict]) -> str:
        """Build CNC-specific context"""
        material = context.get('material') if context else None
        
        context_parts = [f"CNC Manufacturing Expert consulting on: {question}"]
        
        if material and material in self.profile.knowledge_base['materials']:
            mat_data = self.profile.knowledge_base['materials'][material]
            context_parts.append(f"\nMaterial: {material}")
            context_parts.append(f"Recommended cutting speed: {mat_data['cutting_speed_range']} m/min")
            context_parts.append(f"Feed per tooth: {mat_data['feed_per_tooth']} mm")
            context_parts.append(f"Recommended tools: {', '.join(mat_data['recommended_tools'])}")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, question: str, domain_context: str) -> str:
        """Generate CNC-specific response"""
        # Extract material if mentioned
        materials = ['aluminum', 'steel', 'titanium', 'stainless']
        detected_material = None
        
        for mat in materials:
            if mat in question.lower():
                detected_material = mat
                break
        
        response_parts = [f"**CNC Manufacturing Expert - {self.profile.name}**\n"]
        
        if detected_material:
            response_parts.append(f"For {detected_material} machining, I recommend:\n")
            
            if 'aluminum' in detected_material:
                response_parts.append("- **Cutting Speed**: 250-350 m/min for roughing, up to 400 m/min for finishing")
                response_parts.append("- **Feed Rate**: 0.15-0.25 mm/tooth")
                response_parts.append("- **Tool**: 3-4 flute carbide end mill with TiAlN coating")
                response_parts.append("- **Coolant**: Mist or flood for deep pockets")
                response_parts.append("- **DOC**: Up to 2x diameter for roughing, 0.2-0.5mm for finishing")
            
            elif 'titanium' in detected_material:
                response_parts.append("- **Cutting Speed**: 50-70 m/min (titanium machines SLOW)")
                response_parts.append("- **Feed Rate**: 0.08-0.12 mm/tooth")
                response_parts.append("- **Tool**: Solid carbide with low helix angle (30°)")
                response_parts.append("- **Coolant**: High-pressure flood coolant REQUIRED")
                response_parts.append("- **Warning**: Use sharp tools, titanium work-hardens quickly")
        
        else:
            response_parts.append("Please specify the material you're machining for detailed recommendations.")
        
        return "\n".join(response_parts)


# =============================================================================
# 2. SOFTWARE ENGINEERING BOT
# =============================================================================

class SoftwareEngineeringBot(ManufacturingBot):
    """
    Expert in manufacturing software development
    
    SPECIALIZATIONS:
    - API integration (FANUC FOCAS, SolidWorks API)
    - Database design
    - Real-time systems
    - IoT/sensor integration
    - Algorithm optimization
    """
    
    def _initialize_knowledge_base(self):
        """Initialize software knowledge"""
        self.profile.knowledge_base = {
            'api_integrations': {
                'FANUC_FOCAS': {
                    'library': 'pyfocas',
                    'functions': ['cnc_rdspload', 'cnc_rdposition', 'cnc_rdsverr'],
                    'use_cases': ['Real-time monitoring', 'Adaptive control']
                },
                'SolidWorks_API': {
                    'library': 'win32com',
                    'functions': ['NewDocument', 'RunSimulation', 'ExportSTL'],
                    'use_cases': ['Automated design', 'FEA validation']
                }
            },
            'architectures': {
                'microservices': 'Separate services for each manufacturing system',
                'event_driven': 'Sensor data triggers automated responses',
                'real_time': 'RTOS for critical CNC control loops'
            }
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """Software-specific keywords"""
        return [
            'api', 'integration', 'database', 'code', 'algorithm',
            'software', 'programming', 'python', 'rest', 'websocket',
            'real-time', 'iot', 'sensor', 'mqtt', 'modbus',
            'performance', 'optimization', 'architecture'
        ]
    
    def _generate_response(self, question: str, domain_context: str) -> str:
        """Generate software-specific response"""
        response_parts = [f"**Software Engineering Expert - {self.profile.name}**\n"]
        
        if 'api' in question.lower() or 'integration' in question.lower():
            response_parts.append("For manufacturing API integration, I recommend:\n")
            response_parts.append("- **Architecture**: RESTful API with FastAPI for performance")
            response_parts.append("- **Real-time**: WebSocket for sensor streaming")
            response_parts.append("- **Database**: PostgreSQL for relational data, Redis for caching")
            response_parts.append("- **Authentication**: JWT tokens with role-based access")
            response_parts.append("- **Example Code**:")
            response_parts.append("```python")
            response_parts.append("from fastapi import FastAPI, WebSocket")
            response_parts.append("app = FastAPI()")
            response_parts.append("@app.websocket('/ws/sensors')")
            response_parts.append("async def stream_sensors(websocket: WebSocket):")
            response_parts.append("    await websocket.accept()")
            response_parts.append("    while True:")
            response_parts.append("        data = read_sensors()")
            response_parts.append("        await websocket.send_json(data)")
            response_parts.append("```")
        
        elif 'database' in question.lower():
            response_parts.append("For manufacturing database design:\n")
            response_parts.append("- **Structure**: Normalized schema with proper foreign keys")
            response_parts.append("- **Partitioning**: Time-series data partitioned by date")
            response_parts.append("- **Indexing**: Composite indexes on (machine_id, timestamp)")
            response_parts.append("- **Archival**: Move old telemetry to cold storage after 90 days")
        
        else:
            response_parts.append("Please specify: API integration, database design, or performance optimization?")
        
        return "\n".join(response_parts)


# =============================================================================
# 3. QUALITY CONTROL SPECIALIST
# =============================================================================

class QualityControlBot(ManufacturingBot):
    """
    Expert in quality assurance and inspection
    
    SPECIALIZATIONS:
    - Dimensional inspection
    - Statistical Process Control (SPC)
    - ISO/AS standards
    - CMM programming
    - Six Sigma methods
    """
    
    def _initialize_knowledge_base(self):
        """Initialize quality knowledge"""
        self.profile.knowledge_base = {
            'tolerance_standards': {
                'ISO2768-m': 'Medium tolerance class',
                'ISO2768-f': 'Fine tolerance class',
                'ISO2768-v': 'Very fine tolerance class'
            },
            'inspection_methods': {
                'CMM': 'Coordinate Measuring Machine - high precision',
                'Optical': 'Vision system - fast, non-contact',
                'Manual': 'Calipers, micrometers - low cost'
            },
            'spc_limits': {
                'Cpk': {
                    'min_acceptable': 1.33,
                    'good': 1.67,
                    'excellent': 2.0
                }
            }
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """Quality-specific keywords"""
        return [
            'quality', 'inspection', 'tolerance', 'dimension', 'accuracy',
            'precision', 'defect', 'reject', 'cmm', 'measurement',
            'spc', 'cpk', 'six sigma', 'iso', 'standard', 'calibration'
        ]
    
    def _generate_response(self, question: str, domain_context: str) -> str:
        """Generate quality-specific response"""
        response_parts = [f"**Quality Control Specialist - {self.profile.name}**\n"]
        
        if 'tolerance' in question.lower():
            response_parts.append("Regarding tolerances:\n")
            response_parts.append("- **ISO 2768-m (Medium)**: ±0.1mm for dimensions 3-6mm")
            response_parts.append("- **ISO 2768-f (Fine)**: ±0.05mm for dimensions 3-6mm")
            response_parts.append("- **Tighter tolerances**: Require precision machining, increase cost 2-3x")
            response_parts.append("- **Inspection**: Use CMM for ±0.005mm, optical for ±0.01mm")
        
        elif 'defect' in question.lower() or 'reject' in question.lower():
            response_parts.append("For defect analysis:\n")
            response_parts.append("1. **Categorize**: Dimensional, surface finish, or material defect?")
            response_parts.append("2. **Root Cause**: Tool wear, machine drift, or material variation?")
            response_parts.append("3. **SPC**: Plot control charts to identify trends")
            response_parts.append("4. **Cpk Target**: Aim for Cpk > 1.33 minimum, >1.67 preferred")
        
        else:
            response_parts.append("I can help with tolerance selection, inspection methods, or SPC analysis.")
        
        return "\n".join(response_parts)


# =============================================================================
# 4. MANUFACTURING ECONOMIST
# =============================================================================

class ManufacturingEconomistBot(ManufacturingBot):
    """
    Expert in manufacturing economics and cost optimization
    
    SPECIALIZATIONS:
    - Cost estimation
    - ROI analysis
    - Make vs. buy decisions
    - Capacity planning
    - Pricing strategies
    """
    
    def _initialize_knowledge_base(self):
        """Initialize economics knowledge"""
        self.profile.knowledge_base = {
            'cost_factors': {
                'material_cost': 'Raw material + scrap rate',
                'labor_cost': 'Setup time + cycle time',
                'overhead': 'Machine depreciation, utilities, rent',
                'tooling': 'Tool cost / tool life'
            },
            'pricing_models': {
                'cost_plus': 'Total cost + markup %',
                'market_based': 'Competitive market rate',
                'value_based': 'Customer perceived value'
            }
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """Economics-specific keywords"""
        return [
            'cost', 'price', 'profit', 'roi', 'economics', 'budget',
            'savings', 'expensive', 'cheap', 'value', 'margin',
            'make or buy', 'capacity', 'volume', 'break-even'
        ]
    
    def _generate_response(self, question: str, domain_context: str) -> str:
        """Generate economics-specific response"""
        response_parts = [f"**Manufacturing Economist - {self.profile.name}**\n"]
        
        if 'cost' in question.lower():
            response_parts.append("Cost breakdown for CNC parts:\n")
            response_parts.append("- **Material**: 15-30% of total")
            response_parts.append("- **Labor**: 25-40% (setup + operation)")
            response_parts.append("- **Overhead**: 30-45% (machine, facility)")
            response_parts.append(" - **Tooling**: 5-15% (depends on volume)")
            response_parts.append("\n**Optimization Strategies**:")
            response_parts.append("- Increase batch size to amortize setup costs")
            response_parts.append("- Use standard tools instead of custom")
            response_parts.append("- Consider offshore for high-volume, low-complexity parts")
        
        elif 'roi' in question.lower():
            response_parts.append("ROI Analysis for manufacturing investments:\n")
            response_parts.append("- **Payback Period**: Investment / Annual Savings")
            response_parts.append("- **Break-even**: Fixed Costs / (Price - Variable Cost)")
            response_parts.append("- **NPV**: Consider 3-5 year horizon with discount rate")
        
        return "\n".join(response_parts)


# =============================================================================
# 5. MATERIALS SCIENCE ADVISOR
# =============================================================================

class MaterialsScienceBot(ManufacturingBot):
    """
    Expert in materials selection and properties
    
    SPECIALIZATIONS:
    - Material properties
    - Heat treatment
    - Corrosion resistance
    - Strength-to-weight ratios
    - Material substitution
    """
    
    def _initialize_knowledge_base(self):
        """Initialize materials knowledge"""
        self.profile.knowledge_base = {
            'materials': {
                'Aluminum6061': {
                    'density': 2.7,
                    'tensile_strength': 310,
                    'hardness_hb': 95,
                    'applications': ['Aircraft', 'Automotive', 'General purpose'],
                    'cost_factor': 1.0
                },
                'Steel4140': {
                    'density': 7.85,
                    'tensile_strength': 655,
                    'hardness_hb': 250,
                    'applications': ['Gears', 'Shafts', 'High strength'],
                    'cost_factor': 0.6
                },
                'Titanium6Al4V': {
                    'density': 4.43,
                    'tensile_strength': 900,
                    'hardness_hb': 330,
                    'applications': ['Aerospace', 'Medical', 'High performance'],
                    'cost_factor': 8.0
                }
            }
        }
    
    def _get_domain_keywords(self) -> List[str]:
        """Materials-specific keywords"""
        return [
            'material', 'aluminum', 'steel', 'titanium', 'strength',
            'hardness', 'density', 'weight', 'corrosion', 'heat treat',
            'properties', 'alloy', 'metal', 'substitute'
        ]
    
    def _generate_response(self, question: str, domain_context: str) -> str:
        """Generate materials-specific response"""
        response_parts = [f"**Materials Science Advisor - {self.profile.name}**\n"]
        
        if 'substitute' in question.lower() or 'alternative' in question.lower():
            response_parts.append("Material substitution analysis:\n")
            response_parts.append("**Aluminum → Steel**")
            response_parts.append("- Pros: 3x stronger, 1/2 cost")
            response_parts.append("- Cons: 3x heavier, harder to machine")
            response_parts.append("\n**Steel → Titanium**")
            response_parts.append("- Pros: 40% lighter, corrosion resistant")
            response_parts.append("- Cons: 13x more expensive, difficult to machine")
        
        elif 'strength' in question.lower():
            response_parts.append("Strength comparison (Tensile MPa):\n")
            response_parts.append("- Aluminum 6061: 310 MPa")
            response_parts.append("- Steel 4140: 655 MPa (2.1x aluminum)")
            response_parts.append("- Titanium Ti-6Al-4V: 900 MPa (2.9x aluminum)")
            response_parts.append("\n**Strength-to-Weight**:")
            response_parts.append("- Aluminum: 310/2.7 = 115")
            response_parts.append("- Steel: 655/7.85 = 83")
            response_parts.append("- Titanium: 900/4.43 = 203 (BEST)")
        
        return "\n".join(response_parts)


# =============================================================================
# BOT COORDINATOR
# =============================================================================

class BotCoordinator:
    """
    Coordinates multiple specialized bots
    
    RESPONSIBILITIES:
    - Route questions to appropriate bot(s)
    - Handle multi-bot collaborations
    - Aggregate responses
    - Track bot performance
    """
    
    def __init__(self):
        """Initialize bot coordinator"""
        self.bots: Dict[str, ManufacturingBot] = {}
        self._initialize_bots()
    
    def _initialize_bots(self):
        """Initialize all specialized bots"""
        
        # 1. CNC Manufacturing Expert
        cnc_bot = CNCManufacturingBot(BotProfile(
            bot_id='cnc_001',
            name='MachineMaster',
            expertise=ExpertiseDomain.CNC_MANUFACTURING,
            specializations=[
                'speeds and feeds', 'tooling', 'toolpaths',
                'machining parameters', 'fixtures', 'coolant'
            ],
            knowledge_base={},
            personality='Direct and practical, focuses on proven techniques'
        ))
        self.bots['cnc_manufacturing'] = cnc_bot
        
        # 2. Software Engineering Bot
        software_bot = SoftwareEngineeringBot(BotProfile(
            bot_id='soft_001',
            name='CodeCraft',
            expertise=ExpertiseDomain.SOFTWARE_ENGINEERING,
            specializations=[
                'api integration', 'databases', 'real-time systems',
                'IoT', 'algorithms', 'architecture'
            ],
            knowledge_base={},
            personality='Technical and precise, provides code examples'
        ))
        self.bots['software'] = software_bot
        
        # 3. Quality Control Specialist
        quality_bot = QualityControlBot(BotProfile(
            bot_id='qc_001',
            name='PrecisionGuard',
            expertise=ExpertiseDomain.QUALITY_CONTROL,
            specializations=[
                'tolerances', 'inspection', 'SPC', 'CMM',
                'ISO standards', 'six sigma'
            ],
            knowledge_base={},
            personality='Meticulous and standards-focused'
        ))
        self.bots['quality'] = quality_bot
        
        # 4. Manufacturing Economist
        econ_bot = ManufacturingEconomistBot(BotProfile(
            bot_id='econ_001',
            name='CostOptimizer',
            expertise=ExpertiseDomain.ECONOMICS,
            specializations=[
                'cost estimation', 'ROI', 'pricing', 'make vs buy',
                'capacity planning', 'profitability'
            ],
            knowledge_base={},
            personality='Data-driven and ROI-focused'
        ))
        self.bots['economics'] = econ_bot
        
        # 5. Materials Science Advisor
        materials_bot = MaterialsScienceBot(BotProfile(
            bot_id='mat_001',
            name='MaterialExpert',
            expertise=ExpertiseDomain.MATERIALS_SCIENCE,
            specializations=[
                'material properties', 'selection', 'heat treatment',
                'corrosion', 'substitution'
            ],
            knowledge_base={},
            personality='Scientific and analytical'
        ))
        self.bots['materials'] = materials_bot
    
    def route_question(self, question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Route question to appropriate bot(s)
        
        Args:
            question: User's question
            context: Additional context
        
        Returns:
            Consultation response
        """
        # Score all bots
        bot_scores = {}
        for bot_key, bot in self.bots.items():
            score = bot.can_answer(question)
            if score > 0.1:  # Only consider bots with >10% confidence
                bot_scores[bot_key] = score
        
        if not bot_scores:
            return {
                'status': 'no_expert_found',
                'message': 'No specialized bot found for this question. Please rephrase or provide more context.',
                'available_bots': list(self.bots.keys())
            }
        
        # Get top bot
        top_bot_key = max(bot_scores, key=bot_scores.get)
        top_bot = self.bots[top_bot_key]
        
        # Get consultation
        consultation = top_bot.consult(question, context)
        
        # Check if multiple bots should collaborate (score > 0.5)
        collaborating_bots = [k for k, s in bot_scores.items() if s > 0.5 and k != top_bot_key]
        
        collaboration_responses = []
        if collaborating_bots:
            for bot_key in collaborating_bots[:2]:  # Max 2 additional bots
                collab_response = self.bots[bot_key].consult(question, context)
                collaboration_responses.append(collab_response)
        
        return {
            'status': 'success',
            'primary_consultant': consultation,
            'collaborating_consultants': collaboration_responses,
            'bot_scores': bot_scores,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_bot_roster(self) -> List[Dict]:
        """Get list of all available bots"""
        return [
            {
                'bot_id': bot.profile.bot_id,
                'name': bot.profile.name,
                'expertise': bot.profile.expertise.value,
                'specializations': bot.profile.specializations,
                'consultations_given': bot.profile.consultation_count
            }
            for bot in self.bots.values()
        ]


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Bot Manufacturing Consulting System")
    print("=" * 70)
    
    # Create coordinator
    coordinator = BotCoordinator()
    
    # Show bot roster
    print("\n🤖 Available Consultants:")
    for bot_info in coordinator.get_bot_roster():
        print(f"\n  {bot_info['name']} ({bot_info['expertise']})")
        print(f"    Specializations: {', '.join(bot_info['specializations'][:3])}")
    
    # Example consultations
    questions = [
        "What cutting speed should I use for aluminum?",
        "How do I integrate with the FANUC API?",
        "What tolerance should I specify for a bearing housing?",
        "How can I reduce manufacturing costs?",
        "Should I use steel or titanium for a lightweight shaft?"
    ]
    
    print("\n" + "=" * 70)
    print("Example Consultations:")
    print("=" * 70)
    
    for question in questions:
        print(f"\n❓ QUESTION: {question}")
        
        result = coordinator.route_question(question)
        
        if result['status'] == 'success':
            primary = result['primary_consultant']
            print(f"\n✅ PRIMARY CONSULTANT: {primary['bot_name']}")
            print(f"   Confidence: {primary['confidence']*100:.0f}%")
            print(f"\n{primary['response']}")
            
            if result['collaborating_consultants']:
                print(f"\n💡 Additional insights from:")
                for collab in result['collaborating_consultants']:
                    print(f"   - {collab['bot_name']}")
        else:
            print(f"\n❌ {result['message']}")
    
        print("\n" + "-" * 70)
