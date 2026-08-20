"""
Master Manufacturing Orchestrator
Central coordinator for all manufacturing intelligence systems

ARCHITECTURE:
1. Coordinates all subsystems (FANUC, REaaS, Producer Engine, etc.)
2. Routes requests to appropriate systems
3. Aggregates results from multiple systems
4. Handles complete manufacturing workflows
5. Provides unified interface

PARADIGM: Orchestra conductor - each system is an instrument,
orchestrator creates symphony of manufacturing intelligence
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging

# Import all major systems
try:
    from cms.fanuc_wave_controller import FanucWaveController
    FANUC_AVAILABLE = True
except ImportError:
    FANUC_AVAILABLE = False
    FanucWaveController = None

from cms.reverse_engineering_simulator import ReverseEngineeringSimulator, SyntheticProject
from cms.producer_effectiveness_engine import PartOptimizationBot, ProducerStatistics, EconomicTier
from cms.solidworks_simulation_validator import IntegratedOptimizationSystem
from cms.llm_context_system import RealTimeLLMAccessor
from cms.synthetic_data_generator import PartTemplateGenerator, SyntheticOperationDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

@dataclass
class ManufacturingRequest:
    """
    Unified manufacturing request
    
    Can be for: quotation, optimization, reverse engineering,
    simulation, or complete production package
    """
    request_id: str
    request_type: str  # 'quote', 'optimize', 'reaas', 'simulate', 'complete'
    
    # Part description
    part_description: str
    part_type: Optional[str] = None
    material: Optional[str] = None
    quantity: int = 1
    
    # Requirements
    tolerance_mm: Optional[float] = None
    surface_finish: Optional[str] = None
    complexity: Optional[str] = None
    
    # Constraints
    max_cost_per_part: Optional[float] = None
    required_delivery_date: Optional[str] = None
    urgency: str = 'standard'
    
    # Geometry (for validation)
    geometry: Optional[Dict] = None
    loading_conditions: Optional[Dict] = None
    
    # Additional context
    customer_id: Optional[str] = None
    notes: Optional[str] = None
    
    # Metadata
    created_at: str = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ManufacturingResponse:
    """
    Unified manufacturing response
    
    Contains results from all relevant systems
    """
    request_id: str
    status: str  # 'success', 'partial', 'error'
    
    # Results from each system
    llm_understanding: Optional[Dict] = None
    reaas_analysis: Optional[Dict] = None
    optimization_result: Optional[Dict] = None
    solidworks_validation: Optional[Dict] = None
    fanuc_gcode: Optional[Dict] = None
    synthetic_simulation: Optional[Dict] = None
    
    # Aggregated results
    recommended_producer: Optional[str] = None
    estimated_cost: Optional[float] = None
    estimated_time_minutes: Optional[float] = None
    predicted_lifetime_hours: Optional[float] = None
    quality_prediction: Optional[float] = None
    
    # Recommendations
    recommendations: List[str] = None
    warnings: List[str] = None
    
    # Metadata
    processing_time_seconds: float = 0.0
    systems_used: List[str] = None
    completed_at: str = None
    
    def __post_init__(self):
        if not self.completed_at:
            self.completed_at = datetime.now().isoformat()
        if self.recommendations is None:
            self.recommendations = []
        if self.warnings is None:
            self.warnings = []
        if self.systems_used is None:
            self.systems_used = []


# =============================================================================
# MASTER ORCHESTRATOR
# =============================================================================

class MasterManufacturingOrchestrator:
    """
    Central coordinator for all manufacturing intelligence systems
    
    RESPONSIBILITIES:
    1. Route requests to appropriate systems
    2. Coordinate multi-system workflows
    3. Aggregate results
    4. Handle errors gracefully
    5. Provide unified interface
    
    METAPHOR: Orchestra conductor - coordinates all instruments
    to create beautiful symphony of manufacturing intelligence
    """
    
    def __init__(self):
        """Initialize all subsystems"""
        logger.info("🎭 Initializing Master Manufacturing Orchestrator...")
        
        # Initialize subsystems
        try:
            self.llm_accessor = RealTimeLLMAccessor()
            logger.info("  ✅ LLM Context System initialized")
        except Exception as e:
            logger.warning(f"  ⚠️ LLM Context System failed: {e}")
            self.llm_accessor = None
        
        try:
            self.reaas_simulator = ReverseEngineeringSimulator()
            logger.info("  ✅ REaaS Simulator initialized")
        except Exception as e:
            logger.warning(f"  ⚠️ REaaS Simulator failed: {e}")
            self.reaas_simulator = None
        
        try:
            self.producer_engine = PartOptimizationBot()
            logger.info("  ✅ Producer Effectiveness Engine initialized")
        except Exception as e:
            logger.warning(f"  ⚠️ Producer Engine failed: {e}")
            self.producer_engine = None
        
        try:
            self.optimization_system = IntegratedOptimizationSystem()
            logger.info("  ✅ SolidWorks Integration initialized")
        except Exception as e:
            logger.warning(f"  ⚠️ SolidWorks Integration failed: {e}")
            self.optimization_system = None
        
        try:
            self.part_generator = PartTemplateGenerator()
            self.operation_generator = SyntheticOperationDataGenerator()
            logger.info("  ✅ Synthetic Data Generators initialized")
        except Exception as e:
            logger.warning(f"  ⚠️ Synthetic Data failed: {e}")
            self.part_generator = None
            self.operation_generator = None
        
        # FANUC controller requires hardware - initialized on demand
        self.fanuc_controller = None
        
        logger.info("🎭 Master Orchestrator ready!")
    
    # =========================================================================
    # MAIN PROCESSING METHOD
    # =========================================================================
    
    def process_request(self, request: ManufacturingRequest) -> ManufacturingResponse:
        """
        Process manufacturing request through appropriate systems
        
        Args:
            request: Manufacturing request
        
        Returns:
            Complete manufacturing response
        """
        start_time = datetime.now()
        logger.info(f"📥 Processing request: {request.request_id} ({request.request_type})")
        
        response = ManufacturingResponse(
            request_id=request.request_id,
            status='success'
        )
        
        try:
            # Route based on request type
            if request.request_type == 'quote':
                self._process_quote_request(request, response)
            
            elif request.request_type == 'optimize':
                self._process_optimization_request(request, response)
            
            elif request.request_type == 'reaas':
                self._process_reaas_request(request, response)
            
            elif request.request_type == 'simulate':
                self._process_simulation_request(request, response)
            
            elif request.request_type == 'complete':
                self._process_complete_workflow(request, response)
            
            else:
                response.status = 'error'
                response.warnings.append(f"Unknown request type: {request.request_type}")
        
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}")
            response.status = 'error'
            response.warnings.append(f"Processing error: {str(e)}")
        
        # Finalize response
        end_time = datetime.now()
        response.processing_time_seconds = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Request {request.request_id} completed in {response.processing_time_seconds:.2f}s")
        
        return response
    
    # =========================================================================
    # WORKFLOW PROCESSORS
    # =========================================================================
    
    def _process_quote_request(self, request: ManufacturingRequest, response: ManufacturingResponse):
        """Process instant quote request"""
        logger.info("  💰 Generating quote...")
        
        # Step 1: Use LLM to understand requirements
        if self.llm_accessor:
            try:
                llm_result = self.llm_accessor.query_with_context(
                    request.part_description,
                    verbose=False
                )
                response.llm_understanding = llm_result
                response.systems_used.append('LLM Context System')
            except Exception as e:
                logger.warning(f"  ⚠️ LLM failed: {e}")
        
        # Step 2: Optimize and estimate
        if self.producer_engine:
            try:
                optimization = self.producer_engine.optimize_complete_project(
                    part_description=request.part_description,
                    constraints={
                        'max_cost_per_part': request.max_cost_per_part
                    } if request.max_cost_per_part else None
                )
                
                response.optimization_result = optimization
                response.estimated_cost = optimization['optimal_specifications']['estimated_cost']
                response.estimated_time_minutes = optimization['optimal_specifications']['estimated_time']
                response.predicted_lifetime_hours = optimization['optimal_specifications']['lifetime_span_hours']
                response.recommended_producer = optimization.get('recommended_producer')
                response.recommendations = optimization.get('recommendations', [])
                response.systems_used.append('Producer Effectiveness Engine')
                
            except Exception as e:
                logger.warning(f"  ⚠️ Producer Engine failed: {e}")
    
    def _process_optimization_request(self, request: ManufacturingRequest, response: ManufacturingResponse):
        """Process optimization request with optional SolidWorks validation"""
        logger.info("  ⚙️ Optimizing specifications...")
        
        # First, optimize
        self._process_quote_request(request, response)
        
        # Then, validate with SolidWorks if geometry provided
        if request.geometry and request.loading_conditions and self.optimization_system:
            try:
                logger.info("  🔬 Validating with SolidWorks...")
                validation = self.optimization_system.optimize_and_validate(
                    part_description=request.part_description,
                    geometry=request.geometry,
                    loading_conditions=request.loading_conditions,
                    constraints={'max_cost_per_part': request.max_cost_per_part} if request.max_cost_per_part else None
                )
                
                response.solidworks_validation = validation
                
                # Update predictions with validated values
                if validation['validation_status'] == 'simulation_validated':
                    final_specs = validation['final_specs']
                    response.predicted_lifetime_hours = final_specs.get('lifetime_span_hours')
                    response.recommendations.append(
                        f"✅ SolidWorks validation complete - Safety factor: {final_specs.get('safety_factor', 'N/A')}"
                    )
                
                response.systems_used.append('SolidWorks Validation')
                
            except Exception as e:
                logger.warning(f"  ⚠️ SolidWorks validation failed: {e}")
    
    def _process_reaas_request(self, request: ManufacturingRequest, response: ManufacturingResponse):
        """Process reverse engineering request"""
        logger.info("  🔄 Reverse engineering part...")
        
        if not self.reaas_simulator:
            response.warnings.append("REaaS system not available")
            return
        
        try:
            # Generate reverse engineering project
            project = self.reaas_simulator.generate_project_context()
            
            # Simulate production
            telemetry = self.reaas_simulator.simulate_production_run(
                project,
                duration_seconds=min(project.estimated_cycle_time * 60, 600),  # Max 10 min simulation
                sample_rate_hz=1
            )
            
            response.reaas_analysis = {
                'project': asdict(project),
                'telemetry_samples': len(telemetry),
                'final_biochemical_state': {
                    'cortisol': telemetry[-1].cortisol,
                    'dopamine': telemetry[-1].dopamine,
                    'serotonin': telemetry[-1].serotonin
                },
                'final_tool_health': telemetry[-1].tool_health
            }
            
            response.estimated_cost = project.estimated_cycle_time * 1.5  # Simplified
            response.systems_used.append('REaaS Simulator')
            
            # Now optimize the replacement part
            if self.producer_engine:
                optimization = self.producer_engine.optimize_complete_project(
                    part_description=f"Replacement for {project.name}, material {project.material}"
                )
                response.optimization_result = optimization
                response.estimated_cost = optimization['optimal_specifications']['estimated_cost']
                response.systems_used.append('Producer Effectiveness Engine')
        
        except Exception as e:
            logger.warning(f"  ⚠️ REaaS failed: {e}")
            response.warnings.append(f"REaaS error: {str(e)}")
    
    def _process_simulation_request(self, request: ManufacturingRequest, response: ManufacturingResponse):
        """Process synthetic simulation request"""
        logger.info("  🎬 Running synthetic simulation...")
        
        if not self.part_generator or not self.operation_generator:
            response.warnings.append("Synthetic data generators not available")
            return
        
        try:
            # Generate synthetic part
            part = self.part_generator.generate_part(
                request.part_type or 'bearing_housing',
                vendor_requirements={'material': request.material} if request.material else None
            )
            
            # Generate synthetic operation data
            operation = part['operations'][0]
            stream = self.operation_generator.generate_operation_stream(
                operation=operation,
                material=part['material'],
                duration_minutes=min(part['estimated_cycle_time_minutes'], 10),
                sample_rate_hz=1,
                inject_failures=True
            )
            
            response.synthetic_simulation = {
                'part': part,
                'operation_samples': len(stream),
                'final_state': stream[-1] if stream else None
            }
            
            response.estimated_cost = part['estimated_cost_usd']
            response.estimated_time_minutes = part['estimated_cycle_time_minutes']
            response.systems_used.append('Synthetic Data Generator')
        
        except Exception as e:
            logger.warning(f"  ⚠️ Synthetic simulation failed: {e}")
            response.warnings.append(f"Simulation error: {str(e)}")
    
    def _process_complete_workflow(self, request: ManufacturingRequest, response: ManufacturingResponse):
        """
        Process complete manufacturing workflow
        
        1. LLM understands requirements
        2. REaaS if reverse engineering needed
        3. Optimize specifications
        4. Validate with SolidWorks
        5. Generate G-Code (FANUC)
        6. Simulate operation
        """
        logger.info("  🎯 Processing COMPLETE workflow...")
        
        # Step 1: LLM Understanding
        logger.info("    1️⃣ LLM understanding requirements...")
        if self.llm_accessor:
            try:
                llm_result = self.llm_accessor.query_with_context(
                    request.part_description,
                    verbose=False
                )
                response.llm_understanding = llm_result
                response.systems_used.append('LLM Context System')
            except Exception as e:
                logger.warning(f"    ⚠️ LLM failed: {e}")
        
        # Step 2: Check if reverse engineering needed
        if 'replacement' in request.part_description.lower() or 'worn' in request.part_description.lower():
            logger.info("    2️⃣ Reverse engineering detected...")
            self._process_reaas_request(request, response)
        
        # Step 3: Optimization
        logger.info("    3️⃣ Optimizing specifications...")
        self._process_optimization_request(request, response)
        
        # Step 4: Synthetic simulation
        logger.info("    4️⃣ Running synthetic simulation...")
        self._process_simulation_request(request, response)
        
        # Step 5: Generate final recommendations
        self._generate_final_recommendations(request, response)
    
    def _generate_final_recommendations(self, request: ManufacturingRequest, response: ManufacturingResponse):
        """Generate final aggregated recommendations"""
        
        # Cost recommendations
        if response.estimated_cost:
            if request.max_cost_per_part and response.estimated_cost > request.max_cost_per_part:
                response.recommendations.append(
                    f"⚠️ Estimated cost ${response.estimated_cost:.2f} exceeds budget ${request.max_cost_per_part:.2f}"
                )
            else:
                response.recommendations.append(
                    f"✅ Cost estimate ${response.estimated_cost:.2f} within budget"
                )
        
        # Quality recommendations
        if response.quality_prediction and response.quality_prediction < 0.90:
            response.recommendations.append(
                "⚠️ Quality prediction below 90% - consider relaxing tolerances"
            )
        
        # Lifetime recommendations
        if response.predicted_lifetime_hours and response.predicted_lifetime_hours < 10000:
            response.recommendations.append(
                f"⚠️ Predicted lifetime {response.predicted_lifetime_hours:.0f} hours is low - consider design improvements"
            )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_system_status(self) -> Dict:
        """Get status of all subsystems"""
        return {
            'llm_context': self.llm_accessor is not None,
            'reaas_simulator': self.reaas_simulator is not None,
            'producer_engine': self.producer_engine is not None,
            'solidworks_integration': self.optimization_system is not None,
            'synthetic_data': self.part_generator is not None and self.operation_generator is not None,
            'fanuc_controller': self.fanuc_controller is not None
        }
    
    def get_supported_request_types(self) -> List[str]:
        """Get list of supported request types"""
        return ['quote', 'optimize', 'reaas', 'simulate', 'complete']


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Master Manufacturing Orchestrator - Demo")
    print("=" * 70)
    
    # Create orchestrator
    orchestrator = MasterManufacturingOrchestrator()
    
    # Show system status
    status = orchestrator.get_system_status()
    print("\n📊 System Status:")
    for system, available in status.items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {system}")
    
    # Example request 1: Simple quote
    print("\n" + "=" * 70)
    print("REQUEST 1: Simple Quote")
    print("=" * 70)
    
    request1 = ManufacturingRequest(
        request_id="REQ-001",
        request_type="quote",
        part_description="100 aluminum brackets, ±0.05mm tolerance",
        quantity=100,
        max_cost_per_part=100.0
    )
    
    response1 = orchestrator.process_request(request1)
    
    print(f"\n✅ Response Status: {response1.status}")
    print(f"💰 Estimated Cost: ${response1.estimated_cost:.2f}/part" if response1.estimated_cost else "Cost: N/A")
    print(f"⏱️  Estimated Time: {response1.estimated_time_minutes:.1f} min/part" if response1.estimated_time_minutes else "Time: N/A")
    print(f"🏭 Recommended Producer: {response1.recommended_producer}" if response1.recommended_producer else "Producer: N/A")
    print(f"⚙️  Systems Used: {', '.join(response1.systems_used)}")
    
    if response1.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in response1.recommendations:
            print(f"  - {rec}")
    
    # Example request 2: Complete workflow
    print("\n" + "=" * 70)
    print("REQUEST 2: Complete Workflow")
    print("=" * 70)
    
    request2 = ManufacturingRequest(
        request_id="REQ-002",
        request_type="complete",
        part_description="Replacement titanium gear for marine pump",
        part_type="gear",
        material="Titanium6Al4V",
        quantity=10
    )
    
    response2 = orchestrator.process_request(request2)
    
    print(f"\n✅ Response Status: {response2.status}")
    print(f"💰 Estimated Cost: ${response2.estimated_cost:.2f}/part" if response2.estimated_cost else "Cost: N/A")
    print(f"⏱️  Estimated Time: {response2.estimated_time_minutes:.1f} min/part" if response2.estimated_time_minutes else "Time: N/A")
    print(f"🔧 Predicted Lifetime: {response2.predicted_lifetime_hours:.0f} hours" if response2.predicted_lifetime_hours else "Lifetime: N/A")
    print(f"⚙️  Systems Used: {', '.join(response2.systems_used)}")
    print(f"⏱️  Processing Time: {response2.processing_time_seconds:.2f}s")
    
    if response2.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in response2.recommendations:
            print(f"  - {rec}")
