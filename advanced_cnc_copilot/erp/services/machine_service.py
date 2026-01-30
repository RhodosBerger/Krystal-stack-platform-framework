"""
Service Layer - Business Logic Implementation
"""

from typing import Dict, Any, List
from django.utils import timezone
from datetime import timedelta

class BaseService:
    """Base service with common operations"""
    
    def __init__(self, repository):
        self.repository = repository
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate data - override in subclass"""
        return True, []
    
    def execute(self, command: str, **kwargs) -> Any:
        """Execute command pattern"""
        method = getattr(self, f"_{command}", None)
        if method:
            return method(**kwargs)
        raise NotImplementedError(f"Command {command} not implemented")


class MachineService(BaseService):
    """Machine business logic"""
    
    def __init__(self):
        from erp.repositories.base_repository import MachineRepository
        super().__init__(MachineRepository())
    
    def get_machine_status(self, machine_id: int) -> Dict[str, Any]:
        """Get comprehensive machine status"""
        machine = self.repository.get_by_id(machine_id)
        if not machine:
            return {'status': 'NOT_FOUND'}
        
        # Get latest telemetry
        from erp.repositories.base_repository import TelemetryRepository
        telemetry_repo = TelemetryRepository()
        latest_telemetry = telemetry_repo.get_latest_for_machine(machine_id, limit=1)
        
        # Calculate OEE
        oee = machine.get_current_oee()
        
        return {
            'machine': {
                'id': machine.id,
                'name': machine.name,
                'controller': machine.controller_type,
                'is_active': machine.is_active
            },
            'telemetry': latest_telemetry[0] if latest_telemetry else None,
            'oee': oee,
            'status': 'RUNNING' if machine.is_active else 'STOPPED'
        }
    
    def register_machine(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Register new machine"""
        # Validate
        is_valid, errors = self.validate_machine_data(data)
        if not is_valid:
            return {'success': False, 'errors': errors}
        
        # Create
        machine = self.repository.create(**data)
        
        return {
            'success': True,
            'machine_id': machine.id,
            'message': f'Machine {machine.name} registered successfully'
        }
    
    def validate_machine_data(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate machine registration data"""
        errors = []
        
        if not data.get('name'):
            errors.append('Machine name is required')
        
        if not data.get('ip_address'):
            errors.append('IP address is required')
        
        # Check if IP already exists
        if data.get('ip_address'):
            existing = self.repository.get_by_ip(data['ip_address'])
            if existing:
                errors.append(f'Machine with IP {data["ip_address"]} already exists')
        
        return (len(errors) == 0, errors)


class TelemetryService(BaseService):
    """Telemetry processing service"""
    
    def __init__(self):
        from erp.repositories.base_repository import TelemetryRepository
        super().__init__(TelemetryRepository())
    
    def process_telemetry_batch(self, machine_id: int, telemetry_data: List[Dict]) -> Dict:
        """Process batch telemetry data"""
        # Enrich with dopamine/signal
        from cms.dopamine_engine import DopamineEngine
        from cms.signaling_system import SignalingSystem
        
        dopamine_engine = DopamineEngine()
        semaphore = SignalingSystem()
        
        enriched_data = []
        for data in telemetry_data:
            # Calculate dopamine
            dopamine = dopamine_engine.calculate_reward(
                load=data.get('load', 0),
                vibration=data.get('vibration_z', 0),
                temperature=data.get('spindle_temp', 0)
            )
            
            # Determine signal
            signal = semaphore.evaluate({
                'load': data.get('load', 0),
                'vibration': data.get('vibration_z', 0)
            })
            
            enriched_data.append({
                **data,
                'machine_id': machine_id,
                'dopamine': dopamine,
                'cortisol': 100 - dopamine,
                'signal': signal,
                'timestamp': timezone.now()
            })
        
        # Bulk create
        created = self.repository.bulk_create(enriched_data)
        
        return {
            'success': True,
            'count': len(created),
            'latest_signal': enriched_data[-1]['signal'] if enriched_data else None
        }
    
    def get_telemetry_summary(self, machine_id: int, hours: int = 24) -> Dict:
        """Get telemetry summary for time period"""
        start_time = timezone.now() - timedelta(hours=hours)
        end_time = timezone.now()
        
        telemetry = self.repository.get_by_timerange(machine_id, start_time, end_time)
        
        if not telemetry:
            return {'message': 'No data available'}
        
        # Calculate statistics
        loads = [t.load for t in telemetry]
        rpms = [t.rpm for t in telemetry]
        
        return {
            'period': f'Last {hours} hours',
            'data_points': len(telemetry),
            'load': {
                'avg': sum(loads) / len(loads),
                'max': max(loads),
                'min': min(loads)
            },
            'rpm': {
                'avg': sum(rpms) / len(rpms),
                'max': max(rpms),
                'min': min(rpms)
            },
            'signals': {
                'green': self.repository.get_signal_count(machine_id, 'GREEN'),
                'amber': self.repository.get_signal_count(machine_id, 'AMBER'),
                'red': self.repository.get_signal_count(machine_id, 'RED')
            }
        }


class DopamineService(BaseService):
    """Dopamine engine integration service"""
    
    def __init__(self):
        super().__init__(None)  # No repository needed
        from cms.dopamine_engine import DopamineEngine
        self.engine = DopamineEngine()
    
    def evaluate_conditions(self, metrics: Dict[str, float]) -> Dict:
        """Evaluate current conditions and return dopamine score"""
        dopamine = self.engine.calculate_reward(
            load=metrics.get('load', 0),
            vibration=metrics.get('vibration', 0),
            temperature=metrics.get('temperature', 0)
        )
        
        cortisol = 100 - dopamine
        
        # Get reasoning
        reasoning = self.engine.get_reasoning()
        
        return {
            'dopamine': round(dopamine, 2),
            'cortisol': round(cortisol, 2),
            'serotonin': 70,  # Placeholder
            'mood': self._classify_mood(dopamine),
            'reasoning': reasoning,
            'recommendation': self._get_recommendation(dopamine)
        }
    
    def _classify_mood(self, dopamine: float) -> str:
        """Classify system mood"""
        if dopamine >= 80:
            return 'EXCELLENT'
        elif dopamine >= 60:
            return 'GOOD'
        elif dopamine >= 40:
            return 'CAUTION'
        else:
            return 'CRITICAL'
    
    def _get_recommendation(self, dopamine: float) -> str:
        """Get action recommendation"""
        if dopamine >= 70:
            return "Conditions optimal. Consider increasing feed rate by 5%."
        elif dopamine >= 50:
            return "Conditions acceptable. Maintain current parameters."
        elif dopamine >= 30:
            return "Warning: Reduce feed rate by 10% or check vibration."
        else:
            return "ALERT: Stop machining and inspect tool/spindle!"


class EconomicsService(BaseService):
    """Economics calculation service"""
    
    def __init__(self):
        from erp.repositories.base_repository import JobRepository
        super().__init__(JobRepository())
    
    def calculate_job_economics(self, job_id: int) -> Dict:
        """Calculate complete job economics"""
        from erp.models import Job
        from erp.economics import EconomicsCalculator
        
        job = self.repository.get_by_id(job_id)
        if not job:
            return {'error': 'Job not found'}
        
        calculator = EconomicsCalculator(job)
        record = calculator.create_record()
        
        return {
            'job_id': job.job_id,
            'total_cost': float(record.total_cost),
            'cost_per_part': float(record.cost_per_part),
            'breakdown': {
                'machine_time': float(record.machine_time_cost),
                'material': float(record.material_cost),
                'tooling': float(record.tool_cost),
                'energy': float(record.energy_cost)
            },
            'savings': float(record.savings) if record.savings else 0,
            'traditional_cost': float(record.traditional_cost) if record.traditional_cost else 0,
            'roi_percentage': round((float(record.savings) / float(record.traditional_cost)) * 100, 2) if record.savings and record.traditional_cost else 0
        }
