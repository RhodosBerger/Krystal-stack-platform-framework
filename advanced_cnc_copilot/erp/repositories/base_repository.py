"""
Base Repository Pattern Implementation
Foundation for all data access layers
"""

from typing import TypeVar, Generic, List, Optional, Dict, Any
from django.db.models import Model, QuerySet
from django.core.exceptions import ObjectDoesNotExist

T = TypeVar('T', bound=Model)

class BaseRepository(Generic[T]):
    """
    Generic repository providing CRUD operations
    Usage: class MachineRepository(BaseRepository[Machine])
    """
    
    def __init__(self, model_class: type[T]):
        self.model_class = model_class
    
    def get_by_id(self, id: int) -> Optional[T]:
        """Get single entity by ID"""
        try:
            return self.model_class.objects.get(pk=id)
        except ObjectDoesNotExist:
            return None
    
    def get_all(self, **filters) -> QuerySet[T]:
        """Get all entities with optional filters"""
        if filters:
            return self.model_class.objects.filter(**filters)
        return self.model_class.objects.all()
    
    def create(self, **data) -> T:
        """Create new entity"""
        return self.model_class.objects.create(**data)
    
    def update(self, id: int, **data) -> Optional[T]:
        """Update existing entity"""
        instance = self.get_by_id(id)
        if instance:
            for key, value in data.items():
                setattr(instance, key, value)
            instance.save()
        return instance
    
    def delete(self, id: int) -> bool:
        """Delete entity by ID"""
        instance = self.get_by_id(id)
        if instance:
            instance.delete()
            return True
        return False
    
    def find_by(self, **criteria) -> QuerySet[T]:
        """Find entities by criteria"""
        return self.model_class.objects.filter(**criteria)
    
    def count(self, **filters) -> int:
        """Count entities"""
        return self.get_all(**filters).count()
    
    def exists(self, **criteria) -> bool:
        """Check if entity exists"""
        return self.model_class.objects.filter(**criteria).exists()
    
    def bulk_create(self, entities: List[Dict[str, Any]]) -> List[T]:
        """Bulk create entities"""
        instances = [self.model_class(**data) for data in entities]
        return self.model_class.objects.bulk_create(instances)
    
    def paginate(self, page: int = 1, per_page: int = 50, **filters) -> Dict[str, Any]:
        """Paginate results"""
        queryset = self.get_all(**filters)
        total = queryset.count()
        
        start = (page - 1) * per_page
        end = start + per_page
        
        results = list(queryset[start:end])
        
        return {
            'results': results,
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        }


class MachineRepository(BaseRepository):
    """Repository for Machine model"""
    
    def __init__(self):
        from erp.models import Machine
        super().__init__(Machine)
    
    def get_active_machines(self, organization_id: int = None):
        """Get all active machines"""
        filters = {'is_active': True}
        if organization_id:
            filters['organization_id'] = organization_id
        return self.find_by(**filters)
    
    def get_by_ip(self, ip_address: str):
        """Get machine by IP address"""
        return self.find_by(ip_address=ip_address).first()


class TelemetryRepository(BaseRepository):
    """Repository for Telemetry model"""
    
    def __init__(self):
        from erp.models import Telemetry
        super().__init__(Telemetry)
    
    def get_latest_for_machine(self, machine_id: int, limit: int = 100):
        """Get latest telemetry readings"""
        return self.find_by(machine_id=machine_id).order_by('-timestamp')[:limit]
    
    def get_by_timerange(self, machine_id: int, start_time, end_time):
        """Get telemetry within time range"""
        return self.model_class.objects.filter(
            machine_id=machine_id,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).order_by('-timestamp')
    
    def get_signal_count(self, machine_id: int, signal: str):
        """Count specific signal occurrences"""
        return self.find_by(machine_id=machine_id, signal=signal).count()


class ProjectRepository(BaseRepository):
    """Repository for Project model"""
    
    def __init__(self):
        from erp.models import Project
        super().__init__(Project)
    
    def get_successful_projects(self, material: str = None):
        """Get successful projects"""
        filters = {'success': True}
        if material:
            filters['material'] = material
        return self.find_by(**filters)
    
    def search_by_part_number(self, query: str):
        """Search projects by part number"""
        return self.model_class.objects.filter(part_number__icontains=query)


class ToolRepository(BaseRepository):
    """Repository for Tool model"""
    
    def __init__(self):
        from erp.models import Tool
        super().__init__(Tool)
    
    def get_tools_needing_replacement(self, threshold: float = 0.1):
        """Get tools below life threshold"""
        from django.db.models import F
        return self.model_class.objects.filter(
            current_usage__gte=F('expected_life') * (1 - threshold)
        )
    
    def get_by_status(self, status: str):
        """Get tools by status"""
        return self.find_by(status=status)


class JobRepository(BaseRepository):
    """Repository for Job model"""
    
    def __init__(self):
        from erp.models import Job
        super().__init__(Job)
    
    def get_queued_jobs(self, priority: int = None):
        """Get queued jobs ordered by priority"""
        queryset = self.find_by(status='QUEUED').order_by('priority', 'deadline')
        if priority:
            queryset = queryset.filter(priority=priority)
        return queryset
    
    def get_in_progress_jobs(self, machine_id: int = None):
        """Get jobs currently running"""
        filters = {'status': 'IN_PROGRESS'}
        if machine_id:
            filters['machine_id'] = machine_id
        return self.find_by(**filters)
