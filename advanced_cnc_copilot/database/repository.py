"""
Database Repository Layer
Provides CRUD operations for all database models

PATTERN: Repository pattern separates data access logic
from business logic, making code more maintainable and testable
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

from database.models import (
    Vendor, Material, Producer, Part,
    Project, Job, Operation,
    Telemetry, QualityInspection,
    ProjectStatus, JobStatus, OperationStatus, QualityResult
)


# =============================================================================
# MATERIAL REPOSITORY
# =============================================================================

class MaterialRepository:
    """Repository for Material operations"""
    
    @staticmethod
    def get_all(session: Session) -> List[Material]:
        """Get all materials"""
        return session.query(Material).all()
    
    @staticmethod
    def get_by_id(session: Session, material_id: int) -> Optional[Material]:
        """Get material by ID"""
        return session.query(Material).filter(Material.id == material_id).first()
    
    @staticmethod
    def get_by_name(session: Session, name: str) -> Optional[Material]:
        """Get material by name"""
        return session.query(Material).filter(Material.name == name).first()
    
    @staticmethod
    def create(session: Session, name: str, material_type: str, properties: Dict) -> Material:
        """Create new material"""
        material = Material(name=name, type=material_type, properties=properties)
        session.add(material)
        session.flush()
        return material


# =============================================================================
# PROJECT REPOSITORY
# =============================================================================

class ProjectRepository:
    """Repository for Project operations"""
    
    @staticmethod
    def get_all(session: Session, status: Optional[ProjectStatus] = None) -> List[Project]:
        """Get all projects, optionally filtered by status"""
        query = session.query(Project)
        if status:
            query = query.filter(Project.status == status)
        return query.order_by(desc(Project.created_at)).all()
    
    @staticmethod
    def get_by_id(session: Session, project_id: int) -> Optional[Project]:
        """Get project by ID"""
        return session.query(Project).filter(Project.id == project_id).first()
    
    @staticmethod
    def create(session: Session, name: str, description: str = None, **kwargs) -> Project:
        """Create new project"""
        project = Project(name=name, description=description, **kwargs)
        session.add(project)
        session.flush()
        return project
    
    @staticmethod
    def update_status(session: Session, project_id: int, status: ProjectStatus) -> Optional[Project]:
        """Update project status"""
        project = session.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = status
            project.updated_at = datetime.utcnow()
            session.flush()
        return project
    
    @staticmethod
    def get_progress(session: Session, project_id: int) -> Dict[str, Any]:
        """Get project progress statistics"""
        project = session.query(Project).filter(Project.id == project_id).first()
        if not project:
            return None
        
        total_jobs = len(project.jobs)
        completed_jobs = sum(1 for j in project.jobs if j.status == JobStatus.COMPLETED)
        
        return {
            'project_id': project.id,
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'completion_percentage': (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            'status': project.status.value
        }


# =============================================================================
# JOB REPOSITORY
# =============================================================================

class JobRepository:
    """Repository for Job operations"""
    
    @staticmethod
    def get_by_id(session: Session, job_id: int) -> Optional[Job]:
        """Get job by ID"""
        return session.query(Job).filter(Job.id == job_id).first()
    
    @staticmethod
    def get_by_job_id(session: Session, job_id_str: str) -> Optional[Job]:
        """Get job by job_id string"""
        return session.query(Job).filter(Job.job_id == job_id_str).first()
    
    @staticmethod
    def create(session: Session, job_id: str, project_id: int, part_id: int, 
               quantity: int, **kwargs) -> Job:
        """Create new job"""
        job = Job(
            job_id=job_id,
            project_id=project_id,
            part_id=part_id,
            quantity=quantity,
            **kwargs
        )
        session.add(job)
        session.flush()
        return job
    
    @staticmethod
    def get_active_jobs(session: Session) -> List[Job]:
        """Get all active (running/queued) jobs"""
        return session.query(Job).filter(
            Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
        ).order_by(desc(Job.priority)).all()
    
    @staticmethod
    def update_status(session: Session, job_id: int, status: JobStatus) -> Optional[Job]:
        """Update job status"""
        job = session.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = status
            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.utcnow()
            elif status == JobStatus.COMPLETED:
                job.completed_at = datetime.utcnow()
            session.flush()
        return job


# =============================================================================
# OPERATION REPOSITORY
# =============================================================================

class OperationRepository:
    """Repository for Operation operations"""
    
    @staticmethod
    def get_by_id(session: Session, operation_id: int) -> Optional[Operation]:
        """Get operation by ID"""
        return session.query(Operation).filter(Operation.id == operation_id).first()
    
    @staticmethod
    def create(session: Session, job_id: int, name: str, **kwargs) -> Operation:
        """Create new operation"""
        operation = Operation(job_id=job_id, name=name, **kwargs)
        session.add(operation)
        session.flush()
        return operation
    
    @staticmethod
    def get_running_operations(session: Session) -> List[Operation]:
        """Get all currently running operations"""
        return session.query(Operation).filter(
            Operation.status == OperationStatus.RUNNING
        ).all()
    
    @staticmethod
    def update_status(session: Session, operation_id: int, status: OperationStatus) -> Optional[Operation]:
        """Update operation status"""
        operation = session.query(Operation).filter(Operation.id == operation_id).first()
        if operation:
            operation.status = status
            if status == OperationStatus.RUNNING and not operation.started_at:
                operation.started_at = datetime.utcnow()
            elif status == OperationStatus.COMPLETED:
                operation.completed_at = datetime.utcnow()
                operation.actual_time = operation.completed_at - operation.started_at
            session.flush()
        return operation


# =============================================================================
# TELEMETRY REPOSITORY
# =============================================================================

class TelemetryRepository:
    """Repository for Telemetry (time-series) data"""
    
    @staticmethod
    def add_sample(session: Session, operation_id: int, sensor_id: str, 
                   value: float, unit: str = None, **kwargs) -> Telemetry:
        """Add single telemetry sample"""
        telemetry = Telemetry(
            operation_id=operation_id,
            timestamp=datetime.utcnow(),
            sensor_id=sensor_id,
            value=value,
            unit=unit,
            **kwargs
        )
        session.add(telemetry)
        session.flush()
        return telemetry
    
    @staticmethod
    def add_batch(session: Session, samples: List[Dict]) -> int:
        """Add batch of telemetry samples (more efficient)"""
        telemetry_objects = [Telemetry(**sample) for sample in samples]
        session.bulk_save_objects(telemetry_objects)
        session.flush()
        return len(telemetry_objects)
    
    @staticmethod
    def get_latest(session: Session, operation_id: int, sensor_id: str, 
                   limit: int = 100) -> List[Telemetry]:
        """Get latest telemetry samples for operation and sensor"""
        return session.query(Telemetry).filter(
            and_(
                Telemetry.operation_id == operation_id,
                Telemetry.sensor_id == sensor_id
            )
        ).order_by(desc(Telemetry.timestamp)).limit(limit).all()
    
    @staticmethod
    def get_time_range(session: Session, operation_id: int, 
                       start_time: datetime, end_time: datetime) -> List[Telemetry]:
        """Get telemetry within time range"""
        return session.query(Telemetry).filter(
            and_(
                Telemetry.operation_id == operation_id,
                Telemetry.timestamp >= start_time,
                Telemetry.timestamp <= end_time
            )
        ).order_by(Telemetry.timestamp).all()
    
    @staticmethod
    def get_realtime_stream(session: Session, operation_id: int, 
                            last_seconds: int = 10) -> List[Telemetry]:
        """Get real-time telemetry stream (last N seconds)"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=last_seconds)
        return session.query(Telemetry).filter(
            and_(
                Telemetry.operation_id == operation_id,
                Telemetry.timestamp >= cutoff_time
            )
        ).order_by(Telemetry.timestamp).all()


# =============================================================================
# QUALITY INSPECTION REPOSITORY
# =============================================================================

class QualityInspectionRepository:
    """Repository for Quality Inspection operations"""
    
    @staticmethod
    def create(session: Session, operation_id: int, result: QualityResult, **kwargs) -> QualityInspection:
        """Create quality inspection record"""
        inspection = QualityInspection(
            operation_id=operation_id,
            result=result,
            **kwargs
        )
        session.add(inspection)
        session.flush()
        return inspection
    
    @staticmethod
    def get_by_operation(session: Session, operation_id: int) -> List[QualityInspection]:
        """Get all inspections for an operation"""
        return session.query(QualityInspection).filter(
            QualityInspection.operation_id == operation_id
        ).order_by(desc(QualityInspection.inspection_date)).all()
    
    @staticmethod
    def get_pass_rate(session: Session, start_date: datetime, end_date: datetime) -> float:
        """Calculate quality pass rate for date range"""
        total = session.query(QualityInspection).filter(
            and_(
                QualityInspection.inspection_date >= start_date,
                QualityInspection.inspection_date <= end_date
            )
        ).count()
        
        if total == 0:
            return 0.0
        
        passed = session.query(QualityInspection).filter(
            and_(
                QualityInspection.inspection_date >= start_date,
                QualityInspection.inspection_date <= end_date,
                QualityInspection.result == QualityResult.PASS
            )
        ).count()
        
        return (passed / total) * 100


# =============================================================================
# PRODUCER REPOSITORY
# =============================================================================

class ProducerRepository:
    """Repository for Producer operations"""
    
    @staticmethod
    def get_all(session: Session) -> List[Producer]:
        """Get all producers"""
        return session.query(Producer).order_by(desc(Producer.effectiveness_score)).all()
    
    @staticmethod
    def get_by_id(session: Session, producer_id: int) -> Optional[Producer]:
        """Get producer by ID"""
        return session.query(Producer).filter(Producer.id == producer_id).first()
    
    @staticmethod
    def search_by_capability(session: Session, material: str = None, 
                             min_tolerance: float = None) -> List[Producer]:
        """Search producers by capabilities"""
        # This requires JSONB queries - example implementation
        query = session.query(Producer)
        
        # Would use PostgreSQL JSONB operators in real implementation
        # query = query.filter(Producer.capabilities['materials'].contains([material]))
        
        return query.order_by(desc(Producer.effectiveness_score)).all()


# =============================================================================
# PART REPOSITORY
# =============================================================================

class PartRepository:
    """Repository for Part operations"""
    
    @staticmethod
    def get_by_id(session: Session, part_id: int) -> Optional[Part]:
        """Get part by ID"""
        return session.query(Part).filter(Part.id == part_id).first()
    
    @staticmethod
    def get_by_part_number(session: Session, part_number: str) -> Optional[Part]:
        """Get part by part number"""
        return session.query(Part).filter(Part.part_number == part_number).first()
    
    @staticmethod
    def create(session: Session, name: str, part_number: str, **kwargs) -> Part:
        """Create new part"""
        part = Part(name=name, part_number=part_number, **kwargs)
        session.add(part)
        session.flush()
        return part
    
    @staticmethod
    def search(session: Session, search_term: str) -> List[Part]:
        """Search parts by name or part number"""
        search_pattern = f"%{search_term}%"
        return session.query(Part).filter(
            or_(
                Part.name.ilike(search_pattern),
                Part.part_number.ilike(search_pattern)
            )
        ).all()


# Example usage
if __name__ == "__main__":
    from database.connection import init_db, db_manager
    
    # Initialize database
    init_db()
    
    # Example: Create project with jobs
    with db_manager.session_scope() as session:
        # Create project
        project = ProjectRepository.create(
            session,
            name="Test Project",
            description="Example manufacturing project"
        )
        print(f"✅ Created project: {project.name} (ID: {project.id})")
        
        # Get material
        material = MaterialRepository.get_by_name(session, "Aluminum 6061")
        if material:
            print(f"✅ Found material: {material.name}")
        
        # Get all active projects
        active_projects = ProjectRepository.get_all(session, status=ProjectStatus.IN_PROGRESS)
        print(f"📊 Active projects: {len(active_projects)}")
