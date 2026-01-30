"""
Database package initialization
"""

from database.models import (
    Base,
    Vendor, Material, Producer, Part,
    Project, Job, Operation,
    Telemetry, QualityInspection,
    ProjectStatus, JobStatus, OperationStatus, QualityResult
)

from database.connection import (
    DatabaseManager,
    db_manager,
    init_db,
    get_session,
    get_db
)

__all__ = [
    # Models
    'Base',
    'Vendor', 'Material', 'Producer', 'Part',
    'Project', 'Job', 'Operation',
    'Telemetry', 'QualityInspection',
    
    # Enums
    'ProjectStatus', 'JobStatus', 'OperationStatus', 'QualityResult',
    
    # Connection
    'DatabaseManager', 'db_manager',
    'init_db', 'get_session', 'get_db'
]
