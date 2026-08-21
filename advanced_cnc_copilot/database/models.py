"""
Database Models - SQLAlchemy ORM
Complete schema for manufacturing intelligence platform

TABLES:
1. vendors - Supplier information
2. materials - Material properties library
3. parts - Part catalog
4. producers - Manufacturing capability database
5. projects - Manufacturing projects/orders
6. jobs - Production jobs within projects
7. operations - Individual machining operations
8. telemetry - Real-time sensor data (partitioned by date)
9. quality_inspections - QC records
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, 
    Boolean, Text, TIMESTAMP, Interval, Enum as SQLEnum, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB, DOUBLE_PRECISION
from datetime import datetime
import enum

Base = declarative_base()


# =============================================================================
# ENUMS
# =============================================================================

class ProjectStatus(enum.Enum):
    """Project status enum"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class JobStatus(enum.Enum):
    """Job status enum"""
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class OperationStatus(enum.Enum):
    """Operation status enum"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QualityResult(enum.Enum):
    """Quality inspection result"""
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL = "conditional"


# =============================================================================
# MASTER DATA TABLES
# =============================================================================

class Vendor(Base):
    """
    Vendor/Supplier information
    """
    __tablename__ = 'vendors'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    contact_info = Column(JSONB)  # Phone, email, address
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parts = relationship('Part', back_populates='vendor')
    
    def __repr__(self):
        return f"<Vendor(id={self.id}, name='{self.name}')>"


class Material(Base):
    """
    Material properties library
    """
    __tablename__ = 'materials'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    type = Column(String(50))  # Metal, plastic, composite
    properties = Column(JSONB)  # Density, strength, hardness, etc.
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    parts = relationship('Part', back_populates='material')
    
    def __repr__(self):
        return f"<Material(id={self.id}, name='{self.name}')>"


class Producer(Base):
    """
    Manufacturing producer/facility capability database
    """
    __tablename__ = 'producers'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    type = Column(String(100))  # Internal, external, partner
    location = Column(String(255))
    capacity = Column(JSONB)  # Max parts/day, machine count, etc.
    capabilities = Column(JSONB)  # Materials, processes, tolerances
    effectiveness_score = Column(Float)  # Overall effectiveness (0-1)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Producer(id={self.id}, name='{self.name}')>"


class Part(Base):
    """
    Part catalog with specifications
    """
    __tablename__ = 'parts'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    part_number = Column(String(100), unique=True, nullable=False)
    
    # Foreign keys
    vendor_id = Column(Integer, ForeignKey('vendors.id'))
    material_id = Column(Integer, ForeignKey('materials.id'))
    
    # Specifications
    dimensions = Column(JSONB)  # Length, width, height, diameter, etc.
    weight_kg = Column(Float)
    complexity_score = Column(Float)  # 0.0-1.0
    
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    vendor = relationship('Vendor', back_populates='parts')
    material = relationship('Material', back_populates='parts')
    jobs = relationship('Job', back_populates='part')
    
    def __repr__(self):
        return f"<Part(id={self.id}, part_number='{self.part_number}', name='{self.name}')>"


# =============================================================================
# CORE OPERATIONAL TABLES
# =============================================================================

class Project(Base):
    """
    Manufacturing project/order
    Central hub for all production activity
    """
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Dates
    start_date = Column(TIMESTAMP)
    end_date = Column(TIMESTAMP)
    due_date = Column(TIMESTAMP)
    
    # Status
    status = Column(SQLEnum(ProjectStatus), default=ProjectStatus.PENDING)
    
    # LLM analysis
    llm_suggestions = Column(JSONB)  # AI recommendations
    
    # Economic
    estimated_cost = Column(Float)
    actual_cost = Column(Float)
    
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    jobs = relationship('Job', back_populates='project', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}', status='{self.status.value}')>"


class Job(Base):
    """
    Production job within a project
    """
    __tablename__ = 'jobs'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(50), unique=True, nullable=False)  # JOB-000001
    
    # Foreign keys
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    part_id = Column(Integer, ForeignKey('parts.id'), nullable=False)
    producer_id = Column(Integer, ForeignKey('producers.id'))
    
    # Quantity
    quantity = Column(Integer, nullable=False)
    completed_quantity = Column(Integer, default=0)
    
    # Scheduling
    priority = Column(Integer, default=1)  # 1=low, 5=critical
    due_date = Column(TIMESTAMP)
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    
    # Status
    status = Column(SQLEnum(JobStatus), default=JobStatus.QUEUED)
    
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    project = relationship('Project', back_populates='jobs')
    part = relationship('Part', back_populates='jobs')
    operations = relationship('Operation', back_populates='job', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f"<Job(id={self.id}, job_id='{self.job_id}', status='{self.status.value}')>"


class Operation(Base):
    """
    Individual machining operation
    """
    __tablename__ = 'operations'
    
    id = Column(Integer, primary_key=True)
    
    # Foreign key
    job_id = Column(Integer, ForeignKey('jobs.id'), nullable=False)
    
    # Operation details
    name = Column(String(255), nullable=False)
    sequence_order = Column(Integer)  # Order in job
    operation_type = Column(String(100))  # roughing, finishing, drilling, etc.
    
    # Time
    estimated_time = Column(Interval)  # Estimated duration
    actual_time = Column(Interval)  # Actual duration
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    
    # Parameters
    parameters = Column(JSONB)  # Speeds, feeds, tool, etc.
    
    # Status
    status = Column(SQLEnum(OperationStatus), default=OperationStatus.QUEUED)
    
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    job = relationship('Job', back_populates='operations')
    telemetry = relationship('Telemetry', back_populates='operation', cascade='all, delete-orphan')
    quality_inspections = relationship('QualityInspection', back_populates='operation')
    
    def __repr__(self):
        return f"<Operation(id={self.id}, name='{self.name}', status='{self.status.value}')>"


# =============================================================================
# TIME-SERIES & QUALITY TABLES
# =============================================================================

class Telemetry(Base):
    """
    Real-time sensor telemetry data
    
    IMPORTANT: This table should be partitioned by timestamp (date)
    for performance with high-volume data
    
    Example partitions:
    - telemetry_2026_01_24
    - telemetry_2026_01_25
    - etc.
    """
    __tablename__ = 'telemetry'
    
    id = Column(Integer, primary_key=True)
    
    # Foreign key
    operation_id = Column(Integer, ForeignKey('operations.id'), nullable=False)
    
    # Time (part of composite key for partitioning)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)
    
    # Sensor data
    sensor_id = Column(String(100))
    value = Column(DOUBLE_PRECISION)
    unit = Column(String(50))
    
    # Additional data
    metadata = Column(JSONB)  # Flexible for different sensor types
    
    # Biochemical states (if applicable)
    cortisol = Column(Float)
    dopamine = Column(Float)
    serotonin = Column(Float)
    adrenaline = Column(Float)
    
    # Relationships
    operation = relationship('Operation', back_populates='telemetry')
    
    # Index for time-series queries
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (timestamp)'},
    )
    
    def __repr__(self):
        return f"<Telemetry(id={self.id}, sensor='{self.sensor_id}', value={self.value})>"


class QualityInspection(Base):
    """
    Quality control inspection records
    """
    __tablename__ = 'quality_inspections'
    
    id = Column(Integer, primary_key=True)
    
    # Foreign key
    operation_id = Column(Integer, ForeignKey('operations.id'), nullable=False)
    
    # Inspector
    inspector_id = Column(String(100))
    
    # Result
    result = Column(SQLEnum(QualityResult), nullable=False)
    
    # Measurements
    measurements = Column(JSONB)  # Dimensional measurements
    defects = Column(JSONB)  # List of defects found
    
    # Comments
    comments = Column(Text)
    
    # Date
    inspection_date = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    operation = relationship('Operation', back_populates='quality_inspections')
    
    def __repr__(self):
        return f"<QualityInspection(id={self.id}, result='{self.result.value}')>"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_all_tables(engine):
    """
    Create all tables in database
    
    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.create_all(engine)
    print("✅ All tables created successfully")


def drop_all_tables(engine):
    """
    Drop all tables (WARNING: Data loss!)
    
    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.drop_all(engine)
    print("⚠️ All tables dropped")


if __name__ == "__main__":
    # Example: Create tables
    from sqlalchemy import create_engine
    
    # Connection string (update with your credentials)
    DATABASE_URL = "postgresql://user:password@localhost:5432/manufacturing_db"
    
    engine = create_engine(DATABASE_URL, echo=True)
    
    # Create all tables
    create_all_tables(engine)
    
    print("\n📊 Database Schema:")
    print("  1. vendors")
    print("  2. materials")
    print("  3. producers")
    print("  4. parts")
    print("  5. projects")
    print("  6. jobs")
    print("  7. operations")
    print("  8. telemetry (partitioned)")
    print("  9. quality_inspections")
