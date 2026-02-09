from sqlalchemy import Column, Integer, DateTime, Float, String, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class Machine(Base):
    """
    Represents a CNC machine in the system
    """
    __tablename__ = 'machines'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    serial_number = Column(String, unique=True, nullable=False)
    model = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)


class Telemetry(Base):
    """
    TimescaleDB hypertable for storing high-frequency telemetry data
    Includes columns for spindle_load, vibration_x, dopamine_score, and cortisol_level
    as required for the 'Neuro-Safety' reflex at 1kHz ingestion rates
    """
    __tablename__ = 'telemetry'
    
    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(Integer, nullable=False)  # Foreign key reference
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Core telemetry values
    spindle_load = Column(Float, nullable=False)  # Percentage load on spindle
    vibration_x = Column(Float, nullable=False)   # Vibration in X-axis
    vibration_y = Column(Float)                   # Vibration in Y-axis
    vibration_z = Column(Float)                   # Vibration in Z-axis
    
    # Neuro-Safety metrics
    dopamine_score = Column(Float, default=0.0)  # Reward metric (efficiency)
    cortisol_level = Column(Float, default=0.0)  # Stress metric (risk/violation)
    
    # Additional operational metrics
    spindle_rpm = Column(Float)
    feed_rate = Column(Float)
    temperature = Column(Float)
    axis_position_x = Column(Float)
    axis_position_y = Column(Float)
    axis_position_z = Column(Float)
    tool_offset_x = Column(Float)
    tool_offset_y = Column(Float)
    tool_offset_z = Column(Float)
    
    # Create composite index for efficient time-series queries
    __table_args__ = (
        Index('idx_telemetry_machine_time', 'machine_id', 'timestamp'),
        Index('idx_telemetry_cortisol_time', 'cortisol_level', 'timestamp'),
        Index('idx_telemetry_dopamine_time', 'dopamine_score', 'timestamp'),
    )


class Project(Base):
    """
    Represents a manufacturing project/job
    """
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    estimated_duration_hours = Column(Float)
    actual_duration_hours = Column(Float)
    status = Column(String, default='pending')  # pending, running, completed, paused, error


def get_database_url():
    """
    Get database URL from environment variables or use default
    """
    return os.getenv('DATABASE_URL', 'postgresql://username:password@localhost/fanuc_rise')


def create_database_engine():
    """
    Create database engine with connection pooling settings optimized for high-frequency writes
    """
    database_url = get_database_url()
    engine = create_engine(
        database_url,
        pool_size=20,           # Connection pool size for concurrent telemetry writes
        max_overflow=30,        # Maximum overflow connections
        pool_pre_ping=True,     # Verify connections before use
        pool_recycle=3600       # Recycle connections every hour
    )
    return engine


def create_tables(engine):
    """
    Create all tables in the database
    This will also convert the Telemetry table into a Hypertable when using TimescaleDB
    """
    Base.metadata.create_all(bind=engine)


def get_session_local(engine):
    """
    Create a session factory for database operations
    """
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal