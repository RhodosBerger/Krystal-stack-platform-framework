"""
SQLAlchemy models mapping to Django ERP tables.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, JSON, Text
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class User(Base):
    __tablename__ = "erp_riseuser"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, index=True)
    password = Column(String(128))
    email = Column(String(254))
    role = Column(String(20))
    is_active = Column(Boolean, default=True)
    is_staff = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    last_login = Column(DateTime)
    date_joined = Column(DateTime, default=datetime.datetime.utcnow)

class Machine(Base):
    __tablename__ = "erp_machine"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    controller_type = Column(String(50))
    ip_address = Column(String(45))
    is_active = Column(Boolean, default=True)
    axes = Column(Integer)
    max_rpm = Column(Integer)
    work_envelope = Column(JSON)

class Project(Base):
    __tablename__ = "erp_project"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200))
    part_number = Column(String(100), unique=True)
    gcode = Column(Text)
    material = Column(String(50))
    complexity_score = Column(Float)
    estimated_cycle_time = Column(Float)
    dimensions = Column(JSON)
    llm_suggestions = Column(JSON)

class Telemetry(Base):
    __tablename__ = "erp_telemetry"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    machine_id = Column(Integer, ForeignKey("erp_machine.id"))
    rpm = Column(Integer)
    load = Column(Float)
    vibration_x = Column(Float)
    vibration_y = Column(Float)
    vibration_z = Column(Float)
    spindle_temp = Column(Float)
    coolant_temp = Column(Float)
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    dopamine = Column(Float)
    cortisol = Column(Float)
    signal = Column(String(10))
    active_tool = Column(String(10))
    
    machine = relationship("Machine")
