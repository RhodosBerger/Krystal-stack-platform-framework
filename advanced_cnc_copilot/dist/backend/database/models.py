"""
SQLAlchemy Models for FANUC RISE
"""
from sqlalchemy import Boolean, Column, Integer, String, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String, nullable=True)
    password = Column(String)  # Hashed
    role = Column(String, default="operator")  # admin, engineer, manager, operator
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True) # e.g. JOB-123456
    description = Column(String)
    status = Column(String, default="QUEUED")
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="jobs")

class MarketplaceComponent(Base):
    __tablename__ = "marketplace_components"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    category = Column(String) # GCODE, MATERIAL, TOOL, CONFIG
    description = Column(String)
    payload = Column(JSON) # The actual code or data
    author_id = Column(Integer, ForeignKey("users.id"))
    downloads = Column(Integer, default=0)
    rating_sum = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    version = Column(String, default="1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow)

    author = relationship("User", back_populates="shared_components")

User.jobs = relationship("Job", back_populates="owner")
User.shared_components = relationship("MarketplaceComponent", back_populates="author")
