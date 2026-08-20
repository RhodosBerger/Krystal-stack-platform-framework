"""
Database Utility for FANUC RISE Backend
Provides SQLAlchemy engine and session management.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.core.config import settings

DATABASE_URL = settings.database_url

# Check if we are in a container or local
if "localhost" in DATABASE_URL and os.path.exists("/.dockerenv"):
    DATABASE_URL = DATABASE_URL.replace("localhost", "timescaledb")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
