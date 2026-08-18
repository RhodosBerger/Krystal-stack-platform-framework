"""
Database Connection Manager
Handles database connections, sessions, and configuration

FEATURES:
- Connection pooling
- Session management
- Environment-based configuration
- Health checks
- Migration support
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os
from typing import Generator
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages database connections and sessions
    
    SINGLETON pattern - only one instance per application
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.engine = None
        self.SessionLocal = None
        self._initialized = True
    
    def initialize(self, database_url: str = None, echo: bool = False):
        """
        Initialize database connection
        
        Args:
            database_url: PostgreSQL connection string
            echo: Whether to log SQL statements
        """
        if database_url is None:
            # Get from environment variable
            database_url = os.getenv(
                'DATABASE_URL',
                'postgresql://postgres:postgres@localhost:5432/manufacturing_db'
            )
        
        logger.info(f"Initializing database connection to {database_url.split('@')[1] if '@' in database_url else 'database'}")
        
        # Create engine with connection pooling
        self.engine = create_engine(
            database_url,
            echo=echo,
            poolclass=QueuePool,
            pool_size=10,  # Number of connections to keep open
            max_overflow=20,  # Max connections above pool_size
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600  # Recycle connections after 1 hour
        )
        
        # Create session factory
        self.SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
        )
        
        logger.info("✅ Database connection initialized")
    
    def get_session(self):
        """
        Get a database session
        
        Returns:
            SQLAlchemy session
        """
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator:
        """
        Provide a transactional scope for database operations
        
        Usage:
            with db_manager.session_scope() as session:
                session.add(new_object)
                # Automatically commits on success, rolls back on error
        
        Yields:
            SQLAlchemy session
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """
        Check database connection health
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def close(self):
        """Close all database connections"""
        if self.SessionLocal:
            self.SessionLocal.remove()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")


# =============================================================================
# GLOBAL DATABASE MANAGER INSTANCE
# =============================================================================

db_manager = DatabaseManager()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_db() -> Generator:
    """
    FastAPI dependency for getting database session
    
    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    
    Yields:
        Database session
    """
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


def init_db(database_url: str = None, echo: bool = False):
    """
    Initialize database (convenience function)
    
    Args:
        database_url: PostgreSQL connection string
        echo: Whether to log SQL statements
    """
    db_manager.initialize(database_url, echo)


def get_session():
    """
    Get database session (convenience function)
    
    Returns:
        SQLAlchemy session
    """
    return db_manager.get_session()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Initialize database
    init_db(echo=True)
    
    # Health check
    if db_manager.health_check():
        print("✅ Database connection healthy")
    else:
        print("❌ Database connection failed")
    
    # Example: Using session context
    from database.models import Material
    
    try:
        with db_manager.session_scope() as session:
            # Create new material
            aluminum = Material(
                name='Aluminum 6061',
                type='Metal',
                properties={
                    'density': 2.7,
                    'tensile_strength': 310,
                    'hardness_hb': 95
                }
            )
            session.add(aluminum)
            print("✅ Material added successfully")
        
        # Query materials
        with db_manager.session_scope() as session:
            materials = session.query(Material).all()
            print(f"\n📊 Materials in database: {len(materials)}")
            for mat in materials:
                print(f"  - {mat.name}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Close connections
        db_manager.close()
