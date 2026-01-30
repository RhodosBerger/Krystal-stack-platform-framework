from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional
from sqlalchemy.orm import Session

T = TypeVar('T')

class BaseRepository(Generic[T], ABC):
    """
    Abstract base repository implementing common CRUD operations
    """
    
    def __init__(self, session: Session):
        self.session = session
    
    @abstractmethod
    def get_by_id(self, id: int) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """Get all entities with pagination"""
        pass
    
    @abstractmethod
    def create(self, entity: T) -> T:
        """Create a new entity"""
        pass
    
    @abstractmethod
    def update(self, id: int, entity: T) -> Optional[T]:
        """Update an existing entity"""
        pass
    
    @abstractmethod
    def delete(self, id: int) -> bool:
        """Delete an entity by ID"""
        pass
    
    def commit(self):
        """Commit current transaction"""
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
    
    def rollback(self):
        """Rollback current transaction"""
        self.session.rollback()