"""
MongoDB Store - Document and Data Persistence

Provides MongoDB integration for document storage, RAG embeddings,
and application data persistence.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Optional MongoDB import
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.collection import Collection
    from bson import ObjectId
    HAS_MONGO = True
except ImportError:
    HAS_MONGO = False
    logger.warning("pymongo not installed - using in-memory fallback")


@dataclass
class MongoConfig:
    """MongoDB connection configuration."""
    host: str = "localhost"
    port: int = 27017
    database: str = "gamesa_db"
    username: Optional[str] = None
    password: Optional[str] = None
    auth_source: str = "admin"


class CollectionName(Enum):
    DOCUMENTS = "documents"
    EMBEDDINGS = "embeddings"
    ANALYSES = "analyses"
    USERS = "users"
    AUDIT_LOGS = "audit_logs"
    CACHE = "llm_cache"
    AUTOCOMPLETE = "autocomplete_history"


class InMemoryCollection:
    """In-memory fallback when MongoDB is not available."""

    def __init__(self, name: str):
        self.name = name
        self._data: Dict[str, Dict] = {}
        self._counter = 0

    def insert_one(self, document: Dict) -> Any:
        self._counter += 1
        doc_id = f"mem_{self._counter}"
        document["_id"] = doc_id
        self._data[doc_id] = document
        return type("InsertResult", (), {"inserted_id": doc_id})()

    def insert_many(self, documents: List[Dict]) -> Any:
        ids = []
        for doc in documents:
            result = self.insert_one(doc)
            ids.append(result.inserted_id)
        return type("InsertManyResult", (), {"inserted_ids": ids})()

    def find_one(self, filter: Dict = None) -> Optional[Dict]:
        if not filter:
            return next(iter(self._data.values()), None)
        for doc in self._data.values():
            if self._matches(doc, filter):
                return doc
        return None

    def find(self, filter: Dict = None, limit: int = 0) -> List[Dict]:
        results = []
        for doc in self._data.values():
            if not filter or self._matches(doc, filter):
                results.append(doc)
                if limit and len(results) >= limit:
                    break
        return results

    def update_one(self, filter: Dict, update: Dict) -> Any:
        doc = self.find_one(filter)
        if doc:
            if "$set" in update:
                doc.update(update["$set"])
            return type("UpdateResult", (), {"modified_count": 1})()
        return type("UpdateResult", (), {"modified_count": 0})()

    def delete_one(self, filter: Dict) -> Any:
        for doc_id, doc in list(self._data.items()):
            if self._matches(doc, filter):
                del self._data[doc_id]
                return type("DeleteResult", (), {"deleted_count": 1})()
        return type("DeleteResult", (), {"deleted_count": 0})()

    def delete_many(self, filter: Dict) -> Any:
        count = 0
        for doc_id, doc in list(self._data.items()):
            if self._matches(doc, filter):
                del self._data[doc_id]
                count += 1
        return type("DeleteResult", (), {"deleted_count": count})()

    def count_documents(self, filter: Dict = None) -> int:
        if not filter:
            return len(self._data)
        return len(self.find(filter))

    def _matches(self, doc: Dict, filter: Dict) -> bool:
        for key, value in filter.items():
            if key == "_id" and key in doc:
                if doc[key] != value:
                    return False
            elif key not in doc or doc[key] != value:
                return False
        return True


class MongoStore:
    """
    MongoDB store for application data.

    Features:
    - Document CRUD operations
    - Embedding storage for RAG
    - Analysis results caching
    - Autocomplete history
    - Audit logging
    """

    def __init__(self, config: Optional[MongoConfig] = None):
        self.config = config or MongoConfig()
        self._client = None
        self._db = None
        self._collections: Dict[str, Any] = {}

        self._connect()

    def _connect(self):
        """Connect to MongoDB or use fallback."""
        if HAS_MONGO:
            try:
                if self.config.username:
                    uri = f"mongodb://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}?authSource={self.config.auth_source}"
                else:
                    uri = f"mongodb://{self.config.host}:{self.config.port}"

                self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                self._client.admin.command("ping")
                self._db = self._client[self.config.database]
                logger.info(f"Connected to MongoDB: {self.config.host}:{self.config.port}")

                # Create indexes
                self._create_indexes()

            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}, using in-memory fallback")
                self._use_fallback()
        else:
            self._use_fallback()

    def _use_fallback(self):
        """Use in-memory fallback."""
        for name in CollectionName:
            self._collections[name.value] = InMemoryCollection(name.value)

    def _create_indexes(self):
        """Create database indexes."""
        if not self._db:
            return

        # Documents collection
        self._db[CollectionName.DOCUMENTS.value].create_index([
            ("upload_time", DESCENDING)
        ])
        self._db[CollectionName.DOCUMENTS.value].create_index([
            ("status", ASCENDING)
        ])
        self._db[CollectionName.DOCUMENTS.value].create_index([
            ("tags", ASCENDING)
        ])

        # Embeddings collection
        self._db[CollectionName.EMBEDDINGS.value].create_index([
            ("doc_id", ASCENDING)
        ])

        # Analyses collection
        self._db[CollectionName.ANALYSES.value].create_index([
            ("doc_id", ASCENDING),
            ("analysis_type", ASCENDING)
        ])

        # Audit logs
        self._db[CollectionName.AUDIT_LOGS.value].create_index([
            ("timestamp", DESCENDING)
        ])

    def _get_collection(self, name: CollectionName) -> Any:
        """Get collection by name."""
        if self._db:
            return self._db[name.value]
        return self._collections.get(name.value, InMemoryCollection(name.value))

    # Document operations

    def save_document(self, document: Dict) -> str:
        """Save document record."""
        coll = self._get_collection(CollectionName.DOCUMENTS)
        document["created_at"] = datetime.now()
        document["updated_at"] = datetime.now()
        result = coll.insert_one(document)
        return str(result.inserted_id)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID."""
        coll = self._get_collection(CollectionName.DOCUMENTS)
        if HAS_MONGO and self._db:
            try:
                return coll.find_one({"_id": ObjectId(doc_id)})
            except:
                return coll.find_one({"_id": doc_id})
        return coll.find_one({"_id": doc_id})

    def update_document(self, doc_id: str, updates: Dict) -> bool:
        """Update document."""
        coll = self._get_collection(CollectionName.DOCUMENTS)
        updates["updated_at"] = datetime.now()

        if HAS_MONGO and self._db:
            try:
                result = coll.update_one(
                    {"_id": ObjectId(doc_id)},
                    {"$set": updates}
                )
            except:
                result = coll.update_one({"_id": doc_id}, {"$set": updates})
        else:
            result = coll.update_one({"_id": doc_id}, {"$set": updates})

        return result.modified_count > 0

    def delete_document(self, doc_id: str) -> bool:
        """Delete document."""
        coll = self._get_collection(CollectionName.DOCUMENTS)

        if HAS_MONGO and self._db:
            try:
                result = coll.delete_one({"_id": ObjectId(doc_id)})
            except:
                result = coll.delete_one({"_id": doc_id})
        else:
            result = coll.delete_one({"_id": doc_id})

        return result.deleted_count > 0

    def list_documents(
        self,
        filter: Optional[Dict] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict]:
        """List documents with optional filter."""
        coll = self._get_collection(CollectionName.DOCUMENTS)
        query = filter or {}

        if HAS_MONGO and self._db:
            cursor = coll.find(query).sort("created_at", DESCENDING).skip(skip).limit(limit)
            return list(cursor)
        return coll.find(query, limit=limit)

    # Embedding operations for RAG

    def save_embedding(
        self,
        doc_id: str,
        chunk_id: int,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict] = None,
    ) -> str:
        """Save text embedding for RAG."""
        coll = self._get_collection(CollectionName.EMBEDDINGS)
        record = {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
            "created_at": datetime.now(),
        }
        result = coll.insert_one(record)
        return str(result.inserted_id)

    def search_embeddings(
        self,
        doc_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Search embeddings (basic - for vector search use dedicated DB)."""
        coll = self._get_collection(CollectionName.EMBEDDINGS)
        query = {}
        if doc_ids:
            query["doc_id"] = {"$in": doc_ids}
        return coll.find(query, limit=limit)

    # Analysis results

    def save_analysis(
        self,
        doc_id: str,
        analysis_type: str,
        result: Dict,
        tokens_used: int = 0,
    ) -> str:
        """Save LLM analysis result."""
        coll = self._get_collection(CollectionName.ANALYSES)
        record = {
            "doc_id": doc_id,
            "analysis_type": analysis_type,
            "result": result,
            "tokens_used": tokens_used,
            "created_at": datetime.now(),
        }
        result = coll.insert_one(record)
        return str(result.inserted_id)

    def get_analyses(self, doc_id: str) -> List[Dict]:
        """Get all analyses for a document."""
        coll = self._get_collection(CollectionName.ANALYSES)
        return coll.find({"doc_id": doc_id})

    # Autocomplete history

    def save_autocomplete(
        self,
        field_type: str,
        value: str,
        user_id: Optional[str] = None,
    ):
        """Save autocomplete entry."""
        coll = self._get_collection(CollectionName.AUTOCOMPLETE)
        coll.insert_one({
            "field_type": field_type,
            "value": value,
            "user_id": user_id,
            "timestamp": datetime.now(),
        })

    def get_autocomplete_history(
        self,
        field_type: str,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[str]:
        """Get recent autocomplete values."""
        coll = self._get_collection(CollectionName.AUTOCOMPLETE)
        query = {"field_type": field_type}
        if user_id:
            query["user_id"] = user_id

        results = coll.find(query, limit=limit)
        return [r["value"] for r in results]

    # Audit logging

    def log_audit(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        user_id: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        """Log audit event."""
        coll = self._get_collection(CollectionName.AUDIT_LOGS)
        coll.insert_one({
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "user_id": user_id,
            "details": details or {},
            "timestamp": datetime.now(),
        })

    def get_audit_logs(
        self,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get audit logs."""
        coll = self._get_collection(CollectionName.AUDIT_LOGS)
        query = {}
        if entity_type:
            query["entity_type"] = entity_type
        if entity_id:
            query["entity_id"] = entity_id
        return coll.find(query, limit=limit)

    # Statistics

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        for name in CollectionName:
            coll = self._get_collection(name)
            stats[name.value] = coll.count_documents({})
        return stats

    def close(self):
        """Close database connection."""
        if self._client:
            self._client.close()


# Factory function
def create_mongo_store(
    host: str = "localhost",
    port: int = 27017,
    database: str = "gamesa_db",
    **kwargs,
) -> MongoStore:
    """Create MongoDB store."""
    config = MongoConfig(host=host, port=port, database=database, **kwargs)
    return MongoStore(config)
