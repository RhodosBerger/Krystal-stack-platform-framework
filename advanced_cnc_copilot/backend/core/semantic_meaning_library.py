"""
Semantic Meaning Library 💾
Responsibility:
1. Act as the Long-Term Memory (LTM) for the system.
2. Persist 'User Mirror' objects to the SQL Database (TimescaleDB).
3. Retrieve past projects by Intent or Geometry.
"""
import logging
import json
import os
import psycopg2
from datetime import datetime
from backend.cms.protocols.user_mirror import MirrorObject

logger = logging.getLogger("SemanticLibrary")

class SemanticMeaningLibrary:
    def __init__(self):
        # Allow override for testing, default to Docker internal DNS
        self.db_url = os.getenv("DATABASE_URL", "postgresql://user:password@timescaledb:5432/cnc_db")
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect(self.db_url)
            self.conn.autocommit = True
            logger.info("✅ Connected to Long-Term Memory (TimescaleDB)")
            self._init_schema()
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to DB: {e}. Running in Ephemeral Mode.")
            self.conn = None

    def _init_schema(self):
        """Ensure the 'project_memory' table exists"""
        if not self.conn: return
        query = """
        CREATE TABLE IF NOT EXISTS project_memory (
            id SERIAL PRIMARY KEY,
            project_id VARCHAR(50) UNIQUE NOT NULL,
            user_intent TEXT,
            geometry_spec JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
        except Exception as e:
            logger.error(f"Schema Init Failed: {e}")

    def save_mirror(self, mirror: MirrorObject) -> bool:
        """
        Persist a User Mirror Object to the DB.
        """
        if not self.conn:
            logger.info(f"Ephemeral Save: {mirror.project_id}")
            return True
            
        try:
            query = """
            INSERT INTO project_memory (project_id, user_intent, geometry_spec)
            VALUES (%s, %s, %s)
            ON CONFLICT (project_id) DO UPDATE 
            SET user_intent = EXCLUDED.user_intent, geometry_spec = EXCLUDED.geometry_spec;
            """
            with self.conn.cursor() as cur:
                cur.execute(query, (
                    mirror.project_id,
                    mirror.user_will.intent,
                    mirror.geometry.json()
                ))
            logger.info(f"✅ Saved Project {mirror.project_id} to LTM")
            return True
        except Exception as e:
            logger.error(f"Failed to save mirror: {e}")
            return False

    def recall_project(self, project_id: str) -> dict:
        """
        Retrieve a project from LTM.
        """
        if not self.conn: return {}
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT user_intent, geometry_spec FROM project_memory WHERE project_id = %s", (project_id,))
                res = cur.fetchone()
                if res:
                    return {"intent": res[0], "geometry": res[1]}
        except Exception as e:
            logger.error(f"Recall failed: {e}")
        return {}

# Global Instance
semantic_library = SemanticMeaningLibrary()
