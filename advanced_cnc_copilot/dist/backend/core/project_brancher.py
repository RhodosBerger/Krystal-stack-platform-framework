"""
Project Brancher 🌿🔀
Responsibility:
1. Clone project state for "Preview" projects.
2. Apply "Alternative Use" deltas (e.g., optimization bias changes).
3. Manage the branching lineage in the Cortex.
"""
import uuid
import json
import redis
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

class ProjectBrancher:
    def __init__(self):
        self.redis_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
        try:
            self.redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
        except:
            self.redis = None

    def create_branch(self, parent_job_id: str, branch_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Creates an alternative build of a project.
        """
        if not self.redis:
            return {"status": "ERROR", "message": "Redis Offline"}

        # 1. Fetch Parent State
        parent_data = self.redis.hgetall(f"job:{parent_job_id}")
        if not parent_data:
            return {"status": "ERROR", "message": "Parent Project Not Found"}

        # 2. Verify "Preview" State (Mock check: if not SEALED)
        if parent_data.get("status") == "SEALED":
            return {"status": "ERROR", "message": "Cannot branch a SEALED project."}

        # 3. Create Branch State
        branch_id = f"BR-{uuid.uuid4().hex[:8]}"
        branch_data = parent_data.copy()
        branch_data.update({
            "job_id": branch_id,
            "parent_id": parent_job_id,
            "branch_name": branch_name,
            "created_at": datetime.now().isoformat(),
            "status": "PREVIEW",
            "is_alternative": "true"
        })
        
        # Apply Overrides (e.g., different feed rates or biases)
        if overrides:
            current_params = json.loads(branch_data.get("params", "{}"))
            current_params.update(overrides)
            branch_data["params"] = json.dumps(current_params)

        # 4. Save Branch
        self.redis.hset(f"job:{branch_id}", mapping=branch_data)
        
        # 5. Track in Lineage
        self.redis.lpush(f"lineage:{parent_job_id}", branch_id)
        
        return {
            "status": "SUCCESS",
            "branch_id": branch_id,
            "parent_id": parent_job_id,
            "branch_name": branch_name
        }

    def get_branches(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Returns all alternative builds for a project chain.
        """
        if not self.redis: return []
        
        # Check if job_id is a parent or a branch itself
        parent_id = self.redis.hget(f"job:{job_id}", "parent_id") or job_id
        branch_ids = self.redis.lrange(f"lineage:{parent_id}", 0, -1)
        
        branches = []
        # Include Parent
        parent_data = self.redis.hgetall(f"job:{parent_id}")
        if parent_data:
            branches.append({"job_id": parent_id, "name": "Main/Master", "status": parent_data.get("status")})
            
        for bid in branch_ids:
            b_data = self.redis.hgetall(f"job:{bid}")
            if b_data:
                branches.append({
                    "job_id": bid,
                    "name": b_data.get("branch_name", "Alternative Build"),
                    "status": b_data.get("status")
                })
        return branches

# Global Instance
project_brancher = ProjectBrancher()
