"""
Generative Platform - LLM Agents + Admin Controls + Adaptive Generation

Combines KrystalSDK adaptive intelligence with:
- Multi-agent LLM orchestration
- Admin control panel
- Content generation pipelines
- Real-time monitoring & governance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum, auto
from abc import ABC, abstractmethod
import time
import json
import hashlib
import random
import os

# Import LLM client (unified local/API support)
try:
    from .llm_client import LLMClient, Message as LLMMessage, LLMConfig
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    LLMClient = None


# ============================================================
# Core Types
# ============================================================

class AgentRole(Enum):
    """LLM Agent roles in the system."""
    PLANNER = auto()        # High-level task planning
    CODER = auto()          # Code generation
    CRITIC = auto()         # Review and critique
    RESEARCHER = auto()     # Information gathering
    EXECUTOR = auto()       # Task execution
    GUARDIAN = auto()       # Safety and governance
    CREATIVE = auto()       # Creative content
    OPTIMIZER = auto()      # Performance tuning


class ContentType(Enum):
    """Types of generated content."""
    CODE = auto()
    TEXT = auto()
    CONFIG = auto()
    SHADER = auto()
    PRESET = auto()
    POLICY = auto()
    REPORT = auto()
    DECISION = auto()


class AdminLevel(Enum):
    """Admin permission levels."""
    VIEWER = 0
    OPERATOR = 1
    ADMIN = 2
    SUPERADMIN = 3


@dataclass
class GeneratedArtifact:
    """Output from generation pipeline."""
    content_type: ContentType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    agent_id: str = ""
    timestamp: float = field(default_factory=time.time)
    approved: bool = False


@dataclass
class AgentMessage:
    """Message between agents."""
    sender: str
    receiver: str
    content: Any
    message_type: str = "request"
    priority: int = 5
    timestamp: float = field(default_factory=time.time)


# ============================================================
# LLM Agent Framework
# ============================================================

class LLMAgent(ABC):
    """Base class for LLM-powered agents."""

    def __init__(self, agent_id: str, role: AgentRole, model: str = "local"):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.memory: List[Dict] = []
        self.active = True
        self.metrics = {"calls": 0, "tokens": 0, "latency_sum": 0}

    @abstractmethod
    def process(self, input_data: Any, context: Dict = None) -> Any:
        """Process input and generate output."""
        pass

    def remember(self, item: Dict):
        """Add to agent memory."""
        self.memory.append({**item, "timestamp": time.time()})
        if len(self.memory) > 100:
            self.memory.pop(0)

    def get_context(self, n: int = 5) -> List[Dict]:
        """Get recent memory context."""
        return self.memory[-n:]


class PlannerAgent(LLMAgent):
    """Breaks down tasks into subtasks."""

    def __init__(self, agent_id: str = "planner"):
        super().__init__(agent_id, AgentRole.PLANNER)

    def process(self, task: str, context: Dict = None) -> List[Dict]:
        """Plan task execution."""
        self.metrics["calls"] += 1

        # Simulated planning (real impl would call LLM)
        steps = [
            {"step": 1, "action": "analyze", "target": task},
            {"step": 2, "action": "decompose", "subtasks": []},
            {"step": 3, "action": "prioritize", "criteria": "impact"},
            {"step": 4, "action": "execute", "parallel": True},
            {"step": 5, "action": "validate", "metrics": ["quality", "safety"]}
        ]

        self.remember({"task": task, "plan": steps})
        return steps


class CoderAgent(LLMAgent):
    """Generates code artifacts."""

    def __init__(self, agent_id: str = "coder", use_llm: bool = True):
        super().__init__(agent_id, AgentRole.CODER)
        self.templates = {}
        self.use_llm = use_llm and HAS_LLM
        self.llm_client = LLMClient() if self.use_llm else None

    def process(self, spec: Dict, context: Dict = None) -> GeneratedArtifact:
        """Generate code from specification."""
        self.metrics["calls"] += 1
        start = time.time()

        if self.use_llm and self.llm_client:
            # Use real LLM (local or API)
            messages = [
                LLMMessage("system", "You are an expert programmer. Generate clean, working code."),
                LLMMessage("user", f"Generate code for: {spec.get('task', spec.get('name', 'function'))}\nSpec: {json.dumps(spec)}")
            ]
            response = self.llm_client.complete(messages, max_tokens=1024)
            code = response.content
            self.metrics["tokens"] += response.tokens_total
        else:
            # Fallback mock
            code = f"""# Generated by {self.agent_id}
# Spec: {spec.get('name', 'unnamed')}

def generated_function():
    \"\"\"Auto-generated implementation.\"\"\"
    pass
"""

        self.metrics["latency_sum"] += (time.time() - start) * 1000

        artifact = GeneratedArtifact(
            content_type=ContentType.CODE,
            content=code,
            metadata=spec,
            quality_score=0.8 if not self.use_llm else 0.85,
            agent_id=self.agent_id
        )

        self.remember({"spec": spec, "artifact_hash": hashlib.md5(code.encode()).hexdigest()})
        return artifact


class CriticAgent(LLMAgent):
    """Reviews and critiques generated content."""

    def __init__(self, agent_id: str = "critic"):
        super().__init__(agent_id, AgentRole.CRITIC)

    def process(self, artifact: GeneratedArtifact, context: Dict = None) -> Dict:
        """Critique artifact."""
        self.metrics["calls"] += 1

        # Simulated critique
        review = {
            "artifact_id": artifact.agent_id,
            "scores": {
                "correctness": random.uniform(0.7, 1.0),
                "efficiency": random.uniform(0.6, 0.9),
                "readability": random.uniform(0.7, 1.0),
                "safety": random.uniform(0.8, 1.0)
            },
            "issues": [],
            "suggestions": [],
            "approved": True
        }

        overall = sum(review["scores"].values()) / len(review["scores"])
        review["overall_score"] = overall
        review["approved"] = overall > 0.75

        self.remember({"review": review})
        return review


class GuardianAgent(LLMAgent):
    """Safety and governance enforcement."""

    def __init__(self, agent_id: str = "guardian"):
        super().__init__(agent_id, AgentRole.GUARDIAN)
        self.policies: List[Dict] = []
        self.blocked_patterns: List[str] = []

    def process(self, content: Any, context: Dict = None) -> Dict:
        """Check content against policies."""
        self.metrics["calls"] += 1

        result = {
            "safe": True,
            "violations": [],
            "risk_score": 0.0,
            "recommendations": []
        }

        # Check blocked patterns
        content_str = str(content)
        for pattern in self.blocked_patterns:
            if pattern.lower() in content_str.lower():
                result["safe"] = False
                result["violations"].append(f"Blocked pattern: {pattern}")
                result["risk_score"] += 0.5

        self.remember({"checked": type(content).__name__, "result": result["safe"]})
        return result

    def add_policy(self, policy: Dict):
        """Add governance policy."""
        self.policies.append(policy)

    def block_pattern(self, pattern: str):
        """Block content pattern."""
        self.blocked_patterns.append(pattern)


class OptimizerAgent(LLMAgent):
    """Performance optimization agent."""

    def __init__(self, agent_id: str = "optimizer"):
        super().__init__(agent_id, AgentRole.OPTIMIZER)
        # Integrate with KrystalSDK
        self.krystal = None

    def process(self, state: Dict, context: Dict = None) -> Dict:
        """Optimize based on state."""
        self.metrics["calls"] += 1

        # Lazy import to avoid circular deps
        if self.krystal is None:
            try:
                from .krystal_sdk import Krystal
                self.krystal = Krystal()
            except ImportError:
                pass

        if self.krystal:
            self.krystal.observe(state)
            action = self.krystal.decide()
            return {
                "action": action,
                "phase": self.krystal.get_phase(),
                "metrics": self.krystal.get_metrics()
            }

        return {"action": [0.5] * 4, "phase": "LIQUID", "metrics": {}}


# ============================================================
# Agent Orchestrator
# ============================================================

class AgentOrchestrator:
    """Coordinates multiple LLM agents."""

    def __init__(self):
        self.agents: Dict[str, LLMAgent] = {}
        self.message_queue: List[AgentMessage] = []
        self.workflows: Dict[str, List[str]] = {}
        self.running = False

    def register(self, agent: LLMAgent):
        """Register agent."""
        self.agents[agent.agent_id] = agent

    def unregister(self, agent_id: str):
        """Remove agent."""
        self.agents.pop(agent_id, None)

    def send(self, message: AgentMessage):
        """Send message between agents."""
        self.message_queue.append(message)

    def define_workflow(self, name: str, agent_sequence: List[str]):
        """Define agent workflow."""
        self.workflows[name] = agent_sequence

    def execute_workflow(self, name: str, initial_input: Any) -> List[Any]:
        """Execute workflow through agent chain."""
        if name not in self.workflows:
            return []

        results = []
        current_input = initial_input

        for agent_id in self.workflows[name]:
            agent = self.agents.get(agent_id)
            if agent and agent.active:
                result = agent.process(current_input)
                results.append({"agent": agent_id, "result": result})
                current_input = result

        return results

    def broadcast(self, message: Any, exclude: List[str] = None):
        """Broadcast to all agents."""
        exclude = exclude or []
        for agent_id, agent in self.agents.items():
            if agent_id not in exclude and agent.active:
                agent.process(message)

    def get_metrics(self) -> Dict:
        """Aggregate metrics from all agents."""
        return {
            agent_id: agent.metrics
            for agent_id, agent in self.agents.items()
        }


# ============================================================
# Admin Control Panel
# ============================================================

@dataclass
class AdminUser:
    """Admin user account."""
    username: str
    level: AdminLevel
    permissions: List[str] = field(default_factory=list)
    api_key: str = ""
    created: float = field(default_factory=time.time)


class AdminControlPanel:
    """Admin interface for platform governance."""

    def __init__(self):
        self.users: Dict[str, AdminUser] = {}
        self.audit_log: List[Dict] = []
        self.config: Dict[str, Any] = {
            "max_tokens_per_request": 4096,
            "rate_limit_per_minute": 60,
            "content_filtering": True,
            "auto_approve_threshold": 0.9,
            "require_guardian_check": True
        }
        self.feature_flags: Dict[str, bool] = {}

    def add_user(self, user: AdminUser, actor: str = "system"):
        """Add admin user."""
        self.users[user.username] = user
        self._audit(actor, "add_user", {"username": user.username, "level": user.level.name})

    def check_permission(self, username: str, permission: str) -> bool:
        """Check if user has permission."""
        user = self.users.get(username)
        if not user:
            return False

        # Superadmin has all permissions
        if user.level == AdminLevel.SUPERADMIN:
            return True

        return permission in user.permissions

    def set_config(self, key: str, value: Any, actor: str):
        """Update configuration."""
        if not self.check_permission(actor, "config.write"):
            raise PermissionError(f"User {actor} cannot modify config")

        old_value = self.config.get(key)
        self.config[key] = value
        self._audit(actor, "config_change", {"key": key, "old": old_value, "new": value})

    def set_feature_flag(self, flag: str, enabled: bool, actor: str):
        """Toggle feature flag."""
        self.feature_flags[flag] = enabled
        self._audit(actor, "feature_flag", {"flag": flag, "enabled": enabled})

    def _audit(self, actor: str, action: str, details: Dict):
        """Add audit log entry."""
        self.audit_log.append({
            "timestamp": time.time(),
            "actor": actor,
            "action": action,
            "details": details
        })

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get recent audit entries."""
        return self.audit_log[-limit:]

    def get_dashboard_data(self) -> Dict:
        """Get data for admin dashboard."""
        return {
            "user_count": len(self.users),
            "config": self.config,
            "feature_flags": self.feature_flags,
            "recent_audit": self.get_audit_log(10)
        }


# ============================================================
# Generation Pipeline
# ============================================================

class GenerationPipeline:
    """Content generation pipeline with governance."""

    def __init__(self, orchestrator: AgentOrchestrator, admin: AdminControlPanel):
        self.orchestrator = orchestrator
        self.admin = admin
        self.output_queue: List[GeneratedArtifact] = []
        self.pending_approval: List[GeneratedArtifact] = []

    def generate(self, request: Dict, actor: str) -> Optional[GeneratedArtifact]:
        """
        Generate content through agent pipeline.

        Flow:
        1. Planner breaks down request
        2. Coder/Creative generates content
        3. Critic reviews quality
        4. Guardian checks safety
        5. Optimizer tunes output
        6. Admin approval if needed
        """
        # Check permissions
        if not self.admin.check_permission(actor, "generate"):
            return None

        # Get agents
        planner = self.orchestrator.agents.get("planner")
        coder = self.orchestrator.agents.get("coder")
        critic = self.orchestrator.agents.get("critic")
        guardian = self.orchestrator.agents.get("guardian")

        # Plan
        plan = planner.process(request.get("task", "")) if planner else []

        # Generate
        artifact = None
        if coder:
            artifact = coder.process(request)

        if not artifact:
            return None

        # Critique
        if critic:
            review = critic.process(artifact)
            artifact.quality_score = review.get("overall_score", 0)
            artifact.metadata["review"] = review

        # Safety check
        if guardian and self.admin.config.get("require_guardian_check"):
            safety = guardian.process(artifact.content)
            artifact.metadata["safety"] = safety
            if not safety["safe"]:
                artifact.metadata["blocked"] = True
                return artifact

        # Auto-approve or queue
        threshold = self.admin.config.get("auto_approve_threshold", 0.9)
        if artifact.quality_score >= threshold:
            artifact.approved = True
            self.output_queue.append(artifact)
        else:
            self.pending_approval.append(artifact)

        return artifact

    def approve(self, artifact_idx: int, actor: str) -> bool:
        """Manually approve pending artifact."""
        if not self.admin.check_permission(actor, "approve"):
            return False

        if 0 <= artifact_idx < len(self.pending_approval):
            artifact = self.pending_approval.pop(artifact_idx)
            artifact.approved = True
            self.output_queue.append(artifact)
            self.admin._audit(actor, "approve_artifact", {"idx": artifact_idx})
            return True
        return False

    def reject(self, artifact_idx: int, actor: str, reason: str = "") -> bool:
        """Reject pending artifact."""
        if not self.admin.check_permission(actor, "approve"):
            return False

        if 0 <= artifact_idx < len(self.pending_approval):
            artifact = self.pending_approval.pop(artifact_idx)
            self.admin._audit(actor, "reject_artifact", {"idx": artifact_idx, "reason": reason})
            return True
        return False


# ============================================================
# Generative Platform (Main Interface)
# ============================================================

class GenerativePlatform:
    """
    Complete generative platform with agents, admin, and pipelines.

    Usage:
        platform = create_generative_platform()

        # Admin setup
        platform.admin.add_user(AdminUser("admin", AdminLevel.SUPERADMIN))

        # Generate content
        result = platform.generate({"task": "create optimizer"}, actor="admin")

        # Check pending approvals
        pending = platform.get_pending()

        # Approve
        platform.approve(0, actor="admin")
    """

    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        self.admin = AdminControlPanel()
        self.pipeline = GenerationPipeline(self.orchestrator, self.admin)

        # Initialize default agents
        self._setup_default_agents()

        # Define default workflows
        self._setup_workflows()

    def _setup_default_agents(self):
        """Create default agent set."""
        self.orchestrator.register(PlannerAgent())
        self.orchestrator.register(CoderAgent())
        self.orchestrator.register(CriticAgent())
        self.orchestrator.register(GuardianAgent())
        self.orchestrator.register(OptimizerAgent())

    def _setup_workflows(self):
        """Define standard workflows."""
        self.orchestrator.define_workflow(
            "code_generation",
            ["planner", "coder", "critic", "guardian"]
        )
        self.orchestrator.define_workflow(
            "optimization",
            ["planner", "optimizer", "critic"]
        )
        self.orchestrator.define_workflow(
            "review_only",
            ["critic", "guardian"]
        )

    def generate(self, request: Dict, actor: str) -> Optional[GeneratedArtifact]:
        """Generate content."""
        return self.pipeline.generate(request, actor)

    def run_workflow(self, workflow: str, input_data: Any) -> List[Any]:
        """Execute named workflow."""
        return self.orchestrator.execute_workflow(workflow, input_data)

    def get_pending(self) -> List[GeneratedArtifact]:
        """Get pending approvals."""
        return self.pipeline.pending_approval

    def approve(self, idx: int, actor: str) -> bool:
        """Approve artifact."""
        return self.pipeline.approve(idx, actor)

    def reject(self, idx: int, actor: str, reason: str = "") -> bool:
        """Reject artifact."""
        return self.pipeline.reject(idx, actor, reason)

    def get_agent(self, agent_id: str) -> Optional[LLMAgent]:
        """Get agent by ID."""
        return self.orchestrator.agents.get(agent_id)

    def add_agent(self, agent: LLMAgent):
        """Add custom agent."""
        self.orchestrator.register(agent)

    def get_metrics(self) -> Dict:
        """Get platform metrics."""
        return {
            "agents": self.orchestrator.get_metrics(),
            "admin": self.admin.get_dashboard_data(),
            "queue": {
                "pending": len(self.pipeline.pending_approval),
                "completed": len(self.pipeline.output_queue)
            }
        }


# Factory
def create_generative_platform() -> GenerativePlatform:
    """Create configured generative platform."""
    platform = GenerativePlatform()

    # Add default superadmin
    platform.admin.add_user(AdminUser(
        username="system",
        level=AdminLevel.SUPERADMIN,
        permissions=["*"]
    ))

    return platform


# Demo
if __name__ == "__main__":
    print("=== Generative Platform Demo ===\n")

    platform = create_generative_platform()

    # Setup admin
    platform.admin.add_user(AdminUser(
        username="developer",
        level=AdminLevel.ADMIN,
        permissions=["generate", "approve", "config.read"]
    ), actor="system")

    # Generate content
    print("Generating code artifact...")
    artifact = platform.generate({
        "task": "create performance optimizer",
        "name": "perf_optimizer",
        "type": "function"
    }, actor="developer")

    if artifact:
        print(f"Generated: {artifact.content_type.name}")
        print(f"Quality: {artifact.quality_score:.2f}")
        print(f"Approved: {artifact.approved}")
        print(f"Content preview:\n{artifact.content[:200]}...")

    # Run workflow
    print("\n=== Running Workflow ===")
    results = platform.run_workflow("code_generation", "optimize game performance")
    for r in results:
        print(f"  {r['agent']}: {type(r['result']).__name__}")

    # Metrics
    print("\n=== Platform Metrics ===")
    metrics = platform.get_metrics()
    print(f"Agents: {list(metrics['agents'].keys())}")
    print(f"Pending: {metrics['queue']['pending']}")
    print(f"Completed: {metrics['queue']['completed']}")
