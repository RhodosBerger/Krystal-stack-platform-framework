"""
Workflow Automation Engine 🔄
Responsibility:
1. Define and execute automated workflow chains.
2. Link trigger conditions to actions.
"""
import uuid
from typing import Dict, Any, List, Callable, Optional

class WorkflowStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class WorkflowStep:
    def __init__(self, name: str, action: Callable[[Dict], Dict], condition: Optional[Callable[[Dict], bool]] = None):
        self.id = f"STEP-{uuid.uuid4().hex[:4].upper()}"
        self.name = name
        self.action = action
        self.condition = condition or (lambda ctx: True)

class Workflow:
    def __init__(self, name: str, description: str = ""):
        self.id = f"WF-{uuid.uuid4().hex[:6].upper()}"
        self.name = name
        self.description = description
        self.steps: List[WorkflowStep] = []
        self.status = WorkflowStatus.PENDING
        self.execution_log: List[Dict[str, Any]] = []

    def add_step(self, step: WorkflowStep):
        self.steps.append(step)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes all steps in sequence."""
        self.status = WorkflowStatus.RUNNING
        current_context = context.copy()

        for step in self.steps:
            if step.condition(current_context):
                try:
                    result = step.action(current_context)
                    current_context.update(result or {})
                    self.execution_log.append({"step": step.name, "status": "SUCCESS"})
                except Exception as e:
                    self.execution_log.append({"step": step.name, "status": "FAILED", "error": str(e)})
                    self.status = WorkflowStatus.FAILED
                    return {"status": "FAILED", "error": str(e), "log": self.execution_log}
            else:
                self.execution_log.append({"step": step.name, "status": "SKIPPED"})

        self.status = WorkflowStatus.COMPLETED
        return {"status": "COMPLETED", "context": current_context, "log": self.execution_log}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "steps": [{"id": s.id, "name": s.name} for s in self.steps]
        }

class WorkflowRegistry:
    def __init__(self):
        self._workflows: Dict[str, Workflow] = {}
        self._create_default_workflows()

    def _create_default_workflows(self):
        # Sample workflow: Product -> LLM -> Audit
        wf = Workflow("ProductToPayload", "Generate payloads from product and audit")
        wf.add_step(WorkflowStep("ValidateProduct", lambda ctx: {"validated": True}))
        wf.add_step(WorkflowStep("GeneratePayloads", lambda ctx: {"payloads_generated": 4}))
        wf.add_step(WorkflowStep("AuditOutput", lambda ctx: {"audit_passed": True}))
        self._workflows[wf.id] = wf

    def register(self, workflow: Workflow) -> str:
        self._workflows[workflow.id] = workflow
        return workflow.id

    def get(self, workflow_id: str) -> Optional[Workflow]:
        return self._workflows.get(workflow_id)

    def list_all(self) -> List[Workflow]:
        return list(self._workflows.values())

# Global Instance
workflow_registry = WorkflowRegistry()
