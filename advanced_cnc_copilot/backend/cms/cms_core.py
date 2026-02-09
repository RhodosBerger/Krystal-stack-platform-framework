#!/usr/bin/env python3
"""
CMS Core & Knowledge Base
The "Ground Truth" for the Engineering Copilot.

This module is responsible for:
1. Collecting User Documentation (Inputs)
2. Structuring "Engineering Rules" (Predefined Parameters)
3. Serving as the validation source for the Monitoring System
"""

import uuid
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import asyncio

# Import MessageBus for The Shadow Council
try:
    from backend.cms.message_bus import global_bus, Message
except ImportError:
    global_bus = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CMS_CORE] - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EngineeringRule:
    """A rigid rule defined by the engineer (User Input)."""
    rule_id: str
    category: str       # e.g., "Material", "Safety", "Tooling"
    condition: str      # e.g., "Material == Titanium"
    requirement: str    # e.g., "Coolant strict flood, RPM < 2000"
    criticality: str    # "High", "Medium", "Info"

@dataclass
class DocumentationEntry:
    """Freeform documentation or Wiki content."""
    doc_id: str
    title: str
    content: str
    tags: List[str]
    author: str

class KnowledgeBase:
    """
    The Central Management System (CMS) for Engineering Knowledge.
    """
    
    def __init__(self):
        self.rules: Dict[str, EngineeringRule] = {}
        self.docs: Dict[str, DocumentationEntry] = {}
        self._kb_id = f"KB_{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"Initialized Knowledge Base: {self._kb_id}")

    def add_rule(self, category: str, condition: str, requirement: str, criticality: str = "High") -> str:
        """
        User Input Interface: Define a rigid parameter/rule.
        """
        rule_id = f"RULE_{uuid.uuid4().hex[:6].upper()}"
        rule = EngineeringRule(rule_id, category, condition, requirement, criticality)
        self.rules[rule_id] = rule
        logger.info(f"Rule Added: [{category}] If {condition} -> {requirement}")
        return rule_id

    def add_documentation(self, title: str, content: str, author: str = "Engineer") -> str:
        """
        User Input Interface: Add general knowledge.
        """
        doc_id = f"DOC_{uuid.uuid4().hex[:6].upper()}"
        doc = DocumentationEntry(doc_id, title, content, [], author)
        self.docs[doc_id] = doc
        logger.info(f"Documentation Added: '{title}' by {author}")
        return doc_id

    def query_rules(self, context_keywords: List[str]) -> List[EngineeringRule]:
        """
        Used by the Monitoring System to find relevant rules for validation.
        """
        matches = []
        for rule in self.rules.values():
            # Simple keyword matching for prototype
            if any(k.lower() in rule.category.lower() or k.lower() in rule.condition.lower() for k in context_keywords):
                matches.append(rule)
        return matches

    def validate_plan(self, proposed_plan: Dict[str, Any]) -> List[str]:
        """
        The Core Validation Logic.
        Checks a proposed plan (dictionary) against stored rules.
        """
        violations = []
        
        # Example validation logic (Prototype)
        # In a real system, this would parse 'condition' strings or use an embedded rules engine.
        
        material = proposed_plan.get("material", "Unknown")
        rpm = proposed_plan.get("rpm", 0)
        
        # Check against all rules
        for rule in self.rules.values():
            if rule.category == "Material" and material in rule.condition:
                # "Material == Titanium" -> rule checks if "Titanium" is in the plan's material
                
                # Hardcoded logic parser for the demo
                if "RPM <" in rule.requirement:
                    limit = int(rule.requirement.split("<")[1].strip())
                    if rpm >= limit:
                        violations.append(f"[CRITICAL] Rule Violation ({rule.rule_id}): {rule.condition} requires {rule.requirement}, but Plan has RPM={rpm}")
                        
        return violations

# Singleton instance for the app to use
global_kb = KnowledgeBase()

if __name__ == "__main__":
    # Test
    kb = KnowledgeBase()
    kb.add_rule("Material", "Titanium", "RPM < 2000", "High")
    
    plan = {"material": "Titanium", "rpm": 3000}
    errors = kb.validate_plan(plan)
    print(errors)

class AuditorWrapper:
    """
    The Shadow Council Member: 'The Auditor'.
    Listens to the MessageBus and validates plans against the Ground Truth.
    """
    def __init__(self):
        self.kb = global_kb
        # Pre-seed some rules for the demo
        self.kb.add_rule("Material", "Titanium", "RPM < 3000", "High")
        
        if global_bus:
            global_bus.subscribe("DRAFT_PLAN", self.audit_compliance)
            logger.info("The Auditor has joined the Council.")

    async def audit_compliance(self, msg: Message):
        """
        Triggered when 'The Creator' proposes a plan.
        """
        plan = msg.payload
        logger.info(f"Auditor validating plan for compliance...")
        
        violations = self.kb.validate_plan(plan)
        
        if violations:
            status = "FAIL"
            reason = "; ".join(violations)
        else:
            status = "PASS"
            reason = "Compliant with all Engineering Rules."
            
        result = {
            "status": status,
            "reason": reason,
            "violations": violations
        }
        
        await global_bus.publish("VALIDATION_RESULT", result, sender_id="AUDITOR")
