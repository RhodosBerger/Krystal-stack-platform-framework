#!/usr/bin/env python3
"""
DYNAMIC FORM BUILDER - Backend Boolean Logic & Configuration System
Generuje dynamické formuláre na základe CMS definícií.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class FieldType(Enum):
    """Typy UI prvkov"""
    TEXT = "text"
    NUMBER = "number"
    SLIDER = "slider"
    TOGGLE = "toggle"
    SELECT = "select"
    MULTISELECT = "multiselect"
    TAGS = "tags"
    COLOR = "color"
    DATE = "date"
    TIME = "time"
    TEXTAREA = "textarea"
    FILE = "file"

class ValidationRule(Enum):
    """Validačné pravidlá"""
    REQUIRED = "required"
    MIN = "min"
    MAX = "max"
    PATTERN = "pattern"
    EMAIL = "email"
    URL = "url"
    NUMERIC = "numeric"

@dataclass
class FormField:
    """Definícia jedného form fieldu"""
    id: str
    type: FieldType
    label: str
    default_value: Any = None
    placeholder: str = ""
    help_text: str = ""
    required: bool = False
    
    # Type-specific options
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: List[Dict[str, str]] = field(default_factory=list)  # pre select/multiselect
    pattern: Optional[str] = None
    
    # Conditional display
    depends_on: Optional[str] = None  # ID iného fieldu
    show_if: Optional[Any] = None  # Hodnota na ktorú sa má kontrolovať
    
    # Grouping
    section: str = "General"
    order: int = 0

@dataclass
class FormSection:
    """Sekcia formulára (grouping)"""
    id: str
    title: str
    description: str = ""
    icon: str = "⚙️"
    order: int = 0
    collapsed: bool = False
    fields: List[FormField] = field(default_factory=list)

class BooleanLogicEngine:
    """Engine pre vyhodnocovanie boolean logiky"""
    
    @staticmethod
    def evaluate(condition: Dict[str, Any], values: Dict[str, Any]) -> bool:
        """
        Vyhodnotí boolean podmienku.
        
        Condition format:
        {
            "field": "enable_safety",
            "operator": "==",
            "value": True
        }
        
        Advanced:
        {
            "and": [
                {"field": "mode", "operator": "==", "value": "AUTO"},
                {"field": "dopamine", "operator": ">", "value": 50}
            ]
        }
        """
        # Simple condition
        if "field" in condition:
            field_val = values.get(condition["field"])
            operator = condition["operator"]
            expected = condition["value"]
            
            if operator == "==":
                return field_val == expected
            elif operator == "!=":
                return field_val != expected
            elif operator == ">":
                return field_val > expected
            elif operator == "<":
                return field_val < expected
            elif operator == ">=":
                return field_val >= expected
            elif operator == "<=":
                return field_val <= expected
            elif operator == "in":
                return field_val in expected
            elif operator == "not_in":
                return field_val not in expected
        
        # Compound conditions
        if "and" in condition:
            return all(BooleanLogicEngine.evaluate(c, values) for c in condition["and"])
        
        if "or" in condition:
            return any(BooleanLogicEngine.evaluate(c, values) for c in condition["or"])
        
        if "not" in condition:
            return not BooleanLogicEngine.evaluate(condition["not"], values)
        
        return False

class DynamicFormBuilder:
    """Hlavný builder pre dynamické formuláre"""
    
    def __init__(self):
        self.sections: List[FormSection] = []
        self.logic_engine = BooleanLogicEngine()
    
    def add_section(self, section: FormSection):
        """Pridaj sekciu"""
        self.sections.append(section)
        self.sections.sort(key=lambda s: s.order)
    
    def add_field(self, section_id: str, field: FormField):
        """Pridaj field do sekcie"""
        section = next((s for s in self.sections if s.id == section_id), None)
        if section:
            section.fields.append(field)
            section.fields.sort(key=lambda f: f.order)
    
    def build_config(self) -> Dict[str, Any]:
        """
        Vygeneruj JSON config pre frontend.
        """
        return {
            "sections": [
                {
                    "id": section.id,
                    "title": section.title,
                    "description": section.description,
                    "icon": section.icon,
                    "collapsed": section.collapsed,
                    "fields": [
                        {
                            "id": field.id,
                            "type": field.type.value,
                            "label": field.label,
                            "default": field.default_value,
                            "placeholder": field.placeholder,
                            "help": field.help_text,
                            "required": field.required,
                            "min": field.min_value,
                            "max": field.max_value,
                            "step": field.step,
                            "options": field.options,
                            "pattern": field.pattern,
                            "depends_on": field.depends_on,
                            "show_if": field.show_if
                        }
                        for field in section.fields
                    ]
                }
                for section in self.sections
            ]
        }
    
    def validate(self, values: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validuj submittnuté hodnoty.
        Returns: Dict[field_id, List[error_messages]]
        """
        errors = {}
        
        for section in self.sections:
            for field in section.fields:
                value = values.get(field.id)
                field_errors = []
                
                # Check required
                if field.required and (value is None or value == ""):
                    field_errors.append(f"{field.label} je povinný")
                
                # Check type-specific
                if value is not None:
                    if field.type == FieldType.NUMBER or field.type == FieldType.SLIDER:
                        if field.min_value is not None and value < field.min_value:
                            field_errors.append(f"Minimum je {field.min_value}")
                        if field.max_value is not None and value > field.max_value:
                            field_errors.append(f"Maximum je {field.max_value}")
                    
                    if field.pattern and isinstance(value, str):
                        import re
                        if not re.match(field.pattern, value):
                            field_errors.append(f"Neplatný formát")
                
                if field_errors:
                    errors[field.id] = field_errors
        
        return errors

# ===== PRÍKLAD KONFIGURÁCIE =====

def create_safety_config() -> DynamicFormBuilder:
    """Vytvor konfiguráciu pre Safety & Control panel"""
    
    builder = DynamicFormBuilder()
    
    # Sekcia 1: Základné nastavenia
    section_basic = FormSection(
        id="basic",
        title="Základné Nastavenia",
        description="Hlavné safety parametre",
        icon="🛡️",
        order=1
    )
    
    section_basic.fields = [
        FormField(
            id="enable_safety",
            type=FieldType.TOGGLE,
            label="Povoliť Safety Monitoring",
            default_value=True,
            required=True,
            help_text="Ak vypnuté, všetky safety kontroly sú deaktivované!",
            order=1
        ),
        FormField(
            id="safety_level",
            type=FieldType.SELECT,
            label="Safety Level",
            default_value="BALANCED",
            options=[
                {"value": "CONSERVATIVE", "label": "Konzervat  (slowest, safest)"},
                {"value": "BALANCED", "label": "Vyvážený (recommended)"},
                {"value": "AGGRESSIVE", "label": "Agresívny (fastest, risky)"}
            ],
            depends_on="enable_safety",
            show_if=True,
            order=2
        ),
        FormField(
            id="max_load_percent",
            type=FieldType.SLIDER,
            label="Maximálna Záťaž Spindle (%)",
            default_value=95.0,
            min_value=50.0,
            max_value=100.0,
            step=1.0,
            help_text="Červený alarm pri prekročení",
            order=3
        ),
        FormField(
            id="max_vibration_g",
            type=FieldType.SLIDER,
            label="Maximálna Vibrácia (g)",
            default_value=0.2,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            order=4
        )
    ]
    
    builder.add_section(section_basic)
    
    # Sekcia 2: Dopamine Engine
    section_dopamine = FormSection(
        id="dopamine",
        title="Dopamine Engine Tuning",
        description="Pokročilé nastavenia reward systému",
        icon="🧠",
        order=2
    )
    
    section_dopamine.fields = [
        FormField(
            id="dopamine_sensitivity",
            type=FieldType.SLIDER,
            label="Citlivosť Dopamínu",
            default_value=70.0,
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            help_text="Vyššia hodnota = reaguje rýchlejšie na zmeny",
            order=1
        ),
        FormField(
            id="risk_aversion",
            type=FieldType.SLIDER,
            label="Risk Aversion",
            default_value=85.0,
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            help_text="Vyššia hodnota = konzervat approach",
            order=2
        ),
        FormField(
            id="learning_rate",
            type=FieldType.NUMBER,
            label="Learning Rate",
            default_value=0.01,
            min_value=0.001,
            max_value=0.1,
            step=0.001,
            help_text="Rýchlosť učenia sa (advanced)",
            order=3
        )
    ]
    
    builder.add_section(section_dopamine)
    
    # Sekcia 3: Alarmy
    section_alerts = FormSection(
        id="alerts",
        title="Alert Configuration",
        description="Email/SMS notifikácie",
        icon="🔔",
        order=3
    )
    
    section_alerts.fields = [
        FormField(
            id="enable_alerts",
            type=FieldType.TOGGLE,
            label="Povoliť Alarmy",
            default_value=True,
            order=1
        ),
        FormField(
            id="alert_emails",
            type=FieldType.TAGS,
            label="Email Adresy",
            default_value=[],
            placeholder="email@example.com",
            help_text="Pridaj email a stlač Enter",
            depends_on="enable_alerts",
            show_if=True,
            order=2
        ),
        FormField(
            id="alert_levels",
            type=FieldType.MULTISELECT,
            label="Alert Typy",
            default_value=["RED", "AMBER"],
            options=[
                {"value": "RED", "label": "🔴 RED (Critical)"},
                {"value": "AMBER", "label": "🟡 AMBER (Warning)"},
                {"value": "GREEN", "label": "🟢 GREEN (Info)"}
            ],
            depends_on="enable_alerts",
            show_if=True,
            order=3
        )
    ]
    
    builder.add_section(section_alerts)
    
    return builder

# API Endpoint example
if __name__ == "__main__":
    import json
    
    # Vytvor konfiguráciu
    builder = create_safety_config()
    config = builder.build_config()
    
    print("=== GENERATED CONFIG ===")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    # Test validácie
    test_values = {
        "enable_safety": True,
        "safety_level": "BALANCED",
        "max_load_percent": 95,
        "max_vibration_g": 0.2,
        "dopamine_sensitivity": 70,
        "risk_aversion": 85,
        "learning_rate": 0.01,
        "enable_alerts": True,
        "alert_emails": ["admin@factory.sk"],
        "alert_levels": ["RED", "AMBER"]
    }
    
    errors = builder.validate(test_values)
    print("\n=== VALIDATION ===")
    if errors:
        print("Errors:", errors)
    else:
        print("✅ All valid!")
    
    # Test boolean logic
    logic = BooleanLogicEngine()
    condition = {
        "and": [
            {"field": "enable_safety", "operator": "==", "value": True},
            {"field": "max_load_percent", "operator": ">", "value": 90}
        ]
    }
    result = logic.evaluate(condition, test_values)
    print(f"\n=== BOOLEAN LOGIC ===")
    print(f"Condition: Safety enabled AND load > 90%")
    print(f"Result: {result}")
