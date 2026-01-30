"""
Parameter Registry 📋
Responsibility:
1. Store all system parameters with self-documenting metadata.
2. Provide validation, descriptions, and LLM-friendly representations.
3. Enable natural language queries about configuration.
"""
import uuid
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from datetime import datetime, timezone

class ParamType(Enum):
    STRING = "STRING"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    LIST = "LIST"
    ENUM = "ENUM"

class Parameter:
    def __init__(
        self,
        key: str,
        value: Any,
        param_type: ParamType,
        description: str,
        category: str = "general",
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        enum_options: Optional[List[str]] = None,
        unit: Optional[str] = None,
        editable: bool = True,
        llm_hint: Optional[str] = None
    ):
        self.key = key
        self.value = value
        self.param_type = param_type
        self.description = description
        self.category = category
        self.min_val = min_val
        self.max_val = max_val
        self.enum_options = enum_options or []
        self.unit = unit
        self.editable = editable
        self.llm_hint = llm_hint or description
        self.last_modified = datetime.now(timezone.utc).isoformat()
        self.modification_history: List[Dict[str, Any]] = []

    def validate(self, new_value: Any) -> Dict[str, Any]:
        """Validates a value against this parameter's rules."""
        errors = []
        
        if self.param_type == ParamType.INTEGER:
            if not isinstance(new_value, int):
                try: new_value = int(new_value)
                except: errors.append(f"Must be an integer")
            if self.min_val is not None and new_value < self.min_val:
                errors.append(f"Must be >= {self.min_val}")
            if self.max_val is not None and new_value > self.max_val:
                errors.append(f"Must be <= {self.max_val}")
                
        elif self.param_type == ParamType.FLOAT:
            if not isinstance(new_value, (int, float)):
                try: new_value = float(new_value)
                except: errors.append(f"Must be a number")
            if self.min_val is not None and new_value < self.min_val:
                errors.append(f"Must be >= {self.min_val}")
            if self.max_val is not None and new_value > self.max_val:
                errors.append(f"Must be <= {self.max_val}")
                
        elif self.param_type == ParamType.BOOLEAN:
            if not isinstance(new_value, bool):
                if str(new_value).lower() in ['true', '1', 'yes']: new_value = True
                elif str(new_value).lower() in ['false', '0', 'no']: new_value = False
                else: errors.append("Must be true or false")
                
        elif self.param_type == ParamType.ENUM:
            if new_value not in self.enum_options:
                errors.append(f"Must be one of: {', '.join(self.enum_options)}")
        
        return {"valid": len(errors) == 0, "errors": errors, "coerced_value": new_value}

    def set_value(self, new_value: Any) -> Dict[str, Any]:
        """Sets the value after validation."""
        if not self.editable:
            return {"success": False, "error": "Parameter is read-only"}
        
        validation = self.validate(new_value)
        if not validation["valid"]:
            return {"success": False, "errors": validation["errors"]}
        
        old_value = self.value
        self.value = validation["coerced_value"]
        self.last_modified = datetime.now(timezone.utc).isoformat()
        self.modification_history.append({
            "from": old_value,
            "to": self.value,
            "at": self.last_modified
        })
        
        return {"success": True, "old_value": old_value, "new_value": self.value}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "type": self.param_type.value,
            "description": self.description,
            "category": self.category,
            "min": self.min_val,
            "max": self.max_val,
            "enum_options": self.enum_options,
            "unit": self.unit,
            "editable": self.editable,
            "llm_hint": self.llm_hint,
            "last_modified": self.last_modified
        }

    def to_llm_summary(self) -> str:
        """Returns an LLM-friendly description of this parameter."""
        summary = f"**{self.key}** ({self.category}): {self.description}"
        summary += f"\n  - Current Value: {self.value}"
        if self.unit: summary += f" {self.unit}"
        summary += f"\n  - Type: {self.param_type.value}"
        if self.min_val is not None or self.max_val is not None:
            summary += f"\n  - Range: [{self.min_val or '-∞'} to {self.max_val or '∞'}]"
        if self.enum_options:
            summary += f"\n  - Options: {', '.join(self.enum_options)}"
        summary += f"\n  - Editable: {'Yes' if self.editable else 'No (Read-Only)'}"
        return summary


class ParameterRegistry:
    def __init__(self):
        self._params: Dict[str, Parameter] = {}
        self._seed_defaults()

    def _seed_defaults(self):
        """Seeds the registry with default system parameters."""
        defaults = [
            Parameter("max_spindle_rpm", 8000, ParamType.INTEGER, "Maximum spindle RPM for safety", "machining", 100, 24000, unit="RPM", llm_hint="Maximum rotational speed of the spindle"),
            Parameter("default_feed_rate", 500, ParamType.FLOAT, "Default cutting feed rate", "machining", 10, 5000, unit="mm/min"),
            Parameter("safety_mode", True, ParamType.BOOLEAN, "Enable safety checks before G-code execution", "safety"),
            Parameter("coordinate_system", "G90", ParamType.ENUM, "Default coordinate system mode", "machining", enum_options=["G90", "G91"]),
            Parameter("notification_poll_interval", 30, ParamType.INTEGER, "Interval for polling notifications", "ui", 5, 300, unit="seconds"),
            Parameter("max_payload_size_mb", 50, ParamType.INTEGER, "Maximum payload file size", "export", 1, 500, unit="MB"),
            Parameter("llm_temperature", 0.7, ParamType.FLOAT, "LLM generation temperature", "llm", 0.0, 2.0),
            Parameter("audit_retention_days", 90, ParamType.INTEGER, "Days to retain audit logs", "system", 7, 365, unit="days"),
            Parameter("default_material", "Aluminum 6061", ParamType.STRING, "Default material for new operations", "materials"),
            Parameter("precision_decimal_places", 4, ParamType.INTEGER, "Decimal precision for coordinates", "machining", 2, 8),
        ]
        for p in defaults:
            self._params[p.key] = p

    def register(self, param: Parameter) -> str:
        """Registers a new parameter."""
        self._params[param.key] = param
        return param.key

    def get(self, key: str) -> Optional[Parameter]:
        return self._params.get(key)

    def get_value(self, key: str) -> Any:
        param = self._params.get(key)
        return param.value if param else None

    def set_value(self, key: str, value: Any) -> Dict[str, Any]:
        param = self._params.get(key)
        if not param:
            return {"success": False, "error": "Parameter not found"}
        return param.set_value(value)

    def list_all(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._params.values()]

    def list_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._params.values() if p.category == category]

    def get_categories(self) -> List[str]:
        return list(set(p.category for p in self._params.values()))

    def get_llm_summary(self) -> str:
        """Returns a full LLM-readable summary of all parameters."""
        lines = ["# System Configuration Parameters\n"]
        for category in sorted(self.get_categories()):
            lines.append(f"\n## Category: {category.upper()}\n")
            for p in self._params.values():
                if p.category == category:
                    lines.append(p.to_llm_summary() + "\n")
        return "\n".join(lines)

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Searches parameters by key, description, or category."""
        query_lower = query.lower()
        return [p.to_dict() for p in self._params.values() 
                if query_lower in p.key.lower() 
                or query_lower in p.description.lower() 
                or query_lower in p.category.lower()]


# Global Instance
parameter_registry = ParameterRegistry()
