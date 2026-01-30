"""
Element Template Engine 🧱🛠️
Responsibility:
1. Assembly of custom files based on "Elements".
2. Multi-format emitters (G-Code, JSON, Markdown).
3. Dynamic property injection for assembled elements.
"""
import json
import os
from typing import List, Dict, Any, Optional

class Element:
    def __init__(self, id: str, type: str, content: Any, metadata: Dict[str, Any] = None):
        self.id = id
        self.type = type # "GEOMETRY", "TOOL", "PROVENANCE", "METRIC"
        self.content = content
        self.metadata = metadata or {}

class ElementTemplateEngine:
    def __init__(self):
        self.elements: Dict[str, Element] = {}

    def register_element(self, element: Element):
        self.elements[element.id] = element

    def assemble_document(self, element_ids: List[str], format: str = "JSON") -> str:
        """
        Assembles a document from a list of elements.
        """
        selected_elements = [self.elements[eid] for eid in element_ids if eid in self.elements]
        
        if format.upper() == "JSON":
            return self._emit_json(selected_elements)
        elif format.upper() == "GCODE":
            return self._emit_gcode(selected_elements)
        elif format.upper() == "MARKDOWN":
            return self._emit_markdown(selected_elements)
        else:
            return f"Unsupported format: {format}"

    def _emit_json(self, elements: List[Element]) -> str:
        data = {
            "assembly_type": "CUSTOM_ELEMENT_GEN",
            "element_count": len(elements),
            "elements": [
                {"id": e.id, "type": e.type, "content": e.content, "metadata": e.metadata}
                for e in elements
            ]
        }
        return json.dumps(data, indent=4)

    def _emit_gcode(self, elements: List[Element]) -> str:
        lines = [
            "%",
            "(ASSEMBLY: CUSTOM_GENERATED)",
            "G21",
            "G90"
        ]
        for e in elements:
            if e.type == "GEOMETRY":
                lines.append(f"(ELEMENT: {e.id})")
                if isinstance(e.content, list):
                    lines.extend(e.content)
                else:
                    lines.append(str(e.content))
            elif e.type == "PROVENANCE":
                lines.append(f"(ORIGIN: {e.content})")
        
        lines.append("M30")
        lines.append("%")
        return "\n".join(lines)

    def _emit_markdown(self, elements: List[Element]) -> str:
        lines = ["# Custom Manufacturing Document\n"]
        for e in elements:
            lines.append(f"## Element: {e.id} ({e.type})")
            lines.append(f"```json\n{json.dumps(e.content, indent=2)}\n```\n")
        return "\n".join(lines)

# Global Instance
template_engine = ElementTemplateEngine()
