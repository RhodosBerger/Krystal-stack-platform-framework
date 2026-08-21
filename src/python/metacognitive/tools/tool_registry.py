"""
GAMESA Metacognitive - Tool Registry

Registry for managing tools available to the LLM.
Supports dynamic tool registration and invocation.
"""

from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import inspect


@dataclass
class ToolParameter:
    """Parameter definition for a tool."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None


@dataclass
class ToolDefinition:
    """Tool definition for LLM function calling."""
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable


class BaseTool(ABC):
    """Base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool."""
        pass

    def to_openai_format(self) -> Dict:
        """Convert to OpenAI function calling format."""
        parameters = self.get_parameters()

        properties = {}
        required = []

        for param in parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class ToolRegistry:
    """
    Registry for managing tools.

    Supports:
    - Dynamic tool registration
    - Tool invocation
    - Format conversion (OpenAI, Anthropic, etc.)
    """

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        print(f"Tool registered: {tool.name}")

    def unregister(self, tool_name: str):
        """Unregister a tool."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            print(f"Tool unregistered: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        try:
            result = tool.execute(**kwargs)
            return {
                "success": True,
                "result": result,
                "tool": tool_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }

    def to_openai_format(self) -> List[Dict]:
        """Convert all tools to OpenAI function calling format."""
        return [tool.to_openai_format() for tool in self.tools.values()]

    def to_anthropic_format(self) -> List[Dict]:
        """Convert all tools to Anthropic tool format."""
        tools = []
        for tool in self.tools.values():
            parameters = tool.get_parameters()

            properties = {}
            required = []

            for param in parameters:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description
                }
                if param.required:
                    required.append(param.name)

            tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            })

        return tools

    def parse_tool_call(self, tool_call: Dict) -> Dict:
        """
        Parse tool call from LLM response.

        Args:
            tool_call: Tool call dict from LLM

        Returns:
            Parsed arguments
        """
        if "function" in tool_call:
            # OpenAI format
            name = tool_call["function"]["name"]
            args_str = tool_call["function"]["arguments"]
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            return {"name": name, "arguments": args}
        elif "input" in tool_call:
            # Anthropic format
            name = tool_call["name"]
            args = tool_call["input"]
            return {"name": name, "arguments": args}
        else:
            raise ValueError("Unknown tool call format")


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry
