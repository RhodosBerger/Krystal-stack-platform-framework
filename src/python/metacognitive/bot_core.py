"""
GAMESA Metacognitive - Bot Core

Conversation management and orchestration for the metacognitive system.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from .llm_integrations.base_connector import BaseLLMConnector, LLMMessage, LLMResponse
from .tools.tool_registry import ToolRegistry, get_tool_registry


@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    timestamp: float
    role: str
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_results: Optional[List[Dict]] = None


@dataclass
class ConversationContext:
    """Context for a conversation."""
    conversation_id: str
    created_at: float
    turns: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """
    Manages conversations with LLM including tool use.

    Features:
    - Conversation history tracking
    - Tool call orchestration
    - Context window management
    - Multi-turn reasoning
    """

    def __init__(
        self,
        llm_connector: BaseLLMConnector,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        max_history: int = 50
    ):
        """
        Initialize conversation manager.

        Args:
            llm_connector: LLM backend connector
            tool_registry: Tool registry instance
            system_prompt: System prompt for the LLM
            max_history: Maximum conversation turns to keep
        """
        self.llm = llm_connector
        self.tools = tool_registry or get_tool_registry()
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_history = max_history

        self.context = ConversationContext(
            conversation_id=f"conv_{datetime.now().timestamp()}",
            created_at=datetime.now().timestamp()
        )

        # Add system prompt
        self._add_turn("system", self.system_prompt)

    def _default_system_prompt(self) -> str:
        """Default system prompt for GAMESA metacognitive."""
        return """You are the Metacognitive Interface for GAMESA, an adaptive system optimization framework.

Your role is to:
1. Analyze system performance data and telemetry
2. Identify patterns and correlations in resource usage
3. Propose optimization policies based on evidence
4. Explain your reasoning with confidence scores

You have access to tools for:
- Precise calculations (calculator)
- Telemetry analysis (telemetry_analyzer)

When proposing policies, always:
- Provide clear justification based on data
- Include confidence scores (0.0-1.0)
- Consider safety implications
- Suggest shadow evaluation for risky changes

Be concise, analytical, and data-driven."""

    def _add_turn(self, role: str, content: str, **kwargs):
        """Add a turn to conversation history."""
        turn = ConversationTurn(
            timestamp=datetime.now().timestamp(),
            role=role,
            content=content,
            **kwargs
        )
        self.context.turns.append(turn)

        # Trim history if too long (keep system prompt)
        if len(self.context.turns) > self.max_history + 1:
            # Keep system prompt (first turn) + recent turns
            self.context.turns = [self.context.turns[0]] + self.context.turns[-(self.max_history):]

    def _build_messages(self) -> List[LLMMessage]:
        """Build message list from conversation history."""
        messages = []

        for turn in self.context.turns:
            message = LLMMessage(
                role=turn.role,
                content=turn.content,
                tool_calls=turn.tool_calls
            )
            messages.append(message)

            # Add tool results if present
            if turn.tool_results:
                for result in turn.tool_results:
                    tool_message = LLMMessage(
                        role="tool",
                        content=json.dumps(result),
                        tool_call_id=result.get("call_id")
                    )
                    messages.append(tool_message)

        return messages

    def chat(
        self,
        user_message: str,
        enable_tools: bool = True,
        max_tool_iterations: int = 5
    ) -> str:
        """
        Send a message and get response.

        Args:
            user_message: User's message
            enable_tools: Whether to enable tool use
            max_tool_iterations: Maximum tool call iterations

        Returns:
            Assistant's response text
        """
        # Add user message to history
        self._add_turn("user", user_message)

        # Tool use loop
        for iteration in range(max_tool_iterations):
            messages = self._build_messages()

            # Get available tools
            tools = self.tools.to_openai_format() if enable_tools else None

            # Generate response
            response = self.llm.generate(messages, tools=tools)

            # Check if tool calls are requested
            if response.tool_calls and enable_tools:
                # Execute tool calls
                tool_results = []

                for tool_call in response.tool_calls:
                    parsed = self.tools.parse_tool_call(tool_call)
                    tool_name = parsed["name"]
                    tool_args = parsed["arguments"]

                    # Execute tool
                    result = self.tools.execute(tool_name, **tool_args)

                    tool_results.append({
                        "call_id": tool_call.get("id"),
                        "tool": tool_name,
                        "result": result
                    })

                # Add assistant turn with tool calls
                self._add_turn(
                    "assistant",
                    response.content or "",
                    tool_calls=response.tool_calls,
                    tool_results=tool_results
                )

                # Continue loop to get final response

            else:
                # No tool calls, this is final response
                self._add_turn("assistant", response.content)
                return response.content

        # Max iterations reached
        return "Maximum tool iterations reached. Please simplify your request."

    def get_history(self) -> List[ConversationTurn]:
        """Get conversation history."""
        return self.context.turns.copy()

    def clear_history(self):
        """Clear conversation history (keep system prompt)."""
        system_turn = self.context.turns[0] if self.context.turns else None
        self.context.turns = [system_turn] if system_turn else []

    def export_history(self) -> str:
        """Export conversation history as JSON."""
        history_dict = {
            "conversation_id": self.context.conversation_id,
            "created_at": self.context.created_at,
            "turns": [
                {
                    "timestamp": turn.timestamp,
                    "role": turn.role,
                    "content": turn.content,
                    "tool_calls": turn.tool_calls,
                    "tool_results": turn.tool_results
                }
                for turn in self.context.turns
            ]
        }
        return json.dumps(history_dict, indent=2)


class MultiModalInputProcessor:
    """
    Process multimodal inputs (text, image, audio).

    Note: This is a placeholder for future multimodal support.
    Currently focuses on text processing.
    """

    def __init__(self):
        self.supported_modalities = ["text"]

    def process(self, input_data: Any, modality: str = "text") -> Dict:
        """
        Process input based on modality.

        Args:
            input_data: Raw input data
            modality: Input modality type

        Returns:
            Processed input dict
        """
        if modality == "text":
            return {"type": "text", "content": str(input_data)}
        elif modality == "image":
            # Placeholder for image processing
            return {"type": "image", "content": "Image processing not yet implemented"}
        elif modality == "audio":
            # Placeholder for audio processing
            return {"type": "audio", "content": "Audio processing not yet implemented"}
        else:
            raise ValueError(f"Unsupported modality: {modality}")
