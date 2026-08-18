"""
GAMESA Metacognitive Module

LLM-powered metacognitive reasoning for policy generation and self-reflection.
"""

from .metacognitive_engine import (
    MetacognitiveEngine,
    MetacognitiveAnalysis,
    PolicyProposal,
    create_metacognitive_engine
)

from .bot_core import (
    ConversationManager,
    ConversationTurn,
    ConversationContext,
    MultiModalInputProcessor
)

from .llm_integrations.base_connector import (
    BaseLLMConnector,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMProvider,
    LLMConnectorFactory
)

from .tools.tool_registry import (
    ToolRegistry,
    BaseTool,
    ToolParameter,
    get_tool_registry
)

__all__ = [
    # Engine
    "MetacognitiveEngine",
    "MetacognitiveAnalysis",
    "PolicyProposal",
    "create_metacognitive_engine",

    # Bot Core
    "ConversationManager",
    "ConversationTurn",
    "ConversationContext",
    "MultiModalInputProcessor",

    # LLM
    "BaseLLMConnector",
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    "LLMProvider",
    "LLMConnectorFactory",

    # Tools
    "ToolRegistry",
    "BaseTool",
    "ToolParameter",
    "get_tool_registry",
]
