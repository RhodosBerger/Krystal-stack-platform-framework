"""
GAMESA Metacognitive - Base LLM Connector

Abstract base class for LLM backend integrations.
Provides unified interface for different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMMessage:
    """Message in conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    finish_reason: str  # "stop", "length", "tool_calls"
    usage: Dict[str, int]  # {prompt_tokens, completion_tokens, total_tokens}
    model: str
    tool_calls: Optional[List[Dict]] = None


@dataclass
class LLMConfig:
    """Configuration for LLM connector."""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30


class BaseLLMConnector(ABC):
    """
    Abstract base class for LLM connectors.

    All LLM providers must implement this interface.
    """

    def __init__(self, config: LLMConfig):
        """Initialize connector with configuration."""
        self.config = config
        self.conversation_history: List[LLMMessage] = []

    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from LLM.

        Args:
            messages: Conversation history
            tools: Available tools (for function calling)
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse with content and metadata
        """
        pass

    @abstractmethod
    def stream_generate(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ):
        """
        Stream response from LLM (generator).

        Args:
            messages: Conversation history
            tools: Available tools
            **kwargs: Provider-specific parameters

        Yields:
            Chunks of response text
        """
        pass

    def add_message(self, role: str, content: str, **kwargs):
        """Add message to conversation history."""
        message = LLMMessage(role=role, content=content, **kwargs)
        self.conversation_history.append(message)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[LLMMessage]:
        """Get conversation history."""
        return self.conversation_history.copy()

    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate connection to LLM provider.

        Returns:
            True if connection is valid
        """
        pass

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Default implementation: ~4 chars per token (rough estimate).
        Override for provider-specific tokenization.
        """
        return len(text) // 4


class LLMConnectorFactory:
    """Factory for creating LLM connectors."""

    _connectors = {}

    @classmethod
    def register(cls, provider: LLMProvider, connector_class: type):
        """Register a connector class for a provider."""
        cls._connectors[provider] = connector_class

    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMConnector:
        """Create connector instance based on config."""
        connector_class = cls._connectors.get(config.provider)
        if not connector_class:
            raise ValueError(f"No connector registered for provider: {config.provider}")

        return connector_class(config)

    @classmethod
    def list_providers(cls) -> List[LLMProvider]:
        """List all registered providers."""
        return list(cls._connectors.keys())
