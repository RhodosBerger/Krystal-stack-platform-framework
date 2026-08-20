"""
LLM Backends - Connect to real LLM providers

Supports:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local Ollama
- Mock (for testing)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum, auto
import json
import os
import time


class LLMProvider(Enum):
    OPENAI = auto()
    ANTHROPIC = auto()
    OLLAMA = auto()
    MOCK = auto()


@dataclass
class LLMMessage:
    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    content: str
    tokens_used: int = 0
    latency_ms: float = 0
    model: str = ""
    finish_reason: str = "stop"


class LLMBackend(ABC):
    """Base LLM backend."""

    @abstractmethod
    def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend."""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = "https://api.openai.com/v1"

    def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        try:
            import urllib.request
            import urllib.error

            start = time.time()

            data = json.dumps({
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "max_tokens": kwargs.get("max_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.7)
            }).encode()

            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())

            latency = (time.time() - start) * 1000

            return LLMResponse(
                content=result["choices"][0]["message"]["content"],
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
                latency_ms=latency,
                model=self.model,
                finish_reason=result["choices"][0].get("finish_reason", "stop")
            )
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", model=self.model)

    def is_available(self) -> bool:
        return bool(self.api_key)


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""

    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"

    def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        try:
            import urllib.request

            start = time.time()

            # Extract system message
            system = ""
            chat_messages = []
            for m in messages:
                if m.role == "system":
                    system = m.content
                else:
                    chat_messages.append({"role": m.role, "content": m.content})

            data = json.dumps({
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", 1024),
                "system": system,
                "messages": chat_messages
            }).encode()

            req = urllib.request.Request(
                f"{self.base_url}/messages",
                data=data,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())

            latency = (time.time() - start) * 1000

            return LLMResponse(
                content=result["content"][0]["text"],
                tokens_used=result.get("usage", {}).get("input_tokens", 0) +
                           result.get("usage", {}).get("output_tokens", 0),
                latency_ms=latency,
                model=self.model,
                finish_reason=result.get("stop_reason", "stop")
            )
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", model=self.model)

    def is_available(self) -> bool:
        return bool(self.api_key)


class OllamaBackend(LLMBackend):
    """Local Ollama backend."""

    def __init__(self, model: str = "llama2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        try:
            import urllib.request

            start = time.time()

            data = json.dumps({
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False
            }).encode()

            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())

            latency = (time.time() - start) * 1000

            return LLMResponse(
                content=result["message"]["content"],
                tokens_used=result.get("eval_count", 0),
                latency_ms=latency,
                model=self.model
            )
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", model=self.model)

    def is_available(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except:
            return False


class MockBackend(LLMBackend):
    """Mock backend for testing."""

    def __init__(self):
        self.call_count = 0
        self.responses = {}

    def complete(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        self.call_count += 1
        last_msg = messages[-1].content if messages else ""

        # Generate contextual mock response
        if "code" in last_msg.lower() or "function" in last_msg.lower():
            content = "def generated_function():\n    return 'mock result'"
        elif "plan" in last_msg.lower():
            content = "1. Analyze requirements\n2. Design solution\n3. Implement\n4. Test"
        elif "review" in last_msg.lower():
            content = "Code review: Looks good. Score: 8/10"
        else:
            content = f"Mock response to: {last_msg[:50]}..."

        return LLMResponse(
            content=content,
            tokens_used=len(content.split()),
            latency_ms=10,
            model="mock"
        )

    def is_available(self) -> bool:
        return True

    def set_response(self, trigger: str, response: str):
        """Set custom response for trigger."""
        self.responses[trigger] = response


class LLMRouter:
    """Routes requests to available backends."""

    def __init__(self):
        self.backends: Dict[LLMProvider, LLMBackend] = {}
        self.default_provider = LLMProvider.MOCK
        self._setup_backends()

    def _setup_backends(self):
        """Initialize available backends."""
        self.backends[LLMProvider.MOCK] = MockBackend()

        # Try to setup real backends
        openai = OpenAIBackend()
        if openai.is_available():
            self.backends[LLMProvider.OPENAI] = openai
            self.default_provider = LLMProvider.OPENAI

        anthropic = AnthropicBackend()
        if anthropic.is_available():
            self.backends[LLMProvider.ANTHROPIC] = anthropic
            if self.default_provider == LLMProvider.MOCK:
                self.default_provider = LLMProvider.ANTHROPIC

        ollama = OllamaBackend()
        if ollama.is_available():
            self.backends[LLMProvider.OLLAMA] = ollama

    def complete(self, messages: List[LLMMessage],
                 provider: LLMProvider = None, **kwargs) -> LLMResponse:
        """Route completion request."""
        provider = provider or self.default_provider
        backend = self.backends.get(provider, self.backends[LLMProvider.MOCK])
        return backend.complete(messages, **kwargs)

    def available_providers(self) -> List[LLMProvider]:
        """List available providers."""
        return [p for p, b in self.backends.items() if b.is_available()]


# Factory
def create_llm_router() -> LLMRouter:
    return LLMRouter()


# Convenience function
def quick_complete(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Quick completion with auto-routing."""
    router = create_llm_router()
    messages = [
        LLMMessage("system", system),
        LLMMessage("user", prompt)
    ]
    return router.complete(messages).content
