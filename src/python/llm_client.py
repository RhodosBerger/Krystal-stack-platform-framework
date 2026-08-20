"""
Unified LLM Client - Equal support for Local and API providers

Both local (Ollama, LM Studio, vLLM) and API (OpenAI, Claude)
providers share the same interface and capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator, Callable
from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import json
import time
import threading
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class ProviderType(Enum):
    """Provider categories."""
    LOCAL = auto()   # Ollama, LM Studio, vLLM, llama.cpp
    API = auto()     # OpenAI, Anthropic, Cohere


@dataclass
class LLMConfig:
    """Unified configuration for any provider."""
    provider: str = "auto"  # auto, ollama, lmstudio, vllm, openai, anthropic
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: int = 60
    retry_count: int = 3
    retry_delay: float = 1.0
    stream: bool = False

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load config from environment."""
        config = cls()

        # Check for local providers first
        if os.getenv("OLLAMA_HOST") or _check_ollama():
            config.provider = "ollama"
            config.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            config.model = os.getenv("OLLAMA_MODEL", "llama2")
        elif os.getenv("LMSTUDIO_HOST") or _check_lmstudio():
            config.provider = "lmstudio"
            config.base_url = os.getenv("LMSTUDIO_HOST", "http://localhost:1234")
            config.model = os.getenv("LMSTUDIO_MODEL", "local-model")
        elif os.getenv("VLLM_HOST"):
            config.provider = "vllm"
            config.base_url = os.getenv("VLLM_HOST", "http://localhost:8000")
            config.model = os.getenv("VLLM_MODEL", "")
        # Then check API providers
        elif os.getenv("OPENAI_API_KEY"):
            config.provider = "openai"
            config.api_key = os.getenv("OPENAI_API_KEY")
            config.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        elif os.getenv("ANTHROPIC_API_KEY"):
            config.provider = "anthropic"
            config.api_key = os.getenv("ANTHROPIC_API_KEY")
            config.model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            config.provider = "gemini"
            config.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            config.model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        else:
            config.provider = "mock"

        # Override with explicit settings
        if os.getenv("LLM_PROVIDER"):
            config.provider = os.getenv("LLM_PROVIDER")
        if os.getenv("LLM_MODEL"):
            config.model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_MAX_TOKENS"):
            config.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
        if os.getenv("LLM_TEMPERATURE"):
            config.temperature = float(os.getenv("LLM_TEMPERATURE"))

        return config


@dataclass
class Message:
    """Chat message."""
    role: str  # system, user, assistant
    content: str


@dataclass
class Response:
    """LLM response with unified metrics."""
    content: str
    model: str = ""
    provider: str = ""
    provider_type: ProviderType = ProviderType.LOCAL
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0
    finish_reason: str = "stop"
    raw: Dict = field(default_factory=dict)

    @property
    def tokens_total(self) -> int:
        return self.tokens_input + self.tokens_output

    @property
    def is_local(self) -> bool:
        return self.provider_type == ProviderType.LOCAL


class LLMProvider(ABC):
    """Base provider interface."""

    provider_type: ProviderType = ProviderType.LOCAL

    @abstractmethod
    def complete(self, messages: List[Message], **kwargs) -> Response:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        """Default non-streaming implementation."""
        response = self.complete(messages, **kwargs)
        yield response.content


# ============================================================
# Local Providers
# ============================================================

class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    provider_type = ProviderType.LOCAL

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"

    def complete(self, messages: List[Message], **kwargs) -> Response:
        start = time.time()

        data = json.dumps({
            "model": kwargs.get("model", self.config.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }).encode()

        try:
            req = Request(f"{self.base_url}/api/chat", data=data,
                         headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read())

            return Response(
                content=result["message"]["content"],
                model=result.get("model", self.config.model),
                provider="ollama",
                provider_type=ProviderType.LOCAL,
                tokens_input=result.get("prompt_eval_count", 0),
                tokens_output=result.get("eval_count", 0),
                latency_ms=(time.time() - start) * 1000,
                raw=result
            )
        except Exception as e:
            return Response(content=f"Error: {e}", provider="ollama",
                          provider_type=ProviderType.LOCAL)

    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        data = json.dumps({
            "model": kwargs.get("model", self.config.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True
        }).encode()

        try:
            req = Request(f"{self.base_url}/api/chat", data=data,
                         headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=self.config.timeout) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line)
                        if "message" in chunk:
                            yield chunk["message"].get("content", "")
        except Exception as e:
            yield f"Error: {e}"

    def is_available(self) -> bool:
        return _check_ollama(self.base_url)


class LMStudioProvider(LLMProvider):
    """LM Studio local provider (OpenAI-compatible API)."""

    provider_type = ProviderType.LOCAL

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:1234"

    def complete(self, messages: List[Message], **kwargs) -> Response:
        start = time.time()

        data = json.dumps({
            "model": kwargs.get("model", self.config.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False
        }).encode()

        try:
            req = Request(f"{self.base_url}/v1/chat/completions", data=data,
                         headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read())

            usage = result.get("usage", {})
            return Response(
                content=result["choices"][0]["message"]["content"],
                model=result.get("model", self.config.model),
                provider="lmstudio",
                provider_type=ProviderType.LOCAL,
                tokens_input=usage.get("prompt_tokens", 0),
                tokens_output=usage.get("completion_tokens", 0),
                latency_ms=(time.time() - start) * 1000,
                raw=result
            )
        except Exception as e:
            return Response(content=f"Error: {e}", provider="lmstudio",
                          provider_type=ProviderType.LOCAL)

    def is_available(self) -> bool:
        return _check_lmstudio(self.base_url)


class VLLMProvider(LLMProvider):
    """vLLM server provider (OpenAI-compatible)."""

    provider_type = ProviderType.LOCAL

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:8000"

    def complete(self, messages: List[Message], **kwargs) -> Response:
        start = time.time()

        data = json.dumps({
            "model": kwargs.get("model", self.config.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature)
        }).encode()

        try:
            req = Request(f"{self.base_url}/v1/chat/completions", data=data,
                         headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read())

            usage = result.get("usage", {})
            return Response(
                content=result["choices"][0]["message"]["content"],
                model=result.get("model", self.config.model),
                provider="vllm",
                provider_type=ProviderType.LOCAL,
                tokens_input=usage.get("prompt_tokens", 0),
                tokens_output=usage.get("completion_tokens", 0),
                latency_ms=(time.time() - start) * 1000,
                raw=result
            )
        except Exception as e:
            return Response(content=f"Error: {e}", provider="vllm",
                          provider_type=ProviderType.LOCAL)

    def is_available(self) -> bool:
        try:
            req = Request(f"{self.base_url}/health")
            with urlopen(req, timeout=2):
                return True
        except:
            return False


# ============================================================
# API Providers
# ============================================================

class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    provider_type = ProviderType.API

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.openai.com/v1"

    def complete(self, messages: List[Message], **kwargs) -> Response:
        start = time.time()

        data = json.dumps({
            "model": kwargs.get("model", self.config.model) or "gpt-3.5-turbo",
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature)
        }).encode()

        try:
            req = Request(f"{self.base_url}/chat/completions", data=data,
                         headers={
                             "Content-Type": "application/json",
                             "Authorization": f"Bearer {self.config.api_key}"
                         })
            with urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read())

            usage = result.get("usage", {})
            return Response(
                content=result["choices"][0]["message"]["content"],
                model=result.get("model", self.config.model),
                provider="openai",
                provider_type=ProviderType.API,
                tokens_input=usage.get("prompt_tokens", 0),
                tokens_output=usage.get("completion_tokens", 0),
                latency_ms=(time.time() - start) * 1000,
                finish_reason=result["choices"][0].get("finish_reason", "stop"),
                raw=result
            )
        except Exception as e:
            return Response(content=f"Error: {e}", provider="openai",
                          provider_type=ProviderType.API)

    def is_available(self) -> bool:
        return bool(self.config.api_key)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    provider_type = ProviderType.API

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.anthropic.com/v1"

    def complete(self, messages: List[Message], **kwargs) -> Response:
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
            "model": kwargs.get("model", self.config.model) or "claude-3-haiku-20240307",
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "system": system,
            "messages": chat_messages
        }).encode()

        try:
            req = Request(f"{self.base_url}/messages", data=data,
                         headers={
                             "Content-Type": "application/json",
                             "x-api-key": self.config.api_key,
                             "anthropic-version": "2023-06-01"
                         })
            with urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read())

            usage = result.get("usage", {})
            return Response(
                content=result["content"][0]["text"],
                model=result.get("model", self.config.model),
                provider="anthropic",
                provider_type=ProviderType.API,
                tokens_input=usage.get("input_tokens", 0),
                tokens_output=usage.get("output_tokens", 0),
                latency_ms=(time.time() - start) * 1000,
                finish_reason=result.get("stop_reason", "stop"),
                raw=result
            )
        except Exception as e:
            return Response(content=f"Error: {e}", provider="anthropic",
                          provider_type=ProviderType.API)

    def is_available(self) -> bool:
        return bool(self.config.api_key)


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    provider_type = ProviderType.API

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta"

    def complete(self, messages: List[Message], **kwargs) -> Response:
        start = time.time()

        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})

        model = kwargs.get("model", self.config.model) or "gemini-1.5-flash"

        body = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature)
            }
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        data = json.dumps(body).encode()
        url = f"{self.base_url}/models/{model}:generateContent?key={self.config.api_key}"

        try:
            req = Request(url, data=data, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read())

            candidate = result.get("candidates", [{}])[0]
            content = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            usage = result.get("usageMetadata", {})

            return Response(
                content=content,
                model=model,
                provider="gemini",
                provider_type=ProviderType.API,
                tokens_input=usage.get("promptTokenCount", 0),
                tokens_output=usage.get("candidatesTokenCount", 0),
                latency_ms=(time.time() - start) * 1000,
                finish_reason=candidate.get("finishReason", "STOP"),
                raw=result
            )
        except Exception as e:
            return Response(content=f"Error: {e}", provider="gemini",
                          provider_type=ProviderType.API)

    def is_available(self) -> bool:
        return bool(self.config.api_key)


class MockProvider(LLMProvider):
    """Mock provider for testing."""

    provider_type = ProviderType.LOCAL

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.call_count = 0

    def complete(self, messages: List[Message], **kwargs) -> Response:
        self.call_count += 1
        last = messages[-1].content if messages else ""

        # Context-aware mock responses
        if "code" in last.lower():
            content = "```python\ndef solution():\n    return 'implemented'\n```"
        elif "plan" in last.lower():
            content = "1. Analyze\n2. Design\n3. Implement\n4. Test"
        else:
            content = f"Mock response to: {last[:50]}..."

        return Response(
            content=content,
            model="mock",
            provider="mock",
            provider_type=ProviderType.LOCAL,
            tokens_input=len(last.split()),
            tokens_output=len(content.split()),
            latency_ms=5
        )

    def is_available(self) -> bool:
        return True


# ============================================================
# Unified Client
# ============================================================

class LLMClient:
    """
    Unified LLM client with equal support for local and API providers.

    Usage:
        # Auto-detect provider
        client = LLMClient()

        # Explicit local
        client = LLMClient(LLMConfig(provider="ollama", model="llama2"))

        # Explicit API
        client = LLMClient(LLMConfig(provider="openai", api_key="sk-..."))

        # Complete
        response = client.complete([Message("user", "Hello")])
        print(response.content)
        print(f"Provider: {response.provider} ({response.provider_type.name})")
    """

    PROVIDERS = {
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider,
        "vllm": VLLMProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "mock": MockProvider,
    }

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig.from_env()
        self.provider = self._init_provider()
        self.metrics = {
            "calls": 0,
            "tokens_total": 0,
            "latency_total_ms": 0,
            "errors": 0
        }
        self._lock = threading.Lock()

    def _init_provider(self) -> LLMProvider:
        """Initialize provider based on config."""
        provider_name = self.config.provider

        if provider_name == "auto":
            # Try local first, then API
            for name in ["ollama", "lmstudio", "vllm", "openai", "anthropic", "gemini", "mock"]:
                provider_cls = self.PROVIDERS.get(name)
                if provider_cls:
                    provider = provider_cls(self.config)
                    if provider.is_available():
                        return provider
            return MockProvider(self.config)

        provider_cls = self.PROVIDERS.get(provider_name, MockProvider)
        return provider_cls(self.config)

    def complete(self, messages: List[Message], **kwargs) -> Response:
        """Send completion request."""
        with self._lock:
            self.metrics["calls"] += 1

        # Retry logic
        last_error = None
        for attempt in range(self.config.retry_count):
            try:
                response = self.provider.complete(messages, **kwargs)

                with self._lock:
                    self.metrics["tokens_total"] += response.tokens_total
                    self.metrics["latency_total_ms"] += response.latency_ms

                return response

            except Exception as e:
                last_error = e
                with self._lock:
                    self.metrics["errors"] += 1
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        return Response(content=f"Error after {self.config.retry_count} retries: {last_error}",
                       provider=self.config.provider)

    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        """Stream completion response."""
        return self.provider.stream(messages, **kwargs)

    def chat(self, prompt: str, system: str = None) -> str:
        """Simple chat interface."""
        messages = []
        if system:
            messages.append(Message("system", system))
        messages.append(Message("user", prompt))
        return self.complete(messages).content

    def is_local(self) -> bool:
        """Check if using local provider."""
        return self.provider.provider_type == ProviderType.LOCAL

    def is_api(self) -> bool:
        """Check if using API provider."""
        return self.provider.provider_type == ProviderType.API

    def get_metrics(self) -> Dict:
        """Get client metrics."""
        with self._lock:
            return {
                **self.metrics,
                "provider": self.config.provider,
                "provider_type": self.provider.provider_type.name,
                "model": self.config.model,
                "avg_latency_ms": (self.metrics["latency_total_ms"] / self.metrics["calls"]
                                  if self.metrics["calls"] > 0 else 0)
            }


# ============================================================
# Helpers
# ============================================================

def _check_ollama(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running."""
    try:
        req = Request(f"{host}/api/tags")
        with urlopen(req, timeout=2):
            return True
    except:
        return False


def _check_lmstudio(host: str = "http://localhost:1234") -> bool:
    """Check if LM Studio is running."""
    try:
        req = Request(f"{host}/v1/models")
        with urlopen(req, timeout=2):
            return True
    except:
        return False


def create_client(provider: str = "auto", **kwargs) -> LLMClient:
    """Factory for LLM client."""
    config = LLMConfig(provider=provider, **kwargs)
    return LLMClient(config)


def quick_chat(prompt: str, system: str = None) -> str:
    """One-liner chat."""
    return LLMClient().chat(prompt, system)


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM Client CLI")
    parser.add_argument("command", choices=["chat", "status", "bench"],
                       help="Command to run")
    parser.add_argument("--provider", default="auto", help="Provider to use")
    parser.add_argument("--model", default="", help="Model to use")
    parser.add_argument("--prompt", default="Hello!", help="Prompt for chat")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    config = LLMConfig.from_env()
    if args.provider != "auto":
        config.provider = args.provider
    if args.model:
        config.model = args.model

    client = LLMClient(config)

    if args.command == "status":
        status = {
            "provider": config.provider,
            "provider_type": client.provider.provider_type.name,
            "model": config.model,
            "available": client.provider.is_available(),
            "is_local": client.is_local()
        }
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Provider: {status['provider']} ({status['provider_type']})")
            print(f"Model: {status['model']}")
            print(f"Available: {status['available']}")
            print(f"Local: {status['is_local']}")

    elif args.command == "chat":
        response = client.chat(args.prompt)
        if args.json:
            print(json.dumps({"response": response, "metrics": client.get_metrics()}, indent=2))
        else:
            print(response)

    elif args.command == "bench":
        print("Running benchmark (10 calls)...")
        for i in range(10):
            client.chat(f"Count to {i+1}")
        metrics = client.get_metrics()
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            print(f"Calls: {metrics['calls']}")
            print(f"Tokens: {metrics['tokens_total']}")
            print(f"Avg latency: {metrics['avg_latency_ms']:.1f}ms")


if __name__ == "__main__":
    main()
