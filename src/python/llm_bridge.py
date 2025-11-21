"""
LLM Bridge - Pipeline to Local LLM (LM Studio)

Provides automated generation via shared fields from LM Studio.
Supports RAG patterns, code generation, and document analysis.
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field
from enum import Enum
import requests
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class LLMConfig:
    """LLM connection configuration."""
    provider: LLMProvider = LLMProvider.LM_STUDIO
    base_url: str = "http://localhost:1234/v1"
    model: str = "local-model"
    api_key: str = "lm-studio"  # LM Studio doesn't require real key
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120


@dataclass
class LLMResponse:
    """LLM response wrapper."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Cache entry for LLM responses."""
    response: LLMResponse
    timestamp: float
    hits: int = 0


class LLMCache:
    """LRU cache for LLM responses."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()

    def _hash_prompt(self, prompt: str, system: Optional[str] = None) -> str:
        content = f"{system or ''}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, prompt: str, system: Optional[str] = None) -> Optional[LLMResponse]:
        key = self._hash_prompt(prompt, system)
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                if time.time() - entry.timestamp < self.ttl:
                    entry.hits += 1
                    response = entry.response
                    response.cached = True
                    return response
                else:
                    del self._cache[key]
        return None

    def set(self, prompt: str, response: LLMResponse, system: Optional[str] = None):
        key = self._hash_prompt(prompt, system)
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest_key]
            self._cache[key] = CacheEntry(response=response, timestamp=time.time())

    def clear(self):
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_hits = sum(e.hits for e in self._cache.values())
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_hits": total_hits,
                "ttl_seconds": self.ttl,
            }


class LLMBridge:
    """
    Bridge to Local LLM (LM Studio or compatible).

    Features:
    - OpenAI-compatible API
    - Response caching
    - Streaming support
    - RAG context injection
    - Code generation helpers
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.cache = LLMCache()
        self._session = requests.Session()

    def _build_messages(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        if context:
            messages.extend(context)

        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """Generate response from LLM."""

        # Check cache
        if use_cache and not context:
            cached = self.cache.get(prompt, system)
            if cached:
                return cached

        messages = self._build_messages(prompt, system, context)

        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": False,
        }

        start_time = time.time()

        try:
            resp = self._session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=self.config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            latency = (time.time() - start_time) * 1000

            response = LLMResponse(
                content=content,
                model=data.get("model", self.config.model),
                tokens_used=tokens,
                latency_ms=latency,
                metadata={"finish_reason": data["choices"][0].get("finish_reason")},
            )

            # Cache response
            if use_cache and not context:
                self.cache.set(prompt, response, system)

            return response

        except requests.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            raise LLMError(f"Failed to connect to LLM: {e}")

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Generate streaming response from LLM."""

        messages = self._build_messages(prompt, system, context)

        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        try:
            with self._session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=self.config.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                            except json.JSONDecodeError:
                                continue

        except requests.RequestException as e:
            logger.error(f"LLM stream failed: {e}")
            raise LLMError(f"Failed to stream from LLM: {e}")

    # Specialized generation methods

    def analyze_document(self, content: str, doc_type: str = "general") -> LLMResponse:
        """Analyze document content."""
        system = f"""You are a document analyst specializing in {doc_type} documents.
Extract key information, summarize content, and identify important data points.
Return structured JSON when possible."""

        return self.generate(
            prompt=f"Analyze this document:\n\n{content[:8000]}",
            system=system,
            use_cache=True,
        )

    def analyze_financial(self, content: str, fields: List[str] = None) -> LLMResponse:
        """Analyze financial/accounting document."""
        fields_str = ", ".join(fields) if fields else "amounts, dates, VAT, totals, parties"

        system = f"""You are a financial document analyst and auditor.
Extract these fields: {fields_str}
Calculate totals and verify VAT calculations.
Return structured JSON with extracted data and any discrepancies found."""

        return self.generate(
            prompt=f"Analyze this financial document:\n\n{content[:8000]}",
            system=system,
            use_cache=False,  # Financial data should not be cached
        )

    def generate_code(
        self,
        description: str,
        language: str = "python",
        framework: Optional[str] = None,
    ) -> LLMResponse:
        """Generate code from description."""
        framework_str = f" using {framework}" if framework else ""

        system = f"""You are an expert {language} developer{framework_str}.
Generate clean, well-documented code following best practices.
Include type hints and error handling."""

        return self.generate(
            prompt=f"Generate {language} code for:\n{description}",
            system=system,
        )

    def suggest_autocomplete(
        self,
        partial_text: str,
        field_type: str,
        history: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Generate autocomplete suggestions based on context."""
        history_ctx = ""
        if history:
            history_ctx = f"\nRecent entries: {', '.join(history[-5:])}"

        system = f"""You are an autocomplete assistant for {field_type} fields.
Suggest 3-5 completions based on the partial input and context.
Return JSON array of suggestions."""

        return self.generate(
            prompt=f"Complete this {field_type}: '{partial_text}'{history_ctx}",
            system=system,
            use_cache=True,
        )

    def summarize(self, content: str, style: str = "concise") -> LLMResponse:
        """Summarize content."""
        system = f"""Create a {style} summary of the provided content.
Preserve key facts and important details."""

        return self.generate(
            prompt=f"Summarize:\n{content[:10000]}",
            system=system,
        )

    def rag_query(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        max_context_length: int = 4000,
    ) -> LLMResponse:
        """RAG-style query with retrieved documents."""

        # Build context from documents
        context_parts = []
        current_length = 0

        for doc in context_docs:
            doc_text = f"[Source: {doc.get('source', 'unknown')}]\n{doc.get('content', '')}\n"
            if current_length + len(doc_text) > max_context_length:
                break
            context_parts.append(doc_text)
            current_length += len(doc_text)

        context_str = "\n---\n".join(context_parts)

        system = """You are a helpful assistant with access to a document database.
Use the provided context to answer questions accurately.
Cite sources when possible. If information is not in the context, say so."""

        return self.generate(
            prompt=f"Context:\n{context_str}\n\nQuestion: {query}",
            system=system,
            use_cache=False,
        )

    def health_check(self) -> bool:
        """Check if LLM service is available."""
        try:
            resp = self._session.get(
                f"{self.config.base_url}/models",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=5,
            )
            return resp.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            resp = self._session.get(
                f"{self.config.base_url}/models",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except:
            return []


class LLMError(Exception):
    """LLM-related error."""
    pass


# Factory function
def create_llm_bridge(
    provider: str = "lm_studio",
    base_url: str = "http://localhost:1234/v1",
    **kwargs,
) -> LLMBridge:
    """Create LLM bridge with configuration."""
    config = LLMConfig(
        provider=LLMProvider(provider),
        base_url=base_url,
        **kwargs,
    )
    return LLMBridge(config)
