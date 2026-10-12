"""
KRYSTAL-STACK // OPENAI-COMPATIBLE LLM CONNECTOR BRIDGE
=============================================================================
Re-exports the OpenAiLlmConnector from krystal_kernel for economic engine
modules and Mission Control HTTP dispatch.
"""

from krystal_kernel.openai_llm_connector import (
    VITAL_MAX_HP,
    DEFAULT_TIMEOUT_S,
    OpenAiProviderType,
    OpenAiEndpointConfig,
    OpenAiProbeResult,
    OpenAiLlmConnector,
    OpenAIProviderType,
    OpenAIEndpointConfig,
    OpenAIProbeResult,
    OpenAILLMConnector,
    ENDPOINT_PRESETS,
    GLOBAL_OPENAI_CONNECTOR
)

__all__ = [
    "VITAL_MAX_HP",
    "DEFAULT_TIMEOUT_S",
    "OpenAiProviderType",
    "OpenAiEndpointConfig",
    "OpenAiProbeResult",
    "OpenAiLlmConnector",
    "OpenAIProviderType",
    "OpenAIEndpointConfig",
    "OpenAIProbeResult",
    "OpenAILLMConnector",
    "ENDPOINT_PRESETS",
    "GLOBAL_OPENAI_CONNECTOR"
]
