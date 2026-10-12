#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK // TEST SUITE: UNIVERSAL OPENAI LLM CONNECTOR & GATEWAY
==============================================================================
Tests:
  1. Universal Invariants: VITAL_MAX_HP == 6 across all configuration states.
  2. Endpoint Presets: Ollama, LM Studio, Localhost Kernel, OpenAI, OpenRouter, DeepSeek.
  3. Key Masking & Security: Zero raw API key leakage in dicts or status payloads.
  4. Configuration Mutation: Dynamic updates to base_url, model_name, and timeout.
  5. Mock Chat Completion: Robust handling of standard OpenAI chat responses.
  6. Graceful Probe Fallback: Handling offline or unreachable endpoints.
  7. Kernel API Integration: /v1/models merges upstream models while keeping reference fallback.
  8. Economic Engine Re-export Bridge: Proper module exports in economic_engine.
==============================================================================
"""

import unittest
import os
import sys
import json
from io import BytesIO
from unittest.mock import patch, MagicMock
import urllib.error

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_kernel.openai_llm_connector import (
    VITAL_MAX_HP,
    OpenAIEndpointConfig,
    OpenAILLMConnector,
    ENDPOINT_PRESETS,
    GLOBAL_OPENAI_CONNECTOR
)
from krystal_web_hub.economic_engine.openai_llm_bridge import (
    GLOBAL_OPENAI_CONNECTOR as BRIDGE_GLOBAL_CONNECTOR,
    OpenAILLMConnector as BridgeConnector,
    OpenAIEndpointConfig as BridgeConfig
)
from krystal_kernel.kernel import ComputeKernel, KernelParams
from krystal_kernel.openai_api import ApiContext, handle


class TestOpenAILLMConnector(unittest.TestCase):

    def setUp(self):
        self.config = OpenAIEndpointConfig(
            base_url="http://127.0.0.1:11434/v1",
            api_key="sk-test-super-secret-key-1234567890abcdef",
            model_name="llama3:latest",
            timeout_sec=5.0,
            endpoint_type="ollama"
        )
        self.connector = OpenAILLMConnector(config=self.config)
        self.kernel = ComputeKernel(KernelParams())
        self.ctx = ApiContext(self.kernel)

    def test_universal_invariants(self):
        """Verifies VITAL_MAX_HP == 6 invariant in connector and status dictionaries."""
        self.assertEqual(VITAL_MAX_HP, 6)
        status = self.connector.get_status()
        self.assertEqual(status["vital_max_hp_rule"], 6)

        safe_dict = self.config.to_safe_dict()
        self.assertEqual(safe_dict["vital_max_hp_rule"], 6)

    def test_endpoint_presets(self):
        """Validates standard LLM presets for local and cloud runtimes."""
        self.assertIn("ollama", ENDPOINT_PRESETS)
        self.assertIn("lmstudio", ENDPOINT_PRESETS)
        self.assertIn("krystal_local", ENDPOINT_PRESETS)
        self.assertIn("openai", ENDPOINT_PRESETS)
        self.assertIn("openrouter", ENDPOINT_PRESETS)
        self.assertIn("deepseek", ENDPOINT_PRESETS)

        ollama_preset = ENDPOINT_PRESETS["ollama"]
        self.assertEqual(ollama_preset["base_url"], "http://127.0.0.1:11434/v1")
        self.assertEqual(ollama_preset["requires_auth"], False)

        openai_preset = ENDPOINT_PRESETS["openai"]
        self.assertEqual(openai_preset["base_url"], "https://api.openai.com/v1")
        self.assertEqual(openai_preset["requires_auth"], True)

    def test_key_masking_security(self):
        """Verifies that API keys are strictly masked and never leaked in representations."""
        raw_secret = "sk-proj-verysecrettoken1234567890abcdef"
        config = OpenAIEndpointConfig(api_key=raw_secret)
        masked = config.masked_api_key()

        self.assertNotEqual(masked, raw_secret)
        self.assertTrue(masked.startswith("sk-p..."))
        self.assertTrue(masked.endswith("cdef"))

        # Verify safe dictionary
        safe_dict = config.to_safe_dict()
        self.assertNotIn(raw_secret, json.dumps(safe_dict))
        self.assertEqual(safe_dict["masked_api_key"], masked)

        # Empty key masking
        empty_config = OpenAIEndpointConfig(api_key="")
        self.assertEqual(empty_config.masked_api_key(), "(none)")

    def test_configure_mutation(self):
        """Validates dynamic re-configuration of the connector."""
        res = self.connector.configure(
            base_url="http://192.168.1.100:1234/v1",
            api_key="custom-key-xyz",
            model_name="mistral-7b-instruct",
            timeout_sec=12.5,
            endpoint_type="lmstudio"
        )
        self.assertTrue(res["success"])
        self.assertEqual(self.connector.config.base_url, "http://192.168.1.100:1234/v1")
        self.assertEqual(self.connector.config.model_name, "mistral-7b-instruct")
        self.assertEqual(self.connector.config.timeout_sec, 12.5)
        self.assertEqual(self.connector.config.endpoint_type, "LM_STUDIO_LOCAL")

    @patch("urllib.request.urlopen")
    def test_chat_completion_mock(self, mock_urlopen):
        """Validates chat completion request parsing and response handling."""
        mock_response = MagicMock()
        mock_payload = {
            "id": "chatcmpl-mock-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3:latest",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Krystal kernel connection successful."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20
            }
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        res = self.connector.chat_completion(
            messages=[{"role": "user", "content": "Ahoj Krystal!"}],
            temperature=0.7
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["content"], "Krystal kernel connection successful.")
        self.assertEqual(res["model"], "llama3:latest")
        self.assertEqual(res["usage"]["total_tokens"], 20)
        self.assertEqual(res["vital_max_hp_rule"], 6)

    def test_probe_unreachable_endpoint(self):
        """Verifies graceful error handling when probing an offline endpoint."""
        connector = OpenAILLMConnector(
            config=OpenAIEndpointConfig(
                base_url="http://127.0.0.1:59999/v1",  # Unused port
                timeout_sec=0.5
            )
        )
        res = connector.probe_endpoint()
        self.assertFalse(res["success"])
        self.assertFalse(res["reachable"])
        self.assertIn("error", res)
        self.assertEqual(res["vital_max_hp_rule"], 6)

    def test_openai_api_integration(self):
        """Verifies integration with krystal_kernel/openai_api.py."""
        # Pre-populate cached models in global connector
        GLOBAL_OPENAI_CONNECTOR.cached_models = ["upstream-gpt-4o", "upstream-claude-3"]
        s, h, b = handle(self.ctx, "GET", "/v1/models", {}, b"")
        self.assertEqual(s, 200)
        models_data = json.loads(b.decode("utf-8"))
        model_ids = [m["id"] for m in models_data["data"]]

        # Upstream cached models should be listed
        self.assertIn("upstream-gpt-4o", model_ids)
        self.assertIn("upstream-claude-3", model_ids)
        # Built-in reference chat and embeddings must remain
        self.assertIn("krystal-reference-chat", model_ids)
        self.assertIn("krystal-hash-embed-64", model_ids)

        # Standard reference model call still produces honest reference explanation
        headers = {"content-type": "application/json"}
        chat_req = json.dumps({
            "model": "krystal-reference-chat",
            "messages": [{"role": "user", "content": "Hello Krystal"}]
        }).encode("utf-8")
        s_chat, _, b_chat = handle(self.ctx, "POST", "/v1/chat/completions", headers, chat_req)
        self.assertEqual(s_chat, 200)
        ref_call = json.loads(b_chat.decode("utf-8"))
        self.assertIn("no language model", ref_call["choices"][0]["message"]["content"])

    def test_economic_engine_bridge_reexport(self):
        """Verifies that the economic engine re-export bridge properly maps classes."""
        self.assertIs(BRIDGE_GLOBAL_CONNECTOR, GLOBAL_OPENAI_CONNECTOR)
        self.assertIs(BridgeConnector, OpenAILLMConnector)
        self.assertIs(BridgeConfig, OpenAIEndpointConfig)


if __name__ == "__main__":
    unittest.main()
