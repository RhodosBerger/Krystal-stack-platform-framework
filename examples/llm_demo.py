#!/usr/bin/env python3
"""
LLM Demo - Shows equal usage of local and API providers

Demonstrates:
- Auto-detection of providers
- Switching between local (Ollama) and API (OpenAI/Claude/Gemini)
- Unified interface for both
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.python.llm_client import (
    LLMClient, LLMConfig, Message, ProviderType,
    create_client, quick_chat
)


def demo_auto_detect():
    """Demo auto-detection of providers."""
    print("=== Auto-Detection Demo ===\n")

    client = LLMClient()
    print(f"Detected provider: {client.config.provider}")
    print(f"Provider type: {client.provider.provider_type.name}")
    print(f"Model: {client.config.model}")
    print(f"Is local: {client.is_local()}")

    response = client.chat("Say hello in one word")
    print(f"\nResponse: {response}")
    print(f"Metrics: {client.get_metrics()}")


def demo_local_provider():
    """Demo using local Ollama."""
    print("\n=== Local Provider (Ollama) Demo ===\n")

    config = LLMConfig(
        provider="ollama",
        model="llama2",
        base_url="http://localhost:11434"
    )
    client = LLMClient(config)

    if client.provider.is_available():
        print("Ollama is available!")
        response = client.complete([
            Message("system", "You are a helpful assistant."),
            Message("user", "What is 2+2?")
        ])
        print(f"Response: {response.content}")
        print(f"Tokens: {response.tokens_total}")
        print(f"Latency: {response.latency_ms:.0f}ms")
    else:
        print("Ollama not available. Start with: ollama serve")


def demo_api_providers():
    """Demo API providers."""
    print("\n=== API Providers Demo ===\n")

    providers = [
        ("openai", "OPENAI_API_KEY", "gpt-3.5-turbo"),
        ("anthropic", "ANTHROPIC_API_KEY", "claude-3-haiku-20240307"),
        ("gemini", "GEMINI_API_KEY", "gemini-1.5-flash"),
    ]

    for name, env_var, model in providers:
        api_key = os.getenv(env_var)
        if api_key:
            print(f"\n--- {name.upper()} ---")
            config = LLMConfig(provider=name, api_key=api_key, model=model)
            client = LLMClient(config)

            response = client.chat("What is the capital of France? One word.")
            print(f"Response: {response.content.strip()}")
            print(f"Tokens: {response.tokens_total}, Latency: {response.latency_ms:.0f}ms")
        else:
            print(f"\n--- {name.upper()} ---")
            print(f"Not configured (set {env_var})")


def demo_comparison():
    """Compare local vs API responses."""
    print("\n=== Local vs API Comparison ===\n")

    prompt = "Explain quantum computing in one sentence."

    # Try local first
    local_client = create_client(provider="ollama")
    if local_client.provider.is_available():
        local_resp = local_client.chat(prompt)
        print(f"Local (Ollama):")
        print(f"  Response: {local_resp[:100]}...")
        print(f"  Latency: {local_client.get_metrics()['avg_latency_ms']:.0f}ms")
    else:
        print("Local: Not available")

    # Try API
    for provider in ["openai", "anthropic", "gemini"]:
        api_client = create_client(provider=provider)
        if api_client.provider.is_available():
            api_resp = api_client.chat(prompt)
            print(f"\nAPI ({provider}):")
            print(f"  Response: {api_resp[:100]}...")
            print(f"  Latency: {api_client.get_metrics()['avg_latency_ms']:.0f}ms")
            break
    else:
        print("\nAPI: No provider configured")


def demo_with_krystal():
    """Demo LLM + KrystalSDK integration."""
    print("\n=== LLM + KrystalSDK Integration ===\n")

    from src.python.krystal_sdk import Krystal

    client = LLMClient()
    krystal = Krystal()

    # Simulate optimization loop with LLM hints
    for i in range(5):
        # Get current metrics
        krystal.observe({"iteration": i/10, "performance": 0.5 + i*0.1})
        action = krystal.decide()
        reward = sum(action) / len(action)
        krystal.reward(reward)

        # Ask LLM for insight every few iterations
        if i == 4:
            metrics = krystal.get_metrics()
            prompt = f"Given optimization metrics: {metrics}, suggest one improvement."
            hint = client.chat(prompt)
            print(f"LLM Hint: {hint[:150]}...")

    print(f"\nKrystal final state: {krystal}")


def main():
    print("=" * 60)
    print("KrystalSDK LLM Demo - Local & API Providers")
    print("=" * 60)

    demo_auto_detect()
    demo_local_provider()
    demo_api_providers()
    demo_comparison()
    demo_with_krystal()

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
