"""
Configuration Management - YAML/TOML config + .env support

Handles:
- YAML/TOML configuration files
- Environment variable loading (.env)
- Configuration validation
- Default values
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import os
import json


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "mock"  # openai, anthropic, ollama, mock
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: int = 30
    rate_limit: int = 60  # requests per minute


@dataclass
class LayerConfig:
    """Configuration for a system layer."""
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """Full system configuration."""
    # Layers
    hardware: LayerConfig = field(default_factory=LayerConfig)
    signal: LayerConfig = field(default_factory=LayerConfig)
    learning: LayerConfig = field(default_factory=LayerConfig)
    prediction: LayerConfig = field(default_factory=LayerConfig)
    emergence: LayerConfig = field(default_factory=LayerConfig)
    generation: LayerConfig = field(default_factory=LayerConfig)

    # LLM
    llm: LLMConfig = field(default_factory=LLMConfig)

    # General
    log_level: str = "info"
    metrics_enabled: bool = True
    admin_enabled: bool = True


def load_env(path: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file.

    Format:
        KEY=value
        # comment
        ANOTHER_KEY="quoted value"
    """
    env_vars = {}
    env_path = Path(path)

    if not env_path.exists():
        return env_vars

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Remove quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                env_vars[key] = value
                os.environ[key] = value

    return env_vars


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config (minimal parser, no deps)."""
    config = {}
    current_section = None
    indent_stack = [(-1, config)]

    with open(path) as f:
        for line in f:
            # Skip comments and empty lines
            stripped = line.rstrip()
            if not stripped or stripped.lstrip().startswith("#"):
                continue

            # Calculate indent
            indent = len(line) - len(line.lstrip())
            content = stripped.lstrip()

            # Pop stack to find parent
            while indent_stack and indent <= indent_stack[-1][0]:
                indent_stack.pop()

            parent = indent_stack[-1][1] if indent_stack else config

            if ":" in content:
                key, _, value = content.partition(":")
                key = key.strip()
                value = value.strip()

                if value:
                    # Key-value pair
                    # Try to parse value
                    if value.lower() == "true":
                        parent[key] = True
                    elif value.lower() == "false":
                        parent[key] = False
                    elif value.isdigit():
                        parent[key] = int(value)
                    elif value.replace(".", "").isdigit():
                        parent[key] = float(value)
                    else:
                        parent[key] = value.strip('"\'')
                else:
                    # Section header
                    parent[key] = {}
                    indent_stack.append((indent, parent[key]))

    return config


def load_toml(path: str) -> Dict[str, Any]:
    """Load TOML config (minimal parser, no deps)."""
    config = {}
    current_section = config

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Section header
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1]
                parts = section_name.split(".")
                current_section = config
                for part in parts:
                    if part not in current_section:
                        current_section[part] = {}
                    current_section = current_section[part]
                continue

            # Key-value
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()

                # Parse value
                if value.lower() == "true":
                    current_section[key] = True
                elif value.lower() == "false":
                    current_section[key] = False
                elif value.startswith('"') and value.endswith('"'):
                    current_section[key] = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    current_section[key] = value[1:-1]
                elif "." in value:
                    try:
                        current_section[key] = float(value)
                    except ValueError:
                        current_section[key] = value
                else:
                    try:
                        current_section[key] = int(value)
                    except ValueError:
                        current_section[key] = value

    return config


def load_config(path: str = None) -> SystemConfig:
    """
    Load configuration from file or environment.

    Precedence:
    1. Environment variables (highest)
    2. Config file (yaml/toml/json)
    3. Defaults (lowest)
    """
    config = SystemConfig()

    # Load .env first
    load_env()

    # Load config file if provided
    if path:
        path_obj = Path(path)
        if path_obj.exists():
            if path.endswith(".yaml") or path.endswith(".yml"):
                data = load_yaml(path)
            elif path.endswith(".toml"):
                data = load_toml(path)
            elif path.endswith(".json"):
                with open(path) as f:
                    data = json.load(f)
            else:
                data = {}

            # Apply config
            _apply_config(config, data)

    # Override with environment variables
    if os.getenv("OPENAI_API_KEY"):
        config.llm.provider = "openai"
        config.llm.api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("ANTHROPIC_API_KEY"):
        config.llm.provider = "anthropic"
        config.llm.api_key = os.getenv("ANTHROPIC_API_KEY")
    if os.getenv("OLLAMA_HOST"):
        config.llm.provider = "ollama"
        config.llm.base_url = os.getenv("OLLAMA_HOST")
    if os.getenv("LLM_MODEL"):
        config.llm.model = os.getenv("LLM_MODEL")
    if os.getenv("LOG_LEVEL"):
        config.log_level = os.getenv("LOG_LEVEL")

    return config


def _apply_config(config: SystemConfig, data: Dict):
    """Apply dict data to config object."""
    if "llm" in data:
        llm = data["llm"]
        if "provider" in llm:
            config.llm.provider = llm["provider"]
        if "model" in llm:
            config.llm.model = llm["model"]
        if "api_key" in llm:
            config.llm.api_key = llm["api_key"]
        if "max_tokens" in llm:
            config.llm.max_tokens = llm["max_tokens"]

    for layer in ["hardware", "signal", "learning", "prediction", "emergence", "generation"]:
        if layer in data:
            layer_config = getattr(config, layer)
            if "enabled" in data[layer]:
                layer_config.enabled = data[layer]["enabled"]
            if "params" in data[layer]:
                layer_config.params = data[layer]["params"]

    if "log_level" in data:
        config.log_level = data["log_level"]
    if "metrics_enabled" in data:
        config.metrics_enabled = data["metrics_enabled"]


def save_config(config: SystemConfig, path: str):
    """Save configuration to file."""
    data = {
        "llm": {
            "provider": config.llm.provider,
            "model": config.llm.model,
            "max_tokens": config.llm.max_tokens,
            "temperature": config.llm.temperature,
        },
        "hardware": {"enabled": config.hardware.enabled, "params": config.hardware.params},
        "signal": {"enabled": config.signal.enabled, "params": config.signal.params},
        "learning": {"enabled": config.learning.enabled, "params": config.learning.params},
        "prediction": {"enabled": config.prediction.enabled, "params": config.prediction.params},
        "emergence": {"enabled": config.emergence.enabled, "params": config.emergence.params},
        "generation": {"enabled": config.generation.enabled, "params": config.generation.params},
        "log_level": config.log_level,
        "metrics_enabled": config.metrics_enabled,
    }

    with open(path, "w") as f:
        if path.endswith(".json"):
            json.dump(data, f, indent=2)
        else:
            # Simple YAML output
            _write_yaml(f, data)


def _write_yaml(f, data: Dict, indent: int = 0):
    """Write dict as YAML."""
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            f.write(f"{prefix}{key}:\n")
            _write_yaml(f, value, indent + 1)
        elif isinstance(value, bool):
            f.write(f"{prefix}{key}: {str(value).lower()}\n")
        elif isinstance(value, (int, float)):
            f.write(f"{prefix}{key}: {value}\n")
        else:
            f.write(f"{prefix}{key}: \"{value}\"\n")


# Example config template
EXAMPLE_CONFIG = """# KrystalSDK Configuration

llm:
  provider: "mock"  # openai, anthropic, ollama, mock
  model: "gpt-3.5-turbo"
  max_tokens: 1024
  temperature: 0.7

hardware:
  enabled: true
  params:
    poll_interval: 100

signal:
  enabled: true
  params:
    smoothing: 0.9

learning:
  enabled: true
  params:
    learning_rate: 0.1
    gamma: 0.95

prediction:
  enabled: true
  params:
    horizon: 10

emergence:
  enabled: true
  params:
    swarm_size: 10

generation:
  enabled: true
  params:
    quality_threshold: 0.8

log_level: "info"
metrics_enabled: true
admin_enabled: true
"""


if __name__ == "__main__":
    # Demo
    print("=== Config Demo ===\n")

    # Write example config
    with open("/tmp/krystal_config.yaml", "w") as f:
        f.write(EXAMPLE_CONFIG)

    # Load it
    config = load_config("/tmp/krystal_config.yaml")
    print(f"LLM Provider: {config.llm.provider}")
    print(f"Learning Rate: {config.learning.params.get('learning_rate', 'default')}")
    print(f"Log Level: {config.log_level}")
