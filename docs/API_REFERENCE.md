# KrystalSDK API Reference

## krystal_sdk

### Krystal

Main adaptive intelligence class.

```python
from src.python.krystal_sdk import Krystal, KrystalConfig

# Default configuration
k = Krystal()

# Custom configuration
k = Krystal(KrystalConfig(
    state_dim=8,
    action_dim=4,
    learning_rate=0.1,
    swarm_particles=5,
    enable_phase=True,
    enable_swarm=True
))
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `observe(obs)` | `obs: Dict[str, float]` | `self` | Feed current state observation |
| `decide(objective=None)` | `objective: Callable` (optional) | `List[float]` | Get action recommendation |
| `reward(r)` | `r: float` | `float` (TD error) | Provide reward signal |
| `optimize(objective, iterations)` | `objective: Callable`, `iterations: int` | `Tuple[List, float]` | Run swarm optimization |
| `control(setpoint, current)` | `setpoint: float`, `current: float` | `float` | PID control output |
| `get_phase()` | - | `str` | Current phase name |
| `get_metrics()` | - | `Dict` | Performance metrics |
| `save(path)` | `path: str` | - | Save state to file |
| `load(path)` | `path: str` | - | Load state from file |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `cycle` | `int` | Current cycle count |
| `total_reward` | `float` | Cumulative reward |
| `last_action` | `List[float]` | Most recent action |
| `config` | `KrystalConfig` | Configuration object |

---

### KrystalConfig

Configuration dataclass.

```python
@dataclass
class KrystalConfig:
    state_dim: int = 8           # State vector dimensions
    action_dim: int = 4          # Action vector dimensions
    learning_rate: float = 0.1   # TD learning rate
    swarm_particles: int = 5     # PSO particle count
    enable_phase: bool = True    # Enable phase transitions
    enable_swarm: bool = True    # Enable swarm optimizer
    auto_reward: bool = False    # Auto-compute reward
```

---

### Factory Functions

```python
# Pre-configured optimizers
from src.python.krystal_sdk import (
    create_game_optimizer,      # FPS, thermal, quality
    create_server_optimizer,    # CPU, latency, scaling
    create_ml_optimizer,        # Loss, accuracy, hyperparams
    create_iot_optimizer        # Battery, power, throughput
)
```

---

### Phase Enum

```python
from src.python.krystal_sdk import Phase

Phase.SOLID   # Exploitation mode, low exploration
Phase.LIQUID  # Balanced exploration/exploitation
Phase.GAS     # High exploration, random search
Phase.PLASMA  # Breakthrough mode, maximum creativity
```

---

### health_check()

```python
from src.python.krystal_sdk import health_check

result = health_check()
# {
#     "status": "healthy",
#     "version": "0.1.0",
#     "components": {
#         "learner": {"ok": True, ...},
#         "phase": {"ok": True, ...},
#         "swarm": {"ok": True, ...},
#         "krystal": {"ok": True, ...}
#     }
# }
```

---

## llm_client

### LLMClient

Unified LLM client supporting local and API providers.

```python
from src.python.llm_client import LLMClient, LLMConfig, Message

# Auto-detect provider
client = LLMClient()

# Explicit configuration
client = LLMClient(LLMConfig(
    provider="openai",
    api_key="sk-xxx",
    model="gpt-4",
    max_tokens=1024,
    temperature=0.7
))
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `complete(messages, **kwargs)` | `messages: List[Message]` | `Response` | Send completion request |
| `stream(messages, **kwargs)` | `messages: List[Message]` | `Iterator[str]` | Stream response tokens |
| `chat(prompt, system=None)` | `prompt: str`, `system: str` | `str` | Simple chat interface |
| `is_local()` | - | `bool` | Check if using local provider |
| `is_api()` | - | `bool` | Check if using API provider |
| `get_metrics()` | - | `Dict` | Client metrics |

---

### LLMConfig

```python
@dataclass
class LLMConfig:
    provider: str = "auto"      # auto, ollama, openai, anthropic, gemini, mock
    model: str = ""             # Model name
    api_key: str = ""           # API key (for API providers)
    base_url: str = ""          # Custom endpoint URL
    max_tokens: int = 1024      # Maximum output tokens
    temperature: float = 0.7    # Sampling temperature
    timeout: int = 60           # Request timeout (seconds)
    retry_count: int = 3        # Retry attempts
    retry_delay: float = 1.0    # Delay between retries
    stream: bool = False        # Enable streaming

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
```

---

### Message

```python
@dataclass
class Message:
    role: str      # "system", "user", or "assistant"
    content: str   # Message content
```

---

### Response

```python
@dataclass
class Response:
    content: str                           # Response text
    model: str = ""                        # Model used
    provider: str = ""                     # Provider name
    provider_type: ProviderType = LOCAL    # LOCAL or API
    tokens_input: int = 0                  # Input token count
    tokens_output: int = 0                 # Output token count
    latency_ms: float = 0                  # Request latency
    finish_reason: str = "stop"            # Completion reason
    raw: Dict = {}                         # Raw API response

    @property
    def tokens_total(self) -> int          # Total tokens
    @property
    def is_local(self) -> bool             # Using local provider?
```

---

### ProviderType

```python
from src.python.llm_client import ProviderType

ProviderType.LOCAL  # Ollama, LM Studio, vLLM
ProviderType.API    # OpenAI, Anthropic, Gemini
```

---

### Providers

| Provider | Type | Environment Variables |
|----------|------|----------------------|
| `ollama` | LOCAL | `OLLAMA_HOST`, `OLLAMA_MODEL` |
| `lmstudio` | LOCAL | `LMSTUDIO_HOST`, `LMSTUDIO_MODEL` |
| `vllm` | LOCAL | `VLLM_HOST`, `VLLM_MODEL` |
| `openai` | API | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| `anthropic` | API | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| `gemini` | API | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| `mock` | LOCAL | None |

---

## generative_platform

### GenerativePlatform

Multi-agent orchestration system.

```python
from src.python.generative_platform import (
    create_generative_platform,
    GenerativePlatform,
    AdminUser,
    AdminLevel
)

platform = create_generative_platform()

# Add user
platform.admin.add_user(AdminUser(
    username="developer",
    level=AdminLevel.ADMIN,
    permissions=["generate", "approve"]
))

# Generate content
artifact = platform.generate({
    "task": "create function",
    "name": "sort_list"
}, actor="developer")
```

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate(request, actor)` | `request: Dict`, `actor: str` | `GeneratedArtifact` | Generate content |
| `run_workflow(name, input)` | `name: str`, `input: Any` | `List[Any]` | Execute agent workflow |
| `get_pending()` | - | `List[GeneratedArtifact]` | Pending approvals |
| `approve(idx, actor)` | `idx: int`, `actor: str` | `bool` | Approve artifact |
| `reject(idx, actor, reason)` | `idx: int`, `actor: str`, `reason: str` | `bool` | Reject artifact |
| `get_agent(agent_id)` | `agent_id: str` | `LLMAgent` | Get agent by ID |
| `add_agent(agent)` | `agent: LLMAgent` | - | Register custom agent |
| `get_metrics()` | - | `Dict` | Platform metrics |

---

### GeneratedArtifact

```python
@dataclass
class GeneratedArtifact:
    content_type: ContentType   # CODE, TEXT, CONFIG, etc.
    content: str                # Generated content
    metadata: Dict = {}         # Additional data
    quality_score: float = 0.0  # Quality rating (0-1)
    agent_id: str = ""          # Generating agent
    timestamp: float            # Creation time
    approved: bool = False      # Approval status
```

---

### AdminLevel

```python
class AdminLevel(Enum):
    VIEWER = 0      # Read-only access
    OPERATOR = 1    # Basic operations
    ADMIN = 2       # Full management
    SUPERADMIN = 3  # All permissions
```

---

### Built-in Agents

| Agent | Role | Description |
|-------|------|-------------|
| `PlannerAgent` | PLANNER | Task decomposition |
| `CoderAgent` | CODER | Code generation (LLM-powered) |
| `CriticAgent` | CRITIC | Quality review |
| `GuardianAgent` | GUARDIAN | Safety enforcement |
| `OptimizerAgent` | OPTIMIZER | KrystalSDK integration |

---

## unified_system

### UnifiedSystem

Six-level adaptive system.

```python
from src.python.unified_system import (
    create_unified_system,
    UnifiedSystem,
    SystemMode
)

system = create_unified_system()
system.start()

# Run single tick
result = system.tick()
# {
#     "cycle": 1,
#     "telemetry": {...},
#     "signals": {...},
#     "decision": {...},
#     "predictions": {...},
#     "emergence": {...},
#     "reward": 0.5,
#     "mode": "RUNNING"
# }

# Run multiple cycles
results = system.run_cycles(100)

# Get summary
summary = system.get_state_summary()

system.stop()
```

---

### SystemMode

```python
class SystemMode(Enum):
    INIT = auto()       # Initializing
    RUNNING = auto()    # Active operation
    PAUSED = auto()     # Temporarily stopped
    LEARNING = auto()   # Training mode
    INFERENCE = auto()  # Inference only
```

---

## config

### load_config

```python
from src.python.config import load_config, SystemConfig

# Load from file
config = load_config("config.yaml")

# Access settings
print(config.llm.provider)
print(config.learning.params)
```

### load_env

```python
from src.python.config import load_env

# Load .env file
env_vars = load_env(".env")
```

---

## web_dashboard

### run_server

```python
from src.python.web_dashboard import run_server, run_background

# Blocking server
run_server(host="0.0.0.0", port=8000)

# Background server
thread = run_background(port=8000)
```

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML |
| `/api/metrics` | GET | Current metrics |
| `/api/uptime` | GET | Server uptime |
| `/api/decide` | GET | Get action |
| `/api/observe` | POST | Send observation |
| `/api/reward` | POST | Send reward |
| `/api/reset` | POST | Reset system |
| `/api/generate` | POST | Generate content |
| `/api/agents` | GET | List agents |

---

## CLI Reference

### krystal_sdk

```bash
# Health check
python -m src.python.krystal_sdk health [--json]

# Version
python -m src.python.krystal_sdk version

# Benchmark
python -m src.python.krystal_sdk bench [--json]

# Demo
python -m src.python.krystal_sdk demo [--cycles N] [--json]
```

### llm_client

```bash
# Status
python -m src.python.llm_client status [--provider NAME] [--json]

# Chat
python -m src.python.llm_client chat --prompt "Hello" [--provider NAME]

# Benchmark
python -m src.python.llm_client bench [--json]
```

### web_dashboard

```bash
# Start server
python -m src.python.web_dashboard [--host HOST] [--port PORT]
```
