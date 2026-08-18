"""
GAMESA Crystal-Vino Protocol Definitions (GAMESA_JSON_V1)

Defines all message types for Cross-Forex resource market communication:
- Market Ticker (telemetry broadcast)
- Trade Orders (agent requests)
- Clearing Directives (guardian responses)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum, auto
import json
import time
import uuid


# ============================================================
# ENUMS
# ============================================================

class MessageType(Enum):
    """Protocol message types."""
    MARKET_TICKER = "MARKET_TICKER"
    TRADE_ORDER = "TRADE_ORDER"
    DIRECTIVE = "DIRECTIVE"
    AGENT_REGISTER = "AGENT_REGISTER"
    AGENT_HEARTBEAT = "AGENT_HEARTBEAT"
    MARKET_STATUS = "MARKET_STATUS"
    ERROR = "ERROR"


class MarketStatus(Enum):
    """Market operational status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    HALTED = "HALTED"
    EMERGENCY = "EMERGENCY"


class Scenario(Enum):
    """Runtime scenario classification."""
    IDLE = "IDLE"
    DESKTOP = "DESKTOP"
    GAME_MENU = "GAME_MENU"
    GAME_COMBAT = "GAME_COMBAT"
    GAME_CUTSCENE = "GAME_CUTSCENE"
    RENDERING = "RENDERING"
    ML_INFERENCE = "ML_INFERENCE"
    ML_TRAINING = "ML_TRAINING"


class Commodity(Enum):
    """Tradeable resource commodities."""
    HEX_COMPUTE = "hex_compute"
    HEX_MEMORY = "hex_memory"
    HEX_IO = "hex_io"
    THERMAL_HEADROOM_GPU = "thermal_headroom_gpu"
    THERMAL_HEADROOM_CPU = "thermal_headroom_cpu"
    COMPUTE_CREDITS = "compute_credits"
    PRECISION_BUDGET = "precision_budget"
    LATENCY_BUDGET = "latency_budget"


class AgentType(Enum):
    """Agent classifications."""
    IRIS_XE = "AGENT_IRIS_XE"
    SILICON = "AGENT_SILICON"
    NEURAL = "AGENT_NEURAL"
    CACHE = "AGENT_CACHE"
    GUARDIAN = "GUARDIAN"
    EXTERNAL = "EXTERNAL"


class OrderAction(Enum):
    """Trade order action types."""
    REQUEST_FP16_OVERRIDE = "REQUEST_FP16_OVERRIDE"
    REQUEST_FP32_RESTORE = "REQUEST_FP32_RESTORE"
    REQUEST_FREQ_BOOST = "REQUEST_FREQ_BOOST"
    REQUEST_THREAD_PARK = "REQUEST_THREAD_PARK"
    REQUEST_CACHE_PREFETCH = "REQUEST_CACHE_PREFETCH"
    REQUEST_BATCH_REDUCE = "REQUEST_BATCH_REDUCE"
    REQUEST_NPU_SLOT = "REQUEST_NPU_SLOT"
    OFFER_THERMAL_SAVING = "OFFER_THERMAL_SAVING"
    OFFER_LATENCY_BUDGET = "OFFER_LATENCY_BUDGET"


class DirectiveStatus(Enum):
    """Directive response status."""
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    PENDING = "PENDING"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class Precision(Enum):
    """Compute precision levels."""
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    INT4 = "INT4"


# ============================================================
# MESSAGE STRUCTURES
# ============================================================

@dataclass
class CommodityPrices:
    """Current commodity prices/levels."""
    hex_compute: int = 0x00
    hex_memory: int = 0x00
    hex_io: int = 0x00
    thermal_headroom_gpu: float = 20.0
    thermal_headroom_cpu: float = 15.0
    compute_credits: float = 100.0
    precision_budget: float = 1.0
    latency_budget_ms: float = 16.67

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CommodityPrices":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MarketState:
    """Current market state."""
    scenario: str = "IDLE"
    market_status: str = "OPEN"
    tick_interval_ms: int = 16
    guardian_mode: str = "NORMAL"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MarketTicker:
    """
    Market ticker broadcast message.
    Sent by crystal_socketd every 16-50ms.
    """
    ts: int  # Timestamp in microseconds
    commodities: CommodityPrices
    state: MarketState
    cycle: int = 0

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.MARKET_TICKER.value,
            "ts": self.ts,
            "cycle": self.cycle,
            "commodities": self.commodities.to_dict(),
            "state": self.state.to_dict()
        })

    @classmethod
    def from_json(cls, data: Union[str, Dict]) -> "MarketTicker":
        if isinstance(data, str):
            data = json.loads(data)
        return cls(
            ts=data["ts"],
            cycle=data.get("cycle", 0),
            commodities=CommodityPrices.from_dict(data["commodities"]),
            state=MarketState(**data["state"])
        )


@dataclass
class TradeBid:
    """Bid details for a trade order."""
    reason: str
    est_thermal_saving: float = 0.0
    est_latency_impact: float = 0.0
    est_compute_cost: float = 0.0
    priority: int = 5  # 1-10, higher = more urgent

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradeOrder:
    """
    Trade order from agent requesting resource adjustment.
    """
    source: str  # Agent ID
    action: str  # OrderAction value
    bid: TradeBid
    order_id: str = field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:8]}")
    ts: int = field(default_factory=lambda: int(time.time() * 1000000))

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.TRADE_ORDER.value,
            "order_id": self.order_id,
            "ts": self.ts,
            "source": self.source,
            "action": self.action,
            "bid": self.bid.to_dict()
        })

    @classmethod
    def from_json(cls, data: Union[str, Dict]) -> "TradeOrder":
        if isinstance(data, str):
            data = json.loads(data)
        return cls(
            source=data["source"],
            action=data["action"],
            bid=TradeBid(**data["bid"]),
            order_id=data.get("order_id", ""),
            ts=data.get("ts", 0)
        )


@dataclass
class DirectiveParams:
    """Parameters for approved directive."""
    duration_ms: int = 5000
    force_precision: Optional[str] = None
    target_frequency_mhz: Optional[int] = None
    cache_addresses: Optional[List[str]] = None
    batch_size: Optional[int] = None
    thread_mask: Optional[int] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Directive:
    """
    Guardian directive response to trade order.
    """
    target: str  # Target agent
    permit_id: str
    status: str  # DirectiveStatus value
    params: DirectiveParams = field(default_factory=DirectiveParams)
    reason: str = ""
    expires_at: int = 0  # Timestamp when directive expires

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.DIRECTIVE.value,
            "target": self.target,
            "permit_id": self.permit_id,
            "status": self.status,
            "params": self.params.to_dict(),
            "reason": self.reason,
            "expires_at": self.expires_at
        })

    @classmethod
    def from_json(cls, data: Union[str, Dict]) -> "Directive":
        if isinstance(data, str):
            data = json.loads(data)
        return cls(
            target=data["target"],
            permit_id=data["permit_id"],
            status=data["status"],
            params=DirectiveParams(**data.get("params", {})),
            reason=data.get("reason", ""),
            expires_at=data.get("expires_at", 0)
        )


@dataclass
class AgentRegistration:
    """Agent registration message."""
    agent_id: str
    agent_type: str
    capabilities: List[str] = field(default_factory=list)
    subscriptions: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.AGENT_REGISTER.value,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "subscriptions": self.subscriptions
        })


@dataclass
class ErrorMessage:
    """Error response message."""
    code: int
    message: str
    details: Dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.ERROR.value,
            "code": self.code,
            "message": self.message,
            "details": self.details
        })


# ============================================================
# PROTOCOL UTILITIES
# ============================================================

class ProtocolCodec:
    """Encode/decode GAMESA_JSON_V1 messages."""

    @staticmethod
    def encode(message: Any) -> bytes:
        """Encode message to bytes."""
        if hasattr(message, 'to_json'):
            return message.to_json().encode('utf-8')
        return json.dumps(message).encode('utf-8')

    @staticmethod
    def decode(data: bytes) -> Dict:
        """Decode bytes to message dict."""
        return json.loads(data.decode('utf-8'))

    @staticmethod
    def parse(data: Union[str, bytes, Dict]) -> Any:
        """Parse message to appropriate type."""
        if isinstance(data, bytes):
            data = json.loads(data.decode('utf-8'))
        elif isinstance(data, str):
            data = json.loads(data)

        msg_type = data.get("type")

        if msg_type == MessageType.MARKET_TICKER.value:
            return MarketTicker.from_json(data)
        elif msg_type == MessageType.TRADE_ORDER.value:
            return TradeOrder.from_json(data)
        elif msg_type == MessageType.DIRECTIVE.value:
            return Directive.from_json(data)
        else:
            return data

    @staticmethod
    def create_ticker(cycle: int, commodities: Dict, scenario: str = "IDLE",
                      market_status: str = "OPEN") -> MarketTicker:
        """Helper to create market ticker."""
        return MarketTicker(
            ts=int(time.time() * 1000000),
            cycle=cycle,
            commodities=CommodityPrices.from_dict(commodities),
            state=MarketState(scenario=scenario, market_status=market_status)
        )

    @staticmethod
    def create_order(agent: str, action: str, reason: str,
                     thermal_saving: float = 0.0, priority: int = 5) -> TradeOrder:
        """Helper to create trade order."""
        return TradeOrder(
            source=agent,
            action=action,
            bid=TradeBid(reason=reason, est_thermal_saving=thermal_saving, priority=priority)
        )

    @staticmethod
    def create_directive(target: str, order_id: str, approved: bool,
                         duration_ms: int = 5000, **params) -> Directive:
        """Helper to create directive."""
        return Directive(
            target=target,
            permit_id=order_id,
            status=DirectiveStatus.APPROVED.value if approved else DirectiveStatus.DENIED.value,
            params=DirectiveParams(duration_ms=duration_ms, **params),
            expires_at=int(time.time() * 1000000) + duration_ms * 1000
        )


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate protocol usage."""
    print("=== GAMESA Crystal Protocol Demo ===\n")

    codec = ProtocolCodec()

    # Create market ticker
    ticker = codec.create_ticker(
        cycle=100,
        commodities={
            "hex_compute": 0x8F,
            "hex_memory": 0x2A,
            "thermal_headroom_gpu": 12.5
        },
        scenario="GAME_COMBAT"
    )
    print("Market Ticker:")
    print(ticker.to_json())

    # Create trade order
    order = codec.create_order(
        agent=AgentType.IRIS_XE.value,
        action=OrderAction.REQUEST_FP16_OVERRIDE.value,
        reason="HEX_COMPUTE_HIGH",
        thermal_saving=2.0,
        priority=7
    )
    print("\nTrade Order:")
    print(order.to_json())

    # Create directive
    directive = codec.create_directive(
        target=AgentType.IRIS_XE.value,
        order_id=order.order_id,
        approved=True,
        duration_ms=5000,
        force_precision="FP16"
    )
    print("\nDirective:")
    print(directive.to_json())

    # Parse roundtrip
    print("\n--- Roundtrip Test ---")
    encoded = codec.encode(ticker)
    decoded = codec.parse(encoded)
    print(f"Decoded type: {type(decoded).__name__}")
    print(f"Cycle: {decoded.cycle}, Scenario: {decoded.state.scenario}")


if __name__ == "__main__":
    demo()
