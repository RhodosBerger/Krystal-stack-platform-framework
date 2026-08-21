"""
GAMESA Guardian/Hex Engine - Central Bank of Cross-Forex

The regulatory authority that:
- Sets "interest rates" (Hex depth levels)
- Enforces regulations (temperature limits)
- Clears trades (approves/rejects orders)
- Manages market stability
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum, auto
from collections import deque
import time


class GuardianMode(Enum):
    """Guardian operational mode."""
    NORMAL = auto()        # Standard operations
    CONSERVATIVE = auto()  # Tighter restrictions
    AGGRESSIVE = auto()    # Allow more risk
    EMERGENCY = auto()     # Safety-first
    MAINTENANCE = auto()   # Reduced activity


class HexDepth(Enum):
    """Hex depth levels (interest rates)."""
    MINIMAL = 0x10    # Low restriction
    LOW = 0x30
    MODERATE = 0x50
    HIGH = 0x80
    EXTREME = 0xC0
    MAXIMUM = 0xFF    # Maximum restriction


class RegulationType(Enum):
    """Types of market regulations."""
    THERMAL_LIMIT = auto()
    POWER_LIMIT = auto()
    COMPUTE_CAP = auto()
    MEMORY_CAP = auto()
    LATENCY_FLOOR = auto()
    PRECISION_FLOOR = auto()


@dataclass
class Regulation:
    """A market regulation."""
    reg_type: RegulationType
    threshold: float
    action: str
    priority: int = 5
    active: bool = True


@dataclass
class ClearingResult:
    """Result of order clearing."""
    approved: bool
    reason: str
    adjustments: Dict = field(default_factory=dict)
    risk_score: float = 0.0


@dataclass
class MarketMetrics:
    """Aggregate market metrics."""
    total_orders: int = 0
    approved_orders: int = 0
    denied_orders: int = 0
    thermal_violations: int = 0
    stability_score: float = 1.0
    avg_hex_compute: float = 0.0
    avg_thermal_headroom: float = 20.0


class GuardianHexEngine:
    """
    Guardian/Hex Engine - Central Bank.

    Controls the Cross-Forex market through:
    - Hex depth levels (monetary policy)
    - Regulations (safety limits)
    - Order clearing (trade approval)
    - Market interventions
    """

    # Thermal limits
    GPU_TEMP_CRITICAL = 90
    GPU_TEMP_WARNING = 80
    CPU_TEMP_CRITICAL = 85
    CPU_TEMP_WARNING = 75

    def __init__(self):
        self.mode = GuardianMode.NORMAL
        self.hex_depth = HexDepth.MODERATE

        # Regulations
        self._regulations: List[Regulation] = []
        self._init_default_regulations()

        # Market state
        self._current_commodities: Dict = {}
        self._metrics = MarketMetrics()

        # Order history for pattern analysis
        self._order_history: deque = deque(maxlen=10000)
        self._clearing_history: deque = deque(maxlen=10000)

        # Risk tracking per agent
        self._agent_risk: Dict[str, float] = {}

        # Intervention callbacks
        self._on_intervention: List[Callable[[Dict], None]] = []

    def _init_default_regulations(self):
        """Initialize default market regulations."""
        self._regulations = [
            Regulation(RegulationType.THERMAL_LIMIT, self.GPU_TEMP_CRITICAL,
                       "DENY_ALL", priority=10),
            Regulation(RegulationType.THERMAL_LIMIT, self.GPU_TEMP_WARNING,
                       "DENY_THERMAL_INCREASE", priority=8),
            Regulation(RegulationType.POWER_LIMIT, 250.0,
                       "DENY_POWER_INCREASE", priority=7),
            Regulation(RegulationType.LATENCY_FLOOR, 5.0,
                       "WARN_LATENCY", priority=5),
        ]

    # --------------------------------------------------------
    # HEX DEPTH (INTEREST RATES)
    # --------------------------------------------------------

    def set_hex_depth(self, depth: HexDepth):
        """Set current hex depth level."""
        old_depth = self.hex_depth
        self.hex_depth = depth

        if depth.value > old_depth.value:
            self._trigger_intervention("HEX_DEPTH_INCREASE", {
                "old": old_depth.name,
                "new": depth.name
            })

    def calculate_hex_depth(self, commodities: Dict) -> HexDepth:
        """Calculate appropriate hex depth from market conditions."""
        hex_compute = commodities.get("hex_compute", 0)
        thermal_gpu = commodities.get("thermal_headroom_gpu", 20)

        # Emergency conditions
        if thermal_gpu < 5:
            return HexDepth.MAXIMUM

        # High pressure
        if hex_compute > 0xC0 or thermal_gpu < 10:
            return HexDepth.EXTREME

        # Elevated
        if hex_compute > 0x80 or thermal_gpu < 15:
            return HexDepth.HIGH

        # Moderate
        if hex_compute > 0x50:
            return HexDepth.MODERATE

        # Low pressure
        if hex_compute > 0x30:
            return HexDepth.LOW

        return HexDepth.MINIMAL

    # --------------------------------------------------------
    # REGULATIONS
    # --------------------------------------------------------

    def add_regulation(self, regulation: Regulation):
        """Add a market regulation."""
        self._regulations.append(regulation)
        self._regulations.sort(key=lambda r: -r.priority)

    def check_regulations(self, commodities: Dict) -> List[Tuple[Regulation, str]]:
        """Check all regulations against current commodities."""
        violations = []

        for reg in self._regulations:
            if not reg.active:
                continue

            if reg.reg_type == RegulationType.THERMAL_LIMIT:
                gpu_temp = 85 - commodities.get("thermal_headroom_gpu", 20)
                if gpu_temp >= reg.threshold:
                    violations.append((reg, f"GPU temp {gpu_temp}C >= {reg.threshold}C"))

            elif reg.reg_type == RegulationType.POWER_LIMIT:
                power = commodities.get("power_draw", 0)
                if power >= reg.threshold:
                    violations.append((reg, f"Power {power}W >= {reg.threshold}W"))

        return violations

    # --------------------------------------------------------
    # ORDER CLEARING
    # --------------------------------------------------------

    def clear_order(self, order, commodities: Dict) -> ClearingResult:
        """
        Clear (approve/deny) a trade order.

        Implements Guardian arbitration logic.
        """
        self._metrics.total_orders += 1
        self._order_history.append({
            "order": order,
            "commodities": commodities.copy(),
            "ts": time.time()
        })

        # Update commodities
        self._current_commodities = commodities

        # Calculate dynamic hex depth
        new_depth = self.calculate_hex_depth(commodities)
        if new_depth != self.hex_depth:
            self.set_hex_depth(new_depth)

        # Check regulations
        violations = self.check_regulations(commodities)
        if violations:
            for reg, msg in violations:
                if reg.action == "DENY_ALL":
                    self._metrics.denied_orders += 1
                    self._metrics.thermal_violations += 1
                    return ClearingResult(
                        approved=False,
                        reason=f"Regulation violation: {msg}",
                        risk_score=1.0
                    )

        # Assess order risk
        risk = self._assess_risk(order, commodities)

        # Emergency mode - deny all risky orders
        if self.mode == GuardianMode.EMERGENCY and risk > 0.3:
            self._metrics.denied_orders += 1
            return ClearingResult(
                approved=False,
                reason="Emergency mode: risk too high",
                risk_score=risk
            )

        # Conservative mode - higher threshold
        if self.mode == GuardianMode.CONSERVATIVE and risk > 0.5:
            self._metrics.denied_orders += 1
            return ClearingResult(
                approved=False,
                reason="Conservative mode: risk threshold exceeded",
                risk_score=risk
            )

        # Normal clearing logic
        result = self._evaluate_order(order, commodities, risk)

        if result.approved:
            self._metrics.approved_orders += 1
            self._update_agent_risk(order.source, risk * 0.5)
        else:
            self._metrics.denied_orders += 1
            self._update_agent_risk(order.source, risk)

        self._clearing_history.append({
            "order_id": order.order_id,
            "approved": result.approved,
            "risk": risk,
            "ts": time.time()
        })

        return result

    def _assess_risk(self, order, commodities: Dict) -> float:
        """Assess risk of an order."""
        risk = 0.0

        # Thermal risk
        thermal_gpu = commodities.get("thermal_headroom_gpu", 20)
        if thermal_gpu < 10:
            risk += 0.4
        elif thermal_gpu < 15:
            risk += 0.2

        # Compute pressure risk
        hex_compute = commodities.get("hex_compute", 0)
        if hex_compute > 0xC0:
            risk += 0.3
        elif hex_compute > 0x80:
            risk += 0.15

        # Agent history risk
        agent_risk = self._agent_risk.get(order.source, 0.0)
        risk += agent_risk * 0.2

        # Order-specific risk
        action = order.action
        if "BOOST" in action:
            risk += 0.2  # Frequency boost is risky
        elif "FP16" in action:
            risk -= 0.1  # FP16 saves power, reduces risk
        elif "BATCH_REDUCE" in action:
            risk -= 0.1  # Reduced batch is safer

        return min(1.0, max(0.0, risk))

    def _evaluate_order(self, order, commodities: Dict, risk: float) -> ClearingResult:
        """Evaluate order for approval."""
        action = order.action
        bid = order.bid

        # FP16 override - almost always approve (saves power)
        if "FP16" in action:
            if commodities.get("thermal_headroom_gpu", 20) < 15:
                return ClearingResult(
                    approved=True,
                    reason="Approved: thermal mitigation",
                    risk_score=risk
                )
            return ClearingResult(
                approved=True,
                reason="Approved: precision downgrade",
                adjustments={"force_precision": "FP16"},
                risk_score=risk
            )

        # FP32 restore - only when conditions allow
        if "FP32" in action:
            if commodities.get("hex_compute", 0) < 0x50:
                return ClearingResult(
                    approved=True,
                    reason="Approved: compute headroom available",
                    adjustments={"force_precision": "FP32"},
                    risk_score=risk
                )
            return ClearingResult(
                approved=False,
                reason="Denied: compute pressure too high for FP32",
                risk_score=risk
            )

        # Frequency boost - check thermal
        if "FREQ_BOOST" in action:
            if commodities.get("thermal_headroom_cpu", 15) > 10:
                return ClearingResult(
                    approved=True,
                    reason="Approved: thermal headroom available",
                    adjustments={"target_frequency_mhz": 4800},
                    risk_score=risk
                )
            return ClearingResult(
                approved=False,
                reason="Denied: insufficient thermal headroom",
                risk_score=risk
            )

        # Cache prefetch - generally safe
        if "PREFETCH" in action:
            return ClearingResult(
                approved=True,
                reason="Approved: cache prefetch",
                risk_score=risk
            )

        # Thread park - safe
        if "THREAD_PARK" in action:
            return ClearingResult(
                approved=True,
                reason="Approved: thread parking",
                adjustments={"thread_mask": 0x0F},
                risk_score=risk
            )

        # Batch reduce - safe
        if "BATCH" in action:
            return ClearingResult(
                approved=True,
                reason="Approved: batch reduction",
                adjustments={"batch_size": 1},
                risk_score=risk
            )

        # NPU slot - check availability
        if "NPU" in action:
            if commodities.get("hex_compute", 0) < 0x70:
                return ClearingResult(
                    approved=True,
                    reason="Approved: NPU slot",
                    risk_score=risk
                )
            return ClearingResult(
                approved=False,
                reason="Denied: compute pressure too high",
                risk_score=risk
            )

        # Default: approve low-risk orders
        if risk < 0.5:
            return ClearingResult(
                approved=True,
                reason="Approved: acceptable risk",
                risk_score=risk
            )

        return ClearingResult(
            approved=False,
            reason="Denied: risk threshold exceeded",
            risk_score=risk
        )

    def _update_agent_risk(self, agent_id: str, delta: float):
        """Update agent's cumulative risk score."""
        current = self._agent_risk.get(agent_id, 0.0)
        # Decay existing risk, add new
        self._agent_risk[agent_id] = current * 0.95 + delta * 0.05

    # --------------------------------------------------------
    # INTERVENTIONS
    # --------------------------------------------------------

    def _trigger_intervention(self, intervention_type: str, details: Dict):
        """Trigger market intervention."""
        intervention = {
            "type": intervention_type,
            "details": details,
            "ts": time.time(),
            "mode": self.mode.name,
            "hex_depth": self.hex_depth.name
        }

        for callback in self._on_intervention:
            try:
                callback(intervention)
            except Exception:
                pass

    def on_intervention(self, callback: Callable[[Dict], None]):
        """Register intervention callback."""
        self._on_intervention.append(callback)

    def emergency_halt(self):
        """Trigger emergency market halt."""
        self.mode = GuardianMode.EMERGENCY
        self.hex_depth = HexDepth.MAXIMUM
        self._trigger_intervention("EMERGENCY_HALT", {"reason": "manual_trigger"})

    def resume_normal(self):
        """Resume normal operations."""
        self.mode = GuardianMode.NORMAL
        self.hex_depth = HexDepth.MODERATE
        self._trigger_intervention("RESUME_NORMAL", {})

    # --------------------------------------------------------
    # METRICS & REPORTING
    # --------------------------------------------------------

    def update_metrics(self, commodities: Dict):
        """Update aggregate metrics."""
        hex_compute = commodities.get("hex_compute", 0)
        thermal = commodities.get("thermal_headroom_gpu", 20)

        # Running averages
        alpha = 0.1
        self._metrics.avg_hex_compute = (
            (1 - alpha) * self._metrics.avg_hex_compute + alpha * hex_compute
        )
        self._metrics.avg_thermal_headroom = (
            (1 - alpha) * self._metrics.avg_thermal_headroom + alpha * thermal
        )

        # Stability score
        approval_rate = (
            self._metrics.approved_orders / max(1, self._metrics.total_orders)
        )
        self._metrics.stability_score = (
            0.5 * approval_rate +
            0.3 * min(1.0, thermal / 20) +
            0.2 * (1 - min(1.0, hex_compute / 255))
        )

    def get_metrics(self) -> Dict:
        """Get Guardian metrics."""
        return {
            "mode": self.mode.name,
            "hex_depth": self.hex_depth.name,
            "total_orders": self._metrics.total_orders,
            "approved": self._metrics.approved_orders,
            "denied": self._metrics.denied_orders,
            "approval_rate": self._metrics.approved_orders / max(1, self._metrics.total_orders),
            "thermal_violations": self._metrics.thermal_violations,
            "stability_score": self._metrics.stability_score,
            "avg_hex_compute": f"0x{int(self._metrics.avg_hex_compute):02X}",
            "avg_thermal": f"{self._metrics.avg_thermal_headroom:.1f}C"
        }


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate Guardian/Hex Engine."""
    print("=== Guardian/Hex Engine Demo ===\n")

    from .crystal_protocol import TradeOrder, TradeBid

    guardian = GuardianHexEngine()

    # Intervention callback
    guardian.on_intervention(lambda i: print(f"  [Intervention] {i['type']}"))

    # Test scenarios
    scenarios = [
        {
            "name": "Normal conditions",
            "commodities": {"hex_compute": 0x40, "thermal_headroom_gpu": 18},
            "order_action": "REQUEST_FP16_OVERRIDE"
        },
        {
            "name": "High compute",
            "commodities": {"hex_compute": 0x90, "thermal_headroom_gpu": 12},
            "order_action": "REQUEST_FREQ_BOOST"
        },
        {
            "name": "Thermal pressure",
            "commodities": {"hex_compute": 0xB0, "thermal_headroom_gpu": 6},
            "order_action": "REQUEST_FP32_RESTORE"
        },
        {
            "name": "Recovery",
            "commodities": {"hex_compute": 0x30, "thermal_headroom_gpu": 22},
            "order_action": "REQUEST_FP32_RESTORE"
        }
    ]

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Commodities: hex=0x{scenario['commodities']['hex_compute']:02X}, "
              f"thermal={scenario['commodities']['thermal_headroom_gpu']}C")

        order = TradeOrder(
            source="AGENT_IRIS_XE",
            action=scenario["order_action"],
            bid=TradeBid(reason="TEST", priority=5)
        )

        result = guardian.clear_order(order, scenario["commodities"])
        guardian.update_metrics(scenario["commodities"])

        print(f"Order: {order.action}")
        print(f"Result: {'APPROVED' if result.approved else 'DENIED'} - {result.reason}")
        print(f"Risk: {result.risk_score:.2f}, Hex Depth: {guardian.hex_depth.name}")

    print(f"\n--- Guardian Metrics ---")
    for key, val in guardian.get_metrics().items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    demo()
