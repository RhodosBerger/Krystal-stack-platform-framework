"""
Cross-Forex Trading System

Resource trading market for compute, memory, bandwidth, and latency credits.
Agents submit trade orders that Guardian arbitrates in real-time.

Integrates with MAVB for voxel-based memory trading.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum, auto
from collections import deque
import time
import random
import heapq


# ============================================================
# ENUMS
# ============================================================

class Commodity(Enum):
    """Tradeable commodities."""
    COMPUTE_CREDITS = auto()
    THERMAL_HEADROOM = auto()
    LATENCY_BUDGET = auto()
    BANDWIDTH_GBPS = auto()
    MEMORY_MB = auto()
    VRAM_MB = auto()
    POWER_WATTS = auto()
    CACHE_LINES = auto()
    DMA_SLOTS = auto()
    PREFETCH_WINDOW = auto()


class OrderType(Enum):
    """Order types."""
    BUY = auto()
    SELL = auto()
    LIMIT_BUY = auto()
    LIMIT_SELL = auto()


class OrderStatus(Enum):
    """Order status."""
    PENDING = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


class TierDirective(Enum):
    """Domain tier directives from ranking."""
    LEGENDARY = 4
    EPIC = 3
    RARE = 2
    COMMON = 1
    INHIBITED = 0


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TradeOrder:
    """A trade order in the market."""
    order_id: str
    agent_id: str
    commodity: Commodity
    order_type: OrderType
    quantity: float
    price: float  # Credits per unit
    timestamp: float = field(default_factory=time.time)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    voxel_bundle: Optional[Tuple[int, int, int]] = None  # MAVB coupling


@dataclass
class MarketTicker:
    """Real-time market data."""
    commodity: Commodity
    bid: float  # Best buy price
    ask: float  # Best sell price
    last_price: float
    volume_24h: float
    high_24h: float
    low_24h: float
    mavb_congestion: float = 0.0  # From MAVB heatmap


@dataclass
class Trade:
    """Executed trade."""
    trade_id: str
    buyer_id: str
    seller_id: str
    commodity: Commodity
    quantity: float
    price: float
    timestamp: float = field(default_factory=time.time)


# ============================================================
# ORDER BOOK
# ============================================================

class OrderBook:
    """Order book for a single commodity."""

    def __init__(self, commodity: Commodity):
        self.commodity = commodity
        self.buy_orders: List[TradeOrder] = []   # Max heap (negative price)
        self.sell_orders: List[TradeOrder] = []  # Min heap
        self.trades: deque = deque(maxlen=1000)
        self.last_price = 1.0

    def add_order(self, order: TradeOrder) -> List[Trade]:
        """Add order and attempt matching."""
        trades = []

        if order.order_type in (OrderType.BUY, OrderType.LIMIT_BUY):
            trades = self._match_buy(order)
            if order.status != OrderStatus.FILLED:
                heapq.heappush(self.buy_orders, (-order.price, order.timestamp, order))
        else:
            trades = self._match_sell(order)
            if order.status != OrderStatus.FILLED:
                heapq.heappush(self.sell_orders, (order.price, order.timestamp, order))

        return trades

    def _match_buy(self, buy_order: TradeOrder) -> List[Trade]:
        """Match buy order against sell orders."""
        trades = []

        while self.sell_orders and buy_order.filled_quantity < buy_order.quantity:
            best_sell = self.sell_orders[0][2]

            if best_sell.price > buy_order.price:
                break  # No match

            # Calculate fill
            remaining_buy = buy_order.quantity - buy_order.filled_quantity
            remaining_sell = best_sell.quantity - best_sell.filled_quantity
            fill_qty = min(remaining_buy, remaining_sell)

            # Execute trade
            trade = Trade(
                trade_id=f"T_{int(time.time()*1000)}_{len(trades)}",
                buyer_id=buy_order.agent_id,
                seller_id=best_sell.agent_id,
                commodity=self.commodity,
                quantity=fill_qty,
                price=best_sell.price
            )
            trades.append(trade)
            self.trades.append(trade)
            self.last_price = best_sell.price

            # Update orders
            buy_order.filled_quantity += fill_qty
            best_sell.filled_quantity += fill_qty

            if best_sell.filled_quantity >= best_sell.quantity:
                best_sell.status = OrderStatus.FILLED
                heapq.heappop(self.sell_orders)
            else:
                best_sell.status = OrderStatus.PARTIAL

        if buy_order.filled_quantity >= buy_order.quantity:
            buy_order.status = OrderStatus.FILLED
        elif buy_order.filled_quantity > 0:
            buy_order.status = OrderStatus.PARTIAL

        return trades

    def _match_sell(self, sell_order: TradeOrder) -> List[Trade]:
        """Match sell order against buy orders."""
        trades = []

        while self.buy_orders and sell_order.filled_quantity < sell_order.quantity:
            best_buy = self.buy_orders[0][2]

            if best_buy.price < sell_order.price:
                break

            remaining_sell = sell_order.quantity - sell_order.filled_quantity
            remaining_buy = best_buy.quantity - best_buy.filled_quantity
            fill_qty = min(remaining_sell, remaining_buy)

            trade = Trade(
                trade_id=f"T_{int(time.time()*1000)}_{len(trades)}",
                buyer_id=best_buy.agent_id,
                seller_id=sell_order.agent_id,
                commodity=self.commodity,
                quantity=fill_qty,
                price=best_buy.price
            )
            trades.append(trade)
            self.trades.append(trade)
            self.last_price = best_buy.price

            sell_order.filled_quantity += fill_qty
            best_buy.filled_quantity += fill_qty

            if best_buy.filled_quantity >= best_buy.quantity:
                best_buy.status = OrderStatus.FILLED
                heapq.heappop(self.buy_orders)
            else:
                best_buy.status = OrderStatus.PARTIAL

        if sell_order.filled_quantity >= sell_order.quantity:
            sell_order.status = OrderStatus.FILLED
        elif sell_order.filled_quantity > 0:
            sell_order.status = OrderStatus.PARTIAL

        return trades

    def get_ticker(self) -> MarketTicker:
        """Get current ticker data."""
        bid = -self.buy_orders[0][0] if self.buy_orders else 0.0
        ask = self.sell_orders[0][0] if self.sell_orders else float('inf')

        prices = [t.price for t in self.trades]
        volume = sum(t.quantity for t in self.trades)

        return MarketTicker(
            commodity=self.commodity,
            bid=bid,
            ask=ask if ask != float('inf') else 0.0,
            last_price=self.last_price,
            volume_24h=volume,
            high_24h=max(prices) if prices else self.last_price,
            low_24h=min(prices) if prices else self.last_price
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        for heap in [self.buy_orders, self.sell_orders]:
            for i, (_, _, order) in enumerate(heap):
                if order.order_id == order_id:
                    order.status = OrderStatus.CANCELLED
                    return True
        return False


# ============================================================
# DOMAIN RANKING
# ============================================================

@dataclass
class DomainScore:
    """Scores for a cognitive domain."""
    domain_name: str
    throughput: float = 0.5
    importance: float = 0.5
    stability: float = 0.5
    cost: float = 0.5

    def total_score(self) -> float:
        return (self.throughput * 0.3 +
                self.importance * 0.3 +
                self.stability * 0.25 -
                self.cost * 0.15)

    def get_tier(self) -> TierDirective:
        score = self.total_score()
        if score >= 0.8:
            return TierDirective.LEGENDARY
        elif score >= 0.6:
            return TierDirective.EPIC
        elif score >= 0.4:
            return TierDirective.RARE
        elif score >= 0.2:
            return TierDirective.COMMON
        else:
            return TierDirective.INHIBITED


class DomainRanking:
    """Hierarchical domain ranking system."""

    DOMAINS = ["STRATEGIC", "TACTICAL", "CREATIVE", "ANALYTICAL",
               "PROTECTIVE", "REFLECTIVE", "INTUITIVE"]

    def __init__(self):
        self.scores: Dict[str, DomainScore] = {
            d: DomainScore(domain_name=d) for d in self.DOMAINS
        }
        self.history: deque = deque(maxlen=1000)

    def update(self, domain: str, throughput: float = None,
               importance: float = None, stability: float = None,
               cost: float = None):
        """Update domain scores."""
        if domain not in self.scores:
            return

        score = self.scores[domain]
        if throughput is not None:
            score.throughput = 0.9 * score.throughput + 0.1 * throughput
        if importance is not None:
            score.importance = 0.9 * score.importance + 0.1 * importance
        if stability is not None:
            score.stability = 0.9 * score.stability + 0.1 * stability
        if cost is not None:
            score.cost = 0.9 * score.cost + 0.1 * cost

        self.history.append({
            "timestamp": time.time(),
            "domain": domain,
            "score": score.total_score(),
            "tier": score.get_tier().name
        })

    def get_rankings(self) -> List[Tuple[str, TierDirective, float]]:
        """Get all domains ranked by score."""
        rankings = [
            (name, score.get_tier(), score.total_score())
            for name, score in self.scores.items()
        ]
        return sorted(rankings, key=lambda x: -x[2])

    def get_tier_directive(self, domain: str) -> TierDirective:
        """Get tier directive for domain."""
        if domain in self.scores:
            return self.scores[domain].get_tier()
        return TierDirective.COMMON

    def get_resource_multiplier(self, domain: str) -> float:
        """Get resource allocation multiplier based on tier."""
        tier = self.get_tier_directive(domain)
        multipliers = {
            TierDirective.LEGENDARY: 2.0,
            TierDirective.EPIC: 1.5,
            TierDirective.RARE: 1.0,
            TierDirective.COMMON: 0.75,
            TierDirective.INHIBITED: 0.25
        }
        return multipliers.get(tier, 1.0)


# ============================================================
# AMYGDALA THERMAL GUARDIAN
# ============================================================

class ThermalResponse(Enum):
    """Thermal guardian responses."""
    SAFE = auto()
    COOL = auto()
    THROTTLE = auto()
    EMERGENCY = auto()


@dataclass
class ThermalState:
    """Current thermal state."""
    cpu_temp: float = 55.0
    gpu_temp: float = 65.0
    power_draw: float = 150.0
    fan_speed: float = 50.0


class AmygdalaThermalGuardian:
    """
    Thermal guardian - the system's "fight or flight" response.

    Monitors temperature/power and emits throttle/cool/safe responses.
    """

    def __init__(self):
        self.state = ThermalState()
        self.thresholds = {
            "cpu_warning": 75.0,
            "cpu_critical": 85.0,
            "gpu_warning": 80.0,
            "gpu_critical": 90.0,
            "power_warning": 250.0,
            "power_critical": 300.0
        }
        self.response_history: deque = deque(maxlen=1000)
        self.current_response = ThermalResponse.SAFE

    def update(self, cpu_temp: float = None, gpu_temp: float = None,
               power_draw: float = None, fan_speed: float = None):
        """Update thermal state."""
        if cpu_temp is not None:
            self.state.cpu_temp = cpu_temp
        if gpu_temp is not None:
            self.state.gpu_temp = gpu_temp
        if power_draw is not None:
            self.state.power_draw = power_draw
        if fan_speed is not None:
            self.state.fan_speed = fan_speed

    def evaluate(self) -> Tuple[ThermalResponse, Dict]:
        """Evaluate thermal state and emit response."""
        response = ThermalResponse.SAFE
        actions = {}

        # CPU check
        if self.state.cpu_temp >= self.thresholds["cpu_critical"]:
            response = ThermalResponse.EMERGENCY
            actions["cpu"] = "emergency_throttle"
        elif self.state.cpu_temp >= self.thresholds["cpu_warning"]:
            response = max(response, ThermalResponse.THROTTLE, key=lambda x: x.value)
            actions["cpu"] = "throttle"

        # GPU check
        if self.state.gpu_temp >= self.thresholds["gpu_critical"]:
            response = ThermalResponse.EMERGENCY
            actions["gpu"] = "emergency_throttle"
        elif self.state.gpu_temp >= self.thresholds["gpu_warning"]:
            response = max(response, ThermalResponse.THROTTLE, key=lambda x: x.value)
            actions["gpu"] = "throttle"

        # Power check
        if self.state.power_draw >= self.thresholds["power_critical"]:
            response = ThermalResponse.EMERGENCY
            actions["power"] = "limit"
        elif self.state.power_draw >= self.thresholds["power_warning"]:
            response = max(response, ThermalResponse.COOL, key=lambda x: x.value)
            actions["power"] = "reduce"

        # Fan recommendation
        if response in (ThermalResponse.THROTTLE, ThermalResponse.EMERGENCY):
            actions["fan"] = min(100, self.state.fan_speed + 30)
        elif response == ThermalResponse.COOL:
            actions["fan"] = min(100, self.state.fan_speed + 15)

        self.current_response = response
        self.response_history.append({
            "timestamp": time.time(),
            "response": response.name,
            "state": {
                "cpu": self.state.cpu_temp,
                "gpu": self.state.gpu_temp,
                "power": self.state.power_draw
            }
        })

        return response, actions

    def get_thermal_headroom(self) -> Dict[str, float]:
        """Get remaining thermal headroom."""
        return {
            "cpu": max(0, self.thresholds["cpu_warning"] - self.state.cpu_temp),
            "gpu": max(0, self.thresholds["gpu_warning"] - self.state.gpu_temp),
            "power": max(0, self.thresholds["power_warning"] - self.state.power_draw)
        }


# ============================================================
# CROSS-FOREX EXCHANGE
# ============================================================

class CrossForexExchange:
    """
    Main exchange for resource trading.

    Integrates:
    - Order books for all commodities
    - Domain ranking
    - Amygdala thermal guardian
    - MAVB coupling
    """

    def __init__(self):
        self.order_books: Dict[Commodity, OrderBook] = {
            c: OrderBook(c) for c in Commodity
        }
        self.ranking = DomainRanking()
        self.amygdala = AmygdalaThermalGuardian()

        self.agents: Dict[str, Dict] = {}  # Agent balances
        self.order_counter = 0
        self.all_trades: deque = deque(maxlen=10000)

    def register_agent(self, agent_id: str, initial_credits: float = 1000.0):
        """Register a trading agent."""
        self.agents[agent_id] = {
            "credits": initial_credits,
            "holdings": {c: 0.0 for c in Commodity},
            "domain": "TACTICAL"
        }

    def submit_order(self, agent_id: str, commodity: Commodity,
                     order_type: OrderType, quantity: float, price: float,
                     voxel_bundle: Tuple[int, int, int] = None) -> Tuple[TradeOrder, List[Trade]]:
        """Submit a trade order."""
        if agent_id not in self.agents:
            self.register_agent(agent_id)

        # Check thermal state
        response, _ = self.amygdala.evaluate()
        if response == ThermalResponse.EMERGENCY:
            # Reject non-essential trades during emergency
            domain = self.agents[agent_id].get("domain", "TACTICAL")
            tier = self.ranking.get_tier_directive(domain)
            if tier.value < TierDirective.EPIC.value:
                order = TradeOrder(
                    order_id=f"ORD_{self.order_counter}",
                    agent_id=agent_id,
                    commodity=commodity,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    status=OrderStatus.REJECTED
                )
                return order, []

        self.order_counter += 1
        order = TradeOrder(
            order_id=f"ORD_{self.order_counter:06d}",
            agent_id=agent_id,
            commodity=commodity,
            order_type=order_type,
            quantity=quantity,
            price=price,
            voxel_bundle=voxel_bundle
        )

        # Apply domain multiplier to effective quantity
        domain = self.agents[agent_id].get("domain", "TACTICAL")
        multiplier = self.ranking.get_resource_multiplier(domain)

        trades = self.order_books[commodity].add_order(order)

        # Update balances
        for trade in trades:
            self._settle_trade(trade)
            self.all_trades.append(trade)

        return order, trades

    def _settle_trade(self, trade: Trade):
        """Settle a completed trade."""
        buyer = self.agents.get(trade.buyer_id)
        seller = self.agents.get(trade.seller_id)

        if buyer and seller:
            cost = trade.quantity * trade.price
            buyer["credits"] -= cost
            buyer["holdings"][trade.commodity] += trade.quantity
            seller["credits"] += cost
            seller["holdings"][trade.commodity] -= trade.quantity

    def get_ticker(self, commodity: Commodity) -> MarketTicker:
        """Get ticker for commodity."""
        ticker = self.order_books[commodity].get_ticker()
        # Add MAVB congestion if available
        return ticker

    def get_all_tickers(self) -> List[MarketTicker]:
        """Get all market tickers."""
        return [self.get_ticker(c) for c in Commodity]

    def update_domain(self, agent_id: str, domain: str):
        """Update agent's cognitive domain."""
        if agent_id in self.agents:
            self.agents[agent_id]["domain"] = domain

    def update_thermal(self, cpu_temp: float, gpu_temp: float, power: float):
        """Update thermal state."""
        self.amygdala.update(cpu_temp, gpu_temp, power)

    def tick(self) -> Dict:
        """Market tick - cleanup and evaluation."""
        thermal_response, thermal_actions = self.amygdala.evaluate()

        return {
            "thermal_response": thermal_response.name,
            "thermal_actions": thermal_actions,
            "thermal_headroom": self.amygdala.get_thermal_headroom(),
            "domain_rankings": self.ranking.get_rankings(),
            "total_trades": len(self.all_trades),
            "active_orders": sum(
                len(ob.buy_orders) + len(ob.sell_orders)
                for ob in self.order_books.values()
            )
        }

    def get_agent_portfolio(self, agent_id: str) -> Optional[Dict]:
        """Get agent's portfolio."""
        return self.agents.get(agent_id)


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate Cross-Forex exchange."""
    print("=" * 60)
    print("CROSS-FOREX RESOURCE TRADING EXCHANGE")
    print("=" * 60)

    exchange = CrossForexExchange()

    # Register agents
    agents = ["GPU_AGENT", "CPU_AGENT", "MEM_AGENT", "IO_AGENT"]
    for agent in agents:
        exchange.register_agent(agent, initial_credits=10000)

    # Set domains
    exchange.update_domain("GPU_AGENT", "CREATIVE")
    exchange.update_domain("CPU_AGENT", "TACTICAL")
    exchange.update_domain("MEM_AGENT", "ANALYTICAL")
    exchange.update_domain("IO_AGENT", "PROTECTIVE")

    # Update domain rankings
    exchange.ranking.update("CREATIVE", throughput=0.9, importance=0.8)
    exchange.ranking.update("TACTICAL", throughput=0.7, importance=0.9)

    print("\nDomain Rankings:")
    for domain, tier, score in exchange.ranking.get_rankings():
        print(f"  {domain}: {tier.name} ({score:.2f})")

    print("\nSubmitting trades...")

    # GPU wants compute credits
    order, trades = exchange.submit_order(
        "GPU_AGENT", Commodity.COMPUTE_CREDITS,
        OrderType.BUY, quantity=100, price=1.5
    )
    print(f"GPU BUY: {order.status.name}")

    # CPU sells compute credits
    order, trades = exchange.submit_order(
        "CPU_AGENT", Commodity.COMPUTE_CREDITS,
        OrderType.SELL, quantity=50, price=1.4
    )
    print(f"CPU SELL: {order.status.name}, Trades: {len(trades)}")

    # Memory agent wants bandwidth
    order, trades = exchange.submit_order(
        "MEM_AGENT", Commodity.BANDWIDTH_GBPS,
        OrderType.BUY, quantity=10, price=2.0
    )
    print(f"MEM BUY: {order.status.name}")

    # Update thermal
    exchange.update_thermal(cpu_temp=70, gpu_temp=75, power=200)

    # Market tick
    tick_result = exchange.tick()
    print(f"\nThermal Response: {tick_result['thermal_response']}")
    print(f"Thermal Headroom: {tick_result['thermal_headroom']}")

    # Tickers
    print("\nMarket Tickers:")
    for ticker in exchange.get_all_tickers()[:3]:
        print(f"  {ticker.commodity.name}: Bid={ticker.bid:.2f} Ask={ticker.ask:.2f}")

    # Portfolio
    portfolio = exchange.get_agent_portfolio("GPU_AGENT")
    print(f"\nGPU Agent Portfolio:")
    print(f"  Credits: {portfolio['credits']:.2f}")
    print(f"  Compute: {portfolio['holdings'][Commodity.COMPUTE_CREDITS]:.2f}")


if __name__ == "__main__":
    demo()
