"""
economics.py
The Economic Engine of the Industrial Organism.
Handles the Cross-Forex Trading of computational resources.
"""

import time
import uuid


class ResourceAsset:
    """Represents a tradable hardware resource (e.g., VRAM Block, Compute Cycle)."""

    def __init__(self, name: str, quantity: float, base_value: float):
        self.id = uuid.uuid4()
        self.name = name
        self.quantity = quantity
        self.market_value = base_value
        self.timestamp = time.time()


class MarketOrder:
    """A buy/sell order in the internal resource market."""

    def __init__(self, asset_type: str, amount: float, bid_price: float, process_id: str):
        self.asset_type = asset_type
        self.amount = amount
        self.bid_price = bid_price
        self.process_id = process_id
        self.timestamp = time.time()


class CrossForexMarket:
    """
    The central exchange where processes trade resources.
    Implements the 'Economic Resource Trading' logic.
    """

    def __init__(self):
        self.order_book: list[MarketOrder] = []
        self.market_history: list[dict] = []

    def submit_order(self, order: MarketOrder) -> bool:
        """Submits an order to the market."""
        print(
            f"[ECONOMICS] New Order: Process {order.process_id} wants {order.amount} of {order.asset_type} at ${order.bid_price}"
        )
        self.order_book.append(order)
        self._match_orders()
        return True

    def _match_orders(self):
        """Internal clearing mechanism (Simplified matchmaking)."""
        # In a real implementation, this would use a matching engine.
        # For the scaffold, we simulate instant liquidity.

    def get_market_price(self, asset_type: str) -> float:
        """Calculates current market price based on supply/demand."""
        # Stub: returns a fluctuating 'live' price
        base_price = 100.0
        return base_price * (1.0 + (len(self.order_book) * 0.01))
