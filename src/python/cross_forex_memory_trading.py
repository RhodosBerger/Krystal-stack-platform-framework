"""
Cross-forex Memory Trading System

Implements economic trading of memory resources using the 3D grid memory system
within the GAMESA/KrystalStack framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from collections import deque
import time
import uuid
import threading
from decimal import Decimal
from datetime import datetime

from . import (
    ResourceType, Priority, AllocationRequest, Allocation,
    Effect, Capability, create_guardian_checker,
    Contract, create_guardian_validator,
    TelemetrySnapshot, Signal, SignalKind, Domain
)
from .gpu_pipeline_integration import (
    GPUGridMemoryManager, MemoryGridCoordinate, 
    MemoryContext, GPUType, TaskType
)


# Enums
class MemoryResourceType(Enum):
    """Types of memory resources that can be traded."""
    L1_CACHE = "l1_cache"
    L2_CACHE = "l2_cache"
    L3_CACHE = "l3_cache"
    VRAM = "vram"
    SYSTEM_RAM = "system_ram"
    UHD_BUFFER = "uhd_buffer"
    GRID_LOCATION = "grid_location"
    COHERENCE_SLOT = "coherence_slot"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    PAGE_TABLES = "page_tables"


class TradeStatus(Enum):
    """Status of cross-forex memory trades."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"


class MarketOrderType(Enum):
    """Types of market orders."""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


# Data Classes
@dataclass
class MemoryResource:
    """Represents a tradable memory resource."""
    resource_id: str
    resource_type: MemoryResourceType
    quantity: int  # size in bytes or count
    location: Optional[MemoryGridCoordinate] = None
    performance_rating: float = 1.0  # Performance multiplier
    availability: bool = True
    current_owner: Optional[str] = None
    market_price: Decimal = Decimal('0')
    timestamp: float = field(default_factory=time.time)


@dataclass
class CrossForexTrade:
    """A cross-forex trade for memory resources."""
    trade_id: str
    trader_id: str
    order_type: MarketOrderType
    resource_type: MemoryResourceType
    quantity: int
    limit_price: Optional[Decimal] = None
    market_price: Decimal = Decimal('0')
    status: TradeStatus = TradeStatus.PENDING
    bid_credits: Decimal = Decimal('0')  # Economic bidding system
    collateral: Decimal = Decimal('0')  # Required collateral
    fee: Decimal = Decimal('0')  # Transaction fee
    profit: Optional[Decimal] = None  # Calculated profit
    timestamp: float = field(default_factory=time.time)
    expiry: Optional[float] = None  # Trade expiry time
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourcePortfolio:
    """Portfolio of owned memory resources."""
    portfolio_id: str
    owner_id: str
    resources: Dict[str, MemoryResource] = field(default_factory=dict)
    cash_balance: Decimal = Decimal('1000.00')  # Starting balance
    total_value: Decimal = Decimal('1000.00')
    timestamp: float = field(default_factory=time.time)


@dataclass
class MarketQuote:
    """Current market quote for a memory resource type."""
    resource_type: MemoryResourceType
    bid_price: Decimal  # Price buyers are willing to pay
    ask_price: Decimal  # Price sellers are asking for
    spread: Decimal     # Difference between bid and ask
    volume: int         # Trading volume
    timestamp: float = field(default_factory=time.time)


# Core Classes
class MemoryMarketEngine:
    """Main engine for cross-forex memory resource trading."""
    
    def __init__(self):
        self.resources: Dict[str, MemoryResource] = {}
        self.portfolios: Dict[str, ResourcePortfolio] = {}
        self.active_trades: Dict[str, CrossForexTrade] = {}
        self.order_book: Dict[MemoryResourceType, List[CrossForexTrade]] = {}
        self.market_quotes: Dict[MemoryResourceType, MarketQuote] = {}
        self.trade_history: deque = deque(maxlen=10000)
        self.price_history: Dict[MemoryResourceType, deque] = {}
        self.lock = threading.RLock()
        
        # Initialize with some base resources
        self._initialize_base_resources()
        self._initialize_market_quotes()
        
        # Guardian system integration
        self.effect_checker = create_guardian_checker()
        self.contract_validator = create_guardian_validator()
        
    def _initialize_base_resources(self):
        """Initialize base memory resources."""
        # Create some example resources
        for i in range(5):
            resource = MemoryResource(
                resource_id=f"RES_L3_{i:02d}",
                resource_type=MemoryResourceType.L3_CACHE,
                quantity=1024 * 1024 * 1024,  # 1GB
                performance_rating=0.8,
                availability=True
            )
            self.resources[resource.resource_id] = resource
            
        for i in range(5):
            resource = MemoryResource(
                resource_id=f"RES_VRAM_{i:02d}",
                resource_type=MemoryResourceType.VRAM,
                quantity=2048 * 1024 * 1024,  # 2GB
                performance_rating=1.0,
                availability=True
            )
            self.resources[resource.resource_id] = resource
            
        for i in range(5):
            resource = MemoryResource(
                resource_id=f"RES_UHD_{i:02d}",
                resource_type=MemoryResourceType.UHD_BUFFER,
                quantity=256 * 1024 * 1024,  # 256MB
                performance_rating=0.7,
                availability=True
            )
            self.resources[resource.resource_id] = resource
    
    def _initialize_market_quotes(self):
        """Initialize market quotes for all resource types."""
        # Base prices for different memory types
        base_prices = {
            MemoryResourceType.L1_CACHE: Decimal('10.00'),
            MemoryResourceType.L2_CACHE: Decimal('5.00'),
            MemoryResourceType.L3_CACHE: Decimal('2.00'),
            MemoryResourceType.VRAM: Decimal('1.50'),
            MemoryResourceType.SYSTEM_RAM: Decimal('1.00'),
            MemoryResourceType.UHD_BUFFER: Decimal('0.80'),
            MemoryResourceType.GRID_LOCATION: Decimal('0.50'),
            MemoryResourceType.COHERENCE_SLOT: Decimal('0.30'),
            MemoryResourceType.MEMORY_BANDWIDTH: Decimal('0.20'),
            MemoryResourceType.PAGE_TABLES: Decimal('0.10'),
        }
        
        for resource_type, base_price in base_prices.items():
            quote = MarketQuote(
                resource_type=resource_type,
                bid_price=base_price - Decimal('0.05'),
                ask_price=base_price + Decimal('0.05'),
                spread=Decimal('0.10'),
                volume=0
            )
            self.market_quotes[resource_type] = quote
            self.price_history[resource_type] = deque(maxlen=1000)
            # Initialize some price points
            for _ in range(10):
                self.price_history[resource_type].append(float(base_price))
    
    def create_portfolio(self, owner_id: str) -> ResourcePortfolio:
        """Create a new resource trading portfolio."""
        with self.lock:
            portfolio_id = f"PORTFOLIO_{uuid.uuid4().hex[:8].upper()}"
            
            portfolio = ResourcePortfolio(
                portfolio_id=portfolio_id,
                owner_id=owner_id,
                cash_balance=Decimal('10000.00'),  # $10,000 starting capital
                total_value=Decimal('10000.00')
            )
            
            self.portfolios[portfolio_id] = portfolio
            return portfolio
    
    def place_trade(self, trade: CrossForexTrade) -> Tuple[bool, str]:
        """Place a cross-forex memory trade."""
        with self.lock:
            # Validate the trade
            validation_result = self._validate_trade(trade)
            if not validation_result[0]:
                return False, validation_result[1]
            
            # Check portfolio balance
            portfolio = self.portfolios.get(trade.trader_id)
            if not portfolio:
                return False, f"Portfolio {trade.trader_id} not found"
            
            # Calculate trade cost
            trade_cost = self._calculate_trade_cost(trade)
            
            if trade_cost > portfolio.cash_balance:
                return False, f"Insufficient balance: ${portfolio.cash_balance} < ${trade_cost}"
            
            # Execute trade
            success, message = self._execute_trade(trade, portfolio)
            
            if success:
                # Update portfolio
                if trade.order_type in [MarketOrderType.MARKET_BUY, MarketOrderType.LIMIT_BUY]:
                    portfolio.cash_balance -= trade_cost
                else:
                    portfolio.cash_balance += trade_cost
                
                # Update total value
                portfolio.total_value = self._calculate_portfolio_value(portfolio)
                portfolio.timestamp = time.time()
            
            return success, message
    
    def _validate_trade(self, trade: CrossForexTrade) -> Tuple[bool, str]:
        """Validate a cross-forex trade."""
        # Check if trader has valid portfolio
        if not self._portfolio_exists(trade.trader_id):
            return False, f"Portfolio {trade.trader_id} does not exist"
        
        # Check trade parameters
        if trade.quantity <= 0:
            return False, "Quantity must be positive"
        
        if trade.bid_credits < 0:
            return False, "Bid credits cannot be negative"
        
        # Check if resource type is valid
        if trade.resource_type not in self.market_quotes:
            return False, f"Invalid resource type: {trade.resource_type.value}"
        
        # Check if trade is expired
        if trade.expiry and time.time() > trade.expiry:
            return False, "Trade has expired"
        
        # Validate market order
        if trade.order_type in [MarketOrderType.MARKET_BUY, MarketOrderType.MARKET_SELL]:
            # Market orders should not have limit prices
            if trade.limit_price is not None:
                return False, "Market orders cannot have limit prices"
        
        # Validate limit order
        if trade.order_type in [MarketOrderType.LIMIT_BUY, MarketOrderType.LIMIT_SELL]:
            if trade.limit_price is None:
                return False, "Limit orders must have limit prices"
            if trade.limit_price <= 0:
                return False, "Limit price must be positive"
        
        return True, "Valid trade"
    
    def _calculate_trade_cost(self, trade: CrossForexTrade) -> Decimal:
        """Calculate the cost of a trade."""
        # Use market price if available, otherwise use limit price
        if trade.market_price > 0:
            price = trade.market_price
        elif trade.limit_price:
            price = trade.limit_price
        else:
            # Get current market price
            quote = self.market_quotes.get(trade.resource_type)
            if quote:
                price = quote.ask_price if 'BUY' in trade.order_type.name else quote.bid_price
            else:
                price = Decimal('0')
        
        # Calculate base cost
        cost = Decimal(str(trade.quantity)) * price * Decimal('1.001')  # Add small spread
        
        # Add fees
        fee = cost * Decimal('0.001')  # 0.1% fee
        trade.fee = fee
        
        return cost + fee
    
    def _execute_trade(self, trade: CrossForexTrade, portfolio: ResourcePortfolio) -> Tuple[bool, str]:
        """Execute a cross-forex trade."""
        try:
            # Update trade status
            trade.status = TradeStatus.EXECUTED
            trade.execution_time = time.time()
            
            # Get current market price if not provided
            if trade.market_price == 0:
                quote = self.market_quotes.get(trade.resource_type)
                if quote:
                    trade.market_price = quote.ask_price if 'BUY' in trade.order_type.name else quote.bid_price
            
            # Handle buy order
            if 'BUY' in trade.order_type.name:
                # Find available resource
                available_resource = self._find_available_resource(trade.resource_type, trade.quantity)
                if not available_resource:
                    return False, f"No available {trade.resource_type.value} resources"
                
                # Assign resource to portfolio
                resource_copy = MemoryResource(
                    resource_id=available_resource.resource_id,
                    resource_type=available_resource.resource_type,
                    quantity=trade.quantity,
                    performance_rating=available_resource.performance_rating,
                    availability=False,
                    current_owner=portfolio.portfolio_id
                )
                
                portfolio.resources[resource_copy.resource_id] = resource_copy
                available_resource.current_owner = portfolio.portfolio_id
                available_resource.availability = False
                
                # Calculate profit based on resource performance
                trade.profit = self._calculate_trade_profit(trade)
            
            # Handle sell order
            elif 'SELL' in trade.order_type.name:
                # Find resource in portfolio
                resource_to_sell = self._find_resource_in_portfolio(portfolio, trade.resource_type)
                if not resource_to_sell:
                    return False, f"No {trade.resource_type.value} resource found in portfolio"
                
                # Sell the resource
                del portfolio.resources[resource_to_sell.resource_id]
                self.resources[resource_to_sell.resource_id].availability = True
                self.resources[resource_to_sell.resource_id].current_owner = None
                
                # Calculate profit
                trade.profit = self._calculate_trade_profit(trade)
            
            # Add to active trades and history
            self.active_trades[trade.trade_id] = trade
            self.trade_history.append(trade)
            
            # Update market statistics
            self._update_market_statistics(trade)
            
            return True, f"Trade executed successfully: {trade.trade_id}"
            
        except Exception as e:
            trade.status = TradeStatus.FAILED
            return False, f"Trade execution failed: {str(e)}"
    
    def _find_available_resource(self, resource_type: MemoryResourceType, 
                               quantity: int) -> Optional[MemoryResource]:
        """Find an available resource of the specified type."""
        for resource in self.resources.values():
            if (resource.resource_type == resource_type and 
                resource.availability and 
                resource.quantity >= quantity):
                return resource
        return None
    
    def _find_resource_in_portfolio(self, portfolio: ResourcePortfolio, 
                                  resource_type: MemoryResourceType) -> Optional[MemoryResource]:
        """Find a resource of specified type in portfolio."""
        for resource in portfolio.resources.values():
            if resource.resource_type == resource_type:
                return resource
        return None
    
    def _calculate_trade_profit(self, trade: CrossForexTrade) -> Optional[Decimal]:
        """Calculate profit from a trade."""
        # Simplified profit calculation based on resource type and performance
        performance_factor = {
            MemoryResourceType.L1_CACHE: 1.5,
            MemoryResourceType.L2_CACHE: 1.3,
            MemoryResourceType.L3_CACHE: 1.2,
            MemoryResourceType.VRAM: 1.4,
            MemoryResourceType.SYSTEM_RAM: 1.0,
            MemoryResourceType.UHD_BUFFER: 0.9,
            MemoryResourceType.GRID_LOCATION: 1.1,
            MemoryResourceType.COHERENCE_SLOT: 1.6,
            MemoryResourceType.MEMORY_BANDWIDTH: 1.8,
            MemoryResourceType.PAGE_TABLES: 0.8,
        }.get(trade.resource_type, 1.0)
        
        base_profit = (trade.quantity * trade.market_price) * Decimal(str(performance_factor))
        return base_profit - trade.fee
    
    def _update_market_statistics(self, trade: CrossForexTrade):
        """Update market statistics after a trade."""
        # Update market quote
        quote = self.market_quotes.get(trade.resource_type)
        if quote:
            quote.volume += trade.quantity
            
            # Calculate price impact based on trade size and performance
            price_impact = Decimal('0.001') * Decimal(str(trade.quantity / 1000000000))  # Price impact per GB
            if 'BUY' in trade.order_type.name:
                quote.bid_price += price_impact
                quote.ask_price += price_impact
            else:
                quote.bid_price -= price_impact
                quote.ask_price -= price_impact
            
            quote.spread = quote.ask_price - quote.bid_price
            quote.timestamp = time.time()
            
            # Update price history
            if trade.resource_type in self.price_history:
                self.price_history[trade.resource_type].append(float(trade.market_price))
    
    def _portfolio_exists(self, portfolio_id: str) -> bool:
        """Check if a portfolio exists."""
        return portfolio_id in self.portfolios
    
    def _calculate_portfolio_value(self, portfolio: ResourcePortfolio) -> Decimal:
        """Calculate total value of a portfolio."""
        value = portfolio.cash_balance
        
        for resource in portfolio.resources.values():
            quote = self.market_quotes.get(resource.resource_type)
            if quote:
                price_per_unit = quote.bid_price
                resource_value = Decimal(str(resource.quantity)) * price_per_unit
                value += resource_value
        
        return value
    
    def get_portfolio_balance(self, portfolio_id: str) -> Optional[Decimal]:
        """Get balance of a portfolio."""
        portfolio = self.portfolios.get(portfolio_id)
        return portfolio.total_value if portfolio else None
    
    def get_market_quote(self, resource_type: MemoryResourceType) -> Optional[MarketQuote]:
        """Get current market quote for a resource type."""
        return self.market_quotes.get(resource_type)
    
    def get_resource_performance(self, resource: MemoryResource) -> float:
        """Get performance rating of a memory resource."""
        return resource.performance_rating
    
    def integrate_with_gamesa(self):
        """Integrate with GAMESA framework."""
        # Check for valid memory trading capability
        if not self.effect_checker.can_perform("memory_trader", Effect.MEMORY_CONTROL):
            print("Memory trader lacks RESOURCE_CONTROL capability!")
            return False
        
        # Validate safety constraints
        contract_result = self.contract_validator.check_invariants("memory_trading", {
            "total_traded_value": sum(float(trade.market_price * Decimal(str(trade.quantity))) 
                                    for trade in self.active_trades.values()),
            "portfolio_count": len(self.portfolios),
            "resource_availability": sum(1 for r in self.resources.values() if r.availability)
        })
        
        if not contract_result.valid:
            print(f"Memory trading validation failed: {contract_result.errors}")
            return False
        
        print("Memory trading integrated with GAMESA successfully")
        return True


class CrossForexManager:
    """Manager for cross-forex trading operations."""
    
    def __init__(self):
        self.memory_engine = MemoryMarketEngine()
        self.grid_memory_manager = GPUGridMemoryManager()
        
    def request_memory_resources(self, request: AllocationRequest) -> Allocation:
        """Request memory resources through cross-forex trading."""
        # Create a trade for the requested memory
        trade = CrossForexTrade(
            trade_id=f"MEM_{request.request_id}",
            trader_id=request.agent_id,
            order_type=MarketOrderType.MARKET_BUY,
            resource_type=MemoryResourceType.VRAM,  # Default to VRAM
            quantity=int(request.amount),  # Convert to int for quantity
            bid_credits=Decimal(str(request.bid_credits)),
            collateral=Decimal('100.00')  # Default collateral
        )
        
        # Place the trade
        success, message = self.memory_engine.place_trade(trade)
        
        if success:
            # Create allocation based on trade execution
            coord = MemoryGridCoordinate(tier=3, slot=0, depth=16)  # VRAM tier
            
            allocation = Allocation(
                allocation_id=trade.trade_id,
                request_id=request.request_id,
                agent_id=request.agent_id,
                resource_type=request.resource_type,
                amount=request.amount,
                granted_at=time.time(),
                expires_at=time.time() + (request.duration_ms / 1000.0),
                status="active" if success else "failed"
            )
            
            print(f"Memory allocation through cross-forex: {message}")
            return allocation
        else:
            print(f"Memory allocation failed: {message}")
            return Allocation(
                allocation_id=f"FAILED_{request.request_id}",
                request_id=request.request_id,
                agent_id=request.agent_id,
                resource_type=request.resource_type,
                amount=request.amount,
                granted_at=time.time(),
                expires_at=time.time(),
                status="failed"
            )
    
    def optimize_memory_trading(self, context: MemoryContext, amount: int) -> CrossForexTrade:
        """Optimize memory trading based on context."""
        # Determine optimal resource type based on context
        if context.performance_critical and context.compute_intensive:
            resource_type = MemoryResourceType.L1_CACHE
        elif context.performance_critical:
            resource_type = MemoryResourceType.VRAM
        elif context.gpu_preference == GPUType.UHD:
            resource_type = MemoryResourceType.UHD_BUFFER
        else:
            resource_type = MemoryResourceType.SYSTEM_RAM
        
        # Create optimized trade
        trade = CrossForexTrade(
            trade_id=f"OPT_{uuid.uuid4().hex[:8].upper()}",
            trader_id="SYSTEM_OPTIMIZER",
            order_type=MarketOrderType.MARKET_BUY,
            resource_type=resource_type,
            quantity=amount,
            bid_credits=Decimal('50.00'),  # System optimization credits
            metadata={
                "context": context,
                "optimization_goal": "performance",
                "grid_coord": "optimized"
            }
        )
        
        # Execute trade
        success, message = self.memory_engine.place_trade(trade)
        print(f"Optimized memory trading: {message}")
        
        return trade
    
    def get_memory_allocation(self, size: int, context: MemoryContext) -> GPUGridMemoryManager:
        """Get optimized memory allocation using cross-forex system."""
        # Optimize through trading
        trade = self.optimize_memory_trading(context, size)
        
        # Allocate through grid system
        allocation = self.grid_memory_manager.allocate_optimized(size, context)
        print(f"Memory allocated at grid location: {allocation.grid_coordinate}")
        
        return allocation


class MemoryTradingSignalProcessor:
    """Process signals for memory trading decisions."""
    
    def __init__(self, cross_forex_manager: CrossForexManager):
        self.cross_forex_manager = cross_forex_manager
        self.signal_history = deque(maxlen=100)
        
    def process_signal(self, signal: Signal) -> Optional[CrossForexTrade]:
        """Process a signal and generate appropriate trading action."""
        self.signal_history.append(signal)
        
        trade = None
        context = MemoryContext()
        
        if signal.kind == SignalKind.MEMORY_PRESSURE:
            # High memory pressure - trade for more memory
            context.performance_critical = True
            context.access_pattern = "random"
            trade = self.cross_forex_manager.optimize_memory_trading(
                context, int(signal.strength * 1024 * 1024 * 1024)  # Convert to bytes
            )
        
        elif signal.kind == SignalKind.THERMAL_WARNING:
            # Thermal pressure - trade for UHD buffer (cooler)
            context.performance_critical = True
            context.gpu_preference = GPUType.UHD
            context.access_pattern = "sequential"
            trade = self.cross_forex_manager.optimize_memory_trading(
                context, int(512 * 1024 * 1024)  # 512MB
            )
        
        elif signal.kind == SignalKind.USER_BOOST_REQUEST:
            # User requested boost - trade for high-performance memory
            context.performance_critical = True
            context.compute_intensive = True
            context.access_pattern = "burst"
            trade = self.cross_forex_manager.optimize_memory_trading(
                context, int(1024 * 1024 * 1024)  # 1GB
            )
        
        if trade:
            print(f"Signal processed: {signal.kind.value} -> Trade {trade.trade_id}")
        
        return trade


# Demo function
def demo_cross_forex_memory_trading():
    """Demonstrate the cross-forex memory trading system."""
    print("=== Cross-forex Memory Trading System Demo ===\n")
    
    # Initialize the cross-forex manager
    cross_forex_manager = CrossForexManager()
    signal_processor = MemoryTradingSignalProcessor(cross_forex_manager)
    
    # Integrate with GAMESA
    integration_success = cross_forex_manager.memory_engine.integrate_with_gamesa()
    print(f"GAMESA integration: {'Success' if integration_success else 'Failed'}")
    print()
    
    # Create a portfolio
    portfolio = cross_forex_manager.memory_engine.create_portfolio("TRADER_001")
    print(f"Portfolio created: {portfolio.portfolio_id}")
    print(f"Initial balance: ${portfolio.cash_balance}")
    print()
    
    # Simulate different memory allocation contexts
    contexts = [
        MemoryContext(
            access_pattern="sequential",
            performance_critical=True,
            compute_intensive=True,
            gpu_preference=GPUType.UHD
        ),
        MemoryContext(
            access_pattern="random",
            performance_critical=True,
            compute_intensive=False
        ),
        MemoryContext(
            access_pattern="burst",
            performance_critical=False,
            compute_intensive=False
        )
    ]
    
    print("Processing memory allocation requests...")
    for i, context in enumerate(contexts):
        print(f"\nContext {i+1}: Performance={context.performance_critical}, GPU={context.gpu_preference}")
        
        # Get optimized memory allocation
        allocation = cross_forex_manager.get_memory_allocation(
            size=512 * 1024 * 1024,  # 512MB
            context=context
        )
        
        print(f"  Allocated at grid: Tier={allocation.grid_coordinate.tier}, "
              f"Slot={allocation.grid_coordinate.slot}, "
              f"Depth={allocation.grid_coordinate.depth}")
        print(f"  Expected latency: {allocation.performance_metrics['latency']:.2f}ns")
    
    print("\nSimulating trading signals...")
    
    # Simulate signals
    signals = [
        Signal(
            id="SIG001",
            source="TELEMETRY",
            kind=SignalKind.MEMORY_PRESSURE,
            strength=0.8,
            confidence=0.9,
            payload={"pressure_level": "HIGH", "recommended_size": "1GB"}
        ),
        Signal(
            id="SIG002", 
            source="TELEMETRY",
            kind=SignalKind.THERMAL_WARNING,
            strength=0.6,
            confidence=0.85,
            payload={"temperature": 85, "recommendation": "move_to_uhd"}
        ),
        Signal(
            id="SIG003",
            source="USER",
            kind=SignalKind.USER_BOOST_REQUEST,
            strength=1.0,
            confidence=0.99,
            payload={"request_type": "performance_boost", "priority": "high"}
        )
    ]
    
    for signal in signals:
        print(f"\nProcessing signal: {signal.kind.value} (Strength: {signal.strength})")
        trade = signal_processor.process_signal(signal)
        
        if trade:
            print(f"  Trade executed: {trade.trade_id}")
            print(f"  Resource: {trade.resource_type.value}")
            print(f"  Quantity: {trade.quantity} units")
            print(f"  Cost: ${trade.market_price * Decimal(str(trade.quantity)) + trade.fee:.2f}")
        else:
            print("  No trading action taken")
    
    # Show final portfolio status
    final_portfolio = cross_forex_manager.memory_engine.portfolios[portfolio.portfolio_id]
    print(f"\nFinal Portfolio Status:")
    print(f"  Cash Balance: ${final_portfolio.cash_balance}")
    print(f"  Total Value: ${final_portfolio.total_value}")
    print(f"  Owned Resources: {len(final_portfolio.resources)}")
    
    # Show market quotes
    print(f"\nCurrent Market Quotes:")
    for res_type, quote in cross_forex_manager.memory_engine.market_quotes.items():
        print(f"  {res_type.value}: Bid=${quote.bid_price}, Ask=${quote.ask_price}, Spread=${quote.spread}")
    
    print(f"\nCross-forex memory trading demo completed successfully!")


if __name__ == "__main__":
    demo_cross_forex_memory_trading()