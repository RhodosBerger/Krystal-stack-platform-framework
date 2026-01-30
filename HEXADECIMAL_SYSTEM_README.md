# Hexadecimal System with ASCII Rendering and Composition Generator

This module implements a hexadecimal-based system for the GAMESA framework, including an ASCII rendering engine and composition generator that works with the existing essential encoder and OpenVINO integration.

## Components

### 1. Hexadecimal Trading System (`hexadecimal_system.py`)

The hexadecimal system implements a trading protocol using hexadecimal values for resource allocation and management:

- **Hexadecimal Commodities**: Different types of resources represented as hex values
- **Depth Levels**: Trading intensity controls (Minimal to Maximum)
- **Market Regulation**: Safety limits and order clearing mechanisms
- **Trade Execution**: Secure trading with validation

#### Hexadecimal Commodity Types:
- **HEX_COMPUTE** (0x00-0x1F): Compute resources
- **HEX_MEMORY** (0x20-0x3F): Memory resources
- **HEX_IO** (0x40-0x5F): I/O resources
- **HEX_GPU** (0x60-0x7F): GPU resources
- **HEX_NEURAL** (0x80-0x9F): Neural processing
- **HEX_CRYPTO** (0xA0-0xBF): Cryptographic resources
- **HEX_RENDER** (0xC0-0xDF): Rendering resources
- **HEX_SYSTEM** (0xE0-0xFF): System resources

#### Depth Levels:
- **MINIMAL** (0x10): Low restriction
- **LOW** (0x30): Low restriction
- **MODERATE** (0x50): Moderate restriction
- **HIGH** (0x80): High restriction
- **EXTREME** (0xC0): Extreme restriction
- **MAXIMUM** (0xFF): Maximum restriction

### 2. ASCII Rendering Engine

The ASCII rendering engine provides visualization capabilities:

- **Hex Digit Rendering**: ASCII art representation of hexadecimal digits (0-F)
- **Market State Visualization**: ASCII rendering of current market conditions
- **Composition Rendering**: Visual representation of resource compositions
- **Scalable Output**: Adjustable scaling for larger ASCII displays

### 3. Composition Generator

The composition generator creates resource allocations based on market patterns:

- **Pattern Detection**: Identifies optimal hex value patterns based on market conditions
- **Resource Allocation**: Distributes resources based on hex values
- **Efficiency Scoring**: Calculates efficiency scores for compositions
- **Risk Assessment**: Evaluates risk levels for generated compositions
- **Optimization**: Optimizes compositions for specific targets

### 4. Pattern Matching System

The pattern matching system analyzes market data:

- **Market Pattern Detection**: Identifies trends in hex value distributions
- **Optimal Value Selection**: Finds optimal hex values based on market state
- **Volatility Analysis**: Assesses market volatility for pattern matching
- **Demand Pressure Analysis**: Evaluates market demand for pattern selection

## Integration with GAMESA Framework

The hexadecimal system integrates seamlessly with the existing GAMESA framework:

- **Economic Resource Trading**: Hexadecimal commodities traded like financial assets
- **Safety Constraints**: Formal verification of hex trades
- **Telemetry Integration**: Market patterns based on system telemetry
- **Metacognitive Analysis**: AI-driven optimization of hex patterns
- **Cross-Forex Resource Market**: Hex resources traded across different domains

## Installation

The hexadecimal system is part of the broader GAMESA framework and requires:

1. Core dependencies (already included in the framework):
```bash
pip install numpy
```

2. For full functionality:
```bash
pip install openvino  # For OpenVINO integration
```

## Usage

### Basic Hexadecimal Trading

```python
from hexadecimal_system import HexadecimalSystem, HexCommodityType, HexDepthLevel

# Create hexadecimal system
hex_system = HexadecimalSystem()

# Create a compute commodity
compute_commodity = hex_system.create_commodity(
    HexCommodityType.HEX_COMPUTE,
    quantity=100.0,
    depth_level=HexDepthLevel.MODERATE
)

# Execute a trade
trade = hex_system.execute_trade(
    compute_commodity.commodity_id,
    "BuyerAgent",
    "SellerAgent",
    price=150.0
)
```

### ASCII Rendering

```python
from hexadecimal_system import ASCIIHexRenderer

# Create renderer
renderer = ASCIIHexRenderer()

# Render a hex value
hex_art = renderer.render_hex_value(0xAB, scale=1)
print(hex_art)

# Render market state
market_art = renderer.render_market_state(hex_system)
print(market_art)
```

### Composition Generation

```python
from hexadecimal_system import CompositionGenerator

# Create generator
generator = CompositionGenerator()

# Generate composition based on market state
market_state = {
    'demand_pressure': 0.6,
    'volatility': 0.3,
    'trend': 'up'
}

composition = generator.generate_composition(market_state)
print(f"Efficiency: {composition['efficiency_score']:.2f}")
print(f"Risk: {composition['risk_level']}")
```

### Pattern Matching

```python
from hexadecimal_system import HexPatternMatcher

# Create pattern matcher
matcher = HexPatternMatcher()

# Find optimal hex values for current market
optimal_values = matcher.find_optimal_hex_values(market_state)
print(f"Optimal hex values: {[f'0x{v:02X}' for v in optimal_values]}")

# Detect market patterns
patterns = matcher.detect_market_patterns(hex_system)
print(f"Detected patterns: {patterns}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                HEXADECIMAL TRADING SYSTEM                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Commodity   │  │ Market      │  │ Trade Executor      │ │
│  │ Manager     │  │ Regulator   │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                      │            │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Hexadecimal Trading Engine                 │ │
│  │         (Validation, Clearing, Execution)              │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 ASCII RENDERING ENGINE                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Hex Digit   │  │ Market      │  │ Composition         │ │
│  │ Renderer    │  │ Renderer    │  │ Renderer            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                      │            │
│  ┌─────────────────────────────────────────────────────────┤
│  │              ASCII Visualization                        │ │
│  │         (Digit art, Market display, Composition)       │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               COMPOSITION GENERATOR                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Pattern     │  │ Resource    │  │ Optimization        │ │
│  │ Matcher     │  │ Allocator   │  │ Engine              │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                      │            │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Composition Engine                         │ │
│  │         (Generation, Scoring, Risk Assessment)         │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Hexadecimal Trading
- Secure trading with validation and clearing
- Depth-based trading controls
- Market regulation and safety limits
- Comprehensive trade history

### ASCII Rendering
- Visual representation of hex values
- Market state visualization
- Composition display
- Scalable ASCII output

### Composition Generation
- Pattern-based resource allocation
- Efficiency scoring
- Risk assessment
- Optimization capabilities

### Pattern Matching
- Market trend detection
- Optimal value selection
- Volatility analysis
- Demand pressure evaluation

## Integration Points

The system integrates with:
- Essential encoder for data processing
- OpenVINO for hardware acceleration
- GAMESA framework for resource management
- Telemetry systems for market data
- Safety systems for validation

## Testing

Run the comprehensive test suite:

```bash
python test_hexadecimal_system.py
```

This will test all components and verify the integration between hexadecimal trading, ASCII rendering, and composition generation.