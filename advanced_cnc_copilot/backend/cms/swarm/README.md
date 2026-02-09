# Anti-Fragile Marketplace - Fleet Intelligence System

The Anti-Fragile Marketplace is the core intelligence system for the FANUC RISE v2.1 Advanced CNC Copilot. It implements a revolutionary approach to G-Code strategy evaluation that prioritizes resilience over raw speed, using biological metaphors and collective intelligence to create manufacturing systems that become stronger under stress.

## Core Components

### 1. Survivor Ranking System (`survivor_ranking.py`)
The Survivor Ranking System evaluates G-Code strategies based on their ability to withstand chaotic conditions rather than just their performance under ideal circumstances. It implements the "Death Penalty Function" which assigns zero fitness to any strategy that violates physical constraints.

**Key Features:**
- **Stress Testing**: Evaluates strategies under simulated adverse conditions
- **Survivor Badges**: Awards badges based on resilience rather than speed
- **Anti-Fragile Scoring**: Rates strategies on their ability to improve under stress
- **Physics Validation**: Ensures all strategies comply with physical constraints

### 2. Economic Auditor (`economic_auditor.py`)
The Economic Auditor calculates the true value of shared knowledge by quantifying the "Cost of Ignorance" versus the "Value of Shared Knowledge." It produces Fleet Savings Reports that measure the ROI of the Hive Mind system in real dollars.

**Key Features:**
- **Cost of Ignorance Tracking**: Records losses from preventable failures
- **Fleet Savings Calculation**: Quantifies value of shared trauma across machines
- **ROI Metrics**: Calculates return on investment for collective intelligence
- **Economic Impact Analysis**: Measures financial benefits of the system

### 3. Genetic Tracker (`genetic_tracker.py`)
The Genetic Tracker maintains the "Genealogy of Code" - tracking how G-Code strategies evolve and mutate across the fleet. It creates evolution trees showing how toolpaths improve through the collective intelligence of the swarm.

**Key Features:**
- **Mutation Tracking**: Records every change to G-Code strategies
- **Lineage Trees**: Shows evolutionary paths from root strategies
- **Genetic Diversity Metrics**: Measures how strategies have evolved
- **Ancestry Analysis**: Tracks shared origins of different strategies

### 4. Anti-Fragile Marketplace (`anti_fragile_marketplace.py`)
The main orchestration engine that combines all components into a unified system for ranking G-Code strategies by resilience rather than speed.

**Key Features:**
- **Unified Ranking**: Combines survivor scores, economic value, and genetic fitness
- **Marketplace Dynamics**: Creates competitive environment for resilient strategies
- **Collective Learning**: Shares successful strategies across the fleet
- **Performance Tracking**: Monitors improvement over time

## Theoretical Foundations

The system implements several key theoretical concepts:

### 1. The Great Translation
Maps SaaS metrics (Churn, CAC, LTV) to manufacturing physics (Tool Wear, Setup Time, Cycle Time) to create economic optimization algorithms that understand the physics of manufacturing.

### 2. Quadratic Mantinel
Implements physics-informed geometric constraints where Speed = f(Curvature²), ensuring that toolpaths maintain momentum through high-curvature sections without causing servo jerks.

### 3. Shadow Council Governance
Uses three-agent validation (Creator, Auditor, Accountant) where probabilistic AI suggestions are filtered through deterministic physics constraints before execution.

### 4. Neuro-Safety Gradients
Replaces binary safe/unsafe states with continuous dopamine/cortisol gradients that provide nuanced safety responses based on proximity to dangerous conditions.

### 5. Nightmare Training
Enables offline learning where the system replays historical operations during idle time, injecting failure scenarios to improve resilience without risking physical hardware.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Machine A     │    │   Hive Mind      │    │   Machine B     │
│                 │◄──►│  (Shared Memory) │◄──►│                 │
│  ┌─────────────┐│    │                  │    │  ┌─────────────┐│
│  │Survivor Rank││    │ ┌──────────────┐ │    │  │Survivor Rank││
│  │    ing      │├────┼─┤Trauma Registry│ │◄───┤  │    ing      ││
│  └─────────────┘│    │ └──────────────┘ │    │  └─────────────┘│
│  ┌─────────────┐│    │ ┌──────────────┐ │    │  ┌─────────────┐│
│  │Economic Audit││    │ │Strategy Scores │ │    │  │Economic Audit││
│  │     or      │├────┼─┤   (Global)   │ │◄───┤  │     or      ││
│  └─────────────┘│    │ └──────────────┘ │    │  └─────────────┘│
│  ┌─────────────┐│    │ ┌──────────────┐ │    │  ┌─────────────┐│
│  │Genetic Track ││    │ │Mutation Trees│ │    │  │Genetic Track ││
│  │     er      │├────┼─┤   (Shared)   │ │◄───┤  │     er      ││
│  └─────────────┘│    │ └──────────────┘ │    │  └─────────────┘│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Usage Examples

### Submitting a Strategy to the Marketplace

```python
from cms.swarm.anti_fragile_marketplace import AntiFragileMarketplace

# Initialize the marketplace
marketplace = AntiFragileMarketplace()

# Submit a G-Code strategy for evaluation
listing = marketplace.submit_strategy(
    strategy_id="STRAT_INCONEL_FACE_MILL_001",
    strategy_name="Inconel Face Mill - Conservative Approach",
    material="Inconel-718",
    operation_type="face_mill",
    parameters={
        "feed_rate": 1500,
        "rpm": 4000,
        "depth": 1.0,
        "width": 10.0
    },
    author="CNC_Operator_001",
    description="Conservative parameters for machining Inconel-718 to minimize tool wear"
)

print(f"Strategy scored {listing.survivor_badge.survivor_score:.3f} on resilience scale")
print(f"Economic value: {listing.economic_value:.3f}")
print(f"Badge level: {listing.survivor_badge.badge_level}")
```

### Getting Top Strategies

```python
# Get top strategies for a specific material and operation
top_strategies = marketplace.get_top_strategies(
    material="Inconel-718",
    operation_type="face_mill",
    min_survivor_score=0.5,
    limit=5
)

for i, strategy in enumerate(top_strategies, 1):
    print(f"{i}. {strategy.strategy_name}")
    print(f"   Score: {strategy.survivor_badge.survivor_score:.3f}")
    print(f"   Economic Value: {strategy.economic_value:.3f}")
    print(f"   Tags: {', '.join(strategy.tags)}")
```

### Running Marketplace Simulation

```python
# Run a simulation to test resilience under stress conditions
simulation_results = marketplace.run_marketplace_simulation(duration_hours=1.0)

print(f"Simulation completed with {simulation_results['strategies_evaluated']} strategies")
print(f"Stress tests performed: {simulation_results['stress_tests_performed']}")
print(f"Economic impact: ${simulation_results['economic_impact']:,.2f}")

# View top performers from the simulation
for perf in simulation_results['top_performers'][:3]:
    print(f"  Top performer: {perf['name']} (Score: {perf['survivor_score']:.3f})")
```

### Getting Strategy Genealogy

```python
# Get the complete evolution history of a strategy
genealogy = marketplace.get_strategy_genealogy("STRAT_INCONEL_FACE_MILL_001")

print(f"Strategy {genealogy['strategy_id']} has {genealogy['listing_info']['generation_count']} generations")
print(f"Related strategies: {len(genealogy['related_strategies'])}")
print(f"Genetic diversity: {genealogy['genetic_diversity']:.3f}")
```

## Economic Impact

The Anti-Fragile Marketplace delivers measurable economic benefits:

- **Reduced Tool Breakage**: By sharing trauma across the fleet, each tool breakage incident prevents similar incidents across all machines
- **Optimized Parameters**: Collective learning identifies optimal parameters faster than individual machine learning
- **Preventive Maintenance**: Early detection of stress patterns enables proactive maintenance
- **Quality Improvements**: Shared knowledge of successful strategies improves overall product quality

## Integration with Existing Systems

The marketplace integrates with the broader FANUC RISE system:

- **Shadow Council**: Uses the three-agent validation system for strategy approval
- **Dopamine Engine**: Incorporates neuro-safety gradients for nuanced safety responses
- **Economics Engine**: Factors in economic impact when ranking strategies
- **Telemetry System**: Uses real-time data for validation and improvement
- **API Connections**: Integrates with CAD/CAM systems for automatic optimization

## Benefits

1. **Collective Intelligence**: Lessons learned by one machine benefit the entire fleet
2. **Resilience Over Speed**: Prioritizes reliable operation over maximum performance
3. **Economic Optimization**: Balances safety with profitability
4. **Adaptive Learning**: Continuously improves through Nightmare Training
5. **Risk Reduction**: Minimizes costly failures through shared experience
6. **Scalability**: Improves as more machines join the fleet

## Future Enhancements

- **Advanced Genetic Algorithms**: More sophisticated mutation and crossover operations
- **Real-time Adaptation**: Dynamic strategy adjustment based on current conditions
- **Cross-Factory Learning**: Sharing strategies across different manufacturing facilities
- **Predictive Analytics**: Anticipating failure modes before they emerge
- **Automated Optimization**: Self-improving strategies based on performance feedback

The Anti-Fragile Marketplace represents a paradigm shift from individual machine optimization to collective fleet intelligence, creating manufacturing systems that become stronger through adversity rather than brittle under stress.