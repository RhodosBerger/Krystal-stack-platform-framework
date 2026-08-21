# FANUC RISE v2.1 - Operator's Handbook
## Working with the Cognitive Manufacturing System

---

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding the Shadow Council](#understanding-the-shadow-council)
3. [Neuro-Safety Visualization Guide](#neuro-safety-visualization-guide)
4. [Interacting with the System](#interacting-with-the-system)
5. [Troubleshooting Common Scenarios](#troubleshooting-common-scenarios)
6. [Safety Protocols](#safety-protocols)
7. [Performance Optimization Tips](#performance-optimization-tips)

---

## Introduction

Welcome to the FANUC RISE v2.1 Advanced CNC Copilot system! This cognitive manufacturing platform introduces a revolutionary approach to CNC operations using the "Shadow Council" governance pattern. Unlike traditional systems that rely solely on deterministic execution, RISE v2.1 employs probabilistic decision-making while maintaining strict safety protocols.

The system consists of three intelligent agents working together:
- **Creator Agent**: Proposes optimizations based on current conditions
- **Auditor Agent**: Validates proposals against physics constraints and safety requirements
- **Accountant Agent**: Evaluates economic impact of proposed changes

Your role as an operator is to work collaboratively with these agents, interpreting their decisions and providing human oversight when needed.

---

## Understanding the Shadow Council

### Decision-Making Process
The Shadow Council operates on a three-agent consensus model:

1. **Proposal Phase**: The Creator Agent analyzes current conditions and proposes parameter optimizations
2. **Validation Phase**: The Auditor Agent validates proposals against physics constraints using the "Quadratic Mantinel" and "Death Penalty Function"
3. **Economic Evaluation Phase**: The Accountant Agent assesses the financial impact of the proposal
4. **Execution Phase**: If all agents approve, the system executes the optimized parameters

### Decision Outcomes
- **APPROVED**: All agents agreed; optimized parameters are implemented
- **REJECTED**: One or more agents vetoed the proposal; system continues with current parameters
- **MODIFIED**: Auditor approved with parameter adjustments for safety

### Reading Decision Logs
On the Shadow Council Console, you'll see decision traces showing:
- Reasoning for each agent's decision
- Physics constraints that were checked
- Economic impact calculations
- Final approval/rejection status

---

## Neuro-Safety Visualization Guide

### Understanding the Visual Indicators

The system uses neuroscience-inspired visualization to communicate safety and performance states:

#### Dopamine Levels (Reward/Optimization)
- **Visual Indicator**: Green-blue pulsing border
- **Meaning**: System efficiency and economic reward potential
- **Levels**:
  - **Low** (0.0-0.3): Conservative operation, prioritizing safety over efficiency
  - **Medium** (0.3-0.7): Balanced operation, optimizing for both safety and efficiency
  - **High** (0.7-1.0): Aggressive optimization, high efficiency with acceptable safety margins

#### Cortisol Levels (Stress/Risk)
- **Visual Indicator**: Orange-red pulsing border
- **Meaning**: System stress and risk assessment
- **Levels**:
  - **Low** (0.0-0.3): Safe operating conditions, minimal stress
  - **Caution** (0.3-0.6): Elevated stress, system adapting to challenges
  - **Warning** (0.6-0.8): High stress, conservative approach activated
  - **Critical** (0.8-1.0): Dangerous stress levels, safety-first mode

#### Combined Neuro-State
- **Green Dominant**: Optimal conditions, system performing well
- **Yellow Mixed**: Balanced state, system adapting appropriately
- **Red Dominant**: High-risk situation, human attention required

### Interpreting the Glass Brain Interface

The Glass Brain Interface makes the AI's decision-making transparent:

```
┌─────────────────────────────────────────────────────────┐
│  NEURO-SAFETY VISUALIZATION                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  DOPAMINE   │  │   CORTISOL  │  │  BALANCE  │      │
│  │    0.75     │  │    0.22     │  │   0.53    │      │
│  │   (Reward)  │  │   (Stress)  │  │ (Safety)  │      │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

- The **Dopamine** gauge shows the system's reward state (efficiency/profit potential)
- The **Cortisol** gauge shows the system's stress state (risk/safety concerns)
- The **Balance** gauge shows the overall neuro-safety state

### Real-Time Feedback Interpretation

- **Pulsing Animation**: Indicates system activity and decision-making
- **Pulse Speed**: Faster pulses indicate more frequent optimization attempts
- **Pulse Intensity**: Brighter pulses indicate stronger neuro-chemical signals
- **Color Shifts**: Changes in color indicate transitions between operational modes

---

## Interacting with the System

### Operator Dashboard Navigation

The React-based operator dashboard provides real-time insights:

#### 1. Machine Status Panel
- Shows current operational parameters
- Displays Shadow Council decision status
- Provides immediate feedback on system health

#### 2. Neuro-Safety Gradient Display
- Visualizes dopamine and cortisol levels
- Indicates current operational mode
- Shows stress trend analysis

#### 3. Economic Metrics Panel
- Shows current profit rate (Pr = (Sales_Price - Cost) / Time)
- Displays quality yield percentages
- Tracks tool wear and maintenance needs

#### 4. Shadow Council Log
- Shows recent decisions and their reasoning
- Provides transparency into AI decision-making
- Highlights any rejected proposals and reasons

### Submitting Parameter Requests

To request parameter changes:

1. **Identify Opportunity**: Notice conditions that might benefit from optimization
2. **Submit Request**: Use the "Optimization Request" form in the dashboard
3. **Wait for Approval**: Shadow Council will evaluate and respond within 100ms
4. **Monitor Results**: Watch for implementation and performance impact

### Understanding Rejection Reasons

When the Auditor Agent rejects a proposal, common reasons include:

#### Physics Constraint Violations
- **Feed Rate vs. Curvature**: Quadratic Mantinel prevents servo jerk during tight curves
- **Temperature Limits**: Preventing thermal runaway during high-load operations
- **Vibration Thresholds**: Maintaining surface finish quality and tool life
- **Spindle Load Limits**: Preventing tool breakage and motor damage

#### Economic Considerations
- **Tool Wear Impact**: High-speed operations may reduce tool life disproportionately
- **Quality Risk**: Aggressive parameters may compromise surface finish
- **Safety Margins**: Insufficient buffer for unexpected material variations

---

## Troubleshooting Common Scenarios

### Scenario 1: Auditor Rejected Feed Rate Increase
**Problem**: Creator proposed higher feed rate, but Auditor rejected it
**Solution**: 
- Check current material hardness and tool condition
- Verify part geometry for tight radii that might require slower feeds
- Review temperature and vibration levels that may be approaching limits

**Physics Behind It**: The Quadratic Mantinel enforces that feed rate is limited by path curvature (feed ≤ k × √(radius)). During tight curves, the system automatically reduces feed to prevent servo jerk and chatter.

### Scenario 2: High Cortisol (Stress) Levels
**Problem**: System shows elevated stress indicators
**Solution**:
- Check coolant flow and temperature
- Inspect tool for wear or damage
- Verify material clamping and fixturing
- Consider switching to ECONOMY mode temporarily

**Physics Behind It**: Cortisol elevation indicates potential constraint violations. The system is protecting itself from damage while searching for safer operating parameters.

### Scenario 3: Low Dopamine (Reward) Levels
**Problem**: System shows low efficiency/reward indicators
**Solution**:
- Check if conservative parameters are limiting productivity
- Verify material properties match expected values
- Consider if current operation could benefit from optimization
- Monitor for opportunities to improve efficiency safely

**Physics Behind It**: Low dopamine indicates the system is operating below its potential, possibly due to overly conservative parameters or suboptimal conditions.

### Scenario 4: Phantom Trauma Detection
**Problem**: System shows high stress without apparent physical cause
**Solution**:
- Check for sensor drift or calibration issues
- Verify that actual machine conditions match reported values
- May indicate overly sensitive safety thresholds that need adjustment

**Physics Behind It**: Phantom trauma occurs when the system responds to sensor artifacts rather than real physical conditions. The system distinguishes between actual stress events and sensor anomalies.

---

## Safety Protocols

### Emergency Procedures
1. **Immediate Stop**: Press the emergency stop button on the machine or dashboard
2. **System Override**: Use the manual override to bypass Shadow Council decisions
3. **Safe State**: System will automatically revert to conservative parameters
4. **Restart Protocol**: Follow guided restart procedure through dashboard

### Understanding the Death Penalty Function
The system implements a "Death Penalty Function" that immediately rejects any proposal violating critical safety constraints:

- Spindle load > 95% of rated capacity
- Temperature > 75°C 
- Vibration > 4.0G
- Feed rate incompatible with tool geometry
- RPM incompatible with material properties

### Neuro-Safety Overrides
Under extreme conditions, the system may override operator inputs:
- If cortisol levels exceed 0.95, system enters safety-only mode
- Economic optimization is suspended until safe conditions return
- All decisions require unanimous Shadow Council approval

---

## Performance Optimization Tips

### Working with the Creator Agent
- Provide feedback on successful vs. unsuccessful manual adjustments
- Share insights about material properties and optimal parameters
- Use the "Learning Feedback" feature to teach the system

### Collaborating with the Auditor Agent
- Understand that rejections are protective, not punitive
- Learn which parameter combinations are consistently rejected
- Trust the physics-based constraints for long-term reliability

### Supporting the Accountant Agent
- Focus on economic metrics alongside pure technical performance
- Understand how tool life affects long-term profitability
- Balance short-term efficiency with long-term sustainability

### Maximizing System Performance
1. **Consistent Material Properties**: Ensure materials match expected specifications
2. **Proper Tool Maintenance**: Regular inspection and replacement schedules
3. **Optimal Fixturing**: Secure workpiece holding minimizes vibration
4. **Coolant Management**: Proper flow and temperature for optimal thermal control
5. **Environmental Stability**: Minimize external factors affecting machine performance

---

## Frequently Asked Questions

**Q: Why did the system reject my aggressive feed rate proposal?**
A: The Auditor Agent likely detected a potential constraint violation. High feed rates during tight curves can cause servo jerk, compromising surface finish and potentially damaging tools. The Quadratic Mantinel enforces physics-based constraints.

**Q: What does it mean when the borders are pulsing rapidly?**
A: Rapid pulsing indicates the system is actively making optimization decisions. Faster pulses suggest more frequent attempts to adjust parameters. If accompanied by high cortisol, the system may be struggling with challenging conditions.

**Q: How do I know if the system is performing optimally?**
A: Look for sustained high dopamine levels (>0.7) combined with low cortisol levels (<0.4). The "Balance" indicator should be positive, and you should see consistent economic performance in the metrics panel.

**Q: Can I override Shadow Council decisions?**
A: Yes, emergency overrides exist for critical situations. However, frequent overrides may indicate the system needs recalibration or that operating conditions are outside expected parameters. Use sparingly and document reasons.

---

## Contact and Support

For technical support:
- **System Status**: Dashboard → Support Panel → System Health
- **Knowledge Base**: Dashboard → Resources → Knowledge Base
- **Escalation**: Dashboard → Support Panel → Contact Engineer
- **Shadow Council Queries**: Dashboard → Governance → Submit Query

---

*FANUC RISE v2.1 - Transforming deterministic execution into probabilistic creation*
*The Industrial Organism is now awake and ready to work with you*