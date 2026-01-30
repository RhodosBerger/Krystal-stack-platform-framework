# 🧠 LLM Integration Guide
**Connecting Intelligence Everywhere in CNC Copilot**

---

## 🎯 Overview

This system uses **LLM (Large Language Models)** to connect unrelated data across logging sessions and infer practical manufacturing insights.

### **Conda Environment Setup**

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate cnc-copilot-llm

# Verify installation
python -c "from cms.llm_connector import LLMConnector; print('LLM Ready!')"
```

---

## 🔧 Quick Start

### **1. Set API Keys**

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
EOF
```

### **2. Initialize LLM Connector**

```python
from cms.llm_connector import LLMConnector

# Option 1: OpenAI (GPT-4)
llm = LLMConnector()  # Uses OpenAI by default

# Option 2: Anthropic (Claude)
from cms.llm_connector import LLMConfig
config = LLMConfig(provider='anthropic', model='claude-3-opus-20240229')
llm = LLMConnector(config)

# Option 3: Local LLM (Privacy-first)
config = LLMConfig(
    provider='local',
    local_model_path='models/llama-2-7b-chat.gguf'
)
llm = LLMConnector(config)
```

---

## 💡 Core Capabilities

### **1. Connect Unrelated Data**

**Problem:** You have data from different sessions that aren't obviously related

```python
from cms.cross_session_intelligence import CrossSessionIntelligence, DataPoint
from datetime import datetime, timedelta

# Initialize intelligence engine
intelligence = CrossSessionIntelligence()

# Add data from Session A (3 days ago)
intelligence.add_data_point(DataPoint(
    session_id='session_monday',
    timestamp=datetime.now() - timedelta(days=3),
    data_type='tool_wear',
    data={'wear_mm': 0.15, 'cuts_completed': 450},
    machine_id='CNC_001'
))

# Add data from Session B (today - different session!)
intelligence.add_data_point(DataPoint(
    session_id='session_thursday',
    timestamp=datetime.now(),
    data_type='vibration',
    data={'x': 2.5, 'y': 2.8, 'z': 2.3},
    machine_id='CNC_001'
))

# LLM connects them!
connections = intelligence.connect_unrelated_events(time_window_hours=72)
print(connections)
# Output: "High vibration pattern matches pre-failure signature from Monday session"
```

---

### **2. Infer Practical Relationships**

```python
# Two seemingly unrelated data points
data_a = {
    'type': 'ambient_temperature',
    'value': 32,  # Celsius
    'timestamp': '2024-01-20 08:00'
}

data_b = {
    'type': 'tool_wear_rate',
    'value': 0.8,  # mm/hour
    'timestamp': '2024-01-20 10:00'
}

# Ask LLM if they're related
relationship = llm.infer_relationship(data_a, data_b)

print(relationship)
# {
#     'has_relationship': True,
#     'relationship_type': 'causal',
#     'confidence': 0.85,
#     'explanation': 'Higher ambient temperature causes thermal expansion...',
#     'practical_action': 'Compensate cutting parameters by 3% when temp > 30C'
# }
```

---

### **3. Predict Future Events**

```python
# Current indicators
current_state = {
    'vibration_x': 2.6,
    'vibration_y': 2.9,
    'spindle_load': 85,
    'temperature': 65
}

# Predict tool failure
prediction = intelligence.predict_future_event(
    event_type='tool_failure',
    current_indicators=current_state
)

print(f"Failure likelihood: {prediction['prediction_confidence']*100:.0f}%")
print(f"Estimated time: {prediction['estimated_time_minutes']:.0f} minutes")
print(f"Recommended action: {prediction['llm_analysis']}")

# Output:
# Failure likelihood: 82%
# Estimated time: 38 minutes
# Recommended action: Stop machine and change tool immediately
```

---

### **4. Natural Language Queries**

```python
# Ask questions like talking to an expert
answer = intelligence.ask_question("Why did quality drop yesterday at 2 PM?")
print(answer)

# LLM analyzes ALL data and responds:
# "Quality drop at 14:00 correlates with:
#  1. Coolant temperature increased by 5°C at 13:45
#  2. Similar pattern occurred on Jan 15 with same result
#  3. Root cause: Coolant pump degradation
#  4. Recommendation: Replace coolant pump"

# More questions:
intelligence.ask_question("Which machine performs best?")
intelligence.ask_question("Should I change cutting speed?")
intelligence.ask_question("What causes excessive vibration?")
```

---

### **5. Find Similar Historical Sessions**

```python
# Current session data
current = {
    'vibration': {'x': 2.5, 'y': 2.7},
    'load': 78,
    'rpm': 3000
}

# Find similar sessions from history
similar = llm.find_similar_sessions(current, top_k=5)

for session in similar:
    print(f"Session {session['metadata']['session_id']}")
    print(f"  Similarity: {session['similarity_score']:.2f}")
    print(f"  Timestamp: {session['metadata']['timestamp']}")
    print(f"  Outcome: {session['session_data'].get('outcome', 'unknown')}")
```

---

## 🏭 Real-World Use Cases

### **Use Case 1: Root Cause Analysis**

```python
# Multiple unrelated symptoms
symptoms = [
    {'type': 'part_quality', 'value': 'out_of_tolerance', 'time': '08:15'},
    {'type': 'vibration', 'value': 'increased_15%', 'time': '08:10'},
    {'type': 'hvac_temp', 'value': '18C', 'time': '07:45'},
    {'type': 'floor_expansion', 'value': '0.2mm', 'time': '08:00'},
]

# LLM finds the chain
insights = llm.connect_unrelated_data(symptoms, purpose='root_cause_analysis')

# Output:
# "Cold HVAC (18C) → Floor contraction → Machine base shift → 
#  Increased vibration → Part out of tolerance"
```

---

### **Use Case 2: Predictive Maintenance**

```python
# Continuous monitoring
while machining:
    # Collect current data
    current_state = sensor_system.get_current_state()
    
    # Check for early warning signs
    prediction = intelligence.predict_future_event(
        event_type='bearing_failure',
        current_indicators=current_state
    )
    
    if prediction['prediction_confidence'] > 0.75:
        # LLM found pattern matching previous failures
        send_alert(f"Bearing failure likely in {prediction['estimated_time_minutes']} min")
        schedule_maintenance()
```

---

### **Use Case 3: Quality Investigation**

```python
# Quality defect detected
defect_data = {
    'part_id': 'P12345',
    'defect_type': 'surface_finish',
    'measurement': 'Ra=3.2 (spec: Ra<1.6)',
    'timestamp': '2024-01-24 10:30'
}

# Find when this happened before
similar_defects = intelligence.find_related_sessions(defect_data)

# LLM analysis
answer = intelligence.ask_question(
    f"Why does part {defect_data['part_id']} have poor surface finish?"
)

# Answer might reveal:
# "3 previous occurrences, all when:
#  - Coolant flow was low
#  - Spindle run time > 4 hours
#  - Ambient humidity < 30%"
```

---

### **Use Case 4: Process Optimization**

```python
# Weekly insights report
report = intelligence.generate_insights_report(days=7)

print("Key Insights:")
for insight in report['key_insights']:
    print(f"  - {insight}")

print("\nTop Recommendations:")
for rec in report['recommendations']:
    if rec['priority'] == 'high':
        print(f"  [{rec['priority'].upper()}] {rec['action']}")
        print(f"    Expected benefit: {rec['expected_benefit']}")
```

---

## 🔌 Integration Points

### **With CMS (Cortex Manufacturing System)**

```python
from cms.sensory_cortex import SensoryCortex
from cms.cross_session_intelligence import CrossSessionIntelligence

cortex = SensoryCortex()
intelligence = CrossSessionIntelligence()

@cortex.on_sensor_data
def analyze_with_llm(sensor_data):
    """Every sensor reading goes through LLM analysis"""
    
    # Store for cross-session analysis
    intelligence.add_data_point(DataPoint(
        session_id=cortex.current_session_id,
        timestamp=datetime.now(),
        data_type='sensors',
        data=sensor_data,
        machine_id=cortex.machine_id
    ))
    
    # Check for concerning patterns
    if sensor_data['vibration'] > THRESHOLD:
        # Ask LLM: "Have we seen this before?"
        prediction = intelligence.predict_future_event(
            'tool_failure',
            sensor_data
        )
        
        if prediction['prediction_confidence'] > 0.7:
            cortex.emit_signal('high_failure_risk', prediction)
```

---

### **With SolidWorks Bridge**

```python
from cms.solidworks_bridge import SolidWorksBridge

sw_bridge = SolidWorksBridge()
intelligence = CrossSessionIntelligence()

# Analyze manufacturability
analysis = sw_bridge.analyze_part('bracket.SLDPRT')

# Ask LLM for similar parts
answer = intelligence.ask_question(
    f"We've machined parts with {analysis['feature_count']} features before. "
    f"What was the average machining time and cost?"
)

# Get data-driven estimate instead of guessing!
```

---

### **With CIF (AI Inference)**

```python
from cms.cif import CIFCore
from cms.llm_connector import LLMConnector

cif = CIFCore()
llm = LLMConnector()

# CIF runs ML model for prediction
model = cif.load_model('tool_wear.cif')
compiled = cif.compile_model(model)
ml_prediction = compiled(sensor_data)

# LLM explains the prediction
explanation = llm.query_natural_language(
    f"The ML model predicts tool wear of {ml_prediction['wear_mm']:.2f}mm. "
    f"Based on historical data, is this concerning?"
)

print(explanation)
# "This wear rate is 20% higher than normal for this operation.
#  Historical data shows this pattern preceded tool failure in 3/5 cases.
#  Recommend: Monitor closely and prepare replacement tool."
```

---

## 📊 Architecture

```
Manufacturing Data (Sensors, Quality, Operations)
            ↓
LLM Connector (OpenAI/Claude/Local)
            ├─ Embeddings (Vector Search)
            ├─ Relationship Inference
            ├─ Pattern Matching
            └─ Natural Language Interface
            ↓
Vector Database (Cross-Session Memory)
            ├─ Chroma (Local)
            └─ FAISS (Fast similarity)
            ↓
Cross-Session Intelligence
            ├─ Find Related Events
            ├─ Predict Future
            ├─ Connect Dots
            └─ Generate Insights
            ↓
Actionable Intelligence
```

---

## ⚙️ Configuration

### **Choose LLM Provider**

```python
# OpenAI (Best accuracy, cloud-based)
config = LLMConfig(
    provider='openai',
    model='gpt-4-turbo-preview',
    temperature=0.7
)

# Anthropic (Best reasoning, cloud-based)
config = LLMConfig(
    provider='anthropic',
    model='claude-3-opus-20240229',
    temperature=0.7
)

# Local (Privacy-first, no internet required)
config = LLMConfig(
    provider='local',
    local_model_path='models/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
    temperature=0.7
)
```

---

## 🎯 Best Practices

### **1. Always Store Session Data**
```python
# Don't lose data - store everything
intelligence.add_data_point(DataPoint(...))
```

### **2. Use Specific Questions**
```python
# Good
"Why did spindle load increase 15% on Machine #3 yesterday at 2 PM?"

# Bad
"Something wrong?"
```

### **3. Provide Context**
```python
# Better predictions with more context
prediction = intelligence.predict_future_event(
    'tool_failure',
    current_indicators={
        'vibration': 2.5,
        'load': 85,
        'temp': 65,
        'runtime_hours': 4.5,  # More context
        'material': '6061-T6',
        'tool_age_hours': 120
    }
)
```

### **4. Regular Insights Reports**
```python
# Weekly automated analysis
@scheduler.weekly
def weekly_intelligence():
    report = intelligence.generate_insights_report(days=7)
    email_to_team(report)
```

---

## 🚀 Advanced Features

### **Custom Prompts**
```python
# Use custom analysis logic
custom_prompt = """
Analyze this CNC machining data for aerospace standards (AS9100).
Focus on traceability, validation, and quality assurance.
...
"""

llm.query_natural_language(custom_prompt)
```

### **Batch Processing**
```python
# Analyze multiple sessions at once
results = []
for session_id in session_ids:
    result = intelligence.find_related_sessions(session_id)
    results.append(result)
```

---

**The LLM system acts as a time-traveling detective, connecting dots across your entire manufacturing history to surface insights no human would find! 🧠✨**
