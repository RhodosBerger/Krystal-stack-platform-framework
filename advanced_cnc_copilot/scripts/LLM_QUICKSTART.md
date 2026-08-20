# 🚀 Quick Start Scripts for LLM Integration

This directory contains scripts to get started with LLM integration quickly.

---

## 📋 Prerequisites

```bash
# 1. Create conda environment
conda env create -f ../environment.yml

# 2. Activate environment
conda activate cnc-copilot-llm

# 3. Set API key
echo "OPENAI_API_KEY=your-key-here" > ../.env
# OR for local LLM (no API key needed)
# Download model to models/ directory
```

---

## 🎯 Quick Scripts

### **1. Test LLM Connection**

```bash
python test_llm.py
```

**What it does:**
- Tests connection to LLM
- Verifies API key
- Shows available models

### **2. Run Integration Demo**

```bash
python -m cms.llm_integration_examples
```

**What it does:**
- Demonstrates LLM-enhanced cortex
- Shows cross-session intelligence
- Runs AI assistant demo

### **3. Start API Server**

```bash
python -m cms.llm_api_server
```

**Access at:**
- API: http://localhost:8001
- Docs: http://localhost:8001/docs

**Test with curl:**
```bash
# Ask a question
curl -X POST http://localhost:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Why did quality drop?"}'

# Get insights
curl http://localhost:8001/api/insights?days=7

# Predict event
curl -X POST http://localhost:8001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "tool_failure",
    "current_indicators": {"vibration": 2.5, "load": 85}
  }'
```

---

## 📝 Example Scripts

### **test_llm.py**

```python
"""Test LLM connection"""
from cms.llm_connector import LLMConnector

print("Testing LLM connection...")

try:
    llm = LLMConnector()
    
    response = llm.query_natural_language(
        "Say hello and confirm you can help with manufacturing."
    )
    
    print("✅ LLM Connected!")
    print(f"Response: {response}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure:")
    print("1. API key is set in .env")
    print("2. Conda environment is activated")
    print("3. Internet connection is available (for cloud LLMs)")
```

### **quick_demo.py**

```python
"""Quick demonstration of cross-session intelligence"""
from cms.cross_session_intelligence import CrossSessionIntelligence, DataPoint
from datetime import datetime, timedelta

# Initialize
intelligence = CrossSessionIntelligence()

# Add some data
print("Adding historical data...")

# Past session with high vibration → failure
intelligence.add_data_point(DataPoint(
    session_id='past_session',
    timestamp=datetime.now() - timedelta(days=1),
    data_type='vibration',
    data={'x': 2.5, 'y': 2.8},
    machine_id='CNC_001'
))

intelligence.add_data_point(DataPoint(
    session_id='past_session',
    timestamp=datetime.now() - timedelta(hours=23),
    data_type='failure',
    data={'type': 'tool_failure'},
    machine_id='CNC_001'
))

# Current session with similar vibration
intelligence.add_data_point(DataPoint(
    session_id='current',
    timestamp=datetime.now(),
    data_type='vibration',
    data={'x': 2.6, 'y': 2.9},
    machine_id='CNC_001'
))

# Predict
print("\nPredicting future event...")
prediction = intelligence.predict_future_event(
    'tool_failure',
    {'vibration': 2.6}
)

print(f"\n🔮 Prediction:")
print(f"  Confidence: {prediction['prediction_confidence']*100:.0f}%")
print(f"  Time: ~{prediction.get('estimated_time_minutes', 'unknown')} min")

# Ask question
print("\n💬 Asking LLM...")
answer = intelligence.ask_question(
    "Should I stop the machine based on current vibration?"
)

print(f"\n🧠 Answer: {answer}")
```

### **batch_analyze.py**

```python
"""Batch analyze multiple sessions"""
from cms.cross_session_intelligence import CrossSessionIntelligence
import json

intelligence = CrossSessionIntelligence()

# Load sessions from files or database
# (Simulated here)
sessions = [
    'session_001',
    'session_002', 
    'session_003'
]

print("Analyzing sessions...")

for session_id in sessions:
    print(f"\n📊 {session_id}:")
    
    # Find related sessions
    related = intelligence.find_related_sessions(session_id)
    
    print(f"  Related sessions: {related['related_sessions_found']}")
    
    for rel in related.get('relationships', []):
        print(f"    • {rel['session_id']}: {rel['relationship']['relationship_type']}")
        print(f"      Confidence: {rel['relationship']['confidence']:.2f}")
```

---

## 🔧 Configuration Options

### **Use Local LLM (No API Key Required)**

```python
from cms.llm_connector import LLMConnector, LLMConfig

# Use local LLaMA model
config = LLMConfig(
    provider='local',
    local_model_path='models/llama-2-7b-chat.gguf'
)

llm = LLMConnector(config)
```

### **Use Anthropic Claude**

```python
config = LLMConfig(
    provider='anthropic',
    model='claude-3-opus-20240229'
)

llm = LLMConnector(config)
```

### **Adjust Temperature**

```python
config = LLMConfig(
    provider='openai',
    model='gpt-4',
    temperature=0.2  # More deterministic (0.0-1.0)
)
```

---

## 🐛 Troubleshooting

### **"No module named 'langchain'"**
```bash
conda activate cnc-copilot-llm
pip install langchain openai anthropic
```

### **"API key not found"**
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-..." > .env

# Or set environment variable
export OPENAI_API_KEY=sk-...
```

### **"Connection refused"**
Check if API server is running:
```bash
python -m cms.llm_api_server
```

### **Slow responses**
- Use local LLM for faster responses (no network)
- Reduce context window
- Use smaller model (gpt-3.5-turbo instead of gpt-4)

---

## 📚 Next Steps

1. **Explore API**: http://localhost:8001/docs
2. **Read guide**: `../LLM_INTEGRATION_GUIDE.md`
3. **Customize prompts**: Edit `cms/llm_connector.py`
4. **Add your data**: Use API endpoints to add manufacturing data
5. **Build dashboard**: Use API to create web interface

---

## 💡 Tips

- **Start small**: Test with small dataset first
- **Use local LLM**: For privacy-sensitive data
- **Cache responses**: LLM calls are expensive
- **Batch operations**: Process multiple items together
- **Monitor costs**: Track API usage for cloud LLMs

---

**Happy manufacturing! 🏭🤖**
