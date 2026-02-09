"""
LLM Integration with CMS Modules
Demonstrates how to connect LLM throughout the system
"""

from datetime import datetime, timedelta
from backend.cms.llm_connector import LLMConnector, LLMConfig
from backend.cms.cross_session_intelligence import CrossSessionIntelligence, DataPoint
from backend.cms.sensory_cortex import SensoryCortex
from backend.cms.hippocampus_aggregator import HippocampusAggregator
import json


class LLMEnhancedCortex:
    """
    Sensory Cortex enhanced with LLM intelligence
    
    Every sensor reading is analyzed by LLM for:
    - Historical pattern matching
    - Anomaly detection
    - Predictive alerts
    - Natural language explanations
    """
    
    def __init__(self, llm_config: LLMConfig = None):
        """Initialize enhanced cortex"""
        self.cortex = SensoryCortex()
        self.intelligence = CrossSessionIntelligence(
            llm_connector=LLMConnector(llm_config)
        )
        self.current_session_id = self._generate_session_id()
        
        # Register LLM handlers
        self._setup_llm_handlers()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _setup_llm_handlers(self):
        """Set up LLM-enhanced event handlers"""
        
        @self.cortex.on_sensor_data
        def analyze_sensor_with_llm(sensor_data):
            """Every sensor reading analyzed by LLM"""
            
            # Store in cross-session memory
            self.intelligence.add_data_point(DataPoint(
                session_id=self.current_session_id,
                timestamp=datetime.now(),
                data_type='sensor_reading',
                data=sensor_data,
                machine_id=sensor_data.get('machine_id', 'unknown')
            ))
            
            # Check for concerning patterns
            if self._is_concerning(sensor_data):
                # Ask LLM: Have we seen this pattern before?
                similar = self.intelligence.llm.find_similar_sessions(
                    sensor_data, 
                    top_k=3
                )
                
                if similar:
                    # Predict what happens next
                    prediction = self.intelligence.predict_future_event(
                        event_type='anomaly',
                        current_indicators=sensor_data
                    )
                    
                    if prediction['prediction_confidence'] > 0.7:
                        self._send_llm_alert(sensor_data, prediction)
        
        @self.cortex.on_signal
        def llm_explain_signal(signal_name, signal_data):
            """LLM explains why signal was triggered"""
            
            explanation = self.intelligence.ask_question(
                f"Signal '{signal_name}' was triggered with data: {signal_data}. "
                f"What does this mean and what should we do?"
            )
            
            print(f"📢 Signal: {signal_name}")
            print(f"🧠 LLM Explanation: {explanation}")
    
    def _is_concerning(self, sensor_data: dict) -> bool:
        """Quick check if sensor data is concerning"""
        # Simple heuristics - would be more sophisticated in production
        vibration = sensor_data.get('vibration', {})
        if isinstance(vibration, dict):
            if any(v > 2.0 for v in vibration.values()):
                return True
        
        if sensor_data.get('temperature', 0) > 80:
            return True
        
        if sensor_data.get('spindle_load', 0) > 90:
            return True
        
        return False
    
    def _send_llm_alert(self, sensor_data: dict, prediction: dict):
        """Send intelligent alert with LLM analysis"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': 'high' if prediction['prediction_confidence'] > 0.8 else 'medium',
            'current_data': sensor_data,
            'prediction': prediction,
            'llm_recommendation': prediction.get('llm_analysis', 'No recommendation')
        }
        
        print("🚨 ALERT 🚨")
        print(json.dumps(alert, indent=2))
        
        # Could also send email, SMS, etc.
    
    def ask_about_current_state(self, question: str) -> str:
        """Ask LLM question about current machine state"""
        return self.intelligence.ask_question(question)
    
    def get_daily_insights(self) -> dict:
        """Get LLM-generated insights for the day"""
        return self.intelligence.generate_insights_report(days=1)


class LLMEnhancedHippocampus:
    """
    Hippocampus (memory system) enhanced with LLM
    
    Uses LLM to:
    - Summarize long-term memories
    - Find patterns across time
    - Generate learning insights
    - Optimize memory consolidation
    """
    
    def __init__(self):
        """Initialize enhanced hippocampus"""
        self.hippocampus = HippocampusAggregator()
        self.intelligence = CrossSessionIntelligence()
    
    def consolidate_memories_with_llm(self, days: int = 7) -> dict:
        """
        Consolidate memories using LLM intelligence
        
        LLM analyzes all memories and:
        - Identifies important patterns
        - Discards irrelevant data
        - Summarizes key learnings
        - Suggests process improvements
        """
        
        # Get all recent memories
        cutoff = datetime.now() - timedelta(days=days)
        recent_memories = [
            dp for dp in self.intelligence.data_repository
            if dp.timestamp > cutoff
        ]
        
        if not recent_memories:
            return {'status': 'no_memories', 'days': days}
        
        # Ask LLM to consolidate
        memory_data = [
            {
                'timestamp': dp.timestamp.isoformat(),
                'type': dp.data_type,
                'data': dp.data
            }
            for dp in recent_memories
        ]
        
        consolidation_prompt = f"""
Analyze these {len(memory_data)} manufacturing memories from the last {days} days.

Memories:
{json.dumps(memory_data[:100], indent=2)}  # Limit to avoid token overflow

Tasks:
1. Identify the 5 most important patterns
2. Summarize key learnings
3. Recommend what to remember long-term vs. discard
4. Suggest process improvements based on patterns

Provide structured output:
{{
    "key_patterns": ["pattern1", "pattern2", ...],
    "learnings": ["learning1", "learning2", ...],
    "keep_long_term": ["type1", "type2", ...],
    "can_discard": ["type1", "type2", ...],
    "process_improvements": [
        {{"action": "...", "expected_benefit": "..."}}
    ]
}}
"""
        
        result = self.intelligence.llm.query_natural_language(consolidation_prompt)
        
        # Parse result
        try:
            consolidation = json.loads(result)
        except json.JSONDecodeError:
            consolidation = {
                'raw_analysis': result
            }
        
        consolidation['memory_count'] = len(recent_memories)
        consolidation['time_period_days'] = days
        consolidation['consolidated_at'] = datetime.now().isoformat()
        
        return consolidation
    
    def search_memories_semantic(self, query: str, top_k: int = 10) -> list:
        """
        Semantic search through memories using LLM embeddings
        
        Example:
        query = "times when tool failed unexpectedly"
        Returns: All memories semantically similar to tool failures
        """
        
        results = self.intelligence.llm.vector_store.similarity_search(
            query, 
            k=top_k
        )
        
        return [
            {
                'content': json.loads(result.page_content),
                'metadata': result.metadata
            }
            for result in results
        ]


class LLMManufacturingAssistant:
    """
    AI Assistant for manufacturing operations
    
    Natural language interface to entire system
    """
    
    def __init__(self):
        """Initialize assistant"""
        self.intelligence = CrossSessionIntelligence()
        self.enhanced_cortex = LLMEnhancedCortex()
    
    def chat(self, message: str) -> str:
        """
        Chat interface to manufacturing system
        
        Examples:
        - "What's the current machine status?"
        - "Why did quality drop yesterday?"
        - "Should I change the tool?"
        - "Show me trends from last week"
        """
        
        # Determine intent and route to appropriate handler
        answer = self.intelligence.ask_question(message)
        
        return answer
    
    def explain_alert(self, alert_data: dict) -> str:
        """Explain what an alert means in plain language"""
        
        explanation = self.intelligence.ask_question(
            f"An alert was triggered with this data: {json.dumps(alert_data)}. "
            f"Please explain what this means, why it's important, and what actions to take."
        )
        
        return explanation
    
    def recommend_action(self, situation: str) -> dict:
        """Get LLM recommendation for a situation"""
        
        prompt = f"""
Situation: {situation}

As a manufacturing expert, provide:
1. Analysis of the situation
2. Recommended immediate actions
3. Recommended long-term actions
4. Expected outcomes

Format as JSON:
{{
    "analysis": "...",
    "immediate_actions": ["action1", "action2"],
    "long_term_actions": ["action1", "action2"],
    "expected_outcomes": "..."
}}
"""
        
        result = self.intelligence.llm.query_natural_language(prompt)
        
        try:
            recommendation = json.loads(result)
        except json.JSONDecodeError:
            recommendation = {'raw_response': result}
        
        return recommendation
    
    def compare_sessions(self, session_a: str, session_b: str) -> dict:
        """
        Compare two sessions and find differences/similarities
        
        LLM analyzes both and provides insights
        """
        
        # Find session data
        data_a = [
            dp for dp in self.intelligence.data_repository
            if dp.session_id == session_a
        ]
        
        data_b = [
            dp for dp in self.intelligence.data_repository
            if dp.session_id == session_b
        ]
        
        if not data_a or not data_b:
            return {'error': 'Session not found'}
        
        # Ask LLM to compare
        comparison_prompt = f"""
Compare these two manufacturing sessions:

Session A ({session_a}):
{json.dumps([dp.data for dp in data_a[:10]], indent=2)}

Session B ({session_b}):
{json.dumps([dp.data for dp in data_b[:10]], indent=2)}

Provide:
1. Key differences
2. Key similarities
3. Which performed better and why
4. Recommendations to improve

Format as JSON.
"""
        
        result = self.intelligence.llm.query_natural_language(comparison_prompt)
        
        try:
            comparison = json.loads(result)
        except:
            comparison = {'raw_analysis': result}
        
        return comparison


# Demo functions
def demo_llm_enhanced_cortex():
    """Demonstrate LLM-enhanced sensory cortex"""
    
    print("=" * 70)
    print("Demo: LLM-Enhanced Sensory Cortex")
    print("=" * 70)
    
    # Initialize
    enhanced = LLMEnhancedCortex()
    
    # Simulate sensor data
    print("\n📊 Simulating sensor data...")
    
    sensor_data = {
        'machine_id': 'CNC_001',
        'vibration': {'x': 2.5, 'y': 2.8, 'z': 2.3},
        'temperature': 65,
        'spindle_load': 85,
        'timestamp': datetime.now().isoformat()
    }
    
    # This triggers LLM analysis automatically
    enhanced.cortex.emit_sensor_data(sensor_data)
    
    # Ask questions
    print("\n❓ Asking LLM questions...")
    
    answer = enhanced.ask_about_current_state(
        "Based on current vibration levels, should we stop the machine?"
    )
    print(f"\n🧠 LLM Answer: {answer}")


def demo_cross_session_intelligence():
    """Demonstrate cross-session intelligence"""
    
    print("\n" + "=" * 70)
    print("Demo: Cross-Session Intelligence")
    print("=" * 70)
    
    intelligence = CrossSessionIntelligence()
    
    # Add historical data
    print("\n📚 Adding historical data...")
    
    # Session 1: Normal operation (2 days ago)
    intelligence.add_data_point(DataPoint(
        session_id='session_001',
        timestamp=datetime.now() - timedelta(days=2),
        data_type='operation',
        data={'vibration': 0.5, 'quality': 'good', 'cycle_time': 120},
        machine_id='CNC_001'
    ))
    
    # Session 2: High vibration then failure (1 day ago)
    intelligence.add_data_point(DataPoint(
        session_id='session_002',
        timestamp=datetime.now() - timedelta(days=1, hours=2),
        data_type='operation',
        data={'vibration': 2.5, 'quality': 'declining', 'cycle_time': 130},
        machine_id='CNC_001'
    ))
    
    intelligence.add_data_point(DataPoint(
        session_id='session_002',
        timestamp=datetime.now() - timedelta(days=1),
        data_type='failure',
        data={'type': 'tool_failure', 'cause': 'excessive_wear'},
        machine_id='CNC_001'
    ))
    
    # Current session: Similar vibration pattern
    intelligence.add_data_point(DataPoint(
        session_id='session_current',
        timestamp=datetime.now(),
        data_type='operation',
        data={'vibration': 2.6, 'quality': 'good', 'cycle_time': 125},
        machine_id='CNC_001'
    ))
    
    # Predict
    print("\n🔮 Predicting future event...")
    
    prediction = intelligence.predict_future_event(
        event_type='tool_failure',
        current_indicators={'vibration': 2.6, 'cycle_time': 125}
    )
    
    print(f"\n📊 Prediction Results:")
    print(f"  Confidence: {prediction['prediction_confidence']*100:.0f}%")
    print(f"  Time estimate: {prediction.get('estimated_time_minutes', 'N/A')} minutes")
    print(f"  Based on {prediction['historical_cases']} similar cases")
    
    # Generate insights
    print("\n💡 Generating insights report...")
    
    report = intelligence.generate_insights_report(days=7)
    print(f"\n📈 Insights:")
    for insight in report.get('key_insights', [])[:3]:
        print(f"  • {insight}")


def demo_manufacturing_assistant():
    """Demonstrate AI assistant"""
    
    print("\n" + "=" * 70)
    print("Demo: AI Manufacturing Assistant")
    print("=" * 70)
    
    assistant = LLMManufacturingAssistant()
    
    # Example conversations
    questions = [
        "What's the current machine status?",
        "Should I change the cutting tool?",
        "Why might vibration be increasing?",
    ]
    
    print("\n💬 Having conversation with AI assistant...\n")
    
    for question in questions:
        print(f"👤 Human: {question}")
        answer = assistant.chat(question)
        print(f"🤖 Assistant: {answer}\n")


if __name__ == "__main__":
    """Run all demos"""
    
    print("\n" + "🚀 " * 35)
    print("LLM Integration Demo Suite")
    print("🚀 " * 35)
    
    # Run demos
    try:
        demo_llm_enhanced_cortex()
    except Exception as e:
        print(f"Error in cortex demo: {e}")
    
    try:
        demo_cross_session_intelligence()
    except Exception as e:
        print(f"Error in intelligence demo: {e}")
    
    try:
        demo_manufacturing_assistant()
    except Exception as e:
        print(f"Error in assistant demo: {e}")
    
    print("\n" + "✅ " * 35)
    print("Demo Complete!")
    print("✅ " * 35 + "\n")
