"""
Cross-Session Intelligence Engine
Uses LLM to connect unrelated data and infer practical relationships

This is the "brain" that connects dots across time
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from backend.cms.llm_connector import LLMConnector, LLMConfig


@dataclass
class DataPoint:
    """
    Single data point from any logging session
    """
    session_id: str
    timestamp: datetime
    data_type: str  # 'vibration', 'tool_wear', 'quality', 'temperature', etc.
    data: Dict[str, Any]
    machine_id: Optional[str] = None
    part_id: Optional[str] = None
    operator_id: Optional[str] = None


class CrossSessionIntelligence:
    """
    Connects data across different logging sessions
    
    Paradigm: Time-Traveling Detective
    - Looks back through all sessions
    - Finds patterns humans would miss
    - Connects seemingly unrelated events
    - Predicts future based on past patterns
    """
    
    def __init__(self, llm_connector: Optional[LLMConnector] = None):
        """
        Initialize cross-session intelligence
        
        Args:
            llm_connector: LLM connector (creates default if None)
        """
        self.llm = llm_connector or LLMConnector()
        self.data_repository: List[DataPoint] = []
    
    def add_data_point(self, data_point: DataPoint):
        """Add data point to repository"""
        self.data_repository.append(data_point)
        
        # Store in vector DB for semantic search
        self.llm.store_session_data(
            session_data={
                'type': data_point.data_type,
                'data': data_point.data,
                'session_id': data_point.session_id
            },
            metadata={
                'timestamp': data_point.timestamp.isoformat(),
                'machine_id': data_point.machine_id,
                'part_id': data_point.part_id,
                'data_type': data_point.data_type
            }
        )
    
    def find_related_sessions(self, 
                             current_session_id: str,
                             lookback_days: int = 90) -> Dict[str, Any]:
        """
        Find related sessions from history
        
        Example:
        Current session: High vibration detected
        Finds: 3 sessions from last 90 days with similar vibration
        Infers: All had tool failure within 2 hours
        Action: Change tool now!
        
        Args:
            current_session_id: Current session to analyze
            lookback_days: How far back to look
        
        Returns:
            Related sessions with inferred relationships
        """
        # Get current session data
        current_data = [
            dp for dp in self.data_repository 
            if dp.session_id == current_session_id
        ]
        
        if not current_data:
            return {'error': 'Session not found'}
        
        # Get historical data
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        historical_data = [
            dp for dp in self.data_repository
            if dp.timestamp > cutoff_date and dp.session_id != current_session_id
        ]
        
        # Find similar patterns
        current_summary = self._summarize_session(current_data)
        similar_sessions = self.llm.find_similar_sessions(current_summary, top_k=10)
        
        # Analyze relationships
        relationships = []
        for similar in similar_sessions:
            # Get full session data
            similar_session_id = similar['metadata'].get('session_id')
            similar_data = [
                dp for dp in historical_data
                if dp.session_id == similar_session_id
            ]
            
            if similar_data:
                # Infer relationship
                relationship = self.llm.infer_relationship(
                    data_a=current_summary,
                    data_b=self._summarize_session(similar_data),
                    context=f"Similarity score: {similar['similarity_score']:.3f}"
                )
                
                if relationship.get('has_relationship'):
                    relationships.append({
                        'session_id': similar_session_id,
                        'similarity_score': similar['similarity_score'],
                        'relationship': relationship,
                        'timestamp': similar_data[0].timestamp.isoformat()
                    })
        
        return {
            'current_session': current_session_id,
            'related_sessions_found': len(relationships),
            'relationships': relationships,
            'analysis_date': datetime.now().isoformat()
        }
    
    def connect_unrelated_events(self, 
                                 time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Find connections between events that happened in same time window
        
        Example scenario:
        - Session A: Part quality dropped (8:00 AM)
        - Session B: HVAC temperature fluctuation (7:45 AM)
        - Session C: Different machine vibration spike (8:05 AM)
        
        LLM infers: Temperature change → floor expansion → machine vibration → quality drop
        
        Args:
            time_window_hours: Time window to analyze
        
        Returns:
            List of discovered connections
        """
        # Group events by time windows
        windows = self._create_time_windows(time_window_hours)
        
        connections = []
        
        for window_start, window_data in windows.items():
            if len(window_data) < 2:
                continue  # Need at least 2 events to connect
            
            # Prepare data for LLM
            data_points = [
                {
                    'type': dp.data_type,
                    'timestamp': dp.timestamp.isoformat(),
                    'data': dp.data,
                    'machine': dp.machine_id,
                    'session': dp.session_id
                }
                for dp in window_data
            ]
            
            # Ask LLM to find connections
            result = self.llm.connect_unrelated_data(
                data_points=data_points,
                purpose='identify_root_causes_and_predictions'
            )
            
            if result.get('connections_found'):
                connections.append({
                    'time_window_start': window_start.isoformat(),
                    'events_count': len(window_data),
                    'connections': result['connections_found'],
                    'insights': result.get('insights', []),
                    'recommendations': result.get('recommendations', [])
                })
        
        return connections
    
    def predict_future_event(self, 
                            event_type: str,
                            current_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict future event based on current indicators and historical patterns
        
        Example:
        Event type: "tool_failure"
        Current indicators: {vibration: 2.5, load: 85%, temp: 65C}
        
        LLM finds: 5 past sessions with similar indicators
        All 5 had tool failure within 30-45 minutes
        Prediction: 80% chance of failure in next 40 minutes
        
        Args:
            event_type: What to predict ('tool_failure', 'quality_drop', etc.)
            current_indicators: Current sensor/process data
        
        Returns:
            Prediction with confidence and time estimate
        """
        # Find similar historical patterns
        similar_sessions = self.llm.find_similar_sessions(current_indicators, top_k=10)
        
        # Check what happened after these patterns
        outcomes = []
        for session in similar_sessions:
            session_id = session['metadata'].get('session_id')
            session_time = datetime.fromisoformat(session['metadata']['timestamp'])
            
            # Look for events after this session
            followup_events = [
                dp for dp in self.data_repository
                if dp.session_id != session_id
                and dp.timestamp > session_time
                and dp.timestamp < session_time + timedelta(hours=2)
                and event_type in dp.data_type
            ]
            
            if followup_events:
                time_to_event = (followup_events[0].timestamp - session_time).total_seconds() / 60
                outcomes.append({
                    'event_occurred': True,
                    'time_to_event_minutes': time_to_event,
                    'similarity_score': session['similarity_score']
                })
            else:
                outcomes.append({
                    'event_occurred': False,
                    'similarity_score': session['similarity_score']
                })
        
        # Calculate prediction
        if not outcomes:
            return {
                'prediction': 'insufficient_data',
                'confidence': 0.0
            }
        
        events_occurred = sum(1 for o in outcomes if o.get('event_occurred', False))
        confidence = events_occurred / len(outcomes)
        
        if events_occurred > 0:
            avg_time = np.mean([
                o['time_to_event_minutes'] 
                for o in outcomes 
                if o.get('event_occurred', False)
            ])
        else:
            avg_time = None
        
        # Get LLM analysis
        analysis_prompt = f"""
Based on historical data:
- Found {len(similar_sessions)} similar situations
- Event occurred in {events_occurred}/{len(outcomes)} cases ({confidence*100:.1f}%)
- Average time to event: {avg_time:.1f} minutes (when it occurred)

Current indicators:
{json.dumps(current_indicators, indent=2)}

Provide prediction analysis:
{{
    "likelihood": "high/medium/low",
    "reasoning": "explanation",
    "recommended_action": "specific action",
    "urgency": "immediate/soon/monitor"
}}
"""
        
        llm_analysis = self.llm.query_natural_language(analysis_prompt)
        
        return {
            'event_type': event_type,
            'prediction_confidence': confidence,
            'estimated_time_minutes': avg_time,
            'historical_cases': len(outcomes),
            'cases_with_event': events_occurred,
            'llm_analysis': llm_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def ask_question(self, question: str) -> str:
        """
        Ask natural language question about manufacturing data
        
        Example questions:
        - "Why did quality drop yesterday?"
        - "Which machine has best performance?"
        - "What causes tool wear to increase?"
        - "Should I change the tool now?"
        
        Args:
            question: Natural language question
        
        Returns:
            Natural language answer based on all available data
        """
        return self.llm.query_natural_language(question)
    
    def generate_insights_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive insights from recent data
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Report with insights, patterns, and recommendations
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_data = [
            dp for dp in self.data_repository
            if dp.timestamp > cutoff
        ]
        
        if not recent_data:
            return {'error': 'No recent data'}
        
        # Group by data type
        by_type = {}
        for dp in recent_data:
            if dp.data_type not in by_type:
                by_type[dp.data_type] = []
            by_type[dp.data_type].append(dp)
        
        # Prepare all data for LLM
        all_data = [
            {
                'type': dp.data_type,
                'timestamp': dp.timestamp.isoformat(),
                'data': dp.data,
                'machine': dp.machine_id
            }
            for dp in recent_data
        ]
        
        # Get comprehensive analysis
        insights = self.llm.connect_unrelated_data(
            data_points=all_data,
            purpose='comprehensive_manufacturing_optimization'
        )
        
        return {
            'period': f'Last {days} days',
            'data_points_analyzed': len(recent_data),
            'data_types': list(by_type.keys()),
            'connections_found': insights.get('connections_found', []),
            'key_insights': insights.get('insights', []),
            'recommendations': insights.get('recommendations', []),
            'generated_at': datetime.now().isoformat()
        }
    
    def _summarize_session(self, data_points: List[DataPoint]) -> Dict[str, Any]:
        """Create summary of session data"""
        if not data_points:
            return {}
        
        summary = {
            'session_id': data_points[0].session_id,
            'timestamp': data_points[0].timestamp.isoformat(),
            'duration_minutes': (data_points[-1].timestamp - data_points[0].timestamp).total_seconds() / 60,
            'data_types': list(set(dp.data_type for dp in data_points)),
            'machine_id': data_points[0].machine_id,
            'data_summary': {}
        }
        
        # Aggregate data by type
        for data_type in summary['data_types']:
            type_data = [dp.data for dp in data_points if dp.data_type == data_type]
            summary['data_summary'][data_type] = {
                'count': len(type_data),
                'sample': type_data[0] if type_data else {}
            }
        
        return summary
    
    def _create_time_windows(self, window_hours: int) -> Dict[datetime, List[DataPoint]]:
        """Group data into time windows"""
        windows = {}
        
        for dp in self.data_repository:
            # Round timestamp to window start
            window_start = dp.timestamp.replace(
                minute=0,
                second=0,
                microsecond=0
            )
            window_start = window_start - timedelta(
                hours=(window_start.hour % window_hours)
            )
            
            if window_start not in windows:
                windows[window_start] = []
            windows[window_start].append(dp)
        
        return windows


# Example usage functions
def demo_cross_session_intelligence():
    """Demonstrate cross-session intelligence"""
    
    # Initialize
    intelligence = CrossSessionIntelligence()
    
    # Simulate adding data from different sessions
    # Session 1: Normal operation
    intelligence.add_data_point(DataPoint(
        session_id='session_001',
        timestamp=datetime.now() - timedelta(hours=5),
        data_type='vibration',
        data={'x': 0.5, 'y': 0.6, 'z': 0.4},
        machine_id='CNC_001'
    ))
    
    # Session 2: High vibration
    intelligence.add_data_point(DataPoint(
        session_id='session_002',
        timestamp=datetime.now() - timedelta(hours=3),
        data_type='vibration',
        data={'x': 2.5, 'y': 2.8, 'z': 2.3},
        machine_id='CNC_001'
    ))
    
    # Session 3: Tool failure (unrelated session)
    intelligence.add_data_point(DataPoint(
        session_id='session_003',
        timestamp=datetime.now() - timedelta(hours=2, minutes=45),
        data_type='tool_failure',
        data={'tool_id': 'T01', 'reason': 'excessive_wear'},
        machine_id='CNC_001'
    ))
    
    # Current session: Similar vibration pattern
    intelligence.add_data_point(DataPoint(
        session_id='session_current',
        timestamp=datetime.now(),
        data_type='vibration',
        data={'x': 2.6, 'y': 2.9, 'z': 2.4},
        machine_id='CNC_001'
    ))
    
    # Predict future
    prediction = intelligence.predict_future_event(
        event_type='tool_failure',
        current_indicators={'vibration_x': 2.6, 'vibration_y': 2.9}
    )
    
    print("Prediction:", json.dumps(prediction, indent=2))
    
    # Ask question
    answer = intelligence.ask_question("Should I change the tool now?")
    print("\nAnswer:", answer)
    
    return intelligence
