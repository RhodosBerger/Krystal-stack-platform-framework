"""
LLM Connector for CNC Copilot (STUBBED FOR COMPATIBILITY)
Legacy module replaced by backend.core.llm_brain
"""

import os
from typing import List, Dict, Any, Optional

class LLMConfig:
    pass

class LLMConnector:
    """
    Stubbed Universal LLM Connector
    """
    def __init__(self, config: Optional[LLMConfig] = None):
        print("WARNING: Legacy LLMConnector initialized in STUB mode.")
    
    def infer_relationship(self, data_a, data_b, context=None):
        return {
            'has_relationship': False, 
            'explanation': "Legacy AI module disabled in favor of Core Brain.",
            'confidence': 0.0
        }
    
    def find_similar_sessions(self, current_data, top_k=5):
        return []

    def connect_unrelated_data(self, data_points, purpose='optimization'):
        return {'insights': ["System optimized using new Core Brain engine."]}

    def query_natural_language(self, query):
        return "Legacy system replaced. Please use the new Intelligence Dashboard."

def create_llm_connector(provider='openai', model='gpt-4'):
    return LLMConnector()

def infer_relationship_simple(data_a, data_b):
    return False
