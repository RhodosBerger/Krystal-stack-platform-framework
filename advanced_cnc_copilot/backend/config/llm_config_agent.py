"""
LLM Config Agent 🤖
Responsibility:
1. Parse natural language configuration queries.
2. Translate intent to parameter actions.
3. Provide intelligent configuration suggestions.
"""
from typing import Dict, Any, List, Optional
from backend.config.parameter_registry import parameter_registry, Parameter, ParamType

class LLMConfigAgent:
    def __init__(self):
        self.last_query = None
        self.conversation_context = []

    def query(self, natural_language: str) -> Dict[str, Any]:
        """
        Processes a natural language query about configuration.
        Returns structured response with interpretation and action.
        """
        self.last_query = natural_language
        nl_lower = natural_language.lower()
        
        # Intent Detection
        intent = self._detect_intent(nl_lower)
        
        if intent == "list_all":
            return self._handle_list_all()
        elif intent == "search":
            return self._handle_search(nl_lower)
        elif intent == "get_value":
            return self._handle_get_value(nl_lower)
        elif intent == "set_value":
            return self._handle_set_value(natural_language)
        elif intent == "explain":
            return self._handle_explain(nl_lower)
        elif intent == "suggest":
            return self._handle_suggest(nl_lower)
        else:
            return self._handle_unknown(natural_language)

    def _detect_intent(self, query: str) -> str:
        """Detects the user's intent from the query."""
        if any(w in query for w in ["list all", "show all", "všetky parametre", "all parameters"]):
            return "list_all"
        elif any(w in query for w in ["search", "find", "hľadaj", "nájdi"]):
            return "search"
        elif any(w in query for w in ["what is", "aké je", "current value", "hodnota"]):
            return "get_value"
        elif any(w in query for w in ["set", "change", "nastav", "zmeň", "update"]):
            return "set_value"
        elif any(w in query for w in ["explain", "vysvetli", "what does", "čo robí"]):
            return "explain"
        elif any(w in query for w in ["suggest", "recommend", "odporuč", "navrhni"]):
            return "suggest"
        else:
            return "unknown"

    def _handle_list_all(self) -> Dict[str, Any]:
        params = parameter_registry.list_all()
        categories = parameter_registry.get_categories()
        return {
            "intent": "list_all",
            "response": f"Found {len(params)} parameters across {len(categories)} categories.",
            "categories": categories,
            "params": params
        }

    def _handle_search(self, query: str) -> Dict[str, Any]:
        # Extract search term
        for prefix in ["search for", "find", "hľadaj", "nájdi"]:
            if prefix in query:
                term = query.split(prefix)[-1].strip()
                break
        else:
            term = query.split()[-1]  # Last word as fallback
        
        results = parameter_registry.search(term)
        return {
            "intent": "search",
            "search_term": term,
            "response": f"Found {len(results)} parameters matching '{term}'.",
            "results": results
        }

    def _handle_get_value(self, query: str) -> Dict[str, Any]:
        # Try to find parameter key in query
        for param in parameter_registry._params.values():
            if param.key.lower() in query:
                return {
                    "intent": "get_value",
                    "param_key": param.key,
                    "response": f"The value of '{param.key}' is: {param.value}" + (f" {param.unit}" if param.unit else ""),
                    "param": param.to_dict()
                }
        return {
            "intent": "get_value",
            "response": "Could not identify which parameter you're asking about. Please be more specific.",
            "suggestion": "Try: 'What is the max_spindle_rpm?'"
        }

    def _handle_set_value(self, query: str) -> Dict[str, Any]:
        # Parse "set X to Y" pattern
        import re
        patterns = [
            r"set\s+(\w+)\s+to\s+(.+)",
            r"change\s+(\w+)\s+to\s+(.+)",
            r"nastav\s+(\w+)\s+na\s+(.+)",
            r"update\s+(\w+)\s+to\s+(.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                
                # Try to find the parameter
                param = parameter_registry.get(key)
                if not param:
                    # Fuzzy match
                    for p in parameter_registry._params.values():
                        if key in p.key.lower():
                            param = p
                            break
                
                if param:
                    result = param.set_value(value)
                    if result["success"]:
                        return {
                            "intent": "set_value",
                            "param_key": param.key,
                            "response": f"Successfully updated '{param.key}' from {result['old_value']} to {result['new_value']}.",
                            "success": True
                        }
                    else:
                        return {
                            "intent": "set_value",
                            "param_key": param.key,
                            "response": f"Failed to update: {result.get('errors', result.get('error'))}",
                            "success": False
                        }
        
        return {
            "intent": "set_value",
            "response": "Could not parse the set command. Try: 'Set max_spindle_rpm to 10000'",
            "success": False
        }

    def _handle_explain(self, query: str) -> Dict[str, Any]:
        for param in parameter_registry._params.values():
            if param.key.lower() in query:
                return {
                    "intent": "explain",
                    "param_key": param.key,
                    "response": param.to_llm_summary(),
                    "param": param.to_dict()
                }
        return {
            "intent": "explain",
            "response": "Please specify which parameter you want explained."
        }

    def _handle_suggest(self, query: str) -> Dict[str, Any]:
        # Provide intelligent suggestions based on context
        suggestions = []
        
        if "machining" in query or "obrábanie" in query:
            suggestions = [
                "Consider max_spindle_rpm based on your material hardness.",
                "Adjust default_feed_rate for optimal surface finish.",
                "Use G90 (absolute) for most operations."
            ]
        elif "safety" in query or "bezpečnosť" in query:
            suggestions = [
                "Keep safety_mode enabled for production environments.",
                "Set coordinate_system to G90 to avoid positioning errors."
            ]
        else:
            suggestions = [
                "Review all parameters with 'list all parameters'.",
                "Search for specific areas with 'search machining'.",
                "Get help with 'explain max_spindle_rpm'."
            ]
        
        return {
            "intent": "suggest",
            "response": "Here are some suggestions:",
            "suggestions": suggestions
        }

    def _handle_unknown(self, query: str) -> Dict[str, Any]:
        return {
            "intent": "unknown",
            "response": "I'm not sure what you're asking. Here's what I can help with:",
            "capabilities": [
                "List all parameters: 'Show all parameters'",
                "Search: 'Search for spindle'",
                "Get value: 'What is max_spindle_rpm?'",
                "Set value: 'Set max_spindle_rpm to 10000'",
                "Explain: 'Explain default_feed_rate'",
                "Suggest: 'Suggest settings for machining'"
            ]
        }

    def get_full_documentation(self) -> str:
        """Returns complete LLM-readable documentation of all parameters."""
        return parameter_registry.get_llm_summary()


# Global Instance
llm_config_agent = LLMConfigAgent()
