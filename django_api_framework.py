#!/usr/bin/env python3
"""
Django API Framework for Evolutionary Computing System

This module provides Django-based API integration for the evolutionary computing framework.
"""

from django.urls import path
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
import uuid
from typing import Dict, Any, List
import time
import logging
import threading
from functools import wraps


logger = logging.getLogger(__name__)


class ProfileAPIView(View):
    """API view for profile configuration."""
    
    def get(self, request):
        """Get all profiles."""
        # In a real implementation, this would fetch from database
        profiles = [
            {
                'id': 'PROFILE_BALANCED_1A2B3C4D',
                'name': 'Balanced Profile',
                'description': 'Balanced performance and power consumption',
                'parameters': {
                    'cpu_priority': 'normal',
                    'memory_allocation': 'balanced',
                    'gpu_enabled': True,
                    'power_limit': 100.0
                },
                'created_at': time.time(),
                'is_active': True
            },
            {
                'id': 'PROFILE_GAMING_5E6F7G8H',
                'name': 'Gaming Profile',
                'description': 'High performance for gaming applications',
                'parameters': {
                    'cpu_priority': 'high',
                    'memory_allocation': 'aggressive',
                    'gpu_enabled': True,
                    'power_limit': 120.0
                },
                'created_at': time.time(),
                'is_active': False
            }
        ]
        return JsonResponse({'profiles': profiles})
    
    def post(self, request):
        """Create a new profile."""
        try:
            data = json.loads(request.body)
            profile_id = f"PROFILE_{data.get('name', 'CUSTOM').upper()}_{uuid.uuid4().hex[:8].upper()}"
            
            new_profile = {
                'id': profile_id,
                'name': data.get('name', 'Custom Profile'),
                'description': data.get('description', 'Custom profile'),
                'parameters': data.get('parameters', {}),
                'created_at': time.time(),
                'is_active': False
            }
            
            logger.info(f"Created new profile: {new_profile['name']}")
            return JsonResponse({'profile': new_profile, 'status': 'created'})
        except Exception as e:
            logger.error(f"Error creating profile: {e}")
            return JsonResponse({'error': str(e)}, status=400)


class ConditionalLogicAPIView(View):
    """API view for conditional logic operations."""
    
    def post(self, request):
        """Process conditional logic."""
        try:
            data = json.loads(request.body)
            
            # Process conditional logic
            conditions = data.get('conditions', [])
            result = self._evaluate_conditions(conditions)
            
            response = {
                'result': result,
                'conditions_evaluated': len(conditions),
                'timestamp': time.time()
            }
            
            return JsonResponse(response)
        except Exception as e:
            logger.error(f"Error processing conditional logic: {e}")
            return JsonResponse({'error': str(e)}, status=400)
    
    def _evaluate_conditions(self, conditions: List[Dict[str, Any]]) -> bool:
        """Evaluate a list of conditions."""
        # Simple boolean evaluation
        results = []
        for condition in conditions:
            if 'operator' in condition:
                operator = condition['operator']
                left = condition.get('left', 0)
                right = condition.get('right', 0)
                
                if operator == '==':
                    results.append(left == right)
                elif operator == '!=':
                    results.append(left != right)
                elif operator == '>':
                    results.append(left > right)
                elif operator == '<':
                    results.append(left < right)
                elif operator == '>=':
                    results.append(left >= right)
                elif operator == '<=':
                    results.append(left <= right)
                elif operator == 'and':
                    results.append(all(results))
                elif operator == 'or':
                    results.append(any(results))
        
        return all(results) if results else True


class TransformerFunctionAPIView(View):
    """API view for transformer functions."""
    
    def post(self, request):
        """Apply transformer functions to data."""
        try:
            data = json.loads(request.body)
            
            input_data = data.get('input_data', [])
            transformations = data.get('transformations', [])
            
            # Apply transformations
            transformed_data = input_data
            applied_transformations = []
            
            for transform in transformations:
                func_name = transform.get('function', 'identity')
                params = transform.get('parameters', {})
                
                if func_name == 'normalize':
                    transformed_data = self._normalize_data(transformed_data, params)
                    applied_transformations.append({'function': func_name, 'status': 'applied'})
                elif func_name == 'scale':
                    transformed_data = self._scale_data(transformed_data, params)
                    applied_transformations.append({'function': func_name, 'status': 'applied'})
                elif func_name == 'filter':
                    transformed_data = self._filter_data(transformed_data, params)
                    applied_transformations.append({'function': func_name, 'status': 'applied'})
                else:
                    applied_transformations.append({'function': func_name, 'status': 'unsupported'})
            
            response = {
                'original_data': input_data,
                'transformed_data': transformed_data,
                'applied_transformations': applied_transformations,
                'timestamp': time.time()
            }
            
            return JsonResponse(response)
        except Exception as e:
            logger.error(f"Error applying transformations: {e}")
            return JsonResponse({'error': str(e)}, status=400)
    
    def _normalize_data(self, data: List[float], params: Dict[str, Any]) -> List[float]:
        """Normalize data to [0, 1] range."""
        if not data:
            return data
        
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val
        
        if range_val == 0:
            return [0.0] * len(data)
        
        return [(x - min_val) / range_val for x in data]
    
    def _scale_data(self, data: List[float], params: Dict[str, Any]) -> List[float]:
        """Scale data by a factor."""
        factor = params.get('factor', 1.0)
        return [x * factor for x in data]
    
    def _filter_data(self, data: List[float], params: Dict[str, Any]) -> List[float]:
        """Filter data based on criteria."""
        min_val = params.get('min', float('-inf'))
        max_val = params.get('max', float('inf'))
        
        return [x for x in data if min_val <= x <= max_val]


class OptimizationAPIView(View):
    """API view for optimization operations."""
    
    def post(self, request):
        """Run optimization."""
        try:
            data = json.loads(request.body)
            
            optimization_type = data.get('type', 'genetic')
            parameters = data.get('parameters', {})
            
            if optimization_type == 'genetic':
                result = self._run_genetic_optimization(parameters)
            elif optimization_type == 'differential':
                result = self._run_differential_optimization(parameters)
            elif optimization_type == 'particle_swarm':
                result = self._run_particle_swarm_optimization(parameters)
            else:
                result = {'error': f'Unknown optimization type: {optimization_type}'}
            
            result['timestamp'] = time.time()
            result['optimization_type'] = optimization_type
            
            return JsonResponse(result)
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
            return JsonResponse({'error': str(e)}, status=400)
    
    def _run_genetic_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run genetic algorithm optimization."""
        # Simulate genetic algorithm optimization
        generations = parameters.get('generations', 100)
        population_size = parameters.get('population_size', 50)
        
        # Simulate optimization process
        best_fitness = 0.0
        for gen in range(generations):
            # Simulate evolution
            current_best = 0.95 * (1 - math.exp(-gen / 50))  # Simulated improvement curve
            best_fitness = max(best_fitness, current_best)
        
        return {
            'best_fitness': best_fitness,
            'generations_run': generations,
            'population_size': population_size,
            'optimization_success': True,
            'convergence_reached': best_fitness > 0.9
        }
    
    def _run_differential_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run differential evolution optimization."""
        # Simulate differential evolution optimization
        return {
            'best_fitness': 0.87,
            'iterations': parameters.get('iterations', 100),
            'optimization_success': True,
            'convergence_reached': True
        }
    
    def _run_particle_swarm_optimization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run particle swarm optimization."""
        # Simulate particle swarm optimization
        return {
            'best_fitness': 0.91,
            'iterations': parameters.get('iterations', 100),
            'optimization_success': True,
            'convergence_reached': True
        }


# URL patterns
urlpatterns = [
    path('api/profiles/', ProfileAPIView.as_view(), name='profiles'),
    path('api/conditional/', ConditionalLogicAPIView.as_view(), name='conditional'),
    path('api/transform/', TransformerFunctionAPIView.as_view(), name='transform'),
    path('api/optimize/', OptimizationAPIView.as_view(), name='optimize'),
]


# Standalone functions for integration
def create_profile_template(name: str, 
                          default_config: Dict[str, Any],
                          conditional_rules: List[Dict[str, Any]],
                          transformer_rules: List[Dict[str, Any]]) -> str:
    """Create a profile template with conditional logic and transformers."""
    template_id = f"TPL_{name.upper()}_{uuid.uuid4().hex[:8].upper()}"
    
    template = {
        'id': template_id,
        'name': name,
        'default_config': default_config,
        'conditional_rules': conditional_rules,
        'transformer_rules': transformer_rules,
        'created_at': time.time(),
        'version': '1.0'
    }
    
    logger.info(f"Created profile template: {name}")
    return template_id


def process_conditional_logic(data: Dict[str, Any], 
                           rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process conditional logic rules."""
    result = copy.deepcopy(data)
    
    for rule in rules:
        condition = rule.get('condition', {})
        action = rule.get('action', {})
        
        # Evaluate condition
        if _evaluate_condition(data, condition):
            # Apply action
            result = _apply_action(result, action)
    
    return result


def apply_transformers(data: Any, transformer_chain: List[Dict[str, Any]]) -> Any:
    """Apply transformer functions in chain."""
    result = data
    
    for transformer in transformer_chain:
        func_name = transformer.get('function', 'identity')
        params = transformer.get('parameters', {})
        
        if func_name == 'normalize':
            result = _normalize_values(result, params)
        elif func_name == 'aggregate':
            result = _aggregate_values(result, params)
        elif func_name == 'filter':
            result = _filter_values(result, params)
        elif func_name == 'transform':
            result = _transform_values(result, params)
    
    return result


def _evaluate_condition(data: Dict[str, Any], condition: Dict[str, Any]) -> bool:
    """Evaluate a condition against data."""
    # Implementation of condition evaluation
    return True  # Simplified for demonstration


def _apply_action(data: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    """Apply an action to data."""
    # Implementation of action application
    return data  # Simplified for demonstration


def _normalize_values(data: Any, params: Dict[str, Any]) -> Any:
    """Normalize values."""
    # Implementation of normalization
    return data


def _aggregate_values(data: Any, params: Dict[str, Any]) -> Any:
    """Aggregate values."""
    # Implementation of aggregation
    return data


def _filter_values(data: Any, params: Dict[str, Any]) -> Any:
    """Filter values."""
    # Implementation of filtering
    return data


def _transform_values(data: Any, params: Dict[str, Any]) -> Any:
    """Transform values."""
    # Implementation of transformation
    return data


class DjangoAPIFramework:
    """Main Django API framework class."""
    
    def __init__(self):
        self.api_endpoints = {
            'profiles': '/api/profiles/',
            'conditional': '/api/conditional/',
            'transform': '/api/transform/',
            'optimize': '/api/optimize/'
        }
        self.templates = {}
        self.framework_id = f"DJANGO_API_{uuid.uuid4().hex[:8].upper()}"
        self.is_initialized = True
        
        logger.info(f"Initialized Django API framework: {self.framework_id}")
    
    def create_profile_template(self, name: str, 
                              default_config: Dict[str, Any],
                              conditional_rules: List[Dict[str, Any]],
                              transformer_rules: List[Dict[str, Any]]) -> str:
        """Create a profile template."""
        template_id = create_profile_template(name, default_config, conditional_rules, transformer_rules)
        self.templates[template_id] = {
            'name': name,
            'config': default_config,
            'rules': {'conditional': conditional_rules, 'transformer': transformer_rules}
        }
        return template_id
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information."""
        return {
            'framework_id': self.framework_id,
            'endpoints': self.api_endpoints,
            'templates_count': len(self.templates),
            'initialized': self.is_initialized,
            'timestamp': time.time()
        }


# For testing purposes
if __name__ == "__main__":
    # Create framework instance
    framework = DjangoAPIFramework()
    print(f"Django API Framework initialized: {framework.framework_id}")
    
    # Test profile template creation
    template_id = framework.create_profile_template(
        "test_template",
        {"cpu_priority": "high", "memory": "aggressive"},
        [{"condition": {"field": "cpu_usage", "operator": ">", "value": 80}, "action": {"priority": "high"}}],
        [{"function": "normalize", "parameters": {"range": [0, 1]}}]
    )
    print(f"Created template: {template_id}")
    
    # Show API info
    api_info = framework.get_api_info()
    print(f"API Info: {api_info}")