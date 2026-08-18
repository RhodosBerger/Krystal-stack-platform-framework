"""
Similarity Search for Projects
"""

import numpy as np
from django.db.models import Q

def calculate_dimension_similarity(dims1, dims2):
    """Calculate similarity between two dimension dicts"""
    
    # Extract common keys
    common_keys = set(dims1.keys()) & set(dims2.keys())
    
    if not common_keys:
        return 0.0
    
    # Calculate normalized differences
    differences = []
    for key in common_keys:
        try:
            val1 = float(dims1[key])
            val2 = float(dims2[key])
            
            # Normalize by average
            avg = (val1 + val2) / 2
            if avg > 0:
                diff = abs(val1 - val2) / avg
                differences.append(diff)
        except (ValueError, TypeError):
            # Skip non-numeric values
            continue
    
    if not differences:
        return 0.0
    
    # Average difference → similarity
    avg_diff = np.mean(differences)
    similarity = max(0, 1 - avg_diff)
    
    return similarity

def find_similar(project, limit=5):
    """Find similar projects"""
    from .models import Project
    
    # Get all successful projects with same material
    candidates = Project.objects.filter(
        material=project.material,
        success=True
    ).exclude(id=project.id)
    
    similarities = []
    
    for candidate in candidates:
        # Dimension similarity
        dim_sim = calculate_dimension_similarity(
            project.dimensions,
            candidate.dimensions
        )
        
        # Complexity similarity
        complexity_diff = abs(project.complexity_score - candidate.complexity_score) / 10.0
        complexity_sim = max(0, 1 - complexity_diff)
        
        # Combined similarity (weighted average)
        overall_sim = 0.7 * dim_sim + 0.3 * complexity_sim
        
        similarities.append((overall_sim, candidate))
    
    # Sort by similarity (descending)
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    # Return top N
    return [proj for score, proj in similarities[:limit]]
