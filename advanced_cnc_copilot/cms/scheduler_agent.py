"""
Scheduler Agent
Optimizes the production queue using Graph Knowledge.
Goal: Minimize tool changes and setup times by grouping 'compatible' jobs.
"""

import logging
from typing import List, Dict
from cms.graph_knowledge import knowledge_graph

logger = logging.getLogger("SCHEDULER")

class SchedulerAgent:
    def __init__(self):
        self.knowledge = knowledge_graph

    def optimize_schedule(self, jobs: List[Dict]) -> List[Dict]:
        """
        Reorders jobs to maximize flow.
        Jobs is a list of dicts: {'id': 'J1', 'material': 'Aluminum6061', ...}
        """
        logger.info(f"Optimizing schedule for {len(jobs)} jobs...")
        
        # 1. Enlightened Enrichment
        # Ask the Graph for the best setup for each job
        for job in jobs:
            recommendation = self.knowledge.find_optimal_setup(job['material'])
            if recommendation:
                job['recommended_setup'] = recommendation
                job['estimated_confidence'] = recommendation['confidence']
            else:
                job['estimated_confidence'] = 0.5 # Unknown material
        
        # 2. Grouping Strategy (Clustering)
        # Group by Material -> Tool to minimize switchovers
        # Check if 'Aluminum' flows well into 'Titanium' (unlikely) vs 'Aluminum' -> 'Plastic'
        
        # Simple Heuristic Sort: Group by Material, then by Confidence (High confidence first)
        def sort_key(j):
            return (j['material'], -j.get('estimated_confidence', 0))
            
        optimized_jobs = sorted(jobs, key=sort_key)
        
        # 3. Insert specific setup instructions
        last_material = None
        final_sequence = []
        
        for job in optimized_jobs:
            if job['material'] != last_material:
                # Insert a 'Virtual' Changeover Op
                final_sequence.append({
                    "id": "SETUP_CHANGE",
                    "type": "META",
                    "action": f"Prepare for {job['material']}",
                    "duration": 15 # minutes
                })
                last_material = job['material']
            final_sequence.append(job)
            
        logger.info("Schedule Optimized.")
        return final_sequence

# Global Instance
scheduler = SchedulerAgent()
