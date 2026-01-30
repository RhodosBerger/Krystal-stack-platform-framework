"""
Graph Knowledge Base
The Long-Term Memory of the machine.
Stores successful relationships: Material -> Tool -> Parameters -> Success.
Uses NetworkX to find optimal paths.
"""

import networkx as nx
import json
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("GRAPH_KNOWLEDGE")

class GraphKnowledgeBase:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.load_knowledge()

    def load_knowledge(self):
        """Loads (or initializes) the knowledge graph."""
        # In a real app, load from a generic graph DB or serialized file
        # Here we seed with some common sense knowledge
        self.add_path("Aluminum6061", "FaceMill_50mm", {"rpm": 8000, "feed": 3000}, weight=0.95)
        self.add_path("Titanium6Al4V", "EndMill_12mm_Carbide", {"rpm": 2500, "feed": 400}, weight=0.88)
        self.add_path("Steel4140", "Drill_8mm_HSS", {"rpm": 1200, "feed": 150}, weight=0.92)
        logger.info(f"Graph loaded with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def add_path(self, material: str, tool: str, params: Dict, weight: float = 1.0):
        """
        Records a successful connection. 
        Material -> Tool -> Setup
        Weight = Quality Score (0.0 - 1.0)
        """
        # Node structure: Category:Value
        mat_node = f"MAT:{material}"
        tool_node = f"TOOL:{tool}"
        # Store params as a discrete 'Setup' node to allow reuse
        setup_id = f"SETUP:{hash(frozenset(params.items()))}" 
        
        self.graph.add_node(mat_node, type="material")
        self.graph.add_node(tool_node, type="tool")
        self.graph.add_node(setup_id, type="setup", params=params)
        
        # Edges represent flow/compatibility
        # Material -> Tool (This material CAN be cut by this tool)
        if self.graph.has_edge(mat_node, tool_node):
            # Reinforce existing knowledge (simple average for MVP)
            old_w = self.graph[mat_node][tool_node]['weight']
            new_w = (old_w + weight) / 2
            self.graph[mat_node][tool_node]['weight'] = new_w
        else:
            self.graph.add_edge(mat_node, tool_node, weight=weight)
            
        # Tool -> Setup (This tool WORKS WELL with these params)
        self.graph.add_edge(tool_node, setup_id, weight=weight)

    def find_optimal_setup(self, material: str) -> Dict:
        """
        Finds the highest-confidence tool and setup for a material.
        Uses shortest path (where distance = 1 - confidence).
        """
        start = f"MAT:{material}"
        if start not in self.graph:
            return None

        # Logic: Find all reachable setups and pick the one with highest path product
        best_setup = None
        best_score = -1.0
        
        # Simple traversal for MVP (Depth=2)
        for tool in self.graph.successors(start):
            w1 = self.graph[start][tool]['weight']
            for setup in self.graph.successors(tool):
                w2 = self.graph[tool][setup]['weight']
                score = w1 * w2
                
                if score > best_score:
                    best_score = score
                    best_setup = self.graph.nodes[setup]['params']
                    best_setup['tool'] = tool.replace("TOOL:", "")
                    best_setup['confidence'] = score

        return best_setup

    def export_graph(self) -> Dict:
        return nx.node_link_data(self.graph)

# Global Instance
knowledge_graph = GraphKnowledgeBase()
