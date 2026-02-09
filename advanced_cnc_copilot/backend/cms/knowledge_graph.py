#!/usr/bin/env python3
"""
KNOWLEDGE GRAPH (Topology Engine)
The Map of Cause and Effect.
"""

from typing import List, Dict, Optional

class Node:
    def __init__(self, name: str, node_type: str):
        self.name = name
        self.type = node_type # "PARAM", "PROBLEM", "ROOT"
        self.parents = []
        self.children = []

    def add_child(self, node):
        self.children.append(node)
        node.parents.append(self)

    def __repr__(self):
        return f"[{self.type}] {self.name}"

class CausalGraph:
    def __init__(self):
        self.nodes = {}
        self._build_topology()

    def _build_topology(self):
        # --- 1. PARAMETER TOPOLOGY (Forces) ---
        rpm = self._add("RPM", "KNOB")
        feed = self._add("FEED", "KNOB")
        
        mrr = self._add("MRR", "DERIVED")
        heat = self._add("HEAT", "GAUGE")
        vib = self._add("VIBRATION", "GAUGE")
        
        # Connections
        rpm.add_child(mrr) # RPM increases MRR
        feed.add_child(mrr) # Feed increases MRR
        mrr.add_child(heat) # MRR creates Heat
        rpm.add_child(vib) # RPM excites Vibration

        # --- 2. PROBLEM TOPOLOGY (Failures) ---
        unstable = self._add("UNSTABLE_CUT", "ROOT_CAUSE")
        chatter = self._add("CHATTER", "PROBLEM")
        surface_fail = self._add("BAD_SURFACE", "SYMPTOM")
        tool_break = self._add("TOOL_BREAKAGE", "CATASTROPHE")

        # Connections (Casual Chain)
        unstable.add_child(chatter)
        chatter.add_child(surface_fail)
        chatter.add_child(tool_break)

    def _add(self, name, type) -> Node:
        node = Node(name, type)
        self.nodes[name] = node
        return node

    def trace_root_cause(self, symptom_name: str) -> List[str]:
        """
        Given a symptom, find the Root Cause path.
        """
        if symptom_name not in self.nodes:
            return []
            
        start_node = self.nodes[symptom_name]
        path = [start_node.name]
        
        # Simple BFS/DFS up the chain
        # For simplicity, we just grab the first parent recursively
        curr = start_node
        while curr.parents:
            parent = curr.parents[0] # Primary cause
            path.append(parent.name)
            curr = parent
            
        return list(reversed(path))

    def find_knobs_for_gauge(self, gauge_name: str) -> List[str]:
        """
        Given a metric (Heat), find what Knobs control it (RPM, Feed).
        """
        if gauge_name not in self.nodes: return []
        
        knobs = []
        # Traverse up
        def traverse(node):
            if node.type == "KNOB":
                knobs.append(node.name)
            for p in node.parents:
                traverse(p)
                
        traverse(self.nodes[gauge_name])
        return list(set(knobs)) # Dedup

# Usage
if __name__ == "__main__":
    graph = CausalGraph()
    
    print("--- TRACE: Why is the Surface Bad? ---")
    chain = graph.trace_root_cause("BAD_SURFACE")
    print(f"Causal Chain: {' -> '.join(chain)}")
    
    print("\n--- CONTROL: How do I reduce Heat? ---")
    knobs = graph.find_knobs_for_gauge("HEAT")
    print(f"Adjustable Knobs: {knobs}")
