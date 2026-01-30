"""
Resonance Trainer
The "Sleep Mode" of the system.
Reads daily logs, identifies "Wins" (High Dopamine, Stable Flow), and re-trains the Graph.
"""

import logging
import json
import random
from typing import List
from cms.graph_knowledge import knowledge_graph
from cms.log_transformer import LogTransformer

logger = logging.getLogger("RESONANCE_TRAINER")

class ResonanceTrainer:
    def __init__(self):
        self.transformer = LogTransformer()
        self.knowledge = knowledge_graph

    def run_training_cycle(self):
        """
        1. Read Logs.
        2. Identify successful runs (Low Cortisol, High Dopamine).
        3. Update Graph Weights.
        """
        logger.info("Starting Resonance Training Cycle (Sleep Mode)...")
        
        # 1. Simulate reading "Yesterday's" logs (using memory file or transformer)
        # For prototype, we'll use the 'agg_demo_memory.json' if available, or generate synthetic
        try:
            with open("agg_demo_memory.json", "r") as f:
                history = json.load(f)
        except FileNotFoundError:
            logger.warning("No memory file found. Training skipped.")
            return

        training_count = 0
        
        for entry in history:
            # 2. Determine Success
            # Success = High Dopamine (>70) AND Low Cortisol (<40)
            dopamine = entry.get('outcome_dopamine', 0)
            cortisol = entry.get('outcome_cortisol', 100)
            
            if dopamine > 70 and cortisol < 40:
                # 3. Reinforce Graph
                material = entry.get('material', 'Unknown')
                # We need to infer the tool from the params/action in this demo dataset
                # Hypothetical mapping based on speed
                rpm = entry.get('rpm', 0)
                tool = "HighSpeed_EndMill" if rpm > 8000 else "Standard_Cutter"
                
                params = {
                    "rpm": rpm, 
                    "feed": entry.get('feed', 0),
                    "strategy": entry.get('action_taken', 'STANDARD')
                }
                
                # Update the Knowledge Graph
                # We add it with a weight of 1.0 (Successful)
                # The graph engine will average this with existing weights
                self.knowledge.add_path(material, tool, params, weight=1.0)
                training_count += 1

        logger.info(f"Training Complete. Assimilated {training_count} successful experiences into Long-Term Memory.")
        
        # 4. Generate LLM Fine-Tuning Data (Bonus)
        # Convert these wins into Q&A format for future LLM training
        self._generate_llm_dataset(history)
        
    def _generate_llm_dataset(self, history: List):
        """Generates a JSONL file for OpenAI/Llama fine-tuning."""
        training_data = []
        for entry in history:
            dopamine = entry.get('outcome_dopamine', 0)
            if dopamine > 80:
                prompt = f"Suggest parameters for {entry.get('material')} at {entry.get('rpm')} RPM."
                completion = f"Recommended Action: {entry.get('action_taken')}. Result: Stable flow with {dopamine:.1f}% efficiency."
                training_data.append({"prompt": prompt, "completion": completion})
        
        with open("training_data.jsonl", "w") as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Generated {len(training_data)} fine-tuning examples in 'training_data.jsonl'.")

# Global Instance
trainer = ResonanceTrainer()
