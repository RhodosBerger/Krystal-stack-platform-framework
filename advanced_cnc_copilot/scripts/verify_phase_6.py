
import sys
import os
import logging
import json

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from cms.graph_knowledge import knowledge_graph
from cms.scheduler_agent import scheduler
from cms.resonance_trainer import trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY_PHASE_6")

def main():
    logger.info("--- STARTING GRAPH MIND VERIFICATION ---")
    
    # 1. Test Graph & Scheduler
    logger.info("[TEST 1] Testing Knowledge Graph & Scheduler...")
    
    # Mock some incoming jobs
    jobs = [
        {"id": "JOB_UNK", "material": "Kryptonite", "priority": 1},      # Unknown Material
        {"id": "JOB_ALU", "material": "Aluminum6061", "priority": 2},    # High Confidence
        {"id": "JOB_TIN", "material": "Titanium6Al4V", "priority": 3}    # Known but standard
    ]
    
    optimized = scheduler.optimize_schedule(jobs)
    
    logger.info("\n--- OPTIMIZED SCHEDULE ---")
    for step in optimized:
        if step.get('type') == 'META':
            print(f">>> [SETUP CHANGE] {step['action']} ({step['duration']}m)")
        else:
            confidence = step.get('estimated_confidence', 0)
            print(f" - {step['id']} ({step['material']}) | Confidence: {confidence:.2f}")
            if 'recommended_setup' in step:
                print(f"   -> Suggestion: {step['recommended_setup']['tool']} @ {step['recommended_setup']['rpm']} RPM")

    # Assert logic: Aluminum should be high confidence, Kryptonite low
    assert optimized[0]['id'] == 'JOB_ALU', "Scheduler failed to prioritize High Confidence job (Aluminum)"
    assert optimized[-1]['id'] == 'JOB_UNK', "Scheduler failed to deprioritize Unknown material"
    
    
    # 2. Test Feedback Loop (Resonance Trainer)
    logger.info("\n[TEST 2] Testing Resonance Trainer (Feedback Loop)...")
    
    # Inject a fake memory of a NEW discovery for Kryptonite
    fake_memory = [
        {
            "timestamp": 123456789,
            "material": "Kryptonite",
            "rpm": 10000, 
            "feed": 1200, 
            "action_taken": "EXPERIMENTAL_CUT",
            "outcome_dopamine": 95.0, # Huge Win!
            "outcome_cortisol": 10.0
        }
    ]
    
    # Mock the memory file
    with open("agg_demo_memory.json", "w") as f:
        json.dump(fake_memory, f)
        
    # Run the trainer
    trainer.run_training_cycle()
    
    # Verify the graph learned Kryptonite
    logger.info("Querying Graph for 'Kryptonite' after training...")
    knowledge = knowledge_graph.find_optimal_setup("Kryptonite")
    
    assert knowledge is not None, "Graph failed to learn from Resonance Training!"
    assert knowledge['rpm'] == 10000, "Graph learned wrong parameters!"
    
    logger.info(f"✅ SUCCESS! The Matrix learned that Kryptonite needs 10k RPM.")
    
    logger.info("\n--- PHASE 6 VERIFIED ---")

if __name__ == "__main__":
    main()
