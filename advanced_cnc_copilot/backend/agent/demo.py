"""
Coding Agent Demo
Verifies the agent's behavior in Architect (Alpha), Developer (Beta), and Fast (Gamma) modes.
"""

import sys
import os
import logging

# Ensure path is correct
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from augmented_LLM_generator_system.core.openvino_engine import OpenVINOEngine
from augmented_LLM_generator_system.core.llm_processor import LLMProcessor
from augmented_LLM_generator_system.agent.core import CodingAgent

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

def run_demo():
    print("-" * 50)
    print("🤖 INITIALIZING CODING AGENT 🤖")
    print("-" * 50)
    
    # Init components
    engine = OpenVINOEngine(device="CPU")
    # For demo, we don't have a real model, but we set the path to trigger the Mock logic inside Engine
    processor = LLMProcessor(openvino_engine=engine, llama_model_path="models/openllama_3b.gguf")
    agent = CodingAgent(engine, processor)
    
    # Task 1: Architect (Alpha)
    print("\n[TASK 1] Design a complex system (Expected: ALPHA)")
    result_alpha = agent.run_task("Design a microservices architecture for a video processing app")
    print(">> RESULT (Truncated):")
    print(result_alpha[:200] + "...\n")
    
    # Task 2: Developer (Beta)
    print("\n[TASK 2] Write standard function (Expected: BETA)")
    result_beta = agent.run_task("Write a function to calculate Fibonacci numbers")
    print(">> RESULT (Truncated):")
    print(result_beta[:200] + "...\n")
    
    # Task 3: Fast Fix (Gamma)
    print("\n[TASK 3] Quick fix (Expected: GAMMA)")
    result_gamma = agent.run_task("Quick fix: typo in variable nmae")
    print(">> RESULT (Truncated):")
    print(result_gamma + "\n")

if __name__ == "__main__":
    run_demo()
