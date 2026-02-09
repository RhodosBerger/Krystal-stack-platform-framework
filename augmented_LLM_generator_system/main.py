"""
Augmented LLM Generator System
Main Entry Point
Integrates Essential Encoder, OpenVINO Engine, and LLM Processor.
"""

import sys
import os
import argparse
import logging
import json

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmented_LLM_generator_system.core.encoder import EssentialEncoder, EncodingType
from augmented_LLM_generator_system.core.openvino_engine import OpenVINOEngine, CognitiveState
from augmented_LLM_generator_system.core.llm_processor import LLMProcessor, Provider

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AugmentedGenerator")

def main():
    parser = argparse.ArgumentParser(description="Augmented LLM Generator System")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--encode", type=str, default="none", choices=["json", "base64", "hex", "neural", "none"], help="Input encoding strategy")
    parser.add_argument("--device", type=str, default="CPU", help="Target OpenVINO device (CPU, GPU)")
    parser.add_argument("--model", type=str, default=None, help="Path to .gguf model for LlamaCPP")
    parser.add_argument("--simulate-load", action="store_true", help="Simulate high system load (triggers GAMMA state)")
    
    args = parser.parse_args()
    
    # 1. Initialize Components
    logger.info("Initializing System Components...")
    encoder = EssentialEncoder()
    engine = OpenVINOEngine(device=args.device)
    processor = LLMProcessor(openvino_engine=engine, llama_model_path=args.model)
    
    # 2. Simulate Telemetry & Determine Cognitive State
    telemetry = {
        "cpu_load": 95.0 if args.simulate_load else 15.0,
        "memory_usage": 45.0,
        "user_activity": 50.0
    }
    
    current_state = engine.determine_cognitive_state(telemetry)
    logger.info(f"System Cognitive State: {current_state.value.upper()}")
    
    # 3. Optimize System based on State
    engine.optimize_system_for_state(current_state)
    
    # 4. Input Processing (Encoding)
    prompt_data = args.prompt
    if args.encode != "none":
        # Simulate complex input data structure
        input_payload = {"content": args.prompt, "timestamp": 1234567890}
        
        etype = EncodingType(args.encode)
        logger.info(f"Encoding input using {etype.value} strategy...")
        encoded_result = encoder.encode(input_payload, etype)
        
        logger.info(f"Encoded Data Hash: {encoded_result.hash}")
        logger.info(f"Compression/Size Stats: {encoded_result.size_encoded} bytes")
        
        # For the LLM, we normally pass the decoded content, but here we might pass the encoded string 
        # if the LLM is trained to handle it (e.g. Hex). For this demo, we append metadata.
        prompt_data = f"[System: Process {args.encode} encoded data]\n{encoded_result.data}\n\nUser Question: {args.prompt}"

    # 5. LLM Generation
    logger.info("Generating Response...")
    start_time = os.times().elapsed
    
    response = processor.generate(prompt=prompt_data)
    
    duration = os.times().elapsed - start_time
    logger.info(f"Generation Complete in {duration:.2f}s")
    
    print("\n" + "="*40)
    print(" GENERATED OUTPUT")
    print("="*40)
    print(response)
    print("="*40 + "\n")

    # 6. Post-Processing (Optional Neural Encoding of Output)
    # logger.info("Neural Encoding Output for Downstream Tasks...")
    # neural_out = encoder.encode([len(response)], EncodingType.NEURAL)
    # print(f"Output Neural Vector: {neural_out.data}")

if __name__ == "__main__":
    main()
