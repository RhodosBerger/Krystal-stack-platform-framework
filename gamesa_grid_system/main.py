"""
Main entry point for GAMESA Grid System
"""
from src.grid import MemoryGrid3D
from src.guardian import GuardianCharacter
from src.scheduler import AdaptiveScheduler
from src.presets import preset_manager
from src.telemetry import HardwareTelemetry
import time
import random


def simulate_operation():
    """Simulate an operation for testing"""
    return {
        'id': f'op_{random.randint(1000, 9999)}',
        'type': random.choice(['compute', 'memory', 'gpu', 'io']),
        'priority': random.randint(1, 100),
        'resources': {
            'cpu': random.randint(1, 8),
            'memory_mb': random.randint(100, 1000),
            'gpu': random.random() * 0.5
        }
    }


def demo_gamespace_system():
    """Demonstrate the GAMESA Grid System"""
    print("=" * 80)
    print("GAMESA Grid System - 3D Memory Cache for Adaptive Performance Optimization")
    print("=" * 80)
    
    # Create the 3D memory grid
    print("\n1. Creating 3D Memory Grid...")
    grid = MemoryGrid3D(dimensions=(8, 8, 8))
    print(f"   - Grid dimensions: {grid.width}x{grid.height}x{grid.depth}")
    
    # Create the guardian character
    print("\n2. Initializing Guardian Character...")
    guardian = GuardianCharacter()
    print(f"   - Guardian strategy: {guardian.strategy.name}")
    print(f"   - Initial state: {guardian.state.value}")
    
    # Create the adaptive scheduler
    print("\n3. Setting up Adaptive Scheduler...")
    scheduler = AdaptiveScheduler(grid)
    print(f"   - Connected to grid and guardian")
    
    # Create hardware telemetry
    print("\n4. Initializing Hardware Telemetry...")
    telemetry = HardwareTelemetry()
    print(f"   - Collecting system data...")
    
    print("\n5. Starting Demonstration...")
    
    # Simulate scheduling operations
    for i in range(10):
        print(f"\n--- Operation {i+1} ---")
        
        # Create a simulated operation
        operation = simulate_operation()
        print(f"   Operation: {operation['id']} ({operation['type']})")
        
        # Schedule the operation
        success, position = scheduler.schedule_operation(operation)
        
        if success:
            print(f"   [OK] Scheduled at position: {position}")
            print(f"   - Grid occupancy: {grid.calculate_occupancy():.2%}")
        else:
            print(f"   [FAIL] Failed to schedule")
        
        # Show current system state
        system_state = telemetry.collect_telemetry()
        print(f"   System: CPU={system_state['cpu']['utilization']:.1%}, "
              f"GPU={system_state['gpu']['utilization']:.1%}, "
              f"Temp={system_state['thermal']['cpu_temp']:.1f}°C")
        
        time.sleep(0.5)  # Brief pause for demonstration
    
    print("\n" + "=" * 80)
    print("Performance Preset Demonstration")
    print("=" * 80)
    
    # Show preset effectiveness
    for preset_name in ['performance', 'power', 'thermal', 'balanced']:
        analysis = preset_manager.get_effectiveness_analysis(preset_name)
        print(f"\n{preset_name} preset:")
        print(f"  - Records: {analysis.get('history_records', 0)}")
        print(f"  - Avg Performance: {analysis.get('average_performance', 0):.2f}")
        print(f"  - Avg Power: {analysis.get('average_power', 0):.2f}")
        print(f"  - Avg Thermal: {analysis.get('average_thermal', 0):.2f}")
    
    print("\n" + "=" * 80)
    print("GAMESA Grid System Demonstration Complete!")
    print("The system successfully demonstrates:")
    print("- 3D memory grid for adaptive operation scheduling")
    print("- Guardian character using tic-tac-toe inspired strategy")
    print("- Hardware telemetry integration")
    print("- AI-guided placement optimization")
    print("- Performance preset management")
    print("=" * 80)


if __name__ == "__main__":
    demo_gamespace_system()