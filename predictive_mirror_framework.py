import time
import math
import random
from collections import deque
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class PredictionPoint:
    timestamp: float
    value: float
    confidence: float # 0.0 to 1.0

class TemporalPredictor:
    """
    The Oracle Engine.
    Uses historical data to project future states using a simplified Kalman-like approach.
    """
    def __init__(self, history_size=20):
        self.history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size)
        self.last_val = 0.0
        self.last_time = time.time()

    def input_data(self, val: float):
        """Feed current data into the predictor (e.g., CPU Load, Mouse X, Scene Complexity)"""
        now = time.time()
        dt = now - self.last_time
        if dt == 0: return # Prevent div by zero in hyper-fast loops
        
        # Calculate Velocity (Rate of Change)
        velocity = (val - self.last_val) / dt
        self.velocity_history.append(velocity)
        
        self.history.append(val)
        self.last_val = val
        self.last_time = now

    def predict_horizon(self, steps=10, dt_step=0.05) -> List[PredictionPoint]:
        """
        Projects the future path based on momentum and volatility.
        Returns a list of predicted points for visualization.
        """
        if len(self.history) < 2:
            return []

        # 1. Calculate Momentum (Avg Velocity)
        avg_vel = sum(self.velocity_history) / len(self.velocity_history)
        
        # 2. Calculate Volatility (Standard Deviation of Velocity) -> Uncertainty
        variance = sum((v - avg_vel) ** 2 for v in self.velocity_history) / len(self.velocity_history)
        volatility = math.sqrt(variance)
        
        # 3. Project Future
        predictions = []
        current_val = self.last_val
        
        for i in range(steps):
            # Apply momentum
            current_val += avg_vel * dt_step
            
            # Uncertainty grows with time (Cone of Uncertainty)
            # Volatility adds a +/- range, confidence drops as we look further
            uncertainty_factor = volatility * (i + 1) * 0.5
            confidence = max(0.1, 1.0 - (uncertainty_factor * 0.1))
            
            # Simulate "Chaos" in the prediction (Noise)
            # In a real system, this would be interference from other processes
            noise = random.uniform(-uncertainty_factor, uncertainty_factor) * 0.1
            
            pred_point = PredictionPoint(
                timestamp=self.last_time + (i * dt_step),
                value=current_val + noise,
                confidence=confidence
            )
            predictions.append(pred_point)
            
        return predictions

class ResourcePreloader:
    """
    Simulates the 'Action' taken based on prediction.
    If prediction > threshold, it triggers 'Shadow Load'.
    """
    def __init__(self):
        self.shadow_buffer = {} # Stores pre-calculated results
        
    def check_and_preload(self, predictions: List[PredictionPoint]):
        """
        Analyzes the future. If a spike is predicted, pre-load resources.
        """
        actions = []
        if not predictions: return actions
        
        # Find max load in the horizon
        max_pred = max(p.value for p in predictions)
        
        if max_pred > 80.0:
            actions.append("INJECT_VOLTAGE_BOOST")
            self.shadow_buffer["HighLoadAssets"] = "READY"
        
        if max_pred < 20.0:
            actions.append("ENGAGE_ECO_MODE")
            
        return actions

if __name__ == "__main__":
    # Test
    oracle = TemporalPredictor()
    loader = ResourcePreloader()
    
    # Simulate a rising sine wave signal
    print("Training Oracle...")
    val = 0
    for i in range(50):
        val += math.sin(i * 0.2) + (i * 0.1) # Upward trend with oscillation
        oracle.input_data(val)
        time.sleep(0.01)
        
    print(f"Current Value: {oracle.last_val:.2f}")
    
    # Predict
    future = oracle.predict_horizon(steps=5)
    print("\nFuture Prediction:")
    for pt in future:
        print(f"  T+{pt.timestamp - oracle.last_time:.2f}s | Val: {pt.value:.2f} | Conf: {pt.confidence:.0%}")
        
    # Check Actions
    acts = loader.check_and_preload(future)
    print(f"\nSystem Actions: {acts}")