
import sys
import os
import time
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx # type: ignore

# Add backend to path to import Dopamine Engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from advanced_cnc_copilot.backend.cms.services.dopamine_engine import DopamineEngine, NeuroState

class MockRepository:
    def __init__(self):
        self.pregenerated_data = [] # Not used for generation logic directly here

    def get_recent_by_machine(self, machine_id, minutes):
        return []

def generate_synthetic_data(samples=10000):
    print(f"Generatign {samples} samples...")
    repo = MockRepository()
    engine = DopamineEngine(repo)
    
    X = []
    y = []
    
    for _ in range(samples):
        # Random inputs
        load = random.uniform(0, 120)
        temp = random.uniform(20, 100)
        vib = random.uniform(0, 5)
        # History avg - simulated
        hist_avg = random.uniform(0, 1.0)
        
        # Current metrics dict for engine
        metrics = {
            'spindle_load': load,
            'temperature': temp,
            'vibration_x': vib
        }
        
        # Use the logic from detect_phantom_trauma
        # We need to hack the engine slightly or just replicate logic for training label
        # But to be accurate, we should mock the history call inside or just use the logic directly.
        # Let's use the core calculation logic.
        
        # 1. Calc Stress
        stress = engine._calculate_cortisol_response(metrics)
        
        # 2. Logic Check
        physical_safe = (vib < 1.0 and temp < 55 and load < 80)
        
        is_phantom = False
        if physical_safe and stress > 0.6 and stress > hist_avg * 1.2:
            is_phantom = True
            
        # Features: Load, Temp, Vib, HistAvg
        X.append([load, temp, vib, hist_avg])
        y.append(1 if is_phantom else 0)
        
    return np.array(X, dtype=np.float32), np.array(y)

def train_and_export():
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print("Training MLP Classifier...")
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=500)
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("Converting to ONNX...")
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onx = convert_sklearn(clf, initial_types=initial_type)
    
    with open("phantom_net.onnx", "wb") as f:
        f.write(onx.SerializeToString())
        
    print("Saved to phantom_net.onnx")

if __name__ == "__main__":
    train_and_export()
