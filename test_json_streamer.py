import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.cms.services.dopamine_engine import DopamineEngine
from backend.core.json_streamer import JSONCodeStreamer
from test_synapse import MockRepository # Reuse mock

def run_test():
    print("🚀 Starting JSON Profile Streamer Test...")
    
    # Setup
    repo = MockRepository()
    engine = DopamineEngine(repo)
    streamer = JSONCodeStreamer(engine)
    
    # Define an Advanced Profile (JSON)
    profile = {
        "profile_id": "TEST_PROFILE_001",
        "segments": [
            {"id": 1, "target": {"x": 10, "y": 10}, "optimized_feed": 1200},
            {"id": 2, "target": {"x": 20, "y": 20}, "optimized_feed": 1500},
            {"id": 3, "target": {"x": 30, "y": 30}, "optimized_feed": 2000, "simulated_risk": "high"}, # Should Fail
            {"id": 4, "target": {"x": 40, "y": 40}, "optimized_feed": 1000}
        ]
    }
    
    print(f"📄 Executing Profile: {profile['profile_id']}")
    
    for event in streamer.execute_profile(profile):
        print(f"EVENT: {event}")
        
    print("✅ Test Complete.")

if __name__ == "__main__":
    run_test()
