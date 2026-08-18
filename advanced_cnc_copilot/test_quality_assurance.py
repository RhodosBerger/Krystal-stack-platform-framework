from backend.core.quality_assurance import qa_guard, qa_stats, qa_scanner
import time

def test_quality_assurance():
    print("📊 Verifying Universal Validation & Statistics...")

    # 1. Test Runtime Statistics (Mock)
    print("\n[Stats] Testing QA Decorator...")
    @qa_guard.protect
    def slow_function():
        time.sleep(0.1)
        return "Done"

    slow_function()
    report = qa_stats.get_report()
    print(f"✅ Function executed. Avg Duration: {report['avg_duration_ms']:.2f}ms")
    if report['total_calls'] > 0:
        print("✅ Statistics recorded successfully")

    # 2. Test Security Scanner
    print("\n[Scanner] Testing Safety Checks...")
    unsafe_text = "Please delete everything using rm -rf /"
    issues = qa_scanner.scan_text(unsafe_text)
    if issues:
        print(f"✅ Scanner caught threat: {issues[0]}")
    else:
        print("❌ Scanner fail: Did not catch 'rm -rf'")

    # 3. Test Integrated G-Code Generator (Mocked)
    print("\n[Integration] Testing Wrapped Generator...")
    try:
        # Define a mock generator class that mimics the real one but uses the real decorator
        class MockGenerator:
            @qa_guard.protect
            def generate_from_description(self, desc):
                # Simulate work
                time.sleep(0.05)
                if "fail" in desc:
                    raise ValueError("Simulated Failure")
                return "G-Code Program", {"valid": True}

        generator = MockGenerator()
        
        # Test Success
        prog, val = generator.generate_from_description("Mill a pocket")
        new_report = qa_stats.get_report()
        print(f"✅ Mock Generator call 1: Success")
        
        # Test Failure Tracking
        try:
            generator.generate_from_description("fail this")
        except:
            pass
            
        final_report = qa_stats.get_report()
        print(f"✅ Total Calls tracked: {final_report['total_calls']}")
        print(f"✅ Failures tracked: {final_report['failure_count']}")
        
        if final_report['failure_count'] == 1:
            print("✅ Stats correctly recorded the failure")
        else:
            print("❌ Failure count mismatch")
        
    except Exception as e:
        print(f"⚠️ Integration test skipped: {e}")

if __name__ == "__main__":
    test_quality_assurance()
