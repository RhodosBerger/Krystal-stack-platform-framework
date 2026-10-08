import os
import sys
import subprocess

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_script = os.path.join(WORKSPACE_ROOT, "tests", "test_mimicry_compositor.py")
output_file = os.path.join(WORKSPACE_ROOT, "scratch", "urban_test_results.txt")

os.makedirs(os.path.dirname(output_file), exist_ok=True)

print(f"[TEST RUNNER] Executing: {test_script}")
result = subprocess.run([sys.executable, test_script], capture_output=True, text=True, cwd=WORKSPACE_ROOT)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== RETURNCODE: {result.returncode} ===\n")

print(f"[TEST RUNNER] Finished with code {result.returncode}. Output written to {output_file}")
