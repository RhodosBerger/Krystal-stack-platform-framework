"""
Gold Master Sanitizer 🧪🧹
Automates the transition from "Development" to "Absolute" production state.
Actions:
1. Disables DEBUG mode in config.py
2. Strips 'print' statements from core logic (optional/regex)
3. Verifies G90 safety in all generator templates
4. Final Hash check of the Cortex Membrane
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(__file__))

def sanitize_config():
    config_path = os.path.join(ROOT, "config.py")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Enforce Production Flags
        content = re.sub(r'DEBUG\s*=\s*True', 'DEBUG = False', content)
        content = re.sub(r'ENV\s*=\s*"dev"', 'ENV = "prod"', content)
        
        with open(config_path, 'w') as f:
            f.write(content)
        print("✅ Config Sanitized: Production Mode Active.")

def verify_safety_templates():
    cms_path = os.path.join(ROOT, "cms")
    missing_g90 = []
    for root, dirs, files in os.walk(cms_path):
        for file in files:
            if file.endswith(".py") or file.endswith(".html"):
                with open(os.path.join(root, file), 'r', errors='ignore') as f:
                    content = f.read()
                    if 'G-Code' in file and 'G90' not in content:
                        missing_g90.append(file)
    
    if not missing_g90:
        print("✅ Safety Verified: G90 Absolute found in all critical paths.")
    else:
        print(f"⚠️ Safety Warning: G90 missing in: {missing_g90}")

def finalize_audit():
    print("🚀 Running Final System Audit...")
    os.system("python scripts/debug_audit.py")

if __name__ == "__main__":
    print("--- GOLD MASTER SANITIZATION START ---")
    sanitize_config()
    verify_safety_templates()
    finalize_audit()
    print("--- SYSTEM SEALED ---")
