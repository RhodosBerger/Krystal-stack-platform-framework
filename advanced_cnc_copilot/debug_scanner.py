import os
import py_compile
import sys
import importlib.util

def check_syntax(start_path):
    print(f"🔍 Scanning for Syntax Errors in '{start_path}'...")
    error_count = 0
    checked_count = 0
    
    for root, dirs, files in os.walk(start_path):
        if "venv" in root or "node_modules" in root or "__pycache__" in root:
            continue
            
        for file in files:
            if file.endswith(".py"):
                checked_count += 1
                full_path = os.path.join(root, file)
                try:
                    py_compile.compile(full_path, doraise=True)
                except py_compile.PyCompileError as e:
                    print(f"❌ SYNTAX ERROR in {full_path}:")
                    print(e)
                    error_count += 1
                except Exception as e:
                    print(f"⚠️ Check Failed for {full_path}: {e}")
                    
    print(f"✅ Scanned {checked_count} files. Found {error_count} syntax errors.")
    return error_count

def check_critical_imports():
    print("\n🔍 Verifying Critical Module Imports...")
    
    # Add root to path so we can mimic the runtime environment
    root_path = os.getcwd()
    if root_path not in sys.path:
        sys.path.append(root_path)
    
    modules_to_check = [
        "backend.main",
        "backend.core.orchestrator",
        "backend.core.security",
        "cms.interaction_supervisor",
        "cms.auditor_agent",
        "cms.message_bus"
    ]
    
    import_errors = 0
    for mod in modules_to_check:
        try:
            # We use importlib to check findability without necessarily executing everything
            if importlib.util.find_spec(mod) is None:
                print(f"❌ MODULE NOT FOUND: {mod}")
                import_errors += 1
            else:
                print(f"ok: {mod}")
        except Exception as e:
            print(f"⚠️ Import Check Error for {mod}: {e}")
            import_errors += 1
            
    return import_errors

if __name__ == "__main__":
    print("=== 🐛 AUTOMATED CODE DEBUGGER ===")
    s_err = check_syntax(".")
    i_err = check_critical_imports()
    
    if s_err == 0 and i_err == 0:
        print("\n✨ ALL SYSTEMS GO. No static errors detected.")
    else:
        print(f"\n🛑 FOUND ISSUES: {s_err} Syntax Errors, {i_err} Import Errors.")
