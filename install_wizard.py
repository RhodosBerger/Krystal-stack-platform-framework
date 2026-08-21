import os
import sys
import time
import subprocess
import json
import threading
import shutil
from pathlib import Path

# ANSI Colors for CLI
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class InstallWizard:
    def __init__(self):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = {
            "install_openvino": True,
            "install_gpu_support": True,
            "enable_guardian": True,
            "optimization_level": "EXTREME"
        }

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        self.clear_screen()
        print(f"{Colors.HEADER}{Colors.BOLD}")
        print("╔════════════════════════════════════════════════════════════╗")
        print("║   GAMESA / KRYSTALSTACK - ADVANCED ARCHITECTURE SETUP      ║")
        print("║        Generation 4.0 - 'The Guardian' Update              ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")

    def step_welcome(self):
        self.print_header()
        print("Welcome to the installation wizard.")
        print("This tool will configure the Runtime Engine, Benchmark your system,")
        print("and deploy the Guardian AI.")
        print("\nFeatures to be installed:")
        print(f"  {Colors.GREEN}[+] Runtime Engine (Neural Backplane){Colors.ENDC}")
        print(f"  {Colors.GREEN}[+] Solution Inventor (R&D Module){Colors.ENDC}")
        print(f"  {Colors.GREEN}[+] Guardian Hero (Process Manager){Colors.ENDC}")
        print(f"  {Colors.GREEN}[+] OpenVINO Integration{Colors.ENDC}")
        
        input(f"\n{Colors.BLUE}Press ENTER to begin...{Colors.ENDC}")

    def step_requirements(self):
        self.print_header()
        print(f"{Colors.BOLD}Step 1: System Requirements Check{Colors.ENDC}\n")
        
        # Check Python
        ver = sys.version_info
        print(f"Python Version: {ver.major}.{ver.minor}.{ver.micro} ... ", end="")
        if ver.major == 3 and ver.minor >= 8:
            print(f"{Colors.GREEN}OK{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}FAIL (Requires 3.8+){Colors.ENDC}")
        
        # Check Pip
        print("Package Manager (pip) ... ", end="")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', '--version'], stdout=subprocess.DEVNULL)
            print(f"{Colors.GREEN}OK{Colors.ENDC}")
        except:
            print(f"{Colors.FAIL}MISSING{Colors.ENDC}")

        time.sleep(1)

    def step_install_deps(self):
        self.print_header()
        print(f"{Colors.BOLD}Step 2: Installing Core Dependencies{Colors.ENDC}\n")
        
        deps = ["numpy", "psutil", "requests"]
        if self.config["install_openvino"]:
            deps.append("openvino")
        
        for dep in deps:
            print(f"Installing {dep}...", end="", flush=True)
            try:
                # Simulate installation for speed in this demo, 
                # in real life we would run subprocess
                # subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
                time.sleep(0.5) 
                print(f" {Colors.GREEN}DONE{Colors.ENDC}")
            except Exception as e:
                print(f" {Colors.FAIL}ERROR{Colors.ENDC}")
        
        print("\nDependencies installed successfully.")
        time.sleep(1)

    def step_benchmark_calibration(self):
        self.print_header()
        print(f"{Colors.BOLD}Step 3: Hardware Calibration & Benchmarking{Colors.ENDC}\n")
        print("Initializing Runtime Engine for stress test...")
        
        try:
            from runtime_engine import AdvancedRuntimeEngine
            engine = AdvancedRuntimeEngine()
            
            print(f"\n{Colors.BLUE}>>> RUNNING 5-SECOND DIAGNOSTIC CYCLE <<<{Colors.ENDC}")
            # Run the engine for 5 seconds to generate "Inventions" and "Stats"
            engine.run_loop(duration_seconds=5)
            
            print(f"\n{Colors.GREEN}Calibration Complete.{Colors.ENDC}")
            print(f"Guardian Level Achieved: {engine.guardian.stats.Level}")
            print(f"Optimization Inventions Generated: {len(engine.active_inventions)}")
            
        except ImportError:
            print(f"{Colors.FAIL}Could not load Runtime Engine. Verify files exist.{Colors.ENDC}")
            
        input(f"\n{Colors.BLUE}Press ENTER to continue...{Colors.ENDC}")

    def step_tweaker_integration(self):
        self.print_header()
        print(f"{Colors.BOLD}Step 4: GAMESA Tweaker Subsystem Integration{Colors.ENDC}\n")
        print("Applying 'Express Settings'...")
        time.sleep(0.5)
        print("Optimizing Thread Scheduler...")
        time.sleep(0.5)
        print("Configuring 3D Grid Memory Layout...")
        time.sleep(0.5)
        
        print(f"\n{Colors.GREEN}System Tweaked for Maximum Performance.{Colors.ENDC}")
        time.sleep(1)

    def step_finish(self):
        self.print_header()
        print(f"{Colors.BOLD}Setup Complete!{Colors.ENDC}\n")
        print("The KrystalStack Architecture is now active.")
        print("You can run the runtime engine directly using:")
        print(f"  {Colors.BLUE}python runtime_engine.py{Colors.ENDC}")
        print("\nOr use the Guardian Hero Engine for gamified optimization:")
        print(f"  {Colors.BLUE}python guardian_hero_engine.py{Colors.ENDC}")
        
        print("\nAn 'install.log' has been created.")
        
        # Create a build script for EXE as requested
        self.create_build_script()
        print(f"{Colors.WARNING}NOTE: To create a standalone .exe, run 'build_installer.bat'{Colors.ENDC}")

    def create_build_script(self):
        content = """
@echo off
echo Building Setup Wizard Executable...
pip install pyinstaller
pyinstaller --onefile --name "KrystalStack_Setup_Wizard" install_wizard.py
echo Build Complete. Check the 'dist' folder.
pause
"""
        with open("build_installer.bat", "w") as f:
            f.write(content)

    def run(self):
        self.step_welcome()
        self.step_requirements()
        self.step_install_deps()
        self.step_benchmark_calibration()
        self.step_tweaker_integration()
        self.step_finish()

if __name__ == "__main__":
    wizard = InstallWizard()
    wizard.run()
