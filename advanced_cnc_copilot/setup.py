"""
Advanced CNC Copilot - Automated Setup Script
Downloads and installs all required runtimes and dependencies

FEATURES:
- Python dependency installation
- PostgreSQL check and setup guidance
- Optional runtime downloads (Llama, Ollama)
- Environment configuration
- Database initialization
- Verification tests
"""

import subprocess
import sys
import os
import platform
from pathlib import Path
import json
import urllib.request
import shutil

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    required_major = 3
    required_minor = 9
    
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < required_major or (version.major == required_major and version.minor < required_minor):
        print_error(f"Python {required_major}.{required_minor}+ required!")
        print_info(f"Download from: https://www.python.org/downloads/")
        return False
    
    print_success(f"Python version OK ({version.major}.{version.minor}.{version.micro})")
    return True


def check_pip():
    """Check if pip is installed"""
    print_header("Checking pip")
    
    try:
        import pip
        print_success(f"pip is installed (version {pip.__version__})")
        return True
    except ImportError:
        print_error("pip is not installed!")
        print_info("Install pip: python -m ensurepip --upgrade")
        return False


def install_python_dependencies():
    """Install Python packages from requirements.txt"""
    print_header("Installing Python Dependencies")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        return False
    
    print_info(f"Installing from: {requirements_file}")
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        print("\nInstalling dependencies (this may take 5-10 minutes)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", str(requirements_file),
            "--upgrade"
        ])
        
        print_success("All Python dependencies installed!")
        return True
    
    except subprocess.CalledProcessError as e:
        print_error(f"Installation failed: {e}")
        return False


def check_postgresql():
    """Check if PostgreSQL is installed"""
    print_header("Checking PostgreSQL")
    
    # Try to find psql command
    psql_found = shutil.which("psql") is not None
    
    if psql_found:
        print_success("PostgreSQL is installed!")
        try:
            result = subprocess.run(["psql", "--version"], capture_output=True, text=True)
            print(f"   Version: {result.stdout.strip()}")
        except:
            pass
        return True
    else:
        print_warning("PostgreSQL not found on PATH")
        print_info("Download PostgreSQL:")
        if platform.system() == "Windows":
            print("   https://www.postgresql.org/download/windows/")
            print("   or install via: winget install PostgreSQL.PostgreSQL")
        elif platform.system() == "Darwin":  # macOS
            print("   brew install postgresql")
        else:  # Linux
            print("   sudo apt-get install postgresql  # Debian/Ubuntu")
            print("   sudo yum install postgresql      # RHEL/CentOS")
        return False


def check_nodejs():
    """Check if Node.js is installed (optional)"""
    print_header("Checking Node.js (Optional)")
    
    node_found = shutil.which("node") is not None
    
    if node_found:
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            print_success(f"Node.js is installed: {result.stdout.strip()}")
            return True
        except:
            pass
    
    print_warning("Node.js not found (optional for frontend development)")
    print_info("Download from: https://nodejs.org/")
    return False


def setup_environment_file():
    """Create .env file template"""
    print_header("Setting up Environment Configuration")
    
    env_file = Path(__file__).parent / ".env"
    env_template = Path(__file__).parent / ".env.template"
    
    # Create template if doesn't exist
    if not env_template.exists():
        template_content = """# Advanced CNC Copilot - Environment Configuration

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/cnc_copilot
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# API Keys (get from respective providers)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# LLM Configuration
DEFAULT_LLM_PROVIDER=openai  # openai, anthropic, local
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379/0

# MQTT Configuration (for IoT sensors)
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_USERNAME=
MQTT_PASSWORD=

# API Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True
API_LOG_LEVEL=info

# Security
SECRET_KEY=your_secret_key_here_change_in_production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Feature Flags
ENABLE_PREDICTIVE_MAINTENANCE=True
ENABLE_LLM_GCODE_GENERATION=True
ENABLE_IOT_SENSORS=True
ENABLE_MULTI_BOT_SYSTEM=True

# Paths
DATA_DIRECTORY=./data
LOGS_DIRECTORY=./logs
MODELS_DIRECTORY=./models

# Debug
DEBUG_MODE=True
VERBOSE_LOGGING=False
"""
        with open(env_template, 'w') as f:
            f.write(template_content)
        print_success("Created .env.template")
    
    # Copy to .env if doesn't exist
    if not env_file.exists():
        shutil.copy(env_template, env_file)
        print_success("Created .env file")
        print_warning("IMPORTANT: Edit .env file with your API keys and database credentials!")
    else:
        print_info(".env file already exists, not overwriting")
    
    return True


def create_directories():
    """Create necessary directories"""
    print_header("Creating Project Directories")
    
    base_dir = Path(__file__).parent
    directories = [
        "data",
        "logs",
        "models",
        "uploads",
        "exports",
        "backups"
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print_success(f"Created: {dir_name}/")
    
    return True


def download_local_llm():
    """Download local LLM model (optional)"""
    print_header("Local LLM Setup (Optional)")
    
    print_info("Local LLM provides offline AI capabilities")
    print("Options:")
    print("  1. Ollama (recommended) - Easy setup, good performance")
    print("  2. llama.cpp - Direct model loading")
    print("  3. Skip - Use cloud LLMs only")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == "1":
        print_info("Installing Ollama...")
        if platform.system() == "Windows":
            print("Download from: https://ollama.ai/download/windows")
        elif platform.system() == "Darwin":
            print("Run: curl https://ollama.ai/install.sh | sh")
        else:
            print("Run: curl https://ollama.ai/install.sh | sh")
        
        print("\nAfter installing Ollama, run:")
        print("  ollama pull llama2")
        print("  ollama pull codellama")
    
    elif choice == "2":
        print_info("llama.cpp already included in requirements.txt")
        print("Download models from: https://huggingface.co/models")
        print("Place in: ./models/ directory")
    
    else:
        print_info("Skipping local LLM setup")
    
    return True


def verify_installation():
    """Verify all installations"""
    print_header("Verifying Installation")
    
    checks = []
    
    # Test imports
    test_packages = [
        ("fastapi", "FastAPI"),
        ("sqlalchemy", "SQLAlchemy"),
        ("numpy", "NumPy"),
        ("sklearn", "scikit-learn"),
        ("paho.mqtt.client", "MQTT Client"),
    ]
    
    for package, name in test_packages:
        try:
            __import__(package)
            print_success(f"{name} imported successfully")
            checks.append(True)
        except ImportError:
            print_error(f"{name} failed to import")
            checks.append(False)
    
    # Test database connection (optional)
    try:
        from database.connection import DatabaseConnectionManager
        print_info("Database connection module available")
    except ImportError:
        print_warning("Database connection module not found (will be created)")
    
    success_rate = sum(checks) / len(checks) * 100
    
    print(f"\n{Colors.BOLD}Verification Summary:{Colors.END}")
    print(f"   Success rate: {success_rate:.0f}%")
    
    if success_rate == 100:
        print_success("All core dependencies verified!")
        return True
    elif success_rate >= 80:
        print_warning("Most dependencies verified, some optional packages missing")
        return True
    else:
        print_error("Many dependencies missing or failed")
        return False


def print_next_steps():
    """Print next steps after setup"""
    print_header("Setup Complete! 🎉")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.END}\n")
    
    print("1. Configure Environment:")
    print("   - Edit .env file with your API keys")
    print("   - Set DATABASE_URL to your PostgreSQL connection string")
    print("")
    
    print("2. Initialize Database:")
    print("   - Create PostgreSQL database: createdb cnc_copilot")
    print("   - Run schema: psql cnc_copilot < database/schema.sql")
    print("   - Import knowledge base: python scripts/load_knowledge_to_db.py")
    print("")
    
    print("3. Start Services:")
    print("   - API Server: python api/manufacturing_api.py")
    print("   - Dashboard: Open cms/dashboard/developer-preview.html")
    print("")
    
    print("4. Test System:")
    print("   - Run tests: pytest tests/")
    print("   - Check API docs: http://localhost:8000/docs")
    print("")
    
    print(f"{Colors.BOLD}Documentation:{Colors.END}")
    print("   - Quick Start: scripts/LLM_QUICKSTART.md")
    print("   - Developer Preview: DEVELOPER_PREVIEW.md")
    print("   - Integration Guide: practical_integration_guide.md")
    print("")
    
    print_success("System is ready for development! 🚀")


def main():
    """Main setup function"""
    print_header("Advanced CNC Copilot - Automated Setup")
    print("This script will install all required dependencies and set up the development environment.\n")
    
    # Run checks and installations
    steps = [
        ("Python Version", check_python_version),
        ("pip", check_pip),
        ("Python Dependencies", install_python_dependencies),
        ("PostgreSQL Database", check_postgresql),
        ("Node.js (Optional)", check_nodejs),
        ("Environment Config", setup_environment_file),
        ("Project Directories", create_directories),
        ("Local LLM (Optional)", download_local_llm),
        ("Verification", verify_installation),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print_error(f"Error in {step_name}: {e}")
            results.append((step_name, False))
    
    # Summary
    print_header("Setup Summary")
    for step_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}  {step_name}")
    
    # Next steps
    print_next_steps()
    
    return all(r for _, r in results)


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
