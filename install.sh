#!/bin/bash
# Installation script for Advanced Evolutionary Computing Framework

set -e  # Exit on any error

echo "==========================================="
echo "Advanced Evolutionary Computing Framework"
echo "Installation Script"
echo "==========================================="

# Check if running as root/administrator (only required for system-wide installation)
if [[ "$EUID" -eq 0 ]]; then
    echo "Running as root/administrator - proceeding with system-wide installation"
    INSTALL_PREFIX="/usr/local"
    SUDO=""
else
    echo "Running as regular user - installing to user directory"
    INSTALL_PREFIX="$HOME/.local"
    SUDO="sudo"
fi

# Check if Python 3.8+ is available
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.8 or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo "Found Python $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv evolutionary_env
source evolutionary_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core dependencies
echo "Installing core dependencies..."
pip install --upgrade numpy psutil requests scikit-learn matplotlib seaborn pandas

# Install AI/ML dependencies
echo "Installing AI/ML dependencies..."
pip install --upgrade torch torchvision tensorflow

# Install OpenVINO
echo "Installing OpenVINO..."
pip install --upgrade openvino

# Install web framework dependencies
echo "Installing web framework dependencies..."
pip install --upgrade flask django fastapi uvicorn

# Install visualization dependencies
echo "Installing visualization dependencies..."
pip install --upgrade plotly dash bokeh

# Install development dependencies
echo "Installing development dependencies..."
pip install --upgrade pytest pytest-cov black flake8 mypy sphinx

# Install optional GPU dependencies if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected - installing GPU dependencies..."
    pip install --upgrade cupy nvidia-ml-py3 pycuda
else
    echo "No NVIDIA GPU detected - skipping GPU dependencies"
fi

# Install the main package
echo "Installing the evolutionary framework..."
pip install -e .

# Create configuration directory
echo "Creating configuration directory..."
mkdir -p ~/.evolutionary_framework/config
mkdir -p ~/.evolutionary_framework/data
mkdir -p ~/.evolutionary_framework/logs

# Create default configuration
cat > ~/.evolutionary_framework/config/default_config.json << EOF
{
    "system": {
        "platform": "$(uname -s)",
        "architecture": "$(uname -m)",
        "cpu_count": $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4),
        "memory_mb": $(free -m 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 8192),
        "gpu_available": $(python3 -c "import torch; print(torch.cuda.is_available())")
    },
    "optimization": {
        "population_size": 100,
        "generations": 100,
        "mutation_rate": 0.01,
        "crossover_rate": 0.8,
        "elitism_rate": 0.1
    },
    "performance": {
        "max_threads": $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4),
        "memory_limit_mb": 2048,
        "batch_size": 32,
        "precision": "float32"
    },
    "api": {
        "host": "localhost",
        "port": 8080,
        "workers": 4,
        "debug": false
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "~/.evolutionary_framework/logs/app.log"
    }
}
EOF

# Create startup script
cat > evolutionary_framework.sh << EOF
#!/bin/bash
# Startup script for Evolutionary Framework

# Activate virtual environment
source $(pwd)/evolutionary_env/bin/activate

# Run the main application
python -m evolutionary_framework.main "\$@"
EOF

chmod +x evolutionary_framework.sh

# Create systemd service file (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    cat > evolutionary_framework.service << EOF
[Unit]
Description=Advanced Evolutionary Computing Framework
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/evolutionary_env/bin
ExecStart=$(pwd)/evolutionary_env/bin/python -m evolutionary_framework.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    echo "Systemd service file created (requires sudo to install):"
    echo "  sudo cp evolutionary_framework.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable evolutionary_framework"
    echo "  sudo systemctl start evolutionary_framework"
fi

echo ""
echo "==========================================="
echo "Installation Complete!"
echo "==========================================="
echo ""
echo "To use the framework:"
echo "  1. Source the virtual environment: source evolutionary_env/bin/activate"
echo "  2. Run the main application: python -m evolutionary_framework.main"
echo "  3. Or use the startup script: ./evolutionary_framework.sh"
echo ""
echo "Configuration is stored in: ~/.evolutionary_framework/config/"
echo "Logs are stored in: ~/.evolutionary_framework/logs/"
echo ""
echo "For API access, visit: http://localhost:8080/api/"
echo "For documentation, visit: http://localhost:8080/docs/"
echo ""
echo "To run tests: python -m pytest tests/"
echo "To run benchmarks: python -m evolutionary_framework.benchmarks"
echo ""
echo "Enjoy your Advanced Evolutionary Computing Framework!"
echo "==========================================="