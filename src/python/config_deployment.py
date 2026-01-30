"""
GAMESA/KrystalStack Configuration and Deployment Module

Handles system configuration, deployment workflows, and runtime management
"""

import os
import json
import yaml
import toml
from pathlib import Path
import platform
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import uuid
from enum import Enum
import logging
import subprocess
import shutil


class DeploymentTarget(Enum):
    """Target platforms for deployment."""
    DESKTOP_WINDOWS = "desktop_windows"
    DESKTOP_LINUX = "desktop_linux"
    SERVER_LINUX = "server_linux"
    EMBEDDED_ARM = "embedded_arm"
    CLOUD_CONTAINER = "cloud_container"
    DEVELOPMENT = "development"


class ConfigurationLevel(Enum):
    """Levels of configuration."""
    SYSTEM = "system"
    APPLICATION = "application"
    COMPONENT = "component"
    RUNTIME = "runtime"


@dataclass
class SystemRequirements:
    """System requirements for deployment."""
    min_cpu_cores: int = 4
    min_memory_gb: int = 8
    min_storage_gb: int = 25
    gpu_support: List[str] = field(default_factory=lambda: ["nvidia", "amd", "intel"])
    cuda_version: Optional[str] = None
    openvino_version: Optional[str] = "2024.0"
    python_version: str = "3.8+"
    os_compatibility: List[str] = field(default_factory=lambda: ["Windows", "Linux", "macOS"])


@dataclass
class ComponentConfiguration:
    """Configuration for individual components."""
    enabled: bool = True
    max_concurrent_tasks: int = 8
    memory_reservation_mb: int = 256
    threading_mode: str = "async"  # "sync", "async", "threaded"
    performance_target: float = 0.8  # 0.0-1.0
    safety_enabled: bool = True
    telemetry_enabled: bool = True


@dataclass
class DeploymentConfig:
    """Main deployment configuration."""
    # System configuration
    deployment_target: DeploymentTarget = DeploymentTarget.DEVELOPMENT
    system_requirements: SystemRequirements = field(default_factory=SystemRequirements)
    
    # Component configurations
    gpu_pipeline_config: ComponentConfiguration = field(default_factory=ComponentConfiguration)
    memory_coherence_config: ComponentConfiguration = field(default_factory=ComponentConfiguration)
    cross_forex_trading_config: ComponentConfiguration = field(default_factory=ComponentConfiguration)
    signal_processing_config: ComponentConfiguration = field(default_factory=ComponentConfiguration)
    safety_validation_config: ComponentConfiguration = field(default_factory=ComponentConfiguration)
    
    # Performance and optimization
    max_parallel_tasks: int = 16
    memory_alignment_boundary: int = 64
    cache_line_size: int = 64
    enable_gpu_priority_scheduling: bool = True
    enable_memory_compaction: bool = True
    enable_cache_prefetching: bool = True
    cache_prefetch_distance: int = 8
    
    # Resource allocation
    default_trading_capital: float = 1000.0
    max_transaction_size: float = 500.0
    coherence_timeout_ms: float = 1.0
    thermal_threshold_c: float = 85.0
    
    # Integration settings
    enable_uhd_coprocessor: bool = True
    enable_3d_grid_memory: bool = True
    enable_cross_forex_trading: bool = True
    enable_memory_coherence: bool = True
    enable_signal_processing: bool = True
    
    # Logging and telemetry
    log_level: str = "INFO"
    enable_telemetry: bool = True
    telemetry_frequency_hz: float = 10.0
    log_rotation_days: int = 7
    
    # Safety settings
    enable_safety_validation: bool = True
    enable_contract_validation: bool = True
    enable_effect_validation: bool = True
    max_power_draw_w: float = 300.0
    emergency_cooldown_threshold_c: float = 85.0
    
    # Advanced settings
    advanced_gpu_optimizations: bool = True
    enable_dynamic_pricing: bool = True
    trading_history_size: int = 10000
    enable_risk_management: bool = True
    max_risk_percentage: float = 0.1


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self):
        self.config: Optional[DeploymentConfig] = None
        self.config_path: Optional[Path] = None
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> bool:
        """Load configuration from file."""
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Look for config in common locations
            config_locations = [
                "gamesa_config.yaml",
                "gamesa_config.toml",
                "gamesa_config.json",
                "~/.gamesa/config.yaml",
                "./config/gamesa_config.yaml"
            ]
            
            for loc in config_locations:
                path = Path(loc).expanduser()
                if path.exists():
                    self.config_path = path
                    break
        
        if not self.config_path or not self.config_path.exists():
            self.logger.info("No configuration file found, using defaults")
            self.config = DeploymentConfig()
            return True
        
        try:
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                return self._load_yaml_config()
            elif self.config_path.suffix.lower() == '.toml':
                return self._load_toml_config()
            elif self.config_path.suffix.lower() == '.json':
                return self._load_json_config()
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
        
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def _load_yaml_config(self) -> bool:
        """Load YAML configuration."""
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.config = self._merge_config_data(data)
        return True
    
    def _load_toml_config(self) -> bool:
        """Load TOML configuration."""
        with open(self.config_path, 'r') as f:
            data = toml.load(f)
        
        self.config = self._merge_config_data(data)
        return True
    
    def _load_json_config(self) -> bool:
        """Load JSON configuration."""
        with open(self.config_path, 'r') as f:
            data = json.load(f)
        
        self.config = self._merge_config_data(data)
        return True
    
    def _merge_config_data(self, data: Dict[str, Any]) -> DeploymentConfig:
        """Merge loaded configuration data with defaults."""
        config = DeploymentConfig()
        
        # Map data to config fields
        for key, value in data.items():
            if hasattr(config, key):
                if isinstance(getattr(config, key), ComponentConfiguration):
                    # Handle component configuration
                    comp_config = getattr(config, key)
                    if isinstance(value, dict):
                        for comp_key, comp_val in value.items():
                            if hasattr(comp_config, comp_key):
                                setattr(comp_config, comp_key, comp_val)
                else:
                    setattr(config, key, value)
        
        return config
    
    def validate_config(self) -> List[str]:
        """Validate configuration values."""
        errors = []
        
        if not self.config:
            return ["Configuration not loaded"]
        
        # Validate system requirements
        if platform.system() not in self.config.system_requirements.os_compatibility:
            errors.append(f"OS not compatible: {platform.system()}")
        
        # Check memory
        available_memory = psutil.virtual_memory().total / (1024**3)
        if available_memory < self.config.system_requirements.min_memory_gb:
            errors.append(f"Insufficient memory: {available_memory:.1f}GB < {self.config.system_requirements.min_memory_gb}GB")
        
        # Check storage
        storage_free = shutil.disk_usage(".").free / (1024**3)
        if storage_free < self.config.system_requirements.min_storage_gb:
            errors.append(f"Insufficient storage: {storage_free:.1f}GB < {self.config.system_requirements.min_storage_gb}GB")
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count < self.config.system_requirements.min_cpu_cores:
            errors.append(f"Insufficient CPU cores: {cpu_count} < {self.config.system_requirements.min_cpu_cores}")
        
        # Validate component configurations
        for attr_name in dir(self.config):
            attr = getattr(self.config, attr_name)
            if isinstance(attr, ComponentConfiguration):
                if attr.max_concurrent_tasks <= 0:
                    errors.append(f"Invalid max_concurrent_tasks for {attr_name}")
        
        # Validate performance targets
        if not 0.0 <= self.config.memory_coherence_config.performance_target <= 1.0:
            errors.append("Memory coherence performance target must be between 0.0 and 1.0")
        
        if not 0.0 <= self.config.gpu_pipeline_config.performance_target <= 1.0:
            errors.append("GPU pipeline performance target must be between 0.0 and 1.0")
        
        return errors
    
    def get_config(self) -> Optional[DeploymentConfig]:
        """Get current configuration."""
        return self.config


class SystemDeployer:
    """Handles system deployment and installation."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.deployment_path = Path("./deploy")
        self.install_path = Path("./gamesa_install")
    
    def check_system_compatibility(self) -> Dict[str, Any]:
        """Check system compatibility with requirements."""
        hardware_info = {
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'os': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
        }
        
        requirements = self.config.system_requirements
        compatible = True
        issues = []
        
        if hardware_info['cpu_cores'] < requirements.min_cpu_cores:
            compatible = False
            issues.append(f"CPU cores: {hardware_info['cpu_cores']} < {requirements.min_cpu_cores}")
        
        if hardware_info['memory_gb'] < requirements.min_memory_gb:
            compatible = False
            issues.append(f"Memory: {hardware_info['memory_gb']:.1f}GB < {requirements.min_memory_gb}GB")
        
        if hardware_info['os'] not in requirements.os_compatibility:
            compatible = False
            issues.append(f"OS: {hardware_info['os']} not in {requirements.os_compatibility}")
        
        return {
            'compatible': compatible,
            'hardware_info': hardware_info,
            'requirements': requirements,
            'issues': issues
        }
    
    def install_dependencies(self) -> bool:
        """Install system dependencies."""
        try:
            # Check if pip is available
            result = subprocess.run(['pip', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("pip not found")
                return False
            
            # Install required packages
            requirements = [
                'numpy>=1.21.0',
                'psutil>=5.8.0',
                'pyyaml>=6.0',
                'toml>=0.10.0',
                'openvino>=2024.0',
                'pydantic>=2.0',
                'requests>=2.28.0'
            ]
            
            for req in requirements:
                try:
                    # Check if package is already installed
                    subprocess.run(['pip', 'show', req.split('>')[0]], 
                                 capture_output=True, check=True)
                    self.logger.info(f"Package already installed: {req}")
                except subprocess.CalledProcessError:
                    # Install if not found
                    self.logger.info(f"Installing: {req}")
                    subprocess.run(['pip', 'install', req], check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Dependency installation failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Dependency installation error: {e}")
            return False
    
    def configure_environment(self) -> bool:
        """Configure system environment."""
        try:
            env_vars = {
                'GAMESA_CONFIG_PATH': str(self.install_path / 'config'),
                'GAMESA_LOG_LEVEL': self.config.log_level,
                'GAMESA_ENABLE_TELEMETRY': str(self.config.enable_telemetry),
                'GAMESA_MAX_PARALLEL': str(self.config.max_parallel_tasks),
            }
            
            # Create config directory
            (self.install_path / 'config').mkdir(parents=True, exist_ok=True)
            
            # Create environment file
            with open(self.install_path / '.env', 'w') as f:
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            
            self.logger.info("Environment configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment configuration failed: {e}")
            return False
    
    def deploy_system(self) -> bool:
        """Deploy the complete system."""
        self.logger.info(f"Deploying GAMESA/KrystalStack to {self.install_path}")
        
        try:
            # Create installation directory
            self.install_path.mkdir(parents=True, exist_ok=True)
            
            # Check system compatibility
            compat_info = self.check_system_compatibility()
            if not compat_info['compatible']:
                self.logger.error(f"System not compatible: {compat_info['issues']}")
                return False
            
            # Install dependencies
            if not self.install_dependencies():
                return False
            
            # Configure environment
            if not self.configure_environment():
                return False
            
            # Copy configuration to install directory
            if self.config_path:
                shutil.copy(self.config_path, self.install_path / 'config' / 'gamesa_config.yaml')
            
            # Create service files if running as server
            if self.config.deployment_target in [DeploymentTarget.SERVER_LINUX, DeploymentTarget.CLOUD_CONTAINER]:
                self._create_service_files()
            
            self.logger.info(f"System deployed successfully to {self.install_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_service_files(self):
        """Create systemd service files for Linux deployments."""
        service_content = f"""
[Unit]
Description=GAMESA/KrystalStack Service
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'gamesa')}
WorkingDirectory={self.install_path}
ExecStart={sys.executable} -m src.python.main --mode production
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
        """
        
        with open(self.install_path / 'gamesa.service', 'w') as f:
            f.write(service_content)


class RuntimeManager:
    """Manages runtime execution of the system."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.start_time = None
        
        # Initialize logging
        numeric_level = getattr(logging, config.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)
    
    def start(self):
        """Start the runtime system."""
        self.logger.info("Starting GAMESA/KrystalStack runtime...")
        
        # Initialize components based on configuration
        self._initialize_components()
        
        self.start_time = time.time()
        self.is_running = True
        
        self.logger.info("GAMESA/KrystalStack runtime started successfully")
    
    def _initialize_components(self):
        """Initialize components based on configuration."""
        self.logger.info("Initializing components based on configuration...")
        
        # Initialize GPU pipeline if enabled
        if self.config.gpu_pipeline_config.enabled:
            from .functional_layer import GPUGridMemoryManager, GPUCoherenceManager
            from .gamesa_gpu_integration import GAMESAGPUController
            
            self.gpu_controller = GAMESAGPUController()
            self.gpu_controller.initialize()
            self.logger.info("GPU pipeline initialized")
        
        # Initialize cross-forex trading if enabled
        if self.config.cross_forex_trading_config.enabled:
            from .cross_forex_memory_trading import CrossForexManager
            self.cross_forex_manager = CrossForexManager()
            self.logger.info("Cross-forex trading initialized")
        
        # Initialize memory coherence if enabled
        if self.config.memory_coherence_config.enabled:
            from .memory_coherence_protocol import MemoryCoherenceProtocol
            self.memory_coherence = MemoryCoherenceProtocol()
            self.logger.info("Memory coherence protocol initialized")
        
        # Initialize safety validation if enabled
        if self.config.safety_validation_config.enabled:
            from . import create_guardian_checker, create_guardian_validator
            self.guardian_checker = create_guardian_checker()
            self.guardian_validator = create_guardian_validator()
            self.logger.info("Safety validation initialized")
    
    def stop(self):
        """Stop the runtime system."""
        self.logger.info("Stopping GAMESA/KrystalStack runtime...")
        
        self.is_running = False
        
        # Cleanup components
        self._cleanup_components()
        
        self.logger.info("GAMESA/KrystalStack runtime stopped")
    
    def _cleanup_components(self):
        """Cleanup initialized components."""
        if hasattr(self, 'gpu_controller'):
            self.gpu_controller.shutdown()
        
        self.logger.info("Components cleaned up")
    
    def get_status(self) -> Dict[str, Any]:
        """Get runtime status."""
        return {
            'is_running': self.is_running,
            'uptime_seconds': (time.time() - self.start_time) if self.start_time else 0,
            'config_mode': self.config.deployment_target.value,
            'gpu_enabled': self.config.gpu_pipeline_config.enabled,
            'trading_enabled': self.config.cross_forex_trading_config.enabled,
            'coherence_enabled': self.config.memory_coherence_config.enabled,
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('.').percent,
                'process_count': len(psutil.pids())
            }
        }


def create_sample_config(filename: str = "gamesa_config.yaml"):
    """Create a sample configuration file."""
    sample_config = {
        'deployment_target': 'development',
        'system_requirements': {
            'min_cpu_cores': 8,
            'min_memory_gb': 16,
            'min_storage_gb': 50,
            'gpu_support': ['nvidia', 'amd', 'intel'],
            'openvino_version': '2024.0',
            'python_version': '3.9+'
        },
        'gpu_pipeline_config': {
            'enabled': True,
            'max_concurrent_tasks': 16,
            'memory_reservation_mb': 512,
            'threading_mode': 'async',
            'performance_target': 0.95,
            'safety_enabled': True,
            'telemetry_enabled': True
        },
        'memory_coherence_config': {
            'enabled': True,
            'max_concurrent_tasks': 8,
            'memory_reservation_mb': 256,
            'threading_mode': 'async',
            'performance_target': 0.98,
            'safety_enabled': True,
            'telemetry_enabled': True
        },
        'cross_forex_trading_config': {
            'enabled': True,
            'max_concurrent_tasks': 32,
            'memory_reservation_mb': 128,
            'threading_mode': 'async',
            'performance_target': 0.90,
            'safety_enabled': True,
            'telemetry_enabled': True
        },
        'max_parallel_tasks': 32,
        'enable_uhd_coprocessor': True,
        'enable_3d_grid_memory': True,
        'enable_cross_forex_trading': True,
        'enable_memory_coherence': True,
        'default_trading_capital': 5000.0,
        'coherence_timeout_ms': 0.5,
        'thermal_threshold_c': 80.0,
        'log_level': 'INFO',
        'enable_telemetry': True,
        'telemetry_frequency_hz': 60.0,
        'enable_safety_validation': True,
        'max_power_draw_w': 250.0,
        'advanced_gpu_optimizations': True,
        'enable_dynamic_pricing': True,
        'trading_history_size': 50000,
        'enable_risk_management': True,
        'max_risk_percentage': 0.08
    }
    
    with open(filename, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False)
    
    print(f"Sample configuration created: {filename}")


def deploy_system(config_path: Optional[str] = None, target_path: str = "./gamesa_deploy"):
    """Deploy the complete system."""
    print("GAMESA/KrystalStack Deployment Wizard")
    print("=" * 50)
    
    # Load configuration
    config_manager = ConfigManager()
    if not config_manager.load_config(config_path):
        print("Failed to load configuration, using defaults...")
        config_manager.config = DeploymentConfig()
    
    # Validate configuration
    errors = config_manager.validate_config()
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print(f"Configuration loaded successfully with {len(errors)} errors")
    
    # Deploy system
    deployer = SystemDeployer(config_manager.config)
    success = deployer.deploy_system()
    
    if success:
        print("=" * 50)
        print("GAMESA/KrystalStack System Deployed Successfully!")
        print(f"Installation path: {deployer.install_path}")
        print("")
        print("Next Steps:")
        print("  1. Review configuration at gamesa_config.yaml")
        print("  2. Ensure hardware requirements are met")
        print("  3. Run system with: python -m src.python.main --mode production")
        print("  4. Monitor system with: python -m src.python.main --mode benchmark")
        print("=" * 50)
    else:
        print("Deployment failed. Check logs for details.")
    
    return success


def run_system(config_path: Optional[str] = None, mode: str = "demonstration"):
    """Run the system with specified config."""
    from .main import main as main_function
    
    # Set up arguments for main function
    import sys
    sys.argv = [sys.argv[0], f"--mode={mode}"]
    if config_path:
        # Add config path to environment or handle differently
        pass
    
    try:
        main_function()
        return True
    except Exception as e:
        print(f"System execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main deployment and configuration utility."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GAMESA/KrystalStack Configuration and Deployment')
    parser.add_argument('action', choices=['deploy', 'run', 'validate', 'create-config', 'benchmark', 'demo'],
                       help='Action to perform')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--target', type=str, default='./gamesa_deploy', help='Deployment target path')
    parser.add_argument('--mode', type=str, choices=['production', 'demonstration', 'benchmark', 'debug'],
                       default='demonstration', help='Execution mode')
    parser.add_argument('--validate-only', action='store_true', help='Only validate configuration')
    parser.add_argument('--no-deploy', action='store_true', help='Don\'t actually deploy, just test')
    
    args = parser.parse_args()
    
    if args.action == 'create-config':
        create_sample_config()
        return 0
    
    elif args.action == 'validate':
        config_manager = ConfigManager()
        if args.config:
            success = config_manager.load_config(args.config)
        else:
            success = config_manager.load_config()
        
        if success:
            errors = config_manager.validate_config()
            if errors:
                print("Configuration validation failed with errors:")
                for error in errors:
                    print(f"  - {error}")
                return 1
            else:
                print("Configuration validation passed!")
                return 0
        else:
            print("Failed to load configuration")
            return 1
    
    elif args.action == 'deploy':
        success = deploy_system(args.config, args.target)
        return 0 if success else 1
    
    elif args.action == 'run':
        success = run_system(args.config, args.mode)
        return 0 if success else 1
    
    elif args.action == 'demo':
        print("GAMESA/KrystalStack - Complete System Demo")
        print("=" * 60)
        
        config_manager = ConfigManager()
        if args.config:
            config_manager.load_config(args.config)
        else:
            config_manager.config = DeploymentConfig()
        
        # Run a quick integration test
        from .integration_test import run_integration_tests
        success = run_integration_tests()
        
        if success:
            print("\n✓ Demo completed successfully!")
            print("The complete GAMESA/KrystalStack system is working properly.")
            print("Features tested:")
            print("  - GPU Pipeline Integration")
            print("  - 3D Grid Memory System") 
            print("  - Cross-forex Trading")
            print("  - Memory Coherence Protocol")
            print("  - Signal Processing")
            print("  - Safety Validation")
            print("  - Performance Optimization")
        else:
            print("\n✗ Demo failed - system components not working properly")
        
        return 0 if success else 1
    
    elif args.action == 'benchmark':
        print("GAMESA/KrystalStack - Performance Benchmark Suite")
        print("=" * 60)
        
        from .integration_test import run_performance_benchmarks
        run_performance_benchmarks()
        
        print("\nBenchmark suite completed!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    print(f"\nGAMESA/KrystalStack configuration utility completed with code: {exit_code}")
    exit(exit_code)