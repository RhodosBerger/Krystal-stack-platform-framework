import platform
import shutil
import logging
import os

class RuntimeSelector:
    """
    Gamesa Cortex V2: Runtime Selector.
    Maps Hardware to Best Runtimes.
    """
    def __init__(self):
        self.logger = logging.getLogger("RuntimeSelector")
        self.system_info = self._probe_system()

    def _probe_system(self):
        return {
            "os": platform.system(),
            "arch": platform.machine(),
            "has_rust": shutil.which("cargo") is not None,
            "has_vulkan": shutil.which("vulkaninfo") is not None,
            "has_opencl": os.path.exists("/usr/lib/libOpenCL.so") # Simplified check
        }

    def select_best_configuration(self):
        config = {
            "control_plane": "Python",
            "planning": "Python (Fallback)",
            "compute": "Python (Fallback)",
            "ai": "Heuristic (Fallback)"
        }

        # 1. Planning Layer
        if self.system_info["has_rust"]:
            config["planning"] = "Rust (Native)"
        
        # 2. Compute Layer
        if self.system_info["has_vulkan"]:
            config["compute"] = "Vulkan (GPU)"
        elif self.system_info["has_opencl"]:
             config["compute"] = "OpenCL (Accelerated)"
             
        # 3. AI Layer (Heuristic based on arch)
        if "bit" in self.system_info["arch"]: # 64-bit assume capable
             config["ai"] = "OpenVINO/ONNX"
             
        self.logger.info(f"Selected Configuration: {config}")
        return config
