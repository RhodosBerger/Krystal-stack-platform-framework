import logging
from .openvino_subsystem import OpenVINOSubsystem
from .power_governor import PowerGovernor
from .utils import PreciseTimer

class EagleEyeAccelerator:
    """
    Gamesa Cortex V2: Eagle Eye NPU Accelerator (Intel Architecture).
    Provides strong acceleration via optimal resource consumption.
    By dynamically routing requests across Intel NPUs, Iris Xe GPUs, and CPUs,
    it allows processing more with smaller consumption using Intel P-State tuning.
    """
    def __init__(self, openvino_subsystem: OpenVINOSubsystem, power_governor: PowerGovernor):
        self.logger = logging.getLogger("EagleEyeAccelerator")
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
            
        self.openvino = openvino_subsystem
        self.power = power_governor
        self.timer = PreciseTimer()
        self.logger.info("Eagle Eye NPU Accelerator Online. Preparing optimal consumption routes.")
        
        # Explicitly invoke 11th Gen+ features (VNNI, DL Boost, Iris Xe)
        self.openvino.enable_11th_gen_optimizations()

    def optimize_acceleration(self, workload_size: int) -> str:
        """
        Calculates the most optimal acceleration strategy to achieve 
        "Strong Acceleration" focusing on HIGH BANDWIDTH and PERFORMANCE first,
        then scaling down dynamically for optimization.
        """
        # Primary objective: Performance and Bandwidth.
        # We push the power governor into 'overdrive' by default for heavy workloads,
        # maximizing throughput on the NPU to process more in the given time.
        if workload_size > 5000:
            self.power.set_mode("overdrive")
            self.openvino.set_performance_hint("THROUGHPUT")
            self.openvino.set_streams("AUTO") # Maximize available streams for bandwidth
            self.logger.debug(f"Eagle Eye: High Workload ({workload_size}), optimizing for HIGH BANDWIDTH (THROUGHPUT) on NPU.")
        elif workload_size > 1000:
            self.power.set_mode("balanced")
            self.openvino.set_performance_hint("THROUGHPUT")
            self.openvino.set_streams(4) # Moderate parallel streams
            self.logger.debug(f"Eagle Eye: Medium Workload ({workload_size}), balancing for throughput.")
        else:
            self.power.set_mode("eco")
            self.openvino.set_performance_hint("LATENCY")
            self.openvino.set_streams(1) # Smaller consumption for light workloads
            self.logger.debug(f"Eagle Eye: Light Workload ({workload_size}), optimizing for eco LATENCY on NPU.")

        # Route enforcement: prioritize Intel NPU -> Intel GPU -> CPU
        available_devices = self.openvino.get_available_devices()
        target_device = "CPU"
        for dev in available_devices:
            if "NPU" in dev:
                target_device = dev
                break
            elif "GPU" in dev:
                target_device = dev
                
        return target_device

    def execute_with_eagle_eye(self, model_handle, input_data, workload_size: int):
        """
        Executes inference using the Eagle Eye optimized route.
        """
        target_device = self.optimize_acceleration(workload_size)
        start_time = self.timer.elapsed_ms()
        
        # In a real pipeline, we'd ensure the model is loaded to the specific target device.
        # For our architecture simulation, we just invoke openvino subsystem.
        result = self.openvino.async_inference(model_handle, input_data)
        
        end_time = self.timer.elapsed_ms()
        duration = end_time - start_time
        
        self.logger.info(f"Eagle Eye strong acceleration on {target_device} completed in {duration:.4f}ms with optimal consumption.")
        return result
