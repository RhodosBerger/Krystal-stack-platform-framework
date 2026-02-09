import os
import platform
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Any

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


@dataclass
class TelemetrySnapshot:
    event_id: str
    timestamp: float
    cpu_percent: float
    memory_percent: float
    process_count: int
    cpu_count: int
    load_1m: float
    disk_percent: float
    net_bytes_sent: int
    net_bytes_recv: int
    platform_system: str
    platform_release: str
    is_windows: bool
    openvino_device_hint: str
    oneapi_threads: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "process_count": self.process_count,
            "cpu_count": self.cpu_count,
            "load_1m": self.load_1m,
            "disk_percent": self.disk_percent,
            "net_bytes_sent": self.net_bytes_sent,
            "net_bytes_recv": self.net_bytes_recv,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "is_windows": self.is_windows,
            "openvino_device_hint": self.openvino_device_hint,
            "oneapi_threads": self.oneapi_threads,
        }


class TelemetryCollector:
    def sample(self) -> TelemetrySnapshot:
        load_1m = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        cpu_count = max(1, os.cpu_count() or 1)
        net_bytes_sent = 0
        net_bytes_recv = 0
        disk_percent = 0.0

        if psutil is not None:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            process_count = len(psutil.pids())
            disk_percent = psutil.disk_usage("/").percent
            net = psutil.net_io_counters()
            net_bytes_sent = int(net.bytes_sent)
            net_bytes_recv = int(net.bytes_recv)
        else:
            # Lightweight fallback for environments without psutil.
            cpu_percent = max(0.0, min(100.0, load_1m * 100.0 / cpu_count))
            memory_percent = 0.0
            process_count = 0

        return TelemetrySnapshot(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            process_count=process_count,
            cpu_count=cpu_count,
            load_1m=load_1m,
            disk_percent=disk_percent,
            net_bytes_sent=net_bytes_sent,
            net_bytes_recv=net_bytes_recv,
            platform_system=platform.system(),
            platform_release=platform.release(),
            is_windows=platform.system().lower() == "windows",
            openvino_device_hint=os.getenv("OPENVINO_DEVICE", "AUTO"),
            oneapi_threads=os.getenv("ONEAPI_NUM_THREADS", "auto"),
        )
