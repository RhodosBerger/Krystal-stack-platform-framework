"""Krystal Kernel - a hardware-aware, user-space compute kernel (stdlib only).

It is *not* an OS kernel. It is a scheduler + pipeline runtime that routes named compute
kernels to execution lanes (threads, worker processes, optional OpenVINO accelerator),
enforces admission/back-pressure/deadlines, learns routing from its own event log, and
heals itself when a lane or backend misbehaves.

    from krystal_kernel import get_kernel
    k = get_kernel()
    fut = k.submit("hash_embed", {"texts": ["hello"]}, kind="embed", cost=1, deadline_ms=200)
    fut.result()
"""
from .hwprofile import HardwareProfile, detect_profile, calibrate  # noqa: F401
from .params import KernelParams, derive_params  # noqa: F401
from .eventlog import EventLog  # noqa: F401
from .kernel import ComputeKernel, Backpressure, DeadlineMissed, get_kernel, shutdown_kernel  # noqa: F401

__all__ = [
    "HardwareProfile", "detect_profile", "calibrate", "KernelParams", "derive_params",
    "EventLog", "ComputeKernel", "Backpressure", "DeadlineMissed", "get_kernel", "shutdown_kernel",
]
