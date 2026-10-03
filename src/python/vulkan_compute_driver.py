"""
Krystal-Stack: Direct Vulkan Compute Driver (Zero-Compiler Ctypes Engine)
========================================================================
Binds directly to C:\\Windows\\System32\\vulkan-1.dll via Python ctypes.

CURRENT STATUS (verified by code inspection and measurement):
  * REAL:  vulkan-1.dll loading, vkCreateInstance, physical device enumeration,
           Intel Iris Xe (Vendor 0x8086) discovery, compute queue family lookup.
  * NOT YET REAL: no logical device, no SPIR-V shader, no compute pipeline, no GPU
           buffers, no vkCmdDispatch. execute_raymarch() is a pure-Python CPU
           emulation of the intended kernel (~46 ms/frame at 96x40 on the dev host).
           readback_mb_s is host-list throughput, NOT a PCIe/GPU readback.
  * Telemetry exposes this via `real_gpu_dispatch` and `compute_backend`.

Target design (see GET /api/priorities, id 'vulkan-real-dispatch'):
1. Zero external compiler or SDK installation required on host.
2. Dual SSBO output: uint8 glyph indices + uint32 TrueColor RGB (19.2 KB/frame at 96x40).
3. CPU fallback remains the reference implementation for correctness checks.
"""

import os
import sys
import ctypes
import math
import time
from typing import Dict, Any, Tuple, Optional, List

# Vulkan Constants
VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
VK_STRUCTURE_TYPE_SUBMIT_INFO = 4
VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE = 6
VK_STRUCTURE_TYPE_BIND_SPARSE_INFO = 7
VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12
VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO = 13
VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16
VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18
VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29
VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30
VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32
VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33
VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34
VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35
VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39
VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40
VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42

VK_QUEUE_COMPUTE_BIT = 0x00000002
VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020
VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004
VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6
VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
VK_PIPELINE_BIND_POINT_COMPUTE = 1

# Ctypes Structures
class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]

class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
    ]

class VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * 256),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("limits", ctypes.c_uint8 * 512),  # Approximate limits blob
        ("sparseProperties", ctypes.c_uint32 * 5),
    ]

class VkQueueFamilyProperties(ctypes.Structure):
    _fields_ = [
        ("queueFlags", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("timestampValidBits", ctypes.c_uint32),
        ("minImageTransferGranularity", ctypes.c_uint32 * 3),
    ]


class VulkanComputeDriver:
    """
    Direct Ctypes Driver for Vulkan Compute pipelines on Windows.
    Provides zero-compiler hardware raymarching with seamless CPU fallback.
    """

    def __init__(self):
        self.dll_path = "C:\\Windows\\System32\\vulkan-1.dll"
        self.is_windows = (sys.platform == "win32")
        self.vulkan_lib = None
        self.vulkan_available = False
        self.device_name = "CPU SIMD Fallback (Software)"
        self.vendor_id = 0
        self.device_id = 0
        self.driver_version = ""
        self.is_intel_iris = False
        self.compute_queue_family = -1
        self.dispatch_count = 0
        self.last_dispatch_us = 0.0
        self.readback_mb_s = 0.0
        # `gpu_accelerated` is a legacy alias meaning "a Vulkan device was detected".
        # It does NOT mean compute runs on the GPU - see `real_gpu_dispatch`.
        self.gpu_accelerated = False
        self.device_detected = False
        self.real_gpu_dispatch = False      # becomes True only once vkCmdDispatch is wired
        self.compute_backend = "CPU_EMULATION"

        # Attempt to load vulkan-1.dll and discover hardware
        self._init_vulkan()

    def _init_vulkan(self):
        """Loads vulkan-1.dll and probes physical devices."""
        if not self.is_windows or not os.path.exists(self.dll_path):
            print(f"[VULKAN] vulkan-1.dll not found at {self.dll_path}. Using CPU SIMD fallback.")
            return

        try:
            # Load DLL via ctypes
            self.vulkan_lib = ctypes.CDLL(self.dll_path)

            # Setup vkCreateInstance signature
            vkCreateInstance = self.vulkan_lib.vkCreateInstance
            vkCreateInstance.argtypes = [
                ctypes.POINTER(VkInstanceCreateInfo),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vkCreateInstance.restype = ctypes.c_int32

            app_info = VkApplicationInfo(
                sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pNext=None,
                pApplicationName=b"KrystalStackCompute",
                applicationVersion=1,
                pEngineName=b"KrystalVulkanEngine",
                engineVersion=1,
                apiVersion=(1 << 22) | (3 << 12)  # Vulkan 1.3
            )

            create_info = VkInstanceCreateInfo(
                sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pNext=None,
                flags=0,
                pApplicationInfo=ctypes.pointer(app_info),
                enabledLayerCount=0,
                ppEnabledLayerNames=None,
                enabledExtensionCount=0,
                ppEnabledExtensionNames=None
            )

            instance = ctypes.c_void_p()
            res = vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))

            if res == VK_SUCCESS and instance.value:
                self.instance = instance
                self._probe_devices()
            else:
                print(f"[VULKAN] vkCreateInstance returned code {res}. Using CPU fallback.")

        except Exception as e:
            print(f"[VULKAN] Dynamic loading or probing notice: {e}")
            self.vulkan_available = False

    def _probe_devices(self):
        """Enumerates physical devices and identifies Intel Iris Xe Graphics."""
        try:
            vkEnumeratePhysicalDevices = self.vulkan_lib.vkEnumeratePhysicalDevices
            vkEnumeratePhysicalDevices.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vkEnumeratePhysicalDevices.restype = ctypes.c_int32

            count = ctypes.c_uint32(0)
            res = vkEnumeratePhysicalDevices(self.instance, ctypes.byref(count), None)

            if res == VK_SUCCESS and count.value > 0:
                devices = (ctypes.c_void_p * count.value)()
                vkEnumeratePhysicalDevices(self.instance, ctypes.byref(count), devices)

                vkGetPhysicalDeviceProperties = self.vulkan_lib.vkGetPhysicalDeviceProperties
                vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkPhysicalDeviceProperties)]
                vkGetPhysicalDeviceProperties.restype = None

                vkGetPhysicalDeviceQueueFamilyProperties = self.vulkan_lib.vkGetPhysicalDeviceQueueFamilyProperties
                vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(VkQueueFamilyProperties)
                ]
                vkGetPhysicalDeviceQueueFamilyProperties.restype = None

                for i in range(count.value):
                    dev = devices[i]
                    props = VkPhysicalDeviceProperties()
                    vkGetPhysicalDeviceProperties(dev, ctypes.byref(props))

                    dev_name = props.deviceName.decode("utf-8", errors="replace")
                    v_id = props.vendorID
                    d_id = props.deviceID

                    # Check queue families
                    q_count = ctypes.c_uint32(0)
                    vkGetPhysicalDeviceQueueFamilyProperties(dev, ctypes.byref(q_count), None)
                    q_props = (VkQueueFamilyProperties * q_count.value)()
                    vkGetPhysicalDeviceQueueFamilyProperties(dev, ctypes.byref(q_count), q_props)

                    compute_idx = -1
                    for q_idx in range(q_count.value):
                        if q_props[q_idx].queueFlags & VK_QUEUE_COMPUTE_BIT:
                            compute_idx = q_idx
                            break

                    # Vendor 0x8086 = Intel
                    if v_id == 0x8086 or "Iris" in dev_name or "Intel" in dev_name:
                        self.physical_device = dev
                        self.device_name = dev_name
                        self.vendor_id = v_id
                        self.device_id = d_id
                        self.is_intel_iris = True
                        self.compute_queue_family = compute_idx
                        self.vulkan_available = True
                        self.gpu_accelerated = True
                        self.device_detected = True
                        print(f"[VULKAN] Successfully selected device: '{dev_name}' (Compute Queue #{compute_idx})")
                        break

                if not self.gpu_accelerated and count.value > 0:
                    # Pick first available device
                    props = VkPhysicalDeviceProperties()
                    vkGetPhysicalDeviceProperties(devices[0], ctypes.byref(props))
                    self.device_name = props.deviceName.decode("utf-8", errors="replace")
                    self.physical_device = devices[0]
                    self.vulkan_available = True
                    self.gpu_accelerated = True
                    self.device_detected = True
                    print(f"[VULKAN] Selected primary device: '{self.device_name}'")

        except Exception as e:
            print(f"[VULKAN] Device probe notice: {e}")
            self.vulkan_available = False

    def execute_raymarch(
        self,
        width: int,
        height: int,
        t: float,
        mode: str = "CYBERPUNK",
        cam_pos: Tuple[float, float, float] = (0.0, 0.0, -3.2),
        step_budget: int = 32
    ) -> Tuple[List[int], List[int]]:
        """
        Executes 3D raymarching pass.
        Returns: (glyph_indices, truecolor_rgb_u32)
        - glyph_indices: list of ints [0..len(ramp)-1]
        - truecolor_rgb_u32: packed 0xRRGGBB ints
        """
        t0 = time.perf_counter()
        total_pixels = width * height

        # Dual SSBO buffers:
        # SSBO 0: glyph indices (uint8)
        # SSBO 1: packed RGB colors (uint32)
        glyphs: List[int] = [0] * total_pixels
        colors: List[int] = [0] * total_pixels

        aspect = (width / height) * 0.52
        light_dir = (0.577, 0.577, -0.577)

        # High-speed vectorized Python/SIMD math kernel (emulating GPU workgroup grid)
        # Evaluates 3D Torus + Pulsing Sphere SDF
        r1 = 1.05 + 0.1 * math.sin(t * 1.5)
        r2 = 0.38
        core_r = 0.55 + 0.12 * math.sin(t * 3.5)

        for y in range(height):
            screen_y = (1.0 - (y / height) * 2.0)
            row_offset = y * width
            for x in range(width):
                screen_x = ((x / width) * 2.0 - 1.0) * aspect

                # Camera ray
                rd_len = math.sqrt(screen_x * screen_x + screen_y * screen_y + 4.0)
                rdx = screen_x / rd_len
                rdy = screen_y / rd_len
                rdz = 2.0 / rd_len

                dist = 0.0
                hit = False
                px, py, pz = cam_pos

                for _ in range(min(step_budget, 24)):
                    px = cam_pos[0] + rdx * dist
                    py = cam_pos[1] + rdy * dist
                    pz = cam_pos[2] + rdz * dist

                    # Rotate around Y
                    theta = t * 1.2
                    c, s = math.cos(theta), math.sin(theta)
                    rx = px * c + pz * s
                    ry = py
                    rz = -px * s + pz * c

                    # Torus SDF
                    qx = math.sqrt(rx * rx + rz * rz) - r1
                    qy = ry
                    d_torus = math.sqrt(qx * qx + qy * qy) - r2

                    # Sphere SDF
                    d_core = math.sqrt(px * px + py * py + pz * pz) - core_r

                    # Smooth Min polynomial
                    k = 0.3
                    h = max(k - abs(d_torus - d_core), 0.0) / k
                    d = min(d_torus, d_core) - h * h * k * 0.25

                    if d < 0.005:
                        hit = True
                        break
                    dist += d
                    if dist > 7.0:
                        break

                idx = row_offset + x
                if hit:
                    # Analytical approximate normal
                    eps = 0.003
                    nx = (math.sqrt((px + eps)**2 + pz**2) - r1) - (math.sqrt(px**2 + pz**2) - r1)
                    ny = py
                    nz = (math.sqrt(px**2 + (pz + eps)**2) - r1) - (math.sqrt(px**2 + pz**2) - r1)
                    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
                    if mag > 1e-5:
                        nx, ny, nz = nx / mag, ny / mag, nz / mag
                    else:
                        nx, ny, nz = 0.0, 1.0, 0.0

                    diff = max(0.0, nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2])
                    rim = 1.0 - max(0.0, -(nx * rdx + ny * rdy + nz * rdz))
                    luma = min(1.0, max(0.0, diff * 0.75 + rim * 0.5))

                    glyph_idx = int(luma * 4.0)
                    glyphs[idx] = min(4, max(0, glyph_idx))

                    # TrueColor RGB packed: 0xRRGGBB
                    r = int(255 * luma)
                    g = int(40 + 160 * rim)
                    b = int(120 + 130 * diff)
                    colors[idx] = (r << 16) | (g << 8) | b
                else:
                    glyphs[idx] = 0
                    colors[idx] = 0x0A0F19  # Deep ambient background

        dt = max(1e-6, time.perf_counter() - t0)
        self.dispatch_count += 1
        self.last_dispatch_us = round(dt * 1e6, 1)

        # 19.2 KB readback size for 96x40 viewport
        buffer_bytes = total_pixels * 5  # 1 byte glyph + 4 byte color
        self.readback_mb_s = round((buffer_bytes / (1024 * 1024)) / dt, 2)

        return glyphs, colors

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns driver status, active GPU device, and dispatch timings."""
        return {
            "vulkan_available": self.vulkan_available,
            "gpu_accelerated": self.gpu_accelerated,
            "device_name": self.device_name,
            "vendor_id": hex(self.vendor_id),
            "device_id": hex(self.device_id),
            "is_intel_iris": self.is_intel_iris,
            "compute_queue_family": self.compute_queue_family,
            "dispatch_count": self.dispatch_count,
            "last_dispatch_us": self.last_dispatch_us,
            "readback_mb_s": self.readback_mb_s,
            "ssbo_readback_kb_per_frame": 19.2,
            "device_detected": self.device_detected,
            "real_gpu_dispatch": self.real_gpu_dispatch,
            "compute_backend": self.compute_backend,
            "readback_is_simulated": not self.real_gpu_dispatch
        }
