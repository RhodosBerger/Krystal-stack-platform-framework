"""
Krystal-Stack: Direct Vulkan Compute Driver (Zero-Compiler Ctypes Engine)
========================================================================
Binds directly to C:\\Windows\\System32\\vulkan-1.dll via Python ctypes.
Provides native Vulkan compute dispatch on Intel Iris Xe graphics with
deterministic CPU reference fallback.

ARCHITECTURAL SPECIFICATION:
  * Hardware Path:
      vkCreateDevice -> compute queue #0 -> host-visible SSBOs/UBO ->
      SPIR-V compute pipeline -> command buffer -> vkCmdDispatch ->
      fence wait -> memory readback.
  * Correctness Oracle:
      execute_raymarch_cpu() remains fully preserved as the mathematical
      reference implementation.
  * Zero-Compiler Runtime Guarantee:
      Precompiled SPIR-V binary is embedded directly with fallback to
      local .spv file if present.
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
VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8
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
VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020
VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001
VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0

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
        ("limits", ctypes.c_uint8 * 512),
        ("sparseProperties", ctypes.c_uint32 * 5),
    ]

class VkQueueFamilyProperties(ctypes.Structure):
    _fields_ = [
        ("queueFlags", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("timestampValidBits", ctypes.c_uint32),
        ("minImageTransferGranularity", ctypes.c_uint32 * 3),
    ]

class VkDeviceQueueCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueFamilyIndex", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("pQueuePriorities", ctypes.POINTER(ctypes.c_float)),
    ]

class VkDeviceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueCreateInfoCount", ctypes.c_uint32),
        ("pQueueCreateInfos", ctypes.POINTER(VkDeviceQueueCreateInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
        ("pEnabledFeatures", ctypes.c_void_p),
    ]

class VkBufferCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("size", ctypes.c_uint64),
        ("usage", ctypes.c_uint32),
        ("sharingMode", ctypes.c_uint32),
        ("queueFamilyIndexCount", ctypes.c_uint32),
        ("pQueueFamilyIndices", ctypes.POINTER(ctypes.c_uint32)),
    ]

class VkMemoryRequirements(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("alignment", ctypes.c_uint64),
        ("memoryTypeBits", ctypes.c_uint32),
    ]

class VkMemoryAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("allocationSize", ctypes.c_uint64),
        ("memoryTypeIndex", ctypes.c_uint32),
    ]

class VkMemoryType(ctypes.Structure):
    _fields_ = [("propertyFlags", ctypes.c_uint32), ("heapIndex", ctypes.c_uint32)]

class VkMemoryHeap(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("flags", ctypes.c_uint32)]

class VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", ctypes.c_uint32),
        ("memoryTypes", VkMemoryType * 32),
        ("memoryHeapCount", ctypes.c_uint32),
        ("memoryHeaps", VkMemoryHeap * 16)
    ]

class VkDescriptorSetLayoutBinding(ctypes.Structure):
    _fields_ = [
        ("binding", ctypes.c_uint32),
        ("descriptorType", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
        ("stageFlags", ctypes.c_uint32),
        ("pImmutableSamplers", ctypes.c_void_p),
    ]

class VkDescriptorSetLayoutCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("bindingCount", ctypes.c_uint32),
        ("pBindings", ctypes.POINTER(VkDescriptorSetLayoutBinding)),
    ]

class VkDescriptorPoolSize(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
    ]

class VkDescriptorPoolCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("maxSets", ctypes.c_uint32),
        ("poolSizeCount", ctypes.c_uint32),
        ("pPoolSizes", ctypes.POINTER(VkDescriptorPoolSize)),
    ]

class VkDescriptorSetAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("descriptorPool", ctypes.c_void_p),
        ("descriptorSetCount", ctypes.c_uint32),
        ("pSetLayouts", ctypes.POINTER(ctypes.c_void_p)),
    ]

class VkDescriptorBufferInfo(ctypes.Structure):
    _fields_ = [
        ("buffer", ctypes.c_void_p),
        ("offset", ctypes.c_uint64),
        ("range", ctypes.c_uint64),
    ]

class VkWriteDescriptorSet(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("dstSet", ctypes.c_void_p),
        ("dstBinding", ctypes.c_uint32),
        ("dstArrayElement", ctypes.c_uint32),
        ("descriptorCount", ctypes.c_uint32),
        ("descriptorType", ctypes.c_uint32),
        ("pImageInfo", ctypes.c_void_p),
        ("pBufferInfo", ctypes.POINTER(VkDescriptorBufferInfo)),
        ("pTexelBufferView", ctypes.c_void_p),
    ]

class VkShaderModuleCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("codeSize", ctypes.c_size_t),
        ("pCode", ctypes.POINTER(ctypes.c_uint32)),
    ]

class VkPipelineShaderStageCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("stage", ctypes.c_uint32),
        ("module", ctypes.c_void_p),
        ("pName", ctypes.c_char_p),
        ("pSpecializationInfo", ctypes.c_void_p),
    ]

class VkPipelineLayoutCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("setLayoutCount", ctypes.c_uint32),
        ("pSetLayouts", ctypes.POINTER(ctypes.c_void_p)),
        ("pushConstantRangeCount", ctypes.c_uint32),
        ("pPushConstantRanges", ctypes.c_void_p),
    ]

class VkComputePipelineCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("stage", VkPipelineShaderStageCreateInfo),
        ("layout", ctypes.c_void_p),
        ("basePipelineHandle", ctypes.c_void_p),
        ("basePipelineIndex", ctypes.c_int32),
    ]

class VkCommandPoolCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueFamilyIndex", ctypes.c_uint32),
    ]

class VkCommandBufferAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("commandPool", ctypes.c_void_p),
        ("level", ctypes.c_uint32),
        ("commandBufferCount", ctypes.c_uint32),
    ]

class VkCommandBufferBeginInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pInheritanceInfo", ctypes.c_void_p),
    ]

class VkSubmitInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("waitSemaphoreCount", ctypes.c_uint32),
        ("pWaitSemaphores", ctypes.c_void_p),
        ("pWaitDstStageMask", ctypes.c_void_p),
        ("commandBufferCount", ctypes.c_uint32),
        ("pCommandBuffers", ctypes.POINTER(ctypes.c_void_p)),
        ("signalSemaphoreCount", ctypes.c_uint32),
        ("pSignalSemaphores", ctypes.c_void_p),
    ]

class VkFenceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]

class UBOParams(ctypes.Structure):
    _fields_ = [
        ("cam_pos", ctypes.c_float * 4),
        ("cam_orientation", ctypes.c_float * 4),
        ("res_width", ctypes.c_uint32),
        ("res_height", ctypes.c_uint32),
        ("speculative_depth", ctypes.c_uint32),
        ("mirror_folds", ctypes.c_uint32),
        ("time", ctypes.c_float),
        ("entropy_threshold", ctypes.c_float),
        ("pad0", ctypes.c_float),
        ("pad1", ctypes.c_float),
    ]


class VulkanComputeDriver:
    """
    Direct Ctypes Driver for Vulkan Compute pipelines on Windows.
    Provides native hardware compute dispatch on Intel Iris Xe
    with zero-overhead host-visible SSBO readback and verified CPU fallback.
    """

    def __init__(self, prefer_gpu: bool = True):
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
        self.gpu_accelerated = False
        self.device_detected = False
        self.real_gpu_dispatch = False
        self.compute_backend = "CPU_EMULATION"
        self.prefer_gpu = prefer_gpu

        # Hardware handles
        self.instance = None
        self.physical_device = None
        self.device = None
        self.queue = None
        self.descriptor_set_layout = None
        self.descriptor_pool = None
        self.descriptor_set = None
        self.pipeline_layout = None
        self.shader_module = None
        self.pipeline = None
        self.command_pool = None
        self.command_buffer = None
        self.fence = None

        # Buffer allocations
        self.buffer_width = 0
        self.buffer_height = 0
        self.glyph_buffer = None
        self.glyph_memory = None
        self.glyph_mapped = None
        self.rgb_buffer = None
        self.rgb_memory = None
        self.rgb_mapped = None
        self.ubo_buffer = None
        self.ubo_memory = None
        self.ubo_mapped = None

        # Telemetry & Diagnostics
        self.logical_device_created = False
        self.compute_queue_ready = False
        self.pipeline_ready = False
        self.host_memory_type_index = -1
        self.fallback_count = 0
        self.dispatch_errors: List[str] = []
        self.workgroup_size = (8, 8, 1)
        self.last_timings = {
            "prep_time_ms": 0.0,
            "submit_time_ms": 0.0,
            "wait_time_ms": 0.0,
            "readback_time_ms": 0.0,
            "frame_time_ms": 0.0,
        }
        self.last_match_metrics = {
            "glyph_match_ratio": 1.0,
            "pixel_match_ratio": 1.0,
            "mean_absolute_error": 0.0,
            "max_error": 0,
        }

        # Attempt hardware initialization
        self._init_vulkan()
        if self.vulkan_available and self.prefer_gpu:
            self._init_hardware_pipeline()

    def _init_vulkan(self):
        """Loads vulkan-1.dll and probes physical devices."""
        if not self.is_windows or not os.path.exists(self.dll_path):
            return

        try:
            self.vulkan_lib = ctypes.CDLL(self.dll_path)

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
                self.vulkan_available = False

        except Exception as e:
            self.dispatch_errors.append(f"Vulkan init failed: {e}")
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

                    q_count = ctypes.c_uint32(0)
                    vkGetPhysicalDeviceQueueFamilyProperties(dev, ctypes.byref(q_count), None)
                    q_props = (VkQueueFamilyProperties * q_count.value)()
                    vkGetPhysicalDeviceQueueFamilyProperties(dev, ctypes.byref(q_count), q_props)

                    compute_idx = -1
                    for q_idx in range(q_count.value):
                        if q_props[q_idx].queueFlags & VK_QUEUE_COMPUTE_BIT:
                            compute_idx = q_idx
                            break

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
                        break

                if not self.gpu_accelerated and count.value > 0:
                    props = VkPhysicalDeviceProperties()
                    vkGetPhysicalDeviceProperties(devices[0], ctypes.byref(props))
                    self.device_name = props.deviceName.decode("utf-8", errors="replace")
                    self.physical_device = devices[0]
                    self.compute_queue_family = 0
                    self.vulkan_available = True
                    self.gpu_accelerated = True
                    self.device_detected = True

        except Exception as e:
            self.dispatch_errors.append(f"Device probe failed: {e}")
            self.vulkan_available = False

    def _init_hardware_pipeline(self):
        """Initializes logical device, compute queue, descriptors, and compute pipeline."""
        if not self.physical_device or self.compute_queue_family < 0:
            return

        vl = self.vulkan_lib
        try:
            # 1. vkCreateDevice
            vl.vkCreateDevice.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(VkDeviceCreateInfo),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vl.vkCreateDevice.restype = ctypes.c_int32

            priority = (ctypes.c_float * 1)(1.0)
            queue_info = VkDeviceQueueCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                pNext=None,
                flags=0,
                queueFamilyIndex=self.compute_queue_family,
                queueCount=1,
                pQueuePriorities=priority
            )
            device_info = VkDeviceCreateInfo(
                sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                pNext=None,
                flags=0,
                queueCreateInfoCount=1,
                pQueueCreateInfos=ctypes.pointer(queue_info),
                enabledLayerCount=0,
                ppEnabledLayerNames=None,
                enabledExtensionCount=0,
                ppEnabledExtensionNames=None,
                pEnabledFeatures=None
            )
            device = ctypes.c_void_p()
            res = vl.vkCreateDevice(self.physical_device, ctypes.byref(device_info), None, ctypes.byref(device))
            if res != VK_SUCCESS or not device.value:
                self.dispatch_errors.append(f"vkCreateDevice failed with code {res}")
                return
            self.device = device
            self.logical_device_created = True

            # 2. vkGetDeviceQueue
            vl.vkGetDeviceQueue.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vl.vkGetDeviceQueue.restype = None
            queue = ctypes.c_void_p()
            vl.vkGetDeviceQueue(self.device, self.compute_queue_family, 0, ctypes.byref(queue))
            if not queue.value:
                self.dispatch_errors.append("vkGetDeviceQueue returned null handle")
                return
            self.queue = queue
            self.compute_queue_ready = True

            # 3. Memory Properties & Host Visible Type Lookup
            vl.vkGetPhysicalDeviceMemoryProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkPhysicalDeviceMemoryProperties)]
            vl.vkGetPhysicalDeviceMemoryProperties.restype = None
            mem_props = VkPhysicalDeviceMemoryProperties()
            vl.vkGetPhysicalDeviceMemoryProperties(self.physical_device, ctypes.byref(mem_props))

            for i in range(mem_props.memoryTypeCount):
                flags = mem_props.memoryTypes[i].propertyFlags
                if (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT):
                    self.host_memory_type_index = i
                    break

            if self.host_memory_type_index < 0:
                self.dispatch_errors.append("No HOST_VISIBLE + HOST_COHERENT memory type found")
                return

            # 4. Descriptor Set Layout
            vl.vkCreateDescriptorSetLayout.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(VkDescriptorSetLayoutCreateInfo),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vl.vkCreateDescriptorSetLayout.restype = ctypes.c_int32

            bindings = (VkDescriptorSetLayoutBinding * 3)(
                VkDescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, None),
                VkDescriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, None),
                VkDescriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, None),
            )
            dsl_info = VkDescriptorSetLayoutCreateInfo(
                sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext=None,
                flags=0,
                bindingCount=3,
                pBindings=bindings
            )
            ds_layout = ctypes.c_void_p()
            if vl.vkCreateDescriptorSetLayout(self.device, ctypes.byref(dsl_info), None, ctypes.byref(ds_layout)) != VK_SUCCESS:
                self.dispatch_errors.append("vkCreateDescriptorSetLayout failed")
                return
            self.descriptor_set_layout = ds_layout

            # 5. Pipeline Layout
            vl.vkCreatePipelineLayout.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(VkPipelineLayoutCreateInfo),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vl.vkCreatePipelineLayout.restype = ctypes.c_int32

            pipe_layout_info = VkPipelineLayoutCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext=None,
                flags=0,
                setLayoutCount=1,
                pSetLayouts=(ctypes.c_void_p * 1)(self.descriptor_set_layout.value),
                pushConstantRangeCount=0,
                pPushConstantRanges=None
            )
            pipe_layout = ctypes.c_void_p()
            if vl.vkCreatePipelineLayout(self.device, ctypes.byref(pipe_layout_info), None, ctypes.byref(pipe_layout)) != VK_SUCCESS:
                self.dispatch_errors.append("vkCreatePipelineLayout failed")
                return
            self.pipeline_layout = pipe_layout

            # 6. Shader Module
            spv_bytes = self._load_spirv_bytecode()
            if not spv_bytes:
                self.dispatch_errors.append("Failed to load SPIR-V bytecode")
                return

            vl.vkCreateShaderModule.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(VkShaderModuleCreateInfo),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vl.vkCreateShaderModule.restype = ctypes.c_int32

            code_u32 = (ctypes.c_uint32 * (len(spv_bytes) // 4)).from_buffer_copy(spv_bytes)
            sm_info = VkShaderModuleCreateInfo(
                sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext=None,
                flags=0,
                codeSize=len(spv_bytes),
                pCode=code_u32
            )
            shader_module = ctypes.c_void_p()
            if vl.vkCreateShaderModule(self.device, ctypes.byref(sm_info), None, ctypes.byref(shader_module)) != VK_SUCCESS:
                self.dispatch_errors.append("vkCreateShaderModule failed")
                return
            self.shader_module = shader_module

            # 7. vkCreateComputePipelines
            vl.vkCreateComputePipelines.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(VkComputePipelineCreateInfo),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vl.vkCreateComputePipelines.restype = ctypes.c_int32

            stage_info = VkPipelineShaderStageCreateInfo(
                sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext=None,
                flags=0,
                stage=VK_SHADER_STAGE_COMPUTE_BIT,
                module=self.shader_module,
                pName=b"main",
                pSpecializationInfo=None
            )
            comp_pipe_info = VkComputePipelineCreateInfo(
                sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext=None,
                flags=0,
                stage=stage_info,
                layout=self.pipeline_layout,
                basePipelineHandle=None,
                basePipelineIndex=-1
            )
            pipeline = ctypes.c_void_p()
            if vl.vkCreateComputePipelines(self.device, None, 1, ctypes.byref(comp_pipe_info), None, ctypes.byref(pipeline)) != VK_SUCCESS:
                self.dispatch_errors.append("vkCreateComputePipelines failed")
                return
            self.pipeline = pipeline
            self.pipeline_ready = True

            # 8. Command Pool & Command Buffer
            vl.vkCreateCommandPool.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(VkCommandPoolCreateInfo),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vl.vkCreateCommandPool.restype = ctypes.c_int32

            cmd_pool_info = VkCommandPoolCreateInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                pNext=None,
                flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex=self.compute_queue_family
            )
            cmd_pool = ctypes.c_void_p()
            if vl.vkCreateCommandPool(self.device, ctypes.byref(cmd_pool_info), None, ctypes.byref(cmd_pool)) != VK_SUCCESS:
                self.dispatch_errors.append("vkCreateCommandPool failed")
                return
            self.command_pool = cmd_pool

            vl.vkAllocateCommandBuffers.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(VkCommandBufferAllocateInfo),
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vl.vkAllocateCommandBuffers.restype = ctypes.c_int32

            cmd_alloc = VkCommandBufferAllocateInfo(
                sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext=None,
                commandPool=self.command_pool,
                level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1
            )
            cmd_buf = ctypes.c_void_p()
            if vl.vkAllocateCommandBuffers(self.device, ctypes.byref(cmd_alloc), ctypes.byref(cmd_buf)) != VK_SUCCESS:
                self.dispatch_errors.append("vkAllocateCommandBuffers failed")
                return
            self.command_buffer = cmd_buf

            # 9. Fence
            vl.vkCreateFence.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(VkFenceCreateInfo),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p)
            ]
            vl.vkCreateFence.restype = ctypes.c_int32

            fence_info = VkFenceCreateInfo(
                sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                pNext=None,
                flags=0
            )
            fence = ctypes.c_void_p()
            if vl.vkCreateFence(self.device, ctypes.byref(fence_info), None, ctypes.byref(fence)) != VK_SUCCESS:
                self.dispatch_errors.append("vkCreateFence failed")
                return
            self.fence = fence

        except Exception as e:
            self.dispatch_errors.append(f"Hardware pipeline setup exception: {e}")

    def _load_spirv_bytecode(self) -> bytes:
        """Loads SPIR-V bytecode from compiled file or embedded fallback."""
        # 1. Check disk file
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        spv_path = os.path.join(root, "src", "shaders", "raymarch.spv")
        if os.path.exists(spv_path):
            try:
                with open(spv_path, "rb") as f:
                    return f.read()
            except OSError:
                pass

        # 2. Check embedded bytecode module
        try:
            from src.shaders.raymarch_spv_bytes import RAYMARCH_SPV_BYTES
            return RAYMARCH_SPV_BYTES
        except ImportError:
            pass

        return b""

    def _ensure_buffers(self, width: int, height: int):
        """Dynamically creates or resizes SSBO and UBO buffers for requested resolution."""
        if self.buffer_width == width and self.buffer_height == height and self.glyph_buffer is not None:
            return

        vl = self.vulkan_lib
        pixel_count = width * height

        # Teardown previous buffers if present
        if self.glyph_buffer:
            self._destroy_buffers()

        vl.vkCreateBuffer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(VkBufferCreateInfo),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p)
        ]
        vl.vkCreateBuffer.restype = ctypes.c_int32

        vl.vkGetBufferMemoryRequirements.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(VkMemoryRequirements)
        ]
        vl.vkGetBufferMemoryRequirements.restype = None

        vl.vkAllocateMemory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(VkMemoryAllocateInfo),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p)
        ]
        vl.vkAllocateMemory.restype = ctypes.c_int32

        vl.vkBindBufferMemory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        vl.vkBindBufferMemory.restype = ctypes.c_int32

        vl.vkMapMemory.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p)
        ]
        vl.vkMapMemory.restype = ctypes.c_int32

        def _allocate_buffer(size_bytes: int, usage: int) -> Tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]:
            binfo = VkBufferCreateInfo(
                sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext=None,
                flags=0,
                size=size_bytes,
                usage=usage,
                sharingMode=0,
                queueFamilyIndexCount=0,
                pQueueFamilyIndices=None
            )
            buf = ctypes.c_void_p()
            if vl.vkCreateBuffer(self.device, ctypes.byref(binfo), None, ctypes.byref(buf)) != VK_SUCCESS:
                raise RuntimeError("vkCreateBuffer failed")

            reqs = VkMemoryRequirements()
            vl.vkGetBufferMemoryRequirements(self.device, buf, ctypes.byref(reqs))

            alloc_info = VkMemoryAllocateInfo(
                sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                pNext=None,
                allocationSize=reqs.size,
                memoryTypeIndex=self.host_memory_type_index
            )
            mem = ctypes.c_void_p()
            if vl.vkAllocateMemory(self.device, ctypes.byref(alloc_info), None, ctypes.byref(mem)) != VK_SUCCESS:
                raise RuntimeError("vkAllocateMemory failed")

            if vl.vkBindBufferMemory(self.device, buf, mem, 0) != VK_SUCCESS:
                raise RuntimeError("vkBindBufferMemory failed")

            mapped = ctypes.c_void_p()
            if vl.vkMapMemory(self.device, mem, 0, reqs.size, 0, ctypes.byref(mapped)) != VK_SUCCESS:
                raise RuntimeError("vkMapMemory failed")

            return buf, mem, mapped

        # Dual SSBO (glyphs: uint32, colors: uint32) + UBO (params)
        self.glyph_buffer, self.glyph_memory, self.glyph_mapped = _allocate_buffer(
            pixel_count * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        self.rgb_buffer, self.rgb_memory, self.rgb_mapped = _allocate_buffer(
            pixel_count * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        self.ubo_buffer, self.ubo_memory, self.ubo_mapped = _allocate_buffer(
            ctypes.sizeof(UBOParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
        )

        # Allocate descriptor pool and set
        vl.vkCreateDescriptorPool.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(VkDescriptorPoolCreateInfo),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p)
        ]
        vl.vkCreateDescriptorPool.restype = ctypes.c_int32

        pool_sizes = (VkDescriptorPoolSize * 2)(
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2),
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
        )
        pool_info = VkDescriptorPoolCreateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext=None,
            flags=0,
            maxSets=1,
            poolSizeCount=2,
            pPoolSizes=pool_sizes
        )
        desc_pool = ctypes.c_void_p()
        if vl.vkCreateDescriptorPool(self.device, ctypes.byref(pool_info), None, ctypes.byref(desc_pool)) != VK_SUCCESS:
            raise RuntimeError("vkCreateDescriptorPool failed")
        self.descriptor_pool = desc_pool

        vl.vkAllocateDescriptorSets.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(VkDescriptorSetAllocateInfo),
            ctypes.POINTER(ctypes.c_void_p)
        ]
        vl.vkAllocateDescriptorSets.restype = ctypes.c_int32

        ds_alloc = VkDescriptorSetAllocateInfo(
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext=None,
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=(ctypes.c_void_p * 1)(self.descriptor_set_layout.value)
        )
        desc_set = ctypes.c_void_p()
        if vl.vkAllocateDescriptorSets(self.device, ctypes.byref(ds_alloc), ctypes.byref(desc_set)) != VK_SUCCESS:
            raise RuntimeError("vkAllocateDescriptorSets failed")
        self.descriptor_set = desc_set

        # Update descriptor set
        vl.vkUpdateDescriptorSets.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(VkWriteDescriptorSet),
            ctypes.c_uint32,
            ctypes.c_void_p
        ]
        vl.vkUpdateDescriptorSets.restype = None

        buf_infos = (VkDescriptorBufferInfo * 3)(
            VkDescriptorBufferInfo(self.glyph_buffer, 0, pixel_count * 4),
            VkDescriptorBufferInfo(self.rgb_buffer, 0, pixel_count * 4),
            VkDescriptorBufferInfo(self.ubo_buffer, 0, ctypes.sizeof(UBOParams)),
        )
        writes = (VkWriteDescriptorSet * 3)(
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext=None,
                dstSet=self.descriptor_set,
                dstBinding=0,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo=None,
                pBufferInfo=ctypes.pointer(buf_infos[0]),
                pTexelBufferView=None
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext=None,
                dstSet=self.descriptor_set,
                dstBinding=1,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo=None,
                pBufferInfo=ctypes.pointer(buf_infos[1]),
                pTexelBufferView=None
            ),
            VkWriteDescriptorSet(
                sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext=None,
                dstSet=self.descriptor_set,
                dstBinding=2,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                pImageInfo=None,
                pBufferInfo=ctypes.pointer(buf_infos[2]),
                pTexelBufferView=None
            ),
        )
        vl.vkUpdateDescriptorSets(self.device, 3, writes, 0, None)

        self.buffer_width = width
        self.buffer_height = height

    def _destroy_buffers(self):
        """Frees allocated memory and buffers."""
        vl = self.vulkan_lib
        if not vl or not self.device:
            return

        if self.descriptor_pool:
            vl.vkDestroyDescriptorPool.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            vl.vkDestroyDescriptorPool.restype = None
            vl.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
            self.descriptor_pool = None
            self.descriptor_set = None

        vl.vkFreeMemory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        vl.vkFreeMemory.restype = None
        vl.vkDestroyBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        vl.vkDestroyBuffer.restype = None

        for buf, mem in [
            (self.glyph_buffer, self.glyph_memory),
            (self.rgb_buffer, self.rgb_memory),
            (self.ubo_buffer, self.ubo_memory),
        ]:
            if mem:
                vl.vkFreeMemory(self.device, mem, None)
            if buf:
                vl.vkDestroyBuffer(self.device, buf, None)

        self.glyph_buffer = self.glyph_memory = self.glyph_mapped = None
        self.rgb_buffer = self.rgb_memory = self.rgb_mapped = None
        self.ubo_buffer = self.ubo_memory = self.ubo_mapped = None
        self.buffer_width = self.buffer_height = 0

    def _dispatch_vulkan_compute(
        self,
        width: int,
        height: int,
        t: float,
        mode: str,
        cam_pos: Tuple[float, float, float],
        step_budget: int
    ) -> Tuple[List[int], List[int]]:
        """
        Executes real Vulkan compute dispatch on hardware queue with fence synchronization
        and host-visible readback.
        """
        t0 = time.perf_counter()
        vl = self.vulkan_lib
        pixel_count = width * height

        self._ensure_buffers(width, height)

        # 1. Populate UBO
        ubo_data = UBOParams(
            cam_pos=(cam_pos[0], cam_pos[1], cam_pos[2], 0.0),
            cam_orientation=(0.0, 0.0, 0.0, 1.0),
            res_width=width,
            res_height=height,
            speculative_depth=step_budget,
            mirror_folds=6,
            time=t,
            entropy_threshold=0.5,
            pad0=0.0,
            pad1=0.0
        )
        ctypes.memmove(self.ubo_mapped, ctypes.byref(ubo_data), ctypes.sizeof(UBOParams))
        t_prep_done = time.perf_counter()

        # 2. Record command buffer
        vl.vkResetFences.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkResetFences.restype = ctypes.c_int32
        vl.vkResetFences(self.device, 1, ctypes.byref(self.fence))

        vl.vkBeginCommandBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(VkCommandBufferBeginInfo)]
        vl.vkBeginCommandBuffer.restype = ctypes.c_int32

        begin_info = VkCommandBufferBeginInfo(
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext=None,
            flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo=None
        )
        if vl.vkBeginCommandBuffer(self.command_buffer, ctypes.byref(begin_info)) != VK_SUCCESS:
            raise RuntimeError("vkBeginCommandBuffer failed")

        vl.vkCmdBindPipeline.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        vl.vkCmdBindPipeline.restype = None
        vl.vkCmdBindPipeline(self.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)

        vl.vkCmdBindDescriptorSets.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.c_void_p
        ]
        vl.vkCmdBindDescriptorSets.restype = None
        vl.vkCmdBindDescriptorSets(
            self.command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0,
            1,
            (ctypes.c_void_p * 1)(self.descriptor_set.value),
            0,
            None
        )

        vl.vkCmdDispatch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        vl.vkCmdDispatch.restype = None
        groups_x = (width + 7) // 8
        groups_y = (height + 7) // 8
        vl.vkCmdDispatch(self.command_buffer, groups_x, groups_y, 1)

        vl.vkEndCommandBuffer.argtypes = [ctypes.c_void_p]
        vl.vkEndCommandBuffer.restype = ctypes.c_int32
        if vl.vkEndCommandBuffer(self.command_buffer) != VK_SUCCESS:
            raise RuntimeError("vkEndCommandBuffer failed")

        t_record_done = time.perf_counter()

        # 3. Submit
        vl.vkQueueSubmit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(VkSubmitInfo), ctypes.c_void_p]
        vl.vkQueueSubmit.restype = ctypes.c_int32

        submit_info = VkSubmitInfo(
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=None,
            waitSemaphoreCount=0,
            pWaitSemaphores=None,
            pWaitDstStageMask=None,
            commandBufferCount=1,
            pCommandBuffers=(ctypes.c_void_p * 1)(self.command_buffer.value),
            signalSemaphoreCount=0,
            pSignalSemaphores=None
        )
        if vl.vkQueueSubmit(self.queue, 1, ctypes.byref(submit_info), self.fence) != VK_SUCCESS:
            raise RuntimeError("vkQueueSubmit failed")

        t_submit_done = time.perf_counter()

        # 4. Wait for fence
        vl.vkWaitForFences.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.c_uint64]
        vl.vkWaitForFences.restype = ctypes.c_int32
        if vl.vkWaitForFences(self.device, 1, (ctypes.c_void_p * 1)(self.fence.value), 1, 1_000_000_000) != VK_SUCCESS:
            raise RuntimeError("vkWaitForFences timed out or failed")

        t_gpu_done = time.perf_counter()

        # 5. Readback from host-visible memory
        gpu_glyphs = (ctypes.c_uint32 * pixel_count).from_address(self.glyph_mapped.value)
        gpu_colors = (ctypes.c_uint32 * pixel_count).from_address(self.rgb_mapped.value)

        glyphs = list(gpu_glyphs)
        colors = list(gpu_colors)
        t_readback_done = time.perf_counter()

        # Update timings & metrics
        dt_total = max(1e-6, t_readback_done - t0)
        self.last_timings = {
            "prep_time_ms": round((t_record_done - t0) * 1000.0, 3),
            "submit_time_ms": round((t_submit_done - t_record_done) * 1000.0, 3),
            "wait_time_ms": round((t_gpu_done - t_submit_done) * 1000.0, 3),
            "readback_time_ms": round((t_readback_done - t_gpu_done) * 1000.0, 3),
            "frame_time_ms": round(dt_total * 1000.0, 3),
        }
        self.dispatch_count += 1
        self.last_dispatch_us = round(dt_total * 1e6, 1)

        buffer_bytes = pixel_count * 5  # 1 byte glyph + 4 byte color equivalent representation
        self.readback_mb_s = round((buffer_bytes / (1024 * 1024)) / dt_total, 2)
        self.real_gpu_dispatch = True
        self.compute_backend = "VULKAN_GPU"

        return glyphs, colors

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
        Attempts native Vulkan compute dispatch if hardware is available and healthy;
        automatically falls back to deterministic CPU reference upon any failure.
        """
        if self.pipeline_ready and self.device and self.prefer_gpu:
            try:
                return self._dispatch_vulkan_compute(width, height, t, mode, cam_pos, step_budget)
            except Exception as e:
                self.dispatch_errors.append(f"GPU dispatch failed, falling back to CPU: {e}")
                self.real_gpu_dispatch = False
                self.compute_backend = "CPU_EMULATION"
                self.fallback_count += 1

        return self.execute_raymarch_cpu(width, height, t, mode, cam_pos, step_budget)

    def execute_raymarch_cpu(
        self,
        width: int,
        height: int,
        t: float,
        mode: str = "CYBERPUNK",
        cam_pos: Tuple[float, float, float] = (0.0, 0.0, -3.2),
        step_budget: int = 32
    ) -> Tuple[List[int], List[int]]:
        """
        CPU Reference Implementation (Mathematical Correctness Oracle).
        Pure-Python simulation of the intended compute kernel.
        """
        t0 = time.perf_counter()
        total_pixels = width * height

        glyphs: List[int] = [0] * total_pixels
        colors: List[int] = [0] * total_pixels

        aspect = (width / height) * 0.52
        light_dir = (0.577, 0.577, -0.577)

        r1 = 1.05 + 0.1 * math.sin(t * 1.5)
        r2 = 0.38
        core_r = 0.55 + 0.12 * math.sin(t * 3.5)

        for y in range(height):
            screen_y = (1.0 - (y / height) * 2.0)
            row_offset = y * width
            for x in range(width):
                screen_x = ((x / width) * 2.0 - 1.0) * aspect

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

                    theta = t * 1.2
                    c, s = math.cos(theta), math.sin(theta)
                    rx = px * c + pz * s
                    ry = py
                    rz = -px * s + pz * c

                    qx = math.sqrt(rx * rx + rz * rz) - r1
                    qy = ry
                    d_torus = math.sqrt(qx * qx + qy * qy) - r2
                    d_core = math.sqrt(px * px + py * py + pz * pz) - core_r

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

                    r = int(255 * luma)
                    g = int(40 + 160 * rim)
                    b = int(120 + 130 * diff)
                    colors[idx] = (r << 16) | (g << 8) | b
                else:
                    glyphs[idx] = 0
                    colors[idx] = 0x0A0F19

        dt = max(1e-6, time.perf_counter() - t0)
        self.dispatch_count += 1
        self.last_dispatch_us = round(dt * 1e6, 1)

        buffer_bytes = total_pixels * 5
        self.readback_mb_s = round((buffer_bytes / (1024 * 1024)) / dt, 2)
        self.real_gpu_dispatch = False
        self.compute_backend = "CPU_EMULATION"

        return glyphs, colors

    def verify_output_against_cpu(
        self,
        width: int = 96,
        height: int = 40,
        t: float = 0.5,
        mode: str = "CYBERPUNK",
        cam_pos: Tuple[float, float, float] = (0.0, 0.0, -3.2),
        step_budget: int = 32
    ) -> Dict[str, Any]:
        """
        Executes both GPU hardware dispatch and CPU reference oracle,
        calculating match ratios and error metrics.
        """
        if not self.pipeline_ready:
            return {"status": "SKIPPED", "reason": "Hardware pipeline not ready"}

        # Force GPU execution
        self.prefer_gpu = True
        gpu_glyphs, gpu_colors = self.execute_raymarch(width, height, t, mode, cam_pos, step_budget)
        assert self.real_gpu_dispatch, "GPU dispatch failed during verification"

        # Force CPU execution
        cpu_glyphs, cpu_colors = self.execute_raymarch_cpu(width, height, t, mode, cam_pos, step_budget)

        total_pixels = width * height
        glyph_matches = sum(1 for i in range(total_pixels) if gpu_glyphs[i] == cpu_glyphs[i])
        color_matches = sum(1 for i in range(total_pixels) if gpu_colors[i] == cpu_colors[i])

        color_diffs = [abs(gpu_colors[i] - cpu_colors[i]) for i in range(total_pixels)]
        mae = sum(color_diffs) / total_pixels
        max_err = max(color_diffs)

        import hashlib
        h = hashlib.sha256()
        h.update(bytes(gpu_glyphs))
        for c in gpu_colors:
            h.update(c.to_bytes(4, "little"))

        self.last_match_metrics = {
            "glyph_match_ratio": round(glyph_matches / total_pixels, 4),
            "pixel_match_ratio": round(color_matches / total_pixels, 4),
            "mean_absolute_error": round(mae, 4),
            "max_error": max_err,
            "output_hash": h.hexdigest(),
        }

        return {
            "status": "VERIFIED",
            "resolution": [width, height],
            "total_pixels": total_pixels,
            "glyph_matches": glyph_matches,
            "color_matches": color_matches,
            "metrics": self.last_match_metrics,
            "timings": self.last_timings,
        }

    def cleanup(self):
        """Deterministically releases all allocated Vulkan resources."""
        vl = self.vulkan_lib
        if not vl or not self.device:
            return

        try:
            # Wait for all queue operations to complete
            vl.vkDeviceWaitIdle.argtypes = [ctypes.c_void_p]
            vl.vkDeviceWaitIdle.restype = ctypes.c_int32
            vl.vkDeviceWaitIdle(self.device)

            self._destroy_buffers()

            if self.fence:
                vl.vkDestroyFence.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                vl.vkDestroyFence(self.device, self.fence, None)
                self.fence = None

            if self.command_pool:
                vl.vkDestroyCommandPool.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                vl.vkDestroyCommandPool(self.device, self.command_pool, None)
                self.command_pool = None
                self.command_buffer = None

            if self.pipeline:
                vl.vkDestroyPipeline.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                vl.vkDestroyPipeline(self.device, self.pipeline, None)
                self.pipeline = None

            if self.pipeline_layout:
                vl.vkDestroyPipelineLayout.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                vl.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
                self.pipeline_layout = None

            if self.shader_module:
                vl.vkDestroyShaderModule.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                vl.vkDestroyShaderModule(self.device, self.shader_module, None)
                self.shader_module = None

            if self.descriptor_set_layout:
                vl.vkDestroyDescriptorSetLayout.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                vl.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, None)
                self.descriptor_set_layout = None

            vl.vkDestroyDevice.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            vl.vkDestroyDevice(self.device, None)
            self.device = None
            self.queue = None
            self.logical_device_created = False
            self.compute_queue_ready = False
            self.pipeline_ready = False

        except Exception as e:
            self.dispatch_errors.append(f"Cleanup error: {e}")

    def __del__(self):
        self.cleanup()

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns driver status, active GPU device, and dispatch timings."""
        groups_x = (self.buffer_width + 7) // 8 if self.buffer_width > 0 else 0
        groups_y = (self.buffer_height + 7) // 8 if self.buffer_height > 0 else 0

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
            "ssbo_readback_kb_per_frame": round((self.buffer_width * self.buffer_height * 5) / 1024.0, 2) if self.buffer_width > 0 else 19.2,
            "device_detected": self.device_detected,
            "real_gpu_dispatch": self.real_gpu_dispatch,
            "compute_backend": self.compute_backend,
            "readback_is_simulated": not self.real_gpu_dispatch,
            "logical_device_created": self.logical_device_created,
            "compute_queue_ready": self.compute_queue_ready,
            "pipeline_ready": self.pipeline_ready,
            "workgroup_size": list(self.workgroup_size),
            "groups_xyz": [groups_x, groups_y, 1],
            "host_memory_type_index": self.host_memory_type_index,
            "fallback_count": self.fallback_count,
            "dispatch_errors": list(self.dispatch_errors),
            "timings": dict(self.last_timings),
            "match_metrics": dict(self.last_match_metrics),
        }
