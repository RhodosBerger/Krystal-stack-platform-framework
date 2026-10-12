"""
Generic Vulkan compute kernel runner (Phase 5.3 "Real Silicon Evidence").

The existing ``VulkanComputeDriver`` is specialised for the SDF raymarch shader. This module
reuses its ctypes structures and device probing, and adds a small, generic runner for
storage-buffer compute kernels with a fixed 4-binding layout:

    binding 0: Params  (32 bytes: uint n, uint stride, uint aux0, uint aux1, float c[4])
    binding 1: A       (input)
    binding 2: B       (input, may be unused)
    binding 3: Out     (output)

Evidence rules:
  * Every timing is wall-clock (time.perf_counter). Nothing is added to a measurement.
  * Timings are split so callers can see where the cost is: marshal (Python <-> bytes),
    upload (memcpy into host-visible UMA memory), gpu (submit + fence wait), readback.
  * ``available()`` is False with a reason whenever any step fails; callers must then label the
    tier UNAVAILABLE instead of substituting a CPU result.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import struct
import threading
import time
from array import array
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.python import vulkan_compute_driver as vk

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAXONOMY_SHADER_DIR = os.path.join(_ROOT, "src", "shaders", "taxonomy")
LOCAL_SIZE_X = 256
PARAMS_STRUCT = struct.Struct("<4I4f")  # 32 bytes, std430-compatible
_MIN_BUFFER_BYTES = 64
_FENCE_TIMEOUT_NS = 5_000_000_000


@dataclass
class KernelTimings:
    marshal_ms: float = 0.0
    upload_ms: float = 0.0
    gpu_ms: float = 0.0          # record + submit + fence wait
    readback_ms: float = 0.0
    unmarshal_ms: float = 0.0

    @property
    def device_path_ms(self) -> float:
        """Upload + GPU + readback (what a native caller holding raw buffers would pay)."""
        return self.upload_ms + self.gpu_ms + self.readback_ms

    @property
    def end_to_end_ms(self) -> float:
        """Everything a Python caller holding Python lists pays."""
        return self.marshal_ms + self.device_path_ms + self.unmarshal_ms

    def to_dict(self) -> Dict[str, float]:
        return {
            "marshal_ms": round(self.marshal_ms, 4),
            "upload_ms": round(self.upload_ms, 4),
            "gpu_ms": round(self.gpu_ms, 4),
            "readback_ms": round(self.readback_ms, 4),
            "unmarshal_ms": round(self.unmarshal_ms, 4),
            "device_path_ms": round(self.device_path_ms, 4),
            "end_to_end_ms": round(self.end_to_end_ms, 4),
        }


@dataclass
class _Buffer:
    buf: ctypes.c_void_p
    mem: ctypes.c_void_p
    mapped: ctypes.c_void_p
    capacity: int


@dataclass
class _Pipeline:
    name: str
    ds_layout: ctypes.c_void_p
    pipe_layout: ctypes.c_void_p
    module: ctypes.c_void_p
    pipeline: ctypes.c_void_p
    desc_pool: ctypes.c_void_p
    desc_set: ctypes.c_void_p
    buffers: List[Optional[_Buffer]] = field(default_factory=lambda: [None, None, None, None])


class VulkanContext:
    """Process-wide Vulkan device / queue / command buffer for generic compute kernels."""

    _instance: Optional["VulkanContext"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "VulkanContext":
        with cls._lock:
            if cls._instance is None:
                cls._instance = VulkanContext()
            return cls._instance

    def __init__(self) -> None:
        self.ok = False
        self.reason = ""
        self.device_name = ""
        self.vl = None
        self.device = None
        self.queue = None
        self.cmd_pool = None
        self.cmd_buf = None
        self.fence = None
        self.mem_type = -1
        self.pipelines: Dict[str, _Pipeline] = {}
        self._probe = None
        try:
            self._init()
            self.ok = True
            atexit.register(self.close)
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable", with reason
            self.reason = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ init
    def _init(self) -> None:
        probe = vk.VulkanComputeDriver(prefer_gpu=False)  # instance + physical device only
        if not probe.vulkan_available or not probe.physical_device or probe.compute_queue_family < 0:
            raise RuntimeError("no Vulkan compute device: " + "; ".join(probe.dispatch_errors))
        self._probe = probe  # keep the VkInstance alive
        self.vl = vl = probe.vulkan_lib
        self.device_name = probe.device_name
        qf = probe.compute_queue_family

        vl.vkCreateDevice.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkDeviceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkCreateDevice.restype = ctypes.c_int32
        prio = (ctypes.c_float * 1)(1.0)
        qinfo = vk.VkDeviceQueueCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, pNext=None, flags=0,
                                           queueFamilyIndex=qf, queueCount=1, pQueuePriorities=prio)
        dinfo = vk.VkDeviceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, pNext=None, flags=0,
                                      queueCreateInfoCount=1, pQueueCreateInfos=ctypes.pointer(qinfo),
                                      enabledLayerCount=0, ppEnabledLayerNames=None, enabledExtensionCount=0,
                                      ppEnabledExtensionNames=None, pEnabledFeatures=None)
        device = ctypes.c_void_p()
        self._check(vl.vkCreateDevice(probe.physical_device, ctypes.byref(dinfo), None, ctypes.byref(device)), "vkCreateDevice")
        self.device = device

        vl.vkGetDeviceQueue.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkGetDeviceQueue.restype = None
        queue = ctypes.c_void_p()
        vl.vkGetDeviceQueue(device, qf, 0, ctypes.byref(queue))
        if not queue.value:
            raise RuntimeError("vkGetDeviceQueue returned null")
        self.queue = queue

        vl.vkGetPhysicalDeviceMemoryProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkPhysicalDeviceMemoryProperties)]
        vl.vkGetPhysicalDeviceMemoryProperties.restype = None
        props = vk.VkPhysicalDeviceMemoryProperties()
        vl.vkGetPhysicalDeviceMemoryProperties(probe.physical_device, ctypes.byref(props))
        want = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        best, fallback = -1, -1
        for i in range(props.memoryTypeCount):
            fl = props.memoryTypes[i].propertyFlags
            if (fl & want) == want:
                if fl & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT and best < 0:
                    best = i
                elif fallback < 0:
                    fallback = i
        self.mem_type = best if best >= 0 else fallback
        if self.mem_type < 0:
            raise RuntimeError("no HOST_VISIBLE|HOST_COHERENT memory type")

        vl.vkCreateCommandPool.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkCommandPoolCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkCreateCommandPool.restype = ctypes.c_int32
        pinfo = vk.VkCommandPoolCreateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, pNext=None,
                                           flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex=qf)
        pool = ctypes.c_void_p()
        self._check(vl.vkCreateCommandPool(device, ctypes.byref(pinfo), None, ctypes.byref(pool)), "vkCreateCommandPool")
        self.cmd_pool = pool

        vl.vkAllocateCommandBuffers.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkCommandBufferAllocateInfo), ctypes.POINTER(ctypes.c_void_p)]
        vl.vkAllocateCommandBuffers.restype = ctypes.c_int32
        ainfo = vk.VkCommandBufferAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, pNext=None,
                                               commandPool=pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1)
        cbuf = ctypes.c_void_p()
        self._check(vl.vkAllocateCommandBuffers(device, ctypes.byref(ainfo), ctypes.byref(cbuf)), "vkAllocateCommandBuffers")
        self.cmd_buf = cbuf

        vl.vkCreateFence.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkFenceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkCreateFence.restype = ctypes.c_int32
        finfo = vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext=None, flags=0)
        fence = ctypes.c_void_p()
        self._check(vl.vkCreateFence(device, ctypes.byref(finfo), None, ctypes.byref(fence)), "vkCreateFence")
        self.fence = fence

        # Function prototypes used on the hot path.
        vl.vkCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkBufferCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkCreateBuffer.restype = ctypes.c_int32
        vl.vkGetBufferMemoryRequirements.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(vk.VkMemoryRequirements)]
        vl.vkGetBufferMemoryRequirements.restype = None
        vl.vkAllocateMemory.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkMemoryAllocateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkAllocateMemory.restype = ctypes.c_int32
        vl.vkBindBufferMemory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        vl.vkBindBufferMemory.restype = ctypes.c_int32
        vl.vkMapMemory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkMapMemory.restype = ctypes.c_int32
        vl.vkFreeMemory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        vl.vkFreeMemory.restype = None
        vl.vkDestroyBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        vl.vkDestroyBuffer.restype = None
        vl.vkUpdateDescriptorSets.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(vk.VkWriteDescriptorSet), ctypes.c_uint32, ctypes.c_void_p]
        vl.vkUpdateDescriptorSets.restype = None
        vl.vkResetFences.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkResetFences.restype = ctypes.c_int32
        vl.vkBeginCommandBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkCommandBufferBeginInfo)]
        vl.vkBeginCommandBuffer.restype = ctypes.c_int32
        vl.vkCmdBindPipeline.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        vl.vkCmdBindPipeline.restype = None
        vl.vkCmdBindDescriptorSets.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                               ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.c_void_p]
        vl.vkCmdBindDescriptorSets.restype = None
        vl.vkCmdDispatch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        vl.vkCmdDispatch.restype = None
        vl.vkEndCommandBuffer.argtypes = [ctypes.c_void_p]
        vl.vkEndCommandBuffer.restype = ctypes.c_int32
        vl.vkQueueSubmit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(vk.VkSubmitInfo), ctypes.c_void_p]
        vl.vkQueueSubmit.restype = ctypes.c_int32
        vl.vkWaitForFences.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.c_uint64]
        vl.vkWaitForFences.restype = ctypes.c_int32

    @staticmethod
    def _check(res: int, what: str) -> None:
        if res != vk.VK_SUCCESS:
            raise RuntimeError(f"{what} failed with VkResult {res}")

    # ------------------------------------------------------------- pipelines
    def pipeline(self, name: str, spv_path: str) -> _Pipeline:
        if name in self.pipelines:
            return self.pipelines[name]
        vl, dev = self.vl, self.device
        with open(spv_path, "rb") as fh:
            code = fh.read()
        if len(code) % 4 != 0 or len(code) < 20:
            raise RuntimeError(f"invalid SPIR-V: {spv_path}")

        vl.vkCreateDescriptorSetLayout.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkDescriptorSetLayoutCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkCreateDescriptorSetLayout.restype = ctypes.c_int32
        bindings = (vk.VkDescriptorSetLayoutBinding * 4)(*[
            vk.VkDescriptorSetLayoutBinding(i, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, vk.VK_SHADER_STAGE_COMPUTE_BIT, None)
            for i in range(4)
        ])
        dsl_info = vk.VkDescriptorSetLayoutCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, pNext=None,
                                                      flags=0, bindingCount=4, pBindings=bindings)
        ds_layout = ctypes.c_void_p()
        self._check(vl.vkCreateDescriptorSetLayout(dev, ctypes.byref(dsl_info), None, ctypes.byref(ds_layout)), "vkCreateDescriptorSetLayout")

        vl.vkCreatePipelineLayout.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkPipelineLayoutCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkCreatePipelineLayout.restype = ctypes.c_int32
        pl_info = vk.VkPipelineLayoutCreateInfo(sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, pNext=None, flags=0,
                                                setLayoutCount=1, pSetLayouts=(ctypes.c_void_p * 1)(ds_layout.value),
                                                pushConstantRangeCount=0, pPushConstantRanges=None)
        pipe_layout = ctypes.c_void_p()
        self._check(vl.vkCreatePipelineLayout(dev, ctypes.byref(pl_info), None, ctypes.byref(pipe_layout)), "vkCreatePipelineLayout")

        vl.vkCreateShaderModule.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkShaderModuleCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkCreateShaderModule.restype = ctypes.c_int32
        code_u32 = (ctypes.c_uint32 * (len(code) // 4)).from_buffer_copy(code)
        sm_info = vk.VkShaderModuleCreateInfo(sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, pNext=None, flags=0,
                                              codeSize=len(code), pCode=code_u32)
        module = ctypes.c_void_p()
        self._check(vl.vkCreateShaderModule(dev, ctypes.byref(sm_info), None, ctypes.byref(module)), "vkCreateShaderModule")

        vl.vkCreateComputePipelines.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(vk.VkComputePipelineCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkCreateComputePipelines.restype = ctypes.c_int32
        stage = vk.VkPipelineShaderStageCreateInfo(sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, pNext=None, flags=0,
                                                   stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName=b"main",
                                                   pSpecializationInfo=None)
        cp_info = vk.VkComputePipelineCreateInfo(sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, pNext=None, flags=0,
                                                 stage=stage, layout=pipe_layout, basePipelineHandle=None, basePipelineIndex=-1)
        pipeline = ctypes.c_void_p()
        self._check(vl.vkCreateComputePipelines(dev, None, 1, ctypes.byref(cp_info), None, ctypes.byref(pipeline)), "vkCreateComputePipelines")

        vl.vkCreateDescriptorPool.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkDescriptorPoolCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vl.vkCreateDescriptorPool.restype = ctypes.c_int32
        sizes = (vk.VkDescriptorPoolSize * 1)(vk.VkDescriptorPoolSize(vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4))
        dp_info = vk.VkDescriptorPoolCreateInfo(sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, pNext=None, flags=0,
                                                maxSets=1, poolSizeCount=1, pPoolSizes=sizes)
        pool = ctypes.c_void_p()
        self._check(vl.vkCreateDescriptorPool(dev, ctypes.byref(dp_info), None, ctypes.byref(pool)), "vkCreateDescriptorPool")

        vl.vkAllocateDescriptorSets.argtypes = [ctypes.c_void_p, ctypes.POINTER(vk.VkDescriptorSetAllocateInfo), ctypes.POINTER(ctypes.c_void_p)]
        vl.vkAllocateDescriptorSets.restype = ctypes.c_int32
        ds_alloc = vk.VkDescriptorSetAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, pNext=None,
                                                  descriptorPool=pool, descriptorSetCount=1,
                                                  pSetLayouts=(ctypes.c_void_p * 1)(ds_layout.value))
        dset = ctypes.c_void_p()
        self._check(vl.vkAllocateDescriptorSets(dev, ctypes.byref(ds_alloc), ctypes.byref(dset)), "vkAllocateDescriptorSets")

        p = _Pipeline(name, ds_layout, pipe_layout, module, pipeline, pool, dset)
        self.pipelines[name] = p
        return p

    # --------------------------------------------------------------- buffers
    def _alloc(self, size: int) -> _Buffer:
        vl, dev = self.vl, self.device
        size = max(_MIN_BUFFER_BYTES, size)
        binfo = vk.VkBufferCreateInfo(sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, pNext=None, flags=0, size=size,
                                      usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sharingMode=0,
                                      queueFamilyIndexCount=0, pQueueFamilyIndices=None)
        buf = ctypes.c_void_p()
        self._check(vl.vkCreateBuffer(dev, ctypes.byref(binfo), None, ctypes.byref(buf)), "vkCreateBuffer")
        reqs = vk.VkMemoryRequirements()
        vl.vkGetBufferMemoryRequirements(dev, buf, ctypes.byref(reqs))
        ainfo = vk.VkMemoryAllocateInfo(sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, pNext=None,
                                        allocationSize=reqs.size, memoryTypeIndex=self.mem_type)
        mem = ctypes.c_void_p()
        self._check(vl.vkAllocateMemory(dev, ctypes.byref(ainfo), None, ctypes.byref(mem)), "vkAllocateMemory")
        self._check(vl.vkBindBufferMemory(dev, buf, mem, 0), "vkBindBufferMemory")
        mapped = ctypes.c_void_p()
        self._check(vl.vkMapMemory(dev, mem, 0, reqs.size, 0, ctypes.byref(mapped)), "vkMapMemory")
        return _Buffer(buf, mem, mapped, size)

    def _free(self, b: Optional[_Buffer]) -> None:
        if b is None:
            return
        self.vl.vkFreeMemory(self.device, b.mem, None)  # implicitly unmaps
        self.vl.vkDestroyBuffer(self.device, b.buf, None)

    def _ensure_buffers(self, p: _Pipeline, sizes: Sequence[int]) -> None:
        changed = False
        for i, need in enumerate(sizes):
            cur = p.buffers[i]
            if cur is None or cur.capacity < need:
                self._free(cur)
                cap = _MIN_BUFFER_BYTES
                while cap < need:
                    cap *= 2
                p.buffers[i] = self._alloc(cap)
                changed = True
        if not changed:
            return
        infos = (vk.VkDescriptorBufferInfo * 4)(*[vk.VkDescriptorBufferInfo(b.buf, 0, b.capacity) for b in p.buffers])
        writes = (vk.VkWriteDescriptorSet * 4)(*[
            vk.VkWriteDescriptorSet(sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, pNext=None, dstSet=p.desc_set,
                                    dstBinding=i, dstArrayElement=0, descriptorCount=1,
                                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pImageInfo=None,
                                    pBufferInfo=ctypes.pointer(infos[i]), pTexelBufferView=None)
            for i in range(4)
        ])
        self.vl.vkUpdateDescriptorSets(self.device, 4, writes, 0, None)

    # -------------------------------------------------------------- dispatch
    def dispatch(self, p: _Pipeline, params: bytes, a: bytes, b: bytes, out_nbytes: int, threads: int,
                 timings: KernelTimings) -> bytes:
        vl, dev = self.vl, self.device
        t0 = time.perf_counter()
        self._ensure_buffers(p, (len(params), len(a), max(len(b), 4), out_nbytes))
        ctypes.memmove(p.buffers[0].mapped, params, len(params))
        ctypes.memmove(p.buffers[1].mapped, a, len(a))
        if b:
            ctypes.memmove(p.buffers[2].mapped, b, len(b))
        t1 = time.perf_counter()

        vl.vkResetFences(dev, 1, ctypes.byref(self.fence))
        begin = vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext=None,
                                            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, pInheritanceInfo=None)
        self._check(vl.vkBeginCommandBuffer(self.cmd_buf, ctypes.byref(begin)), "vkBeginCommandBuffer")
        vl.vkCmdBindPipeline(self.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline)
        vl.vkCmdBindDescriptorSets(self.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe_layout, 0, 1,
                                   (ctypes.c_void_p * 1)(p.desc_set.value), 0, None)
        vl.vkCmdDispatch(self.cmd_buf, max(1, (threads + LOCAL_SIZE_X - 1) // LOCAL_SIZE_X), 1, 1)
        self._check(vl.vkEndCommandBuffer(self.cmd_buf), "vkEndCommandBuffer")
        submit = vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, pNext=None, waitSemaphoreCount=0,
                                 pWaitSemaphores=None, pWaitDstStageMask=None, commandBufferCount=1,
                                 pCommandBuffers=(ctypes.c_void_p * 1)(self.cmd_buf.value),
                                 signalSemaphoreCount=0, pSignalSemaphores=None)
        self._check(vl.vkQueueSubmit(self.queue, 1, ctypes.byref(submit), self.fence), "vkQueueSubmit")
        self._check(vl.vkWaitForFences(dev, 1, (ctypes.c_void_p * 1)(self.fence.value), 1, _FENCE_TIMEOUT_NS), "vkWaitForFences")
        t2 = time.perf_counter()

        out = ctypes.string_at(p.buffers[3].mapped, out_nbytes)
        t3 = time.perf_counter()
        timings.upload_ms += (t1 - t0) * 1000.0
        timings.gpu_ms += (t2 - t1) * 1000.0
        timings.readback_ms += (t3 - t2) * 1000.0
        return out

    def close(self) -> None:
        if not self.ok or not self.device:
            return
        vl, dev = self.vl, self.device
        try:
            vl.vkDeviceWaitIdle.argtypes = [ctypes.c_void_p]
            vl.vkDeviceWaitIdle(dev)
            for p in self.pipelines.values():
                for b in p.buffers:
                    self._free(b)
                for fn, handle in (("vkDestroyDescriptorPool", p.desc_pool), ("vkDestroyPipeline", p.pipeline),
                                   ("vkDestroyShaderModule", p.module), ("vkDestroyPipelineLayout", p.pipe_layout),
                                   ("vkDestroyDescriptorSetLayout", p.ds_layout)):
                    f = getattr(vl, fn)
                    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                    f.restype = None
                    f(dev, handle, None)
            for fn, handle in (("vkDestroyFence", self.fence), ("vkDestroyCommandPool", self.cmd_pool)):
                f = getattr(vl, fn)
                f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                f.restype = None
                f(dev, handle, None)
            vl.vkDestroyDevice.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            vl.vkDestroyDevice.restype = None
            vl.vkDestroyDevice(dev, None)
        except Exception:  # noqa: BLE001 - best-effort teardown at exit
            pass
        self.pipelines.clear()
        self.device = None
        self.ok = False


# ============================================================================
# Taxonomy kernels (semantics identical to the T0 reference in
# crystalsec_v3_placement_calibration.TaxonomyExecutionEngine._run_directsim)
# ============================================================================

# workload class value -> shader basename
TAXONOMY_KERNELS: Dict[str, str] = {
    "INT8_DOT_PRODUCT": "int8_dot",
    "MIXED_COMPUTE_MEMORY": "int8_dot",   # T0 MIXED currently consumes only vector_a/vector_b
    "FP32_VECTOR_MATH": "fp32_poly",
    "MEMORY_BOUND_STREAM": "mem_stream",
    "BRANCH_HEAVY_CONTROL": "branch_heavy",
}


def _pack_uint8(values: Sequence[int]) -> bytes:
    raw = array("B", [int(x) & 0xFF for x in values]).tobytes()
    pad = (-len(raw)) % 4
    return raw + b"\x00" * pad


def _pack_int8(values: Sequence[int]) -> bytes:
    raw = array("b", [max(-128, min(127, int(x))) for x in values]).tobytes()
    pad = (-len(raw)) % 4
    return raw + b"\x00" * pad


class TaxonomyGpuKernels:
    """Runs taxonomy workloads on the real Vulkan device. Returns (outputs, KernelTimings)."""

    def __init__(self, context: Optional[VulkanContext] = None, shader_dir: str = TAXONOMY_SHADER_DIR):
        self.shader_dir = shader_dir
        self._ctx = context
        self._reason = ""

    @property
    def ctx(self) -> Optional[VulkanContext]:
        if self._ctx is None:
            self._ctx = VulkanContext.get()
        return self._ctx

    def available(self) -> bool:
        ctx = self.ctx
        if ctx is None or not ctx.ok:
            self._reason = ctx.reason if ctx is not None else "no context"
            return False
        missing = [k for k in set(TAXONOMY_KERNELS.values())
                   if not os.path.exists(os.path.join(self.shader_dir, k + ".spv"))]
        if missing:
            self._reason = f"missing SPIR-V: {missing} (run scripts/build_shaders.ps1)"
            return False
        return True

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    @property
    def device_name(self) -> str:
        ctx = self.ctx
        return ctx.device_name if ctx is not None else ""

    def supports(self, workload_class: str) -> bool:
        return workload_class in TAXONOMY_KERNELS

    def run(self, workload_class: str, data: Dict[str, Any]) -> Tuple[List[float], KernelTimings]:
        if workload_class not in TAXONOMY_KERNELS:
            raise ValueError(f"no GPU kernel for {workload_class}")
        ctx = self.ctx
        if ctx is None or not ctx.ok:
            raise RuntimeError(f"Vulkan unavailable: {ctx.reason if ctx else 'no context'}")
        kname = TAXONOMY_KERNELS[workload_class]
        pipe = ctx.pipeline(kname, os.path.join(self.shader_dir, kname + ".spv"))
        t = KernelTimings()

        tm = time.perf_counter()
        if kname == "int8_dot":
            va, vb = data["vector_a"], data["vector_b"]
            n = min(len(va), len(vb))
            a, b = _pack_uint8(va[:n]), _pack_int8(vb[:n])
            params = PARAMS_STRUCT.pack(n, 1, 0, 0, 0.0, 0.0, 0.0, 0.0)
        elif kname == "fp32_poly":
            xs, c = data["inputs"], list(data["coeffs"]) + [0.0] * 4
            n = len(xs)
            a, b = array("f", xs).tobytes(), b""
            params = PARAMS_STRUCT.pack(n, 1, 0, 0, *[float(v) for v in c[:4]])
        elif kname == "mem_stream":
            buf, stride = data["buffer"], int(data["stride"])
            n = (len(buf) + stride - 1) // stride
            a, b = array("f", buf).tobytes(), b""
            params = PARAMS_STRUCT.pack(n, stride, 0, 0, 0.0, 0.0, 0.0, 0.0)
        elif kname == "branch_heavy":
            keys = data["branch_keys"]
            n = len(keys)
            a, b = array("i", keys).tobytes(), b""
            params = PARAMS_STRUCT.pack(n, 1, 0, 0, 0.0, 0.0, 0.0, 0.0)
        else:  # pragma: no cover - guarded by TAXONOMY_KERNELS
            raise ValueError(kname)
        t.marshal_ms = (time.perf_counter() - tm) * 1000.0

        raw = ctx.dispatch(pipe, params, a, b, n * 4, n, t)

        tu = time.perf_counter()
        out = array("f")
        out.frombytes(raw)
        result = out.tolist()
        t.unmarshal_ms = (time.perf_counter() - tu) * 1000.0
        return result, t


def verify_outputs(reference: Sequence[float], candidate: Sequence[float],
                   rel_tol: float = 1e-5, abs_tol: float = 1e-4) -> Dict[str, Any]:
    """Element-wise comparison against the T0 reference. Exact classes still use the tolerance
    path, but integer-valued results below 2**24 are represented exactly in float32."""
    if len(reference) != len(candidate):
        return {"match": False, "reason": f"length {len(candidate)} != {len(reference)}",
                "max_abs_error": None, "mismatches": None}
    mism, max_err = 0, 0.0
    for r, c in zip(reference, candidate):
        err = abs(r - c)
        if err > max_err:
            max_err = err
        if err > max(abs_tol, rel_tol * abs(r)):
            mism += 1
    return {"match": mism == 0, "reason": "" if mism == 0 else f"{mism} elements outside tolerance",
            "max_abs_error": max_err, "mismatches": mism}
