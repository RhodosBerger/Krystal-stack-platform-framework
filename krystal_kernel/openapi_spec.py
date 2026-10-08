#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: OPENAPI 3.1.0 SPECIFICATION & DOCUMENTATION GENERATOR
==============================================================================
Module: krystal_kernel/openapi_spec.py
Description: Generates the official OpenAPI 3.1.0 specification describing all
             inputs, outputs, schemas, and endpoints for Cortex Process
             Prioritization, OpenVINO Inference, and Kernel Telemetry.

System Invariant: VITAL_MAX_HP = 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import json
from typing import Dict, Any

VITAL_MAX_HP: int = 6


def get_openapi_specification() -> Dict[str, Any]:
    """Generates the OpenAPI 3.1.0 specification dictionary."""
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Krystal-Stack Cortex & Windows API Process Prioritization API",
            "version": "2.4.0",
            "description": (
                "A high-performance system API orchestrating Windows process prioritization "
                "via kernel32.dll, supervised by the Cortex Cognitive Decision Algorithm "
                "and accelerated by Intel OpenVINO neural prioritization models. "
                "Maintains system invariant VITAL_MAX_HP = 6."
            ),
            "contact": {
                "name": "Dušan Kopecký & Krystal Architecture Council",
                "url": "http://localhost:8080"
            },
            "license": {
                "name": "Krystal Stack Enterprise License (2026)"
            }
        },
        "servers": [
            {
                "url": "http://localhost:8080",
                "description": "Localhost Production Hub"
            }
        ],
        "paths": {
            "/api/cortex/integrity": {
                "get": {
                    "summary": "Retrieve Cortex Algorithm System Integrity Decision",
                    "description": "Evaluates neuromorphic entropy, coherence, and thread thrashing to emit an authoritative decision on system health and priority mandates.",
                    "operationId": "getCortexIntegrity",
                    "responses": {
                        "200": {
                            "description": "Authoritative Cortex Integrity Decision",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/CortexIntegrityVerdict"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/cortex/prioritize": {
                "post": {
                    "summary": "Compile Policy and Prioritize Process via Windows API",
                    "description": "Synthesizes Cortex integrity, executes OpenVINO model inference, and invokes kernel32.dll!SetPriorityClass and SetProcessAffinityMask.",
                    "operationId": "prioritizeProcess",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ProcessPrioritizationRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prioritization outcome, applied Windows API codes, and compiled plan.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ProcessPrioritizationResponse"}
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid parameters or PID"
                        }
                    }
                }
            },
            "/api/cortex/compile": {
                "post": {
                    "summary": "Compile Scheduling Policy to Bytecode & Polyglot Stagers",
                    "description": "Compiles a process prioritization policy into an 8-operation bytecode plan, standalone C# Win32 stager, and C++ dispatcher without modifying OS state.",
                    "operationId": "compilePolicy",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CompilePolicyRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Compiled execution plan with generated polyglot source code.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/CortexCompiledPlan"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/cortex/openvino_infer": {
                "post": {
                    "summary": "Execute OpenVINO Neural Process Prioritization Model",
                    "description": "Performs forward tensor pass on 8 process telemetry features, returning Softmax probability distribution over 6 Windows Priority Classes.",
                    "operationId": "openvinoInfer",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/OpenVINOInferenceRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "OpenVINO model inference result and class probabilities.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/OpenVINOInferenceResult"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/cortex/processes": {
                "get": {
                    "summary": "List Active Windows Processes",
                    "description": "Enumerates running system processes with PID, working set memory, and priority class.",
                    "operationId": "listProcesses",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "description": "Maximum number of processes to return",
                            "required": False,
                            "schema": {"type": "integer", "default": 50}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of active processes",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "total_processes": {"type": "integer"},
                                            "processes": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/ProcessSummary"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/kernel/integrity": {
                "get": {
                    "summary": "Low-level Kernel Telemetry & Context-Switch Rates",
                    "description": "Direct telemetry from Windows NT kernel (ntdll) or Linux /proc/stat measuring real-time Context Switches, System Calls, and Thrashing Index.",
                    "operationId": "getKernelIntegrity",
                    "responses": {
                        "200": {
                            "description": "Live processor integrity telemetry report",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ProcessorIntegrityReport"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/copilot/generate": {
                "post": {
                    "summary": "Synthesize Polyglot Visual Copilot Source Code",
                    "description": "Generates source code for Microsoft Visual C#, C++20, Python Markov ML, and Vulkan shaders.",
                    "operationId": "generateCopilot",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CopilotGenerateRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Generated polyglot sources and behavioral predictions"
                        }
                    }
                }
            },
            "/api/asahi/power": {
                "get": {
                    "summary": "Asahi-Inspired Power Governor & Thermal Fuse Telemetry",
                    "description": "Returns multi-domain voltage balancing (V_core, V_gt), P-state, thermal headroom, and hardware acceleration ratio.",
                    "operationId": "getAsahiPower",
                    "responses": {
                        "200": {
                            "description": "Current power governor state",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PowerGovernorTelemetry"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/iris_xe/uma": {
                "get": {
                    "summary": "Intel Iris Xe Unified Shared Memory (UMA) Status",
                    "description": "Returns active memory quotient tier (32MB -> 512MB), effective bandwidth, and VSync pacing lock.",
                    "operationId": "getIrisXeUma",
                    "responses": {
                        "200": {
                            "description": "Active UMA quotient and memory bandwidth status",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/IrisXeUmaStatus"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/iris_xe/pace_frame": {
                "post": {
                    "summary": "Dynamic Frame Pacing & UMA Quotient Scaling",
                    "description": "Dynamically scales Iris Xe memory quotient and triggers Asahi Burst to guarantee VSync lock without dropping frames.",
                    "operationId": "paceFrameAndScaleQuotient",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/FramePacingRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Updated UMA quotient and VSync pacing report",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/IrisXeUmaStatus"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/self_healing/status": {
                "get": {
                    "summary": "Self-Healing Telemetry Governor Status",
                    "description": "Monitors program health, reports tolerated transient telemetric deviations, and active self-healing actions.",
                    "operationId": "getSelfHealingStatus",
                    "responses": {
                        "200": {
                            "description": "Self-healing status and active repair patterns",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SelfHealingStatusReport"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/self_healing/inject_telemetry": {
                "post": {
                    "summary": "Evaluate Telemetric Surge & Run Self-Healing Patterns",
                    "description": "Evaluates live or injected telemetric surge (CS storm, frame jitter, thermal burst) against the grace window.",
                    "operationId": "injectTelemetryAndHeal",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TelemetryInjectionRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Self-healing response and tolerance status",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SelfHealingStatusReport"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/wsl/coprocessor": {
                "get": {
                    "summary": "Retrieve WSL2 Coprocessor Status",
                    "description": "Returns status of the virtualized Linux coprocessor, Ubuntu distribution, and automated Bash scripts.",
                    "operationId": "getWslCoprocessorStatus",
                    "responses": {
                        "200": {
                            "description": "WSL2 Coprocessor status report",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/WSLCoprocessorStatus"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/wsl/diagnose_process": {
                "post": {
                    "summary": "Automated Failing Process Diagnostic via WSL Coprocessor",
                    "description": "Dispatches failing process (PID, crash signature) to WSL coprocessor for forensic analysis and automated admin remediation.",
                    "operationId": "diagnoseFailingProcessWsl",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ProcessDiagnosticRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Process diagnostic report with automated admin remediation",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ProcessDiagnosticReport"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/wsl/ufs_triage": {
                "post": {
                    "summary": "Run Automated UFS Log Triage Pipeline",
                    "description": "Executes Unix File System log triage, severity sorting, inode check, and automated socket cleanup.",
                    "operationId": "runUfsLogTriage",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/UfsLogTriageRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "UFS log triage and filesystem health verdict",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UfsLogTriageResult"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/wsl/remediate": {
                "post": {
                    "summary": "Execute Automated Administrative Remediation",
                    "description": "Executes automated system administrator remediation action without manual human intervention.",
                    "operationId": "executeAdminRemediation",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AutomatedAdminRemediationRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Remediation execution result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AutomatedAdminRemediationResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/wsl/gui_status": {
                "get": {
                    "summary": "Get WSL2 GUI Compositor Status",
                    "description": "Returns status of the low-latency cross-OS UMA bridge, active Wayland surfaces, and hypervisor transport state.",
                    "operationId": "getWslGuiStatus",
                    "responses": {
                        "200": {
                            "description": "Status of the WSL2 GUI bridge",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "active_mode": {"type": "string"},
                                            "active_surfaces_count": {"type": "integer"},
                                            "direct_d3d12_dxg_bridge": {"type": "boolean"},
                                            "vital_max_hp": {"type": "integer", "default": 6}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/wsl/benchmark_gui_latency": {
                "post": {
                    "summary": "Benchmark WSL2 GUI Latency vs Standard WSLg",
                    "description": "Compares standard Microsoft WSLg FreeRDP rail latency against Krystal Direct UMA Zero-Copy surface composition.",
                    "operationId": "benchmarkWslGuiLatency",
                    "responses": {
                        "200": {
                            "description": "Cross-OS composition benchmark results",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/WslGuiBenchmarkComparison"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/wsl/create_shared_surface": {
                "post": {
                    "summary": "Create Cross-OS Shared Surface for Linux Window",
                    "description": "Allocates a zero-copy D3D12/Vulkan shared texture handle accessible directly by both Windows DWM and WSL2 Linux Wayland client.",
                    "operationId": "createCrossOsSharedSurface",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/WslGuiCreateSurfaceRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Allocated cross-OS surface descriptor",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/CrossOsSurfaceDescriptor"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/bytecode/predict_power": {
                "post": {
                    "summary": "Predict Bytecode Power & Current Alternation Schedule",
                    "description": "Analyzes instruction dependency DAGs and computes a phase-staggered schedule that alternates current between Willow Cove and Iris Xe to eliminate voltage droop.",
                    "operationId": "predictBytecodePower",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/BytecodePowerPredictionRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Phase-staggered power schedule report",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/BytecodePowerScheduleReport"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/bytecode/micro_slice": {
                "post": {
                    "summary": "Dispatch Micro-Slice Step with Priority Action Injection",
                    "description": "Dispatches a sub-3ms micro-slice and dynamically injects high-priority preemptive action without dropping VSync frames.",
                    "operationId": "dispatchMicroSlice",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {"type": "object", "properties": {"inject_priority": {"type": "boolean", "default": True}}}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Micro-sliced execution result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/BytecodePowerScheduleReport"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/nss/reconstruct": {
                "post": {
                    "summary": "Execute K-NSS Open Neural Super-Sampling Benchmark",
                    "description": "Community-modifiable open alternative to Nvidia DLSS / Intel XeSS, upscaling 540p/720p to 1080p/4K using Iris Xe DP4A tensor instructions.",
                    "operationId": "reconstructFrameKnss",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/NssReconstructionRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "K-NSS reconstruction metrics and speedup report",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/NssReconstructionResult"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/nss/glsl_shader": {
                "get": {
                    "summary": "Export Open-Source K-NSS Vulkan GLSL Compute Shader",
                    "description": "Exports community-modifiable, permissive Apache-2.0 Vulkan 1.3 GLSL compute shader source code.",
                    "operationId": "exportKnssGlslShader",
                    "responses": {
                        "200": {
                            "description": "Vulkan GLSL shader source code export",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/NssGlslExportResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/llm/vram_unlock": {
                "post": {
                    "summary": "Unlock Intel Iris Xe VRAM Aperture for Local LLMs",
                    "description": "Unlocks host-coherent zero-copy UMA memory pool (2GB to 8GB), eliminating the 128MB Windows WDDM aperture clamping for quantized models.",
                    "operationId": "unlockVramAperture",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/VramUnlockRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "VRAM aperture unlock report",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/VramUnlockResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/llm/benchmark_tokens": {
                "post": {
                    "summary": "Benchmark Local Quantized LLM Token Generation",
                    "description": "Measures token generation speed (tokens/sec), latency, and bandwidth on Iris Xe comparing clamped 128MB vs unlocked UMA.",
                    "operationId": "benchmarkLocalLlmTokens",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/LlmTokenBenchmarkRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Token throughput benchmark results",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/LlmTokenBenchmarkResult"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/llm/optimal_plan": {
                "post": {
                    "summary": "Optimal Process Calculator & Chip Lifespan Plan",
                    "description": "Calculates matrix combinatorics of voltage, frequency, and quantization to maximize throughput while preserving silicon lifespan via Arrhenius model.",
                    "operationId": "getOptimalProcessPlan",
                    "responses": {
                        "200": {
                            "description": "Optimal process plan output",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/OptimalProcessPlan"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/isa/speculate": {
                "post": {
                    "summary": "Execute K-ISA Speculative Instruction Pipeline",
                    "description": "Dispatches speculative micro-opcodes (K_SPEC_PREFETCH_UMA, K_SPEC_INTERPOLATE_FRAME, K_SAFE_VOLT_CLAMP) to hide pipeline stalls and token latencies.",
                    "operationId": "executeKIsaSpeculation",
                    "responses": {
                        "200": {
                            "description": "Speculative pipeline execution report",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/KIsaSpeculationReport"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/godot/render_package": {
                "get": {
                    "summary": "Export Godot 4.x Render Engine Integration Package",
                    "description": "Exports integration GDScript, K-NSS viewport shader, and live telemetry connection parameters for Godot 4.x.",
                    "operationId": "exportGodotRenderPackage",
                    "responses": {
                        "200": {
                            "description": "Godot integration package metadata",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GodotRenderPackageResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/janet/decode_binary": {
                "post": {
                    "summary": "Decode Raw Binary Bytecode Stream & Synthesize Kernel Alerts (Janet)",
                    "description": "Parses 64-bit aligned KSYN binary stream via Janet engine semantics, emitting dual-representation hex audit trace, command profile, and structured severity alerts.",
                    "operationId": "decodeJanetBinaryStream",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "binary_hex": {"type": "string", "description": "Hexadecimal byte stream (e.g. 4B53594E...)"},
                                        "instructions": {"type": "array", "description": "Optional list of raw instruction dicts"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Decoded binary payload, command mappings, and synthesized alerts",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/JanetBinaryDecodeResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/janet/render_profile_svg": {
                "get": {
                    "summary": "Render Decoded Bytecode Profile Blueprint (SVG)",
                    "description": "Generates vector SVG blueprint of decoded instruction power bars, architectural domains, and timeline distributions.",
                    "operationId": "renderJanetProfileSvg",
                    "responses": {
                        "200": {
                            "description": "SVG vector graphic blueprint or JSON with embedded SVG",
                            "content": {
                                "image/svg+xml": {"schema": {"type": "string"}},
                                "application/json": {"schema": {"type": "object"}}
                            }
                        }
                    }
                }
            },
            "/api/wsl/hypervisor_port": {
                "get": {
                    "summary": "Retrieve KPHP Hypervisor Port & Direct UMA Status",
                    "description": "Returns status of AF_VSOCK port 19283, roundtrip latency, explorer suspension status, and reclaimed memory.",
                    "operationId": "getHypervisorPortStatus",
                    "responses": {
                        "200": {
                            "description": "Hypervisor port telemetry",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HypervisorPortStatus"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/wsl/toggle_gnome_overlay": {
                "post": {
                    "summary": "Toggle On-Demand GNOME Overlay / Explorer State",
                    "description": "Switches between GNOME_PURE_SOVEREIGN, HYBRID_SEAMLESS_OVERLAY, and WINDOWS_CLASSIC_BYPASS without reboot.",
                    "operationId": "toggleGnomeOverlay",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "target_mode": {"type": "string", "enum": ["GNOME_PURE_SOVEREIGN", "HYBRID_SEAMLESS_OVERLAY", "WINDOWS_CLASSIC_BYPASS"]}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Updated desktop mode and memory reclamation status",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HypervisorPortStatus"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/wsl/registry_inspect": {
                "get": {
                    "summary": "Inspect Windows Registry and Enumerate Win32 Apps for GNOME",
                    "description": "Exposes Windows Registry query capabilities and generates GNOME .desktop entry mappings for native Win32 apps.",
                    "operationId": "inspectWindowsRegistry",
                    "responses": {
                        "200": {
                            "description": "Mapped Win32 apps and registry values",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/openvino/hardware_bypass_status": {
                "get": {
                    "summary": "Retrieve Tiger Lake PL1/PL2 Bypass Telemetry & Voltage Clamps",
                    "description": "Inspects MSR 0x610 and HWP EPP state to confirm unblocking of OEM thermal clamps on 11th Gen Tiger Lake.",
                    "operationId": "getHardwareBypassStatus",
                    "responses": {
                        "200": {
                            "description": "Tiger Lake power limit bypass telemetry",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/TigerLakeBypassProfile"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/openvino/accelerate": {
                "post": {
                    "summary": "Apply OpenVINO DP4A INT8 & Multi-Stream Acceleration Pack",
                    "description": "Configures CUMULATIVE_THROUGHPUT, 4 GPU streams, model priority HIGH, and U8 KV-cache.",
                    "operationId": "applyOpenVinoAcceleration",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "use_dp4a_int8": {"type": "boolean", "default": True},
                                        "streams_count": {"type": "integer", "default": 4},
                                        "enable_kv_u8": {"type": "boolean", "default": True}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Updated OpenVINO execution parameters",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/OpenVinoExtendedConfig"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/openvino/benchmark_dp4a": {
                "get": {
                    "summary": "Execute Empirical Benchmark Proving Unlocked Core i5 Exceeds Stock Core i7",
                    "description": "Runs comparative throughput benchmark across Core i5 stock (15W), Core i7 stock (28W), and Krystal unlocked Core i5 (32W, DP4A).",
                    "operationId": "runOpenVinoDp4aBenchmark",
                    "responses": {
                        "200": {
                            "description": "Empirical benchmark comparison results",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ChipBenchmarkComparison"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/accelerator/predictions": {
                "get": {
                    "summary": "Retrieve Hardware Performance Predictions & Component Assistance",
                    "description": "Returns quantified FPS scaling, 1% low stutter reduction, 48.7x latency compression, and component assistance attribution breakdown.",
                    "operationId": "getAcceleratorPredictions",
                    "responses": {
                        "200": {
                            "description": "Hardware performance predictions and component breakdown",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GraphicsAccelerationBenchmarkReport"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/accelerator/dispatch_bytecode": {
                "post": {
                    "summary": "Dispatch Bytecode Stream to UMA Predictive Graphics Pipeline",
                    "description": "Ingests 64-bit aligned KSYN bytecode stream, performs 2nd-order Markov branch prediction, pre-stages UMA memory slabs, and dispatches GPU passes.",
                    "operationId": "dispatchBytecodeAccelerator",
                    "requestBody": {
                        "required": False,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "opcodes": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "default": [1, 2, 3, 161, 162, 165, 166]
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Accelerated graphics passes, UMA allocations, and frame synthesis metrics",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/BytecodeDispatchResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/api/accelerator/benchmark_suite": {
                "get": {
                    "summary": "Execute Empirical Graphics & Compute Benchmark Suite",
                    "description": "Full empirical benchmark comparing stock Tiger Lake 15W profile against Krystal 32W UMA DP4A profile with vital max HP lock.",
                    "operationId": "runAcceleratorBenchmarkSuite",
                    "responses": {
                        "200": {
                            "description": "Empirical benchmark suite results",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GraphicsAccelerationBenchmarkReport"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "TigerLakeBypassProfile": {
                    "type": "object",
                    "properties": {
                        "pl1_limit_watts": {"type": "number"},
                        "pl2_limit_watts": {"type": "number"},
                        "tau_window_seconds": {"type": "number"},
                        "hwp_epp_hex": {"type": "string"},
                        "voltage_offset_mv": {"type": "number"},
                        "voltage_clamp_max_v": {"type": "number"},
                        "junction_temp_c": {"type": "number"},
                        "arrhenius_wear_factor": {"type": "number"},
                        "pl_clamp_bypassed": {"type": "boolean"},
                        "performance_gain_pct": {"type": "number"},
                        "vital_max_hp": {"type": "integer", "default": 6},
                        "timestamp_iso": {"type": "string"}
                    }
                },
                "OpenVinoExtendedConfig": {
                    "type": "object",
                    "properties": {
                        "performance_hint": {"type": "string"},
                        "inference_precision_hint": {"type": "string"},
                        "execution_mode_hint": {"type": "string"},
                        "gpu_throughput_streams": {"type": "integer"},
                        "model_priority": {"type": "string"},
                        "gpu_host_task_priority": {"type": "string"},
                        "kv_cache_precision": {"type": "string"},
                        "enable_mmap_weights": {"type": "boolean"},
                        "cache_dir": {"type": "string"},
                        "multi_device_priorities": {"type": "string"},
                        "predicted_tokens_per_sec": {"type": "number"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "ChipBenchmarkComparison": {
                    "type": "object",
                    "properties": {
                        "benchmark_id": {"type": "string"},
                        "workload_name": {"type": "string"},
                        "i5_stock_15w_tokens_sec": {"type": "number"},
                        "i7_stock_28w_tokens_sec": {"type": "number"},
                        "krystal_i5_unlocked_32w_tokens_sec": {"type": "number"},
                        "krystal_i7_unlocked_35w_tokens_sec": {"type": "number"},
                        "i5_unlocked_vs_i7_stock_speedup_pct": {"type": "number"},
                        "arrhenius_longevity_guarantee": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "HypervisorPortStatus": {
                    "type": "object",
                    "properties": {
                        "port_number": {"type": "integer"},
                        "transport_protocol": {"type": "string"},
                        "active_mode": {"type": "string"},
                        "shared_memory_size_mb": {"type": "integer"},
                        "roundtrip_latency_ms": {"type": "number"},
                        "explorer_process_state": {"type": "string"},
                        "reclaimed_ram_mb": {"type": "integer"},
                        "active_win32_apps_in_gnome": {"type": "integer"},
                        "vital_max_hp": {"type": "integer", "default": 6},
                        "timestamp_iso": {"type": "string"}
                    }
                },
                "JanetBinaryDecodeResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "severity": {"type": "string"},
                        "title": {"type": "string"},
                        "instruction_count": {"type": "integer"},
                        "total_power_watts": {"type": "number"},
                        "alerts": {"type": "array"},
                        "decoded_commands": {"type": "array"},
                        "hex_trace_log": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "CortexIntegrityVerdict": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "number"},
                        "cortex_integrity_score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Integrity decided by the Cortex Algorithm"},
                        "coherence_index": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "neuromorphic_entropy": {"type": "number"},
                        "hemispheric_balance": {"type": "number"},
                        "verdict_status": {"type": "string", "enum": ["OPTIMAL_SYNAPSE", "BALANCED_EXECUTION", "SYNAPTIC_OVERLOAD", "CORTEX_COLLAPSE"]},
                        "decision_mandate": {"type": "string", "enum": ["BOOST_ALLOWED", "MAINTAIN", "FORCE_THROTTLE", "ISOLATE_CORES"]},
                        "recommended_windows_priority": {"type": "string", "enum": ["IDLE", "BELOW_NORMAL", "NORMAL", "ABOVE_NORMAL", "HIGH", "REALTIME"]},
                        "recommended_affinity_mask": {"type": "integer", "description": "Bitmask for CPU cores"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    },
                    "required": ["cortex_integrity_score", "verdict_status", "decision_mandate", "vital_max_hp"]
                },
                "ProcessPrioritizationRequest": {
                    "type": "object",
                    "properties": {
                        "pid": {"type": "integer", "description": "Target Process ID (PID)"},
                        "process_name": {"type": "string", "description": "Executable name e.g. render_worker.exe"},
                        "user_priority_intent": {"type": "string", "enum": ["IDLE", "BELOW_NORMAL", "NORMAL", "ABOVE_NORMAL", "HIGH", "REALTIME"], "nullable": True},
                        "demand_low_latency": {"type": "boolean", "default": False},
                        "cpu_load": {"type": "number", "default": 50.0},
                        "cs_rate": {"type": "number", "default": 6500.0},
                        "thrashing_index": {"type": "number", "default": 1.1},
                        "working_set_mb": {"type": "number", "default": 256.0},
                        "thread_count": {"type": "integer", "default": 8}
                    },
                    "required": ["pid"]
                },
                "ProcessPrioritizationResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "pid": {"type": "integer"},
                        "priority_applied": {"type": "string"},
                        "win32_code": {"type": "string"},
                        "affinity_applied": {"type": "string"},
                        "cortex_verdict": {"$ref": "#/components/schemas/CortexIntegrityVerdict"},
                        "openvino_inference": {"$ref": "#/components/schemas/OpenVINOInferenceResult"},
                        "compiled_plan_id": {"type": "string"},
                        "execution_details": {"type": "object"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "CompilePolicyRequest": {
                    "type": "object",
                    "properties": {
                        "pid": {"type": "integer"},
                        "process_name": {"type": "string"},
                        "user_priority_intent": {"type": "string"},
                        "demand_low_latency": {"type": "boolean", "default": False}
                    },
                    "required": ["pid"]
                },
                "CortexCompiledPlan": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "target_pid": {"type": "integer"},
                        "target_process_name": {"type": "string"},
                        "compiled_operations": {"type": "array", "items": {"type": "object"}},
                        "resolved_win32_priority_class": {"type": "string"},
                        "resolved_win32_priority_code": {"type": "integer"},
                        "resolved_affinity_mask": {"type": "integer"},
                        "generated_csharp_stager": {"type": "string"},
                        "generated_cpp_dispatcher": {"type": "string"},
                        "generated_powershell_cmd": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "OpenVINOInferenceRequest": {
                    "type": "object",
                    "properties": {
                        "pid": {"type": "integer", "default": 1000},
                        "name": {"type": "string", "default": "worker.exe"},
                        "cpu_pct": {"type": "number", "default": 50.0},
                        "cs_rate": {"type": "number", "default": 6500.0},
                        "page_faults_per_sec": {"type": "number", "default": 120.0},
                        "working_set_mb": {"type": "number", "default": 256.0},
                        "thread_count": {"type": "integer", "default": 8},
                        "io_ops_per_sec": {"type": "number", "default": 50.0},
                        "kernel_time_ratio": {"type": "number", "default": 0.15},
                        "thrashing_index": {"type": "number", "default": 1.1}
                    }
                },
                "OpenVINOInferenceResult": {
                    "type": "object",
                    "properties": {
                        "target_pid": {"type": "integer"},
                        "process_name": {"type": "string"},
                        "input_feature_vector": {"type": "array", "items": {"type": "number"}},
                        "predicted_priority_class": {"type": "string"},
                        "win32_priority_code": {"type": "integer"},
                        "confidence_score": {"type": "number"},
                        "class_probabilities": {"type": "object"},
                        "inference_device": {"type": "string"},
                        "inference_latency_us": {"type": "number"},
                        "openvino_backend": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "ProcessSummary": {
                    "type": "object",
                    "properties": {
                        "pid": {"type": "integer"},
                        "name": {"type": "string"},
                        "memory_mb": {"type": "number"},
                        "current_priority": {"type": "string"}
                    }
                },
                "ProcessorIntegrityReport": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "number"},
                        "cpu_utilization_pct": {"type": "number"},
                        "context_switches_per_sec": {"type": "number"},
                        "system_calls_per_sec": {"type": "number"},
                        "thrashing_index": {"type": "number"},
                        "integrity_score": {"type": "number"},
                        "status": {"type": "string"},
                        "behavioral_diagnosis": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "CopilotGenerateRequest": {
                    "type": "object",
                    "properties": {
                        "context_switches_per_sec": {"type": "number"},
                        "cpu_utilization_pct": {"type": "number"},
                        "thrashing_index": {"type": "number"}
                    }
                },
                "PowerGovernorTelemetry": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "number"},
                        "current_p_state": {"type": "string"},
                        "voltage_core_v": {"type": "number"},
                        "voltage_gpu_gt_v": {"type": "number"},
                        "clock_cpu_mhz": {"type": "integer"},
                        "clock_gpu_mhz": {"type": "integer"},
                        "junction_temp_c": {"type": "number"},
                        "thermal_headroom_c": {"type": "number"},
                        "thermal_fuse_breached": {"type": "boolean"},
                        "hardware_acceleration_ratio": {"type": "number"},
                        "estimated_power_watts": {"type": "number"},
                        "governor_diagnosis": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "IrisXeUmaStatus": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "number"},
                        "active_quotient_tier": {"type": "string", "enum": ["Q32_32MB", "Q64_64MB", "Q128_128MB", "Q256_256MB", "Q512_512MB"]},
                        "allocated_uma_mb": {"type": "integer"},
                        "max_uma_budget_mb": {"type": "integer"},
                        "effective_bandwidth_gbps": {"type": "number"},
                        "vsync_target_hz": {"type": "integer"},
                        "vsync_budget_ms": {"type": "number"},
                        "last_frame_time_ms": {"type": "number"},
                        "frame_drop_risk_pct": {"type": "number"},
                        "vsync_locked": {"type": "boolean"},
                        "render_efficiency_score": {"type": "number"},
                        "power_governor_p_state": {"type": "string"},
                        "junction_temp_c": {"type": "number"},
                        "thermal_fuse_headroom_c": {"type": "number"},
                        "status_diagnosis": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "FramePacingRequest": {
                    "type": "object",
                    "properties": {
                        "frame_time_ms": {"type": "number", "default": 7.5},
                        "complexity_factor": {"type": "number", "default": 1.0},
                        "vsync_target_hz": {"type": "integer", "default": 120}
                    }
                },
                "SelfHealingStatusReport": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "number"},
                        "system_operational_mode": {"type": "string", "enum": ["OPTIMAL_STEADY", "TOLERATING_TRANSIENT", "ACTIVE_HEALING", "THERMAL_EMERGENCY"]},
                        "tolerated_deviations_count": {"type": "integer"},
                        "deviations": {"type": "array", "items": {"type": "object"}},
                        "active_healing_actions": {"type": "array", "items": {"type": "string"}},
                        "thermal_fuse_headroom_c": {"type": "number"},
                        "thermal_fuse_safe": {"type": "boolean"},
                        "hardware_acceleration_granted": {"type": "boolean"},
                        "vsync_locked": {"type": "boolean"},
                        "cortex_vital_hp": {"type": "integer", "default": 6}
                    }
                },
                "TelemetryInjectionRequest": {
                    "type": "object",
                    "properties": {
                        "cs_rate": {"type": "number", "default": 35000.0},
                        "thrashing_index": {"type": "number", "default": 1.8},
                        "frame_time_ms": {"type": "number", "default": 7.9},
                        "junction_temp_c": {"type": "number", "default": 78.0},
                        "vsync_budget_ms": {"type": "number", "default": 8.333}
                    }
                },
                "WSLCoprocessorStatus": {
                    "type": "object",
                    "properties": {
                        "wsl_installed": {"type": "boolean"},
                        "default_distro": {"type": "string"},
                        "coprocessor_engine": {"type": "string"},
                        "execution_mode": {"type": "string"},
                        "scripts_available": {"type": "object"},
                        "subsystem_features": {"type": "array", "items": {"type": "string"}},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "ProcessDiagnosticRequest": {
                    "type": "object",
                    "properties": {
                        "pid": {"type": "integer", "default": 4412},
                        "process_name": {"type": "string", "default": "vulkan_renderer.exe"},
                        "crash_signature": {"type": "string", "default": "STATUS_ACCESS_VIOLATION (0xC0000005)"}
                    }
                },
                "ProcessDiagnosticReport": {
                    "type": "object",
                    "properties": {
                        "target_pid": {"type": "integer"},
                        "process_name": {"type": "string"},
                        "failure_type": {"type": "string"},
                        "crash_signature": {"type": "string"},
                        "confidence_pct": {"type": "integer"},
                        "virtual_coprocessor_backend": {"type": "string"},
                        "diagnostic_details": {"type": "string"},
                        "prescribed_admin_action": {"type": "string"},
                        "admin_intervention_required": {"type": "boolean"},
                        "vital_max_hp": {"type": "integer", "default": 6},
                        "timestamp_iso": {"type": "string"}
                    }
                },
                "UfsLogTriageRequest": {
                    "type": "object",
                    "properties": {
                        "log_directory": {"type": "string"},
                        "auto_heal": {"type": "boolean", "default": True}
                    }
                },
                "UfsLogTriageResult": {
                    "type": "object",
                    "properties": {
                        "triage_id": {"type": "string"},
                        "total_log_entries": {"type": "integer"},
                        "severity_breakdown": {"type": "object"},
                        "detected_crash_signatures": {"type": "object"},
                        "ufs_inode_health": {"type": "object"},
                        "automated_maintenance": {"type": "object"},
                        "subsystem": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6},
                        "timestamp_iso": {"type": "string"}
                    }
                },
                "AutomatedAdminRemediationRequest": {
                    "type": "object",
                    "properties": {
                        "action_name": {"type": "string", "default": "QUARANTINE_PROCESS_AND_STAGE_DUMP"}
                    }
                },
                "AutomatedAdminRemediationResponse": {
                    "type": "object",
                    "properties": {
                        "remediation_id": {"type": "string"},
                        "timestamp": {"type": "string"},
                        "remediation": {"type": "object"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "WslGuiCreateSurfaceRequest": {
                    "type": "object",
                    "properties": {
                        "window_title": {"type": "string", "default": "Krystal Linux Shell (Wayland)"},
                        "linux_pid": {"type": "integer", "default": 4096},
                        "width": {"type": "integer", "default": 1920},
                        "height": {"type": "integer", "default": 1080},
                        "mode": {"type": "string", "default": "KRYSTAL_DIRECT_UMA"}
                    }
                },
                "CrossOsSurfaceDescriptor": {
                    "type": "object",
                    "properties": {
                        "surface_id": {"type": "string"},
                        "window_title": {"type": "string"},
                        "linux_pid": {"type": "integer"},
                        "resolution_w": {"type": "integer"},
                        "resolution_h": {"type": "integer"},
                        "pixel_format": {"type": "string"},
                        "dxgi_shared_handle": {"type": "string"},
                        "latency_mode": {"type": "string"},
                        "render_latency_ms": {"type": "number"},
                        "framerate_fps": {"type": "number"},
                        "frame_jitter_ms": {"type": "number"},
                        "zero_copy_active": {"type": "boolean"},
                        "vital_max_hp": {"type": "integer", "default": 6},
                        "timestamp_iso": {"type": "string"}
                    }
                },
                "WslGuiBenchmarkComparison": {
                    "type": "object",
                    "properties": {
                        "benchmark_id": {"type": "string"},
                        "standard_wslg_latency_ms": {"type": "number"},
                        "standard_wslg_fps": {"type": "number"},
                        "standard_wslg_jitter_ms": {"type": "number"},
                        "krystal_uma_latency_ms": {"type": "number"},
                        "krystal_uma_fps": {"type": "number"},
                        "krystal_uma_jitter_ms": {"type": "number"},
                        "latency_reduction_factor": {"type": "number"},
                        "bandwidth_saved_pct": {"type": "number"},
                        "kisa_speculation_active": {"type": "boolean"},
                        "frame_drops_prevented": {"type": "integer"},
                        "system_invariant_intact": {"type": "boolean"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "BytecodePowerPredictionRequest": {
                    "type": "object",
                    "properties": {
                        "inject_priority_action": {"type": "boolean", "default": True},
                        "custom_instructions": {"type": "array", "items": {"type": "object"}}
                    }
                },
                "MicroSliceStep": {
                    "type": "object",
                    "properties": {
                        "slice_index": {"type": "integer"},
                        "phase_name": {"type": "string"},
                        "active_instructions": {"type": "array", "items": {"type": "integer"}},
                        "active_domains": {"type": "array", "items": {"type": "string"}},
                        "total_current_amperes": {"type": "number"},
                        "voltage_droop_risk_pct": {"type": "number"},
                        "time_budget_us": {"type": "number"},
                        "priority_action_injected": {"type": "boolean"},
                        "log_audit_signature": {"type": "string"}
                    }
                },
                "BytecodePowerScheduleReport": {
                    "type": "object",
                    "properties": {
                        "schedule_id": {"type": "string"},
                        "total_instructions": {"type": "integer"},
                        "uncoordinated_peak_current_a": {"type": "number"},
                        "staggered_peak_current_a": {"type": "number"},
                        "current_reduction_pct": {"type": "number"},
                        "voltage_droop_prevented": {"type": "boolean"},
                        "total_latency_us": {"type": "number"},
                        "response_budget_ms": {"type": "number"},
                        "micro_slices": {"type": "array", "items": {"$ref": "#/components/schemas/MicroSliceStep"}},
                        "step_log_trace": {"type": "array", "items": {"type": "string"}},
                        "vital_max_hp": {"type": "integer", "default": 6},
                        "timestamp_iso": {"type": "string"}
                    }
                },
                "NssReconstructionRequest": {
                    "type": "object",
                    "properties": {
                        "target_width": {"type": "integer", "default": 1920},
                        "target_height": {"type": "integer", "default": 1080},
                        "profile": {"type": "string", "enum": ["ULTRA_PERFORMANCE", "PERFORMANCE", "BALANCED", "QUALITY", "ULTRA_QUALITY"], "default": "PERFORMANCE"}
                    }
                },
                "NssReconstructionResult": {
                    "type": "object",
                    "properties": {
                        "profile": {"type": "string"},
                        "resolution": {"type": "object"},
                        "native_frame_time_ms": {"type": "number"},
                        "knss_frame_time_ms": {"type": "number"},
                        "effective_fps_native": {"type": "number"},
                        "effective_fps_knss": {"type": "number"},
                        "speedup_multiplier": {"type": "number"},
                        "latency_saved_ms": {"type": "number"},
                        "dp4a_tensor_cycles": {"type": "integer"},
                        "vram_bandwidth_saved_pct": {"type": "number"},
                        "open_source_license": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6},
                        "timestamp_iso": {"type": "string"}
                    }
                },
                "NssGlslExportResponse": {
                    "type": "object",
                    "properties": {
                        "shader_name": {"type": "string"},
                        "vulkan_version": {"type": "string"},
                        "license": {"type": "string"},
                        "glsl_source": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "VramUnlockRequest": {
                    "type": "object",
                    "properties": {
                        "tier": {"type": "string", "enum": ["128MB_CLAMPED", "2GB_UNLOCKED", "4GB_UNLOCKED", "8GB_UNLOCKED"], "default": "4GB_UNLOCKED"}
                    }
                },
                "VramUnlockResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "tier": {"type": "string"},
                        "allocated_vram_gb": {"type": "number"},
                        "host_coherent_virtual_address": {"type": "string"},
                        "pcie_ring_bus_clamping_eliminated": {"type": "boolean"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "LlmTokenBenchmarkRequest": {
                    "type": "object",
                    "properties": {
                        "tier": {"type": "string", "enum": ["128MB_CLAMPED", "2GB_UNLOCKED", "4GB_UNLOCKED", "8GB_UNLOCKED"], "default": "4GB_UNLOCKED"},
                        "quantization": {"type": "string", "enum": ["INT4_GGUF_AWQ", "INT8_VNNI_DP4A", "FP16_HALF", "FP32_NATIVE"], "default": "INT4_GGUF_AWQ"}
                    }
                },
                "LlmTokenBenchmarkResult": {
                    "type": "object",
                    "properties": {
                        "aperture_tier": {"type": "string"},
                        "quantization": {"type": "string"},
                        "allocated_vram_gb": {"type": "number"},
                        "effective_bandwidth_gbps": {"type": "number"},
                        "tokens_per_second": {"type": "number"},
                        "time_to_first_token_ms": {"type": "number"},
                        "token_speedup_vs_clamped": {"type": "number"},
                        "ring_bus_pcie_stalls_per_sec": {"type": "integer"},
                        "junction_temp_c": {"type": "number"},
                        "voltage_core_v": {"type": "number"},
                        "voltage_gpu_gt_v": {"type": "number"},
                        "projected_lifespan_years": {"type": "number"},
                        "vital_max_hp": {"type": "integer", "default": 6},
                        "timestamp_iso": {"type": "string"}
                    }
                },
                "OptimalProcessPlan": {
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "target_throughput_tokens_sec": {"type": "number"},
                        "recommended_aperture": {"type": "string"},
                        "recommended_quantization": {"type": "string"},
                        "balanced_vcore_v": {"type": "number"},
                        "balanced_vgt_v": {"type": "number"},
                        "target_frequency_mhz": {"type": "integer"},
                        "thermal_envelope_c": {"type": "number"},
                        "chip_lifespan_index": {"type": "number"},
                        "safeguard_active": {"type": "boolean"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "KIsaSpeculationReport": {
                    "type": "object",
                    "properties": {
                        "block_id": {"type": "string"},
                        "total_instructions": {"type": "integer"},
                        "stall_events_detected": {"type": "integer"},
                        "stalls_neutralized": {"type": "integer"},
                        "stall_suppression_rate_pct": {"type": "number"},
                        "cumulative_latency_hidden_ms": {"type": "number"},
                        "peak_safe_voltage_v": {"type": "number"},
                        "chip_lifespan_preserved_years": {"type": "number"},
                        "dispatched_entries": {"type": "array", "items": {"type": "object"}},
                        "vital_max_hp": {"type": "integer", "default": 6},
                        "timestamp_iso": {"type": "string"}
                    }
                },
                "GodotRenderPackageResponse": {
                    "type": "object",
                    "properties": {
                        "engine_name": {"type": "string"},
                        "engine_version": {"type": "string"},
                        "integration_script": {"type": "string"},
                        "viewport_shader": {"type": "string"},
                        "rest_hub_url": {"type": "string"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "BytecodeDispatchResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "instructions_processed": {"type": "integer"},
                        "dispatched_passes_count": {"type": "integer"},
                        "passes": {"type": "array", "items": {"type": "object"}},
                        "predictions_count": {"type": "integer"},
                        "predictions": {"type": "array", "items": {"type": "object"}},
                        "total_gpu_duration_ms": {"type": "number"},
                        "effective_framerate_fps": {"type": "number"},
                        "render_latency_ms": {"type": "number"},
                        "uma_allocated_mb": {"type": "number"},
                        "vital_max_hp_verified": {"type": "boolean"},
                        "vital_max_hp": {"type": "integer", "default": 6}
                    }
                },
                "GraphicsAccelerationBenchmarkReport": {
                    "type": "object",
                    "properties": {
                        "benchmark_id": {"type": "string"},
                        "baseline_stock_fps": {"type": "number"},
                        "accelerated_fps": {"type": "number"},
                        "fps_increase_pct": {"type": "number"},
                        "baseline_1pct_low_fps": {"type": "number"},
                        "accelerated_1pct_low_fps": {"type": "number"},
                        "low_fps_increase_pct": {"type": "number"},
                        "baseline_latency_ms": {"type": "number"},
                        "accelerated_latency_ms": {"type": "number"},
                        "latency_reduction_factor": {"type": "number"},
                        "raw_compute_tops_int8": {"type": "number"},
                        "raw_compute_gflops_cpu": {"type": "number"},
                        "component_assistance": {"type": "object"},
                        "vital_max_hp": {"type": "integer", "default": 6},
                        "timestamp_iso": {"type": "string"}
                    }
                }
            }
        }
    }


def render_swagger_ui_html() -> str:
    """Generates an embedded, self-contained Swagger UI HTML page."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Krystal Stack // OpenAPI 3.1.0 Interactive Explorer</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    body {
      margin: 0;
      background: #090d16;
      color: #e2e8f0;
      font-family: 'Inter', sans-serif;
    }
    .top-header {
      background: #04060c;
      padding: 16px 32px;
      border-bottom: 1px solid rgba(0, 240, 255, 0.2);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .top-header h1 {
      font-size: 1.15rem;
      margin: 0;
      color: #00f0ff;
      font-family: 'Fira Code', monospace;
    }
    .badge {
      background: rgba(0, 255, 136, 0.15);
      border: 1px solid #00ff88;
      color: #00ff88;
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-family: 'Fira Code', monospace;
    }
    .swagger-ui {
      filter: invert(88%) hue-rotate(180deg);
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }
  </style>
</head>
<body>
  <div class="top-header">
    <h1>⬡ KRYSTAL // CORTEX & OPENVINO OPENAPI 3.1.0</h1>
    <span class="badge">VITAL_MAX_HP = 6</span>
  </div>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => {
      SwaggerUIBundle({
        url: '/api/openapi.json',
        dom_id: '#swagger-ui',
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIBundle.SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout",
        deepLinking: true
      });
    };
  </script>
</body>
</html>
"""


generate_openapi_spec = get_openapi_specification


if __name__ == "__main__":
    spec = get_openapi_specification()
    print(f"Generated OpenAPI 3.1.0 Specification with {len(spec['paths'])} endpoints and {len(spec['components']['schemas'])} schemas.")
    assert spec["components"]["schemas"]["CortexIntegrityVerdict"]["properties"]["vital_max_hp"]["default"] == 6
