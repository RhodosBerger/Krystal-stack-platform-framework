/**
 * openvino_agent.hpp — Brain Cortex V3
 * ============================================================
 * OpenVINO 2024.x C++ API wrapper for the Cortex Bot.
 *
 * Encapsulates ov::Core, model compilation, async inference
 * requests, and dynamic device/precision/hint switching.
 *
 * Supported devices on i5-1135G7:
 *   CPU       — always available, DL Boost VNNI (INT8/BF16)
 *   GPU.0     — Iris Xe 80EU (FP16/FP32 via OpenCL/LevelZero)
 *   AUTO      — OpenVINO AUTO plugin (latency-optimised)
 *   HETERO:GPU,CPU  — heterogeneous split
 *
 * Note: 11th Gen does NOT have a discrete NPU.
 *       The GNA mentioned in Intel docs for 11th Gen is a
 *       weak co-processor for audio/speech only (GNA 3.0).
 *       We correctly map "NPU" to the HETERO fallback here.
 *
 * Build requirement:
 *   find_package(OpenVINO REQUIRED)
 *   target_link_libraries(cortex_bot PRIVATE openvino::runtime)
 *
 * If OpenVINO is NOT installed we compile in stub mode
 * (CORTEX_NO_OPENVINO=1), which uses synthetic latencies.
 */

#pragma once
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>

// ── Try to include real OpenVINO ──────────────────────────────
#ifndef CORTEX_NO_OPENVINO
  #if __has_include(<openvino/openvino.hpp>)
    #include <openvino/openvino.hpp>
    #define CORTEX_OV_AVAILABLE 1
  #else
    #define CORTEX_OV_AVAILABLE 0
  #endif
#else
  #define CORTEX_OV_AVAILABLE 0
#endif

namespace cortex {

// ─────────────────────────────────────────────────────────────
struct InferRequest {
    std::string device;      // actual device used
    std::string precision;   // f32 | f16 | bf16 | int8
    std::string hint;        // LATENCY | THROUGHPUT
    int         num_streams;
};

struct InferResult {
    float       latency_ms;
    float       throughput_fps;
    std::string device_used;
    bool        ok;
};

// ─────────────────────────────────────────────────────────────
class OpenVINOAgent {
public:
    explicit OpenVINOAgent(bool simulate = false)
        : simulate_(simulate)
    {
#if CORTEX_OV_AVAILABLE
        if (!simulate_) {
            try {
                core_ = std::make_unique<ov::Core>();
                auto devs = core_->get_available_devices();
                std::cout << "[OV] Available devices: ";
                for (auto& d : devs) std::cout << d << " ";
                std::cout << "\n";
            } catch (const std::exception& e) {
                std::cerr << "[OV] Init failed: " << e.what() << " → sim mode\n";
                simulate_ = true;
            }
        }
#else
        simulate_ = true;
        std::cout << "[OV] OpenVINO not found → stub mode\n";
#endif
        rng_.seed(42);
    }

    // ── Load & compile a model onto a device ─────────────────
    bool load_model(const std::string& model_path,
                    const std::string& device,
                    const std::string& hint = "LATENCY",
                    const std::string& precision = "bf16",
                    int num_streams = 1)
    {
        current_device_    = resolve_device(device);
        current_precision_ = precision;
        current_hint_      = hint;
        current_streams_   = num_streams;

#if CORTEX_OV_AVAILABLE
        if (!simulate_) {
            try {
                ov::AnyMap props;
                props[ov::hint::performance_mode.name()] =
                    (hint == "THROUGHPUT") ? ov::hint::PerformanceMode::THROUGHPUT
                                           : ov::hint::PerformanceMode::LATENCY;
                props[ov::inference_num_threads.name()] = num_streams * 2;

                // Precision mapping
                if (precision == "f16")
                    props["INFERENCE_PRECISION_HINT"] = "f16";
                else if (precision == "int8" || precision == "bf16")
                    props["ENFORCE_BF16"] = (precision == "bf16") ? "YES" : "NO";

                auto model    = core_->read_model(model_path);
                compiled_     = std::make_shared<ov::CompiledModel>(
                    core_->compile_model(model, current_device_, props));
                infer_req_    = std::make_shared<ov::InferRequest>(
                    compiled_->create_infer_request());

                std::cout << "[OV] Loaded " << model_path
                          << " → " << current_device_
                          << " (" << precision << ", " << hint << ")\n";
                return true;
            } catch (const std::exception& e) {
                std::cerr << "[OV] Load failed: " << e.what() << "\n";
                return false;
            }
        }
#endif
        // Stub mode
        std::cout << "[OV-SIM] Loaded " << model_path
                  << " → " << current_device_
                  << " (" << precision << ", " << hint << ")\n";
        model_loaded_ = true;
        return true;
    }

    // ── Run one synchronous inference pass ───────────────────
    InferResult infer(const std::vector<float>& input) {
        auto t0 = std::chrono::high_resolution_clock::now();

#if CORTEX_OV_AVAILABLE
        if (!simulate_ && infer_req_) {
            try {
                auto tensor = ov::Tensor(ov::element::f32,
                    {1, input.size()},
                    const_cast<float*>(input.data()));
                infer_req_->set_input_tensor(tensor);
                infer_req_->infer();
            } catch (const std::exception& e) {
                std::cerr << "[OV] Infer error: " << e.what() << "\n";
                return {999.0f, 0.0f, current_device_, false};
            }
        }
#endif

        auto t1 = std::chrono::high_resolution_clock::now();
        float real_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // In stub: synthesise realistic latency per device+precision
        float latency_ms = simulate_ ? synth_latency() : real_ms;
        float fps = 1000.0f / std::max(latency_ms, 0.1f);

        return {latency_ms, fps, current_device_, true};
    }

    // ── Dynamic re-configuration (no model reload needed) ────
    // OpenVINO supports changing properties on a running compiled
    // model via set_property() on the InferRequest (OV 2024+).
    void set_hint(const std::string& hint) {
        current_hint_ = hint;
#if CORTEX_OV_AVAILABLE
        if (!simulate_ && compiled_) {
            compiled_->set_property(
                {{ov::hint::performance_mode.name(),
                  hint == "THROUGHPUT"
                    ? ov::hint::PerformanceMode::THROUGHPUT
                    : ov::hint::PerformanceMode::LATENCY}});
        }
#endif
        std::cout << "[OV] Hint → " << hint << "\n";
    }

    // ── Switch to HETERO split (GPU offload heavy layers) ─────
    void switch_to_hetero() {
        current_device_ = "HETERO:GPU,CPU";
        std::cout << "[OV] Switched to HETERO:GPU,CPU\n";
        // In real usage: reload model on HETERO device
    }

    // ── Getters ───────────────────────────────────────────────
    const std::string& device()    const { return current_device_;    }
    const std::string& precision() const { return current_precision_; }
    const std::string& hint()      const { return current_hint_;      }

private:
    bool        simulate_;
    bool        model_loaded_ = false;
    std::string current_device_    = "CPU";
    std::string current_precision_ = "bf16";
    std::string current_hint_      = "LATENCY";
    int         current_streams_   = 1;
    std::mt19937 rng_;

#if CORTEX_OV_AVAILABLE
    std::unique_ptr<ov::Core>                 core_;
    std::shared_ptr<ov::CompiledModel>        compiled_;
    std::shared_ptr<ov::InferRequest>         infer_req_;
#endif

    // Realistic latency table for Tiger Lake (stub mode)
    float synth_latency() {
        float base = 25.0f;
        if      (current_device_ == "GPU.0")          base = 9.0f;
        else if (current_device_ == "HETERO:GPU,CPU") base = 12.0f;
        else if (current_device_ == "AUTO")            base = 16.0f;
        else                                           base = 22.0f; // CPU

        if (current_precision_ == "int8") base *= 0.55f;
        else if (current_precision_ == "bf16") base *= 0.75f;
        else if (current_precision_ == "f16") base *= 0.65f;

        std::normal_distribution<float> nd(base, base * 0.1f);
        return std::max(1.0f, nd(rng_));
    }

    // Resolve logical device name → OpenVINO device string
    static std::string resolve_device(const std::string& d) {
        if (d == "igpu" || d == "gpu")            return "GPU.0";
        if (d == "npu" || d == "hetero")          return "HETERO:GPU,CPU";
        if (d == "auto")                           return "AUTO";
        return "CPU";
    }
};

} // namespace cortex
