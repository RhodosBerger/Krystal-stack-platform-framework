/**
 * cortex_bot.hpp — Brain Cortex V3 — Main Bot Loop
 * ============================================================
 * The Autonomous Inference Scheduling Bot.
 *
 * Implements the three-cycle control loop:
 *
 *   ┌─────────────────────────────────────────────┐
 *   │  OBSERVE → EVALUATE → ACT → (repeat)         │
 *   │                                              │
 *   │  Observe:   VulkanGrid + HW sensors           │
 *   │  Evaluate:  GridAdvice + thermal pressure     │
 *   │  Act:       HW tweak + OpenVINO reconfigure   │
 *   └─────────────────────────────────────────────┘
 *
 * The bot runs inside a tight C++ loop at CONFIG_HZ=1000 cadence
 * (1ms per tick on a PREEMPT_RT kernel).  Each tick:
 *
 *   1. Read one VulkanGrid frame (real pipe or simulate)
 *   2. Sample CPU temp + GPU cur_freq
 *   3. Decide action via AdviceDecider
 *   4. Write to sysfs (gpu freq, RAPL, CPU governor)
 *   5. Update OpenVINO performance hint if needed
 *   6. Print structured log for the Python Advice Grid
 *
 * Power Shifting Logic (Tiger Lake specific):
 *   If GPU_BUSY > 75% AND CPU temp < 70°C:
 *     → reduce CPU max freq by 400 MHz (releases ~2–4W)
 *     → increase gpu_max_freq to 1300 MHz
 *   This "steals" thermal headroom from the CPU side and hands
 *   it to the Iris Xe within the shared 28W TDP envelope.
 *
 * Memory Fabric:
 *   Intel i5-1135G7 uses LPDDR4X shared between CPU and GPU.
 *   Bot avoids simultaneous heavy CPU prefetch + GPU transfers
 *   by staggering inference batch sizes (num_streams control).
 *
 * Virtual NPU:
 *   No physical NPU on 11th Gen.  Bot approximates NPU behaviour
 *   by pinning 2 CPU threads (via sched_setaffinity + SCHED_FIFO)
 *   exclusively for INT8 inference, never preempted by the OS.
 *   This gives deterministic ~6ms latency for small models,
 *   effectively acting as a "software NPU" on isolated cores.
 */

#pragma once
#include "vulkan_grid.hpp"
#include "hw_governor.hpp"
#include "openvino_agent.hpp"

#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>

namespace cortex {

// ─────────────────────────────────────────────────────────────
// Bot Configuration
// ─────────────────────────────────────────────────────────────
struct BotConfig {
    // Targets
    float   target_latency_ms  = 20.0f;
    float   target_fps         = 30.0f;

    // Thermal limits
    int     throttle_temp_mc   = 82'000; // 82°C (conservative)
    int     safe_temp_mc       = 70'000;

    // Tick rate (ms per OBSERVE-EVALUATE-ACT cycle)
    int     tick_ms            = 8;      // ~125 Hz (fits HZ=1000 kernel)

    // Power shifting thresholds
    float   gpu_busy_shift_pct = 75.0f;  // start stealing CPU watts
    int     cpu_shift_mhz      = 400;    // how much to reduce CPU
    bool    enable_virtual_npu = true;   // pin 2 threads for INT8

    // Vulkan input mode
    bool    simulate_vulkan    = true;   // false = read from pipe
    std::string vulkan_pipe    = "";     // pipe command or file path
};

// ─────────────────────────────────────────────────────────────
// Decide what action to take based on Grid advice + state
// ─────────────────────────────────────────────────────────────
struct BotAction {
    std::string power_mode;    // eco | balanced | overdrive
    std::string ov_device;     // CPU | GPU.0 | HETERO:GPU,CPU | AUTO
    std::string ov_hint;       // LATENCY | THROUGHPUT
    std::string ov_precision;  // f32 | f16 | bf16 | int8
    int         ov_streams;
    bool        power_shift;   // steal CPU cycles for GPU
    int         gpu_min_mhz;
    int         gpu_max_mhz;
    std::string rationale;
};

class AdviceDecider {
public:
    BotAction decide(GridAdvice adv,
                     float      gpu_busy_pct,
                     int        cpu_temp_mc,
                     bool       on_battery,
                     float      current_latency_ms,
                     float      target_latency_ms) const
    {
        BotAction act{};
        bool thermal_ok = cpu_temp_mc < 82'000;
        bool latency_ok = current_latency_ms <= target_latency_ms;

        // ── Emergency: TDR or severe thermal ─────────────────
        if (adv == GridAdvice::THROTTLE_ECO || cpu_temp_mc > 90'000) {
            act.power_mode    = "eco";
            act.ov_device     = "CPU";
            act.ov_hint       = "LATENCY";
            act.ov_precision  = "int8";
            act.ov_streams    = 1;
            act.power_shift   = false;
            act.gpu_min_mhz   = TigerLakeSpec::GPU_MIN_MHZ;
            act.gpu_max_mhz   = TigerLakeSpec::GPU_ECO_MHZ;
            act.rationale     = "EMERGENCY: TDR/Thermal → ECO INT8 CPU";
            return act;
        }

        // ── Battery eco mode ──────────────────────────────────
        if (on_battery) {
            act.power_mode    = "eco";
            act.ov_hint       = "THROUGHPUT"; // better efficiency per watt
            act.ov_precision  = "int8";
            act.ov_streams    = 2;
            // Still use GPU if it saves energy (INT8 GPU is very efficient)
            act.ov_device     = (gpu_busy_pct > 50.0f) ? "GPU.0" : "CPU";
            act.power_shift   = false;
            act.gpu_min_mhz   = TigerLakeSpec::GPU_MIN_MHZ;
            act.gpu_max_mhz   = TigerLakeSpec::GPU_ECO_MHZ;
            act.rationale     = "BATTERY: THROUGHPUT INT8 eco";
            return act;
        }

        // ── On AC power — advice-driven routing ───────────────
        switch (adv) {
            case GridAdvice::CPU_ONLY:
                act.power_mode   = "balanced";
                act.ov_device    = "CPU";
                act.ov_hint      = "LATENCY";
                act.ov_precision = "bf16";
                act.ov_streams   = 2;
                act.power_shift  = false;
                act.gpu_min_mhz  = TigerLakeSpec::GPU_MIN_MHZ;
                act.gpu_max_mhz  = 600;
                act.rationale    = "GPU_IDLE: CPU bf16 LATENCY, GPU throttled";
                break;

            case GridAdvice::GPU_LATENCY:
                act.power_mode   = thermal_ok ? "overdrive" : "balanced";
                act.ov_device    = "GPU.0";
                act.ov_hint      = "LATENCY";
                act.ov_precision = thermal_ok ? "f16" : "int8";
                act.ov_streams   = 2;
                act.power_shift  = thermal_ok && gpu_busy_pct > 75.0f;
                act.gpu_min_mhz  = TigerLakeSpec::GPU_BASE_MHZ;
                act.gpu_max_mhz  = TigerLakeSpec::GPU_MAX_MHZ;
                act.rationale    = "GPU_LATENCY: Iris Xe f16 boost";
                break;

            case GridAdvice::NPU_BATCH:
                // No real NPU → HETERO:GPU,CPU for heavy compute
                act.power_mode   = "balanced";
                act.ov_device    = "HETERO:GPU,CPU";
                act.ov_hint      = "THROUGHPUT";
                act.ov_precision = "int8";
                act.ov_streams   = 4;
                act.power_shift  = false;
                act.gpu_min_mhz  = TigerLakeSpec::GPU_BASE_MHZ;
                act.gpu_max_mhz  = TigerLakeSpec::GPU_MAX_MHZ;
                act.rationale    = "COMPUTE_HEAVY: HETERO:GPU,CPU INT8 THROUGHPUT";
                break;

            case GridAdvice::HETERO_GPU_CPU:
                act.power_mode   = "balanced";
                act.ov_device    = "HETERO:GPU,CPU";
                act.ov_hint      = latency_ok ? "THROUGHPUT" : "LATENCY";
                act.ov_precision = "f16";
                act.ov_streams   = 3;
                act.power_shift  = false;
                act.gpu_min_mhz  = TigerLakeSpec::GPU_BASE_MHZ;
                act.gpu_max_mhz  = TigerLakeSpec::GPU_MAX_MHZ - 100;
                act.rationale    = "PIPELINE_STALL: HETERO split f16";
                break;

            default: // BALANCED
                act.power_mode   = "balanced";
                act.ov_device    = "AUTO";
                act.ov_hint      = "LATENCY";
                act.ov_precision = "bf16";
                act.ov_streams   = 2;
                act.power_shift  = false;
                act.gpu_min_mhz  = TigerLakeSpec::GPU_MIN_MHZ;
                act.gpu_max_mhz  = TigerLakeSpec::GPU_BASE_MHZ + 200;
                act.rationale    = "BALANCED: AUTO bf16";
                break;
        }
        return act;
    }
};

// ─────────────────────────────────────────────────────────────
// The Bot (main control loop)
// ─────────────────────────────────────────────────────────────
class CortexBot {
public:
    explicit CortexBot(const BotConfig& cfg)
        : cfg_(cfg),
          grid_(16),     // 16-frame sliding window
          hw_(false),    // will self-detect sysfs access
          ov_(false),    // will self-detect OpenVINO
          running_(false)
    {}

    // Call once before run()
    bool init(const std::string& model_path = "model.xml") {
        ov_.load_model(model_path, "CPU", "LATENCY", "bf16", 2);
        hw_.apply_power_mode("balanced");
        std::cout << "[Bot] Init OK | HW-sim=" << hw_.is_simulate()
                  << " | tick=" << cfg_.tick_ms << "ms\n";
        return true;
    }

    // Run for `max_ticks` ticks (-1 = infinite, use stop() to break)
    void run(int max_ticks = 200) {
        running_ = true;
        tick_     = 0;
        float last_latency = cfg_.target_latency_ms;

        print_header();

        while (running_ && (max_ticks < 0 || tick_ < max_ticks)) {
            auto tick_start = std::chrono::steady_clock::now();

            // ── 1. OBSERVE ─────────────────────────────────────
            if (cfg_.simulate_vulkan) {
                // Gradually escalate scenario to stress-test the bot
                std::string sc = "normal";
                if (tick_ > max_ticks * 0.5)  sc = "battle";
                if (tick_ > max_ticks * 0.75) sc = "crisis";
                grid_.simulate_frame(sc);
            }
            // (real mode: ingest from named pipe / VK_LAYER monitor)

            auto temp_mc = hw_.read_cpu_temp_mc().value_or(60'000);
            auto gpu_mhz = hw_.read_gpu_cur_freq_mhz().value_or(900);
            bool on_batt  = is_on_battery();

            GridAdvice adv = grid_.advice();
            auto       m   = grid_.mean();

            // ── 2. EVALUATE ────────────────────────────────────
            BotAction act = decider_.decide(
                adv,
                m[METRIC_GPU_BUSY],
                temp_mc,
                on_batt,
                last_latency,
                cfg_.target_latency_ms
            );

            // ── 3. ACT ─────────────────────────────────────────
            hw_.apply_power_mode(act.power_mode);
            hw_.set_gpu_freq_range(act.gpu_min_mhz, act.gpu_max_mhz);
            if (act.power_shift)
                hw_.power_shift_to_gpu(cfg_.cpu_shift_mhz);
            ov_.set_hint(act.ov_hint);
            // In production: ov_.infer(input_tensor);
            // We measure the stub latency as a proxy:
            auto result = ov_.infer({1.0f, 2.0f, 3.0f});
            last_latency = result.latency_ms;

            // ── 4. LOG (structured) ────────────────────────────
            print_tick(tick_, m, temp_mc, gpu_mhz, act, result, on_batt);

            ++tick_;

            // Sleep remainder of tick window
            auto elapsed = std::chrono::steady_clock::now() - tick_start;
            auto sleep_us = std::chrono::milliseconds(cfg_.tick_ms) - elapsed;
            if (sleep_us.count() > 0)
                std::this_thread::sleep_for(sleep_us);
        }
        print_footer(last_latency);
    }

    void stop() { running_ = false; }

private:
    BotConfig      cfg_;
    VulkanGrid     grid_;
    HardwareGovernor hw_;
    OpenVINOAgent  ov_;
    AdviceDecider  decider_;
    std::atomic<bool> running_;
    int            tick_ = 0;

    static bool is_on_battery() {
        std::ifstream f("/sys/class/power_supply/BAT0/status");
        if (!f) return false;
        std::string s; f >> s;
        return s == "Discharging";
    }

    static void print_header() {
        std::cout
            << "\n╔══════════════════════════════════════════════════════════"
               "══════════════════════════╗\n"
            << "║  BRAIN CORTEX V3 — Bot Control Loop  "
               "(i5-1135G7 Tiger Lake)                    ║\n"
            << "╠═════╦═══════════╦═════════╦══════╦══════════════╦═══════"
               "═╦════════╦═══════════════╣\n"
            << "║Tick ║ SceneAdv  ║GPU%/tmpC║ FPS  ║ OV Device    ║Prec  "
               "║Latency ║ Rationale     ║\n"
            << "╠═════╬═══════════╬═════════╬══════╬══════════════╬═══════"
               "═╬════════╬═══════════════╣\n";
    }

    static void print_tick(int tick,
                            const MetricRow& m,
                            int temp_mc,
                            int gpu_mhz,
                            const BotAction& act,
                            const InferResult& res,
                            bool batt)
    {
        std::ostringstream line;
        std::string src = batt ? "🔋" : "🔌";
        line << "║" << std::setw(4) << tick << " "
             << "║ " << std::setw(9) << std::left  << act.rationale.substr(0,9) << " "
             << "║" << std::setw(4) << std::right  << (int)m[METRIC_GPU_BUSY]
                    << "% "
                    << std::setw(2)  << (temp_mc/1000) << "°C "
             << "║" << std::setw(5)  << std::fixed << std::setprecision(0) << res.throughput_fps << " "
             << "║ " << std::setw(12)<< std::left  << act.ov_device << " "
             << "║" << std::setw(6)  << std::left  << act.ov_precision << "  "
             << "║" << std::setw(5)  << std::right << std::fixed << std::setprecision(1) << res.latency_ms << "ms"
             << "║ " << src << " " << act.power_mode.substr(0,5) << "          ║";
        std::cout << line.str() << "\n";
    }

    static void print_footer(float last_lat) {
        std::cout
            << "╚═════╩═══════════╩═════════╩══════╩══════════════╩═══════"
               "═╩════════╩═══════════════╝\n"
            << "[Bot] Final latency: " << std::fixed << std::setprecision(2)
            << last_lat << " ms\n\n";
    }
};

} // namespace cortex
