/**
 * vulkan_grid.hpp — Brain Cortex V3
 * ============================================================
 * Reads Vulkan telemetry from VK_LAYER_LUNARG_monitor JSON
 * output (or simulated data) and maps it into a 2D float
 * tensor (VulkanGrid) that the bot can reason over.
 *
 * Grid layout (rows = time steps, cols = metrics):
 *   [0] gpu_busy_pct       0..100
 *   [1] pipeline_stalls    0..100  (% of frames with stalls)
 *   [2] vram_throughput_gb 0..N    (GB/s, unified for Tiger Lake)
 *   [3] draw_calls         normalised 0..1
 *   [4] compute_invocations  normalised 0..1
 *   [5] frame_time_ms      raw ms
 *   [6] tdr_flag           0 or 1
 *
 * On Tiger Lake (i5-1135G7), VRAM = system RAM (LPDDR4X up to
 * 51.2 GB/s), so vram_throughput reflects L3/RAM bandwidth.
 *
 * Data source:
 *   VK_LAYER_LUNARG_monitor writes GPU overlay metrics to stderr
 *   when enabled.  We parse its JSON‑like output or fall back
 *   to /sys/class/drm/card0/gt_cur_freq_mhz + perf events.
 *
 * Usage:
 *   VulkanGrid grid(16);           // 16-row sliding window
 *   grid.ingest_log_line(line);    // from popen() pipe
 *   auto tensor = grid.tensor();   // row-major float array
 *   auto advice = grid.advice();   // quick heuristic
 */

#pragma once
#include <array>
#include <deque>
#include <string>
#include <sstream>
#include <regex>
#include <random>
#include <chrono>
#include <cmath>
#include <optional>
#include <iostream>

namespace cortex {

// ── Metric indices ────────────────────────────────────────────
constexpr int METRIC_GPU_BUSY        = 0;
constexpr int METRIC_PIPELINE_STALL  = 1;
constexpr int METRIC_VRAM_TPUT_GB    = 2;
constexpr int METRIC_DRAW_CALLS_NORM = 3;
constexpr int METRIC_COMPUTE_NORM    = 4;
constexpr int METRIC_FRAME_TIME_MS   = 5;
constexpr int METRIC_TDR_FLAG        = 6;
constexpr int METRIC_COUNT           = 7;

using MetricRow = std::array<float, METRIC_COUNT>;

// ── Advice emitted by the grid ────────────────────────────────
enum class GridAdvice {
    CPU_ONLY,           // GPU barely used → offload to CPU
    GPU_LATENCY,        // GPU underutilised → push work, low latency
    NPU_BATCH,          // heavy compute shaders → NPU batch mode
    HETERO_GPU_CPU,     // stalled GPU → hybrid split
    THROTTLE_ECO,       // TDR or thermal → emergency ECO
    BALANCED,           // nominal
};

inline std::string advice_str(GridAdvice a) {
    switch (a) {
        case GridAdvice::CPU_ONLY:      return "CPU_ONLY";
        case GridAdvice::GPU_LATENCY:   return "GPU_LATENCY";
        case GridAdvice::NPU_BATCH:     return "NPU_BATCH";
        case GridAdvice::HETERO_GPU_CPU:return "HETERO:GPU,CPU";
        case GridAdvice::THROTTLE_ECO:  return "THROTTLE_ECO";
        default:                        return "BALANCED";
    }
}

// ── The Vulkan Grid class ─────────────────────────────────────
class VulkanGrid {
public:
    explicit VulkanGrid(size_t window = 16) : window_(window) {
        rng_.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    // ── Ingest one log line (VK_LAYER_LUNARG_monitor format) ──
    // Example line:
    //  {"frame":1042,"gpu_busy":72,"stalls":5,"vram_gb":18.3,
    //   "dc":650,"compute":120,"ft_ms":14.2,"tdr":0}
    bool ingest_log_line(const std::string& line) {
        MetricRow row{};
        try {
            row[METRIC_GPU_BUSY]        = extract_float(line, "gpu_busy",  0.0f);
            row[METRIC_PIPELINE_STALL]  = extract_float(line, "stalls",    0.0f);
            row[METRIC_VRAM_TPUT_GB]    = extract_float(line, "vram_gb",   0.0f);
            float dc                    = extract_float(line, "dc",        0.0f);
            float cmp                   = extract_float(line, "compute",   0.0f);
            row[METRIC_DRAW_CALLS_NORM] = std::min(1.0f, dc   / 5000.0f);
            row[METRIC_COMPUTE_NORM]    = std::min(1.0f, cmp  / 2000.0f);
            row[METRIC_FRAME_TIME_MS]   = extract_float(line, "ft_ms",     0.0f);
            row[METRIC_TDR_FLAG]        = extract_float(line, "tdr",       0.0f);
        } catch (...) {
            return false;
        }
        push(row);
        return true;
    }

    // ── Simulate a frame (when no real Vulkan hook is running) ─
    void simulate_frame(const std::string& scenario = "normal") {
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        MetricRow row{};
        if (scenario == "menu") {
            row = {15.0f, 1.0f, 4.0f, 0.01f, 0.0f, 8.0f, 0.0f};
        } else if (scenario == "battle") {
            row = {85.0f, 30.0f, 32.0f, 0.7f, 0.6f, 28.0f, 0.0f};
        } else if (scenario == "crisis") {
            bool tdr = (uni(rng_) < 0.12f);
            row = {99.0f, 70.0f, 48.0f, 1.0f, 1.0f, 120.0f, tdr ? 1.0f : 0.0f};
        } else { // normal
            row = {
                45.0f + uni(rng_)*30.0f,  // gpu_busy 45–75
                5.0f  + uni(rng_)*15.0f,  // stalls 5–20
                12.0f + uni(rng_)*10.0f,  // vram 12–22 GB/s
                0.2f  + uni(rng_)*0.4f,   // draw_calls
                0.0f  + uni(rng_)*0.2f,   // compute
                12.0f + uni(rng_)*8.0f,   // frame_time 12–20ms
                0.0f
            };
        }
        push(row);
    }

    // ── Return row-major tensor (rows × METRIC_COUNT) ─────────
    std::vector<float> tensor() const {
        std::vector<float> out;
        out.reserve(rows_.size() * METRIC_COUNT);
        for (const auto& r : rows_)
            for (float v : r) out.push_back(v);
        return out;
    }

    // ── Column mean over the sliding window ───────────────────
    MetricRow mean() const {
        MetricRow acc{};
        if (rows_.empty()) return acc;
        for (const auto& r : rows_)
            for (int i = 0; i < METRIC_COUNT; ++i)
                acc[i] += r[i];
        for (float& v : acc) v /= static_cast<float>(rows_.size());
        return acc;
    }

    // ── Heuristic advice from the current window ───────────────
    GridAdvice advice() const {
        if (rows_.empty()) return GridAdvice::BALANCED;
        const auto m = mean();
        const auto& last = rows_.back();

        if (last[METRIC_TDR_FLAG] > 0.5f)             return GridAdvice::THROTTLE_ECO;
        if (m[METRIC_COMPUTE_NORM] > 0.5f)            return GridAdvice::NPU_BATCH;
        if (m[METRIC_PIPELINE_STALL] > 40.0f)         return GridAdvice::HETERO_GPU_CPU;
        if (m[METRIC_GPU_BUSY] > 80.0f)               return GridAdvice::GPU_LATENCY;
        if (m[METRIC_GPU_BUSY] < 15.0f)               return GridAdvice::CPU_ONLY;
        return GridAdvice::BALANCED;
    }

    size_t size() const { return rows_.size(); }

private:
    size_t window_;
    std::deque<MetricRow> rows_;
    std::mt19937 rng_;

    void push(const MetricRow& r) {
        rows_.push_back(r);
        if (rows_.size() > window_) rows_.pop_front();
    }

    static float extract_float(const std::string& s,
                                const std::string& key,
                                float def) {
        // Minimal JSON‑key extractor — no full JSON dep needed
        std::string pattern = "\"" + key + "\"\\s*:\\s*([0-9.]+)";
        std::regex  re(pattern);
        std::smatch m;
        if (std::regex_search(s, m, re))
            return std::stof(m[1].str());
        return def;
    }
};

} // namespace cortex
