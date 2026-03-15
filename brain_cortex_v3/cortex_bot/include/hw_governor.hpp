// hw_governor.hpp -- Brain Cortex V3
// ============================================================
// Direct Linux sysfs hardware control for Intel i5-1135G7 (Tiger Lake).
//
// Controlled interfaces:
//   CPU P-State (passive mode)
//     /sys/devices/system/cpu/intel_pstate/status        -> "passive"
//     /sys/devices/system/cpu/cpuN/cpufreq/scaling_governor
//     /sys/devices/system/cpu/cpuN/cpufreq/scaling_min_freq
//     /sys/devices/system/cpu/cpuN/cpufreq/scaling_max_freq
//
//   Intel Iris Xe GPU (i915 DRM)
//     /sys/class/drm/card0/gt_min_freq_mhz    <- bot writes
//     /sys/class/drm/card0/gt_max_freq_mhz    <- bot writes
//     /sys/class/drm/card0/gt_cur_freq_mhz    <- bot reads
//     /sys/class/drm/card0/gt_boost_freq_mhz
//
//   Intel RAPL (package energy cap)
//     /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw
//     /sys/class/powercap/intel-rapl:0/constraint_1_power_limit_uw
//
//   Thermal zones (milli-Celsius)
//     /sys/class/thermal/thermal_zoneN/temp
//
// IMPORTANT: Writing to gt_min/max_freq_mhz requires CAP_SYS_ADMIN.
//   sudo setcap cap_sys_admin+ep ./cortex_bot
//
// i5-1135G7 Iris Xe 80EU: 400-1300 MHz
// i5-1135G7 CPU: 2400 MHz base, 4200 MHz boost
// Tiger Lake shared TDP: PBP1=28W, PBP2=44W

#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <iostream>
#include <cstdint>
#include <optional>

namespace cortex {

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────
// Hardware limits for i5-1135G7 Tiger Lake
// ─────────────────────────────────────────────────────────────
struct TigerLakeSpec {
    // CPU
    static constexpr int CPU_CORES         = 4;
    static constexpr int CPU_THREADS       = 8;
    static constexpr int CPU_BASE_MHZ      = 2400;
    static constexpr int CPU_BOOST_MHZ     = 4200;
    static constexpr int CPU_ECO_MHZ       = 800;

    // Iris Xe 80EU GPU
    static constexpr int GPU_MIN_MHZ       = 400;
    static constexpr int GPU_BASE_MHZ      = 900;
    static constexpr int GPU_MAX_MHZ       = 1300;
    static constexpr int GPU_ECO_MHZ       = 500;

    // RAPL (microwatts)
    static constexpr uint64_t TDP_PBP1_UW  = 28'000'000;  // 28W sustained
    static constexpr uint64_t TDP_PBP2_UW  = 44'000'000;  // 44W burst
    static constexpr uint64_t TDP_ECO_UW   = 15'000'000;  // 15W battery eco

    // Thermal throttle threshold (milli-Celsius)
    static constexpr int THROTTLE_TEMP_MC  = 85'000;
};

// ─────────────────────────────────────────────────────────────
class HardwareGovernor {
public:
    // Simulation mode: no actual sysfs writes, just log intent
    explicit HardwareGovernor(bool simulate = false)
        : simulate_(simulate) {
        if (!simulate_) {
            // Detect whether we have write access to the key paths
            simulate_ = !can_write("/sys/class/drm/card0/gt_min_freq_mhz");
            if (simulate_)
                std::cerr << "[HWGov] No sysfs write access → simulation mode\n";
        }
        // Switch to passive P-State mode so bot has full control
        if (!simulate_)
            write_sysfs("/sys/devices/system/cpu/intel_pstate/status", "passive");
    }

    // ── GPU frequency control (Iris Xe) ──────────────────────
    bool set_gpu_freq_range(int min_mhz, int max_mhz) {
        min_mhz = clamp(min_mhz, TigerLakeSpec::GPU_MIN_MHZ, TigerLakeSpec::GPU_MAX_MHZ);
        max_mhz = clamp(max_mhz, min_mhz,                    TigerLakeSpec::GPU_MAX_MHZ);
        log("GPU freq → min=" + std::to_string(min_mhz) + " max=" + std::to_string(max_mhz));
        bool ok = write_sysfs("/sys/class/drm/card0/gt_min_freq_mhz", std::to_string(min_mhz));
             ok &= write_sysfs("/sys/class/drm/card0/gt_max_freq_mhz", std::to_string(max_mhz));
        return ok;
    }

    // ── CPU frequency control (all cores) ────────────────────
    bool set_cpu_max_freq(int max_mhz) {
        int khz = clamp(max_mhz, TigerLakeSpec::CPU_ECO_MHZ, TigerLakeSpec::CPU_BOOST_MHZ) * 1000;
        bool ok = true;
        for (int cpu = 0; cpu < TigerLakeSpec::CPU_THREADS; ++cpu) {
            std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
                             + "/cpufreq/scaling_max_freq";
            ok &= write_sysfs(path, std::to_string(khz));
        }
        log("CPU max freq → " + std::to_string(max_mhz) + " MHz");
        return ok;
    }

    bool set_cpu_governor(const std::string& gov) {
        bool ok = true;
        for (int cpu = 0; cpu < TigerLakeSpec::CPU_THREADS; ++cpu) {
            std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu)
                             + "/cpufreq/scaling_governor";
            ok &= write_sysfs(path, gov);
        }
        log("CPU governor → " + gov);
        return ok;
    }

    // ── RAPL energy cap ───────────────────────────────────────
    bool set_rapl_package_limit(uint64_t limit_uw) {
        log("RAPL pkg limit → " + std::to_string(limit_uw / 1'000'000) + "W");
        return write_sysfs(
            "/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw",
            std::to_string(limit_uw)
        );
    }

    // ── Named power mode presets ──────────────────────────────
    void apply_power_mode(const std::string& mode) {
        if (mode == "eco") {
            set_gpu_freq_range(TigerLakeSpec::GPU_MIN_MHZ, TigerLakeSpec::GPU_ECO_MHZ);
            set_cpu_max_freq(TigerLakeSpec::CPU_BASE_MHZ);
            set_cpu_governor("powersave");
            set_rapl_package_limit(TigerLakeSpec::TDP_ECO_UW);
        } else if (mode == "overdrive") {
            set_gpu_freq_range(TigerLakeSpec::GPU_BASE_MHZ, TigerLakeSpec::GPU_MAX_MHZ);
            set_cpu_max_freq(TigerLakeSpec::CPU_BOOST_MHZ);
            set_cpu_governor("performance");
            set_rapl_package_limit(TigerLakeSpec::TDP_PBP2_UW);
        } else { // balanced
            set_gpu_freq_range(TigerLakeSpec::GPU_MIN_MHZ, TigerLakeSpec::GPU_BASE_MHZ + 200);
            set_cpu_max_freq(TigerLakeSpec::CPU_BASE_MHZ + 800);
            set_cpu_governor("schedutil");
            set_rapl_package_limit(TigerLakeSpec::TDP_PBP1_UW);
        }
        current_mode_ = mode;
    }

    // ── Power shifting: steal watts from CPU for GPU burst ────
    // On Tiger Lake, CPU and GPU share the same package TDP.
    // By capping CPU max freq we release thermal headroom for
    // the Iris Xe to sustain higher frequency.
    void power_shift_to_gpu(int reduction_mhz = 400) {
        int new_cpu_max = TigerLakeSpec::CPU_BOOST_MHZ - reduction_mhz;
        set_cpu_max_freq(new_cpu_max);
        set_gpu_freq_range(TigerLakeSpec::GPU_BASE_MHZ, TigerLakeSpec::GPU_MAX_MHZ);
        log("Power shifted to GPU: CPU capped at " + std::to_string(new_cpu_max) + " MHz");
    }

    // ── Sensor reads ──────────────────────────────────────────
    std::optional<int> read_cpu_temp_mc() {
        // Returns milli-Celsius, or nullopt if unavailable
        for (int zone = 0; zone < 10; ++zone) {
            std::string path = "/sys/class/thermal/thermal_zone"
                             + std::to_string(zone) + "/temp";
            auto val = read_sysfs_int(path);
            if (val && *val > 10'000) return val; // sanity: > 10°C
        }
        return std::nullopt;
    }

    std::optional<int> read_gpu_cur_freq_mhz() {
        return read_sysfs_int("/sys/class/drm/card0/gt_cur_freq_mhz");
    }

    std::optional<uint64_t> read_rapl_energy_uj() {
        return read_sysfs_u64("/sys/class/powercap/intel-rapl:0/energy_uj");
    }

    bool is_simulate() const { return simulate_; }
    const std::string& mode() const { return current_mode_; }

private:
    bool        simulate_;
    std::string current_mode_ = "balanced";

    static bool can_write(const std::string& path) {
        std::ofstream f(path, std::ios::app);
        return f.good();
    }

    bool write_sysfs(const std::string& path, const std::string& value) {
        if (simulate_) {
            std::cout << "[SIM-HW] " << path << " <- " << value << "\n";
            return true;
        }
        std::ofstream f(path);
        if (!f) {
            std::cerr << "[HWGov] WRITE FAIL: " << path << "\n";
            return false;
        }
        f << value;
        return true;
    }

    static std::optional<int> read_sysfs_int(const std::string& path) {
        std::ifstream f(path);
        if (!f) return std::nullopt;
        int v; f >> v;
        return f ? std::optional<int>{v} : std::nullopt;
    }

    static std::optional<uint64_t> read_sysfs_u64(const std::string& path) {
        std::ifstream f(path);
        if (!f) return std::nullopt;
        uint64_t v; f >> v;
        return f ? std::optional<uint64_t>{v} : std::nullopt;
    }

    static int clamp(int val, int lo, int hi) {
        return val < lo ? lo : (val > hi ? hi : val);
    }

    static void log(const std::string& msg) {
        std::cout << "[HWGov] " << msg << "\n";
    }
};

} // namespace cortex
