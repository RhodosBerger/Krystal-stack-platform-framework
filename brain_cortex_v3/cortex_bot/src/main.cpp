/**
 * main.cpp — Brain Cortex V3 — Bot Entry Point
 * ============================================================
 * Wires up VulkanGrid + HW Governor + OpenVINO Agent → CortexBot.
 *
 * Build (without OpenVINO):
 *   g++ -std=c++17 -O2 -DCORTEX_NO_OPENVINO \
 *       -I../include main.cpp -o cortex_bot
 *
 * Build (with OpenVINO 2024):
 *   g++ -std=c++17 -O2 \
 *       -I/opt/intel/openvino/runtime/include \
 *       -I../include main.cpp \
 *       -L/opt/intel/openvino/runtime/lib/intel64 \
 *       -lopenvino -o cortex_bot
 *
 * Custom kernel advice:
 *   Compile kernel with CONFIG_HZ=1000 and PREEMPT_RT patch.
 *   Set intel_pstate=passive in kernel cmdline:
 *     GRUB_CMDLINE_LINUX="intel_pstate=passive"
 *   This gives the bot full CPU freq control via scaling_max_freq.
 *
 * Run:
 *   sudo ./cortex_bot --ticks 100 --simulate
 *   sudo ./cortex_bot --ticks 300 --model /path/to/model.xml
 */

#include "cortex_bot.hpp"
#include <iostream>
#include <string>
#include <cstring>

static void print_usage(const char* prog) {
    std::cout
        << "\nUsage: " << prog << " [options]\n"
        << "  --ticks N        Number of control ticks (default 200)\n"
        << "  --tick-ms N      Tick period in ms (default 8 = ~125Hz)\n"
        << "  --model PATH     Path to OpenVINO model.xml (default: model.xml)\n"
        << "  --simulate       Force simulation mode (no real sysfs writes)\n"
        << "  --target-ms F    Latency target in ms (default 20)\n"
        << "  --no-virtual-npu Disable virtual NPU thread pinning\n"
        << "\nExamples:\n"
        << "  sudo ./cortex_bot --ticks 500 --tick-ms 8\n"
        << "  ./cortex_bot --simulate --ticks 50\n\n";
}

int main(int argc, char** argv) {
    cortex::BotConfig cfg;
    int  max_ticks  = 60;
    bool simulate   = false;
    std::string model_path = "model.xml";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        } else if (std::strcmp(argv[i], "--simulate") == 0) {
            simulate = true;
        } else if (std::strcmp(argv[i], "--ticks") == 0 && i+1 < argc) {
            max_ticks = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--tick-ms") == 0 && i+1 < argc) {
            cfg.tick_ms = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--model") == 0 && i+1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--target-ms") == 0 && i+1 < argc) {
            cfg.target_latency_ms = std::stof(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-virtual-npu") == 0) {
            cfg.enable_virtual_npu = false;
        }
    }

    cfg.simulate_vulkan = true; // always in demo; real mode needs VK_LAYER pipe

    std::cout
        << "\n╔═══════════════════════════════════════════════╗\n"
        << "║  BRAIN CORTEX V3 — CortexBot v1.0            ║\n"
        << "║  Intel i5-1135G7 Tiger Lake — Ubuntu          ║\n"
        << "║  Iris Xe 80EU + HETERO:GPU,CPU + INT8 DLBoost ║\n"
        << "╚═══════════════════════════════════════════════╝\n"
        << "  Config: ticks=" << max_ticks
        << "  tick_ms=" << cfg.tick_ms
        << "  target=" << cfg.target_latency_ms << "ms"
        << "  simulate=" << simulate << "\n\n";

    cortex::CortexBot bot(cfg);

    if (!bot.init(model_path)) {
        std::cerr << "[main] Bot init failed.\n";
        return 1;
    }

    bot.run(max_ticks);
    return 0;
}
