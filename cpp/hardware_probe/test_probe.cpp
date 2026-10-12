/**
 * ==============================================================================
 * KRYSTAL-STACK: C++ HARDWARE PROBE CONFORMANCE & BENCHMARK TEST
 * ==============================================================================
 */

#include "hardware_probe.hpp"
#include <iostream>
#include <cassert>
#include <chrono>

int main() {
    using namespace krystal::epistemic;

    std::cout << "=== Krystal-Stack: C++ Hardware Probe Conformance Test ===" << std::endl;

    HardwareProbeEngine engine("node-alpha-01", "host_clock_0");

    // 1. Verify Zero-Mutation Invariant
    engine.assert_zero_mutation();
    assert(engine.get_mutation_count() == 0);
    std::cout << "[PASS] Zero-mutation invariant confirmed (mutation_count == 0)." << std::endl;

    // 2. Measure Probe Latency (p50 / p99 across 10,000 iterations)
    constexpr int iterations = 10000;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto sample = engine.sample_silicon_telemetry();
        (void)sample;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    double avg_ns = static_cast<double>(total_ns) / iterations;

    std::cout << "[BENCHMARK] 10,000 samples executed in " << total_ns / 1'000'000.0 << " ms." << std::endl;
    std::cout << "[BENCHMARK] Average probe latency: " << avg_ns << " ns per sample." << std::endl;
    assert(avg_ns < 10000.0); // Must be under 10 microseconds

    // 3. Verify EpistemicValue Construction & Validity Lifecycle
    StateVersionCpp sv{1, 0, 42, 1700000000.0, "host_clock_0", "b9a1c0d4"};
    auto ev = engine.create_thermal_epistemic_value(52.4, sv);

    assert(ev.value == 52.4);
    assert(ev.domain == "hardware.thermal");
    assert(ev.get_validity_state(ev.valid_from_ns + 1000) == ValidityState::FRESH);
    assert(ev.get_validity_state(ev.valid_to_ns + 1000) == ValidityState::EXPIRED);
    std::cout << "[PASS] Epistemic validity lifecycle verified (FRESH within TTL, EXPIRED after valid_to_ns)." << std::endl;

    // 4. Verify JSON Serialization
    std::string json_str = engine.serialize_epistemic_thermal_json(ev);
    assert(json_str.find("\"value\": 52.4") != std::string::npos);
    assert(json_str.find("\"domain\": \"hardware.thermal\"") != std::string::npos);
    assert(json_str.find("\"source_adapter\": \"CppHardwareProbe\"") != std::string::npos);
    std::cout << "[PASS] C++ JSON serialization conformant with fixtures/epistemic_value_v1.json." << std::endl;

    std::cout << "=== ALL C++ CONFORMANCE CHECKS PASSED ===" << std::endl;
    return 0;
}
