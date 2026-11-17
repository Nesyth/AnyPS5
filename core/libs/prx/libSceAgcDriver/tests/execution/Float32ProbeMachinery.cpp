#include "prx/libSceAgcDriver/Execution/include/Float32ProbeProgram.hpp"
#include "prx/libSceAgcDriver/Execution/include/Float32UnaryProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/FloatFlushProbe.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string Describe(const AgcDriver::Float32Probe::Verdict& verdict) {
    char text[384];
    std::snprintf(text, sizeof(text), "%u runs, %u programs not compiled, %u runs failed their control counts, sentinels %s, %llu of %llu results differ in today's form and %llu with the form, checksums differ in checks 0x%x, %u dispatches, proven 0x%x", verdict.runs, verdict.notCompiled, verdict.failedRuns, verdict.sentinels ? "found" : "missing", static_cast<unsigned long long>(verdict.referenceMismatches), static_cast<unsigned long long>(verdict.compared), static_cast<unsigned long long>(verdict.mismatches), verdict.checksumDifferences, verdict.dispatches, verdict.proven);
    return text;
}

void RequireMachinery(const std::string& probe, const AgcDriver::Float32Probe::Verdict& verdict, std::uint32_t runs, std::uint32_t chunks) {
    const auto text = Describe(verdict);
    std::printf("%s: %s\n", probe.c_str(), text.c_str());
    if (verdict.notCompiled != 0u || verdict.failedRuns != 0u || verdict.runs != runs || !verdict.sentinels || verdict.referenceMismatches != 0u || verdict.dispatches != runs * 2u * chunks) {
        throw std::runtime_error(probe + ": the probe machinery failed: " + text);
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto target = device->Target();
        const auto sweep = AgcDriver::Float32Probe::EndsSweep();
        const auto chunks = static_cast<std::uint32_t>(sweep.bases.size());
        const std::array<bool, 1> flush{target.float32FlushToZero};
        RequireMachinery("f32 native unary probe", AgcDriver::ProbeFloat32NativeUnary(*device, target, flush, sweep), 3u, chunks);
        RequireMachinery("f32 fract hardware flush probe", AgcDriver::ProbeFloat32HardwareFlushOps(*device, target, sweep), 3u, chunks);
        std::puts("f32 probe machinery tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
