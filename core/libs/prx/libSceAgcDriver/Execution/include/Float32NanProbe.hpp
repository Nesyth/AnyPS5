#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FLOAT32NANPROBE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FLOAT32NANPROBE_HPP

#include "Recompiler.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace AgcDriver {

class VulkanDevice;

struct Float32NanProbeResult {
    std::uint32_t mismatches = 0;
    std::uint32_t compared = 0;
    std::uint32_t fusedMismatches = 0;
    std::uint32_t fusedCompared = 0;
    std::array<std::uint32_t, 32> byResult{};
    std::array<std::uint32_t, 3> byMode{};
    struct Row {
        std::uint32_t mode = 0;
        std::uint32_t result = 0;
        std::array<std::uint32_t, 3> operands{};
        std::uint32_t selected = 0;
        std::uint32_t host = 0;
    };
    std::vector<Row> rows;
};

struct Float32MinMaxProbeResult {
    std::uint32_t mismatches = 0;
    std::uint32_t compared = 0;
    std::uint32_t skippedModes = 0;
    std::array<std::uint32_t, 32> byResult{};
    std::array<std::uint32_t, 6> byMode{};
    std::vector<Float32NanProbeResult::Row> rows;
};

Float32NanProbeResult ProbeFloat32HostNans(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, bool detail = false, bool fused = false);
Float32MinMaxProbeResult ProbeFloat32NativeMinMax(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, bool detail = false);

}

#endif
