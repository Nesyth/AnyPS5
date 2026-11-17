#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_ROBUSTLOADPROBE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_ROBUSTLOADPROBE_HPP

#include "Recompiler.hpp"

#include <cstdint>
#include <optional>

namespace AgcDriver {

class VulkanDevice;

struct RobustLoadMismatch {
    std::uint32_t range = 0;
    std::uint32_t offset = 0;
    std::uint32_t outOfRangeMode = 0;
    bool scalar = false;
    bool rangeEndsBuffer = false;
    std::uint32_t today = 0;
    std::uint32_t exact = 0;
};

struct RobustLoadProbeResult {
    std::uint32_t compared = 0;
    std::uint32_t mismatches = 0;
    std::optional<RobustLoadMismatch> first;
};

RobustLoadProbeResult ProbeRobustLoads(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target);

}

#endif
