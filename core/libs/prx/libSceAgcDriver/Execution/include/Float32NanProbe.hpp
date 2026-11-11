#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FLOAT32NANPROBE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FLOAT32NANPROBE_HPP

#include "Recompiler.hpp"
#include <cstdint>

namespace AgcDriver {

class VulkanDevice;

struct Float32NanProbeResult {
    std::uint32_t mismatches = 0;
    std::uint32_t compared = 0;
};

Float32NanProbeResult ProbeFloat32HostNans(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target);

}

#endif
