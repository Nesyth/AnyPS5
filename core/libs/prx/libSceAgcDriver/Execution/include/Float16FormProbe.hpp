#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FLOAT16FORMPROBE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FLOAT16FORMPROBE_HPP

#include "Recompiler.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace AgcDriver {

class VulkanDevice;

struct Float16FormProbeResult {
    std::uint64_t compared = 0;
    std::uint64_t mismatches = 0;
    std::uint32_t modes = 0;
    std::array<std::uint64_t, 2> byOperation{};
    std::uint64_t groupsCounted = 0;
    std::uint64_t groupsExpected = 0;
};

std::span<const std::uint32_t> Float16MinMaxProbeCode();
ShaderRecompiler::RecompileResult CompileFloat16FormProbe(const ShaderRecompiler::SpirvTarget& target, std::span<const std::uint32_t> code, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode);
Float16FormProbeResult ProbeFloat16NativeMinMax(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, bool native);

}

#endif
