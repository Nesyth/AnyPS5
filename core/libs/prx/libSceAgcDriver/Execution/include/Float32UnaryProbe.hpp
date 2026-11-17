#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FLOAT32UNARYPROBE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FLOAT32UNARYPROBE_HPP

#include "prx/libSceAgcDriver/Execution/include/Float32ProbeProgram.hpp"
#include "Recompiler.hpp"

#include <span>

namespace AgcDriver {

class VulkanDevice;

Float32Probe::Verdict ProbeFloat32NativeUnary(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, std::span<const bool> flushSettings, const Float32Probe::Sweep& sweep);

}

#endif
