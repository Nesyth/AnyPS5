#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_FLOAT32EXACTNESS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_EXECUTION_FLOAT32EXACTNESS_HPP

#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "Recompiler.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace Float32Exactness {

inline void Dispatch(AgcDriver::VulkanDevice& device, const ShaderRecompiler::RecompileRequest& request, std::array<std::uint32_t, 3> groups) {
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, groups[0], groups[1], groups[2], {}, request.shader.codeAddress);
    device.WaitIdle();
}

inline void Recheck(AgcDriver::VulkanDevice& device, const ShaderRecompiler::RecompileRequest& request, std::span<std::uint32_t> output, std::array<std::uint32_t, 3> groups, const std::string& name) {
    const auto proven = device.Target();
    if (proven.float32HardwareFlushOps == 0u && proven.float32NativeUnaryOps == 0u) return;
    const std::vector<std::uint32_t> kept(output.begin(), output.end());
    const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 4> modes{request.context.floatMode, std::nullopt, ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false}, ShaderRecompiler::ShaderFloatMode{0xc0u, true, true, false}};
    std::uint32_t compared = 0u;
    for (std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        if (std::find(modes.begin(), modes.begin() + static_cast<std::ptrdiff_t>(modeIndex), modes[modeIndex]) != modes.begin() + static_cast<std::ptrdiff_t>(modeIndex)) continue;
        auto today = request;
        today.context.floatMode = modes[modeIndex];
        today.target.float32HardwareFlushOps = 0u;
        today.target.float32NativeUnaryOps = 0u;
        try {
            Dispatch(device, today, groups);
        } catch (const std::exception&) {
            if (modeIndex == 0u) throw;
            continue;
        }
        const std::vector<std::uint32_t> expected(output.begin(), output.end());
        auto changed = today;
        changed.target.float32HardwareFlushOps = proven.float32HardwareFlushOps;
        changed.target.float32NativeUnaryOps = proven.float32NativeUnaryOps;
        Dispatch(device, changed, groups);
        ++compared;
        for (std::size_t index = 0; index < output.size(); ++index) {
            if (output[index] == expected[index]) continue;
            char text[224];
            std::snprintf(text, sizeof(text), ": float mode %zu, word %zu is 0x%08x with the f32 exactness fields (hardware flush 0x%x, native unary 0x%03x) and 0x%08x without", modeIndex, index, output[index], proven.float32HardwareFlushOps, proven.float32NativeUnaryOps, expected[index]);
            throw std::runtime_error(name + text);
        }
    }
    std::printf("%s: f32 exactness fields (hardware flush 0x%x, native unary 0x%03x) match today's forms in %u float modes\n", name.c_str(), proven.float32HardwareFlushOps, proven.float32NativeUnaryOps, compared);
    std::copy(kept.begin(), kept.end(), output.begin());
}

}

#endif
