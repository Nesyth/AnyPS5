#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWWRITERANGES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWWRITERANGES_HPP

#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ShaderRecompiler {
struct IrResourcePlan;
struct ResourceSnapshot;
}

namespace AgcDriver::DriverDetail {

std::vector<std::pair<std::uint64_t, std::uint64_t>> DrawWriteRanges(const Graphics::State& graphics, std::span<const Graphics::CompiledShader> stages, bool exact);
std::vector<std::pair<std::uint64_t, std::uint64_t>> DispatchWriteRanges(const ShaderRecompiler::RecompileResult& compiled, bool& bounded);
enum class CapturedWrites : std::uint8_t { Named, NoPlan, Width, TableImage };
CapturedWrites CapturedWriteRanges(const ShaderRecompiler::IrResourcePlan* plan, const ShaderRecompiler::ResourceSnapshot& snapshot, std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges);
bool ExactDrawWrites();
bool StagesMayWriteBuffers(std::span<const Graphics::CompiledShader> stages);
std::vector<std::pair<std::uint64_t, std::uint64_t>> QueuedDrawReadRanges(const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> stages);

}

#endif
