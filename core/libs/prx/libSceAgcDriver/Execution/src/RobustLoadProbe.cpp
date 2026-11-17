#include "prx/libSceAgcDriver/Execution/include/RobustLoadProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver {
namespace {

using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t BufferTail = 4160;
constexpr std::array<std::uint32_t, 17> Ranges{1u, 2u, 3u, 4u, 6u, 8u, 12u, 13u, 16u, 20u, 36u, 64u, 68u, 100u, 260u, 1028u, 4100u};
constexpr std::array<std::uint32_t, 2> OutOfRangeModes{3u, 2u};

alignas(256) constexpr std::array<std::uint32_t, 27> Code{
    0x8f118210u, 0xf4200484u, 0x22000000u, 0xbf8cc07fu, 0xf42004c0u, 0x24000004u, 0x34020082u, 0x8f148710u,
    0x4a040214u, 0xe0301000u, 0x80030302u, 0xbf8c3f70u, 0xe0301000u, 0x80000403u, 0xbf8c0070u, 0x7e0a0213u,
    0x8f158810u, 0x4a0c0215u, 0xe0701004u, 0x80010506u, 0xe0701084u, 0x80010406u, 0x260e0081u, 0x4c0e0e81u,
    0xe0c80000u, 0x80010700u, 0xbf810000u,
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes, std::uint32_t outOfRangeMode) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, (0x31016facu & ~0x30000000u) | (outOfRangeMode << 28u)};
}

std::uint32_t Word(std::uint32_t index) {
    return ((index + 1u) * 0x9e3779b1u) | 1u;
}

std::vector<std::uint32_t> Offsets(std::uint32_t range, std::uint32_t bufferBytes) {
    const std::uint32_t whole = range & ~3u;
    const std::uint32_t up = (range + 3u) & ~3u;
    std::vector<std::uint32_t> offsets{0u, 4u, (range / 2u) & ~3u, whole >= 4u ? whole - 4u : 0u, whole, up, up + 4u, up + 12u, up + 16u, up + 60u, up + 64u, up + 4096u, bufferBytes, bufferBytes + 4u, bufferBytes + 4096u, 0x7ffffffcu, 0xfffffff8u, 0xfffffffcu};
    std::ranges::sort(offsets);
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
    return offsets;
}

void Note(RobustLoadProbeResult& result, std::uint32_t range, std::uint32_t offset, std::uint32_t outOfRangeMode, bool scalar, bool endsBuffer, std::uint32_t today, std::uint32_t exact) {
    ++result.compared;
    if (today == exact) return;
    ++result.mismatches;
    if (!result.first) result.first = RobustLoadMismatch{range, offset, outOfRangeMode, scalar, endsBuffer, today, exact};
}

void RunRange(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, std::uint32_t range, bool endsBuffer, std::uint32_t outOfRangeMode, RobustLoadProbeResult& result) {
    std::vector<std::uint32_t> input(((endsBuffer ? range : range + BufferTail) + 3u) / 4u);
    for (std::uint32_t index = 0; index < input.size(); ++index) input[index] = Word(index);
    const auto bufferBytes = static_cast<std::uint32_t>(input.size() * 4u);
    const auto offsets = Offsets(range, bufferBytes);
    const auto groups = static_cast<std::uint32_t>(offsets.size());
    std::vector<std::uint32_t> scalarTable(groups);
    std::vector<std::uint32_t> vectorTable(std::size_t{groups} * Lanes);
    for (std::uint32_t group = 0; group < groups; ++group) {
        scalarTable[group] = offsets[group] - 4u;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) vectorTable[std::size_t{group} * Lanes + lane] = offsets[(group + lane) % groups];
    }
    std::vector<std::uint32_t> output(1u + std::size_t{groups} * Lanes * 2u);
    const std::array<std::array<std::uint32_t, 4>, 4> descriptors{
        BufferDescriptor(input.data(), range, outOfRangeMode),
        BufferDescriptor(output.data(), static_cast<std::uint32_t>(output.size() * 4u), 3u),
        BufferDescriptor(scalarTable.data(), static_cast<std::uint32_t>(scalarTable.size() * 4u), 3u),
        BufferDescriptor(vectorTable.data(), static_cast<std::uint32_t>(vectorTable.size() * 4u), 3u),
    };
    std::vector<std::uint32_t> userData(16, 0u);
    for (std::size_t slot = 0; slot < descriptors.size(); ++slot) std::copy(descriptors[slot].begin(), descriptors[slot].end(), userData.begin() + static_cast<std::ptrdiff_t>(slot * 4u));
    const std::span<const std::uint32_t> code(Code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0u, {true, false, false}, false, 1};
    auto today = target;
    today.exactRobustLoads = false;
    today.exactRobustLoadsAlign4 = false;
    auto exact = target;
    exact.exactRobustLoads = true;
    exact.exactRobustLoadsAlign4 = true;
    const auto run = [&](const ShaderRecompiler::SpirvTarget& chosen, std::vector<std::uint32_t>& words) {
        std::fill(output.begin(), output.end(), 0xdeadbeefu);
        output[0] = 0u;
        ShaderRecompiler::RecompileRequest request{
            {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
            {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
            chosen,
            {0, 0, 0, 128}
        };
        request.useCache = false;
        request.context.floatMode = ShaderRecompiler::ShaderFloatMode{0xf0u, true, false, false};
        const auto compiled = ShaderRecompiler::Recompile(request);
        words = compiled.spirv.Copy();
        device.Dispatch(compiled, groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
        device.WaitIdle();
        if (output[0] != groups) throw std::runtime_error("exact robust load probe: " + std::to_string(output[0]) + " of " + std::to_string(groups) + " workgroups counted themselves for a range of " + std::to_string(range) + " bytes");
        return std::vector<std::uint32_t>(output.begin() + 1, output.end());
    };
    std::vector<std::uint32_t> todayWords;
    std::vector<std::uint32_t> exactWords;
    const auto checked = run(today, todayWords);
    const auto direct = run(exact, exactWords);
    if (todayWords == exactWords) throw std::runtime_error("exact robust load probe: the exact form compiled to the same module as today's checked loads");
    const auto inside = [&](std::uint32_t offset) { return std::uint64_t{offset} + 4u <= range; };
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint32_t address = scalarTable[group] + 4u;
        const bool scalarInside = address >= scalarTable[group] && inside(address);
        const auto base = std::size_t{group} * Lanes * 2u;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto vectorOffset = vectorTable[std::size_t{group} * Lanes + lane];
            Note(result, range, address, outOfRangeMode, true, endsBuffer, checked[base + lane], direct[base + lane]);
            Note(result, range, vectorOffset, outOfRangeMode, false, endsBuffer, checked[base + Lanes + lane], direct[base + Lanes + lane]);
            if (scalarInside && checked[base + lane] != input[address / 4u]) throw std::runtime_error("exact robust load probe: today's scalar load at byte " + std::to_string(address) + " of a range of " + std::to_string(range) + " bytes did not return the stored word");
            if (inside(vectorOffset) && checked[base + Lanes + lane] != input[vectorOffset / 4u]) throw std::runtime_error("exact robust load probe: today's vector load at byte " + std::to_string(vectorOffset) + " of a range of " + std::to_string(range) + " bytes did not return the stored word");
        }
    }
}

}

RobustLoadProbeResult ProbeRobustLoads(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target) {
    RobustLoadProbeResult result;
    for (const auto range : Ranges) {
        for (const bool endsBuffer : {false, true}) {
            for (const auto outOfRangeMode : OutOfRangeModes) RunRange(device, target, range, endsBuffer, outOfRangeMode, result);
        }
    }
    return result;
}

}
