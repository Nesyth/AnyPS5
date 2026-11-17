#include "prx/libSceAgcDriver/Execution/include/Float16ConversionProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"

#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver {
namespace {

using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 32;
constexpr std::uint32_t Halves = 1u << 16u;

alignas(256) constexpr std::array<std::uint32_t, 10> WidenCode{
    0x34020082u, 0x8f098708u, 0x4a020209u, 0xe0301000u, 0x80000401u, 0xbf8c3f70u, 0x7e0a1704u, 0xe0701000u,
    0x80010501u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 10> NarrowCode{
    0x34020082u, 0x8f098708u, 0x4a020209u, 0xe0301000u, 0x80000401u, 0xbf8c3f70u, 0x7e0a1504u, 0xe0701000u,
    0x80010501u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 11> RoundTripCode{
    0x34020082u, 0x8f098708u, 0x4a020209u, 0xe0301000u, 0x80000401u, 0xbf8c3f70u, 0x7e0a1504u, 0x7e0c1705u,
    0xe0701000u, 0x80010601u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 10> NarrowTowardZeroCode{
    0x34020082u, 0x8f098708u, 0x4a020209u, 0xe0301000u, 0x80000401u, 0xbf8c3f70u, 0x5e0a0904u, 0xe0701000u,
    0x80010501u, 0xbf810000u,
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

bool Supported(const ShaderRecompiler::SpirvTarget& target, spv::Capability capability) {
    return std::find(target.supportedCapabilities.begin(), target.supportedCapabilities.end(), static_cast<std::uint32_t>(capability)) != target.supportedCapabilities.end();
}

}

std::vector<std::uint32_t> Float16NarrowProbeInputs() {
    std::vector<std::uint32_t> inputs;
    constexpr std::array<std::uint32_t, 4> nanLow{0x0000u, 0x0001u, 0x1000u, 0x1fffu};
    constexpr std::array<std::uint32_t, 16> mantissas{0x000000u, 0x000001u, 0x000fffu, 0x001000u, 0x001001u, 0x001fffu, 0x002000u, 0x003000u, 0x400000u, 0x7fe000u, 0x7ff000u, 0x7fefffu, 0x7ff001u, 0x7fffffu, 0x155555u, 0x2aaaaau};
    for (const std::uint32_t sign : {0u, 0x80000000u}) {
        for (std::uint32_t top = 0; top < 1024u; ++top) {
            for (const std::uint32_t low : nanLow) {
                if (top != 0u || low != 0u) inputs.push_back(sign | 0x7f800000u | (top << 13u) | low);
            }
        }
        for (std::uint32_t exponent = 0; exponent < 256u; ++exponent) {
            for (const std::uint32_t mantissa : mantissas) inputs.push_back(sign | (exponent << 23u) | mantissa);
        }
        for (std::uint32_t exponent = 100u; exponent < 114u; ++exponent) {
            for (std::uint32_t bit = 0; bit < 23u; ++bit) {
                for (const std::uint32_t mantissa : {1u << bit, (3u << bit) & 0x7fffffu, (1u << bit) - 1u}) inputs.push_back(sign | (exponent << 23u) | mantissa);
            }
        }
    }
    inputs.resize((inputs.size() + Threads - 1u) / Threads * Threads, 0u);
    return inputs;
}

namespace {

std::vector<std::uint32_t> Run(VulkanDevice& device, ShaderRecompiler::SpirvTarget target, bool widen, bool narrow, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode, std::span<const std::uint32_t> code, std::span<const std::uint32_t> input) {
    std::vector<std::uint32_t> output(input.size(), 0xdeadbeefu);
    std::vector<std::uint32_t> userData(8, 0u);
    const auto inputDescriptor = BufferDescriptor(input.data(), static_cast<std::uint32_t>(input.size_bytes()));
    const auto outputDescriptor = BufferDescriptor(output.data(), static_cast<std::uint32_t>(output.size() * 4u));
    std::copy(inputDescriptor.begin(), inputDescriptor.end(), userData.begin());
    std::copy(outputDescriptor.begin(), outputDescriptor.end(), userData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {true, false, false}, false, 1};
    target.float16WidenNans = widen;
    target.float16NarrowNans = narrow;
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    request.context.floatMode = floatMode;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, static_cast<std::uint32_t>(input.size() / Threads), 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    return output;
}

void Compare(Float16ConversionProbeDirection& direction, VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, bool widen, bool narrow, std::span<const std::uint32_t> code, std::span<const std::uint32_t> input) {
    const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 4> modes{std::nullopt, ShaderRecompiler::ShaderFloatMode{0xf0u, true, false, false}, ShaderRecompiler::ShaderFloatMode{0x00u, false, true, false}, ShaderRecompiler::ShaderFloatMode{0xf4u, true, false, true}};
    for (const auto& mode : modes) {
        const auto emulated = Run(device, target, false, false, mode, code, input);
        const auto native = Run(device, target, widen, narrow, mode, code, input);
        for (std::size_t lane = 0; lane < input.size(); ++lane) {
            ++direction.compared;
            if (emulated[lane] == native[lane]) continue;
            ++direction.mismatches;
            if (!direction.first) direction.first = Float16ConversionMismatch{input[lane], emulated[lane], native[lane]};
        }
    }
}

}

Float16ConversionProbeResult ProbeFloat16Conversions(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target) {
    Float16ConversionProbeResult result;
    if (target.float16DenormPreserve) {
        result.widen.probed = true;
        std::vector<std::uint32_t> halves(Halves);
        for (std::uint32_t lane = 0; lane < Halves; ++lane) halves[lane] = lane | ((lane ^ 0xa5a5u) << 16u);
        Compare(result.widen, device, target, true, false, WidenCode, halves);
    }
    if (Supported(target, spv::CapabilityFloat16) && Supported(target, spv::CapabilityRoundingModeRTE) && Supported(target, spv::CapabilityDenormPreserve)) {
        result.narrow.probed = true;
        const bool widen = result.widen.probed && result.widen.mismatches == 0u;
        const auto singles = Float16NarrowProbeInputs();
        Compare(result.narrow, device, target, widen, true, NarrowCode, singles);
        Compare(result.narrow, device, target, widen, true, NarrowTowardZeroCode, singles);
        Compare(result.narrow, device, target, widen, true, RoundTripCode, singles);
    }
    return result;
}

namespace {

constexpr std::uint32_t PackThreads = 256;
constexpr std::uint32_t PackGroups = 4096;
constexpr std::uint32_t PackLanes = PackThreads * PackGroups;
constexpr std::uint32_t PackIterations = 4096;

alignas(256) constexpr std::array<std::uint32_t, 32> PackDifferenceCode{
    0x8f058804u, 0x4a020005u, 0x7e0e0280u, 0x7e1202c1u, 0x7e140281u, 0x34040283u, 0xbe8603ffu, PackIterations,
    0x3a0602ffu, 0x80000000u, 0x5e080701u, 0x7d8a0880u, 0x020a1480u, 0x4a0e0b07u, 0x020c02c1u, 0x26121306u,
    0x4a0202ffu, PackLanes, 0x80868106u, 0xbf078006u, 0xbf85fff3u, 0xe0701000u, 0x80000702u, 0xe0701004u,
    0x80000902u, 0xbe8703ffu, PackLanes * 8u, 0x26100081u, 0x4c101081u, 0xe0c80000u, 0x07000800u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 14> WidenCountedCode{
    0x34020082u, 0x8f0d870cu, 0x4a02020du, 0xe0301000u, 0x80000401u, 0xbf8c3f70u, 0x7e0a1704u, 0xe0701000u,
    0x80010501u, 0x260c0081u, 0x4c0c0c81u, 0xe0c80000u, 0x80020600u, 0xbf810000u,
};

std::vector<std::uint32_t> RunWidenCounted(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode, std::span<const std::uint32_t> input) {
    std::vector<std::uint32_t> output(input.size(), 0xdeadbeefu);
    std::vector<std::uint32_t> counter(1, 0u);
    std::vector<std::uint32_t> userData(12, 0u);
    const auto inputDescriptor = BufferDescriptor(input.data(), static_cast<std::uint32_t>(input.size_bytes()));
    const auto outputDescriptor = BufferDescriptor(output.data(), static_cast<std::uint32_t>(output.size() * 4u));
    const auto counterDescriptor = BufferDescriptor(counter.data(), 4u);
    std::copy(inputDescriptor.begin(), inputDescriptor.end(), userData.begin());
    std::copy(outputDescriptor.begin(), outputDescriptor.end(), userData.begin() + 4);
    std::copy(counterDescriptor.begin(), counterDescriptor.end(), userData.begin() + 8);
    const std::span<const std::uint32_t> code(WidenCountedCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    request.context.floatMode = floatMode;
    const auto result = ShaderRecompiler::Recompile(request);
    const auto groups = static_cast<std::uint32_t>(input.size() / Threads);
    device.Dispatch(result, groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    if (counter[0] != groups) throw std::runtime_error("f16 to f32 widening probe: " + std::to_string(counter[0]) + " of " + std::to_string(groups) + " workgroups counted themselves");
    return output;
}

struct PackTally {
    std::uint64_t mismatches = 0;
    std::optional<std::uint32_t> first;
};

PackTally RunPackDifference(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode, std::vector<std::uint32_t>& output, std::vector<std::pair<std::vector<std::uint32_t>, PackTally>>& seen) {
    std::fill(output.begin(), output.end(), 0xffffffffu);
    output.back() = 0u;
    std::vector<std::uint32_t> userData(4, 0u);
    const auto outputDescriptor = BufferDescriptor(output.data(), static_cast<std::uint32_t>(output.size() * 4u));
    std::copy(outputDescriptor.begin(), outputDescriptor.end(), userData.begin());
    const std::span<const std::uint32_t> code(PackDifferenceCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{PackThreads, 1, 1}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    request.context.floatMode = floatMode;
    const auto result = ShaderRecompiler::Recompile(request);
    auto words = result.spirv.Copy();
    for (const auto& entry : seen) {
        if (entry.first == words) return entry.second;
    }
    device.Dispatch(result, PackGroups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    if (output.back() != PackGroups) throw std::runtime_error("f16 toward-zero narrowing probe: " + std::to_string(output.back()) + " of " + std::to_string(PackGroups) + " workgroups counted themselves");
    PackTally tally;
    for (std::uint32_t lane = 0; lane < PackLanes; ++lane) {
        const auto count = output[lane * 2u];
        const auto first = output[lane * 2u + 1u];
        if (count > PackIterations) throw std::runtime_error("f16 toward-zero narrowing probe: a lane reported more differences than iterations");
        tally.mismatches += count;
        if (count != 0u && (!tally.first || first < *tally.first)) tally.first = first;
    }
    seen.emplace_back(std::move(words), tally);
    return tally;
}

}

Float16ConversionProbeDirection ProbeFloat16NativeWiden(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target) {
    Float16ConversionProbeDirection direction;
    direction.probed = true;
    std::vector<std::uint32_t> halves(Halves);
    for (std::uint32_t lane = 0; lane < Halves; ++lane) halves[lane] = lane | ((lane ^ 0xa5a5u) << 16u);
    auto decoded = target;
    decoded.float16DenormPreserve = false;
    decoded.float16NativeWiden = false;
    auto native = target;
    native.float16NativeWiden = true;
    const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 4> modes{std::nullopt, ShaderRecompiler::ShaderFloatMode{0xf0u, true, false, false}, ShaderRecompiler::ShaderFloatMode{0x00u, false, true, false}, ShaderRecompiler::ShaderFloatMode{0xf4u, true, false, true}};
    for (const auto& mode : modes) {
        const auto emulated = RunWidenCounted(device, decoded, mode, halves);
        const auto unpacked = RunWidenCounted(device, native, mode, halves);
        for (std::size_t lane = 0; lane < halves.size(); ++lane) {
            ++direction.compared;
            if (emulated[lane] == unpacked[lane] && emulated[lane] != 0xdeadbeefu) continue;
            ++direction.mismatches;
            if (!direction.first) direction.first = Float16ConversionMismatch{halves[lane], emulated[lane], unpacked[lane]};
        }
    }
    return direction;
}

Float16PackProbeResult ProbeFloat16PackTowardZero(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target) {
    Float16PackProbeResult result;
    if (!Supported(target, spv::CapabilityFloat16) || !Supported(target, spv::CapabilityRoundingModeRTE) || !Supported(target, spv::CapabilityDenormPreserve)) return result;
    const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 5> modes{std::nullopt, ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false}, ShaderRecompiler::ShaderFloatMode{0xd0u, true, false, false}, ShaderRecompiler::ShaderFloatMode{0xe0u, true, false, false}, ShaderRecompiler::ShaderFloatMode{0xf0u, true, false, false}};
    auto probed = target;
    probed.float16PackTowardZero = ShaderRecompiler::PackTowardZeroHalf2x16 | ShaderRecompiler::PackTowardZeroDifference;
    std::vector<std::uint32_t> output(std::size_t{PackLanes} * 2u + 1u);
    std::vector<std::pair<std::vector<std::uint32_t>, PackTally>> seen;
    for (const auto& mode : modes) {
        const auto tally = RunPackDifference(device, probed, mode, output, seen);
        ++result.modes;
        result.compared += std::uint64_t{PackLanes} * PackIterations;
        result.mismatches += tally.mismatches;
        if (tally.first && (!result.first || *tally.first < *result.first)) result.first = tally.first;
    }
    result.form = result.mismatches == 0u ? ShaderRecompiler::PackTowardZeroHalf2x16 : 0u;
    return result;
}

}
