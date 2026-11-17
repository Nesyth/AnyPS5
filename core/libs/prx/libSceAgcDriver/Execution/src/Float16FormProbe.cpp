#include "prx/libSceAgcDriver/Execution/include/Float16FormProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestCodeWords.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver {
namespace {

using namespace GuestCode;
using ShaderRecompiler::ShaderFloatMode;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 256;
constexpr std::uint32_t GroupsX = 4096;
constexpr std::uint32_t GroupsY = 2048;
constexpr std::uint32_t Halves = 2;
constexpr std::uint32_t CounterStride = 4;
constexpr std::uint64_t GroupsPerHalf = static_cast<std::uint64_t>(GroupsX) * GroupsY;

constexpr std::uint32_t PackedAshrI16 = 0x06u;
constexpr std::uint32_t PackedMaxI16 = 0x07u;
constexpr std::uint32_t PackedMinI16 = 0x08u;
constexpr std::uint32_t PackedMin = 0x11u;
constexpr std::uint32_t PackedMax = 0x12u;
constexpr std::uint32_t VectorMov = 0x01u;
constexpr std::uint32_t VectorLshr = 0x16u;
constexpr std::uint32_t VectorAnd = 0x1bu;
constexpr std::uint32_t VectorOr = 0x1cu;
constexpr std::uint32_t VectorXor = 0x1du;
constexpr std::uint32_t VectorAdd = 0x25u;
constexpr std::uint32_t VectorMaxF16 = 0x39u;
constexpr std::uint32_t VectorMinF16 = 0x3au;
constexpr std::uint32_t CompareExecNotEqual = 0xd5u;
constexpr std::uint32_t BitfieldInsert = 0x14au;
constexpr std::uint32_t ShiftLeftOr = 0x36fu;
constexpr std::uint32_t ScalarOr = 0x10u;
constexpr std::uint32_t ScalarShiftLeft = 0x1eu;
constexpr std::uint32_t ScalarMov = 0x03u;
constexpr std::uint32_t BufferAtomicAdd = 0x32u;
constexpr std::uint32_t CounterSgpr = 19u;
constexpr std::uint32_t CounterVgpr = 37u;

alignas(256) std::array<std::uint32_t, Halves * CounterStride> Counters{};

alignas(256) constexpr auto MinMaxCode = std::to_array<std::uint32_t>({
    Sop2(ScalarShiftLeft, 11u, 10u, Inline(11u)),
    Sop2(ScalarOr, 11u, 11u, 9u),
    Sop2(ScalarShiftLeft, 11u, 11u, Inline(4u)),
    Sop2(ScalarShiftLeft, 12u, 8u, Inline(4u)),
    Sop2(ScalarShiftLeft, CounterSgpr, 10u, Inline(4u)),
    Vop2(VectorLshr, 1u, Inline(4u), 0u),
    Vop2(VectorOr, 1u, 11u, 1u),
    Vop2(VectorAnd, 2u, Inline(15u), 0u),
    Vop2(VectorOr, 2u, 12u, 2u),
    Vop2(VectorXor, 3u, LiteralSource, 2u), 0x0000a5a5u,
    Vop2(VectorXor, 4u, LiteralSource, 1u), 0x00005a5au,
    Vop3Low(ShiftLeftOr, 5u, 0u, 0u, false), Vop3High(Vgpr(3u), Inline(16u), Vgpr(1u), 0u),
    Vop3Low(ShiftLeftOr, 6u, 0u, 0u, false), Vop3High(Vgpr(4u), Inline(16u), Vgpr(2u), 0u),
    Vop2(VectorLshr, 7u, Inline(16u), 5u),
    Vop2(VectorLshr, 8u, Inline(16u), 6u),
    Vop3pLow(PackedMin, 10u, 0u, 7u, 0u, false), Vop3pHigh(Vgpr(5u), Vgpr(6u), 0u, 7u, 0u),
    Vop3pLow(PackedMax, 11u, 0u, 7u, 0u, false), Vop3pHigh(Vgpr(5u), Vgpr(6u), 0u, 7u, 0u),
    Vop2(VectorMinF16, 12u, Vgpr(5u), 6u),
    Vop2(VectorMaxF16, 13u, Vgpr(5u), 6u),
    Vop2(VectorMinF16, 14u, Vgpr(7u), 8u),
    Vop2(VectorMaxF16, 15u, Vgpr(7u), 8u),
    Vop3pLow(PackedAshrI16, 20u, 0u, 6u, 0u, false), Vop3pHigh(Inline(15u), Vgpr(5u), 0u, 6u, 0u),
    Vop2(VectorAnd, 20u, LiteralSource, 20u), 0x7fff7fffu,
    Vop2(VectorXor, 20u, Vgpr(5u), 20u),
    Vop3pLow(PackedAshrI16, 21u, 0u, 6u, 0u, false), Vop3pHigh(Inline(15u), Vgpr(6u), 0u, 6u, 0u),
    Vop2(VectorAnd, 21u, LiteralSource, 21u), 0x7fff7fffu,
    Vop2(VectorXor, 21u, Vgpr(6u), 21u),
    Vop3pLow(PackedMinI16, 22u, 0u, 7u, 0u, false), Vop3pHigh(Vgpr(20u), Vgpr(21u), 0u, 7u, 0u),
    Vop3pLow(PackedMaxI16, 23u, 0u, 7u, 0u, false), Vop3pHigh(Vgpr(20u), Vgpr(21u), 0u, 7u, 0u),
    Vop3pLow(PackedAshrI16, 24u, 0u, 6u, 0u, false), Vop3pHigh(Inline(15u), Vgpr(22u), 0u, 6u, 0u),
    Vop2(VectorAnd, 24u, LiteralSource, 24u), 0x7fff7fffu,
    Vop2(VectorXor, 24u, Vgpr(22u), 24u),
    Vop3pLow(PackedAshrI16, 25u, 0u, 6u, 0u, false), Vop3pHigh(Inline(15u), Vgpr(23u), 0u, 6u, 0u),
    Vop2(VectorAnd, 25u, LiteralSource, 25u), 0x7fff7fffu,
    Vop2(VectorXor, 25u, Vgpr(23u), 25u),
    Vop2(VectorAnd, 26u, LiteralSource, 5u), 0x7fff7fffu,
    Vop2(VectorAdd, 26u, LiteralSource, 26u), 0x03ff03ffu,
    Vop2(VectorAnd, 26u, LiteralSource, 26u), 0x80008000u,
    Vop3pLow(PackedAshrI16, 26u, 0u, 6u, 0u, false), Vop3pHigh(Inline(15u), Vgpr(26u), 0u, 6u, 0u),
    Vop2(VectorAnd, 27u, LiteralSource, 6u), 0x7fff7fffu,
    Vop2(VectorAdd, 27u, LiteralSource, 27u), 0x03ff03ffu,
    Vop2(VectorAnd, 27u, LiteralSource, 27u), 0x80008000u,
    Vop3pLow(PackedAshrI16, 27u, 0u, 6u, 0u, false), Vop3pHigh(Inline(15u), Vgpr(27u), 0u, 6u, 0u),
    Vop3Low(BitfieldInsert, 28u, 0u, 0u, false), Vop3High(Vgpr(26u), Vgpr(6u), Vgpr(24u), 0u),
    Vop3Low(BitfieldInsert, 28u, 0u, 0u, false), Vop3High(Vgpr(27u), Vgpr(5u), Vgpr(28u), 0u),
    Vop3Low(BitfieldInsert, 29u, 0u, 0u, false), Vop3High(Vgpr(26u), Vgpr(6u), Vgpr(25u), 0u),
    Vop3Low(BitfieldInsert, 29u, 0u, 0u, false), Vop3High(Vgpr(27u), Vgpr(5u), Vgpr(29u), 0u),
    Vop2(VectorXor, 30u, Vgpr(10u), 28u),
    Vop2(VectorXor, 31u, Vgpr(12u), 28u),
    Vop2(VectorAnd, 31u, LiteralSource, 31u), 0x0000ffffu,
    Vop2(VectorOr, 30u, Vgpr(30u), 31u),
    Vop2(VectorLshr, 31u, Inline(16u), 28u),
    Vop2(VectorXor, 31u, Vgpr(14u), 31u),
    Vop2(VectorAnd, 31u, LiteralSource, 31u), 0x0000ffffu,
    Vop2(VectorOr, 30u, Vgpr(30u), 31u),
    Vop2(VectorXor, 32u, Vgpr(11u), 29u),
    Vop2(VectorXor, 33u, Vgpr(13u), 29u),
    Vop2(VectorAnd, 33u, LiteralSource, 33u), 0x0000ffffu,
    Vop2(VectorOr, 32u, Vgpr(32u), 33u),
    Vop2(VectorLshr, 33u, Inline(16u), 29u),
    Vop2(VectorXor, 33u, Vgpr(15u), 33u),
    Vop2(VectorAnd, 33u, LiteralSource, 33u), 0x0000ffffu,
    Vop2(VectorOr, 32u, Vgpr(32u), 33u),
    Vop1(VectorMov, 35u, Inline(1u)),
    Vop1(VectorMov, CounterVgpr, CounterSgpr),
    Sop1(ScalarMov, 18u, ExecLo),
    Vopc(CompareExecNotEqual, Inline(0u), 30u), MubufLow(BufferAtomicAdd, 0u, true), MubufHigh(CounterVgpr, 35u, 1u, Inline(0u)),
    Sop1(ScalarMov, ExecLo, 18u),
    Vopc(CompareExecNotEqual, Inline(0u), 32u), MubufLow(BufferAtomicAdd, 4u, true), MubufHigh(CounterVgpr, 35u, 1u, Inline(0u)),
    Sop1(ScalarMov, ExecLo, 18u),
    Vop2(VectorAdd, 36u, LiteralSource, 0u), 0x000000ffu,
    Vop2(VectorLshr, 36u, Inline(8u), 36u),
    Vop2(VectorXor, 36u, Inline(1u), 36u),
    Vopc(CompareExecNotEqual, Inline(0u), 36u), MubufLow(BufferAtomicAdd, 8u, true), MubufHigh(CounterVgpr, 35u, 1u, Inline(0u)),
    Endpgm,
});

const std::array<std::optional<ShaderFloatMode>, 5> MinMaxModes{
    std::nullopt,
    ShaderFloatMode{0xc0u, true, false, false},
    ShaderFloatMode{0xc0u, false, false, true},
    ShaderFloatMode{0xf0u, true, false, false},
    ShaderFloatMode{0xccu, true, false, false},
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

}

std::span<const std::uint32_t> Float16MinMaxProbeCode() {
    return MinMaxCode;
}

ShaderRecompiler::RecompileResult CompileFloat16FormProbe(const ShaderRecompiler::SpirvTarget& target, std::span<const std::uint32_t> code, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode) {
    std::vector<std::uint32_t> userData(8, 0u);
    const auto counters = BufferDescriptor(Counters.data(), static_cast<std::uint32_t>(Counters.size() * 4u));
    std::copy(counters.begin(), counters.end(), userData.begin());
    std::copy(counters.begin(), counters.end(), userData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {true, true, true}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    request.context.floatMode = floatMode;
    return ShaderRecompiler::Recompile(request);
}

Float16FormProbeResult ProbeFloat16NativeMinMax(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, bool native) {
    Float16FormProbeResult result;
    auto candidate = target;
    candidate.float16NativeMinMax = native;
    const std::span<const std::uint32_t> code(MinMaxCode);
    constexpr std::uint64_t casesPerHalf = GroupsPerHalf * Threads * 2u;
    for (std::size_t mode = 0; mode < MinMaxModes.size(); ++mode) {
        const auto compiled = CompileFloat16FormProbe(candidate, code, MinMaxModes[mode]);
        Counters.fill(0u);
        device.Dispatch(compiled, GroupsX, GroupsY, Halves, {}, reinterpret_cast<std::uintptr_t>(code.data()));
        device.WaitIdle();
        ++result.modes;
        for (std::uint32_t half = 0; half < Halves; ++half) {
            const std::uint32_t* counts = &Counters[half * CounterStride];
            result.groupsCounted += counts[2];
            result.groupsExpected += GroupsPerHalf;
            if (counts[2] != GroupsPerHalf) {
                throw std::runtime_error("f16 min/max probe: float mode " + std::to_string(mode) + ", half " + std::to_string(half) + " counted " + std::to_string(counts[2]) + " of its " + std::to_string(GroupsPerHalf) + " groups");
            }
            result.compared += casesPerHalf;
            result.byOperation[0] += counts[0];
            result.byOperation[1] += counts[1];
            result.mismatches += static_cast<std::uint64_t>(counts[0]) + counts[1];
        }
        if (result.mismatches != 0u) return result;
    }
    return result;
}

}
