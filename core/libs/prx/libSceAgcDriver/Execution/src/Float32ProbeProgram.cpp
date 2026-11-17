#include "prx/libSceAgcDriver/Execution/include/Float32ProbeProgram.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <stdexcept>

namespace AgcDriver::Float32Probe {
namespace {

using ShaderRecompiler::ShaderStage;

alignas(256) std::array<std::uint32_t, 32768> Code{};
alignas(256) std::array<std::uint32_t, 64> Constants{};
alignas(256) std::array<std::uint32_t, 64> Results{};

constexpr std::uint32_t Sop1Mov = 0x03u;
constexpr std::uint32_t Sop1Bcnt1 = 0x0fu;
constexpr std::uint32_t Sop2Add = 0x00u;
constexpr std::uint32_t Sop2Cselect = 0x0au;
constexpr std::uint32_t Sop2Or = 0x10u;
constexpr std::uint32_t Sop2Lshl = 0x1eu;
constexpr std::uint32_t SopcLg = 0x07u;
constexpr std::uint32_t SopcLt = 0x0au;
constexpr std::uint32_t BufferAtomicAdd = 0x32u;
constexpr std::uint32_t BufferAtomicOr = 0x3au;
constexpr std::uint32_t WaveLanes = 32u;
constexpr std::uint32_t LaneScratch = 10u;
constexpr std::uint32_t Counter = 11u;
constexpr std::uint32_t Scratch = 12u;
constexpr std::uint32_t Processed = 13u;
constexpr std::uint32_t LoopBase = 14u;
constexpr std::uint32_t Mismatches = 15u;
constexpr std::uint32_t Base = 16u;
constexpr std::uint32_t AccumulatorBase = 48u;
constexpr std::uint32_t ChecksumSums = 81u;
constexpr std::uint32_t ChecksumLane = 97u;
constexpr std::uint32_t MaskSgpr = 100u;
constexpr std::uint32_t BitSgpr = 101u;
constexpr std::uint32_t ChecksumBase = 64u;
constexpr std::uint32_t HashValue = 80u;
constexpr std::uint32_t HashShifted = 81u;
constexpr std::uint32_t HashKey = 82u;
constexpr std::uint32_t HashFirst = 83u;
constexpr std::uint32_t HashSecond = 84u;
constexpr std::uint32_t HashGolden = 85u;
constexpr std::uint32_t ReportVgpr = 250u;
constexpr std::uint32_t ResultMask = 0u;
constexpr std::uint32_t ResultMismatches = 1u;
constexpr std::uint32_t ResultProcessed = 2u;
constexpr std::uint32_t ResultBase = 3u;
constexpr std::uint32_t ResultChecksums = 16u;

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

Sweep Covering(std::uint32_t groupsXLog2, std::uint32_t groupsYLog2) {
    Sweep sweep{groupsXLog2, groupsYLog2, {}};
    for (std::uint64_t base = 0; base < (1ull << 32u); base += sweep.ChunkInputs()) sweep.bases.push_back(static_cast<std::uint32_t>(base));
    return sweep;
}

}

Sweep FullSweep() {
    return Covering(5u, 5u);
}

Sweep EndsSweep() {
    Sweep sweep{0u, 0u, {}};
    sweep.bases = {0u, static_cast<std::uint32_t>((1ull << 32u) - sweep.ChunkInputs())};
    return sweep;
}

void Program::literalIf(std::uint32_t operand, std::uint32_t literal) {
    if (operand == Lit) words.push_back(literal);
}

void Program::vop1(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t literal) {
    words.push_back(0x7e000000u | (vdst << 17u) | (op << 9u) | src0);
    literalIf(src0, literal);
}

void Program::vop2(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t vsrc1, std::uint32_t literal) {
    words.push_back((op << 25u) | (vdst << 17u) | (vsrc1 << 9u) | src0);
    literalIf(src0, literal);
}

void Program::vop3(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t src1, std::uint32_t src2) {
    words.push_back(0xd4000000u | (op << 16u) | vdst);
    words.push_back((src2 << 18u) | (src1 << 9u) | src0);
}

void Program::vopc(std::uint32_t op, std::uint32_t src0, std::uint32_t vsrc1, std::uint32_t literal) {
    words.push_back(0x7c000000u | (op << 17u) | (vsrc1 << 9u) | src0);
    literalIf(src0, literal);
}

void Program::sop1(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0, std::uint32_t literal) {
    words.push_back(0xbe800000u | (sdst << 16u) | (op << 8u) | ssrc0);
    literalIf(ssrc0, literal);
}

void Program::sop2(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0, std::uint32_t ssrc1, std::uint32_t literal) {
    words.push_back(0x80000000u | (op << 23u) | (sdst << 16u) | (ssrc1 << 8u) | ssrc0);
    if (ssrc0 == Lit || ssrc1 == Lit) words.push_back(literal);
}

void Program::sopc(std::uint32_t op, std::uint32_t ssrc0, std::uint32_t ssrc1, std::uint32_t literal) {
    words.push_back(0xbf000000u | (op << 16u) | (ssrc1 << 8u) | ssrc0);
    if (ssrc0 == Lit || ssrc1 == Lit) words.push_back(literal);
}

void Program::move(std::uint32_t vdst, std::uint32_t value) {
    if (value <= 64u) vop1(Vop1::Mov, vdst, Int(value));
    else vop1(Vop1::Mov, vdst, Lit, value);
}

void Program::flush(std::uint32_t vdst, std::uint32_t vsrc, std::uint32_t scratch) {
    vop2(Vop2::And, scratch, Lit, vsrc, 0x7f800000u);
    vopc(Vopc::EqU32, Zero, scratch);
    vop2(Vop2::And, scratch + 1u, Lit, vsrc, 0x80000000u);
    vop2(Vop2::Cndmask, vdst, V(vsrc), scratch + 1u);
}

void Program::flag(std::uint32_t vdst, std::uint32_t one) {
    vop2(Vop2::Cndmask, vdst, Zero, one);
}

void Program::select(std::uint32_t vdst, std::uint32_t flagVgpr, std::uint32_t valueVgpr) {
    vopc(Vopc::NeU32, Zero, flagVgpr);
    vop2(Vop2::Cndmask, vdst, V(vdst), valueVgpr);
}

void Program::begin() {
    for (std::uint32_t slot = 0; slot < Slots; ++slot) sop1(Sop1Mov, AccumulatorBase + slot, Zero);
    sop1(Sop1Mov, Mismatches, Zero);
    sop1(Sop1Mov, Processed, Zero);
    for (std::uint32_t slot = 0; slot < ChecksumSlots; ++slot) vop1(Vop1::Mov, ChecksumBase + slot, Zero);
    move(HashFirst, 0x85ebca6bu);
    move(HashSecond, 0xc2b2ae35u);
    move(HashGolden, 0x9e3779b1u);
    words.push_back(0xf4200000u | (Base << 6u));
    words.push_back(0xfa000000u);
    words.push_back(0xbf8c0070u);
}

void Program::laneIndex(std::uint32_t vdst, std::uint32_t groupsXLog2) {
    sop2(Sop2Lshl, LaneScratch, 9u, Int(groupsXLog2));
    sop2(Sop2Add, LaneScratch, LaneScratch, 8u);
    sop2(Sop2Lshl, LaneScratch, LaneScratch, Int(ThreadShift));
    vop2(Vop2::AddNc, vdst, LaneScratch, 0u);
}

std::size_t Program::loopBegin() {
    sop1(Sop1Mov, Counter, Zero);
    return words.size();
}

void Program::loopInput(std::uint32_t vinput, std::uint32_t vlaneBase) {
    sop2(Sop2Add, LoopBase, Base, Counter);
    vop2(Vop2::AddNc, vinput, LoopBase, vlaneBase);
    sop1(Sop1Bcnt1, Scratch, ExecLo);
    sop2(Sop2Add, Processed, Processed, Scratch);
    vop2(Vop2::AddNc, ChecksumBase + ControlChecksum, V(vinput), ChecksumBase + ControlChecksum);
    vop3(Vop3::MulLo, HashKey, V(vinput), V(HashGolden));
}

void Program::loopEnd(std::size_t start, std::uint32_t iterations) {
    sop2(Sop2Add, Counter, Counter, Int(1u));
    if (iterations <= 64u) sopc(SopcLt, Counter, Int(iterations));
    else sopc(SopcLt, Counter, Lit, iterations);
    const std::int32_t offset = static_cast<std::int32_t>(start) - static_cast<std::int32_t>(words.size() + 1u);
    words.push_back(0xbf850000u | (static_cast<std::uint32_t>(offset) & 0xffffu));
}

void Program::check(std::uint32_t vresult, std::uint32_t vexpected, std::uint32_t slot) {
    vopc(Vopc::NeU32, V(vresult), vexpected);
    sop2(Sop2Or, AccumulatorBase + slot, AccumulatorBase + slot, VccLo);
    sop1(Sop1Bcnt1, Scratch, VccLo);
    sop2(Sop2Add, Mismatches, Mismatches, Scratch);
}

void Program::checksum(std::uint32_t vresult, std::uint32_t slot) {
    vop2(Vop2::Xor, HashValue, V(vresult), HashKey);
    vop3(Vop3::MulLo, HashValue, V(HashValue), V(HashFirst));
    vop2(Vop2::Lshrrev, HashShifted, Int(13u), HashValue);
    vop2(Vop2::Xor, HashValue, V(HashShifted), HashValue);
    vop3(Vop3::MulLo, HashValue, V(HashValue), V(HashSecond));
    vop2(Vop2::Lshrrev, HashShifted, Int(16u), HashValue);
    vop2(Vop2::Xor, HashValue, V(HashShifted), HashValue);
    vop2(Vop2::AddNc, ChecksumBase + slot, V(HashValue), ChecksumBase + slot);
}

void Program::sentinel(std::uint32_t vinput, std::uint32_t value, std::uint32_t slot) {
    vopc(Vopc::EqU32, Lit, vinput, value);
    sop2(Sop2Or, AccumulatorBase + slot, AccumulatorBase + slot, VccLo);
}

void Program::atomic(std::uint32_t op, std::uint32_t offset, std::uint32_t sgpr, std::uint32_t vgpr) {
    vop1(Vop1::Mov, vgpr, sgpr);
    words.push_back(0xe0000000u | (op << 18u) | offset);
    words.push_back(0x80010000u | (vgpr << 8u));
}

void Program::report(std::uint32_t checksums) {
    if (checksums > DataChecksums) throw std::runtime_error("a float probe program asks for more checksums than it has slots");
    for (std::uint32_t slot = 0; slot < ChecksumSlots; ++slot) {
        if (slot >= checksums && slot != ControlChecksum) continue;
        vop3(Vop3::Readlane, ChecksumSums + slot, V(ChecksumBase + slot), Int(0u));
        for (std::uint32_t lane = 1; lane < WaveLanes; ++lane) {
            vop3(Vop3::Readlane, ChecksumLane, V(ChecksumBase + slot), Int(lane));
            sop2(Sop2Add, ChecksumSums + slot, ChecksumSums + slot, ChecksumLane);
        }
    }
    sop1(Sop1Mov, MaskSgpr, Zero);
    for (std::uint32_t slot = 0; slot < Slots; ++slot) {
        sopc(SopcLg, AccumulatorBase + slot, Zero);
        sop2(Sop2Cselect, BitSgpr, Lit, Zero, 1u << slot);
        sop2(Sop2Or, MaskSgpr, MaskSgpr, BitSgpr);
    }
    sop1(Sop1Mov, ExecLo, Int(1u));
    atomic(BufferAtomicOr, ResultMask * 4u, MaskSgpr, ReportVgpr);
    atomic(BufferAtomicAdd, ResultMismatches * 4u, Mismatches, ReportVgpr + 1u);
    atomic(BufferAtomicAdd, ResultProcessed * 4u, Processed, ReportVgpr + 2u);
    atomic(BufferAtomicOr, ResultBase * 4u, Base, ReportVgpr + 3u);
    for (std::uint32_t slot = 0; slot < ChecksumSlots; ++slot) {
        if (slot >= checksums && slot != ControlChecksum) continue;
        atomic(BufferAtomicAdd, (ResultChecksums + slot) * 4u, ChecksumSums + slot, ReportVgpr + 4u);
    }
    words.push_back(0xbf8c0070u);
    words.push_back(0xbf810000u);
}

Outcome Run(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, const std::optional<ShaderRecompiler::ShaderFloatMode>& mode, const Program& program, const Sweep& sweep) {
    Outcome outcome;
    const auto words = program.Words();
    if (words.size() > Code.size()) throw std::runtime_error("a float probe program does not fit its buffer");
    std::copy(words.begin(), words.end(), Code.begin());
    const std::span<const std::uint32_t> code(Code.data(), words.size());
    std::vector<std::uint32_t> userData(8, 0u);
    const auto input = BufferDescriptor(Constants.data(), static_cast<std::uint32_t>(Constants.size() * 4u));
    const auto output = BufferDescriptor(Results.data(), static_cast<std::uint32_t>(Results.size() * 4u));
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {true, true, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    request.context.floatMode = mode;
    std::optional<ShaderRecompiler::RecompileResult> result;
    try {
        result.emplace(ShaderRecompiler::Recompile(request));
    } catch (const std::exception&) {
        return outcome;
    }
    outcome.compiled = true;
    const std::uint64_t inputs = sweep.ChunkInputs();
    for (const std::uint32_t base : sweep.bases) {
        Constants.fill(0u);
        Constants[0] = base;
        Results.fill(0u);
        try {
            device.Dispatch(*result, 1u << sweep.groupsXLog2, 1u << sweep.groupsYLog2, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
            device.WaitIdle();
        } catch (const std::exception&) {
            return outcome;
        }
        ++outcome.dispatches;
        const std::uint64_t sum = inputs * base + inputs * (inputs - 1u) / 2u;
        if (Results[ResultProcessed] != static_cast<std::uint32_t>(inputs) || Results[ResultBase] != base || Results[ResultChecksums + ControlChecksum] != static_cast<std::uint32_t>(sum)) return outcome;
        outcome.mask |= Results[ResultMask];
        outcome.count += Results[ResultMismatches];
        for (std::uint32_t slot = 0; slot < ChecksumSlots; ++slot) outcome.checksums[slot] += Results[ResultChecksums + slot];
    }
    outcome.ran = !sweep.bases.empty();
    return outcome;
}

std::uint32_t Compare(Verdict& verdict, const Outcome& reference, const Outcome& candidate, std::uint32_t checks, std::uint64_t results) {
    const std::uint32_t slots = (1u << checks) - 1u;
    verdict.dispatches += reference.dispatches + candidate.dispatches;
    if (!reference.compiled || (reference.ran && !candidate.compiled)) {
        ++verdict.notCompiled;
        return slots;
    }
    if (!reference.ran || !candidate.ran) {
        ++verdict.failedRuns;
        return slots;
    }
    ++verdict.runs;
    verdict.sentinels = verdict.sentinels && (reference.mask & SentinelMask) == SentinelMask && (candidate.mask & SentinelMask) == SentinelMask;
    std::uint32_t differing = 0u;
    for (std::uint32_t slot = 0; slot < checks; ++slot) {
        if (reference.checksums[slot] != candidate.checksums[slot]) differing |= 1u << slot;
    }
    verdict.checksumDifferences |= differing;
    verdict.compared += results;
    verdict.mismatches += candidate.count;
    verdict.referenceMismatches += reference.count;
    return ((reference.mask | candidate.mask) & slots) | differing;
}

}
