#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FLOAT32PROBEPROGRAM_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FLOAT32PROBEPROGRAM_HPP

#include "Recompiler.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace AgcDriver {

class VulkanDevice;

namespace Float32Probe {

constexpr std::uint32_t V(std::uint32_t index) { return 256u + index; }
constexpr std::uint32_t Int(std::uint32_t value) { return 128u + value; }
inline constexpr std::uint32_t Zero = 128u;
inline constexpr std::uint32_t Lit = 255u;
inline constexpr std::uint32_t VccLo = 106u;
inline constexpr std::uint32_t ExecLo = 126u;
inline constexpr std::uint32_t ThreadShift = 8u;
inline constexpr std::uint32_t Threads = 1u << ThreadShift;
inline constexpr std::uint32_t IterationShift = 8u;
inline constexpr std::uint32_t Iterations = 1u << IterationShift;
inline constexpr std::uint32_t Slots = 32u;
inline constexpr std::uint32_t FirstSentinel = 30u;
inline constexpr std::uint32_t LastSentinel = 31u;
inline constexpr std::uint32_t SentinelMask = (1u << FirstSentinel) | (1u << LastSentinel);
inline constexpr std::uint32_t ChecksumSlots = 16u;
inline constexpr std::uint32_t ControlChecksum = ChecksumSlots - 1u;
inline constexpr std::uint32_t DataChecksums = ControlChecksum;

namespace Vop1 {
inline constexpr std::uint32_t Mov = 0x01u, Fract = 0x20u, Trunc = 0x21u, Ceil = 0x22u, Rndne = 0x23u, Floor = 0x24u, Exp = 0x25u, Log = 0x27u, Rcp = 0x2au, RcpIflag = 0x2bu, Rsq = 0x2eu, Sqrt = 0x33u, Sin = 0x35u, Cos = 0x36u, Not = 0x37u;
}
namespace Vop2 {
inline constexpr std::uint32_t Cndmask = 0x01u, MinU32 = 0x13u, MaxU32 = 0x14u, Lshrrev = 0x16u, Lshlrev = 0x1au, And = 0x1bu, Or = 0x1cu, Xor = 0x1du, AddNc = 0x25u, SubrevNc = 0x27u;
}
namespace Vop3 {
inline constexpr std::uint32_t MulLo = 0x169u, Readlane = 0x360u;
}
namespace Vopc {
inline constexpr std::uint32_t LtU32 = 0xc1u, EqU32 = 0xc2u, LeU32 = 0xc3u, GtU32 = 0xc4u, NeU32 = 0xc5u;
}

struct Sweep {
    std::uint32_t groupsXLog2 = 0;
    std::uint32_t groupsYLog2 = 0;
    std::vector<std::uint32_t> bases;

    [[nodiscard]] std::uint64_t ChunkInputs() const { return 1ull << (groupsXLog2 + groupsYLog2 + ThreadShift + IterationShift); }
    [[nodiscard]] std::uint64_t Inputs() const { return ChunkInputs() * bases.size(); }
};

Sweep FullSweep();
Sweep EndsSweep();

class Program {
public:
    void vop1(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t literal = 0u);
    void vop2(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t vsrc1, std::uint32_t literal = 0u);
    void vop3(std::uint32_t op, std::uint32_t vdst, std::uint32_t src0, std::uint32_t src1, std::uint32_t src2 = Zero);
    void vopc(std::uint32_t op, std::uint32_t src0, std::uint32_t vsrc1, std::uint32_t literal = 0u);
    void sop1(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0, std::uint32_t literal = 0u);
    void sop2(std::uint32_t op, std::uint32_t sdst, std::uint32_t ssrc0, std::uint32_t ssrc1, std::uint32_t literal = 0u);
    void sopc(std::uint32_t op, std::uint32_t ssrc0, std::uint32_t ssrc1, std::uint32_t literal = 0u);
    void move(std::uint32_t vdst, std::uint32_t value);
    void flush(std::uint32_t vdst, std::uint32_t vsrc, std::uint32_t scratch);
    void flag(std::uint32_t vdst, std::uint32_t one);
    void select(std::uint32_t vdst, std::uint32_t flagVgpr, std::uint32_t valueVgpr);
    void begin();
    void laneIndex(std::uint32_t vdst, std::uint32_t groupsXLog2);
    std::size_t loopBegin();
    void loopInput(std::uint32_t vinput, std::uint32_t vlaneBase);
    void loopEnd(std::size_t start, std::uint32_t iterations);
    void check(std::uint32_t vresult, std::uint32_t vexpected, std::uint32_t slot);
    void checksum(std::uint32_t vresult, std::uint32_t slot);
    void sentinel(std::uint32_t vinput, std::uint32_t value, std::uint32_t slot);
    void report(std::uint32_t checksums);

    [[nodiscard]] std::span<const std::uint32_t> Words() const { return words; }

private:
    void literalIf(std::uint32_t operand, std::uint32_t literal);
    void atomic(std::uint32_t op, std::uint32_t offset, std::uint32_t sgpr, std::uint32_t vgpr);
    std::vector<std::uint32_t> words;
};

struct Outcome {
    bool compiled = false;
    bool ran = false;
    std::uint32_t dispatches = 0;
    std::uint32_t mask = 0;
    std::uint64_t count = 0;
    std::array<std::uint32_t, ChecksumSlots> checksums{};
};

struct Verdict {
    std::uint32_t proven = 0;
    std::uint64_t compared = 0;
    std::uint64_t mismatches = 0;
    std::uint64_t referenceMismatches = 0;
    std::uint32_t checksumDifferences = 0;
    std::uint32_t notCompiled = 0;
    std::uint32_t failedRuns = 0;
    std::uint32_t runs = 0;
    std::uint32_t dispatches = 0;
    bool sentinels = true;

    [[nodiscard]] bool Passed() const { return runs != 0u && notCompiled == 0u && failedRuns == 0u && sentinels; }
};

Outcome Run(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, const std::optional<ShaderRecompiler::ShaderFloatMode>& mode, const Program& program, const Sweep& sweep);
std::uint32_t Compare(Verdict& verdict, const Outcome& reference, const Outcome& candidate, std::uint32_t checks, std::uint64_t results);

}

}

#endif
