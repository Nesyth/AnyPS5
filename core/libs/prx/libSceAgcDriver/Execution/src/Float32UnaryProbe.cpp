#include "prx/libSceAgcDriver/Execution/include/Float32UnaryProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace AgcDriver {
namespace {

constexpr std::uint32_t X = 3u;
constexpr std::uint32_t Magnitude = 4u;
constexpr std::uint32_t Exponent = 5u;
constexpr std::uint32_t Sign = 6u;
constexpr std::uint32_t SignedInfinity = 7u;
constexpr std::uint32_t Quiet = 8u;
constexpr std::uint32_t IsNan = 9u;
constexpr std::uint32_t IsInfinity = 10u;
constexpr std::uint32_t ZeroExponent = 11u;
constexpr std::uint32_t FullExponent = 12u;
constexpr std::uint32_t Negative = 13u;
constexpr std::uint32_t IsOne = 14u;
constexpr std::uint32_t Overflow = 15u;
constexpr std::uint32_t Underflow = 16u;
constexpr std::uint32_t FlushedX = 17u;
constexpr std::uint32_t FlushedZero = 18u;
constexpr std::uint32_t One = 200u;
constexpr std::uint32_t ConstantZero = 201u;
constexpr std::uint32_t Invalid = 202u;
constexpr std::uint32_t Infinity = 203u;
constexpr std::uint32_t NegativeInfinity = 204u;
constexpr std::uint32_t ConstantOne = 205u;
constexpr std::uint32_t Result = 40u;
constexpr std::uint32_t Passed = 41u;
constexpr std::uint32_t Special = 42u;
constexpr std::uint32_t Poison = 43u;
constexpr std::uint32_t Expected = 44u;

enum class Pass { None, Exp, Fract };

struct UnaryCheck {
    std::uint32_t opcode;
    std::array<std::uint32_t, 4> special;
    Pass pass;
    std::array<std::array<std::uint32_t, 2>, 5> rules;
};

namespace Op = Float32Probe::Vop1;

constexpr std::array<UnaryCheck, 13> Checks{{
    {Op::Rcp, {ZeroExponent, FullExponent, 0u, 0u}, Pass::None, {{{IsInfinity, Sign}, {ZeroExponent, SignedInfinity}, {IsNan, Quiet}, {0u, 0u}, {0u, 0u}}}},
    {Op::RcpIflag, {ZeroExponent, FullExponent, 0u, 0u}, Pass::None, {{{IsInfinity, Sign}, {ZeroExponent, SignedInfinity}, {IsNan, Quiet}, {0u, 0u}, {0u, 0u}}}},
    {Op::Rsq, {ZeroExponent, FullExponent, Negative, 0u}, Pass::None, {{{IsInfinity, ConstantZero}, {ZeroExponent, SignedInfinity}, {Negative, Invalid}, {IsNan, Quiet}, {0u, 0u}}}},
    {Op::Sqrt, {ZeroExponent, FullExponent, Negative, 0u}, Pass::None, {{{IsInfinity, X}, {ZeroExponent, Sign}, {Negative, Invalid}, {IsNan, Quiet}, {0u, 0u}}}},
    {Op::Log, {ZeroExponent, FullExponent, Negative, IsOne}, Pass::None, {{{IsInfinity, X}, {IsOne, ConstantZero}, {ZeroExponent, NegativeInfinity}, {Negative, Invalid}, {IsNan, Quiet}}}},
    {Op::Exp, {ZeroExponent, FullExponent, Overflow, Underflow}, Pass::Exp, {{{Overflow, Infinity}, {Underflow, ConstantZero}, {ZeroExponent, ConstantOne}, {IsNan, Quiet}, {0u, 0u}}}},
    {Op::Fract, {FullExponent, 0u, 0u, 0u}, Pass::Fract, {{{IsInfinity, Invalid}, {IsNan, Quiet}, {0u, 0u}, {0u, 0u}, {0u, 0u}}}},
    {Op::Sin, {FullExponent, FlushedZero, 0u, 0u}, Pass::None, {{{FlushedZero, FlushedX}, {IsInfinity, Invalid}, {IsNan, Quiet}, {0u, 0u}, {0u, 0u}}}},
    {Op::Cos, {FullExponent, 0u, 0u, 0u}, Pass::None, {{{IsInfinity, Invalid}, {IsNan, Quiet}, {0u, 0u}, {0u, 0u}, {0u, 0u}}}},
    {Op::Floor, {IsNan, 0u, 0u, 0u}, Pass::None, {{{IsNan, Quiet}, {0u, 0u}, {0u, 0u}, {0u, 0u}, {0u, 0u}}}},
    {Op::Ceil, {IsNan, 0u, 0u, 0u}, Pass::None, {{{IsNan, Quiet}, {0u, 0u}, {0u, 0u}, {0u, 0u}, {0u, 0u}}}},
    {Op::Trunc, {IsNan, 0u, 0u, 0u}, Pass::None, {{{IsNan, Quiet}, {0u, 0u}, {0u, 0u}, {0u, 0u}, {0u, 0u}}}},
    {Op::Rndne, {IsNan, 0u, 0u, 0u}, Pass::None, {{{IsNan, Quiet}, {0u, 0u}, {0u, 0u}, {0u, 0u}, {0u, 0u}}}},
}};

struct KindChecks {
    ShaderRecompiler::Float32UnaryKind kind;
    std::uint32_t first;
    std::uint32_t count;
};

using Kind = ShaderRecompiler::Float32UnaryKind;

constexpr std::array<KindChecks, 12> Kinds{{
    {Kind::Rcp, 0u, 2u}, {Kind::Rsq, 2u, 1u}, {Kind::Sqrt, 3u, 1u}, {Kind::Log, 4u, 1u}, {Kind::Exp, 5u, 1u}, {Kind::Fract, 6u, 1u},
    {Kind::Sin, 7u, 1u}, {Kind::Cos, 8u, 1u}, {Kind::Floor, 9u, 1u}, {Kind::Ceil, 10u, 1u}, {Kind::Trunc, 11u, 1u}, {Kind::Rndne, 12u, 1u},
}};

Float32Probe::Program UnaryProgram(const Float32Probe::Sweep& sweep, bool ieee) {
    using namespace Float32Probe;
    Program program;
    program.begin();
    program.laneIndex(1u, sweep.groupsXLog2);
    program.vop2(Vop2::Lshlrev, 2u, Int(IterationShift), 1u);
    program.move(One, 1u);
    program.move(ConstantZero, 0u);
    program.move(Invalid, 0xffc00000u);
    program.move(Infinity, 0x7f800000u);
    program.move(NegativeInfinity, 0xff800000u);
    program.move(ConstantOne, 0x3f800000u);
    const auto start = program.loopBegin();
    program.loopInput(X, 2u);
    program.sentinel(X, 0u, FirstSentinel);
    program.sentinel(X, 0xffffffffu, LastSentinel);
    program.vop2(Vop2::And, Magnitude, Lit, X, 0x7fffffffu);
    program.vop2(Vop2::Lshrrev, Exponent, Int(23u), X);
    program.vop2(Vop2::And, Exponent, Lit, Exponent, 0xffu);
    program.vop2(Vop2::And, Sign, Lit, X, 0x80000000u);
    program.vop2(Vop2::Or, SignedInfinity, Lit, Sign, 0x7f800000u);
    if (ieee) program.vop2(Vop2::Or, Quiet, Lit, X, 0x00400000u);
    else program.vop1(Vop1::Mov, Quiet, V(X));
    program.vopc(Vopc::LtU32, Lit, Magnitude, 0x7f800000u);
    program.flag(IsNan, One);
    program.vopc(Vopc::EqU32, Lit, Magnitude, 0x7f800000u);
    program.flag(IsInfinity, One);
    program.vopc(Vopc::EqU32, Zero, Exponent);
    program.flag(ZeroExponent, One);
    program.vop2(Vop2::Cndmask, FlushedX, V(X), Sign);
    program.vopc(Vopc::EqU32, Lit, Exponent, 0xffu);
    program.flag(FullExponent, One);
    program.vopc(Vopc::LeU32, Lit, X, 0x80800000u);
    program.flag(Negative, One);
    program.vopc(Vopc::EqU32, Lit, X, 0x3f800000u);
    program.flag(IsOne, One);
    program.vop2(Vop2::SubrevNc, 30u, Lit, X, 0x43000000u);
    program.vopc(Vopc::GtU32, Lit, 30u, 0x3d000000u);
    program.flag(Overflow, One);
    program.vopc(Vopc::LtU32, Lit, X, 0xc2fc0000u);
    program.flag(Underflow, One);
    program.vop2(Vop2::And, 31u, Lit, FlushedX, 0x7fffffffu);
    program.vopc(Vopc::EqU32, Zero, 31u);
    program.flag(FlushedZero, One);
    for (std::uint32_t index = 0; index < Checks.size(); ++index) {
        const UnaryCheck& unary = Checks[index];
        program.vop1(unary.opcode, Result, V(X));
        switch (unary.pass) {
        case Pass::None: program.vop1(Vop1::Mov, Passed, V(Result)); break;
        case Pass::Exp: program.vop2(Vop2::MaxU32, Passed, Lit, Result, 0x00800000u); break;
        case Pass::Fract:
            program.vop2(Vop2::And, Passed, Lit, Result, 0x7fffffffu);
            program.vop2(Vop2::MinU32, Passed, Lit, Passed, 0x3f7fffffu);
            break;
        }
        program.vop1(Vop1::Mov, Special, V(unary.special[0]));
        for (std::size_t flagIndex = 1; flagIndex < unary.special.size(); ++flagIndex) {
            if (unary.special[flagIndex] != 0u) program.vop2(Vop2::Or, Special, V(unary.special[flagIndex]), Special);
        }
        program.vop1(Vop1::Not, Poison, V(Result));
        program.vop1(Vop1::Mov, Expected, V(Passed));
        program.select(Expected, Special, Poison);
        for (const auto& rule : unary.rules) {
            if (rule[0] != 0u) program.select(Expected, rule[0], rule[1]);
        }
        program.check(Result, Expected, index);
        program.checksum(Result, index);
    }
    program.loopEnd(start, Iterations);
    program.report(static_cast<std::uint32_t>(Checks.size()));
    return program;
}

}

Float32Probe::Verdict ProbeFloat32NativeUnary(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, std::span<const bool> flushSettings, const Float32Probe::Sweep& sweep) {
    Float32Probe::Verdict verdict;
    const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 3> modes{std::nullopt, ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false}, ShaderRecompiler::ShaderFloatMode{0xc0u, true, true, false}};
    const std::array<Float32Probe::Program, 2> programs{UnaryProgram(sweep, false), UnaryProgram(sweep, true)};
    const auto checks = static_cast<std::uint32_t>(Checks.size());
    std::uint32_t failing = 0u;
    for (const bool flush : flushSettings) {
        auto reference = target;
        reference.float32FlushToZero = flush;
        reference.float32NativeUnaryOps = 0u;
        auto candidate = reference;
        candidate.float32NativeUnaryOps = ShaderRecompiler::Float32NativeUnaryAll;
        for (const auto& mode : modes) {
            const auto& program = programs[mode.has_value() && mode->ieeeMode ? 1u : 0u];
            const auto expected = Float32Probe::Run(device, reference, mode, program, sweep);
            const auto actual = expected.compiled && expected.ran ? Float32Probe::Run(device, candidate, mode, program, sweep) : Float32Probe::Outcome{};
            failing |= Float32Probe::Compare(verdict, expected, actual, checks, sweep.Inputs() * checks);
        }
    }
    if (!verdict.Passed()) return verdict;
    for (const KindChecks& entry : Kinds) {
        bool exact = true;
        for (std::uint32_t index = entry.first; index < entry.first + entry.count; ++index) exact = exact && ((failing >> index) & 1u) == 0u;
        if (exact) verdict.proven |= ShaderRecompiler::Float32NativeUnaryBit(entry.kind);
    }
    return verdict;
}

}
