#include <spirv/unified1/spirv.hpp>
#include "Recompiler.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct UnaryOp {
    const char* name;
    std::uint32_t opcode;
    bool shrinks;
};

constexpr std::array<UnaryOp, 13> Ops{{
    {"v_rcp_f32", 0x2au, true}, {"v_rcp_iflag_f32", 0x2bu, true}, {"v_rsq_f32", 0x2eu, true}, {"v_sqrt_f32", 0x33u, true},
    {"v_log_f32", 0x27u, true}, {"v_exp_f32", 0x25u, true}, {"v_fract_f32", 0x20u, true}, {"v_sin_f32", 0x35u, false},
    {"v_cos_f32", 0x36u, false}, {"v_floor_f32", 0x24u, false}, {"v_ceil_f32", 0x22u, false}, {"v_trunc_f32", 0x21u, false},
    {"v_rndne_f32", 0x23u, false},
}};

constexpr std::uint32_t FractOpcode = 0x20u;

struct Config {
    std::optional<ShaderRecompiler::ShaderFloatMode> mode;
    bool flushToZero = true;
    std::uint32_t hardwareFlushOps = 0u;
    std::uint32_t nativeUnaryOps = 0u;
};

std::vector<std::uint32_t> Program(std::uint32_t opcode, bool literal) {
    std::vector<std::uint32_t> words{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u};
    if (literal) {
        words.push_back(0x7e0600ffu | (opcode << 9u));
        words.push_back(0x80000001u);
    } else {
        words.push_back(0x7e060102u | (opcode << 9u));
    }
    words.insert(words.end(), {0xe0701000u, 0x80000301u, 0xbf810000u});
    return words;
}

std::vector<std::uint32_t> Recompiled(std::span<const std::uint32_t> code, const Config& config) {
    const std::array<std::uint32_t, 4> userData{0x10000000u, 0x00100000u, 0x100u, 0x00027facu};
    const std::array<std::uint32_t, 3> capabilities{1u, 61u, static_cast<std::uint32_t>(spv::CapabilityDenormFlushToZero)};
    const std::array<std::string_view, 1> extensions{"SPV_KHR_storage_buffer_storage_class"};
    ShaderRecompiler::RecompileRequest request{};
    request.shader = {ShaderRecompiler::ShaderStage::Compute, 0x20000u, code, 0, {}};
    request.context.waveSize = 32;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderRecompiler::ShaderComputeStageInfo{{64u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    request.context.floatMode = config.mode;
    request.target.vulkanVersion = 0x00403000u;
    request.target.spirvVersion = 0x00010600u;
    request.target.subgroupSize = 32u;
    request.target.bdaAbiVersion = 1;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.target.maxWorkgroupSize = {1024u, 1024u, 64u};
    request.target.maxWorkgroupInvocations = 1024;
    request.target.maxWorkgroupSharedMemoryBytes = 49152;
    request.target.float32FlushToZero = config.flushToZero;
    request.target.float32HardwareFlushOps = config.hardwareFlushOps;
    request.target.float32NativeUnaryOps = config.nativeUnaryOps;
    request.layout = {0, 0, 0, 128};
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    const auto& words = result.spirv.Words();
    return {words.begin(), words.end()};
}

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr ShaderRecompiler::ShaderFloatMode Probed{0xc0u, true, false, false};
constexpr ShaderRecompiler::ShaderFloatMode ProbedIeee{0xc0u, true, true, false};
constexpr ShaderRecompiler::ShaderFloatMode Kept{0xf0u, true, false, false};
constexpr ShaderRecompiler::ShaderFloatMode Flushed{0x00u, true, false, false};

void RequireSame(const std::vector<std::uint32_t>& code, Config config, std::uint32_t hardwareFlushOps, std::uint32_t nativeUnaryOps, const std::string& what) {
    const auto today = Recompiled(code, config);
    config.hardwareFlushOps = hardwareFlushOps;
    config.nativeUnaryOps = nativeUnaryOps;
    Require(Recompiled(code, config) == today, what + ": the module changed although the form does not apply");
}

void RequireSmaller(const std::vector<std::uint32_t>& code, Config config, std::uint32_t hardwareFlushOps, std::uint32_t nativeUnaryOps, bool shrinks, const std::string& what) {
    const auto today = Recompiled(code, config);
    config.hardwareFlushOps = hardwareFlushOps;
    config.nativeUnaryOps = nativeUnaryOps;
    const auto changed = Recompiled(code, config);
    std::printf("%s: %zu SPIR-V words today, %zu with the form\n", what.c_str(), today.size(), changed.size());
    if (shrinks) Require(changed != today && changed.size() < today.size(), what + ": the form does not shrink the module");
}

}

int main() {
    try {
        const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 3> probedModes{std::nullopt, Probed, ProbedIeee};
        for (const UnaryOp& op : Ops) {
            const auto code = Program(op.opcode, false);
            const auto literal = Program(op.opcode, true);
            const std::string name = op.name;
            for (const auto& mode : probedModes) {
                for (const bool flush : {false, true}) {
                    const std::string what = name + (mode.has_value() ? (mode->ieeeMode ? " in 0xc0 with IEEE_MODE" : " in 0xc0") : " undeclared") + (flush ? " with the hardware flush" : " with the emulated flush");
                    RequireSmaller(code, {mode, flush, 0u, 0u}, 0u, ShaderRecompiler::Float32NativeUnaryAll, op.shrinks, what + ", native unary");
                    RequireSame(literal, {mode, flush, 0u, 0u}, ShaderRecompiler::Float32HardwareFlushAll, ShaderRecompiler::Float32NativeUnaryAll, what + ", literal operand");
                }
            }
            for (const auto& mode : {Kept, Flushed}) {
                for (const bool flush : {false, true}) {
                    RequireSame(code, {mode, flush, 0u, 0u}, ShaderRecompiler::Float32HardwareFlushAll, ShaderRecompiler::Float32NativeUnaryAll, name + " in float mode " + std::to_string(mode.floatMode) + (flush ? " with the hardware flush" : " with the emulated flush"));
                }
            }
            if (op.opcode != FractOpcode) RequireSame(code, {Probed, true, 0u, 0u}, ShaderRecompiler::Float32HardwareFlushAll, 0u, name + ", hardware fract flush on another operation");
        }
        const auto fract = Program(FractOpcode, false);
        for (const auto& mode : probedModes) {
            const std::string what = std::string("v_fract_f32") + (mode.has_value() ? (mode->ieeeMode ? " in 0xc0 with IEEE_MODE" : " in 0xc0") : " undeclared");
            RequireSmaller(fract, {mode, true, 0u, 0u}, ShaderRecompiler::Float32HardwareFlushAll, 0u, true, what + ", hardware flush");
            RequireSmaller(fract, {mode, true, 0u, 0u}, ShaderRecompiler::Float32HardwareFlushAll, ShaderRecompiler::Float32NativeUnaryAll, true, what + ", hardware flush and native unary");
            RequireSame(fract, {mode, false, 0u, 0u}, ShaderRecompiler::Float32HardwareFlushAll, 0u, what + ", hardware flush without the hardware f32 flush");
        }
        std::puts("PASS: the f32 exactness forms apply only where they are probed, compile, and shrink the modules they change");
        return 0;
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
}
