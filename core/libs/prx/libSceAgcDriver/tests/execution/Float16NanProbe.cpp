#include "prx/libSceAgcDriver/Execution/include/Float16NanProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/Float16FormProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestCodeWords.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <spirv/unified1/GLSL.std.450.h>
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 64;

alignas(256) constexpr std::array<std::uint32_t, 26> PackedCode{
    0x34020084u, 0x34060085u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u,
    0xcc0f400au, 0x18020b04u,
    0xcc10400bu, 0x18020b04u,
    0xcc0e400cu, 0x1c1a0b04u,
    0xcc0f4a0du, 0x30020b04u,
    0xcc10c00eu, 0x18020d04u,
    0xcc0e330fu, 0x8c1a0b04u,
    0x7e200306u, 0x78200b04u,
    0xcc0ec011u, 0x3c1a0f04u,
    0xe0781000u, 0x80010a03u, 0xe0781010u, 0x80010e03u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 20> ScalarFmaCode{
    0x34020084u, 0x34060085u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u,
    0xd74b000au, 0x041a0b04u,
    0xd74b980bu, 0x041a0b04u,
    0x7e180306u, 0x6c180b04u,
    0x6e1a0b04u, 0x00003d55u,
    0x701c0b04u, 0x00000001u,
    0xe0781000u, 0x80010a03u, 0xe0701010u, 0x80010e03u, 0xbf810000u,
};

alignas(256) constexpr auto ConstantMinMaxCode = std::to_array<std::uint32_t>({
    0x34020084u, 0x34060085u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u,
    AgcDriver::GuestCode::Sop1(0x03u, 20u, AgcDriver::GuestCode::LiteralSource), 0x7e01bc00u,
    AgcDriver::GuestCode::Vop2(0x3au, 10u, 242u, 5u),
    AgcDriver::GuestCode::Vop2(0x39u, 11u, AgcDriver::GuestCode::LiteralSource, 5u), 0x00007e01u,
    AgcDriver::GuestCode::Vop2(0x3au, 12u, 20u, 5u),
    AgcDriver::GuestCode::Vop3pLow(0x12u, 13u, 0u, 7u, 0u, false), AgcDriver::GuestCode::Vop3pHigh(AgcDriver::GuestCode::Vgpr(4u), 20u, 0u, 7u, 0u),
    AgcDriver::GuestCode::Vop3pLow(0x11u, 14u, 0u, 7u, 0u, false), AgcDriver::GuestCode::Vop3pHigh(AgcDriver::GuestCode::Vgpr(4u), 240u, 0u, 7u, 0u),
    AgcDriver::GuestCode::Vop3Low(0x351u, 15u, 0u, 0u, false), AgcDriver::GuestCode::Vop3High(AgcDriver::GuestCode::Vgpr(4u), AgcDriver::GuestCode::Vgpr(5u), 242u, 0u),
    AgcDriver::GuestCode::Vop3Low(0x354u, 16u, 0u, 0u, false), AgcDriver::GuestCode::Vop3High(20u, 20u, AgcDriver::GuestCode::Vgpr(5u), 0u),
    0xe0781000u, 0x80010a03u, 0xe0781010u, 0x80010e03u, 0xbf810000u,
});

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::uint32_t Selects(std::span<const std::uint32_t> words) {
    std::uint32_t selects = 0;
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        Require(count != 0 && cursor + count <= words.size(), "f16 NaN probe: malformed SPIR-V");
        if ((words[cursor] & 0xffffu) == spv::OpSelect) ++selects;
        cursor += count;
    }
    return selects;
}

std::uint32_t GlslInstructions(std::span<const std::uint32_t> words, std::uint32_t instruction) {
    std::uint32_t found = 0;
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        Require(count != 0 && cursor + count <= words.size(), "f16 form probe: malformed SPIR-V");
        if ((words[cursor] & 0xffffu) == spv::OpExtInst && count > 4u && words[cursor + 4u] == instruction) ++found;
        cursor += count;
    }
    return found;
}

std::uint32_t Instructions(std::span<const std::uint32_t> words, spv::Op opcode) {
    std::uint32_t found = 0;
    for (std::size_t cursor = 5; cursor < words.size();) {
        const auto count = words[cursor] >> 16u;
        Require(count != 0 && cursor + count <= words.size(), "f16 form probe: malformed SPIR-V");
        if ((words[cursor] & 0xffffu) == static_cast<std::uint32_t>(opcode)) ++found;
        cursor += count;
    }
    return found;
}

void VerifyFormSpirv(ShaderRecompiler::SpirvTarget target) {
    const std::optional<ShaderRecompiler::ShaderFloatMode> mode = ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false};
    target.float16HostNans = true;
    const auto compile = [&](std::span<const std::uint32_t> code, bool wholeRegister, bool scalarOnVector, bool nativeMinMax) {
        auto formTarget = target;
        formTarget.float16WholeRegister = wholeRegister;
        formTarget.float16ScalarOnVector = scalarOnVector;
        formTarget.float16NativeMinMax = nativeMinMax;
        return AgcDriver::CompileFloat16FormProbe(formTarget, code, mode).spirv.Copy();
    };
    const auto packedToday = compile(PackedCode, false, false, false);
    const auto packedWords = compile(PackedCode, true, false, false);
    Require(Instructions(packedWords, spv::OpIsNan) > Instructions(packedToday, spv::OpIsNan) && Instructions(packedWords, spv::OpVectorShuffle) > Instructions(packedToday, spv::OpVectorShuffle), "f16 form probe: the whole-register forms do not take the vector NaN select and the lane shuffle");
    Require(compile(PackedCode, false, true, true) == packedToday, "f16 form probe: float16ScalarOnVector or float16NativeMinMax changed the packed add, mul, fma and fmac forms");
    const auto scalarToday = compile(ScalarFmaCode, false, false, false);
    const auto scalarWords = compile(ScalarFmaCode, false, true, false);
    Require(Instructions(scalarWords, spv::OpVectorShuffle) > Instructions(scalarToday, spv::OpVectorShuffle), "f16 form probe: the scalar fma forms on whole registers do not merge their lane with a shuffle");
    Require(compile(ScalarFmaCode, true, false, true) == scalarToday, "f16 form probe: float16WholeRegister or float16NativeMinMax changed the scalar fma forms");
    const auto minMaxCode = AgcDriver::Float16MinMaxProbeCode();
    const auto integer = compile(minMaxCode, true, true, false);
    const auto native = compile(minMaxCode, true, true, true);
    Require(GlslInstructions(integer, GLSLstd450NMin) == 0u && GlslInstructions(integer, GLSLstd450NMax) == 0u && Instructions(integer, spv::OpIsNan) == 0u, "f16 form probe: the integer min/max form uses a GLSL min or max or a float NaN test");
    const auto nativeOperations = GlslInstructions(native, GLSLstd450NMin) + GlslInstructions(native, GLSLstd450NMax);
    Require(GlslInstructions(native, GLSLstd450NMin) != 0u && GlslInstructions(native, GLSLstd450NMax) != 0u && GlslInstructions(native, GLSLstd450FMin) == 0u && GlslInstructions(native, GLSLstd450FMax) == 0u, "f16 form probe: float16NativeMinMax does not emit NMin and NMax alone");
    Require(Instructions(native, spv::OpIsNan) >= nativeOperations, "f16 form probe: a native f16 min or max lacks the second operand's NaN select");
    const auto constants = compile(ConstantMinMaxCode, false, false, true);
    Require(GlslInstructions(constants, GLSLstd450NMin) == 0u && GlslInstructions(constants, GLSLstd450NMax) == 0u, "f16 form probe: an f16 min or max with a literal, inline or SGPR-held constant operand emits NMin or NMax");
}

void VerifySpirv() {
    constexpr std::array<std::uint32_t, 12> capabilities{
        spv::CapabilityShader, spv::CapabilityGroupNonUniform, spv::CapabilityGroupNonUniformBallot,
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess,
        spv::CapabilityDenormPreserve, spv::CapabilitySignedZeroInfNanPreserve, spv::CapabilityFloat16,
        spv::CapabilityRoundingModeRTE, spv::CapabilityInt16, spv::CapabilityFMAKHR,
    };
    constexpr std::array<std::string_view, 6> extensions{
        "SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage", "SPV_KHR_float_controls",
        "SPV_KHR_storage_buffer_storage_class", "SPV_KHR_16bit_storage", "SPV_KHR_fma",
    };
    std::vector<std::uint32_t> input(Threads * 4u), output(Threads * 8u);
    std::vector<std::uint32_t> userData(8, 0u);
    const auto inputDescriptor = BufferDescriptor(input.data(), static_cast<std::uint32_t>(input.size() * 4u));
    const auto outputDescriptor = BufferDescriptor(output.data(), static_cast<std::uint32_t>(output.size() * 4u));
    std::copy(inputDescriptor.begin(), inputDescriptor.end(), userData.begin());
    std::copy(outputDescriptor.begin(), outputDescriptor.end(), userData.begin() + 4);
    ShaderRecompiler::SpirvTarget target{};
    target.vulkanVersion = 0x00403000u;
    target.spirvVersion = 0x00010600u;
    target.subgroupSize = 32;
    target.bdaAbiVersion = 1;
    target.supportedCapabilities = capabilities;
    target.supportedExtensions = extensions;
    target.maxWorkgroupSize = {1024, 1024, 64};
    target.maxWorkgroupInvocations = 1024;
    target.maxWorkgroupSharedMemoryBytes = 49152;
    target.float16DenormPreserve = true;
    const auto compile = [&](std::span<const std::uint32_t> code, bool hostNans, bool ieee) {
        const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
        target.float16HostNans = hostNans;
        ShaderRecompiler::RecompileRequest request{
            {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
            {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
            target,
            {0, 0, 0, 128}
        };
        request.useCache = false;
        request.context.floatMode = ShaderRecompiler::ShaderFloatMode{0xc0u, false, ieee, true};
        return ShaderRecompiler::Recompile(request).spirv.Copy();
    };
    for (const auto& [code, name] : {std::pair{std::span<const std::uint32_t>(PackedCode), "packed add, mul, fma and fmac"}, std::pair{std::span<const std::uint32_t>(ScalarFmaCode), "scalar fma forms"}}) {
        const auto selected = compile(code, false, false);
        const auto host = compile(code, true, false);
        Require(Selects(host) < Selects(selected), std::string("f16 NaN probe: the ") + name + " with float16HostNans and IEEE_MODE clear have " + std::to_string(Selects(host)) + " selects, without " + std::to_string(Selects(selected)));
        Require(compile(code, true, true) == compile(code, false, true), std::string("f16 NaN probe: float16HostNans changed the ") + name + " with IEEE_MODE set");
    }
    VerifyFormSpirv(target);
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--verify-spirv") {
            VerifySpirv();
            std::puts("f16 NaN results drop their selection only when the target says the device returns RDNA's NaN with IEEE_MODE clear");
            return 0;
        }
        RequestFloat16FormProbes();
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto target = device->Target();
        const auto supports = [&](spv::Capability capability) {
            return std::ranges::find(target.supportedCapabilities, static_cast<std::uint32_t>(capability)) != target.supportedCapabilities.end();
        };
        if (!supports(spv::CapabilityFloat16) || !supports(spv::CapabilityRoundingModeRTE) || !supports(spv::CapabilityDenormPreserve)) {
            std::puts("f16 NaN probe: skipped, the device has no native f16 with RTE rounding and kept denormals");
            return VulkanTestSkipped;
        }
        const auto probe = AgcDriver::ProbeFloat16HostNans(*device, target);
        std::printf("f16 NaN probe: %u of %u results differ from the selected NaN; the device %s its own NaN results\n", probe.mismatches, probe.compared, target.float16HostNans ? "keeps" : "does not keep");
        Require(probe.compared != 0u, "f16 NaN probe: nothing was compared");
        Require(!target.float16HostNans || probe.mismatches == 0u, "f16 NaN probe: the device keeps its NaN results although they differ from RDNA's");
        Require(target.float16HostNans || probe.mismatches != 0u || std::getenv("APS5_NO_HOST_F16_NANS") != nullptr, "f16 NaN probe: the device matches RDNA but keeps the selection");
        const auto compute = device->ComputeTarget(32u);
        Require(!target.float16NativeMinMax, "f16 min/max probe: the graphics target carries the compute-only native f16 min and max");
        const auto integer = AgcDriver::ProbeFloat16NativeMinMax(*device, WithoutFloat16Forms(compute), false);
        std::printf("f16 min/max probe with the integer order: %llu of %llu probe cases differ in %u float modes, %llu of %llu dispatched groups counted\n", static_cast<unsigned long long>(integer.mismatches), static_cast<unsigned long long>(integer.compared), integer.modes, static_cast<unsigned long long>(integer.groupsCounted), static_cast<unsigned long long>(integer.groupsExpected));
        Require(integer.compared != 0u && integer.mismatches == 0u && integer.groupsCounted == integer.groupsExpected, "f16 min/max probe: the probe's integer reference differs from today's f16 min and max");
        const auto native = AgcDriver::ProbeFloat16NativeMinMax(*device, WithoutFloat16Forms(compute), true);
        std::printf("f16 min/max probe with NMin and NMax: %llu of %llu probe cases differ (min %llu, max %llu) in %u float modes; the device %s them in compute programs\n", static_cast<unsigned long long>(native.mismatches), static_cast<unsigned long long>(native.compared), static_cast<unsigned long long>(native.byOperation[0]), static_cast<unsigned long long>(native.byOperation[1]), native.modes, compute.float16NativeMinMax ? "uses" : "does not use");
        Require(compute.float16NativeMinMax == (native.compared != 0u && native.mismatches == 0u), "f16 min/max probe: the device's verdict differs from a second run of the probe");
        std::puts("f16 NaN probe tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
