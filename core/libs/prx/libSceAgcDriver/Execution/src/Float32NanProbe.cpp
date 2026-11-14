#include "prx/libSceAgcDriver/Execution/include/Float32NanProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <algorithm>
#include <array>
#include <exception>
#include <optional>
#include <span>
#include <vector>

namespace AgcDriver {

namespace {

using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 256;
constexpr std::uint32_t Inputs = 4;
constexpr std::uint32_t Results = 32;
constexpr std::uint32_t Combinations = 4096;

alignas(256) std::array<std::uint32_t, Threads * Inputs> Input{};
alignas(256) std::array<std::uint32_t, Threads * Results> Output{};

alignas(256) constexpr std::array<std::uint32_t, 50> Code{
    0x34020084u, 0x34060087u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u,
    0x06140b04u,
    0x08160b04u,
    0x0a180b04u,
    0x101a0b04u,
    0xd54b000eu, 0x041a0b04u,
    0xd541000fu, 0x041a0b04u,
    0xd54b0010u, 0x04160906u,
    0x06220905u,
    0x10240d05u,
    0x7e260306u, 0x56260b04u,
    0xd5030014u, 0x20020b04u,
    0xd5080115u, 0x00020b04u,
    0xd54b0016u, 0x841a0b04u,
    0x082e0906u,
    0x10300d06u,
    0xd54b0019u, 0x04120b05u,
    0x063408f2u,
    0x083608f2u,
    0x0a3808f2u,
    0x103a08f4u,
    0x103c0880u,
    0x103e08f6u,
    0x064008f3u,
    0x084208f0u,
    0xe0781000u, 0x80010a03u, 0xe0781010u, 0x80010e03u, 0xe0781020u, 0x80011203u, 0xe0781030u, 0x80011603u, 0xe0781040u, 0x80011a03u, 0xe0781050u, 0x80011e03u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 70> FusedCode{
    0x34020084u, 0x34060087u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u,
    0x06140b04u,
    0x08160b04u,
    0x0a180b04u,
    0x101a0b04u,
    0xd54b000eu, 0x041a0b04u,
    0xd541000fu, 0x041a0b04u,
    0xd54b0010u, 0x04160906u,
    0x06220905u,
    0x10240d05u,
    0x7e260306u, 0x56260b04u,
    0xd5030014u, 0x20020b04u,
    0xd5080115u, 0x00020b04u,
    0xd54b0016u, 0x841a0b04u,
    0x082e0906u,
    0x10300d06u,
    0xd54b0019u, 0x04120b05u,
    0x063408f2u,
    0x083608f2u,
    0x0a3808f2u,
    0x103a08f4u,
    0x103c0880u,
    0x103e08f6u,
    0x064008f3u,
    0x084208f0u,
    0xd54b0022u, 0x041a0af2u,
    0xd54b0023u, 0x0419e704u,
    0xd54b0024u, 0x03ca0b04u,
    0xd54b0025u, 0x0419e904u,
    0x584c0d04u, 0x7f800000u,
    0x584e0d04u, 0x00000001u,
    0x5a500b04u, 0x80000000u,
    0xd54b0029u, 0x041a0904u,
    0xe0781000u, 0x80010a03u, 0xe0781010u, 0x80010e03u, 0xe0781020u, 0x80011203u, 0xe0781030u, 0x80011603u, 0xe0781040u, 0x80011a03u, 0xe0781050u, 0x80011e03u,
    0xe0781060u, 0x80012203u, 0xe0781070u, 0x80012603u, 0xbf810000u,
};

constexpr std::uint32_t FusedResults = (1u << 4u) | (1u << 5u) | (1u << 6u) | (1u << 9u) | (1u << 12u) | (1u << 15u) | 0xff000000u;

alignas(256) constexpr std::array<std::uint32_t, 49> MinMaxCode{
    0x34020084u, 0x34060087u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u,
    0x1e140b04u,
    0x20160b04u,
    0x1e180905u,
    0x201a0905u,
    0xd551000eu, 0x041a0b04u,
    0xd554000fu, 0x041a0b04u,
    0xd5570010u, 0x041a0b04u,
    0xd5570011u, 0x04160906u,
    0xd5510012u, 0x04120d05u,
    0xd5540013u, 0x04160906u,
    0x20280880u,
    0x1e2a08f2u,
    0xd5570016u, 0x03c90104u,
    0xd5570017u, 0x03ce08f2u,
    0x20300904u,
    0xd50f0219u, 0x20020b04u,
    0xd551001au, 0x04150104u,
    0xd557001bu, 0x02020b04u,
    0xd510001cu, 0x00010104u,
    0xd557001du, 0x0201e504u,
    0xe0781000u, 0x80010a03u, 0xe0781010u, 0x80010e03u, 0xe0781020u, 0x80011203u, 0xe0781030u, 0x80011603u, 0xe0781040u, 0x80011a03u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 16> Values{
    0x7fc00000u, 0xffc00000u, 0x7fc12345u, 0xffe54321u, 0x7f800001u, 0xff800001u, 0x7fa00000u, 0x7f800000u,
    0xff800000u, 0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0x00800000u, 0x00000001u,
};

constexpr std::array<std::uint32_t, 16> MinMaxValues{
    0x7fc00000u, 0xffc00000u, 0x7fc12345u, 0xff812345u, 0x7f800001u, 0x7f800000u, 0xff800000u, 0x00000000u,
    0x80000000u, 0x3f800000u, 0xbf800000u, 0x40400000u, 0x00000001u, 0x80000001u, 0x00800000u, 0x7f7fffffu,
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

void FillRows(std::uint32_t first, const std::array<std::uint32_t, 16>& values) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const std::uint32_t combination = (first + tid) % Combinations;
        for (std::uint32_t operand = 0; operand < 3u; ++operand) Input[tid * Inputs + operand] = values[(combination >> (operand * 4u)) & 15u];
        Input[tid * Inputs + 3u] = 0u;
    }
}

std::vector<std::uint32_t> Run(VulkanDevice& device, const ShaderRecompiler::RecompileResult& result, std::span<const std::uint32_t> code, const std::array<std::uint32_t, 16>& values, std::uint32_t first) {
    FillRows(first, values);
    Output.fill(0xdeadbeefu);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    return {Output.begin(), Output.end()};
}

ShaderRecompiler::RecompileResult Compile(const ShaderRecompiler::SpirvTarget& target, std::span<const std::uint32_t> code, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode) {
    std::vector<std::uint32_t> userData(8, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
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

}

Float32NanProbeResult ProbeFloat32HostNans(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, bool detail, bool fused) {
    Float32NanProbeResult result;
    const std::span<const std::uint32_t> code = fused ? std::span<const std::uint32_t>(FusedCode) : std::span<const std::uint32_t>(Code);
    auto selectedTarget = target;
    selectedTarget.float32HostNans = false;
    selectedTarget.float32HostFmaNans = false;
    auto hostTarget = target;
    hostTarget.float32HostNans = true;
    hostTarget.float32HostFmaNans = fused;
    const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 3> modes{std::nullopt, ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false}, ShaderRecompiler::ShaderFloatMode{0xc0u, false, false, false}};
    for (std::uint32_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        const auto selected = Compile(selectedTarget, code, mode);
        const auto host = Compile(hostTarget, code, mode);
        for (std::uint32_t first = 0; first < Combinations; first += Threads) {
            const auto expected = Run(device, selected, code, Values, first);
            const auto actual = Run(device, host, code, Values, first);
            for (std::uint32_t index = 0; index < expected.size(); ++index) {
                const std::uint32_t slot = index % Results;
                const bool fusedSlot = fused && ((FusedResults >> slot) & 1u) != 0u;
                if (fusedSlot) ++result.fusedCompared;
                else ++result.compared;
                if (expected[index] == actual[index]) continue;
                if (fusedSlot) ++result.fusedMismatches;
                else ++result.mismatches;
                ++result.byResult[slot];
                ++result.byMode[modeIndex];
                if (detail && result.rows.size() < 48u && result.byResult[slot] <= 3u) {
                    const std::uint32_t combination = (first + index / Results) % Combinations;
                    result.rows.push_back({modeIndex, slot, {Values[combination & 15u], Values[(combination >> 4u) & 15u], Values[(combination >> 8u) & 15u]}, expected[index], actual[index]});
                }
            }
            if (result.mismatches != 0u && !detail) return result;
        }
    }
    return result;
}

Float32MinMaxProbeResult ProbeFloat32NativeMinMax(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, bool detail) {
    Float32MinMaxProbeResult result;
    const std::span<const std::uint32_t> code(MinMaxCode);
    auto selectedTarget = target;
    selectedTarget.float32NativeMinMax = false;
    auto nativeTarget = target;
    nativeTarget.float32NativeMinMax = true;
    const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 6> modes{
        std::nullopt,
        ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false},
        ShaderRecompiler::ShaderFloatMode{0xc0u, false, false, false},
        ShaderRecompiler::ShaderFloatMode{0xf0u, true, false, false},
        ShaderRecompiler::ShaderFloatMode{0xc0u, true, true, false},
        ShaderRecompiler::ShaderFloatMode{0xf0u, true, true, false},
    };
    for (std::uint32_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        std::optional<ShaderRecompiler::RecompileResult> selected;
        try {
            selected.emplace(Compile(selectedTarget, code, mode));
        } catch (const std::exception&) {
            ++result.skippedModes;
            continue;
        }
        std::optional<ShaderRecompiler::RecompileResult> native;
        try {
            native.emplace(Compile(nativeTarget, code, mode));
        } catch (const std::exception&) {
            ++result.mismatches;
            return result;
        }
        for (std::uint32_t first = 0; first < Combinations; first += Threads) {
            const auto expected = Run(device, *selected, code, MinMaxValues, first);
            const auto actual = Run(device, *native, code, MinMaxValues, first);
            for (std::uint32_t index = 0; index < expected.size(); ++index) {
                ++result.compared;
                if (expected[index] == actual[index]) continue;
                const std::uint32_t slot = index % Results;
                ++result.mismatches;
                ++result.byResult[slot];
                ++result.byMode[modeIndex];
                if (detail && result.rows.size() < 48u && result.byResult[slot] <= 3u) {
                    const std::uint32_t combination = (first + index / Results) % Combinations;
                    result.rows.push_back({modeIndex, slot, {MinMaxValues[combination & 15u], MinMaxValues[(combination >> 4u) & 15u], MinMaxValues[(combination >> 8u) & 15u]}, expected[index], actual[index]});
                }
            }
            if (result.mismatches != 0u && !detail) return result;
        }
    }
    return result;
}

}
