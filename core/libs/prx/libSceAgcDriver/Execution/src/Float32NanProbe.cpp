#include "prx/libSceAgcDriver/Execution/include/Float32NanProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <algorithm>
#include <array>
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

constexpr std::array<std::uint32_t, 16> Values{
    0x7fc00000u, 0xffc00000u, 0x7fc12345u, 0xffe54321u, 0x7f800001u, 0xff800001u, 0x7fa00000u, 0x7f800000u,
    0xff800000u, 0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0x00800000u, 0x00000001u,
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

void FillRows(std::uint32_t first) {
    for (std::uint32_t tid = 0; tid < Threads; ++tid) {
        const std::uint32_t combination = (first + tid) % Combinations;
        for (std::uint32_t operand = 0; operand < 3u; ++operand) Input[tid * Inputs + operand] = Values[(combination >> (operand * 4u)) & 15u];
        Input[tid * Inputs + 3u] = 0u;
    }
}

std::vector<std::uint32_t> Run(VulkanDevice& device, const ShaderRecompiler::RecompileResult& result, std::uint32_t first) {
    FillRows(first);
    Output.fill(0xdeadbeefu);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(Code.data()));
    device.WaitIdle();
    return {Output.begin(), Output.end()};
}

ShaderRecompiler::RecompileResult Compile(ShaderRecompiler::SpirvTarget target, bool hostNans, const std::optional<ShaderRecompiler::ShaderFloatMode>& floatMode) {
    std::vector<std::uint32_t> userData(8, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(Code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    target.float32HostNans = hostNans;
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

Float32NanProbeResult ProbeFloat32HostNans(VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, bool detail) {
    Float32NanProbeResult result;
    const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 3> modes{std::nullopt, ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false}, ShaderRecompiler::ShaderFloatMode{0xc0u, false, false, false}};
    for (std::uint32_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex) {
        const auto& mode = modes[modeIndex];
        const auto selected = Compile(target, false, mode);
        const auto host = Compile(target, true, mode);
        for (std::uint32_t first = 0; first < Combinations; first += Threads) {
            const auto expected = Run(device, selected, first);
            const auto actual = Run(device, host, first);
            for (std::uint32_t index = 0; index < expected.size(); ++index) {
                ++result.compared;
                if (expected[index] == actual[index]) continue;
                ++result.mismatches;
                ++result.byResult[index % Results];
                ++result.byMode[modeIndex];
                if (detail && result.rows.size() < 48u && result.byResult[index % Results] <= 3u) {
                    const std::uint32_t combination = (first + index / Results) % Combinations;
                    result.rows.push_back({modeIndex, index % Results, {Values[combination & 15u], Values[(combination >> 4u) & 15u], Values[(combination >> 8u) & 15u]}, expected[index], actual[index]});
                }
            }
            if (result.mismatches != 0u && !detail) return result;
        }
    }
    return result;
}

}
