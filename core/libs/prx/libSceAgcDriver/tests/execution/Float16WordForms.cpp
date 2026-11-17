#include "prx/libSceAgcDriver/Execution/include/GuestCodeWords.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;
using namespace AgcDriver::GuestCode;

constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t Inputs = 4;
constexpr std::uint32_t Results = 64;
constexpr std::uint32_t Batches = 3;
constexpr std::uint32_t FirstResult = 10;
alignas(256) std::array<std::uint32_t, Threads * Inputs> Input{};
alignas(256) std::array<std::uint32_t, Threads * Results> Output{};

struct Variant {
    std::vector<std::uint32_t> words;
    std::string name;
    bool constantMinMax = false;
};

std::uint64_t Seed = 0x2545f4914f6cdd1dull;

std::uint32_t Next() {
    Seed += 0x9e3779b97f4a7c15ull;
    std::uint64_t z = Seed;
    z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
    return static_cast<std::uint32_t>(z ^ (z >> 31u));
}

std::uint32_t Half() {
    const std::uint32_t bits = Next();
    const std::uint32_t sign = (bits >> 31u) << 15u;
    const std::uint32_t mantissa = bits & 0x3ffu;
    switch ((bits >> 16u) % 12u) {
    case 0u: return sign;
    case 1u: return sign | std::max(mantissa, 1u);
    case 2u: return sign | 0x7c00u;
    case 3u: return sign | 0x7c00u | std::max(mantissa, 1u);
    case 4u: return sign | 0x7e00u | mantissa;
    case 5u: return sign | ((0x1du + (bits >> 20u) % 2u) << 10u) | mantissa;
    case 6u: return sign | ((1u + (bits >> 20u) % 4u) << 10u) | mantissa;
    case 7u: return sign | ((0x0du + (bits >> 20u) % 5u) << 10u) | mantissa;
    default: return bits & 0xffffu;
    }
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

std::string Hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

void AddPacked(std::vector<Variant>& variants, const char* name, std::uint32_t op, std::uint32_t sources, std::uint32_t opSel, std::uint32_t opSelHi, std::uint32_t neg, std::uint32_t negHi, bool clamp, std::uint32_t second = Vgpr(5u), std::optional<std::uint32_t> literal = std::nullopt) {
    const std::uint32_t used = sources == 3u ? 7u : 3u;
    const std::uint32_t selectHigh = (opSelHi & used) | (sources == 3u ? 0u : 4u);
    Variant variant{{Vop3pLow(op, 0u, opSel & used, selectHigh, negHi & used, clamp), Vop3pHigh(Vgpr(4u), second, sources == 3u ? Vgpr(6u) : 0u, selectHigh, neg & used)}, name};
    if (literal.has_value()) variant.words.push_back(*literal);
    variant.name += " op_sel " + std::to_string(opSel & used) + " op_sel_hi " + std::to_string(selectHigh) + " neg_lo " + std::to_string(neg & used) + " neg_hi " + std::to_string(negHi & used) + (clamp ? " clamp" : "");
    variants.push_back(std::move(variant));
}

void AddScalar(std::vector<Variant>& variants, const char* name, std::uint32_t op, bool ternary, std::uint32_t opSel, std::uint32_t abs, std::uint32_t neg, bool clamp) {
    const std::uint32_t used = ternary ? 7u : 3u;
    Variant variant{{Vop3Low(op, 0u, ternary ? opSel : 0u, abs & used, clamp), Vop3High(Vgpr(4u), Vgpr(5u), ternary ? Vgpr(6u) : 0u, neg & used)}, name};
    variant.name += " op_sel " + std::to_string(ternary ? opSel : 0u) + " abs " + std::to_string(abs & used) + " neg " + std::to_string(neg & used) + (clamp ? " clamp" : "");
    variants.push_back(std::move(variant));
}

std::vector<Variant> Variants() {
    std::vector<Variant> variants;
    for (const auto& [op, name] : {std::pair{0x0fu, "v_pk_add_f16"}, std::pair{0x10u, "v_pk_mul_f16"}}) {
        for (std::uint32_t combo = 0; combo < 512u; ++combo) {
            AddPacked(variants, name, op, 2u, combo & 3u, (combo >> 2u) & 3u, (combo >> 4u) & 3u, (combo >> 6u) & 3u, combo >= 256u);
        }
    }
    for (std::uint32_t source = 0; source < 3u; ++source) {
        for (std::uint32_t combo = 0; combo < 16u; ++combo) {
            for (const bool clamp : {false, true}) {
                const std::uint32_t others = (combo * 5u + source * 3u + (clamp ? 7u : 0u)) & 15u;
                const auto field = [&](std::uint32_t bit) {
                    std::uint32_t value = 0u;
                    for (std::uint32_t index = 0; index < 3u; ++index) value |= (((index == source ? combo : others) >> bit) & 1u) << index;
                    return value;
                };
                AddPacked(variants, "v_pk_fma_f16", 0x0eu, 3u, field(0u), field(1u), field(2u), field(3u), clamp);
            }
        }
    }
    for (std::uint32_t combo = 0; combo < 64u; ++combo) {
        const std::uint32_t bits = Next();
        AddPacked(variants, "v_pk_fma_f16", 0x0eu, 3u, bits & 7u, (bits >> 3u) & 7u, (bits >> 6u) & 7u, (bits >> 9u) & 7u, (bits >> 12u) & 1u);
    }
    const std::array<std::pair<std::uint32_t, std::optional<std::uint32_t>>, 5> constants{{{LiteralSource, 0x3c00c400u}, {LiteralSource, 0x7e00bc01u}, {LiteralSource, 0x83ff7bffu}, {242u, std::nullopt}, {245u, std::nullopt}}};
    for (const auto& [second, literal] : constants) {
        for (std::uint32_t select = 0; select < 4u; ++select) {
            AddPacked(variants, "v_pk_add_f16 with a constant", 0x0fu, 2u, (select & 1u) << 1u, (select & 2u) | 1u, select & 1u, (select >> 1u) & 1u, false, second, literal);
            AddPacked(variants, "v_pk_mul_f16 with a constant", 0x10u, 2u, (select & 1u) << 1u, (select & 2u) | 1u, (select >> 1u) & 1u, select & 1u, select == 3u, second, literal);
        }
    }
    variants.push_back({{Vop2(0x3cu, 0u, Vgpr(4u), 5u)}, "v_pk_fmac_f16"});
    for (const auto& [op, name] : {std::pair{0x132u, "v_add_f16"}, std::pair{0x133u, "v_sub_f16"}, std::pair{0x134u, "v_subrev_f16"}, std::pair{0x135u, "v_mul_f16"}}) {
        for (std::uint32_t combo = 0; combo < 32u; ++combo) AddScalar(variants, name, op, false, 0u, combo & 3u, (combo >> 2u) & 3u, combo >= 16u);
    }
    for (const auto& [op, name] : {std::pair{0x32u, "v_add_f16 e32"}, std::pair{0x33u, "v_sub_f16 e32"}, std::pair{0x34u, "v_subrev_f16 e32"}, std::pair{0x35u, "v_mul_f16 e32"}}) {
        variants.push_back({{Vop2(op, 0u, Vgpr(4u), 5u)}, name});
        variants.push_back({{Vop2(op, 0u, 242u, 5u)}, std::string(name) + " with 1.0"});
        variants.push_back({{Vop2(op, 0u, LiteralSource, 5u), 0x00003555u}, std::string(name) + " with a literal"});
    }
    variants.push_back({{Vop2(0x36u, 0u, Vgpr(4u), 5u)}, "v_fmac_f16"});
    variants.push_back({{Vop2(0x37u, 0u, Vgpr(4u), 5u), 0x00003d55u}, "v_fmamk_f16"});
    variants.push_back({{Vop2(0x38u, 0u, Vgpr(4u), 5u), 0x00000001u}, "v_fmaak_f16"});
    for (std::uint32_t opSel = 0; opSel < 16u; ++opSel) {
        for (std::uint32_t abs = 0; abs < 8u; ++abs) AddScalar(variants, "v_fma_f16", 0x34bu, true, opSel, abs, (opSel * 3u + abs) & 7u, ((opSel ^ abs) & 1u) != 0u);
    }
    for (const auto& [op, name] : {std::pair{0x11u, "v_pk_min_f16"}, std::pair{0x12u, "v_pk_max_f16"}}) {
        for (std::uint32_t combo = 0; combo < 512u; combo += 8u) {
            AddPacked(variants, name, op, 2u, combo & 3u, (combo >> 2u) & 3u, (combo >> 4u) & 3u, (combo >> 6u) & 3u, ((combo >> 3u) & 1u) != 0u);
        }
    }
    for (const auto& [op, name] : {std::pair{0x13au, "v_min_f16"}, std::pair{0x139u, "v_max_f16"}}) {
        for (std::uint32_t combo = 0; combo < 32u; ++combo) AddScalar(variants, name, op, false, 0u, combo & 3u, (combo >> 2u) & 3u, combo >= 16u);
    }
    variants.push_back({{Vop2(0x3au, 0u, Vgpr(4u), 5u)}, "v_min_f16 e32"});
    variants.push_back({{Vop2(0x39u, 0u, Vgpr(4u), 5u)}, "v_max_f16 e32"});
    for (const auto& [op, name] : {std::pair{0x351u, "v_min3_f16"}, std::pair{0x354u, "v_max3_f16"}, std::pair{0x357u, "v_med3_f16"}}) {
        for (std::uint32_t combo = 0; combo < 16u; ++combo) AddScalar(variants, name, op, true, 0u, combo & 7u, (combo * 3u) & 7u, combo >= 8u);
    }
    variants.push_back({{Vop2(0x3au, 0u, 242u, 5u)}, "v_min_f16 e32 with 1.0", true});
    variants.push_back({{Vop2(0x39u, 0u, LiteralSource, 5u), 0x00007e01u}, "v_max_f16 e32 with a NaN literal", true});
    variants.push_back({{Vop2(0x3au, 0u, 20u, 5u)}, "v_min_f16 e32 with an SGPR constant", true});
    variants.push_back({{Vop2(0x39u, 0u, 20u, 5u)}, "v_max_f16 e32 with an SGPR constant", true});
    AddPacked(variants, "v_pk_max_f16 with an SGPR constant", 0x12u, 2u, 0u, 3u, 0u, 0u, false, 20u);
    variants.back().constantMinMax = true;
    AddPacked(variants, "v_pk_min_f16 with an SGPR constant", 0x11u, 2u, 1u, 2u, 2u, 1u, true, 20u);
    variants.back().constantMinMax = true;
    AddPacked(variants, "v_pk_min_f16 with 0.5", 0x11u, 2u, 0u, 3u, 0u, 0u, true, 240u);
    variants.back().constantMinMax = true;
    variants.push_back({{Vop3Low(0x351u, 0u, 0u, 0u, false), Vop3High(Vgpr(4u), Vgpr(5u), 242u, 0u)}, "v_min3_f16 with 1.0", true});
    variants.push_back({{Vop3Low(0x354u, 0u, 0u, 0u, true), Vop3High(20u, 20u, Vgpr(5u), 0u)}, "v_max3_f16 clamp with an SGPR constant twice", true});
    return variants;
}

std::vector<std::uint32_t> Program(std::span<const Variant> variants) {
    std::vector<std::uint32_t> code{Sop1(0x03u, 20u, LiteralSource), 0x7e01bc00u, Vop2(0x1au, 1u, Inline(4u), 0u), Vop2(0x1au, 3u, Inline(8u), 0u), MubufLow(0x0eu, 0u, true), MubufHigh(1u, 4u, 0u, Inline(0u)), WaitVmcnt0};
    for (std::uint32_t index = 0; index < Results; ++index) code.push_back(Vop1(0x01u, FirstResult + index, Vgpr(4u + index % 4u)));
    for (std::uint32_t index = 0; index < variants.size(); ++index) {
        auto words = variants[index].words;
        const std::uint32_t destination = FirstResult + index;
        const std::uint32_t family = words[0] >> 26u;
        if ((words[0] >> 31u) == 0u) words[0] |= destination << 17u;
        else if (family == 0x35u || (words[0] >> 24u) == 0xccu) words[0] |= destination;
        code.insert(code.end(), words.begin(), words.end());
    }
    for (std::uint32_t group = 0; group < Results / 4u; ++group) {
        code.push_back(MubufLow(0x1eu, group * 16u, true));
        code.push_back(MubufHigh(3u, FirstResult + group * 4u, 1u, Inline(0u)));
    }
    code.push_back(Endpgm);
    return code;
}

}

int main() {
    try {
        RequestFloat16FormProbes();
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto deviceTarget = device->Target();
        if (!TargetHasCapability(deviceTarget, spv::CapabilityFloat16) || !TargetHasCapability(deviceTarget, spv::CapabilityRoundingModeRTE) || !TargetHasCapability(deviceTarget, spv::CapabilityDenormPreserve)) {
            std::puts("f16 word forms: skipped, the device has no native f16 with RTE rounding and kept denormals");
            return VulkanTestSkipped;
        }
        const auto baseline = WithoutFloat16Forms(deviceTarget);
        const auto forms = WithFloat16Forms(*device);
        auto forced = forms;
        forced.float16NativeMinMax = true;
        const auto variants = Variants();
        std::vector<Variant> constantVariants;
        std::copy_if(variants.begin(), variants.end(), std::back_inserter(constantVariants), [](const Variant& variant) { return variant.constantMinMax; });
        const std::array<std::optional<ShaderRecompiler::ShaderFloatMode>, 5> modes{
            std::nullopt,
            ShaderRecompiler::ShaderFloatMode{0xc0u, true, false, false},
            ShaderRecompiler::ShaderFloatMode{0xc0u, false, false, true},
            ShaderRecompiler::ShaderFloatMode{0xf0u, true, false, true},
            ShaderRecompiler::ShaderFloatMode{0xc0u, true, true, false},
        };
        std::vector<std::uint32_t> userData(8, 0u);
        const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
        const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
        std::copy(input.begin(), input.end(), userData.begin());
        std::copy(output.begin(), output.end(), userData.begin() + 4);
        const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
        const auto check = [&](std::span<const Variant> selected, const ShaderRecompiler::SpirvTarget& formTarget, const char* label) {
            std::uint64_t compared = 0u;
            for (std::size_t first = 0; first < selected.size(); first += Results) {
                const std::span<const Variant> chunk(selected.data() + first, std::min<std::size_t>(Results, selected.size() - first));
                const auto code = Program(chunk);
                const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(std::span<const std::uint32_t>(code))}}};
                const auto compile = [&](const ShaderRecompiler::SpirvTarget& target, const std::optional<ShaderRecompiler::ShaderFloatMode>& mode) {
                    ShaderRecompiler::RecompileRequest request{
                        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
                        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
                        target,
                        {0, 0, 0, 128}
                    };
                    request.useCache = false;
                    request.context.floatMode = mode;
                    return ShaderRecompiler::Recompile(request);
                };
                const auto run = [&](const ShaderRecompiler::RecompileResult& result) {
                    Output.fill(0xdeadbeefu);
                    device->Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
                    device->WaitIdle();
                    return Output;
                };
                for (const auto& mode : modes) {
                    const auto today = compile(baseline, mode);
                    const auto word = compile(formTarget, mode);
                    for (std::uint32_t batch = 0; batch < Batches; ++batch) {
                        for (auto& value : Input) value = Half() | (Half() << 16u);
                        const auto expected = run(today);
                        const auto actual = run(word);
                        for (std::uint32_t tid = 0; tid < Threads; ++tid) {
                            const std::uint32_t* row = &Input[tid * Inputs];
                            for (std::uint32_t index = 0; index < chunk.size(); ++index) {
                                const std::uint32_t slot = tid * Results + index;
                                ++compared;
                                if (actual[slot] == expected[slot]) continue;
                                Require(false, std::string("f16 word forms") + label + ": " + chunk[index].name + (mode.has_value() ? " in FLOAT_MODE " + Hex(mode->floatMode) + (mode->dx10Clamp ? " dx10" : "") + (mode->ieeeMode ? " ieee" : "") + (mode->fp16Overflow ? " fp16 overflow" : "") : std::string(" without a float mode")) + " with v4=" + Hex(row[0]) + " v5=" + Hex(row[1]) + " v6=" + Hex(row[2]) + " v7=" + Hex(row[3]) + " s20=0x7e01bc00 is " + Hex(actual[slot]) + " with the f16 word forms, " + Hex(expected[slot]) + " without");
                            }
                        }
                    }
                }
            }
            return compared;
        };
        const std::uint64_t compared = check(variants, forms, "");
        const std::uint64_t constantCompared = check(constantVariants, forced, " with native f16 min and max forced on");
        std::printf("f16 word forms tests passed (%zu instruction forms, %llu results; native f16 min and max %s by the device's probe; %zu forms with constant min/max operands, %llu results with native f16 min and max forced on)\n", variants.size(), static_cast<unsigned long long>(compared), forms.float16NativeMinMax ? "taken" : "not taken", constantVariants.size(), static_cast<unsigned long long>(constantCompared));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
