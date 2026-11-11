#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectDrawReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/StagePlanReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::ShaderResources;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;
using AgcDriver::Graphics::RuntimeSwitch;
namespace StagePlanReport = AgcDriver::DriverDetail::StagePlans;
using PlanCounter = StagePlanReport::Counter;
namespace DirectReport = AgcDriver::DriverDetail::DirectDraws;
using DirectCounter = DirectReport::Counter;

constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t ColorBytes = 65536;
constexpr std::uint32_t RenderTargetTileMode = 0x1b;
constexpr std::uint32_t ColorInfo = 0x28028;
constexpr std::uint32_t ForeignColorInfo = 0x28014;
constexpr std::size_t ConstantsOffset = 0xc00;
constexpr std::uint32_t TextureUserWords = 12;
constexpr std::uint32_t Draws = 6;

alignas(256) constexpr std::array<std::uint32_t, 16> PlainVertex{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
};

alignas(256) constexpr std::array<std::uint32_t, 10> ConstantPixel{0xf4000100u, 0xfa000000u, 0xbf8cc07fu, 0x7e000204u, 0x06000002u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u};

alignas(256) constexpr std::array<std::uint32_t, 12> SamplePixel{0x7e0002f0u, 0x7e0202f0u, 0xf09c0f08u, 0x00400000u, 0xbf8c0000u, 0xf800180fu, 0x03020100u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u};

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header PlainVertexHeader{};
alignas(8) Header ConstantPixelHeader{};
alignas(8) Header SamplePixelHeader{};

void Register(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, BlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + BlockBytes != begin + BlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "clean resident views: cannot allocate a guest block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint64_t At(std::size_t offset) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block + offset)); }
    std::uint8_t* Bytes(std::size_t offset) const { return block + offset; }

private:
    std::uint8_t* block = nullptr;
};

using RegisterWrites = std::vector<std::pair<std::uint32_t, std::uint32_t>>;

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

void SubmitDraws(std::uint64_t vertex, std::uint64_t pixel, const RegisterWrites& userConfig, const RegisterWrites& shaderRegisters, const GuestBlock& target, std::uint32_t draws, std::uint32_t colorInfo) {
    const auto color = target.At(0);
    const float scale = 32.0f;
    const RegisterWrites context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, colorInfo},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000u | (RenderTargetTileMode << 14u)}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x318, static_cast<std::uint32_t>(color >> 8u)}, {0x390, static_cast<std::uint32_t>(color >> 40u)},
        {0x10f, std::bit_cast<std::uint32_t>(scale)}, {0x110, std::bit_cast<std::uint32_t>(scale)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
    };
    std::vector<std::uint32_t> commands;
    for (const auto& [offset, value] : userConfig) Append(commands, 0x79, {offset, value});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    Append(commands, 0x76, {0xc8, static_cast<std::uint32_t>(vertex >> 8u)});
    Append(commands, 0x76, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)});
    Append(commands, 0x76, {0x008, static_cast<std::uint32_t>(pixel >> 8u), static_cast<std::uint32_t>(pixel >> 40u)});
    for (const auto& [offset, value] : shaderRegisters) Append(commands, 0x76, {offset, value});
    for (std::uint32_t draw = 0; draw < draws; ++draw) Append(commands, 0x2d, {3u, 2});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "clean resident views: a draw submission failed");
    AgcDriverWaitIdle_nid_postfix();
}

std::array<std::uint32_t, 8> ViewedDescriptor(std::uint64_t address) {
    return {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (56u << 20u) | (3u << 30u), 15u | (3u << 14u), 0xfacu | (RenderTargetTileMode << 20u) | (9u << 28u), 0u, 0u, 0u, 0u};
}

struct Scene {
    GuestBlock data;
    GuestBlock viewed;
    GuestBlock target;
};

Scene* Blocks = nullptr;

enum class Step { Steady, CpuWrite, StoredBack, Foreign };

const char* StepName(Step step) {
    switch (step) {
    case Step::Steady: return "steady";
    case Step::CpuWrite: return "a CPU write under the pending surface";
    case Step::StoredBack: return "the surface stored back";
    default: return "a new pending surface over it";
    }
}

struct Submission {
    std::uint32_t value = 0;
    Step step = Step::Steady;
};

std::vector<std::uint8_t> Image(const GuestBlock& target) {
    AgcDriver::GuestMemory::FlushGpuWrites(static_cast<std::uintptr_t>(target.At(0)), ColorBytes);
    return {target.Bytes(0), target.Bytes(0) + ColorBytes};
}

std::vector<std::uint8_t> Render(const Submission& submission) {
    auto& blocks = *Blocks;
    const float constant = 0.125f * static_cast<float>(1u + submission.value % 6u);
    std::memcpy(blocks.data.Bytes(ConstantsOffset), &constant, sizeof(constant));
    const auto pointer = blocks.data.At(ConstantsOffset);
    SubmitDraws(reinterpret_cast<std::uintptr_t>(PlainVertex.data()), reinterpret_cast<std::uintptr_t>(ConstantPixel.data()), {{0x242, 4}}, {{0x8b, 0}, {0x8c, 0}, {0x00b, 3u << 1u}, {0x00c, static_cast<std::uint32_t>(pointer)}, {0x00d, static_cast<std::uint32_t>(pointer >> 32u)}, {0x00e, std::bit_cast<std::uint32_t>(0.25f)}}, blocks.viewed, 1, submission.step == Step::Foreign ? ForeignColorInfo : ColorInfo);
    if (submission.step == Step::CpuWrite) std::memset(blocks.viewed.Bytes(0), static_cast<int>(0x40u + 0x11u * submission.value), ColorBytes);
    if (submission.step == Step::StoredBack) AgcDriver::GuestMemory::FlushGpuWrites(static_cast<std::uintptr_t>(blocks.viewed.At(0)), ColorBytes);
    std::memset(blocks.target.Bytes(0), 0, ColorBytes);
    const auto image = ViewedDescriptor(blocks.viewed.At(0));
    RegisterWrites registers{{0x8b, 0}, {0x8c, 0}, {0x00b, TextureUserWords << 1u}};
    for (std::uint32_t word = 0; word < TextureUserWords; ++word) registers.emplace_back(0x00cu + word, word < image.size() ? image[word] : 0u);
    SubmitDraws(reinterpret_cast<std::uintptr_t>(PlainVertex.data()), reinterpret_cast<std::uintptr_t>(SamplePixel.data()), {{0x242, 4}, {0x24a, 0}}, registers, blocks.target, Draws, ColorInfo);
    return Image(blocks.target);
}

struct Totals {
    std::array<std::uint64_t, static_cast<std::size_t>(PlanCounter::Count)> plans{};
    std::array<std::uint64_t, static_cast<std::size_t>(DirectCounter::Count)> direct{};
    ShaderResources::CleanRuleCounts rules{};

    static Totals Now() {
        Totals totals;
        for (std::size_t index = 0; index < totals.plans.size(); ++index) totals.plans[index] = StagePlanReport::Total(static_cast<PlanCounter>(index));
        for (std::size_t index = 0; index < totals.direct.size(); ++index) totals.direct[index] = DirectReport::Total(static_cast<DirectCounter>(index));
        totals.rules = ShaderResources::CleanRuleCounted();
        return totals;
    }

    std::uint64_t Plans(const Totals& base, PlanCounter counter) const { return plans[static_cast<std::size_t>(counter)] - base.plans[static_cast<std::size_t>(counter)]; }
    std::uint64_t Direct(const Totals& base, DirectCounter counter) const { return direct[static_cast<std::size_t>(counter)] - base.direct[static_cast<std::size_t>(counter)]; }
    std::uint64_t Skipped(const Totals& base) const { return Plans(base, PlanCounter::ProofsSkipped) + Plans(base, PlanCounter::ProofsRequeried) + Direct(base, DirectCounter::ProofsSkipped); }
    std::uint64_t Verified(const Totals& base) const { return Plans(base, PlanCounter::ProofsVerified); }
    std::uint64_t Differed(const Totals& base) const { return Plans(base, PlanCounter::ProofsDiffered); }
};

std::string Describe(const Totals& now, const Totals& base) {
    std::string text;
    for (std::size_t index = 0; index < now.plans.size(); ++index) {
        const auto moved = now.Plans(base, static_cast<PlanCounter>(index));
        if (moved != 0) text += std::string("; ") + StagePlanReport::CounterName(static_cast<PlanCounter>(index)) + " " + std::to_string(moved);
    }
    for (std::size_t index = 0; index < now.direct.size(); ++index) {
        const auto moved = now.Direct(base, static_cast<DirectCounter>(index));
        if (moved != 0) text += std::string("; direct ") + DirectReport::CounterName(static_cast<DirectCounter>(index)) + " " + std::to_string(moved);
    }
    const auto rules = [&](const char* name, std::uint64_t current, std::uint64_t previous) {
        if (current != previous) text += std::string("; ") + name + " " + std::to_string(current - previous);
    };
    rules("resident-view hits", now.rules.viewHits, base.rules.viewHits);
    rules("pending-range hits", now.rules.pendingHits, base.rules.pendingHits);
    rules("pending-range records", now.rules.exemptRecords, base.rules.exemptRecords);
    rules("resident-view drops", now.rules.viewDrops, base.rules.viewDrops);
    rules("element proof drops", now.rules.sourceDrops, base.rules.sourceDrops);
    rules("rule hits verified", now.rules.verified, base.rules.verified);
    rules("rule hits differing", now.rules.differs, base.rules.differs);
    rules("other ranges not clean", now.rules.requiredNotClean, base.rules.requiredNotClean);
    rules("pending-range records written during the proof", now.rules.exemptUnsettled, base.rules.exemptUnsettled);
    rules("records without a viewed surface", now.rules.noSources, base.rules.noSources);
    rules("records with a snapshot over a viewed surface", now.rules.snapshotOverView, base.rules.snapshotOverView);
    rules("records with unrecorded elements", now.rules.unrecordedViews, base.rules.unrecordedViews);
    return text;
}

enum class Mode { Plans, Direct };

void Paths(Mode mode, bool on) {
    RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, true);
    RuntimeSwitches::Assign(RuntimeSwitch::DrawRecords, false);
    RuntimeSwitches::Assign(RuntimeSwitch::StagePlans, on && mode == Mode::Plans);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectDraws, on && mode == Mode::Direct);
    RuntimeSwitches::Assign(RuntimeSwitch::DirectCleanProofs, on && mode == Mode::Direct);
}

void Rules(bool views, bool pending) {
    RuntimeSwitches::Assign(RuntimeSwitch::CleanResidentViews, views);
    RuntimeSwitches::Assign(RuntimeSwitch::CleanPendingRanges, pending);
}

Totals RunSet(Mode mode, const std::string& name, const std::vector<Submission>& submissions) {
    Paths(mode, false);
    std::vector<std::vector<std::uint8_t>> today;
    for (const auto& submission : submissions) today.push_back(Render(submission));
    Paths(mode, true);
    const auto base = Totals::Now();
    for (std::size_t index = 0; index < submissions.size(); ++index) {
        const auto image = Render(submissions[index]);
        Require(image == today[index], name + ": submission " + std::to_string(index) + " (" + StepName(submissions[index].step) + ") rendered differently from today's path" + Describe(Totals::Now(), base));
    }
    Paths(mode, false);
    const auto now = Totals::Now();
    std::cout << name << ":" << Describe(now, base) << '\n';
    Require(now.rules.differs == base.rules.differs && now.Differed(base) == 0, name + ": the clean-currency verify found a hit of a new rule that disagrees with the full proof" + Describe(now, base));
    return now;
}

std::vector<Submission> SteadySet() {
    std::vector<Submission> submissions;
    for (std::uint32_t index = 0; index < 6; ++index) submissions.push_back({index % 2u, Step::Steady});
    return submissions;
}

std::vector<Submission> DropSet(Step step) {
    return {{0, Step::Steady}, {1, Step::Steady}, {2, step}, {3, Step::Steady}, {4, step}, {5, Step::Steady}};
}

int Run(Mode mode, bool ring) {
    const auto plans = mode == Mode::Plans;
    const bool verifyEvery = std::getenv("APS5_VERIFY_CLEAN_CURRENCY") != nullptr && std::string_view(std::getenv("APS5_VERIFY_CLEAN_CURRENCY")) == "1";
    {
        Paths(mode, false);
        Rules(false, false);
        const auto first = Render({0, Step::Steady});
        const auto second = Render({1, Step::Steady});
        Require(first != second, "clean resident views: two values of the sampled render target rendered the same image");
    }
#ifdef _WIN32
    constexpr bool cleanRanges = true;
#else
    constexpr bool cleanRanges = false;
#endif
    const bool checks = ring && plans && cleanRanges;
    Rules(false, false);
    const auto offBase = Totals::Now();
    const auto off = RunSet(mode, "without the new rules, draws sampling a pending render target", SteadySet());
    if (checks) {
        Require(off.rules.pendingHits == offBase.rules.pendingHits && off.rules.viewHits == offBase.rules.viewHits && off.rules.exemptRecords == offBase.rules.exemptRecords, "clean resident views: a new rule answered with its switch unset" + Describe(off, offBase));
    }
    Rules(false, true);
    const auto pendingBase = Totals::Now();
    const auto pending = RunSet(mode, "pending ranges, draws sampling a pending render target", SteadySet());
    if (checks) {
        Require(pending.rules.exemptRecords > pendingBase.rules.exemptRecords, "clean resident views: no template over a pending render target was recorded clean with its pending range exempt" + Describe(pending, pendingBase));
        if (verifyEvery) Require(pending.rules.verified > pendingBase.rules.verified, "clean resident views: the verify checked no pending-range hit" + Describe(pending, pendingBase));
        else Require(pending.rules.pendingHits > pendingBase.rules.pendingHits && pending.Skipped(pendingBase) > off.Skipped(offBase), "clean resident views: no draw sampling a pending render target skipped its proof through the pending-range rule" + Describe(pending, pendingBase));
        Require(pending.rules.viewHits == pendingBase.rules.viewHits, "clean resident views: the resident-view rule answered with its switch unset" + Describe(pending, pendingBase));
    }
    Rules(true, true);
    const auto bothBase = Totals::Now();
    const auto both = RunSet(mode, "resident views and pending ranges, draws sampling a pending render target", SteadySet());
    if (checks && !verifyEvery) Require(both.rules.pendingHits > bothBase.rules.pendingHits, "clean resident views: with both rules no draw hit through the pending-range rule" + Describe(both, bothBase));
    for (const auto step : {Step::CpuWrite, Step::StoredBack, Step::Foreign}) {
        const auto before = Totals::Now();
        const auto after = RunSet(mode, std::string("resident views and pending ranges, ") + StepName(step), DropSet(step));
        if (checks) Require(after.rules.sourceDrops > before.rules.sourceDrops || after.rules.viewDrops > before.rules.viewDrops, std::string("clean resident views: ") + StepName(step) + " dropped no kept proof" + Describe(after, before));
    }
    Rules(true, false);
    const auto viewsBase = Totals::Now();
    const auto views = RunSet(mode, "resident views alone, draws sampling a pending render target", SteadySet());
    if (checks) Require(views.rules.pendingHits == viewsBase.rules.pendingHits && views.rules.exemptRecords == viewsBase.rules.exemptRecords, "clean resident views: the pending-range rule answered with its switch unset" + Describe(views, viewsBase));
    Rules(false, false);
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        const bool direct = argc > 1 && std::string_view(argv[1]) == "--direct";
        using AgcDriver::DriverDetail::DrawFinisher;
        using AgcDriver::DriverDetail::DrawPipeline;
        const bool ring = DrawFinisher::Enabled() && DrawFinisher::Overlap() && DrawPipeline::Depth() != 0;
        const bool lean = RuntimeSwitches::Set(RuntimeSwitch::LeanFront);
        Register(PlainVertexHeader, PlainVertex.data(), sizeof(PlainVertex), 2);
        Register(ConstantPixelHeader, ConstantPixel.data(), sizeof(ConstantPixel), 1);
        Register(SamplePixelHeader, SamplePixel.data(), sizeof(SamplePixel), 1);
        Scene scene;
        Blocks = &scene;
        Run(direct ? Mode::Direct : Mode::Plans, ring);
        StagePlanReport::Report(true);
        DirectReport::Report(true);
        RuntimeSwitches::Assign(RuntimeSwitch::LeanFront, lean);
        AgcDriverShutdown_nid_postfix();
        std::cout << "draws sampling a render target an earlier pass left pending render as today's path with the resident-view and pending-range clean rules, and drop their kept proofs for a CPU write under the surface, a store-back and a new pending surface" << (ring ? "" : " (no finisher ring: plans never ran)") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { AgcDriverShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
