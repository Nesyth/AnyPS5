#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectDraws.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/SlimMutex.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/WriteValidation.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/SrtWalker.hpp"
#include "Optimization/SrtWalker/SrtAddressArithmetic.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include <cstdio>
#include <atomic>
#include <algorithm>
#include <new>
#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

using DirectDraws::Counter;
using DirectDraws::Step;
using DirectDraws::TrustEvent;
using DirectDraws::TrustSite;
using DirectDraws::TrustUse;
using Role = ShaderRecompiler::ProgramRole;
using Stage = ShaderRecompiler::ShaderStage;

constexpr std::size_t InvocationSlots = 1024;
constexpr std::size_t WideInvocationSlots = 16384;
constexpr std::uint64_t WalkPageBytes = 0x1000u;

std::size_t mixedSlot(std::uint64_t first, std::uint64_t second, std::size_t slots) {
    auto mixed = first * 0x9e3779b97f4a7c15ull ^ second;
    mixed ^= mixed >> 29u;
    return static_cast<std::size_t>(mixed % slots);
}

bool trusted() {
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpTrustFinisher)) return false;
    struct TrustedStorage {};
    auto& held = HostThreadLocal<std::uint64_t, TrustedStorage>();
    if (++held == 256u) {
        DirectDraws::AddTrusted(held);
        held = 0;
    }
    return true;
}

bool auditOn() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::TrustAudit) && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpTrustFinisher);
}

std::uint32_t auditEvery() {
    static const auto every = DirectDraws::TrustAuditEvery();
    return every;
}

std::uint32_t auditHold() {
    static const auto hold = DirectDraws::TrustAuditHold();
    return hold;
}

bool presentBefore(std::uint32_t until) {
    return static_cast<std::int32_t>(until - static_cast<std::uint32_t>(Graphics::Recorder::Presents())) > 0;
}

std::atomic<std::uint32_t> auditSeed{0x9e3779b9u};

std::uint32_t auditStart() {
    const auto next = auditSeed.load(std::memory_order_relaxed) * 1664525u + 1013904223u;
    auditSeed.store(next, std::memory_order_relaxed);
    return (next >> 16) % ((auditEvery() + 1u) / 2u);
}

std::uint32_t heldUntil() {
    const auto until = static_cast<std::uint32_t>(Graphics::Recorder::Presents()) + auditHold();
    return until != 0 ? until : 1u;
}

TrustUse auditUse(DirectDraws::TrustState& state) {
    if (!auditOn()) return TrustUse::Test;
    if (const auto until = state.volatileUntil.load(std::memory_order_relaxed); until != 0) {
        if (presentBefore(until)) return TrustUse::Volatile;
        auto expected = until;
        state.volatileUntil.compare_exchange_strong(expected, 0u, std::memory_order_relaxed);
    }
    const auto uses = state.uses.load(std::memory_order_relaxed) + 1u;
    const bool audit = uses >= auditEvery();
    state.uses.store(audit ? auditStart() : uses, std::memory_order_relaxed);
    return audit ? TrustUse::Audit : TrustUse::Skip;
}

TrustUse auditUse(DirectDraws::LocalTrustState& state) {
    if (!auditOn()) return TrustUse::Test;
    if (state.volatileUntil != 0) {
        if (presentBefore(state.volatileUntil)) return TrustUse::Volatile;
        state.volatileUntil = 0;
    }
    const auto uses = state.uses + 1u;
    const bool audit = uses >= auditEvery();
    state.uses = audit ? auditStart() : uses;
    return audit ? TrustUse::Audit : TrustUse::Skip;
}

bool audited(TrustUse use) {
    return use == TrustUse::Audit || use == TrustUse::Volatile;
}

void noteUse(TrustSite site, TrustUse use) {
    if (use == TrustUse::Skip) DirectDraws::AddTrust(site, TrustEvent::Trusted);
    else if (use == TrustUse::Audit) DirectDraws::AddTrust(site, TrustEvent::Audited);
    else if (use == TrustUse::Volatile) DirectDraws::AddTrust(site, TrustEvent::Volatile);
}

void noteFailed(TrustSite site, TrustUse use, std::uint32_t before) {
    DirectDraws::AddTrust(site, use == TrustUse::Audit ? TrustEvent::AuditFailed : TrustEvent::VolatileFailed);
    if (before == 0 || !presentBefore(before)) DirectDraws::AddTrust(site, TrustEvent::Demoted);
}

void noteFailed(DirectDraws::TrustState& state, TrustSite site, TrustUse use) {
    if (!audited(use)) return;
    noteFailed(site, use, state.volatileUntil.exchange(heldUntil(), std::memory_order_relaxed));
}

void noteFailed(DirectDraws::LocalTrustState& state, TrustSite site, TrustUse use) {
    if (!audited(use)) return;
    noteFailed(site, use, std::exchange(state.volatileUntil, heldUntil()));
}

bool rectVerifyDue() {
    const auto every = DirectDraws::RectVerifyEvery();
    if (every == 0) return false;
    struct RectVerifyStorage {};
    auto& ordinal = HostThreadLocal<std::uint64_t, RectVerifyStorage>();
    return ++ordinal % every == 0 && !trusted();
}

// Why a direct draw's read ranges were not settled (the three tests of RangesSettled, in its order), counted
// when the pipeline statistics are on: one line per 4096 such draws.
void noteUnsettled(const AddressRanges& ranges, const Graphics::Recorder::WriteRanges* before, std::uint64_t generation) {
    static const bool on = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!on) return;
    static std::atomic<std::uint64_t> queued{0}, recorded{0}, stamped{0}, total{0};
    if (DrawPipeline::Active() && DrawPipeline::Queue0().Busy() && DrawPipeline::Queue0().OverlapsAny(ranges)) queued.fetch_add(1, std::memory_order_relaxed);
    else if (!Graphics::Recorder::ReadsSettled(ranges, before)) recorded.fetch_add(1, std::memory_order_relaxed);
    else stamped.fetch_add(1, std::memory_order_relaxed);
    static_cast<void>(generation);
    if ((total.fetch_add(1, std::memory_order_relaxed) + 1u) % 4096u == 0u) std::fprintf(stderr, "[direct-unsettled] cumulative: %llu over a write queued in the draw pipeline, %llu over a recorder write not settled, %llu with a newer memory stamp; %llu ranges under a newer block stamp settled by their pages\n", static_cast<unsigned long long>(queued.load(std::memory_order_relaxed)), static_cast<unsigned long long>(recorded.load(std::memory_order_relaxed)), static_cast<unsigned long long>(stamped.load(std::memory_order_relaxed)), static_cast<unsigned long long>(PreciseSettles.load(std::memory_order_relaxed)));
}

bool stampsSettled(std::uint64_t begin, std::size_t bytes, std::uint64_t generation) {
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle) && GuestMemory::TrackerGenerationNow() <= generation) return true;
    if (GuestMemory::NewestStamp(begin, bytes) <= generation) return true;
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PreciseSettle) && GuestMemory::UnchangedSince(begin, bytes, generation);
}

std::uint64_t IndirectClassWord(const std::optional<Pm4::DrawParameters::IndirectDraw>& indirect) {
    if (!indirect) return 0;
    using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
    std::uint64_t word = 1;
    word |= static_cast<std::uint64_t>(indirect->opcode & 0xffu) << 1u;
    word |= static_cast<std::uint64_t>(indirect->recordBytes & 0xffu) << 9u;
    word |= static_cast<std::uint64_t>(indirect->countIndirect ? 1u : 0u) << 17u;
    word |= static_cast<std::uint64_t>(indirect->count > 1u ? 1u : 0u) << 18u;
    word |= static_cast<std::uint64_t>(indirect->drawIndexEnabled ? 1u : 0u) << 19u;
    word |= static_cast<std::uint64_t>(indirect->vertexRule == Rule::InPlace ? 1u : 0u) << 20u;
    word |= static_cast<std::uint64_t>(indirect->instanceRule == Rule::InPlace ? 1u : 0u) << 21u;
    word |= static_cast<std::uint64_t>(indirect->baseVertexLocation & 0xfffu) << 22u;
    word |= static_cast<std::uint64_t>(indirect->startInstanceLocation & 0xfffu) << 34u;
    word |= static_cast<std::uint64_t>(indirect->drawIndexLocation & 0xfffu) << 46u;
    return word;
}

std::optional<Counter> classifyDirectIndirect(const LeanDrawJob& job, const LeanAnchor& anchor, DirectStages& resolved) {
    const auto refuse = [](Counter why) {
        DirectDraws::Add(why);
        return std::optional<Counter>(Counter::FallbackShape);
    };
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectGpuIndirect) || resolved.decode == nullptr || resolved.decode->programs.empty()) return refuse(Counter::ShapeIndirect);
    const auto stage = std::ranges::find_if(resolved.stages, [](const DirectResolvedStage& candidate) { return candidate.program == 0; });
    if (stage == resolved.stages.end() || stage->Shared() == nullptr || stage->Shared()->donor == nullptr || !resolved.refreshed) return refuse(Counter::ShapeIndirect);
    auto parameters = job.parameters;
    const auto& program = resolved.decode->programs.front();
    const auto cpu = Driver::ClassifyIndirectDraw(*stage->Shared()->donor, resolved.decode->state, program, anchor.device, parameters, false, &stage->Side());
    NoteIndirectProgramPath(program.binary.codeAddress, cpu.has_value());
    if (cpu) return refuse(Counter::ShapeIndirectCpu);
    resolved.indirect = parameters.indirect;
    return std::nullopt;
}

std::optional<Counter> DirectShapeRefusal(const LeanDrawJob& job) {
    const auto& decode = *job.shape->decode;
    const auto& state = decode.state;
    if (job.parameters.indirect && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectGpuIndirect)) return Counter::ShapeIndirect;
    if (state.quadList) return Counter::ShapeQuadList;
    if (state.rectList) {
        if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectRectLists)) return Counter::ShapeRectList;
        if (job.parameters.indirect) return Counter::ShapeRectIndirect;
        if (decode.programs.size() != 2 || decode.roles.size() != 2 || decode.programs[0].binary.stage != Stage::Vertex || decode.programs[1].binary.stage != Stage::Fragment) return Counter::ShapeRectPrograms;
    }
    if (state.stages.path != Graphics::ShaderPath::Vertex) return Counter::ShapePath;
    if (state.stages.mesh) return Counter::ShapeMesh;
    if (state.stages.tessellation || state.stages.tessellationGeometry) return Counter::ShapeTessellation;
    if (state.tessellationCapture != nullptr) return Counter::ShapeCapture;
    if (!Graphics::DrawsRecordable()) return Counter::ShapeNotRecordable;
    if (decode.programs.empty() || decode.programs.size() > DirectClassKey::MaxStages || decode.roles.size() != decode.programs.size()) return Counter::ShapeStageCount;
    for (std::size_t i = 0; i < decode.programs.size(); ++i) {
        const auto& program = decode.programs[i];
        const auto stage = program.binary.stage;
        if (stage != Stage::Vertex && stage != Stage::Fragment) return Counter::ShapeStageKind;
        if (decode.roles[i] != (stage == Stage::Fragment ? Role::Fragment : Role::Main)) return Counter::ShapeRole;
        if (SkippedShader(program.binary.codeAddress)) return Counter::ShapeSkipped;
    }
    return std::nullopt;
}

struct InvocationSlot {
    std::shared_ptr<const ShaderSnapshot> snapshot;
    std::size_t codeOffset = 0;
    std::shared_ptr<const ShaderRecompiler::SourceHandle> handle;
    bool convertedSamples = false;
    bool layerRowSamples = false;
    std::uint64_t variant = 0;
    DirectDraws::LocalTrustState trust;
};

struct InvocationCache {
    std::array<InvocationSlot, InvocationSlots> slots;
    std::array<std::uint8_t, InvocationSlots / 2> recent{};
    std::vector<std::uint64_t> key;
    std::vector<InvocationSlot> wide;

    InvocationSlot& For(const DrawProgram& program) {
        const auto snapshot = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(program.snapshot.get()));
        if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::InvocationSlotsWide)) {
            if (wide.empty()) wide.resize(WideInvocationSlots);
            return wide[mixedSlot(snapshot, program.codeOffset, WideInvocationSlots)];
        }
        if (!wide.empty()) {
            wide.clear();
            wide.shrink_to_fit();
        }
        if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoInvocationWays)) return slots[mixedSlot(snapshot, program.codeOffset, InvocationSlots)];
        const auto set = mixedSlot(snapshot, program.codeOffset, InvocationSlots / 2u);
        auto* const ways = slots.data() + set * 2u;
        for (std::uint8_t way = 0; way < 2u; ++way) {
            if (ways[way].snapshot.get() != program.snapshot.get() || ways[way].codeOffset != program.codeOffset) continue;
            recent[set] = way;
            return ways[way];
        }
        return ways[recent[set] ^ 1u];
    }
};

struct InvocationCacheStorage {};

bool InvocationCacheable(const DrawProgram& program, const ShaderRecompiler::RecompileRequest& request) {
    const auto stage = request.shader.stage;
    if (stage == Stage::Compute || (ShaderRecompiler::GcnGuestShaders() && stage == Stage::Vertex)) return false;
    return (stage != Stage::Vertex && stage != Stage::Fragment) || !request.shader.header.empty() || !program.snapshot->userDataTailRegister;
}

struct DirectReads {
    std::vector<std::pair<std::uint64_t, std::uint32_t>> words;
    std::uint64_t page = ~std::uint64_t{0};
    bool failed = false;
    std::vector<std::uint32_t> nodes;

    void Reset() {
        words.clear();
        nodes.clear();
        page = ~std::uint64_t{0};
        failed = false;
    }
};

struct DirectReadsStorage {};

bool readDirectWord(void* context, std::uint64_t address, std::uint32_t* value) {
    auto& reads = *static_cast<DirectReads*>(context);
    const auto page = address & ~(WalkPageBytes - 1u);
    const bool crossesPage = (address & (WalkPageBytes - 1u)) > WalkPageBytes - sizeof(*value);
    if (crossesPage || page != reads.page) {
        if (!crossesPage && GuestMemory::Accessible(reinterpret_cast<const void*>(page), WalkPageBytes)) {
            reads.page = page;
        } else if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), sizeof(*value))) {
            reads.failed = true;
            return false;
        }
    }
    std::memcpy(value, reinterpret_cast<const void*>(address), sizeof(*value));
    reads.words.emplace_back(address, *value);
    return true;
}

bool directAccessible(void* context, std::uint64_t address, std::uint64_t bytes) {
    if (GuestMemory::Accessible(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes))) return true;
    static_cast<DirectReads*>(context)->failed = true;
    return false;
}

bool directViewsCurrent(void*, std::span<const ShaderRecompiler::RegisteredImageView> views) {
    return CaptureRegisteredViewsCurrent(views);
}

std::shared_ptr<const ShaderRecompiler::ResourceCapture> walkDirect(const ShaderRecompiler::PreparedShaderInvocation& invocation, DirectReads& reads) {
    const auto& request = invocation.Request();
    ShaderRecompiler::SrtRuntime runtime;
    runtime.userData = request.context.userData;
    runtime.shaderBase = request.shader.codeAddress;
    runtime.userContext = &reads;
    runtime.readMemory = &readDirectWord;
    runtime.readSpecializationMemory = &readDirectWord;
    runtime.accessible = &directAccessible;
    ShaderMemory::SetImageViewHooks(runtime);
    runtime.snapshotCurrent = &directViewsCurrent;
    runtime.readNodes = &reads.nodes;
    return invocation.Capture(runtime);
}

bool readRanges(DirectReads& reads, AddressRanges& ranges, std::vector<std::uint32_t>& values) {
    auto& words = reads.words;
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());
    ranges.clear();
    values.clear();
    for (std::size_t index = 0; index < words.size(); ++index) {
        const auto [address, value] = words[index];
        if (index != 0 && words[index - 1].first == address) return false;
        if (!ranges.empty() && ranges.back().second == address) ranges.back().second += sizeof(std::uint32_t);
        else ranges.emplace_back(address, address + sizeof(std::uint32_t));
        values.push_back(value);
    }
    return true;
}

bool imageTables(const ShaderRecompiler::ResourceSnapshot& snapshot) {
    return !snapshot.tables.tables.empty() || !snapshot.tables.nested.empty() || !snapshot.tables.ranges.empty();
}

struct StageFrontSlot {
    std::uint64_t variant = 0;
    std::uint64_t key = 0;
    std::uint64_t clears = 0;
    std::shared_ptr<const DirectStage> stage;
};

struct StageFront {
    std::array<StageFrontSlot, DirectStageStore::FrontSlots> slots;
};

struct StageFrontStorage {};

struct ClassFrontSlot {
    std::uint64_t probe = 0;
    std::uint64_t clears = 0;
    std::shared_ptr<const DirectClass> entry;
};

struct ClassFront {
    std::array<ClassFrontSlot, DirectClassStore::FrontSlots> slots;
};

struct ClassFrontStorage {};

bool ExactClassKeys() {
    static const bool exact = std::getenv("APS5_DIRECT_DRAWS_EXACT_KEY") != nullptr;
    return exact;
}

struct HitPool {
    static constexpr std::size_t Slots = 256;
    std::vector<std::shared_ptr<DrawRecordHit>> slots;
    std::size_t next = 0;
    std::uint64_t device = 0;
};

struct HitPoolStorage {};

void keepCleared(Graphics::ShaderResources::PreparedBuffers& held) {
    auto moved = std::move(held.moved);
    auto selected = std::move(held.selected);
    auto blob = std::move(held.blob);
    moved.clear();
    selected.clear();
    blob.clear();
    held = {};
    held.moved = std::move(moved);
    held.selected = std::move(selected);
    held.blob = std::move(blob);
}

void keepCleared(Graphics::PreparedInputs& held) {
    auto vertexInput = std::move(held.vertexInput);
    auto vertexHandles = std::move(held.vertexHandles);
    auto vertexOffsets = std::move(held.vertexOffsets);
    auto inPlaceReads = std::move(held.inPlaceReads);
    auto ranges = std::move(held.ranges);
    vertexInput.bindings.clear();
    vertexInput.attributes.clear();
    vertexHandles.clear();
    vertexOffsets.clear();
    inPlaceReads.clear();
    ranges.clear();
    held = {};
    held.vertexInput = std::move(vertexInput);
    held.vertexHandles = std::move(vertexHandles);
    held.vertexOffsets = std::move(vertexOffsets);
    held.inPlaceReads = std::move(inPlaceReads);
    held.ranges = std::move(ranges);
}

void recycleHit(DrawRecordHit& hit) {
    auto sides = std::move(hit.sides);
    auto stages = std::move(hit.stages);
    auto captures = std::move(hit.captures);
    auto buffers = hit.preparedBuffers.has_value() ? std::move(*hit.preparedBuffers) : std::move(hit.spareBuffers);
    auto inputs = hit.preparedInputs.has_value() ? std::move(*hit.preparedInputs) : std::move(hit.spareInputs);
    stages.clear();
    captures.clear();
    keepCleared(buffers);
    keepCleared(inputs);
    hit = {};
    hit.sides = std::move(sides);
    hit.stages = std::move(stages);
    hit.captures = std::move(captures);
    hit.spareBuffers = std::move(buffers);
    hit.spareInputs = std::move(inputs);
    hit.pooled = true;
}

std::shared_ptr<DrawRecordHit> pooledHit(std::uint64_t device) {
    auto& pool = HostThreadLocal<HitPool, HitPoolStorage>();
    if (pool.device != device) {
        pool.slots.clear();
        pool.next = 0;
        pool.device = device;
    }
    if (pool.slots.size() < HitPool::Slots) {
        auto& made = pool.slots.emplace_back(std::make_shared<DrawRecordHit>());
        made->pooled = true;
        return made;
    }
    for (std::size_t tries = 0; tries < 4; ++tries) {
        auto& slot = pool.slots[pool.next];
        pool.next = (pool.next + 1) % HitPool::Slots;
        if (slot.use_count() != 1) continue;
        recycleHit(*slot);
        return slot;
    }
    return std::make_shared<DrawRecordHit>();
}

const Graphics::Recorder::WriteRanges* pendingSince(const DirectStages& resolved, std::shared_ptr<const Graphics::Recorder::WriteRanges>& held) {
    if (!resolved.serialed) {
        held = Graphics::Recorder::PendingWriteSnapshot();
        return held.get();
    }
    if (Graphics::Recorder::PublishGeneration() != resolved.beforeSerial) {
        held = Graphics::Recorder::PendingWriteSnapshot();
        DirectDraws::Add(Counter::SettleLoadsTaken);
        return held.get();
    }
    DirectDraws::Add(Counter::SettleLoadsSkipped);
    if (const auto every = DirectDraws::BufferVerifyEvery(); every != 0) {
        struct OrdinalStorage {};
        auto& ordinal = HostThreadLocal<std::uint64_t, OrdinalStorage>();
        if (++ordinal % every == 0) {
            const auto loaded = Graphics::Recorder::PendingWriteSnapshot();
            if (Graphics::Recorder::PublishGeneration() == resolved.beforeSerial) {
                DirectDraws::Add(Counter::SettleVerifyChecked);
                if (loaded.get() != resolved.before.get()) DirectDraws::Add(Counter::SettleVerifyDiffering);
            }
        }
    }
    return resolved.before.get();
}

bool settledSince(const AddressRanges& ranges, const DirectStages& resolved, const Graphics::Recorder::WriteRanges* also = nullptr) {
    if (resolved.serialed) return RangesSettledSince(ranges, resolved.before.get(), resolved.generation, also, resolved.beforeSerial);
    return RangesSettled(ranges, resolved.before.get(), resolved.generation, also);
}

bool readsSettledSince(const AddressRanges& ranges, const DirectStages& resolved, const Graphics::Recorder::WriteRanges* pending) {
    if (resolved.serialed) return Graphics::Recorder::ReadsSettledSince(ranges, pending, nullptr, resolved.beforeSerial, resolved.before.get());
    return Graphics::Recorder::ReadsSettled(ranges, pending);
}

namespace Validation = Graphics::WriteValidation;

constexpr std::size_t FinisherSlots = 2048;
constexpr std::uint64_t FinisherRangeWindows = 8;
constexpr std::size_t FinisherWindowLimit = 64;
constexpr std::uint8_t FinisherLiveSkip = 64;
constexpr std::uint8_t SlotNone = 0;
constexpr std::uint8_t SlotClear = 1;
constexpr std::uint8_t SlotLive = 2;
constexpr unsigned ArmSessionShift = 48;
constexpr std::uint64_t ArmEpochMask = (std::uint64_t{1} << ArmSessionShift) - 1u;
constexpr std::uint64_t ArmSessionLimit = std::uint64_t{1} << (64u - ArmSessionShift);
constexpr std::size_t ProgramArmSlots = 256;
constexpr std::uint8_t ArmBackoff = 64;

struct FinisherSlot {
    std::uint64_t window = ~std::uint64_t{0};
    std::uint64_t armed = 0;
    std::uint64_t retired = 0;
    std::uint32_t epoch = 0;
    std::uint8_t state = SlotNone;
    std::uint8_t liveSkip = 0;
};

struct FinisherTable {
    std::uint64_t session = 0;
    std::uint64_t switches = 0;
    std::uint32_t epoch = 0;
    std::array<FinisherSlot, FinisherSlots> slots{};
};

struct FinisherTableHandle {
    std::unique_ptr<FinisherTable> table;
};

struct FinisherTableStorage {};

FinisherTable& finisherTable(std::uint64_t session) {
    auto& handle = HostThreadLocal<FinisherTableHandle, FinisherTableStorage>();
    if (handle.table == nullptr) handle.table = std::make_unique<FinisherTable>();
    auto& table = *handle.table;
    const auto switches = Graphics::RuntimeSwitches::Generation();
    if (table.session != session || table.switches != switches) {
        table.session = session;
        table.switches = switches;
        if (++table.epoch == 0) {
            table.slots.fill(FinisherSlot{});
            table.epoch = 1;
        }
    }
    return table;
}

std::size_t finisherSlot(std::uint64_t window) {
    return static_cast<std::size_t>((window * 0x9E3779B97F4A7C15ull) >> 53);
}

struct FinisherWindows {
    std::array<std::uint64_t, FinisherWindowLimit> windows{};
    std::size_t count = 0;
};

struct FinisherWindowsStorage {};

bool collectWindows(const AddressRanges& ranges, FinisherWindows& out) {
    out.count = 0;
    for (const auto& [begin, end] : ranges) {
        if (end <= begin) continue;
        const auto first = begin >> Validation::WindowShift;
        const auto last = (end - 1u) >> Validation::WindowShift;
        if (last - first >= FinisherRangeWindows) return false;
        for (auto window = first;; ++window) {
            const auto* const known = out.windows.data();
            if (std::find(known, known + out.count, window) == known + out.count) {
                if (out.count == out.windows.size()) return false;
                out.windows[out.count++] = window;
            }
            if (window == last) break;
        }
    }
    return out.count != 0;
}

bool windowClocksHeld(std::uint64_t window, std::uint64_t armed, bool clean) {
    if (Validation::Clock(Validation::Class::W, window) > armed) return false;
    if (Validation::Clock(Validation::Class::Q, window) > armed) return false;
    if (Validation::Clock(Validation::Class::L, window) > armed) return false;
    return !clean || Validation::Clock(Validation::Class::C, window) <= armed;
}

bool rangeClocksHeld(std::uint64_t begin, std::uint64_t end, std::uint64_t armed) {
    if (end <= begin) return true;
    const auto last = (end - 1u) >> Validation::WindowShift;
    for (auto window = begin >> Validation::WindowShift;; ++window) {
        if (!windowClocksHeld(window, armed, true)) return false;
        if (window == last) break;
    }
    return true;
}

bool settleProof(std::uint64_t window) {
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> span{{{window << Validation::WindowShift, (window + 1u) << Validation::WindowShift}}};
    if (!DrawFinisher::Get().PendingClearForValidation(span)) return false;
    if (DrawPipeline::Active() && !DrawPipeline::Queue0().QueuedClearForValidation(span)) return false;
    return Graphics::Recorder::ReadsSettled(span, nullptr, nullptr);
}

bool capturedSnapshotsClear(const AddressRanges& ranges, const DirectStages& resolved, const Graphics::Recorder::WriteRanges* also) {
    const auto* before = resolved.before.get();
    for (const auto& [begin, end] : ranges) {
        if (end <= begin) continue;
        const auto bytes = static_cast<std::size_t>(end - begin);
        if (Graphics::Recorder::SnapshotOverlaps(before, begin, bytes) || Graphics::Recorder::SnapshotOverlaps(also, begin, bytes)) return false;
    }
    return true;
}

bool copiedWatchClear(const AddressRanges& ranges, const DrawPipeline::Watch& watch) {
    if (!watch.snapshot) return true;
    const auto count = std::min<std::size_t>(watch.snapshotCount, watch.snapshotWrites.size());
    for (std::size_t at = 0; at < count; ++at) {
        const auto& [first, last] = watch.snapshotWrites[at];
        for (const auto& [begin, end] : ranges) {
            if (begin < last && first < end) return false;
        }
    }
    return true;
}

Validation::Answer finisherAnswer(const AddressRanges& ranges, const DirectStages& resolved, const Graphics::Recorder::WriteRanges* also, const DrawPipeline::Watch* watch, Validation::Test test, bool quiet) {
    const auto session = Validation::Session();
    if (session == 0) return Validation::Answer::Today;
    const auto note = [&](Validation::Outcome outcome) {
        if (!quiet) Validation::Note(test, outcome);
    };
    const auto today = [&](Validation::Outcome outcome) {
        note(outcome);
        return Validation::Answer::Today;
    };
    if (resolved.wvSession != session) return today(Validation::Outcome::Session);
    auto& windows = HostThreadLocal<FinisherWindows, FinisherWindowsStorage>();
    if (!collectWindows(ranges, windows)) return today(Validation::Outcome::Wide);
    if (DrawFinisher::UnknownWritesAhead()) return today(Validation::Outcome::Unknown);
    auto& table = finisherTable(session);
    auto answer = Validation::Answer::Generations;
    for (std::size_t at = 0; at < windows.count; ++at) {
        const auto window = windows.windows[at];
        auto& slot = table.slots[finisherSlot(window)];
        if (slot.window != window || slot.epoch != table.epoch) {
            if (quiet) return Validation::Answer::Today;
            if (slot.epoch == table.epoch && slot.state == SlotClear) note(Validation::Outcome::Evicted);
            slot = FinisherSlot{};
            slot.window = window;
            slot.epoch = table.epoch;
        }
        if (slot.state == SlotClear && !windowClocksHeld(window, slot.armed, false)) {
            if (quiet) return Validation::Answer::Today;
            slot.state = SlotNone;
            note(Validation::Outcome::Moved);
        }
        if (slot.state == SlotLive) {
            if (quiet) return Validation::Answer::Today;
            if (slot.liveSkip != 0) {
                --slot.liveSkip;
                return today(Validation::Outcome::Live);
            }
            slot.state = SlotNone;
        }
        if (slot.state == SlotNone) {
            if (quiet) return Validation::Answer::Today;
            const auto before = Validation::Epoch();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const bool proved = settleProof(window);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (!proved) {
                slot.state = SlotLive;
                slot.liveSkip = FinisherLiveSkip;
                return today(Validation::Outcome::Live);
            }
            slot.state = SlotClear;
            slot.armed = before;
            slot.retired = DrawPipeline::Active() ? DrawPipeline::Queue0().Retired() : 0;
            answer = Validation::Answer::Proved;
        }
        if (watch != nullptr && watch->retired < slot.retired) return today(Validation::Outcome::Watch);
    }
    if (answer == Validation::Answer::Proved) {
        for (std::size_t at = 0; at < windows.count; ++at) {
            const auto window = windows.windows[at];
            const auto& slot = table.slots[finisherSlot(window)];
            if (slot.window != window || slot.epoch != table.epoch || slot.state != SlotClear || !windowClocksHeld(window, slot.armed, false)) return today(Validation::Outcome::Live);
        }
    }
    if (watch != nullptr && DrawPipeline::Active()) {
        const auto retiredNow = DrawPipeline::Queue0().Retired();
        if (retiredNow != watch->retired && retiredNow - watch->retired > watch->items) return today(Validation::Outcome::Watch);
    }
    if (watch != nullptr && !copiedWatchClear(ranges, *watch)) return today(Validation::Outcome::Watch);
    if (!capturedSnapshotsClear(ranges, resolved, also)) return today(Validation::Outcome::Snapshot);
    if (!RangesStamped(ranges, resolved.generation)) return today(Validation::Outcome::Stamps);
    note(answer == Validation::Answer::Generations ? Validation::Outcome::Answered : Validation::Outcome::Armed);
    return answer;
}

template<typename TToday>
bool settledOr(const AddressRanges& ranges, const DirectStages& resolved, const Graphics::Recorder::WriteRanges* also, const DrawPipeline::Watch* watch, Validation::Test test, TToday&& today) {
    const auto answer = finisherAnswer(ranges, resolved, also, watch, test, false);
    if (answer == Validation::Answer::Today) return today();
    if (answer == Validation::Answer::Proved || !Validation::VerifyDue(test)) return true;
    if (today()) {
        Validation::NoteVerify(test, false, nullptr);
        return true;
    }
    const bool differs = finisherAnswer(ranges, resolved, also, watch, test, true) == Validation::Answer::Generations;
    Validation::NoteVerify(test, differs, "today's settle test refused ranges whose windows raised no queued, pending, label or store event since their proof");
    return false;
}

bool settledValidated(const AddressRanges& ranges, const DirectStages& resolved, Validation::Test test) {
    return settledOr(ranges, resolved, nullptr, nullptr, test, [&] { return settledSince(ranges, resolved); });
}

void raiseCleanForget(std::uintptr_t start, std::uintptr_t end) {
    Validation::Raise(Validation::Class::C, Validation::Source::CleanForget, start, end);
}

void ensureCleanForgetHook() {
    static const bool hooked = [] {
        GuestMemory::SetCleanForgetHook(&raiseCleanForget);
        return true;
    }();
    static_cast<void>(hooked);
}

std::uint64_t packArm(std::uint64_t session, std::uint64_t epoch) {
    return (session << ArmSessionShift) | (epoch & ArmEpochMask);
}

bool armable(std::uint64_t session, std::uint64_t epoch) {
    return session != 0 && session < ArmSessionLimit && epoch <= ArmEpochMask;
}

struct ProgramArm {
    std::weak_ptr<const void> owner;
    const void* raw = nullptr;
    std::uint64_t packed = 0;
    std::uint8_t skips = 0;
};

struct ProgramArms {
    std::array<ProgramArm, ProgramArmSlots> slots{};
};

struct ProgramArmsHandle {
    std::unique_ptr<ProgramArms> arms;
};

struct ProgramArmsStorage {};
struct ProgramRangesStorage {};

ProgramArms& programArms() {
    auto& handle = HostThreadLocal<ProgramArmsHandle, ProgramArmsStorage>();
    if (handle.arms == nullptr) handle.arms = std::make_unique<ProgramArms>();
    return *handle.arms;
}

std::size_t programArmSlot(const void* raw) {
    return static_cast<std::size_t>((static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(raw)) * 0x9E3779B97F4A7C15ull) >> 56);
}

template<typename TPointer>
bool armOwns(const ProgramArm& arm, const TPointer& owner) {
    return owner != nullptr && arm.raw == static_cast<const void*>(owner.get()) && !arm.owner.owner_before(owner) && !owner.owner_before(arm.owner);
}

const DrawProgram* programOfRegion(const DirectStages& resolved, const ShaderRecompiler::MemoryRegion& region) {
    for (const auto& program : resolved.decode->programs) {
        if (program.snapshot == nullptr) continue;
        for (const auto& held : program.memory) {
            if (!held.bytes.empty() && held.guestAddress == region.guestAddress && held.bytes.data() == region.bytes.data() && held.bytes.size() == region.bytes.size()) return &program;
        }
    }
    return nullptr;
}

bool programArmsHeld(const DirectStages& resolved, std::span<const ShaderRecompiler::MemoryRegion> regions, std::size_t prefix, std::uint64_t session) {
    auto& arms = programArms();
    for (std::size_t at = 0; at < prefix; ++at) {
        const auto& region = regions[at];
        const auto* program = programOfRegion(resolved, region);
        if (program == nullptr) return false;
        const auto& arm = arms.slots[programArmSlot(program->snapshot.get())];
        if (!armOwns(arm, program->snapshot) || arm.packed == 0 || (arm.packed >> ArmSessionShift) != session) return false;
        if (!rangeClocksHeld(region.guestAddress, region.guestAddress + region.bytes.size(), arm.packed & ArmEpochMask)) return false;
    }
    return true;
}

std::size_t programMemoryHeld(const DirectStages& resolved, std::span<const ShaderRecompiler::MemoryRegion> regions, std::size_t compared, bool& verify) {
    constexpr auto test = Validation::Test::CapturedMemory;
    verify = false;
    const auto session = Validation::Session();
    if (session == 0) return 0;
    const auto prefix = resolved.memory.size();
    if (prefix == 0 || prefix > compared || prefix > regions.size() || resolved.decode == nullptr) return 0;
    if (resolved.wvSession != session) {
        Validation::Note(test, Validation::Outcome::Session);
        return 0;
    }
    if (DrawFinisher::UnknownWritesAhead()) {
        Validation::Note(test, Validation::Outcome::Unknown);
        return 0;
    }
    if (!programArmsHeld(resolved, regions, prefix, session)) return 0;
    Validation::Note(test, Validation::Outcome::Answered);
    verify = Validation::VerifyDue(test);
    return prefix;
}

void armProgramMemory(const DirectStages& resolved) {
    constexpr auto test = Validation::Test::CapturedMemory;
    const auto session = resolved.wvSession;
    if (resolved.memory.empty() || resolved.decode == nullptr || session == 0 || session != Validation::Session()) return;
    auto& arms = programArms();
    auto& ranges = HostThreadLocal<AddressRanges, ProgramRangesStorage>();
    auto& windows = HostThreadLocal<FinisherWindows, FinisherWindowsStorage>();
    for (const auto& program : resolved.decode->programs) {
        if (program.snapshot == nullptr) continue;
        ranges.clear();
        for (const auto& region : program.memory) {
            if (!region.bytes.empty()) ranges.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
        }
        if (ranges.empty()) continue;
        auto& arm = arms.slots[programArmSlot(program.snapshot.get())];
        const bool owned = armOwns(arm, program.snapshot);
        if (owned && arm.packed != 0 && (arm.packed >> ArmSessionShift) == session) {
            bool held = true;
            for (const auto& [begin, end] : ranges) held = held && rangeClocksHeld(begin, end, arm.packed & ArmEpochMask);
            if (held) continue;
        }
        if (owned && arm.skips != 0) {
            --arm.skips;
            continue;
        }
        arm.owner = program.snapshot;
        arm.raw = program.snapshot.get();
        arm.packed = 0;
        arm.skips = ArmBackoff;
        if (!collectWindows(ranges, windows)) {
            Validation::Note(test, Validation::Outcome::Wide);
            continue;
        }
        ensureCleanForgetHook();
        const auto before = Validation::Epoch();
        if (!armable(session, before)) continue;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (!GuestMemory::AllClean(ranges)) {
            Validation::Note(test, Validation::Outcome::NotClean);
            continue;
        }
        bool proved = true;
        for (std::size_t at = 0; proved && at < windows.count; ++at) proved = settleProof(windows.windows[at]);
        if (!proved) {
            Validation::Note(test, Validation::Outcome::Live);
            continue;
        }
        bool equal = true;
        for (const auto& region : program.memory) {
            if (!region.bytes.empty() && GuestMemory::CompareMapped(region.guestAddress, region.bytes) != GuestMemory::Compare::Equal) equal = false;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (!equal) continue;
        arm.packed = packArm(session, before);
        arm.skips = 0;
        Validation::Note(test, Validation::Outcome::Armed);
    }
}

Counter BufferRefusal(Graphics::ShaderResources::PreparedBuffers::Refusal refusal) {
    using Refusal = Graphics::ShaderResources::PreparedBuffers::Refusal;
    switch (refusal) {
    case Refusal::Inaccessible: return Counter::BuffersInaccessible;
    case Refusal::UploadLimit: return Counter::BuffersUploadLimit;
    case Refusal::Budget: return Counter::BuffersBudget;
    case Refusal::NotGuest: return Counter::BuffersNotGuest;
    case Refusal::Written: return Counter::BuffersWritten;
    case Refusal::Oversize: return Counter::BuffersOversize;
    default: return Counter::BuffersClassification;
    }
}

bool stepSplits() {
    static const bool on = std::getenv("APS5_STEP_SPLITS") != nullptr;
    return on;
}

bool repeatCensus() {
    static const bool on = std::getenv("APS5_BUFFER_REPEAT_CENSUS") != nullptr;
    return on;
}

constexpr std::uint64_t SettleMergeGap = 4096;

const AddressRanges& mergedRanges(const AddressRanges& ranges) {
    struct MergedStorage {};
    auto& merged = HostThreadLocal<AddressRanges, MergedStorage>();
    merged.assign(ranges.begin(), ranges.end());
    for (std::size_t at = 1; at < merged.size(); ++at) {
        const auto value = merged[at];
        std::size_t to = at;
        for (; to > 0 && merged[to - 1].first > value.first; --to) merged[to] = merged[to - 1];
        merged[to] = value;
    }
    std::size_t kept = 0;
    for (std::size_t at = 1; at < merged.size(); ++at) {
        if (merged[at].first <= merged[kept].second + SettleMergeGap) merged[kept].second = std::max(merged[kept].second, merged[at].second);
        else merged[++kept] = merged[at];
    }
    if (!merged.empty()) merged.resize(kept + 1);
    return merged;
}

struct RepeatCensus {
    static constexpr std::uint64_t Every = 256;
    static constexpr std::uint32_t Window = 16;
    static constexpr std::size_t Entries = 64;
    static constexpr std::size_t ArenaBytes = 65536;
    struct Entry {
        std::uint64_t begin = 0;
        std::size_t bytes = 0;
        std::size_t offset = 0;
    };
    std::uint64_t hits = 0;
    std::uint32_t window = 0;
    std::array<Entry, Entries> entries{};
    std::size_t used = 0;
    std::vector<std::byte> arena;
    std::size_t arenaUsed = 0;
};

void noteRepeatCensus(const Graphics::ShaderResources::PreparedBuffers& prepared) {
    struct CensusStorage {};
    auto& census = HostThreadLocal<RepeatCensus, CensusStorage>();
    const auto ordinal = census.hits++;
    if (census.window == 0) {
        if (ordinal % RepeatCensus::Every != 0) return;
        census.window = RepeatCensus::Window;
        census.used = 0;
        census.arenaUsed = 0;
        if (census.arena.size() != RepeatCensus::ArenaBytes) census.arena.resize(RepeatCensus::ArenaBytes);
    }
    --census.window;
    std::uint64_t ranges = 0;
    std::uint64_t bytes = 0;
    std::uint64_t repeatRanges = 0;
    std::uint64_t repeatBytes = 0;
    for (const auto& item : prepared.selected) {
        if (item.inPlace || item.bytes == 0 || item.offset > prepared.blob.size() || item.bytes > prepared.blob.size() - item.offset) continue;
        const auto* source = prepared.blob.data() + item.offset;
        ++ranges;
        bytes += item.bytes;
        if (item.begin == 0) continue;
        auto* const first = census.entries.data();
        auto* const last = first + census.used;
        auto* const entry = std::find_if(first, last, [&](const RepeatCensus::Entry& candidate) { return candidate.begin == item.begin && candidate.bytes == item.bytes; });
        if (entry != last) {
            if (std::memcmp(census.arena.data() + entry->offset, source, item.bytes) == 0) {
                ++repeatRanges;
                repeatBytes += item.bytes;
            } else {
                std::memcpy(census.arena.data() + entry->offset, source, item.bytes);
            }
            continue;
        }
        if (census.used == RepeatCensus::Entries || item.bytes > census.arena.size() - census.arenaUsed) continue;
        census.entries[census.used++] = {item.begin, item.bytes, census.arenaUsed};
        std::memcpy(census.arena.data() + census.arenaUsed, source, item.bytes);
        census.arenaUsed += item.bytes;
    }
    DirectDraws::Add(Counter::CensusHits);
    DirectDraws::Add(Counter::CensusRanges, ranges);
    DirectDraws::Add(Counter::CensusBytes, bytes);
    DirectDraws::Add(Counter::CensusRepeatRanges, repeatRanges);
    DirectDraws::Add(Counter::CensusRepeatBytes, repeatBytes);
    if (ranges != 0 && repeatRanges == ranges) DirectDraws::Add(Counter::CensusHitsAllRepeat);
}

// The hit's buffers from its record's buffer plan, taken against the colour target the commit
// will draw with (commitRecordDraw hands the committer the job's decode state). The guest bytes
// are read here, ahead of the commit, so they follow the walk's settled rule: a write queued or
// pending before this draw may land before the commit, and once it retired no committer test
// sees it. The rule also covers the read itself: a write pending or queued when it began that
// retired by the end.
void prepareBuffers(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectStages& resolved, DrawRecordHit& hit, DirectStepClock* splits, DirectDraws::TrustState& classTrust, TrustUse classUse) {
    auto plan = hit.record->commit->bufferPlan.load(std::memory_order_acquire);
    if (plan == nullptr) {
        DirectDraws::Add(Counter::BuffersNoPlan);
        return;
    }
    auto& prepared = hit.pooled ? hit.preparedBuffers.emplace(std::move(hit.spareBuffers)) : hit.preparedBuffers.emplace();
    prepared.generation = resolved.generation;
    const bool settleTrusted = trusted() || classUse == TrustUse::Skip;
    bool settleNoted = false;
    const auto settleFailed = [&] {
        if (!settleNoted) noteFailed(classTrust, TrustSite::BuffersSettle, classUse);
        settleNoted = true;
    };
    std::shared_ptr<const Graphics::Recorder::WriteRanges> pendingHeld;
    const auto* const pending = settleTrusted ? nullptr : pendingSince(resolved, pendingHeld);
    std::optional<DrawPipeline::Watch> queued;
    if (!settleTrusted && DrawPipeline::Active()) queued = DrawPipeline::Queue0().WatchWrites();
    bool done = false;
    try {
        done = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectBuffersInPlace) ? anchor.device->PrepareDirectBuffers(*plan, hit.stages, &job.shape->decode->state.color, prepared) : Graphics::ShaderResources::PrepareDirectBuffers(*plan, hit.stages, &job.shape->decode->state.color, prepared);
    } catch (const std::exception&) {
        prepared.refusal = Graphics::ShaderResources::PreparedBuffers::Refusal::Classification;
    }
    if (splits != nullptr) splits->Split(Step::BuffersPrepare);
    if (!done) {
        DirectDraws::Add(BufferRefusal(prepared.refusal));
        hit.preparedBuffers.reset();
        return;
    }
    noteUse(TrustSite::BuffersSettle, classUse);
    const bool deferred = (prepared.shortcuts & Graphics::ShaderResources::PreparedBuffers::DeferredImports) != 0;
    if (prepared.lookupsDeferred != 0) DirectDraws::Add(Counter::LookupsDeferred, prepared.lookupsDeferred);
    if (prepared.lookupsForced != 0) DirectDraws::Add(Counter::LookupsForced, prepared.lookupsForced);
    const auto todayUnsettled = [&](const AddressRanges& ranges) { return !settledSince(ranges, resolved, Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle) ? pending : nullptr) || (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle) && !readsSettledSince(ranges, resolved, pending)) || (queued.has_value() && DrawPipeline::Queue0().WrittenSince(*queued, ranges)); };
    const auto unsettled = [&](const AddressRanges& ranges) { return !settledOr(ranges, resolved, pending, queued.has_value() ? &*queued : nullptr, Validation::Test::BuffersSettle, [&] { return !todayUnsettled(ranges); }); };
    struct ReadStorage {};
    auto& read = HostThreadLocal<AddressRanges, ReadStorage>();
    read.clear();
    bool candidates = false;
    for (const auto& item : prepared.selected) {
        if (item.begin == 0 || item.inPlace) continue;
        read.emplace_back(item.begin, item.begin + item.bytes);
        candidates = candidates || item.handle != VK_NULL_HANDLE || (deferred && item.moved);
    }
    bool check = !settleTrusted && !read.empty();
    bool mergedRefused = false;
    if (check && read.size() > 2 && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::MergedSettle)) {
        struct RecheckStorage {};
        auto& recheckOrdinal = HostThreadLocal<std::uint64_t, RecheckStorage>();
        const auto recheckEvery = DirectDraws::BufferVerifyEvery();
        const bool recheck = recheckEvery != 0 && ++recheckOrdinal % recheckEvery == 0;
        const bool separateRefused = recheck && unsettled(read);
        const auto& merged = mergedRanges(read);
        DirectDraws::Add(Counter::SettleRangesBefore, read.size());
        DirectDraws::Add(Counter::SettleRangesAfter, merged.size());
        if (!unsettled(merged)) {
            DirectDraws::Add(Counter::MergedSettlesPassed);
            if (recheck) DirectDraws::Add(Counter::MergedSettlesRechecked);
            if (separateRefused) {
                DirectDraws::Add(Counter::MergedSettlesRecheckRefused);
            } else {
                check = false;
                prepared.shortcuts |= Graphics::ShaderResources::PreparedBuffers::MergedSettle;
            }
        } else {
            mergedRefused = true;
            DirectDraws::Add(Counter::MergedSettlesRefused);
        }
    }
    if (check && unsettled(read)) {
        settleFailed();
        bool held = candidates;
        for (std::size_t index = 0; held && index < prepared.selected.size(); ++index) {
            const auto& item = prepared.selected[index];
            if (item.begin == 0 || item.inPlace) continue;
            struct RangeStorage {};
            auto& single = HostThreadLocal<AddressRanges, RangeStorage>();
            single.assign(1, {item.begin, item.begin + item.bytes});
            if (!unsettled(single)) continue;
            if (deferred && item.moved && item.handle == VK_NULL_HANDLE) {
                DirectDraws::Add(Counter::LookupsAtSettle);
                bool found = false;
                try {
                    found = anchor.device->FindPreparedImport(*plan, prepared, index);
                } catch (const std::exception&) {
                    found = false;
                }
                if (found) DirectDraws::Add(Counter::LookupsAtSettleFound);
            }
            held = Graphics::ShaderResources::TakePreparedInPlace(prepared, index);
        }
        if (!held) {
            DirectDraws::Add(Counter::BuffersUnsettled);
            hit.preparedBuffers.reset();
            return;
        }
    } else if (mergedRefused) {
        DirectDraws::Add(Counter::MergedSettlesRecovered);
    }
    if (prepared.inPlace) {
        std::uint64_t ranges = 0;
        for (const auto& item : prepared.selected) {
            if (!item.inPlace) continue;
            if (!settleTrusted && !stampsSettled(item.begin, item.bytes, resolved.generation)) {
                settleFailed();
                DirectDraws::Add(Counter::BuffersUnsettled);
                hit.preparedBuffers.reset();
                return;
            }
            ++ranges;
        }
        DirectDraws::Add(Counter::BuffersInPlace);
        DirectDraws::Add(Counter::BuffersInPlaceRanges, ranges);
    }
    if (splits != nullptr) splits->Split(Step::BuffersSettle);
    prepared.plan = std::move(plan);
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::AliasesAtPrepare)) prepared.aliasesClear = Graphics::DrawBufferAliasesClear(hit.stages, job.shape->decode->state.color, hit.parameters.indexAddress, static_cast<std::uint64_t>(hit.parameters.indexCount) * hit.parameters.indexSize);
    DirectDraws::Add(Counter::BuffersPrepared);
    DirectDraws::Add(Counter::BuffersBlobBytes, prepared.blob.size());
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::BufferBlobAppend)) DirectDraws::Add(Counter::BlobAppends);
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitBindings)) return;
    bool placed = false;
    try {
        placed = anchor.device->PlaceDirectBuffers(prepared);
    } catch (const std::exception&) {
        prepared.placed.reset();
    }
    DirectDraws::Add(placed ? Counter::BindingsPlaced : Counter::BindingsNotPlaced);
    if (splits != nullptr) splits->Split(Step::BuffersPlace);
    if (placed && repeatCensus()) noteRepeatCensus(prepared);
}

void feedZoneCard(Graphics::ShaderResources::PreparedBuffers& prepared) {
    if (!Graphics::CommitFeedOn(Graphics::CommitFeedCards)) {
        prepared.card.valid = false;
        return;
    }
    using Build = Graphics::ShaderResources::ZoneCardBuild;
    const auto built = Graphics::ShaderResources::BuildZoneCard(prepared);
    DirectDraws::Add(built == Build::Made ? Counter::FeedCardsMade : built == Build::OverCapacity ? Counter::FeedCardsOverCapacity : Counter::FeedCardsRefused);
}

void noteGuardAnswer(const Graphics::GuardAnswerNote& note, bool inputs) {
    using Made = Graphics::GuardAnswerMade;
    switch (note.made) {
    case Made::Off:
        return;
    case Made::Made:
        if (inputs) {
            DirectDraws::Add(Counter::FollowAheadInputsMade);
            if (note.clear != 0) DirectDraws::Add(Counter::FollowAheadInputsClear);
            DirectDraws::Add(Counter::FollowAheadInputRanges, note.depthRanges);
            return;
        }
        DirectDraws::Add(Counter::FollowAheadBuffersMade);
        if ((note.clear & Graphics::GuardAnswer::DepthClear) != 0) DirectDraws::Add(Counter::FollowAheadBuffersDepthClear);
        if ((note.clear & Graphics::GuardAnswer::MultisampleClear) != 0) DirectDraws::Add(Counter::FollowAheadBuffersMultisampleClear);
        DirectDraws::Add(Counter::FollowAheadGuardEntries, note.depthRanges);
        DirectDraws::Add(Counter::FollowAheadMovedEntries, note.multisampleRanges);
        return;
    case Made::SpansOff:
        DirectDraws::Add(Counter::FollowAheadSpansOff);
        return;
    case Made::ProofPending:
        DirectDraws::Add(Counter::FollowAheadProofPending);
        return;
    case Made::StampMoved:
        DirectDraws::Add(Counter::FollowAheadStampMoved);
        return;
    case Made::BoardStale:
        DirectDraws::Add(Counter::FollowAheadBoardStale);
        return;
    case Made::GenerationMoved:
        DirectDraws::Add(Counter::FollowAheadGenerationMoved);
        return;
    }
}

Counter InputRefusal(Graphics::PreparedInputs::Refusal refusal) {
    using Refusal = Graphics::PreparedInputs::Refusal;
    switch (refusal) {
    case Refusal::NoRecipe: return Counter::InputsNoRecipe;
    case Refusal::NotInPlace: return Counter::InputsNotInPlace;
    case Refusal::Inaccessible: return Counter::InputsInaccessible;
    case Refusal::Thrown: return Counter::InputsThrown;
    default: return Counter::InputsShape;
    }
}

void prepareInputs(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectStages& resolved, DrawRecordHit& hit, DirectDraws::TrustState& classTrust, TrustUse classUse) {
    auto recipe = hit.record->commit->inputsRecipe.load(std::memory_order_acquire);
    if (recipe == nullptr) {
        DirectDraws::Add(Counter::InputsNoRecipe);
        return;
    }
    auto& prepared = hit.pooled ? hit.preparedInputs.emplace(std::move(hit.spareInputs)) : hit.preparedInputs.emplace();
    const bool settleTrusted = trusted() || classUse == TrustUse::Skip;
    std::shared_ptr<const Graphics::Recorder::WriteRanges> pendingHeld;
    const auto* const pending = settleTrusted ? nullptr : pendingSince(resolved, pendingHeld);
    std::optional<DrawPipeline::Watch> queued;
    if (!settleTrusted && DrawPipeline::Active()) queued = DrawPipeline::Queue0().WatchWrites();
    bool done = false;
    try {
        done = anchor.device->PrepareDrawInputs(job.shape->decode->state, hit.parameters, hit.stages, std::move(recipe), prepared);
    } catch (const std::exception&) {
        prepared.refusal = Graphics::PreparedInputs::Refusal::Thrown;
    }
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::InputsMemo)) {
        const auto note = Graphics::TakeInputsMemoNote();
        if (note.outcome == Graphics::InputsMemoOutcome::Hit) DirectDraws::Add(Counter::InputsMemoHits);
        else if (note.outcome == Graphics::InputsMemoOutcome::MissKey) DirectDraws::Add(Counter::InputsMemoMissKey);
        else if (note.outcome == Graphics::InputsMemoOutcome::MissSerial) DirectDraws::Add(Counter::InputsMemoMissSerial);
        else if (note.outcome == Graphics::InputsMemoOutcome::Indirect) DirectDraws::Add(Counter::InputsMemoIndirect);
        if (note.stored) DirectDraws::Add(Counter::InputsMemoStored);
        if (note.verified) DirectDraws::Add(Counter::InputsMemoVerifyChecked);
        if (note.differed) DirectDraws::Add(Counter::InputsMemoVerifyDiffering);
    }
    if (!done) {
        DirectDraws::Add(InputRefusal(prepared.refusal));
        hit.preparedInputs.reset();
        return;
    }
    struct ReadStorage {};
    auto& read = HostThreadLocal<AddressRanges, ReadStorage>();
    read.clear();
    if (hit.parameters.indexed) read.emplace_back(prepared.ranges.front().begin, prepared.ranges.front().end);
    if (!read.empty()) noteUse(TrustSite::IndexSettle, classUse);
    const auto indexUnsettled = [&] { return !settledSince(read, resolved, Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle) ? pending : nullptr) || (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle) && !readsSettledSince(read, resolved, pending)) || (queued.has_value() && DrawPipeline::Queue0().WrittenSince(*queued, read)); };
    if (!settleTrusted && !read.empty() && !settledOr(read, resolved, pending, queued.has_value() ? &*queued : nullptr, Validation::Test::IndexSettle, [&] { return !indexUnsettled(); })) {
        noteFailed(classTrust, TrustSite::IndexSettle, classUse);
        DirectDraws::Add(Counter::InputsUnsettled);
        hit.preparedInputs.reset();
        return;
    }
    DirectDraws::Add(Counter::InputsPrepared);
}

void recordFragment(const LeanDrawJob& job, const LeanAnchor& anchor, DrawRecordHit& hit) {
    if (!hit.preparedBuffers.has_value() || hit.preparedBuffers->placed == nullptr || !hit.preparedInputs.has_value()) return;
    const auto block = hit.record->commit->fragmentBlock.load(std::memory_order_acquire);
    if (block == nullptr) {
        DirectDraws::Add(Counter::FragmentsNoBlock);
        return;
    }
    struct SampleStorage {};
    auto& sampled = HostThreadLocal<std::uint64_t, SampleStorage>();
    const bool timed = (sampled++ & 7u) == 0;
    const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& fragment = hit.fragment.emplace();
    auto refusal = DrawFragmentRefusal::Thrown;
    bool made = false;
    try {
        made = anchor.device->RecordDrawFragment(job.shape->decode->state, hit.parameters, hit.stages, *block, *hit.preparedBuffers, *hit.preparedInputs, fragment, refusal);
    } catch (const std::exception&) {
        refusal = DrawFragmentRefusal::Thrown;
        made = false;
    }
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FragmentPlans)) {
        const auto note = Graphics::ShaderResources::TakeFragmentPlanNote();
        if (note.built != 0) DirectDraws::Add(Counter::FragmentSetPlansBuilt, note.built);
        if (note.hit != 0) DirectDraws::Add(Counter::FragmentSetPlansHit, note.hit);
        if (note.pushed != 0) DirectDraws::Add(Counter::FragmentPushPlans, note.pushed);
        if (note.checked != 0) DirectDraws::Add(Counter::FragmentPlanVerifyChecked, note.checked);
        if (note.differing != 0) DirectDraws::Add(Counter::FragmentPlanVerifyDiffering, note.differing);
    }
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FragmentParkedSets)) {
        const auto note = Graphics::ShaderResources::TakeParkedSetsNote();
        if (note.taken != 0) DirectDraws::Add(Counter::ParkedTakes, note.taken);
        if (note.fresh != 0) DirectDraws::Add(Counter::ParkedFreshSets, note.fresh);
        if (note.today != 0) DirectDraws::Add(Counter::ParkedTodayWrites, note.today);
        if (note.parkedBuffers != 0) DirectDraws::Add(Counter::ParkedBufferDescriptors, note.parkedBuffers);
        if (note.parkedOthers != 0) DirectDraws::Add(Counter::ParkedOtherDescriptors, note.parkedOthers);
        if (note.freshBuffers != 0) DirectDraws::Add(Counter::FreshBufferDescriptors, note.freshBuffers);
        if (note.freshOthers != 0) DirectDraws::Add(Counter::FreshOtherDescriptors, note.freshOthers);
        if (note.checked != 0) DirectDraws::Add(Counter::ParkedVerifyChecked, note.checked);
        if (note.differing != 0) DirectDraws::Add(Counter::ParkedVerifyDiffering, note.differing);
    }
    if (!made) {
        hit.fragment.reset();
        const auto reason = refusal == DrawFragmentRefusal::None || refusal == DrawFragmentRefusal::Count ? DrawFragmentRefusal::Thrown : refusal;
        DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::FragmentsNoBlock) + static_cast<std::size_t>(reason) - static_cast<std::size_t>(DrawFragmentRefusal::NoBlock)));
        return;
    }
    DirectDraws::Add(Counter::FragmentsMade);
    if (fragment.written) DirectDraws::Add(Counter::FragmentsWritten);
    if (!timed) return;
    DirectDraws::Add(Counter::FragmentsFinisherNanoseconds, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count()));
    DirectDraws::Add(Counter::FragmentsFinisherTimed);
}

struct ReplayKey {
    const DrawDecode* decode = nullptr;
    std::uint64_t shape = 0;
    std::uint64_t deviceSerial = 0;
    std::uint32_t indexSize = 0;
    std::uint32_t flags = 0;
    bool indexed = false;

    static ReplayKey For(const LeanDrawJob& job) {
        ReplayKey key;
        key.decode = job.shape->decode.get();
        key.shape = job.shapeKey;
        key.deviceSerial = job.deviceSerial;
        key.indexSize = job.parameters.indexSize;
        key.flags = job.parameters.flags;
        key.indexed = job.parameters.indexed;
        return key;
    }

    bool operator==(const ReplayKey&) const = default;
};

struct ReplayKept {
    std::shared_ptr<const DrawRecord> record;
    std::shared_ptr<const void> owner;
    std::shared_ptr<const DrawDecode> decode;
    std::vector<std::shared_ptr<const ShaderRecompiler::ResourceCapture>> captures;
    std::vector<ShaderRecompiler::HeapStageWords> sides;
};

struct ReplaySource {
    ReplayKey key;
    bool held = false;
    const VulkanDevice* device = nullptr;
    std::uint64_t clears = 0;
    std::shared_ptr<const ReplayKept> kept;
    std::vector<Graphics::CompiledShader> stages;
    Pm4::DrawParameters parameters{};
    std::vector<DrawPipeline::Range> reads;
    Graphics::ShaderResources::PreparedBuffers buffers;
    Graphics::PreparedInputs inputs;
};

struct ReplayStore {
    static constexpr std::size_t Slots = 1024;
    std::vector<ReplaySource> slots;
    bool filled = false;
    bool holding = false;
    std::uint64_t deviceSerial = 0;
    std::uint64_t tag = 0;
    std::uint64_t generation = 0;
    std::uint64_t timing = 0;

    ReplaySource& Slot(const ReplayKey& key) {
        if (slots.empty()) slots.resize(Slots);
        const auto first = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(key.decode)) ^ key.shape;
        const auto second = (static_cast<std::uint64_t>(key.indexSize) << 33u) ^ (static_cast<std::uint64_t>(key.flags) << 1u) ^ (key.indexed ? 1u : 0u);
        return slots[mixedSlot(first, second, Slots)];
    }
};

struct ReplayStoreStorage {};

std::atomic<std::uint32_t> replayHolders{0};

void clearReplayStore(ReplayStore& store) {
    if (store.filled) DirectDraws::Add(Counter::ReplayCleared);
    store.slots.clear();
    store.filled = false;
}

void syncReplayStore(ReplayStore& store, std::uint64_t deviceSerial, std::uint64_t tag) {
    const auto generation = Graphics::RuntimeSwitches::Generation();
    if (store.deviceSerial == deviceSerial && store.tag == tag && store.generation == generation) return;
    clearReplayStore(store);
    store.deviceSerial = deviceSerial;
    store.tag = tag;
    store.generation = generation;
}

void dropReplayStore() {
    auto& store = HostThreadLocal<ReplayStore, ReplayStoreStorage>();
    if (!store.holding) return;
    clearReplayStore(store);
    store.generation = 0;
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PooledHits)) HostThreadLocal<HitPool, HitPoolStorage>().slots.clear();
    store.holding = false;
    replayHolders.fetch_sub(1, std::memory_order_relaxed);
}

bool replayClassHeld(const DirectClassStore& classes, const std::shared_ptr<const void>& owner) {
    const auto* entry = static_cast<const DirectClass*>(owner.get());
    return entry != nullptr && entry->record != nullptr && classes.Find(entry->key, entry->probe).get() == entry;
}

bool wordsWithin(std::span<const std::uint32_t> words, const std::vector<std::uint32_t>& held) {
    const std::less_equal<const std::uint32_t*> notAfter{};
    return !held.empty() && notAfter(held.data(), words.data()) && notAfter(words.data() + words.size(), held.data() + held.size());
}

void keepReplaySource(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectStages& resolved, const DrawRecordHit& hit) {
    auto& store = HostThreadLocal<ReplayStore, ReplayStoreStorage>();
    syncReplayStore(store, job.deviceSerial, resolved.tag);
    const auto key = ReplayKey::For(job);
    auto& source = store.Slot(key);
    const auto refuse = [&](Counter why) {
        DirectDraws::Add(why);
        if (!(source.key == key)) return;
        source.held = false;
        source.kept.reset();
    };
    if (resolved.rect != nullptr || hit.addressed || hit.parameters.indirect || job.parameters.indirect) return refuse(Counter::ReplaySourceShape);
    if (hit.record == nullptr || !hit.fragment.has_value() || !hit.preparedBuffers.has_value() || !hit.preparedInputs.has_value()) return refuse(Counter::ReplaySourceNoFragment);
    if (hit.fragment->resources == nullptr || hit.fragment->resources->ChecksFaults()) return refuse(Counter::ReplaySourceFaults);
    const auto& classes = DirectStores::Get().classes;
    const auto clears = classes.Clears();
    if (!replayClassHeld(classes, hit.owner) || static_cast<const DirectClass*>(hit.owner.get())->record != hit.record) return refuse(Counter::ReplaySourceClass);
    source.held = false;
    source.kept.reset();
    auto kept = std::make_shared<ReplayKept>();
    kept->record = hit.record;
    kept->owner = hit.owner;
    kept->decode = job.shape->decode;
    kept->captures = hit.captures;
    kept->sides = hit.sides;
    source.stages.assign(hit.stages.begin(), hit.stages.end());
    for (auto& stage : source.stages) {
        for (std::size_t index = 0; index < hit.sides.size(); ++index) {
            if (stage.heap != &hit.sides[index]) continue;
            stage.heap = &kept->sides[index];
            break;
        }
    }
    source.buffers = *hit.preparedBuffers;
    source.buffers.placed.reset();
    source.buffers.placedDevice = VK_NULL_HANDLE;
    for (auto& entry : source.buffers.moved) {
        if (entry.words.empty()) continue;
        bool rewired = false;
        for (std::size_t side = 0; side < hit.sides.size() && !rewired; ++side) {
            const auto& bindings = hit.sides[side].bindings;
            for (std::size_t binding = 0; binding < bindings.size(); ++binding) {
                if (!wordsWithin(entry.words, bindings[binding])) continue;
                entry.words = std::span<const std::uint32_t>(kept->sides[side].bindings[binding].data() + (entry.words.data() - bindings[binding].data()), entry.words.size());
                rewired = true;
                break;
            }
        }
    }
    source.inputs = *hit.preparedInputs;
    source.parameters = hit.parameters;
    source.reads.assign(hit.reads.begin(), hit.reads.end());
    source.key = key;
    source.device = anchor.device.get();
    source.clears = clears;
    source.kept = std::move(kept);
    source.held = true;
    store.filled = true;
    if (!store.holding) {
        store.holding = true;
        replayHolders.fetch_add(1, std::memory_order_relaxed);
    }
    DirectDraws::Add(Counter::ReplaySourcesKept);
}

}

std::size_t DirectStageStore::KeyHash::operator()(const std::pair<std::uint64_t, std::uint64_t>& key) const {
    return static_cast<std::size_t>(key.first * 0x9e3779b97f4a7c15ull ^ key.second);
}

std::shared_ptr<const DirectStage> DirectStageStore::Find(std::uint64_t variant, std::uint64_t key) const {
    auto& slot = HostThreadLocal<StageFront, StageFrontStorage>().slots[mixedSlot(variant, key, FrontSlots)];
    if (slot.stage != nullptr && slot.variant == variant && slot.key == key && slot.clears == clears.load(std::memory_order_acquire)) return slot.stage;
    std::lock_guard lock(mutex);
    const auto found = stages.find({variant, key});
    if (found == stages.end()) return nullptr;
    slot = {variant, key, clears.load(std::memory_order_relaxed), found->second};
    return found->second;
}

std::shared_ptr<const DirectStage> DirectStageStore::Install(std::shared_ptr<const DirectStage> stage, std::uint64_t tag, bool& installed) {
    installed = false;
    bool overflowed = false;
    {
        std::lock_guard lock(mutex);
        if (!DirectStores::Get().Current(tag)) return stage;
        const std::pair key{stage->variant, stage->key};
        if (const auto found = stages.find(key); found != stages.end()) return found->second;
        if (stages.size() >= Bound) {
            DirectDraws::Add(Counter::StagesEvicted, stages.size());
            stages.clear();
            clears.fetch_add(1, std::memory_order_release);
            overflowed = true;
        }
        stages.emplace(key, stage);
        installed = true;
        DirectDraws::NoteStages(stages.size());
    }
    // A class key names its stages by pointer, so no class can hit once its stages are gone.
    if (overflowed) DirectStores::Get().classes.Evict();
    return stage;
}

void DirectStageStore::Clear() {
    std::lock_guard lock(mutex);
    stages.clear();
    clears.fetch_add(1, std::memory_order_release);
    DirectDraws::NoteStages(0);
}

std::size_t DirectStageStore::Size() const {
    std::lock_guard lock(mutex);
    return stages.size();
}

DirectClassKey DirectClassKey::For(const LeanDrawJob& job, const DirectStages& resolved) {
    const auto& parameters = job.parameters;
    DirectClassKey key;
    key.shape = job.shapeKey;
    if (resolved.layoutHashed) {
        key.layout = resolved.layoutHash;
    } else {
        const auto layout = HashDrawRecordWords(resolved.vertexLayout, {});
        key.layout = layout[0] ^ (layout[1] * 0x9e3779b97f4a7c15ull);
    }
    key.indexSize = parameters.indexSize;
    key.flags = parameters.flags;
    key.indexed = parameters.indexed;
    key.count = static_cast<std::uint32_t>(resolved.stages.size());
    for (std::size_t i = 0; i < resolved.stages.size(); ++i) key.stages[i] = resolved.stages[i].Shared().get();
    key.indirect = IndirectClassWord(resolved.indirect);
    key.generated = resolved.rectControl;
    if (!ExactClassKeys()) return key;
    key.indexAddress = parameters.indexAddress;
    key.indexCount = parameters.indexCount;
    key.instanceCount = parameters.instanceCount;
    key.firstVertex = parameters.firstVertex;
    key.firstInstance = parameters.firstInstance;
    return key;
}

std::uint64_t DirectClassKey::Probe() const {
    std::uint64_t probe = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) { probe = (probe ^ value) * 0x100000001b3ull; };
    mix(shape);
    mix(layout);
    mix(indexAddress);
    mix((static_cast<std::uint64_t>(indexCount) << 32u) | indexSize);
    mix((static_cast<std::uint64_t>(instanceCount) << 32u) | flags);
    mix((static_cast<std::uint64_t>(firstVertex) << 32u) | firstInstance);
    mix((static_cast<std::uint64_t>(count) << 1u) | (indexed ? 1u : 0u));
    for (std::uint32_t i = 0; i < count && i < MaxStages; ++i) mix(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(stages[i])));
    if (indirect != 0) mix(indirect);
    if (generated != nullptr) mix(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(generated)));
    return probe ^ (probe >> 29u);
}

std::size_t DirectClass::Bytes() const {
    return sizeof(DirectClass) + (record != nullptr ? record->Bytes() : 0);
}

namespace {

bool spreadClasses() {
    return Graphics::SpreadDestruction::Queueing();
}

template<typename Map>
void releaseClasses(Map& classes, std::size_t bytes) noexcept {
    if (classes.empty()) return;
    std::vector<std::shared_ptr<const void>> owners;
    try {
        owners.reserve(classes.size() + 1);
        for (auto& entry : classes) owners.push_back(std::move(entry.second));
        owners.push_back(std::make_shared<Map>(std::move(classes)));
    } catch (...) {
        return;
    }
    Graphics::SpreadDestruction::Release(owners, bytes);
}

}

std::shared_ptr<const DirectClass> DirectClassStore::Find(const DirectClassKey& key, std::uint64_t probe) const {
    auto& slot = HostThreadLocal<ClassFront, ClassFrontStorage>().slots[static_cast<std::size_t>(probe % FrontSlots)];
    if (slot.entry != nullptr && slot.probe == probe && slot.clears == clears.load(std::memory_order_acquire) && slot.entry->key == key) return slot.entry;
    std::lock_guard lock(mutex);
    const auto found = classes.find(key);
    if (found == classes.end()) return nullptr;
    slot = {probe, clears.load(std::memory_order_relaxed), found->second};
    return found->second;
}

std::shared_ptr<const DirectClass> DirectClassStore::Install(std::shared_ptr<const DirectClass> entry, std::uint64_t tag, bool& installed) {
    installed = false;
    decltype(classes) evicted;
    std::size_t evictedBytes = 0;
    {
        std::lock_guard lock(mutex);
        if (!DirectStores::Get().Current(tag)) return entry;
        if (const auto found = classes.find(entry->key); found != classes.end()) return found->second;
        if (classes.size() >= Bound) {
            DirectDraws::Add(Counter::ClassesEvicted, kept);
            if (spreadClasses()) {
                classes.swap(evicted);
                evictedBytes = bytes;
            } else {
                classes.clear();
            }
            kept = 0;
            bytes = 0;
            clears.fetch_add(1, std::memory_order_release);
        }
        if (entry->record != nullptr) ++kept;
        bytes += entry->Bytes();
        classes.emplace(entry->key, entry);
        installed = true;
        note();
    }
    releaseClasses(evicted, evictedBytes);
    return entry;
}

bool DirectClassStore::Refuse(const DirectClass& found, Counter reason) {
    auto refused = std::make_shared<DirectClass>();
    refused->key = found.key;
    refused->probe = found.probe;
    refused->stages = found.stages;
    refused->generated = found.generated;
    refused->refusal = reason;
    std::lock_guard lock(mutex);
    const auto held = classes.find(found.key);
    if (held == classes.end() || held->second.get() != &found) return false;
    if (found.record != nullptr) --kept;
    bytes = bytes - found.Bytes() + refused->Bytes();
    held->second = std::move(refused);
    clears.fetch_add(1, std::memory_order_release);
    note();
    return true;
}

void DirectClassStore::Evict() {
    decltype(classes) evicted;
    std::size_t evictedBytes = 0;
    {
        std::lock_guard lock(mutex);
        DirectDraws::Add(Counter::ClassesEvicted, kept);
        if (spreadClasses()) {
            classes.swap(evicted);
            evictedBytes = bytes;
        } else {
            classes.clear();
        }
        kept = 0;
        bytes = 0;
        clears.fetch_add(1, std::memory_order_release);
        note();
    }
    releaseClasses(evicted, evictedBytes);
}

void DirectClassStore::Clear() {
    decltype(classes) evicted;
    std::size_t evictedBytes = 0;
    {
        std::lock_guard lock(mutex);
        if (spreadClasses()) {
            classes.swap(evicted);
            evictedBytes = bytes;
        } else {
            classes.clear();
        }
        kept = 0;
        bytes = 0;
        clears.fetch_add(1, std::memory_order_release);
        note();
    }
    releaseClasses(evicted, evictedBytes);
}

std::size_t DirectClassStore::Size() const {
    std::lock_guard lock(mutex);
    return classes.size();
}

void DirectClassStore::note() const {
    DirectDraws::NoteClasses(kept, bytes);
}

void RefuseDirectClass(const std::shared_ptr<const void>& owner) {
    if (owner == nullptr) return;
    const auto& found = *static_cast<const DirectClass*>(owner.get());
    DirectDraws::Add(Counter::CommitDemoted);
    if (DirectStores::Get().classes.Refuse(found, Counter::RefusedVerify)) DirectDraws::Add(Counter::RefusedVerify);
}

DirectStores& DirectStores::Get() {
    static auto* stores = new DirectStores;
    return *stores;
}

std::uint64_t DirectStores::Follow(std::uint64_t serial) {
    struct Seen {
        std::uint64_t generation = 0;
        std::uint64_t deviceSerial = 0;
        std::uint64_t epoch = 0;
    };
    struct SeenStorage {};
    auto& seen = HostThreadLocal<Seen, SeenStorage>();
    const auto generation = Graphics::RuntimeSwitches::Generation();
    const auto current = epoch.load(std::memory_order_acquire);
    if (seen.epoch == current && seen.generation == generation && seen.deviceSerial == serial) return current;
    std::lock_guard lock(mutex);
    auto live = Graphics::RuntimeSwitches::Snapshot();
    // Switches that change only what the committer does with a hit, or only dispatch packets,
    // leave the stages and classes as they are, so an in-run A/B of one keeps the stores.
    for (const auto id : {Graphics::RuntimeSwitch::DirectDraws, Graphics::RuntimeSwitch::DirectCleanProofs, Graphics::RuntimeSwitch::DirectPassTargets, Graphics::RuntimeSwitch::CleanCollects, Graphics::RuntimeSwitch::DirectBuffers, Graphics::RuntimeSwitch::PipelineDispatch, Graphics::RuntimeSwitch::PipelineDispatchMisses, Graphics::RuntimeSwitch::XpCommitFloor, Graphics::RuntimeSwitch::MsaaRecords, Graphics::RuntimeSwitch::XpInputsAsIs, Graphics::RuntimeSwitch::XpStampsByEpoch, Graphics::RuntimeSwitch::MsaaRecordHold, Graphics::RuntimeSwitch::TemplateTouch, Graphics::RuntimeSwitch::DeferredEncode, Graphics::RuntimeSwitch::XpProofAsIs, Graphics::RuntimeSwitch::DirectInputs, Graphics::RuntimeSwitch::DirectRepeat, Graphics::RuntimeSwitch::CleanResidentViews, Graphics::RuntimeSwitch::CleanPendingRanges, Graphics::RuntimeSwitch::RefusedCollectMemo, Graphics::RuntimeSwitch::DirectRetry, Graphics::RuntimeSwitch::ItemsByTurn, Graphics::RuntimeSwitch::XpNoCommits, Graphics::RuntimeSwitch::FourthFinisher, Graphics::RuntimeSwitch::FifthFinisher, Graphics::RuntimeSwitch::DirectRepeatSets, Graphics::RuntimeSwitch::DirectRepeatStores, Graphics::RuntimeSwitch::DirectRepeatShared, Graphics::RuntimeSwitch::WorkerCore, Graphics::RuntimeSwitch::LeanListBatch, Graphics::RuntimeSwitch::LeanListDirect, Graphics::RuntimeSwitch::FinisherDeposit, Graphics::RuntimeSwitch::CommitBindings, Graphics::RuntimeSwitch::CommitBindingsStamps, Graphics::RuntimeSwitch::CommitBindingsSets, Graphics::RuntimeSwitch::CommitBindingsDynamic, Graphics::RuntimeSwitch::CommitFragments, Graphics::RuntimeSwitch::ExpressCommits, Graphics::RuntimeSwitch::XpStreamSegments, Graphics::RuntimeSwitch::XpNoEagerStore, Graphics::RuntimeSwitch::LabelOrderedStores, Graphics::RuntimeSwitch::DeeperRing, Graphics::RuntimeSwitch::BesideSampleDumps, Graphics::RuntimeSwitch::DispatchBeside, Graphics::RuntimeSwitch::FaultFragments, Graphics::RuntimeSwitch::StageMemo, Graphics::RuntimeSwitch::KeepFaultSlots, Graphics::RuntimeSwitch::SiblingFinishers, Graphics::RuntimeSwitch::FastSettle, Graphics::RuntimeSwitch::XpSlowFinisher, Graphics::RuntimeSwitch::XpSlowCommitter, Graphics::RuntimeSwitch::XpSlowWorker, Graphics::RuntimeSwitch::XpExpressFloor, Graphics::RuntimeSwitch::LeanExpress, Graphics::RuntimeSwitch::XpAppendCommits, Graphics::RuntimeSwitch::PreciseSettle, Graphics::RuntimeSwitch::AppendCommits, Graphics::RuntimeSwitch::StageMemoPatchVertex, Graphics::RuntimeSwitch::StageMemoPatchBuffers, Graphics::RuntimeSwitch::StageMemoPatchData, Graphics::RuntimeSwitch::RunFollow, Graphics::RuntimeSwitch::TurnWaitsByJob, Graphics::RuntimeSwitch::ExpressKeepState, Graphics::RuntimeSwitch::StageMemoFront, Graphics::RuntimeSwitch::StageMemoAdmitThird, Graphics::RuntimeSwitch::CommitStamps, Graphics::RuntimeSwitch::PassReadsOnce, Graphics::RuntimeSwitch::ReapThrottle, Graphics::RuntimeSwitch::OcclusionWindows, Graphics::RuntimeSwitch::ExpressPassStamps, Graphics::RuntimeSwitch::PassBarriersMerged, Graphics::RuntimeSwitch::LabelReadbackMemo, Graphics::RuntimeSwitch::ExpressTargetUploads, Graphics::RuntimeSwitch::DrawRuns, Graphics::RuntimeSwitch::LargeBatches, Graphics::RuntimeSwitch::ReplayStateDedupe, Graphics::RuntimeSwitch::ReplayStateAcrossBinds, Graphics::RuntimeSwitch::RunFollowNoCarry, Graphics::RuntimeSwitch::RunFollowPlainLoop, Graphics::RuntimeSwitch::FinisherParkSlots, Graphics::RuntimeSwitch::ListHooksOnce, Graphics::RuntimeSwitch::DispatchEnvOnce, Graphics::RuntimeSwitch::PlainRegisterPackets, Graphics::RuntimeSwitch::UserWordBlocks, Graphics::RuntimeSwitch::XpDeviceUploads, Graphics::RuntimeSwitch::XpDepthLayouts, Graphics::RuntimeSwitch::XpTargetRestarts, Graphics::RuntimeSwitch::TargetRestarts, Graphics::RuntimeSwitch::MovedInPlace, Graphics::RuntimeSwitch::IndirectInPass, Graphics::RuntimeSwitch::FragmentOwnSets, Graphics::RuntimeSwitch::DirectBuffersInPlace, Graphics::RuntimeSwitch::DirectGpuIndirect, Graphics::RuntimeSwitch::ExpressGpuIndirect, Graphics::RuntimeSwitch::RegisterRuns, Graphics::RuntimeSwitch::ListRegisterPackets, Graphics::RuntimeSwitch::PacketClassTable, Graphics::RuntimeSwitch::RangeMemo, Graphics::RuntimeSwitch::PendingIndexKept, Graphics::RuntimeSwitch::FillImageIndex, Graphics::RuntimeSwitch::WatchedReaps, Graphics::RuntimeSwitch::DeferredBatchStamps, Graphics::RuntimeSwitch::StageViews, Graphics::RuntimeSwitch::StageMemoLeanStores, Graphics::RuntimeSwitch::SettleSerial, Graphics::RuntimeSwitch::FragmentPlans, Graphics::RuntimeSwitch::BufferBlobAppend, Graphics::RuntimeSwitch::ShapePrologue, Graphics::RuntimeSwitch::DirectHandoff, Graphics::RuntimeSwitch::InputsMemo, Graphics::RuntimeSwitch::WorkerHelps, Graphics::RuntimeSwitch::DispatchMovedWrites, Graphics::RuntimeSwitch::DispatchRelocate, Graphics::RuntimeSwitch::DispatchMovedRecipes, Graphics::RuntimeSwitch::StageMemoPatchMerged, Graphics::RuntimeSwitch::StageMemoChained, Graphics::RuntimeSwitch::StageMemoAdmitWays, Graphics::RuntimeSwitch::StageMemoInvocationWays, Graphics::RuntimeSwitch::ImportSnapshots, Graphics::RuntimeSwitch::FollowWindows, Graphics::RuntimeSwitch::KeepBindingsMoved, Graphics::RuntimeSwitch::FollowWider, Graphics::RuntimeSwitch::ExpressDepthIntent, Graphics::RuntimeSwitch::LazyInPlaceImports, Graphics::RuntimeSwitch::PreparedCapacity, Graphics::RuntimeSwitch::MergedSettle, Graphics::RuntimeSwitch::IdleHitRelease, Graphics::RuntimeSwitch::FragmentParkedSets, Graphics::RuntimeSwitch::UnlockedStamps, Graphics::RuntimeSwitch::WriteSnapshotFronts, Graphics::RuntimeSwitch::PendingFronts, Graphics::RuntimeSwitch::InvocationSlotsWide, Graphics::RuntimeSwitch::XpTrustFinisher, Graphics::RuntimeSwitch::XpTrustCommitter, Graphics::RuntimeSwitch::TemplateKeeps, Graphics::RuntimeSwitch::DepthCheckLean, Graphics::RuntimeSwitch::DispatchRelocateTrials, Graphics::RuntimeSwitch::DirectRectLists, Graphics::RuntimeSwitch::DirectAddressed, Graphics::RuntimeSwitch::TrustAudit, Graphics::RuntimeSwitch::ProofKeeps, Graphics::RuntimeSwitch::ViewTemplateKeeps, Graphics::RuntimeSwitch::LeanFallbackDraws, Graphics::RuntimeSwitch::LabelRuns, Graphics::RuntimeSwitch::DispatchCommitLean, Graphics::RuntimeSwitch::RecordHeavyDraws, Graphics::RuntimeSwitch::InPlaceExact, Graphics::RuntimeSwitch::XpJobReplay, Graphics::RuntimeSwitch::XpFlow, Graphics::RuntimeSwitch::DeepFlow, Graphics::RuntimeSwitch::TailsAhead, Graphics::RuntimeSwitch::FixupsByTurn, Graphics::RuntimeSwitch::FrameFlow, Graphics::RuntimeSwitch::XpNoResolveWaits, Graphics::RuntimeSwitch::WriteValidation, Graphics::RuntimeSwitch::OrderedSyncs, Graphics::RuntimeSwitch::DeepShallow, Graphics::RuntimeSwitch::HtileCarryFills, Graphics::RuntimeSwitch::HostDeepCache, Graphics::RuntimeSwitch::CommitRing, Graphics::RuntimeSwitch::PublicationFromFills, Graphics::RuntimeSwitch::CommitFeed, Graphics::RuntimeSwitch::DeepRuns, Graphics::RuntimeSwitch::FollowAhead, Graphics::RuntimeSwitch::SpreadDestruction, Graphics::RuntimeSwitch::StorageCopyRelease, Graphics::RuntimeSwitch::SkipPendingPipelines, Graphics::RuntimeSwitch::ImportAhead}) live.Assign(id, false);
    if (!tagged || deviceSerial != serial || !(live == switches)) {
        tagged = true;
        deviceSerial = serial;
        switches = live;
        epoch.fetch_add(1, std::memory_order_acq_rel);
        stages.Clear();
        classes.Clear();
    }
    seen = {generation, serial, epoch.load(std::memory_order_relaxed)};
    return seen.epoch;
}

void DirectStepClock::Lap(Step step) {
    const auto now = std::chrono::steady_clock::now();
    const auto index = static_cast<std::size_t>(step);
    nanoseconds[index] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - mark).count());
    reached[index] = true;
    mark = now;
    splitMark = now;
}

void DirectStepClock::Split(Step step) {
    const auto now = std::chrono::steady_clock::now();
    const auto index = static_cast<std::size_t>(step);
    nanoseconds[index] += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - splitMark).count());
    reached[index] = true;
    splitMark = now;
}

void DirectStepClock::Report() const {
    for (std::size_t index = 0; index < nanoseconds.size(); ++index) {
        if (reached[index]) DirectDraws::AddStep(static_cast<Step>(index), nanoseconds[index]);
    }
}

namespace {

bool resolveCensus() {
    static const bool on = std::getenv("APS5_RESOLVE_CENSUS") != nullptr;
    return on;
}

struct ResolveCensusShard {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::uint64_t> seen;
};

void noteResolveCensus(const LeanDrawJob& job, const DirectStages& resolved) {
    constexpr std::uint64_t Prime = 1099511628211ull;
    const auto mix = [](std::uint64_t& hash, std::uint64_t value) { hash = (hash ^ value) * Prime; };
    auto key = LeanExactKey(*job.shape, job.words);
    const auto& parameters = job.parameters;
    for (const std::uint64_t value : {parameters.indexAddress, std::uint64_t{parameters.indexCount}, std::uint64_t{parameters.indexSize}, std::uint64_t{parameters.instanceCount}, std::uint64_t{parameters.flags}, std::uint64_t{parameters.indexed}, std::uint64_t{parameters.firstVertex}, std::uint64_t{parameters.firstInstance}}) mix(key, value);
    std::uint64_t values = 1469598103934665603ull;
    std::uint64_t content = job.shape->finalShape;
    for (const std::uint64_t value : {std::uint64_t{parameters.indexCount}, std::uint64_t{parameters.indexSize}, std::uint64_t{parameters.instanceCount}, std::uint64_t{parameters.flags}, std::uint64_t{parameters.indexed}, std::uint64_t{parameters.firstVertex}, std::uint64_t{parameters.firstInstance}}) mix(content, value);
    for (std::size_t index = resolved.memory.size(); index < resolved.regions.size(); ++index) {
        const auto& region = resolved.regions[index];
        mix(values, region.guestAddress);
        mix(content, region.bytes.size());
        const auto bytes = region.bytes;
        std::size_t at = 0;
        for (; at + sizeof(std::uint64_t) <= bytes.size(); at += sizeof(std::uint64_t)) {
            std::uint64_t word = 0;
            std::memcpy(&word, bytes.data() + at, sizeof(word));
            mix(values, word);
            mix(content, word);
        }
        for (; at < bytes.size(); ++at) {
            mix(values, std::to_integer<std::uint64_t>(bytes[at]));
            mix(content, std::to_integer<std::uint64_t>(bytes[at]));
        }
    }
    static std::array<ResolveCensusShard, 16> shards;
    auto& shard = shards[static_cast<std::size_t>((key >> 7u) & 15u)];
    auto outcome = Counter::ResolveCensusNew;
    {
        std::lock_guard lock(shard.mutex);
        if (shard.seen.size() > 262144u) shard.seen.clear();
        const auto [found, inserted] = shard.seen.try_emplace(key, values);
        if (!inserted) {
            outcome = found->second == values ? Counter::ResolveCensusSame : Counter::ResolveCensusChanged;
            found->second = values;
        }
    }
    DirectDraws::Add(outcome);
    static std::array<ResolveCensusShard, 16> contents;
    auto& contentShard = contents[static_cast<std::size_t>((content >> 7u) & 15u)];
    bool seen = false;
    {
        std::lock_guard lock(contentShard.mutex);
        if (contentShard.seen.size() > 262144u) contentShard.seen.clear();
        seen = !contentShard.seen.try_emplace(content, 1u).second;
    }
    DirectDraws::Add(seen ? Counter::ResolveCensusContentSeen : Counter::ResolveCensusContentNew);
}

void noteStageCensus(const ShaderRecompiler::PreparedShaderInvocation& invocation, const ShaderRecompiler::ResourceCapture& capture, const DirectReads& reads, std::span<const Graphics::DecodeRead> decodeReads) {
    constexpr std::uint64_t Prime = 1099511628211ull;
    const auto mix = [](std::uint64_t& hash, std::uint64_t value) { hash = (hash ^ value) * Prime; };
    const auto& request = invocation.Request();
    const auto userData = request.context.userData;
    const auto& use = capture.userDataUse;
    std::uint64_t content = invocation.VariantId() * 0x9e3779b97f4a7c15ull + request.layout.pushConstantOffsetBytes;
    for (const auto& read : decodeReads) {
        mix(content, read.size);
        for (std::uint32_t at = 0; at < read.size; ++at) mix(content, std::to_integer<std::uint64_t>(read.data[at]));
    }
    bool matched = true;
    for (const auto& [address, origin] : use.reads) {
        const auto found = std::lower_bound(reads.words.begin(), reads.words.end(), std::pair<std::uint64_t, std::uint32_t>{address, 0u});
        if (found == reads.words.end() || found->first != address) {
            matched = false;
            break;
        }
        const bool based = origin >= 0 && static_cast<std::size_t>(origin) < userData.size();
        mix(content, static_cast<std::uint64_t>(static_cast<std::int64_t>(origin)));
        mix(content, based ? address - (static_cast<std::uint64_t>(userData[static_cast<std::size_t>(origin)]) & ~std::uint64_t{3}) : address);
        mix(content, found->second);
    }
    DirectDraws::Add(Counter::ResolveCensusStages);
    std::vector<std::uint64_t> traced;
    traced.reserve(use.reads.size());
    for (const auto& read : use.reads) traced.push_back(read.first);
    std::sort(traced.begin(), traced.end());
    traced.erase(std::unique(traced.begin(), traced.end()), traced.end());
    if (!matched || traced.size() != reads.words.size()) {
        DirectDraws::Add(Counter::ResolveCensusStageUntraced);
        return;
    }
    auto rule = content;
    std::uint32_t valueWords = 0;
    for (std::size_t index = 0; index < userData.size(); ++index) {
        const bool tracked = index < ShaderRecompiler::SrtUserDataUse::TrackedWords;
        const bool value = !tracked || ((use.asValue >> index) & 1u) != 0;
        if (!value) continue;
        ++valueWords;
        mix(rule, index);
        mix(rule, userData[index]);
    }
    static std::array<ResolveCensusShard, 16> rules;
    static std::array<ResolveCensusShard, 16> contents;
    const auto seenIn = [](std::array<ResolveCensusShard, 16>& shards, std::uint64_t key) {
        auto& shard = shards[static_cast<std::size_t>((key >> 7u) & 15u)];
        std::lock_guard lock(shard.mutex);
        if (shard.seen.size() > 262144u) shard.seen.clear();
        return !shard.seen.try_emplace(key, 1u).second;
    };
    const bool ruleSeen = seenIn(rules, rule);
    const bool contentSeen = seenIn(contents, content);
    if (ruleSeen) DirectDraws::Add(Counter::ResolveCensusStageRuleSeen);
    if (contentSeen) DirectDraws::Add(Counter::ResolveCensusStageContentSeen);
    if (contentSeen && !ruleSeen) {
        static std::atomic<std::uint32_t> printed{0};
        if (printed.fetch_add(1, std::memory_order_relaxed) < 24) std::fprintf(stderr, "[resolve-census] stage content seen, memo rule new: stage %u, %zu user words, %u compared as values, value mask %016llx, base mask %016llx, %zu reads, %zu decode reads\n", static_cast<unsigned>(request.shader.stage), userData.size(), valueWords, static_cast<unsigned long long>(use.asValue), static_cast<unsigned long long>(use.asBase), use.reads.size(), decodeReads.size());
    }
}

bool stageMemoOn() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemo);
}

std::uint64_t stageMemoVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_STAGE_MEMO");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

struct StageMemoRoot {
    std::uint64_t offset = 0;
    std::uint16_t origin = 0;
    std::uint8_t aligned = 0;
    std::uint8_t words = 1;
    std::uint8_t chained = 0;

    bool operator==(const StageMemoRoot& other) const = default;
};

struct StageMemoWord {
    std::uint32_t key = 0xffffffffu;
    std::uint8_t kind = 0;
    std::uint8_t merged = 0;

    bool operator==(const StageMemoWord& other) const = default;
};

struct StageMemoPatch {
    std::uint32_t word = 0;
    std::uint32_t binding = 0;
    std::uint32_t position = 0;
    std::uint32_t mask = 0;
};

constexpr std::uint8_t StageMemoVertexKind = 1;
constexpr std::uint8_t StageMemoBaseKind = 5;
constexpr std::uint8_t StageMemoMergedKind = 6;
constexpr std::uint32_t StageMemoSideKinds = (1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::BufferAddress)) | (1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::BufferSize)) | (1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::FlatWord));
constexpr std::uint32_t StageMemoWalkKinds = StageMemoSideKinds | (1u << StageMemoBaseKind);
constexpr std::uint32_t StageMemoGuardedKinds = (1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::BufferAddress)) | (1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::BufferSize)) | (1u << StageMemoBaseKind);
constexpr std::uint32_t StageMemoArmBit = 1u << 31u;

bool stageMemoGuarded(std::uint8_t kind) {
    return kind < 32u && ((StageMemoGuardedKinds >> kind) & 1u) != 0u;
}

std::uint8_t stageMemoWordKind(std::uint32_t kinds) {
    const auto real = kinds & StageMemoSideKinds;
    if (std::popcount(real) > 1) return static_cast<std::uint8_t>(ShaderRecompiler::SideReadKind::BufferAddress);
    if (real == 0u) return StageMemoBaseKind;
    const auto kind = static_cast<std::uint8_t>(std::countr_zero(real));
    return (kinds & (1u << StageMemoBaseKind)) != 0u && !stageMemoGuarded(kind) ? StageMemoBaseKind : kind;
}

std::uint32_t stageMemoPatchKinds() {
    using Graphics::RuntimeSwitch;
    using Graphics::RuntimeSwitches::Set;
    return (Set(RuntimeSwitch::StageMemoPatchVertex) ? 1u << StageMemoVertexKind : 0u) | (Set(RuntimeSwitch::StageMemoPatchBuffers) ? 1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::BufferAddress) : 0u) |
           (Set(RuntimeSwitch::StageMemoPatchData) ? (1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::BufferSize)) | (1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::FlatWord)) : 0u) |
           (Set(RuntimeSwitch::StageMemoChained) ? 1u << StageMemoBaseKind : 0u) | (Set(RuntimeSwitch::StageMemoPatchMerged) ? 1u << StageMemoMergedKind : 0u);
}

std::uint32_t stageMemoArm(std::uint32_t enabledKinds) {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoAdmitWays) ? enabledKinds | StageMemoArmBit : enabledKinds & ((1u << StageMemoBaseKind) | (1u << StageMemoMergedKind));
}

std::size_t stageMemoWayLimit() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoPatchMerged) || Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoChained) ? 8u : 4u;
}

std::uint32_t stageMemoForeignKinds(std::uint32_t enabledKinds) {
    return ((1u << StageMemoBaseKind) | (1u << StageMemoMergedKind)) & ~enabledKinds;
}

struct StageMemoPattern {
    std::uint64_t relocatable = 0;
    std::uint64_t values = 0;
    std::uint64_t bases = 0;
    std::uint64_t addresses = 0;
    std::uint32_t userWords = 0;
    std::uint32_t decodeRoots = 0;
    std::uint32_t patchKinds = 0;
    std::uint32_t arm = 0;
    std::vector<StageMemoWord> words;
    std::vector<StageMemoRoot> roots;

    bool operator==(const StageMemoPattern& other) const = default;
};

struct NestedArm {
    std::atomic<std::uint64_t> validated{0};
    std::atomic<std::uint64_t> forget{0};
    std::atomic<std::uint32_t> skips{0};

    NestedArm() = default;
    NestedArm(const NestedArm&) noexcept {}
    NestedArm& operator=(const NestedArm&) noexcept { return *this; }
};

struct StageMemoEntry {
    std::shared_ptr<const DrawDecode> decode;
    std::uint32_t stage = 0;
    std::uint32_t way = 0;
    std::uint32_t cursor = 0;
    std::uint32_t pushOffset = 0;
    std::uint32_t decodeWords = 0;
    std::uint64_t tag = 0;
    std::shared_ptr<const DirectStage> previous;
    std::uint64_t indexAddress = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t indexSize = 0;
    std::uint32_t instanceCount = 0;
    std::uint64_t variant = 0;
    bool convertedSamples = false;
    bool layerRowSamples = false;
    std::vector<std::uint32_t> userData;
    std::vector<std::uint32_t> rootValues;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> nested;
    std::vector<std::uint32_t> order;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;
    std::shared_ptr<const DirectStage> shared;
    ShaderRecompiler::HeapStageWords side;
    std::shared_ptr<const ShaderRecompiler::SideForwards> forwards;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertexInfo;
    std::vector<Graphics::DecodeRead> decodeReads;
    std::vector<StageMemoPatch> patches;
    bool viewable = false;
    bool attributesKept = false;
    std::vector<std::uint64_t> addresses;
    AddressRanges ranges;
    std::vector<std::uint32_t> values;
    AddressRanges decodeRanges;
    AddressRanges settled;
    std::vector<DrawPipeline::Range> bufferReads;
    std::vector<DrawPipeline::Range> attributeReads;
    std::vector<std::uint32_t> layoutWords;
    std::vector<std::uint32_t> vertexInputs;
    std::uint64_t layoutHash = 0;
    mutable DirectDraws::TrustState trust;
    mutable NestedArm nestedArm;
};

void viewStageEntry(StageMemoEntry& entry, const StageMemoPattern& pattern, std::span<const std::uint32_t> userData, ShaderRecompiler::ShaderStage stage, std::size_t count) {
    entry.addresses.clear();
    for (const auto& root : pattern.roots) {
        if (root.chained != 0) {
            entry.addresses.push_back((static_cast<std::uint64_t>(entry.rootValues[root.origin]) & ~std::uint64_t{3}) + root.offset);
            continue;
        }
        const auto word = userData[root.origin];
        entry.addresses.push_back((root.aligned != 0 ? static_cast<std::uint64_t>(word) & ~std::uint64_t{3} : static_cast<std::uint64_t>(word)) + root.offset);
    }
    const auto walkRoots = pattern.roots.size() - pattern.decodeRoots;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> sorted;
    sorted.reserve(entry.order.size());
    bool increasing = true;
    for (const auto slotIndex : entry.order) {
        const bool isRoot = slotIndex < walkRoots;
        const auto address = isRoot ? entry.addresses[pattern.decodeRoots + slotIndex] : entry.nested[slotIndex - walkRoots].first;
        const auto value = isRoot ? entry.rootValues[entry.decodeWords + slotIndex] : entry.nested[slotIndex - walkRoots].second;
        if (!sorted.empty() && address <= sorted.back().first) increasing = false;
        sorted.emplace_back(address, value);
    }
    if (!increasing) return;
    entry.ranges.clear();
    entry.values.clear();
    for (const auto& [address, value] : sorted) {
        if (!entry.ranges.empty() && entry.ranges.back().second == address) entry.ranges.back().second += sizeof(std::uint32_t);
        else entry.ranges.emplace_back(address, address + sizeof(std::uint32_t));
        entry.values.push_back(value);
    }
    entry.decodeRanges.clear();
    for (std::size_t read = 0; read < entry.decodeReads.size(); ++read) entry.decodeRanges.emplace_back(entry.addresses[read], entry.addresses[read] + entry.decodeReads[read].size);
    entry.settled.assign(entry.ranges.begin(), entry.ranges.end());
    entry.settled.insert(entry.settled.end(), entry.decodeRanges.begin(), entry.decodeRanges.end());
    const Graphics::CompiledShader own{stage, entry.shared->donor.get(), entry.pushOffset, &entry.capture->snapshot.registeredImages, &entry.side};
    const std::array<Graphics::CompiledShader, 2> buffersOnly{Graphics::CompiledShader{stage, nullptr, 0}, own};
    entry.bufferReads = QueuedDrawReadRanges(Pm4::DrawParameters{}, buffersOnly);
    const auto all = QueuedDrawReadRanges(Pm4::DrawParameters{}, std::span<const Graphics::CompiledShader>(&own, 1));
    if (all.size() < entry.bufferReads.size()) return;
    entry.attributeReads.assign(all.begin() + static_cast<std::ptrdiff_t>(entry.bufferReads.size()), all.end());
    struct InfosStorage {};
    auto& infos = HostThreadLocal<std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>, InfosStorage>();
    infos.assign(count, std::nullopt);
    if (entry.stage < count) infos[entry.stage] = entry.vertexInfo;
    DrawVertexLayout(infos, entry.layoutWords, entry.vertexInputs);
    infos.clear();
    const auto layout = HashDrawRecordWords(entry.layoutWords, {});
    entry.layoutHash = layout[0] ^ (layout[1] * 0x9e3779b97f4a7c15ull);
    entry.attributesKept = !entry.vertexInfo.has_value() || std::ranges::all_of(entry.side.vertexAttributes, [&](const ShaderRecompiler::VertexAttribute& attribute) { return attribute.location < entry.vertexInfo->resources.size() && attribute.resource.fields == entry.vertexInfo->resources[attribute.location].fields; });
    entry.viewable = entry.stage < count;
}

using StageForwards = ShaderRecompiler::SideForwards;

std::shared_ptr<const StageForwards> stageForwards(const std::shared_ptr<const DirectStage>& shared, const ShaderRecompiler::PreparedShaderInvocation& invocation, const ShaderRecompiler::ResourceCapture& capture) {
    struct Known {
        std::shared_ptr<const DirectStage> stage;
        decltype(ShaderRecompiler::ResourceCapture::plan) plan;
        std::shared_ptr<const StageForwards> forwards;
    };
    static std::mutex mutex;
    static std::unordered_map<const DirectStage*, Known> known;
    {
        std::lock_guard lock(mutex);
        if (const auto found = known.find(shared.get()); found != known.end() && found->second.plan.get() == capture.plan.get()) return found->second.forwards;
    }
    auto traced = invocation.TraceSideForwards(capture, *shared->donor, *shared->patch);
    std::shared_ptr<const StageForwards> result = traced.traced ? std::make_shared<const StageForwards>(std::move(traced)) : nullptr;
    std::lock_guard lock(mutex);
    if (known.size() > 65536u) known.clear();
    auto& slot = known[shared.get()];
    slot = Known{shared, capture.plan, result};
    return slot.forwards;
}

using StageMemoMap = std::unordered_map<std::uint64_t, std::shared_ptr<const StageMemoEntry>>;

struct StageMemoShard {
    Graphics::SlimMutex mutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<const std::vector<StageMemoPattern>>> patterns;
    std::unordered_map<std::uint64_t, std::shared_ptr<const StageMemoEntry>> current;
    std::unordered_map<std::uint64_t, std::shared_ptr<const StageMemoEntry>> previous;
};

class StageMemoStore {
public:
    static constexpr std::size_t Ways = 4;
    static constexpr std::size_t Shards = 64;
    static constexpr std::size_t ShardBound = 512;

    std::shared_ptr<const std::vector<StageMemoPattern>> Patterns(std::uint64_t key) {
        auto& held = shard(key);
        std::lock_guard lock(held.mutex);
        const auto found = held.patterns.find(key);
        return found != held.patterns.end() ? found->second : nullptr;
    }

    std::shared_ptr<const StageMemoEntry> Find(std::uint64_t key) {
        auto& held = shard(key);
        StageMemoMap* retired = nullptr;
        std::shared_ptr<const StageMemoEntry> entry;
        {
            std::lock_guard lock(held.mutex);
            if (const auto found = held.current.find(key); found != held.current.end()) return found->second;
            const auto old = held.previous.find(key);
            if (old == held.previous.end()) return nullptr;
            entry = std::move(old->second);
            held.previous.erase(old);
            retired = place(held, key, entry);
        }
        if (retired != nullptr) forget(*retired);
        return entry;
    }

    std::optional<std::size_t> Way(std::uint64_t key, const StageMemoPattern& pattern, std::size_t limit, std::uint32_t foreign) {
        auto& held = shard(key);
        std::lock_guard lock(held.mutex);
        auto& ways = held.patterns[key];
        if (ways != nullptr) {
            const auto way = static_cast<std::size_t>(std::ranges::find(*ways, pattern) - ways->begin());
            if (way != ways->size()) return way;
            if (static_cast<std::size_t>(std::ranges::count_if(*ways, [&](const StageMemoPattern& known) { return known.arm == pattern.arm && (known.patchKinds & foreign) == 0u; })) >= limit) return std::nullopt;
        }
        auto grown = ways != nullptr ? std::make_shared<std::vector<StageMemoPattern>>(*ways) : std::make_shared<std::vector<StageMemoPattern>>();
        grown->push_back(pattern);
        const auto way = grown->size() - 1u;
        ways = std::move(grown);
        patternVersion.fetch_add(1, std::memory_order_release);
        return way;
    }

    std::uint64_t PatternVersion() const { return patternVersion.load(std::memory_order_acquire); }

    bool Admit(std::uint64_t key, bool third) {
        auto& cell = seen[static_cast<std::size_t>((key ^ (key >> 29u)) % SeenKeys)];
        const auto once = key & ~std::uint64_t{1};
        const auto twice = key | std::uint64_t{1};
        const auto held = cell.load(std::memory_order_relaxed);
        if (held == twice) return true;
        if (held == once) {
            if (!third) return true;
            cell.store(twice, std::memory_order_relaxed);
            return false;
        }
        cell.store(once, std::memory_order_relaxed);
        return false;
    }

    enum class MissCause : std::uint8_t { First, Second, Admitted, Missing, Rotated };

    MissCause Cause(std::uint64_t key, bool third, bool first) const {
        const auto index = cell(key);
        if (storedKeys[index].load(std::memory_order_relaxed) == key) return rotatedKeys[index].load(std::memory_order_relaxed) == key ? MissCause::Rotated : MissCause::Missing;
        if (first) return MissCause::Admitted;
        const auto held = seen[index].load(std::memory_order_relaxed);
        if (held == (key | std::uint64_t{1}) || (held == (key & ~std::uint64_t{1}) && !third)) return MissCause::Admitted;
        return held == (key & ~std::uint64_t{1}) ? MissCause::Second : MissCause::First;
    }

    void Insert(std::uint64_t key, std::shared_ptr<const StageMemoEntry> entry) {
        auto& held = shard(key);
        StageMemoMap* retired = nullptr;
        StageMemoMap::node_type replaced;
        std::shared_ptr<const StageMemoEntry> dropped;
        {
            std::lock_guard lock(held.mutex);
            if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoLeanStores)) {
                replaced = held.previous.extract(key);
                if (const auto found = held.current.find(key); found != held.current.end()) dropped = std::exchange(found->second, std::move(entry));
                else retired = place(held, key, entry);
            } else {
                held.previous.erase(key);
                retired = place(held, key, entry);
            }
        }
        storedKeys[cell(key)].store(key, std::memory_order_relaxed);
        if (retired != nullptr) forget(*retired);
    }

private:
    StageMemoShard& shard(std::uint64_t key) { return shards[static_cast<std::size_t>((key ^ (key >> 23u)) % Shards)]; }

    static std::size_t cell(std::uint64_t key) { return static_cast<std::size_t>((key ^ (key >> 29u)) % SeenKeys); }

    void note(const StageMemoMap& dropped) {
        for (const auto& [key, entry] : dropped) rotatedKeys[cell(key)].store(key, std::memory_order_relaxed);
    }

    void forget(StageMemoMap& dropped) {
        note(dropped);
        dropped.clear();
    }

    StageMemoMap* place(StageMemoShard& held, std::uint64_t key, const std::shared_ptr<const StageMemoEntry>& entry) {
        StageMemoMap* retired = nullptr;
        if (held.current.size() >= ShardBound && !held.current.contains(key)) {
            if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoLeanStores)) {
                struct RetiredStorage {};
                retired = &HostThreadLocal<StageMemoMap, RetiredStorage>();
                retired->clear();
                retired->swap(held.previous);
            } else {
                note(held.previous);
            }
            held.previous = std::move(held.current);
            held.current.clear();
            DirectDraws::Add(Counter::StageMemoCleared);
        }
        held.current.insert_or_assign(key, entry);
        return retired;
    }

    static constexpr std::size_t SeenKeys = std::size_t{1} << 17u;
    std::array<StageMemoShard, Shards> shards;
    std::atomic<std::uint64_t> patternVersion{1};
    std::unique_ptr<std::atomic<std::uint64_t>[]> seen{new std::atomic<std::uint64_t>[SeenKeys]()};
    std::unique_ptr<std::atomic<std::uint64_t>[]> storedKeys{new std::atomic<std::uint64_t>[SeenKeys]()};
    std::unique_ptr<std::atomic<std::uint64_t>[]> rotatedKeys{new std::atomic<std::uint64_t>[SeenKeys]()};
};

StageMemoStore& stageMemoStore() {
    static auto* store = new StageMemoStore;
    return *store;
}

struct StageMemoFrontSlot {
    std::uint64_t key = 0;
    std::shared_ptr<const StageMemoEntry> entry;
    std::uint32_t uses = 0;
};

struct StageMemo {
    static constexpr std::size_t FrontSets = 8192;
    std::vector<StageMemoFrontSlot> front;
    std::vector<std::uint32_t> gathered;
    std::vector<std::uint64_t> addresses;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> sorted;
    AddressRanges decodeRanges;
    AddressRanges settled;
    std::uint64_t verifyOrdinal = 0;
    std::uint64_t patchedOrdinal = 0;
    bool patched = false;
    std::uint64_t missKey = 0;
    std::uint64_t patternVersion = 0;
    std::unordered_map<std::uint64_t, std::shared_ptr<const std::vector<StageMemoPattern>>> patterns;
    static constexpr std::size_t MissWays = 8;
    static constexpr std::size_t RefusalSlots = 4096;
    std::uint64_t mergedOrdinal = 0;
    std::uint64_t relocatedOrdinal = 0;
    std::uint64_t stageKey = 0;
    std::array<std::uint64_t, MissWays> missKeys{};
    std::uint32_t missCount = 0;
    std::uint32_t tried = 0;
    std::uint32_t armWays = 0;
    bool merged = false;
    bool relocated = false;
    bool inputsTag = false;
    bool noEntryPlain = false;
    bool noEntryPatterned = false;
    bool forced = false;
    std::vector<std::uint8_t> refusals;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> conflicts;
    std::uint64_t volatileKey = 0;
    std::uint32_t volatileUntil = 0;
};

struct StageMemoStorage {};

enum class StageMemoOutcome : std::uint8_t { Hit, NoPattern, Gather, NoEntry, Inputs, Values, Views, Order, Unsettled, Side };

Counter stageMemoCounter(StageMemoOutcome outcome) {
    return static_cast<Counter>(static_cast<std::size_t>(Counter::StageMemoHits) + (outcome == StageMemoOutcome::Hit ? 0u : static_cast<std::size_t>(outcome) + 1u));
}

constexpr std::uint64_t StageMemoPrime = 1099511628211ull;

std::uint64_t stageMemoKey(const DrawDecode* decode, std::size_t stage) {
    auto key = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(decode)) * 0x9e3779b97f4a7c15ull;
    key = (key ^ stage) * StageMemoPrime;
    return key ^ (key >> 31u);
}

bool stageMemoRelocates(std::uint64_t mask, std::size_t word) {
    return word < ShaderRecompiler::SrtUserDataUse::TrackedWords && ((mask >> word) & 1u) != 0;
}

bool stageMemoRead(std::uint64_t address, std::uint32_t& value, std::uint64_t& page) {
    const auto base = address & ~(WalkPageBytes - 1u);
    const bool crossesPage = (address & (WalkPageBytes - 1u)) > WalkPageBytes - sizeof(value);
    if (crossesPage || base != page) {
        if (!crossesPage && GuestMemory::Accessible(reinterpret_cast<const void*>(base), WalkPageBytes)) {
            page = base;
        } else if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), sizeof(value))) {
            return false;
        }
    }
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    return true;
}

struct NestedRangesStorage {};

bool nestedClocksHeld(const StageMemoEntry& entry, std::uint64_t armed) {
    std::uint64_t last = ~std::uint64_t{0};
    for (const auto& word : entry.nested) {
        const auto first = word.first >> Validation::WindowShift;
        const auto end = (word.first + sizeof(std::uint32_t) - 1u) >> Validation::WindowShift;
        for (auto window = first;; ++window) {
            if (window != last && !windowClocksHeld(window, armed, true)) return false;
            last = window;
            if (window == end) break;
        }
    }
    return true;
}

bool nestedValuesSame(const StageMemoEntry& entry) {
    std::uint64_t page = ~std::uint64_t{0};
    for (const auto& [address, value] : entry.nested) {
        std::uint32_t live = 0;
        if (!stageMemoRead(address, live, page) || live != value) return false;
    }
    return true;
}

bool nestedArmHolds(const StageMemoEntry& entry, const DirectStages& resolved) {
    constexpr auto test = Validation::Test::NestedValues;
    const auto session = resolved.wvSession;
    if (session == 0 || session != Validation::Session()) return false;
    const auto packed = entry.nestedArm.validated.load(std::memory_order_acquire);
    if (packed == 0 || (packed >> ArmSessionShift) != session) return false;
    const auto forget = entry.nestedArm.forget.load(std::memory_order_acquire);
    if ((forget & 1u) != 0 || forget != GuestMemory::ForgetSerial()) {
        Validation::Note(test, Validation::Outcome::Moved);
        return false;
    }
    if (DrawFinisher::UnknownWritesAhead()) {
        Validation::Note(test, Validation::Outcome::Unknown);
        return false;
    }
    const auto armed = packed & ArmEpochMask;
    if (!nestedClocksHeld(entry, armed)) {
        Validation::Note(test, Validation::Outcome::Moved);
        return false;
    }
    Validation::Note(test, Validation::Outcome::Answered);
    if (!Validation::VerifyDue(test)) return true;
    if (nestedValuesSame(entry)) {
        Validation::NoteVerify(test, false, nullptr);
        return true;
    }
    Validation::NoteVerify(test, nestedClocksHeld(entry, armed), "a nested descriptor word changed although its clean, settled windows raised no event since its arm");
    return false;
}

void armNested(const StageMemoEntry& entry, const DirectStages& resolved) {
    constexpr auto test = Validation::Test::NestedValues;
    const auto session = resolved.wvSession;
    if (session == 0 || session != Validation::Session()) return;
    auto& arm = entry.nestedArm;
    if (const auto skips = arm.skips.load(std::memory_order_relaxed); skips != 0) {
        arm.skips.store(skips - 1u, std::memory_order_relaxed);
        return;
    }
    const auto forget = GuestMemory::ForgetSerial();
    if ((forget & 1u) != 0) return;
    auto& ranges = HostThreadLocal<AddressRanges, NestedRangesStorage>();
    ranges.clear();
    for (const auto& word : entry.nested) ranges.emplace_back(word.first, word.first + sizeof(std::uint32_t));
    auto& windows = HostThreadLocal<FinisherWindows, FinisherWindowsStorage>();
    const auto backoff = [&](Validation::Outcome outcome) {
        arm.skips.store(ArmBackoff, std::memory_order_relaxed);
        Validation::Note(test, outcome);
    };
    if (!collectWindows(ranges, windows)) return backoff(Validation::Outcome::Wide);
    ensureCleanForgetHook();
    const auto before = Validation::Epoch();
    if (!armable(session, before)) return;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!GuestMemory::AllClean(ranges)) return backoff(Validation::Outcome::NotClean);
    for (std::size_t at = 0; at < windows.count; ++at) {
        if (!settleProof(windows.windows[at])) return backoff(Validation::Outcome::Live);
    }
    if (!nestedValuesSame(entry)) return;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (GuestMemory::ForgetSerial() != forget) return;
    arm.forget.store(forget, std::memory_order_relaxed);
    arm.validated.store(packArm(session, before), std::memory_order_release);
    Validation::Note(test, Validation::Outcome::Armed);
}

std::uint64_t stageMemoContentKey(std::uint64_t key, std::size_t way, const StageMemoPattern& pattern, std::span<const std::uint32_t> userData, std::span<const std::uint32_t> rootValues, const Pm4::DrawParameters& parameters, std::uint32_t cursor, const DirectStage* previous) {
    auto hash = (key ^ way) * StageMemoPrime;
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoNoIndex)) {
        for (const std::uint64_t value : {parameters.indexAddress, std::uint64_t{parameters.indexCount}, std::uint64_t{parameters.indexSize}, std::uint64_t{parameters.instanceCount}}) hash = (hash ^ value) * StageMemoPrime;
    }
    for (const std::uint64_t value : {std::uint64_t{cursor}, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(previous))}) hash = (hash ^ value) * StageMemoPrime;
    for (std::size_t word = 0; word < userData.size(); ++word) {
        if (!stageMemoRelocates(pattern.relocatable, word)) hash = (hash ^ userData[word]) * StageMemoPrime;
    }
    for (std::size_t at = 0; at < rootValues.size(); ++at) {
        if (pattern.words.empty()) {
            hash = (hash ^ rootValues[at]) * StageMemoPrime;
            continue;
        }
        const auto& word = pattern.words[at];
        const bool guarded = stageMemoGuarded(word.kind);
        const std::uint64_t live = guarded && (rootValues[at] & ~word.key) != 0u ? std::uint64_t{1} << 32u : 0u;
        hash = (hash ^ (rootValues[at] & word.key) ^ live) * StageMemoPrime;
    }
    return hash;
}

int stageMemoSameRoots(const StageMemoPattern& pattern, std::span<const std::uint32_t> stored, std::span<const std::uint32_t> live) {
    if (stored.size() != live.size()) return 0;
    if (pattern.words.empty()) return std::equal(stored.begin(), stored.end(), live.begin()) ? 1 : 0;
    if (pattern.words.size() != live.size()) return 0;
    bool patched = false;
    for (std::size_t at = 0; at < live.size(); ++at) {
        const auto& word = pattern.words[at];
        if (((stored[at] ^ live[at]) & word.key) != 0u) return 0;
        const bool guarded = stageMemoGuarded(word.kind);
        if (guarded && (live[at] & ~word.key) == 0u) return 0;
        patched = patched || stored[at] != live[at];
    }
    return patched ? 2 : 1;
}

bool stageMemoCacheable(const DrawProgram& program) {
    const auto stage = program.binary.stage;
    if (stage != Stage::Vertex && stage != Stage::Fragment) return false;
    if (ShaderRecompiler::GcnGuestShaders() && stage == Stage::Vertex) return false;
    if (program.tailContinuation != nullptr) return false;
    return !program.binary.header.empty() || !program.snapshot->userDataTailRegister;
}

bool stageMemoDiag() {
    static const bool on = std::getenv("APS5_STAGE_MEMO_DIAG") != nullptr;
    return on;
}

struct StageMemoRootHistory {
    std::array<std::vector<std::uint32_t>, 16> kept;
    std::size_t next = 0;
};

enum class StageMemoGrant : std::uint8_t { Granted, DecodeKey, NoNodes, NoForwards, NodeForwards, Mixed, KindOff, AllKey, GuardedZero, NoPosition, NotVerbatim, BaseAbsolute, Count };

enum class StageMemoKeyReason : std::uint8_t { None, NoTape, UniformFill, GcnGuest, Interpreter, Uncovered, Condition, Arithmetic, NestedOperand, DwordKept, Clean, BaseChained, BaseAbsolute, NoUse, Count };

struct StageMemoDiagState {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, StageMemoRootHistory> lastRoots;
    std::array<std::uint64_t, static_cast<std::size_t>(StageMemoGrant::Count)> grants{};
    std::array<std::uint64_t, static_cast<std::size_t>(StageMemoKeyReason::Count)> reasons{};
    std::uint64_t stores = 0;
    std::uint64_t compared = 0;
    std::uint64_t full = 0;
    std::uint64_t patterned = 0;
    std::uint64_t plain = 0;
};

StageMemoDiagState& stageMemoDiagState() {
    static auto* state = new StageMemoDiagState;
    return *state;
}

StageMemoKeyReason stageMemoKeyReason(const ShaderRecompiler::SideForwards& forwards, const DirectReads& reads, std::span<const std::pair<std::uint64_t, std::int32_t>> walked, std::uint64_t address, bool chainedChild) {
    using Use = ShaderRecompiler::SideNodeUse;
    if (forwards.tape == ShaderRecompiler::SideTape::None) return StageMemoKeyReason::NoTape;
    if (forwards.tape == ShaderRecompiler::SideTape::UniformFill) return StageMemoKeyReason::UniformFill;
    if (forwards.tape == ShaderRecompiler::SideTape::GcnGuest) return StageMemoKeyReason::GcnGuest;
    std::uint32_t uses = 0;
    for (std::size_t at = 0; at < walked.size(); ++at) {
        if (walked[at].first != address) continue;
        if (reads.nodes.size() != walked.size() || reads.nodes[at] == ShaderRecompiler::SrtNoReadNode) return StageMemoKeyReason::Interpreter;
        const auto node = reads.nodes[at];
        if (node < forwards.nodeUses.size()) uses |= forwards.nodeUses[node];
    }
    const auto has = [&](Use use) { return (uses & static_cast<std::uint32_t>(use)) != 0u; };
    if (has(Use::Uncovered)) return StageMemoKeyReason::Uncovered;
    if (has(Use::Condition)) return StageMemoKeyReason::Condition;
    if (has(Use::Argument)) return StageMemoKeyReason::Arithmetic;
    if (has(Use::ReadArgument)) return StageMemoKeyReason::NestedOperand;
    if (has(Use::DwordKept)) return StageMemoKeyReason::DwordKept;
    if (has(Use::Clean)) return StageMemoKeyReason::Clean;
    if (has(Use::Base)) return chainedChild ? StageMemoKeyReason::BaseChained : StageMemoKeyReason::BaseAbsolute;
    return has(Use::Forward) ? StageMemoKeyReason::None : StageMemoKeyReason::NoUse;
}

void stageMemoKeyReasons(const StageMemoPattern& pattern, const StageMemoEntry& entry, std::span<const std::uint32_t> userData, const DirectReads& reads, std::span<const std::pair<std::uint64_t, std::int32_t>> walked, std::span<const StageMemoGrant> grants, std::vector<StageMemoKeyReason>& reasons) {
    reasons.assign(grants.size(), StageMemoKeyReason::None);
    for (std::size_t root = pattern.decodeRoots; root < pattern.roots.size(); ++root) {
        const auto wordIndex = entry.decodeWords + (root - pattern.decodeRoots);
        if (wordIndex >= grants.size() || grants[wordIndex] == StageMemoGrant::Granted || grants[wordIndex] == StageMemoGrant::DecodeKey) continue;
        const auto& placed = pattern.roots[root];
        const auto base = placed.chained != 0 ? static_cast<std::uint64_t>(entry.rootValues[placed.origin]) : static_cast<std::uint64_t>(userData[placed.origin]);
        const auto address = (base & ~std::uint64_t{3}) + placed.offset;
        const bool chainedChild = std::ranges::any_of(pattern.roots, [&](const StageMemoRoot& other) { return other.chained != 0 && other.origin == wordIndex; });
        reasons[wordIndex] = stageMemoKeyReason(*entry.forwards, reads, walked, address, chainedChild);
    }
}

void stageMemoGrantCensus(std::uint64_t key, std::size_t way, std::span<const std::uint32_t> rootValues, std::span<const StageMemoGrant> grants, std::span<const StageMemoKeyReason> keyReasons, bool full) {
    static constexpr std::array<const char*, static_cast<std::size_t>(StageMemoGrant::Count)> names{"granted", "decode word kept as key", "no read nodes", "no read forwards in the stage", "a read without forwards", "mixed kinds", "kind not enabled", "all bits key", "guarded zero", "side position missing", "side word not verbatim", "base of a read kept absolute"};
    static constexpr std::array<const char*, static_cast<std::size_t>(StageMemoKeyReason::Count)> reasonNames{"", "stage without a tape", "uniform fill", "GCN guest stage", "interpreter read", "uncovered side position", "condition root", "arithmetic, select or compare input", "high half, offset or records of a nested read", "descriptor dword without forward", "clean slot", "base of a nested read, child chained", "base of a nested read, child kept absolute", "no use found"};
    auto& state = stageMemoDiagState();
    std::lock_guard lock(state.mutex);
    ++state.stores;
    if (full) {
        ++state.full;
    } else if (const auto found = state.lastRoots.find(key ^ (way * StageMemoPrime)); found != state.lastRoots.end()) {
        const std::vector<std::uint32_t>* closest = nullptr;
        std::size_t differing = 0;
        for (const auto& candidate : found->second.kept) {
            if (candidate.size() != rootValues.size()) continue;
            std::size_t count = 0;
            for (std::size_t at = 0; at < rootValues.size(); ++at) count += candidate[at] != rootValues[at] ? 1u : 0u;
            if (count != 0 && (closest == nullptr || count < differing)) {
                closest = &candidate;
                differing = count;
            }
        }
        if (closest != nullptr) {
            ++state.compared;
            for (std::size_t at = 0; at < rootValues.size() && at < grants.size(); ++at) {
                if ((*closest)[at] == rootValues[at]) continue;
                ++state.grants[static_cast<std::size_t>(grants[at])];
                if (at < keyReasons.size() && keyReasons[at] != StageMemoKeyReason::None) ++state.reasons[static_cast<std::size_t>(keyReasons[at])];
            }
        }
    }
    if (state.stores % 20000u != 0u) return;
    std::string text;
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (state.grants[index] == 0) continue;
        text += text.empty() ? " " : ", ";
        text += names[index];
        text += ' ';
        text += std::to_string(state.grants[index]);
    }
    std::string finer;
    for (std::size_t index = 1; index < reasonNames.size(); ++index) {
        if (state.reasons[index] == 0) continue;
        finer += finer.empty() ? "; the key words among them by the recompiler's node uses: " : ", ";
        finer += reasonNames[index];
        finer += ' ';
        finer += std::to_string(state.reasons[index]);
    }
    std::fprintf(stderr, "[stage-memo-diag] %llu stores (%llu refused for full patterns), %llu with an earlier sighting of other root values; their differing root words by grant decision:%s; no-entry lookups under a way with patch words %llu, without %llu%s\n", static_cast<unsigned long long>(state.stores), static_cast<unsigned long long>(state.full), static_cast<unsigned long long>(state.compared), text.c_str(), static_cast<unsigned long long>(state.patterned), static_cast<unsigned long long>(state.plain), finer.c_str());
}

struct StageMemoDiagLast {
    std::vector<std::uint32_t> userData;
    std::uint64_t indexAddress = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t indexSize = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t cursor = 0;
    const DirectStage* previous = nullptr;
};

void stageMemoDiagnose(std::uint64_t key, std::size_t way, const StageMemoPattern& pattern, std::span<const std::uint32_t> userData, std::span<const std::uint32_t> gathered, const Pm4::DrawParameters& parameters, std::uint32_t cursor, const DirectStage* previous) {
    enum Cause : std::size_t { First, Same, IndexAddress, IndexCount, IndexSize, InstanceCount, Cursor, Previous, ValueWord, BaseValueWord, FixedBaseWord, IdleWord, Causes };
    static constexpr std::array<const char*, Causes> names{"first sight", "same key", "index address", "index count", "index size", "instance count", "push cursor", "previous stage", "value word", "base word read as a value", "base word kept fixed", "idle word"};
    auto& state = stageMemoDiagState();
    static std::unordered_map<std::uint64_t, StageMemoDiagLast> last;
    static std::array<std::uint64_t, Causes> causes{};
    static std::uint64_t total = 0;
    static std::uint32_t printed = 0;
    auto identity = (key ^ way) * StageMemoPrime;
    for (const auto value : gathered) identity = (identity ^ value) * StageMemoPrime;
    std::lock_guard lock(state.mutex);
    ++(pattern.patchKinds != 0u ? state.patterned : state.plain);
    if (last.size() > 400000u) last.clear();
    const auto [found, inserted] = last.try_emplace(identity);
    auto& kept = found->second;
    auto cause = First;
    std::size_t word = 0;
    if (!inserted) {
        cause = Same;
        if (kept.indexAddress != parameters.indexAddress) cause = IndexAddress;
        else if (kept.indexCount != parameters.indexCount) cause = IndexCount;
        else if (kept.indexSize != parameters.indexSize) cause = IndexSize;
        else if (kept.instanceCount != parameters.instanceCount) cause = InstanceCount;
        else if (kept.cursor != cursor) cause = Cursor;
        else if (kept.previous != previous) cause = Previous;
        else {
            for (; word < userData.size() && word < kept.userData.size(); ++word) {
                if (stageMemoRelocates(pattern.relocatable, word) || kept.userData[word] == userData[word]) continue;
                const bool tracked = word < ShaderRecompiler::SrtUserDataUse::TrackedWords;
                const bool value = !tracked || ((pattern.values >> word) & 1u) != 0;
                const bool base = tracked && ((pattern.bases >> word) & 1u) != 0;
                cause = value && base ? BaseValueWord : value ? ValueWord : base ? FixedBaseWord : IdleWord;
                break;
            }
        }
        if (cause >= ValueWord && printed < 48u) {
            ++printed;
            std::fprintf(stderr, "[stage-memo-diag] same roots, key differs in a %s: word %zu of %zu, %08x -> %08x; movable %016llx, values %016llx, bases %016llx, %zu roots (%u decode)\n", names[cause], word, userData.size(), kept.userData[word], userData[word], static_cast<unsigned long long>(pattern.relocatable), static_cast<unsigned long long>(pattern.values), static_cast<unsigned long long>(pattern.bases), pattern.roots.size(), pattern.decodeRoots);
        }
    }
    auto& lastRoots = state.lastRoots;
    static std::array<std::uint64_t, 7> rootBuckets{};
    static std::uint32_t rootsPrinted = 0;
    if (cause == First) {
        if (lastRoots.size() > 100000u) lastRoots.clear();
        auto& history = lastRoots[key ^ (way * StageMemoPrime)];
        const std::vector<std::uint32_t>* closest = nullptr;
        std::size_t differing = 0;
        bool any = false;
        for (const auto& candidate : history.kept) {
            if (candidate.empty()) continue;
            any = true;
            if (candidate.size() != gathered.size()) continue;
            std::size_t count = 0;
            for (std::size_t at = 0; at < gathered.size(); ++at) count += candidate[at] != gathered[at] ? 1u : 0u;
            if (closest == nullptr || count < differing) {
                closest = &candidate;
                differing = count;
            }
        }
        static const std::vector<std::uint32_t> none;
        const auto& before = closest != nullptr ? *closest : none;
        if (!any) {
            ++rootBuckets[0];
        } else if (closest == nullptr) {
            ++rootBuckets[6];
        } else {
            ++rootBuckets[differing <= 2 ? 1 : differing <= 4 ? 2 : differing <= 8 ? 3 : differing <= 16 ? 4 : 5];
            if (differing != 0 && rootsPrinted < 120u && (rootsPrinted < 40u || total > 1500000u)) {
                ++rootsPrinted;
                std::string text;
                std::size_t at = 0;
                std::size_t listed = 0;
                for (std::size_t root = 0; root < pattern.roots.size(); ++root) {
                    for (std::uint32_t word = 0; word < pattern.roots[root].words && at < gathered.size(); ++word, ++at) {
                        if (before[at] == gathered[at] || listed >= 6u) continue;
                        ++listed;
                        char line[128];
                        std::snprintf(line, sizeof(line), " root %zu%s %s %u +0x%llx word %u: %08x -> %08x;", root, root < pattern.decodeRoots ? " (decode)" : "", pattern.roots[root].chained != 0 ? "chained on word" : "user", static_cast<unsigned>(pattern.roots[root].origin), static_cast<unsigned long long>(pattern.roots[root].offset), static_cast<unsigned>(word), before[at], gathered[at]);
                        text += line;
                    }
                }
                std::fprintf(stderr, "[stage-memo-diag] first sight of root values, %zu of %zu words differ from the closest of the pattern's last 16 sightings:%s\n", differing, gathered.size(), text.c_str());
            }
        }
        history.kept[history.next++ % history.kept.size()].assign(gathered.begin(), gathered.end());
    }
    ++causes[cause];
    kept.userData.assign(userData.begin(), userData.end());
    kept.indexAddress = parameters.indexAddress;
    kept.indexCount = parameters.indexCount;
    kept.indexSize = parameters.indexSize;
    kept.instanceCount = parameters.instanceCount;
    kept.cursor = cursor;
    kept.previous = previous;
    enum MaskedCause : std::size_t { MaskedNew, MaskedSame, MaskedIndex, MaskedCursor, MaskedPrevious, MaskedWord, MaskedCauses };
    static constexpr std::array<const char*, MaskedCauses> maskedNames{"new", "seen with the same inputs", "index parameters", "push cursor", "previous stage", "user value word"};
    static std::unordered_map<std::uint64_t, StageMemoDiagLast> masked;
    static std::array<std::uint64_t, MaskedCauses> maskedCauses{};
    static std::array<std::uint64_t, ShaderRecompiler::SrtUserDataUse::TrackedWords> maskedWords{};
    if (!pattern.words.empty() && pattern.words.size() == gathered.size()) {
        auto maskedIdentity = (key ^ way ^ 0x9e3779b97f4a7c15ull) * StageMemoPrime;
        for (std::size_t at = 0; at < gathered.size(); ++at) {
            const auto& held = pattern.words[at];
            const std::uint64_t live = stageMemoGuarded(held.kind) && (gathered[at] & ~held.key) != 0u ? std::uint64_t{1} << 32u : 0u;
            maskedIdentity = (maskedIdentity ^ (gathered[at] & held.key) ^ live) * StageMemoPrime;
        }
        if (masked.size() > 400000u) masked.clear();
        const auto [seenAt, fresh] = masked.try_emplace(maskedIdentity);
        auto& before = seenAt->second;
        auto maskedCause = MaskedNew;
        if (!fresh) {
            maskedCause = MaskedSame;
            if (before.indexAddress != parameters.indexAddress || before.indexCount != parameters.indexCount || before.indexSize != parameters.indexSize || before.instanceCount != parameters.instanceCount) maskedCause = MaskedIndex;
            else if (before.cursor != cursor) maskedCause = MaskedCursor;
            else if (before.previous != previous) maskedCause = MaskedPrevious;
            else {
                for (std::size_t at = 0; at < userData.size() && at < before.userData.size(); ++at) {
                    if (stageMemoRelocates(pattern.relocatable, at) || before.userData[at] == userData[at]) continue;
                    maskedCause = MaskedWord;
                    if (at < maskedWords.size()) ++maskedWords[at];
                    break;
                }
            }
        }
        ++maskedCauses[maskedCause];
        before.userData.assign(userData.begin(), userData.end());
        before.indexAddress = parameters.indexAddress;
        before.indexCount = parameters.indexCount;
        before.indexSize = parameters.indexSize;
        before.instanceCount = parameters.instanceCount;
        before.cursor = cursor;
        before.previous = previous;
    }
    if (total % 200000u == 199999u) {
        std::string text;
        for (std::size_t index = 0; index < MaskedCauses; ++index) {
            if (maskedCauses[index] == 0) continue;
            text += text.empty() ? " " : ", ";
            text += maskedNames[index];
            text += ' ';
            text += std::to_string(maskedCauses[index]);
        }
        std::string words;
        for (std::size_t index = 0; index < maskedWords.size(); ++index) {
            if (maskedWords[index] == 0) continue;
            words += words.empty() ? " (" : ", ";
            words += "word ";
            words += std::to_string(index);
            words += ' ';
            words += std::to_string(maskedWords[index]);
        }
        if (!words.empty()) words += ')';
        std::fprintf(stderr, "[stage-memo-diag] no-entry misses under ways with patch words, by their masked root content:%s%s\n", text.c_str(), words.c_str());
    }
    if (++total % 200000u == 0u) {
        std::string text;
        for (std::size_t index = 0; index < Causes; ++index) {
            if (causes[index] == 0) continue;
            text += text.empty() ? " " : ", ";
            text += names[index];
            text += ' ';
            text += std::to_string(causes[index]);
        }
        std::fprintf(stderr, "[stage-memo-diag] %llu misses without an entry:%s; first sights by root words that differ from the closest of the pattern's last 16 sightings: no sighting before %llu, 1 to 2 words %llu, 3 to 4 %llu, 5 to 8 %llu, 9 to 16 %llu, more %llu, another word count %llu\n", static_cast<unsigned long long>(total), text.c_str(), static_cast<unsigned long long>(rootBuckets[0]), static_cast<unsigned long long>(rootBuckets[1]), static_cast<unsigned long long>(rootBuckets[2]), static_cast<unsigned long long>(rootBuckets[3]), static_cast<unsigned long long>(rootBuckets[4]), static_cast<unsigned long long>(rootBuckets[5]), static_cast<unsigned long long>(rootBuckets[6]));
    }
}

void entryFailed(StageMemo& memo, std::uint64_t contentKey, const StageMemoEntry& entry, TrustSite site, TrustUse use) {
    if (!audited(use)) return;
    noteFailed(entry.trust, site, use);
    memo.volatileKey = contentKey;
    memo.volatileUntil = entry.trust.volatileUntil.load(std::memory_order_relaxed);
}

StageMemoOutcome adoptViews(StageMemo& memo, const StageMemoEntry& entry, const std::shared_ptr<const StageMemoEntry>& owning, const StageMemoPattern& pattern, std::span<const std::uint32_t> userData, int sameRoots, DirectStages& resolved, std::size_t index, std::uint64_t contentKey, TrustUse entryUse) {
    auto& stage = resolved.stages[index];
    const auto& gathered = memo.gathered;
    const auto& addresses = memo.addresses;
    const bool sameAddresses = addresses == entry.addresses;
    const bool sameValues = sameRoots == 1;
    const bool vertexPatched = (pattern.patchKinds & (1u << StageMemoVertexKind)) != 0u;
    const auto walkRoots = pattern.roots.size() - pattern.decodeRoots;
    if (sameAddresses) {
        if (!sameValues) {
            stage.values.clear();
            for (const auto slotIndex : entry.order) stage.values.push_back(slotIndex < walkRoots ? gathered[entry.decodeWords + slotIndex] : entry.nested[slotIndex - walkRoots].second);
        }
    } else {
        auto& sorted = memo.sorted;
        sorted.clear();
        bool increasing = true;
        for (const auto slotIndex : entry.order) {
            const bool isRoot = slotIndex < walkRoots;
            const auto address = isRoot ? addresses[pattern.decodeRoots + slotIndex] : entry.nested[slotIndex - walkRoots].first;
            const auto value = isRoot ? gathered[entry.decodeWords + slotIndex] : entry.nested[slotIndex - walkRoots].second;
            if (!sorted.empty() && address <= sorted.back().first) increasing = false;
            sorted.emplace_back(address, value);
        }
        if (!increasing) {
            std::sort(sorted.begin(), sorted.end());
            bool torn = false;
            for (std::size_t at = 1; at < sorted.size(); ++at) torn = torn || sorted[at].first == sorted[at - 1].first;
            if (torn) return StageMemoOutcome::Order;
        }
        stage.ranges.clear();
        stage.values.clear();
        for (const auto& [address, value] : sorted) {
            if (!stage.ranges.empty() && stage.ranges.back().second == address) stage.ranges.back().second += sizeof(std::uint32_t);
            else stage.ranges.emplace_back(address, address + sizeof(std::uint32_t));
            stage.values.push_back(value);
        }
    }
    const auto& ranges = sameAddresses ? entry.ranges : stage.ranges;
    const bool settleTrusted = trusted() || entryUse == TrustUse::Skip;
    noteUse(TrustSite::ViewSettle, entryUse);
    const auto viewUnsettled = [&] {
        entryFailed(memo, contentKey, entry, TrustSite::ViewSettle, entryUse);
        return StageMemoOutcome::Unsettled;
    };
    if (!settleTrusted && entry.decodeReads.empty()) {
        if (!settledValidated(ranges, resolved, Validation::Test::ViewSettle)) return viewUnsettled();
    } else if (!settleTrusted) {
        auto& decodeRanges = memo.decodeRanges;
        if (!sameAddresses) {
            decodeRanges.clear();
            for (std::size_t read = 0; read < entry.decodeReads.size(); ++read) decodeRanges.emplace_back(addresses[read], addresses[read] + entry.decodeReads[read].size);
        }
        if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle)) {
            if (!settledValidated(ranges, resolved, Validation::Test::ViewSettle) || !settledValidated(sameAddresses ? entry.decodeRanges : decodeRanges, resolved, Validation::Test::ViewSettle)) return viewUnsettled();
        } else if (sameAddresses) {
            if (!settledValidated(entry.settled, resolved, Validation::Test::ViewSettle)) return viewUnsettled();
        } else {
            auto& settled = memo.settled;
            settled.assign(ranges.begin(), ranges.end());
            settled.insert(settled.end(), decodeRanges.begin(), decodeRanges.end());
            if (!settledValidated(settled, resolved, Validation::Test::ViewSettle)) return viewUnsettled();
        }
    }
    bool forwardsSame = true;
    for (const auto& forward : entry.forwards->forwards) {
        if (forward.word >= userData.size()) return StageMemoOutcome::Side;
        forwardsSame = forwardsSame && userData[forward.word] == entry.userData[forward.word];
    }
    const bool sideSame = forwardsSame && (sameValues || (entry.patches.empty() && !vertexPatched)) && (!vertexPatched || entry.attributesKept);
    if (!sideSame) {
        stage.side = entry.side;
        for (const auto& forward : entry.forwards->forwards) {
            const auto stored = entry.userData[forward.word];
            if (forward.push) {
                if (forward.position < stage.side.pushWords.size() && stage.side.pushWords[forward.position].second == stored) stage.side.pushWords[forward.position].second = userData[forward.word];
            } else if (forward.binding < stage.side.bindings.size() && forward.position < stage.side.bindings[forward.binding].size()) {
                auto& held = stage.side.bindings[forward.binding][forward.position];
                if (held == stored) held = userData[forward.word];
            }
        }
        for (const auto& patch : entry.patches) {
            auto& word = stage.side.bindings[patch.binding][patch.position];
            word = (word & ~patch.mask) | (gathered[patch.word] & patch.mask);
        }
    }
    memo.patched = sameRoots == 2;
    GuardRanges(ranges);
    const bool vertexSame = !vertexPatched || sameValues;
    const bool decodeSame = sameAddresses && vertexSame;
    if (!decodeSame) {
        auto& decodeReads = resolved.decodeReads[index];
        decodeReads = entry.decodeReads;
        for (std::size_t read = 0; read < decodeReads.size(); ++read) decodeReads[read].address = addresses[read];
        if (vertexPatched && entry.vertexInfo.has_value()) {
            std::size_t at = 0;
            for (std::size_t read = 0; read < decodeReads.size(); ++read) {
                auto& decoded = decodeReads[read];
                std::memcpy(decoded.data.data(), gathered.data() + at, decoded.size);
                at += decoded.size / sizeof(std::uint32_t);
            }
        }
    }
    if (!vertexSame) {
        resolved.vertexInfos[index] = entry.vertexInfo;
        if (resolved.vertexInfos[index].has_value()) {
            auto& info = *resolved.vertexInfos[index];
            std::size_t at = 0;
            for (std::size_t read = 0; read < entry.decodeReads.size(); ++read) {
                if ((read & 1u) != 0) info.resources[read / 2u].fields = Graphics::VertexResourceFields(std::span<const std::uint32_t, 4>(gathered.data() + at, 4), gathered[at - 1u]);
                at += entry.decodeReads[read].size / sizeof(std::uint32_t);
            }
        }
    }
    if (decodeSame) stage.decodeView = &entry.decodeReads;
    if (vertexSame) stage.vertexView = &entry.vertexInfo;
    if (!sideSame && vertexPatched && resolved.VertexInfo(index).has_value()) {
        const auto& info = *resolved.VertexInfo(index);
        for (auto& attribute : stage.side.vertexAttributes) attribute.resource = info.resources[attribute.location];
    }
    stage.sharedView = &entry.shared;
    stage.captureView = &entry.capture;
    stage.pushOffset = entry.pushOffset;
    if (sideSame) stage.sideView = &entry.side;
    if (sameAddresses) stage.rangesView = &entry.ranges;
    if (sameAddresses && sameValues) stage.valuesView = &entry.values;
    stage.memo = &entry;
    stage.pin = owning;
    if (sideSame) DirectDraws::Add(Counter::StageViewSides);
    if (sameAddresses) DirectDraws::Add(Counter::StageViewRanges);
    if (sameAddresses && sameValues) DirectDraws::Add(Counter::StageViewValues);
    if (decodeSame && !entry.decodeReads.empty()) DirectDraws::Add(Counter::StageViewDecodes);
    if (vertexSame && entry.vertexInfo.has_value()) DirectDraws::Add(Counter::StageViewVertices);
    return StageMemoOutcome::Hit;
}

StageMemoOutcome adoptStage(StageMemo& memo, const LeanDrawJob& job, std::span<const std::uint32_t> userData, std::size_t index, std::uint32_t cursor, const InvocationSlot& slot, DirectStages& resolved) {
    const auto& decode = job.shape->decode;
    const auto key = stageMemoKey(decode.get(), index);
    memo.stageKey = key;
    memo.missCount = 0;
    memo.tried = 0;
    memo.armWays = 0;
    memo.merged = false;
    memo.relocated = false;
    memo.inputsTag = false;
    memo.noEntryPlain = false;
    memo.noEntryPatterned = false;
    auto& store = stageMemoStore();
    if (const auto version = store.PatternVersion(); memo.patternVersion != version) {
        memo.patterns.clear();
        memo.patternVersion = version;
    }
    auto cached = memo.patterns.find(key);
    if (cached == memo.patterns.end()) {
        auto fetched = store.Patterns(key);
        if (fetched == nullptr) return StageMemoOutcome::NoPattern;
        cached = memo.patterns.emplace(key, std::move(fetched)).first;
    }
    const auto* const ways = cached->second.get();
    const auto enabledKinds = stageMemoPatchKinds();
    const auto arm = stageMemoArm(enabledKinds);
    const auto foreign = stageMemoForeignKinds(enabledKinds);
    memo.patched = false;
    auto& stage = resolved.stages[index];
    const DirectStage* previous = index != 0 ? resolved.stages[index - 1].Shared().get() : nullptr;
    auto last = StageMemoOutcome::NoEntry;
    for (std::size_t way = 0; way < ways->size(); ++way) {
        const auto& pattern = (*ways)[way];
        if (pattern.arm == arm && (pattern.patchKinds & foreign) == 0u) ++memo.armWays;
        if (pattern.userWords != userData.size() || (pattern.patchKinds & ~enabledKinds) != 0u || pattern.arm != arm) continue;
        ++memo.tried;
        auto& gathered = memo.gathered;
        auto& addresses = memo.addresses;
        gathered.clear();
        addresses.clear();
        std::uint64_t page = ~std::uint64_t{0};
        bool readable = true;
        for (const auto& root : pattern.roots) {
            std::uint64_t address = 0;
            if (root.chained != 0) {
                if (root.origin >= gathered.size()) {
                    readable = false;
                    break;
                }
                const auto at = static_cast<std::int64_t>(static_cast<std::uint64_t>(gathered[root.origin]) & ~std::uint64_t{3}) + static_cast<std::int64_t>(root.offset);
                if (at < 0 || static_cast<std::uint64_t>(at) > ShaderRecompiler::Detail::AddressMask) {
                    readable = false;
                    break;
                }
                address = static_cast<std::uint64_t>(at);
            } else {
                const auto word = userData[root.origin];
                address = (root.aligned != 0 ? static_cast<std::uint64_t>(word) & ~std::uint64_t{3} : static_cast<std::uint64_t>(word)) + root.offset;
            }
            addresses.push_back(address);
            for (std::uint32_t at = 0; readable && at < root.words; ++at) {
                std::uint32_t value = 0;
                readable = stageMemoRead(address + at * sizeof(std::uint32_t), value, page);
                gathered.push_back(value);
            }
            if (!readable) break;
        }
        if (!readable) {
            last = StageMemoOutcome::Gather;
            continue;
        }
        const auto contentKey = stageMemoContentKey(key, way, pattern, userData, gathered, job.parameters, cursor, previous);
        std::shared_ptr<const StageMemoEntry> fetched;
        const StageMemoEntry* held = nullptr;
        const std::shared_ptr<const StageMemoEntry>* owning = &fetched;
        if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoFront)) {
            if (memo.front.empty()) memo.front.resize(StageMemo::FrontSets * 2u);
            auto* set = memo.front.data() + static_cast<std::size_t>((contentKey ^ (contentKey >> 19u)) % StageMemo::FrontSets) * 2u;
            auto* slot = set[0].key == contentKey && set[0].entry != nullptr ? &set[0] : set[1].key == contentKey && set[1].entry != nullptr ? &set[1] : nullptr;
            if (slot != nullptr) {
                held = slot->entry.get();
                owning = &slot->entry;
                if ((++slot->uses & 63u) == 0u && store.Find(contentKey) == nullptr) store.Insert(contentKey, slot->entry);
            } else if ((fetched = store.Find(contentKey)) != nullptr) {
                auto& victim = set[0].uses <= set[1].uses ? set[0] : set[1];
                set[0].uses >>= 1u;
                set[1].uses >>= 1u;
                victim.key = contentKey;
                victim.uses = 1;
                victim.entry = std::move(fetched);
                held = victim.entry.get();
                owning = &victim.entry;
            }
        } else {
            fetched = store.Find(contentKey);
            held = fetched.get();
        }
        if (held == nullptr) {
            last = StageMemoOutcome::NoEntry;
            memo.missKey = contentKey;
            if (memo.missCount < StageMemo::MissWays) memo.missKeys[memo.missCount++] = contentKey;
            (pattern.patchKinds != 0u ? memo.noEntryPatterned : memo.noEntryPlain) = true;
            if (stageMemoDiag()) stageMemoDiagnose(key, way, pattern, userData, gathered, job.parameters, cursor, previous);
            continue;
        }
        const auto& entry = *held;
        const auto& parameters = job.parameters;
        const auto entryUse = auditUse(entry.trust);
        const bool inputsTrusted = trusted() || entryUse == TrustUse::Skip;
        noteUse(TrustSite::StageInputs, entryUse);
        if (resolved.wvSession != 0) Validation::Note(Validation::Test::StageInputs, Validation::Outcome::Kept);
        const auto sameInputs = [&](bool trustedInputs) {
            bool same = trustedInputs ? (entry.tag == resolved.tag && entry.variant == slot.variant && entry.convertedSamples == slot.convertedSamples && entry.layerRowSamples == slot.layerRowSamples && entry.userData.size() == userData.size()) : entry.decode.get() == decode.get() && entry.stage == index && entry.way == way && entry.cursor == cursor && entry.tag == resolved.tag && entry.previous.get() == previous && (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoNoIndex) || (entry.indexAddress == parameters.indexAddress && entry.indexCount == parameters.indexCount && entry.indexSize == parameters.indexSize && entry.instanceCount == parameters.instanceCount)) && entry.variant == slot.variant && entry.convertedSamples == slot.convertedSamples && entry.layerRowSamples == slot.layerRowSamples && entry.userData.size() == userData.size();
            for (std::size_t word = 0; same && word < userData.size(); ++word) {
                if (!stageMemoRelocates(pattern.relocatable, word)) same = trustedInputs || entry.userData[word] == userData[word];
                else if (((pattern.addresses >> word) & 1u) != 0) same = userData[word] != 0u && ((userData[word] ^ entry.userData[word]) & 3u) == 0u;
                else if (((pattern.bases >> word) & 1u) != 0) same = userData[word] != 0u;
            }
            return same;
        };
        if (!sameInputs(inputsTrusted)) {
            if (audited(entryUse) && sameInputs(true)) entryFailed(memo, contentKey, entry, TrustSite::StageInputs, entryUse);
            last = StageMemoOutcome::Inputs;
            memo.inputsTag = entry.tag != resolved.tag;
            continue;
        }
        const auto sameRoots = stageMemoSameRoots(pattern, entry.rootValues, gathered);
        if (sameRoots == 0) {
            last = StageMemoOutcome::Values;
            continue;
        }
        bool nestedSame = true;
        const bool nestedTrusted = !entry.nested.empty() && (trusted() || entryUse == TrustUse::Skip);
        if (!entry.nested.empty()) noteUse(TrustSite::StageNested, entryUse);
        const bool nestedHeld = !nestedTrusted && !entry.nested.empty() && nestedArmHolds(entry, resolved);
        for (const auto& [address, value] : entry.nested) {
            if (nestedTrusted || nestedHeld) break;
            std::uint32_t live = 0;
            if (!stageMemoRead(address, live, page) || live != value) {
                nestedSame = false;
                break;
            }
        }
        if (nestedSame && !nestedTrusted && !nestedHeld && !entry.nested.empty()) armNested(entry, resolved);
        if (!nestedSame) {
            entryFailed(memo, contentKey, entry, TrustSite::StageNested, entryUse);
            last = StageMemoOutcome::Values;
            continue;
        }
        if (!CaptureRegisteredViewsCurrent(entry.capture->snapshot.registeredImages)) {
            last = StageMemoOutcome::Views;
            continue;
        }
        memo.merged = false;
        memo.relocated = false;
        if (sameRoots == 2 && (pattern.patchKinds & ((1u << StageMemoMergedKind) | (1u << StageMemoBaseKind))) != 0u) {
            for (std::size_t at = 0; at < gathered.size() && at < pattern.words.size(); ++at) memo.merged = memo.merged || (pattern.words[at].merged != 0 && gathered[at] != entry.rootValues[at]);
            for (const auto& root : pattern.roots) memo.relocated = memo.relocated || (root.chained != 0 && ((gathered[root.origin] ^ entry.rootValues[root.origin]) & ~3u) != 0u);
        }
        if (entry.viewable && resolved.viewing) {
            const auto viewed = adoptViews(memo, entry, *owning, pattern, userData, sameRoots, resolved, index, contentKey, entryUse);
            if (viewed != StageMemoOutcome::Order) return viewed;
            last = viewed;
            continue;
        }
        const auto walkRoots = pattern.roots.size() - pattern.decodeRoots;
        auto& sorted = memo.sorted;
        sorted.clear();
        bool increasing = true;
        for (const auto slotIndex : entry.order) {
            const bool isRoot = slotIndex < walkRoots;
            const auto address = isRoot ? addresses[pattern.decodeRoots + slotIndex] : entry.nested[slotIndex - walkRoots].first;
            const auto value = isRoot ? gathered[entry.decodeWords + slotIndex] : entry.nested[slotIndex - walkRoots].second;
            if (!sorted.empty() && address <= sorted.back().first) increasing = false;
            sorted.emplace_back(address, value);
        }
        if (!increasing) {
            std::sort(sorted.begin(), sorted.end());
            bool torn = false;
            for (std::size_t at = 1; at < sorted.size(); ++at) torn = torn || sorted[at].first == sorted[at - 1].first;
            if (torn) {
                last = StageMemoOutcome::Order;
                continue;
            }
        }
        stage.ranges.clear();
        stage.values.clear();
        for (const auto& [address, value] : sorted) {
            if (!stage.ranges.empty() && stage.ranges.back().second == address) stage.ranges.back().second += sizeof(std::uint32_t);
            else stage.ranges.emplace_back(address, address + sizeof(std::uint32_t));
            stage.values.push_back(value);
        }
        auto& decodeRanges = memo.decodeRanges;
        decodeRanges.clear();
        const bool settleTrusted = trusted() || entryUse == TrustUse::Skip;
        noteUse(TrustSite::StageSettle, entryUse);
        const auto entryUnsettled = [&] {
            entryFailed(memo, contentKey, entry, TrustSite::StageSettle, entryUse);
            return StageMemoOutcome::Unsettled;
        };
        for (std::size_t read = 0; !settleTrusted && read < entry.decodeReads.size(); ++read) decodeRanges.emplace_back(addresses[read], addresses[read] + entry.decodeReads[read].size);
        if (!settleTrusted && decodeRanges.empty()) {
            if (!settledValidated(stage.ranges, resolved, Validation::Test::StageSettle)) return entryUnsettled();
        } else if (!settleTrusted && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle)) {
            if (!settledValidated(stage.ranges, resolved, Validation::Test::StageSettle) || !settledValidated(decodeRanges, resolved, Validation::Test::StageSettle)) return entryUnsettled();
        } else if (!settleTrusted) {
            auto& settled = memo.settled;
            settled.assign(stage.ranges.begin(), stage.ranges.end());
            settled.insert(settled.end(), decodeRanges.begin(), decodeRanges.end());
            if (!settledValidated(settled, resolved, Validation::Test::StageSettle)) return entryUnsettled();
        }
        stage.side = entry.side;
        bool forwarded = true;
        for (const auto& forward : entry.forwards->forwards) {
            if (forward.word >= userData.size()) {
                forwarded = false;
                continue;
            }
            const auto stored = entry.userData[forward.word];
            if (forward.push) {
                if (forward.position < stage.side.pushWords.size() && stage.side.pushWords[forward.position].second == stored) stage.side.pushWords[forward.position].second = userData[forward.word];
            } else if (forward.binding < stage.side.bindings.size() && forward.position < stage.side.bindings[forward.binding].size()) {
                auto& held = stage.side.bindings[forward.binding][forward.position];
                if (held == stored) held = userData[forward.word];
            }
        }
        if (!forwarded) return StageMemoOutcome::Side;
        for (const auto& patch : entry.patches) {
            auto& word = stage.side.bindings[patch.binding][patch.position];
            word = (word & ~patch.mask) | (gathered[patch.word] & patch.mask);
        }
        memo.patched = sameRoots == 2;
        GuardRanges(stage.ranges);
        auto& decodeReads = resolved.decodeReads[index];
        decodeReads = entry.decodeReads;
        for (std::size_t read = 0; read < decodeReads.size(); ++read) decodeReads[read].address = addresses[read];
        resolved.vertexInfos[index] = entry.vertexInfo;
        if ((pattern.patchKinds & (1u << StageMemoVertexKind)) != 0u && resolved.vertexInfos[index].has_value()) {
            auto& info = *resolved.vertexInfos[index];
            std::size_t at = 0;
            for (std::size_t read = 0; read < decodeReads.size(); ++read) {
                auto& decoded = decodeReads[read];
                std::memcpy(decoded.data.data(), gathered.data() + at, decoded.size);
                if ((read & 1u) != 0) info.resources[read / 2u].fields = Graphics::VertexResourceFields(std::span<const std::uint32_t, 4>(gathered.data() + at, 4), gathered[at - 1u]);
                at += decoded.size / sizeof(std::uint32_t);
            }
            for (auto& attribute : stage.side.vertexAttributes) attribute.resource = info.resources[attribute.location];
        }
        stage.capture = entry.capture;
        stage.shared = entry.shared;
        stage.pushOffset = entry.pushOffset;
        return StageMemoOutcome::Hit;
    }
    return memo.armWays == 0 && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoAdmitWays) ? StageMemoOutcome::NoPattern : last;
}

struct StageMemoPlaced {
    std::uint64_t address;
    std::uint32_t slot;
    bool nested;
};

struct StageMemoByNode {
    bool operator()(const ShaderRecompiler::SideReadForward& forward, std::uint32_t node) const { return forward.node < node; }
    bool operator()(std::uint32_t node, const ShaderRecompiler::SideReadForward& forward) const { return node < forward.node; }
};

struct StageMemoByParent {
    bool operator()(const ShaderRecompiler::SideReadBase& base, std::uint32_t node) const { return base.parent < node; }
    bool operator()(std::uint32_t node, const ShaderRecompiler::SideReadBase& base) const { return node < base.parent; }
};

std::optional<Counter> stageMemoFinish(std::span<const std::uint32_t> userData, const ShaderRecompiler::SrtUserDataUse& use, std::uint64_t addresses, std::uint64_t relocatable, StageMemoPattern& pattern, StageMemoEntry& entry, std::vector<StageMemoWord>& words, std::vector<StageMemoPlaced>& placed) {
    if (pattern.patchKinds != 0u) pattern.words = std::move(words);
    for (std::size_t word = 0; word < userData.size() && word < ShaderRecompiler::SrtUserDataUse::TrackedWords; ++word) {
        const auto bit = std::uint64_t{1} << word;
        if (((use.asBase | addresses) & bit) != 0 && userData[word] == 0u) relocatable &= ~bit;
    }
    pattern.relocatable = relocatable;
    pattern.values = use.asValue;
    pattern.bases = use.asBase;
    pattern.addresses = addresses & relocatable;
    const auto walkRoots = static_cast<std::uint32_t>(pattern.roots.size() - pattern.decodeRoots);
    std::sort(placed.begin(), placed.end(), [](const StageMemoPlaced& a, const StageMemoPlaced& b) { return a.address < b.address; });
    entry.order.reserve(placed.size());
    for (const auto& item : placed) entry.order.push_back(item.nested ? walkRoots + item.slot : item.slot);
    return std::nullopt;
}

std::optional<Counter> stageMemoShapeChained(std::span<const std::uint32_t> userData, const DirectResolvedStage& stage, const DirectReads& reads, std::uint32_t enabledKinds, StageMemoPattern& pattern, StageMemoEntry& entry, std::vector<StageMemoWord>& words, std::vector<StageMemoPlaced>& placed, std::vector<StageMemoGrant>* grants) {
    const auto& use = stage.capture->userDataUse;
    const auto& forwards = *entry.forwards;
    constexpr std::uint32_t None = 0xffffffffu;
    constexpr std::uint32_t Conflict = 0xfffffffeu;
    enum class Formula : std::uint8_t { User, Chained, Absolute };
    struct Group {
        std::uint64_t address = 0;
        std::uint64_t offset = 0;
        std::uint32_t value = 0;
        std::uint32_t first = 0;
        std::uint32_t begin = 0;
        std::uint32_t end = 0;
        std::uint32_t root = None;
        std::uint32_t parent = None;
        std::int32_t origin = -1;
        Formula formula = Formula::Absolute;
    };
    struct Scratch {
        std::vector<std::pair<std::uint64_t, std::uint32_t>> byAddress;
        std::vector<Group> groups;
        std::vector<std::uint32_t> readGroup;
        std::vector<std::uint32_t> order;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> nodeGroups;
    };
    struct ScratchStorage {};
    auto& scratch = HostThreadLocal<Scratch, ScratchStorage>();
    auto& byAddress = scratch.byAddress;
    auto& groups = scratch.groups;
    auto& readGroup = scratch.readGroup;
    auto& order = scratch.order;
    auto& nodeGroups = scratch.nodeGroups;
    byAddress.clear();
    for (std::size_t at = 0; at < use.reads.size(); ++at) byAddress.emplace_back(use.reads[at].first, static_cast<std::uint32_t>(at));
    std::sort(byAddress.begin(), byAddress.end());
    groups.clear();
    readGroup.assign(use.reads.size(), None);
    for (std::size_t at = 0; at < byAddress.size(); ++at) {
        if (at == 0 || byAddress[at].first != byAddress[at - 1].first) {
            const auto found = std::lower_bound(reads.words.begin(), reads.words.end(), std::pair<std::uint64_t, std::uint32_t>{byAddress[at].first, 0u});
            if (found == reads.words.end() || found->first != byAddress[at].first) return Counter::StageMemoUntraced;
            Group group;
            group.address = byAddress[at].first;
            group.value = found->second;
            group.first = byAddress[at].second;
            group.begin = static_cast<std::uint32_t>(at);
            groups.push_back(group);
        }
        groups.back().end = static_cast<std::uint32_t>(at + 1u);
        readGroup[byAddress[at].second] = static_cast<std::uint32_t>(groups.size() - 1u);
    }
    if (groups.size() != reads.words.size()) return Counter::StageMemoUntraced;
    nodeGroups.clear();
    for (std::size_t at = 0; at < use.reads.size(); ++at) {
        if (reads.nodes[at] != ShaderRecompiler::SrtNoReadNode) nodeGroups.emplace_back(reads.nodes[at], readGroup[at]);
    }
    std::sort(nodeGroups.begin(), nodeGroups.end());
    nodeGroups.erase(std::unique(nodeGroups.begin(), nodeGroups.end()), nodeGroups.end());
    const auto groupOfNode = [&](std::uint32_t node) {
        const auto found = std::lower_bound(nodeGroups.begin(), nodeGroups.end(), std::pair<std::uint32_t, std::uint32_t>{node, 0u});
        if (found == nodeGroups.end() || found->first != node) return None;
        if (std::next(found) != nodeGroups.end() && std::next(found)->first == node) return Conflict;
        return found->second;
    };
    const auto parentOf = [&](std::uint32_t node) {
        const auto found = std::lower_bound(forwards.bases.begin(), forwards.bases.end(), node, [](const ShaderRecompiler::SideReadBase& base, std::uint32_t child) { return base.child < child; });
        return found != forwards.bases.end() && found->child == node ? found->parent : None;
    };
    order.resize(groups.size());
    for (std::size_t at = 0; at < order.size(); ++at) order[at] = static_cast<std::uint32_t>(at);
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) { return groups[a].first < groups[b].first; });
    bool chained = false;
    for (const auto index : order) {
        auto& group = groups[index];
        for (auto at = group.begin; at < group.end; ++at) {
            const auto read = byAddress[at].second;
            const auto origin = use.reads[read].second;
            auto formula = Formula::Absolute;
            std::uint64_t offset = 0;
            std::uint32_t parent = None;
            if (origin >= 0) {
                if (static_cast<std::size_t>(origin) >= userData.size()) return Counter::StageMemoUntraced;
                formula = Formula::User;
                offset = group.address - (static_cast<std::uint64_t>(userData[static_cast<std::size_t>(origin)]) & ~std::uint64_t{3});
            } else if (const auto node = reads.nodes[read]; node != ShaderRecompiler::SrtNoReadNode && groupOfNode(node) == index) {
                const auto parentNode = parentOf(node);
                const auto parentGroup = parentNode != None ? groupOfNode(parentNode) : None;
                if (parentGroup < groups.size() && groups[parentGroup].root != None && groups[parentGroup].root <= 0xffffu) {
                    formula = Formula::Chained;
                    parent = parentGroup;
                    offset = group.address - (static_cast<std::uint64_t>(groups[parentGroup].value) & ~std::uint64_t{3});
                }
            }
            if (at == group.begin) {
                group.formula = formula;
                group.offset = offset;
                group.parent = parent;
                group.origin = origin;
            } else if (formula != group.formula || offset != group.offset || parent != group.parent || (formula == Formula::User && origin != group.origin)) {
                return Counter::StageMemoChainConflict;
            }
        }
        if (group.formula == Formula::Absolute) {
            placed.push_back({group.address, static_cast<std::uint32_t>(entry.nested.size()), true});
            entry.nested.emplace_back(group.address, group.value);
            continue;
        }
        placed.push_back({group.address, static_cast<std::uint32_t>(pattern.roots.size() - pattern.decodeRoots), false});
        if (group.formula == Formula::User) {
            pattern.roots.push_back({group.offset, static_cast<std::uint16_t>(group.origin), 1, 1, 0});
        } else {
            pattern.roots.push_back({group.offset, static_cast<std::uint16_t>(groups[group.parent].root), 1, 1, 1});
            chained = true;
        }
        group.root = static_cast<std::uint32_t>(entry.rootValues.size());
        entry.rootValues.push_back(group.value);
        words.push_back({});
    }
    if (chained) pattern.patchKinds |= 1u << StageMemoBaseKind;
    pattern.arm |= 1u << StageMemoBaseKind;
    const auto& readForwards = forwards.chainedReads;
    const bool walkable = (enabledKinds & StageMemoWalkKinds) != 0u && (!readForwards.empty() || !forwards.children.empty());
    if (grants != nullptr) {
        grants->assign(entry.rootValues.size(), StageMemoGrant::Granted);
        for (std::size_t at = 0; at < entry.decodeWords; ++at) {
            if (words[at].key == 0xffffffffu) (*grants)[at] = StageMemoGrant::DecodeKey;
        }
        const auto walkReason = (enabledKinds & StageMemoWalkKinds) == 0u ? StageMemoGrant::KindOff : walkable ? StageMemoGrant::Granted : StageMemoGrant::NoForwards;
        for (std::size_t at = entry.decodeWords; at < grants->size(); ++at) (*grants)[at] = walkReason;
    }
    if (!walkable) return std::nullopt;
    const bool merging = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoPatchMerged);
    for (const auto index : order) {
        const auto& group = groups[index];
        if (group.root == None) continue;
        std::uint32_t key = 0;
        std::uint32_t kinds = 0;
        std::uint8_t kind = 0;
        bool pinned = false;
        auto cause = StageMemoGrant::Granted;
        const auto pin = [&](StageMemoGrant why) {
            if (!pinned) cause = why;
            pinned = true;
        };
        for (auto at = group.begin; at < group.end; ++at) {
            const auto node = reads.nodes[byAddress[at].second];
            if (node == ShaderRecompiler::SrtNoReadNode) {
                pin(StageMemoGrant::NodeForwards);
                continue;
            }
            bool chainedChild = false;
            const auto [firstChild, lastChild] = std::equal_range(forwards.children.begin(), forwards.children.end(), node, StageMemoByParent{});
            for (auto child = firstChild; child != lastChild; ++child) {
                const auto childGroup = groupOfNode(child->child);
                if (childGroup == None) continue;
                if (childGroup < groups.size() && groups[childGroup].formula == Formula::Chained && groups[childGroup].parent == index) chainedChild = true;
                else pin(StageMemoGrant::BaseAbsolute);
            }
            const auto [first, last] = std::equal_range(readForwards.begin(), readForwards.end(), node, StageMemoByNode{});
            if (first == last && !chainedChild) {
                pin(StageMemoGrant::NodeForwards);
                continue;
            }
            for (auto forward = first; forward != last; ++forward) {
                const auto forwardKind = static_cast<std::uint8_t>(forward->kind);
                if (!merging && kind != 0u && kind != forwardKind) pin(StageMemoGrant::Mixed);
                kind = forwardKind;
                kinds |= 1u << forwardKind;
                key |= forward->keyMask;
            }
            if (chainedChild) kinds |= 1u << StageMemoBaseKind;
        }
        const auto wordIndex = group.root;
        const auto note = [&](StageMemoGrant why) {
            if (grants != nullptr && wordIndex < grants->size()) (*grants)[wordIndex] = why;
        };
        if (pinned || kinds == 0u || (kinds & ~enabledKinds) != 0u || key == 0xffffffffu) {
            note(pinned ? cause : kinds == 0u ? StageMemoGrant::NodeForwards : (kinds & ~enabledKinds) != 0u ? StageMemoGrant::KindOff : StageMemoGrant::AllKey);
            continue;
        }
        const auto value = entry.rootValues[wordIndex];
        const auto patchMask = ~key;
        if ((kinds & StageMemoGuardedKinds) != 0u && (value & patchMask) == 0u) {
            note(StageMemoGrant::GuardedZero);
            continue;
        }
        bool fits = true;
        auto why = StageMemoGrant::Granted;
        const auto patchesBefore = entry.patches.size();
        for (auto at = group.begin; fits && at < group.end; ++at) {
            const auto [first, last] = std::equal_range(readForwards.begin(), readForwards.end(), reads.nodes[byAddress[at].second], StageMemoByNode{});
            for (auto forward = first; fits && forward != last; ++forward) {
                const bool held = forward->binding < stage.side.bindings.size() && forward->position < stage.side.bindings[forward->binding].size();
                fits = held && ((stage.side.bindings[forward->binding][forward->position] ^ value) & patchMask) == 0u;
                if (!held) why = StageMemoGrant::NoPosition;
                else if (!fits) why = StageMemoGrant::NotVerbatim;
                if (fits) entry.patches.push_back({wordIndex, forward->binding, forward->position, patchMask});
            }
        }
        if (!fits) {
            entry.patches.resize(patchesBefore);
            note(why);
            continue;
        }
        note(StageMemoGrant::Granted);
        const bool mergedWord = std::popcount(kinds & StageMemoSideKinds) > 1;
        words[wordIndex] = {key, stageMemoWordKind(kinds), static_cast<std::uint8_t>(mergedWord ? 1u : 0u)};
        pattern.patchKinds |= kinds | (mergedWord ? 1u << StageMemoMergedKind : 0u);
    }
    return std::nullopt;
}

std::optional<Counter> stageMemoShape(bool lean, const DrawProgram& program, std::span<const std::uint32_t> userData, const DirectResolvedStage& stage, const DirectReads& reads, const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& info, const std::vector<Graphics::DecodeRead>& decodeReads, StageMemoPattern& pattern, StageMemoEntry& entry, std::vector<StageMemoGrant>* grants) {
    const auto& use = stage.capture->userDataUse;
    const auto enabledKinds = stageMemoPatchKinds();
    std::vector<StageMemoWord> words;
    pattern.userWords = static_cast<std::uint32_t>(userData.size());
    const auto addresses = entry.forwards->addressWords;
    auto relocatable = ~use.asValue | addresses;
    std::uint64_t pinned = 0;
    if (!decodeReads.empty()) {
        if (!info.has_value() || !info->fetchEmbedded || decodeReads.size() != 2u * static_cast<std::size_t>(info->resourcesNum)) return Counter::StageMemoUntraced;
        const auto hidden = 8u - program.firstUserSgpr;
        const std::size_t attribWord = info->fetchAttribReg + hidden;
        const std::size_t bufferWord = info->fetchBufferReg + hidden;
        const auto tracked = static_cast<std::size_t>(ShaderRecompiler::SrtUserDataUse::TrackedWords);
        if (attribWord + 1u >= userData.size() || bufferWord + 1u >= userData.size() || attribWord + 1u >= tracked || bufferWord + 1u >= tracked) return Counter::StageMemoUntraced;
        for (std::size_t read = 0; read < decodeReads.size(); ++read) {
            const auto& decoded = decodeReads[read];
            const bool sharp = (read & 1u) != 0;
            if (decoded.size != (sharp ? 16u : 4u)) return Counter::StageMemoUntraced;
            const auto word = sharp ? bufferWord : attribWord;
            pattern.roots.push_back({decoded.address - userData[word], static_cast<std::uint16_t>(word), 0, static_cast<std::uint8_t>(sharp ? 4 : 1)});
            for (std::uint32_t at = 0; at < decoded.size; at += sizeof(std::uint32_t)) {
                std::uint32_t value = 0;
                std::memcpy(&value, decoded.data.data() + at, sizeof(value));
                entry.rootValues.push_back(value);
                StageMemoWord word;
                if (sharp && at < 12u && (enabledKinds & (1u << StageMemoVertexKind)) != 0u) {
                    word = {at == 4u ? 0xffff0000u : 0u, StageMemoVertexKind};
                    pattern.patchKinds |= 1u << StageMemoVertexKind;
                }
                words.push_back(word);
            }
        }
        for (const auto word : {attribWord, bufferWord}) {
            if (((use.asValue >> word) & 1u) == 0) relocatable |= std::uint64_t{1} << word;
            pinned |= std::uint64_t{1} << (word + 1u);
        }
    }
    relocatable &= ~pinned;
    pattern.decodeRoots = static_cast<std::uint32_t>(pattern.roots.size());
    entry.decodeWords = static_cast<std::uint32_t>(entry.rootValues.size());
    using Placed = StageMemoPlaced;
    std::vector<Placed> placed;
    placed.reserve(use.reads.size());
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoChained) && reads.nodes.size() == use.reads.size() && entry.forwards->tape == ShaderRecompiler::SideTape::Traced) {
        if (const auto refused = stageMemoShapeChained(userData, stage, reads, enabledKinds, pattern, entry, words, placed, grants)) return refused;
        return stageMemoFinish(userData, use, addresses, relocatable, pattern, entry, words, placed);
    }
    const auto placeRead = [&](std::uint64_t address, std::int32_t origin) {
        const auto found = std::lower_bound(reads.words.begin(), reads.words.end(), std::pair<std::uint64_t, std::uint32_t>{address, 0u});
        if (found == reads.words.end() || found->first != address) return false;
        if (origin >= 0) {
            if (static_cast<std::size_t>(origin) >= userData.size()) return false;
            placed.push_back({address, static_cast<std::uint32_t>(pattern.roots.size() - pattern.decodeRoots), false});
            pattern.roots.push_back({address - (static_cast<std::uint64_t>(userData[static_cast<std::size_t>(origin)]) & ~std::uint64_t{3}), static_cast<std::uint16_t>(origin), 1, 1});
            entry.rootValues.push_back(found->second);
            words.push_back({});
        } else {
            placed.push_back({address, static_cast<std::uint32_t>(entry.nested.size()), true});
            entry.nested.emplace_back(address, found->second);
        }
        return true;
    };
    struct ByAddressStorage {};
    std::vector<std::pair<std::uint64_t, std::uint32_t>> none;
    auto& byAddress = lean ? HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint32_t>>, ByAddressStorage>() : none;
    if (!lean) {
        for (const auto& [address, origin] : use.reads) {
            if (std::ranges::any_of(placed, [&](const Placed& known) { return known.address == address; })) continue;
            if (!placeRead(address, origin)) return Counter::StageMemoUntraced;
        }
    } else {
        byAddress.clear();
        for (std::size_t at = 0; at < use.reads.size(); ++at) byAddress.emplace_back(use.reads[at].first, static_cast<std::uint32_t>(at));
        std::sort(byAddress.begin(), byAddress.end());
        struct FirstsStorage {};
        auto& firsts = HostThreadLocal<std::vector<std::uint32_t>, FirstsStorage>();
        firsts.clear();
        for (std::size_t at = 0; at < byAddress.size(); ++at) {
            if (at == 0 || byAddress[at].first != byAddress[at - 1].first) firsts.push_back(byAddress[at].second);
        }
        std::sort(firsts.begin(), firsts.end());
        for (const auto at : firsts) {
            if (!placeRead(use.reads[at].first, use.reads[at].second)) return Counter::StageMemoUntraced;
        }
    }
    if (placed.size() != reads.words.size()) return Counter::StageMemoUntraced;
    const auto& readForwards = entry.forwards->reads;
    if (grants != nullptr) {
        grants->assign(entry.rootValues.size(), StageMemoGrant::Granted);
        for (std::size_t at = 0; at < entry.decodeWords; ++at) {
            if (words[at].key == 0xffffffffu) (*grants)[at] = StageMemoGrant::DecodeKey;
        }
        const auto walkReason = (enabledKinds & StageMemoWalkKinds) == 0u ? StageMemoGrant::KindOff : readForwards.empty() ? StageMemoGrant::NoForwards : reads.nodes.size() != use.reads.size() ? StageMemoGrant::NoNodes : StageMemoGrant::Granted;
        for (std::size_t at = entry.decodeWords; at < grants->size(); ++at) (*grants)[at] = walkReason;
    }
    if ((enabledKinds & StageMemoWalkKinds) != 0u && !readForwards.empty() && reads.nodes.size() == use.reads.size()) {
        const bool merging = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoPatchMerged);
        struct Granted {
            std::uint64_t address;
            std::uint32_t key;
            std::uint8_t kind;
            bool pinned;
            StageMemoGrant cause = StageMemoGrant::Granted;
            std::uint32_t kinds = 0;
        };
        struct ByNode {
            bool operator()(const ShaderRecompiler::SideReadForward& forward, std::uint32_t node) const { return forward.node < node; }
            bool operator()(std::uint32_t node, const ShaderRecompiler::SideReadForward& forward) const { return node < forward.node; }
        };
        const auto pin = [](Granted& slot, StageMemoGrant cause) {
            if (!slot.pinned) slot.cause = cause;
            slot.pinned = true;
        };
        const auto grantRead = [&](Granted& slot, std::size_t at) {
            const auto node = reads.nodes[at];
            const auto [first, last] = std::equal_range(readForwards.begin(), readForwards.end(), node, ByNode{});
            if (node == ShaderRecompiler::SrtNoReadNode || first == last) {
                pin(slot, StageMemoGrant::NodeForwards);
                return;
            }
            for (auto forward = first; forward != last; ++forward) {
                const auto kind = static_cast<std::uint8_t>(forward->kind);
                if (!merging && slot.kind != 0u && slot.kind != kind) pin(slot, StageMemoGrant::Mixed);
                slot.kind = kind;
                slot.kinds |= 1u << kind;
                slot.key |= forward->keyMask;
            }
        };
        std::vector<Granted> granted;
        if (!lean) {
            for (std::size_t at = 0; at < use.reads.size(); ++at) {
                const auto address = use.reads[at].first;
                auto slot = std::find_if(granted.begin(), granted.end(), [&](const Granted& known) { return known.address == address; });
                if (slot == granted.end()) slot = granted.insert(granted.end(), Granted{address, 0u, 0u, false});
                grantRead(*slot, at);
            }
        }
        const auto groupOf = [&](std::uint64_t address) {
            const auto begin = std::lower_bound(byAddress.begin(), byAddress.end(), std::pair<std::uint64_t, std::uint32_t>{address, 0u});
            auto end = begin;
            while (end != byAddress.end() && end->first == address) ++end;
            return std::pair{begin, end};
        };
        for (const auto& item : placed) {
            if (item.nested) continue;
            Granted leanSlot{item.address, 0u, 0u, false};
            Granted* slot = nullptr;
            if (lean) {
                const auto [begin, end] = groupOf(item.address);
                for (auto read = begin; read != end; ++read) grantRead(leanSlot, read->second);
                slot = begin != end ? &leanSlot : nullptr;
            } else {
                const auto found = std::find_if(granted.begin(), granted.end(), [&](const Granted& known) { return known.address == item.address; });
                slot = found != granted.end() ? &*found : nullptr;
            }
            const auto wordIndex = entry.decodeWords + item.slot;
            const auto note = [&](StageMemoGrant why) {
                if (grants != nullptr && wordIndex < grants->size()) (*grants)[wordIndex] = why;
            };
            if (slot == nullptr || slot->pinned || slot->kinds == 0u || (slot->kinds & ~enabledKinds) != 0u || slot->key == 0xffffffffu) {
                note(slot == nullptr || slot->kinds == 0u ? StageMemoGrant::NodeForwards : slot->pinned ? slot->cause : (slot->kinds & ~enabledKinds) != 0u ? StageMemoGrant::KindOff : StageMemoGrant::AllKey);
                continue;
            }
            const auto value = entry.rootValues[wordIndex];
            const auto patchMask = ~slot->key;
            const bool guarded = (slot->kinds & StageMemoGuardedKinds) != 0u;
            if (guarded && (value & patchMask) == 0u) {
                note(StageMemoGrant::GuardedZero);
                continue;
            }
            bool fits = true;
            auto why = StageMemoGrant::Granted;
            const auto patchesBefore = entry.patches.size();
            const auto fitRead = [&](std::size_t at) {
                const auto [first, last] = std::equal_range(readForwards.begin(), readForwards.end(), reads.nodes[at], ByNode{});
                for (auto forward = first; fits && forward != last; ++forward) {
                    const bool held = forward->binding < stage.side.bindings.size() && forward->position < stage.side.bindings[forward->binding].size();
                    fits = held && ((stage.side.bindings[forward->binding][forward->position] ^ value) & patchMask) == 0u;
                    if (!held) why = StageMemoGrant::NoPosition;
                    else if (!fits) why = StageMemoGrant::NotVerbatim;
                    if (fits) entry.patches.push_back({wordIndex, forward->binding, forward->position, patchMask});
                }
            };
            if (lean) {
                const auto [begin, end] = groupOf(item.address);
                for (auto read = begin; fits && read != end; ++read) fitRead(read->second);
            } else {
                for (std::size_t at = 0; fits && at < use.reads.size(); ++at) {
                    if (use.reads[at].first != item.address) continue;
                    fitRead(at);
                }
            }
            if (!fits) {
                entry.patches.resize(patchesBefore);
                note(why);
                continue;
            }
            note(StageMemoGrant::Granted);
            const bool mergedWord = std::popcount(slot->kinds & StageMemoSideKinds) > 1;
            words[wordIndex] = {slot->key, stageMemoWordKind(slot->kinds), static_cast<std::uint8_t>(mergedWord ? 1u : 0u)};
            pattern.patchKinds |= slot->kinds | (mergedWord ? 1u << StageMemoMergedKind : 0u);
        }
    }
    return stageMemoFinish(userData, use, addresses, relocatable, pattern, entry, words, placed);
}

const char* stageMemoStoreDifference(const std::optional<Counter>& refused, const StageMemoPattern& pattern, const StageMemoEntry& entry, const StageMemoPattern& todayPattern, const StageMemoEntry& today) {
    if (refused.has_value()) return "the refusal";
    if (!(pattern == todayPattern)) return "the pattern";
    if (entry.decodeWords != today.decodeWords || entry.rootValues != today.rootValues) return "the root values";
    if (entry.nested != today.nested) return "the nested reads";
    if (entry.order != today.order) return "the order";
    if (entry.patches.size() != today.patches.size()) return "the patches";
    for (std::size_t at = 0; at < entry.patches.size(); ++at) {
        const auto& a = entry.patches[at];
        const auto& b = today.patches[at];
        if (a.word != b.word || a.binding != b.binding || a.position != b.position || a.mask != b.mask) return "the patches";
    }
    return nullptr;
}

std::optional<Counter> rememberStage(const LeanDrawJob& job, const DrawProgram& program, std::size_t index, std::uint32_t cursor, const InvocationSlot& slot, const ShaderRecompiler::PreparedShaderInvocation& invocation, const DirectReads& reads, DirectStages& resolved, std::uint64_t& storedKey) {
    auto& stage = resolved.stages[index];
    const std::span<const std::uint32_t> userData = program.userData;
    if (userData.size() > 0xffffu) return Counter::StageMemoUntraced;
    StageMemoPattern pattern;
    StageMemoEntry entry;
    entry.forwards = stageForwards(stage.shared, invocation, *stage.capture);
    if (entry.forwards == nullptr) return Counter::StageMemoUntraced;
    const bool lean = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoLeanStores);
    const bool diag = stageMemoDiag() && stageMemoPatchKinds() != 0u;
    struct GrantsStorage {};
    auto* const grants = diag ? &HostThreadLocal<std::vector<StageMemoGrant>, GrantsStorage>() : nullptr;
    const auto& info = resolved.vertexInfos[index];
    const auto& decodeReads = resolved.decodeReads[index];
    if (const auto refused = stageMemoShape(lean, program, userData, stage, reads, info, decodeReads, pattern, entry, grants)) return refused;
    if (lean) {
        DirectDraws::Add(Counter::StageMemoLeanStores);
        if (const auto every = stageMemoVerifyEvery(); every != 0) {
            struct OrdinalStorage {};
            auto& ordinal = HostThreadLocal<std::uint64_t, OrdinalStorage>();
            if (++ordinal % every == 0) {
                StageMemoPattern todayPattern;
                StageMemoEntry today;
                today.forwards = entry.forwards;
                const auto refused = stageMemoShape(false, program, userData, stage, reads, info, decodeReads, todayPattern, today, nullptr);
                DirectDraws::Add(Counter::StageMemoStoreVerifyChecked);
                if (const char* what = stageMemoStoreDifference(refused, pattern, entry, todayPattern, today); what != nullptr) {
                    DirectDraws::Add(Counter::StageMemoStoreVerifyDiffering);
                    static std::atomic<std::uint32_t> printed{0};
                    if (printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[stage-memo-verify] store differs in %s: stage %zu, %zu reads, %zu roots lean and %zu today\n", what, index, stage.capture->userDataUse.reads.size(), pattern.roots.size(), todayPattern.roots.size());
                }
            }
        }
    }
    const auto key = stageMemoKey(job.shape->decode.get(), index);
    auto& store = stageMemoStore();
    const auto storeKinds = stageMemoPatchKinds();
    pattern.arm |= stageMemoArm(storeKinds);
    std::span<const StageMemoKeyReason> keyReasons;
    if (grants != nullptr) {
        struct ReasonsStorage {};
        auto& held = HostThreadLocal<std::vector<StageMemoKeyReason>, ReasonsStorage>();
        stageMemoKeyReasons(pattern, entry, userData, reads, stage.capture->userDataUse.reads, *grants, held);
        keyReasons = held;
    }
    const auto placedWay = store.Way(key, pattern, stageMemoWayLimit(), stageMemoForeignKinds(storeKinds));
    if (!placedWay.has_value()) {
        if (grants != nullptr) stageMemoGrantCensus(key, 0, entry.rootValues, *grants, keyReasons, true);
        return Counter::StageMemoPatternsFull;
    }
    const auto way = *placedWay;
    if (grants != nullptr) stageMemoGrantCensus(key, way, entry.rootValues, *grants, keyReasons, false);
    entry.decode = job.shape->decode;
    entry.stage = static_cast<std::uint32_t>(index);
    entry.way = static_cast<std::uint32_t>(way);
    entry.cursor = cursor;
    entry.pushOffset = stage.pushOffset;
    entry.tag = resolved.tag;
    if (index != 0) entry.previous = resolved.stages[index - 1].Shared();
    entry.indexAddress = job.parameters.indexAddress;
    entry.indexCount = job.parameters.indexCount;
    entry.indexSize = job.parameters.indexSize;
    entry.instanceCount = job.parameters.instanceCount;
    entry.variant = slot.variant;
    entry.convertedSamples = slot.convertedSamples;
    entry.layerRowSamples = slot.layerRowSamples;
    entry.userData.assign(userData.begin(), userData.end());
    entry.capture = stage.capture;
    entry.shared = stage.shared;
    const bool viewing = resolved.viewing;
    const bool moving = lean && viewing;
    if (moving) {
        entry.side = std::move(stage.side);
        entry.vertexInfo = std::move(resolved.vertexInfos[index]);
        entry.decodeReads = std::move(resolved.decodeReads[index]);
    } else {
        entry.side = stage.side;
        entry.vertexInfo = info;
        entry.decodeReads = decodeReads;
    }
    if (viewing) {
        struct ViewSampleStorage {};
        auto& sampled = HostThreadLocal<std::uint64_t, ViewSampleStorage>();
        const bool timed = (sampled++ & 7u) == 0;
        const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        viewStageEntry(entry, pattern, userData, stage.stage, resolved.stages.size());
        if (timed) {
            DirectDraws::Add(Counter::StageViewBuildNanoseconds, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - began).count()));
            DirectDraws::Add(Counter::StageViewBuildsTimed);
        }
    }
    const auto contentKey = stageMemoContentKey(key, way, pattern, userData, entry.rootValues, job.parameters, cursor, entry.previous.get());
    auto made = std::make_shared<const StageMemoEntry>(std::move(entry));
    if (auditOn()) {
        const auto& memo = HostThreadLocal<StageMemo, StageMemoStorage>();
        if (memo.volatileKey == contentKey && memo.volatileUntil != 0) made->trust.volatileUntil.store(memo.volatileUntil, std::memory_order_relaxed);
    }
    if (moving) {
        stage.sideView = &made->side;
        stage.decodeView = &made->decodeReads;
        stage.vertexView = &made->vertexInfo;
        stage.memo = made.get();
        stage.pin = made;
    }
    store.Insert(contentKey, std::move(made));
    storedKey = contentKey;
    return Counter::StageMemoStored;
}

bool stageMemoAdmitted(StageMemo& memo) {
    using Graphics::RuntimeSwitch;
    using Graphics::RuntimeSwitches::Set;
    using Cause = StageMemoStore::MissCause;
    auto& store = stageMemoStore();
    const bool first = Set(RuntimeSwitch::StageMemoAdmitFirst);
    const bool third = Set(RuntimeSwitch::StageMemoAdmitThird);
    const bool ways = Set(RuntimeSwitch::StageMemoAdmitWays);
    memo.forced = false;
    if (memo.tried == 0) {
        DirectDraws::Add(Counter::StageMemoNoEntryUntried);
    } else {
        auto cause = Cause::First;
        if (ways) {
            for (std::uint32_t at = 0; at < memo.missCount; ++at) cause = std::max(cause, store.Cause(memo.missKeys[at], third, first));
        } else {
            cause = store.Cause(memo.missKey, third, first);
        }
        DirectDraws::Add(static_cast<Counter>(static_cast<std::size_t>(Counter::StageMemoNoEntryFirst) + static_cast<std::size_t>(cause)));
    }
    if (memo.noEntryPlain && memo.noEntryPatterned) DirectDraws::Add(Counter::StageMemoNoEntryBothWays);
    if (first) return true;
    if (!ways) return store.Admit(memo.missKey, third);
    if (memo.tried == 0) {
        if (memo.armWays >= stageMemoWayLimit()) return false;
        DirectDraws::Add(Counter::StageMemoAdmitUntried);
        return true;
    }
    bool admitted = false;
    for (std::uint32_t at = 0; at < memo.missCount; ++at) admitted = store.Admit(memo.missKeys[at], third) || admitted;
    if (memo.refusals.empty()) memo.refusals.assign(StageMemo::RefusalSlots, 0u);
    auto& refused = memo.refusals[static_cast<std::size_t>((memo.stageKey ^ (memo.stageKey >> 17u)) % StageMemo::RefusalSlots)];
    if (admitted) {
        refused = 0;
        return true;
    }
    if (++refused < 64u) return false;
    refused = 0;
    memo.forced = true;
    DirectDraws::Add(Counter::StageMemoAdmitRefusals);
    return true;
}

bool stageMemoLookedUp(const StageMemo& memo, std::uint64_t key) {
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoAdmitWays)) return key == memo.missKey;
    return std::find(memo.missKeys.begin(), memo.missKeys.begin() + memo.missCount, key) != memo.missKeys.begin() + memo.missCount;
}

std::pair<std::uint64_t, std::uint32_t>& stageMemoConflictSlot(StageMemo& memo, std::uint64_t key) {
    if (memo.conflicts.empty()) memo.conflicts.assign(StageMemo::RefusalSlots, {});
    return memo.conflicts[static_cast<std::size_t>((key ^ (key >> 17u)) % StageMemo::RefusalSlots)];
}

bool stageMemoConflictWaits(StageMemo& memo, std::uint64_t key) {
    if (memo.conflicts.empty() || !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoChained)) return false;
    auto& held = stageMemoConflictSlot(memo, key);
    if (held.first != key || held.second == 0u) return false;
    --held.second;
    return true;
}

struct StageMemoKept {
    DirectResolvedStage stage;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertexInfo;
    std::vector<Graphics::DecodeRead> decodeReads;
    bool patched = false;
    bool merged = false;
    bool relocated = false;
};

const char* stageMemoDifference(const StageMemoKept& kept, const DirectResolvedStage& stage, const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertexInfo, const std::vector<Graphics::DecodeRead>& decodeReads, std::size_t count) {
    if (kept.stage.Shared().get() != stage.Shared().get()) return "the shared stage";
    if (kept.stage.pushOffset != stage.pushOffset) return "the push offset";
    if (kept.stage.Ranges() != stage.Ranges()) return "the read ranges";
    if (kept.stage.Values() != stage.Values()) return "the read values";
    const auto& a = kept.stage.Side();
    const auto& b = stage.Side();
    if (a.userData != b.userData || a.flattenedSrt != b.flattenedSrt || a.bindings != b.bindings || a.pushWords != b.pushWords) return "the side words";
    if (a.vertexAttributes.size() != b.vertexAttributes.size()) return "the side vertex attributes";
    for (std::size_t at = 0; at < a.vertexAttributes.size(); ++at) {
        const auto& x = a.vertexAttributes[at];
        const auto& y = b.vertexAttributes[at];
        if (x.location != y.location || x.components != y.components || x.fetchIndex != y.fetchIndex || x.formatComponents != y.formatComponents || x.resource.fields != y.resource.fields) return "the side vertex attributes";
    }
    const auto& ours = kept.stage.Capture()->snapshot.registeredImages;
    const auto& theirs = stage.Capture()->snapshot.registeredImages;
    if (ours.size() != theirs.size()) return "the registered images";
    for (std::size_t at = 0; at < ours.size(); ++at) {
        if (ours[at].resource != theirs[at].resource || ours[at].record != theirs[at].record || ours[at].address != theirs[at].address || ours[at].backingSerial != theirs[at].backingSerial) return "the registered images";
    }
    if (kept.vertexInfo.has_value() != vertexInfo.has_value()) return "the vertex info";
    if (vertexInfo.has_value()) {
        const auto& x = *kept.vertexInfo;
        const auto& y = *vertexInfo;
        if (x.resourcesNum != y.resourcesNum || x.fetchAttribReg != y.fetchAttribReg || x.fetchBufferReg != y.fetchBufferReg || x.fetchEmbedded != y.fetchEmbedded || x.paClVsOutCntl != y.paClVsOutCntl || x.depthRemapMode != y.depthRemapMode) return "the vertex info";
        for (std::uint32_t at = 0; at < x.resourcesNum && at < x.resources.size(); ++at) {
            if (x.resources[at].fields != y.resources[at].fields || x.resourcesDst[at].registerStart != y.resourcesDst[at].registerStart || x.resourcesDst[at].registersNum != y.resourcesDst[at].registersNum || x.resourcesDst[at].attrId != y.resourcesDst[at].attrId || x.resourcesDst[at].fetchIndex != y.resourcesDst[at].fetchIndex) return "the vertex info";
        }
    }
    if (kept.decodeReads.size() != decodeReads.size()) return "the decode reads";
    for (std::size_t at = 0; at < decodeReads.size(); ++at) {
        if (kept.decodeReads[at].address != decodeReads[at].address || kept.decodeReads[at].size != decodeReads[at].size || kept.decodeReads[at].data != decodeReads[at].data) return "the decode reads";
    }
    if (kept.stage.memo == nullptr || !static_cast<const StageMemoEntry*>(kept.stage.memo)->viewable) return nullptr;
    const auto& entry = *static_cast<const StageMemoEntry*>(kept.stage.memo);
    if (kept.stage.sideView != nullptr) {
        const Graphics::CompiledShader fresh{stage.stage, stage.Shared()->donor.get(), stage.pushOffset, &stage.Capture()->snapshot.registeredImages, &stage.Side()};
        const std::array<Graphics::CompiledShader, 2> buffersOnly{Graphics::CompiledShader{stage.stage, nullptr, 0}, fresh};
        if (QueuedDrawReadRanges(Pm4::DrawParameters{}, buffersOnly) != entry.bufferReads) return "the entry's buffer reads";
        const auto all = QueuedDrawReadRanges(Pm4::DrawParameters{}, std::span<const Graphics::CompiledShader>(&fresh, 1));
        if (all.size() < entry.bufferReads.size() || !std::equal(all.begin() + static_cast<std::ptrdiff_t>(entry.bufferReads.size()), all.end(), entry.attributeReads.begin(), entry.attributeReads.end())) return "the entry's attribute reads";
    }
    if (kept.stage.vertexView == nullptr) return nullptr;
    std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>> infos(count);
    if (entry.stage < count) infos[entry.stage] = vertexInfo;
    std::vector<std::uint32_t> layout;
    std::vector<std::uint32_t> inputs;
    DrawVertexLayout(infos, layout, inputs);
    if (layout != entry.layoutWords || inputs != entry.vertexInputs) return "the entry's vertex layout";
    return nullptr;
}

}

namespace {

struct RefreshedDecode {
    std::shared_ptr<const DrawDecode> source;
    std::optional<DrawDecode> decode;
};
struct RefreshedDecodeStorage {};

std::shared_ptr<const DrawDecode> refreshedDecode(const std::shared_ptr<const DrawDecode>& from, const DrawUserWords& words) {
    auto& scratch = HostThreadLocal<RefreshedDecode, RefreshedDecodeStorage>();
    if (scratch.source != from) {
        if (scratch.decode.has_value()) *scratch.decode = *from;
        else scratch.decode.emplace(*from);
        scratch.source = from;
    }
    for (auto& program : scratch.decode->programs) {
        ReadUserWordsFrom(words, program);
        if (program.merged) InitializeMergedFrom(words, program, program.mergedPointer, program.mergedPointerRequired);
    }
    return std::shared_ptr<const DrawDecode>(std::shared_ptr<void>(), &*scratch.decode);
}

struct ShapePrologueSlot {
    std::shared_ptr<const DrawDecode> decode;
    bool direct = false;
    bool plain = false;
    std::vector<ShaderRecompiler::MemoryRegion> memory;
};

struct ShapePrologueCache {
    static constexpr std::size_t Slots = 256;
    std::array<ShapePrologueSlot, Slots> slots;
    std::uint64_t device = 0;
};

struct ShapePrologueStorage {};

bool directShapeStatic(const DrawDecode& decode) {
    const auto& state = decode.state;
    if (state.quadList || state.rectList || state.stages.path != Graphics::ShaderPath::Vertex || state.stages.mesh || state.stages.tessellation || state.stages.tessellationGeometry || state.tessellationCapture != nullptr) return false;
    if (decode.programs.empty() || decode.programs.size() > DirectClassKey::MaxStages || decode.roles.size() != decode.programs.size()) return false;
    for (std::size_t i = 0; i < decode.programs.size(); ++i) {
        const auto stage = decode.programs[i].binary.stage;
        if ((stage != Stage::Vertex && stage != Stage::Fragment) || decode.roles[i] != (stage == Stage::Fragment ? Role::Fragment : Role::Main)) return false;
    }
    return true;
}

bool plainUserWords(const DrawDecode& decode) {
    return std::ranges::none_of(decode.programs, [](const DrawProgram& program) { return program.merged || ((program.binary.stage == Stage::Vertex || program.binary.stage == Stage::Fragment) && program.snapshot->header.empty() && program.snapshot->userDataTailRegister); });
}

const ShapePrologueSlot& shapePrologue(const LeanDrawJob& job) {
    auto& cache = HostThreadLocal<ShapePrologueCache, ShapePrologueStorage>();
    if (cache.device != job.deviceSerial) {
        for (auto& slot : cache.slots) slot = {};
        cache.device = job.deviceSerial;
    }
    const auto* const decode = job.shape->decode.get();
    auto& slot = cache.slots[mixedSlot(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(decode)), 0, ShapePrologueCache::Slots)];
    if (slot.decode.get() == decode) {
        DirectDraws::Add(Counter::PrologueCached);
        return slot;
    }
    slot.decode = job.shape->decode;
    slot.direct = directShapeStatic(*decode);
    slot.plain = plainUserWords(*decode);
    slot.memory.clear();
    AppendProgramMemory(slot.memory, decode->programs);
    DirectDraws::Add(Counter::PrologueBuilt);
    if (!slot.plain) DirectDraws::Add(Counter::PrologueToday);
    return slot;
}

bool prologueDirect(const LeanDrawJob& job, const ShapePrologueSlot& slot) {
    if ((job.parameters.indirect && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectGpuIndirect)) || !slot.direct || !Graphics::DrawsRecordable()) return false;
    return std::ranges::none_of(job.shape->decode->programs, [](const DrawProgram& program) { return SkippedShader(program.binary.codeAddress); });
}

void prologueWords(const DrawUserWords& words, const DrawProgram& program, std::vector<std::uint32_t>& out) {
    const auto resources = program.nullPixel ? 0u : words.Value(program.resourceRegister);
    const auto userCount = ((resources >> 1u) & 0x1fu) | (((resources >> 27u) & 1u) << 5u);
    require(userCount <= 32, "graphics user SGPR count exceeds the register bank");
    out.clear();
    for (std::uint32_t i = 0; i < userCount; ++i) out.push_back(words.Value(program.userDataBase + i));
}

constexpr std::uint64_t CapturePageBytes = 0x1000u;

std::optional<Counter> addressedRefusal(const DirectStage& stage) {
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectAddressed)) return Counter::AddressedSwitchOff;
    if (stage.donor->bdaStores) return Counter::AddressedStores;
    return std::nullopt;
}

struct AddressedVerifyStorage {};

bool addressedVerifyDue() {
    const auto every = DirectDraws::AddressedVerifyEvery();
    if (every == 0) return false;
    auto& ordinal = HostThreadLocal<std::uint64_t, AddressedVerifyStorage>();
    return ++ordinal % every == 0;
}

int addressedProgramWord(std::span<const Graphics::GuestMemorySnapshot> program, std::uint64_t address) {
    constexpr std::uint64_t word = sizeof(std::uint32_t);
    if (program.empty() || address + word <= program.front().address || address >= program.back().address + program.back().bytes.size()) return 0;
    const auto next = std::upper_bound(program.begin(), program.end(), address, [](std::uint64_t value, const Graphics::GuestMemorySnapshot& region) { return value < region.address; });
    if (next != program.begin()) {
        const auto& previous = *std::prev(next);
        const auto offset = address - previous.address;
        if (offset < previous.bytes.size()) return previous.bytes.size() - offset < word ? -1 : 1;
    }
    if (next != program.end() && next->address - address < word) return -1;
    return 0;
}

struct AddressedWordsStorage {};
struct AddressedProgramStorage {};
struct AddressedRunsStorage {};

std::optional<Counter> addressedSnapshots(const DirectStages& resolved, DrawRecordHit& hit) {
    auto& pairs = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint32_t>>, AddressedWordsStorage>();
    auto& program = HostThreadLocal<std::vector<Graphics::GuestMemorySnapshot>, AddressedProgramStorage>();
    auto& runs = HostThreadLocal<std::vector<Graphics::GuestMemorySnapshot>, AddressedRunsStorage>();
    pairs.clear();
    program.clear();
    runs.clear();
    for (const auto& stage : resolved.stages) {
        const auto& values = stage.Values();
        std::size_t at = 0;
        for (const auto& [begin, end] : stage.Ranges()) {
            for (auto address = begin; address < end; address += sizeof(std::uint32_t)) {
                if (at == values.size()) return Counter::FallbackWalk;
                pairs.emplace_back(address, values[at++]);
            }
        }
    }
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    for (std::size_t index = 1; index < pairs.size(); ++index) {
        if (pairs[index].first == pairs[index - 1].first) return Counter::FallbackTornReads;
    }
    for (const auto& region : resolved.memory) program.push_back({region.guestAddress, region.bytes});
    std::stable_sort(program.begin(), program.end(), [](const Graphics::GuestMemorySnapshot& left, const Graphics::GuestMemorySnapshot& right) { return left.address < right.address; });
    hit.words.assign(1, {});
    auto& words = hit.words.front();
    words.reserve(pairs.size());
    std::uint64_t runBegin = 0;
    std::size_t runFirst = 0;
    bool open = false;
    const auto close = [&] {
        if (open) runs.push_back({runBegin, std::as_bytes(std::span<const std::uint32_t>(words.data() + runFirst, words.size() - runFirst))});
        open = false;
    };
    for (const auto& [address, value] : pairs) {
        const auto served = addressedProgramWord(program, address);
        if (served < 0) {
            DirectDraws::Add(Counter::AddressedOverlap);
            return Counter::FallbackAddressed;
        }
        if (served > 0) continue;
        if (!open || address != runBegin + (words.size() - runFirst) * sizeof(std::uint32_t) || address % CapturePageBytes == 0) {
            close();
            runBegin = address;
            runFirst = words.size();
            open = true;
        }
        words.push_back(value);
    }
    close();
    const auto count = resolved.stages.size();
    hit.decodeReads.assign(count, {});
    std::size_t decoded = 0;
    for (std::size_t i = 0; i < count; ++i) {
        hit.decodeReads[i] = resolved.DecodeReads(i);
        decoded += hit.decodeReads[i].size();
    }
    auto& snapshots = hit.snapshots;
    snapshots.clear();
    snapshots.reserve(program.size() + runs.size() + decoded);
    std::merge(program.begin(), program.end(), runs.begin(), runs.end(), std::back_inserter(snapshots), [](const Graphics::GuestMemorySnapshot& left, const Graphics::GuestMemorySnapshot& right) { return left.address < right.address; });
    for (const auto& reads : hit.decodeReads) {
        for (const auto& read : reads) snapshots.push_back({read.address, read.Bytes()});
    }
    hit.addressed = true;
    return std::nullopt;
}

void refreshStages(const LeanDrawJob& job, DirectStages& resolved) {
    if (resolved.refreshed) return;
    resolved.decode = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::LeanPrepare) ? refreshedDecode(job.shape->decode, job.words) : LeanRefreshedDecode(*job.shape->decode, job.words);
    resolved.refreshed = true;
    const auto& decode = *resolved.decode;
    resolved.linked.clear();
    for (std::size_t i = 0; i < decode.programs.size(); ++i) {
        const auto& program = decode.programs[i];
        resolved.linked.push_back({decode.roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
    }
    DirectDraws::Add(Counter::PrologueRefreshes);
}

bool layoutFromEntry(DirectStages& resolved) {
    const StageMemoEntry* only = nullptr;
    for (std::size_t i = 0; i < resolved.stages.size(); ++i) {
        const auto& stage = resolved.stages[i];
        if (!resolved.VertexInfo(i).has_value()) continue;
        if (only != nullptr || stage.vertexView == nullptr || stage.memo == nullptr) return false;
        const auto* entry = static_cast<const StageMemoEntry*>(stage.memo);
        if (!entry->viewable || &entry->vertexInfo != stage.vertexView) return false;
        only = entry;
    }
    if (only == nullptr) return false;
    resolved.vertexLayout.assign(only->layoutWords.begin(), only->layoutWords.end());
    resolved.vertexInputs.assign(only->vertexInputs.begin(), only->vertexInputs.end());
    resolved.layoutHash = only->layoutHash;
    resolved.layoutHashed = true;
    DirectDraws::Add(Counter::StageViewLayouts);
    return true;
}

void appendStageRegions(DirectStages& resolved, bool adopted) {
    auto& regions = resolved.regions;
    for (std::size_t i = 0; i < resolved.stages.size(); ++i) {
        if (resolved.stages[i].adopted != adopted) continue;
        for (const auto& read : resolved.DecodeReads(i)) regions.push_back({read.address, read.Bytes()});
    }
    for (const auto& stage : resolved.stages) {
        if (stage.adopted != adopted) continue;
        const auto& values = stage.Values();
        std::size_t at = 0;
        for (const auto& [begin, end] : stage.Ranges()) {
            const auto words = static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
            regions.push_back({begin, std::as_bytes(std::span(values.data() + at, words))});
            at += words;
        }
    }
}

}

std::optional<Counter> Driver::resolveDirectStages(const LeanDrawJob& job, const LeanAnchor& anchor, DirectStages& resolved, DirectStepClock* clock) {
    DrawPipeline::FollowEpoch(job.epoch);
    DrawPipeline::Active() = true;
    GuestMemory::SetCurrentPacket(job.opcode, job.queue);
    const DrawFinisher::AheadScope ahead;
    if (clock != nullptr) clock->Start();
    const auto lap = [&](Step step) {
        if (clock != nullptr) clock->Lap(step);
    };
    resolved.rect.reset();
    resolved.rectControl = nullptr;
    resolved.addressed = false;
    const auto* const prologue = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::ShapePrologue) ? &shapePrologue(job) : nullptr;
    if (prologue == nullptr || !prologueDirect(job, *prologue)) {
        if (const auto shape = DirectShapeRefusal(job)) {
            DirectDraws::Add(*shape);
            return Counter::FallbackShape;
        }
    }
    auto& stores = DirectStores::Get();
    resolved.tag = stores.Follow(job.deviceSerial);
    const auto& shape = *job.shape;
    const auto& device = *anchor.device;
    auto* const memo = stageMemoOn() ? &HostThreadLocal<StageMemo, StageMemoStorage>() : nullptr;
    resolved.viewing = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageViews);
    resolved.layoutHashed = false;
    const bool lazy = prologue != nullptr && prologue->plain && memo != nullptr && !job.parameters.indirect;
    if (prologue != nullptr && prologue->plain && memo != nullptr && job.parameters.indirect) DirectDraws::Add(Counter::PrologueIndirect);
    resolved.refreshed = !lazy;
    resolved.worded = lazy;
    if (lazy) {
        resolved.decode = std::shared_ptr<const DrawDecode>(std::shared_ptr<void>(), shape.decode.get());
        const auto& programs = shape.decode->programs;
        resolved.userWords.resize(programs.size());
        for (std::size_t i = 0; i < programs.size(); ++i) prologueWords(job.words, programs[i], resolved.userWords[i]);
        resolved.memory.assign(prologue->memory.begin(), prologue->memory.end());
        resolved.linked.clear();
        if ((job.recordFlags & DrawRecordFlags::VerifyDue) != 0 && !trusted()) {
            refreshStages(job, resolved);
            DirectDraws::Add(Counter::PrologueVerifyChecked);
            const auto todayShape = DirectShapeRefusal(job);
            const bool direct = !todayShape.has_value();
            bool same = direct;
            for (std::size_t i = 0; i < programs.size(); ++i) same = same && resolved.userWords[i] == resolved.decode->programs[i].userData;
            if (!same) {
                DirectDraws::Add(Counter::PrologueVerifyDiffering);
                static std::atomic<std::uint32_t> printed{0};
                if (printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[direct-draws] prologue verify: %s differs from today's for a shape with %zu programs\n", direct ? "a program's user words" : "the direct shape test", programs.size());
                for (std::size_t i = 0; i < programs.size(); ++i) resolved.userWords[i] = resolved.decode->programs[i].userData;
            }
            if (!direct) {
                DirectDraws::Add(*todayShape);
                return Counter::FallbackShape;
            }
        }
    } else {
        resolved.decode = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::LeanPrepare) ? refreshedDecode(shape.decode, job.words) : LeanRefreshedDecode(*shape.decode, job.words);
    }
    const auto& decode = *resolved.decode;
    const auto& graphics = decode.state;
    const auto& programs = decode.programs;
    const auto& roles = decode.roles;
    const auto count = programs.size();
    if (!lazy) {
        resolved.memory.clear();
        AppendProgramMemory(resolved.memory, programs);
        resolved.linked.clear();
        for (std::size_t i = 0; i < count; ++i) {
            const auto& program = programs[i];
            resolved.linked.push_back({roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
        }
    }
    resolved.vertexInfos.assign(count, std::nullopt);
    resolved.decodeReads.resize(count);
    const auto decodeVertex = [&](std::size_t i) {
        const auto& program = resolved.decode->programs[i];
        auto& reads = resolved.decodeReads[i];
        reads.clear();
        auto& info = resolved.vertexInfos[i];
        info.reset();
        if (program.binary.stage == Stage::Fragment) return;
        info = DecodeOrderedVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, reads, 8u - program.firstUserSgpr);
        info->paClVsOutCntl = PositionExportLayout(program.snapshot.get(), program.codeOffset, program.binary.code, shape.reg207, program.tailContinuation.get()) | graphics.layerExports;
        info->depthRemapMode = graphics.depthRemap.mode;
        Graphics::ApplyPointSize(graphics, *info);
    };
    if (memo != nullptr) {
        for (auto& reads : resolved.decodeReads) reads.clear();
    } else {
        for (std::size_t i = 0; i < count; ++i) decodeVertex(i);
    }
    const bool snapshotTrusted = trusted();
    resolved.serialed = !snapshotTrusted && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::SettleSerial);
    if (resolved.serialed) resolved.beforeSerial = Graphics::Recorder::PublishGeneration();
    resolved.wvSession = Validation::FinisherSide() ? Validation::Session() : 0;
    if (resolved.wvSession != 0) Validation::Note(Validation::Test::PendingSnapshot, Validation::Outcome::Kept);
    if (snapshotTrusted) resolved.before.reset();
    else resolved.before = Graphics::Recorder::PendingWriteSnapshot();
    resolved.generation = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitStamps) ? GuestMemory::TrackerGenerationSettled() : GuestMemory::TrackerGeneration();
    resolved.stages.resize(count);
    resolved.regions.clear();
    lap(Step::Request);
    auto& cache = HostThreadLocal<InvocationCache, InvocationCacheStorage>();
    auto& reads = HostThreadLocal<DirectReads, DirectReadsStorage>();
    const bool libraries = device.GraphicsPipelineLibraries();
    const ShaderRecompiler::RecompileResult* previous = nullptr;
    std::uint32_t cursor = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& head = programs[i];
        auto& stage = resolved.stages[i];
        stage.Unview();
        cursor = Graphics::StagePushOffset(cursor, head.binary.stage, libraries);
        stage.program = i;
        stage.stage = head.binary.stage;
        stage.requestOffset = cursor;
        stage.pushOffset = 0;
        stage.shared.reset();
        stage.capture.reset();
        const auto cursorBefore = cursor;
        std::optional<StageMemoKept> memoKept;
        bool memoAdmitted = true;
        bool memoMissed = false;
        bool memoUnsettled = false;
        const bool memoStage = memo != nullptr && stageMemoCacheable(head);
        if (memo != nullptr) {
            const auto& known = cache.For(head);
            if (memoStage && known.handle != nullptr && known.snapshot.get() == head.snapshot.get() && known.codeOffset == head.codeOffset) {
                const auto words = resolved.worded ? std::span<const std::uint32_t>(resolved.userWords[i]) : std::span<const std::uint32_t>(head.userData);
                const auto outcome = adoptStage(*memo, job, words, i, cursor, known, resolved);
                DirectDraws::Add(stageMemoCounter(outcome));
                memoUnsettled = outcome == StageMemoOutcome::Unsettled;
                if (outcome == StageMemoOutcome::NoEntry) {
                    memoAdmitted = stageMemoAdmitted(*memo);
                    memoMissed = true;
                }
                if (outcome == StageMemoOutcome::Inputs && memo->inputsTag) DirectDraws::Add(Counter::StageMemoInputsOlderTag);
                const bool addressedRefused = outcome == StageMemoOutcome::Hit && stage.Shared()->addressed && addressedRefusal(*stage.Shared()).has_value();
                if (addressedRefused) {
                    stage.shared.reset();
                    stage.capture.reset();
                    stage.Unview();
                }
                if (outcome == StageMemoOutcome::Hit && !addressedRefused) {
                    const auto every = stageMemoVerifyEvery();
                    if (memo->patched) DirectDraws::Add(Counter::StageMemoPatched);
                    if (memo->merged) DirectDraws::Add(Counter::StageMemoMerged);
                    if (memo->relocated) DirectDraws::Add(Counter::StageMemoRelocated);
                    auto& ordinal = memo->relocated ? memo->relocatedOrdinal : memo->merged ? memo->mergedOrdinal : memo->patched ? memo->patchedOrdinal : memo->verifyOrdinal;
                    if (every == 0 || ++ordinal % every != 0 || trusted()) {
                        const auto& adopted = *stage.Shared()->donor;
                        cursor += static_cast<std::uint32_t>(adopted.pushConstants.size());
                        previous = &adopted;
                        stage.adopted = true;
                        lap(Step::Side);
                        continue;
                    }
                    memoKept.emplace(StageMemoKept{stage, resolved.VertexInfo(i), resolved.DecodeReads(i), memo->patched, memo->merged, memo->relocated});
                    stage.shared.reset();
                    stage.capture.reset();
                    stage.Unview();
                }
            } else if (memoStage) {
                DirectDraws::Add(known.handle == nullptr ? Counter::StageMemoNotLookedUpEmpty : Counter::StageMemoNotLookedUpOther);
            }
            refreshStages(job, resolved);
            decodeVertex(i);
        }
        const auto& program = resolved.decode->programs[i];
        const auto pixel = program.binary.stage == Stage::Fragment && previous != nullptr ? Graphics::DefaultUnexportedInputs(decode.pixel, previous->parameterExports) : decode.pixel;
        const bool hasFragment = std::ranges::any_of(resolved.decode->programs, [](const DrawProgram& value) { return value.binary.stage == Stage::Fragment; });
        const auto request = [&] { return drawStageRequest(program, graphics, pixel, resolved.vertexInfos[i], resolved.memory, resolved.linked, device, cursor, job.parameters, std::nullopt, hasFragment); };
        auto built = request();
        lap(Step::Request);
        if (built.target.gpuResolvedImages) return Counter::FallbackHeap;

        std::optional<ShaderRecompiler::PreparedShaderInvocation> invocation;
        InvocationSlot* slot = nullptr;
        try {
            if (InvocationCacheable(program, built)) slot = &cache.For(program);
            const auto rebuilt = slot == nullptr ? Counter::InvocationsRebuiltUncached : slot->handle == nullptr ? Counter::InvocationsRebuiltEmpty : slot->snapshot.get() != program.snapshot.get() || slot->codeOffset != program.codeOffset ? Counter::InvocationsRebuiltOther : Counter::InvocationsRebuiltRefused;
            if (slot != nullptr && slot->handle != nullptr && slot->snapshot.get() == program.snapshot.get() && slot->codeOffset == program.codeOffset) {
                built.context.convertedSamples = slot->convertedSamples;
                built.context.layerRowSamples = slot->layerRowSamples;
                ShaderRecompiler::BuildPreparedShaderKey(built, cache.key);
                built.shader.code = ShaderRecompiler::GetPreparedCode(*slot->handle);
                invocation = ShaderRecompiler::PreparedShaderInvocation::TryCreate(std::move(built), slot->handle, cache.key);
            }
            if (!invocation) {
                const auto fresh = request();
                invocation.emplace(InvocationFor(*program.snapshot, program.codeOffset, fresh));
                DirectDraws::Add(Counter::InvocationsRebuilt);
                DirectDraws::Add(rebuilt);
                if (slot != nullptr) {
                    const auto& context = invocation->Request().context;
                    const auto trust = slot->snapshot.get() == program.snapshot.get() && slot->codeOffset == program.codeOffset ? slot->trust : DirectDraws::LocalTrustState{};
                    *slot = {program.snapshot, program.codeOffset, invocation->Handle(), context.convertedSamples, context.layerRowSamples, invocation->VariantId(), trust};
                }
            }
        } catch (const std::exception&) {
            if (slot != nullptr) slot->handle.reset();
            return Counter::FallbackInvocation;
        }
        lap(Step::Invocation);

        reads.Reset();
        try {
            stage.capture = walkDirect(*invocation, reads);
        } catch (const ShaderRecompiler::ConvertedSampleRequired&) {
            if (slot != nullptr) slot->handle.reset();
            return Counter::FallbackSampleRetry;
        } catch (const ShaderRecompiler::LayerRowSamplesRequired&) {
            if (slot != nullptr) slot->handle.reset();
            return Counter::FallbackSampleRetry;
        } catch (const std::exception&) {
            return Counter::FallbackWalk;
        }
        if (reads.failed || stage.capture == nullptr) return Counter::FallbackWalk;
        const auto& snapshot = stage.capture->snapshot;
        if (imageTables(snapshot)) return Counter::FallbackTables;
        if (!readRanges(reads, stage.ranges, stage.values)) return Counter::FallbackTornReads;
        if (!CaptureRegisteredViewsCurrent(snapshot.registeredImages)) return Counter::FallbackViews;
        const auto walkUse = slot != nullptr && !memoUnsettled ? auditUse(slot->trust) : TrustUse::Test;
        noteUse(TrustSite::WalkSettle, walkUse);
        if (!trusted() && walkUse != TrustUse::Skip && !settledValidated(stage.ranges, resolved, Validation::Test::WalkSettle)) {
            if (slot != nullptr) noteFailed(slot->trust, TrustSite::WalkSettle, walkUse);
            noteUnsettled(stage.ranges, resolved.before.get(), resolved.generation);
            return Counter::FallbackUnsettled;
        }
        GuardRanges(stage.ranges);
        lap(Step::Walk);

        const auto key = invocation->SharedKey(*stage.capture);
        if (!key) return Counter::FallbackWalk;
        const auto variant = invocation->VariantId();
        stage.shared = stores.stages.Find(variant, *key);
        if (stage.shared == nullptr) {
            ShaderRecompiler::SharedStage shared;
            try {
                shared = invocation->ShareStage(*stage.capture);
            } catch (const std::exception&) {
                return Counter::FallbackNoPatch;
            }
            if (!shared) return Counter::FallbackNoPatch;
            auto made = std::make_shared<DirectStage>();
            made->variant = variant;
            made->key = *key;
            made->addressed = std::ranges::any_of(shared.donor->bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; });
            made->donor = std::move(shared.donor);
            made->patch = std::move(shared.patch);
            bool installed = false;
            stage.shared = stores.stages.Install(std::move(made), resolved.tag, installed);
            if (installed) DirectDraws::Add(Counter::StagesBuilt);
        }
        const auto& donor = *stage.shared->donor;
        if (donor.gpuResolvedImages || donor.deviceImageTables) return Counter::FallbackHeap;
        if (stage.shared->addressed) {
            if (const auto refused = addressedRefusal(*stage.shared)) {
                DirectDraws::Add(*refused);
                return Counter::FallbackAddressed;
            }
        }
        if (snapshot.srtPoison.size() + snapshot.nullRootReads != donor.poisonedSrtReads) return Counter::FallbackPoisoned;
        lap(Step::Key);

        if (!invocation->FillSharedSide(*stage.capture, donor, *stage.shared->patch, stage.side)) return Counter::FallbackSide;
        const auto slotEnd = (cursor / Graphics::PipelinePushSlotBytes + 1u) * Graphics::PipelinePushSlotBytes;
        if (donor.pushConstants.size() > slotEnd - cursor) return Counter::FallbackPush;
        stage.pushOffset = donor.pushConstants.empty() ? 0u : cursor;
        cursor += static_cast<std::uint32_t>(donor.pushConstants.size());
        previous = &donor;
        if (memoKept.has_value()) {
            DirectDraws::Add(Counter::StageMemoVerifyChecked);
            if (memoKept->patched) DirectDraws::Add(Counter::StageMemoVerifyPatched);
            if (memoKept->merged) DirectDraws::Add(Counter::StageMemoVerifyMerged);
            if (memoKept->relocated) DirectDraws::Add(Counter::StageMemoVerifyRelocated);
            if (const char* what = stageMemoDifference(*memoKept, stage, resolved.VertexInfo(i), resolved.DecodeReads(i), count); what != nullptr) {
                DirectDraws::Add(Counter::StageMemoVerifyDiffering);
                static std::atomic<std::uint32_t> printed{0};
                if (printed.fetch_add(1, std::memory_order_relaxed) < 32) {
                    const auto* kept = memoKept->stage.Shared().get();
                    const auto* fresh = stage.Shared().get();
                    const auto& keptCapture = memoKept->stage.Capture();
                    const auto& freshCapture = stage.Capture();
                    const auto wide = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
                    std::fprintf(stderr, "[stage-memo-verify] %s differs: stage %zu of a draw with %zu stages (kept variant %llx key %llx, fresh variant %llx key %llx, donor %s, patch %s, read ranges %s, read values %s, registered images %zu and %zu)\n", what, i, count, wide(kept != nullptr ? kept->variant : 0), wide(kept != nullptr ? kept->key : 0), wide(fresh != nullptr ? fresh->variant : 0), wide(fresh != nullptr ? fresh->key : 0), kept != nullptr && fresh != nullptr && kept->donor.get() == fresh->donor.get() ? "same" : "other", kept != nullptr && fresh != nullptr && kept->patch.get() == fresh->patch.get() ? "same" : "other", memoKept->stage.Ranges() == stage.Ranges() ? "same" : "other", memoKept->stage.Values() == stage.Values() ? "same" : "other", keptCapture != nullptr ? keptCapture->snapshot.registeredImages.size() : std::size_t{0}, freshCapture != nullptr ? freshCapture->snapshot.registeredImages.size() : std::size_t{0});
                }
            }
        } else if (memoStage && memoAdmitted && slot != nullptr && slot->handle != nullptr) {
            const auto memoKey = stageMemoKey(job.shape->decode.get(), i);
            if (stageMemoConflictWaits(*memo, memoKey)) {
                DirectDraws::Add(Counter::StageMemoChainConflict);
            } else {
                std::uint64_t storedKey = 0;
                const auto stored = rememberStage(job, program, i, cursorBefore, *slot, *invocation, reads, resolved, storedKey);
                if (stored) DirectDraws::Add(*stored);
                if (stored == Counter::StageMemoChainConflict) stageMemoConflictSlot(*memo, memoKey) = {memoKey, 64u};
                if (memoMissed && memo->tried != 0 && !memo->forced && stored == Counter::StageMemoStored && !stageMemoLookedUp(*memo, storedKey)) DirectDraws::Add(Counter::StageMemoAdmitOtherKey);
            }
        }
        if (resolveCensus()) noteStageCensus(*invocation, *stage.Capture(), reads, resolved.DecodeReads(i));
        lap(Step::Side);
    }
    if (graphics.rectList) {
        if (count != 2) {
            DirectDraws::Add(Counter::ShapeRectPrograms);
            return Counter::FallbackShape;
        }
        const auto& vertex = *resolved.stages[0].Shared()->donor;
        const auto& fragment = *resolved.stages[1].Shared()->donor;
        resolved.rect = PreparedRectangleShaders(*shape.decode->programs[0].snapshot, vertex.variantId, fragment.variantId);
        if (resolved.rect == nullptr) {
            DirectDraws::Add(Counter::ShapeRectUnprepared);
            return Counter::FallbackShape;
        }
        resolved.rectControl = graphics.provokingVertexMode == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT ? &resolved.rect->lastVertexControl : &resolved.rect->control;
    }
    resolved.addressed = std::ranges::any_of(resolved.stages, [](const DirectResolvedStage& stage) { return stage.Shared()->addressed; });
    if (!resolved.viewing || !layoutFromEntry(resolved)) {
        for (std::size_t i = 0; i < count; ++i) {
            if (resolved.stages[i].vertexView != nullptr) resolved.vertexInfos[i] = *resolved.stages[i].vertexView;
        }
        DrawVertexLayout(resolved.vertexInfos, resolved.vertexLayout, resolved.vertexInputs);
    }
    auto& regions = resolved.regions;
    resolved.checksUse = auditUse(job.shape->trust);
    const bool regionsTrusted = !resolveCensus() && (trusted() || resolved.checksUse == TrustUse::Skip);
    if (regionsTrusted) regions.clear();
    else regions.assign(resolved.memory.begin(), resolved.memory.end());
    if (regionsTrusted) {
        resolved.compared = 0;
    } else if (!resolved.viewing) {
        for (const auto& decodeReads : resolved.decodeReads) {
            for (const auto& read : decodeReads) regions.push_back({read.address, read.Bytes()});
        }
        for (const auto& stage : resolved.stages) {
            std::size_t at = 0;
            for (const auto& [begin, end] : stage.ranges) {
                const auto words = static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
                regions.push_back({begin, std::as_bytes(std::span(stage.values.data() + at, words))});
                at += words;
            }
        }
        resolved.compared = regions.size();
    } else {
        appendStageRegions(resolved, false);
        resolved.compared = regions.size();
        appendStageRegions(resolved, true);
        std::size_t adopted = 0;
        for (const auto& stage : resolved.stages) adopted += stage.adopted ? 1u : 0u;
        if (adopted != 0) DirectDraws::Add(Counter::StageViewUncompared, adopted);
    }
    if (resolveCensus()) noteResolveCensus(job, resolved);
    lap(Step::Checks);
    return std::nullopt;
}

std::optional<Counter> Driver::assembleDirectHit(const LeanDrawJob& job, DirectStages& resolved, DrawRecordHit& hit) {
    const auto count = resolved.stages.size();
    hit.sides.resize(count);
    hit.captures.resize(count);
    hit.stages.clear();
    hit.stages.reserve(resolved.rect != nullptr ? count + 2u : count);
    bool entryReads = resolved.viewing;
    for (std::size_t i = 0; i < count; ++i) {
        auto& stage = resolved.stages[i];
        const auto* side = stage.sideView;
        if (side == nullptr) {
            if (hit.pooled) hit.sides[i] = stage.side;
            else hit.sides[i] = std::move(stage.side);
            side = &hit.sides[i];
        }
        entryReads = entryReads && stage.sideView != nullptr && stage.memo != nullptr && static_cast<const StageMemoEntry*>(stage.memo)->viewable;
        if (stage.pin != nullptr) hit.captures[i] = std::shared_ptr<const ShaderRecompiler::ResourceCapture>(std::move(stage.pin), stage.Capture().get());
        else hit.captures[i] = stage.Capture();
        hit.stages.push_back({stage.stage, stage.Shared()->donor.get(), stage.pushOffset, &hit.captures[i]->snapshot.registeredImages, side});
    }
    if (resolved.rect != nullptr) {
        const std::array<Graphics::CompiledShader, 2> generated{{{Stage::TessellationControl, resolved.rectControl, 0u}, {Stage::TessellationEvaluation, &resolved.rect->evaluation, 0u}}};
        hit.stages.insert(hit.stages.begin() + 1, generated.begin(), generated.end());
    }
    hit.parameters = job.parameters;
    const auto& front = *resolved.stages.front().Shared()->donor;
    const auto& program = resolved.decode->programs.front();
    const auto words = resolved.worded ? std::span<const std::uint32_t>(resolved.userWords.front()) : std::span<const std::uint32_t>(program.userData);
    const auto userWord = [&](std::int32_t sgpr, std::uint32_t& value) {
        if (sgpr < 0 || static_cast<std::uint32_t>(sgpr) < program.firstUserSgpr) return false;
        const auto index = static_cast<std::uint32_t>(sgpr) - program.firstUserSgpr;
        if (index >= words.size()) return false;
        value = words[index];
        return true;
    };
    std::uint32_t offset = 0;
    if (resolved.indirect) hit.parameters.indirect = resolved.indirect;
    if (!hit.parameters.indirect && front.vertexOffsetSgpr >= 0) {
        if (!userWord(front.vertexOffsetSgpr, offset) || offset > std::numeric_limits<std::uint32_t>::max() - hit.parameters.firstVertex) return Counter::FallbackOverflow;
        hit.parameters.firstVertex += offset;
    }
    if (!hit.parameters.indirect && front.instanceOffsetSgpr >= 0) {
        if (!userWord(front.instanceOffsetSgpr, offset)) return Counter::FallbackOverflow;
        hit.parameters.firstInstance = offset;
    }
    if (entryReads) {
        hit.reads = QueuedDrawReadRanges(hit.parameters, {});
        for (const auto& stage : resolved.stages) {
            const auto& held = static_cast<const StageMemoEntry*>(stage.memo)->bufferReads;
            hit.reads.insert(hit.reads.end(), held.begin(), held.end());
        }
        const auto& attributes = static_cast<const StageMemoEntry*>(resolved.stages.front().memo)->attributeReads;
        hit.reads.insert(hit.reads.end(), attributes.begin(), attributes.end());
        DirectDraws::Add(Counter::StageViewReads);
    } else {
        hit.reads = QueuedDrawReadRanges(hit.parameters, hit.stages);
    }
    runCaptureCheckHook();
    const std::span<const ShaderRecompiler::MemoryRegion> regions(resolved.regions);
    const auto compared = resolved.viewing ? std::min(resolved.compared, regions.size()) : regions.size();
    const auto checksUse = resolved.checksUse;
    const bool checksTrusted = trusted() || checksUse == TrustUse::Skip;
    noteUse(TrustSite::CapturedMemory, checksUse);
    bool programVerify = false;
    auto programHeld = checksTrusted ? std::size_t{0} : programMemoryHeld(resolved, regions, compared, programVerify);
    if (programVerify) {
        const bool changed = capturedMemoryChanged(regions.first(programHeld));
        Validation::NoteVerify(Validation::Test::CapturedMemory, changed && programArmsHeld(resolved, regions, programHeld, Validation::Session()), "program memory changed although its clean, settled windows raised no event since its arm");
        if (changed) programHeld = 0;
    }
    if (!checksTrusted && capturedMemoryChanged(regions.subspan(programHeld, compared - programHeld))) {
        noteFailed(job.shape->trust, TrustSite::CapturedMemory, checksUse);
        return Counter::FallbackMemoryChanged;
    }
    if (!checksTrusted && programHeld == 0) armProgramMemory(resolved);
    if (!checksTrusted && compared < regions.size()) {
        struct OrdinalStorage {};
        auto& ordinal = HostThreadLocal<std::uint64_t, OrdinalStorage>();
        if (const auto every = stageMemoVerifyEvery(); every != 0 && ++ordinal % every == 0) {
            DirectDraws::Add(Counter::StageViewMemoryChecked);
            if (capturedMemoryChanged(regions.subspan(compared))) {
                noteFailed(job.shape->trust, TrustSite::CapturedMemory, checksUse);
                DirectDraws::Add(Counter::StageViewMemoryChanged);
                return Counter::FallbackMemoryChanged;
            }
        }
    }
    if (resolved.wvSession != 0) Validation::Note(Validation::Test::LabelOverlap, Validation::Outcome::Kept);
    if (recordQueuedLabelsAfterCapture(job.queue, checksTrusted ? std::span<const ShaderRecompiler::MemoryRegion>() : regions)) {
        noteFailed(job.shape->trust, TrustSite::CapturedMemory, checksUse);
        return Counter::FallbackLabels;
    }
    if (resolved.addressed) return addressedSnapshots(resolved, hit);
    return std::nullopt;
}

const DrawRecordSink* Driver::finishDirectAnswer(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectStages& resolved, const DrawRecordHit& hit, DrawRecordView& answer) {
    bool answered = false;
    try {
        answer.Capture(hit.stages, hit.parameters);
        answer.reads = hit.reads;
        answer.contentKey = anchor.device->DrawContentKey(resolved.decode->state, hit.parameters, hit.stages);
        if (hit.addressed) {
            answer.addressed = true;
            answer.CaptureSnapshots(hit.snapshots);
        }
        answered = true;
    } catch (const std::exception&) {
    }
    struct SinkStorage {};
    auto& sink = HostThreadLocal<DrawRecordSink, SinkStorage>();
    sink.Reset();
    sink.verify = true;
    struct SinkScope {
        ~SinkScope() { DrawRecordSinkSlot::Set(nullptr); }
    } scope;
    if (answered) DrawRecordSinkSlot::Set(&sink);
    finishLeanDraw(job, anchor);
    DrawRecordSinkSlot::Set(nullptr);
    return answered ? &sink : nullptr;
}

void Driver::buildDirectClass(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectClassKey& key, std::uint64_t probe, const DirectStages& resolved, const DrawRecordHit& hit) {
    DrawRecordView answer;
    const auto* today = finishDirectAnswer(job, anchor, resolved, hit, answer);
    if (today == nullptr) return;
    const auto& sink = *today;
    auto entry = std::make_shared<DirectClass>();
    entry->key = key;
    entry->probe = probe;
    for (std::size_t i = 0; i < resolved.stages.size(); ++i) entry->stages[i] = resolved.stages[i].Shared();
    entry->generated = resolved.rect;
    if (!sink.queued) {
        entry->refusal = Counter::RefusedNotQueued;
    } else if (sink.writes != job.shape->writes) {
        entry->refusal = Counter::RefusedWrites;
    } else {
        auto record = DrawRecord::Build(DrawRecordKey::For(job.shapeKey, job.parameters), job.shape->shared_from_this(), job.deviceSerial, job.recordGeneration, Graphics::Recorder::Presents(), sink);
        record->contentKey.clear();
        record->vertexLayout = resolved.vertexLayout;
        answer.writes = record->writes;
        answer.fragmentOutputs = record->fragmentOutputs;
        if (compareRecordAnswer(job, *record, answer, sink, &DirectDraws::NoteBuildDifference, &DirectDraws::NoteAllowed)) entry->refusal = Counter::RefusedDiffers;
        else entry->record = std::move(record);
    }
    const auto outcome = entry->record != nullptr ? Counter::ClassesBuilt : entry->refusal;
    bool installed = false;
    DirectStores::Get().classes.Install(std::move(entry), resolved.tag, installed);
    if (installed) DirectDraws::Add(outcome);
    if (installed && resolved.addressed) DirectDraws::Add(outcome == Counter::ClassesBuilt ? Counter::AddressedClassesBuilt : Counter::AddressedClassesRefused);
}

void Driver::verifyDirectHit(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectClass& found, const DirectStages& resolved, const DrawRecordHit& hit) {
    DrawRecordView answer;
    const auto* today = finishDirectAnswer(job, anchor, resolved, hit, answer);
    if (today == nullptr) return;
    DirectDraws::Add(Counter::VerifyChecked);
    const auto& record = *found.record;
    answer.writes = record.writes;
    answer.fragmentOutputs = record.fragmentOutputs;
    if (!compareRecordAnswer(job, record, answer, *today, &DirectDraws::NoteDifference, &DirectDraws::NoteAllowed)) return;
    DirectDraws::Add(Counter::VerifyDiffering);
    if (DirectStores::Get().classes.Refuse(found, Counter::RefusedVerify)) DirectDraws::Add(Counter::RefusedVerify);
}

void Driver::verifyDirectRectList(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectClass& found, const DirectStages& resolved, const DrawRecordHit& hit) {
    DrawRecordView answer;
    const auto* today = finishDirectAnswer(job, anchor, resolved, hit, answer);
    if (today == nullptr) return;
    DirectDraws::Add(Counter::RectListsVerifyChecked);
    const auto& record = *found.record;
    answer.writes = record.writes;
    answer.fragmentOutputs = record.fragmentOutputs;
    if (!compareRecordAnswer(job, record, answer, *today, &DirectDraws::NoteRectDifference, &DirectDraws::NoteAllowed)) return;
    DirectDraws::Add(Counter::RectListsVerifyDiffering);
    if (DirectStores::Get().classes.Refuse(found, Counter::RefusedVerify)) DirectDraws::Add(Counter::RefusedVerify);
}

void Driver::verifyAddressedHit(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectClass& found, const DirectStages& resolved, const DrawRecordHit& hit) {
    DrawRecordView answer;
    const auto* today = finishDirectAnswer(job, anchor, resolved, hit, answer);
    if (today == nullptr) return;
    DirectDraws::Add(Counter::AddressedVerifyChecked);
    const auto& record = *found.record;
    answer.writes = record.writes;
    answer.fragmentOutputs = record.fragmentOutputs;
    if (!compareRecordAnswer(job, record, answer, *today, &DirectDraws::NoteAddressedDifference, &DirectDraws::NoteAllowed)) return;
    DirectDraws::Add(Counter::AddressedVerifyDiffering);
    if (DirectStores::Get().classes.Refuse(found, Counter::RefusedVerify)) DirectDraws::Add(Counter::RefusedVerify);
}

void Driver::finishDirectDraw(const LeanDrawJob& job) {
    const SampledHitch hitch("finisher job");
    const auto& anchor = leanDraws.AnchorAt(job.anchor);
    DirectDraws::Add(Counter::Draws);
    const bool replaying = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpJobReplay);
    std::optional<Counter> replayMiss;
    if (replaying) {
        replayMiss = replayDirectJob(job, anchor);
        if (!replayMiss) return;
        if (*replayMiss != Counter::ReplayTodayNoEarlier) DirectDraws::Add(*replayMiss);
    } else if (replayHolders.load(std::memory_order_relaxed) != 0) {
        dropReplayStore();
    }
    const bool rectCensus = job.shape->decode->state.rectList && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectRectLists);
    struct ResolvedStorage {};
    auto& resolved = HostThreadLocal<DirectStages, ResolvedStorage>();
    struct SampleStorage {};
    auto& sampled = HostThreadLocal<std::uint64_t, SampleStorage>();
    std::optional<DirectStepClock> clock;
    if (sampled++ % DirectDraws::StepSampleEvery == 0) clock.emplace();
    auto* timer = clock.has_value() ? &*clock : nullptr;
    const auto lap = [&](Step step) {
        if (timer != nullptr) timer->Lap(step);
    };
    std::optional<Counter> reason;
    DirectClassKey key;
    std::uint64_t probe = 0;
    std::shared_ptr<const DirectClass> found;
    std::shared_ptr<DrawRecordHit> hit;
    bool rectVerify = false;
    bool addressedVerify = false;
    try {
        reason = resolveDirectStages(job, anchor, resolved, timer);
        for (std::uint32_t again = 0; again < 2u && reason == Counter::FallbackUnsettled && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectRetry); ++again) reason = resolveDirectStages(job, anchor, resolved, timer);
        resolved.indirect.reset();
        if (!reason && job.parameters.indirect) reason = classifyDirectIndirect(job, anchor, resolved);
        if (!reason) {
            key = DirectClassKey::For(job, resolved);
            probe = key.Probe();
            found = DirectStores::Get().classes.Find(key, probe);
            if (found != nullptr && found->record == nullptr) reason = Counter::FallbackClassRefused;
            else if (found != nullptr && found->record->vertexLayout != resolved.vertexLayout) found = nullptr;
            lap(Step::Class);
        }
        if (!reason) {
            hit = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PooledHits) ? pooledHit(job.deviceSerial) : std::make_shared<DrawRecordHit>();
            reason = assembleDirectHit(job, resolved, *hit);
            lap(Step::Checks);
        }
        if (!reason && found == nullptr) reason = Counter::FallbackClassNew;
        else if (!reason && (job.recordFlags & DrawRecordFlags::VerifyDue) != 0 && !trusted()) reason = Counter::FallbackVerify;
        else if (!reason && resolved.rect != nullptr && rectVerifyDue()) {
            reason = Counter::FallbackVerify;
            rectVerify = true;
        }
        else if (!reason && resolved.addressed && addressedVerifyDue()) {
            addressedVerify = true;
            reason = Counter::FallbackVerify;
        }
        if (!reason && resolved.addressed) hit->contentKey = anchor.device->DrawContentKey(resolved.decode->state, hit->parameters, hit->stages);
    } catch (const std::exception&) {
        reason = Counter::FallbackWalk;
    }
    if (!reason) {
        hit->record = found->record;
        auto& classTrust = found->trust;
        if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectHandoff)) hit->owner = std::move(found);
        else hit->owner = found;
        hit->direct = true;
        if (hit->parameters.indirect) DirectDraws::Add(Counter::IndirectHits);
        auto* const splits = timer != nullptr && stepSplits() ? timer : nullptr;
        if (hit->addressed) {
            DirectDraws::Add(Counter::AddressedServed);
            DirectDraws::Add(Counter::AddressedSnapshotRegions, hit->snapshots.size());
            std::uint64_t snapshotBytes = 0;
            for (const auto& snapshot : hit->snapshots) snapshotBytes += snapshot.bytes.size();
            DirectDraws::Add(Counter::AddressedSnapshotBytes, snapshotBytes);
        }
        const auto classUse = !hit->addressed && (job.recordFlags & (DrawRecordFlags::DirectBuffers | DrawRecordFlags::DirectInputs)) != 0 ? auditUse(classTrust) : TrustUse::Test;
        if (!hit->addressed && (job.recordFlags & DrawRecordFlags::DirectBuffers) != 0) {
            prepareBuffers(job, anchor, resolved, *hit, splits, classTrust, classUse);
            if (hit->preparedBuffers.has_value()) {
                feedZoneCard(*hit->preparedBuffers);
                noteGuardAnswer(Graphics::AnswerPreparedGuards(*hit->preparedBuffers), false);
            }
            lap(Step::Buffers);
        }
        if (!hit->addressed && (job.recordFlags & DrawRecordFlags::DirectInputs) != 0) {
            prepareInputs(job, anchor, resolved, *hit, classTrust, classUse);
            if (hit->preparedInputs.has_value()) noteGuardAnswer(Graphics::AnswerPreparedInputGuards(*hit->preparedInputs), true);
            lap(Step::Inputs);
        }
        if (!hit->addressed && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitFragments)) recordFragment(job, anchor, *hit);
        if (splits != nullptr) splits->Split(Step::Fragment);
        if (replaying) {
            try {
                keepReplaySource(job, anchor, resolved, *hit);
            } catch (const std::exception&) {
            }
            if (replayMiss == Counter::ReplayTodayNoEarlier) DirectDraws::Add(Counter::ReplayTodayNoEarlier);
        }
        commitRecordDraw(job, anchor, std::move(hit));
        if (splits != nullptr) splits->Split(Step::Commit);
        lap(Step::HandOff);
        if (timer != nullptr) timer->Report();
        DirectDraws::Add(Counter::Direct);
        if (rectCensus) {
            DirectDraws::Add(Counter::RectListsSeen);
            DirectDraws::Add(Counter::RectListsDirect);
        }
        ThreadReport::MarkItem(ThreadReport::ItemMark::Planned);
        return;
    }
    if (timer != nullptr) timer->Report();
    struct FallbackMark {
        ~FallbackMark() { ThreadReport::MarkItem(ThreadReport::ItemMark::PlanFallback); }
    } fallbackMark;
    DirectDraws::Add(*reason);
    if (replayMiss == Counter::ReplayTodayNoEarlier) DirectDraws::Add(*reason == Counter::FallbackClassNew || *reason == Counter::FallbackClassRefused ? Counter::ReplayTodayClassNew : Counter::ReplayTodayNotDirect);
    if (rectCensus) {
        DirectDraws::Add(Counter::RectListsSeen);
        DirectDraws::Add(*reason == Counter::FallbackShape ? Counter::RectListsShape : *reason == Counter::FallbackClassNew ? Counter::RectListsClassNew : *reason == Counter::FallbackVerify ? Counter::RectListsVerify : Counter::RectListsOther);
    }
    if (resolved.addressed && *reason != Counter::FallbackAddressed && *reason != Counter::FallbackVerify) DirectDraws::Add(Counter::AddressedLater);
    if (*reason == Counter::FallbackClassNew) {
        buildDirectClass(job, anchor, key, probe, resolved, *hit);
        return;
    }
    if (*reason == Counter::FallbackVerify) {
        if (rectVerify) verifyDirectRectList(job, anchor, *found, resolved, *hit);
        if (addressedVerify) verifyAddressedHit(job, anchor, *found, resolved, *hit);
        else verifyDirectHit(job, anchor, *found, resolved, *hit);
        return;
    }
    finishLeanDraw(job, anchor);
}

std::optional<Counter> Driver::replayDirectJob(const LeanDrawJob& job, const LeanAnchor& anchor) {
    if ((job.recordFlags & DrawRecordFlags::VerifyDue) != 0) return Counter::ReplayKeptVerify;
    constexpr auto prepared = DrawRecordFlags::DirectBuffers | DrawRecordFlags::DirectInputs;
    if ((job.recordFlags & prepared) != prepared || !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitBindings) || !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitFragments)) return Counter::ReplayKeptFlags;
    if (job.parameters.indirect) return Counter::ReplayKeptIndirect;
    auto& store = HostThreadLocal<ReplayStore, ReplayStoreStorage>();
    const bool timed = (store.timing & 7u) == 0;
    const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::shared_ptr<DrawRecordHit> hit;
    try {
        syncReplayStore(store, job.deviceSerial, DirectStores::Get().Follow(job.deviceSerial));
        const auto key = ReplayKey::For(job);
        auto& source = store.Slot(key);
        if (!source.held || source.kept == nullptr || !(source.key == key)) return Counter::ReplayTodayNoEarlier;
        if (source.device != anchor.device.get()) return Counter::ReplayKeptStores;
        const auto& classes = DirectStores::Get().classes;
        if (const auto clears = classes.Clears(); clears != source.clears) {
            if (!replayClassHeld(classes, source.kept->owner)) {
                source.held = false;
                source.kept.reset();
                return Counter::ReplayKeptClass;
            }
            source.clears = clears;
        }
        const auto forget = GuestMemory::ForgetSerial();
        const auto imports = Graphics::HostImportSetGeneration();
        const auto mapping = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        const auto& buffers = source.buffers;
        const auto& inputs = source.inputs;
        if ((forget & 1u) != 0 || buffers.forget != forget || inputs.forget != forget || inputs.imports != imports || inputs.mapping != mapping || (buffers.imports != 0 && buffers.imports != imports) || (buffers.mapping != 0 && buffers.mapping != mapping)) return Counter::ReplayKeptSerials;
        const auto& commit = *source.kept->record->commit;
        if (commit.bufferPlan.load(std::memory_order_acquire) != buffers.plan || commit.inputsRecipe.load(std::memory_order_acquire) != inputs.recipe) return Counter::ReplayKeptRecipe;
        DrawPipeline::FollowEpoch(job.epoch);
        DrawPipeline::Active() = true;
        GuestMemory::SetCurrentPacket(job.opcode, job.queue);
        static_cast<void>(recordQueuedLabelsAfterCapture(job.queue, {}));
        hit = pooledHit(job.deviceSerial);
        auto& made = *hit;
        made.record = source.kept->record;
        made.owner = source.kept->owner;
        made.replayed = source.kept;
        made.direct = true;
        made.parameters = source.parameters;
        made.stages.assign(source.stages.begin(), source.stages.end());
        made.reads.assign(source.reads.begin(), source.reads.end());
        auto& madeBuffers = made.pooled ? made.preparedBuffers.emplace(std::move(made.spareBuffers)) : made.preparedBuffers.emplace();
        madeBuffers = buffers;
        madeBuffers.depthGuards = Graphics::DepthGuardsGeneration_nid_no_patch();
        madeBuffers.guardAnswer = {};
        auto& madeInputs = made.pooled ? made.preparedInputs.emplace(std::move(made.spareInputs)) : made.preparedInputs.emplace();
        madeInputs = inputs;
        madeInputs.guardAnswer = {};
        madeInputs.state = &job.shape->decode->state;
        madeInputs.draw = &made.parameters;
        madeInputs.shaders = made.stages.data();
        madeInputs.stages = made.stages.size();
        if (!anchor.device->PlaceDirectBuffers(madeBuffers)) return Counter::ReplayKeptPlace;
    } catch (const std::exception&) {
        return Counter::ReplayKeptPlace;
    }
    const auto placed = timed ? std::chrono::steady_clock::now() : began;
    recordFragment(job, anchor, *hit);
    const bool fragmented = hit->fragment.has_value();
    const auto recorded = timed ? std::chrono::steady_clock::now() : began;
    commitRecordDraw(job, anchor, std::move(hit));
    ++store.timing;
    if (timed) {
        const auto ended = std::chrono::steady_clock::now();
        const auto between = [](std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count()); };
        DirectDraws::Add(Counter::ReplayNanoseconds, between(began, ended));
        DirectDraws::Add(Counter::ReplayCopyNanoseconds, between(began, placed));
        DirectDraws::Add(Counter::ReplayFragmentNanoseconds, between(placed, recorded));
        DirectDraws::Add(Counter::ReplayHandoffNanoseconds, between(recorded, ended));
        DirectDraws::Add(Counter::ReplayTimed);
    }
    DirectDraws::Add(Counter::ReplayJobs);
    if (!fragmented) DirectDraws::Add(Counter::ReplayNoFragment);
    DirectDraws::Add(Counter::Direct);
    ThreadReport::MarkItem(ThreadReport::ItemMark::Planned);
    return std::nullopt;
}

void Driver::runLeanRun(const std::byte* payload) {
    auto& driver = Get();
    const auto& header = *std::launder(reinterpret_cast<const LeanRunHeader*>(payload));
    if (header.count > 1 && ThreadReport::Enabled()) ThreadReport::Add(ThreadReport::Count::FinisherJobs, header.count - 1);
    for (std::uint32_t index = 0; index < header.count; ++index) driver.finishDirectDraw(header.jobs[index]);
}

}
