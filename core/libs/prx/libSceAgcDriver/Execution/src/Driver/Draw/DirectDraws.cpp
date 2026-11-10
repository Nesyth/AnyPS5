#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectDraws.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecords.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/SrtWalker.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <span>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

using DirectDraws::Counter;
using DirectDraws::Step;
using Role = ShaderRecompiler::ProgramRole;
using Stage = ShaderRecompiler::ShaderStage;

constexpr std::size_t InvocationSlots = 1024;
constexpr std::uint64_t WalkPageBytes = 0x1000u;

std::size_t mixedSlot(std::uint64_t first, std::uint64_t second, std::size_t slots) {
    auto mixed = first * 0x9e3779b97f4a7c15ull ^ second;
    mixed ^= mixed >> 29u;
    return static_cast<std::size_t>(mixed % slots);
}

bool DirectShape(const LeanDrawJob& job) {
    const auto& decode = *job.shape->decode;
    const auto& state = decode.state;
    if (job.parameters.indirect || state.quadList || state.rectList || state.stages.path != Graphics::ShaderPath::Vertex || state.stages.mesh || state.stages.tessellation || state.stages.tessellationGeometry || state.tessellationCapture != nullptr || !Graphics::DrawsRecordable()) return false;
    if (decode.programs.empty() || decode.programs.size() > DirectClassKey::MaxStages || decode.roles.size() != decode.programs.size()) return false;
    for (std::size_t i = 0; i < decode.programs.size(); ++i) {
        const auto& program = decode.programs[i];
        const auto stage = program.binary.stage;
        if ((stage != Stage::Vertex && stage != Stage::Fragment) || decode.roles[i] != (stage == Stage::Fragment ? Role::Fragment : Role::Main) || SkippedShader(program.binary.codeAddress)) return false;
    }
    return true;
}

struct InvocationSlot {
    std::shared_ptr<const ShaderSnapshot> snapshot;
    std::size_t codeOffset = 0;
    std::shared_ptr<const ShaderRecompiler::SourceHandle> handle;
    bool convertedSamples = false;
    bool layerRowSamples = false;
};

struct InvocationCache {
    std::array<InvocationSlot, InvocationSlots> slots;
    std::vector<std::uint64_t> key;

    InvocationSlot& For(const DrawProgram& program) { return slots[mixedSlot(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(program.snapshot.get())), program.codeOffset, InvocationSlots)]; }
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

    void Reset() {
        words.clear();
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
    std::lock_guard lock(mutex);
    if (!DirectStores::Get().Current(tag)) return stage;
    const std::pair key{stage->variant, stage->key};
    if (const auto found = stages.find(key); found != stages.end()) return found->second;
    if (stages.size() >= Bound) {
        DirectDraws::Add(Counter::StagesEvicted, stages.size());
        stages.clear();
        clears.fetch_add(1, std::memory_order_release);
    }
    stages.emplace(key, stage);
    installed = true;
    DirectDraws::NoteStages(stages.size());
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
    const auto layout = HashDrawRecordWords(resolved.vertexLayout, {});
    key.layout = layout[0] ^ (layout[1] * 0x9e3779b97f4a7c15ull);
    key.indexSize = parameters.indexSize;
    key.flags = parameters.flags;
    key.indexed = parameters.indexed;
    key.count = static_cast<std::uint32_t>(resolved.stages.size());
    for (std::size_t i = 0; i < resolved.stages.size(); ++i) key.stages[i] = resolved.stages[i].shared.get();
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
    return probe ^ (probe >> 29u);
}

std::size_t DirectClass::Bytes() const {
    return sizeof(DirectClass) + (record != nullptr ? record->Bytes() : 0);
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
    std::lock_guard lock(mutex);
    if (!DirectStores::Get().Current(tag)) return entry;
    if (const auto found = classes.find(entry->key); found != classes.end()) return found->second;
    if (classes.size() >= Bound) {
        DirectDraws::Add(Counter::ClassesEvicted, kept);
        classes.clear();
        kept = 0;
        bytes = 0;
        clears.fetch_add(1, std::memory_order_release);
    }
    if (entry->record != nullptr) ++kept;
    bytes += entry->Bytes();
    classes.emplace(entry->key, entry);
    installed = true;
    note();
    return entry;
}

bool DirectClassStore::Refuse(const DirectClass& found, Counter reason) {
    auto refused = std::make_shared<DirectClass>();
    refused->key = found.key;
    refused->probe = found.probe;
    refused->stages = found.stages;
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

void DirectClassStore::Clear() {
    std::lock_guard lock(mutex);
    classes.clear();
    kept = 0;
    bytes = 0;
    clears.fetch_add(1, std::memory_order_release);
    note();
}

std::size_t DirectClassStore::Size() const {
    std::lock_guard lock(mutex);
    return classes.size();
}

void DirectClassStore::note() const {
    DirectDraws::NoteClasses(kept, bytes);
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
    live.Assign(Graphics::RuntimeSwitch::DirectDraws, false);
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
}

void DirectStepClock::Report() const {
    for (std::size_t index = 0; index < nanoseconds.size(); ++index) {
        if (reached[index]) DirectDraws::AddStep(static_cast<Step>(index), nanoseconds[index]);
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
    if (!DirectShape(job)) return Counter::FallbackShape;
    auto& stores = DirectStores::Get();
    resolved.tag = stores.Follow(job.deviceSerial);
    const auto& shape = *job.shape;
    const auto& device = *anchor.device;
    resolved.decode = LeanRefreshedDecode(*shape.decode, job.words);
    const auto& decode = *resolved.decode;
    const auto& graphics = decode.state;
    const auto& programs = decode.programs;
    const auto& roles = decode.roles;
    const auto count = programs.size();
    resolved.memory.clear();
    AppendProgramMemory(resolved.memory, programs);
    resolved.linked.clear();
    for (std::size_t i = 0; i < count; ++i) {
        const auto& program = programs[i];
        resolved.linked.push_back({roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
    }
    resolved.vertexInfos.assign(count, std::nullopt);
    resolved.decodeReads.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& program = programs[i];
        auto& reads = resolved.decodeReads[i];
        reads.clear();
        if (program.binary.stage == Stage::Fragment) continue;
        auto& info = resolved.vertexInfos[i];
        info = DecodeOrderedVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, reads, 8u - program.firstUserSgpr);
        info->paClVsOutCntl = PositionExportLayout(program.snapshot.get(), program.codeOffset, program.binary.code, shape.reg207, program.tailContinuation.get()) | graphics.layerExports;
        info->depthRemapMode = graphics.depthRemap.mode;
        Graphics::ApplyPointSize(graphics, *info);
    }
    resolved.before = Graphics::Recorder::PendingWriteSnapshot();
    resolved.generation = GuestMemory::TrackerGeneration();
    resolved.stages.resize(count);
    resolved.regions.clear();
    lap(Step::Request);
    auto& cache = HostThreadLocal<InvocationCache, InvocationCacheStorage>();
    auto& reads = HostThreadLocal<DirectReads, DirectReadsStorage>();
    const bool libraries = device.GraphicsPipelineLibraries();
    const ShaderRecompiler::RecompileResult* previous = nullptr;
    std::uint32_t cursor = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& program = programs[i];
        auto& stage = resolved.stages[i];
        cursor = Graphics::StagePushOffset(cursor, program.binary.stage, libraries);
        stage.program = i;
        stage.stage = program.binary.stage;
        stage.requestOffset = cursor;
        stage.pushOffset = 0;
        stage.shared.reset();
        stage.capture.reset();
        const auto pixel = program.binary.stage == Stage::Fragment && previous != nullptr ? Graphics::DefaultUnexportedInputs(decode.pixel, previous->parameterExports) : decode.pixel;
        const auto request = [&] { return drawStageRequest(program, graphics, pixel, resolved.vertexInfos[i], resolved.memory, resolved.linked, device, cursor, job.parameters, std::nullopt); };
        auto built = request();
        lap(Step::Request);
        if (built.target.gpuResolvedImages) return Counter::FallbackHeap;

        std::optional<ShaderRecompiler::PreparedShaderInvocation> invocation;
        InvocationSlot* slot = nullptr;
        try {
            if (InvocationCacheable(program, built)) slot = &cache.For(program);
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
                if (slot != nullptr) {
                    const auto& context = invocation->Request().context;
                    *slot = {program.snapshot, program.codeOffset, invocation->Handle(), context.convertedSamples, context.layerRowSamples};
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
        if (!RangesSettled(stage.ranges, resolved.before.get(), resolved.generation)) return Counter::FallbackUnsettled;
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
        if (stage.shared->addressed) return Counter::FallbackAddressed;
        if (snapshot.srtPoison.size() + snapshot.nullRootReads != donor.poisonedSrtReads) return Counter::FallbackPoisoned;
        lap(Step::Key);

        if (!invocation->FillSharedSide(*stage.capture, donor, *stage.shared->patch, stage.side)) return Counter::FallbackSide;
        const auto slotEnd = (cursor / Graphics::PipelinePushSlotBytes + 1u) * Graphics::PipelinePushSlotBytes;
        if (donor.pushConstants.size() > slotEnd - cursor) return Counter::FallbackPush;
        stage.pushOffset = donor.pushConstants.empty() ? 0u : cursor;
        cursor += static_cast<std::uint32_t>(donor.pushConstants.size());
        previous = &donor;
        lap(Step::Side);
    }
    DrawVertexLayout(resolved.vertexInfos, resolved.vertexLayout, resolved.vertexInputs);
    auto& regions = resolved.regions;
    regions.assign(resolved.memory.begin(), resolved.memory.end());
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
    lap(Step::Checks);
    return std::nullopt;
}

void Driver::finishDirectDraw(const LeanDrawJob& job) {
    const auto& anchor = leanDraws.AnchorAt(job.anchor);
    DirectDraws::Add(Counter::Draws);
    struct ResolvedStorage {};
    auto& resolved = HostThreadLocal<DirectStages, ResolvedStorage>();
    struct SampleStorage {};
    auto& sampled = HostThreadLocal<std::uint64_t, SampleStorage>();
    std::optional<DirectStepClock> clock;
    if (sampled++ % DirectDraws::StepSampleEvery == 0) clock.emplace();
    std::optional<Counter> reason;
    try {
        reason = resolveDirectStages(job, anchor, resolved, clock.has_value() ? &*clock : nullptr);
    } catch (const std::exception&) {
        reason = Counter::FallbackWalk;
    }
    if (clock.has_value()) clock->Report();
    struct FallbackMark {
        ~FallbackMark() { ThreadReport::MarkItem(ThreadReport::ItemMark::PlanFallback); }
    } fallbackMark;
    DirectDraws::Add(reason.value_or(Counter::FallbackPending));
    finishLeanDraw(job, anchor);
}

}
