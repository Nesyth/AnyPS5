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
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/SlimMutex.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include "Optimization/SrtWalker.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include <cstdio>
#include <atomic>
#include <algorithm>
#include <new>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
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
    std::uint64_t variant = 0;
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

Counter BufferRefusal(Graphics::ShaderResources::PreparedBuffers::Refusal refusal) {
    using Refusal = Graphics::ShaderResources::PreparedBuffers::Refusal;
    switch (refusal) {
    case Refusal::Inaccessible: return Counter::BuffersInaccessible;
    case Refusal::UploadLimit: return Counter::BuffersUploadLimit;
    case Refusal::Budget: return Counter::BuffersBudget;
    default: return Counter::BuffersClassification;
    }
}

// The hit's buffers from its record's buffer plan, taken against the colour target the commit
// will draw with (commitRecordDraw hands the committer the job's decode state). The guest bytes
// are read here, ahead of the commit, so they follow the walk's settled rule: a write queued or
// pending before this draw may land before the commit, and once it retired no committer test
// sees it. The rule also covers the read itself: a write pending or queued when it began that
// retired by the end.
void prepareBuffers(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectStages& resolved, DrawRecordHit& hit) {
    auto plan = hit.record->commit->bufferPlan.load(std::memory_order_acquire);
    if (plan == nullptr) {
        DirectDraws::Add(Counter::BuffersNoPlan);
        return;
    }
    auto& prepared = hit.pooled ? hit.preparedBuffers.emplace(std::move(hit.spareBuffers)) : hit.preparedBuffers.emplace();
    prepared.generation = resolved.generation;
    const auto pending = Graphics::Recorder::PendingWriteSnapshot();
    std::optional<DrawPipeline::Watch> queued;
    if (DrawPipeline::Active()) queued = DrawPipeline::Queue0().WatchWrites();
    bool done = false;
    try {
        done = Graphics::ShaderResources::PrepareDirectBuffers(*plan, hit.stages, &job.shape->decode->state.color, prepared);
    } catch (const std::exception&) {
        prepared.refusal = Graphics::ShaderResources::PreparedBuffers::Refusal::Classification;
    }
    if (!done) {
        DirectDraws::Add(BufferRefusal(prepared.refusal));
        hit.preparedBuffers.reset();
        return;
    }
    struct ReadStorage {};
    auto& read = HostThreadLocal<AddressRanges, ReadStorage>();
    read.clear();
    for (const auto& item : prepared.selected) {
        if (item.begin != 0) read.emplace_back(item.begin, item.begin + item.bytes);
    }
    if (!read.empty() && (!RangesSettled(read, resolved.before.get(), resolved.generation, Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle) ? pending.get() : nullptr) || (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle) && !Graphics::Recorder::ReadsSettled(read, pending.get())) || (queued.has_value() && DrawPipeline::Queue0().WrittenSince(*queued, read)))) {
        DirectDraws::Add(Counter::BuffersUnsettled);
        hit.preparedBuffers.reset();
        return;
    }
    prepared.plan = std::move(plan);
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::AliasesAtPrepare)) prepared.aliasesClear = Graphics::DrawBufferAliasesClear(hit.stages, job.shape->decode->state.color, hit.parameters.indexAddress, static_cast<std::uint64_t>(hit.parameters.indexCount) * hit.parameters.indexSize);
    DirectDraws::Add(Counter::BuffersPrepared);
    DirectDraws::Add(Counter::BuffersBlobBytes, prepared.blob.size());
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitBindings)) return;
    bool placed = false;
    try {
        placed = anchor.device->PlaceDirectBuffers(prepared);
    } catch (const std::exception&) {
        prepared.placed.reset();
    }
    DirectDraws::Add(placed ? Counter::BindingsPlaced : Counter::BindingsNotPlaced);
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

void prepareInputs(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectStages& resolved, DrawRecordHit& hit) {
    auto recipe = hit.record->commit->inputsRecipe.load(std::memory_order_acquire);
    if (recipe == nullptr) {
        DirectDraws::Add(Counter::InputsNoRecipe);
        return;
    }
    auto& prepared = hit.pooled ? hit.preparedInputs.emplace(std::move(hit.spareInputs)) : hit.preparedInputs.emplace();
    const auto pending = Graphics::Recorder::PendingWriteSnapshot();
    std::optional<DrawPipeline::Watch> queued;
    if (DrawPipeline::Active()) queued = DrawPipeline::Queue0().WatchWrites();
    bool done = false;
    try {
        done = anchor.device->PrepareDrawInputs(job.shape->decode->state, hit.parameters, hit.stages, std::move(recipe), prepared);
    } catch (const std::exception&) {
        prepared.refusal = Graphics::PreparedInputs::Refusal::Thrown;
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
    if (!read.empty() && (!RangesSettled(read, resolved.before.get(), resolved.generation, Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle) ? pending.get() : nullptr) || (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle) && !Graphics::Recorder::ReadsSettled(read, pending.get())) || (queued.has_value() && DrawPipeline::Queue0().WrittenSince(*queued, read)))) {
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

void DirectClassStore::Evict() {
    std::lock_guard lock(mutex);
    DirectDraws::Add(Counter::ClassesEvicted, kept);
    classes.clear();
    kept = 0;
    bytes = 0;
    clears.fetch_add(1, std::memory_order_release);
    note();
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
    for (const auto id : {Graphics::RuntimeSwitch::DirectDraws, Graphics::RuntimeSwitch::DirectCleanProofs, Graphics::RuntimeSwitch::DirectPassTargets, Graphics::RuntimeSwitch::CleanCollects, Graphics::RuntimeSwitch::DirectBuffers, Graphics::RuntimeSwitch::PipelineDispatch, Graphics::RuntimeSwitch::PipelineDispatchMisses, Graphics::RuntimeSwitch::XpCommitFloor, Graphics::RuntimeSwitch::MsaaRecords, Graphics::RuntimeSwitch::XpInputsAsIs, Graphics::RuntimeSwitch::XpStampsByEpoch, Graphics::RuntimeSwitch::MsaaRecordHold, Graphics::RuntimeSwitch::TemplateTouch, Graphics::RuntimeSwitch::DeferredEncode, Graphics::RuntimeSwitch::XpProofAsIs, Graphics::RuntimeSwitch::DirectInputs, Graphics::RuntimeSwitch::DirectRepeat, Graphics::RuntimeSwitch::CleanResidentViews, Graphics::RuntimeSwitch::CleanPendingRanges, Graphics::RuntimeSwitch::RefusedCollectMemo, Graphics::RuntimeSwitch::DirectRetry, Graphics::RuntimeSwitch::ItemsByTurn, Graphics::RuntimeSwitch::XpNoCommits, Graphics::RuntimeSwitch::FourthFinisher, Graphics::RuntimeSwitch::FifthFinisher, Graphics::RuntimeSwitch::DirectRepeatSets, Graphics::RuntimeSwitch::DirectRepeatStores, Graphics::RuntimeSwitch::DirectRepeatShared, Graphics::RuntimeSwitch::WorkerCore, Graphics::RuntimeSwitch::LeanListBatch, Graphics::RuntimeSwitch::LeanListDirect, Graphics::RuntimeSwitch::FinisherDeposit, Graphics::RuntimeSwitch::CommitBindings, Graphics::RuntimeSwitch::CommitBindingsStamps, Graphics::RuntimeSwitch::CommitBindingsSets, Graphics::RuntimeSwitch::CommitBindingsDynamic, Graphics::RuntimeSwitch::CommitFragments, Graphics::RuntimeSwitch::ExpressCommits, Graphics::RuntimeSwitch::XpStreamSegments, Graphics::RuntimeSwitch::XpNoEagerStore, Graphics::RuntimeSwitch::LabelOrderedStores, Graphics::RuntimeSwitch::DeeperRing, Graphics::RuntimeSwitch::BesideSampleDumps, Graphics::RuntimeSwitch::DispatchBeside, Graphics::RuntimeSwitch::FaultFragments, Graphics::RuntimeSwitch::StageMemo, Graphics::RuntimeSwitch::KeepFaultSlots, Graphics::RuntimeSwitch::SiblingFinishers, Graphics::RuntimeSwitch::FastSettle, Graphics::RuntimeSwitch::XpSlowFinisher, Graphics::RuntimeSwitch::XpSlowCommitter, Graphics::RuntimeSwitch::XpSlowWorker, Graphics::RuntimeSwitch::XpExpressFloor, Graphics::RuntimeSwitch::LeanExpress, Graphics::RuntimeSwitch::XpAppendCommits, Graphics::RuntimeSwitch::PreciseSettle, Graphics::RuntimeSwitch::AppendCommits, Graphics::RuntimeSwitch::StageMemoPatchVertex, Graphics::RuntimeSwitch::StageMemoPatchBuffers, Graphics::RuntimeSwitch::StageMemoPatchData, Graphics::RuntimeSwitch::RunFollow, Graphics::RuntimeSwitch::TurnWaitsByJob, Graphics::RuntimeSwitch::ExpressKeepState, Graphics::RuntimeSwitch::StageMemoFront, Graphics::RuntimeSwitch::StageMemoAdmitThird, Graphics::RuntimeSwitch::CommitStamps, Graphics::RuntimeSwitch::PassReadsOnce, Graphics::RuntimeSwitch::ReapThrottle, Graphics::RuntimeSwitch::OcclusionWindows, Graphics::RuntimeSwitch::ExpressPassStamps, Graphics::RuntimeSwitch::PassBarriersMerged, Graphics::RuntimeSwitch::LabelReadbackMemo, Graphics::RuntimeSwitch::ExpressTargetUploads, Graphics::RuntimeSwitch::DrawRuns, Graphics::RuntimeSwitch::LargeBatches, Graphics::RuntimeSwitch::ReplayStateDedupe, Graphics::RuntimeSwitch::RunFollowNoCarry, Graphics::RuntimeSwitch::RunFollowPlainLoop, Graphics::RuntimeSwitch::FinisherParkSlots, Graphics::RuntimeSwitch::ListHooksOnce, Graphics::RuntimeSwitch::DispatchEnvOnce, Graphics::RuntimeSwitch::PlainRegisterPackets, Graphics::RuntimeSwitch::UserWordBlocks, Graphics::RuntimeSwitch::XpDeviceUploads, Graphics::RuntimeSwitch::XpDepthLayouts, Graphics::RuntimeSwitch::XpTargetRestarts, Graphics::RuntimeSwitch::TargetRestarts}) live.Assign(id, false);
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

    bool operator==(const StageMemoRoot& other) const = default;
};

struct StageMemoWord {
    std::uint32_t key = 0xffffffffu;
    std::uint8_t kind = 0;

    bool operator==(const StageMemoWord& other) const = default;
};

struct StageMemoPatch {
    std::uint32_t word = 0;
    std::uint32_t binding = 0;
    std::uint32_t position = 0;
    std::uint32_t mask = 0;
};

constexpr std::uint8_t StageMemoVertexKind = 1;

std::uint32_t stageMemoPatchKinds() {
    using Graphics::RuntimeSwitch;
    using Graphics::RuntimeSwitches::Set;
    return (Set(RuntimeSwitch::StageMemoPatchVertex) ? 1u << StageMemoVertexKind : 0u) | (Set(RuntimeSwitch::StageMemoPatchBuffers) ? 1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::BufferAddress) : 0u) |
           (Set(RuntimeSwitch::StageMemoPatchData) ? (1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::BufferSize)) | (1u << static_cast<std::uint32_t>(ShaderRecompiler::SideReadKind::FlatWord)) : 0u);
}

struct StageMemoPattern {
    std::uint64_t relocatable = 0;
    std::uint64_t values = 0;
    std::uint64_t bases = 0;
    std::uint64_t addresses = 0;
    std::uint32_t userWords = 0;
    std::uint32_t decodeRoots = 0;
    std::uint32_t patchKinds = 0;
    std::vector<StageMemoWord> words;
    std::vector<StageMemoRoot> roots;

    bool operator==(const StageMemoPattern& other) const = default;
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
};

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
        std::lock_guard lock(held.mutex);
        if (const auto found = held.current.find(key); found != held.current.end()) return found->second;
        const auto old = held.previous.find(key);
        if (old == held.previous.end()) return nullptr;
        auto entry = std::move(old->second);
        held.previous.erase(old);
        place(held, key, entry);
        return entry;
    }

    std::optional<std::size_t> Way(std::uint64_t key, const StageMemoPattern& pattern) {
        auto& held = shard(key);
        std::lock_guard lock(held.mutex);
        auto& ways = held.patterns[key];
        if (ways != nullptr) {
            const auto way = static_cast<std::size_t>(std::ranges::find(*ways, pattern) - ways->begin());
            if (way != ways->size()) return way;
            if (ways->size() >= Ways) return std::nullopt;
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

    void Insert(std::uint64_t key, std::shared_ptr<const StageMemoEntry> entry) {
        auto& held = shard(key);
        std::lock_guard lock(held.mutex);
        held.previous.erase(key);
        place(held, key, entry);
    }

private:
    StageMemoShard& shard(std::uint64_t key) { return shards[static_cast<std::size_t>((key ^ (key >> 23u)) % Shards)]; }

    void place(StageMemoShard& held, std::uint64_t key, const std::shared_ptr<const StageMemoEntry>& entry) {
        if (held.current.size() >= ShardBound && !held.current.contains(key)) {
            held.previous = std::move(held.current);
            held.current.clear();
            DirectDraws::Add(Counter::StageMemoCleared);
        }
        held.current.insert_or_assign(key, entry);
    }

    static constexpr std::size_t SeenKeys = std::size_t{1} << 17u;
    std::array<StageMemoShard, Shards> shards;
    std::atomic<std::uint64_t> patternVersion{1};
    std::unique_ptr<std::atomic<std::uint64_t>[]> seen{new std::atomic<std::uint64_t>[SeenKeys]()};
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
        const bool guarded = word.kind == static_cast<std::uint8_t>(ShaderRecompiler::SideReadKind::BufferAddress) || word.kind == static_cast<std::uint8_t>(ShaderRecompiler::SideReadKind::BufferSize);
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
        const bool guarded = word.kind == static_cast<std::uint8_t>(ShaderRecompiler::SideReadKind::BufferAddress) || word.kind == static_cast<std::uint8_t>(ShaderRecompiler::SideReadKind::BufferSize);
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
    static std::mutex mutex;
    static std::unordered_map<std::uint64_t, StageMemoDiagLast> last;
    static std::array<std::uint64_t, Causes> causes{};
    static std::uint64_t total = 0;
    static std::uint32_t printed = 0;
    auto identity = (key ^ way) * StageMemoPrime;
    for (const auto value : gathered) identity = (identity ^ value) * StageMemoPrime;
    std::lock_guard lock(mutex);
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
    struct RootHistory {
        std::array<std::vector<std::uint32_t>, 16> kept;
        std::size_t next = 0;
    };
    static std::unordered_map<std::uint64_t, RootHistory> lastRoots;
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
                        std::snprintf(line, sizeof(line), " root %zu%s user %u +0x%llx word %u: %08x -> %08x;", root, root < pattern.decodeRoots ? " (decode)" : "", static_cast<unsigned>(pattern.roots[root].origin), static_cast<unsigned long long>(pattern.roots[root].offset), static_cast<unsigned>(word), before[at], gathered[at]);
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

StageMemoOutcome adoptStage(StageMemo& memo, const LeanDrawJob& job, const DrawProgram& program, std::size_t index, std::uint32_t cursor, const InvocationSlot& slot, DirectStages& resolved) {
    const auto& decode = job.shape->decode;
    const auto key = stageMemoKey(decode.get(), index);
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
    memo.patched = false;
    const std::span<const std::uint32_t> userData = program.userData;
    auto& stage = resolved.stages[index];
    const DirectStage* previous = index != 0 ? resolved.stages[index - 1].shared.get() : nullptr;
    auto last = StageMemoOutcome::NoEntry;
    for (std::size_t way = 0; way < ways->size(); ++way) {
        const auto& pattern = (*ways)[way];
        if (pattern.userWords != userData.size() || (pattern.patchKinds & ~enabledKinds) != 0u) continue;
        auto& gathered = memo.gathered;
        auto& addresses = memo.addresses;
        gathered.clear();
        addresses.clear();
        std::uint64_t page = ~std::uint64_t{0};
        bool readable = true;
        for (const auto& root : pattern.roots) {
            const auto word = userData[root.origin];
            const auto address = (root.aligned != 0 ? static_cast<std::uint64_t>(word) & ~std::uint64_t{3} : static_cast<std::uint64_t>(word)) + root.offset;
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
        if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoFront)) {
            if (memo.front.empty()) memo.front.resize(StageMemo::FrontSets * 2u);
            auto* set = memo.front.data() + static_cast<std::size_t>((contentKey ^ (contentKey >> 19u)) % StageMemo::FrontSets) * 2u;
            auto* slot = set[0].key == contentKey && set[0].entry != nullptr ? &set[0] : set[1].key == contentKey && set[1].entry != nullptr ? &set[1] : nullptr;
            if (slot != nullptr) {
                held = slot->entry.get();
                if ((++slot->uses & 63u) == 0u && store.Find(contentKey) == nullptr) store.Insert(contentKey, slot->entry);
            } else if ((fetched = store.Find(contentKey)) != nullptr) {
                auto& victim = set[0].uses <= set[1].uses ? set[0] : set[1];
                set[0].uses >>= 1u;
                set[1].uses >>= 1u;
                victim.key = contentKey;
                victim.uses = 1;
                victim.entry = std::move(fetched);
                held = victim.entry.get();
            }
        } else {
            fetched = store.Find(contentKey);
            held = fetched.get();
        }
        if (held == nullptr) {
            last = StageMemoOutcome::NoEntry;
            memo.missKey = contentKey;
            if (stageMemoDiag()) stageMemoDiagnose(key, way, pattern, userData, gathered, job.parameters, cursor, previous);
            continue;
        }
        const auto& entry = *held;
        const auto& parameters = job.parameters;
        bool same = entry.decode.get() == decode.get() && entry.stage == index && entry.way == way && entry.cursor == cursor && entry.tag == resolved.tag && entry.previous.get() == previous && (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoNoIndex) || (entry.indexAddress == parameters.indexAddress && entry.indexCount == parameters.indexCount && entry.indexSize == parameters.indexSize && entry.instanceCount == parameters.instanceCount)) && entry.variant == slot.variant && entry.convertedSamples == slot.convertedSamples && entry.layerRowSamples == slot.layerRowSamples && entry.userData.size() == userData.size();
        for (std::size_t word = 0; same && word < userData.size(); ++word) {
            if (!stageMemoRelocates(pattern.relocatable, word)) same = entry.userData[word] == userData[word];
            else if (((pattern.addresses >> word) & 1u) != 0) same = userData[word] != 0u && ((userData[word] ^ entry.userData[word]) & 3u) == 0u;
            else if (((pattern.bases >> word) & 1u) != 0) same = userData[word] != 0u;
        }
        if (!same) {
            last = StageMemoOutcome::Inputs;
            continue;
        }
        const auto sameRoots = stageMemoSameRoots(pattern, entry.rootValues, gathered);
        if (sameRoots == 0) {
            last = StageMemoOutcome::Values;
            continue;
        }
        bool nestedSame = true;
        for (const auto& [address, value] : entry.nested) {
            std::uint32_t live = 0;
            if (!stageMemoRead(address, live, page) || live != value) {
                nestedSame = false;
                break;
            }
        }
        if (!nestedSame) {
            last = StageMemoOutcome::Values;
            continue;
        }
        if (!CaptureRegisteredViewsCurrent(entry.capture->snapshot.registeredImages)) {
            last = StageMemoOutcome::Views;
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
        for (std::size_t read = 0; read < entry.decodeReads.size(); ++read) decodeRanges.emplace_back(addresses[read], addresses[read] + entry.decodeReads[read].size);
        if (decodeRanges.empty()) {
            if (!RangesSettled(stage.ranges, resolved.before.get(), resolved.generation)) return StageMemoOutcome::Unsettled;
        } else if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle)) {
            if (!RangesSettled(stage.ranges, resolved.before.get(), resolved.generation) || !RangesSettled(decodeRanges, resolved.before.get(), resolved.generation)) return StageMemoOutcome::Unsettled;
        } else {
            auto& settled = memo.settled;
            settled.assign(stage.ranges.begin(), stage.ranges.end());
            settled.insert(settled.end(), decodeRanges.begin(), decodeRanges.end());
            if (!RangesSettled(settled, resolved.before.get(), resolved.generation)) return StageMemoOutcome::Unsettled;
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
    return last;
}

std::optional<Counter> rememberStage(const LeanDrawJob& job, const DrawProgram& program, std::size_t index, std::uint32_t cursor, const InvocationSlot& slot, const ShaderRecompiler::PreparedShaderInvocation& invocation, const DirectReads& reads, const DirectStages& resolved) {
    const auto& stage = resolved.stages[index];
    const auto& use = stage.capture->userDataUse;
    const std::span<const std::uint32_t> userData = program.userData;
    if (userData.size() > 0xffffu) return Counter::StageMemoUntraced;
    StageMemoPattern pattern;
    StageMemoEntry entry;
    const auto enabledKinds = stageMemoPatchKinds();
    std::vector<StageMemoWord> words;
    entry.forwards = stageForwards(stage.shared, invocation, *stage.capture);
    if (entry.forwards == nullptr) return Counter::StageMemoUntraced;
    pattern.userWords = static_cast<std::uint32_t>(userData.size());
    const auto addresses = entry.forwards->addressWords;
    auto relocatable = ~use.asValue | addresses;
    std::uint64_t pinned = 0;
    const auto& info = resolved.vertexInfos[index];
    const auto& decodeReads = resolved.decodeReads[index];
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
    struct Placed {
        std::uint64_t address;
        std::uint32_t slot;
        bool nested;
    };
    std::vector<Placed> placed;
    placed.reserve(use.reads.size());
    for (const auto& [address, origin] : use.reads) {
        if (std::ranges::any_of(placed, [&](const Placed& known) { return known.address == address; })) continue;
        const auto found = std::lower_bound(reads.words.begin(), reads.words.end(), std::pair<std::uint64_t, std::uint32_t>{address, 0u});
        if (found == reads.words.end() || found->first != address) return Counter::StageMemoUntraced;
        if (origin >= 0) {
            if (static_cast<std::size_t>(origin) >= userData.size()) return Counter::StageMemoUntraced;
            placed.push_back({address, static_cast<std::uint32_t>(pattern.roots.size() - pattern.decodeRoots), false});
            pattern.roots.push_back({address - (static_cast<std::uint64_t>(userData[static_cast<std::size_t>(origin)]) & ~std::uint64_t{3}), static_cast<std::uint16_t>(origin), 1, 1});
            entry.rootValues.push_back(found->second);
            words.push_back({});
        } else {
            placed.push_back({address, static_cast<std::uint32_t>(entry.nested.size()), true});
            entry.nested.emplace_back(address, found->second);
        }
    }
    if (placed.size() != reads.words.size()) return Counter::StageMemoUntraced;
    const auto& readForwards = entry.forwards->reads;
    if ((enabledKinds & ~(1u << StageMemoVertexKind)) != 0u && !readForwards.empty() && reads.nodes.size() == use.reads.size()) {
        struct Granted {
            std::uint64_t address;
            std::uint32_t key;
            std::uint8_t kind;
            bool pinned;
        };
        struct ByNode {
            bool operator()(const ShaderRecompiler::SideReadForward& forward, std::uint32_t node) const { return forward.node < node; }
            bool operator()(std::uint32_t node, const ShaderRecompiler::SideReadForward& forward) const { return node < forward.node; }
        };
        std::vector<Granted> granted;
        for (std::size_t at = 0; at < use.reads.size(); ++at) {
            const auto address = use.reads[at].first;
            auto slot = std::find_if(granted.begin(), granted.end(), [&](const Granted& known) { return known.address == address; });
            if (slot == granted.end()) slot = granted.insert(granted.end(), Granted{address, 0u, 0u, false});
            const auto node = reads.nodes[at];
            const auto [first, last] = std::equal_range(readForwards.begin(), readForwards.end(), node, ByNode{});
            if (node == ShaderRecompiler::SrtNoReadNode || first == last) {
                slot->pinned = true;
                continue;
            }
            for (auto forward = first; forward != last; ++forward) {
                const auto kind = static_cast<std::uint8_t>(forward->kind);
                if (slot->kind != 0u && slot->kind != kind) slot->pinned = true;
                slot->kind = kind;
                slot->key |= forward->keyMask;
            }
        }
        for (const auto& item : placed) {
            if (item.nested) continue;
            const auto slot = std::find_if(granted.begin(), granted.end(), [&](const Granted& known) { return known.address == item.address; });
            if (slot == granted.end() || slot->pinned || slot->kind == 0u || (enabledKinds & (1u << slot->kind)) == 0u || slot->key == 0xffffffffu) continue;
            const auto wordIndex = entry.decodeWords + item.slot;
            const auto value = entry.rootValues[wordIndex];
            const auto patchMask = ~slot->key;
            const bool guarded = slot->kind == static_cast<std::uint8_t>(ShaderRecompiler::SideReadKind::BufferAddress) || slot->kind == static_cast<std::uint8_t>(ShaderRecompiler::SideReadKind::BufferSize);
            if (guarded && (value & patchMask) == 0u) continue;
            bool fits = true;
            const auto patchesBefore = entry.patches.size();
            for (std::size_t at = 0; fits && at < use.reads.size(); ++at) {
                if (use.reads[at].first != item.address) continue;
                const auto [first, last] = std::equal_range(readForwards.begin(), readForwards.end(), reads.nodes[at], ByNode{});
                for (auto forward = first; fits && forward != last; ++forward) {
                    fits = forward->binding < stage.side.bindings.size() && forward->position < stage.side.bindings[forward->binding].size() && ((stage.side.bindings[forward->binding][forward->position] ^ value) & patchMask) == 0u;
                    if (fits) entry.patches.push_back({wordIndex, forward->binding, forward->position, patchMask});
                }
            }
            if (!fits) {
                entry.patches.resize(patchesBefore);
                continue;
            }
            words[wordIndex] = {slot->key, slot->kind};
            pattern.patchKinds |= 1u << slot->kind;
        }
    }
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
    std::sort(placed.begin(), placed.end(), [](const Placed& a, const Placed& b) { return a.address < b.address; });
    entry.order.reserve(placed.size());
    for (const auto& item : placed) entry.order.push_back(item.nested ? walkRoots + item.slot : item.slot);
    const auto key = stageMemoKey(job.shape->decode.get(), index);
    auto& store = stageMemoStore();
    const auto placedWay = store.Way(key, pattern);
    if (!placedWay.has_value()) return Counter::StageMemoPatternsFull;
    const auto way = *placedWay;
    entry.decode = job.shape->decode;
    entry.stage = static_cast<std::uint32_t>(index);
    entry.way = static_cast<std::uint32_t>(way);
    entry.cursor = cursor;
    entry.pushOffset = stage.pushOffset;
    entry.tag = resolved.tag;
    if (index != 0) entry.previous = resolved.stages[index - 1].shared;
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
    entry.side = stage.side;
    entry.vertexInfo = info;
    entry.decodeReads = decodeReads;
    const auto contentKey = stageMemoContentKey(key, way, pattern, userData, entry.rootValues, job.parameters, cursor, entry.previous.get());
    store.Insert(contentKey, std::make_shared<const StageMemoEntry>(std::move(entry)));
    return Counter::StageMemoStored;
}

struct StageMemoKept {
    DirectResolvedStage stage;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertexInfo;
    std::vector<Graphics::DecodeRead> decodeReads;
    bool patched = false;
};

const char* stageMemoDifference(const StageMemoKept& kept, const DirectResolvedStage& stage, const std::optional<ShaderRecompiler::ShaderVertexStageInfo>& vertexInfo, const std::vector<Graphics::DecodeRead>& decodeReads) {
    if (kept.stage.shared.get() != stage.shared.get()) return "the shared stage";
    if (kept.stage.pushOffset != stage.pushOffset) return "the push offset";
    if (kept.stage.ranges != stage.ranges) return "the read ranges";
    if (kept.stage.values != stage.values) return "the read values";
    const auto& a = kept.stage.side;
    const auto& b = stage.side;
    if (a.userData != b.userData || a.flattenedSrt != b.flattenedSrt || a.bindings != b.bindings || a.pushWords != b.pushWords) return "the side words";
    if (a.vertexAttributes.size() != b.vertexAttributes.size()) return "the side vertex attributes";
    for (std::size_t at = 0; at < a.vertexAttributes.size(); ++at) {
        const auto& x = a.vertexAttributes[at];
        const auto& y = b.vertexAttributes[at];
        if (x.location != y.location || x.components != y.components || x.fetchIndex != y.fetchIndex || x.formatComponents != y.formatComponents || x.resource.fields != y.resource.fields) return "the side vertex attributes";
    }
    const auto& ours = kept.stage.capture->snapshot.registeredImages;
    const auto& theirs = stage.capture->snapshot.registeredImages;
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
    resolved.decode = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::LeanPrepare) ? refreshedDecode(shape.decode, job.words) : LeanRefreshedDecode(*shape.decode, job.words);
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
    const auto decodeVertex = [&](std::size_t i) {
        const auto& program = programs[i];
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
    auto* const memo = stageMemoOn() ? &HostThreadLocal<StageMemo, StageMemoStorage>() : nullptr;
    if (memo != nullptr) {
        for (auto& reads : resolved.decodeReads) reads.clear();
    } else {
        for (std::size_t i = 0; i < count; ++i) decodeVertex(i);
    }
    resolved.before = Graphics::Recorder::PendingWriteSnapshot();
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
        const auto& program = programs[i];
        auto& stage = resolved.stages[i];
        cursor = Graphics::StagePushOffset(cursor, program.binary.stage, libraries);
        stage.program = i;
        stage.stage = program.binary.stage;
        stage.requestOffset = cursor;
        stage.pushOffset = 0;
        stage.shared.reset();
        stage.capture.reset();
        const auto cursorBefore = cursor;
        std::optional<StageMemoKept> memoKept;
        bool memoAdmitted = true;
        const bool memoStage = memo != nullptr && stageMemoCacheable(program);
        if (memo != nullptr) {
            const auto& known = cache.For(program);
            if (memoStage && known.handle != nullptr && known.snapshot.get() == program.snapshot.get() && known.codeOffset == program.codeOffset) {
                const auto outcome = adoptStage(*memo, job, program, i, cursor, known, resolved);
                DirectDraws::Add(stageMemoCounter(outcome));
                if (outcome == StageMemoOutcome::NoEntry) memoAdmitted = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoAdmitFirst) || stageMemoStore().Admit(memo->missKey, Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::StageMemoAdmitThird));
                if (outcome == StageMemoOutcome::Hit) {
                    const auto every = stageMemoVerifyEvery();
                    if (memo->patched) DirectDraws::Add(Counter::StageMemoPatched);
                    if (every == 0 || (memo->patched ? ++memo->patchedOrdinal : ++memo->verifyOrdinal) % every != 0) {
                        const auto& adopted = *stage.shared->donor;
                        cursor += static_cast<std::uint32_t>(adopted.pushConstants.size());
                        previous = &adopted;
                        lap(Step::Side);
                        continue;
                    }
                    memoKept.emplace(StageMemoKept{stage, resolved.vertexInfos[i], resolved.decodeReads[i], memo->patched});
                    stage.shared.reset();
                    stage.capture.reset();
                }
            }
            decodeVertex(i);
        }
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
                    *slot = {program.snapshot, program.codeOffset, invocation->Handle(), context.convertedSamples, context.layerRowSamples, invocation->VariantId()};
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
        if (!RangesSettled(stage.ranges, resolved.before.get(), resolved.generation)) {
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
        if (stage.shared->addressed) return Counter::FallbackAddressed;
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
            if (const char* what = stageMemoDifference(*memoKept, stage, resolved.vertexInfos[i], resolved.decodeReads[i]); what != nullptr) {
                DirectDraws::Add(Counter::StageMemoVerifyDiffering);
                static std::atomic<std::uint32_t> printed{0};
                if (printed.fetch_add(1, std::memory_order_relaxed) < 32) {
                    const auto* kept = memoKept->stage.shared.get();
                    const auto* fresh = stage.shared.get();
                    const auto wide = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
                    std::fprintf(stderr, "[stage-memo-verify] %s differs: stage %zu of a draw with %zu stages (kept variant %llx key %llx, fresh variant %llx key %llx, donor %s, patch %s, read ranges %s, read values %s, registered images %zu and %zu)\n", what, i, count, wide(kept != nullptr ? kept->variant : 0), wide(kept != nullptr ? kept->key : 0), wide(fresh != nullptr ? fresh->variant : 0), wide(fresh != nullptr ? fresh->key : 0), kept != nullptr && fresh != nullptr && kept->donor.get() == fresh->donor.get() ? "same" : "other", kept != nullptr && fresh != nullptr && kept->patch.get() == fresh->patch.get() ? "same" : "other", memoKept->stage.ranges == stage.ranges ? "same" : "other", memoKept->stage.values == stage.values ? "same" : "other", memoKept->stage.capture != nullptr ? memoKept->stage.capture->snapshot.registeredImages.size() : std::size_t{0}, stage.capture != nullptr ? stage.capture->snapshot.registeredImages.size() : std::size_t{0});
                }
            }
        } else if (memoStage && memoAdmitted && slot != nullptr && slot->handle != nullptr) {
            if (const auto stored = rememberStage(job, program, i, cursorBefore, *slot, *invocation, reads, resolved)) DirectDraws::Add(*stored);
        }
        if (resolveCensus()) noteStageCensus(*invocation, *stage.capture, reads, resolved.decodeReads[i]);
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
    if (resolveCensus()) noteResolveCensus(job, resolved);
    lap(Step::Checks);
    return std::nullopt;
}

std::optional<Counter> Driver::assembleDirectHit(const LeanDrawJob& job, DirectStages& resolved, DrawRecordHit& hit) {
    const auto count = resolved.stages.size();
    hit.sides.resize(count);
    hit.captures.resize(count);
    hit.stages.clear();
    hit.stages.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto& stage = resolved.stages[i];
        if (hit.pooled) hit.sides[i] = stage.side;
        else hit.sides[i] = std::move(stage.side);
        hit.captures[i] = stage.capture;
        hit.stages.push_back({stage.stage, stage.shared->donor.get(), stage.pushOffset, &hit.captures[i]->snapshot.registeredImages, &hit.sides[i]});
    }
    hit.parameters = job.parameters;
    const auto& front = *resolved.stages.front().shared->donor;
    const auto& program = resolved.decode->programs.front();
    const auto userWord = [&](std::int32_t sgpr, std::uint32_t& value) {
        if (sgpr < 0 || static_cast<std::uint32_t>(sgpr) < program.firstUserSgpr) return false;
        const auto index = static_cast<std::uint32_t>(sgpr) - program.firstUserSgpr;
        if (index >= program.userData.size()) return false;
        value = program.userData[index];
        return true;
    };
    std::uint32_t offset = 0;
    if (front.vertexOffsetSgpr >= 0) {
        if (!userWord(front.vertexOffsetSgpr, offset) || offset > std::numeric_limits<std::uint32_t>::max() - hit.parameters.firstVertex) return Counter::FallbackOverflow;
        hit.parameters.firstVertex += offset;
    }
    if (front.instanceOffsetSgpr >= 0) {
        if (!userWord(front.instanceOffsetSgpr, offset)) return Counter::FallbackOverflow;
        hit.parameters.firstInstance = offset;
    }
    hit.reads = QueuedDrawReadRanges(hit.parameters, hit.stages);
    runCaptureCheckHook();
    if (capturedMemoryChanged(resolved.regions)) return Counter::FallbackMemoryChanged;
    if (recordQueuedLabelsAfterCapture(job.queue, resolved.regions)) return Counter::FallbackLabels;
    return std::nullopt;
}

const DrawRecordSink* Driver::finishDirectAnswer(const LeanDrawJob& job, const LeanAnchor& anchor, const DirectStages& resolved, const DrawRecordHit& hit, DrawRecordView& answer) {
    bool answered = false;
    try {
        answer.Capture(hit.stages, hit.parameters);
        answer.reads = hit.reads;
        answer.contentKey = anchor.device->DrawContentKey(resolved.decode->state, hit.parameters, hit.stages);
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
    for (std::size_t i = 0; i < resolved.stages.size(); ++i) entry->stages[i] = resolved.stages[i].shared;
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

void Driver::finishDirectDraw(const LeanDrawJob& job) {
    const auto& anchor = leanDraws.AnchorAt(job.anchor);
    DirectDraws::Add(Counter::Draws);
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
    try {
        reason = resolveDirectStages(job, anchor, resolved, timer);
        for (std::uint32_t again = 0; again < 2u && reason == Counter::FallbackUnsettled && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectRetry); ++again) reason = resolveDirectStages(job, anchor, resolved, timer);
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
        else if (!reason && (job.recordFlags & DrawRecordFlags::VerifyDue) != 0) reason = Counter::FallbackVerify;
    } catch (const std::exception&) {
        reason = Counter::FallbackWalk;
    }
    if (!reason) {
        hit->record = found->record;
        hit->owner = found;
        hit->direct = true;
        if ((job.recordFlags & DrawRecordFlags::DirectBuffers) != 0) {
            prepareBuffers(job, anchor, resolved, *hit);
            lap(Step::Buffers);
        }
        if ((job.recordFlags & DrawRecordFlags::DirectInputs) != 0) {
            prepareInputs(job, anchor, resolved, *hit);
            lap(Step::Inputs);
        }
        if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitFragments)) recordFragment(job, anchor, *hit);
        commitRecordDraw(job, anchor, std::move(hit));
        lap(Step::HandOff);
        if (timer != nullptr) timer->Report();
        DirectDraws::Add(Counter::Direct);
        ThreadReport::MarkItem(ThreadReport::ItemMark::Planned);
        return;
    }
    if (timer != nullptr) timer->Report();
    struct FallbackMark {
        ~FallbackMark() { ThreadReport::MarkItem(ThreadReport::ItemMark::PlanFallback); }
    } fallbackMark;
    DirectDraws::Add(*reason);
    if (*reason == Counter::FallbackClassNew) {
        buildDirectClass(job, anchor, key, probe, resolved, *hit);
        return;
    }
    if (*reason == Counter::FallbackVerify) {
        verifyDirectHit(job, anchor, *found, resolved, *hit);
        return;
    }
    finishLeanDraw(job, anchor);
}

void Driver::runLeanRun(const std::byte* payload) {
    auto& driver = Get();
    const auto& header = *std::launder(reinterpret_cast<const LeanRunHeader*>(payload));
    if (header.count > 1 && ThreadReport::Enabled()) ThreadReport::Add(ThreadReport::Count::FinisherJobs, header.count - 1);
    for (std::uint32_t index = 0; index < header.count; ++index) driver.finishDirectDraw(header.jobs[index]);
}

}
