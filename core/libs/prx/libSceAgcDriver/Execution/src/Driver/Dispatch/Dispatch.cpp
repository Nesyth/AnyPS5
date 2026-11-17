#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/HeapStages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BindlessHeap.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/BufferCopy.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DwordPatternFill.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

bool DispatchItemsEnabled() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PipelineDispatch);
}

bool dumpsDispatchProgram(std::uint64_t address) {
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DispatchEnvOnce)) {
        static const std::optional<std::uint64_t> watched = []() -> std::optional<std::uint64_t> {
            const char* text = std::getenv("APS5_DUMP_DISPATCH_PROGRAM");
            if (text == nullptr) return std::nullopt;
            return std::strtoull(text, nullptr, 0);
        }();
        return watched.has_value() && *watched == address;
    }
    const char* watched = std::getenv("APS5_DUMP_DISPATCH_PROGRAM");
    return watched != nullptr && std::strtoull(watched, nullptr, 0) == address;
}

bool awaitRegionWriters(std::span<const ShaderRecompiler::MemoryRegion> regions, DeepSite site) {
    std::vector<DrawPipeline::Range> ranges;
    ranges.reserve(regions.size());
    for (const auto& region : regions) ranges.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
    auto& pipeline = DrawPipeline::Queue0();
    while (pipeline.Busy() && pipeline.OverlapsAny(ranges)) pipeline.AwaitWriters(ranges, DrawPipeline::DrainReason::Packet, 0x15, site);
    return true;
}

enum class ItemFallback : std::size_t { Miss, Reads, NoRecipe, Writes, Verify, Count };
constexpr std::array<const char*, static_cast<std::size_t>(ItemFallback::Count)> ItemFallbackNames{"miss", "reads", "no recipe", "writes", "verify"};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ItemFallback::Count)> itemFallbacks{};
std::atomic<std::uint64_t> itemCommits{0};
std::atomic<std::uint64_t> itemRebuilds{0};
std::atomic<std::uint64_t> itemMisses{0};
std::atomic<std::uint64_t> itemRecipeLess{0};
std::atomic<std::uint64_t> itemRelocated{0};
std::atomic<std::uint64_t> itemNotPlain{0};
std::atomic<std::uint64_t> itemFixups{0};
enum class NotPlain : std::size_t { NoResult, HtileClear, KeysFill, DepthClear, Count };
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(NotPlain::Count)> itemNotPlainKinds{};

bool CensusOn() {
    static const bool on = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return on;
}

template<typename TEnum>
constexpr std::size_t CensusIndex(TEnum value) {
    return static_cast<std::size_t>(value);
}

enum class CensusOutcome : std::size_t { Other, Clear, Fill, Copy, Adopted, Exact, Data, Patched, Relocated, Miss, Count };
constexpr std::array<const char*, CensusIndex(CensusOutcome::Count)> CensusOutcomeNames{"other", "multisample clear", "fill", "copy", "adopted ahead", "exact hit", "data hit", "patched address hit", "relocated hit", "miss"};
enum class CensusMiss : std::size_t { Other, CacheOff, NotCacheable, Evicted, NewProgram, NotKept, NoEntry, Differing, ViewsChanged, QueuedStores, PendingWrites, QueuedLabel, Unmapped, Forgotten, DeviceImages, Count };
constexpr std::array<const char*, CensusIndex(CensusMiss::Count)> CensusMissNames{"other", "cache off", "not cacheable", "evicted", "new program", "key seen but never kept", "no entry", "memory differs", "image views changed", "queued stores", "pending GPU writes", "queued label", "unmapped", "forgotten", "device image fallback"};
enum class CensusUser : std::size_t { KeyField, TablePointer, WalkedValue, Forwarded, RegisterRead, Unused, Untracked, Count };
constexpr std::array<const char*, CensusIndex(CensusUser::Count)> CensusUserNames{"no user word", "table pointer", "walked value", "forwarded data", "register read", "unused", "untracked"};
enum class CensusWord : std::size_t { Equal, ReadAddress, WrittenAddress, BufferShape, Texture, Sampler, Data, Program, OtherBinding, Gone, Count };
constexpr std::array<const char*, CensusIndex(CensusWord::Count)> CensusWordNames{"result equal", "read-only address", "written address", "buffer size or format", "texture", "sampler", "data", "program", "other binding", "reference gone"};
enum class CensusRecipe : std::size_t { Verify, Htile, Transient, Refused, NeverAttached, Count };
constexpr std::array<const char*, CensusIndex(CensusRecipe::Count)> CensusRecipeNames{"stamp validation", "HTILE clear", "relocated or patched copy", "pre-check refused", "never attached"};
enum class CensusRoute : std::size_t { QueuedRecipe, QueuedRecipeLess, QueuedMiss, QueuedRelocated, InlineRecipe, InlineRecipeLess, InlineMiss, Count };
constexpr std::size_t CensusWords = CensusIndex(CensusWord::Count);
constexpr std::size_t CensusRecipes = CensusIndex(CensusRecipe::Count);
constexpr std::size_t CensusNotPlain = CensusIndex(ItemFallback::Count);
constexpr std::size_t CensusDrainOther = CensusNotPlain + 1;
constexpr std::size_t CensusDrains = CensusDrainOther + 1;

struct CensusCounts {
    std::array<std::atomic<std::uint64_t>, CensusIndex(CensusOutcome::Count)> outcomes{};
    std::array<std::atomic<std::uint64_t>, CensusIndex(CensusMiss::Count)> misses{};
    std::array<std::atomic<std::uint64_t>, CensusIndex(CensusUser::Count) * CensusWords> noEntry{};
    std::array<std::atomic<std::uint64_t>, CensusWords> differing{};
    std::array<std::atomic<std::uint64_t>, CensusWords> present{};
    std::array<std::atomic<std::uint64_t>, CensusIndex(CensusRoute::Count)> routes{};
    std::array<std::atomic<std::uint64_t>, 2 * CensusRecipes> recipeLess{};
    std::array<std::atomic<std::uint64_t>, CensusDrains> drained{};
    std::array<std::atomic<std::uint64_t>, 3> redispatched{};
    std::atomic<std::uint64_t> packets{0}, indirect{0}, resolved{0}, walkChanged{0}, notCandidate{0}, captures{0}, captureNs{0}, missPaths{0}, missNs{0};
};

std::array<CensusCounts, 2>& CensusQueues() {
    static std::array<CensusCounts, 2> queues;
    return queues;
}

struct CensusReference {
    bool used = false;
    std::uint64_t key = 0;
    std::uint64_t serial = 0;
    std::vector<std::uint32_t> userData;
    std::weak_ptr<const DispatchVariant> variant;
};

struct CensusProgram {
    std::array<CensusReference, 4> ring;
    std::size_t next = 0;
    std::uint64_t serial = 0;
};

struct CensusMemory {
    AgcDriver::Mutex mutex;
    std::unordered_map<std::uint64_t, CensusProgram> programs;
    std::unordered_set<std::uint64_t> inserted;
};

CensusMemory& CensusState() {
    static CensusMemory state;
    return state;
}

struct CensusNearest {
    bool found = false;
    bool evicted = false;
    std::uint64_t key = 0;
    std::vector<std::uint32_t> userData;
    std::shared_ptr<const DispatchVariant> variant;
};

CensusNearest CensusFind(std::uint64_t program, std::uint64_t key, std::span<const std::uint32_t> userData) {
    CensusNearest nearest;
    auto& state = CensusState();
    std::lock_guard lock(state.mutex);
    nearest.evicted = state.inserted.contains(key);
    const auto found = state.programs.find(program);
    if (nearest.evicted || found == state.programs.end()) return nearest;
    const CensusReference* best = nullptr;
    std::size_t bestDistance = 0;
    for (const auto& held : found->second.ring) {
        if (!held.used) continue;
        std::size_t distance = 0;
        if (held.userData.size() != userData.size()) {
            distance = std::max(held.userData.size(), userData.size()) + 1u;
        } else {
            for (std::size_t index = 0; index < userData.size(); ++index) distance += held.userData[index] != userData[index] ? 1u : 0u;
        }
        if (best == nullptr || distance < bestDistance || (distance == bestDistance && held.serial > best->serial)) {
            best = &held;
            bestDistance = distance;
        }
    }
    if (best == nullptr) return nearest;
    nearest.found = true;
    nearest.key = best->key;
    nearest.userData = best->userData;
    nearest.variant = best->variant.lock();
    return nearest;
}

void CensusRemember(std::uint64_t program, std::uint64_t key, std::span<const std::uint32_t> userData, const std::shared_ptr<DispatchVariant>& variant, bool inserted) {
    auto& state = CensusState();
    std::unordered_map<std::uint64_t, CensusProgram> dropped;
    std::unordered_set<std::uint64_t> forgotten;
    std::lock_guard lock(state.mutex);
    if (inserted) {
        if (state.inserted.size() >= (std::size_t{1} << 17u)) forgotten.swap(state.inserted);
        state.inserted.insert(key);
    }
    if (state.programs.size() >= 4096 && !state.programs.contains(program)) dropped.swap(state.programs);
    auto& entry = state.programs[program];
    auto at = entry.ring.size();
    for (std::size_t index = 0; index < entry.ring.size(); ++index) {
        if (entry.ring[index].used && entry.ring[index].key == key) at = index;
    }
    if (at == entry.ring.size()) {
        at = entry.next;
        entry.next = (entry.next + 1u) % entry.ring.size();
    }
    auto& slot = entry.ring[at];
    slot.used = true;
    slot.key = key;
    slot.serial = ++entry.serial;
    slot.userData.assign(userData.begin(), userData.end());
    slot.variant = variant;
}

CensusWord CensusWordKind(const ShaderRecompiler::DescriptorBinding& binding, std::size_t word, std::uint32_t changed) {
    using Role = ShaderRecompiler::DescriptorRole;
    switch (binding.role) {
        case Role::GuestBuffers: {
            const auto element = word / 4u;
            const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
            if (word % 4u == 0u || (word % 4u == 1u && (changed & 0xffffu) != 0u)) return written ? CensusWord::WrittenAddress : CensusWord::ReadAddress;
            return CensusWord::BufferShape;
        }
        case Role::GuestImages:
        case Role::ImageTable:
        case Role::ImageTableMap:
        case Role::ImageStorageTable:
            return CensusWord::Texture;
        case Role::GuestSamplers:
        case Role::SamplerTable:
            return CensusWord::Sampler;
        case Role::FlattenedSrt:
        case Role::ShaderData:
            return CensusWord::Data;
        default:
            return CensusWord::OtherBinding;
    }
}

CensusWord CensusCompare(const ShaderRecompiler::RecompileResult& before, const ShaderRecompiler::RecompileResult& now, const DispatchVariant* masks, std::uint32_t& present) {
    auto first = CensusWord::Count;
    const auto note = [&](CensusWord kind) {
        present |= std::uint32_t{1} << CensusIndex(kind);
        if (first == CensusWord::Count) first = kind;
    };
    if (before.PipelineVariantId() != now.PipelineVariantId() || before.bindings.size() != now.bindings.size()) {
        note(CensusWord::Program);
        return first;
    }
    for (std::size_t index = 0; index < now.bindings.size(); ++index) {
        const auto& old = before.bindings[index];
        const auto& live = now.bindings[index];
        if (old.role != live.role || old.count != live.count || old.guestDescriptor.size() != live.guestDescriptor.size()) {
            note(CensusWord::Program);
            continue;
        }
        const bool masked = masks != nullptr && masks->flatBinding == index;
        for (std::size_t word = 0; word < live.guestDescriptor.size(); ++word) {
            const auto changed = old.guestDescriptor[word] ^ live.guestDescriptor[word];
            if (changed == 0u) continue;
            if (masked && std::find(masks->dataSlots.begin(), masks->dataSlots.end(), static_cast<std::uint32_t>(word)) != masks->dataSlots.end()) continue;
            note(CensusWordKind(live, word, changed));
        }
    }
    if (before.pushConstants != now.pushConstants) note(CensusWord::Data);
    return first == CensusWord::Count ? CensusWord::Equal : first;
}

CensusUser CensusUserKind(std::span<const std::uint32_t> before, std::span<const std::uint32_t> now, const ShaderRecompiler::ResourceCapture* capture, const ShaderRecompiler::RecompileResult& compiled, const ShaderSnapshot& shader) {
    if (before.size() != now.size()) return CensusUser::KeyField;
    std::size_t index = 0;
    while (index < now.size() && before[index] == now[index]) ++index;
    if (index == now.size()) return CensusUser::KeyField;
    if (index >= 64 || capture == nullptr) return CensusUser::Untracked;
    const auto bit = std::uint64_t{1} << index;
    if ((capture->userDataUse.asBase & bit) != 0) return CensusUser::TablePointer;
    if ((capture->userDataUse.asValue & bit) != 0) return CensusUser::WalkedValue;
    if (std::find(compiled.shaderDataUserWords.begin(), compiled.shaderDataUserWords.end(), static_cast<std::uint32_t>(index)) != compiled.shaderDataUserWords.end()) return CensusUser::Forwarded;
    const bool tail = shader.userDataTailRegister.has_value() && (index == *shader.userDataTailRegister || index == *shader.userDataTailRegister + 1u);
    if ((compiled.userDataRegisterReads & bit) != 0 || tail) return CensusUser::RegisterRead;
    return CensusUser::Unused;
}

bool CensusSameWalk(const DispatchVariant& variant, std::span<const ShaderRecompiler::MemoryRegion> captured, std::span<const ShaderRecompiler::MemoryRegion> memory) {
    std::size_t at = 0;
    for (const auto& region : captured) {
        if (std::any_of(memory.begin(), memory.end(), [&](const ShaderRecompiler::MemoryRegion& known) { return region.guestAddress >= known.guestAddress && region.guestAddress < known.guestAddress + known.bytes.size(); })) continue;
        if (at >= variant.runs.size() || variant.runs[at].first != region.guestAddress || variant.runs[at].second != region.guestAddress + region.bytes.size()) return false;
        ++at;
    }
    return at == variant.runs.size();
}

bool CensusLiveRecipe(const DispatchVariant& variant) {
    const auto recipe = variant.recipe.load(std::memory_order_acquire);
    return recipe != nullptr && (!recipe->moves || Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DispatchMovedRecipes));
}

struct OutcomeCensus {
    OutcomeCensus(std::uint32_t queueId, bool resolvingAhead, bool indirect);
    ~OutcomeCensus();
    OutcomeCensus(const OutcomeCensus&) = delete;
    OutcomeCensus& operator=(const OutcomeCensus&) = delete;

    std::chrono::steady_clock::time_point Clock() const { return on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}; }
    void Captured(std::chrono::steady_clock::time_point start) const;
    void MissPath(std::chrono::steady_clock::time_point start) const;
    void Served(bool fromAhead, bool cached, const std::shared_ptr<DispatchVariant>& variant, bool dataHit, std::uint64_t address, const ShaderSnapshot* shader, std::uint64_t dispatchKey, std::span<const std::uint32_t> userData);
    void ClassifyMiss(bool cacheOff, const DispatchEntry* entry, const ShaderRecompiler::ResourceCapture* capture, const ShaderRecompiler::RecompileResult& compiled, const ShaderSnapshot& shader, std::span<const std::uint32_t> userData, std::span<const ShaderRecompiler::MemoryRegion> captured, std::span<const ShaderRecompiler::MemoryRegion> memory);
    void Inserted(const std::shared_ptr<DispatchVariant>& variant, std::span<const std::uint32_t> userData) const;
    void Queued(bool recipe, bool late, bool relocated, const DispatchVariant* variant) const;
    void Inline(bool candidate, bool cached, const DispatchVariant* variant, bool recipeHit, bool htile, bool stamp) const;

    bool on = false;
    bool resolving = false;
    std::size_t queue = 0;
    bool notPlain = false;
    bool lookupFound = false;
    bool walkChanged = false;
    std::size_t compared = 0;
    std::size_t drain = CensusDrainOther;
    std::uint32_t present = 0;
    std::uint64_t program = 0;
    std::uint64_t key = 0;
    EntryOutcome lookup = EntryOutcome::Differing;
    CensusOutcome outcome = CensusOutcome::Other;
    CensusMiss miss = CensusMiss::Other;
    CensusUser user = CensusUser::KeyField;
    CensusWord word = CensusWord::Equal;
    OutcomeCensus* parent = nullptr;
};

OutcomeCensus*& CurrentCensus() {
    struct CurrentCensusStorage {};
    return HostThreadLocal<OutcomeCensus*, CurrentCensusStorage>();
}

OutcomeCensus::OutcomeCensus(std::uint32_t queueId, bool resolvingAhead, bool indirect) : on(CensusOn()), resolving(resolvingAhead), queue(queueId == 0 ? 0u : 1u) {
    if (!on) return;
    auto& current = CurrentCensus();
    parent = current;
    current = this;
    auto& counts = CensusQueues()[queue];
    if (resolving) counts.resolved.fetch_add(1, std::memory_order_relaxed);
    else if (parent == nullptr && indirect) counts.indirect.fetch_add(1, std::memory_order_relaxed);
}

OutcomeCensus::~OutcomeCensus() {
    if (!on) return;
    CurrentCensus() = parent;
    if (resolving) return;
    auto& counts = CensusQueues()[queue];
    if (parent != nullptr) {
        const bool hit = outcome == CensusOutcome::Adopted || outcome == CensusOutcome::Exact || outcome == CensusOutcome::Data || outcome == CensusOutcome::Patched || outcome == CensusOutcome::Relocated;
        counts.redispatched[hit ? 0u : outcome == CensusOutcome::Miss ? 1u : 2u].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    counts.packets.fetch_add(1, std::memory_order_relaxed);
    counts.outcomes[CensusIndex(outcome)].fetch_add(1, std::memory_order_relaxed);
    if (outcome != CensusOutcome::Miss) return;
    counts.misses[CensusIndex(miss)].fetch_add(1, std::memory_order_relaxed);
    if (miss == CensusMiss::NoEntry) counts.noEntry[CensusIndex(user) * CensusWords + CensusIndex(word)].fetch_add(1, std::memory_order_relaxed);
    if (miss == CensusMiss::Differing) {
        counts.differing[CensusIndex(word)].fetch_add(1, std::memory_order_relaxed);
        if (walkChanged) counts.walkChanged.fetch_add(1, std::memory_order_relaxed);
    }
    for (std::size_t kind = 0; kind < CensusWords; ++kind) {
        if (((present >> kind) & 1u) != 0) counts.present[kind].fetch_add(1, std::memory_order_relaxed);
    }
}

void OutcomeCensus::Captured(std::chrono::steady_clock::time_point start) const {
    if (!on) return;
    auto& counts = CensusQueues()[queue];
    counts.captures.fetch_add(1, std::memory_order_relaxed);
    counts.captureNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
}

void OutcomeCensus::MissPath(std::chrono::steady_clock::time_point start) const {
    if (!on) return;
    auto& counts = CensusQueues()[queue];
    counts.missPaths.fetch_add(1, std::memory_order_relaxed);
    counts.missNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
}

void OutcomeCensus::Served(bool fromAhead, bool cached, const std::shared_ptr<DispatchVariant>& variant, bool dataHit, std::uint64_t address, const ShaderSnapshot* shader, std::uint64_t dispatchKey, std::span<const std::uint32_t> userData) {
    if (!on || resolving) return;
    program = (address * 0x9e3779b97f4a7c15ull) ^ static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(shader));
    key = dispatchKey;
    const bool lookupHit = lookupFound && (lookup == EntryOutcome::Equal || lookup == EntryOutcome::EqualData);
    if (fromAhead) outcome = CensusOutcome::Adopted;
    else if (!cached) outcome = CensusOutcome::Miss;
    else if (!lookupHit) outcome = CensusOutcome::Relocated;
    else if (variant != nullptr && variant->relocated) outcome = CensusOutcome::Patched;
    else outcome = dataHit ? CensusOutcome::Data : CensusOutcome::Exact;
    if ((outcome == CensusOutcome::Exact || outcome == CensusOutcome::Data) && variant != nullptr) CensusRemember(program, key, userData, variant, true);
}

void OutcomeCensus::ClassifyMiss(bool cacheOff, const DispatchEntry* entry, const ShaderRecompiler::ResourceCapture* capture, const ShaderRecompiler::RecompileResult& compiled, const ShaderSnapshot& shader, std::span<const std::uint32_t> userData, std::span<const ShaderRecompiler::MemoryRegion> captured, std::span<const ShaderRecompiler::MemoryRegion> memory) {
    if (!on || resolving || outcome != CensusOutcome::Miss) return;
    if (cacheOff) {
        miss = CensusMiss::CacheOff;
        return;
    }
    if (lookupFound) {
        switch (lookup) {
            case EntryOutcome::Differing: break;
            case EntryOutcome::QueuedStores: miss = CensusMiss::QueuedStores; return;
            case EntryOutcome::FlushingImage:
            case EntryOutcome::PublishMoved:
            case EntryOutcome::PendingMoved: miss = CensusMiss::PendingWrites; return;
            case EntryOutcome::QueuedLabel: miss = CensusMiss::QueuedLabel; return;
            case EntryOutcome::Inaccessible: miss = CensusMiss::Unmapped; return;
            case EntryOutcome::ForgetMoved: miss = CensusMiss::Forgotten; return;
            case EntryOutcome::DeviceImageFallback: miss = CensusMiss::DeviceImages; return;
            default: miss = CensusMiss::Other; return;
        }
        if (entry == nullptr || entry->variants.empty() || entry->variants.front()->compiled == nullptr) {
            miss = CensusMiss::Other;
            return;
        }
        if (compared == 0) {
            miss = CensusMiss::ViewsChanged;
            return;
        }
        const auto& reference = *entry->variants.front();
        miss = CensusMiss::Differing;
        word = CensusCompare(*reference.compiled, compiled, &reference, present);
        walkChanged = !CensusSameWalk(reference, captured, memory);
        return;
    }
    if (!CacheableResult(compiled)) {
        miss = CensusMiss::NotCacheable;
        return;
    }
    const auto nearest = CensusFind(program, key, userData);
    if (nearest.evicted) {
        miss = CensusMiss::Evicted;
        return;
    }
    if (!nearest.found) {
        miss = CensusMiss::NewProgram;
        return;
    }
    if (nearest.key == key) {
        miss = CensusMiss::NotKept;
        return;
    }
    miss = CensusMiss::NoEntry;
    user = CensusUserKind(nearest.userData, userData, capture, compiled, shader);
    word = nearest.variant != nullptr && nearest.variant->compiled != nullptr ? CensusCompare(*nearest.variant->compiled, compiled, nearest.variant.get(), present) : CensusWord::Gone;
}

void OutcomeCensus::Inserted(const std::shared_ptr<DispatchVariant>& variant, std::span<const std::uint32_t> userData) const {
    if (!on || resolving || variant == nullptr || outcome != CensusOutcome::Miss) return;
    CensusRemember(program, key, userData, variant, true);
}

void OutcomeCensus::Queued(bool recipe, bool late, bool relocated, const DispatchVariant* variant) const {
    if (!on) return;
    auto& counts = CensusQueues()[queue];
    const auto route = late ? (relocated ? CensusRoute::QueuedRelocated : CensusRoute::QueuedMiss) : recipe ? CensusRoute::QueuedRecipe : CensusRoute::QueuedRecipeLess;
    counts.routes[CensusIndex(route)].fetch_add(1, std::memory_order_relaxed);
    if (route != CensusRoute::QueuedRecipeLess) return;
    const auto why = variant != nullptr && variant->relocated ? CensusRecipe::Transient : CensusRecipe::NeverAttached;
    counts.recipeLess[CensusIndex(why)].fetch_add(1, std::memory_order_relaxed);
}

void OutcomeCensus::Inline(bool candidate, bool cached, const DispatchVariant* variant, bool recipeHit, bool htile, bool stamp) const {
    if (!on) return;
    auto& counts = CensusQueues()[queue];
    if (!candidate) counts.notCandidate.fetch_add(1, std::memory_order_relaxed);
    else counts.drained[notPlain ? CensusNotPlain : drain].fetch_add(1, std::memory_order_relaxed);
    const auto route = !cached ? CensusRoute::InlineMiss : recipeHit ? CensusRoute::InlineRecipe : CensusRoute::InlineRecipeLess;
    counts.routes[CensusIndex(route)].fetch_add(1, std::memory_order_relaxed);
    if (route != CensusRoute::InlineRecipeLess) return;
    const auto why = stamp ? CensusRecipe::Verify : htile ? CensusRecipe::Htile : variant != nullptr && CensusLiveRecipe(*variant) ? CensusRecipe::Refused : variant != nullptr && variant->relocated ? CensusRecipe::Transient : CensusRecipe::NeverAttached;
    counts.recipeLess[CensusRecipes + CensusIndex(why)].fetch_add(1, std::memory_order_relaxed);
}

void NoteCensusFallback(ItemFallback reason) {
    if (!CensusOn()) return;
    if (auto* census = CurrentCensus(); census != nullptr) census->drain = CensusIndex(reason);
}

void ReportDispatchCensus() {
    if (!CensusOn()) return;
    const auto take = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    for (std::size_t queue = 0; queue < 2; ++queue) {
        auto& counts = CensusQueues()[queue];
        const auto packets = take(counts.packets);
        const auto indirect = take(counts.indirect);
        const auto resolved = take(counts.resolved);
        std::array<unsigned long long, 3> again{};
        for (std::size_t index = 0; index < again.size(); ++index) again[index] = take(counts.redispatched[index]);
        char item[256];
        std::string text;
        std::snprintf(item, sizeof(item), "[dispatch-census] (10 s) %s: %llu packets (%llu indirect, %llu ahead resolves; re-dispatched: hit %llu, miss %llu, other %llu):", queue == 0 ? "queue 0" : "other queues", packets, indirect, resolved, again[0], again[1], again[2]);
        text += item;
        for (std::size_t index = 0; index < CensusOutcomeNames.size(); ++index) {
            std::snprintf(item, sizeof(item), "%s %s %llu", index == 0 ? "" : ",", CensusOutcomeNames[index], take(counts.outcomes[index]));
            text += item;
        }
        text += "; misses:";
        for (std::size_t index = 0; index < CensusMissNames.size(); ++index) {
            std::snprintf(item, sizeof(item), "%s %s %llu", index == 0 ? "" : ",", CensusMissNames[index], take(counts.misses[index]));
            text += item;
        }
        text += "; no entry by first differing user word / first differing result word:";
        bool any = false;
        for (std::size_t user = 0; user < CensusUserNames.size(); ++user) {
            for (std::size_t word = 0; word < CensusWords; ++word) {
                const auto count = take(counts.noEntry[user * CensusWords + word]);
                if (count == 0) continue;
                std::snprintf(item, sizeof(item), "%s %s/%s %llu", any ? "," : "", CensusUserNames[user], CensusWordNames[word], count);
                text += item;
                any = true;
            }
        }
        if (!any) text += " none";
        text += "; memory differs by first differing result word:";
        any = false;
        for (std::size_t word = 0; word < CensusWords; ++word) {
            const auto count = take(counts.differing[word]);
            if (count == 0) continue;
            std::snprintf(item, sizeof(item), "%s %s %llu", any ? "," : "", CensusWordNames[word], count);
            text += item;
            any = true;
        }
        std::snprintf(item, sizeof(item), "%s (walk changed %llu); result words differing in classified misses:", any ? "" : " none", take(counts.walkChanged));
        text += item;
        for (std::size_t word = CensusIndex(CensusWord::ReadAddress); word < CensusIndex(CensusWord::Gone); ++word) {
            std::snprintf(item, sizeof(item), "%s %s %llu", word == CensusIndex(CensusWord::ReadAddress) ? "" : ",", CensusWordNames[word], take(counts.present[word]));
            text += item;
        }
        take(counts.present[CensusIndex(CensusWord::Equal)]);
        take(counts.present[CensusIndex(CensusWord::Gone)]);
        const auto captures = take(counts.captures);
        const auto captureMs = static_cast<double>(take(counts.captureNs)) / 1e6;
        const auto missPaths = take(counts.missPaths);
        const auto missMs = static_cast<double>(take(counts.missNs)) / 1e6;
        std::snprintf(item, sizeof(item), "; captures %llu in %.1f ms (%.1f us each), miss paths %llu in %.1f ms (%.1f us each)", captures, captureMs, captures != 0 ? captureMs * 1000.0 / static_cast<double>(captures) : 0.0, missPaths, missMs, missPaths != 0 ? missMs * 1000.0 / static_cast<double>(missPaths) : 0.0);
        text += item;
        std::array<unsigned long long, CensusIndex(CensusRoute::Count)> routes{};
        for (std::size_t index = 0; index < routes.size(); ++index) routes[index] = take(counts.routes[index]);
        std::snprintf(item, sizeof(item), "; queued: recipe %llu, after a miss %llu, relocated %llu, without a recipe %llu (", routes[CensusIndex(CensusRoute::QueuedRecipe)], routes[CensusIndex(CensusRoute::QueuedMiss)], routes[CensusIndex(CensusRoute::QueuedRelocated)], routes[CensusIndex(CensusRoute::QueuedRecipeLess)]);
        text += item;
        for (std::size_t index = 0; index < CensusRecipes; ++index) {
            std::snprintf(item, sizeof(item), "%s%s %llu", index == 0 ? "" : ", ", CensusRecipeNames[index], take(counts.recipeLess[index]));
            text += item;
        }
        std::snprintf(item, sizeof(item), "); inline: recipe %llu, miss %llu, without a recipe %llu (", routes[CensusIndex(CensusRoute::InlineRecipe)], routes[CensusIndex(CensusRoute::InlineMiss)], routes[CensusIndex(CensusRoute::InlineRecipeLess)]);
        text += item;
        for (std::size_t index = 0; index < CensusRecipes; ++index) {
            std::snprintf(item, sizeof(item), "%s%s %llu", index == 0 ? "" : ", ", CensusRecipeNames[index], take(counts.recipeLess[CensusRecipes + index]));
            text += item;
        }
        std::snprintf(item, sizeof(item), "), not a pipeline candidate %llu, drained candidates:", take(counts.notCandidate));
        text += item;
        for (std::size_t index = 0; index < CensusDrains; ++index) {
            std::snprintf(item, sizeof(item), "%s %s %llu", index == 0 ? "" : ",", index < ItemFallbackNames.size() ? ItemFallbackNames[index] : index == CensusNotPlain ? "not plain" : "other", take(counts.drained[index]));
            text += item;
        }
        if (queue == 0 || packets != 0 || resolved != 0 || captures != 0) std::fprintf(stderr, "%s\n", text.c_str());
    }
}

bool DispatchMissItemsEnabled() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PipelineDispatchMisses);
}

void NoteItemFallback(ItemFallback reason) {
    itemFallbacks[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    NoteCensusFallback(reason);
}

void ReportDispatchItems(bool profile) {
    static auto lastReport = std::chrono::steady_clock::now();
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!profile && !stats) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::string drained;
    for (std::size_t i = 0; i < ItemFallbackNames.size(); ++i) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s %llu", ItemFallbackNames[i], static_cast<unsigned long long>(itemFallbacks[i].load(std::memory_order_relaxed)));
        drained += item;
    }
    std::fprintf(stderr, "[dispatch-item] %llu commit items (%llu recipes rebuilt on the committer, %llu queued after a cache miss, %llu queued without a recipe, %llu cached ones kept off the queue as not plain; not plain by kind: no result %llu, HTILE clear %llu, key fill %llu, depth plane clear %llu; ZMASK fix-ups queued %llu); drained instead, cumulative:%s\n", static_cast<unsigned long long>(itemCommits.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemRebuilds.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemMisses.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemRecipeLess.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemNotPlain.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemNotPlainKinds[0].load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemNotPlainKinds[1].load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemNotPlainKinds[2].load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemNotPlainKinds[3].load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemFixups.load(std::memory_order_relaxed)), drained.c_str());
    if (const auto relocated = itemRelocated.load(std::memory_order_relaxed); relocated != 0) std::fprintf(stderr, "[dispatch-item] %llu more commit items queued after an exact cache miss that a relocated entry served (not among the misses above), cumulative\n", static_cast<unsigned long long>(relocated));
    ReportDispatchCensus();
}

bool SameDispatchResult(const ShaderRecompiler::RecompileResult& a, const ShaderRecompiler::RecompileResult& b) {
    if (&a == &b) return true;
    if (a.PipelineVariantId() != b.PipelineVariantId() || a.pushConstants != b.pushConstants || a.bindings.size() != b.bindings.size()) return false;
    for (std::size_t index = 0; index < a.bindings.size(); ++index) {
        if (a.bindings[index].guestDescriptor != b.bindings[index].guestDescriptor) return false;
    }
    return true;
}

bool SameBindingShape(const ShaderRecompiler::RecompileResult& a, const ShaderRecompiler::RecompileResult& b) {
    if (a.bindings.size() != b.bindings.size()) return false;
    for (std::size_t index = 0; index < a.bindings.size(); ++index) {
        if (a.bindings[index].role != b.bindings[index].role || a.bindings[index].binding != b.bindings[index].binding || a.bindings[index].count != b.bindings[index].count) return false;
    }
    return true;
}

std::shared_ptr<const Recipe> LiveRecipe(DispatchVariant& variant) {
    auto recipe = variant.recipe.load(std::memory_order_acquire);
    if (recipe == nullptr || !recipe->moves || Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DispatchMovedRecipes)) return recipe;
    static_cast<void>(variant.recipe.compare_exchange_strong(recipe, std::shared_ptr<const Recipe>{}));
    return nullptr;
}

}

bool Driver::capturedCurrent(std::span<const ShaderRecompiler::MemoryRegion> captured, const ShaderSnapshot& code, const PreparedShaderState::UserDataLink* linking) {
    const auto registered = [&](const ShaderRecompiler::MemoryRegion& region) {
        const auto address = region.guestAddress;
        return ((address >= code.codeAddress && address < code.codeAddress + code.code.size() * sizeof(std::uint32_t)) || (!code.header.empty() && address >= code.headerAddress && address < code.headerAddress + code.header.size())) && (linking == nullptr || !linking->Overlaps(address, region.bytes.size()));
    };
    for (const auto& region : captured) {
        const auto address = region.guestAddress;
        const auto bytes = region.bytes.size();
        if (bytes == 0) continue;
        if (registered(region)) continue;
        if (Driver::QueryPendingSpan(address, bytes, {}) != ShaderMemory::PendingWrite::None || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) return false;
        GuestMemory::GuardRead(address, bytes);
        GuestMemory::FlushGpuWrites(address, bytes);
    }
    std::vector<VouchedRegions> vouched;
    if (linking != nullptr && vouchedValidation() && !validateLegacy()) linking->VouchRegions(captured, vouched);
    for (std::size_t index = 0, range = 0; index < captured.size(); ++index) {
        while (range < vouched.size() && index >= vouched[range].end) ++range;
        if ((range < vouched.size() && index >= vouched[range].begin) || registered(captured[index])) continue;
        if (GuestMemory::CompareMapped(captured[index].guestAddress, captured[index].bytes) != GuestMemory::Compare::Equal) return false;
    }
    return true;
}

void NoteDispatchLookup(EntryOutcome outcome, std::size_t compared) {
    if (!CensusOn()) return;
    auto* census = CurrentCensus();
    if (census == nullptr) return;
    census->lookupFound = true;
    census->lookup = outcome;
    census->compared = compared;
}

bool Driver::pipelineDispatch() {
    return DispatchItemsEnabled();
}

void Driver::dispatch(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::uint64_t indirectArguments) {
    PerformanceTimer timing("Driver.Dispatch");
    const auto address = (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20c)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20d) & 0xffu) << 40u);
    if (SkippedShader(address) || RefusedDispatch(address)) return;
    OutcomeCensus census(submission.queue, resolvingAhead(), indirectArguments != 0);
    auto registeredShader = RegisteredProgram(*submission.shaders, address, {0});
    if (!registeredShader) registeredShader = ReadRawShader(address);
    const auto& snapshot = *registeredShader;
    require(snapshot.type == 0, "compute program refers to a non-compute shader");
    const auto userCount = (readRegister(queue.shader, 0x213) >> 1u) & 0x1fu;
    std::vector<std::uint32_t> userData;
    for (std::uint32_t i = 0; i < userCount; ++i) {
        userData.push_back(readUserData(queue.shader, 0x240 + i));
    }
    auto compute = Graphics::DecodeComputeStageInfo(queue.shader, snapshot.header);
    const bool resolving = resolvingAhead();
    const bool orderedIndirect = indirectArguments != 0 && !resolving && submission.queue == 0 && indirectSource().ordered && indirectSource().arguments == indirectArguments && DrawFinisher::OrderedSyncs();
    std::vector<ShaderRecompiler::MemoryRegion> memory{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}};
    if (!snapshot.header.empty()) memory.push_back({snapshot.headerAddress, snapshot.header});

    static const bool unlockedDevice = std::getenv("APS5_NO_UNLOCKED_DEVICE") == nullptr;
    std::shared_ptr<VulkanDevice> localDevice = unlockedDevice ? device.Load() : nullptr;
    if (localDevice == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    const auto codeOffset = static_cast<std::size_t>((address - snapshot.codeAddress) / sizeof(std::uint32_t));
    std::array<std::uint32_t, 5> resolved{};
    ResolvedDispatch ahead;
    bool fromAhead = false;
    auto& groupStats = groupCaptureStats();
    if (!resolving) {
        auto& parked = resolvedAhead();
        if (const auto found = parked.find(currentPacketOffset()); found != parked.end()) {
            ahead = std::move(found->second);
            parked.erase(found);
            fromAhead = ahead.address == address;
            if (!fromAhead) ++groupStats.addressMismatches;
        }
    }
    if (fromAhead) {
        resolved = ahead.packet;
        packet = resolved;
        indirectArguments = ahead.indirectArguments;
    } else if (indirectArguments != 0 && !resolving && matchesFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) {
        indirectSource().path = IndirectFillKernel;
        if (orderedIndirect) awaitIndirectArguments({indirectArguments, indirectArguments + 12u});
        const auto readStart = std::chrono::steady_clock::now();
        resolved = readDispatchArguments(submission.queue, indirectArguments, packet[4]);
        countIndirect(IndirectFillKernel, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        packet = resolved;
        indirectArguments = 0;
    }
    std::array<std::uint32_t, 3> partialGroups{};
    if (indirectArguments == 0 && (packet[4] & 0x2u) != 0) {
        partialGroups = Pm4::PartialGroupThreads({packet[1], packet[2], packet[3]}, {readRegister(queue.shader, 0x207), readRegister(queue.shader, 0x208), readRegister(queue.shader, 0x209)});
    }
    const bool pipelineCandidate = !resolving && !fromAhead && DispatchItemsEnabled() && DrawPipeline::Active() && (indirectArguments == 0 || orderedIndirect) && deferredLabels().labels.empty() && !matchesCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) && !MatchesStreamCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) && !MatchesTableCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) && !MatchesIndexedCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute);
    const bool partialLastGroup = partialGroups != std::array<std::uint32_t, 3>{};
    std::optional<std::pair<std::uint64_t, std::size_t>> htileClearCompletion;
    Graphics::HtileMaskClear htileMask;
    std::optional<Graphics::HtileZmaskFixup> htileFixup;
    if (resolving) {
        if (indirectArguments == 0 && Graphics::MatchMultisampledSingleClear(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads)) return;
        if (!partialLastGroup && matchesFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) return;
        if (!partialLastGroup && indirectArguments == 0 && (MatchDwordPatternFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchConstantDwordFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchClampedGlobalFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchMemoryValueBufferFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute))) return;
        if (!partialLastGroup && indirectArguments == 0 && (matchesCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchesStreamCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchesIndexedCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchesTableCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute))) return;
    } else {
        if (orderedIndirect && indirectArguments != 0 && Graphics::HtileDepthClearAddress(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads) != 0) {
            DrawFinisher::NoteFrameKept(FrameKept::IndirectNote);
            if (DrawPipeline::Queue0().Busy()) {
                DrawFinisher::NoteDeepKept(DeepKept::NoteHtileDepthClear);
                DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x16);
            }
        }
        if (DrawFinisher::Busy() && !DrawFinisher::OnFinisherThread() && (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DispatchBeside) || DrawFinisher::Flow() || DrawFinisher::Deep())) {
            const auto code = std::span(snapshot.code).subspan(codeOffset);
            const bool deep = DrawFinisher::Deep();
            const bool fixupsByTurn = deep && DispatchMissItemsEnabled() && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FixupsByTurn);
            const auto fills = [&] {
                return (!partialLastGroup && matchesFillKernel(code, userData, compute))
                    || (!partialLastGroup && indirectArguments == 0 && (MatchDwordPatternFill(packet, code, userData, compute) || MatchConstantDwordFill(packet, code, userData, compute) || MatchClampedGlobalFill(packet, code, userData, compute) || MatchMemoryValueBufferFill(packet, code, userData, compute)));
            };
            const auto notes = [&] {
                return !pipelineCandidate
                    || (indirectArguments == 0 && Graphics::MatchMultisampledSingleClear(code, userData, compute.numThreads))
                    || (indirectArguments == 0 && (packet[4] & 0x20u) == 0 && Graphics::MatchMemoryValueFill(code, userData, compute.numThreads, {packet[1], packet[2], packet[3]}).has_value())
                    || Graphics::HtileDepthClearAddress(code, userData, compute.numThreads) != 0
                    || Graphics::MatchHtileBufferMaskClear(code, userData, compute, packet, indirectArguments).has_value()
                    || (!fixupsByTurn && Graphics::MatchHtileZmaskFixup(code, userData, compute, packet, indirectArguments).has_value());
            };
            const auto noteKind = [&] {
                if (!pipelineCandidate) return DeepKept::NoteNotCandidate;
                if (indirectArguments == 0 && Graphics::MatchMultisampledSingleClear(code, userData, compute.numThreads)) return DeepKept::NoteMultisampleClear;
                if (indirectArguments == 0 && (packet[4] & 0x20u) == 0 && Graphics::MatchMemoryValueFill(code, userData, compute.numThreads, {packet[1], packet[2], packet[3]}).has_value()) return DeepKept::NoteMemoryValueFill;
                if (Graphics::HtileDepthClearAddress(code, userData, compute.numThreads) != 0) return DeepKept::NoteHtileDepthClear;
                if (Graphics::MatchHtileBufferMaskClear(code, userData, compute, packet, indirectArguments).has_value()) return DeepKept::NoteHtileMaskClear;
                return DeepKept::NoteHtileZmaskFixup;
            };
            const bool noted = deep && notes();
            const bool filledOnly = deep && !noted && fills();
            const bool sideEffects = deep ? noted || filledOnly : !pipelineCandidate
                || (indirectArguments == 0 && Graphics::MatchMultisampledSingleClear(code, userData, compute.numThreads))
                || (!partialLastGroup && matchesFillKernel(code, userData, compute))
                || (!partialLastGroup && indirectArguments == 0 && (MatchDwordPatternFill(packet, code, userData, compute) || MatchConstantDwordFill(packet, code, userData, compute) || MatchClampedGlobalFill(packet, code, userData, compute) || MatchMemoryValueBufferFill(packet, code, userData, compute)))
                || (indirectArguments == 0 && (packet[4] & 0x20u) == 0 && Graphics::MatchMemoryValueFill(code, userData, compute.numThreads, {packet[1], packet[2], packet[3]}).has_value())
                || Graphics::HtileDepthClearAddress(code, userData, compute.numThreads) != 0
                || Graphics::MatchHtileBufferMaskClear(code, userData, compute, packet, indirectArguments).has_value()
                || Graphics::MatchHtileZmaskFixup(code, userData, compute, packet, indirectArguments).has_value();
            if (sideEffects && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpNoPacketDrains) && !DrawFinisher::FlowSkips(DrawFinisher::FlowSite::Dispatch)) {
                if (filledOnly) {
                    DrawFinisher::NoteDeepSite(DeepSite::FillByTurn);
                } else {
                    DrawFinisher::NoteDeepKept(noted ? noteKind() : DeepKept::WorkerNote);
                    DrawFinisher::Get().Drain();
                    if ((DrawFinisher::Get().FlipsQueued() != DrawFinisher::Get().FlipsCommitted() || DrawFinisher::Get().IndirectsInFlight()) && DrawPipeline::Queue0().Busy()) {
                        DrawFinisher::NoteFrameKept(DrawFinisher::Get().FlipsQueued() != DrawFinisher::Get().FlipsCommitted() ? FrameKept::WorkerBehindFlip : FrameKept::WorkerBehindIndirect);
                        DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
                    }
                }
            }
        }
        if (indirectArguments == 0 && Graphics::TakeMultisampledSingleClear(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads)) {
            census.outcome = CensusOutcome::Clear;
            pendingDispatchPhases().outcome = DispatchOutcome::FillHle;
            if (resolvedAhead().empty()) gateOpened() = true;
            return;
        }
        if (indirectArguments == 0 && (packet[4] & 0x20u) == 0) {
            if (const auto fill = Graphics::MatchMemoryValueFill(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads, {packet[1], packet[2], packet[3]})) Graphics::NoteMemoryValueFill(*fill);
        }
        if (!partialLastGroup && fillBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice)) {
            census.outcome = CensusOutcome::Fill;
            pendingDispatchPhases().outcome = DispatchOutcome::FillHle;
            if (resolvedAhead().empty()) gateOpened() = true;
            return;
        }
        if (DrawPipeline::Active() && !pipelineCandidate) {
            if (orderedIndirect && indirectArguments != 0) DrawFinisher::NoteFrameKept(FrameKept::IndirectNotCandidate);
            const auto copyCode = std::span(snapshot.code).subspan(codeOffset);
            DrawFinisher::NoteDeepKept(DrawFinisher::FrameCensus() && (matchesCopyKernel(copyCode, userData, compute) || MatchesStreamCopyKernel(copyCode, userData, compute) || MatchesIndexedCopyKernel(copyCode, userData, compute)) ? DeepKept::WorkerDispatchCopy : DeepKept::WorkerDispatch);
            DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        }
        if (dumpsDispatchProgram(address)) {
            static std::atomic<unsigned> programDumps{0};
            if (programDumps.fetch_add(1, std::memory_order_relaxed) < 3) {
                const auto code = std::span(snapshot.code).subspan(codeOffset);
                std::string text;
                for (std::size_t index = 0; index < code.size() && index < 160; ++index) {
                    char word[12];
                    std::snprintf(word, sizeof(word), " %08x", code[index]);
                    text += word;
                }
                std::string user;
                for (const auto word : userData) {
                    char item[12];
                    std::snprintf(item, sizeof(item), " %08x", word);
                    user += item;
                }
                std::fprintf(stderr, "[dispatch-program] 0x%llx: packet %08x %08x %08x %08x, threads %u %u %u, user data%s; code (%zu dwords):%s\n", static_cast<unsigned long long>(address), packet[1], packet[2], packet[3], packet[4], compute.numThreads[0], compute.numThreads[1], compute.numThreads[2], user.c_str(), code.size(), text.c_str());
            }
        }
        if (const auto htile = Graphics::HtileDepthClearAddress(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads); htile != 0) {
            const auto threads = (packet[4] & 0x20u) != 0 ? static_cast<std::uint64_t>(packet[1]) : static_cast<std::uint64_t>(packet[1]) * compute.numThreads[0];
            const auto nativeBytes = Graphics::HtileDepthClearBytes_nid_no_patch(std::span(snapshot.code).subspan(codeOffset), userData, compute, packet, indirectArguments);
            const auto bytes = static_cast<std::size_t>(threads * 16u);
            if (Graphics::NoteHtileDepthClear(htile, bytes, nativeBytes)) htileClearCompletion = std::pair{htile, bytes};
            if (nativeBytes != 0u) htileMask = {htile, nativeBytes, userData[5], userData[4]};
            static std::atomic<unsigned> htileReports{0};
            if (htileReports.fetch_add(1, std::memory_order_relaxed) < 12) std::fprintf(stderr, "[htile-clear] dispatch: HTILE 0x%llx, packet %08x %08x %08x %08x, user data %zu words [4] %08x [5] %08x, threads %llu, bytes 0x%zx, native bytes 0x%zx, partial groups %d, group id x %d, thread id components %u, indirect %llu, completion %d\n", static_cast<unsigned long long>(htile), packet[1], packet[2], packet[3], packet[4], userData.size(), userData.size() > 4 ? userData[4] : 0u, userData.size() > 5 ? userData[5] : 0u, static_cast<unsigned long long>(threads), bytes, nativeBytes, compute.PartialGroups() ? 1 : 0, compute.groupIdEnable[0] ? 1 : 0, compute.threadIdComponentCount, static_cast<unsigned long long>(indirectArguments), htileClearCompletion.has_value() ? 1 : 0);
        } else if (const auto buffered = Graphics::MatchHtileBufferMaskClear(std::span(snapshot.code).subspan(codeOffset), userData, compute, packet, indirectArguments)) {
            if (Graphics::NoteHtileDepthClear(buffered->address, buffered->bytes, buffered->bytes)) htileClearCompletion = std::pair{buffered->address, buffered->bytes};
            htileMask = *buffered;
        } else {
            htileFixup = Graphics::MatchHtileZmaskFixup(std::span(snapshot.code).subspan(codeOffset), userData, compute, packet, indirectArguments);
        }
        if (!partialLastGroup && indirectArguments == 0 && copyBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice, address)) {
            census.outcome = CensusOutcome::Copy;
            pendingDispatchPhases().outcome = DispatchOutcome::CopyHle;
            if (resolvedAhead().empty()) gateOpened() = true;
            return;
        }
    }
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {
        const std::array<std::uint32_t, 3> threads{packet[1], packet[2], packet[3]};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            if (threads[axis] % compute.numThreads[axis] != 0) compute.partialThreads = threads;
        }
    }
    if (partialLastGroup) compute.partialThreads = partialGroups;
    std::array<std::uint32_t, 3> threadGroup{};
    if (indirectArguments != 0 && (packet[4] & 0x20u) != 0) {
        for (std::uint32_t axis = 0; axis < 3; ++axis) threadGroup[axis] = std::max(readRegister(queue.shader, 0x207 + axis) & 0xffffu, 1u);
        compute.partialThreads = {~0u, ~0u, ~0u};
    }
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, address, std::span(snapshot.code).subspan(codeOffset), snapshot.headerAddress, snapshot.header},
        {(packet[4] & 0x8000u) != 0 ? 32u : 64u, 0, userData, compute, std::nullopt, std::nullopt, memory, DispatchFloatMode(snapshot, queue.shader)},
        localDevice->ComputeTarget((packet[4] & 0x8000u) != 0 ? 32u : 64u),
        {0, 0, 0, threadGroup != std::array<std::uint32_t, 3>{} ? 0u : 128u}
    };
    if (request.target.deviceImageTables) request.target.deviceImageTables = localDevice->DeviceImagesServe(address);
    const std::size_t computeHeapKey = HeapFragmentKey(codeOffset, HeapKeyClass::Compute);
    if (request.target.gpuResolvedImages && (HeapRefused() || HeapStageExcluded(ShaderRecompiler::ShaderStage::Compute) || HeapFragmentKnown(&snapshot, computeHeapKey) == std::optional(false))) request.target.gpuResolvedImages = false;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static std::atomic<double> captureMs{0}, keyMs{0}, recompileMs{0}, deviceMs{0};
    static std::atomic<std::uint64_t> cacheHits{0};
    static std::atomic<std::uint64_t> dispatches{0};
    auto lap = std::chrono::steady_clock::now();

    std::array<double, DriverPhaseCount> phaseMs{};
    auto phaseLap = lap;
    DispatchPhaseTiming phaseTiming{profile, lap, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[PhasePrologue] = std::chrono::duration<double, std::milli>(lap - packetStartedAt()).count();

    static const bool noDispatchCacheEnv = std::getenv("APS5_NO_DISPATCH_CACHE") != nullptr;

    static const std::pair<std::uint64_t, std::uint64_t> probeDispatch = [] {
        const char* text = std::getenv("APS5_PROBE_DISPATCH");
        if (text == nullptr) return std::pair<std::uint64_t, std::uint64_t>{0, 0};
        char* end = nullptr;
        const auto probeAddress = std::strtoull(text, &end, 16);
        const auto index = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 10) : 0ull;
        return std::pair<std::uint64_t, std::uint64_t>{probeAddress, index};
    }();
    bool probeThis = false;

    if (probeDispatch.first != 0 && (address & 0xfffffffffull) == (probeDispatch.first & 0xfffffffffull)) {
        static std::atomic<std::uint64_t> dispatchesSeen{0};
        probeThis = dispatchesSeen.fetch_add(1) == probeDispatch.second;
        if (probeThis) std::fprintf(stderr, "[gpu] probing dispatch %llu of 0x%llx\n", static_cast<unsigned long long>(probeDispatch.second), static_cast<unsigned long long>(address));
    }

    struct ProbeScope {
        bool active;
        explicit ProbeScope(bool active) : active(active) { if (active) ShaderRecompiler::SetDebugProbeActive(true); }
        ~ProbeScope() { if (active) ShaderRecompiler::SetDebugProbeActive(false); }
    } probeScope{probeThis};
    const bool noDispatchCache = noDispatchCacheEnv || probeThis;
    std::uint64_t key = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    mix(address);
    mix(packet[4] & 0x8000u);
    for (const auto threads : compute.partialThreads) mix(threads);
    for (const auto word : userData) mix(word);

    static const bool keyHygiene = std::getenv("APS5_NO_DISPATCH_KEY_HYGIENE") == nullptr;
    if (keyHygiene) {
        for (const auto offset : {0x207u, 0x208u, 0x209u, 0x212u, 0x213u, 0x228u}) {
            const auto found = queue.shader.find(offset);

            mix(found == queue.shader.end() ? (1ull << 32u) : found->second);
        }
    } else {
        for (const auto& [offset, value] : queue.shader) {
            mix(offset);
            mix(value);
        }
    }
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiledResult;

    std::shared_ptr<DispatchVariant> keepVariant;

    std::shared_ptr<DispatchVariant> attachVariant;

    std::shared_ptr<ShaderMemory> shaderMemory;
    std::shared_ptr<const PreparedShaderState::UserDataLink> linking;
    std::vector<ShaderRecompiler::MemoryRegion> captured;

    std::vector<std::uint32_t> liveWords;
    bool dataHit = false;
    bool cached = false;
    bool validated = false;

    std::shared_ptr<DispatchEntry> missedEntry;
    bool missedDiffering = false;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;

    static const bool traceCache = std::getenv("APS5_TRACE_DISPATCH_CACHE") != nullptr;
    if (traceCache) {
        std::lock_guard traceLock(dispatchCacheMutex);
        struct Last { std::vector<std::uint32_t> userData; std::map<std::uint32_t, std::uint32_t> shader; std::uint64_t key; };
        static std::map<std::uint64_t, Last> last;
        static int reports = 0;
        auto& previous = last[address];
        if (previous.key != 0 && previous.key != key && reports < 200) {
            std::string what;
            for (std::size_t i = 0; i < userData.size(); ++i) {
                if (i >= previous.userData.size() || previous.userData[i] != userData[i]) {
                    char text[48];
                    std::snprintf(text, sizeof(text), " user[%zu] %08x->%08x", i, i < previous.userData.size() ? previous.userData[i] : 0u, userData[i]);
                    what += text;
                }
            }
            for (const auto& [offset, value] : queue.shader) {
                const auto old = previous.shader.find(offset);
                if (old == previous.shader.end() || old->second != value) {
                    char text[48];
                    std::snprintf(text, sizeof(text), " sh[%x] %08x->%08x", offset, old == previous.shader.end() ? 0u : old->second, value);
                    what += text;
                }
            }
            ++reports;
            std::fprintf(stderr, "[dispatch-cache] 0x%llx key changed:%s\n", static_cast<unsigned long long>(address), what.c_str());
        }
        previous.userData = userData;
        previous.shader = std::map<std::uint32_t, std::uint32_t>(queue.shader.begin(), queue.shader.end());
        previous.key = key;
    }

    mix(reinterpret_cast<std::uintptr_t>(registeredShader.get()));
    mix(request.target.gpuResolvedImages);
    const bool relocate = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DispatchRelocate) && !stampValidate() && !noDispatchCache;
    DispatchProvenance provenance;
    const auto traceProvenance = [&] {
        std::uint64_t identity = 0x84222325cbf29ce4ull;
        for (const std::uint64_t value : {address, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(registeredShader.get())), static_cast<std::uint64_t>(userCount)}) {
            identity ^= value;
            identity *= 0x100000001b3ull;
        }
        std::vector<std::uint64_t> head{address, packet[4] & 0x8000u};
        for (const auto threads : compute.partialThreads) head.push_back(threads);
        std::vector<std::uint64_t> tail;
        if (keyHygiene) {
            for (const auto offset : {0x207u, 0x208u, 0x209u, 0x212u, 0x213u, 0x228u}) {
                const auto found = queue.shader.find(offset);
                tail.push_back(found == queue.shader.end() ? (1ull << 32u) : found->second);
            }
        } else {
            for (const auto& [offset, value] : queue.shader) {
                tail.push_back(offset);
                tail.push_back(value);
            }
        }
        tail.push_back(reinterpret_cast<std::uintptr_t>(registeredShader.get()));
        tail.push_back(request.target.gpuResolvedImages);
        provenance.identity = identity;
        provenance.userData = userData;
        provenance.aliasOf = [head = std::move(head), tail = std::move(tail), words = userData](const DispatchRelocation& shape) {
            std::uint64_t alias = 0xcbf29ce484222325ull;
            const auto mixAlias = [&](std::uint64_t value) {
                alias ^= value;
                alias *= 0x100000001b3ull;
            };
            for (const auto value : head) mixAlias(value);
            for (std::size_t index = 0; index < words.size(); ++index) {
                const auto bit = index < 64 ? std::uint64_t{1} << index : std::uint64_t{0};
                if ((shape.mask & bit) == 0) {
                    mixAlias(words[index]);
                    continue;
                }
                mixAlias(0x52454c4fu);
                if ((shape.aligned & bit) != 0) mixAlias(words[index] & 3u);
            }
            for (const auto value : tail) mixAlias(value);
            return alias;
        };
    };
    if (fromAhead && key != ahead.key) {
        fromAhead = false;
        ++groupStats.keyMismatches;
    }
    if (fromAhead) {
        if (const auto hook = aheadAdoptHook.load(std::memory_order_acquire)) hook();
        const auto* aheadViews = ahead.capture != nullptr ? &ahead.capture->snapshot.registeredImages : ahead.keepVariant != nullptr && ahead.keepVariant->registeredViews != nullptr ? ahead.keepVariant->registeredViews.get() : nullptr;
        const auto* aheadLinking = ahead.linking != nullptr ? ahead.linking.get() : ahead.keepVariant != nullptr ? ahead.keepVariant->linking.get() : nullptr;
        if (!capturedCurrent(ahead.captured, snapshot, aheadLinking) || (aheadViews != nullptr && !CaptureRegisteredViewsCurrent(*aheadViews))) {
            fromAhead = false;
            ++dispatchesRejectedAhead;
            ++groupStats.memoryMismatches;
        }
    }
    phaseTiming.Phase(PhaseKey);
    timing.Mark("decode_key_device_setup");
    bool readsClear = false;
    const bool queueMisses = pipelineCandidate && DispatchMissItemsEnabled() && !noDispatchCache && !stampValidate() && !verifyDataHits();
    bool lateItem = false;
    bool recipeLess = false;
    if (pipelineCandidate) {
        std::vector<ShaderRecompiler::MemoryRegion> reads;
        bool entryFound = false;
        {
            std::lock_guard cacheLock(dispatchCacheMutex);
            if (const auto found = dispatchCache.find(key); found != dispatchCache.end()) {
                entryFound = true;
                for (const auto& variant : found->second->variants) appendEntryRegions(*variant, reads);
            }
        }
        if (!entryFound && queueMisses) {
            lateItem = true;
        } else if (!entryFound || noDispatchCache || stampValidate()) {
            NoteItemFallback(ItemFallback::Miss);
        } else if (verifyDataHits()) {
            NoteItemFallback(ItemFallback::Verify);
        } else {
            readsClear = std::none_of(reads.begin(), reads.end(), [](const auto& region) { return DrawPipeline::Queue0().Overlaps(region.guestAddress, region.bytes.size()); });
            if (!readsClear && DrawFinisher::FlowSkips(DrawFinisher::FlowSite::DispatchReads)) readsClear = true;
            if (!readsClear && DrawFinisher::Deep()) readsClear = awaitRegionWriters(reads, DeepSite::DispatchReads);
            if (!readsClear) NoteItemFallback(ItemFallback::Reads);
        }
        if (!readsClear && !lateItem && orderedIndirect && indirectArguments != 0) DrawFinisher::NoteFrameKept(FrameKept::IndirectFallback);
        if (!readsClear && !lateItem && DrawPipeline::Queue0().Busy()) {
            DrawFinisher::NoteDeepKept(DeepKept::WorkerDispatchReads);
            DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        }
        ReportDispatchItems(profile);
    }
    if (fromAhead) {
        compiledResult = std::move(ahead.compiledResult);
        keepVariant = std::move(ahead.keepVariant);
        attachVariant = std::move(ahead.attachVariant);
        shaderMemory = std::move(ahead.shaderMemory);
        linking = std::move(ahead.linking);
        captured = std::move(ahead.captured);
        liveWords = std::move(ahead.liveWords);
        capture = std::move(ahead.capture);
        dataHit = ahead.dataHit;
        cached = ahead.cached;
        validated = ahead.validated;
        ++groupStats.adopted;
        ++dispatchesAdoptedAhead;
        const auto age = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ahead.resolvedAt).count();
        groupStats.ageMs += age;
        groupStats.ageMaxMs = std::max(groupStats.ageMaxMs, age);
        phaseTiming.Phase(PhaseLookup);
    } else {
        lookupDispatch(address, submission, key, noDispatchCache, traceCache, profile, memory, phaseTiming, phaseMs, compiledResult, keepVariant, captured, liveWords, dataHit, cached, validated, missedEntry, missedDiffering, relocate);
        if (relocate && !cached) traceProvenance();
        if (relocate && !cached && missedEntry == nullptr && relocateDispatch(address, submission, provenance.identity, provenance.aliasOf, registeredShader, userData, memory, compiledResult, keepVariant, captured, liveWords)) {
            cached = true;
            validated = true;
            dataHit = true;
        }
    }
    census.Served(fromAhead, cached, keepVariant, dataHit, address, registeredShader.get(), key, userData);
    timing.Mark("lookup_cache");
    std::vector<DrawPipeline::Range> itemWrites;
    bool pipelined = false;
    bool writesUnbounded = false;
    if (readsClear) {
        if (cached && keepVariant != nullptr && LiveRecipe(*keepVariant) != nullptr) {
            bool bounded = false;
            itemWrites = DispatchWriteRanges(*compiledResult, bounded);
            pipelined = bounded;
            writesUnbounded = !bounded;
            if (!bounded) NoteItemFallback(ItemFallback::Writes);
        } else if (queueMisses && cached && keepVariant != nullptr) {
            recipeLess = true;
        } else if (queueMisses && !cached) {
            lateItem = true;
        } else {
            NoteItemFallback(ItemFallback::NoRecipe);
        }
        if (!pipelined && !recipeLess && !lateItem && orderedIndirect && indirectArguments != 0) DrawFinisher::NoteFrameKept(writesUnbounded ? FrameKept::IndirectUnbounded : FrameKept::IndirectFallback);
        if (!pipelined && !recipeLess && !lateItem && DrawPipeline::Queue0().Busy()) {
            DrawFinisher::NoteDeepKept(DeepKept::WorkerDispatchWrites);
            DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        }
    }
    std::uint64_t forgetAtCapture = 0;
    if (cached) {
        require(keepVariant != nullptr && keepVariant->shader == registeredShader, "dispatch cache belongs to another registered shader");
        if (profile) captureMs.fetch_add(phaseTiming.Elapsed(), std::memory_order_relaxed);
    } else if (!fromAhead) {
        const auto missStart = census.Clock();
        shaderMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
        if (relocate && !resolving) shaderMemory->RecordReadNodes(&provenance.readNodes);

        static const bool dumpShaders = std::getenv("APS5_DUMP_SHADERS") != nullptr;
        try {
            auto invocation = InvocationFor(snapshot, codeOffset, request, AsyncDrawCompile(), &linking);
            timing.Mark("prepared_invocation");

            const auto waitedBefore = traceCapSync() ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
            forgetAtCapture = GuestMemory::ForgetSerial();
            const auto captureStart = census.Clock();
            capture = [&] {
                const SampledReadScope sampling(evidenceReads);
                return CaptureInvocation(*shaderMemory, snapshot, codeOffset, request, invocation, &linking);
            }();
            census.Captured(captureStart);
            captured = shaderMemory->Regions();
            request.context.memory = captured;
            if (traceCapSync()) traceCapture("dispatch-capture", address, submission.queue, captured, Graphics::Recorder::ThreadWaitedMs() - waitedBefore);
            if (profile) captureMs.fetch_add(phaseTiming.Elapsed(), std::memory_order_relaxed);
            phaseTiming.Phase(PhaseCapture);
            if (dumpShaders) static_cast<void>(dumpRequest(address, request));
            const auto started = std::chrono::steady_clock::now();


            timing.Mark("capture_resources");
            compiledResult = MaterializeInBackground(invocation, capture, request.context.memory, address);
            if (compiledResult->gpuResolvedImages && !HeapStageMappable(*compiledResult, compiledResult->workgroupMemoryDwords != 0)) {
                Graphics::NoteHeapFallback(Graphics::HeapFallback::DispatchUnmapped);
                NoteHeapFragment(&snapshot, computeHeapKey, false);
                request.target.gpuResolvedImages = false;
                invocation = InvocationFor(snapshot, codeOffset, request);
                const auto recaptureStart = census.Clock();
                capture = shaderMemory->Capture(invocation);
                census.Captured(recaptureStart);
                captured = shaderMemory->Regions();
                request.context.memory = captured;
                compiledResult = invocation.Materialize(*capture);
            }
            if (relocate && !resolving && capture != nullptr && compiledResult != nullptr && CacheableResult(*compiledResult)) {
                try {
                    const auto share = invocation.ShareStage(*capture);
                    if (share && share.patch != nullptr && SameBindingShape(*share.donor, *compiledResult)) {
                        auto traced = invocation.TraceSideForwards(*capture, *share.donor, *share.patch);
                        if (traced.traced) provenance.sides = std::make_shared<const ShaderRecompiler::SideForwards>(std::move(traced));
                    }
                } catch (const std::exception&) {
                    provenance.sides = nullptr;
                }
            }
            if (relocate) shaderMemory->RecordReadNodes(nullptr);
            timing.Mark("materialize");
            if (profile && compiledResult->cacheHit) cacheHits.fetch_add(1, std::memory_order_relaxed);
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            static std::atomic<double> totalMs{0};
            if (profile) {
                const auto total = totalMs.fetch_add(elapsed, std::memory_order_relaxed) + elapsed;
                if (elapsed > 200) std::fprintf(stderr, "[gpu] compute shader 0x%llx recompile took %.0f ms (%zu SPIR-V words, %zu captured regions, total %.1f s)\n", static_cast<unsigned long long>(address), elapsed, compiledResult->spirv.size(), captured.size(), total / 1000);
            }
            if (const auto limit = MaxComputeSpirvWords(); limit != 0 && compiledResult->spirv.size() > limit)
                throw std::runtime_error(std::to_string(compiledResult->spirv.size()) + " SPIR-V words, above APS5_MAX_COMPUTE_SPIRV_WORDS (" + std::to_string(limit) + ")");
        } catch (const DrawCompilePending&) {
            return;
        } catch (const std::exception& error) {
            std::string dump;
            if (dumpShaders) {
                const auto regions = shaderMemory->Regions();
                auto failed = request;
                failed.context.memory = regions;
                dump = dumpRequest(address, failed, "_failed");
            }

            std::string reason = error.what();
            if (const auto newline = reason.find('\n'); newline != std::string::npos) reason.resize(newline);
            if (SkipRefusedDispatches()) {
                ReportRefusedDispatch(address, reason);
                return;
            }
            char where[96];
            if (dump.empty()) std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
            else std::snprintf(where, sizeof(where), "compute shader 0x%llx (%s): ", static_cast<unsigned long long>(address), dump.c_str());
            throw std::runtime_error(where + reason);
        }
        if (profile) recompileMs.fetch_add(phaseTiming.Elapsed(), std::memory_order_relaxed);
        phaseTiming.Phase(PhaseRecompile);
        census.ClassifyMiss(noDispatchCache, missedEntry.get(), capture.get(), *compiledResult, snapshot, userData, captured, memory);
        if (!resolving) insertDispatch(address, key, noDispatchCache || !CacheableResult(*compiledResult), profile, registeredShader, forgetAtCapture, memory, shaderMemory, captured, capture, compiledResult, missedEntry, missedDiffering, attachVariant, phaseTiming, linking, relocate ? &provenance : nullptr);
        if (linking != nullptr) linking->AppendRegions(captured);
        census.Inserted(attachVariant, userData);
        census.MissPath(missStart);
    }
    const auto* registeredViews = capture != nullptr ? &capture->snapshot.registeredImages : keepVariant != nullptr && keepVariant->registeredViews != nullptr ? keepVariant->registeredViews.get() : nullptr;
    const auto* captureLinking = linking != nullptr ? linking.get() : keepVariant != nullptr ? keepVariant->linking.get() : nullptr;
    if (resolving) {
        auto& parked = resolvedAhead()[currentPacketOffset()];
        parked = ResolvedDispatch{};
        parked.address = address;
        parked.key = key;
        parked.indirectArguments = indirectArguments;
        parked.forgetAtCapture = forgetAtCapture;
        std::copy_n(packet.begin(), std::min(packet.size(), parked.packet.size()), parked.packet.begin());
        parked.registeredShader = registeredShader;
        parked.compiledResult = compiledResult;
        parked.keepVariant = keepVariant;
        parked.attachVariant = attachVariant;
        parked.missedEntry = missedEntry;
        parked.shaderMemory = shaderMemory;
        parked.linking = linking;
        parked.captured = std::move(captured);
        parked.liveWords = std::move(liveWords);
        parked.capture = capture;
        parked.dataHit = dataHit;
        parked.cached = cached;
        parked.validated = validated;
        parked.missedDiffering = missedDiffering;
        parked.noDispatchCache = noDispatchCache || (compiledResult != nullptr && !CacheableResult(*compiledResult));
        parked.resolvedAt = std::chrono::steady_clock::now();
        ++groupStats.resolved;
        if (cached) ++groupStats.resolvedHits;
        if (!cached && compiledResult != nullptr && !localDevice->HasComputePipeline(*compiledResult)) {
            SubmitComputePrebuild([localDevice, compiledResult, address] {
                try { localDevice->PrebuildComputePipeline(*compiledResult, address); } catch (const std::exception&) {}
            });
        }
        return;
    }
    const bool relocatedHit = keepVariant != nullptr && keepVariant->relocated;
    if (verifyDataHits() && dataHit && !relocatedHit) verifyDataHit(snapshot, codeOffset, request, memory, address, *keepVariant, liveWords, *compiledResult);
    if (relocatedHit) verifyRelocatedHit(snapshot, codeOffset, request, memory, address, *keepVariant, liveWords, *compiledResult);
    std::shared_ptr<DispatchVariant> relocatedAttach;
    if (relocatedHit && keepVariant->donor != nullptr && keepVariant->donor->compiled != nullptr && SameDispatchResult(*keepVariant->compiled, *keepVariant->donor->compiled)) relocatedAttach = keepVariant->donor;

    if (recordQueuedLabelsAfterCapture(submission.queue, captured)) {
        dispatch(queue, packet, submission, indirectArguments);
        return;
    }
    phaseTiming.Phase(PhaseQueuedLabels);
    const auto& compiled = *compiledResult;
    if (!ExpectDeviceImages(*localDevice, userData, address, compiled, true)) {
        dispatch(queue, packet, submission, indirectArguments);
        return;
    }
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    for (const auto& region : captured) snapshots.push_back({region.guestAddress, region.bytes});
    std::array<std::uint32_t, 3> groups{packet[1], packet[2], packet[3]};
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {

        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            const auto threads = std::max(readRegister(queue.shader, 0x207 + axis) & 0xffffu, 1u);
            groups[axis] = ThreadDimensionGroups(groups[axis], threads);
        }
    }
    const std::uint64_t dispatchThreads = indirectArguments != 0 ? 0 : std::uint64_t{groups[0]} * groups[1] * groups[2] * compute.numThreads[0] * compute.numThreads[1] * compute.numThreads[2];
    static const bool traceIo = std::getenv("APS5_TRACE_DISPATCH_IO") != nullptr;
    if (traceIo) {

        std::string words;
        if (std::getenv("APS5_TRACE_DISPATCH_IO")[0] == '2') {
            for (const auto word : userData) {
                char text[12];
                std::snprintf(text, sizeof(text), " %08x", word);
                words += text;
            }
        }
        std::fprintf(stderr, "[dispatch-io] queue=0x%x submission=%llu shader 0x%llx%s\n", submission.queue, static_cast<unsigned long long>(submission.serial), static_cast<unsigned long long>(address), words.c_str());
    }

    const auto rethrow = [&](const std::exception& error) {
        if (fromAhead) {
            ++groupStats.deviceRejects;
            dispatch(queue, packet, submission, indirectArguments);
            return true;
        }
        char where[64];
        std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
        throw std::runtime_error(where + std::string(error.what()) + describeIndirectSource());
    };
    phaseTiming.Phase(PhaseSnapshots);

    bool plain = false;
    if ((pipelined && (DispatchMissItemsEnabled() || indirectArguments != 0)) || lateItem || recipeLess) {
        NotPlain why = NotPlain::Count;
        if (compiledResult == nullptr) why = NotPlain::NoResult;
        else if (htileClearCompletion.has_value() || htileMask.bytes != 0u || (indirectArguments != 0 && htileFixup.has_value())) why = NotPlain::HtileClear;
        else if (!partialLastGroup && MatchSgprPatternFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute)) why = NotPlain::KeysFill;
        else if (!partialLastGroup) {
            if (const auto clear = Graphics::MatchStorageImageClear(std::span(snapshot.code).subspan(codeOffset), userData, compute, groups)) {
                const auto clearAddress = Graphics::StorageImageClearAddress(*clear);
                if (clearAddress != 0 && Graphics::DepthSurfaceAt(clearAddress)) why = NotPlain::DepthClear;
            }
        }
        plain = why == NotPlain::Count;
        if (!plain) itemNotPlainKinds[static_cast<std::size_t>(why)].fetch_add(1, std::memory_order_relaxed);
        else if (htileFixup.has_value()) itemFixups.fetch_add(1, std::memory_order_relaxed);
        if (!plain) census.notPlain = true;
    }
    if (pipelined && (DispatchMissItemsEnabled() || indirectArguments != 0) && !plain) {
        pipelined = false;
        itemWrites.clear();
        itemNotPlain.fetch_add(1, std::memory_order_relaxed);
        if (orderedIndirect && indirectArguments != 0) DrawFinisher::NoteFrameKept(FrameKept::IndirectNotPlain);
        if (DrawPipeline::Queue0().Busy()) {
            DrawFinisher::NoteDeepKept(DeepKept::WorkerDispatchNotPlain);
            DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        }
    }
    if (lateItem || recipeLess) {
        bool queued = plain;
        if (queued && lateItem) {
            queued = std::none_of(captured.begin(), captured.end(), [](const auto& region) { return DrawPipeline::Queue0().Overlaps(region.guestAddress, region.bytes.size()); });
            if (!queued && DrawFinisher::FlowSkips(DrawFinisher::FlowSite::DispatchReads)) queued = true;
            if (!queued && DrawFinisher::Deep()) queued = awaitRegionWriters(captured, DeepSite::LateDispatchReads);
            if (!queued) NoteItemFallback(ItemFallback::Reads);
        } else if (!plain) {
            NoteItemFallback(lateItem ? ItemFallback::Miss : ItemFallback::NoRecipe);
        }
        if (queued) {
            bool bounded = false;
            itemWrites = DispatchWriteRanges(*compiledResult, bounded);
            pipelined = bounded;
            writesUnbounded = !bounded;
            if (!bounded) NoteItemFallback(ItemFallback::Writes);
        }
        if (!pipelined && orderedIndirect && indirectArguments != 0) DrawFinisher::NoteFrameKept(!plain ? FrameKept::IndirectNotPlain : writesUnbounded ? FrameKept::IndirectUnbounded : FrameKept::IndirectFallback);
        if (pipelined) {
            (lateItem ? (relocatedHit ? itemRelocated : itemMisses) : itemRecipeLess).fetch_add(1, std::memory_order_relaxed);
        } else if (DrawPipeline::Queue0().Busy()) {
            DrawFinisher::NoteDeepKept(DeepKept::WorkerDispatchLate);
            DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        }
    }
    if (pipelined) {
        std::shared_ptr<IndirectProbe> indirectProbe;
        if (indirectArguments != 0) indirectProbe = makeIndirectProbe(indirectArguments, packet[4]);
        runCaptureCheckHook();
        if (capturedMemoryChanged(captured, {}, captureLinking)) {
            const RecaptureScope recapture;
            dispatch(queue, packet, submission, indirectArguments);
            return;
        }
        const auto recipe = keepVariant != nullptr ? LiveRecipe(*keepVariant) : nullptr;
        const auto attach = stampValidate() ? nullptr : keepVariant != nullptr ? (relocatedHit ? relocatedAttach : keepVariant) : attachVariant;
        const std::uint64_t prepareThreads = lateItem || recipeLess ? dispatchThreads : 0;
        const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        const auto queueId = submission.queue;
        itemCommits.fetch_add(1, std::memory_order_relaxed);
        census.Queued(recipe != nullptr, lateItem, relocatedHit, keepVariant.get());
        if (indirectArguments != 0) {
            std::optional<DrawPipeline::OrderProbe> orderProbe;
            if (DrawFinisher::FrameVerifyDue(FrameCheck::IndirectOrder)) orderProbe.emplace(FrameCheck::IndirectOrder);
            DrawFinisher::Get().NoteIndirectQueued();
            DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::IndirectDispatch, [this, localDevice = std::move(localDevice), result = compiledResult, capture, registeredViews, snapshots = std::move(snapshots), liveWords = std::move(liveWords), groups, address, queueId, recipe, keepVariant, attach, dataHit, epoch, htileClearCompletion, indirectArguments, threadGroup, text = describeIndirectSource(), probe = std::move(indirectProbe)] {
                struct Committed {
                    ~Committed() { DrawFinisher::Get().NoteIndirectCommitted(); }
                } committed;
                DrawPipeline::FollowEpoch(epoch);
                GuestMemory::SetCurrentPacket(0x16, queueId);
                VulkanDevice::CommitTraceScope traced;
                verifyIndirectCommit(probe);
                const auto& compiledCommit = *result;
                const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
                try {
                    std::shared_ptr<RecipeHit> recipeHit = htileClearCompletion.has_value() ? nullptr : localDevice->PrepareRecipe(recipe, true);
                    traced.Step(VulkanDevice::CommitStep::Prepare);
                    std::shared_ptr<const Recipe> builtRecipe;
                    bool writersNoted = false;
                    for (;;) {
                        std::shared_ptr<PreparedDispatch> prepared;
                        if (recipeHit == nullptr || VulkanDevice::VerifyRecipes()) prepared = localDevice->PrepareDispatch(compiledCommit, snapshots, 0, address, htileClearCompletion, registeredViews);
                        traced.Step(VulkanDevice::CommitStep::Prepare);
                        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Indirect);
                        std::lock_guard gpuLock(GuestMemory::GpuMutex());
                        GuestMemory::BreakProofs(GuestMemory::ProofBreak::GpuLock);
                        traced.Step(VulkanDevice::CommitStep::Lock);
                        recordLabelsForPacket(localDevice.get(), queueId);
                        traced.Step(VulkanDevice::CommitStep::Labels);
                        if (noteWrites && writerKeyedEvidence() && !writersNoted) {
                            noteWrittenBuffers(address, queueId, compiledCommit);
                            writersNoted = true;
                        }
                        if (recipeHit != nullptr) {
                            VulkanDevice::IndirectOutcome outcome{0, 0};
                            const auto recipeResult = localDevice->DispatchRecipe(compiledCommit, groups[0], groups[1], groups[2], indirectArguments, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit, threadGroup);
                            traced.Skip();
                            if (recipeResult == RecipeOutcome::Rebuild) {
                                recipeHit = nullptr;
                                VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, true);
                                continue;
                            }
                            countIndirect(outcome.cpuReason, outcome.argumentReadMs);
                        } else {
                            const auto outcome = localDevice->DispatchIndirect(compiledCommit, indirectArguments, snapshots, address, std::move(prepared), attach == nullptr ? nullptr : &builtRecipe, threadGroup, registeredViews);
                            traced.Skip();
                            countIndirect(outcome.cpuReason, outcome.argumentReadMs);
                        }
                        traced.Step(VulkanDevice::CommitStep::Notes);
                        break;
                    }
                    if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, queueId, compiledCommit);
                    if (builtRecipe != nullptr) {
                        if (dataHit) {
                            auto own = std::make_shared<Recipe>(*builtRecipe);
                            own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                            builtRecipe = std::move(own);
                        }
                        attach->recipe.store(std::move(builtRecipe), std::memory_order_release);
                        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, true);
                    }
                    traced.Step(VulkanDevice::CommitStep::Attach);
                } catch (const std::exception& error) {
                    char where[64];
                    std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
                    throw std::runtime_error(where + std::string(error.what()) + text);
                }
            }, std::move(itemWrites));
            orderProbe.reset();
            DrawFinisher::NoteFrameSite(FrameSite::IndirectQueued);
            ReportDispatchItems(profile);
            return;
        }
        DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Dispatch, [this, localDevice = std::move(localDevice), result = compiledResult, capture, registeredViews, snapshots = std::move(snapshots), liveWords = std::move(liveWords), groups, address, queueId, recipe, keepVariant, attach, prepareThreads, dataHit, epoch, htileClearCompletion, htileMask, itemFixup = DispatchMissItemsEnabled() ? htileFixup : std::nullopt] {
            DrawPipeline::FollowEpoch(epoch);
            GuestMemory::SetCurrentPacket(0x15, queueId);
            VulkanDevice::CommitTraceScope traced;
            if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpDispatchNoCommit)) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                recordLabelsForPacket(localDevice.get(), queueId);
                return;
            }
            const auto& compiledCommit = *result;
            const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
            try {
                std::shared_ptr<RecipeHit> recipeHit = htileClearCompletion.has_value() ? nullptr : localDevice->PrepareRecipe(recipe, false);
                traced.Step(VulkanDevice::CommitStep::Prepare);
                std::shared_ptr<const Recipe> builtRecipe;
                bool writersNoted = false;
                for (;;) {
                    std::shared_ptr<PreparedDispatch> prepared;
                    if (recipeHit == nullptr || VulkanDevice::VerifyRecipes() || htileClearCompletion.has_value()) prepared = localDevice->PrepareDispatch(compiledCommit, snapshots, prepareThreads, address, htileClearCompletion, registeredViews);
                    traced.Step(VulkanDevice::CommitStep::Prepare);
                    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                    std::lock_guard gpuLock(GuestMemory::GpuMutex());
                    traced.Step(VulkanDevice::CommitStep::Lock);
                    recordLabelsForPacket(localDevice.get(), queueId);
                    traced.Step(VulkanDevice::CommitStep::Labels);
                    if (noteWrites && writerKeyedEvidence() && !writersNoted) {
                        noteWrittenBuffers(address, queueId, compiledCommit);
                        writersNoted = true;
                    }
                    const bool fixupCarried = itemFixup.has_value() && localDevice->HtileClearStateCurrent(*itemFixup);
                    traced.Step(VulkanDevice::CommitStep::Notes);
                    if (recipeHit != nullptr) {
                        VulkanDevice::IndirectOutcome outcome{0, 0};
                        const auto recipeResult = localDevice->DispatchRecipe(compiledCommit, groups[0], groups[1], groups[2], 0, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit);
                        traced.Skip();
                        if (recipeResult == RecipeOutcome::Rebuild) {
                            recipeHit = nullptr;
                            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, false);
                            itemRebuilds.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                    } else {
                        localDevice->Dispatch(compiledCommit, groups[0], groups[1], groups[2], snapshots, address, std::move(prepared), htileClearCompletion.has_value() || attach == nullptr ? nullptr : &builtRecipe, htileClearCompletion, registeredViews);
                        traced.Skip();
                    }
                    if (htileMask.bytes != 0u) localDevice->NoteHtileMask(htileMask);
                    if (itemFixup.has_value()) localDevice->NoteHtileZmaskFixup(*itemFixup, fixupCarried);
                    traced.Step(VulkanDevice::CommitStep::Notes);
                    break;
                }
                if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, queueId, compiledCommit);
                if (builtRecipe != nullptr) {
                    if (dataHit) {
                        auto own = std::make_shared<Recipe>(*builtRecipe);
                        own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                        builtRecipe = std::move(own);
                    }
                    attach->recipe.store(std::move(builtRecipe), std::memory_order_release);
                    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, false);
                }
                traced.Step(VulkanDevice::CommitStep::Attach);
            } catch (const std::exception& error) {
                char where[64];
                std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
                throw std::runtime_error(where + std::string(error.what()));
            }
        }, std::move(itemWrites));
        ReportDispatchItems(profile);
        return;
    }

    std::shared_ptr<RecipeHit> recipeHit;
    if (cached && keepVariant != nullptr && !stampValidate()) {
        recipeHit = htileClearCompletion.has_value() ? nullptr : localDevice->PrepareRecipe(LiveRecipe(*keepVariant), indirectArguments != 0);
        phaseTiming.Phase(PhaseRecipePrecheck);
    }
    census.Inline(pipelineCandidate, cached, keepVariant.get(), recipeHit != nullptr, htileClearCompletion.has_value(), stampValidate());

    std::shared_ptr<const Recipe> builtRecipe;
    auto* const attachTo = stampValidate() ? nullptr : keepVariant != nullptr ? (relocatedHit ? relocatedAttach.get() : keepVariant.get()) : attachVariant.get();
    bool writersNoted = false;
    const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
    const auto keysFill = indirectArguments == 0 && !partialLastGroup ? MatchSgprPatternFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute) : std::nullopt;
    const auto filledKeys = keysFill ? KeysFillMemo(keysFill->pattern, false, true) : std::nullopt;
    if (keysFill) Graphics::RetireMultisampleColorTargetsOnFmaskWrite(keysFill->base, keysFill->bytes);

    std::optional<Graphics::DepthPlaneClear> depthPlaneClear;
    if (indirectArguments == 0 && !partialLastGroup) {
        if (const auto clear = Graphics::MatchStorageImageClear(std::span(snapshot.code).subspan(codeOffset), userData, compute, groups)) {
            const auto clearAddress = Graphics::StorageImageClearAddress(*clear);
            if (clearAddress != 0 && Graphics::DepthSurfaceAt(clearAddress)) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                std::lock_guard clearLock(GuestMemory::GpuMutex());
                recordLabelsForPacket(localDevice.get(), submission.queue);
                depthPlaneClear.emplace(*clear);
            }
        }
    }
    bool captureChecked = fromAhead;
    // Under range proofs a direct dispatch whose writes are all known breaks template proofs over
    // those writes and the labels it records, instead of every proof through its GPU lock
    // (APS5_NO_RANGED_DISPATCH_BREAKS turns it off). The copied write-backs and storage results land
    // inside those ranges, and their completions break where they store. Depth plane, HTILE and DCC
    // key clears change surface state outside the ranges, so they keep the full break, as do
    // dispatches whose writes are not bounded and indirect ones.
    std::vector<DrawPipeline::Range> rangedBreaks;
    bool rangedDispatch = false;
    if (indirectArguments == 0 && GuestMemory::RangeProofs() && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoRangedDispatchBreaks) && !htileClearCompletion.has_value() && htileMask.bytes == 0u && !depthPlaneClear.has_value() && !keysFill) {
        rangedBreaks = DispatchWriteRanges(compiled, rangedDispatch);
    }

    for (;;) {

        std::shared_ptr<PreparedDispatch> prepared;
        if (recipeHit == nullptr || VulkanDevice::VerifyRecipes()) {
            try {
                prepared = localDevice->PrepareDispatch(compiled, snapshots, dispatchThreads, address, htileClearCompletion, registeredViews);
            } catch (const std::exception& error) {
                if (rethrow(error)) return;
            }
            if (profile) {
                const auto now = std::chrono::steady_clock::now();
                const auto prepareMs = std::chrono::duration<double, std::milli>(now - phaseLap).count();
                phaseLap = now;
                double parts = 0;
                if (prepared != nullptr) {
                    const auto phases = VulkanDevice::PreparePhaseMs(*prepared);
                    phaseMs[PhasePrepareKey] += phases[0];
                    phaseMs[PhasePrepareFind] += phases[1];
                    phaseMs[PhasePreparePrecollect] += phases[2];
                    phaseMs[PhasePreparePresync] += phases[3];
                    phaseMs[PhasePrepareStageA] += phases[4];
                    for (const auto part : phases) parts += part;
                }
                phaseMs[PhasePrepareOther] += std::max(0.0, prepareMs - parts);
            }
        }
        GuestMemory::TagGpuLockSite(rangedDispatch ? GuestMemory::GpuLockSite::RangedDispatch : indirectArguments != 0 ? GuestMemory::GpuLockSite::Indirect : GuestMemory::GpuLockSite::Dispatch);
        std::unique_lock gpuLock(GuestMemory::GpuMutex());
        phaseTiming.Phase(PhaseLockWait);
        if (!captureChecked) {
            captureChecked = true;
            runCaptureCheckHook();
            if (capturedMemoryChanged(captured, snapshots, captureLinking)) {
                gpuLock.unlock();
                const RecaptureScope recapture;
                dispatch(queue, packet, submission, indirectArguments);
                return;
            }
        }

        if (rangedDispatch) {
            for (const auto& label : deferredLabels().labels) rangedBreaks.emplace_back(label.address, label.address + label.size);
        }
        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(PhaseLabels);
        if (noteWrites && writerKeyedEvidence() && !writersNoted) {
            noteWrittenBuffers(address, submission.queue, compiled);
            writersNoted = true;
        }
        phaseTiming.Phase(PhaseNoteWriters);
        auto* const keysRecorder = filledKeys ? Graphics::Recorder::Active() : nullptr;
        const auto keysNoteBefore = keysRecorder != nullptr ? keysRecorder->LastWriteNote(keysFill->base, keysFill->bytes) : 0;
        const bool htileFixupCarried = htileFixup.has_value() && localDevice->HtileClearStateCurrent(*htileFixup);
        try {
            if (recipeHit != nullptr) {
                VulkanDevice::IndirectOutcome outcome{0, 0};
                const auto result = localDevice->DispatchRecipe(compiled, groups[0], groups[1], groups[2], indirectArguments, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit, threadGroup);
                if (result == RecipeOutcome::Rebuild) {

                    recipeHit = nullptr;
                    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, indirectArguments != 0);
                    continue;
                }
                if (indirectArguments != 0) countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else if (indirectArguments != 0) {
                const auto outcome = localDevice->DispatchIndirect(compiled, indirectArguments, snapshots, address, std::move(prepared), attachTo != nullptr ? &builtRecipe : nullptr, threadGroup, registeredViews);
                countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else {
                localDevice->Dispatch(compiled, groups[0], groups[1], groups[2], snapshots, address, std::move(prepared), htileClearCompletion.has_value() ? nullptr : (attachTo != nullptr ? &builtRecipe : nullptr), htileClearCompletion, registeredViews);
            }
        } catch (const std::exception& error) {
            // A failed dispatch may have changed more than its writes before it threw.
            if (rangedDispatch) GuestMemory::BreakProofs(GuestMemory::ProofBreak::Dispatch);
            if (rethrow(error)) return;
        }
        if (rangedDispatch) {
            for (const auto& [begin, end] : rangedBreaks) GuestMemory::BreakProofs(GuestMemory::ProofBreak::Dispatch, begin, end);
        }
        if (keysRecorder != nullptr && keysRecorder == Graphics::Recorder::Active()) {
            const auto note = keysRecorder->LastWriteNote(keysFill->base, keysFill->bytes);
            if (note != 0 && note != keysNoteBefore) Graphics::NoteKeysFillOnGpu(keysFill->base, keysFill->bytes, *filledKeys);
        }
        if (htileMask.bytes != 0u) localDevice->NoteHtileMask(htileMask);
        if (htileFixup.has_value()) localDevice->NoteHtileZmaskFixup(*htileFixup, htileFixupCarried);
        if (builtRecipe != nullptr) {
            if (dataHit) {

                auto own = std::make_shared<Recipe>(*builtRecipe);
                own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                builtRecipe = std::move(own);
            }
            attachTo->recipe.store(std::move(builtRecipe), std::memory_order_release);
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, indirectArguments != 0);
        }
        break;
    }
    phaseTiming.Phase(PhaseDevice);
    if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, submission.queue, compiled);
    if (profile) deviceMs.fetch_add(phaseTiming.Elapsed(), std::memory_order_relaxed);
    phaseTiming.Phase(PhaseTail);
    if (profile) {
        auto& pending = pendingDispatchPhases();
        pending.outcome = DispatchOutcome::Real;
        pending.phases = true;
        pending.hit = cached;
        pending.validated = validated;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
    }
    if (profile && dispatches.fetch_add(1, std::memory_order_relaxed) % 100 == 99) {
        std::uint64_t hits = 0;
        std::uint64_t evictions = 0;
        {
            std::lock_guard cacheLock(dispatchCacheMutex);
            hits = dispatchCacheHits;
            evictions = dispatchCacheEvictions;
        }
        AgcDriver::ProfilePrint_nid_no_patch( "[gpu] %llu dispatches (%llu dispatch cache hits, %llu evictions, %llu recompile cache hits): capture %.1f s, cache key %.1f s, recompile %.1f s, device %.1f s\n", static_cast<unsigned long long>(dispatches.load(std::memory_order_relaxed)), static_cast<unsigned long long>(hits), static_cast<unsigned long long>(evictions), static_cast<unsigned long long>(cacheHits.load(std::memory_order_relaxed)), captureMs.load(std::memory_order_relaxed) / 1000, keyMs.load(std::memory_order_relaxed) / 1000, recompileMs.load(std::memory_order_relaxed) / 1000, deviceMs.load(std::memory_order_relaxed) / 1000);
    }
}

}
