#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/LeanDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectDrawReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanFront.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::uint64_t FnvPrime = 0x100000001b3ull;
constexpr std::array<std::pair<std::uint32_t, std::size_t>, 4> KeyedBases{{{0x00cu, 0}, {0x04cu, 1}, {0x08cu, 2}, {0x10cu, 4}}};
constexpr std::array<std::pair<std::size_t, std::size_t>, 5> IdentityBases{{{0, 0}, {2, 2}, {3, 3}, {4, 4}, {5, 5}}};

std::uint64_t fold(std::uint64_t key, std::span<const std::uint64_t> values) {
    for (const auto value : values) key = (key ^ value) * FnvPrime;
    return key;
}

std::uint32_t userWordCount(std::uint32_t resources) {
    return ((resources >> 1u) & 0x1fu) | (((resources >> 27u) & 1u) << 5u);
}

bool allUserWords() {
    static const bool all = std::getenv("APS5_DRAW_KEY_ALL_USER_WORDS") != nullptr;
    return all;
}

std::uint32_t userWordRegister(std::size_t index) {
    if (index < 128) return DrawUserWords::Bases[index / 32] + static_cast<std::uint32_t>(index % 32);
    return index < 130 ? 0x082u + static_cast<std::uint32_t>(index - 128) : 0x102u + static_cast<std::uint32_t>(index - 130);
}

bool userWordPresent(const DrawUserWords& words, std::size_t index) {
    return ((words.present[index / 64] >> (index % 64)) & 1u) != 0;
}

void captureUserWords(DrawUserWords& words, const Registers& shader, bool blocks) {
    if (!blocks) {
        words.Capture(shader);
        return;
    }
    words.CaptureBlocks(shader);
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_USER_WORD_BLOCKS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    static std::atomic<std::uint64_t> ordinal{0}, checked{0}, differing{0};
    static std::atomic<std::int64_t> window{-1};
    if (every == 0 || ordinal.fetch_add(1, std::memory_order_relaxed) % every != 0) return;
    checked.fetch_add(1, std::memory_order_relaxed);
    DrawUserWords today;
    today.Capture(shader);
    std::size_t at = 0;
    while (at < DrawUserWords::Count && today.values[at] == words.values[at] && userWordPresent(today, at) == userWordPresent(words, at)) ++at;
    std::size_t resource = 0;
    while (resource < today.resources.size() && today.resources[resource] == words.resources[resource]) ++resource;
    if (at != DrawUserWords::Count || resource != today.resources.size() || today.present != words.present) {
        if (differing.fetch_add(1, std::memory_order_relaxed) < 16) {
            if (at != DrawUserWords::Count) std::fprintf(stderr, "[user-word-blocks-verify] register 0x%x differs: blocks 0x%08x (%s), today 0x%08x (%s)\n", userWordRegister(at), words.values[at], userWordPresent(words, at) ? "set" : "unset", today.values[at], userWordPresent(today, at) ? "set" : "unset");
            else if (resource != today.resources.size()) std::fprintf(stderr, "[user-word-blocks-verify] resource register 0x%x differs: blocks 0x%08x, today 0x%08x\n", DrawUserWords::ResourceRegisters[resource], words.resources[resource], today.resources[resource]);
            else std::fprintf(stderr, "[user-word-blocks-verify] the present bits differ past the captured words\n");
        }
        words = today;
    }
    const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (window.exchange(now, std::memory_order_relaxed) != now) std::fprintf(stderr, "[user-word-blocks-verify] %llu draws checked, %llu differing\n", static_cast<unsigned long long>(checked.load(std::memory_order_relaxed)), static_cast<unsigned long long>(differing.load(std::memory_order_relaxed)));
}

std::uint64_t leanDrawKey(const LeanShape& shape, const DrawUserWords& words, bool unordered) {
    const auto key = LeanExactKey(shape, words, unordered);
    if (!unordered) return key;
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_UNORDERED_DRAW_KEYS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    static std::atomic<std::uint64_t> ordinal{0};
    if (every == 0 || ordinal.fetch_add(1, std::memory_order_relaxed) % every != 0) return key;
    struct Seen {
        std::uint64_t key = 0;
        std::uint64_t other = 0;
    };
    constexpr std::size_t Slots = 4096;
    static std::array<Seen, Slots> byNew{}, byOld{};
    static std::atomic<std::uint64_t> checked{0}, collisions{0};
    static std::atomic<std::int64_t> window{-1};
    const auto old = LeanExactKey(shape, words, false);
    checked.fetch_add(1, std::memory_order_relaxed);
    auto& sameNew = byNew[static_cast<std::size_t>(key % Slots)];
    if (sameNew.key == key && sameNew.other != old && collisions.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[unordered-keys-verify] new key 0x%llx old keys 0x%llx and 0x%llx\n", static_cast<unsigned long long>(key), static_cast<unsigned long long>(sameNew.other), static_cast<unsigned long long>(old));
    sameNew = {key, old};
    auto& sameOld = byOld[static_cast<std::size_t>(old % Slots)];
    if (sameOld.key == old && sameOld.other != key && collisions.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[unordered-keys-verify] old key 0x%llx new keys 0x%llx and 0x%llx\n", static_cast<unsigned long long>(old), static_cast<unsigned long long>(sameOld.other), static_cast<unsigned long long>(key));
    sameOld = {old, key};
    const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (window.exchange(now, std::memory_order_relaxed) != now) std::fprintf(stderr, "[unordered-keys-verify] %llu draws checked, %llu collisions\n", static_cast<unsigned long long>(checked.load(std::memory_order_relaxed)), static_cast<unsigned long long>(collisions.load(std::memory_order_relaxed)));
    return key;
}

bool runtimeRegister(std::uint32_t offset) {
    return offset - 0x00cu < 32u || offset - 0x04cu < 16u || offset - 0x08cu < 32u || offset - 0x10cu < 32u || offset == 0x082u || offset == 0x083u || offset == 0x102u || offset == 0x103u;
}

std::uint64_t presentRun(const DrawUserWords& words, std::size_t index, std::uint32_t count) {
    if (count == 0) return 0;
    const auto bits = words.present[index / 64] >> (index % 64);
    return count >= 64 ? bits : bits & ((std::uint64_t{1} << count) - 1u);
}

template <typename TVisit>
void forEachPresent(const DrawUserWords& words, std::uint32_t base, std::size_t index, std::uint32_t count, TVisit&& visit) {
    for (auto bits = presentRun(words, index, count); bits != 0; bits &= bits - 1) {
        const auto word = static_cast<std::uint32_t>(std::countr_zero(bits));
        visit(base + word, words.values[index + word]);
    }
}

std::size_t verdictIndex(const Pm4::DrawParameters& parameters) {
    if (!parameters.indexed) return 0;
    return parameters.indexSize == 2 ? 1 : parameters.indexSize == 4 ? 2 : 3;
}

std::string where(const Submission& submission, std::size_t packetOffset) {
    return "draw at submission " + std::to_string(submission.serial) + " DWORD " + std::to_string(packetOffset);
}

std::string hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

bool leanQualified(const DrawDecode& decode) {
    if (decode.roles.size() != decode.programs.size()) return false;
    for (std::size_t i = 0; i < decode.programs.size(); ++i) {
        const auto& program = decode.programs[i];
        if (program.snapshot == nullptr || program.snapshot->header.empty() || program.tailContinuation != nullptr) return false;
        if (program.userData.size() < program.hiddenUserWords || program.userData.size() - program.hiddenUserWords > 32u) return false;
        if (decode.roles[i] != ShaderRecompiler::ProgramRole::GeometryBack && ProgramWritesMemory(*program.snapshot, program.codeOffset)) return false;
    }
    return true;
}

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

class WorkerClock {
public:
    explicit WorkerClock(std::optional<LeanFallbacks::Reason> reason) : reason(reason), start(LeanFallbacks::Stats() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    ~WorkerClock() {
        if (start == std::chrono::steady_clock::time_point{}) return;
        const auto spent = elapsedNs(start);
        if (reason) LeanFallbacks::AddToday(*reason, spent);
        else LeanFallbacks::AddLivePrecheck(spent);
    }
    WorkerClock(const WorkerClock&) = delete;
    WorkerClock& operator=(const WorkerClock&) = delete;

private:
    std::optional<LeanFallbacks::Reason> reason;
    std::chrono::steady_clock::time_point start;
};

class FallbackScope {
public:
    explicit FallbackScope(LeanFallbackProbe& probe) : probe(probe), start(LeanFallbacks::Stats() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {}
    ~FallbackScope() {
        probe.force = false;
        if (route != LeanFallbacks::Route::Count) LeanFallbacks::AddMoved(route, start == std::chrono::steady_clock::time_point{} ? 0 : elapsedNs(start));
    }
    FallbackScope(const FallbackScope&) = delete;
    FallbackScope& operator=(const FallbackScope&) = delete;
    void Moved(LeanFallbacks::Route taken) { route = taken; }

private:
    LeanFallbackProbe& probe;
    std::chrono::steady_clock::time_point start;
    LeanFallbacks::Route route = LeanFallbacks::Route::Count;
};

LeanFallbacks::Reason fallbackReason(LeanFallbackCause cause) {
    switch (cause) {
        case LeanFallbackCause::ColorMetadata: return LeanFallbacks::Reason::ColorMetadata;
        case LeanFallbackCause::DepthCopy: return LeanFallbacks::Reason::DepthCopy;
        case LeanFallbackCause::RenderControl: return LeanFallbacks::Reason::RenderControl;
        case LeanFallbackCause::Maintenance: return LeanFallbacks::Reason::Maintenance;
        case LeanFallbackCause::Rejection: return LeanFallbacks::Reason::Rejection;
        default: return LeanFallbacks::Reason::Other;
    }
}

void noteFallbackVerify(LeanFallbackProbe& probe, const Submission& submission, std::size_t packetOffset) {
    const bool differed = std::exchange(probe.differed, false);
    LeanFallbacks::AddVerify(probe.route, differed, differed ? where(submission, packetOffset) + ": the lean answer differs from today's draw (the [lean-front-verify] lines name the field)" : std::string());
}

}

std::shared_ptr<DrawDecode> LeanRefreshedDecode(const DrawDecode& from, const DrawUserWords& words) {
    auto refreshed = std::make_shared<DrawDecode>(from);
    for (auto& program : refreshed->programs) {
        ReadUserWordsFrom(words, program);
        if (program.merged) InitializeMergedFrom(words, program, program.mergedPointer, program.mergedPointerRequired);
    }
    return refreshed;
}

std::uint64_t LeanExactKey(const LeanShape& shape, const DrawUserWords& words, bool unordered) {
    if (unordered) {
        std::array<std::uint64_t, KeyedBases.size()> banks{};
        for (std::size_t base = 0; base < KeyedBases.size(); ++base) {
            const auto [first, resource] = KeyedBases[base];
            const auto limit = first == 0x04cu ? 16u : 32u;
            auto& part = banks[base];
            forEachPresent(words, first, 32u * base, allUserWords() ? limit : std::min(userWordCount(words.resources[resource]), limit), [&](std::uint32_t offset, std::uint32_t value) { part ^= Registers::Contribution(offset, value); });
        }
        std::uint64_t auxiliary = 0;
        const auto auxiliaryWord = [&](std::uint32_t offset, std::uint32_t value) { auxiliary ^= Registers::Contribution(offset, value); };
        forEachPresent(words, 0x082u, 128, 2, auxiliaryWord);
        forEachPresent(words, 0x102u, 130, 2, auxiliaryWord);
        return fold(UnorderedUserWordKey(shape.shape, banks[0] ^ banks[1] ^ banks[2] ^ banks[3], auxiliary), shape.keySuffix);
    }
    auto exact = shape.shape;
    const auto mix = [&](std::uint32_t offset, std::uint32_t value) {
        exact = (exact ^ offset) * FnvPrime;
        exact = (exact ^ value) * FnvPrime;
    };
    for (std::size_t base = 0; base < KeyedBases.size(); ++base) {
        const auto [first, resource] = KeyedBases[base];
        const auto limit = first == 0x04cu ? 16u : 32u;
        forEachPresent(words, first, 32u * base, allUserWords() ? limit : std::min(userWordCount(words.resources[resource]), limit), mix);
    }
    forEachPresent(words, 0x082u, 128, 2, mix);
    forEachPresent(words, 0x102u, 130, 2, mix);
    return fold(exact, shape.keySuffix);
}

std::uint64_t LeanIdentityKey(const LeanShape& shape, const DrawUserWords& words) {
    std::uint64_t key = 0x84222325cbf29ce4ull;
    const auto mix = [&](std::uint64_t value) { key = (key ^ value) * FnvPrime; };
    const auto& set = shape.shaderSet;
    for (const auto [index, resource] : IdentityBases) {
        if (!set.present[index]) {
            mix(0);
            continue;
        }
        mix(set.low[index]);
        mix(set.high[index]);
        mix(words.resources[resource]);
        for (const auto value : set.mixes[index]) mix(value);
    }
    return key == 0 ? 1 : key;
}

void LeanAliasKey(const LeanShape& shape, const DrawUserWords& words, const DrawRelocation* relocation, std::uint64_t deviceSerial, std::uint64_t& alias) {
    if (relocation == nullptr || relocation->words.empty()) return;
    bool readable = true;
    std::uint64_t other = 0x6c62272e07bb0142ull;
    const auto mixOther = [&](std::uint64_t value) { other = (other ^ value) * FnvPrime; };
    mixOther(shape.shape);
    const auto visit = [&](std::uint32_t offset, std::uint32_t value) {
        mixOther(offset);
        const auto word = std::find_if(relocation->words.begin(), relocation->words.end(), [&](const RelocatableUserWord& candidate) { return candidate.reg == offset; });
        mixOther(word != relocation->words.end() ? Driver::relocatedContentKey(*word, value, readable) : value);
    };
    for (const std::size_t base : {std::size_t{0}, std::size_t{2}, std::size_t{3}}) {
        const auto [first, resource] = KeyedBases[base];
        forEachPresent(words, first, 32u * base, allUserWords() ? 32u : std::min(userWordCount(words.resources[resource]), 32u), visit);
    }
    forEachPresent(words, 0x082u, 128, 2, visit);
    forEachPresent(words, 0x102u, 130, 2, visit);
    mixOther(deviceSerial);
    other = fold(other, shape.aliasSuffix);
    if (readable) alias = other == 0 ? 1 : other;
}

ShapeRecord* LeanDraws::Find(std::uint64_t shape, std::uint64_t deviceSerial) {
    if (records.empty()) return nullptr;
    const auto set = static_cast<std::size_t>((shape ^ (deviceSerial * 0x9e3779b97f4a7c15ull)) % Sets) * Ways;
    for (std::size_t way = 0; way < Ways; ++way) {
        auto& record = records[set + way];
        if (!record.used || record.shape != shape || record.deviceSerial != deviceSerial) continue;
        record.lastUse = ++uses;
        return &record;
    }
    return nullptr;
}

ShapeRecord& LeanDraws::Place(std::uint64_t shape, std::uint64_t deviceSerial, std::uint64_t front) {
    if (records.empty()) records.resize(Sets * Ways);
    const auto set = static_cast<std::size_t>((shape ^ (deviceSerial * 0x9e3779b97f4a7c15ull)) % Sets) * Ways;
    auto* victim = &records[set];
    for (std::size_t way = 0; way < Ways; ++way) {
        auto& record = records[set + way];
        if (!record.used) {
            victim = &record;
            break;
        }
        if (record.lastUse < victim->lastUse) victim = &record;
    }
    if (victim->used) {
        Retire(*victim, front);
        LeanFront::AddOnWorker(LeanFront::Counter::ShapeRecordsEvicted);
    }
    *victim = ShapeRecord{};
    victim->used = true;
    victim->shape = shape;
    victim->deviceSerial = deviceSerial;
    victim->lastUse = ++uses;
    return *victim;
}

void LeanDraws::Retire(ShapeRecord& record, std::uint64_t front) {
    if (record.built != nullptr && record.lastSeq >= front) retired.emplace_back(std::move(record.built), record.lastSeq);
    record.built = nullptr;
    RetirePlans(record, front);
    record.plansRefused = false;
}

void LeanDraws::RetirePlans(ShapeRecord& record, std::uint64_t front) {
    if (record.plans == nullptr) return;
    planBytes -= record.plans->bytes;
    --planShapes;
    stagePlans -= record.plans->PlanCount();
    drawPlans -= record.plans->draws.size();
    if (record.plansSeq >= front) retiredPlans.emplace_back(std::move(record.plans), record.plansSeq);
    record.plans = nullptr;
    notePlans();
}

bool LeanDraws::InstallPlans(ShapeRecord& record, std::shared_ptr<const StagePlanSet> set, std::uint64_t front, std::uint64_t present) {
    const auto cap = DrawRecordStore::ConfiguredByteCap();
    const auto held = [&] { return record.plans != nullptr ? record.plans->bytes : std::size_t{0}; };
    if (planBytes - held() + set->bytes > cap) {
        sweepPlans(present, front, &record);
        if (planBytes - held() + set->bytes > cap) return false;
    }
    RetirePlans(record, front);
    planBytes += set->bytes;
    ++planShapes;
    stagePlans += set->PlanCount();
    drawPlans += set->draws.size();
    record.plans = std::move(set);
    record.plansPresent = present;
    notePlans();
    return true;
}

void LeanDraws::sweepPlans(std::uint64_t present, std::uint64_t front, const ShapeRecord* keep) {
    if (sweptPlansPresent == present) return;
    sweptPlansPresent = present;
    for (auto& record : records) {
        if (!record.used || &record == keep || record.plans == nullptr || present <= record.plansPresent + StagePlanLimits::KeepPresents) continue;
        RetirePlans(record, front);
        StagePlans::AddOnWorker(StagePlans::Counter::SetsSwept);
    }
}

void LeanDraws::notePlans() const {
    StagePlans::NoteStore(planShapes, stagePlans, drawPlans, planBytes);
}

void LeanDraws::Clear(std::uint64_t front) {
    std::uint64_t cleared = 0;
    for (auto& record : records) {
        if (!record.used) continue;
        if (record.plans != nullptr) ++cleared;
        Retire(record, front);
        record = ShapeRecord{};
    }
    if (cleared != 0) StagePlans::AddOnWorker(StagePlans::Counter::SetsCleared, cleared);
    sweptPlansPresent = ~std::uint64_t{0};
    identities.Clear(front);
}

void LeanDraws::Release(std::uint64_t front, std::uint64_t issued) {
    if (!retired.empty()) std::erase_if(retired, [&](const auto& entry) { return entry.second < front; });
    if (!retiredPlans.empty()) std::erase_if(retiredPlans, [&](const auto& entry) { return entry.second < front; });
    identities.Release(front);
    while (anchorTail - anchorHead > 1 && anchors[static_cast<std::size_t>((anchorHead + 1) % Anchors)].firstSeq <= front) {
        anchors[static_cast<std::size_t>(anchorHead % Anchors)] = LeanAnchor{};
        ++anchorHead;
    }
    if (anchorTail - anchorHead == 1 && front >= issued && anchors[static_cast<std::size_t>(anchorHead % Anchors)].device != device) {
        anchors[static_cast<std::size_t>(anchorHead % Anchors)] = LeanAnchor{};
        ++anchorHead;
    }
}

void LeanDraws::ReleaseDevice() {
    for (; anchorHead != anchorTail; ++anchorHead) anchors[static_cast<std::size_t>(anchorHead % Anchors)] = LeanAnchor{};
    device = nullptr;
    deviceStores = ~std::uint64_t{0};
}

void LeanDraws::Reset() {
    ReleaseDevice();
    identities.Reset();
    identitiesSerial = 0;
    records.clear();
    retired.clear();
    retiredPlans.clear();
    planBytes = 0;
    planShapes = 0;
    stagePlans = 0;
    drawPlans = 0;
    sweptPlansPresent = ~std::uint64_t{0};
    notePlans();
    registry = nullptr;
    ++registryEpoch;
}

std::optional<std::uint64_t> LeanDraws::AnchorFor(const std::shared_ptr<const ShaderRegistry>& heldRegistry, const std::shared_ptr<VulkanDevice>& heldDevice, std::uint64_t issued) {
    if (anchorTail != anchorHead) {
        const auto& current = anchors[static_cast<std::size_t>((anchorTail - 1) % Anchors)];
        if (current.registry == heldRegistry && current.device == heldDevice) return anchorTail - 1;
    }
    if (anchorTail - anchorHead >= Anchors) return std::nullopt;
    anchors[static_cast<std::size_t>(anchorTail % Anchors)] = LeanAnchor{heldRegistry, heldDevice, issued};
    return anchorTail++;
}

void Driver::prepareLeanDraws() {
    auto& finisher = DrawFinisher::Get();
    finisher.Adopt(true);
    const auto front = finisher.Front();
    if (frontLatch.Renew(leanDraws.stamp)) leanDraws.Clear(front);
    if (leanDraws.device != nullptr && device.Stores() != leanDraws.deviceStores) leanDraws.device = nullptr;
    leanDraws.Release(front, finisher.Issued());
}

bool Driver::leanDrawDevice() {
    auto& lean = leanDraws;
    const auto stores = device.Stores();
    if (stores != lean.deviceStores || lean.device == nullptr) {
        lean.device = device.Load();
        lean.deviceStores = stores;
        if (lean.device != nullptr && lean.device->Serial() != lean.identitiesSerial) {
            lean.identities.Clear(DrawFinisher::Get().Front());
            lean.identitiesSerial = lean.device->Serial();
        }
    }
    return lean.device != nullptr;
}

bool Driver::leanShapeCurrent(const LeanShape& shape, const QueueState& queue, const ShaderRegistry& registry) {
    if (ReadDrawShaderSet(queue, registry) != shape.shaderSet) return false;
    return std::all_of(shape.decode->programs.begin(), shape.decode->programs.end(), [&](const DrawProgram& program) { return ProgramSnapshot(registry, program.binary.codeAddress, {program.snapshot->type}) == program.snapshot; });
}

ShapeRecord* Driver::leanShapeRecord(const QueueState& queue, const Submission& submission) {
    auto& lean = leanDraws;
    if (submission.shaders.get() != lean.registry.get()) {
        lean.registry = submission.shaders;
        ++lean.registryEpoch;
    }
    const auto serial = lean.device->Serial();
    const auto shape = RegisterShapeKey(queue);
    if (auto* record = lean.Find(shape, serial)) {
        if (record->epoch == lean.registryEpoch) return record;
        record->epoch = lean.registryEpoch;
        record->refused = false;
        if (record->built == nullptr) return record;
        try {
            if (leanShapeCurrent(*record->built, queue, *submission.shaders)) {
                LeanFront::AddOnWorker(LeanFront::Counter::ShapeRecordsRestamped);
                return record;
            }
        } catch (const std::exception&) {
        }
        lean.Retire(*record, DrawFinisher::Get().Front());
        return record;
    }
    auto& record = lean.Place(shape, serial, DrawFinisher::Get().Front());
    record.epoch = lean.registryEpoch;
    return &record;
}

LeanVerdict Driver::leanVerdict(const QueueState& queue, ShapeRecord& record, const Pm4::DrawParameters& parameters, bool afterControl) {
    const auto index = verdictIndex(parameters);
    if (index >= record.verdicts.size()) return LeanVerdict::Live;
    auto& verdict = (afterControl ? record.afterControl : record.verdicts)[index];
    if (verdict != LeanVerdict::Unknown) return verdict;
    std::vector<Graphics::RegisterRead> reads;
    {
        const Graphics::RegisterReadLogScope log(&reads);
        verdict = precheckVerdict(queue, parameters.indexed, parameters.indexSize, afterControl ? nullptr : &record.fallbackCauses[index], afterControl);
    }
    if (verdict == LeanVerdict::Normal || verdict == LeanVerdict::Nothing) {
        const bool keyed = std::all_of(reads.begin(), reads.end(), [](const Graphics::RegisterRead& read) { return Graphics::DrawKeyCovers(read) && !(read.bank == Graphics::RegisterBank::Shader && runtimeRegister(read.offset)); });
        if (!keyed) verdict = LeanVerdict::Live;
    }
    return verdict;
}

bool Driver::buildLeanShape(const QueueState& queue, const Submission& submission, ShapeRecord& record) {
    if (record.built != nullptr) return true;
    if (record.refused) return false;
    auto built = std::make_shared<LeanShape>();
    try {
        std::vector<Graphics::RegisterRead> reads;
        std::shared_ptr<DrawDecode> decode;
        {
            const Graphics::RegisterReadLogScope log(&reads);
            decode = decodeDraw(queue, submission, leanDraws.device->graphicsContext());
        }
        const bool keyed = std::all_of(reads.begin(), reads.end(), [](const Graphics::RegisterRead& read) { return Graphics::DrawKeyCovers(read); });
        if (!keyed || !leanQualified(*decode)) {
            record.refused = true;
            LeanFront::AddOnWorker(LeanFront::Counter::ShapesRefused);
            return false;
        }
        built->shape = record.shape;
        built->shaderSet = ReadDrawShaderSet(queue, *submission.shaders);
        built->keySuffix.push_back(record.deviceSerial);
        for (std::size_t index = 0; index < DrawShaderSet::Bases.size(); ++index) {
            const auto& set = built->shaderSet;
            if (!set.present[index]) {
                built->keySuffix.push_back(0);
                continue;
            }
            built->keySuffix.insert(built->keySuffix.end(), set.mixes[index].begin(), set.mixes[index].end());
            if (set.registered[index]) built->aliasSuffix.insert(built->aliasSuffix.end(), set.mixes[index].begin(), set.mixes[index].end());
        }
        built->finalShape = fold(record.shape, built->keySuffix);
        built->reg207 = readRegister(queue.context, 0x207);
        built->skipped = std::any_of(decode->programs.begin(), decode->programs.end(), [](const DrawProgram& program) { return SkippedShader(program.binary.codeAddress); });
        built->writes = DrawWriteRanges(decode->state, {}, ExactDrawWrites());
        built->decode = std::move(decode);
    } catch (const std::exception&) {
        record.refused = true;
        LeanFront::AddOnWorker(LeanFront::Counter::ShapesRefused);
        return false;
    }
    record.built = std::move(built);
    LeanFront::AddOnWorker(LeanFront::Counter::ShapeRecordsBuilt);
    return true;
}

namespace {
std::uint32_t leanRunLimit() {
    static const std::uint32_t limit = [] {
        const char* text = std::getenv("APS5_DRAW_RUN_CAP");
        const auto value = text != nullptr ? std::strtoul(text, nullptr, 10) : 8ul;
        return static_cast<std::uint32_t>(std::clamp<unsigned long>(value, 1ul, LeanRunCap));
    }();
    return limit;
}
}

DrawVerdict Driver::drawLean(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::size_t packetOffset, std::string& rejected) {
    using LeanFront::Counter;
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpSlowWorker)) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(1);
        while (std::chrono::steady_clock::now() < until) {}
    }
    static const bool allowed = [] {
        for (const char* name : {"APS5_PROFILE_DRAW", "APS5_LOCKED_DRAW_PREPARE", "APS5_DUMP_DRAW_SHADERS", "APS5_DUMP_DRAW_SLOT1", "APS5_TRACE_DRAW_PROGRAMS", "APS5_TRACE_SHADER_CORRELATION", "APS5_VERIFY_REGISTER_KEYS"}) {
            if (std::getenv(name) != nullptr) return false;
        }
        return drawEntries() && registerKeyEnabled() && !verifyDrawDataHits() && !verifyDrawRecipe() && !verifyDrawEntries() && !traceDrawCache() && !drawRelocationDrain() && DrawFinisher::Enabled() && DrawFinisher::Overlap();
    }();
    const bool records = frontLatch.records;
    const bool plans = frontLatch.stagePlans;
    const bool direct = frontLatch.directDraws;
    const auto recordsOff = [&](DrawRecords::Counter reason) {
        if (plans) {
            using Off = StagePlans::Counter;
            StagePlans::AddOnWorker(Off::Draws);
            StagePlans::AddOnWorker(reason == DrawRecords::Counter::OffIndirect ? Off::OffIndirect : reason == DrawRecords::Counter::OffLeanVerify ? Off::OffLeanVerify : reason == DrawRecords::Counter::OffInline ? Off::OffInline : Off::OffTodaysDraw);
            return;
        }
        if (!records) return;
        DrawRecords::AddOnWorker(DrawRecords::Counter::Draws);
        DrawRecords::AddOnWorker(reason);
    };
    bool afterControl = false;
    bool movedControl = false;
    const auto today = [&](Counter counter, LeanFallbacks::Reason reason) {
        LeanFront::AddOnWorker(counter);
        recordsOff(counter == Counter::IndirectDrawsOnTodaysPath ? DrawRecords::Counter::OffIndirect : DrawRecords::Counter::OffTodaysDraw);
        if (movedControl) LeanFallbacks::AddLater(LeanFallbacks::Route::RenderControl);
        const WorkerClock clock(reason);
        return draw(queue, packet, submission, packetOffset, rejected);
    };
    auto& finisher = DrawFinisher::Get();
    if (!allowed || !finisher.RingStore() || !DrawPipeline::Active() || ShaderRecompiler::DebugProbeActive()) return today(Counter::DrawsOnTodaysPath, LeanFallbacks::Reason::Other);
    Pm4::DrawParameters parameters{};
    try {
        parameters = Pm4::ResolveDraw(packet, queue);
    } catch (const std::exception&) {
        return today(Counter::DrawsOnTodaysPath, LeanFallbacks::Reason::Other);
    }
    static const bool pipelinedIndirect = std::getenv("APS5_NO_PIPELINED_INDIRECT") == nullptr;
    if (parameters.indirect && !(direct && pipelinedIndirect && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DirectGpuIndirect) && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PipelinedGpuIndirect))) return today(Counter::IndirectDrawsOnTodaysPath, LeanFallbacks::Reason::GpuIndirect);
    const auto todayCounter = parameters.indirect ? Counter::IndirectDrawsOnTodaysPath : Counter::DrawsOnTodaysPath;
    if (!parameters.indirect && !parameters.indexed && (parameters.indexCount == 0 || parameters.instanceCount == 0)) return DrawVerdict::Nothing;
    if (!deferredLabels().labels.empty()) return today(todayCounter, LeanFallbacks::Reason::DeferredLabels);
    if (!leanDrawDevice()) return today(todayCounter, LeanFallbacks::Reason::Other);
    const auto verify = [&](ShapeRecord& checked, LeanVerdict shown, const LeanDrawJob* job, bool forced) {
        const WorkerClock clock(LeanFallbacks::Reason::Verify);
        const auto result = verifyLeanDraw(queue, packet, submission, packetOffset, rejected, checked, shown, parameters, job, afterControl, forced ? &leanFallbackProbe.differed : nullptr);
        if (forced) noteFallbackVerify(leanFallbackProbe, submission, packetOffset);
        return result;
    };
    auto& record = *leanShapeRecord(queue, submission);
    auto verdict = leanVerdict(queue, record, parameters);
    std::optional<FallbackScope> fallbackScope;
    if (verdict == LeanVerdict::Fallback) {
        const auto slot = verdictIndex(parameters);
        const auto cause = slot < record.fallbackCauses.size() ? record.fallbackCauses[slot] : LeanFallbackCause::None;
        const auto reason = fallbackReason(cause);
        const bool head = cause == LeanFallbackCause::ColorMetadata || cause == LeanFallbackCause::DepthCopy || cause == LeanFallbackCause::RenderControl;
        if (!head || !LeanFront::SwitchSet(frontLatch, Graphics::RuntimeSwitch::LeanFallbackDraws)) return today(todayCounter, reason);
        if (parameters.indirect) {
            LeanFallbacks::AddKept(LeanFallbacks::Kept::Indirect);
            return today(todayCounter, reason);
        }
        fallbackScope.emplace(leanFallbackProbe);
        std::optional<Graphics::ColorMetadataPass> pass;
        std::optional<Graphics::DepthCopyPass> copy;
        std::optional<Graphics::DepthRenderControl> control;
        try {
            pass = Graphics::DecodeColorMetadataPass(queue);
            if (!pass) copy = Graphics::DecodeDepthCopyPass(queue);
            if (!pass && !copy) control = Graphics::DecodeDepthRenderControl(queue);
        } catch (const std::exception&) {
            LeanFallbacks::AddKept(LeanFallbacks::Kept::HeadThrew);
            return today(todayCounter, reason);
        }
        if (pass) {
            std::vector<DrawPipeline::Range> writes;
            auto localDevice = pipelinedColorMetadataDevice(*pass, writes);
            if (localDevice == nullptr) {
                LeanFallbacks::AddKept(LeanFallbacks::Kept::MetadataNotPipelined);
                return today(todayCounter, LeanFallbacks::Reason::ColorMetadata);
            }
            if (LeanFallbacks::VerifyDue()) {
                const auto queued = DrawPipeline::PipelinedMetadataPasses();
                const auto result = today(todayCounter, LeanFallbacks::Reason::Verify);
                const bool differed = result != DrawVerdict::Drawn || DrawPipeline::PipelinedMetadataPasses() != queued + 1;
                LeanFallbacks::AddVerify(LeanFallbacks::Route::ColorMetadata, differed, differed ? where(submission, packetOffset) + ": the switch queues the pass through the finisher ring, today's precheck did not queue it as a pipelined pass" : std::string());
                return result;
            }
            fallbackScope->Moved(LeanFallbacks::Route::ColorMetadata);
            recordsOff(DrawRecords::Counter::OffTodaysDraw);
            queueLeanColorMetadataPass(*pass, std::move(writes), std::move(localDevice), submission.queue, (packet[0] >> 8u) & 0xffu);
            return DrawVerdict::Drawn;
        }
        if (copy) {
            const bool checking = LeanFallbacks::VerifyDue();
            if (checking) leanFallbackProbe = {true, false, LeanFallbacks::Route::DepthCopy};
            else fallbackScope->Moved(LeanFallbacks::Route::DepthCopy);
            const auto todays = LeanFront::Total(Counter::DrawsOnTodaysPath);
            const auto result = drawLeanDepthCopy(queue, *copy, packet, submission, packetOffset, rejected);
            if (!checking && LeanFront::Total(Counter::DrawsOnTodaysPath) != todays) LeanFallbacks::AddLater(LeanFallbacks::Route::DepthCopy);
            return result;
        }
        if (!control) {
            LeanFallbacks::AddKept(LeanFallbacks::Kept::HeadChanged);
            return today(todayCounter, reason);
        }
        if (control->samples != 1u) {
            LeanFallbacks::AddKept(LeanFallbacks::Kept::MultisampleControl);
            return today(todayCounter, LeanFallbacks::Reason::RenderControl);
        }
        verdict = leanVerdict(queue, record, parameters, true);
        if (verdict == LeanVerdict::Fallback) {
            LeanFallbacks::AddKept(LeanFallbacks::Kept::AfterControlFallback);
            return today(todayCounter, LeanFallbacks::Reason::RenderControl);
        }
        afterControl = true;
        if (LeanFallbacks::VerifyDue()) {
            leanFallbackProbe = {true, false, LeanFallbacks::Route::RenderControl};
        } else {
            fallbackScope->Moved(LeanFallbacks::Route::RenderControl);
            movedControl = true;
        }
    }
    if (verdict == LeanVerdict::Nothing) {
        const bool forced = std::exchange(leanFallbackProbe.force, false);
        if (LeanFront::Due(frontLatch, LeanFront::Item::Draw) || forced) return verify(record, verdict, nullptr, forced);
        LeanFront::AddOnWorker(Counter::LeanDraws);
        return DrawVerdict::Nothing;
    }
    if (!buildLeanShape(queue, submission, record)) return today(todayCounter, LeanFallbacks::Reason::RefusedShape);
    const auto& shape = *record.built;
    if (parameters.indirect) {
        const auto& decode = *shape.decode;
        const auto& graphics = decode.state;
        if (decode.programs.empty() || decode.roles.size() != decode.programs.size() || !KnownGpuIndirectProgram(decode.programs.front().binary.codeAddress) || graphics.quadList || graphics.rectList || graphics.stages.path != Graphics::ShaderPath::Vertex || graphics.stages.mesh || graphics.stages.tessellation || graphics.stages.tessellationGeometry || graphics.tessellationCapture != nullptr) return today(Counter::IndirectDrawsOnTodaysPath, LeanFallbacks::Reason::GpuIndirect);
        auto& indirect = *parameters.indirect;
        const auto sgprOf = [&](std::uint32_t location) -> std::int32_t {
            const auto words = UserDataWords(decode.programs, decode.roles, location);
            if (words.empty() || words.front().first != 0) return -1;
            return static_cast<std::int32_t>(decode.programs.front().firstUserSgpr + words.front().second);
        };
        indirect.baseVertexSgpr = sgprOf(indirect.baseVertexLocation);
        indirect.startInstanceSgpr = sgprOf(indirect.startInstanceLocation);
        indirect.drawIndexSgpr = sgprOf(indirect.drawIndexLocation);
    }
    const bool forcedCheck = std::exchange(leanFallbackProbe.force, false);
    if (LeanFront::Due(frontLatch, LeanFront::Item::Draw) || forcedCheck) {
        LeanDrawJob job{};
        job.shape = &shape;
        job.serial = submission.serial;
        job.queue = submission.queue;
        job.packetOffset = static_cast<std::uint64_t>(packet.data() - submission.commands.data());
        job.opcode = (packet[0] >> 8u) & 0xffu;
        job.parameters = parameters;
        captureUserWords(job.words, queue.shader, LeanFront::SwitchSet(frontLatch, Graphics::RuntimeSwitch::UserWordBlocks));
        job.drawKey = leanDrawKey(shape, job.words, unorderedDrawKeys.load(std::memory_order_relaxed));
        job.shapeKey = shape.finalShape;
        job.deviceSerial = record.deviceSerial;
        job.records = nullptr;
        job.plans = nullptr;
        job.recordGeneration = 0;
        job.recordFlags = 0;
        recordsOff(DrawRecords::Counter::OffLeanVerify);
        return verify(record, verdict, &job, forcedCheck);
    }
    if (verdict == LeanVerdict::Live) {
        LeanFront::AddOnWorker(Counter::PrecheckedLeanDraws);
        bool traceIndirect = false;
        const auto prechecked = [&] {
            const WorkerClock clock(std::nullopt);
            return precheckDraw(queue, submission, packetOffset, packet, parameters, rejected, traceIndirect);
        }();
        if (prechecked) {
            if (*prechecked == DrawVerdict::Drawn) recordsOff(DrawRecords::Counter::OffTodaysDraw);
            return *prechecked;
        }
    }
    if (!parameters.indirect && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpHalfWorkerDraws) && HalfWorkerDrawDropped()) return DrawVerdict::Drawn;
    if (shape.skipped) return DrawVerdict::Drawn;
    const bool runs = direct && LeanFront::SwitchSet(frontLatch, Graphics::RuntimeSwitch::DrawRuns);
    std::byte* payload = nullptr;
    std::optional<std::uint64_t> anchor;
    if (leanRun.count != 0) {
        if (runs) anchor = leanDraws.AnchorFor(submission.shaders, leanDraws.device, finisher.Issued());
        if (runs && anchor && *anchor == leanRun.anchor) payload = leanRun.payload;
        else publishLeanRun();
    }
    if (payload == nullptr) {
        payload = finisher.ReserveLean();
        anchor = payload != nullptr ? leanDraws.AnchorFor(submission.shaders, leanDraws.device, finisher.Issued()) : std::nullopt;
    }
    if (parameters.indirect && (payload == nullptr || !anchor)) return today(Counter::IndirectDrawsOnTodaysPath, LeanFallbacks::Reason::GpuIndirect);
    std::optional<LeanDrawJob> local;
    LeanDrawJob* job = nullptr;
    if (payload == nullptr || !anchor) {
        job = &local.emplace();
    } else if (runs) {
        if (leanRun.count == 0) openLeanRun(payload, *anchor, finisher.Issued());
        job = ::new (static_cast<void*>(leanRun.jobs + leanRun.count)) LeanDrawJob;
    } else {
        job = ::new (payload) LeanDrawJob;
    }
    job->shape = &shape;
    job->anchor = anchor.value_or(0);
    job->serial = submission.serial;
    job->queue = submission.queue;
    job->epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
    job->packetOffset = static_cast<std::uint64_t>(packet.data() - submission.commands.data());
    job->opcode = (packet[0] >> 8u) & 0xffu;
    job->parameters = parameters;
    captureUserWords(job->words, queue.shader, LeanFront::SwitchSet(frontLatch, Graphics::RuntimeSwitch::UserWordBlocks));
    job->drawKey = leanDrawKey(shape, job->words, unorderedDrawKeys.load(std::memory_order_relaxed));
    job->shapeKey = shape.finalShape;
    job->deviceSerial = record.deviceSerial;
    job->records = nullptr;
    job->plans = nullptr;
    job->recordGeneration = 0;
    job->recordFlags = 0;
    LeanFront::AddOnWorker(Counter::LeanDraws);
    if (local.has_value()) {
        recordsOff(DrawRecords::Counter::OffInline);
        finishLeanDraw(*local, LeanAnchor{submission.shaders, leanDraws.device, 0});
        return DrawVerdict::Drawn;
    }
    if (direct) {
        job->recordFlags = DrawRecordFlags::Direct;
        if (frontLatch.directBuffers) job->recordFlags |= DrawRecordFlags::DirectBuffers;
        if (frontLatch.directInputs) job->recordFlags |= DrawRecordFlags::DirectInputs;
        if (DirectDraws::Due(frontLatch)) job->recordFlags |= DrawRecordFlags::VerifyDue;
    } else if (plans) {
        probeStagePlans(*job, record);
    } else if (records) {
        probeDrawRecord(*job, shape);
    }
    record.lastSeq = finisher.Issued();
    if (runs) {
        if (++leanRun.count == 1) {
            leanRun.key = job->drawKey;
            leanRun.shape = job->shapeKey;
        }
        for (const auto& range : shape.writes) {
            if (std::find(leanRun.writes.begin(), leanRun.writes.end(), range) != leanRun.writes.end()) continue;
            leanRun.writes.push_back(range);
            finisher.HoldOpenRunWrite(range);
        }
        if (leanRun.count >= leanRunLimit()) publishLeanRun();
        return DrawVerdict::Drawn;
    }
    const bool answered = !direct && (job->recordFlags & (DrawRecordFlags::BuildWanted | DrawRecordFlags::VerifyDue)) != 0;
    finisher.PublishLean(&Driver::runLeanDraw, shape.writes, job->drawKey, job->shapeKey, !answered ? nullptr : plans ? &Driver::retireStagePlans : &Driver::retireDrawRecord);
    return DrawVerdict::Drawn;
}

void Driver::openLeanRun(std::byte* payload, std::uint64_t anchor, std::uint64_t seq) {
    if (leanRunJobs.empty()) leanRunJobs.resize(DrawFinisher::LeanSlots * static_cast<std::size_t>(LeanRunCap));
    leanRun.payload = payload;
    leanRun.jobs = leanRunJobs.data() + static_cast<std::size_t>(seq % DrawFinisher::LeanSlots) * LeanRunCap;
    leanRun.count = 0;
    leanRun.anchor = anchor;
    leanRun.writes.clear();
    DrawFinisher::Get().SetOpenRun(&Driver::flushLeanRun);
}

void Driver::publishLeanRun() {
    auto& finisher = DrawFinisher::Get();
    finisher.SetOpenRun(nullptr);
    const auto count = std::exchange(leanRun.count, 0u);
    if (count == 0) return;
    auto* const header = ::new (static_cast<void*>(leanRun.payload)) LeanRunHeader;
    header->count = count;
    header->jobs = leanRun.jobs;
    const bool single = count == 1;
    finisher.PublishLean(&Driver::runLeanRun, leanRun.writes, single ? leanRun.key : PublishedKeys::AnyKey, single ? leanRun.shape : 0, nullptr, true);
}

void Driver::flushLeanRun() {
    Get().publishLeanRun();
}

DrawVerdict Driver::verifyLeanDraw(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::size_t packetOffset, std::string& rejected, ShapeRecord& record, LeanVerdict verdict, const Pm4::DrawParameters& parameters, const LeanDrawJob* job, bool afterControl, bool* differed) {
    using LeanFront::Field;
    LeanDrawSink sink;
    struct SinkScope {
        LeanDraws& lean;
        ~SinkScope() { lean.sink = nullptr; }
    } scope{leanDraws};
    leanDraws.sink = &sink;
    const auto result = draw(queue, packet, submission, packetOffset, rejected);
    leanDraws.sink = nullptr;
    LeanFront::NoteChecked(LeanFront::Item::Draw);
    const auto at = where(submission, packetOffset);
    const auto variant = verdictIndex(parameters);
    bool demote = false;
    const auto demoteVerdict = [&] {
        auto& verdicts = afterControl ? record.afterControl : record.verdicts;
        if (variant < verdicts.size()) verdicts[variant] = LeanVerdict::Live;
        if (differed != nullptr) *differed = true;
    };
    if (verdict == LeanVerdict::Normal && (!sink.prechecked || !sink.passed)) {
        LeanFront::NoteDifference(Field::DrawVerdicts, at + ": the shape record says the precheck passes, today's precheck " + (sink.prechecked ? "returned verdict " + std::to_string(static_cast<int>(sink.verdict)) : std::string("did not run")));
        demoteVerdict();
    } else if (verdict == LeanVerdict::Nothing && (!sink.prechecked || sink.passed || sink.verdict != DrawVerdict::Nothing)) {
        LeanFront::NoteDifference(Field::DrawVerdicts, at + ": the shape record says the draw draws nothing, today's precheck " + (!sink.prechecked ? std::string("did not run") : sink.passed ? std::string("passed") : "returned verdict " + std::to_string(static_cast<int>(sink.verdict))));
        demoteVerdict();
    }
    if (job == nullptr || !sink.keyed) return result;
    const auto& shape = *job->shape;
    if (job->drawKey != sink.key || job->shapeKey != sink.shape) {
        LeanFront::NoteDifference(Field::DrawKeys, at + ": lean key " + hex(job->drawKey) + " shape " + hex(job->shapeKey) + ", today's key " + hex(sink.key) + " shape " + hex(sink.shape));
        demote = true;
    }
    if (drawRelocationEnabled()) {
        const auto identity = LeanIdentityKey(shape, job->words);
        if (identity != sink.identity) {
            LeanFront::NoteDifference(Field::DrawIdentity, at + ": lean identity " + hex(identity) + ", today's " + hex(sink.identity));
            demote = true;
        }
    }
    if (sink.aliasAsked) {
        std::uint64_t alias = 0;
        LeanAliasKey(shape, job->words, sink.relocation.get(), job->deviceSerial, alias);
        if (alias != sink.alias) {
            LeanFront::NoteDifference(Field::AliasKeys, at + ": lean alias " + hex(alias) + ", today's " + hex(sink.alias));
            demote = true;
        } else if (alias != 0) {
            std::shared_ptr<DrawEntry> todayEntry, leanEntry;
            std::shared_ptr<const DrawDecode> todayDonor, leanDonor;
            std::vector<DrawUserPatch> todayPatches, leanPatches;
            {
                std::lock_guard cacheLock(drawCacheMutex);
                relocateDrawEntry(queue, alias, todayEntry, todayDonor, todayPatches);
                relocateDrawEntry(job->words, alias, leanEntry, leanDonor, leanPatches);
            }
            const bool samePatches = std::equal(todayPatches.begin(), todayPatches.end(), leanPatches.begin(), leanPatches.end(), [](const DrawUserPatch& a, const DrawUserPatch& b) { return a.program == b.program && a.index == b.index && a.value == b.value; });
            if (todayDonor != leanDonor || !samePatches) {
                LeanFront::NoteDifference(Field::Patches, at + ": relocating alias " + hex(alias) + " with the job's user words gives " + std::to_string(leanPatches.size()) + " patches" + (todayDonor != leanDonor ? " from another donor" : "") + ", with the queue's registers " + std::to_string(todayPatches.size()));
                demote = true;
            }
        }
    }
    for (std::uint32_t offset = 0; offset < 0x200u; ++offset) {
        if (DrawUserWords::Index(offset) < 0 && DrawUserWords::ResourceIndex(offset) < 0) continue;
        const bool held = queue.shader.contains(offset);
        const auto value = readUserData(queue.shader, offset);
        if ((DrawUserWords::Index(offset) >= 0 && job->words.Contains(offset) != held) || job->words.Value(offset) != value) {
            LeanFront::NoteDifference(Field::UserWords, at + ": user word register " + hex(offset) + " captured as " + hex(job->words.Value(offset)) + (job->words.Contains(offset) ? "" : " (absent)") + ", the queue holds " + hex(value) + (held ? "" : " (absent)"));
            demote = true;
            break;
        }
    }
    try {
        const auto lean = LeanRefreshedDecode(*shape.decode, job->words);
        const auto fresh = decodeDraw(queue, submission, leanDraws.device->graphicsContext());
        if (!sameDecode(*lean, *fresh)) {
            LeanFront::NoteDifference(Field::Decodes, at + ": the shape record's decode with the draw's user words differs from a fresh decode (shape " + hex(job->shapeKey) + ")");
            demote = true;
        }
    } catch (const std::exception& error) {
        LeanFront::NoteDifference(Field::Decodes, at + ": decoding the draw for the check failed: " + error.what());
        demote = true;
    }
    if (sink.handed) {
        if (sink.held.unknown) {
            LeanFront::NoteAllowed(LeanFront::Allowed::HeldTargetsOnly);
        } else if (sink.held.writes != shape.writes) {
            LeanFront::NoteDifference(Field::WriteTemplates, at + ": the shape's write template holds " + std::to_string(shape.writes.size()) + " ranges, today's held writes " + std::to_string(sink.held.writes.size()));
            demote = true;
        }
        const auto& today = sink.parameters;
        const auto& mine = job->parameters;
        const auto sameIndirect = [](const Pm4::DrawParameters::IndirectDraw& a, const Pm4::DrawParameters::IndirectDraw& b) { return a.arguments == b.arguments && a.opcode == b.opcode && a.recordBytes == b.recordBytes && a.stride == b.stride && a.count == b.count && a.countIndirect == b.countIndirect && a.countAddress == b.countAddress && a.baseVertexLocation == b.baseVertexLocation && a.startInstanceLocation == b.startInstanceLocation && a.drawIndexLocation == b.drawIndexLocation && a.drawIndexEnabled == b.drawIndexEnabled && a.indxOffset == b.indxOffset && a.indexLimitKnown == b.indexLimitKnown && a.baseVertexSgpr == b.baseVertexSgpr && a.startInstanceSgpr == b.startInstanceSgpr && a.drawIndexSgpr == b.drawIndexSgpr; };
        const bool sameParameters = today.indexAddress == mine.indexAddress && today.indexCount == mine.indexCount && today.indexSize == mine.indexSize && today.instanceCount == mine.instanceCount && today.flags == mine.flags && today.indexed == mine.indexed && today.firstVertex == mine.firstVertex && today.firstInstance == mine.firstInstance && today.indirect.has_value() == mine.indirect.has_value() && (!mine.indirect || sameIndirect(*today.indirect, *mine.indirect));
        if (!sameParameters || sink.held.key != job->drawKey || sink.held.shape != job->shapeKey || sink.reg207 != shape.reg207) {
            LeanFront::NoteDifference(Field::Jobs, at + ": the lean job's " + (!sameParameters ? "draw parameters" : sink.reg207 != shape.reg207 ? "PA_CL_VS_OUT_CNTL" : "held key") + " differ from today's hand-off");
            demote = true;
        }
        if (sink.device != leanDraws.device.get()) {
            LeanFront::NoteDifference(Field::Jobs, at + ": the lean job's device differs from the device today's draw used");
            if (differed != nullptr) *differed = true;
        }
        if (sink.dataEntry != nullptr && sink.decode != nullptr) {
            try {
                if (!sameDecode(*LeanRefreshedDecode(*sink.dataEntry->decode, job->words), *sink.decode)) {
                    LeanFront::NoteDifference(Field::Decodes, at + ": the data entry's decode refreshed with the job's user words differs from today's refresh (shape " + hex(job->shapeKey) + ")");
                    demote = true;
                }
            } catch (const std::exception& error) {
                LeanFront::NoteDifference(Field::Decodes, at + ": refreshing the data entry's decode with the job's user words failed: " + error.what());
                demote = true;
            }
        }
    }
    if (demote) {
        if (differed != nullptr) *differed = true;
        leanDraws.Retire(record, DrawFinisher::Get().Front());
        record.refused = true;
    }
    return result;
}

}
