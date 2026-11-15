#include "prx/libSceAgcDriver/Graphics/include/WriteValidation.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuardSpans.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <memory>
#include <mutex>
#include <vector>

namespace AgcDriver::Graphics::WriteValidation {

bool FinisherSide() {
    static const bool on = std::getenv("APS5_WRITE_VALIDATION_FINISHER") != nullptr;
    return on;
}

namespace Detail {

std::atomic<std::uint64_t> session{Unlatched};
std::atomic<std::uint64_t> epoch{0};
std::atomic<ClassClocks*> clocks{nullptr};

}

namespace {

constexpr std::array<const char*, TestCount> TestNames{"follower inputs", "follower zones", "follower stores", "follower pass", "follower selection", "follower guards", "leader inputs", "leader zones", "leader in-place", "leader guards", "leader moves", "leader stores", "leader selection", "record moves", "record in-place", "template current", "view settle", "stage settle", "walk settle", "buffers settle", "index settle", "captured memory", "nested values", "stage inputs", "pending snapshot", "label overlap"};
constexpr std::array<const char*, OutcomeCount> OutcomeNames{"answered", "armed", "live", "session", "wide", "evicted", "moved", "snapshot", "watch", "unknown", "stamps", "not clean", "kept"};
constexpr std::array<const char*, SourceCount> SourceNames{"recorder note", "in-flight note", "storage pending", "shadow live", "queue item", "finisher job", "run write", "pending job", "deposit", "recorded label", "wide label", "queued label", "behind completion", "queued store", "queued dump", "key store", "CPU publish", "clean forget"};

constexpr std::uint64_t MaxWindows = 16;
constexpr std::uint64_t SpreadLimit = 64;
constexpr std::uint8_t LiveSkip = 64;
constexpr std::size_t CommitterSlots = 4096;

struct Latching {
    std::mutex mutex;
    std::uint64_t sessions = 0;
    bool hooked = false;
};

Latching& latching() {
    static auto* kept = new Latching;
    return *kept;
}

bool wanted() {
    return RuntimeSwitches::Set(RuntimeSwitch::WriteValidation) && !RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter) && !RuntimeSwitches::Set(RuntimeSwitch::XpTrustFinisher) && !RuntimeSwitches::Set(RuntimeSwitch::TrustAudit);
}

ClassClocks* allocateClocks() {
    auto* all = new ClassClocks[ClassCount];
    for (std::size_t kind = 0; kind < ClassCount; ++kind) {
        for (auto& slot : all[kind].blocks) slot.store(0, std::memory_order_relaxed);
        for (auto& slot : all[kind].supers) slot.store(0, std::memory_order_relaxed);
        all[kind].floor.store(0, std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_release);
    return all;
}

void onAssign(RuntimeSwitch id) {
    if (id != RuntimeSwitch::WriteValidation && id != RuntimeSwitch::XpTrustCommitter && id != RuntimeSwitch::XpTrustFinisher && id != RuntimeSwitch::TrustAudit) return;
    Relatch();
}

void relatchLocked(Latching& state) {
    if (!state.hooked) {
        RuntimeSwitches::SetAssignHook(&onAssign);
        state.hooked = true;
    }
    const auto current = Detail::session.load(std::memory_order_seq_cst);
    if (!wanted()) {
        if (current != 0) Detail::session.store(0, std::memory_order_seq_cst);
        return;
    }
    if (current != 0 && current != Unlatched) return;
    if (Detail::clocks.load(std::memory_order_acquire) == nullptr) Detail::clocks.store(allocateClocks(), std::memory_order_seq_cst);
    Detail::session.store(++state.sessions, std::memory_order_seq_cst);
}

void raiseSlot(std::atomic<std::uint64_t>& slot, std::uint64_t stamp) {
    auto value = slot.load(std::memory_order_seq_cst);
    while (value < stamp && !slot.compare_exchange_weak(value, stamp, std::memory_order_seq_cst, std::memory_order_seq_cst)) {
    }
}

struct Counts {
    std::array<std::array<std::atomic<std::uint64_t>, OutcomeCount>, TestCount> outcomes{};
    std::array<std::atomic<std::uint64_t>, SourceCount> raises{};
    std::array<std::atomic<std::uint64_t>, SpanKinds> spans{};
    std::array<std::uint64_t, TestCount> verifyOrdinal{};
    std::uint32_t notes = 0;
};

struct CountsRegistry {
    std::mutex mutex;
    std::vector<std::unique_ptr<Counts>> threads;
};

CountsRegistry& countsRegistry() {
    static auto* kept = new CountsRegistry;
    return *kept;
}

struct CountsHandle {
    Counts* counts = nullptr;
};

struct CountsStorage {};

Counts& threadCounts() {
    auto& handle = HostThreadLocal<CountsHandle, CountsStorage>();
    if (handle.counts != nullptr) return *handle.counts;
    auto created = std::make_unique<Counts>();
    auto* counts = created.get();
    {
        auto& registry = countsRegistry();
        std::lock_guard lock(registry.mutex);
        registry.threads.push_back(std::move(created));
    }
    handle.counts = counts;
    return *counts;
}

void bump(std::atomic<std::uint64_t>& counter) {
    counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

std::array<std::atomic<std::uint64_t>, SourceCount> sharedRaises{};
std::array<std::atomic<std::uint64_t>, SpanKinds> sharedSpans{};

void countRaise(Source source, std::size_t span) {
    if (source == Source::CleanForget) {
        sharedRaises[static_cast<std::size_t>(source)].fetch_add(1, std::memory_order_relaxed);
        sharedSpans[span].fetch_add(1, std::memory_order_relaxed);
        return;
    }
    auto& counts = threadCounts();
    bump(counts.raises[static_cast<std::size_t>(source)]);
    bump(counts.spans[span]);
}

std::array<std::atomic<std::uint64_t>, TestCount> verifiedCounts{};
std::array<std::atomic<std::uint64_t>, TestCount> differingCounts{};
std::atomic<std::uint64_t> differingAll{0};

struct CommitterSlot {
    std::uint64_t window = ~std::uint64_t{0};
    std::uint64_t armedW = 0;
    std::uint64_t armedL = 0;
    std::uint64_t depthGeneration = 0;
    std::uint64_t multisampleGeneration = 0;
    std::uint32_t epoch = 0;
    std::uint16_t clear = 0;
    std::uint16_t live = 0;
    std::uint8_t liveSkip = 0;
    bool depthAny = false;
};

struct CommitterTable {
    std::uint64_t session = 0;
    std::uint64_t recorder = 0;
    std::uint64_t switches = 0;
    std::uint32_t epoch = 0;
    std::array<CommitterSlot, CommitterSlots> slots{};
};

struct CommitterHandle {
    std::unique_ptr<CommitterTable> table;
};

struct CommitterStorage {};

CommitterTable& committerTable() {
    auto& handle = HostThreadLocal<CommitterHandle, CommitterStorage>();
    if (handle.table == nullptr) handle.table = std::make_unique<CommitterTable>();
    return *handle.table;
}

std::size_t committerSlot(std::uint64_t window) {
    return static_cast<std::size_t>((window * 0x9E3779B97F4A7C15ull) >> 52) & (CommitterSlots - 1u);
}

std::uint16_t without(std::uint16_t facts, std::uint16_t dropped) {
    return static_cast<std::uint16_t>(facts & static_cast<std::uint16_t>(~dropped));
}

bool committerStillClear(const CommitterTable& table, std::uint64_t first, std::uint64_t last, std::uint16_t facts) {
    const bool guards = (facts & Fact::OfG) != 0;
    const auto depthGeneration = guards ? DepthGuardsGeneration_nid_no_patch() : 0u;
    const auto multisampleGeneration = guards ? MultisampleGuardsGeneration() : 0u;
    if ((facts & Fact::DepthReads) != 0 && DepthOverwriteProofsPending_nid_no_patch() != 0u) return false;
    const bool depthAny = (facts & Fact::DepthReads) != 0 && HasResidentDepthGuards_nid_no_patch();
    for (auto window = first;; ++window) {
        const auto& slot = table.slots[committerSlot(window)];
        if (slot.window != window || slot.epoch != table.epoch || (slot.clear & facts) != facts) return false;
        if ((facts & Fact::OfW) != 0 && Clock(Class::W, window) > slot.armedW) return false;
        if ((facts & Fact::OfL) != 0 && Clock(Class::L, window) > slot.armedL) return false;
        if (guards && (slot.depthGeneration != depthGeneration || slot.multisampleGeneration != multisampleGeneration)) return false;
        if ((facts & Fact::DepthReads) != 0 && !slot.depthAny && depthAny) return false;
        if (window == last) break;
    }
    return true;
}

bool proveFact(const Recorder& recorder, std::uint16_t fact, std::uint64_t address, bool& depthAny) {
    constexpr std::size_t bytes = std::size_t{1} << WindowShift;
    switch (fact) {
    case Fact::Pending: return !recorder.PendingWriteOverlaps(address, bytes) && !PendingStorageOverlaps(address, bytes, nullptr);
    case Fact::Writes: return !recorder.PendingWriteOverlaps(address, bytes);
    case Fact::Storage: return !PendingStorageOverlaps(address, bytes, nullptr);
    case Fact::Shadows: return !AnyShadowedOverlaps(address, bytes);
    case Fact::Labels: return !recorder.PendingLabelIn(address, bytes);
    case Fact::Stores: return !recorder.QueuedStoreOverlaps(address, bytes) && !recorder.QueuedKeyStoreOverlaps(address, bytes);
    case Fact::InPlaceExtra: return !CpuPublishPendingOverlaps(address, bytes) && !recorder.CompletionLabelIn(address, bytes);
    case Fact::DepthSpans: return !MayOverlapDepthGuards_nid_no_patch(address, bytes);
    case Fact::MultisampleSpans: return !MayOverlapMultisampleGuards(address, bytes);
    case Fact::DepthReads: {
        if (DepthOverwriteProofsPending_nid_no_patch() != 0u) return false;
        depthAny = HasResidentDepthGuards_nid_no_patch();
        return !depthAny || !MayOverlapDepthGuards_nid_no_patch(address, bytes);
    }
    default: return false;
    }
}

}

namespace Detail {

void Raise(Class kind, Source source, std::uint64_t begin, std::uint64_t end) {
    auto* all = clocks.load(std::memory_order_acquire);
    if (all == nullptr) return;
    auto& target = all[static_cast<std::size_t>(kind)];
    const auto stamp = epoch.fetch_add(1, std::memory_order_seq_cst) + 1;
    const auto first = begin >> WindowShift;
    const auto last = (end - 1u) >> WindowShift;
    if (last - first < SpreadLimit) {
        for (auto block = first;; ++block) {
            raiseSlot(target.blocks[BlockSlot(block)], stamp);
            if (block == last) break;
        }
        countRaise(source, 0);
    } else if (const auto firstSuper = first >> SuperShift, lastSuper = last >> SuperShift; lastSuper - firstSuper < SpreadLimit) {
        for (auto super = firstSuper;; ++super) {
            raiseSlot(target.supers[SuperSlot(super)], stamp);
            if (super == lastSuper) break;
        }
        countRaise(source, 1);
    } else {
        raiseSlot(target.floor, stamp);
        countRaise(source, 2);
    }
}

void RaiseAll(Class kind, Source source) {
    auto* all = clocks.load(std::memory_order_acquire);
    if (all == nullptr) return;
    const auto stamp = epoch.fetch_add(1, std::memory_order_seq_cst) + 1;
    raiseSlot(all[static_cast<std::size_t>(kind)].floor, stamp);
    countRaise(source, 2);
}

}

std::uint64_t Latch() {
    auto& state = latching();
    std::lock_guard lock(state.mutex);
    if (Detail::session.load(std::memory_order_seq_cst) == Unlatched) relatchLocked(state);
    return Detail::session.load(std::memory_order_seq_cst);
}

void Relatch() {
    auto& state = latching();
    std::lock_guard lock(state.mutex);
    relatchLocked(state);
}

void Note(Test test, Outcome outcome) {
    auto& counts = threadCounts();
    bump(counts.outcomes[static_cast<std::size_t>(test)][static_cast<std::size_t>(outcome)]);
    if ((++counts.notes & 255u) == 0) Report();
}

std::uint64_t VerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_WRITE_VALIDATION");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

bool VerifyDue(Test test) {
    const auto every = VerifyEvery();
    if (every == 0) return false;
    auto& counts = threadCounts();
    return ++counts.verifyOrdinal[static_cast<std::size_t>(test)] % every == 0;
}

void NoteVerify(Test test, bool differs, const char* detail) {
    const auto index = static_cast<std::size_t>(test);
    verifiedCounts[index].fetch_add(1, std::memory_order_relaxed);
    if (!differs) return;
    differingCounts[index].fetch_add(1, std::memory_order_relaxed);
    const auto seen = differingAll.fetch_add(1, std::memory_order_relaxed);
    if (seen < 16u || seen % 4096u == 0) std::fprintf(stderr, "[write-validation-verify] %s differs: %s\n", TestName(test), detail != nullptr ? detail : "today's test refused");
}

Totals Collect() {
    Totals totals;
    {
        auto& registry = countsRegistry();
        std::lock_guard lock(registry.mutex);
        for (const auto& counts : registry.threads) {
            for (std::size_t test = 0; test < TestCount; ++test) {
                for (std::size_t outcome = 0; outcome < OutcomeCount; ++outcome) totals.outcomes[test][outcome] += counts->outcomes[test][outcome].load(std::memory_order_relaxed);
            }
            for (std::size_t source = 0; source < SourceCount; ++source) totals.raises[source] += counts->raises[source].load(std::memory_order_relaxed);
            for (std::size_t kind = 0; kind < SpanKinds; ++kind) totals.spans[kind] += counts->spans[kind].load(std::memory_order_relaxed);
        }
    }
    for (std::size_t source = 0; source < SourceCount; ++source) totals.raises[source] += sharedRaises[source].load(std::memory_order_relaxed);
    for (std::size_t kind = 0; kind < SpanKinds; ++kind) totals.spans[kind] += sharedSpans[kind].load(std::memory_order_relaxed);
    for (std::size_t test = 0; test < TestCount; ++test) {
        totals.verified[test] = verifiedCounts[test].load(std::memory_order_relaxed);
        totals.differing[test] = differingCounts[test].load(std::memory_order_relaxed);
    }
    return totals;
}

std::uint64_t CountForTest(Test test, Outcome outcome) {
    return Collect().outcomes[static_cast<std::size_t>(test)][static_cast<std::size_t>(outcome)];
}

const char* TestName(Test test) {
    const auto index = static_cast<std::size_t>(test);
    return index < TestNames.size() ? TestNames[index] : "unnamed";
}

const char* OutcomeName(Outcome outcome) {
    const auto index = static_cast<std::size_t>(outcome);
    return index < OutcomeNames.size() ? OutcomeNames[index] : "unnamed";
}

const char* SourceName(Source source) {
    const auto index = static_cast<std::size_t>(source);
    return index < SourceNames.size() ? SourceNames[index] : "unnamed";
}

namespace {

struct Reported {
    std::mutex mutex;
    Totals totals;
    std::uint64_t presents = 0;
};

Reported& reported() {
    static auto* kept = new Reported;
    return *kept;
}

std::atomic<std::int64_t> reportedWindow{-1};

unsigned long long wide(std::uint64_t value) {
    return static_cast<unsigned long long>(value);
}

void appendCount(std::string& text, const char* name, std::uint64_t count) {
    char item[96];
    std::snprintf(item, sizeof(item), "%s%s %llu", text.empty() ? "" : ", ", name, wide(count));
    text += item;
}

}

void Report() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!stats) return;
    const std::int64_t window = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (reportedWindow.exchange(window, std::memory_order_relaxed) == window) return;
    auto& state = reported();
    std::unique_lock lock(state.mutex, std::try_to_lock);
    if (!lock.owns_lock()) return;
    const auto now = Collect();
    const auto presentsNow = Recorder::Presents();
    const auto presents = presentsNow - state.presents;
    state.presents = presentsNow;
    const auto delta = [](std::uint64_t current, std::uint64_t before) { return current >= before ? current - before : current; };
    std::array<std::array<std::uint64_t, OutcomeCount>, TestCount> outcomes{};
    std::array<std::uint64_t, TestCount> verified{};
    std::array<std::uint64_t, TestCount> differing{};
    std::uint64_t answered = 0;
    std::uint64_t armed = 0;
    std::uint64_t today = 0;
    std::uint64_t kept = 0;
    std::uint64_t checked = 0;
    std::uint64_t differs = 0;
    const auto counted = [](std::size_t outcome) {
        const auto kind = static_cast<Outcome>(outcome);
        return kind != Outcome::Answered && kind != Outcome::Armed && kind != Outcome::Kept && kind != Outcome::Moved && kind != Outcome::Evicted;
    };
    for (std::size_t test = 0; test < TestCount; ++test) {
        for (std::size_t outcome = 0; outcome < OutcomeCount; ++outcome) {
            outcomes[test][outcome] = delta(now.outcomes[test][outcome], state.totals.outcomes[test][outcome]);
            if (counted(outcome)) today += outcomes[test][outcome];
        }
        answered += outcomes[test][static_cast<std::size_t>(Outcome::Answered)];
        armed += outcomes[test][static_cast<std::size_t>(Outcome::Armed)];
        kept += outcomes[test][static_cast<std::size_t>(Outcome::Kept)];
        verified[test] = delta(now.verified[test], state.totals.verified[test]);
        differing[test] = delta(now.differing[test], state.totals.differing[test]);
        checked += verified[test];
        differs += differing[test];
    }
    std::array<std::uint64_t, SourceCount> raises{};
    for (std::size_t source = 0; source < SourceCount; ++source) raises[source] = delta(now.raises[source], state.totals.raises[source]);
    std::array<std::uint64_t, SpanKinds> spans{};
    for (std::size_t kind = 0; kind < SpanKinds; ++kind) spans[kind] = delta(now.spans[kind], state.totals.spans[kind]);
    state.totals = now;
    lock.unlock();
    AgcDriver::ProfilePrint_nid_no_patch("[write-validation] 10 s, %llu presents, session %llu: answered by generations %llu, proved %llu, today's %llu, kept %llu; verify checked %llu, differing %llu (since start %llu)\n", wide(presents), wide(Session()), wide(answered), wide(armed), wide(today), wide(kept), wide(checked), wide(differs), wide(differingAll.load(std::memory_order_relaxed)));
    for (std::size_t test = 0; test < TestCount; ++test) {
        std::uint64_t any = verified[test];
        for (const auto count : outcomes[test]) any += count;
        if (any == 0) continue;
        std::string todays;
        for (std::size_t outcome = 0; outcome < OutcomeCount; ++outcome) {
            if (!counted(outcome) || outcomes[test][outcome] == 0) continue;
            appendCount(todays, OutcomeName(static_cast<Outcome>(outcome)), outcomes[test][outcome]);
        }
        const auto& row = outcomes[test];
        AgcDriver::ProfilePrint_nid_no_patch("[write-validation] %s: answered %llu, armed %llu, today's (%s), kept %llu; moved %llu, evicted %llu; verify %llu checked, %llu differing\n", TestName(static_cast<Test>(test)), wide(row[static_cast<std::size_t>(Outcome::Answered)]), wide(row[static_cast<std::size_t>(Outcome::Armed)]), todays.empty() ? "none" : todays.c_str(), wide(row[static_cast<std::size_t>(Outcome::Kept)]), wide(row[static_cast<std::size_t>(Outcome::Moved)]), wide(row[static_cast<std::size_t>(Outcome::Evicted)]), wide(verified[test]), wide(differing[test]));
    }
    std::string sources;
    for (std::size_t source = 0; source < SourceCount; ++source) {
        if (raises[source] != 0) appendCount(sources, SourceName(static_cast<Source>(source)), raises[source]);
    }
    AgcDriver::ProfilePrint_nid_no_patch("[write-validation] raises by source: %s; by reach: windows %llu, supers %llu, floors %llu; epoch %llu\n", sources.empty() ? "none" : sources.c_str(), wide(spans[0]), wide(spans[1]), wide(spans[2]), wide(Epoch()));
}

Answer CommitterClear(const Recorder& recorder, std::uint64_t begin, std::uint64_t bytes, std::uint16_t facts, Test test) {
    const auto session = Session();
    if (session == 0 || bytes == 0 || facts == 0) return Answer::Today;
    constexpr auto top = std::numeric_limits<std::uint64_t>::max();
    if (begin > top - bytes) {
        Note(test, Outcome::Wide);
        return Answer::Today;
    }
    const auto first = begin >> WindowShift;
    const auto last = (begin + bytes - 1u) >> WindowShift;
    if (last - first >= MaxWindows || last >= (top >> WindowShift)) {
        Note(test, Outcome::Wide);
        return Answer::Today;
    }
    const bool guards = (facts & Fact::OfG) != 0;
    if (guards && (VerifyGuardSpans() || RuntimeSwitches::Set(RuntimeSwitch::NoGuardSpans))) {
        Note(test, Outcome::Kept);
        return Answer::Today;
    }
    if ((facts & Fact::DepthReads) != 0 && DepthOverwriteProofsPending_nid_no_patch() != 0u) {
        Note(test, Outcome::Live);
        return Answer::Today;
    }
    auto& table = committerTable();
    const auto switches = RuntimeSwitches::Generation();
    if (table.session != session || table.recorder != recorder.Id() || table.switches != switches) {
        table.session = session;
        table.recorder = recorder.Id();
        table.switches = switches;
        if (++table.epoch == 0) {
            table.slots.fill(CommitterSlot{});
            table.epoch = 1;
        }
    }
    const auto depthGeneration = guards ? DepthGuardsGeneration_nid_no_patch() : 0u;
    const auto multisampleGeneration = guards ? MultisampleGuardsGeneration() : 0u;
    const bool depthAnyNow = (facts & Fact::DepthReads) != 0 && HasResidentDepthGuards_nid_no_patch();
    auto answer = Answer::Generations;
    for (auto window = first;; ++window) {
        auto& slot = table.slots[committerSlot(window)];
        if (slot.window != window || slot.epoch != table.epoch) {
            if (slot.epoch == table.epoch && slot.clear != 0) Note(test, Outcome::Evicted);
            slot = CommitterSlot{};
            slot.window = window;
            slot.epoch = table.epoch;
        }
        if ((facts & Fact::OfW) != 0 && (slot.clear & Fact::OfW) != 0 && Clock(Class::W, window) > slot.armedW) {
            slot.clear = without(slot.clear, Fact::OfW);
            Note(test, Outcome::Moved);
        }
        if ((facts & Fact::OfL) != 0 && (slot.clear & Fact::OfL) != 0 && Clock(Class::L, window) > slot.armedL) {
            slot.clear = without(slot.clear, Fact::OfL);
            Note(test, Outcome::Moved);
        }
        if (guards && (slot.clear & Fact::OfG) != 0 && (slot.depthGeneration != depthGeneration || slot.multisampleGeneration != multisampleGeneration)) {
            slot.clear = without(slot.clear, Fact::OfG);
            Note(test, Outcome::Moved);
        }
        if ((facts & Fact::DepthReads) != 0 && (slot.clear & Fact::DepthReads) != 0 && !slot.depthAny && depthAnyNow) {
            slot.clear = without(slot.clear, Fact::DepthReads);
            Note(test, Outcome::Moved);
        }
        const auto need = without(facts, slot.clear);
        if (need != 0) {
            if ((slot.live & need) != 0) {
                if (slot.liveSkip != 0) {
                    --slot.liveSkip;
                    Note(test, Outcome::Live);
                    return Answer::Today;
                }
                slot.live = without(slot.live, need);
            }
            const auto before = Epoch();
            const bool provesGuards = (need & Fact::OfG) != 0;
            const auto depthBefore = provesGuards ? DepthGuardsGeneration_nid_no_patch() : 0u;
            const auto multisampleBefore = provesGuards ? MultisampleGuardsGeneration() : 0u;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            std::uint16_t proved = 0;
            std::uint16_t failed = 0;
            bool depthAny = true;
            const auto address = window << WindowShift;
            for (std::uint16_t remaining = need; remaining != 0;) {
                const auto fact = static_cast<std::uint16_t>(remaining & static_cast<std::uint16_t>(-static_cast<std::int32_t>(remaining)));
                remaining = without(remaining, fact);
                if (proveFact(recorder, fact, address, depthAny)) proved = static_cast<std::uint16_t>(proved | fact);
                else failed = static_cast<std::uint16_t>(failed | fact);
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if ((proved & Fact::OfW) != 0) slot.armedW = (slot.clear & Fact::OfW) != 0 ? std::min(slot.armedW, before) : before;
            if ((proved & Fact::OfL) != 0) slot.armedL = (slot.clear & Fact::OfL) != 0 ? std::min(slot.armedL, before) : before;
            if ((proved & Fact::OfG) != 0) {
                if ((slot.clear & Fact::OfG) != 0 && (slot.depthGeneration != depthBefore || slot.multisampleGeneration != multisampleBefore)) slot.clear = without(slot.clear, Fact::OfG);
                slot.depthGeneration = depthBefore;
                slot.multisampleGeneration = multisampleBefore;
            }
            if ((proved & Fact::DepthReads) != 0) slot.depthAny = depthAny;
            slot.clear = static_cast<std::uint16_t>(slot.clear | proved);
            if (failed != 0) {
                slot.live = static_cast<std::uint16_t>(slot.live | failed);
                slot.liveSkip = LiveSkip;
                Note(test, Outcome::Live);
                return Answer::Today;
            }
            answer = Answer::Proved;
        }
        if (window == last) break;
    }
    if (answer == Answer::Proved && !committerStillClear(table, first, last, facts)) {
        Note(test, Outcome::Live);
        return Answer::Today;
    }
    Note(test, answer == Answer::Generations ? Outcome::Answered : Outcome::Armed);
    return answer;
}

}
