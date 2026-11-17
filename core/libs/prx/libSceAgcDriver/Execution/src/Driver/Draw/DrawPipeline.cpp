#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectDrawReport.hpp"
#include "prx/libc/include/HostThread.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadPriority.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/WriteValidation.hpp"
#include "prx/libSceAgcDriver/Graphics/include/CommandEncoder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostAllocator.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileCounter.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/LockProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BindlessHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <array>
#include <limits>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::array<const char*, static_cast<std::size_t>(DrawPipeline::DrainReason::Count)> DrainNames{"packet", "flush", "labels", "capture", "indirect", "submission"};

constexpr std::array<const char*, static_cast<std::size_t>(DrawPipeline::CommitKind::Count)> KindNames{"other", "draw", "indirect", "colormeta", "dispatch", "sampledump", "dmafill", "dmawrite", "dmacopy", "store", "endofpipe", "seal", "submitend", "completion", "bufferfill", "flip", "suspend", "indirectdispatch"};

std::atomic<unsigned> runningCommitters{0};
std::atomic<unsigned> runningFinishers{0};
ProfileCounter drawKeys;
ProfileCounter drawKeysSameAsPrevious;
ProfileCounter drawKeysSeenRecently;
ProfileCounter drawShapesSameAsPrevious;
ProfileCounter drawShapesSeenRecently;

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

ProfileCounter heldReadRanges;
ProfileCounter heldReadQueries;
ProfileCounter heldReadHits;

template<typename TItem>
struct ClearOnExit {
    std::vector<TItem>& items;
    ~ClearOnExit() { items.clear(); }
};

struct LabelRunCounts {
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(LabelRunCensus::Gap::Count)> gaps{};
    std::array<std::atomic<std::uint64_t>, 256> gapOpcodes{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(LabelRunCensus::End::Count)> ends{};
    std::array<std::atomic<std::uint64_t>, 256> endOpcodes{};
    std::array<std::atomic<std::uint64_t>, 4> lengths{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawPipeline::CommitKind::Count) + 1> committedAfter{};
    std::atomic<std::uint64_t> queued{0};
    std::atomic<std::uint64_t> queuedItems{0};
    std::atomic<std::uint64_t> runs{0};
    std::atomic<std::uint64_t> runPackets{0};
    std::atomic<std::uint64_t> runItems{0};
    std::atomic<std::uint64_t> singles{0};
    std::atomic<std::uint64_t> committedItems{0};
    std::atomic<std::uint64_t> committedLabels{0};
    std::atomic<std::uint64_t> verifiedRuns{0};
    std::atomic<std::uint64_t> verifiedPackets{0};
    std::atomic<std::uint64_t> verifyDiffering{0};
};

LabelRunCounts& labelRunCounts() {
    static auto* counts = new LabelRunCounts;
    return *counts;
}

std::string takeTopOpcodes(std::array<std::atomic<std::uint64_t>, 256>& opcodes) {
    std::array<std::uint64_t, 256> taken{};
    for (std::size_t index = 0; index < taken.size(); ++index) taken[index] = opcodes[index].exchange(0, std::memory_order_relaxed);
    std::string text;
    for (int shown = 0; shown < 4; ++shown) {
        const auto top = std::max_element(taken.begin(), taken.end());
        if (*top == 0) break;
        char item[40];
        std::snprintf(item, sizeof(item), "%s0x%02x x%llu", text.empty() ? "" : " ", static_cast<unsigned>(top - taken.begin()), static_cast<unsigned long long>(*top));
        text += item;
        *top = 0;
    }
    return text.empty() ? std::string("none") : text;
}

}

LabelRun::Lookup LabelRun::Find(std::uint64_t address, std::size_t size, std::uint64_t& value) const {
    const auto end = address + size;
    for (auto label = labels.rbegin(); label != labels.rend(); ++label) {
        if (!label->stores) continue;
        const auto last = label->address + label->size;
        if (address >= last || label->address >= end) continue;
        if (label->size == 0 || address < label->address || end > last || size > sizeof(value)) return Lookup::Unknown;
        value = 0;
        std::memcpy(&value, bytes.data() + label->offset + (address - label->address), size);
        return Lookup::Value;
    }
    return Lookup::NotHeld;
}

namespace LabelRunCensus {

bool Enabled() {
    static const bool enabled = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return enabled;
}

void NoteQueued(Gap gap, std::uint32_t otherOpcode, bool item) {
    auto& counts = labelRunCounts();
    counts.queued.fetch_add(1, std::memory_order_relaxed);
    if (item) counts.queuedItems.fetch_add(1, std::memory_order_relaxed);
    counts.gaps[static_cast<std::size_t>(gap)].fetch_add(1, std::memory_order_relaxed);
    if (gap == Gap::Other) counts.gapOpcodes[otherOpcode & 0xffu].fetch_add(1, std::memory_order_relaxed);
}

void NoteRun(std::size_t packets, std::size_t items, End end, std::uint32_t otherOpcode) {
    auto& counts = labelRunCounts();
    counts.ends[static_cast<std::size_t>(end)].fetch_add(1, std::memory_order_relaxed);
    if (end == End::Other) counts.endOpcodes[otherOpcode & 0xffu].fetch_add(1, std::memory_order_relaxed);
    if (packets < 2) {
        counts.singles.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    counts.runs.fetch_add(1, std::memory_order_relaxed);
    counts.runPackets.fetch_add(packets, std::memory_order_relaxed);
    counts.runItems.fetch_add(items, std::memory_order_relaxed);
    counts.lengths[packets == 2 ? 0 : packets <= 4 ? 1 : packets <= 8 ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
}

void NoteVerify(std::size_t packets, std::size_t differing) {
    auto& counts = labelRunCounts();
    counts.verifiedRuns.fetch_add(1, std::memory_order_relaxed);
    counts.verifiedPackets.fetch_add(packets, std::memory_order_relaxed);
    counts.verifyDiffering.fetch_add(differing, std::memory_order_relaxed);
}

void NoteCommitted(DrawPipeline::CommitKind previous, bool first, std::uint32_t labels) {
    auto& counts = labelRunCounts();
    counts.committedItems.fetch_add(1, std::memory_order_relaxed);
    counts.committedLabels.fetch_add(labels, std::memory_order_relaxed);
    counts.committedAfter[first ? static_cast<std::size_t>(DrawPipeline::CommitKind::Count) : static_cast<std::size_t>(previous)].fetch_add(1, std::memory_order_relaxed);
}

std::string Take() {
    auto& counts = labelRunCounts();
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto queued = take(counts.queued);
    const auto committedItems = take(counts.committedItems);
    const auto committedLabels = take(counts.committedLabels);
    const auto verifiedRuns = take(counts.verifiedRuns);
    if (queued == 0 && committedItems == 0 && verifiedRuns == 0) return {};
    const auto queuedItems = take(counts.queuedItems);
    std::array<unsigned long long, static_cast<std::size_t>(Gap::Count)> gaps{};
    for (std::size_t index = 0; index < gaps.size(); ++index) gaps[index] = take(counts.gaps[index]);
    std::array<unsigned long long, static_cast<std::size_t>(End::Count)> ends{};
    for (std::size_t index = 0; index < ends.size(); ++index) ends[index] = take(counts.ends[index]);
    std::array<unsigned long long, 4> lengths{};
    for (std::size_t index = 0; index < lengths.size(); ++index) lengths[index] = take(counts.lengths[index]);
    const auto gapOpcodes = takeTopOpcodes(counts.gapOpcodes);
    const auto endOpcodes = takeTopOpcodes(counts.endOpcodes);
    const auto at = [](auto value) { return static_cast<std::size_t>(value); };
    char head[1024];
    std::snprintf(head, sizeof(head), "label packets queued on the lean loop %llu (label items %llu, storing nothing %llu); since the previous queued label of the segment: none %llu, no packet %llu, register packets only %llu, waits and register packets only %llu, draws %llu, dispatches %llu, other packets %llu (first other opcode: %s), more than 32 packets %llu; ",
        queued, queuedItems, queued - std::min(queued, queuedItems), gaps[at(Gap::First)], gaps[at(Gap::Adjacent)], gaps[at(Gap::Registers)], gaps[at(Gap::Waits)], gaps[at(Gap::Draws)], gaps[at(Gap::Dispatches)], gaps[at(Gap::Other)], gapOpcodes.c_str(), gaps[at(Gap::Far)]);
    char runs[1024];
    std::snprintf(runs, sizeof(runs), "runs %llu of %llu packets with %llu label items (2 packets %llu, 3-4 %llu, 5-8 %llu, 9+ %llu), single labels %llu; runs and single labels ended by: segment end %llu, a draw %llu, a dispatch %llu, a wait %llu, register packets %llu, another packet %llu (%s), a label not queued %llu, an interrupting label %llu, the failure check %llu, the cap %llu; ",
        take(counts.runs), take(counts.runPackets), take(counts.runItems), lengths[0], lengths[1], lengths[2], lengths[3], take(counts.singles), ends[at(End::SegmentEnd)], ends[at(End::Draw)], ends[at(End::Dispatch)], ends[at(End::Wait)], ends[at(End::Registers)], ends[at(End::Other)], endOpcodes.c_str(), ends[at(End::Unqueued)], ends[at(End::Interrupt)], ends[at(End::Check)], ends[at(End::Cap)]);
    std::string after;
    for (std::size_t index = 0; index < counts.committedAfter.size(); ++index) {
        const auto count = take(counts.committedAfter[index]);
        if (count == 0) continue;
        char item[48];
        std::snprintf(item, sizeof(item), " %s %llu", index < KindNames.size() ? KindNames[index] : "first", count);
        after += item;
    }
    char tail[512];
    std::snprintf(tail, sizeof(tail), "committer label items %llu holding %llu labels, by the item committed before:%s; verify: runs %llu, packets %llu, differing %llu", committedItems, committedLabels, after.empty() ? " none" : after.c_str(), verifiedRuns, take(counts.verifiedPackets), take(counts.verifyDiffering));
    return std::string(head) + runs + tail;
}

}

namespace {
struct FinisherThreadSlot {};
struct FinisherProducerSlot {};
struct FinisherSeqSlot {};
struct FinisherAheadSlot {};
struct FinisherDepositSlot {};
using FinisherThread = HostThreadSlot<bool, FinisherThreadSlot>;
using FinisherProducer = HostThreadSlot<bool, FinisherProducerSlot>;
using FinisherSeq = HostThreadSlot<std::uint64_t, FinisherSeqSlot>;
using FinisherAhead = HostThreadSlot<bool, FinisherAheadSlot>;
using FinisherDeposits = HostThreadSlot<bool, FinisherDepositSlot>;
struct FinisherRunLoopSlot {};
struct FinisherRetiredSlot {};
using FinisherRunLoop = HostThreadSlot<bool, FinisherRunLoopSlot>;
using FinisherRetired = HostThreadSlot<bool, FinisherRetiredSlot>;

struct RetiredHits {
    static constexpr std::size_t Cap = 256;
    std::size_t head = 0;
    std::size_t count = 0;
    std::array<std::array<std::shared_ptr<const void>, 3>, Cap> held;
};
struct RetiredHitsStorage {};
std::atomic<std::uint64_t> deepestRetiredHits{0};

enum LeanSlotState : std::uint8_t { LeanFree, LeanReady, LeanDone, LeanDeposited, LeanForwarding };

std::atomic<std::uint64_t> depositedItems{0};
std::atomic<std::uint64_t> forwardedItems{0};
std::atomic<std::uint64_t> turnItems{0};

std::uint64_t depositVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_FINISHER_DEPOSIT");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

std::atomic<std::uint64_t> orderChecks{0};
std::atomic<std::uint64_t> orderDifferences{0};
std::atomic<std::uint64_t> drainChecks{0};
std::atomic<std::uint64_t> drainDifferences{0};
std::atomic<std::uint64_t> drainOrdinal{0};
std::atomic<std::uint64_t> depositPrinted{0};
std::atomic<std::int64_t> depositPrintWindow{0};

void noteDepositDifference(const std::string& text) {
    const auto window = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (depositPrintWindow.exchange(window, std::memory_order_relaxed) != window) depositPrinted.store(0, std::memory_order_relaxed);
    if (depositPrinted.fetch_add(1, std::memory_order_relaxed) < 8) std::fprintf(stderr, "[finisher-deposit-verify] %s\n", text.c_str());
}

thread_local std::uint64_t keyBase = 0;
thread_local std::uint64_t keyOrdinal = 0;

std::uint64_t orderKey() {
    std::uint64_t base = 0;
    if (const auto seq = FinisherSeq::Get(); seq != 0) base = seq * 2 + 1;
    else if (FinisherProducer::Get() && !FinisherThread::Get()) base = DrawFinisher::Get().Issued() * 2;
    if (base == 0) return 0;
    if (keyBase != base) {
        keyBase = base;
        keyOrdinal = 0;
    }
    ++keyOrdinal;
    return (base << 16) | std::min<std::uint64_t>(keyOrdinal, 0xffff);
}

std::string orderText(std::uint64_t order) {
    const auto base = order >> 16;
    return std::string(base % 2 != 0 ? "job " : "worker item before job ") + std::to_string(base / 2) + " (item " + std::to_string(order & 0xffff) + ")";
}

std::atomic<std::uint64_t> helpedRoomJobs{0};
std::atomic<std::uint64_t> helpedDrainJobs{0};
std::atomic<std::uint64_t> helpedNs{0};
std::atomic<std::int64_t> helpedWindow{-1};

void noteHelped(bool room, std::uint64_t jobs, std::chrono::steady_clock::time_point start) {
    const auto end = std::chrono::steady_clock::now();
    (room ? helpedRoomJobs : helpedDrainJobs).fetch_add(jobs, std::memory_order_relaxed);
    helpedNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()), std::memory_order_relaxed);
    const std::int64_t window = std::chrono::duration_cast<std::chrono::seconds>(end.time_since_epoch()).count() / 10;
    if (helpedWindow.exchange(window, std::memory_order_relaxed) == window) return;
    std::fprintf(stderr, "[worker-helps] %llu finisher jobs run by the queue worker while it waited for ring room, %llu while it drained the ring, %.1f ms in them\n", static_cast<unsigned long long>(helpedRoomJobs.load(std::memory_order_relaxed)), static_cast<unsigned long long>(helpedDrainJobs.load(std::memory_order_relaxed)), static_cast<double>(helpedNs.load(std::memory_order_relaxed)) / 1e6);
}

struct HelpScope {
    bool thread = FinisherThread::Get();
    bool producer = FinisherProducer::Get();
    bool ahead = FinisherAhead::Get();
    bool deposits = FinisherDeposits::Get();
    bool active = DrawPipeline::Active();
    GuestMemory::PacketTag packet = GuestMemory::CurrentPacket();
    std::uint64_t base = keyBase;
    std::uint64_t ordinal = keyOrdinal;

    HelpScope() {
        FinisherThread::Set(true);
        FinisherProducer::Set(false);
        FinisherAhead::Set(false);
        FinisherDeposits::Set(false);
    }

    HelpScope(const HelpScope&) = delete;
    HelpScope& operator=(const HelpScope&) = delete;

    ~HelpScope() {
        FinisherThread::Set(thread);
        FinisherProducer::Set(producer);
        FinisherAhead::Set(ahead);
        FinisherDeposits::Set(deposits);
        DrawPipeline::Active() = active;
        GuestMemory::SetCurrentPacket(packet.opcode, packet.queue);
        keyBase = base;
        keyOrdinal = ordinal;
    }
};

bool finisherTakes(unsigned index) {
    if (index == 0) return true;
    if (index >= 5) return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::SiblingFinishers);
    constexpr Graphics::RuntimeSwitch switches[4] = {Graphics::RuntimeSwitch::SecondFinisher, Graphics::RuntimeSwitch::ThirdFinisher, Graphics::RuntimeSwitch::FourthFinisher, Graphics::RuntimeSwitch::FifthFinisher};
    return Graphics::RuntimeSwitches::Set(switches[index - 1]);
}

constexpr const char* FinisherNames[9] = {"draw finisher", "draw finisher 2", "draw finisher 3", "draw finisher 4", "draw finisher 5", "draw finisher 6", "draw finisher 7", "draw finisher 8", "draw finisher 9"};

constexpr unsigned PublishedReadAttempts = 8;

unsigned takingFinishers(std::size_t extras) {
    unsigned wanted = 1u;
    for (unsigned index = 1; index <= extras; ++index) wanted |= finisherTakes(index) ? 1u << index : 0u;
    return wanted;
}

std::uint64_t parkVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_FINISHER_PARK");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

std::atomic<std::uint64_t> parkVerifyOrdinal{0};
std::atomic<std::uint64_t> parkVerifyWaits{0};
std::atomic<std::uint64_t> parkVerifySlices{0};
std::atomic<std::uint64_t> parkVerifyDiffering{0};
std::atomic<std::int64_t> parkVerifyWindow{-1};

bool parkVerifyDue() {
    const auto every = parkVerifyEvery();
    return every != 0 && parkVerifyOrdinal.fetch_add(1, std::memory_order_relaxed) % every == 0;
}

void noteParkVerifyCounts() {
    const std::int64_t window = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (parkVerifyWindow.exchange(window, std::memory_order_relaxed) == window) return;
    std::fprintf(stderr, "[finisher-park-verify] %llu worker waits checked, %llu slices of 50 ms, %llu differing\n", static_cast<unsigned long long>(parkVerifyWaits.load(std::memory_order_relaxed)), static_cast<unsigned long long>(parkVerifySlices.load(std::memory_order_relaxed)), static_cast<unsigned long long>(parkVerifyDiffering.load(std::memory_order_relaxed)));
}

#ifdef _WIN32
const bool hostLargeCacheHooked = [] {
    SetHostLargeCacheQuery(+[] { return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::HostLargeCache); });
    SetHostPrefetchQuery(+[] { return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::HostPrefetch); });
    SetHostDeepCacheQuery(+[] { return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::HostDeepCache); });
    return true;
}();
#endif

bool lockless() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::LocklessHazards);
}

bool deepQueues() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DeepQueues);
}

std::uint64_t finisherDepth() {
    if (!deepQueues()) return 16;
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DeeperRing) ? PublishedKeys::Capacity : PublishedKeys::DeepDepth;
}

bool pipelineStats() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return stats;
}

bool spinHandoff() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::SpinHandoff);
}

constexpr std::chrono::microseconds HandoffLockSpin{20};

struct ProbedOwner {
    std::atomic<const void*> lock{nullptr};
    std::atomic<std::uint32_t> thread{0};
};

std::atomic<std::uint32_t>& probedOwner(const void* lock) {
    static std::array<ProbedOwner, 8> owners;
    for (auto& entry : owners) {
        const void* held = entry.lock.load(std::memory_order_acquire);
        if (held == lock) return entry.thread;
        if (held == nullptr && entry.lock.compare_exchange_strong(held, lock)) return entry.thread;
        if (held == lock) return entry.thread;
    }
    return owners.back().thread;
}

template<typename TMutex>
std::unique_lock<TMutex> lockFor(TMutex& mutex) {
    if (LockProbeMs() > 0.0) {
        static const std::atomic<std::uint32_t> none{0};
        auto& owner = probedOwner(&mutex);
        if (!mutex.try_lock()) AwaitProbedLock("pipeline or finisher mutex", &mutex, owner, none, [&] { return mutex.try_lock(); });
        owner.store(LockProbeThread(), std::memory_order_relaxed);
        return std::unique_lock<TMutex>(mutex, std::adopt_lock);
    }
    if (!spinHandoff()) return std::unique_lock<TMutex>(mutex);
    LockSpinning(mutex, HandoffLockSpin);
    return std::unique_lock<TMutex>(mutex, std::adopt_lock);
}

bool countRange(std::vector<std::pair<DrawPipeline::Range, std::uint32_t>>& counted, const DrawPipeline::Range& range, bool added) {
    const auto found = std::find_if(counted.begin(), counted.end(), [&](const auto& entry) { return entry.first == range; });
    if (added) {
        if (found != counted.end()) {
            ++found->second;
            return false;
        }
        counted.emplace_back(range, 1u);
        return true;
    }
    if (found == counted.end() || --found->second != 0) return false;
    *found = counted.back();
    counted.pop_back();
    return true;
}
}

namespace {
enum class DeepLookup : std::uint32_t { PendingWrites, PendingLabel, PendingKey, EarlierKey, AheadWrites, QueuedWrites, StoredBytes, QueuedLabel, Watch, HeldReads, Count };
constexpr std::array<const char*, static_cast<std::size_t>(DeepLookup::Count)> DeepLookupNames{"pending writes", "pending label", "pending key", "earlier key", "ahead writes", "queued writes", "stored bytes", "queued label", "watch", "held reads"};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DeepLookup::Count)> deepAnswered{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DeepLookup::Count)> deepFellBack{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DeepLookup::Count)> deepVerifyOrdinals{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DeepSite::Count)> deepSiteJobWaits{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DeepSite::Count)> deepSiteItemWaits{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawFinisher::DeepCount::Count)> deepTotals{};
std::array<std::uint64_t, 4> deepTotalsReported{};
std::atomic<std::uint64_t> deepTurnWrites{0};
std::atomic<std::uint64_t> deepTurnSkipped{0};
std::atomic<std::uint64_t> deepTurnConservative{0};
std::atomic<std::uint64_t> deepTurnFull{0};
std::atomic<std::uint64_t> deepTurnUnknown{0};
std::atomic<std::uint64_t> deepVerifyChecked{0};
std::atomic<std::uint64_t> deepVerifyDiffering{0};
std::atomic<std::uint64_t> deepVerifyConservative{0};
std::atomic<std::uint64_t> deepVerifyPrinted{0};
std::atomic<std::int64_t> deepVerifyWindow{0};

std::uint64_t deepVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_DEEP_FLOW");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

bool deepVerifyDue(DeepLookup lookup) {
    const auto every = deepVerifyEvery();
    return every != 0 && deepVerifyOrdinals[static_cast<std::size_t>(lookup)].fetch_add(1, std::memory_order_relaxed) % every == 0;
}

void noteDeepLookup(DeepLookup lookup, bool fellBack) {
    if (!pipelineStats()) return;
    (fellBack ? deepFellBack : deepAnswered)[static_cast<std::size_t>(lookup)].fetch_add(1, std::memory_order_relaxed);
}

void noteDeepTotal(DrawFinisher::DeepCount count) {
    deepTotals[static_cast<std::size_t>(count)].fetch_add(1, std::memory_order_relaxed);
}

void noteDeepSiteWait(DeepSite site, bool job) {
    if (!pipelineStats()) return;
    (job ? deepSiteJobWaits : deepSiteItemWaits)[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
}

void raiseQueued(Graphics::WriteValidation::Source source, std::span<const std::pair<std::uint64_t, std::uint64_t>> writes) {
    if (writes.empty() || Graphics::WriteValidation::Session() == 0 || !Graphics::WriteValidation::FinisherSide()) return;
    for (const auto& [begin, end] : writes) Graphics::WriteValidation::Raise(Graphics::WriteValidation::Class::Q, source, begin, end);
}

bool rangesOverlap(std::span<const std::pair<std::uint64_t, std::uint64_t>> writes, std::span<const std::pair<std::uint64_t, std::uint64_t>> reads) {
    for (const auto& [first, last] : writes) {
        for (const auto& [begin, end] : reads) {
            if (begin < last && first < end) return true;
        }
    }
    return false;
}

DeepFound deepLater(const DeepFound& left, const DeepFound& right) {
    if (left.answer == DeepAnswer::Unsure || right.answer == DeepAnswer::Unsure) return {DeepAnswer::Unsure};
    if (right.answer != DeepAnswer::Hit) return left;
    if (left.answer != DeepAnswer::Hit || right.stamp > left.stamp) return right;
    return left;
}

void noteDeepVerify(DeepLookup lookup, const DeepFound& index, const DeepFound& scan, bool stamped, std::uint64_t front, std::uint64_t asker, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    deepVerifyChecked.fetch_add(1, std::memory_order_relaxed);
    const bool hit = index.answer == DeepAnswer::Hit;
    const bool found = scan.answer == DeepAnswer::Hit;
    const bool differs = (!hit && found) || (stamped && hit && found && scan.stamp > index.stamp);
    if (!differs) {
        if ((hit && !found) || (stamped && hit && found && index.stamp > scan.stamp)) deepVerifyConservative.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    deepVerifyDiffering.fetch_add(1, std::memory_order_relaxed);
    const auto window = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (deepVerifyWindow.exchange(window, std::memory_order_relaxed) != window) deepVerifyPrinted.store(0, std::memory_order_relaxed);
    if (deepVerifyPrinted.fetch_add(1, std::memory_order_relaxed) >= 8) return;
    const auto wide = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto range = ranges.empty() ? std::pair<std::uint64_t, std::uint64_t>{} : ranges.front();
    std::fprintf(stderr, "[deep-flow-verify] %s differs: index %s %llu, scan %s %llu; front %llu, asker %llu, range %llx..%llx\n", DeepLookupNames[static_cast<std::size_t>(lookup)], hit ? "hit" : "clear", wide(index.stamp), found ? "hit" : "clear", wide(scan.stamp), wide(front), wide(asker), wide(range.first), wide(range.second));
}

std::size_t readRegionSlot(std::uint64_t region) {
    return static_cast<std::size_t>((region * 0x9E3779B97F4A7C15ull) >> 52u);
}
}

namespace {
enum class RingScan : std::uint32_t { QueuedItems, QueuedWriter, QueuedScan, DeepVerify, HeldReadsVerify, HeldReads, RunningReads, QueuedLabel, StoredBytes, Watch, WrittenSince, QueuedWrites, ValidationClear, Count };
constexpr std::array<const char*, static_cast<std::size_t>(RingScan::Count)> RingScanNames{"queued items", "queued writer", "queued scan", "deep verify", "held reads verify", "held reads", "running reads", "queued label", "stored bytes", "watch", "written since", "queued writes", "validation clear"};
static_assert(RingScanNames.back() != nullptr, "every committer ring scan needs a name");
constexpr std::uint64_t NoRingWake = ~std::uint64_t{0};
constexpr std::uint32_t RingLockSpins = 4096;
static_assert(DrawPipeline::CommitRingSlots >= DrawFinisher::LeanSlots && (DrawPipeline::CommitRingSlots & (DrawPipeline::CommitRingSlots - 1)) == 0, "the committer ring must hold the deepest committer queue, in a power-of-two number of slots");
struct alignas(64) RingTotalLine {
    std::atomic<std::uint64_t> value{0};
};
std::array<RingTotalLine, static_cast<std::size_t>(DrawPipeline::RingCount::Count)> ringTotals{};
std::array<std::uint64_t, static_cast<std::size_t>(DrawPipeline::RingCount::Count)> ringReported{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(RingScan::Count)> ringScanTotals{};
std::array<std::uint64_t, static_cast<std::size_t>(RingScan::Count)> ringScanReported{};
std::atomic<std::uint64_t> ringHighest{0};
std::atomic<std::uint64_t> ringScanOrdinal{0};
std::atomic<std::uint64_t> ringVerifyPrinted{0};
std::atomic<std::int64_t> ringVerifyWindow{0};

std::uint32_t ringScanId(RingScan scan) {
    return static_cast<std::uint32_t>(scan);
}

void bumpRing(DrawPipeline::RingCount count, std::uint64_t by = 1) {
    ringTotals[static_cast<std::size_t>(count)].value.fetch_add(by, std::memory_order_relaxed);
}

void bumpOwnedRing(DrawPipeline::RingCount count, std::uint64_t by = 1) {
    auto& total = ringTotals[static_cast<std::size_t>(count)].value;
    total.store(total.load(std::memory_order_relaxed) + by, std::memory_order_relaxed);
}

std::uint64_t ringVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_COMMIT_RING");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

bool ringWanted() {
    return DrawFinisher::Deep() && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitRing);
}

void noteRingVerify(const char* check, bool differs, const char* detail) {
    if (!differs) return;
    bumpRing(DrawPipeline::RingCount::VerifyDiffering);
    const auto window = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (ringVerifyWindow.exchange(window, std::memory_order_relaxed) != window) ringVerifyPrinted.store(0, std::memory_order_relaxed);
    if (ringVerifyPrinted.fetch_add(1, std::memory_order_relaxed) >= 8) return;
    std::fprintf(stderr, "[commit-ring-verify] %s differs: %s\n", check, detail);
}
}

class DrawPipeline::RingLock {
public:
    explicit RingLock(DrawPipeline& pipeline) : owner(pipeline) { Lock(); }
    ~RingLock() {
        if (held) Unlock();
    }
    RingLock(const RingLock&) = delete;
    RingLock& operator=(const RingLock&) = delete;

    void Lock() {
        if (owner.ringLocked.exchange(true, std::memory_order_acquire)) {
            bumpRing(RingCount::Contention);
            std::uint32_t spins = 0;
            for (;;) {
                while (owner.ringLocked.load(std::memory_order_relaxed)) {
                    if (spins < RingLockSpins) {
                        ++spins;
                        SpinPause();
                    } else {
                        std::this_thread::yield();
                    }
                }
                if (!owner.ringLocked.exchange(true, std::memory_order_acquire)) break;
            }
        }
        held = true;
    }

    void Unlock() {
        held = false;
        owner.ringLocked.store(false, std::memory_order_release);
    }

private:
    DrawPipeline& owner;
    bool held = false;
};

class DrawPipeline::QueueScan {
public:
    QueueScan(DrawPipeline& pipeline, std::uint32_t caller) : owner(pipeline), lock(lockFor(pipeline.mutex)), view(pipeline.items) {
        if (!owner.commitRing.load(std::memory_order_relaxed)) {
            retired = owner.retiredItems.load(std::memory_order_relaxed);
            return;
        }
        owner.ringScans.fetch_add(1, std::memory_order_seq_cst);
        frozen = true;
        for (std::uint32_t spins = 0; owner.committerRetiring.load(std::memory_order_seq_cst); ++spins) {
            if (spins < RingLockSpins) SpinPause();
            else std::this_thread::yield();
        }
        retired = owner.retiredItems.load(std::memory_order_seq_cst);
        end = owner.enqueuedItems.load(std::memory_order_seq_cst);
        view = QueuedView(owner.ring->data(), retired, end);
        bumpRing(RingCount::SlowScans);
        ringScanTotals[std::min<std::size_t>(caller, ringScanTotals.size() - 1)].fetch_add(1, std::memory_order_relaxed);
        const auto every = ringVerifyEvery();
        verifying = every != 0 && ringScanOrdinal.fetch_add(1, std::memory_order_relaxed) % every == 0;
    }
    ~QueueScan() {
        if (frozen) owner.ringScans.fetch_sub(1, std::memory_order_seq_cst);
    }
    QueueScan(const QueueScan&) = delete;
    QueueScan& operator=(const QueueScan&) = delete;

    const QueuedView& Items() const { return view; }
    std::uint64_t Retired() const { return retired; }
    std::uint64_t Enqueued() const { return frozen ? end : owner.enqueuedItems.load(std::memory_order_relaxed); }
    bool Frozen() const { return frozen; }

    template<typename TScan>
    void Check(const char* check, TScan&& scan) const {
        if (!verifying) return;
        const auto first = scan(view);
        const RingLock stop(owner);
        bumpRing(RingCount::VerifyScans);
        const auto retiredNow = owner.retiredItems.load(std::memory_order_seq_cst);
        std::uint64_t moved = 0;
        for (auto ordinal = retired; ordinal < end; ++ordinal) {
            if (owner.ringSlot(ordinal).ringOrdinal != ordinal + 1) ++moved;
        }
        const auto again = scan(view);
        const bool same = first == again;
        char text[320];
        std::snprintf(text, sizeof(text), "items %llu to %llu scanned with the committer held; under the full stop %llu retired, %llu of the scanned slots no longer hold their item, the repeated scan %s", static_cast<unsigned long long>(retired), static_cast<unsigned long long>(end), static_cast<unsigned long long>(retiredNow), static_cast<unsigned long long>(moved), same ? "agrees" : "answers otherwise");
        noteRingVerify(check, retiredNow != retired || moved != 0 || !same, text);
    }

private:
    DrawPipeline& owner;
    std::unique_lock<std::mutex> lock;
    bool frozen = false;
    bool verifying = false;
    std::uint64_t retired = 0;
    std::uint64_t end = 0;
    QueuedView view;
};

void PublishedRanges::Publish(std::span<const std::pair<Range, std::uint32_t>> entries, std::uint32_t unknownHeld) {
    const auto before = seq.load(std::memory_order_relaxed);
    seq.store(before + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    const auto size = entries.size() <= Capacity ? static_cast<std::uint32_t>(entries.size()) : Capacity + 1;
    std::uint64_t low = ~std::uint64_t{0};
    std::uint64_t high = 0;
    for (std::uint32_t index = 0; index < size && index < Capacity; ++index) {
        bounds[index * 2].store(entries[index].first.first, std::memory_order_relaxed);
        bounds[index * 2 + 1].store(entries[index].first.second, std::memory_order_relaxed);
        low = std::min(low, entries[index].first.first);
        high = std::max(high, entries[index].first.second);
    }
    lowest.store(low, std::memory_order_relaxed);
    highest.store(high, std::memory_order_relaxed);
    count.store(size, std::memory_order_relaxed);
    unknown.store(unknownHeld, std::memory_order_relaxed);
    seq.store(before + 2, std::memory_order_release);
}

void PublishedRanges::PublishUnknownJobs(std::span<const std::uint64_t> jobs) {
    const auto before = seq.load(std::memory_order_relaxed);
    seq.store(before + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    const auto size = jobs.size() <= UnknownCapacity ? static_cast<std::uint32_t>(jobs.size()) : UnknownCapacity + 1;
    for (std::uint32_t index = 0; index < size && index < UnknownCapacity; ++index) unknownJobs[index].store(jobs[index], std::memory_order_relaxed);
    unknownJobCount.store(size, std::memory_order_relaxed);
    seq.store(before + 2, std::memory_order_release);
}

bool PublishedRanges::UnknownAhead(std::uint64_t front, std::uint64_t asker, std::uint64_t& blocker) const {
    for (unsigned attempt = 0; attempt < PublishedReadAttempts; ++attempt) {
        const auto before = seq.load(std::memory_order_acquire);
        if ((before & 1u) != 0) {
            SpinPause();
            continue;
        }
        const auto held = unknown.load(std::memory_order_relaxed);
        const auto size = unknownJobCount.load(std::memory_order_relaxed);
        bool found = false;
        std::uint64_t last = 0;
        for (std::uint32_t index = 0; index < size && index < UnknownCapacity; ++index) {
            const auto job = unknownJobs[index].load(std::memory_order_relaxed);
            if (job < front || job >= asker) continue;
            found = true;
            last = std::max(last, job);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (seq.load(std::memory_order_relaxed) != before) continue;
        if (size > UnknownCapacity || (held != 0 && size == 0)) break;
        blocker = last;
        return found;
    }
    blocker = asker - 1;
    return true;
}

bool PublishedRanges::UnknownHeld() const {
    for (unsigned attempt = 0; attempt < PublishedReadAttempts; ++attempt) {
        const auto before = seq.load(std::memory_order_acquire);
        if ((before & 1u) != 0) {
            SpinPause();
            continue;
        }
        const auto held = unknown.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (seq.load(std::memory_order_relaxed) != before) continue;
        return held != 0;
    }
    return true;
}

PublishedRanges::Answer PublishedRanges::Overlaps(std::span<const Range> ranges, Range* pendingHit, bool* unknownHeld, bool ignoreUnknown) const {
    for (unsigned attempt = 0; attempt < PublishedReadAttempts; ++attempt) {
        const auto before = seq.load(std::memory_order_acquire);
        if ((before & 1u) != 0) {
            SpinPause();
            continue;
        }
        const auto held = ignoreUnknown ? 0u : unknown.load(std::memory_order_relaxed);
        const auto size = count.load(std::memory_order_relaxed);
        const auto low = lowest.load(std::memory_order_relaxed);
        const auto high = highest.load(std::memory_order_relaxed);
        const bool outside = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FastSettle) && std::none_of(ranges.begin(), ranges.end(), [&](const Range& range) { return range.first < high && low < range.second; });
        bool hit = false;
        Range found{};
        for (std::uint32_t index = 0; held == 0 && !outside && index < size && index < Capacity && !hit; ++index) {
            const auto first = bounds[index * 2].load(std::memory_order_relaxed);
            const auto last = bounds[index * 2 + 1].load(std::memory_order_relaxed);
            hit = std::any_of(ranges.begin(), ranges.end(), [&](const Range& range) { return range.first < last && first < range.second; });
            if (hit) found = {first, last};
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (seq.load(std::memory_order_relaxed) != before) continue;
        if (unknownHeld != nullptr) *unknownHeld = held != 0;
        if (held != 0) return Answer::Hit;
        if (size > Capacity) return Answer::Unsure;
        if (hit && pendingHit != nullptr) *pendingHit = found;
        return hit ? Answer::Hit : Answer::Clear;
    }
    return Answer::Unsure;
}

bool PublishedRanges::Copy(std::array<Range, Capacity>& out, std::uint32_t& size) const {
    for (unsigned attempt = 0; attempt < PublishedReadAttempts; ++attempt) {
        const auto before = seq.load(std::memory_order_acquire);
        if ((before & 1u) != 0) {
            SpinPause();
            continue;
        }
        const auto held = unknown.load(std::memory_order_relaxed);
        const auto stored = count.load(std::memory_order_relaxed);
        for (std::uint32_t index = 0; held == 0 && index < stored && index < Capacity; ++index) {
            out[index] = {bounds[index * 2].load(std::memory_order_relaxed), bounds[index * 2 + 1].load(std::memory_order_relaxed)};
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (seq.load(std::memory_order_relaxed) != before) continue;
        if (held != 0 || stored > Capacity) return false;
        size = stored;
        return true;
    }
    return false;
}

void PublishedKeys::Push(std::uint64_t seqNumber, std::uint64_t key, std::uint64_t shape, bool inserts) {
    const auto before = seq.load(std::memory_order_relaxed);
    seq.store(before + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    const auto slot = static_cast<std::size_t>(seqNumber % Capacity) * 3;
    slots[slot].store(key, std::memory_order_relaxed);
    slots[slot + 1].store(shape, std::memory_order_relaxed);
    slots[slot + 2].store(inserts ? 1u : 0u, std::memory_order_relaxed);
    tail.store(seqNumber + 1, std::memory_order_relaxed);
    seq.store(before + 2, std::memory_order_release);
}

void PublishedKeys::Retire(std::uint64_t frontSeq) {
    if (head.load(std::memory_order_relaxed) == frontSeq) return;
    const auto before = seq.load(std::memory_order_relaxed);
    seq.store(before + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    head.store(frontSeq, std::memory_order_relaxed);
    seq.store(before + 2, std::memory_order_release);
}

PublishedKeys::Answer PublishedKeys::Find(std::uint64_t key, std::uint64_t shape, bool insertsOnly) const {
    for (unsigned attempt = 0; attempt < PublishedReadAttempts; ++attempt) {
        const auto before = seq.load(std::memory_order_acquire);
        if ((before & 1u) != 0) {
            SpinPause();
            continue;
        }
        const auto first = head.load(std::memory_order_relaxed);
        const auto last = tail.load(std::memory_order_relaxed);
        const bool fits = first <= last && last - first <= Capacity;
        bool hit = false;
        for (auto number = first; fits && number < last && !hit; ++number) {
            const auto slot = static_cast<std::size_t>(number % Capacity) * 3;
            const auto heldKey = slots[slot].load(std::memory_order_relaxed);
            const auto heldShape = slots[slot + 1].load(std::memory_order_relaxed);
            const bool heldInserts = slots[slot + 2].load(std::memory_order_relaxed) != 0;
            hit = (heldKey == AnyKey || heldKey == key || (shape != 0 && heldShape == shape)) && (heldInserts || !insertsOnly);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (seq.load(std::memory_order_relaxed) != before) continue;
        if (!fits) return Answer::Unsure;
        return hit ? Answer::Hit : Answer::Clear;
    }
    return Answer::Unsure;
}

PublishedKeys::Answer PublishedKeys::FindBefore(std::uint64_t key, std::uint64_t shape, bool insertsOnly, std::uint64_t before) const {
    for (unsigned attempt = 0; attempt < PublishedReadAttempts; ++attempt) {
        const auto version = seq.load(std::memory_order_acquire);
        if ((version & 1u) != 0) {
            SpinPause();
            continue;
        }
        const auto first = head.load(std::memory_order_relaxed);
        const auto pushed = tail.load(std::memory_order_relaxed);
        const auto last = std::min(pushed, before);
        const bool fits = (first > last || last - first <= Capacity) && pushed >= first && pushed - first <= Capacity;
        bool hit = false;
        for (auto number = first; fits && number < last && !hit; ++number) {
            const auto slot = static_cast<std::size_t>(number % Capacity) * 3;
            const auto heldKey = slots[slot].load(std::memory_order_relaxed);
            const auto heldShape = slots[slot + 1].load(std::memory_order_relaxed);
            const bool heldInserts = slots[slot + 2].load(std::memory_order_relaxed) != 0;
            hit = (heldKey == AnyKey || heldKey == key || (shape != 0 && heldShape == shape)) && (heldInserts || !insertsOnly);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (seq.load(std::memory_order_relaxed) != version) continue;
        if (!fits) return Answer::Unsure;
        return hit ? Answer::Hit : Answer::Clear;
    }
    return Answer::Unsure;
}

DrawFinisher& DrawFinisher::Get() {
    static DrawFinisher* finisher = new DrawFinisher();
    return *finisher;
}

bool DrawFinisher::Enabled() {
    return Graphics::DrawPipelineEnabled();
}

bool DrawPipeline::MetadataPassesPipelined() {
    return Graphics::DrawPipelineEnabled();
}

namespace {
ProfileCounter pipelinedMetadataPasses;
}

std::uint64_t DrawPipeline::PipelinedMetadataPasses() {
    return pipelinedMetadataPasses.Sum();
}

void DrawPipeline::CountPipelinedMetadataPass() {
    pipelinedMetadataPasses.Add(1);
}

bool DrawFinisher::Overlap() {
    return Graphics::DrawPipelineEnabled();
}

bool DrawFinisher::PendingOverlaps(std::uint64_t begin, std::uint64_t end) {
    if (!Busy()) return false;
    if (!FinisherProducer::Get()) {
        const Range range{begin, end};
        return aheadWait({&range, 1});
    }
    const bool ring = ringMode.load(std::memory_order_acquire);
    if (ring) retireLean();
    const bool deep = ring && deepRing.load(std::memory_order_relaxed);
    if (deep) {
        const Range range{begin, end};
        if (const auto answer = deepPendingHit({&range, 1}); answer != DeepAnswer::Unsure) return answer == DeepAnswer::Hit;
        if (rangesOverlap(openRunWrites, {&range, 1})) return true;
    }
    const bool flow = Flow();
    if (!deep && (lockless() || flow)) {
        const Range range{begin, end};
        const auto answer = published.Overlaps({&range, 1}, nullptr, nullptr, flow);
        if (answer != PublishedRanges::Answer::Unsure) return answer == PublishedRanges::Answer::Hit;
        if (flow) {
            NoteFlowLookup(FlowLookup::PendingWrites);
            return false;
        }
    }
    if (ring) {
        for (auto seq = frontSeq.load(std::memory_order_acquire); seq < nextSeq; ++seq) {
            const auto& held = leanSlot(seq).pending;
            if (held.unknown) return true;
            for (const auto& [first, last] : held.writes) {
                if (begin < last && first < end) return true;
            }
        }
        return false;
    }
    const auto lock = lockFor(mutex);
    for (const auto& held : pending) {
        if (held.unknown) return true;
        for (const auto& [first, last] : held.writes) {
            if (begin < last && first < end) return true;
        }
    }
    return false;
}

bool DrawFinisher::PendingLabelBytes(std::uint64_t address, std::span<std::byte> out) {
    if (out.empty() || (address & 3u) != 0 || (out.size() & 3u) != 0 || !Busy() || !FinisherProducer::Get()) return false;
    for (std::size_t offset = 0; offset < out.size(); offset += 4) {
        std::uint64_t value = 0;
        if (PendingLabel(address + offset, 4, value) != LabelLookup::Value) return false;
        std::memcpy(out.data() + offset, &value, 4);
    }
    return true;
}

bool DrawFinisher::PendingLabelSpan(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > 64 || (address & 3u) != 0 || (bytes & 3u) != 0 || !Busy() || !FinisherProducer::Get()) return false;
    bool known = false;
    for (std::size_t offset = 0; offset < bytes; offset += 4) {
        std::uint64_t value = 0;
        const auto found = PendingLabel(address + offset, 4, value);
        if (found == LabelLookup::Unknown) return false;
        known = known || found == LabelLookup::Value;
    }
    return known;
}

DrawFinisher::LabelLookup DrawFinisher::PendingLabel(std::uint64_t address, std::size_t bytes, std::uint64_t& value) {
    if (!Busy()) return LabelLookup::NotHeld;
    const auto end = address + bytes;
    if (!FinisherProducer::Get()) {
        const Range range{address, end};
        aheadWait({&range, 1});
        return LabelLookup::NotHeld;
    }
    const bool ring = ringMode.load(std::memory_order_acquire);
    if (ring) retireLean();
    const bool flow = Flow();
    const bool deep = ring && deepRing.load(std::memory_order_relaxed);
    if (!deep && (lockless() || flow)) {
        const Range range{address, end};
        if (published.Overlaps({&range, 1}, nullptr, nullptr, flow) == PublishedRanges::Answer::Clear) return LabelLookup::NotHeld;
    }
    const auto lookup = [&](const Pending& held) -> std::optional<LabelLookup> {
        if (held.unknown && !flow) return LabelLookup::Unknown;
        const bool overlaps = std::any_of(held.writes.begin(), held.writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
        if (!overlaps) return std::nullopt;
        if (held.labelRun != nullptr) return held.labelRun->Find(address, bytes, value) == LabelRun::Lookup::Value ? LabelLookup::Value : LabelLookup::Unknown;
        if (held.labelBytes.empty() || address < held.labelAddress || end > held.labelAddress + held.labelBytes.size()) return LabelLookup::Unknown;
        value = 0;
        std::memcpy(&value, held.labelBytes.data() + (address - held.labelAddress), bytes);
        return LabelLookup::Value;
    };
    if (deep) {
        const Range range{address, end};
        const auto front = frontSeq.load(std::memory_order_acquire);
        const auto unknown = lastUnknown.load(std::memory_order_relaxed);
        const auto found = deepWrites->Newest({&range, 1}, front);
        if (found.answer != DeepAnswer::Unsure && (found.answer != DeepAnswer::Hit || found.stamp < nextSeq)) {
            const bool byUnknown = unknown >= front && (found.answer != DeepAnswer::Hit || unknown > found.stamp);
            if (deepVerifyDue(DeepLookup::PendingLabel)) noteDeepVerify(DeepLookup::PendingLabel, byUnknown ? DeepFound{DeepAnswer::Hit, unknown} : found, scanWrites({&range, 1}, frontSeq.load(std::memory_order_acquire), nextSeq), true, front, 0, {&range, 1});
            if (byUnknown) {
                noteDeepLookup(DeepLookup::PendingLabel, false);
                return LabelLookup::Unknown;
            }
            if (found.answer != DeepAnswer::Hit) {
                noteDeepLookup(DeepLookup::PendingLabel, false);
                return LabelLookup::NotHeld;
            }
            if (const auto answer = lookup(leanSlot(found.stamp).pending)) {
                noteDeepLookup(DeepLookup::PendingLabel, false);
                return *answer;
            }
        }
        noteDeepLookup(DeepLookup::PendingLabel, true);
    }
    if (ring) {
        const auto front = frontSeq.load(std::memory_order_acquire);
        for (auto seq = nextSeq; seq-- > front;) {
            if (const auto answer = lookup(leanSlot(seq).pending)) return *answer;
        }
        return LabelLookup::NotHeld;
    }
    const auto lock = lockFor(mutex);
    for (auto held = pending.rbegin(); held != pending.rend(); ++held) {
        if (const auto answer = lookup(*held)) return *answer;
    }
    return LabelLookup::NotHeld;
}

bool DrawFinisher::PendingOverlapsAny(std::span<const Range> ranges) {
    if (!Busy()) return false;
    if (!FinisherProducer::Get()) return aheadWait(ranges);
    const bool ring = ringMode.load(std::memory_order_acquire);
    if (ring) retireLean();
    const bool deep = ring && deepRing.load(std::memory_order_relaxed);
    if (deep) {
        if (const auto answer = deepPendingHit(ranges); answer != DeepAnswer::Unsure) return answer == DeepAnswer::Hit;
        if (rangesOverlap(openRunWrites, ranges)) return true;
    }
    const bool flow = Flow();
    if (!deep && (lockless() || flow)) {
        const auto answer = published.Overlaps(ranges, nullptr, nullptr, flow);
        if (answer != PublishedRanges::Answer::Unsure) return answer == PublishedRanges::Answer::Hit;
        if (flow) {
            NoteFlowLookup(FlowLookup::PendingWrites);
            return false;
        }
    }
    if (ring) {
        for (auto seq = frontSeq.load(std::memory_order_acquire); seq < nextSeq; ++seq) {
            const auto& held = leanSlot(seq).pending;
            if (held.unknown) return true;
            for (const auto& [first, last] : held.writes) {
                for (const auto& [begin, end] : ranges) {
                    if (begin < last && first < end) return true;
                }
            }
        }
        return false;
    }
    const auto lock = lockFor(mutex);
    for (const auto& held : pending) {
        if (held.unknown) return true;
        for (const auto& [first, last] : held.writes) {
            for (const auto& [begin, end] : ranges) {
                if (begin < last && first < end) return true;
            }
        }
    }
    return false;
}

bool DrawFinisher::PendingClearForValidation(std::span<const Range> ranges) {
    if (FinisherDeposits::Get() && depositsWrite(ranges)) return false;
    if (ringMode.load(std::memory_order_acquire) && deepRing.load(std::memory_order_acquire)) {
        if (deepWrites == nullptr) return false;
        const auto front = frontSeq.load(std::memory_order_seq_cst);
        if (lastUnknown.load(std::memory_order_seq_cst) >= front) return false;
        return deepWrites->Newest(ranges, front).answer == DeepAnswer::Clear;
    }
    return published.Overlaps(ranges, nullptr, nullptr, Flow()) == PublishedRanges::Answer::Clear;
}

bool DrawFinisher::UnknownWritesAhead() {
    if (!Busy()) return false;
    auto& finisher = Get();
    if (finisher.ringMode.load(std::memory_order_acquire) && finisher.deepRing.load(std::memory_order_acquire)) return finisher.lastUnknown.load(std::memory_order_seq_cst) >= finisher.frontSeq.load(std::memory_order_seq_cst);
    if (Flow()) return false;
    return finisher.published.UnknownHeld();
}

bool DrawFinisher::PendingKey(std::uint64_t key, std::uint64_t shape) {
    FlushOpenRun();
    if (!Busy() || !FinisherProducer::Get()) return false;
    if (Flow()) {
        NoteFlowLookup(FlowLookup::SameKey);
        return false;
    }
    const bool precise = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PreciseLookupDrain);
    const bool ring = ringMode.load(std::memory_order_acquire);
    if (ring) retireLean();
    if (ring && deepRing.load(std::memory_order_relaxed)) {
        const auto front = frontSeq.load(std::memory_order_acquire);
        const auto found = keysBefore(key, shape, precise, front, ~std::uint64_t{0});
        noteDeepLookup(DeepLookup::PendingKey, found.answer == DeepAnswer::Unsure);
        if (found.answer != DeepAnswer::Unsure) {
            if (deepVerifyDue(DeepLookup::PendingKey)) noteDeepVerify(DeepLookup::PendingKey, found, scanKeys(key, shape, precise, frontSeq.load(std::memory_order_acquire), nextSeq), true, front, 0, {});
            return found.answer == DeepAnswer::Hit;
        }
        return scanKeys(key, shape, precise, frontSeq.load(std::memory_order_acquire), nextSeq).answer == DeepAnswer::Hit;
    }
    if (lockless()) {
        const auto answer = keys.Find(key, shape, precise);
        if (answer != PublishedKeys::Answer::Unsure) return answer == PublishedKeys::Answer::Hit;
    }
    const auto matches = [&](const Pending& held) { return (held.key == key || (shape != 0 && held.shape == shape)) && (held.inserts || !precise); };
    if (ring) {
        for (auto seq = frontSeq.load(std::memory_order_acquire); seq < nextSeq; ++seq) {
            if (matches(leanSlot(seq).pending)) return true;
        }
        return false;
    }
    const auto lock = lockFor(mutex);
    return std::any_of(pending.begin(), pending.end(), matches);
}

PublishedKeys::Answer DrawFinisher::KeyBefore(std::uint64_t key, std::uint64_t shape, bool insertsOnly) const {
    const auto seq = FinisherSeq::Get();
    if (seq == 0 || frontSeq.load(std::memory_order_acquire) >= seq) return PublishedKeys::Answer::Clear;
    if (Flow()) {
        NoteFlowLookup(FlowLookup::EarlierKey);
        return PublishedKeys::Answer::Clear;
    }
    if (deepRing.load(std::memory_order_acquire)) {
        const auto front = frontSeq.load(std::memory_order_acquire);
        const auto found = keysBefore(key, shape, insertsOnly, front, seq);
        noteDeepLookup(DeepLookup::EarlierKey, found.answer == DeepAnswer::Unsure);
        if (found.answer == DeepAnswer::Unsure) return PublishedKeys::Answer::Unsure;
        if (deepVerifyDue(DeepLookup::EarlierKey)) {
            const std::lock_guard lock(verifySlots);
            noteDeepVerify(DeepLookup::EarlierKey, found, scanKeys(key, shape, insertsOnly, frontSeq.load(std::memory_order_acquire), seq), true, front, seq, {});
        }
        return found.answer == DeepAnswer::Hit ? PublishedKeys::Answer::Hit : PublishedKeys::Answer::Clear;
    }
    return keys.FindBefore(key, shape, insertsOnly, seq);
}

bool DrawFinisher::AheadBusy() const {
    const auto seq = FinisherSeq::Get();
    return seq != 0 && frontSeq.load(std::memory_order_acquire) < seq;
}

bool DrawFinisher::AheadOfTurn() {
    return FinisherAhead::Get() && Get().AheadBusy();
}

DrawFinisher::AheadScope::AheadScope() : previous(FinisherAhead::Get()) {
    FinisherAhead::Set(true);
}

DrawFinisher::AheadScope::~AheadScope() {
    FinisherAhead::Set(previous);
}

namespace {
void noteTurnWait(PublishedRanges::Answer answer, bool unknownHeld, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, const std::pair<std::uint64_t, std::uint64_t>& pendingHit) {
    static std::atomic<std::uint64_t> total{0};
    static std::atomic<std::uint64_t> unknown{0};
    static std::atomic<std::uint64_t> crowded{0};
    static std::atomic<std::uint64_t> overlapping{0};
    static std::atomic<std::uint32_t> printed{0};
    static const bool examples = std::getenv("APS5_TURN_WAIT_DIAG") != nullptr;
    const auto wide = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    if (unknownHeld) {
        unknown.fetch_add(1, std::memory_order_relaxed);
    } else if (answer == PublishedRanges::Answer::Unsure) {
        crowded.fetch_add(1, std::memory_order_relaxed);
    } else {
        overlapping.fetch_add(1, std::memory_order_relaxed);
        if (examples && printed.fetch_add(1, std::memory_order_relaxed) < 32u) {
            const auto asked = std::find_if(ranges.begin(), ranges.end(), [&](const auto& range) { return range.first < pendingHit.second && pendingHit.first < range.second; });
            const auto read = asked != ranges.end() ? *asked : std::pair<std::uint64_t, std::uint64_t>{};
            std::fprintf(stderr, "[turn-waits] a read of %llx..%llx (%zu ranges asked) overlaps the pending write %llx..%llx\n", wide(read.first), wide(read.second), ranges.size(), wide(pendingHit.first), wide(pendingHit.second));
        }
    }
    if ((total.fetch_add(1, std::memory_order_relaxed) + 1u) % 8192u == 0u) {
        std::fprintf(stderr, "[turn-waits] %llu waits: a job with unknown writes outstanding %llu, a pending write over the read %llu, more than %u pending write ranges %llu\n", wide(total.load(std::memory_order_relaxed)), wide(unknown.load(std::memory_order_relaxed)), wide(overlapping.load(std::memory_order_relaxed)), PublishedRanges::Capacity, wide(crowded.load(std::memory_order_relaxed)));
    }
}
}

namespace {
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawFinisher::FlowSite::Count)> flowSkips{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawFinisher::FlowLookup::Count)> flowLookups{};
std::atomic<std::uint64_t> flowTurnSkipsCrowded{0};
std::atomic<std::uint64_t> flowTurnSkipsUnknown{0};
std::atomic<std::uint64_t> flowTurnWaitsKept{0};
std::atomic<std::uint64_t> ringDepthHighest{0};
std::atomic<std::uint64_t> ringDepthSum{0};
std::atomic<std::uint64_t> ringDepthSamples{0};

void noteRingDepth(std::uint64_t depth) {
    ringDepthSum.fetch_add(depth, std::memory_order_relaxed);
    ringDepthSamples.fetch_add(1, std::memory_order_relaxed);
    if (depth > ringDepthHighest.load(std::memory_order_relaxed)) ringDepthHighest.store(depth, std::memory_order_relaxed);
}
}

namespace {
constexpr std::array<const char*, static_cast<std::size_t>(DeepKept::Count)> DeepKeptNames{"flip", "rendering wait", "wait", "worker label", "packet verdict", "segment end", "submission", "settle", "deferred labels", "worker draw", "worker dispatch", "metadata pass", "worker note", "relocation", "indirect cpu", "dispatch not pipelined", "fallback", "shutdown", "other", "note: dispatch not a candidate", "note: multisampled clear", "note: memory value fill", "note: HTILE depth clear", "note: HTILE mask clear", "note: HTILE zmask fix-up", "note: depth render control", "worker dispatch: no entry or reads", "worker dispatch: writes or recipe", "worker dispatch: not plain", "worker dispatch: late item refused", "worker dispatch: copy kernel"};
constexpr std::array<const char*, static_cast<std::size_t>(DeepSite::Count)> DeepSiteNames{"packet verdict", "predication", "indirect arguments", "fill inputs", "fill by turn", "dispatch reads", "late dispatch reads", "decode", "heap stages", "variant", "write evidence", "shader code", "same key", "earlier key", "items by turn", "items with reads by turn", "packet finisher"};
static_assert(DeepKeptNames.back() != nullptr && DeepSiteNames.back() != nullptr, "every deep flow census entry needs a name");
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DeepKept::Count)> deepKept{};
std::array<std::atomic<std::uint64_t>, 257> deepKeptVerdicts{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DeepSite::Count)> deepSites{};
std::atomic<std::uint64_t> deepDepthHighest{0};
std::atomic<std::uint64_t> deepDepthSum{0};
std::atomic<std::uint64_t> deepDepthSamples{0};

void noteDeepDepth(std::uint64_t depth) {
    deepDepthSum.fetch_add(depth, std::memory_order_relaxed);
    deepDepthSamples.fetch_add(1, std::memory_order_relaxed);
    if (depth > deepDepthHighest.load(std::memory_order_relaxed)) deepDepthHighest.store(depth, std::memory_order_relaxed);
}
}

namespace {
constexpr std::array<const char*, static_cast<std::size_t>(FrameKept::Count)> FrameKeptNames{"not latched", "ab phase", "sync flip", "timing log", "predicated", "label form", "deferred labels", "full loop segment end", "full loop settle", "front change", "exception", "indirect gpu", "index inconsistent", "dma copy", "dma to gds", "dma refused", "unknown chain limit", "exact index unsure", "worker dma, sample dump or note behind a flip", "suspend: not latched", "suspend: timing log", "suspend: drain asked", "suspend: deferred labels", "suspend: unordered completion", "indirect: not latched", "indirect: predicated", "indirect: no address", "indirect: deferred labels", "indirect: dispatch items off", "indirect: not a candidate", "indirect: not plain", "indirect: unbounded writes", "indirect: no entry, reads or recipe", "indirect: dispatch note", "indirect: argument wait fell back", "held draw: no decode", "held draw: stage count", "held draw: unbounded writes", "held draw: scratch writes", "held draw: BDA stores", "held draw: descriptor heap", "held draw: walk failed", "held draw: walk incomplete", "held draw: walk limit", "worker dma, sample dump or note behind an indirect dispatch", "held draw: program not prepared", "held draw: written image in a table", "held draw: descriptor behind a queued write", "held draw: descriptor behind GPU work", "held draw: walk read through a sync", "held draw: walk reads an indirect record word"};
constexpr std::array<const char*, static_cast<std::size_t>(FrameSite::Count)> FrameSiteNames{"ordered flip", "rendering wait", "segment end", "exit wait", "exit free", "argument wait", "write scan", "key scan", "queued scan", "dma beside", "register list wait", "resolved unknown jobs passed", "resolve wait", "suspend ordered", "indirect queued", "indirect argument wait", "held writes named", "exit owned"};
static_assert(FrameKeptNames.back() != nullptr && FrameSiteNames.back() != nullptr, "every frame flow census entry needs a name");
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FrameKept::Count)> frameKept{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FrameSite::Count)> frameSites{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FrameSite::Count)> frameSiteNs{};
std::atomic<std::uint64_t> flipDistances{0};
std::atomic<std::uint64_t> flipJobsSum{0};
std::atomic<std::uint64_t> flipJobsHighest{0};
std::atomic<std::uint64_t> flipItemsSum{0};
std::atomic<std::uint64_t> flipItemsHighest{0};
std::atomic<std::uint64_t> flipsInFlightSum{0};
std::atomic<std::uint64_t> flipsInFlightHighest{0};
std::atomic<std::uint64_t> heldNamedTotal{0};
std::atomic<std::uint64_t> heldUnknownTotal{0};
std::atomic<std::uint64_t> heldNamedWindow{0};
std::atomic<std::uint64_t> heldUnknownWindow{0};
std::atomic<std::uint64_t> heldRangesWindow{0};
std::atomic<std::uint64_t> heldNamingNs{0};
std::atomic<std::uint64_t> heldWalksNamed{0};
std::atomic<std::uint64_t> heldWalkNs{0};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FrameKept::Count)> heldWalksKept{};

void noteFrameHighest(std::atomic<std::uint64_t>& highest, std::uint64_t value) {
    if (value > highest.load(std::memory_order_relaxed)) highest.store(value, std::memory_order_relaxed);
}

constexpr std::array<const char*, static_cast<std::size_t>(FrameCheck::Count)> FrameCheckNames{"flip order", "item order", "lifetime", "flip state", "dispatch arguments", "fallback scan", "segment end", "ab phase", "register list", "dma order", "suspend order", "suspend completion", "indirect order", "indirect arguments", "held writes"};
static_assert(FrameCheckNames.back() != nullptr, "every frame flow check needs a name");
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FrameCheck::Count)> frameCheckOrdinals{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FrameCheck::Count)> frameChecked{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FrameCheck::Count)> frameDiffering{};
std::atomic<std::uint64_t> frameVerifyPrinted{0};
std::atomic<std::int64_t> frameVerifyWindow{0};

constexpr std::size_t TailKeptReasons = static_cast<std::size_t>(DeepKept::Count);
constexpr std::size_t TailSiteReasons = static_cast<std::size_t>(DeepSite::Count);
enum class TailExtra : std::uint32_t { RingRoom, QueueRoom, Exit, Resolve, Other, RenderingWait, RewindTail, NoSubmission, Count };
constexpr std::array<const char*, static_cast<std::size_t>(TailExtra::Count)> TailExtraNames{"finisher ring room", "committer queue room", "exit", "unknown writes resolving", "other", "rendering wait", "rewind tail release", "no submission"};
static_assert(TailExtraNames.back() != nullptr, "every tail wait reason needs a name");
constexpr std::size_t TailReasons = TailKeptReasons + TailSiteReasons + static_cast<std::size_t>(TailExtra::Count);
std::array<std::atomic<std::uint64_t>, TailReasons> tailWaits{};
std::array<std::atomic<std::uint64_t>, TailReasons> tailWaitNs{};
struct TailReasonSlot {};
using TailReason = HostThreadSlot<std::uint32_t, TailReasonSlot>;

std::uint32_t tailIndex(TailExtra extra) {
    return static_cast<std::uint32_t>(TailKeptReasons + TailSiteReasons + static_cast<std::size_t>(extra));
}

class TailWatch {
public:
    explicit TailWatch(bool active, std::optional<TailExtra> fixed = std::nullopt) {
        if (!active || !pipelineStats() || !FinisherProducer::Get() || FinisherThread::Get()) return;
        const auto& finisher = DrawFinisher::Get();
        if (finisher.FlipsQueued() == finisher.FlipsCommitted()) return;
        const auto held = TailReason::Get();
        reason = fixed.has_value() ? tailIndex(*fixed) : held != 0 && held <= TailReasons ? held - 1u : tailIndex(TailExtra::Other);
        start = std::chrono::steady_clock::now();
    }
    ~TailWatch() {
        if (start == std::chrono::steady_clock::time_point{}) return;
        tailWaits[reason].fetch_add(1, std::memory_order_relaxed);
        tailWaitNs[reason].fetch_add(elapsedNs(start), std::memory_order_relaxed);
    }
    TailWatch(const TailWatch&) = delete;
    TailWatch& operator=(const TailWatch&) = delete;

private:
    std::chrono::steady_clock::time_point start{};
    std::uint32_t reason = 0;
};

std::mutex rawReadersMutex;
std::vector<std::pair<const void*, std::uint64_t>> rawReaders;
std::mutex flipJobsMutex;
std::deque<std::uint64_t> flipJobs;
std::atomic<std::uint32_t> flipJobsHeld{0};

void frameStateRead(Graphics::FrameState state) {
    if (flipJobsHeld.load(std::memory_order_acquire) == 0 || GuestMemory::OnProofThread()) return;
    std::uint64_t oldest = 0;
    {
        const std::lock_guard lock(flipJobsMutex);
        if (flipJobs.empty()) return;
        oldest = flipJobs.front();
    }
    const char* what = state == Graphics::FrameState::FlipSerial ? "the texture flip serial" : "the import frame floor";
    if (FinisherThread::Get()) {
        const auto seq = FinisherSeq::Get();
        const bool later = seq != 0 && seq >= oldest;
        DrawFinisher::NoteFrameCheck(FrameCheck::FlipState, later, later ? "finisher job " + std::to_string(seq) + " read " + what + " before the flip queued ahead of job " + std::to_string(oldest) + " was committed" : std::string());
        return;
    }
    if (FinisherProducer::Get()) {
        DrawFinisher::NoteFrameCheck(FrameCheck::FlipState, true, std::string("the queue worker read ") + what + " before the flip queued ahead of job " + std::to_string(oldest) + " was committed");
        return;
    }
    DrawFinisher::NoteFrameCheck(FrameCheck::FlipState, false, {});
}
}

DrawFinisher::TailScope::TailScope(FrameTail cause) {
    if (!pipelineStats() || !FinisherProducer::Get() || FinisherThread::Get()) return;
    const auto& finisher = Get();
    if (finisher.FlipsQueued() == finisher.FlipsCommitted()) return;
    reason = tailIndex(cause == FrameTail::RenderingWait ? TailExtra::RenderingWait : cause == FrameTail::RewindTail ? TailExtra::RewindTail : TailExtra::NoSubmission);
    start = std::chrono::steady_clock::now();
}

DrawFinisher::TailScope::~TailScope() {
    if (start == std::chrono::steady_clock::time_point{}) return;
    tailWaits[reason].fetch_add(1, std::memory_order_relaxed);
    tailWaitNs[reason].fetch_add(elapsedNs(start), std::memory_order_relaxed);
}

bool DrawFinisher::aheadWait(std::span<const Range> ranges) {
    if (!FinisherAhead::Get() || !AheadBusy()) return false;
    if (deepRing.load(std::memory_order_acquire)) {
        const auto seq = FinisherSeq::Get();
        const auto front = frontSeq.load(std::memory_order_acquire);
        const auto found = deepWrites->Before(ranges, front, seq);
        if (found.answer == DeepAnswer::Unsure) {
            noteDeepLookup(DeepLookup::AheadWrites, true);
            noteDeepTotal(DeepCount::IndexFull);
            if (pipelineStats()) deepTurnFull.fetch_add(1, std::memory_order_relaxed);
            noteTurnWait(PublishedRanges::Answer::Unsure, false, ranges, {});
            LeanFront::Add(LeanFront::Counter::AheadWaits);
            AwaitTurn();
            return false;
        }
        noteDeepLookup(DeepLookup::AheadWrites, false);
        const auto unknown = leanSlot(seq).unknownBefore;
        const bool byUnknown = unknown >= front && (found.answer != DeepAnswer::Hit || unknown >= found.stamp);
        const auto blocker = byUnknown ? unknown : found.answer == DeepAnswer::Hit ? found.stamp : std::uint64_t{0};
        if (deepVerifyDue(DeepLookup::AheadWrites)) {
            const std::lock_guard lock(verifySlots);
            noteDeepVerify(DeepLookup::AheadWrites, blocker == 0 ? DeepFound{} : DeepFound{DeepAnswer::Hit, blocker}, scanWrites(ranges, frontSeq.load(std::memory_order_acquire), seq), true, front, seq, ranges);
        }
        if (blocker == 0) return false;
        if (pipelineStats()) {
            if (byUnknown) {
                deepTurnUnknown.fetch_add(1, std::memory_order_relaxed);
            } else if (found.conservative) {
                deepTurnConservative.fetch_add(1, std::memory_order_relaxed);
            } else {
                deepTurnWrites.fetch_add(1, std::memory_order_relaxed);
                deepTurnSkipped.fetch_add(seq - 1 - blocker, std::memory_order_relaxed);
            }
        }
        noteTurnWait(PublishedRanges::Answer::Hit, byUnknown, ranges, found.range);
        LeanFront::Add(LeanFront::Counter::AheadWaits);
        if (blocker + 1 >= seq) AwaitTurn();
        else awaitFrontPast(blocker);
        return false;
    }
    Range pendingHit{};
    bool unknownHeld = false;
    auto answer = published.Overlaps(ranges, &pendingHit, &unknownHeld);
    if (answer == PublishedRanges::Answer::Clear) return false;
    if (Flow()) {
        if (unknownHeld) answer = published.Overlaps(ranges, &pendingHit, nullptr, true);
        if (answer != PublishedRanges::Answer::Hit) {
            (answer == PublishedRanges::Answer::Unsure ? flowTurnSkipsCrowded : flowTurnSkipsUnknown).fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        unknownHeld = false;
        flowTurnWaitsKept.fetch_add(1, std::memory_order_relaxed);
    }
    if (unknownHeld && ringMode.load(std::memory_order_acquire) && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::TurnWaitsByJob)) {
        const auto seq = FinisherSeq::Get();
        std::uint64_t blocker = 0;
        const bool blocked = published.UnknownAhead(frontSeq.load(std::memory_order_acquire), seq, blocker);
        if (!blocked || blocker + 1 < seq) {
            if (blocked) {
                noteTurnWait(PublishedRanges::Answer::Hit, true, ranges, pendingHit);
                LeanFront::Add(LeanFront::Counter::AheadWaits);
                awaitFrontPast(blocker);
            }
            unknownHeld = false;
            answer = published.Overlaps(ranges, &pendingHit, nullptr, true);
            if (answer == PublishedRanges::Answer::Clear) return false;
        }
    }
    noteTurnWait(answer, unknownHeld, ranges, pendingHit);
    LeanFront::Add(LeanFront::Counter::AheadWaits);
    AwaitTurn();
    return false;
}

namespace {
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawFinisher::DrainSite::Count)> finisherDrains{}, finisherSkips{};
}

void DrawFinisher::NoteDrain(DrainSite site, bool drained) {
    (drained ? finisherDrains : finisherSkips)[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
}

namespace {
std::array<std::atomic<std::uint64_t>, 257> drainedOpcodes{}, drainedBeside{};
}

void DrawFinisher::NoteDrainedOpcode(std::uint32_t opcode, bool besideCommits) {
    drainedOpcodes[std::min<std::uint32_t>(opcode, 256u)].fetch_add(1, std::memory_order_relaxed);
    if (besideCommits) drainedBeside[std::min<std::uint32_t>(opcode, 256u)].fetch_add(1, std::memory_order_relaxed);
}

std::string DrawFinisher::DrainText() {
    constexpr std::array<const char*, static_cast<std::size_t>(DrainSite::Count)> names{"lookup", "reads", "packet", "kept draw"};
    std::string text = "; worker drains of the finisher (drained/skipped):";
    for (std::size_t i = 0; i < names.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu/%llu", names[i], static_cast<unsigned long long>(finisherDrains[i].exchange(0)), static_cast<unsigned long long>(finisherSkips[i].exchange(0)));
        text += item;
    }
    text += "; drained by opcode (of them beside commits):";
    for (int shown = 0; shown < 6; ++shown) {
        std::size_t top = 0;
        for (std::size_t i = 1; i < drainedOpcodes.size(); ++i) {
            if (drainedOpcodes[i].load() > drainedOpcodes[top].load()) top = i;
        }
        const auto count = drainedOpcodes[top].exchange(0);
        if (count == 0) break;
        char item[48];
        std::snprintf(item, sizeof(item), " 0x%02zx x%llu (%llu)", top, static_cast<unsigned long long>(count), static_cast<unsigned long long>(drainedBeside[top].exchange(0)));
        text += item;
    }
    return text;
}

bool DrawFinisher::Flow() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpFlow);
}

bool DrawFinisher::FlowSkips(FlowSite site) {
    if (!Flow()) return false;
    flowSkips[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
    return true;
}

void DrawFinisher::NoteFlowLookup(FlowLookup lookup) {
    flowLookups[static_cast<std::size_t>(lookup)].fetch_add(1, std::memory_order_relaxed);
}

std::string DrawFinisher::FlowText() {
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto samples = take(ringDepthSamples);
    const auto sum = take(ringDepthSum);
    const auto highest = take(ringDepthHighest);
    std::array<unsigned long long, static_cast<std::size_t>(FlowSite::Count)> skipped{};
    for (std::size_t index = 0; index < skipped.size(); ++index) skipped[index] = take(flowSkips[index]);
    std::array<unsigned long long, static_cast<std::size_t>(FlowLookup::Count)> unscanned{};
    for (std::size_t index = 0; index < unscanned.size(); ++index) unscanned[index] = take(flowLookups[index]);
    const auto crowded = take(flowTurnSkipsCrowded);
    const auto unknown = take(flowTurnSkipsUnknown);
    const auto kept = take(flowTurnWaitsKept);
    const auto site = [&](FlowSite at) { return skipped[static_cast<std::size_t>(at)]; };
    const auto lookup = [&](FlowLookup at) { return unscanned[static_cast<std::size_t>(at)]; };
    char text[1024];
    std::snprintf(text, sizeof(text), "switch %s, ring span %llu slots; outstanding finisher jobs at each publish: highest %llu, mean %.1f over %llu jobs; drains skipped (capture ones on finisher jobs too): packet finisher %llu, packet committer %llu, predication %llu, dispatch %llu, dispatch reads %llu, indirect arguments %llu, relocation %llu, capture %llu; lookups answered without a scan: same key %llu, earlier key %llu, pending writes %llu, queued writes %llu, held reads %llu, write watches %llu; turn waits of jobs ahead skipped: crowded write list %llu, unknown writes %llu; kept for a published write over the read %llu",
        Flow() ? "set" : "unset", static_cast<unsigned long long>(Get().leanMask.load(std::memory_order_relaxed) + 1), highest, samples == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(samples), samples,
        site(FlowSite::PacketFinisher), site(FlowSite::PacketCommitter), site(FlowSite::Predication), site(FlowSite::Dispatch), site(FlowSite::DispatchReads), site(FlowSite::Indirect), site(FlowSite::Relocation), site(FlowSite::Capture),
        lookup(FlowLookup::SameKey), lookup(FlowLookup::EarlierKey), lookup(FlowLookup::PendingWrites), lookup(FlowLookup::QueuedWrites), lookup(FlowLookup::HeldReads), lookup(FlowLookup::Watch), crowded, unknown, kept);
    return text;
}

bool DrawFinisher::Deep() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DeepFlow) && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpFlow) && deepQueues();
}

void DrawFinisher::NoteDeepKept(DeepKept kept, std::uint32_t opcode) {
    if (!pipelineStats() || !Deep()) return;
    TailReason::Set(static_cast<std::uint32_t>(kept) + 1u);
    if (!Busy() && DrawPipeline::Queue0().Outstanding() == 0) return;
    deepKept[static_cast<std::size_t>(kept)].fetch_add(1, std::memory_order_relaxed);
    if (kept == DeepKept::PacketVerdict) deepKeptVerdicts[std::min<std::uint32_t>(opcode, 256u)].fetch_add(1, std::memory_order_relaxed);
}

void DrawFinisher::NoteDeepSite(DeepSite site) {
    if (!pipelineStats() || !Deep()) return;
    if (site != DeepSite::ItemsByTurn && site != DeepSite::ItemsWithReadsByTurn && site != DeepSite::FillByTurn) TailReason::Set(static_cast<std::uint32_t>(TailKeptReasons + static_cast<std::size_t>(site)) + 1u);
    deepSites[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
}

std::string DrawFinisher::DeepText() {
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto samples = take(deepDepthSamples);
    const auto sum = take(deepDepthSum);
    const auto highest = take(deepDepthHighest);
    char head[320];
    std::snprintf(head, sizeof(head), "switch %s, ring latch %s, ring span %llu slots; outstanding finisher jobs at a publish: highest %llu, mean %.1f over %llu jobs",
        Deep() ? "set" : "unset", Get().deepRing.load(std::memory_order_relaxed) ? "on" : "off", static_cast<unsigned long long>(Get().leanMask.load(std::memory_order_relaxed) + 1), highest, samples == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(samples), samples);
    std::string text = head;
    const auto list = [&](const char* title, auto& counts, const auto& names) {
        text += title;
        for (std::size_t index = 0; index < counts.size(); ++index) {
            char item[96];
            std::snprintf(item, sizeof(item), "%s %s %llu", index == 0 ? "" : ",", names[index], take(counts[index]));
            text += item;
        }
    };
    list("; drains kept:", deepKept, DeepKeptNames);
    text += "; kept packet verdicts by opcode:";
    bool shown = false;
    for (int round = 0; round < 6; ++round) {
        std::size_t top = 0;
        for (std::size_t index = 1; index < deepKeptVerdicts.size(); ++index) {
            if (deepKeptVerdicts[index].load(std::memory_order_relaxed) > deepKeptVerdicts[top].load(std::memory_order_relaxed)) top = index;
        }
        const auto count = take(deepKeptVerdicts[top]);
        if (count == 0) break;
        char item[48];
        std::snprintf(item, sizeof(item), " 0x%02zx x%llu", top, count);
        text += item;
        shown = true;
    }
    if (!shown) text += " none";
    for (auto& count : deepKeptVerdicts) count.store(0, std::memory_order_relaxed);
    list("; ordered or awaited instead of drained:", deepSites, DeepSiteNames);
    const auto writes = take(deepTurnWrites);
    const auto skipped = take(deepTurnSkipped);
    char turns[320];
    std::snprintf(turns, sizeof(turns), "; turn waits of jobs ahead: a write over the read %llu (%.1f jobs skipped on average), conservative turn %llu, index full %llu, unknown writes %llu", writes, writes == 0 ? 0.0 : static_cast<double>(skipped) / static_cast<double>(writes), take(deepTurnConservative), take(deepTurnFull), take(deepTurnUnknown));
    text += turns;
    text += "; lookups answered by the index (fallbacks to the scan):";
    for (std::size_t index = 0; index < deepAnswered.size(); ++index) {
        char item[96];
        std::snprintf(item, sizeof(item), "%s %s %llu (%llu)", index == 0 ? "" : ",", DeepLookupNames[index], take(deepAnswered[index]), take(deepFellBack[index]));
        text += item;
    }
    text += "; targeted waits by site (finisher jobs/committer items):";
    for (std::size_t index = 0; index < deepSiteJobWaits.size(); ++index) {
        char item[96];
        std::snprintf(item, sizeof(item), "%s %s %llu/%llu", index == 0 ? "" : ",", DeepSiteNames[index], take(deepSiteJobWaits[index]), take(deepSiteItemWaits[index]));
        text += item;
    }
    const std::array<std::uint64_t, 4> totals{DeepTotal(DeepCount::WriteOverflows), DeepTotal(DeepCount::KeyOverflows), DeepTotal(DeepCount::QueuedOverflows), DeepTotal(DeepCount::WorkerFallbacks)};
    std::array<unsigned long long, 4> since{};
    for (std::size_t index = 0; index < totals.size(); ++index) {
        since[index] = static_cast<unsigned long long>(totals[index] - deepTotalsReported[index]);
        deepTotalsReported[index] = totals[index];
    }
    char tail[320];
    std::snprintf(tail, sizeof(tail), "; index overflows: finisher writes %llu, keys %llu, queued writes %llu; worker lookups scanned for an unsure index %llu; verify every %llu: checked %llu, differing %llu, conservative %llu", since[0], since[1], since[2], since[3], static_cast<unsigned long long>(deepVerifyEvery()), take(deepVerifyChecked), take(deepVerifyDiffering), take(deepVerifyConservative));
    text += tail;
    return text;
}

bool DrawFinisher::FrameFlowWanted() {
    return Deep() && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FrameFlow);
}

bool DrawFinisher::FrameFlow() {
    const auto& finisher = Get();
    return finisher.frameLatch.load(std::memory_order_acquire) && finisher.frameGeneration.load(std::memory_order_relaxed) == Graphics::RuntimeSwitches::Generation() && FrameFlowWanted();
}

bool DrawFinisher::OrderedSyncsWanted() {
    return FrameFlowWanted() && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::OrderedSyncs);
}

bool DrawFinisher::OrderedSyncs() {
    return FrameFlow() && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::OrderedSyncs);
}

void DrawFinisher::NoteHeldNaming(bool named, std::size_t ranges, std::uint64_t ns) {
    (named ? heldNamedTotal : heldUnknownTotal).fetch_add(1, std::memory_order_relaxed);
    if (!pipelineStats()) return;
    (named ? heldNamedWindow : heldUnknownWindow).fetch_add(1, std::memory_order_relaxed);
    if (named) heldRangesWindow.fetch_add(ranges, std::memory_order_relaxed);
    heldNamingNs.fetch_add(ns, std::memory_order_relaxed);
}

void DrawFinisher::NoteHeldWalk(bool named, FrameKept reason, std::uint64_t ns) {
    if (!pipelineStats()) return;
    if (named) heldWalksNamed.fetch_add(1, std::memory_order_relaxed);
    else if (reason < FrameKept::Count) heldWalksKept[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    heldWalkNs.fetch_add(ns, std::memory_order_relaxed);
}

std::uint64_t DrawFinisher::HeldNamedTotal() {
    return heldNamedTotal.load(std::memory_order_relaxed);
}

std::uint64_t DrawFinisher::HeldUnknownTotal() {
    return heldUnknownTotal.load(std::memory_order_relaxed);
}

bool DrawFinisher::FrameCensus() {
    return pipelineStats();
}

void DrawFinisher::LatchFrameFlow() {
    if (!FrameFlowWanted()) {
        if (frameLatch.load(std::memory_order_relaxed)) frameLatch.store(false, std::memory_order_release);
        return;
    }
    const auto generation = Graphics::RuntimeSwitches::Generation();
    if (frameLatch.load(std::memory_order_relaxed) && frameGeneration.load(std::memory_order_relaxed) == generation) return;
    FlushOpenRun();
    if (Busy() || DrawPipeline::Queue0().Outstanding() != 0) return;
    frameLatch.store(false, std::memory_order_relaxed);
    frameGeneration.store(generation, std::memory_order_relaxed);
    frameLatch.store(true, std::memory_order_release);
}

void DrawFinisher::NoteFrameKept(FrameKept kept) {
    if (!pipelineStats()) return;
    frameKept[static_cast<std::size_t>(kept)].fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t DrawFinisher::FrameVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_FRAME_FLOW");
        const std::uint64_t value = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
        if (value != 0) Graphics::SetFrameStateReader(&frameStateRead);
        return value;
    }();
    return every;
}

bool DrawFinisher::FrameVerifyDue(FrameCheck check) {
    const auto every = FrameVerifyEvery();
    return every != 0 && frameCheckOrdinals[static_cast<std::size_t>(check)].fetch_add(1, std::memory_order_relaxed) % every == 0;
}

void DrawFinisher::NoteFrameCheck(FrameCheck check, bool differs, const std::string& detail) {
    const auto index = static_cast<std::size_t>(check);
    frameChecked[index].fetch_add(1, std::memory_order_relaxed);
    if (!differs) return;
    frameDiffering[index].fetch_add(1, std::memory_order_relaxed);
    const auto window = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (frameVerifyWindow.exchange(window, std::memory_order_relaxed) != window) frameVerifyPrinted.store(0, std::memory_order_relaxed);
    if (frameVerifyPrinted.fetch_add(1, std::memory_order_relaxed) >= 8) return;
    std::fprintf(stderr, "[frame-flow-verify] %s differs: %s\n", FrameCheckNames[index], detail.c_str());
}

void DrawFinisher::NoteFlipJobs(std::uint64_t jobs) {
    if (FrameVerifyEvery() == 0) return;
    const std::lock_guard lock(flipJobsMutex);
    flipJobs.push_back(jobs);
    flipJobsHeld.store(static_cast<std::uint32_t>(flipJobs.size()), std::memory_order_release);
}

void DrawFinisher::RetireFlipJobs() {
    if (FrameVerifyEvery() == 0) return;
    const std::lock_guard lock(flipJobsMutex);
    if (!flipJobs.empty()) flipJobs.pop_front();
    flipJobsHeld.store(static_cast<std::uint32_t>(flipJobs.size()), std::memory_order_release);
}

DrawFinisher::RawReaderHold::RawReaderHold(const void* owner) : submission(owner) {
    const std::lock_guard lock(rawReadersMutex);
    const auto found = std::find_if(rawReaders.begin(), rawReaders.end(), [owner](const auto& entry) { return entry.first == owner; });
    if (found != rawReaders.end()) ++found->second;
    else rawReaders.emplace_back(owner, 1u);
}

DrawFinisher::RawReaderHold::~RawReaderHold() {
    Release();
}

void DrawFinisher::RawReaderHold::Release() {
    if (!held.exchange(false, std::memory_order_acq_rel)) return;
    const std::lock_guard lock(rawReadersMutex);
    const auto owner = submission;
    const auto found = std::find_if(rawReaders.begin(), rawReaders.end(), [owner](const auto& entry) { return entry.first == owner; });
    if (found == rawReaders.end()) return;
    if (--found->second != 0) return;
    *found = rawReaders.back();
    rawReaders.pop_back();
}

std::shared_ptr<DrawFinisher::RawReaderHold> DrawFinisher::HoldRawReader(const void* submission) {
    if (FrameVerifyEvery() == 0 || !FrameFlowWanted()) return nullptr;
    return std::make_shared<RawReaderHold>(submission);
}

void DrawFinisher::CheckRawReaders(const void* submission, const char* where) {
    if (FrameVerifyEvery() == 0) return;
    std::uint64_t holding = 0;
    {
        const std::lock_guard lock(rawReadersMutex);
        const auto found = std::find_if(rawReaders.begin(), rawReaders.end(), [submission](const auto& entry) { return entry.first == submission; });
        if (found != rawReaders.end()) holding = found->second;
    }
    NoteFrameCheck(FrameCheck::Lifetime, holding != 0, std::string(where) + ": " + std::to_string(holding) + " finisher jobs still hold a raw pointer to the submission");
}

std::string DrawFinisher::FrameText() {
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto ms = [](unsigned long long ns) { return static_cast<double>(ns) / 1e6; };
    auto& finisher = Get();
    std::array<unsigned long long, static_cast<std::size_t>(FrameSite::Count)> sites{};
    std::array<unsigned long long, static_cast<std::size_t>(FrameSite::Count)> siteNs{};
    for (std::size_t index = 0; index < sites.size(); ++index) {
        sites[index] = take(frameSites[index]);
        siteNs[index] = take(frameSiteNs[index]);
    }
    char head[320];
    std::snprintf(head, sizeof(head), "switch %s, latch %s; flips ordered %llu (%.1f ms queueing them on the worker), queued %llu and committed %llu in all",
        FrameFlowWanted() ? "set" : "unset", FrameFlow() ? "on" : "off", sites[0], ms(siteNs[0]), static_cast<unsigned long long>(finisher.FlipsQueued()), static_cast<unsigned long long>(finisher.FlipsCommitted()));
    std::string text = head;
    const auto namedHeld = take(heldNamedWindow);
    const auto unknownHeld = take(heldUnknownWindow);
    const auto namedRanges = take(heldRangesWindow);
    char syncs[256];
    std::snprintf(syncs, sizeof(syncs), "; ordered syncs %s, %s; held draws named %llu (%.1f ranges each), still unknown %llu, %.2f ms naming them on the worker",
        Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::OrderedSyncs) ? "set" : "unset", OrderedSyncs() ? "on" : "off", namedHeld, namedHeld == 0 ? 0.0 : static_cast<double>(namedRanges) / static_cast<double>(namedHeld), unknownHeld, ms(take(heldNamingNs)));
    text += syncs;
    const auto presents = sites[0];
    const auto perPresent = [&](unsigned long long count) { return presents == 0 ? 0.0 : static_cast<double>(count) / static_cast<double>(presents); };
    const auto walksNamed = take(heldWalksNamed);
    const auto walkNs = take(heldWalkNs);
    std::string walksKeptText;
    unsigned long long walksKept = 0;
    for (std::size_t index = 0; index < heldWalksKept.size(); ++index) {
        const auto count = take(heldWalksKept[index]);
        if (count == 0) continue;
        walksKept += count;
        char item[112];
        std::snprintf(item, sizeof(item), "%s %s %llu (%.2f)", walksKeptText.empty() ? "" : ",", FrameKeptNames[index], count, perPresent(count));
        walksKeptText += item;
    }
    char walks[320];
    std::snprintf(walks, sizeof(walks), "; held walks (per present over %llu ordered flips): named %llu (%.2f), kept unknown %llu (%.2f), %.2f ms walking on the worker (%.3f); walks kept by reason:",
        presents, walksNamed, perPresent(walksNamed), walksKept, perPresent(walksKept), ms(walkNs), presents == 0 ? 0.0 : ms(walkNs) / static_cast<double>(presents));
    text += walks;
    text += walksKeptText.empty() ? std::string(" none") : walksKeptText;
    text += "; kept drains by reason:";
    bool shown = false;
    for (std::size_t index = 0; index < frameKept.size(); ++index) {
        const auto count = take(frameKept[index]);
        if (count == 0) continue;
        char item[96];
        std::snprintf(item, sizeof(item), "%s %s %llu", shown ? "," : "", FrameKeptNames[index], count);
        text += item;
        shown = true;
    }
    if (!shown) text += " none";
    const auto distances = take(flipDistances);
    const auto mean = [&](unsigned long long sum) { return distances == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(distances); };
    const auto jobsSum = take(flipJobsSum);
    const auto itemsSum = take(flipItemsSum);
    const auto flightSum = take(flipsInFlightSum);
    char distance[320];
    std::snprintf(distance, sizeof(distance), "; at an ordered flip packet (%llu): finisher jobs outstanding mean %.1f highest %llu, committer items mean %.1f highest %llu, flips in flight mean %.2f highest %llu",
        distances, mean(jobsSum), take(flipJobsHighest), mean(itemsSum), take(flipItemsHighest), mean(flightSum), take(flipsInFlightHighest));
    text += distance;
    text += "; drains replaced (count, ms):";
    shown = false;
    for (std::size_t index = 1; index < sites.size(); ++index) {
        if (sites[index] == 0) continue;
        char item[112];
        std::snprintf(item, sizeof(item), "%s %s %llu (%.2f ms)", shown ? "," : "", FrameSiteNames[index], sites[index], ms(siteNs[index]));
        text += item;
        shown = true;
    }
    if (!shown) text += " none";
    text += "; worker waits with a flip in flight (count, ms):";
    shown = false;
    for (std::size_t index = 0; index < tailWaits.size(); ++index) {
        const auto count = take(tailWaits[index]);
        const auto ns = take(tailWaitNs[index]);
        if (count == 0) continue;
        const char* kind = index < TailKeptReasons ? "drain " : index < TailKeptReasons + TailSiteReasons ? "wait " : "";
        const char* name = index < TailKeptReasons ? DeepKeptNames[index] : index < TailKeptReasons + TailSiteReasons ? DeepSiteNames[index - TailKeptReasons] : TailExtraNames[index - TailKeptReasons - TailSiteReasons];
        char item[128];
        std::snprintf(item, sizeof(item), "%s %s%s %llu (%.2f ms)", shown ? "," : "", kind, name, count, ms(ns));
        text += item;
        shown = true;
    }
    if (!shown) text += " none";
    if (FrameVerifyEvery() != 0) {
        char every[64];
        std::snprintf(every, sizeof(every), "; verify every %llu (checked/differing):", static_cast<unsigned long long>(FrameVerifyEvery()));
        text += every;
        for (std::size_t index = 0; index < frameChecked.size(); ++index) {
            char item[96];
            std::snprintf(item, sizeof(item), "%s %s %llu/%llu", index == 0 ? "" : ",", FrameCheckNames[index], take(frameChecked[index]), take(frameDiffering[index]));
            text += item;
        }
    }
    return text;
}

void DrawFinisher::NoteFrameSite(FrameSite site, std::uint64_t ns) {
    if (!pipelineStats()) return;
    frameSites[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
    frameSiteNs[static_cast<std::size_t>(site)].fetch_add(ns, std::memory_order_relaxed);
}

void DrawFinisher::NoteFlipDistance(std::uint64_t jobs, std::uint64_t items, std::uint64_t inFlight) {
    if (!pipelineStats()) return;
    flipDistances.fetch_add(1, std::memory_order_relaxed);
    flipJobsSum.fetch_add(jobs, std::memory_order_relaxed);
    flipItemsSum.fetch_add(items, std::memory_order_relaxed);
    flipsInFlightSum.fetch_add(inFlight, std::memory_order_relaxed);
    noteFrameHighest(flipJobsHighest, jobs);
    noteFrameHighest(flipItemsHighest, items);
    noteFrameHighest(flipsInFlightHighest, inFlight);
}

void DrawFinisher::AwaitKeyed(std::uint64_t key, std::uint64_t shape, bool insertsOnly) {
    if (OnFinisherThread()) {
        NoteDeepSite(DeepSite::EarlierKey);
        const auto seq = FinisherSeq::Get();
        if (seq == 0 || !deepRing.load(std::memory_order_acquire)) {
            AwaitTurn();
            return;
        }
        const auto blocker = KeyBlocker(key, shape, insertsOnly);
        if (blocker == 0) return;
        noteDeepSiteWait(DeepSite::EarlierKey, true);
        if (blocker == ~std::uint64_t{0} || blocker + 1 >= seq) AwaitTurn();
        else awaitFrontPast(blocker);
        return;
    }
    NoteDeepSite(DeepSite::SameKey);
    FlushOpenRun();
    if (!Busy()) return;
    if (!FinisherProducer::Get() || !ringMode.load(std::memory_order_acquire) || !deepRing.load(std::memory_order_relaxed)) {
        Drain();
        return;
    }
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher drain under the GPU lock");
    retireLean();
    auto blocker = KeyBlocker(key, shape, insertsOnly);
    if (blocker == ~std::uint64_t{0} && FrameFlow()) blocker = keyScanBlocker(key, shape, insertsOnly);
    if (blocker == 0) return;
    noteDeepSiteWait(DeepSite::SameKey, true);
    if (blocker == ~std::uint64_t{0} || blocker >= nextSeq || !awaitProducerFront(blocker)) {
        if (blocker != ~std::uint64_t{0} && blocker >= nextSeq && FrameFlow()) NoteFrameKept(FrameKept::IndexInconsistent);
        NoteDeepKept(DeepKept::Fallback);
        Drain();
    }
}

std::uint64_t DrawFinisher::KeyBlocker(std::uint64_t key, std::uint64_t shape, bool insertsOnly) {
    if (!deepRing.load(std::memory_order_acquire)) return ~std::uint64_t{0};
    const auto seq = FinisherSeq::Get();
    const auto front = frontSeq.load(std::memory_order_acquire);
    if (seq != 0 && front >= seq) return 0;
    const auto found = keysBefore(key, shape, insertsOnly, front, seq != 0 ? seq : ~std::uint64_t{0});
    if (found.answer == DeepAnswer::Unsure) return ~std::uint64_t{0};
    return found.answer == DeepAnswer::Hit ? found.stamp : 0;
}

void DrawFinisher::AwaitPending(std::span<const Range> ranges, DeepSite site) {
    NoteDeepSite(site);
    FlushOpenRun();
    if (!Busy() || OnFinisherThread()) return;
    if (!FinisherProducer::Get() || !ringMode.load(std::memory_order_acquire) || !deepRing.load(std::memory_order_relaxed)) {
        Drain();
        return;
    }
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher drain under the GPU lock");
    for (;;) {
        retireLean();
        const auto front = frontSeq.load(std::memory_order_acquire);
        const auto found = deepWrites->Newest(ranges, front);
        std::uint64_t blocker = 0;
        bool scanned = false;
        if (found.answer == DeepAnswer::Unsure) {
            noteDeepTotal(DeepCount::WorkerFallbacks);
            if (!FrameFlow()) {
                NoteDeepKept(DeepKept::Fallback);
                Drain();
                return;
            }
            blocker = scanBlocker(ranges, FrameSite::WriteScan);
            scanned = true;
        } else {
            blocker = found.answer == DeepAnswer::Hit ? found.stamp : std::uint64_t{0};
            const auto unknown = lastUnknown.load(std::memory_order_relaxed);
            if (unknown >= front) blocker = std::max(blocker, unknown);
        }
        if (blocker == 0) return;
        noteDeepSiteWait(site, true);
        if (blocker >= nextSeq || !awaitProducerFront(blocker)) {
            if (blocker >= nextSeq && FrameFlow()) NoteFrameKept(FrameKept::IndexInconsistent);
            NoteDeepKept(DeepKept::Fallback);
            Drain();
            return;
        }
        if (scanned) verifyScanned(ranges, blocker);
    }
}

std::string& DrawFinisher::NamingTrace() {
    struct NamingTraceStorage {};
    return HostThreadLocal<std::string, NamingTraceStorage>();
}

bool& DrawFinisher::NamingWalk() {
    struct NamingWalkStorage {};
    return HostThreadLocal<bool, NamingWalkStorage>();
}

DrawFinisher::NamingRefused& DrawFinisher::NamingRefusal() {
    struct NamingRefusalStorage {};
    return HostThreadLocal<NamingRefused, NamingRefusalStorage>();
}

std::uint32_t DrawFinisher::describePendingWriter(std::uint64_t begin, std::uint64_t end) {
    const Range range{begin, end};
    if (deepRing.load(std::memory_order_relaxed)) {
        const auto front = frontSeq.load(std::memory_order_acquire);
        const auto found = deepWrites->Newest({&range, 1}, front);
        if (found.answer == DeepAnswer::Unsure) return 5000u;
        if (found.answer == DeepAnswer::Hit) {
            const auto& slot = leanSlot(found.stamp);
            static std::atomic<std::uint32_t> samples{0};
            if (samples.fetch_add(1, std::memory_order_relaxed) < 40u) {
                const auto& held = slot.pending;
                AgcDriver::ProfilePrint_nid_no_patch("[held-naming] sample: asked 0x%llx +%llu; writer seq %llu (front %llu, next %llu) origin %u kind %u, label 0x%llx +%zu bytes, label run %d, %zu write ranges, first [0x%llx, 0x%llx)\n",
                    static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end - begin), static_cast<unsigned long long>(found.stamp), static_cast<unsigned long long>(front), static_cast<unsigned long long>(nextSeq),
                    static_cast<unsigned>(held.origin), static_cast<unsigned>(held.originKind), static_cast<unsigned long long>(held.labelAddress), held.labelBytes.size(), held.labelRun != nullptr ? 1 : 0, held.writes.size(),
                    static_cast<unsigned long long>(held.writes.empty() ? 0 : held.writes.front().first), static_cast<unsigned long long>(held.writes.empty() ? 0 : held.writes.front().second));
            }
            return 1000u + static_cast<std::uint32_t>(slot.pending.origin) * 100u + slot.pending.originKind;
        }
        if (lastUnknown.load(std::memory_order_relaxed) >= front) return 2000u;
    }
    if (FinisherDeposits::Get() && depositsWrite({&range, 1})) return 3000u;
    return DrawPipeline::Queue0().DescribeQueuedWriter(begin, end);
}

void DrawFinisher::NoteEvidenceWait(std::uint64_t begin, std::uint64_t end, bool buffered) {
    if (!pipelineStats() || NamingWalk() || OnFinisherThread()) return;
    static std::mutex mutex;
    static std::map<std::uint64_t, std::uint64_t> waits;
    static std::uint64_t total = 0;
    const auto code = Get().describePendingWriter(begin, end);
    const auto bytes = end - begin;
    const std::uint64_t sizeClass = bytes == 4096 && (begin & 4095u) == 0 ? 3 : bytes > 64 ? 2 : buffered ? 0 : 1;
    std::lock_guard lock(mutex);
    if (total < 40) AgcDriver::ProfilePrint_nid_no_patch("[write-evidence] sample: asked 0x%llx +%llu (%s); blocker %u\n", static_cast<unsigned long long>(begin), static_cast<unsigned long long>(bytes), buffered ? "word" : "span", code);
    ++waits[(sizeClass << 32u) | code];
    if (++total % 512u != 0) return;
    std::string text;
    for (const auto& [key, count] : waits) {
        char item[64];
        std::snprintf(item, sizeof(item), " class %u blocker %u x%llu", static_cast<unsigned>(key >> 32u), static_cast<unsigned>(key & 0xffffffffu), static_cast<unsigned long long>(count));
        text += item;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[write-evidence] waits on the worker by size class (0 word, 1 span up to 64 bytes, 2 larger span, 3 a whole page) and blocker (as [held-naming]):%s\n", text.c_str());
    waits.clear();
}

void DrawFinisher::NoteNamingBlocker(std::uint64_t begin, std::uint64_t end) {
    if (!pipelineStats() || !NamingWalk()) return;
    static std::mutex mutex;
    static std::map<std::uint32_t, std::uint64_t> blockers;
    static std::uint64_t total = 0;
    const auto code = Get().describePendingWriter(begin, end);
    std::lock_guard lock(mutex);
    ++blockers[code];
    if (++total % 512u != 0) return;
    std::string text;
    for (const auto& [key, count] : blockers) {
        char item[48];
        std::snprintf(item, sizeof(item), " %u x%llu", key, static_cast<unsigned long long>(count));
        text += item;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[held-naming] blockers of the resource walk (1000 + origin * 100 + kind in the ring, 2000 a job with unknown writes, 3000 deposits, 4000 + kind queued, 5000 unsure, 6000 none):%s\n", text.c_str());
    blockers.clear();
}

std::uint32_t DrawPipeline::DescribeQueuedWriter(std::uint64_t begin, std::uint64_t end) {
    const QueueScan scan(*this, ringScanId(RingScan::QueuedWriter));
    const auto describe = [&](const QueuedView& queued) {
        for (std::size_t index = queued.size(); index-- > 0;) {
            const auto& item = queued[index];
            if (std::any_of(item.writes.begin(), item.writes.end(), [&](const Range& write) { return begin < write.second && write.first < end; })) return 4000u + static_cast<std::uint32_t>(item.kind);
        }
        return 6000u;
    };
    const auto code = describe(scan.Items());
    scan.Check("queued writer scan", describe);
    return code;
}

DrawFinisher::ExactFound DrawFinisher::exactPending(std::span<const Range> ranges) {
    constexpr std::uint32_t ResolveSteps = 64;
    ExactFound result;
    const auto front = frontSeq.load(std::memory_order_acquire);
    const auto found = deepWrites->Newest(ranges, front);
    if (found.answer == DeepAnswer::Unsure) {
        result.unsure = true;
        return result;
    }
    if (found.answer == DeepAnswer::Hit) result.blocker = found.stamp;
    const auto newest = lastUnknown.load(std::memory_order_relaxed);
    std::uint32_t steps = 0;
    for (auto unknown = newest; unknown != 0 && unknown >= front;) {
        if (++steps > ResolveSteps) {
            result.limited = true;
            result.blocker = std::max(result.blocker, newest);
            break;
        }
        const auto& slot = leanSlot(unknown);
        if (slot.resolved.load(std::memory_order_acquire) != 1) {
            if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpNoResolveWaits)) ++result.skipped;
            else result.unresolved = std::max(result.unresolved, unknown);
        }
        else if (rangesOverlap(slot.resolvedWrites, ranges)) result.blocker = std::max(result.blocker, unknown);
        else ++result.skipped;
        const auto before = slot.unknownBefore;
        if (before >= unknown) break;
        unknown = before;
    }
    return result;
}

bool DrawFinisher::awaitResolved(std::uint64_t job) {
    auto& slot = leanSlot(job);
    const auto done = [&] { return frontSeq.load(std::memory_order_seq_cst) > job || slot.resolved.load(std::memory_order_seq_cst) == 1; };
    if (done()) return true;
    const TailWatch watch(true, TailExtra::Resolve);
    const auto start = pipelineStats() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    {
        const ThreadReport::WorkerStateScope draining(ThreadReport::WorkerState::DrainFinisher, ThreadReport::Enabled() && ThreadReport::OnQueue0Worker());
        if (!SpinUntil(HandoffSpinWindow() * 2, done)) {
            resolveWaiters.fetch_add(1, std::memory_order_seq_cst);
            frontWaiters.fetch_add(1, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            {
                auto lock = lockFor(mutex);
                turnWake.wait(lock, [&] { return closing || done(); });
            }
            frontWaiters.fetch_sub(1, std::memory_order_relaxed);
            resolveWaiters.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    if (start != std::chrono::steady_clock::time_point{}) NoteFrameSite(FrameSite::ResolveWait, elapsedNs(start));
    retireLean();
    rethrowLeanFailure();
    return done();
}

void DrawFinisher::AwaitExactPending(std::span<const Range> ranges, DeepSite site, FrameSite timed) {
    if (!FrameFlow() || !FinisherProducer::Get() || OnFinisherThread() || !ringMode.load(std::memory_order_acquire) || !deepRing.load(std::memory_order_relaxed)) {
        AwaitPending(ranges, site);
        return;
    }
    NoteDeepSite(site);
    FlushOpenRun();
    if (!Busy()) return;
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher drain under the GPU lock");
    const auto start = FrameCensus() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    bool waited = false;
    bool passed = false;
    for (;;) {
        retireLean();
        const auto found = exactPending(ranges);
        if (found.unsure) {
            noteDeepTotal(DeepCount::WorkerFallbacks);
            const auto blocker = scanBlocker(ranges, FrameSite::WriteScan);
            if (blocker == 0) break;
            waited = true;
            noteDeepSiteWait(site, true);
            if (blocker >= nextSeq || !awaitProducerFront(blocker)) {
                if (blocker >= nextSeq) NoteFrameKept(FrameKept::IndexInconsistent);
                NoteDeepKept(DeepKept::Fallback);
                Drain();
                break;
            }
            verifyScanned(ranges, blocker);
            continue;
        }
        if (found.limited) NoteFrameKept(FrameKept::ResolveLimit);
        passed = passed || found.skipped != 0;
        if (found.blocker != 0) {
            waited = true;
            noteDeepSiteWait(site, true);
            if (found.blocker >= nextSeq || !awaitProducerFront(found.blocker)) {
                if (found.blocker >= nextSeq) NoteFrameKept(FrameKept::IndexInconsistent);
                NoteDeepKept(DeepKept::Fallback);
                Drain();
                break;
            }
            continue;
        }
        if (found.unresolved != 0) {
            waited = true;
            if (!awaitResolved(found.unresolved)) {
                NoteDeepKept(DeepKept::Fallback);
                Drain();
                break;
            }
            continue;
        }
        break;
    }
    if (passed) NoteFrameSite(FrameSite::ResolvedSkip);
    if (waited && start != std::chrono::steady_clock::time_point{}) NoteFrameSite(timed, elapsedNs(start));
}

std::uint64_t DrawFinisher::scanBlocker(std::span<const Range> ranges, FrameSite site) {
    const auto start = FrameCensus() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto front = frontSeq.load(std::memory_order_acquire);
    std::uint64_t blocker = 0;
    for (auto seq = front; seq < nextSeq; ++seq) {
        const auto& slot = leanSlot(seq);
        if (slot.pending.unknown || slot.batch || rangesOverlap(slot.pending.writes, ranges)) blocker = seq;
    }
    NoteFrameSite(site, start == std::chrono::steady_clock::time_point{} ? 0 : elapsedNs(start));
    return blocker;
}

std::uint64_t DrawFinisher::keyScanBlocker(std::uint64_t key, std::uint64_t shape, bool insertsOnly) {
    const auto start = FrameCensus() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto front = frontSeq.load(std::memory_order_acquire);
    std::uint64_t blocker = 0;
    for (auto seq = front; seq < nextSeq; ++seq) {
        const auto& slot = leanSlot(seq);
        const auto& held = slot.pending;
        if (slot.batch || ((held.key == PublishedKeys::AnyKey || held.key == key || (shape != 0 && held.shape == shape)) && (held.inserts || !insertsOnly))) blocker = seq;
    }
    NoteFrameSite(FrameSite::KeyScan, start == std::chrono::steady_clock::time_point{} ? 0 : elapsedNs(start));
    return blocker;
}

void DrawFinisher::verifyScanned(std::span<const Range> ranges, std::uint64_t blocker) {
    if (!FrameVerifyDue(FrameCheck::FallbackScan)) return;
    const auto front = frontSeq.load(std::memory_order_acquire);
    std::uint64_t again = 0;
    for (auto seq = front; seq < nextSeq; ++seq) {
        const auto& slot = leanSlot(seq);
        if (slot.pending.unknown || slot.batch || rangesOverlap(slot.pending.writes, ranges)) again = seq;
    }
    NoteFrameCheck(FrameCheck::FallbackScan, again != 0, "a scan after the wait for job " + std::to_string(blocker) + " still finds writer " + std::to_string(again) + " at front " + std::to_string(front));
}

std::uint64_t DrawFinisher::DeepTotal(DeepCount count) {
    auto& finisher = Get();
    auto& pipeline = DrawPipeline::Queue0();
    const bool ring = finisher.deepIndexed.load(std::memory_order_acquire);
    const bool queue = pipeline.deepIndexed.load(std::memory_order_acquire);
    switch (count) {
        case DeepCount::WriteOverflows: return ring ? finisher.deepWrites->Overflows() : 0;
        case DeepCount::KeyOverflows: return ring ? finisher.deepKeys->Overflows() + finisher.deepShapes->Overflows() : 0;
        case DeepCount::QueuedOverflows: return queue ? pipeline.deepStores->Overflows() + pipeline.deepOthers->Overflows() : 0;
        default: return deepTotals[std::min<std::size_t>(static_cast<std::size_t>(count), deepTotals.size() - 1)].load(std::memory_order_relaxed);
    }
}

DeepAnswer DrawFinisher::deepPendingHit(std::span<const Range> ranges) {
    const auto front = frontSeq.load(std::memory_order_acquire);
    const auto unknown = lastUnknown.load(std::memory_order_relaxed);
    auto found = deepWrites->Newest(ranges, front);
    if (found.answer == DeepAnswer::Unsure) {
        noteDeepTotal(DeepCount::WorkerFallbacks);
        noteDeepLookup(DeepLookup::PendingWrites, true);
        return unknown >= front ? DeepAnswer::Hit : DeepAnswer::Unsure;
    }
    if (unknown >= front && (found.answer != DeepAnswer::Hit || unknown > found.stamp)) found = {DeepAnswer::Hit, unknown};
    noteDeepLookup(DeepLookup::PendingWrites, false);
    if (deepVerifyDue(DeepLookup::PendingWrites)) {
        auto scan = scanWrites(ranges, frontSeq.load(std::memory_order_acquire), nextSeq);
        if (rangesOverlap(openRunWrites, ranges)) scan = {DeepAnswer::Hit, nextSeq};
        noteDeepVerify(DeepLookup::PendingWrites, found, scan, true, front, 0, ranges);
    }
    return found.answer;
}

DeepFound DrawFinisher::keysBefore(std::uint64_t key, std::uint64_t shape, bool insertsOnly, std::uint64_t live, std::uint64_t before) const {
    auto found = deepLater(deepKeys->BeforeAny(insertsOnly, live, before), deepKeys->Before(key, insertsOnly, live, before));
    if (shape != 0) found = deepLater(found, deepShapes->Before(shape, insertsOnly, live, before));
    return found;
}

DeepFound DrawFinisher::scanWrites(std::span<const Range> ranges, std::uint64_t from, std::uint64_t to) const {
    DeepFound scan;
    for (auto seq = from; seq < to; ++seq) {
        const auto& held = (*lean)[SlotIndex(seq)].pending;
        if (held.unknown || rangesOverlap(held.writes, ranges)) scan = {DeepAnswer::Hit, seq};
    }
    return scan;
}

DeepFound DrawFinisher::scanKeys(std::uint64_t key, std::uint64_t shape, bool insertsOnly, std::uint64_t from, std::uint64_t to) const {
    DeepFound scan;
    for (auto seq = from; seq < to; ++seq) {
        const auto& held = (*lean)[SlotIndex(seq)].pending;
        if ((held.key == PublishedKeys::AnyKey || held.key == key || (shape != 0 && held.shape == shape)) && (held.inserts || !insertsOnly)) scan = {DeepAnswer::Hit, seq};
    }
    return scan;
}

bool DrawFinisher::awaitProducerFront(std::uint64_t job) {
    const auto passed = [&] { return frontSeq.load(std::memory_order_seq_cst) > job; };
    if (passed()) return true;
    const TailWatch watch(true);
    const auto start = ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
    {
        const ThreadReport::WorkerStateScope draining(ThreadReport::WorkerState::DrainFinisher, start != ThreadReport::Clock::time_point{} && ThreadReport::OnQueue0Worker());
        if (!SpinUntil(HandoffSpinWindow() * 2, passed)) {
            frontWaiters.fetch_add(1, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            {
                auto lock = lockFor(mutex);
                turnWake.wait(lock, [&] { return closing || passed(); });
            }
            frontWaiters.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    if (start != ThreadReport::Clock::time_point{}) {
        const bool worker = ThreadReport::OnQueue0Worker();
        ThreadReport::Add(worker ? ThreadReport::Count::WorkerFinisherDrains : ThreadReport::Count::OtherDrains, 1);
        ThreadReport::Add(worker ? ThreadReport::Count::WorkerFinisherDrainNs : ThreadReport::Count::OtherDrainNs, ThreadReport::Nanoseconds(start, ThreadReport::Clock::now()));
    }
    retireLean();
    rethrowLeanFailure();
    return passed();
}

bool DrawFinisher::OnFinisherThread() {
    return FinisherThread::Get();
}

bool DrawFinisher::DeferRelease(std::array<std::shared_ptr<const void>, 3>& held) noexcept {
    if (!FinisherRunLoop::Get() || !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::IdleHitRelease)) return false;
    try {
        auto& retired = HostThreadLocal<RetiredHits, RetiredHitsStorage>();
        if (retired.count == RetiredHits::Cap) {
            DirectDraws::Add(DirectDraws::Counter::HitsReleasedAtCap);
            return false;
        }
        auto& slot = retired.held[(retired.head + retired.count) % RetiredHits::Cap];
        for (std::size_t index = 0; index < held.size(); ++index) slot[index] = std::move(held[index]);
        ++retired.count;
        FinisherRetired::Set(true);
        const std::uint64_t depth = retired.count;
        auto seen = deepestRetiredHits.load(std::memory_order_relaxed);
        while (seen < depth && !deepestRetiredHits.compare_exchange_weak(seen, depth, std::memory_order_relaxed)) {
        }
        return true;
    } catch (...) {
        return false;
    }
}

std::uint64_t DrawFinisher::TakeDeepestRelease() {
    return deepestRetiredHits.exchange(0, std::memory_order_relaxed);
}

bool DrawFinisher::releaseRetiredHits(bool stopForJobs) {
    if (!FinisherRetired::Get()) return true;
    auto& retired = HostThreadLocal<RetiredHits, RetiredHitsStorage>();
    const auto began = std::chrono::steady_clock::now();
    std::uint64_t released = 0;
    while (retired.count != 0 && !leanReady() && (!stopForJobs || untakenJobs.load(std::memory_order_acquire) == 0)) {
        auto entry = std::move(retired.held[retired.head]);
        retired.head = (retired.head + 1) % RetiredHits::Cap;
        --retired.count;
        for (auto& part : entry) part.reset();
        ++released;
    }
    if (released != 0) {
        DirectDraws::Add(DirectDraws::Counter::HitsReleasedIdle, released);
        DirectDraws::Add(DirectDraws::Counter::HitsReleaseNanoseconds, elapsedNs(began));
    }
    if (retired.count != 0) return false;
    FinisherRetired::Set(false);
    return true;
}

void DrawFinisher::releaseAllRetiredHits() {
    if (!FinisherRetired::Get()) return;
    auto& retired = HostThreadLocal<RetiredHits, RetiredHitsStorage>();
    const auto began = std::chrono::steady_clock::now();
    std::uint64_t released = 0;
    while (retired.count != 0) {
        auto entry = std::move(retired.held[retired.head]);
        retired.head = (retired.head + 1) % RetiredHits::Cap;
        --retired.count;
        for (auto& part : entry) part.reset();
        ++released;
    }
    FinisherRetired::Set(false);
    if (released == 0) return;
    DirectDraws::Add(DirectDraws::Counter::HitsReleasedIdle, released);
    DirectDraws::Add(DirectDraws::Counter::HitsReleaseNanoseconds, elapsedNs(began));
}

DrawFinisher::Slot* DrawFinisher::untaken() {
    for (auto& slot : jobs) {
        if (!slot.taken) return &slot;
    }
    return nullptr;
}

void DrawFinisher::AwaitTurn() {
    const auto seq = FinisherSeq::Get();
    if (seq == 0) return;
    if (frontSeq.load(std::memory_order_acquire) != seq) awaitTurn(seq);
    if (FinisherDeposits::Get()) forwardOwn(seq);
}

void DrawFinisher::awaitFrontPast(std::uint64_t job) {
    struct Counted {
        ThreadReport::Clock::time_point start;
        ~Counted() {
            if (start == ThreadReport::Clock::time_point{}) return;
            ThreadReport::Add(ThreadReport::Count::FinisherTurnWaits, 1);
            ThreadReport::Add(ThreadReport::Count::FinisherTurnNs, ThreadReport::Nanoseconds(start, ThreadReport::Clock::now()));
        }
    } counted{ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{}};
    const auto passed = [&] { return frontSeq.load(std::memory_order_seq_cst) > job; };
    if (spinHandoff() && SpinUntil(HandoffSpinWindow(), passed)) return;
    frontWaiters.fetch_add(1, std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    {
        auto lock = lockFor(mutex);
        turnWake.wait(lock, [&] { return closing || passed(); });
    }
    frontWaiters.fetch_sub(1, std::memory_order_relaxed);
}

void DrawFinisher::awaitTurn(std::uint64_t seq) {
    struct Counted {
        ThreadReport::Clock::time_point start;
        ~Counted() {
            if (start == ThreadReport::Clock::time_point{}) return;
            ThreadReport::Add(ThreadReport::Count::FinisherTurnWaits, 1);
            ThreadReport::Add(ThreadReport::Count::FinisherTurnNs, ThreadReport::Nanoseconds(start, ThreadReport::Clock::now()));
        }
    } counted{ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{}};
    if (spinHandoff() && SpinUntil(HandoffSpinWindow(), [&] { return frontSeq.load(std::memory_order_acquire) == seq; })) return;
    if (ringMode.load(std::memory_order_acquire)) {
        frontWaiters.fetch_add(1, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        {
            auto lock = lockFor(mutex);
            turnWake.wait(lock, [&] { return closing || frontSeq.load(std::memory_order_seq_cst) == seq; });
        }
        frontWaiters.fetch_sub(1, std::memory_order_relaxed);
        return;
    }
    auto lock = lockFor(mutex);
    ++turnSleepers;
    turnWake.wait(lock, [&] { return closing || jobs.empty() || jobs.front().seq == seq; });
    --turnSleepers;
}

unsigned DrawFinisher::IdleState() const {
    if (outstanding.load(std::memory_order_acquire) == 0) return 0;
    if (leanTake.load(std::memory_order_acquire) < leanTail.load(std::memory_order_acquire)) return parked.load(std::memory_order_acquire) != 0 || parkedSlots.load(std::memory_order_acquire) != 0 ? 1 : 2;
    return 3;
}

namespace {
constexpr std::size_t IdleStates = 4;
constexpr std::size_t IdleLengths = 4;
std::array<std::array<std::atomic<std::uint64_t>, IdleLengths>, IdleStates> committerIdles{}, committerIdleNs{};

void noteCommitterIdle(unsigned state, std::uint64_t ns) {
    const std::size_t length = ns < 20000u ? 0 : ns < 100000u ? 1 : ns < 500000u ? 2 : 3;
    committerIdles[state][length].fetch_add(1, std::memory_order_relaxed);
    committerIdleNs[state][length].fetch_add(ns, std::memory_order_relaxed);
}

std::string committerIdleText() {
    static constexpr const char* names[IdleStates] = {"finisher empty", "jobs untaken with a finisher parked", "jobs untaken with every finisher busy", "jobs in flight only"};
    std::string text;
    for (std::size_t state = 0; state < IdleStates; ++state) {
        std::uint64_t count = 0;
        std::uint64_t ns = 0;
        std::array<std::uint64_t, IdleLengths> counts{}, times{};
        for (std::size_t length = 0; length < IdleLengths; ++length) {
            counts[length] = committerIdles[state][length].exchange(0, std::memory_order_relaxed);
            times[length] = committerIdleNs[state][length].exchange(0, std::memory_order_relaxed);
            count += counts[length];
            ns += times[length];
        }
        char item[320];
        std::snprintf(item, sizeof(item), "%s %s %llu (%.1f ms; under 20 us %llu, %.1f ms; 20 to 100 us %llu, %.1f ms; 100 to 500 us %llu, %.1f ms; longer %llu, %.1f ms)", state == 0 ? "" : ";", names[state], static_cast<unsigned long long>(count), static_cast<double>(ns) / 1e6,
            static_cast<unsigned long long>(counts[0]), static_cast<double>(times[0]) / 1e6, static_cast<unsigned long long>(counts[1]), static_cast<double>(times[1]) / 1e6, static_cast<unsigned long long>(counts[2]), static_cast<double>(times[2]) / 1e6, static_cast<unsigned long long>(counts[3]), static_cast<double>(times[3]) / 1e6);
        text += item;
    }
    return text;
}
}

unsigned DrawFinisher::Finishers() {
    return runningFinishers.load(std::memory_order_acquire);
}

bool DrawFinisher::countPending(const Pending& held, bool added) {
    if (held.unknown) {
        const bool before = pendingUnknown != 0;
        pendingUnknown = added ? pendingUnknown + 1 : pendingUnknown - 1;
        return before != (pendingUnknown != 0);
    }
    bool changed = false;
    for (const auto& range : held.writes) changed = countRange(pendingWrites, range, added) || changed;
    return changed;
}

bool DrawFinisher::Busy() {
    return outstanding.load(std::memory_order_acquire) != 0 || (runOpen.load(std::memory_order_relaxed) && FinisherProducer::Get());
}

void DrawFinisher::FlushOpenRun() {
    if (!runOpen.load(std::memory_order_relaxed) || !FinisherProducer::Get()) return;
    if (const auto flush = std::exchange(openRun, nullptr); flush != nullptr) flush();
}

void DrawFinisher::HoldOpenRunWrite(const Range& range) {
    const Graphics::WriteValidation::RaiseScope raised(Graphics::WriteValidation::Class::Q, Graphics::WriteValidation::Source::RunWrite, range.first, Graphics::WriteValidation::FinisherSide() ? range.second : range.first);
    if (deepRing.load(std::memory_order_relaxed)) {
        deepWrites->Insert({&range, 1}, nextSeq, frontSeq.load(std::memory_order_acquire));
        openRunWrites.push_back(range);
        return;
    }
    if (countRange(pendingWrites, range, true)) published.Publish(pendingWrites, pendingUnknown != 0 ? 1u : 0u);
}

void DrawFinisher::Enqueue(Job job) {
    Enqueue(std::move(job), Pending{});
}

void DrawFinisher::Enqueue(Job job, Pending held) {
    FinisherProducer::Set(true);
    FlushOpenRun();
    if (ringMode.load(std::memory_order_acquire)) {
        rethrowLeanFailure();
        if (closed.load(std::memory_order_acquire)) {
            job();
            return;
        }
        awaitLeanRoom();
        if (closed.load(std::memory_order_acquire)) {
            job();
            return;
        }
        auto& slot = leanSlot(nextSeq);
        std::unique_lock<std::mutex> verifying;
        if (deepVerifyEvery() != 0) verifying = std::unique_lock<std::mutex>(verifySlots);
        slot.runner = nullptr;
        slot.retirer = nullptr;
        slot.batch = false;
        slot.answer = nullptr;
        slot.closure = std::move(job);
        slot.pending = std::move(held);
        publishLean(slot);
        return;
    }
    const auto depth = static_cast<std::size_t>(finisherDepth());
    struct ReclaimStorage {};
    auto& reclaim = HostThreadLocal<std::vector<Job>, ReclaimStorage>();
    const ClearOnExit<Job> clearReclaim{reclaim};
    auto lock = lockFor(mutex);
    if (!ran.empty()) std::swap(ran, reclaim);
    if (failure != nullptr) std::rethrow_exception(std::exchange(failure, nullptr));
    if (closing) {
        lock.unlock();
        job();
        return;
    }
    if (jobs.size() >= depth && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher full while the enqueuing thread holds the GPU lock");
    const TailWatch watch(jobs.size() >= depth, TailExtra::RingRoom);
    const auto roomStart = jobs.size() >= depth && ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
    std::optional<ThreadReport::WorkerStateScope> room;
    if (roomStart != ThreadReport::Clock::time_point{} && ThreadReport::OnQueue0Worker()) room.emplace(ThreadReport::WorkerState::RoomFinisher);
    ++idleWaiters;
    idle.wait(lock, [&] { return jobs.size() < depth; });
    --idleWaiters;
    room.reset();
    if (roomStart != ThreadReport::Clock::time_point{}) {
        const bool worker = ThreadReport::OnQueue0Worker();
        ThreadReport::Add(worker ? ThreadReport::Count::WorkerFinisherRoomWaits : ThreadReport::Count::OtherRoomWaits, 1);
        ThreadReport::Add(worker ? ThreadReport::Count::WorkerFinisherRoomNs : ThreadReport::Count::OtherRoomNs, ThreadReport::Nanoseconds(roomStart, ThreadReport::Clock::now()));
    }
    jobs.push_back({std::move(job), nextSeq++});
    issued.store(nextSeq, std::memory_order_relaxed);
    frontSeq.store(jobs.front().seq, std::memory_order_release);
    untakenJobs.fetch_add(1, std::memory_order_release);
    if (!held.unknown) raiseQueued(Graphics::WriteValidation::Source::PendingJob, held.writes);
    if (countPending(held, true)) published.Publish(pendingWrites, pendingUnknown != 0 ? 1u : 0u);
    keys.Push(jobs.back().seq, held.key, held.shape, held.inserts);
    pending.push_back(std::move(held));
    if (!pending.back().unknown) raiseQueued(Graphics::WriteValidation::Source::PendingJob, pending.back().writes);
    outstanding.fetch_add(1, std::memory_order_release);
    startFinishers();
    if (const auto asleep = dormant.load(std::memory_order_acquire); asleep != 0) {
        for (unsigned index = 1; index <= extra.size(); ++index) {
            if ((asleep & (1u << index)) == 0 || !finisherTakes(index)) continue;
            dormantWake.notify_all();
            break;
        }
    }
    if (sleepers == 0) return;
    if (spinHandoff() && sleepers == 1) wake.notify_one();
    else wake.notify_all();
}

void DrawFinisher::startFinishers() {
    if (!running) {
        running = true;
        thread = std::thread([this] {
            FinisherThread::Set(true);
            RaiseWorkerThreadPriority("draw finisher");
            PlaceFinisherThread("draw finisher");
            run(0);
        });
        runningFinishers.fetch_add(1, std::memory_order_acq_rel);
        started.fetch_or(1u, std::memory_order_release);
    }
    for (unsigned index = 1; index <= extra.size(); ++index) {
        if (extraRunning[index - 1] || !finisherTakes(index)) continue;
        extraRunning[index - 1] = true;
        extra[index - 1] = std::thread([this, index] {
            FinisherThread::Set(true);
            RaiseWorkerThreadPriority(FinisherNames[index]);
            if (index >= 5) PlaceFinisherSiblingThread(FinisherNames[index]);
            else if (index < 3 || std::getenv("APS5_EXTRA_FINISHERS_UNPINNED") == nullptr) PlaceFinisherThread(FinisherNames[index]);
            run(index);
        });
        runningFinishers.fetch_add(1, std::memory_order_acq_rel);
        started.fetch_or(1u << index, std::memory_order_release);
    }
}

bool DrawFinisher::Adopt(bool ring) {
    if (ringMode.load(std::memory_order_relaxed) == ring) return false;
    if (outstanding.load(std::memory_order_acquire) != 0) return false;
    const auto lock = lockFor(mutex);
    if (outstanding.load(std::memory_order_acquire) != 0 || !jobs.empty() || closing || failure != nullptr) return false;
    if (ringMode.load(std::memory_order_relaxed)) retireLean();
    if (!pendingWrites.empty() || pendingUnknown != 0) {
        pendingWrites.clear();
        pendingUnknown = 0;
        unknownSeqs.clear();
        published.PublishUnknownJobs(unknownSeqs);
        published.Publish(pendingWrites, 0u);
    }
    const bool deep = ring && Deep();
    if (ring) {
        if (lean == nullptr) lean = std::make_unique<std::array<LeanSlot, LeanSlots>>();
        leanMask.store((Flow() || deep ? LeanSlots : NarrowLeanSlots) - 1, std::memory_order_seq_cst);
        leanTake.store(nextSeq, std::memory_order_seq_cst);
        leanTail.store(nextSeq, std::memory_order_seq_cst);
        leanRetired = nextSeq;
    }
    if (deepRing.load(std::memory_order_relaxed) != deep) latchDeepRing(deep);
    frontSeq.store(nextSeq, std::memory_order_release);
    leanFailed.store(false, std::memory_order_release);
    ringMode.store(ring, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (const auto slotted = parkedSlots.load(std::memory_order_seq_cst); slotted != 0) signalSlots(slotted);
    LeanFront::Add(LeanFront::Counter::StoreSwitches);
    return true;
}

void DrawFinisher::retireLean() {
    const auto front = frontSeq.load(std::memory_order_acquire);
    if (leanRetired >= front) return;
    bool republish = false;
    bool unlisted = false;
    for (; leanRetired < front; ++leanRetired) {
        auto& slot = leanSlot(leanRetired);
        if (slot.counted) {
            republish = countPending(slot.pending, false) || republish;
            if (slot.pending.unknown) {
                if (const auto listed = std::find(unknownSeqs.begin(), unknownSeqs.end(), leanRetired); listed != unknownSeqs.end()) {
                    unknownSeqs.erase(listed);
                    unlisted = true;
                }
            }
        }
        if (const auto retirer = std::exchange(slot.retirer, nullptr); retirer != nullptr) retirer(slot.payload.data(), std::move(slot.answer));
        slot.answer = nullptr;
    }
    if (republish) published.Publish(pendingWrites, pendingUnknown != 0 ? 1u : 0u);
    if (unlisted) published.PublishUnknownJobs(unknownSeqs);
    keys.Retire(front);
}

void DrawFinisher::rethrowLeanFailure() {
    if (!leanFailed.load(std::memory_order_acquire)) return;
    std::exception_ptr error;
    {
        const auto lock = lockFor(mutex);
        error = std::exchange(failure, nullptr);
        leanFailed.store(false, std::memory_order_release);
    }
    if (error != nullptr) std::rethrow_exception(error);
}

bool DrawFinisher::takeHelped(std::uint64_t& seq) {
    for (;;) {
        auto take = leanTake.load(std::memory_order_acquire);
        if (take >= leanTail.load(std::memory_order_acquire)) return false;
        if (leanSlot(take).runner == nullptr) return false;
        if (leanTake.compare_exchange_weak(take, take + 1, std::memory_order_acq_rel)) {
            seq = take;
            return true;
        }
    }
}

template<typename TWaiting>
void DrawFinisher::helpLean(TWaiting&& waiting, bool room) {
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::WorkerHelps) || !ringMode.load(std::memory_order_acquire) || FinisherThread::Get() || !FinisherProducer::Get()) return;
    if (closed.load(std::memory_order_acquire) || leanFailed.load(std::memory_order_acquire) || GuestMemory::GpuMutex().HeldByThisThread()) return;
    std::uint64_t seq = 0;
    if (!waiting() || !takeHelped(seq)) return;
    const auto start = std::chrono::steady_clock::now();
    std::uint64_t jobs = 0;
    {
        const HelpScope scope;
        do {
            if (parkedSlots.load(std::memory_order_relaxed) != 0) wakeParked(parkedSlots.load(std::memory_order_seq_cst), takingFinishers(extra.size()));
            std::chrono::steady_clock::time_point lastEnd{};
            runLean(seq, lastEnd);
            ++jobs;
        } while (waiting() && takeHelped(seq));
    }
    noteHelped(room, jobs, start);
}

void DrawFinisher::fitLeanSpan() {
    const bool deep = Deep();
    const std::uint64_t wanted = Flow() || deep ? LeanSlots : NarrowLeanSlots;
    if ((leanMask.load(std::memory_order_relaxed) + 1 == wanted && deepRing.load(std::memory_order_relaxed) == deep) || outstanding.load(std::memory_order_seq_cst) != 0) return;
    retireLean();
    if (leanRetired != nextSeq) return;
    leanMask.store(wanted - 1, std::memory_order_seq_cst);
    if (deepRing.load(std::memory_order_relaxed) != deep) latchDeepRing(deep);
}

void DrawFinisher::latchDeepRing(bool deep) {
    if (deep && deepWrites == nullptr) {
        deepWrites = std::make_unique<StampedRanges>();
        deepWrites->Name("finisher writes");
        deepKeys = std::make_unique<StampedKeys>();
        deepShapes = std::make_unique<StampedKeys>();
        deepIndexed.store(true, std::memory_order_release);
    }
    openRunWrites.clear();
    deepRing.store(deep, std::memory_order_seq_cst);
}

namespace {
std::uint64_t shallowCount(const char* name, std::uint64_t fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == 0) return fallback;
    const auto value = std::strtoull(text, nullptr, 10);
    return std::clamp<std::uint64_t>(value, 16, DrawFinisher::LeanSlots);
}

std::uint64_t shallowJobs() {
    static const std::uint64_t jobs = shallowCount("APS5_DEEP_SHALLOW_JOBS", 128);
    return jobs;
}

std::uint64_t shallowItems() {
    static const std::uint64_t items = shallowCount("APS5_DEEP_SHALLOW_ITEMS", shallowJobs());
    return items;
}
}

void DrawFinisher::awaitLeanRoom(bool helping) {
    static_assert(PublishedKeys::Capacity <= NarrowLeanSlots && 16 <= NarrowLeanSlots && NarrowLeanSlots <= LeanSlots && (NarrowLeanSlots & (NarrowLeanSlots - 1)) == 0 && (LeanSlots & (LeanSlots - 1)) == 0, "the finisher ring must hold the deepest finisher queue, in a power-of-two number of slots");
    fitLeanSpan();
    const std::uint64_t deepDepth = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DeepShallow) && !Flow() ? shallowJobs() : LeanSlots;
    const std::uint64_t depth = (Flow() || (Deep() && deepRing.load(std::memory_order_relaxed))) && leanMask.load(std::memory_order_relaxed) + 1 == LeanSlots ? deepDepth : finisherDepth();
    const auto full = [&] { return nextSeq - frontSeq.load(std::memory_order_seq_cst) >= depth && !closed.load(std::memory_order_acquire); };
    if (!full()) {
        retireLean();
        return;
    }
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher full while the enqueuing thread holds the GPU lock");
    const TailWatch watch(true, TailExtra::RingRoom);
    const auto roomStart = ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
    std::optional<ThreadReport::WorkerStateScope> room;
    if (roomStart != ThreadReport::Clock::time_point{} && ThreadReport::OnQueue0Worker()) room.emplace(ThreadReport::WorkerState::RoomFinisher);
    LeanFront::AddOnWorker(LeanFront::Counter::RingRoomWaits);
    if (helping) helpLean(full, true);
    if (full() && (!spinHandoff() || !SpinUntil(HandoffSpinWindow(), [&] { return !full(); }))) {
        frontWaiters.fetch_add(1, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        {
            auto lock = lockFor(mutex);
            if (parkVerifyDue()) verifiedIdleWait(lock, [&] { return !full(); });
            else idle.wait(lock, [&] { return !full(); });
        }
        frontWaiters.fetch_sub(1, std::memory_order_relaxed);
    }
    room.reset();
    if (roomStart != ThreadReport::Clock::time_point{}) {
        const bool worker = ThreadReport::OnQueue0Worker();
        ThreadReport::Add(worker ? ThreadReport::Count::WorkerFinisherRoomWaits : ThreadReport::Count::OtherRoomWaits, 1);
        ThreadReport::Add(worker ? ThreadReport::Count::WorkerFinisherRoomNs : ThreadReport::Count::OtherRoomNs, ThreadReport::Nanoseconds(roomStart, ThreadReport::Clock::now()));
    }
    retireLean();
}

void DrawFinisher::awaitLeanIdle() {
    if (outstanding.load(std::memory_order_seq_cst) == 0) return;
    frontWaiters.fetch_add(1, std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    {
        auto lock = lockFor(mutex);
        if (parkVerifyDue()) verifiedIdleWait(lock, [&] { return outstanding.load(std::memory_order_seq_cst) == 0; });
        else idle.wait(lock, [&] { return outstanding.load(std::memory_order_seq_cst) == 0; });
    }
    frontWaiters.fetch_sub(1, std::memory_order_relaxed);
}

void DrawFinisher::verifiedIdleWait(std::unique_lock<std::mutex>& lock, const std::function<bool()>& ready) {
    parkVerifyWaits.fetch_add(1, std::memory_order_relaxed);
    bool stuckBefore = false;
    std::uint64_t stuckTake = 0;
    while (!idle.wait_for(lock, std::chrono::milliseconds(50), ready)) {
        parkVerifySlices.fetch_add(1, std::memory_order_relaxed);
        const auto take = leanTake.load(std::memory_order_seq_cst);
        const auto tail = leanTail.load(std::memory_order_seq_cst);
        const auto slotted = parkedSlots.load(std::memory_order_seq_cst);
        const auto taking = takingFinishers(extra.size()) & started.load(std::memory_order_acquire);
        const bool stuck = tail > take && idleSpinners.load(std::memory_order_seq_cst) == 0 && wakesInFlight.load(std::memory_order_seq_cst) == 0 && taking != 0 && (slotted & taking) == taking;
        if (!stuck || !stuckBefore || take != stuckTake) {
            stuckBefore = stuck;
            stuckTake = take;
            continue;
        }
        stuckBefore = false;
        if (parkVerifyDiffering.fetch_add(1, std::memory_order_relaxed) < 16) {
            std::fprintf(stderr, "[finisher-park-verify] jobs ready for two slices while every taking finisher sleeps in its park slot, none spins and no wake is in flight: front %llu, take %llu, tail %llu, parked 0x%x, taking 0x%x; waking every slot\n", static_cast<unsigned long long>(frontSeq.load(std::memory_order_acquire)), static_cast<unsigned long long>(take), static_cast<unsigned long long>(tail), slotted, taking);
        }
        signalSlots(slotted);
    }
    noteParkVerifyCounts();
}

void DrawFinisher::signalSlots(std::uint32_t bits) {
    for (unsigned index = 0; bits != 0 && index < parkSlots.size(); ++index) {
        const std::uint32_t bit = 1u << index;
        if ((bits & bit) == 0) continue;
        bits &= ~bit;
        wakesInFlight.fetch_add(1, std::memory_order_seq_cst);
        if ((parkedSlots.fetch_and(~bit, std::memory_order_seq_cst) & bit) == 0) {
            wakesInFlight.fetch_sub(1, std::memory_order_seq_cst);
            continue;
        }
        auto& park = parkSlots[index];
        {
            const std::lock_guard lock(park.mutex);
            park.signaled = true;
        }
        park.wake.notify_one();
    }
}

void DrawFinisher::wakeParked(std::uint32_t slotted, unsigned wanted) {
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FinisherParkSlots)) {
        signalSlots(slotted);
        return;
    }
    if ((slotted & ~wanted) != 0) signalSlots(slotted & ~wanted);
    const std::uint32_t taking = slotted & wanted;
    if (taking == 0) return;
    const auto take = leanTake.load(std::memory_order_seq_cst);
    const auto tail = leanTail.load(std::memory_order_seq_cst);
    if (tail <= take) return;
    const std::uint64_t awake = std::uint64_t{idleSpinners.load(std::memory_order_seq_cst)} + wakesInFlight.load(std::memory_order_seq_cst);
    if (tail - take > awake) signalSlots(taking & (~taking + 1u));
}

bool DrawFinisher::parkInSlot(unsigned index) {
    auto& park = parkSlots[index];
    const std::uint32_t bit = 1u << index;
    std::unique_lock lock(park.mutex);
    parkedSlots.fetch_or(bit, std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const bool stays = !closed.load(std::memory_order_seq_cst) && !leanReady() && untakenJobs.load(std::memory_order_seq_cst) == 0 && finisherTakes(index) && ringMode.load(std::memory_order_seq_cst) && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FinisherParkSlots);
    if (!stays && (parkedSlots.fetch_and(~bit, std::memory_order_seq_cst) & bit) != 0) return false;
    park.wake.wait(lock, [&] { return park.signaled; });
    park.signaled = false;
    return true;
}

void DrawFinisher::publishLean(LeanSlot& slot) {
    const auto seq = nextSeq;
    const bool raises = !slot.pending.unknown;
    if (raises) raiseQueued(Graphics::WriteValidation::Source::FinisherJob, slot.pending.writes);
    if (deepRing.load(std::memory_order_relaxed)) {
        slot.counted = false;
        slot.resolved.store(0, std::memory_order_relaxed);
        if (!slot.resolvedWrites.empty()) slot.resolvedWrites.clear();
        slot.unknownBefore = lastUnknown.load(std::memory_order_relaxed);
        const auto live = frontSeq.load(std::memory_order_acquire);
        const auto& held = slot.pending;
        if (slot.batch) openRunWrites.clear();
        else if (held.unknown) lastUnknown.store(seq, std::memory_order_release);
        else deepWrites->Insert(held.writes, seq, live);
        if (held.key == PublishedKeys::AnyKey) deepKeys->InsertAny(seq, held.inserts, live);
        else deepKeys->Insert(held.key, seq, held.inserts, live);
        if (held.shape != 0) deepShapes->Insert(held.shape, seq, held.inserts, live);
    } else {
        slot.counted = slot.batch || !Flow() || (pendingWrites.size() <= PublishedRanges::Capacity && unknownSeqs.size() <= PublishedRanges::UnknownCapacity);
        if (!slot.batch && slot.counted && slot.pending.unknown) {
            unknownSeqs.push_back(seq);
            published.PublishUnknownJobs(unknownSeqs);
        }
        if (!slot.batch && slot.counted && countPending(slot.pending, true)) published.Publish(pendingWrites, pendingUnknown != 0 ? 1u : 0u);
        keys.Push(seq, slot.pending.key, slot.pending.shape, slot.pending.inserts);
    }
    if (raises) raiseQueued(Graphics::WriteValidation::Source::FinisherJob, slot.pending.writes);
    slot.seq.store(seq, std::memory_order_relaxed);
    slot.state.store(LeanReady, std::memory_order_relaxed);
    outstanding.fetch_add(1, std::memory_order_acq_rel);
    nextSeq = seq + 1;
    issued.store(nextSeq, std::memory_order_relaxed);
    leanTail.store(nextSeq, std::memory_order_seq_cst);
    if (pipelineStats()) {
        const auto depth = nextSeq - frontSeq.load(std::memory_order_relaxed);
        noteRingDepth(depth);
        noteDeepDepth(depth);
    }
    unsigned wanted = 1u;
    for (unsigned index = 1; index <= extra.size(); ++index) wanted |= finisherTakes(index) ? 1u << index : 0u;
    if ((started.load(std::memory_order_acquire) & wanted) != wanted) {
        const auto lock = lockFor(mutex);
        startFinishers();
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if ((dormant.load(std::memory_order_acquire) & wanted) != 0) {
        const auto lock = lockFor(mutex);
        dormantWake.notify_all();
    }
    if (const auto slotted = parkedSlots.load(std::memory_order_seq_cst); slotted != 0) wakeParked(slotted, wanted);
    if (parked.load(std::memory_order_relaxed) == 0) return;
    const auto lock = lockFor(mutex);
    if (spinHandoff() && sleepers == 1) wake.notify_one();
    else wake.notify_all();
}

std::byte* DrawFinisher::ReserveLean() {
    FinisherProducer::Set(true);
    FlushOpenRun();
    rethrowLeanFailure();
    if (closed.load(std::memory_order_acquire)) return nullptr;
    awaitLeanRoom(true);
    if (closed.load(std::memory_order_acquire)) return nullptr;
    return leanSlot(nextSeq).payload.data();
}

void DrawFinisher::PublishLean(LeanRunner runner, std::span<const Range> writes, std::uint64_t key, std::uint64_t shape, LeanRetirer retirer, bool batch) {
    auto& slot = leanSlot(nextSeq);
    std::unique_lock<std::mutex> verifying;
    if (deepVerifyEvery() != 0) verifying = std::unique_lock<std::mutex>(verifySlots);
    slot.runner = runner;
    slot.retirer = retirer;
    slot.batch = batch;
    slot.answer = nullptr;
    auto& held = slot.pending;
    held.key = key;
    held.shape = shape;
    held.inserts = true;
    held.unknown = false; held.origin = 1;
    held.named = false;
    held.writes.assign(writes.begin(), writes.end());
    held.labelAddress = 0;
    held.labelBytes.clear();
    held.labelRun.reset();
    publishLean(slot);
}

bool DrawFinisher::AnswerLean(std::shared_ptr<const void> answer) {
    const auto seq = FinisherSeq::Get();
    auto& finisher = Get();
    if (seq == 0 || finisher.lean == nullptr || !finisher.ringMode.load(std::memory_order_acquire)) return false;
    auto& slot = finisher.leanSlot(seq);
    if (slot.seq.load(std::memory_order_acquire) != seq || slot.retirer == nullptr) return false;
    slot.answer = std::move(answer);
    return true;
}

bool DrawFinisher::takeLean(unsigned index, std::uint64_t& seq) {
    for (;;) {
        auto take = leanTake.load(std::memory_order_acquire);
        if (take >= leanTail.load(std::memory_order_acquire)) return false;
        if (!finisherTakes(index) && !closed.load(std::memory_order_acquire)) return false;
        if (leanTake.compare_exchange_weak(take, take + 1, std::memory_order_acq_rel)) {
            seq = take;
            return true;
        }
    }
}

void DrawFinisher::runLean(std::uint64_t seq, std::chrono::steady_clock::time_point& lastEnd) {
    auto& slot = leanSlot(seq);
    FinisherSeq::Set(seq);
    const auto jobStart = ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
    if (jobStart != ThreadReport::Clock::time_point{}) {
        ThreadReport::FinisherBusy(true, jobStart);
        static_cast<void>(ThreadReport::TakeItemMark());
    }
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::XpSlowFinisher)) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(2);
        while (std::chrono::steady_clock::now() < until) {}
    }
    std::exception_ptr error;
    bool ran = false;
    try {
        if (!leanFailed.load(std::memory_order_acquire)) {
            ran = true;
            if (slot.runner != nullptr) slot.runner(slot.payload.data());
            else slot.closure();
        }
    } catch (...) {
        error = std::current_exception();
    }
    slot.closure = nullptr;
    FinisherSeq::Set(0);
    if (jobStart != ThreadReport::Clock::time_point{}) {
        const auto end = ThreadReport::Clock::now();
        ThreadReport::FinisherBusy(false, end);
        ThreadReport::Add(ThreadReport::Count::FinisherJobs, 1);
        ThreadReport::Add(ThreadReport::Count::FinisherJobNs, ThreadReport::Nanoseconds(jobStart, end));
        std::uint64_t waited = 0;
        const auto mark = ThreadReport::TakeItemMark(&waited);
        ThreadReport::NoteFinisherJob(mark, ThreadReport::Nanoseconds(jobStart, end), waited);
        if (lastEnd != ThreadReport::Clock::time_point{}) ThreadReport::Add(ThreadReport::Count::FinisherIdleNs, ThreadReport::Nanoseconds(lastEnd, jobStart));
        lastEnd = end;
        ThreadReport::Tick(end);
    }
    if (error != nullptr) {
        const auto lock = lockFor(mutex);
        if (failure == nullptr) failure = error;
        leanFailed.store(true, std::memory_order_release);
    }
    const bool deposited = !slot.deposits.empty();
    const bool resolving = slot.pending.unknown && !slot.batch && deepRing.load(std::memory_order_relaxed) && FrameFlowWanted();
    if (resolving) {
        auto& resolved = slot.resolvedWrites;
        resolved.clear();
        for (const auto& held : slot.deposits) {
            resolved.insert(resolved.end(), held.item.writes.begin(), held.item.writes.end());
            if (held.item.labelAddress != 0 && !held.item.labelBytes.empty()) resolved.emplace_back(held.item.labelAddress, held.item.labelAddress + held.item.labelBytes.size());
        }
        slot.resolved.store(ran && error == nullptr ? 1 : 2, std::memory_order_seq_cst);
    }
    FinisherDeposits::Set(false);
    slot.state.store(deposited ? LeanDeposited : LeanDone, std::memory_order_seq_cst);
    if (resolving) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (resolveWaiters.load(std::memory_order_relaxed) != 0) {
            const auto lock = lockFor(mutex);
            turnWake.notify_all();
        }
    }
    const auto forwardNs = advanceLeanFront();
    if (forwardNs != 0 && lastEnd != ThreadReport::Clock::time_point{}) lastEnd += std::chrono::nanoseconds(forwardNs);
}

std::uint64_t DrawFinisher::advanceLeanFront() {
    bool moved = false;
    std::uint64_t forwardNs = 0;
    for (;;) {
        auto front = frontSeq.load(std::memory_order_seq_cst);
        if (front >= leanTail.load(std::memory_order_seq_cst)) break;
        auto& slot = leanSlot(front);
        if (slot.seq.load(std::memory_order_seq_cst) != front) break;
        const auto state = slot.state.load(std::memory_order_seq_cst);
        if (state == LeanDeposited) {
            if (!forwardLean(slot, front, forwardNs)) break;
            continue;
        }
        if (state != LeanDone) break;
        if (!frontSeq.compare_exchange_strong(front, front + 1, std::memory_order_seq_cst)) continue;
        outstanding.fetch_sub(1, std::memory_order_seq_cst);
        moved = true;
    }
    if (!moved) return forwardNs;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (frontWaiters.load(std::memory_order_relaxed) == 0) return forwardNs;
    const auto lock = lockFor(mutex);
    idle.notify_all();
    turnWake.notify_all();
    return forwardNs;
}

DrawFinisher::DepositCounts DrawFinisher::Deposits() {
    return {depositedItems.load(std::memory_order_relaxed), forwardedItems.load(std::memory_order_relaxed), turnItems.load(std::memory_order_relaxed)};
}

bool DrawFinisher::deposit(DrawPipeline& pipeline, DrawPipeline::Item& item) {
    const auto seq = FinisherSeq::Get();
    if (seq == 0 || !ringMode.load(std::memory_order_acquire)) return false;
    auto& slot = leanSlot(seq);
    if (frontSeq.load(std::memory_order_acquire) == seq && !(slot.batch && item.draw && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DrawRuns))) {
        turnItems.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    raiseQueued(Graphics::WriteValidation::Source::Deposit, item.writes);
    slot.deposits.push_back({&pipeline, std::move(item)});
    raiseQueued(Graphics::WriteValidation::Source::Deposit, slot.deposits.back().item.writes);
    FinisherDeposits::Set(true);
    depositedItems.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void DrawFinisher::forwardMerged(std::vector<Deposit>& deposits, std::size_t& sent) {
    const bool merging = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DrawRuns);
    for (std::size_t first = 0; first < deposits.size();) {
        auto& head = deposits[first];
        const auto joins = [&](const Deposit& held) {
            const auto& item = held.item;
            return held.pipeline == head.pipeline && item.draw && !item.store && !item.quiet && item.labelAddress == 0 && item.labelBytes.empty() && item.order == 0 && item.kind == head.item.kind;
        };
        std::size_t last = first + 1;
        if (merging && joins(head)) {
            while (last < deposits.size() && joins(deposits[last])) ++last;
        }
        if (last - first < 2) {
            head.pipeline->queueItem(std::move(head.item));
            ++sent;
            ++first;
            continue;
        }
        DrawPipeline::Item merged;
        merged.draw = true;
        merged.kind = head.item.kind;
        merged.parts = static_cast<std::uint32_t>(last - first);
        auto parts = std::make_shared<std::vector<DrawPipeline::Item>>();
        parts->reserve(last - first);
        struct UnparkParts {
            std::vector<std::uint64_t> ids;
            ~UnparkParts() {
                for (const auto id : ids) DrawFinisher::Get().UnparkReads(id);
            }
        } unparkParts;
        for (std::size_t at = first; at < last; ++at) {
            auto& item = deposits[at].item;
            if (item.parked != 0) unparkParts.ids.push_back(std::exchange(item.parked, 0));
            for (const auto& range : item.writes) {
                if (std::find(merged.writes.begin(), merged.writes.end(), range) == merged.writes.end()) merged.writes.push_back(range);
            }
            merged.reads.insert(merged.reads.end(), item.reads.begin(), item.reads.end());
            parts->push_back(std::move(item));
        }
        merged.commit = [parts] {
            for (auto& part : *parts) part.commit();
        };
        auto* const pipeline = head.pipeline;
        pipeline->queueItem(std::move(merged));
        sent += last - first;
        first = last;
    }
}

void DrawFinisher::forwardOwn(std::uint64_t seq) {
    FinisherDeposits::Set(false);
    auto& slotDeposits = leanSlot(seq).deposits;
    std::vector<Deposit> deposits;
    deposits.swap(slotDeposits);
    struct Restore {
        std::vector<Deposit>& taken;
        std::vector<Deposit>& slot;
        ~Restore() {
            taken.clear();
            if (slot.empty()) taken.swap(slot);
        }
    } restore{deposits, slotDeposits};
    const auto start = ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
    std::size_t sent = 0;
    const auto count = [&] {
        forwardedItems.fetch_add(sent, std::memory_order_relaxed);
        if (start == ThreadReport::Clock::time_point{}) return;
        ThreadReport::Add(ThreadReport::Count::FinisherForwards, sent);
        ThreadReport::Add(ThreadReport::Count::FinisherForwardNs, ThreadReport::Nanoseconds(start, ThreadReport::Clock::now()));
    };
    try {
        forwardMerged(deposits, sent);
    } catch (...) {
        for (auto& held : deposits) {
            if (held.item.parked != 0) UnparkReads(std::exchange(held.item.parked, 0));
        }
        count();
        throw;
    }
    count();
}

bool DrawFinisher::forwardLean(LeanSlot& slot, std::uint64_t seq, std::uint64_t& forwardNs) {
    auto expected = static_cast<std::uint8_t>(LeanDeposited);
    if (!slot.state.compare_exchange_strong(expected, static_cast<std::uint8_t>(LeanForwarding), std::memory_order_seq_cst)) return false;
    if (slot.seq.load(std::memory_order_seq_cst) != seq) {
        slot.state.store(LeanDeposited, std::memory_order_seq_cst);
        return true;
    }
    const auto start = ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
    std::exception_ptr error;
    std::size_t sent = 0;
    try {
        forwardMerged(slot.deposits, sent);
    } catch (...) {
        error = std::current_exception();
        for (auto& held : slot.deposits) {
            if (held.item.parked != 0) UnparkReads(std::exchange(held.item.parked, 0));
        }
    }
    slot.deposits.clear();
    forwardedItems.fetch_add(sent, std::memory_order_relaxed);
    if (start != ThreadReport::Clock::time_point{}) {
        const auto ns = ThreadReport::Nanoseconds(start, ThreadReport::Clock::now());
        ThreadReport::Add(ThreadReport::Count::FinisherForwards, sent);
        ThreadReport::Add(ThreadReport::Count::FinisherForwardNs, ns);
        forwardNs += ns;
    }
    if (error != nullptr) {
        const auto lock = lockFor(mutex);
        if (failure == nullptr) failure = error;
        leanFailed.store(true, std::memory_order_release);
    }
    slot.state.store(LeanDone, std::memory_order_seq_cst);
    return true;
}

const std::vector<DrawFinisher::Deposit>& DrawFinisher::ownDeposits() {
    return leanSlot(FinisherSeq::Get()).deposits;
}

bool DrawFinisher::depositsWrite(std::span<const Range> ranges) {
    for (const auto& held : ownDeposits()) {
        for (const auto& [first, last] : held.item.writes) {
            for (const auto& [begin, end] : ranges) {
                if (begin < last && first < end) return true;
            }
        }
    }
    return false;
}

std::uint64_t DrawFinisher::ParkReads(const std::vector<Range>& reads) {
    std::lock_guard lock(parkedReadsMutex);
    const auto id = nextParkedReads++;
    parkedReads.emplace_back(id, reads);
    parkedReadCount.fetch_add(1, std::memory_order_release);
    return id;
}

void DrawFinisher::UnparkReads(std::uint64_t id) {
    std::lock_guard lock(parkedReadsMutex);
    const auto found = std::ranges::find_if(parkedReads, [id](const auto& entry) { return entry.first == id; });
    if (found == parkedReads.end()) return;
    parkedReads.erase(found);
    parkedReadCount.fetch_sub(1, std::memory_order_release);
}

bool DrawFinisher::ParkedReadsOverlap(std::uint64_t address, std::size_t bytes) {
    if (parkedReadCount.load(std::memory_order_acquire) == 0) return false;
    const auto end = address + bytes;
    std::lock_guard lock(parkedReadsMutex);
    for (const auto& [id, reads] : parkedReads) {
        for (const auto& [begin, limit] : reads) {
            if (address < limit && begin < end) return true;
        }
    }
    return false;
}

bool DrawFinisher::depositsRead(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    for (const auto& held : ownDeposits()) {
        for (const auto& [begin, limit] : held.item.reads) {
            if (address < limit && begin < end) return true;
        }
    }
    return false;
}

void DrawFinisher::checkDrainDeposits() {
    if (lean == nullptr || drainOrdinal.fetch_add(1, std::memory_order_relaxed) % depositVerifyEvery() != 0) return;
    drainChecks.fetch_add(1, std::memory_order_relaxed);
    std::size_t left = 0;
    std::size_t slots = 0;
    for (const auto& slot : *lean) {
        if (slot.deposits.empty()) continue;
        left += slot.deposits.size();
        ++slots;
    }
    if (left == 0) return;
    drainDifferences.fetch_add(1, std::memory_order_relaxed);
    noteDepositDifference("drain end differs: " + std::to_string(left) + " deposited items were left in " + std::to_string(slots) + " ring slots after the finisher drained (front " + std::to_string(frontSeq.load(std::memory_order_acquire)) + ")");
}

void DrawFinisher::run(unsigned index) {
    if (index == 0) NameDriverThread("draw-finish");
    PerformanceTrace::BindThread(FinisherNames[index]);
#ifndef _WIN32
    if (index == 0) ReserveFinisherCore();
#endif
    if (index == 0) StartFinisherSampler();
    FinisherRunLoop::Set(true);
    ThreadReport::Clock::time_point lastEnd{};
    bool woken = false;
    for (;;) {
        std::uint64_t leanSeq = 0;
        const bool took = takeLean(index, leanSeq);
        const bool wasWoken = std::exchange(woken, false);
        if (wasWoken) wakesInFlight.fetch_sub(1, std::memory_order_seq_cst);
        if (took) {
            if (parkedSlots.load(std::memory_order_relaxed) != 0) wakeParked(parkedSlots.load(std::memory_order_seq_cst), takingFinishers(extra.size()));
            runLean(leanSeq, lastEnd);
            continue;
        }
        Slot* slot = nullptr;
        bool failed = false;
        const bool slots = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FinisherParkSlots) && ringMode.load(std::memory_order_acquire) && finisherTakes(index) && !closed.load(std::memory_order_acquire);
        if (slots) idleSpinners.fetch_add(1, std::memory_order_seq_cst);
        static_cast<void>(releaseRetiredHits(true));
        if (finisherTakes(index) && spinHandoff()) SpinUntil(HandoffSpinWindow(), [&] { return untakenJobs.load(std::memory_order_acquire) != 0 || leanReady(); });
        else if (finisherTakes(index)) SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
        if (slots) idleSpinners.fetch_sub(1, std::memory_order_seq_cst);
        if (takeLean(index, leanSeq)) {
            if (parkedSlots.load(std::memory_order_relaxed) != 0) wakeParked(parkedSlots.load(std::memory_order_seq_cst), takingFinishers(extra.size()));
            runLean(leanSeq, lastEnd);
            continue;
        }
        if (!releaseRetiredHits(true) && finisherTakes(index) && untakenJobs.load(std::memory_order_acquire) == 0) continue;
        if (slots && finisherTakes(index)) {
            woken = parkInSlot(index);
            continue;
        }
        if (slots || wasWoken) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (const auto slotted = parkedSlots.load(std::memory_order_seq_cst); slotted != 0 && leanReady()) wakeParked(slotted, takingFinishers(extra.size()));
        }
        {
            auto lock = lockFor(mutex);
            if (!finisherTakes(index) && !closing) {
                lock.unlock();
                releaseAllRetiredHits();
                ReleaseFinisherThread(FinisherNames[index]);
                lock.lock();
                dormant.fetch_or(1u << index, std::memory_order_seq_cst);
                dormantWake.wait(lock, [&] { return closing || finisherTakes(index); });
                dormant.fetch_and(~(1u << index), std::memory_order_seq_cst);
                const bool leaving = closing;
                lock.unlock();
                if (!leaving) {
                    if (index >= 5) PlaceFinisherSiblingThread(FinisherNames[index]);
                    else if (index < 3 || std::getenv("APS5_EXTRA_FINISHERS_UNPINNED") == nullptr) PlaceFinisherThread(FinisherNames[index]);
                }
                continue;
            }
            if (FinisherRetired::Get() && !closing && untaken() == nullptr && !leanReady()) {
                lock.unlock();
                continue;
            }
            ++sleepers;
            parked.fetch_add(1, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            wake.wait(lock, [&] { return closing || !finisherTakes(index) || untaken() != nullptr || leanReady(); });
            parked.fetch_sub(1, std::memory_order_relaxed);
            --sleepers;
            if (!finisherTakes(index) && !closing) continue;
            if (leanReady()) continue;
            slot = untaken();
            if (slot == nullptr) {
                if (closing) break;
                continue;
            }
            slot->taken = true;
            untakenJobs.fetch_sub(1, std::memory_order_relaxed);
            failed = failure != nullptr;
        }
        FinisherSeq::Set(slot->seq);
        const auto jobStart = ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
        if (jobStart != ThreadReport::Clock::time_point{}) {
            ThreadReport::FinisherBusy(true, jobStart);
            static_cast<void>(ThreadReport::TakeItemMark());
        }
        std::exception_ptr error;
        try {
            if (!failed) slot->job();
        } catch (...) {
            error = std::current_exception();
        }
        FinisherSeq::Set(0);
        if (jobStart != ThreadReport::Clock::time_point{}) {
            const auto end = ThreadReport::Clock::now();
            ThreadReport::FinisherBusy(false, end);
            ThreadReport::Add(ThreadReport::Count::FinisherJobs, 1);
            ThreadReport::Add(ThreadReport::Count::FinisherJobNs, ThreadReport::Nanoseconds(jobStart, end));
            std::uint64_t waited = 0;
            const auto mark = ThreadReport::TakeItemMark(&waited);
            ThreadReport::NoteFinisherJob(mark, ThreadReport::Nanoseconds(jobStart, end), waited);
            if (lastEnd != ThreadReport::Clock::time_point{}) ThreadReport::Add(ThreadReport::Count::FinisherIdleNs, ThreadReport::Nanoseconds(lastEnd, jobStart));
            lastEnd = end;
            ThreadReport::Tick(end);
        }
        const auto lock = lockFor(mutex);
        if (error != nullptr && failure == nullptr) failure = error;
        slot->done = true;
        bool republish = false;
        while (!jobs.empty() && jobs.front().done) {
            ran.push_back(std::move(jobs.front().job));
            jobs.pop_front();
            republish = countPending(pending.front(), false) || republish;
            pending.pop_front();
            outstanding.fetch_sub(1, std::memory_order_release);
        }
        if (republish) published.Publish(pendingWrites, pendingUnknown != 0 ? 1u : 0u);
        keys.Retire(jobs.empty() ? nextSeq : jobs.front().seq);
        frontSeq.store(jobs.empty() ? nextSeq : jobs.front().seq, std::memory_order_release);
        if (idleWaiters != 0) idle.notify_all();
        if (turnSleepers != 0) turnWake.notify_all();
    }
    releaseAllRetiredHits();
    FinisherRunLoop::Set(false);
    runningFinishers.fetch_sub(1, std::memory_order_acq_rel);
}

void DrawFinisher::Settle() {
    FlushOpenRun();
    if (!Busy() || OnFinisherThread()) return;
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher settle under the GPU lock");
    const TailWatch watch(true);
    if (ringMode.load(std::memory_order_acquire)) {
        awaitLeanIdle();
        if (FinisherProducer::Get()) retireLean();
        return;
    }
    auto lock = lockFor(mutex);
    ++idleWaiters;
    idle.wait(lock, [&] { return jobs.empty(); });
    --idleWaiters;
}

bool DrawFinisher::AwaitRawReaders() {
    FlushOpenRun();
    if (!Busy() || OnFinisherThread()) return false;
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher settle under the GPU lock");
    if (!ringMode.load(std::memory_order_acquire)) {
        auto lock = lockFor(mutex);
        if (jobs.empty()) return false;
        ++idleWaiters;
        idle.wait(lock, [&] { return jobs.empty(); });
        --idleWaiters;
        return true;
    }
    const auto reader = rawReader;
    const auto passed = [&] { return frontSeq.load(std::memory_order_seq_cst) > reader; };
    if (reader == 0 || passed()) {
        if (FinisherProducer::Get()) retireLean();
        return false;
    }
    {
        const TailWatch watch(true, TailExtra::Exit);
        const ThreadReport::WorkerStateScope draining(ThreadReport::WorkerState::DrainFinisher, ThreadReport::Enabled() && ThreadReport::OnQueue0Worker());
        if (!SpinUntil(HandoffSpinWindow() * 2, passed)) {
            frontWaiters.fetch_add(1, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            {
                auto lock = lockFor(mutex);
                turnWake.wait(lock, [&] { return closing || passed(); });
            }
            frontWaiters.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    if (FinisherProducer::Get()) retireLean();
    return true;
}

void DrawFinisher::Drain(bool helping) {
    if (OnFinisherThread()) return;
    FlushOpenRun();
    if (Busy() && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher drain under the GPU lock");
    const auto drainStart = ThreadReport::Enabled() && Busy() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
    const TailWatch watch(Busy());
    const ThreadReport::WorkerStateScope draining(ThreadReport::WorkerState::DrainFinisher, drainStart != ThreadReport::Clock::time_point{} && ThreadReport::OnQueue0Worker());
    if (helping) helpLean([&] { return outstanding.load(std::memory_order_acquire) != 0; }, false);
    SpinUntil(HandoffSpinWindow() * 2, [&] { return outstanding.load(std::memory_order_acquire) == 0; });
    if (ringMode.load(std::memory_order_acquire)) {
        awaitLeanIdle();
        if (drainStart != ThreadReport::Clock::time_point{}) {
            const bool worker = ThreadReport::OnQueue0Worker();
            ThreadReport::Add(worker ? ThreadReport::Count::WorkerFinisherDrains : ThreadReport::Count::OtherDrains, 1);
            ThreadReport::Add(worker ? ThreadReport::Count::WorkerFinisherDrainNs : ThreadReport::Count::OtherDrainNs, ThreadReport::Nanoseconds(drainStart, ThreadReport::Clock::now()));
        }
        if (FinisherProducer::Get()) retireLean();
        if (depositVerifyEvery() != 0 && FinisherProducer::Get()) checkDrainDeposits();
        rethrowLeanFailure();
        return;
    }
    auto lock = lockFor(mutex);
    if (!jobs.empty()) {
        ++idleWaiters;
        idle.wait(lock, [&] { return jobs.empty(); });
        --idleWaiters;
    }
    if (drainStart != ThreadReport::Clock::time_point{}) {
        const bool worker = ThreadReport::OnQueue0Worker();
        ThreadReport::Add(worker ? ThreadReport::Count::WorkerFinisherDrains : ThreadReport::Count::OtherDrains, 1);
        ThreadReport::Add(worker ? ThreadReport::Count::WorkerFinisherDrainNs : ThreadReport::Count::OtherDrainNs, ThreadReport::Nanoseconds(drainStart, ThreadReport::Clock::now()));
    }
    if (failure != nullptr) std::rethrow_exception(std::exchange(failure, nullptr));
}

void DrawFinisher::Stop() {
    FlushOpenRun();
    std::thread finishers[9];
    {
        const auto lock = lockFor(mutex);
        closing = true;
        closed.store(true, std::memory_order_release);
        finishers[0] = std::move(thread);
        for (std::size_t index = 0; index < extra.size(); ++index) finishers[index + 1] = std::move(extra[index]);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (const auto slotted = parkedSlots.load(std::memory_order_seq_cst); slotted != 0) signalSlots(slotted);
    wake.notify_all();
    dormantWake.notify_all();
    turnWake.notify_all();
    idle.notify_all();
    for (auto& finisher : finishers) {
        if (!finisher.joinable()) continue;
        if (finisher.get_id() == std::this_thread::get_id()) finisher.detach();
        else finisher.join();
    }
}

void DrawFinisher::ReleaseRan() {
    std::vector<Job> done;
    std::vector<std::shared_ptr<const void>> answers;
    const auto lock = lockFor(mutex);
    std::swap(done, ran);
    if (!closed.load(std::memory_order_acquire) || lean == nullptr) return;
    for (auto& slot : *lean) {
        slot.retirer = nullptr;
        if (slot.answer != nullptr) answers.push_back(std::move(slot.answer));
    }
}

DrawPipeline& DrawPipeline::Queue0() {
    static DrawPipeline* pipeline = new DrawPipeline();
    return *pipeline;
}

std::size_t DrawPipeline::Depth() {
    static const std::size_t depth = [] {
        const char* text = std::getenv("APS5_PIPELINED_DRAWS");
        if (std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr || std::getenv("APS5_DRAW_DRAIN") != nullptr || std::getenv("APS5_DRAIN_ALL") != nullptr || std::getenv("APS5_NO_WORDWISE_CAPTURE") != nullptr) return std::size_t{0};
        if (text == nullptr) return std::size_t{32};
        const auto value = std::strtoull(text, nullptr, 10);
        return value == 0 ? std::size_t{0} : value == 1 ? std::size_t{8} : static_cast<std::size_t>(std::min<unsigned long long>(value, 256));
    }();
    return depth;
}

std::size_t DrawPipeline::QueuedItems() {
    auto& pipeline = Queue0();
    const QueueScan scan(pipeline, ringScanId(RingScan::QueuedItems));
    const auto count = [](const QueuedView& queued) { return queued.size(); };
    const auto size = count(scan.Items());
    scan.Check("queued items scan", count);
    return size;
}

bool& DrawPipeline::Active() {
    struct ActiveStorage {};
    return HostThreadLocal<bool, ActiveStorage>();
}

std::atomic<std::uint64_t>& DrawPipeline::EpochToken() {
    static std::atomic<std::uint64_t> token{1};
    return token;
}

namespace {

struct EpochSince {
    std::atomic<std::uint64_t> token{0};
    std::atomic<std::uint64_t> since{0};
};

std::array<EpochSince, 4096>& epochSinces() {
    static std::array<EpochSince, 4096> sinces;
    return sinces;
}

}

void DrawPipeline::NoteEpochSince(std::uint64_t token, std::uint64_t since) {
    auto& slot = epochSinces()[token % epochSinces().size()];
    slot.token.store(0, std::memory_order_relaxed);
    slot.since.store(since, std::memory_order_relaxed);
    slot.token.store(token, std::memory_order_release);
}

void DrawPipeline::FollowEpoch(std::uint64_t token) {
    struct FollowedSlot {};
    using Followed = HostThreadSlot<std::uint64_t, FollowedSlot>;
    if (Followed::Get() == token) return;
    Followed::Set(token);
    const auto& slot = epochSinces()[token % epochSinces().size()];
    const bool noted = slot.token.load(std::memory_order_acquire) == token;
    const auto since = slot.since.load(std::memory_order_relaxed);
    GuestMemory::BumpCollectEpochSince(noted && slot.token.load(std::memory_order_acquire) == token ? since : 0);
}

unsigned DrawPipeline::Committers() {
    return runningCommitters.load(std::memory_order_acquire);
}

void DrawPipeline::rethrowFailure() {
    if (failure == nullptr) return;
    auto error = std::exchange(failure, nullptr);
    failurePending.store(false, std::memory_order_release);
    std::rethrow_exception(error);
}

void DrawPipeline::Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, std::vector<Range> reads, bool draw, Ahead ahead, const void* aheadObject) {
    enqueue({.commit = std::move(commit), .writes = std::move(writes), .labelAddress = labelAddress, .labelBytes = std::move(labelBytes), .reads = std::move(reads), .draw = draw, .ahead = ahead, .aheadObject = aheadObject});
}

void DrawPipeline::EnqueueQuiet(Commit commit) {
    enqueue({std::move(commit), {}, 0, {}, {}, false, false, CommitKind::Other, true});
}

void DrawPipeline::EnqueueKept(Commit commit, std::vector<Range> writes, std::vector<Range> reads, std::shared_ptr<const void> keep) {
    enqueue({std::move(commit), std::move(writes), 0, {}, std::move(reads), true, false, CommitKind::Other, false, 0, 1, std::move(keep)});
}

void DrawPipeline::Enqueue(CommitKind kind, Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, std::vector<Range> reads, bool draw, Ahead ahead, const void* aheadObject) {
    enqueue({.commit = std::move(commit), .writes = std::move(writes), .labelAddress = labelAddress, .labelBytes = std::move(labelBytes), .reads = std::move(reads), .draw = draw, .store = false, .kind = kind, .ahead = ahead, .aheadObject = aheadObject});
}

void DrawPipeline::NoteDrawKey(std::uint64_t key, std::uint64_t shape) {
    constexpr std::size_t RecentSlots = 4096;
    thread_local std::array<std::uint64_t, RecentSlots> recent{};
    thread_local std::uint64_t previous = 0;
    thread_local std::array<std::uint64_t, RecentSlots> recentShapes{};
    thread_local std::uint64_t previousShape = 0;
    const auto shapeSlot = static_cast<std::size_t>((shape ^ (shape >> 29u)) % RecentSlots);
    const auto slot = static_cast<std::size_t>((key ^ (key >> 29u)) % RecentSlots);
    drawKeys.Add(1);
    if (key == previous) drawKeysSameAsPrevious.Add(1);
    if (recent[slot] == key) drawKeysSeenRecently.Add(1);
    if (shape == previousShape) drawShapesSameAsPrevious.Add(1);
    if (recentShapes[shapeSlot] == shape) drawShapesSeenRecently.Add(1);
    recent[slot] = key;
    previous = key;
    recentShapes[shapeSlot] = shape;
    previousShape = shape;
}

void DrawPipeline::EnqueueStore(Commit commit, std::uint64_t address, std::vector<std::byte> bytes) {
    const auto end = address + bytes.size();
    enqueue({std::move(commit), {{address, end}}, address, std::move(bytes), {}, false, true, CommitKind::Store});
}

void DrawPipeline::EnqueueLabelRun(Commit commit, std::vector<Range> writes, std::shared_ptr<const LabelRun> run) {
    Item item;
    item.commit = std::move(commit);
    item.writes = std::move(writes);
    item.kind = CommitKind::EndOfPipe;
    item.quiet = true;
    item.parts = static_cast<std::uint32_t>(std::max<std::size_t>(run->labels.size(), 1));
    item.labelRun = std::move(run);
    enqueue(std::move(item));
}

void DrawPipeline::EnqueueFlip(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes) {
    Item item;
    item.commit = std::move(commit);
    item.writes = std::move(writes);
    item.labelAddress = labelAddress;
    item.labelBytes = std::move(labelBytes);
    item.kind = CommitKind::Flip;
    item.quiet = true;
    enqueue(std::move(item));
}

void DrawPipeline::EnqueueSuspend(Commit commit) {
    Item item;
    item.commit = std::move(commit);
    item.kind = CommitKind::Suspend;
    item.quiet = true;
    enqueue(std::move(item));
}

namespace {
struct ItemTurnSlot {};
using ItemTurn = HostThreadSlot<bool, ItemTurnSlot>;
std::atomic<std::uint64_t> itemsByTurn{0};
struct FlipCommitSlot {};
using FlipCommitting = HostThreadSlot<bool, FlipCommitSlot>;
}

DrawPipeline::FlipCommitScope::FlipCommitScope() : previous(FlipCommitting::Get()) {
    FlipCommitting::Set(true);
}

DrawPipeline::FlipCommitScope::~FlipCommitScope() {
    FlipCommitting::Set(previous);
}

namespace {
struct OrderProbeSlot {};
using OrderProbing = HostThreadSlot<std::uint32_t, OrderProbeSlot>;
}

DrawPipeline::OrderProbe::OrderProbe(FrameCheck check) : previous(OrderProbing::Get()) {
    OrderProbing::Set(static_cast<std::uint32_t>(check) + 1u);
}

DrawPipeline::OrderProbe::~OrderProbe() {
    OrderProbing::Set(previous);
}

void DrawFinisher::verifyNamedWrites(const DrawPipeline::Item& item) {
    const auto seq = FinisherSeq::Get();
    if (seq == 0 || lean == nullptr || !ringMode.load(std::memory_order_acquire)) return;
    const auto& slot = leanSlot(seq);
    if (slot.seq.load(std::memory_order_acquire) != seq || !slot.pending.named) return;
    const bool walked = slot.pending.origin == 4 && slot.pending.originKind != 0;
    if (!walked && !FrameVerifyDue(FrameCheck::HeldWrites)) return;
    const auto& named = slot.pending.writes;
    const auto covered = [&](const Range& range) {
        if (range.first >= range.second) return true;
        return std::any_of(named.begin(), named.end(), [&](const Range& held) { return held.first <= range.first && range.second <= held.second; });
    };
    std::optional<Range> missing;
    for (const auto& write : item.writes) {
        if (covered(write)) continue;
        missing = write;
        break;
    }
    if (!missing.has_value() && item.labelAddress != 0 && !item.labelBytes.empty()) {
        const Range label{item.labelAddress, item.labelAddress + item.labelBytes.size()};
        if (!covered(label)) missing = label;
    }
    char text[320];
    std::snprintf(text, sizeof(text), "the draw job %llu, published with %zu write ranges named on the worker%s, queued a %s item that writes [0x%llx, 0x%llx) outside them", static_cast<unsigned long long>(seq), named.size(), walked ? " by a resource walk" : "", KindNames[static_cast<std::size_t>(item.kind)], static_cast<unsigned long long>(missing.has_value() ? missing->first : 0), static_cast<unsigned long long>(missing.has_value() ? missing->second : 0));
    NoteFrameCheck(FrameCheck::HeldWrites, missing.has_value(), text);
}

void DrawPipeline::enqueue(Item item) {
    if (!DrawFinisher::OnFinisherThread()) DrawFinisher::Get().FlushOpenRun();
    else if (DrawFinisher::FrameVerifyEvery() != 0) DrawFinisher::Get().verifyNamedWrites(item);
    if (const auto probe = OrderProbing::Get(); probe != 0 && !DrawFinisher::OnFinisherThread()) {
        OrderProbing::Set(0);
        const auto check = static_cast<FrameCheck>(probe - 1u);
        const auto jobs = DrawFinisher::Get().Issued();
        const auto before = enqueuedItems.load(std::memory_order_acquire);
        item.commit = [this, inner = std::move(item.commit), check, jobs, before] {
            const auto front = DrawFinisher::Get().Front();
            const auto retiredNow = Retired();
            char text[256];
            std::snprintf(text, sizeof(text), "an item queued with the finisher at job %llu and %llu committer items before it committed with the front at %llu and %llu items retired", static_cast<unsigned long long>(jobs), static_cast<unsigned long long>(before), static_cast<unsigned long long>(front), static_cast<unsigned long long>(retiredNow));
            DrawFinisher::NoteFrameCheck(check, front < jobs || retiredNow < before, text);
            inner();
        };
    }
    if (depositVerifyEvery() != 0 || (DrawFinisher::FrameVerifyEvery() != 0 && DrawFinisher::FrameFlow())) item.order = orderKey();
    if (DrawFinisher::OnFinisherThread()) {
        auto& finisher = DrawFinisher::Get();
        if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FinisherDeposit) && finisher.deposit(*this, item)) return;
        finisher.AwaitTurn();
    }
    if (DrawFinisher::Busy() && !DrawFinisher::OnFinisherThread()) {
        const bool flow = DrawFinisher::Flow();
        const bool deep = DrawFinisher::Deep();
        const bool dispatchTurn = item.kind == CommitKind::Dispatch && (flow || deep || Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DispatchBeside));
        if ((item.reads.empty() || dispatchTurn || flow || deep) && !ItemTurn::Get() && FinisherProducer::Get() && DrawFinisher::Overlap() && !GuestMemory::GpuMutex().HeldByThisThread() && (dispatchTurn || flow || deep || Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::ItemsByTurn))) {
            DrawFinisher::Pending held;
            held.unknown = false; held.origin = 2;
            held.originKind = static_cast<std::uint8_t>(item.kind);
            held.writes = item.writes;
            held.labelAddress = item.labelAddress;
            held.labelBytes = item.labelBytes;
            held.labelRun = item.labelRun;
            itemsByTurn.fetch_add(1, std::memory_order_relaxed);
            if (deep) DrawFinisher::NoteDeepSite(item.reads.empty() ? DeepSite::ItemsByTurn : DeepSite::ItemsWithReadsByTurn);
            const auto parked = item.reads.empty() || flow ? std::uint64_t{0} : DrawFinisher::Get().ParkReads(item.reads);
            if (deep) item.parked = parked;
            const auto unparkAtRun = deep ? std::uint64_t{0} : parked;
            auto shared = std::make_shared<Item>(std::move(item));
            DrawFinisher::Get().Enqueue([this, shared, parked = unparkAtRun] {
                struct Scope {
                    bool previous = ItemTurn::Get();
                    std::uint64_t parked;
                    explicit Scope(std::uint64_t id) : parked(id) { ItemTurn::Set(true); }
                    ~Scope() {
                        ItemTurn::Set(previous);
                        if (parked != 0) DrawFinisher::Get().UnparkReads(parked);
                    }
                } scope{parked};
                enqueue(std::move(*shared));
            }, std::move(held));
            return;
        }
        DrawFinisher::NoteDeepKept(DeepKept::Other);
        DrawFinisher::Get().Drain();
    }
    queueItem(std::move(item));
}

void DrawPipeline::queueItem(Item item) {
    struct Unpark {
        std::uint64_t id;
        ~Unpark() {
            if (id != 0) DrawFinisher::Get().UnparkReads(id);
        }
    } unpark{std::exchange(item.parked, 0)};
    for (;;) {
        if (commitRing.load(std::memory_order_acquire) && ringEnqueue(item)) return;
        if (queueLocked(item)) return;
    }
}

bool DrawPipeline::queueLocked(Item& item) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    struct ReclaimStorage {};
    auto& reclaim = HostThreadLocal<std::vector<Item>, ReclaimStorage>();
    const ClearOnExit<Item> clearReclaim{reclaim};
    std::vector<Item> stranded;
    const auto lockStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto lock = lockFor(mutex);
    if (commitRing.load(std::memory_order_relaxed)) return false;
    if (!retired.empty()) std::swap(retired, reclaim);
    if (profile) lockWaitNs += elapsedNs(lockStart);
    rethrowFailure();
    if (closing) {
        lock.unlock();
        item.commit();
        return true;
    }
    const auto roomLimit = roomLimitNow();
    if (items.size() >= roomLimit) {
        if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline full while the enqueuing thread holds the GPU lock");
        const TailWatch watch(true, TailExtra::QueueRoom);
        const auto start = std::chrono::steady_clock::now();
        const ThreadReport::WorkerStateScope room(ThreadReport::WorkerState::RoomCommitter, ThreadReport::OnQueue0Worker());
        resumeAt = roomLimit - 1;
        ++idleWaiters;
        idle.wait(lock, [&] { return items.size() <= resumeAt; });
        --idleWaiters;
        resumeAt = 0;
        const auto waited = elapsedNs(start);
        fullWaitNs += waited;
        if (ThreadReport::Enabled()) {
            const bool finisher = DrawFinisher::OnFinisherThread();
            const bool forwarding = finisher && FinisherSeq::Get() == 0;
            const bool worker = !finisher && ThreadReport::OnQueue0Worker();
            ThreadReport::Add(forwarding ? ThreadReport::Count::FinisherForwardRoomWaits : finisher ? ThreadReport::Count::FinisherRoomWaits : worker ? ThreadReport::Count::WorkerCommitterRoomWaits : ThreadReport::Count::OtherRoomWaits, 1);
            ThreadReport::Add(forwarding ? ThreadReport::Count::FinisherForwardRoomNs : finisher ? ThreadReport::Count::FinisherRoomNs : worker ? ThreadReport::Count::WorkerCommitterRoomNs : ThreadReport::Count::OtherRoomNs, waited);
        }
        rethrowFailure();
        if (commitRing.load(std::memory_order_relaxed)) return false;
    }
    if (items.empty()) {
        const bool deep = DrawFinisher::Deep();
        if (deepQueue.load(std::memory_order_relaxed) != deep) latchDeepQueue(deep);
        if (deep && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitRing)) {
            std::swap(stranded, retired);
            latchRing();
            return false;
        }
    }
    depthSum += items.size();
    ++enqueued;
    heldReadRanges.Add(item.reads.size());
    raiseQueued(Graphics::WriteValidation::Source::QueueItem, item.writes);
    if (deepQueue.load(std::memory_order_relaxed)) {
        item.counted = false;
        const auto stamp = enqueuedItems.load(std::memory_order_relaxed) + 1;
        (item.store ? *deepStores : *deepOthers).Insert(item.writes, stamp, retiredItems.load(std::memory_order_relaxed) + 1);
        if (!item.reads.empty()) noteDeepReads(item.reads, stamp);
    } else {
        item.counted = !DrawFinisher::Flow() || queuedWrites.size() <= PublishedRanges::Capacity;
        if (item.counted) countWrites(item.writes, true);
    }
    raiseQueued(Graphics::WriteValidation::Source::QueueItem, item.writes);
    enqueuedItems.fetch_add(1, std::memory_order_release);
    items.push_back(std::move(item));
    if (pipelineStats()) deepestItems = std::max(deepestItems, items.size());
    outstanding.fetch_add(1, std::memory_order_release);
    startCommitter();
    if (committerWaiting) {
        ++wakes;
        wake.notify_one();
    }
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!profile && !stats) return true;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport >= std::chrono::seconds(10)) report(now);
    return true;
}

void DrawPipeline::startCommitter() {
    if (thread.joinable()) return;
    thread = Libc::StartHostThread([this] {
        GuestMemory::MarkProofThread();
        RaiseWorkerThreadPriority("draw committer");
        PlaceReservedThread("draw committer");
        run();
    });
    runningCommitters.fetch_add(1, std::memory_order_acq_rel);
}

std::size_t DrawPipeline::roomLimitNow() const {
    const std::size_t deepLimit = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DeepShallow) && !DrawFinisher::Flow() ? static_cast<std::size_t>(shallowItems()) : DrawFinisher::LeanSlots;
    return DrawFinisher::Flow() || (DrawFinisher::Deep() && deepQueue.load(std::memory_order_relaxed)) ? deepLimit : deepQueues() ? std::max<std::size_t>(Depth(), 32) : Depth();
}

DrawPipeline::Item& DrawPipeline::ringSlot(std::uint64_t ordinal) {
    return (*ring)[static_cast<std::size_t>(ordinal & (CommitRingSlots - 1))];
}

void DrawPipeline::latchRing() {
    if (commitRing.load(std::memory_order_relaxed)) return;
    if (ring == nullptr) ring = std::make_unique<std::array<Item, CommitRingSlots>>();
    startCommitter();
    const RingLock ringLock(*this);
    ringReclaimed = enqueuedItems.load(std::memory_order_relaxed);
    commitRing.store(true, std::memory_order_seq_cst);
    bumpRing(RingCount::Latches);
    if (committerWaiting) {
        ++wakes;
        wake.notify_one();
    }
}

void DrawPipeline::unlatchRing() {
    std::vector<Item> released;
    const auto lock = lockFor(mutex);
    const RingLock ringLock(*this);
    if (!commitRing.load(std::memory_order_relaxed) || ringWanted()) return;
    const auto retiredNow = retiredItems.load(std::memory_order_seq_cst);
    if (retiredNow != enqueuedItems.load(std::memory_order_relaxed)) return;
    reclaimRing(retiredNow, released);
    commitRing.store(false, std::memory_order_seq_cst);
    bumpRing(RingCount::Unlatches);
    if (committerWaiting) {
        ++wakes;
        wake.notify_one();
    }
}

void DrawPipeline::reclaimRing(std::uint64_t upTo, std::vector<Item>& out) {
    for (; ringReclaimed < upTo; ++ringReclaimed) {
        auto& slot = ringSlot(ringReclaimed);
        out.push_back(std::move(slot));
        slot.commit = nullptr;
    }
}

bool DrawPipeline::ringEmpty() const {
    const auto retiredNow = retiredItems.load(std::memory_order_seq_cst);
    return retiredNow == enqueuedItems.load(std::memory_order_seq_cst);
}

void DrawPipeline::lowerRingWake(std::uint64_t target) {
    if (target < ringWakeAt.load(std::memory_order_relaxed)) ringWakeAt.store(target, std::memory_order_seq_cst);
}

bool DrawPipeline::queueEmptyLocked() {
    if (!commitRing.load(std::memory_order_relaxed)) return items.empty();
    const auto drained = [&] { return ringEmpty() && outstanding.load(std::memory_order_seq_cst) == 0; };
    if (drained()) return true;
    lowerRingWake(enqueuedItems.load(std::memory_order_seq_cst));
    return drained();
}

void DrawPipeline::raiseReadsLive(std::uint64_t live) {
    auto current = readsLive.load(std::memory_order_relaxed);
    while (current < live && !readsLive.compare_exchange_weak(current, live, std::memory_order_release, std::memory_order_relaxed)) {
    }
}

void DrawPipeline::awaitRingRoom(std::uint64_t resumeFill) {
    const auto room = [&] {
        const auto retiredNow = retiredItems.load(std::memory_order_seq_cst);
        return enqueuedItems.load(std::memory_order_seq_cst) - retiredNow <= resumeFill || !commitRing.load(std::memory_order_seq_cst);
    };
    ringIdleWaiters.fetch_add(1, std::memory_order_seq_cst);
    if (!SpinUntil(HandoffSpinWindow(), room)) {
        auto lock = lockFor(mutex);
        for (;;) {
            if (commitRing.load(std::memory_order_relaxed)) {
                const auto end = enqueuedItems.load(std::memory_order_seq_cst);
                lowerRingWake(end > resumeFill ? end - resumeFill : 0);
            }
            if (room()) break;
            idle.wait(lock);
        }
    }
    ringIdleWaiters.fetch_sub(1, std::memory_order_seq_cst);
}

bool DrawPipeline::ringEnqueue(Item& item) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    struct RingReclaimStorage {};
    auto& reclaim = HostThreadLocal<std::vector<Item>, RingReclaimStorage>();
    const ClearOnExit<Item> clearReclaim{reclaim};
    std::optional<TailWatch> watch;
    std::optional<ThreadReport::WorkerStateScope> room;
    auto roomStart = std::chrono::steady_clock::time_point{};
    bool waited = false;
    for (;;) {
        const auto lockStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        RingLock lock(*this);
        if (!commitRing.load(std::memory_order_relaxed)) return false;
        if (profile) lockWaitNs += elapsedNs(lockStart);
        if (failurePending.load(std::memory_order_acquire)) {
            lock.Unlock();
            {
                const auto held = lockFor(mutex);
                rethrowFailure();
            }
            continue;
        }
        if (!waited && ringClosing.load(std::memory_order_acquire)) {
            lock.Unlock();
            item.commit();
            return true;
        }
        const auto end = enqueuedItems.load(std::memory_order_relaxed);
        const auto retiredNow = retiredItems.load(std::memory_order_seq_cst);
        if (retiredNow == end && !ringWanted()) {
            lock.Unlock();
            unlatchRing();
            return false;
        }
        const auto roomLimit = roomLimitNow();
        if (end - retiredNow >= roomLimit) {
            if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline full while the enqueuing thread holds the GPU lock");
            if (!waited) {
                waited = true;
                watch.emplace(true, TailExtra::QueueRoom);
                roomStart = std::chrono::steady_clock::now();
                room.emplace(ThreadReport::WorkerState::RoomCommitter, ThreadReport::OnQueue0Worker());
            }
            lock.Unlock();
            awaitRingRoom(static_cast<std::uint64_t>(roomLimit - 1));
            continue;
        }
        if (waited) {
            room.reset();
            watch.reset();
            const auto waitedNs = elapsedNs(roomStart);
            fullWaitNs += waitedNs;
            bumpRing(RingCount::RoomWaits);
            bumpRing(RingCount::RoomWaitNs, waitedNs);
            if (ThreadReport::Enabled()) {
                const bool finisher = DrawFinisher::OnFinisherThread();
                const bool forwarding = finisher && FinisherSeq::Get() == 0;
                const bool worker = !finisher && ThreadReport::OnQueue0Worker();
                ThreadReport::Add(forwarding ? ThreadReport::Count::FinisherForwardRoomWaits : finisher ? ThreadReport::Count::FinisherRoomWaits : worker ? ThreadReport::Count::WorkerCommitterRoomWaits : ThreadReport::Count::OtherRoomWaits, 1);
                ThreadReport::Add(forwarding ? ThreadReport::Count::FinisherForwardRoomNs : finisher ? ThreadReport::Count::FinisherRoomNs : worker ? ThreadReport::Count::WorkerCommitterRoomNs : ThreadReport::Count::OtherRoomNs, waitedNs);
            }
        } else {
            bumpRing(RingCount::Enqueues);
        }
        reclaimRing(retiredNow, reclaim);
        const auto queued = static_cast<std::size_t>(end - retiredNow);
        depthSum += queued;
        ++enqueued;
        heldReadRanges.Add(item.reads.size());
        raiseQueued(Graphics::WriteValidation::Source::QueueItem, item.writes);
        item.counted = false;
        const auto stamp = end + 1;
        (item.store ? *deepStores : *deepOthers).Insert(item.writes, stamp, retiredNow + 1);
        if (!item.reads.empty()) noteDeepReads(item.reads, stamp);
        item.ringOrdinal = stamp;
        auto& slot = ringSlot(end);
        slot = std::move(item);
        if (pipelineStats()) {
            deepestItems = std::max(deepestItems, queued + 1);
            if (ringHighest.load(std::memory_order_relaxed) < queued + 1) ringHighest.store(queued + 1, std::memory_order_relaxed);
        }
        outstanding.fetch_add(1, std::memory_order_seq_cst);
        enqueuedItems.store(stamp, std::memory_order_seq_cst);
        raiseQueued(Graphics::WriteValidation::Source::QueueItem, slot.writes);
        lock.Unlock();
        if (committerSleeping.load(std::memory_order_seq_cst) && committerSleeping.exchange(false, std::memory_order_seq_cst)) {
            const auto held = lockFor(mutex);
            ++wakes;
            wake.notify_one();
        }
        if (profile || stats) ringReport();
        return true;
    }
}

void DrawPipeline::ringReport() {
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() < ringReportAt.load(std::memory_order_relaxed)) return;
    const auto lock = lockFor(mutex);
    const RingLock ringLock(*this);
    if (now - lastReport >= std::chrono::seconds(10)) report(now);
    ringReportAt.store(std::chrono::duration_cast<std::chrono::nanoseconds>((lastReport + std::chrono::seconds(10)).time_since_epoch()).count(), std::memory_order_relaxed);
}

std::uint64_t DrawPipeline::RingTotal(RingCount count) {
    return ringTotals[static_cast<std::size_t>(count)].value.load(std::memory_order_relaxed);
}

std::string DrawPipeline::ringText() {
    std::array<std::uint64_t, static_cast<std::size_t>(RingCount::Count)> window{};
    for (std::size_t index = 0; index < window.size(); ++index) {
        const auto total = ringTotals[index].value.load(std::memory_order_relaxed);
        window[index] = total - ringReported[index];
        ringReported[index] = total;
    }
    const auto at = [&](RingCount count) { return static_cast<unsigned long long>(window[static_cast<std::size_t>(count)]); };
    std::string scans;
    for (std::size_t index = 0; index < ringScanTotals.size(); ++index) {
        const auto total = ringScanTotals[index].load(std::memory_order_relaxed);
        const auto count = total - ringScanReported[index];
        ringScanReported[index] = total;
        if (count == 0) continue;
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu", RingScanNames[index], static_cast<unsigned long long>(count));
        scans += item;
    }
    char text[2048];
    std::snprintf(text, sizeof(text), "switch %s, latch %s, ring of %zu slots; pops without the lock %llu, with the lock while a scan ran %llu, on the locked queue %llu; ring enqueues without a room wait %llu, after one %llu (%.1f ms); enqueuer contention %llu; committer sleeps %llu (%.1f ms asleep); waiter wakes %llu; latched %llu, released %llu; scans on the slow path %llu by caller:%s; highest fill %llu; verify: pops checked %llu, scans checked %llu, differing %llu",
        Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::CommitRing) ? "set" : "unset", commitRing.load(std::memory_order_relaxed) ? "on" : "off", CommitRingSlots, at(RingCount::FastPops), at(RingCount::LockedPops), at(RingCount::MutexPops), at(RingCount::Enqueues), at(RingCount::RoomWaits), static_cast<double>(window[static_cast<std::size_t>(RingCount::RoomWaitNs)]) / 1e6, at(RingCount::Contention), at(RingCount::Sleeps), static_cast<double>(window[static_cast<std::size_t>(RingCount::SleepNs)]) / 1e6, at(RingCount::WaiterWakes), at(RingCount::Latches), at(RingCount::Unlatches), at(RingCount::SlowScans), scans.empty() ? " none" : scans.c_str(), static_cast<unsigned long long>(ringHighest.exchange(0, std::memory_order_relaxed)), at(RingCount::VerifyPops), at(RingCount::VerifyScans), at(RingCount::VerifyDiffering));
    return text;
}

void DrawPipeline::report(std::chrono::steady_clock::time_point now) {
    lastReport = now;
    const auto ringCommitted = ringCommits.load(std::memory_order_relaxed);
    const auto ringSpent = ringCommitNs.load(std::memory_order_relaxed);
    const auto ringFailed = ringCommitErrors.load(std::memory_order_relaxed);
    commits += ringCommitted - reportedRingCommits;
    commitNs += ringSpent - reportedRingCommitNs;
    commitErrors += ringFailed - reportedRingCommitErrors;
    reportedRingCommits = ringCommitted;
    reportedRingCommitNs = ringSpent;
    reportedRingCommitErrors = ringFailed;
    std::string drainText;
    for (std::size_t i = 0; i < drains.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f ms)", DrainNames[i], static_cast<unsigned long long>(drains[i]), static_cast<double>(drainWaitNs[i]) / 1e6);
        drainText += item;
    }
    drainText += "; packet drains by opcode:";
    for (int shown = 0; shown < 5; ++shown) {
        const auto top = std::max_element(drainOpcodes.begin(), drainOpcodes.begin() + 0x100);
        if (*top == 0) break;
        char item[32];
        std::snprintf(item, sizeof(item), " 0x%02x x%llu", static_cast<unsigned>(top - drainOpcodes.begin()), static_cast<unsigned long long>(*top));
        drainText += item;
        *top = 0;
    }
    drainOpcodes = {};
    const auto perCommit = commits == 0 ? 0.0 : static_cast<double>(commitNs) / 1e3 / static_cast<double>(commits);
    const auto perDraw = enqueued == 0 ? 0.0 : static_cast<double>(fullWaitNs) / 1e3 / static_cast<double>(enqueued);
    const auto queued = enqueued == 0 ? 0.0 : static_cast<double>(depthSum) / static_cast<double>(enqueued);
    const auto decodeDrainCount = decodeDrains.load(std::memory_order_relaxed);
    const auto depositCount = depositedItems.load(std::memory_order_relaxed);
    const auto turnCount = turnItems.load(std::memory_order_relaxed);
    char reads[256];
    std::snprintf(reads, sizeof(reads), " %llu read ranges, %llu stores held, %llu decode drains, %llu items by turn, %llu finisher items deposited, %llu queued at their turn", static_cast<unsigned long long>(heldReadRanges.Sum()), static_cast<unsigned long long>(heldReadHits.Take()), static_cast<unsigned long long>(decodeDrainCount - reportedDecodeDrains), static_cast<unsigned long long>(itemsByTurn.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(depositCount - reportedDeposits), static_cast<unsigned long long>(turnCount - reportedTurnItems));
    reportedDecodeDrains = decodeDrainCount;
    reportedDeposits = depositCount;
    reportedTurnItems = turnCount;
    if (depositVerifyEvery() != 0) AgcDriver::ProfilePrint_nid_no_patch("[finisher-deposit-verify] 10 s: every %llu; items checked %llu, out of order %llu; drains checked %llu, with deposits left %llu\n", static_cast<unsigned long long>(depositVerifyEvery()), static_cast<unsigned long long>(orderChecks.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(orderDifferences.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(drainChecks.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(drainDifferences.exchange(0, std::memory_order_relaxed)));
    AgcDriver::ProfilePrint_nid_no_patch("[draw] pipelined (10 s, depth %zu): %llu commits at %.1f us each on the committer, %llu failed; %llu enqueued behind %.2f on average, waiting %.1f us per draw for room, %.2f us for the lock; %llu committer wakes;%s; drains:%s\n", Depth(), static_cast<unsigned long long>(commits), perCommit, static_cast<unsigned long long>(commitErrors), static_cast<unsigned long long>(enqueued), queued, perDraw, enqueued == 0 ? 0.0 : static_cast<double>(lockWaitNs) / 1e3 / static_cast<double>(enqueued), static_cast<unsigned long long>(wakes), reads, drainText.c_str());
    if (pipelineStats()) AgcDriver::ProfilePrint_nid_no_patch("[xp-flow] 10 s, %s; committer queue: highest %zu items, mean %.2f items ahead of an enqueued one\n", DrawFinisher::FlowText().c_str(), deepestItems, queued);
    if (pipelineStats()) AgcDriver::ProfilePrint_nid_no_patch("[deep-flow] 10 s, %s; committer queue: latch %s, highest %zu items\n", DrawFinisher::DeepText().c_str(), deepQueue.load(std::memory_order_relaxed) ? "on" : "off", deepestItems);
    if (pipelineStats()) AgcDriver::ProfilePrint_nid_no_patch("[frame-flow] 10 s, %s\n", DrawFinisher::FrameText().c_str());
    if (pipelineStats()) AgcDriver::ProfilePrint_nid_no_patch("[commit-ring] 10 s: %s\n", ringText().c_str());
    deepestItems = 0;
    if (const auto encoded = Graphics::CommandEncoders::TotalsText(); !encoded.empty()) AgcDriver::ProfilePrint_nid_no_patch("[encode] (cumulative): %s\n", encoded.c_str());
    AgcDriver::ProfilePrint_nid_no_patch("[indirect-draws] (cumulative): %s\n", Graphics::IndirectDrawTotalsText().c_str());
    // The census of what moves template proofs (APS5_STAGE_STATS counts them): breaks by site, why
    // the epoch shortcut missed, and the objects taken from another template's proof.
    static const bool census = std::getenv("APS5_STAGE_STATS") != nullptr;
    if (census) AgcDriver::ProfilePrint_nid_no_patch("[proof-breaks] 10 s: BreakProofs by site:%s; epoch shortcut misses: %s; fast-proof stamps taken so far %llu; %s\n", GuestMemory::TakeProofBreakReport().c_str(), Graphics::ShaderResources::TakeEpochMissReport().c_str(), static_cast<unsigned long long>(Graphics::ShaderResources::FastProofStampsTaken()), GuestMemory::SweepCostReport().c_str());
    std::string kindText;
    for (std::size_t i = 0; i < kindCommits.size(); ++i) {
        const auto count = kindCommits[i].exchange(0, std::memory_order_relaxed);
        const auto ns = kindNs[i].exchange(0, std::memory_order_relaxed);
        if (count == 0) continue;
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f us)", KindNames[i], static_cast<unsigned long long>(count), static_cast<double>(ns) / 1e3 / static_cast<double>(count));
        kindText += item;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[commits] by kind (10 s):%s\n", kindText.c_str());
    if (LabelRunCensus::Enabled()) {
        if (const auto labelRuns = LabelRunCensus::Take(); !labelRuns.empty()) AgcDriver::ProfilePrint_nid_no_patch("[label-runs] 10 s: %s\n", labelRuns.c_str());
    }
    if (const auto lookups = Graphics::TakeFullDrawLookupText(); !lookups.empty()) AgcDriver::ProfilePrint_nid_no_patch("[commits] full draws by the template lookup (10 s, on any thread):%s\n", lookups.c_str());
    if (const auto steps = Graphics::TakeFullDrawStepText(); !steps.empty()) AgcDriver::ProfilePrint_nid_no_patch("[commits] full draws with a template hit by cause, microseconds per draw by step (10 s, on any thread):%s\n", steps.c_str());
    AgcDriver::ProfilePrint_nid_no_patch("[record-heavy-draws] 10 s, %s\n", Graphics::TakeRecordHeavyText().c_str());
    if (const auto builds = Graphics::TakeTemplateBuildText(); !builds.empty()) AgcDriver::ProfilePrint_nid_no_patch("[template-builds] 10 s, %s\n", builds.c_str());
    if (Graphics::ResourceCache::FateCensus()) AgcDriver::ProfilePrint_nid_no_patch("[template-keeps] 10 s, %s\n", Graphics::TakeTemplateKeepText().c_str());
    if (Graphics::ResourceCache::FateCensus()) AgcDriver::ProfilePrint_nid_no_patch("[view-template-keeps] 10 s, %s\n", Graphics::TakeViewTemplateKeepText().c_str());
    AgcDriver::ProfilePrint_nid_no_patch("[committer-idle] 10 s, by the finisher's state when the idle began:%s\n", committerIdleText().c_str());
    const auto vk = Graphics::TakeVkDrawCounts();
    AgcDriver::ProfilePrint_nid_no_patch("[vkdraw] (10 s): %llu vkCmdDraw, %llu vkCmdDrawIndexed, %llu vkCmdDrawIndirect calls, %llu mesh; draw keys %llu, same as the previous draw %llu, seen within the last 4096 draws %llu; draw shapes (state without per-draw user words) same as the previous %llu, seen within the last 4096 %llu\n", static_cast<unsigned long long>(vk.direct), static_cast<unsigned long long>(vk.indexed), static_cast<unsigned long long>(vk.indirect), static_cast<unsigned long long>(vk.mesh), static_cast<unsigned long long>(drawKeys.Take()), static_cast<unsigned long long>(drawKeysSameAsPrevious.Take()), static_cast<unsigned long long>(drawKeysSeenRecently.Take()), static_cast<unsigned long long>(drawShapesSameAsPrevious.Take()), static_cast<unsigned long long>(drawShapesSeenRecently.Take()));
    commits = commitNs = commitErrors = enqueued = fullWaitNs = depthSum = wakes = lockWaitNs = 0;
    drains = {};
    drainWaitNs = {};
}


void DrawPipeline::run() {
    NameDriverThread("draw-commit");
    PerformanceTrace::BindThread("draw committer");
#ifndef _WIN32
    ReserveCommitterCore();
#endif
    StartCommitterSampler();
    // A run of draws under one GPU lock hold (APS5_NO_LEAN_RUNS turns it off): the hold that
    // commits one draw stays for the next while the next item is a draw already queued, up to
    // LeanRunCap draws and while no other thread is blocked on the lock. A non-draw item, an empty
    // queue, a failure and a waiting thread end the run. Each draw still takes its own (nested)
    // hold, labels and submit check in commitDraw.
    constexpr std::size_t LeanRunCap = 32;
    std::optional<std::unique_lock<GuestMemory::GpuMutexType>> runHold;
    std::size_t runLength = 0;
    const auto dropHold = [&](Graphics::ExpressRunEnd reason) {
        if (!runHold) return;
        Graphics::EndExpressRun(reason);
        runHold.reset();
    };
    ThreadReport::Clock::time_point lastEnd{};
    ThreadReport::StateTimes idleFrom{};
    Commit* commit = nullptr;
    bool isDraw = false;
    bool rangedItem = false;
    bool quietItem = false;
    std::uint32_t itemParts = 1;
    std::vector<Range> itemRanges;
    bool taken = false;
    auto previousKind = CommitKind::Count;
    bool ringTurn = false;
    std::uint64_t head = 0;
    std::uint64_t lastPopped = 0;
    std::uint64_t ringPops = 0;
    const auto takeItem = [&](Item& item) {
        if (depositVerifyEvery() != 0 || (DrawFinisher::FrameVerifyEvery() != 0 && DrawFinisher::FrameFlow())) verifyOrder(item.order);
        commit = stopped ? nullptr : &item.commit;
        isDraw = item.draw;
        quietItem = item.quiet;
        itemParts = item.parts;
        rangedItem = !isDraw && GuestMemory::RangeProofs();
        itemRanges.clear();
        if (rangedItem) {
            itemRanges = item.writes;
            if (item.labelAddress != 0 && !item.labelBytes.empty()) itemRanges.emplace_back(item.labelAddress, item.labelAddress + item.labelBytes.size());
        }
    };
    const auto takeFront = [&] { takeItem(items.front()); };
    struct AheadHint {
        Ahead ahead = nullptr;
        const void* object = nullptr;
    };
    std::array<AheadHint, 3> aheadHints{};
    bool aheadOn = false;
    const bool aheadCensus = Graphics::CommitFeedCensus();
    const auto noteAhead = [&](const auto& itemAt, std::uint64_t available) {
        aheadOn = Graphics::CommitFeedOn(Graphics::CommitFeedAhead);
        if (!aheadOn) return;
        for (std::size_t distance = 1; distance <= aheadHints.size(); ++distance) {
            auto& hint = aheadHints[distance - 1];
            hint = {};
            if (distance >= available) continue;
            const Item& ahead = itemAt(distance);
            hint.ahead = ahead.ahead;
            hint.object = ahead.aheadObject;
            if (aheadCensus) Graphics::NoteCommitFeed(hint.ahead != nullptr ? Graphics::CommitFeedCount::AheadEntries : Graphics::CommitFeedCount::AheadNoHint);
        }
    };
    const auto noteAheadQueued = [&] { noteAhead([&](std::size_t distance) -> const Item& { return items[distance]; }, static_cast<std::uint64_t>(items.size())); };
    const auto noteAheadRing = [&] {
        const auto enqueuedNow = enqueuedItems.load(std::memory_order_acquire);
        noteAhead([&](std::size_t distance) -> const Item& { return ringSlot(head + distance); }, enqueuedNow > head ? enqueuedNow - head : 0u);
    };
    const auto stepAhead = [&](bool after) {
        for (std::size_t distance = 1; distance <= aheadHints.size(); ++distance) {
            const auto& hint = aheadHints[distance - 1];
            if (hint.ahead == nullptr) continue;
            const auto level = static_cast<std::uint32_t>(2u * (aheadHints.size() - distance) + (after ? 1u : 0u));
            hint.ahead(hint.object, level);
            if (aheadCensus) Graphics::NoteCommitFeed(static_cast<Graphics::CommitFeedCount>(static_cast<std::size_t>(Graphics::CommitFeedCount::AheadLevel0) + level));
        }
    };
    for (;;) {
        if (!taken) {
            static const bool idleCensus = std::getenv("APS5_PIPELINE_STATS") != nullptr;
            const bool idleNow = idleCensus && outstanding.load(std::memory_order_acquire) == 0;
            const unsigned idleState = idleNow ? DrawFinisher::Get().IdleState() : 0;
            const auto idleStart = idleNow ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (ringTurn) {
                const auto ready = [&] { return enqueuedItems.load(std::memory_order_seq_cst) != head || !commitRing.load(std::memory_order_seq_cst) || ringClosing.load(std::memory_order_seq_cst); };
                if (!ready()) SpinUntil(HandoffSpinWindow(), ready);
                if (!ready()) {
                    const auto asleep = std::chrono::steady_clock::now();
                    auto lock = lockFor(mutex);
                    committerWaiting = true;
                    bool slept = false;
                    for (;;) {
                        committerSleeping.store(true, std::memory_order_seq_cst);
                        if (closing || enqueuedItems.load(std::memory_order_seq_cst) != head || !commitRing.load(std::memory_order_relaxed)) break;
                        slept = true;
                        wake.wait(lock);
                    }
                    committerSleeping.store(false, std::memory_order_seq_cst);
                    committerWaiting = false;
                    if (slept) {
                        bumpOwnedRing(RingCount::Sleeps);
                        bumpOwnedRing(RingCount::SleepNs, elapsedNs(asleep));
                    }
                }
                if (idleNow) noteCommitterIdle(idleState, elapsedNs(idleStart));
                const bool arrived = enqueuedItems.load(std::memory_order_seq_cst) != head;
                if (!commitRing.load(std::memory_order_seq_cst)) {
                    ringTurn = false;
                    continue;
                }
                if (!arrived) {
                    if (ringClosing.load(std::memory_order_seq_cst) && enqueuedItems.load(std::memory_order_seq_cst) == head) break;
                    continue;
                }
                takeItem(ringSlot(head));
                noteAheadRing();
            } else {
                SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
                auto lock = lockFor(mutex);
                committerWaiting = true;
                wake.wait(lock, [&] { return !items.empty() || closing || commitRing.load(std::memory_order_relaxed); });
                committerWaiting = false;
                if (idleNow) noteCommitterIdle(idleState, elapsedNs(idleStart));
                if (items.empty()) {
                    if (!commitRing.load(std::memory_order_relaxed)) break;
                    ringTurn = true;
                    head = retiredItems.load(std::memory_order_relaxed);
                    lastPopped = head;
                    continue;
                }
                takeFront();
                noteAheadQueued();
            }
        }
        taken = false;
        const SampledHitch hitch(isDraw ? "committer draw" : "committer item");
        const auto start = std::chrono::steady_clock::now();
        const auto idleTo = ThreadReport::Enabled() && lastEnd != ThreadReport::Clock::time_point{} ? ThreadReport::SnapshotStates(start) : ThreadReport::StateTimes{};
        std::exception_ptr fatal;
        bool failed = false;
        bool shutdown = false;
        if (!isDraw || commit == nullptr || (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoLeanRuns) && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::RunFollow))) {
            dropHold(!isDraw ? Graphics::ExpressRunEnd::NotDraw : commit == nullptr ? Graphics::ExpressRunEnd::Stop : Graphics::ExpressRunEnd::Switch);
            if (!isDraw && commit != nullptr && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::RunFollow)) Graphics::NoteExpressRunItem();
        } else if (!runHold) {
            const bool holdTimed = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::RunFollow);
            const auto holdBegan = holdTimed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
            runHold.emplace(GuestMemory::GpuMutex());
            Graphics::BeginExpressRun();
            runLength = 0;
            if (holdTimed) Graphics::NoteExpressRunHold(elapsedNs(holdBegan));
        }
        if (aheadOn && commit != nullptr) stepAhead(false);
        try {
            if (commit != nullptr) (*commit)();
            if (quietItem) {
            } else if (rangedItem) {
                for (const auto& [begin, end] : itemRanges) GuestMemory::BreakProofs(GuestMemory::ProofBreak::NonDrawItem, begin, end);
            } else if (!isDraw) {
                GuestMemory::BreakProofs(GuestMemory::ProofBreak::NonDrawItem);
            }
        } catch (const std::exception&) {
            failed = true;
            fatal = std::current_exception();
        } catch (const ProcessShutdown&) {
            shutdown = true;
            fatal = std::current_exception();
        } catch (...) {
            fatal = std::current_exception();
        }
        if (aheadOn && commit != nullptr && fatal == nullptr) stepAhead(true);
        const auto spent = elapsedNs(start);
        if (ThreadReport::Enabled()) {
            const auto end = start + std::chrono::nanoseconds(spent);
            ThreadReport::Add(isDraw ? ThreadReport::Count::CommitterDraws : ThreadReport::Count::CommitterOther, itemParts);
            ThreadReport::Add(isDraw ? ThreadReport::Count::CommitterDrawNs : ThreadReport::Count::CommitterOtherNs, spent);
            ThreadReport::NoteCommitterDraw(ThreadReport::TakeItemMark(), spent);
            if (lastEnd != ThreadReport::Clock::time_point{}) {
                ThreadReport::Add(ThreadReport::Count::CommitterIdleNs, ThreadReport::Nanoseconds(lastEnd, start));
                ThreadReport::NoteCommitterIdle(idleFrom, idleTo);
            }
            lastEnd = end;
            idleFrom = ThreadReport::SnapshotStates(end);
            ThreadReport::Tick(end);
        }
        if (fatal != nullptr) dropHold(Graphics::ExpressRunEnd::Stop);
        if (fatal != nullptr && !stopped) {
            stopped = true;
            if (!shutdown) Driver::Get().ReportFailure(fatal);
        }
        bool keepRun = false;
        auto ended = Graphics::ExpressRunEnd::LockWaiter;
        const bool mergedTake = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::RunFollow) && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::RunFollowPlainLoop);
        if (ringTurn) {
            auto& slot = ringSlot(head);
            const auto kind = slot.kind;
            const auto kindIndex = static_cast<std::size_t>(kind);
            const bool counted = slot.counted;
            const auto previousPop = std::exchange(lastPopped, slot.ringOrdinal);
            if (const auto every = ringVerifyEvery(); every != 0 && ++ringPops % every == 0) {
                bumpOwnedRing(RingCount::VerifyPops);
                const auto retiredBefore = retiredItems.load(std::memory_order_seq_cst);
                const auto enqueuedNow = enqueuedItems.load(std::memory_order_seq_cst);
                char text[256];
                std::snprintf(text, sizeof(text), "the committer took the slot of item %llu holding item %llu after item %llu, with %llu items retired and %llu enqueued", static_cast<unsigned long long>(head + 1), static_cast<unsigned long long>(slot.ringOrdinal), static_cast<unsigned long long>(previousPop), static_cast<unsigned long long>(retiredBefore), static_cast<unsigned long long>(enqueuedNow));
                noteRingVerify("pop order", slot.ringOrdinal != head + 1 || slot.ringOrdinal != previousPop + 1 || retiredBefore != head || enqueuedNow <= head, text);
            }
            kindCommits[kindIndex].fetch_add(itemParts, std::memory_order_relaxed);
            kindNs[kindIndex].fetch_add(spent, std::memory_order_relaxed);
            if (LabelRunCensus::Enabled()) {
                if (kind == CommitKind::EndOfPipe) LabelRunCensus::NoteCommitted(previousKind, previousKind == CommitKind::Count, itemParts);
                previousKind = kind;
            }
            ringCommits.store(ringCommits.load(std::memory_order_relaxed) + itemParts, std::memory_order_relaxed);
            ringCommitNs.store(ringCommitNs.load(std::memory_order_relaxed) + spent, std::memory_order_relaxed);
            if (failed) ringCommitErrors.store(ringCommitErrors.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
            if (fatal != nullptr) {
                const auto lock = lockFor(mutex);
                if (failure == nullptr) failure = fatal;
                failurePending.store(true, std::memory_order_seq_cst);
            }
            committerRetiring.store(true, std::memory_order_seq_cst);
            const bool locked = counted || ringScans.load(std::memory_order_seq_cst) != 0;
            std::unique_lock<std::mutex> held;
            if (locked) {
                committerRetiring.store(false, std::memory_order_seq_cst);
                held = lockFor(mutex);
                if (counted) countWrites(slot.writes, false);
            }
            const auto retiredNow = head + 1;
            retiredItems.store(retiredNow, std::memory_order_seq_cst);
            raiseReadsLive(retiredNow + 1);
            outstanding.fetch_sub(1, std::memory_order_seq_cst);
            if (locked) held.unlock();
            else committerRetiring.store(false, std::memory_order_seq_cst);
            bumpOwnedRing(locked ? RingCount::LockedPops : RingCount::FastPops);
            head = retiredNow;
            if (ringWakeAt.load(std::memory_order_seq_cst) <= retiredNow) {
                const auto lock = lockFor(mutex);
                ringWakeAt.store(NoRingWake, std::memory_order_seq_cst);
                idle.notify_all();
                bumpOwnedRing(RingCount::WaiterWakes);
            }
            const bool queued = enqueuedItems.load(std::memory_order_acquire) != head && commitRing.load(std::memory_order_acquire);
            const bool closingNow = ringClosing.load(std::memory_order_acquire);
            keepRun = runHold.has_value() && !stopped && !closingNow && queued && ringSlot(head).draw && ++runLength < LeanRunCap && ringIdleWaiters.load(std::memory_order_relaxed) == 0;
            if (!keepRun) ended = stopped || closingNow ? Graphics::ExpressRunEnd::Stop : !queued ? Graphics::ExpressRunEnd::NotQueued : !ringSlot(head).draw ? Graphics::ExpressRunEnd::NotDraw : runLength >= LeanRunCap ? Graphics::ExpressRunEnd::Cap : Graphics::ExpressRunEnd::IdleWaiter;
            if (mergedTake && queued) {
                takeItem(ringSlot(head));
                noteAheadRing();
                taken = true;
            }
        } else {
            const auto lock = lockFor(mutex);
            const auto retiredNow = retiredItems.fetch_add(1, std::memory_order_release) + 1;
            if (readsLive.load(std::memory_order_relaxed) < retiredNow + 1) readsLive.store(retiredNow + 1, std::memory_order_release);
            if (items.front().counted) countWrites(items.front().writes, false);
            const auto kindIndex = static_cast<std::size_t>(items.front().kind);
            kindCommits[kindIndex].fetch_add(itemParts, std::memory_order_relaxed);
            kindNs[kindIndex].fetch_add(spent, std::memory_order_relaxed);
            if (LabelRunCensus::Enabled()) {
                if (items.front().kind == CommitKind::EndOfPipe) LabelRunCensus::NoteCommitted(previousKind, previousKind == CommitKind::Count, itemParts);
                previousKind = items.front().kind;
            }
            retired.push_back(std::move(items.front()));
            items.pop_front();
            commits += itemParts;
            commitNs += spent;
            if (failed) ++commitErrors;
            if (fatal != nullptr && failure == nullptr) failure = fatal;
            if (fatal != nullptr) failurePending.store(true, std::memory_order_release);
            bumpOwnedRing(RingCount::MutexPops);
            outstanding.fetch_sub(1, std::memory_order_release);
            if ((idleWaiters != 0 && (items.empty() || items.size() <= resumeAt)) || retireWaiters != 0) idle.notify_all();
            keepRun = runHold.has_value() && !stopped && !closing && !items.empty() && items.front().draw && ++runLength < LeanRunCap && idleWaiters == 0;
            if (!keepRun) ended = stopped || closing ? Graphics::ExpressRunEnd::Stop : items.empty() ? Graphics::ExpressRunEnd::NotQueued : !items.front().draw ? Graphics::ExpressRunEnd::NotDraw : runLength >= LeanRunCap ? Graphics::ExpressRunEnd::Cap : Graphics::ExpressRunEnd::IdleWaiter;
            if (mergedTake && !items.empty()) {
                takeFront();
                noteAheadQueued();
                taken = true;
            }
        }
        if (runHold && (!keepRun || GuestMemory::GpuMutex().HasWaiters())) dropHold(ended);
    }
    dropHold(Graphics::ExpressRunEnd::Stop);
    runningCommitters.fetch_sub(1, std::memory_order_acq_rel);
}

void DrawPipeline::verifyOrder(std::uint64_t order) {
    const auto every = depositVerifyEvery() != 0 ? depositVerifyEvery() : DrawFinisher::FrameVerifyEvery();
    if (order == 0 || every == 0 || DrawFinisher::Get().closed.load(std::memory_order_acquire) || ++orderOrdinal % every != 0) return;
    orderChecks.fetch_add(1, std::memory_order_relaxed);
    const bool differs = order <= verifiedOrder;
    if (differs && depositVerifyEvery() != 0) {
        orderDifferences.fetch_add(1, std::memory_order_relaxed);
        noteDepositDifference("item order differs: the committer took an item of " + orderText(order) + " after an item of " + orderText(verifiedOrder));
    }
    if (DrawFinisher::FrameVerifyEvery() != 0) DrawFinisher::NoteFrameCheck(FrameCheck::ItemOrder, differs, differs ? "the committer took an item of " + orderText(order) + " after an item of " + orderText(verifiedOrder) : std::string());
    verifiedOrder = order;
}

void DrawPipeline::Stop() {
    std::thread committer;
    {
        const auto lock = lockFor(mutex);
        closing = true;
        committer = std::move(thread);
        std::optional<RingLock> ringLock;
        if (commitRing.load(std::memory_order_relaxed)) ringLock.emplace(*this);
        ringClosing.store(true, std::memory_order_seq_cst);
    }
    wake.notify_all();
    if (!committer.joinable()) return;
    if (committer.get_id() == std::this_thread::get_id()) committer.detach();
    else committer.join();
}

void DrawPipeline::ReleaseRetired() {
    std::vector<Item> done;
    std::vector<Item> slots;
    const auto lock = lockFor(mutex);
    std::swap(done, retired);
    if (!commitRing.load(std::memory_order_relaxed)) return;
    const RingLock ringLock(*this);
    reclaimRing(retiredItems.load(std::memory_order_seq_cst), slots);
}

void DrawPipeline::Drain(DrainReason reason, std::uint32_t opcode) {
    if (DrawFinisher::OnFinisherThread()) DrawFinisher::Get().AwaitTurn();
    else DrawFinisher::Get().FlushOpenRun();
    drainRequests[std::min<std::uint32_t>(opcode, 0x100)].fetch_add(1, std::memory_order_relaxed);
    reasonRequests[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    drainAll(reason, opcode);
}

void DrawPipeline::drainAll(DrainReason reason, std::uint32_t opcode) {
    const bool finisherBusy = DrawFinisher::Busy() && !DrawFinisher::OnFinisherThread();
    if ((finisherBusy || Busy()) && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline drain under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    if (finisherBusy) DrawFinisher::Get().Drain();
    if (!Busy()) return;
    const TailWatch watch(true);
    const ThreadReport::WorkerStateScope draining(ThreadReport::WorkerState::DrainCommitter, ThreadReport::OnQueue0Worker());
    const auto start = std::chrono::steady_clock::now();
    SpinUntil(HandoffSpinWindow() * 2, [&] { return !Busy(); });
    auto lock = lockFor(mutex);
    ++idleWaiters;
    ringIdleWaiters.fetch_add(1, std::memory_order_seq_cst);
    while (!queueEmptyLocked()) idle.wait(lock);
    ringIdleWaiters.fetch_sub(1, std::memory_order_seq_cst);
    --idleWaiters;
    const auto index = static_cast<std::size_t>(reason);
    ++drains[index];
    ++drainOpcodes[std::min<std::uint32_t>(opcode, 0x100)];
    const auto waited = elapsedNs(start);
    drainWaitNs[index] += waited;
    if (ThreadReport::Enabled()) {
        const bool finisher = DrawFinisher::OnFinisherThread();
        const bool worker = !finisher && ThreadReport::OnQueue0Worker();
        ThreadReport::Add(finisher ? ThreadReport::Count::FinisherCommitterDrains : worker ? ThreadReport::Count::WorkerCommitterDrains : ThreadReport::Count::OtherDrains, 1);
        ThreadReport::Add(finisher ? ThreadReport::Count::FinisherCommitterDrainNs : worker ? ThreadReport::Count::WorkerCommitterDrainNs : ThreadReport::Count::OtherDrainNs, waited);
    }
    rethrowFailure();
}

void DrawPipeline::AwaitWriters(std::span<const Range> ranges, DrainReason reason, std::uint32_t opcode, DeepSite site) {
    DrawFinisher::NoteDeepSite(site);
    auto& finisher = DrawFinisher::Get();
    const bool onFinisher = DrawFinisher::OnFinisherThread();
    if (!onFinisher && !FinisherProducer::Get()) {
        Drain(reason, opcode);
        return;
    }
    if (!onFinisher) finisher.FlushOpenRun();
    drainRequests[std::min<std::uint32_t>(opcode, 0x100)].fetch_add(1, std::memory_order_relaxed);
    reasonRequests[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    const bool finisherBusy = !onFinisher && DrawFinisher::Busy();
    if ((finisherBusy || Busy()) && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline wait under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    const auto fallback = [&] {
        DrawFinisher::NoteDeepKept(DeepKept::Fallback);
        if (onFinisher) finisher.AwaitTurn();
        drainAll(reason, opcode);
    };
    for (;;) {
        if (onFinisher) {
            if (FinisherDeposits::Get() && finisher.depositsWrite(ranges)) {
                noteDeepSiteWait(site, true);
                finisher.AwaitTurn();
                continue;
            }
            if (DrawFinisher::Overlap() && FinisherAhead::Get()) static_cast<void>(finisher.aheadWait(ranges));
            else finisher.AwaitTurn();
        } else if (DrawFinisher::Overlap() && DrawFinisher::Busy()) {
            if (finisher.ringMode.load(std::memory_order_acquire) && finisher.deepRing.load(std::memory_order_relaxed)) {
                finisher.retireLean();
                const auto front = finisher.frontSeq.load(std::memory_order_acquire);
                const auto found = finisher.deepWrites->Newest(ranges, front);
                std::uint64_t blocker = 0;
                bool scanned = false;
                if (found.answer == DeepAnswer::Unsure) {
                    noteDeepTotal(DrawFinisher::DeepCount::WorkerFallbacks);
                    if (!DrawFinisher::FrameFlow()) {
                        fallback();
                        return;
                    }
                    blocker = finisher.scanBlocker(ranges, FrameSite::WriteScan);
                    scanned = true;
                } else {
                    const auto unknown = finisher.lastUnknown.load(std::memory_order_relaxed);
                    blocker = found.answer == DeepAnswer::Hit ? found.stamp : std::uint64_t{0};
                    if (unknown >= front) blocker = std::max(blocker, unknown);
                }
                if (blocker != 0) {
                    noteDeepSiteWait(site, true);
                    if (blocker >= finisher.nextSeq || !finisher.awaitProducerFront(blocker)) {
                        if (blocker >= finisher.nextSeq && DrawFinisher::FrameFlow()) DrawFinisher::NoteFrameKept(FrameKept::IndexInconsistent);
                        fallback();
                        return;
                    }
                    if (scanned) finisher.verifyScanned(ranges, blocker);
                    continue;
                }
            } else if (finisher.PendingOverlapsAny(ranges)) {
                noteDeepSiteWait(site, true);
                finisher.Drain();
                continue;
            }
        }
        if (deepQueue.load(std::memory_order_acquire)) {
            auto found = deepQueued(ranges, retiredItems.load(std::memory_order_acquire) + 1);
            if (found.answer == DeepAnswer::Unsure) {
                if (!DrawFinisher::FrameFlow()) {
                    fallback();
                    return;
                }
                found = scanQueued(ranges);
            }
            if (found.answer != DeepAnswer::Hit) return;
            noteDeepSiteWait(site, false);
            if (!awaitRetired(found.stamp - 1)) {
                fallback();
                return;
            }
            continue;
        }
        if (queuedOverlap(ranges)) fallback();
        return;
    }
}

DeepFound DrawPipeline::scanQueued(std::span<const Range> ranges) {
    const auto start = DrawFinisher::FrameCensus() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    DeepFound found;
    {
        const QueueScan scan(*this, ringScanId(RingScan::QueuedScan));
        const auto newest = [&](const QueuedView& queued) {
            const auto answer = newestQueued(queued, scan.Retired(), ranges);
            return std::pair<int, std::uint64_t>(static_cast<int>(answer.answer), answer.stamp);
        };
        found = newestQueued(scan.Items(), scan.Retired(), ranges);
        scan.Check("queued scan", newest);
    }
    DrawFinisher::NoteFrameSite(FrameSite::QueuedScan, start == std::chrono::steady_clock::time_point{} ? 0 : elapsedNs(start));
    return found;
}

bool DrawPipeline::AwaitExactWriters(std::span<const Range> ranges, DrainReason reason, std::uint32_t opcode, DeepSite site, FrameSite timed, FrameCheck check) {
    auto& finisher = DrawFinisher::Get();
    if (!DrawFinisher::FrameFlow() || DrawFinisher::OnFinisherThread() || !FinisherProducer::Get() || !deepQueue.load(std::memory_order_acquire)) return false;
    finisher.FlushOpenRun();
    const bool ring = finisher.ringMode.load(std::memory_order_acquire) && finisher.deepRing.load(std::memory_order_relaxed);
    if (DrawFinisher::Overlap() && DrawFinisher::Busy() && !ring) return false;
    DrawFinisher::NoteDeepSite(site);
    if ((DrawFinisher::Busy() || Busy()) && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline wait under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    const auto start = DrawFinisher::FrameCensus() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    bool waited = false;
    bool passed = false;
    const auto counted = [&] {
        if (waited) return;
        waited = true;
        drainRequests[std::min<std::uint32_t>(opcode, 0x100)].fetch_add(1, std::memory_order_relaxed);
        reasonRequests[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    };
    const auto gaveUp = [](FrameKept why) {
        DrawFinisher::NoteFrameKept(why);
        return false;
    };
    for (;;) {
        if (DrawFinisher::Overlap() && DrawFinisher::Busy()) {
            finisher.retireLean();
            const auto found = finisher.exactPending(ranges);
            if (found.unsure) return gaveUp(FrameKept::ExactUnsure);
            if (found.limited) DrawFinisher::NoteFrameKept(FrameKept::ResolveLimit);
            passed = passed || found.skipped != 0;
            if (found.blocker != 0) {
                counted();
                noteDeepSiteWait(site, true);
                if (found.blocker >= finisher.nextSeq) return gaveUp(FrameKept::IndexInconsistent);
                if (!finisher.awaitProducerFront(found.blocker)) return gaveUp(FrameKept::ExactUnsure);
                continue;
            }
            if (found.unresolved != 0) {
                counted();
                if (!finisher.awaitResolved(found.unresolved)) return gaveUp(FrameKept::ExactUnsure);
                continue;
            }
        }
        if (!deepQueue.load(std::memory_order_acquire)) return gaveUp(FrameKept::ExactUnsure);
        auto queued = deepQueued(ranges, retiredItems.load(std::memory_order_acquire) + 1);
        if (queued.answer == DeepAnswer::Unsure) queued = scanQueued(ranges);
        if (queued.answer == DeepAnswer::Hit) {
            counted();
            noteDeepSiteWait(site, false);
            if (!awaitRetired(queued.stamp - 1)) return gaveUp(FrameKept::ExactUnsure);
            continue;
        }
        break;
    }
    if (passed) DrawFinisher::NoteFrameSite(FrameSite::ResolvedSkip);
    if (waited && start != std::chrono::steady_clock::time_point{}) DrawFinisher::NoteFrameSite(timed, elapsedNs(start));
    if (DrawFinisher::FrameVerifyDue(check)) verifyExactBytes(ranges, reason, opcode, check);
    return true;
}

void DrawPipeline::verifyExactBytes(std::span<const Range> ranges, DrainReason reason, std::uint32_t opcode, FrameCheck check) {
    std::size_t total = 0;
    for (const auto& [begin, end] : ranges) total += static_cast<std::size_t>(end - begin);
    if (total == 0 || total > (std::size_t{1} << 20u) || ranges.empty()) return;
    const auto readAll = [&] {
        std::vector<std::byte> bytes(total);
        std::size_t at = 0;
        for (const auto& [begin, end] : ranges) {
            const auto size = static_cast<std::size_t>(end - begin);
            GuestMemory::Read(begin, std::span<std::byte>(bytes).subspan(at, size));
            at += size;
        }
        return bytes;
    };
    std::vector<std::byte> before;
    std::vector<std::byte> after;
    try {
        before = readAll();
    } catch (const std::exception&) {
        return;
    }
    drainAll(reason, opcode);
    try {
        after = readAll();
    } catch (const std::exception&) {
        return;
    }
    char text[256];
    std::snprintf(text, sizeof(text), "the %zu bytes at 0x%llx the worker read beside the pipeline (opcode 0x%02x) differ from the bytes after a full drain", total, static_cast<unsigned long long>(ranges.front().first), opcode);
    DrawFinisher::NoteFrameCheck(check, before != after, text);
}

bool DrawPipeline::awaitRetired(std::uint64_t item) {
    const auto passed = [&] { return retiredItems.load(std::memory_order_acquire) > item; };
    if (passed()) return true;
    const TailWatch watch(true);
    if (SpinUntil(HandoffSpinWindow(), passed)) return true;
    const ThreadReport::WorkerStateScope waiting(ThreadReport::WorkerState::DrainCommitter, ThreadReport::OnQueue0Worker());
    auto lock = lockFor(mutex);
    ++retireWaiters;
    for (;;) {
        if (commitRing.load(std::memory_order_relaxed)) {
            lowerRingWake(item + 1);
            if (retiredItems.load(std::memory_order_seq_cst) > item) break;
        }
        if (closing || passed()) break;
        idle.wait(lock);
    }
    --retireWaiters;
    rethrowFailure();
    return passed();
}

void DrawPipeline::latchDeepQueue(bool deep) {
    if (deep && deepStores == nullptr) {
        deepStores = std::make_unique<StampedRanges>();
        deepOthers = std::make_unique<StampedRanges>();
        deepStores->Name("queued stores");
        deepOthers->Name("queued others");
        readRegions = std::make_unique<std::array<std::atomic<std::uint64_t>, ReadRegionSlots>>();
        deepIndexed.store(true, std::memory_order_release);
    }
    deepEpoch.fetch_add(1, std::memory_order_seq_cst);
    deepQueue.store(deep, std::memory_order_seq_cst);
}

DeepFound DrawPipeline::deepQueued(std::span<const Range> ranges, std::uint64_t live) const {
    return deepLater(deepStores->Newest(ranges, live), deepOthers->Newest(ranges, live));
}

bool DrawPipeline::deepQueuedWithin(std::span<const Range> ranges, std::uint64_t from, std::uint64_t to) const {
    return deepStores->Within(ranges, from, to) || deepOthers->Within(ranges, from, to);
}

DeepFound DrawPipeline::newestQueued(const QueuedView& queued, std::uint64_t retiredNow, std::span<const Range> ranges) const {
    for (std::size_t index = queued.size(); index-- > 0;) {
        if (rangesOverlap(queued[index].writes, ranges)) return {DeepAnswer::Hit, retiredNow + index + 1};
    }
    return {};
}

void DrawPipeline::verifyQueued(std::uint32_t lookup, std::span<const Range> ranges) {
    const QueueScan scan(*this, ringScanId(RingScan::DeepVerify));
    if (!deepQueue.load(std::memory_order_relaxed)) return;
    const auto live = scan.Retired() + 1;
    const auto found = deepQueued(ranges, live);
    if (found.answer == DeepAnswer::Unsure) return;
    noteDeepVerify(static_cast<DeepLookup>(lookup), found, newestQueued(scan.Items(), scan.Retired(), ranges), true, live, 0, ranges);
}

void DrawPipeline::noteDeepReads(const std::vector<Range>& reads, std::uint64_t stamp) {
    for (const auto& read : reads) {
        std::uint64_t first = 0;
        std::uint64_t last = 0;
        if (!StampedRanges::CoarseRegions(read, StampedRanges::CoarseQueryRegions, first, last)) {
            wideReads.store(stamp, std::memory_order_release);
            continue;
        }
        for (auto region = first; region <= last; ++region) (*readRegions)[readRegionSlot(region)].store(stamp, std::memory_order_release);
    }
}

bool DrawPipeline::readsMaybe(std::uint64_t address, std::size_t bytes) const {
    const auto live = readsLive.load(std::memory_order_acquire);
    if (wideReads.load(std::memory_order_acquire) >= live) return true;
    std::uint64_t first = 0;
    std::uint64_t last = 0;
    if (!StampedRanges::CoarseRegions({address, address + bytes}, StampedRanges::CoarseQueryRegions, first, last)) return true;
    for (auto region = first; region <= last; ++region) {
        if ((*readRegions)[readRegionSlot(region)].load(std::memory_order_acquire) >= live) return true;
    }
    return false;
}

void DrawPipeline::verifyHeldReads(std::uint64_t address, std::size_t bytes) {
    const QueueScan scan(*this, ringScanId(RingScan::HeldReadsVerify));
    if (!deepQueue.load(std::memory_order_relaxed)) return;
    const auto end = address + bytes;
    const auto retiredNow = scan.Retired();
    const auto& queued = scan.Items();
    DeepFound found;
    for (std::size_t index = queued.size(); index-- > 0 && found.answer != DeepAnswer::Hit;) {
        for (const auto& [begin, limit] : queued[index].reads) {
            if (address >= limit || begin >= end) continue;
            found = {DeepAnswer::Hit, retiredNow + index + 1};
            break;
        }
    }
    const Range range{address, end};
    noteDeepVerify(DeepLookup::HeldReads, readsMaybe(address, bytes) ? DeepFound{DeepAnswer::Hit} : DeepFound{}, found, false, readsLive.load(std::memory_order_relaxed), 0, {&range, 1});
}

std::uint64_t DrawPipeline::DrainRequestsByOpcode(std::uint32_t opcode) {
    return Queue0().drainRequests[std::min<std::uint32_t>(opcode, 0x100)].load(std::memory_order_relaxed);
}

bool DrawPipeline::HeldReads(std::uint64_t address, std::size_t bytes) {
    static const bool unordered = std::getenv("APS5_NO_QUEUED_DRAW_READ_ORDER") != nullptr;
    if (bytes == 0 || unordered || FlipCommitting::Get()) return false;
    if (FinisherDeposits::Get() && DrawFinisher::Get().depositsRead(address, bytes)) return true;
    if (DrawFinisher::Get().ParkedReadsOverlap(address, bytes)) return true;
    auto& pipeline = Queue0();
    if (pipeline.deepQueue.load(std::memory_order_acquire)) {
        const bool maybe = pipeline.readsMaybe(address, bytes);
        noteDeepLookup(DeepLookup::HeldReads, maybe);
        if (deepVerifyDue(DeepLookup::HeldReads)) pipeline.verifyHeldReads(address, bytes);
        if (!maybe) return false;
    }
    if (DrawFinisher::Flow()) {
        DrawFinisher::NoteFlowLookup(DrawFinisher::FlowLookup::HeldReads);
        return false;
    }
    const auto end = address + bytes;
    const QueueScan scan(pipeline, ringScanId(RingScan::HeldReads));
    heldReadQueries.Add(1);
    const auto held = [&](const QueuedView& queued) {
        for (std::size_t index = 0; index < queued.size(); ++index) {
            for (const auto& [begin, limit] : queued[index].reads) {
                if (address >= limit || begin >= end) continue;
                return true;
            }
        }
        return false;
    };
    const bool found = held(scan.Items());
    scan.Check("held reads scan", held);
    if (found) heldReadHits.Add(1);
    return found;
}

std::uint64_t DrawPipeline::HeldReadQueries() {
    return heldReadQueries.Sum();
}

void DrawPipeline::ReleaseRunningReads() {
    const QueueScan scan(*this, ringScanId(RingScan::RunningReads));
    const auto& queued = scan.Items();
    if (queued.empty()) return;
    queued.front().reads.clear();
    const auto live = scan.Retired() + 2;
    if (scan.Frozen()) raiseReadsLive(live);
    else if (readsLive.load(std::memory_order_relaxed) < live) readsLive.store(live, std::memory_order_release);
}

std::uint64_t DrawPipeline::DrainRequestsByReason(DrainReason reason) {
    return Queue0().reasonRequests[static_cast<std::size_t>(reason)].load(std::memory_order_relaxed);
}

std::optional<std::uint64_t> DrawPipeline::PendingLabel(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > 8) return std::nullopt;
    if (FinisherDeposits::Get()) {
        const Range range{address, address + bytes};
        if (DrawFinisher::Get().depositsWrite({&range, 1})) return std::nullopt;
    }
    if (DrawFinisher::Overlap()) {
        std::uint64_t held = 0;
        const auto lookup = DrawFinisher::Get().PendingLabel(address, bytes, held);
        if (lookup == DrawFinisher::LabelLookup::Value) return held;
        if (lookup == DrawFinisher::LabelLookup::Unknown) return std::nullopt;
    }
    const auto end = address + bytes;
    const Range range{address, end};
    const auto writesOver = [&](const Item& item) { return std::any_of(item.writes.begin(), item.writes.end(), [&](const Range& write) { return address < write.second && write.first < end; }); };
    const auto valueOf = [&](const Item& item) -> std::optional<std::uint64_t> {
        if (item.labelRun != nullptr) {
            std::uint64_t held = 0;
            if (item.labelRun->Find(address, bytes, held) != LabelRun::Lookup::Value) return std::nullopt;
            return held;
        }
        if (item.labelBytes.empty() || address < item.labelAddress || end > item.labelAddress + item.labelBytes.size()) return std::nullopt;
        std::uint64_t value = 0;
        std::memcpy(&value, item.labelBytes.data() + (address - item.labelAddress), bytes);
        return value;
    };
    const auto newestValue = [&](const QueuedView& queued) -> std::optional<std::uint64_t> {
        for (std::size_t index = queued.size(); index-- > 0;) {
            if (writesOver(queued[index])) return valueOf(queued[index]);
        }
        return std::nullopt;
    };
    if (deepQueue.load(std::memory_order_acquire)) {
        const std::span<const Range> asked(&range, 1);
        if (deepVerifyDue(DeepLookup::QueuedLabel)) verifyQueued(static_cast<std::uint32_t>(DeepLookup::QueuedLabel), asked);
        const auto queued = deepQueued(asked, retiredItems.load(std::memory_order_acquire) + 1);
        noteDeepLookup(DeepLookup::QueuedLabel, queued.answer == DeepAnswer::Unsure);
        if (queued.answer == DeepAnswer::Clear) return std::nullopt;
        const QueueScan scan(*this, ringScanId(RingScan::QueuedLabel));
        scan.Check("queued label scan", newestValue);
        const auto& listed = scan.Items();
        const auto retiredNow = scan.Retired();
        const auto held = deepQueued(asked, retiredNow + 1);
        if (held.answer == DeepAnswer::Clear) return std::nullopt;
        if (held.answer == DeepAnswer::Hit && held.stamp > retiredNow) {
            const auto at = static_cast<std::size_t>(held.stamp - 1 - retiredNow);
            if (at < listed.size() && writesOver(listed[at])) return valueOf(listed[at]);
        }
        return newestValue(listed);
    }
    if (lockless() || DrawFinisher::Flow()) {
        if (published.Overlaps({&range, 1}) == PublishedRanges::Answer::Clear) return std::nullopt;
    }
    const QueueScan scan(*this, ringScanId(RingScan::QueuedLabel));
    scan.Check("queued label scan", newestValue);
    return newestValue(scan.Items());
}

DrawPipeline::QueuedWrite DrawPipeline::StoredBytes(std::uint64_t address, std::size_t bytes, std::span<std::byte> out) {
    const auto end = address + bytes;
    if (FinisherDeposits::Get()) {
        const Range range{address, end};
        if (DrawFinisher::Get().depositsWrite({&range, 1})) return QueuedWrite::Unknown;
    }
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlaps(address, end)) return QueuedWrite::Unknown;
    const Range range{address, end};
    const auto writesOver = [&](const Item& item) { return std::any_of(item.writes.begin(), item.writes.end(), [&](const Range& write) { return address < write.second && write.first < end; }); };
    const auto storedFrom = [&](const Item& item, std::span<std::byte> into) {
        if (into.size() != bytes || address < item.labelAddress || end > item.labelAddress + item.labelBytes.size()) return QueuedWrite::Unknown;
        std::memcpy(into.data(), item.labelBytes.data() + (address - item.labelAddress), bytes);
        return QueuedWrite::Stored;
    };
    const auto newestStored = [&](const QueuedView& queued, std::span<std::byte> into) {
        auto answer = QueuedWrite::None;
        for (std::size_t index = queued.size(); index-- > 0;) {
            const auto& item = queued[index];
            if (!writesOver(item)) continue;
            if (!item.store) return QueuedWrite::Unknown;
            if (into.empty()) {
                answer = QueuedWrite::Stored;
                continue;
            }
            return storedFrom(item, into);
        }
        return answer;
    };
    const auto storedScan = [&](const QueuedView& queued) {
        std::vector<std::byte> copy(out.size());
        const auto answer = newestStored(queued, copy);
        return std::pair<QueuedWrite, std::vector<std::byte>>(answer, std::move(copy));
    };
    if (deepQueue.load(std::memory_order_acquire)) {
        const std::span<const Range> asked(&range, 1);
        if (deepVerifyDue(DeepLookup::StoredBytes)) verifyQueued(static_cast<std::uint32_t>(DeepLookup::StoredBytes), asked);
        const auto live = retiredItems.load(std::memory_order_acquire) + 1;
        const auto others = deepOthers->Newest(asked, live);
        const auto stores = deepStores->Newest(asked, live);
        const bool sure = others.answer != DeepAnswer::Unsure && stores.answer != DeepAnswer::Unsure;
        noteDeepLookup(DeepLookup::StoredBytes, !sure);
        if (sure) {
            const bool other = others.answer == DeepAnswer::Hit;
            const bool store = stores.answer == DeepAnswer::Hit;
            if (!other && !store) return QueuedWrite::None;
            if (out.empty()) return other ? QueuedWrite::Unknown : QueuedWrite::Stored;
            if (other && (!store || others.stamp > stores.stamp)) return QueuedWrite::Unknown;
        }
        const QueueScan scan(*this, ringScanId(RingScan::StoredBytes));
        scan.Check("stored bytes scan", storedScan);
        const auto& listed = scan.Items();
        const auto retiredNow = scan.Retired();
        const auto heldOthers = deepOthers->Newest(asked, retiredNow + 1);
        const auto heldStores = deepStores->Newest(asked, retiredNow + 1);
        if (heldOthers.answer != DeepAnswer::Unsure && heldStores.answer != DeepAnswer::Unsure) {
            const bool other = heldOthers.answer == DeepAnswer::Hit;
            const bool store = heldStores.answer == DeepAnswer::Hit;
            if (!other && !store) return QueuedWrite::None;
            if (out.empty()) return other ? QueuedWrite::Unknown : QueuedWrite::Stored;
            if (other && (!store || heldOthers.stamp > heldStores.stamp)) return QueuedWrite::Unknown;
            const auto at = static_cast<std::size_t>(heldStores.stamp - 1 - retiredNow);
            if (heldStores.stamp > retiredNow && at < listed.size() && listed[at].store && writesOver(listed[at])) return storedFrom(listed[at], out);
        }
        return newestStored(listed, out);
    }
    if (lockless() || DrawFinisher::Flow()) {
        const auto queued = published.Overlaps({&range, 1});
        if (queued == PublishedRanges::Answer::Clear) return QueuedWrite::None;
        if (queued == PublishedRanges::Answer::Unsure && DrawFinisher::Flow()) {
            DrawFinisher::NoteFlowLookup(DrawFinisher::FlowLookup::QueuedWrites);
            return QueuedWrite::None;
        }
    }
    const QueueScan scan(*this, ringScanId(RingScan::StoredBytes));
    scan.Check("stored bytes scan", storedScan);
    return newestStored(scan.Items(), out);
}

void DrawPipeline::countWrites(const std::vector<Range>& writes, bool added) {
    bool changed = false;
    for (const auto& range : writes) changed = countRange(queuedWrites, range, added) || changed;
    if (changed) published.Publish(queuedWrites, 0);
}

DrawPipeline::Watch DrawPipeline::WatchWrites() {
    if (!Busy()) return {Retired(), 0, {}};
    const bool own = FinisherDeposits::Get();
    const bool flow = DrawFinisher::Flow();
    bool raced = false;
    if (deepQueue.load(std::memory_order_acquire)) {
        const auto epoch = deepEpoch.load(std::memory_order_acquire);
        Watch watch;
        const auto enqueuedBefore = enqueuedItems.load(std::memory_order_acquire);
        watch.retired = retiredItems.load(std::memory_order_acquire);
        if (deepQueue.load(std::memory_order_acquire) && deepEpoch.load(std::memory_order_acquire) == epoch) {
            watch.items = enqueuedBefore > watch.retired ? static_cast<std::size_t>(enqueuedBefore - watch.retired) : 0;
            watch.enqueued = enqueuedBefore;
            watch.deep = true;
            if (deepVerifyDue(DeepLookup::Watch)) {
                const QueueScan scan(*this, ringScanId(RingScan::Watch));
                if (deepQueue.load(std::memory_order_relaxed) && deepEpoch.load(std::memory_order_relaxed) == epoch) {
                    const auto& listed = scan.Items();
                    watch.retired = scan.Retired();
                    watch.enqueued = scan.Enqueued();
                    watch.items = listed.size();
                    for (std::size_t index = 0; index < listed.size(); ++index) watch.verifyWrites.insert(watch.verifyWrites.end(), listed[index].writes.begin(), listed[index].writes.end());
                    watch.verified = true;
                }
            }
            if (own) {
                for (const auto& held : DrawFinisher::Get().ownDeposits()) {
                    ++watch.items;
                    watch.writes.insert(watch.writes.end(), held.item.writes.begin(), held.item.writes.end());
                }
            }
            noteDeepLookup(DeepLookup::Watch, false);
            return watch;
        }
        noteDeepLookup(DeepLookup::Watch, true);
        raced = true;
    }
    if ((lockless() || flow) && !own && !raced) {
        Watch watch;
        const auto enqueuedBefore = enqueuedItems.load(std::memory_order_acquire);
        const bool copied = published.Copy(watch.snapshotWrites, watch.snapshotCount);
        if (copied || flow) {
            if (!copied) {
                watch.snapshotCount = 0;
                DrawFinisher::NoteFlowLookup(DrawFinisher::FlowLookup::Watch);
            }
            watch.retired = retiredItems.load(std::memory_order_acquire);
            watch.items = enqueuedBefore > watch.retired ? static_cast<std::size_t>(enqueuedBefore - watch.retired) : 0;
            watch.snapshot = true;
            return watch;
        }
    }
    const QueueScan scan(*this, ringScanId(RingScan::Watch));
    const auto& listed = scan.Items();
    Watch watch{scan.Retired(), listed.size(), {}};
    if (!flow) {
        for (std::size_t index = 0; index < listed.size(); ++index) watch.writes.insert(watch.writes.end(), listed[index].writes.begin(), listed[index].writes.end());
    } else {
        DrawFinisher::NoteFlowLookup(DrawFinisher::FlowLookup::Watch);
    }
    scan.Check("watch scan", [](const QueuedView& queued) {
        std::vector<Range> writes;
        for (std::size_t index = 0; index < queued.size(); ++index) writes.insert(writes.end(), queued[index].writes.begin(), queued[index].writes.end());
        return writes;
    });
    if (own) {
        for (const auto& held : DrawFinisher::Get().ownDeposits()) {
            ++watch.items;
            watch.writes.insert(watch.writes.end(), held.item.writes.begin(), held.item.writes.end());
        }
    }
    return watch;
}

bool DrawPipeline::WrittenSince(const Watch& watch, const std::vector<Range>& reads) {
    const auto overlaps = [&reads](std::span<const Range> writes) {
        return std::any_of(reads.begin(), reads.end(), [&writes](const Range& read) {
            return std::any_of(writes.begin(), writes.end(), [&read](const Range& write) { return read.first < write.second && write.first < read.second; });
        });
    };
    if (FinisherDeposits::Get() && DrawFinisher::Get().depositsWrite(reads)) return true;
    const auto watched = watch.snapshot ? std::span<const Range>(watch.snapshotWrites.data(), watch.snapshotCount) : std::span<const Range>(watch.writes);
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlapsAny(reads)) return true;
    const auto since = [&](std::uint64_t retiredNow) {
        if (retiredNow == watch.retired) return false;
        if (retiredNow - watch.retired > watch.items) return true;
        if (!watch.deep) return overlaps(watched);
        const bool within = deepQueuedWithin(reads, watch.retired + 1, watch.enqueued + 1);
        if (watch.verified) noteDeepVerify(DeepLookup::Watch, within ? DeepFound{DeepAnswer::Hit} : DeepFound{}, overlaps(watch.verifyWrites) ? DeepFound{DeepAnswer::Hit} : DeepFound{}, false, watch.retired, watch.enqueued, reads);
        return within || overlaps(watched);
    };
    if (deepQueue.load(std::memory_order_acquire)) {
        const auto queued = deepQueued(reads, retiredItems.load(std::memory_order_acquire) + 1);
        if (queued.answer == DeepAnswer::Hit) return true;
        if (queued.answer == DeepAnswer::Clear) return since(retiredItems.load(std::memory_order_acquire));
    } else if (lockless() || DrawFinisher::Flow()) {
        const auto queued = published.Overlaps(reads);
        if (queued == PublishedRanges::Answer::Hit) return true;
        if (queued == PublishedRanges::Answer::Clear || DrawFinisher::Flow()) {
            if (queued != PublishedRanges::Answer::Clear) DrawFinisher::NoteFlowLookup(DrawFinisher::FlowLookup::QueuedWrites);
            return since(retiredItems.load(std::memory_order_acquire));
        }
    }
    const QueueScan scan(*this, ringScanId(RingScan::WrittenSince));
    const auto written = [&](const QueuedView& queued) {
        for (std::size_t index = 0; index < queued.size(); ++index) {
            if (overlaps(queued[index].writes)) return true;
        }
        return false;
    };
    const bool found = written(scan.Items());
    scan.Check("written since scan", written);
    if (found) return true;
    return since(scan.Retired());
}

bool DrawPipeline::Busy() const {
    return (FinisherProducer::Get() && DrawFinisher::Busy() && DrawFinisher::Overlap()) || outstanding.load(std::memory_order_acquire) != 0 || DrawFinisher::AheadOfTurn() || FinisherDeposits::Get();
}

bool DrawPipeline::queuedOverlap(std::span<const Range> ranges) {
    if (deepQueue.load(std::memory_order_acquire)) {
        if (deepVerifyDue(DeepLookup::QueuedWrites)) verifyQueued(static_cast<std::uint32_t>(DeepLookup::QueuedWrites), ranges);
        const auto found = deepQueued(ranges, retiredItems.load(std::memory_order_acquire) + 1);
        noteDeepLookup(DeepLookup::QueuedWrites, found.answer == DeepAnswer::Unsure);
        if (found.answer != DeepAnswer::Unsure) return found.answer == DeepAnswer::Hit;
        const QueueScan scan(*this, ringScanId(RingScan::QueuedWrites));
        const auto newest = [&](const QueuedView& queued) { return newestQueued(queued, scan.Retired(), ranges).answer == DeepAnswer::Hit; };
        const bool hit = newest(scan.Items());
        scan.Check("queued writes scan", newest);
        return hit;
    }
    if (lockless() || DrawFinisher::Flow()) {
        const auto answer = published.Overlaps(ranges);
        if (answer != PublishedRanges::Answer::Unsure) return answer == PublishedRanges::Answer::Hit;
        if (DrawFinisher::Flow()) {
            DrawFinisher::NoteFlowLookup(DrawFinisher::FlowLookup::QueuedWrites);
            return false;
        }
    }
    const auto lock = lockFor(mutex);
    for (const auto& [range, count] : queuedWrites) {
        for (const auto& [address, end] : ranges) {
            if (address < range.second && range.first < end) return true;
        }
    }
    return false;
}

bool DrawPipeline::QueuedClearForValidation(std::span<const Range> ranges) {
    if (deepQueue.load(std::memory_order_seq_cst)) {
        const auto found = deepQueued(ranges, retiredItems.load(std::memory_order_acquire) + 1);
        if (found.answer != DeepAnswer::Unsure) return found.answer == DeepAnswer::Clear;
    } else if (lockless() && !DrawFinisher::Flow()) {
        const auto answer = published.Overlaps(ranges);
        if (answer != PublishedRanges::Answer::Unsure) return answer == PublishedRanges::Answer::Clear;
    }
    const QueueScan scan(*this, ringScanId(RingScan::ValidationClear));
    const auto clear = [&](const QueuedView& queued) {
        for (std::size_t index = 0; index < queued.size(); ++index) {
            if (rangesOverlap(queued[index].writes, ranges)) return false;
        }
        return true;
    };
    const bool answer = clear(scan.Items());
    scan.Check("validation clear scan", clear);
    return answer;
}

bool DrawPipeline::OverlapsAny(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (FinisherDeposits::Get() && DrawFinisher::Get().depositsWrite(ranges)) return true;
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlapsAny(ranges)) return true;
    return queuedOverlap(ranges);
}

bool DrawPipeline::Overlaps(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    const Range range{address, end};
    if (FinisherDeposits::Get() && DrawFinisher::Get().depositsWrite({&range, 1})) return true;
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlaps(address, end)) return true;
    return queuedOverlap({&range, 1});
}

}
