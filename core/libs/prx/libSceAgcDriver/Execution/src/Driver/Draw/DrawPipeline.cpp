#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadPriority.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/CommandEncoder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostAllocator.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BindlessHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include <algorithm>
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

constexpr std::array<const char*, static_cast<std::size_t>(DrawPipeline::CommitKind::Count)> KindNames{"other", "draw", "indirect", "colormeta", "dispatch", "sampledump", "dmafill", "dmawrite", "dmacopy", "store", "endofpipe", "seal", "submitend", "completion", "bufferfill"};

std::atomic<unsigned> runningCommitters{0};
std::atomic<unsigned> runningFinishers{0};
std::atomic<std::uint64_t> drawKeys{0};
std::atomic<std::uint64_t> drawKeysSameAsPrevious{0};
std::atomic<std::uint64_t> drawKeysSeenRecently{0};
std::atomic<std::uint64_t> drawShapesSameAsPrevious{0};
std::atomic<std::uint64_t> drawShapesSeenRecently{0};

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

std::atomic<std::uint64_t> heldReadRanges{0};
std::atomic<std::uint64_t> heldReadQueries{0};
std::atomic<std::uint64_t> heldReadHits{0};

template<typename TItem>
struct ClearOnExit {
    std::vector<TItem>& items;
    ~ClearOnExit() { items.clear(); }
};

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

std::uint64_t orderKey() {
    thread_local std::uint64_t keyBase = 0;
    thread_local std::uint64_t keyOrdinal = 0;
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

bool finisherTakes(unsigned index) {
    if (index == 0) return true;
    constexpr Graphics::RuntimeSwitch switches[4] = {Graphics::RuntimeSwitch::SecondFinisher, Graphics::RuntimeSwitch::ThirdFinisher, Graphics::RuntimeSwitch::FourthFinisher, Graphics::RuntimeSwitch::FifthFinisher};
    return Graphics::RuntimeSwitches::Set(switches[index - 1]);
}

constexpr const char* FinisherNames[5] = {"draw finisher", "draw finisher 2", "draw finisher 3", "draw finisher 4", "draw finisher 5"};

constexpr unsigned PublishedReadAttempts = 8;

#ifdef _WIN32
const bool hostLargeCacheHooked = [] {
    SetHostLargeCacheQuery(+[] { return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::HostLargeCache); });
    SetHostPrefetchQuery(+[] { return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::HostPrefetch); });
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

bool spinHandoff() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::SpinHandoff);
}

constexpr std::chrono::microseconds HandoffLockSpin{20};

template<typename TMutex>
std::unique_lock<TMutex> lockFor(TMutex& mutex) {
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

void PublishedRanges::Publish(std::span<const std::pair<Range, std::uint32_t>> entries, std::uint32_t unknownHeld) {
    const auto before = seq.load(std::memory_order_relaxed);
    seq.store(before + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    const auto size = entries.size() <= Capacity ? static_cast<std::uint32_t>(entries.size()) : Capacity + 1;
    for (std::uint32_t index = 0; index < size && index < Capacity; ++index) {
        bounds[index * 2].store(entries[index].first.first, std::memory_order_relaxed);
        bounds[index * 2 + 1].store(entries[index].first.second, std::memory_order_relaxed);
    }
    count.store(size, std::memory_order_relaxed);
    unknown.store(unknownHeld, std::memory_order_relaxed);
    seq.store(before + 2, std::memory_order_release);
}

PublishedRanges::Answer PublishedRanges::Overlaps(std::span<const Range> ranges) const {
    for (unsigned attempt = 0; attempt < PublishedReadAttempts; ++attempt) {
        const auto before = seq.load(std::memory_order_acquire);
        if ((before & 1u) != 0) {
            SpinPause();
            continue;
        }
        const auto held = unknown.load(std::memory_order_relaxed);
        const auto size = count.load(std::memory_order_relaxed);
        bool hit = false;
        for (std::uint32_t index = 0; held == 0 && index < size && index < Capacity && !hit; ++index) {
            const auto first = bounds[index * 2].load(std::memory_order_relaxed);
            const auto last = bounds[index * 2 + 1].load(std::memory_order_relaxed);
            hit = std::any_of(ranges.begin(), ranges.end(), [&](const Range& range) { return range.first < last && first < range.second; });
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (seq.load(std::memory_order_relaxed) != before) continue;
        if (held != 0) return Answer::Hit;
        if (size > Capacity) return Answer::Unsure;
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
            hit = (heldKey == key || (shape != 0 && heldShape == shape)) && (heldInserts || !insertsOnly);
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
        const auto last = std::min(tail.load(std::memory_order_relaxed), before);
        const bool fits = first > last || last - first <= Capacity;
        bool hit = false;
        for (auto number = first; fits && number < last && !hit; ++number) {
            const auto slot = static_cast<std::size_t>(number % Capacity) * 3;
            const auto heldKey = slots[slot].load(std::memory_order_relaxed);
            const auto heldShape = slots[slot + 1].load(std::memory_order_relaxed);
            const bool heldInserts = slots[slot + 2].load(std::memory_order_relaxed) != 0;
            hit = (heldKey == key || (shape != 0 && heldShape == shape)) && (heldInserts || !insertsOnly);
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
std::atomic<std::uint64_t> pipelinedMetadataPasses{0};
}

std::uint64_t DrawPipeline::PipelinedMetadataPasses() {
    return pipelinedMetadataPasses.load(std::memory_order_relaxed);
}

void DrawPipeline::CountPipelinedMetadataPass() {
    pipelinedMetadataPasses.fetch_add(1, std::memory_order_relaxed);
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
    if (lockless()) {
        const Range range{begin, end};
        const auto answer = published.Overlaps({&range, 1});
        if (answer != PublishedRanges::Answer::Unsure) return answer == PublishedRanges::Answer::Hit;
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
    if (lockless()) {
        const Range range{address, end};
        if (published.Overlaps({&range, 1}) == PublishedRanges::Answer::Clear) return LabelLookup::NotHeld;
    }
    const auto lookup = [&](const Pending& held) -> std::optional<LabelLookup> {
        if (held.unknown) return LabelLookup::Unknown;
        const bool overlaps = std::any_of(held.writes.begin(), held.writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
        if (!overlaps) return std::nullopt;
        if (held.labelBytes.empty() || address < held.labelAddress || end > held.labelAddress + held.labelBytes.size()) return LabelLookup::Unknown;
        value = 0;
        std::memcpy(&value, held.labelBytes.data() + (address - held.labelAddress), bytes);
        return LabelLookup::Value;
    };
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
    if (lockless()) {
        const auto answer = published.Overlaps(ranges);
        if (answer != PublishedRanges::Answer::Unsure) return answer == PublishedRanges::Answer::Hit;
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

bool DrawFinisher::PendingKey(std::uint64_t key, std::uint64_t shape) {
    if (!Busy() || !FinisherProducer::Get()) return false;
    const bool precise = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PreciseLookupDrain);
    const bool ring = ringMode.load(std::memory_order_acquire);
    if (ring) retireLean();
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

bool DrawFinisher::aheadWait(std::span<const Range> ranges) {
    if (!FinisherAhead::Get() || !AheadBusy()) return false;
    if (published.Overlaps(ranges) == PublishedRanges::Answer::Clear) return false;
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

bool DrawFinisher::OnFinisherThread() {
    return FinisherThread::Get();
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
    if (leanTake.load(std::memory_order_acquire) < leanTail.load(std::memory_order_acquire)) return parked.load(std::memory_order_acquire) != 0 ? 1 : 2;
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

void DrawFinisher::Enqueue(Job job) {
    Enqueue(std::move(job), Pending{});
}

void DrawFinisher::Enqueue(Job job, Pending held) {
    FinisherProducer::Set(true);
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
        slot.runner = nullptr;
        slot.retirer = nullptr;
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
    if (countPending(held, true)) published.Publish(pendingWrites, pendingUnknown != 0 ? 1u : 0u);
    keys.Push(jobs.back().seq, held.key, held.shape, held.inserts);
    pending.push_back(std::move(held));
    outstanding.fetch_add(1, std::memory_order_release);
    startFinishers();
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
            if (index < 3 || std::getenv("APS5_EXTRA_FINISHERS_UNPINNED") == nullptr) PlaceFinisherThread(FinisherNames[index]);
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
        published.Publish(pendingWrites, 0u);
    }
    if (ring) {
        if (lean == nullptr) lean = std::make_unique<std::array<LeanSlot, LeanSlots>>();
        leanTake.store(nextSeq, std::memory_order_seq_cst);
        leanTail.store(nextSeq, std::memory_order_seq_cst);
        leanRetired = nextSeq;
    }
    frontSeq.store(nextSeq, std::memory_order_release);
    leanFailed.store(false, std::memory_order_release);
    ringMode.store(ring, std::memory_order_release);
    LeanFront::Add(LeanFront::Counter::StoreSwitches);
    return true;
}

void DrawFinisher::retireLean() {
    const auto front = frontSeq.load(std::memory_order_acquire);
    if (leanRetired >= front) return;
    bool republish = false;
    for (; leanRetired < front; ++leanRetired) {
        auto& slot = leanSlot(leanRetired);
        republish = countPending(slot.pending, false) || republish;
        if (const auto retirer = std::exchange(slot.retirer, nullptr); retirer != nullptr) retirer(slot.payload.data(), std::move(slot.answer));
        slot.answer = nullptr;
    }
    if (republish) published.Publish(pendingWrites, pendingUnknown != 0 ? 1u : 0u);
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

void DrawFinisher::awaitLeanRoom() {
    static_assert(PublishedKeys::Capacity <= LeanSlots && 16 <= LeanSlots, "the finisher ring must hold the deepest finisher queue");
    const std::uint64_t depth = finisherDepth();
    const auto full = [&] { return nextSeq - frontSeq.load(std::memory_order_seq_cst) >= depth && !closed.load(std::memory_order_acquire); };
    if (!full()) {
        retireLean();
        return;
    }
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher full while the enqueuing thread holds the GPU lock");
    const auto roomStart = ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
    std::optional<ThreadReport::WorkerStateScope> room;
    if (roomStart != ThreadReport::Clock::time_point{} && ThreadReport::OnQueue0Worker()) room.emplace(ThreadReport::WorkerState::RoomFinisher);
    LeanFront::AddOnWorker(LeanFront::Counter::RingRoomWaits);
    if (!spinHandoff() || !SpinUntil(HandoffSpinWindow(), [&] { return !full(); })) {
        frontWaiters.fetch_add(1, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        {
            auto lock = lockFor(mutex);
            idle.wait(lock, [&] { return !full(); });
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
        idle.wait(lock, [&] { return outstanding.load(std::memory_order_seq_cst) == 0; });
    }
    frontWaiters.fetch_sub(1, std::memory_order_relaxed);
}

void DrawFinisher::publishLean(LeanSlot& slot) {
    const auto seq = nextSeq;
    if (countPending(slot.pending, true)) published.Publish(pendingWrites, pendingUnknown != 0 ? 1u : 0u);
    keys.Push(seq, slot.pending.key, slot.pending.shape, slot.pending.inserts);
    slot.seq.store(seq, std::memory_order_relaxed);
    slot.state.store(LeanReady, std::memory_order_relaxed);
    outstanding.fetch_add(1, std::memory_order_acq_rel);
    nextSeq = seq + 1;
    issued.store(nextSeq, std::memory_order_relaxed);
    leanTail.store(nextSeq, std::memory_order_seq_cst);
    const unsigned wanted = 1u | (finisherTakes(1) ? 2u : 0u) | (finisherTakes(2) ? 4u : 0u) | (finisherTakes(3) ? 8u : 0u) | (finisherTakes(4) ? 16u : 0u);
    if ((started.load(std::memory_order_acquire) & wanted) != wanted) {
        const auto lock = lockFor(mutex);
        startFinishers();
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (parked.load(std::memory_order_relaxed) == 0) return;
    const auto lock = lockFor(mutex);
    if (spinHandoff() && sleepers == 1) wake.notify_one();
    else wake.notify_all();
}

std::byte* DrawFinisher::ReserveLean() {
    FinisherProducer::Set(true);
    rethrowLeanFailure();
    if (closed.load(std::memory_order_acquire)) return nullptr;
    awaitLeanRoom();
    if (closed.load(std::memory_order_acquire)) return nullptr;
    return leanSlot(nextSeq).payload.data();
}

void DrawFinisher::PublishLean(LeanRunner runner, std::span<const Range> writes, std::uint64_t key, std::uint64_t shape, LeanRetirer retirer) {
    auto& slot = leanSlot(nextSeq);
    slot.runner = runner;
    slot.retirer = retirer;
    slot.answer = nullptr;
    auto& held = slot.pending;
    held.key = key;
    held.shape = shape;
    held.inserts = true;
    held.unknown = false;
    held.writes.assign(writes.begin(), writes.end());
    held.labelAddress = 0;
    held.labelBytes.clear();
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
    std::exception_ptr error;
    try {
        if (!leanFailed.load(std::memory_order_acquire)) {
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
    FinisherDeposits::Set(false);
    slot.state.store(deposited ? LeanDeposited : LeanDone, std::memory_order_seq_cst);
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
    if (frontSeq.load(std::memory_order_acquire) == seq) {
        turnItems.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    leanSlot(seq).deposits.push_back({&pipeline, std::move(item)});
    FinisherDeposits::Set(true);
    depositedItems.fetch_add(1, std::memory_order_relaxed);
    return true;
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
        for (auto& held : deposits) {
            held.pipeline->queueItem(std::move(held.item));
            ++sent;
        }
    } catch (...) {
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
        for (auto& held : slot.deposits) {
            held.pipeline->queueItem(std::move(held.item));
            ++sent;
        }
    } catch (...) {
        error = std::current_exception();
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
    PerformanceTrace::BindThread(FinisherNames[index]);
    if (index == 0) StartFinisherSampler();
    ThreadReport::Clock::time_point lastEnd{};
    for (;;) {
        std::uint64_t leanSeq = 0;
        if (takeLean(index, leanSeq)) {
            runLean(leanSeq, lastEnd);
            continue;
        }
        Slot* slot = nullptr;
        bool failed = false;
        if (finisherTakes(index) && spinHandoff()) SpinUntil(HandoffSpinWindow(), [&] { return untakenJobs.load(std::memory_order_acquire) != 0 || leanReady(); });
        else if (finisherTakes(index)) SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
        if (takeLean(index, leanSeq)) {
            runLean(leanSeq, lastEnd);
            continue;
        }
        {
            auto lock = lockFor(mutex);
            ++sleepers;
            parked.fetch_add(1, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            wake.wait(lock, [&] { return closing || ((untaken() != nullptr || leanReady()) && finisherTakes(index)); });
            parked.fetch_sub(1, std::memory_order_relaxed);
            --sleepers;
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
    runningFinishers.fetch_sub(1, std::memory_order_acq_rel);
}

void DrawFinisher::Settle() {
    if (!Busy() || OnFinisherThread()) return;
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher settle under the GPU lock");
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

void DrawFinisher::Drain() {
    if (OnFinisherThread()) return;
    if (Busy() && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher drain under the GPU lock");
    const auto drainStart = ThreadReport::Enabled() && Busy() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
    const ThreadReport::WorkerStateScope draining(ThreadReport::WorkerState::DrainFinisher, drainStart != ThreadReport::Clock::time_point{} && ThreadReport::OnQueue0Worker());
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
    std::thread finishers[5];
    {
        const auto lock = lockFor(mutex);
        closing = true;
        closed.store(true, std::memory_order_release);
        finishers[0] = std::move(thread);
        for (std::size_t index = 0; index < extra.size(); ++index) finishers[index + 1] = std::move(extra[index]);
    }
    wake.notify_all();
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
        if (text == nullptr) return std::size_t{8};
        const auto value = std::strtoull(text, nullptr, 10);
        return value == 0 ? std::size_t{0} : value == 1 ? std::size_t{8} : static_cast<std::size_t>(std::min<unsigned long long>(value, 256));
    }();
    return depth;
}

std::size_t DrawPipeline::QueuedItems() {
    auto& pipeline = Queue0();
    const auto lock = lockFor(pipeline.mutex);
    return pipeline.items.size();
}

bool& DrawPipeline::Active() {
    struct ActiveStorage {};
    return HostThreadLocal<bool, ActiveStorage>();
}

std::atomic<std::uint64_t>& DrawPipeline::EpochToken() {
    static std::atomic<std::uint64_t> token{1};
    return token;
}

void DrawPipeline::FollowEpoch(std::uint64_t token) {
    struct FollowedSlot {};
    using Followed = HostThreadSlot<std::uint64_t, FollowedSlot>;
    if (Followed::Get() == token) return;
    Followed::Set(token);
    GuestMemory::BumpCollectEpoch();
}

unsigned DrawPipeline::Committers() {
    return runningCommitters.load(std::memory_order_acquire);
}

void DrawPipeline::rethrowFailure() {
    if (failure == nullptr) return;
    auto error = std::exchange(failure, nullptr);
    std::rethrow_exception(error);
}

void DrawPipeline::Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, std::vector<Range> reads, bool draw) {
    enqueue({std::move(commit), std::move(writes), labelAddress, std::move(labelBytes), std::move(reads), draw});
}

void DrawPipeline::EnqueueQuiet(Commit commit) {
    enqueue({std::move(commit), {}, 0, {}, {}, false, false, CommitKind::Other, true});
}

void DrawPipeline::Enqueue(CommitKind kind, Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, std::vector<Range> reads, bool draw) {
    enqueue({std::move(commit), std::move(writes), labelAddress, std::move(labelBytes), std::move(reads), draw, false, kind});
}

void DrawPipeline::NoteDrawKey(std::uint64_t key, std::uint64_t shape) {
    constexpr std::size_t RecentSlots = 4096;
    thread_local std::array<std::uint64_t, RecentSlots> recent{};
    thread_local std::uint64_t previous = 0;
    thread_local std::array<std::uint64_t, RecentSlots> recentShapes{};
    thread_local std::uint64_t previousShape = 0;
    const auto shapeSlot = static_cast<std::size_t>((shape ^ (shape >> 29u)) % RecentSlots);
    const auto slot = static_cast<std::size_t>((key ^ (key >> 29u)) % RecentSlots);
    drawKeys.fetch_add(1, std::memory_order_relaxed);
    if (key == previous) drawKeysSameAsPrevious.fetch_add(1, std::memory_order_relaxed);
    if (recent[slot] == key) drawKeysSeenRecently.fetch_add(1, std::memory_order_relaxed);
    if (shape == previousShape) drawShapesSameAsPrevious.fetch_add(1, std::memory_order_relaxed);
    if (recentShapes[shapeSlot] == shape) drawShapesSeenRecently.fetch_add(1, std::memory_order_relaxed);
    recent[slot] = key;
    previous = key;
    recentShapes[shapeSlot] = shape;
    previousShape = shape;
}

void DrawPipeline::EnqueueStore(Commit commit, std::uint64_t address, std::vector<std::byte> bytes) {
    const auto end = address + bytes.size();
    enqueue({std::move(commit), {{address, end}}, address, std::move(bytes), {}, false, true, CommitKind::Store});
}

namespace {
struct ItemTurnSlot {};
using ItemTurn = HostThreadSlot<bool, ItemTurnSlot>;
std::atomic<std::uint64_t> itemsByTurn{0};
}

void DrawPipeline::enqueue(Item item) {
    if (depositVerifyEvery() != 0) item.order = orderKey();
    if (DrawFinisher::OnFinisherThread()) {
        auto& finisher = DrawFinisher::Get();
        if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::FinisherDeposit) && finisher.deposit(*this, item)) return;
        finisher.AwaitTurn();
    }
    if (DrawFinisher::Busy() && !DrawFinisher::OnFinisherThread()) {
        const bool dispatchTurn = item.kind == CommitKind::Dispatch && Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DispatchBeside);
        if ((item.reads.empty() || dispatchTurn) && !ItemTurn::Get() && FinisherProducer::Get() && DrawFinisher::Overlap() && !GuestMemory::GpuMutex().HeldByThisThread() && (dispatchTurn || Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::ItemsByTurn))) {
            DrawFinisher::Pending held;
            held.unknown = false;
            held.writes = item.writes;
            held.labelAddress = item.labelAddress;
            held.labelBytes = item.labelBytes;
            itemsByTurn.fetch_add(1, std::memory_order_relaxed);
            const auto parked = item.reads.empty() ? std::uint64_t{0} : DrawFinisher::Get().ParkReads(item.reads);
            auto shared = std::make_shared<Item>(std::move(item));
            DrawFinisher::Get().Enqueue([this, shared, parked] {
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
        DrawFinisher::Get().Drain();
    }
    queueItem(std::move(item));
}

void DrawPipeline::queueItem(Item item) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    struct ReclaimStorage {};
    auto& reclaim = HostThreadLocal<std::vector<Item>, ReclaimStorage>();
    const ClearOnExit<Item> clearReclaim{reclaim};
    const auto lockStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto lock = lockFor(mutex);
    if (!retired.empty()) std::swap(retired, reclaim);
    if (profile) lockWaitNs += elapsedNs(lockStart);
    rethrowFailure();
    if (closing) {
        lock.unlock();
        item.commit();
        return;
    }
    const auto roomLimit = deepQueues() ? std::max<std::size_t>(Depth(), 32) : Depth();
    if (items.size() >= roomLimit) {
        if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline full while the enqueuing thread holds the GPU lock");
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
    }
    depthSum += items.size();
    ++enqueued;
    heldReadRanges.fetch_add(item.reads.size(), std::memory_order_relaxed);
    countWrites(item.writes, true);
    enqueuedItems.fetch_add(1, std::memory_order_release);
    items.push_back(std::move(item));
    outstanding.fetch_add(1, std::memory_order_release);
    if (!thread.joinable()) {
        thread = std::thread([this] {
            GuestMemory::MarkProofThread();
            RaiseWorkerThreadPriority("draw committer");
            PlaceReservedThread("draw committer");
            run();
        });
        runningCommitters.fetch_add(1, std::memory_order_acq_rel);
    }
    if (committerWaiting) {
        ++wakes;
        wake.notify_one();
    }
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!profile && !stats) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport >= std::chrono::seconds(10)) report(now);
}

void DrawPipeline::report(std::chrono::steady_clock::time_point now) {
    lastReport = now;
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
    std::snprintf(reads, sizeof(reads), " %llu read ranges, %llu stores held, %llu decode drains, %llu items by turn, %llu finisher items deposited, %llu queued at their turn", static_cast<unsigned long long>(heldReadRanges.load(std::memory_order_relaxed)), static_cast<unsigned long long>(heldReadHits.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(decodeDrainCount - reportedDecodeDrains), static_cast<unsigned long long>(itemsByTurn.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(depositCount - reportedDeposits), static_cast<unsigned long long>(turnCount - reportedTurnItems));
    reportedDecodeDrains = decodeDrainCount;
    reportedDeposits = depositCount;
    reportedTurnItems = turnCount;
    if (depositVerifyEvery() != 0) AgcDriver::ProfilePrint_nid_no_patch("[finisher-deposit-verify] 10 s: every %llu; items checked %llu, out of order %llu; drains checked %llu, with deposits left %llu\n", static_cast<unsigned long long>(depositVerifyEvery()), static_cast<unsigned long long>(orderChecks.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(orderDifferences.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(drainChecks.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(drainDifferences.exchange(0, std::memory_order_relaxed)));
    AgcDriver::ProfilePrint_nid_no_patch("[draw] pipelined (10 s, depth %zu): %llu commits at %.1f us each on the committer, %llu failed; %llu enqueued behind %.2f on average, waiting %.1f us per draw for room, %.2f us for the lock; %llu committer wakes;%s; drains:%s\n", Depth(), static_cast<unsigned long long>(commits), perCommit, static_cast<unsigned long long>(commitErrors), static_cast<unsigned long long>(enqueued), queued, perDraw, enqueued == 0 ? 0.0 : static_cast<double>(lockWaitNs) / 1e3 / static_cast<double>(enqueued), static_cast<unsigned long long>(wakes), reads, drainText.c_str());
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
    AgcDriver::ProfilePrint_nid_no_patch("[committer-idle] 10 s, by the finisher's state when the idle began:%s\n", committerIdleText().c_str());
    const auto vk = Graphics::TakeVkDrawCounts();
    AgcDriver::ProfilePrint_nid_no_patch("[vkdraw] (10 s): %llu vkCmdDraw, %llu vkCmdDrawIndexed, %llu vkCmdDrawIndirect calls, %llu mesh; draw keys %llu, same as the previous draw %llu, seen within the last 4096 draws %llu; draw shapes (state without per-draw user words) same as the previous %llu, seen within the last 4096 %llu\n", static_cast<unsigned long long>(vk.direct), static_cast<unsigned long long>(vk.indexed), static_cast<unsigned long long>(vk.indirect), static_cast<unsigned long long>(vk.mesh), static_cast<unsigned long long>(drawKeys.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(drawKeysSameAsPrevious.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(drawKeysSeenRecently.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(drawShapesSameAsPrevious.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(drawShapesSeenRecently.exchange(0, std::memory_order_relaxed)));
    commits = commitNs = commitErrors = enqueued = fullWaitNs = depthSum = wakes = lockWaitNs = 0;
    drains = {};
    drainWaitNs = {};
}


void DrawPipeline::run() {
    PerformanceTrace::BindThread("draw committer");
    StartCommitterSampler();
    // A run of draws under one GPU lock hold (APS5_NO_LEAN_RUNS turns it off): the hold that
    // commits one draw stays for the next while the next item is a draw already queued, up to
    // LeanRunCap draws and while no other thread is blocked on the lock. A non-draw item, an empty
    // queue, a failure and a waiting thread end the run. Each draw still takes its own (nested)
    // hold, labels and submit check in commitDraw.
    constexpr std::size_t LeanRunCap = 32;
    std::optional<std::unique_lock<GuestMemory::GpuMutexType>> runHold;
    std::size_t runLength = 0;
    ThreadReport::Clock::time_point lastEnd{};
    ThreadReport::StateTimes idleFrom{};
    for (;;) {
        Commit* commit = nullptr;
        bool isDraw = false;
        bool rangedItem = false;
        bool quietItem = false;
        std::vector<Range> itemRanges;
        static const bool idleCensus = std::getenv("APS5_PIPELINE_STATS") != nullptr;
        const bool idleNow = idleCensus && outstanding.load(std::memory_order_acquire) == 0;
        const unsigned idleState = idleNow ? DrawFinisher::Get().IdleState() : 0;
        const auto idleStart = idleNow ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
        {
            auto lock = lockFor(mutex);
            committerWaiting = true;
            wake.wait(lock, [&] { return !items.empty() || closing; });
            committerWaiting = false;
            if (idleNow) noteCommitterIdle(idleState, elapsedNs(idleStart));
            if (items.empty()) break;
            if (depositVerifyEvery() != 0) verifyOrder(items.front().order);
            commit = stopped ? nullptr : &items.front().commit;
            isDraw = items.front().draw;
            quietItem = items.front().quiet;
            rangedItem = !isDraw && GuestMemory::RangeProofs();
            if (rangedItem) {
                itemRanges = items.front().writes;
                if (items.front().labelAddress != 0 && !items.front().labelBytes.empty()) itemRanges.emplace_back(items.front().labelAddress, items.front().labelAddress + items.front().labelBytes.size());
            }
        }
        const auto start = std::chrono::steady_clock::now();
        const auto idleTo = ThreadReport::Enabled() && lastEnd != ThreadReport::Clock::time_point{} ? ThreadReport::SnapshotStates(start) : ThreadReport::StateTimes{};
        std::exception_ptr fatal;
        bool failed = false;
        bool shutdown = false;
        if (!isDraw || commit == nullptr || Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoLeanRuns)) {
            runHold.reset();
        } else if (!runHold) {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
            runHold.emplace(GuestMemory::GpuMutex());
            runLength = 0;
        }
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
        const auto spent = elapsedNs(start);
        if (ThreadReport::Enabled()) {
            const auto end = start + std::chrono::nanoseconds(spent);
            ThreadReport::Add(isDraw ? ThreadReport::Count::CommitterDraws : ThreadReport::Count::CommitterOther, 1);
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
        if (fatal != nullptr) runHold.reset();
        if (fatal != nullptr && !stopped) {
            stopped = true;
            if (!shutdown) Driver::Get().ReportFailure(fatal);
        }
        bool keepRun = false;
        {
            const auto lock = lockFor(mutex);
            retiredItems.fetch_add(1, std::memory_order_release);
            countWrites(items.front().writes, false);
            const auto kindIndex = static_cast<std::size_t>(items.front().kind);
            kindCommits[kindIndex].fetch_add(1, std::memory_order_relaxed);
            kindNs[kindIndex].fetch_add(spent, std::memory_order_relaxed);
            retired.push_back(std::move(items.front()));
            items.pop_front();
            ++commits;
            commitNs += spent;
            if (failed) ++commitErrors;
            if (fatal != nullptr && failure == nullptr) failure = fatal;
            outstanding.fetch_sub(1, std::memory_order_release);
            if (idleWaiters != 0 && (items.empty() || items.size() <= resumeAt)) idle.notify_all();
            keepRun = runHold.has_value() && !stopped && !closing && !items.empty() && items.front().draw && ++runLength < LeanRunCap && idleWaiters == 0;
        }
        if (runHold && (!keepRun || GuestMemory::GpuMutex().HasWaiters())) runHold.reset();
    }
    runHold.reset();
    runningCommitters.fetch_sub(1, std::memory_order_acq_rel);
}

void DrawPipeline::verifyOrder(std::uint64_t order) {
    if (order == 0 || DrawFinisher::Get().closed.load(std::memory_order_acquire) || ++orderOrdinal % depositVerifyEvery() != 0) return;
    orderChecks.fetch_add(1, std::memory_order_relaxed);
    if (order <= verifiedOrder) {
        orderDifferences.fetch_add(1, std::memory_order_relaxed);
        noteDepositDifference("item order differs: the committer took an item of " + orderText(order) + " after an item of " + orderText(verifiedOrder));
    }
    verifiedOrder = order;
}

void DrawPipeline::Stop() {
    std::thread committer;
    {
        const auto lock = lockFor(mutex);
        closing = true;
        committer = std::move(thread);
    }
    wake.notify_all();
    if (!committer.joinable()) return;
    if (committer.get_id() == std::this_thread::get_id()) committer.detach();
    else committer.join();
}

void DrawPipeline::ReleaseRetired() {
    std::vector<Item> done;
    const auto lock = lockFor(mutex);
    std::swap(done, retired);
}

void DrawPipeline::Drain(DrainReason reason, std::uint32_t opcode) {
    if (DrawFinisher::OnFinisherThread()) DrawFinisher::Get().AwaitTurn();
    drainRequests[std::min<std::uint32_t>(opcode, 0x100)].fetch_add(1, std::memory_order_relaxed);
    reasonRequests[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    const bool finisherBusy = DrawFinisher::Busy() && !DrawFinisher::OnFinisherThread();
    if ((finisherBusy || Busy()) && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline drain under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    if (finisherBusy) DrawFinisher::Get().Drain();
    if (!Busy()) return;
    const ThreadReport::WorkerStateScope draining(ThreadReport::WorkerState::DrainCommitter, ThreadReport::OnQueue0Worker());
    const auto start = std::chrono::steady_clock::now();
    SpinUntil(HandoffSpinWindow() * 2, [&] { return !Busy(); });
    auto lock = lockFor(mutex);
    ++idleWaiters;
    idle.wait(lock, [&] { return items.empty(); });
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

std::uint64_t DrawPipeline::DrainRequestsByOpcode(std::uint32_t opcode) {
    return Queue0().drainRequests[std::min<std::uint32_t>(opcode, 0x100)].load(std::memory_order_relaxed);
}

bool DrawPipeline::HeldReads(std::uint64_t address, std::size_t bytes) {
    static const bool unordered = std::getenv("APS5_NO_QUEUED_DRAW_READ_ORDER") != nullptr;
    if (bytes == 0 || unordered) return false;
    if (FinisherDeposits::Get() && DrawFinisher::Get().depositsRead(address, bytes)) return true;
    if (DrawFinisher::Get().ParkedReadsOverlap(address, bytes)) return true;
    const auto end = address + bytes;
    auto& pipeline = Queue0();
    const auto lock = lockFor(pipeline.mutex);
    heldReadQueries.fetch_add(1, std::memory_order_relaxed);
    for (const auto& item : pipeline.items) {
        for (const auto& [begin, limit] : item.reads) {
            if (address >= limit || begin >= end) continue;
            heldReadHits.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

std::uint64_t DrawPipeline::HeldReadQueries() {
    return heldReadQueries.load(std::memory_order_relaxed);
}

void DrawPipeline::ReleaseRunningReads() {
    const auto lock = lockFor(mutex);
    if (!items.empty()) items.front().reads.clear();
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
    if (lockless()) {
        const Range range{address, end};
        if (published.Overlaps({&range, 1}) == PublishedRanges::Answer::Clear) return std::nullopt;
    }
    const auto lock = lockFor(mutex);
    for (auto item = items.rbegin(); item != items.rend(); ++item) {
        const bool overlaps = std::any_of(item->writes.begin(), item->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
        if (!overlaps) continue;
        if (item->labelBytes.empty() || address < item->labelAddress || end > item->labelAddress + item->labelBytes.size()) return std::nullopt;
        std::uint64_t value = 0;
        std::memcpy(&value, item->labelBytes.data() + (address - item->labelAddress), bytes);
        return value;
    }
    return std::nullopt;
}

DrawPipeline::QueuedWrite DrawPipeline::StoredBytes(std::uint64_t address, std::size_t bytes, std::span<std::byte> out) {
    const auto end = address + bytes;
    if (FinisherDeposits::Get()) {
        const Range range{address, end};
        if (DrawFinisher::Get().depositsWrite({&range, 1})) return QueuedWrite::Unknown;
    }
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlaps(address, end)) return QueuedWrite::Unknown;
    if (lockless()) {
        const Range range{address, end};
        if (published.Overlaps({&range, 1}) == PublishedRanges::Answer::Clear) return QueuedWrite::None;
    }
    auto answer = QueuedWrite::None;
    const auto lock = lockFor(mutex);
    for (auto item = items.rbegin(); item != items.rend(); ++item) {
        if (std::none_of(item->writes.begin(), item->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; })) continue;
        if (!item->store) return QueuedWrite::Unknown;
        if (out.empty()) {
            answer = QueuedWrite::Stored;
            continue;
        }
        if (out.size() != bytes || address < item->labelAddress || end > item->labelAddress + item->labelBytes.size()) return QueuedWrite::Unknown;
        std::memcpy(out.data(), item->labelBytes.data() + (address - item->labelAddress), bytes);
        return QueuedWrite::Stored;
    }
    return answer;
}

void DrawPipeline::countWrites(const std::vector<Range>& writes, bool added) {
    bool changed = false;
    for (const auto& range : writes) changed = countRange(queuedWrites, range, added) || changed;
    if (changed) published.Publish(queuedWrites, 0);
}

DrawPipeline::Watch DrawPipeline::WatchWrites() {
    if (!Busy()) return {Retired(), 0, {}};
    const bool own = FinisherDeposits::Get();
    if (lockless() && !own) {
        Watch watch;
        const auto enqueuedBefore = enqueuedItems.load(std::memory_order_acquire);
        if (published.Copy(watch.snapshotWrites, watch.snapshotCount)) {
            watch.retired = retiredItems.load(std::memory_order_acquire);
            watch.items = enqueuedBefore > watch.retired ? static_cast<std::size_t>(enqueuedBefore - watch.retired) : 0;
            watch.snapshot = true;
            return watch;
        }
    }
    const auto lock = lockFor(mutex);
    Watch watch{retiredItems.load(std::memory_order_relaxed), items.size(), {}};
    for (const auto& item : items) watch.writes.insert(watch.writes.end(), item.writes.begin(), item.writes.end());
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
    if (lockless()) {
        const auto queued = published.Overlaps(reads);
        if (queued == PublishedRanges::Answer::Hit) return true;
        if (queued == PublishedRanges::Answer::Clear) {
            const auto retiredNow = retiredItems.load(std::memory_order_acquire);
            if (retiredNow == watch.retired) return false;
            return retiredNow - watch.retired > watch.items || overlaps(watched);
        }
    }
    const auto lock = lockFor(mutex);
    if (std::any_of(items.begin(), items.end(), [&](const Item& item) { return overlaps(item.writes); })) return true;
    const auto retiredNow = retiredItems.load(std::memory_order_relaxed);
    if (retiredNow == watch.retired) return false;
    return retiredNow - watch.retired > watch.items || overlaps(watched);
}

bool DrawPipeline::Busy() const {
    return (FinisherProducer::Get() && DrawFinisher::Busy() && DrawFinisher::Overlap()) || outstanding.load(std::memory_order_acquire) != 0 || DrawFinisher::AheadOfTurn() || FinisherDeposits::Get();
}

bool DrawPipeline::OverlapsAny(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (FinisherDeposits::Get() && DrawFinisher::Get().depositsWrite(ranges)) return true;
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlapsAny(ranges)) return true;
    if (lockless()) {
        const auto answer = published.Overlaps(ranges);
        if (answer != PublishedRanges::Answer::Unsure) return answer == PublishedRanges::Answer::Hit;
    }
    const auto lock = lockFor(mutex);
    for (const auto& [range, count] : queuedWrites) {
        for (const auto& [address, end] : ranges) {
            if (address < range.second && range.first < end) return true;
        }
    }
    return false;
}

bool DrawPipeline::Overlaps(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    if (FinisherDeposits::Get()) {
        const Range range{address, end};
        if (DrawFinisher::Get().depositsWrite({&range, 1})) return true;
    }
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlaps(address, end)) return true;
    if (lockless()) {
        const Range range{address, end};
        const auto answer = published.Overlaps({&range, 1});
        if (answer != PublishedRanges::Answer::Unsure) return answer == PublishedRanges::Answer::Hit;
    }
    const auto lock = lockFor(mutex);
    for (const auto& [range, count] : queuedWrites) {
        if (address < range.second && range.first < end) return true;
    }
    return false;
}

}
