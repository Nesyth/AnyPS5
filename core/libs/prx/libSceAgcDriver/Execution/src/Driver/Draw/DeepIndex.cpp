#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DeepIndex.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::uint64_t Golden = 0x9E3779B97F4A7C15ull;
constexpr std::uint64_t ListMix = 0xC2B2AE3D27D4EB4Full;
constexpr std::uint64_t NewestSlack = 64;
constexpr std::uint64_t Unbounded = ~std::uint64_t{0};
constexpr std::int64_t ReportNs = 10'000'000'000;
constexpr std::uint64_t UnrecordedSamples = 8;
constexpr std::uint64_t AllRuns = StampedRanges::RunEntries == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << (StampedRanges::RunEntries % 64)) - 1;

std::size_t hashed(std::uint64_t value, unsigned bits) {
    return static_cast<std::size_t>((value * Golden) >> (64u - bits));
}

bool censusOn() {
    static const bool on = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return on;
}

bool runsSwitched() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::DeepRuns);
}

unsigned long long wideCount(std::uint64_t value) {
    return static_cast<unsigned long long>(value);
}

std::int64_t steadyNs() {
    return static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct MapLayout {
    std::array<std::size_t, StampedRanges::Levels> own{};
    std::array<std::size_t, StampedRanges::Levels> deep{};
    std::size_t huge = 0;
    std::size_t total = 0;
};

constexpr MapLayout makeLayout() {
    MapLayout layout{};
    std::size_t at = 0;
    for (std::size_t level = 0; level < StampedRanges::Levels; ++level) {
        layout.own[level] = at;
        at += std::size_t{1} << StampedRanges::OwnBits[level];
    }
    for (std::size_t level = 0; level < StampedRanges::Levels; ++level) {
        layout.deep[level] = at;
        if (level != 0) at += std::size_t{1} << StampedRanges::DeepBits[level];
    }
    layout.huge = at;
    layout.total = at + 1;
    return layout;
}

constexpr MapLayout Layout = makeLayout();

std::size_t ownMap(std::size_t level, std::uint64_t block) {
    return Layout.own[level] + hashed(block, StampedRanges::OwnBits[level]);
}

std::size_t deepMap(std::size_t level, std::uint64_t block) {
    return level == 0 ? ownMap(0, block) : Layout.deep[level] + hashed(block, StampedRanges::DeepBits[level]);
}

std::uint64_t lastByte(const StampedRanges::Range& range) {
    return range.second > range.first ? range.second - 1 : range.first;
}

std::size_t ownerLevel(const StampedRanges::Range& range) {
    if (range.second < range.first) return StampedRanges::Levels;
    const auto last = lastByte(range);
    for (std::size_t level = 0; level < StampedRanges::Levels; ++level) {
        const auto shift = StampedRanges::Shifts[level];
        if ((last >> shift) - (range.first >> shift) < StampedRanges::WriteBlocks) return level;
    }
    return StampedRanges::Levels;
}

template<typename TVisit>
void forEachPlacement(const StampedRanges::Range& range, TVisit&& visit) {
    const auto level = ownerLevel(range);
    if (level == StampedRanges::Levels) {
        visit(Layout.huge);
        return;
    }
    const auto last = lastByte(range);
    const auto shift = StampedRanges::Shifts[level];
    const auto ownFirst = range.first >> shift;
    const auto ownLast = last >> shift;
    visit(ownMap(level, ownFirst));
    if (ownLast != ownFirst) visit(ownMap(level, ownLast));
    for (std::size_t deep = std::max<std::size_t>(level, 1); deep < StampedRanges::Levels; ++deep) {
        const auto deepShift = StampedRanges::Shifts[deep];
        const auto deepFirst = range.first >> deepShift;
        const auto deepLast = last >> deepShift;
        visit(deepMap(deep, deepFirst));
        if (deepLast != deepFirst) visit(deepMap(deep, deepLast));
    }
}

bool overlapsAsked(const StampedRanges::Range& write, std::span<const StampedRanges::Range> asked) {
    for (const auto& [begin, end] : asked) {
        if (begin < write.second && write.first < end) return true;
    }
    return false;
}

bool invertedAny(std::span<const StampedRanges::Range> ranges) {
    for (const auto& [begin, end] : ranges) {
        if (end < begin) return true;
    }
    return false;
}

void raiseNewest(std::atomic<std::uint64_t>& newest, std::uint64_t stamp) {
    if (newest.load(std::memory_order_relaxed) >= stamp) return;
    newest.store(stamp > Unbounded - NewestSlack ? Unbounded : stamp + NewestSlack, std::memory_order_relaxed);
}

void raiseOver(std::atomic<std::uint64_t>* table, unsigned shift, unsigned bits, const StampedRanges::Range& range, std::uint64_t stamp) {
    const auto last = lastByte(range) >> shift;
    for (auto at = range.first >> shift; at <= last; ++at) raiseNewest(table[hashed(at, bits)], stamp);
}

bool touchedSince(const std::atomic<std::uint64_t>* table, unsigned shift, unsigned bits, std::uint64_t limit, std::uint64_t first, std::uint64_t last, std::uint64_t lo) {
    const auto low = first >> shift;
    const auto high = last >> shift;
    if (high - low >= limit) return true;
    for (auto at = low; at <= high; ++at) {
        if (table[hashed(at, bits)].load(std::memory_order_relaxed) >= lo) return true;
    }
    return false;
}

bool runRegions(const StampedRanges::Range& range, std::uint64_t limit, std::uint64_t& first, std::uint64_t& last) {
    if (range.second < range.first) return false;
    first = range.first >> StampedRanges::NearShift;
    last = lastByte(range) >> StampedRanges::NearShift;
    return last - first < limit;
}

std::uint64_t listKey(std::span<const StampedRanges::Range> ranges) {
    auto key = static_cast<std::uint64_t>(ranges.size()) * Golden;
    for (const auto& [begin, end] : ranges) {
        key = (key ^ begin) * Golden;
        key = (key ^ end) * ListMix;
        key ^= key >> 29;
    }
    return key;
}

bool sameRanges(const std::array<StampedRanges::Range, StampedRanges::RunRanges>& held, std::uint32_t count, std::span<const StampedRanges::Range> ranges) {
    if (count != ranges.size()) return false;
    for (std::size_t at = 0; at < ranges.size(); ++at) {
        if (held[at] != ranges[at]) return false;
    }
    return true;
}

std::uint64_t spanBits(std::uint64_t from, std::uint64_t to, std::uint64_t word) {
    const auto base = word * 64;
    const auto low = std::max(from, base);
    const auto high = std::min(to, base + 64);
    if (low >= high) return 0;
    const auto width = high - low;
    const auto ones = width >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << width) - 1;
    return ones << (low - base);
}

std::uint32_t wordsBetween(std::uint64_t first, std::uint64_t last) {
    return static_cast<std::uint32_t>(((std::uint64_t{2} << last) - 1) & ~((std::uint64_t{1} << first) - 1));
}

template<typename TCandidates, typename TTest>
void walkNewestFirst(std::uint64_t hi, const TCandidates& candidates, TTest&& test, std::uint64_t& tested) {
    constexpr std::size_t words = StampedRanges::Capacity / 64;
    const auto top = static_cast<std::size_t>((hi - 1) & (StampedRanges::Capacity - 1));
    const auto topWord = top >> 6;
    const auto topBit = top & 63;
    const auto below = topBit == 63 ? ~std::uint64_t{0} : (std::uint64_t{2} << topBit) - 1;
    for (std::size_t step = 0; step <= words; ++step) {
        const auto word = (topWord + words - step % words) % words;
        if ((candidates.words & (std::uint32_t{1} << word)) == 0) continue;
        auto bits = candidates.bits[word];
        if (step == 0) bits &= below;
        else if (step == words) bits &= ~below;
        while (bits != 0) {
            const auto bit = static_cast<std::size_t>(63 - std::countl_zero(bits));
            bits &= ~(std::uint64_t{1} << bit);
            ++tested;
            if (test(word * 64 + bit)) return;
        }
    }
}

struct CensusStripeTag {};
std::atomic<std::uint32_t> nextCensusStripe{0};

const char* const LostNames[] = {"beyond capacity", "out of order", "spill pool full"};

}

StampedRanges::StampedRanges()
    : metas(std::make_unique<Meta[]>(Layout.total)),
      maps(std::make_unique<Map[]>(Layout.total)),
      slots(std::make_unique<Slot[]>(Capacity)),
      blocks(std::make_unique<Block[]>(PoolBlocks)),
      freeBlocks(std::make_unique<std::uint32_t[]>(PoolBlocks)),
      marks(std::make_unique<std::uint64_t[]>(Layout.total)),
      nears(std::make_unique<Near[]>(std::size_t{1} << NearBits)),
      pages(std::make_unique<std::atomic<std::uint64_t>[]>(std::size_t{1} << PageBits)),
      runs(std::make_unique<RunSlot[]>(RunEntries)),
      runStates(std::make_unique<RunState[]>(RunEntries)),
      freeCount(PoolBlocks) {
    for (std::size_t at = 0; at < PoolBlocks; ++at) freeBlocks[at] = static_cast<std::uint32_t>(PoolBlocks - 1 - at);
}

StampedRanges::~StampedRanges() {
    if (censusOn() && written.inserts != 0) report();
}

bool StampedRanges::Regions(const Range& range, std::uint64_t limit, std::uint64_t& first, std::uint64_t& last) {
    if (range.second < range.first) return false;
    first = range.first >> RegionShift;
    last = (range.second > range.first ? range.second - 1 : range.first) >> RegionShift;
    return last - first < limit;
}

bool StampedRanges::CoarseRegions(const Range& range, std::uint64_t limit, std::uint64_t& first, std::uint64_t& last) {
    if (range.second < range.first) return false;
    first = range.first >> CoarseShift;
    last = (range.second > range.first ? range.second - 1 : range.first) >> CoarseShift;
    return last - first < limit;
}

StampedRanges::Window StampedRanges::makeWindow(std::uint64_t lo, std::uint64_t hi) {
    Window window;
    window.lo = lo;
    window.hi = hi;
    window.first = lo & SlotMask;
    window.length = std::min<std::uint64_t>(hi - lo, Capacity);
    if (window.length >= Capacity) {
        window.words = static_cast<std::uint32_t>((std::uint64_t{1} << Words) - 1);
        return window;
    }
    const auto end = window.first + window.length;
    const auto firstWord = window.first >> 6;
    const auto lastWord = ((end - 1) & SlotMask) >> 6;
    window.words = end <= Capacity ? wordsBetween(firstWord, lastWord) : wordsBetween(firstWord, Words - 1) | wordsBetween(0, lastWord);
    return window;
}

std::uint64_t StampedRanges::windowBits(const Window& window, std::size_t word) {
    if (window.length >= Capacity) return ~std::uint64_t{0};
    const auto end = window.first + window.length;
    if (end <= Capacity) return spanBits(window.first, end, word);
    return spanBits(window.first, Capacity, word) | spanBits(0, end - Capacity, word);
}

std::size_t StampedRanges::blocksFor(std::size_t count) {
    return count > Inline ? (count - Inline + BlockRanges - 1) / BlockRanges : 0;
}

bool StampedRanges::occupiedAt(std::size_t slot) const {
    return ((occupied[slot >> 6] >> (slot & 63)) & 1u) != 0;
}

void StampedRanges::setOccupied(std::size_t slot, bool held) {
    const auto bit = std::uint64_t{1} << (slot & 63);
    if (held) occupied[slot >> 6] |= bit;
    else occupied[slot >> 6] &= ~bit;
}

void StampedRanges::setBit(std::size_t map, std::size_t slot, std::uint64_t stamp) {
    auto& meta = metas[map];
    const auto word = slot >> 6;
    const auto bit = std::uint64_t{1} << (slot & 63);
    auto& held = maps[map].words[word];
    const auto value = held.load(std::memory_order_relaxed);
    if ((value & bit) == 0) {
        held.store(value | bit, std::memory_order_relaxed);
        if (value == 0) meta.summary.store(meta.summary.load(std::memory_order_relaxed) | (std::uint64_t{1} << word), std::memory_order_relaxed);
        ++written.bitsSet;
    }
    raiseNewest(meta.newest, stamp);
}

void StampedRanges::clearBit(std::size_t map, std::size_t slot) {
    const auto word = slot >> 6;
    const auto bit = std::uint64_t{1} << (slot & 63);
    auto& held = maps[map].words[word];
    const auto value = held.load(std::memory_order_relaxed);
    if ((value & bit) == 0) return;
    const auto left = value & ~bit;
    held.store(left, std::memory_order_relaxed);
    if (left == 0) {
        auto& meta = metas[map];
        meta.summary.store(meta.summary.load(std::memory_order_relaxed) & ~(std::uint64_t{1} << word), std::memory_order_relaxed);
    }
    ++written.bitsCleared;
}

void StampedRanges::releaseSlot(std::size_t slot, bool keepMarked) {
    const auto& held = slots[slot];
    const auto count = static_cast<std::size_t>(held.count.load(std::memory_order_relaxed));
    const auto clear = [&](std::size_t map) {
        if (!keepMarked || marks[map] != markSerial) clearBit(map, slot);
    };
    const auto inlined = std::min(count, Inline);
    for (std::size_t at = 0; at < inlined; ++at) {
        const Range range{held.cells[at].begin.load(std::memory_order_relaxed), held.cells[at].end.load(std::memory_order_relaxed)};
        forEachPlacement(range, clear);
    }
    auto block = held.head.load(std::memory_order_relaxed);
    std::size_t left = count > Inline ? count - Inline : 0;
    while (left != 0 && block != NoBlock) {
        const auto& cells = blocks[block].cells;
        const auto take = std::min(left, BlockRanges);
        for (std::size_t at = 0; at < take; ++at) {
            const Range range{cells[at].begin.load(std::memory_order_relaxed), cells[at].end.load(std::memory_order_relaxed)};
            forEachPlacement(range, clear);
        }
        left -= take;
        const auto next = blocks[block].next.load(std::memory_order_relaxed);
        freeBlocks[freeCount++] = block;
        block = next;
    }
    setOccupied(slot, false);
}

void StampedRanges::evictSlot(std::size_t slot) {
    releaseSlot(slot, false);
    auto& held = slots[slot];
    const auto version = held.version.load(std::memory_order_relaxed);
    held.version.store(version + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    held.count.store(0, std::memory_order_relaxed);
    held.head.store(NoBlock, std::memory_order_relaxed);
    held.version.store(version + 2, std::memory_order_release);
    ++written.evicted;
}

void StampedRanges::evictSkipped(std::uint64_t first, std::uint64_t length) {
    if (length == 0) return;
    const auto skipped = makeWindow(first, first + length);
    auto words = skipped.words;
    while (words != 0) {
        const auto word = static_cast<std::size_t>(std::countr_zero(words));
        words &= words - 1;
        auto bits = occupied[word] & windowBits(skipped, word);
        while (bits != 0) {
            const auto bit = static_cast<std::size_t>(std::countr_zero(bits));
            bits &= bits - 1;
            evictSlot(word * 64 + bit);
        }
    }
}

std::size_t StampedRanges::heldSkipped(std::uint64_t first, std::uint64_t length) const {
    if (length == 0) return 0;
    const auto skipped = makeWindow(first, first + length);
    auto words = skipped.words;
    std::size_t held = 0;
    while (words != 0) {
        const auto word = static_cast<std::size_t>(std::countr_zero(words));
        words &= words - 1;
        auto bits = occupied[word] & windowBits(skipped, word);
        while (bits != 0) {
            const auto bit = static_cast<std::size_t>(std::countr_zero(bits));
            bits &= bits - 1;
            held += blocksFor(static_cast<std::size_t>(slots[word * 64 + bit].count.load(std::memory_order_relaxed)));
        }
    }
    return held;
}

void StampedRanges::writeCells(Slot& slot, std::size_t from, std::span<const Range> ranges) {
    auto count = from;
    for (const auto& range : ranges) {
        if (count < Inline) {
            slot.cells[count].begin.store(range.first, std::memory_order_relaxed);
            slot.cells[count].end.store(range.second, std::memory_order_relaxed);
        } else {
            const auto spill = count - Inline;
            if (spill % BlockRanges == 0) {
                const auto block = freeBlocks[--freeCount];
                blocks[block].next.store(NoBlock, std::memory_order_relaxed);
                if (spill == 0) slot.head.store(block, std::memory_order_relaxed);
                else blocks[currentTail].next.store(block, std::memory_order_relaxed);
                currentTail = block;
            }
            auto& cell = blocks[currentTail].cells[spill % BlockRanges];
            cell.begin.store(range.first, std::memory_order_relaxed);
            cell.end.store(range.second, std::memory_order_relaxed);
        }
        ++count;
    }
    slot.count.store(static_cast<std::uint32_t>(count), std::memory_order_relaxed);
    written.spillHighest = std::max<std::uint64_t>(written.spillHighest, PoolBlocks - freeCount);
}

void StampedRanges::raiseNears(std::atomic<std::uint64_t> Near::* word, const Range& range, std::uint64_t stamp) {
    const auto last = lastByte(range) >> NearShift;
    for (auto region = range.first >> NearShift; region <= last; ++region) raiseNewest(nears[hashed(region, NearBits)].*word, stamp);
}

void StampedRanges::placeRanges(std::span<const Range> ranges, std::size_t slot, std::uint64_t stamp) {
    for (const auto& range : ranges) {
        const auto level = ownerLevel(range);
        raiseNewest(classes.newest[level], stamp);
        ++written.ranges[level];
        if (level < FineLevels) {
            raiseOver(pages.get(), PageShift, PageBits, range, stamp);
            raiseNears(&Near::fine, range, stamp);
        } else if (level != Levels) {
            raiseNears(&Near::coarse, range, stamp);
        }
        forEachPlacement(range, [&](std::size_t map) { setBit(map, slot, stamp); });
    }
}

void StampedRanges::unrecord(std::uint64_t stamp, std::uint64_t live, std::size_t count, Lost reason) {
    const auto lost = stamp == Unbounded ? Unbounded : stamp + 1;
    if (front.lost.load(std::memory_order_relaxed) < lost) front.lost.store(lost, std::memory_order_release);
    if (stamp == current) currentRecorded = false;
    overflows.store(overflows.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    const auto total = ++written.unrecorded[static_cast<std::size_t>(reason)];
    if (!censusOn() || total > UnrecordedSamples) return;
    std::fprintf(stderr, "[deep-index] %s sample: stamp %llu not recorded (%s): oldest live %llu, newest recorded %llu, %llu ranges, %llu spill blocks free\n", label, wideCount(stamp), LostNames[static_cast<std::size_t>(reason)], wideCount(live), wideCount(front.newest.load(std::memory_order_relaxed)), wideCount(count), wideCount(freeCount));
}

void StampedRanges::append(std::span<const Range> ranges, std::uint64_t stamp, std::uint64_t live) {
    const auto index = static_cast<std::size_t>(stamp & SlotMask);
    auto& slot = slots[index];
    const auto count = static_cast<std::size_t>(slot.count.load(std::memory_order_relaxed));
    if (ranges.size() > MaxRanges - count || blocksFor(count + ranges.size()) - blocksFor(count) > freeCount) {
        unrecord(stamp, live, count + ranges.size(), Lost::Pool);
        return;
    }
    const auto version = slot.version.load(std::memory_order_relaxed);
    slot.version.store(version + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    writeCells(slot, count, ranges);
    slot.version.store(version + 2, std::memory_order_release);
    placeRanges(ranges, index, stamp);
    ++written.appends;
}

void StampedRanges::appendFresh(std::span<const Range> ranges, std::uint64_t stamp, std::uint64_t live) {
    if (ranges.size() > MaxRanges || blocksFor(ranges.size()) > freeCount) {
        unrecord(stamp, live, ranges.size(), Lost::Pool);
        return;
    }
    const auto index = static_cast<std::size_t>(stamp & SlotMask);
    auto& slot = slots[index];
    const auto version = slot.version.load(std::memory_order_relaxed);
    slot.version.store(version + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    slot.stamp.store(stamp, std::memory_order_relaxed);
    slot.head.store(NoBlock, std::memory_order_relaxed);
    writeCells(slot, 0, ranges);
    slot.version.store(version + 2, std::memory_order_release);
    setOccupied(index, true);
    placeRanges(ranges, index, stamp);
    currentInSlot = true;
    ++written.appends;
    ++written.runFreshSlots;
}

std::uint32_t StampedRanges::findOpenRun(std::span<const Range> ranges, std::uint64_t key, std::uint64_t stamp) {
    if (runHint != NoRun && ((runsOpen >> runHint) & 1u) != 0) {
        const auto& state = runStates[runHint];
        if (state.key == key && sameRanges(state.ranges, state.count, ranges)) return runHint;
    }
    auto open = runsOpen;
    while (open != 0) {
        const auto entry = static_cast<std::uint32_t>(std::countr_zero(open));
        open &= open - 1;
        const auto& state = runStates[entry];
        if (stamp - state.last > RunGap) {
            closeRun(entry, Closed::Gap);
            continue;
        }
        if (entry == runHint) continue;
        if (state.key == key && sameRanges(state.ranges, state.count, ranges)) return entry;
    }
    return NoRun;
}

std::size_t StampedRanges::findRecent(std::span<const Range> ranges, std::uint64_t key) const {
    for (std::size_t at = 0; at < recentCount; ++at) {
        const auto& held = recent[recentOrder[at]];
        if (held.key == key && sameRanges(held.ranges, held.count, ranges)) return recentOrder[at];
    }
    return RecentLists;
}

void StampedRanges::noteRecent(std::span<const Range> ranges, std::uint64_t key, std::uint64_t stamp) {
    const auto seen = findRecent(ranges, key);
    std::size_t position = 0;
    std::size_t entry = 0;
    if (seen != RecentLists) {
        entry = seen;
        while (recentOrder[position] != seen) ++position;
    } else {
        if (recentCount < RecentLists) {
            position = recentCount;
            entry = recentCount;
            ++recentCount;
        } else {
            position = RecentLists - 1;
            entry = recentOrder[position];
        }
        auto& held = recent[entry];
        held.key = key;
        held.count = static_cast<std::uint32_t>(ranges.size());
        std::copy(ranges.begin(), ranges.end(), held.ranges.begin());
    }
    recent[entry].stamp = stamp;
    for (; position > 0; --position) recentOrder[position] = recentOrder[position - 1];
    recentOrder[0] = static_cast<std::uint8_t>(entry);
}

bool StampedRanges::runAvailable(std::uint64_t stamp) const {
    if ((~runsUsed & AllRuns) != 0) return true;
    if (runOldest + Capacity > stamp) return false;
    auto used = runsUsed;
    while (used != 0) {
        const auto entry = static_cast<std::size_t>(std::countr_zero(used));
        used &= used - 1;
        if (runStates[entry].last + Capacity <= stamp) return true;
    }
    return false;
}

void StampedRanges::noteRunStamp(std::uint64_t stamp) {
    const auto previous = front.newest.load(std::memory_order_relaxed);
    std::uint64_t first = previous + 1;
    std::uint64_t length = stamp > previous ? stamp - previous : 0;
    if (length > Capacity) {
        length = Capacity;
        first = stamp + 1 - Capacity;
    }
    front.newest.store(stamp, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    evictSkipped(first, length);
    if (runsUsed != 0 && runOldest + Capacity <= stamp) reclaimRuns(stamp);
    currentRecorded = true;
    ++written.runStamps;
}

void StampedRanges::openRun(std::uint32_t entry, std::span<const Range> ranges, std::uint64_t key, std::uint64_t stamp) {
    auto& run = runs[entry];
    const auto version = run.version.load(std::memory_order_relaxed);
    run.version.store(version + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    run.first.store(stamp, std::memory_order_relaxed);
    run.last.store(stamp, std::memory_order_relaxed);
    run.count.store(static_cast<std::uint32_t>(ranges.size()), std::memory_order_relaxed);
    for (std::size_t at = 0; at < ranges.size(); ++at) {
        run.cells[at].begin.store(ranges[at].first, std::memory_order_relaxed);
        run.cells[at].end.store(ranges[at].second, std::memory_order_relaxed);
    }
    run.version.store(version + 2, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    const auto bit = std::uint64_t{1} << entry;
    auto& state = runStates[entry];
    state.key = key;
    state.first = stamp;
    state.last = stamp;
    state.count = static_cast<std::uint32_t>(ranges.size());
    state.wide = false;
    std::copy(ranges.begin(), ranges.end(), state.ranges.begin());
    for (const auto& range : ranges) {
        std::uint64_t first = 0;
        std::uint64_t last = 0;
        if (!runRegions(range, RunRegionLimit, first, last)) {
            state.wide = true;
            continue;
        }
        for (auto region = first; region <= last; ++region) {
            auto& word = nears[hashed(region, NearBits)].runs;
            word.store(word.load(std::memory_order_relaxed) | bit, std::memory_order_relaxed);
        }
    }
    if (state.wide) front.wideRuns.store(front.wideRuns.load(std::memory_order_relaxed) | bit, std::memory_order_relaxed);
    runFront.held.store(runFront.held.load(std::memory_order_relaxed) | bit, std::memory_order_relaxed);
    if (runsUsed == 0) runOldest = stamp;
    runsUsed |= bit;
    runsOpen |= bit;
    runHint = entry;
    ++written.runsOpened;
    written.runsHighest = std::max<std::uint64_t>(written.runsHighest, static_cast<std::uint64_t>(std::popcount(runsUsed)));
}

void StampedRanges::closeRun(std::uint32_t entry, Closed reason) {
    runsOpen &= ~(std::uint64_t{1} << entry);
    if (runHint == entry) runHint = NoRun;
    ++written.runsClosed[static_cast<std::size_t>(reason)];
}

void StampedRanges::reclaimRuns(std::uint64_t stamp) {
    std::uint64_t retired = 0;
    std::uint64_t oldest = Unbounded;
    auto used = runsUsed;
    while (used != 0) {
        const auto entry = static_cast<std::uint32_t>(std::countr_zero(used));
        used &= used - 1;
        const auto& state = runStates[entry];
        if (state.last + Capacity > stamp) {
            oldest = std::min(oldest, state.last);
            continue;
        }
        retired |= std::uint64_t{1} << entry;
        raiseNewest(runFront.recycled, state.last);
        for (std::size_t at = 0; at < state.count; ++at) {
            std::uint64_t first = 0;
            std::uint64_t last = 0;
            if (!runRegions(state.ranges[at], RunRegionLimit, first, last)) {
                raiseNewest(runFront.wideBound, state.last);
                continue;
            }
            for (auto region = first; region <= last; ++region) raiseNewest(nears[hashed(region, NearBits)].runBound, state.last);
        }
    }
    runOldest = oldest == Unbounded ? stamp : oldest;
    if (retired == 0) return;
    std::atomic_thread_fence(std::memory_order_release);
    used = retired;
    while (used != 0) {
        const auto entry = static_cast<std::uint32_t>(std::countr_zero(used));
        used &= used - 1;
        const auto bit = std::uint64_t{1} << entry;
        const auto& state = runStates[entry];
        for (std::size_t at = 0; at < state.count; ++at) {
            std::uint64_t first = 0;
            std::uint64_t last = 0;
            if (!runRegions(state.ranges[at], RunRegionLimit, first, last)) continue;
            for (auto region = first; region <= last; ++region) {
                auto& word = nears[hashed(region, NearBits)].runs;
                word.store(word.load(std::memory_order_relaxed) & ~bit, std::memory_order_relaxed);
            }
        }
    }
    const auto wideRuns = front.wideRuns.load(std::memory_order_relaxed);
    if ((wideRuns & retired) != 0) front.wideRuns.store(wideRuns & ~retired, std::memory_order_relaxed);
    runFront.held.store(runFront.held.load(std::memory_order_relaxed) & ~retired, std::memory_order_relaxed);
    runsUsed &= ~retired;
    runsOpen &= ~retired;
    if (runHint != NoRun && ((retired >> runHint) & 1u) != 0) runHint = NoRun;
    written.runsReclaimed += static_cast<std::uint64_t>(std::popcount(retired));
}

bool StampedRanges::recordInRun(std::span<const Range> ranges, std::uint64_t key, std::uint64_t stamp, std::uint64_t live) {
    const auto open = findOpenRun(ranges, key, stamp);
    if (open != NoRun) {
        auto& state = runStates[open];
        if (state.last >= live && stamp - state.last <= RunGap) {
            noteRunStamp(stamp);
            runs[open].last.store(stamp, std::memory_order_release);
            state.last = stamp;
            runHint = open;
            ++written.runsExtended;
            return true;
        }
        closeRun(open, state.last < live ? Closed::Live : Closed::Gap);
    }
    const auto seen = findRecent(ranges, key);
    if (seen == RecentLists) return false;
    const auto& last = recent[seen];
    if (last.stamp < live || stamp - last.stamp > RunGap) return false;
    if (!runAvailable(stamp)) {
        ++written.runsNoEntry;
        return false;
    }
    noteRunStamp(stamp);
    if ((~runsUsed & AllRuns) == 0) reclaimRuns(stamp);
    const auto free = ~runsUsed & AllRuns;
    if (free == 0) return false;
    openRun(static_cast<std::uint32_t>(std::countr_zero(free)), ranges, key, stamp);
    return true;
}

void StampedRanges::noteInsert() {
    ++written.inserts;
    if (!censusOn() || (written.inserts & 1023u) != 0) return;
    const auto now = steadyNs();
    if (written.reportedAt == 0) {
        written.reportedAt = now;
        return;
    }
    if (now - written.reportedAt < ReportNs) return;
    written.reportedAt = now;
    report();
}

void StampedRanges::Insert(std::span<const Range> ranges, std::uint64_t stamp, std::uint64_t live) {
    if (ranges.empty()) return;
    noteInsert();
    if (started && stamp < current) {
        unrecord(stamp, live, ranges.size(), Lost::Disorder);
        return;
    }
    if (started && stamp == current) {
        if (!currentRecorded) return;
        if (currentInSlot) append(ranges, stamp, live);
        else appendFresh(ranges, stamp, live);
        return;
    }
    started = true;
    current = stamp;
    currentRecorded = false;
    currentInSlot = false;
    currentTail = NoBlock;
    if (stamp >= live && stamp - live >= Capacity) {
        unrecord(stamp, live, ranges.size(), Lost::Capacity);
        return;
    }
    const bool switched = runsSwitched();
    const bool listed = switched && ranges.size() <= RunRanges;
    if (switched && !listed) ++written.runsLongLists;
    std::uint64_t key = 0;
    if (listed) {
        key = listKey(ranges);
        if (recordInRun(ranges, key, stamp, live)) return;
    }
    const auto index = static_cast<std::size_t>(stamp & SlotMask);
    const auto previous = front.newest.load(std::memory_order_relaxed);
    std::uint64_t skippedFirst = previous + 1;
    std::uint64_t skipped = stamp > previous ? stamp - previous - 1 : 0;
    if (skipped > Capacity - 1) {
        skipped = Capacity - 1;
        skippedFirst = stamp - skipped;
    }
    const auto needed = blocksFor(ranges.size());
    auto available = freeCount;
    if (occupiedAt(index)) available += blocksFor(static_cast<std::size_t>(slots[index].count.load(std::memory_order_relaxed)));
    if (needed > available) available += heldSkipped(skippedFirst, skipped);
    if (ranges.size() > MaxRanges || needed > available) {
        unrecord(stamp, live, ranges.size(), Lost::Pool);
        return;
    }
    front.newest.store(stamp, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    evictSkipped(skippedFirst, skipped);
    if (runsUsed != 0 && runOldest + Capacity <= stamp) reclaimRuns(stamp);
    if (occupiedAt(index)) {
        ++markSerial;
        for (const auto& range : ranges) forEachPlacement(range, [&](std::size_t map) { marks[map] = markSerial; });
        releaseSlot(index, true);
    }
    auto& slot = slots[index];
    const auto version = slot.version.load(std::memory_order_relaxed);
    slot.version.store(version + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    slot.stamp.store(stamp, std::memory_order_relaxed);
    slot.head.store(NoBlock, std::memory_order_relaxed);
    writeCells(slot, 0, ranges);
    slot.version.store(version + 2, std::memory_order_release);
    setOccupied(index, true);
    placeRanges(ranges, index, stamp);
    currentRecorded = true;
    currentInSlot = true;
    ++written.stamps;
    if (listed) noteRecent(ranges, key, stamp);
}

StampedRanges::ReadCensus* StampedRanges::readCensus() const {
    if (!censusOn()) return nullptr;
    auto stripe = HostThreadSlot<std::uint32_t, CensusStripeTag>::Get();
    if (stripe == 0) {
        stripe = static_cast<std::uint32_t>(nextCensusStripe.fetch_add(1, std::memory_order_relaxed) % CensusStripes + 1);
        HostThreadSlot<std::uint32_t, CensusStripeTag>::Set(stripe);
    }
    return &reads[stripe - 1];
}

void StampedRanges::noteTested(ReadCensus* census, std::uint64_t tested) {
    if (census == nullptr) return;
    const std::size_t bucket = tested == 0 ? 0 : tested == 1 ? 1 : tested < 4 ? 2 : tested < 8 ? 3 : tested < 16 ? 4 : tested < 64 ? 5 : 6;
    census->tested[bucket].fetch_add(1, std::memory_order_relaxed);
    census->testedTotal.fetch_add(tested, std::memory_order_relaxed);
}

StampedRanges::ClassBounds StampedRanges::classBounds() const {
    ClassBounds bounds;
    for (std::size_t at = 0; at < Classes; ++at) bounds.newest[at] = classes.newest[at].load(std::memory_order_relaxed);
    for (std::size_t at = 0; at < Levels; ++at) bounds.deep[at] = at == 0 ? bounds.newest[0] : std::max(bounds.deep[at - 1], bounds.newest[at]);
    return bounds;
}

template<typename TVisit>
StampedRanges::Gates StampedRanges::probeMaps(const Range& range, std::uint64_t lo, const ClassBounds& bounds, TVisit&& visit) const {
    Gates gates;
    const auto last = lastByte(range);
    std::size_t level = Levels;
    for (std::size_t at = 0; at < Levels; ++at) {
        if ((last >> Shifts[at]) - (range.first >> Shifts[at]) < QueryBlocks) {
            level = at;
            break;
        }
    }
    if (level == Levels) {
        gates.every = true;
        return gates;
    }
    bool coarse = true;
    bool fine = true;
    const auto regionFirst = range.first >> NearShift;
    const auto regionLast = last >> NearShift;
    if (regionLast - regionFirst < NearRegions) {
        coarse = false;
        fine = false;
        for (auto region = regionFirst; region <= regionLast; ++region) {
            const auto& words = nears[hashed(region, NearBits)];
            if (words.coarse.load(std::memory_order_relaxed) >= lo) coarse = true;
            if (words.fine.load(std::memory_order_relaxed) >= lo) fine = true;
        }
    }
    if (fine && (last >> PageShift) - (range.first >> PageShift) < PageQueryPages) fine = touchedSince(pages.get(), PageShift, PageBits, PageQueryPages, range.first, last, lo);
    gates.coarseSkipped = !coarse;
    gates.fineSkipped = !fine;
    const auto visitGrain = [&](std::size_t at, bool deep) {
        const auto shift = Shifts[at];
        const auto end = last >> shift;
        for (auto block = range.first >> shift; block <= end; ++block) visit(deep ? deepMap(at, block) : ownMap(at, block));
    };
    if (level < FineLevels) {
        if (fine && bounds.deep[level] >= lo) visitGrain(level, true);
        if (level == 0 && fine && bounds.newest[1] >= lo) visitGrain(1, false);
        if (coarse) {
            for (auto at = FineLevels; at < Levels; ++at) {
                if (bounds.newest[at] >= lo) visitGrain(at, false);
            }
        }
        return gates;
    }
    if ((fine || coarse) && bounds.deep[level] >= lo) visitGrain(level, true);
    if (coarse) {
        for (auto at = level + 1; at < Levels; ++at) {
            if (bounds.newest[at] >= lo) visitGrain(at, false);
        }
    }
    return gates;
}

void StampedRanges::gather(std::size_t map, const Window& window, Candidates& candidates) const {
    const auto& meta = metas[map];
    if (meta.newest.load(std::memory_order_relaxed) < window.lo) return;
    auto words = static_cast<std::uint32_t>(meta.summary.load(std::memory_order_relaxed)) & window.words;
    const auto& held = maps[map];
    while (words != 0) {
        const auto word = static_cast<std::size_t>(std::countr_zero(words));
        words &= words - 1;
        const auto bits = held.words[word].load(std::memory_order_relaxed) & windowBits(window, word);
        if (bits == 0) continue;
        candidates.bits[word] |= bits;
        candidates.words |= std::uint32_t{1} << word;
    }
}

bool StampedRanges::collect(std::span<const Range> ranges, const Window& window, Candidates& candidates, ReadCensus* census) const {
    const auto bounds = classBounds();
    for (const auto& range : ranges) {
        if (range.second < range.first) return false;
        const auto gates = probeMaps(range, window.lo, bounds, [&](std::size_t map) { gather(map, window, candidates); });
        if (gates.every) {
            candidates.every = true;
            continue;
        }
        if (census == nullptr) continue;
        if (gates.fineSkipped) census->fineSkips.fetch_add(1, std::memory_order_relaxed);
        if (gates.coarseSkipped) census->coarseSkips.fetch_add(1, std::memory_order_relaxed);
    }
    if (bounds.newest[HugeLevel] >= window.lo) gather(Layout.huge, window, candidates);
    if (candidates.every) {
        for (std::size_t word = 0; word < Words; ++word) {
            const auto bits = windowBits(window, word);
            if (bits == 0) continue;
            candidates.bits[word] = bits;
            candidates.words |= std::uint32_t{1} << word;
        }
    }
    return true;
}

bool StampedRanges::slotsMayHold(std::span<const Range> ranges, std::uint64_t lo) const {
    const auto bounds = classBounds();
    for (const auto& range : ranges) {
        bool held = false;
        const auto gates = probeMaps(range, lo, bounds, [&](std::size_t map) { held = held || metas[map].newest.load(std::memory_order_relaxed) >= lo; });
        if (gates.every) {
            for (const auto newest : bounds.newest) {
                if (newest >= lo) return true;
            }
            continue;
        }
        if (held) return true;
    }
    return bounds.newest[HugeLevel] >= lo && metas[Layout.huge].newest.load(std::memory_order_relaxed) >= lo;
}

bool StampedRanges::oldWindowClear(std::span<const Range> ranges, std::uint64_t from) const {
    if (runFront.recycled.load(std::memory_order_relaxed) >= from) {
        if (runFront.wideBound.load(std::memory_order_relaxed) >= from) return false;
        for (const auto& range : ranges) {
            std::uint64_t first = 0;
            std::uint64_t last = 0;
            if (!runRegions(range, RunQueryRegions, first, last)) return false;
            for (auto region = first; region <= last; ++region) {
                if (nears[hashed(region, NearBits)].runBound.load(std::memory_order_relaxed) >= from) return false;
            }
        }
    }
    return !slotsMayHold(ranges, from);
}

StampedRanges::Seen StampedRanges::probe(std::size_t index, std::span<const Range> asked, const Window& window, std::uint64_t& stamp, Range& range) const {
    const auto& slot = slots[index];
    for (unsigned attempt = 0; attempt < ReadAttempts; ++attempt) {
        const auto version = slot.version.load(std::memory_order_acquire);
        if ((version & 1u) != 0) {
            SpinPause();
            continue;
        }
        const auto held = slot.stamp.load(std::memory_order_relaxed);
        const auto count = std::min<std::size_t>(slot.count.load(std::memory_order_relaxed), MaxRanges);
        const bool inside = held >= window.lo && held < window.hi;
        bool found = false;
        bool broken = false;
        Range seen{};
        if (inside) {
            const auto inlined = std::min(count, Inline);
            for (std::size_t at = 0; at < inlined && !found; ++at) {
                const Range cell{slot.cells[at].begin.load(std::memory_order_relaxed), slot.cells[at].end.load(std::memory_order_relaxed)};
                if (!overlapsAsked(cell, asked)) continue;
                found = true;
                seen = cell;
            }
            auto block = slot.head.load(std::memory_order_relaxed);
            std::size_t left = count > Inline ? count - Inline : 0;
            while (left != 0 && !found) {
                if (block >= PoolBlocks) {
                    broken = true;
                    break;
                }
                const auto& cells = blocks[block].cells;
                const auto take = std::min(left, BlockRanges);
                for (std::size_t at = 0; at < take && !found; ++at) {
                    const Range cell{cells[at].begin.load(std::memory_order_relaxed), cells[at].end.load(std::memory_order_relaxed)};
                    if (!overlapsAsked(cell, asked)) continue;
                    found = true;
                    seen = cell;
                }
                left -= take;
                block = blocks[block].next.load(std::memory_order_relaxed);
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (slot.version.load(std::memory_order_relaxed) != version || broken) continue;
        if (!inside || !found) return Seen::Miss;
        stamp = held;
        range = seen;
        return Seen::Hit;
    }
    return Seen::Torn;
}

StampedRanges::Seen StampedRanges::runLookup(std::span<const Range> ranges, std::uint64_t lo, std::uint64_t hi, bool any, RunFound& found, ReadCensus* census) const {
    auto candidates = front.wideRuns.load(std::memory_order_relaxed);
    for (const auto& range : ranges) {
        std::uint64_t first = 0;
        std::uint64_t last = 0;
        if (!runRegions(range, RunQueryRegions, first, last)) {
            candidates |= runFront.held.load(std::memory_order_relaxed);
            continue;
        }
        for (auto region = first; region <= last; ++region) candidates |= nears[hashed(region, NearBits)].runs.load(std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (candidates == 0) {
        if (census != nullptr) census->runFiltered.fetch_add(1, std::memory_order_relaxed);
        return Seen::Miss;
    }
    std::uint64_t tested = 0;
    const auto done = [&](Seen seen) {
        if (census != nullptr) census->runTested.fetch_add(tested, std::memory_order_relaxed);
        return seen;
    };
    while (candidates != 0) {
        const auto entry = static_cast<std::size_t>(std::countr_zero(candidates));
        candidates &= candidates - 1;
        ++tested;
        const auto& run = runs[entry];
        bool read = false;
        for (unsigned attempt = 0; attempt < ReadAttempts && !read; ++attempt) {
            const auto version = run.version.load(std::memory_order_acquire);
            if ((version & 1u) != 0) {
                SpinPause();
                continue;
            }
            const auto first = run.first.load(std::memory_order_relaxed);
            const auto last = run.last.load(std::memory_order_relaxed);
            const auto count = std::min<std::size_t>(run.count.load(std::memory_order_relaxed), RunRanges);
            const bool meets = last >= lo && first < hi;
            bool overlaps = false;
            Range seen{};
            for (std::size_t at = 0; meets && at < count && !overlaps; ++at) {
                const Range cell{run.cells[at].begin.load(std::memory_order_relaxed), run.cells[at].end.load(std::memory_order_relaxed)};
                if (!overlapsAsked(cell, ranges)) continue;
                overlaps = true;
                seen = cell;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (run.version.load(std::memory_order_relaxed) != version) continue;
            read = true;
            if (!meets || !overlaps) break;
            if (any) {
                found = RunFound{true, false, last, seen};
                return done(Seen::Hit);
            }
            const auto stamp = std::min(last, hi - 1);
            const bool conservative = stamp != last && stamp != first;
            if (!found.hit || stamp > found.stamp || (stamp == found.stamp && found.conservative && !conservative)) found = RunFound{true, conservative, stamp, seen};
        }
        if (!read) return done(Seen::Torn);
    }
    return done(found.hit ? Seen::Hit : Seen::Miss);
}

DeepFound StampedRanges::Before(std::span<const Range> ranges, std::uint64_t live, std::uint64_t before) const {
    auto* const census = readCensus();
    const std::size_t kind = before == Unbounded ? 1u : 0u;
    if (census != nullptr) census->lookups[kind].fetch_add(1, std::memory_order_relaxed);
    const auto unsure = [census](Doubt doubt) {
        if (census != nullptr) census->unsure[static_cast<std::size_t>(doubt)].fetch_add(1, std::memory_order_relaxed);
        return DeepFound{DeepAnswer::Unsure};
    };
    if (ranges.empty() || live >= before) {
        noteTested(census, 0);
        return {};
    }
    const auto newest = front.newest.load(std::memory_order_acquire);
    const auto lost = front.lost.load(std::memory_order_acquire);
    std::uint64_t oldest = 0;
    if (newest + 1 >= Capacity) oldest = newest + 1 - Capacity;
    const auto lo = std::max(live, oldest);
    if (lo >= before) {
        noteTested(census, 0);
        return {};
    }
    if (lost > lo) return unsure(Doubt::Unrecorded);
    const auto hi = std::min(before, newest + 1);
    if (lo >= hi) {
        noteTested(census, 0);
        return {};
    }
    if (invertedAny(ranges)) return unsure(Doubt::Inverted);
    RunFound run;
    if (runLookup(ranges, lo, hi, false, run, census) == Seen::Torn) return unsure(Doubt::Torn);
    const auto window = makeWindow(run.hit ? std::max(lo, run.stamp) : lo, hi);
    Candidates candidates;
    if (!collect(ranges, window, candidates, census)) return unsure(Doubt::Inverted);
    if (census != nullptr && candidates.every) census->everySlot.fetch_add(1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    DeepFound found;
    bool torn = false;
    std::uint64_t tested = 0;
    walkNewestFirst(window.hi, candidates, [&](std::size_t slot) {
        std::uint64_t stamp = 0;
        Range range{};
        const auto seen = probe(slot, ranges, window, stamp, range);
        if (seen == Seen::Torn) {
            torn = true;
            return true;
        }
        if (seen != Seen::Hit) return false;
        found = DeepFound{DeepAnswer::Hit, stamp, false, range};
        return true;
    }, tested);
    noteTested(census, tested);
    if (torn) return unsure(Doubt::Torn);
    if (found.answer == DeepAnswer::Hit || !run.hit) return found;
    if (census != nullptr) {
        census->runAnswers[kind].fetch_add(1, std::memory_order_relaxed);
        if (run.conservative) census->runConservative.fetch_add(1, std::memory_order_relaxed);
    }
    return DeepFound{DeepAnswer::Hit, run.stamp, run.conservative, run.range};
}

DeepFound StampedRanges::Newest(std::span<const Range> ranges, std::uint64_t live) const {
    return Before(ranges, live, Unbounded);
}

bool StampedRanges::Within(std::span<const Range> ranges, std::uint64_t from, std::uint64_t to) const {
    auto* const census = readCensus();
    if (census != nullptr) census->lookups[2].fetch_add(1, std::memory_order_relaxed);
    const auto kept = [census](Kept reason) {
        if (census != nullptr) census->kept[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
        return true;
    };
    const auto clear = [census] {
        if (census != nullptr) census->oldClear.fetch_add(1, std::memory_order_relaxed);
        noteTested(census, 0);
        return false;
    };
    if (from >= to || ranges.empty()) {
        noteTested(census, 0);
        return false;
    }
    const auto newest = front.newest.load(std::memory_order_acquire);
    const auto lost = front.lost.load(std::memory_order_acquire);
    if (lost > from) return kept(Kept::Unrecorded);
    const bool old = newest >= from && newest - from >= Capacity;
    const bool exactOld = old && runsSwitched();
    if (old && !exactOld) return kept(Kept::Recycled);
    const auto hi = std::min(to, newest + 1);
    if (from >= hi) {
        noteTested(census, 0);
        return false;
    }
    if (invertedAny(ranges)) return kept(Kept::Inverted);
    RunFound run;
    const auto byRun = runLookup(ranges, from, hi, true, run, census);
    if (byRun == Seen::Torn) return kept(Kept::Torn);
    if (byRun == Seen::Hit) {
        if (census != nullptr) census->runWithin.fetch_add(1, std::memory_order_relaxed);
        noteTested(census, 0);
        return true;
    }
    if (exactOld) {
        if (!oldWindowClear(ranges, from)) return kept(Kept::Recycled);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (front.lost.load(std::memory_order_relaxed) > from) return kept(Kept::Unrecorded);
        return clear();
    }
    const auto window = makeWindow(from, hi);
    Candidates candidates;
    if (!collect(ranges, window, candidates, census)) return kept(Kept::Inverted);
    if (census != nullptr && candidates.every) census->everySlot.fetch_add(1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    bool hit = false;
    bool torn = false;
    std::uint64_t tested = 0;
    walkNewestFirst(window.hi, candidates, [&](std::size_t slot) {
        std::uint64_t stamp = 0;
        Range range{};
        const auto seen = probe(slot, ranges, window, stamp, range);
        if (seen == Seen::Miss) return false;
        hit = seen == Seen::Hit;
        torn = seen == Seen::Torn;
        return true;
    }, tested);
    noteTested(census, tested);
    if (hit) return true;
    if (torn) return kept(Kept::Torn);
    std::atomic_thread_fence(std::memory_order_acquire);
    const auto after = front.newest.load(std::memory_order_relaxed);
    const bool aged = after >= from && after - from >= Capacity;
    if (aged && !(runsSwitched() && oldWindowClear(ranges, from))) return kept(Kept::Recycled);
    if (front.lost.load(std::memory_order_relaxed) > from) return kept(Kept::Unrecorded);
    if (aged && census != nullptr) census->oldClear.fetch_add(1, std::memory_order_relaxed);
    return false;
}

void StampedRanges::report() const {
    std::array<std::uint64_t, 3> lookups{};
    std::array<std::uint64_t, TestedBuckets> tested{};
    std::array<std::uint64_t, static_cast<std::size_t>(Doubt::Count)> unsure{};
    std::array<std::uint64_t, static_cast<std::size_t>(Kept::Count)> kept{};
    std::array<std::uint64_t, 2> runAnswers{};
    std::uint64_t testedTotal = 0;
    std::uint64_t everySlot = 0;
    std::uint64_t runConservative = 0;
    std::uint64_t runWithin = 0;
    std::uint64_t runTested = 0;
    std::uint64_t runFiltered = 0;
    std::uint64_t fineSkips = 0;
    std::uint64_t coarseSkips = 0;
    std::uint64_t oldClear = 0;
    for (const auto& stripe : reads) {
        for (std::size_t at = 0; at < lookups.size(); ++at) lookups[at] += stripe.lookups[at].load(std::memory_order_relaxed);
        for (std::size_t at = 0; at < tested.size(); ++at) tested[at] += stripe.tested[at].load(std::memory_order_relaxed);
        for (std::size_t at = 0; at < unsure.size(); ++at) unsure[at] += stripe.unsure[at].load(std::memory_order_relaxed);
        for (std::size_t at = 0; at < kept.size(); ++at) kept[at] += stripe.kept[at].load(std::memory_order_relaxed);
        for (std::size_t at = 0; at < runAnswers.size(); ++at) runAnswers[at] += stripe.runAnswers[at].load(std::memory_order_relaxed);
        testedTotal += stripe.testedTotal.load(std::memory_order_relaxed);
        everySlot += stripe.everySlot.load(std::memory_order_relaxed);
        runConservative += stripe.runConservative.load(std::memory_order_relaxed);
        runWithin += stripe.runWithin.load(std::memory_order_relaxed);
        runTested += stripe.runTested.load(std::memory_order_relaxed);
        runFiltered += stripe.runFiltered.load(std::memory_order_relaxed);
        fineSkips += stripe.fineSkips.load(std::memory_order_relaxed);
        coarseSkips += stripe.coarseSkips.load(std::memory_order_relaxed);
        oldClear += stripe.oldClear.load(std::memory_order_relaxed);
    }
    std::uint64_t answered = 0;
    for (const auto count : tested) answered += count;
    const auto mean = answered != 0 ? static_cast<double>(testedTotal) / static_cast<double>(answered) : 0.0;
    const auto& ranges = written.ranges;
    const auto& lost = written.unrecorded;
    char runText[1024];
    std::snprintf(runText, sizeof(runText), "; runs (APS5_DEEP_RUNS): numbers recorded only in a run %llu, opened %llu, extended %llu, closed when live passed %llu, closed over a gap %llu, reclaimed %llu, in use %llu, highest %llu of %llu, lists that found no free run %llu, lists too long for a run %llu, writes added to a number of a run %llu; lookups answered by a run: before %llu, newest %llu, conservative %llu, within %llu; runs tested %llu; lookups with no run near %llu; probes skipped: fine by page or region %llu, coarse by region %llu; old watches proven not written %llu",
        wideCount(written.runStamps), wideCount(written.runsOpened), wideCount(written.runsExtended),
        wideCount(written.runsClosed[0]), wideCount(written.runsClosed[1]), wideCount(written.runsReclaimed),
        wideCount(static_cast<std::uint64_t>(std::popcount(runsUsed))), wideCount(written.runsHighest), wideCount(RunEntries),
        wideCount(written.runsNoEntry), wideCount(written.runsLongLists), wideCount(written.runFreshSlots),
        wideCount(runAnswers[0]), wideCount(runAnswers[1]), wideCount(runConservative), wideCount(runWithin),
        wideCount(runTested), wideCount(runFiltered),
        wideCount(fineSkips), wideCount(coarseSkips),
        wideCount(oldClear));
    std::fprintf(stderr, "[deep-index] %s (cumulative): %llu stamps recorded from %llu inserts, %llu appends; ranges by grain: 64 B %llu, 1 KiB %llu, 16 KiB %llu, 256 KiB %llu, 4 MiB %llu, 64 MiB %llu, wider %llu; not recorded: beyond capacity %llu, out of order %llu, spill pool full %llu; slots of skipped stamps evicted %llu; spill blocks in use %llu, highest %llu of %llu; bits set %llu, cleared %llu; lookups: before %llu, newest %llu, within %llu; slots tested per lookup: none %llu, 1 %llu, 2-3 %llu, 4-7 %llu, 8-15 %llu, 16-63 %llu, 64 or more %llu, mean %.2f; lookups over every slot %llu; unsure: unrecorded stamp %llu, torn read %llu, inverted range %llu; within true untested: recycled window %llu, unrecorded stamp %llu, torn read %llu, inverted range %llu%s\n",
        label, wideCount(written.stamps), wideCount(written.inserts), wideCount(written.appends),
        wideCount(ranges[0]), wideCount(ranges[1]), wideCount(ranges[2]), wideCount(ranges[3]), wideCount(ranges[4]), wideCount(ranges[5]), wideCount(ranges[6]),
        wideCount(lost[0]), wideCount(lost[1]), wideCount(lost[2]),
        wideCount(written.evicted), wideCount(PoolBlocks - freeCount), wideCount(written.spillHighest), wideCount(PoolBlocks),
        wideCount(written.bitsSet), wideCount(written.bitsCleared),
        wideCount(lookups[0]), wideCount(lookups[1]), wideCount(lookups[2]),
        wideCount(tested[0]), wideCount(tested[1]), wideCount(tested[2]), wideCount(tested[3]), wideCount(tested[4]), wideCount(tested[5]), wideCount(tested[6]), mean,
        wideCount(everySlot),
        wideCount(unsure[0]), wideCount(unsure[1]), wideCount(unsure[2]),
        wideCount(kept[0]), wideCount(kept[1]), wideCount(kept[2]), wideCount(kept[3]),
        runText);
}

StampedKeys::Bucket& StampedKeys::bucketOf(std::uint64_t key) {
    return buckets[hashed(key, BucketBits)];
}

const StampedKeys::Bucket& StampedKeys::bucketOf(std::uint64_t key) const {
    return buckets[hashed(key, BucketBits)];
}

void StampedKeys::touch(Stamps& stamps, std::uint64_t stamp, bool inserts, std::uint64_t live) {
    if (stamps.last.load(std::memory_order_relaxed) < live) stamps.first.store(stamp, std::memory_order_release);
    stamps.last.store(stamp, std::memory_order_release);
    if (!inserts) return;
    if (stamps.lastInserts.load(std::memory_order_relaxed) < live) stamps.firstInserts.store(stamp, std::memory_order_release);
    stamps.lastInserts.store(stamp, std::memory_order_release);
}

void StampedKeys::rewrite(Entry& entry, std::uint64_t key, std::uint64_t stamp, bool inserts) {
    const auto version = entry.version.load(std::memory_order_relaxed);
    entry.version.store(version + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    entry.key.store(key, std::memory_order_relaxed);
    entry.stamps.first.store(stamp, std::memory_order_relaxed);
    entry.stamps.last.store(stamp, std::memory_order_relaxed);
    entry.stamps.firstInserts.store(inserts ? stamp : 0, std::memory_order_relaxed);
    entry.stamps.lastInserts.store(inserts ? stamp : 0, std::memory_order_relaxed);
    entry.version.store(version + 2, std::memory_order_release);
}

void StampedKeys::Insert(std::uint64_t key, std::uint64_t stamp, bool inserts, std::uint64_t live) {
    if (recent != nullptr && recentKey == key && recent->key.load(std::memory_order_relaxed) == key && recent->stamps.last.load(std::memory_order_relaxed) >= live) {
        touch(recent->stamps, stamp, inserts, live);
        return;
    }
    auto& bucket = bucketOf(key);
    Entry* free = nullptr;
    for (auto& way : bucket.ways) {
        const auto held = way.key.load(std::memory_order_relaxed);
        if (way.stamps.last.load(std::memory_order_relaxed) >= live) {
            if (held != key) continue;
            touch(way.stamps, stamp, inserts, live);
            recent = &way;
            recentKey = key;
            return;
        }
        if (free == nullptr || (held == key && free->key.load(std::memory_order_relaxed) != key)) free = &way;
    }
    if (free != nullptr) {
        rewrite(*free, key, stamp, inserts);
        recent = free;
        recentKey = key;
        return;
    }
    touch(bucket.overflow, stamp, inserts, live);
    overflows.store(overflows.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

void StampedKeys::InsertAny(std::uint64_t stamp, bool inserts, std::uint64_t live) {
    touch(any, stamp, inserts, live);
}

void StampedKeys::consider(std::uint64_t first, std::uint64_t last, std::uint64_t live, std::uint64_t before, DeepFound& found) {
    if (last < live) return;
    std::uint64_t candidate = last;
    bool conservative = false;
    if (last >= before) {
        if (first >= before) return;
        candidate = before - 1;
        conservative = true;
    }
    if (candidate < live) return;
    if (found.answer != DeepAnswer::Hit || candidate > found.stamp || (candidate == found.stamp && found.conservative && !conservative)) found = {DeepAnswer::Hit, candidate, conservative, {}};
}

DeepFound StampedKeys::Before(std::uint64_t key, bool insertsOnly, std::uint64_t live, std::uint64_t before) const {
    DeepFound found;
    const auto& bucket = bucketOf(key);
    for (const auto& way : bucket.ways) {
        if (way.stamps.last.load(std::memory_order_acquire) < live) continue;
        std::uint64_t held = 0;
        std::uint64_t first = 0;
        std::uint64_t last = 0;
        std::uint64_t firstInserts = 0;
        std::uint64_t lastInserts = 0;
        bool read = false;
        for (unsigned attempt = 0; attempt < ReadAttempts && !read; ++attempt) {
            const auto version = way.version.load(std::memory_order_acquire);
            if ((version & 1u) != 0) {
                SpinPause();
                continue;
            }
            held = way.key.load(std::memory_order_relaxed);
            last = way.stamps.last.load(std::memory_order_acquire);
            first = way.stamps.first.load(std::memory_order_relaxed);
            lastInserts = way.stamps.lastInserts.load(std::memory_order_acquire);
            firstInserts = way.stamps.firstInserts.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            read = way.version.load(std::memory_order_relaxed) == version;
        }
        if (!read) return {DeepAnswer::Unsure};
        if (held != key) continue;
        if (insertsOnly) consider(firstInserts, lastInserts, live, before, found);
        else consider(first, last, live, before, found);
    }
    const auto& overflow = bucket.overflow;
    const auto overflowLast = (insertsOnly ? overflow.lastInserts : overflow.last).load(std::memory_order_acquire);
    if (overflowLast >= live) {
        const auto overflowFirst = (insertsOnly ? overflow.firstInserts : overflow.first).load(std::memory_order_acquire);
        if (overflowFirst > overflowLast || overflowFirst < before) return {DeepAnswer::Unsure};
    }
    return found;
}

DeepFound StampedKeys::BeforeAny(bool insertsOnly, std::uint64_t live, std::uint64_t before) const {
    DeepFound found;
    const auto last = (insertsOnly ? any.lastInserts : any.last).load(std::memory_order_acquire);
    const auto first = (insertsOnly ? any.firstInserts : any.first).load(std::memory_order_acquire);
    consider(first, last, live, before, found);
    return found;
}

}
