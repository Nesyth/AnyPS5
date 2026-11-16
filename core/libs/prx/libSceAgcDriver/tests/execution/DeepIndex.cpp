#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DeepIndex.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DeepAnswer;
using AgcDriver::DriverDetail::DeepFound;
using AgcDriver::DriverDetail::StampedRanges;
using AgcDriver::Graphics::RuntimeSwitch;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;
using Range = StampedRanges::Range;

constexpr std::uint64_t Capacity = StampedRanges::Capacity;
constexpr std::uint64_t Unbounded = ~std::uint64_t{0};
constexpr std::uint64_t MiB = std::uint64_t{1} << 20;
constexpr std::uint64_t CommandBase = 0x100000000;
constexpr std::uint64_t CommandSpan = 0x400000;
constexpr std::uint64_t TargetBase = 0x200000000;
constexpr std::uint64_t CrowdBase = 0x600000000;
constexpr std::uint64_t HugeBase = 0x800000000;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void SetRuns(bool on) {
    RuntimeSwitches::Assign(RuntimeSwitch::DeepRuns, on);
}

class Random {
public:
    explicit Random(std::uint64_t seed) : state(seed != 0 ? seed : 1) {}

    std::uint64_t Next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    std::uint64_t Below(std::uint64_t bound) {
        return bound == 0 ? 0 : Next() % bound;
    }

    bool Chance(std::uint64_t percent) {
        return Below(100) < percent;
    }

private:
    std::uint64_t state;
};

std::uint64_t Mix(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

bool Overlaps(const Range& write, const Range& read) {
    return read.first < write.second && write.first < read.second;
}

bool OverlapsAny(const Range& write, std::span<const Range> asked) {
    return std::any_of(asked.begin(), asked.end(), [&write](const Range& read) { return Overlaps(write, read); });
}

std::optional<Range> FirstOverlap(std::span<const Range> writes, std::span<const Range> asked) {
    for (const auto& write : writes) {
        if (OverlapsAny(write, asked)) return write;
    }
    return std::nullopt;
}

std::string RangeText(const Range& range) {
    return "[" + std::to_string(range.first) + ", " + std::to_string(range.second) + ")";
}

std::string RangesText(std::span<const Range> ranges) {
    std::string text;
    for (const auto& range : ranges) text += (text.empty() ? "" : " ") + RangeText(range);
    return text;
}

std::string FoundText(const DeepFound& found) {
    if (found.answer == DeepAnswer::Unsure) return "unsure";
    if (found.answer == DeepAnswer::Clear) return "clear";
    return std::string(found.conservative ? "conservative hit " : "hit ") + std::to_string(found.stamp) + " " + RangeText(found.range);
}

bool SameFound(const DeepFound& left, const DeepFound& right) {
    if (left.answer != right.answer) return false;
    if (left.answer != DeepAnswer::Hit) return true;
    return left.stamp == right.stamp && left.range == right.range && !left.conservative && !right.conservative;
}

class Model {
public:
    enum class Reason : std::size_t { Beyond, Disorder, Pool, Count };
    enum class Watch : std::uint8_t { False, True, Either };

    struct Expected {
        DeepAnswer answer = DeepAnswer::Clear;
        std::uint64_t stamp = 0;
        bool conservative = false;
        std::vector<Range> ranges;
    };

    void Insert(std::span<const Range> ranges, std::uint64_t stamp, std::uint64_t live, bool runsOn) {
        if (ranges.empty()) return;
        if (started && stamp < current) {
            unrecord(stamp, Reason::Disorder);
            return;
        }
        if (started && stamp == current) {
            if (!currentRecorded) return;
            auto& held = recorded[stamp].slot;
            const auto more = blocksFor(held.size() + ranges.size()) - blocksFor(held.size());
            if (ranges.size() > StampedRanges::MaxRanges - held.size() || blocksHeld(newest) + more > StampedRanges::PoolBlocks) {
                unrecord(stamp, Reason::Pool);
                return;
            }
            if (recorded[stamp].run != NoRun && held.empty()) ++freshSlots;
            held.insert(held.end(), ranges.begin(), ranges.end());
            return;
        }
        started = true;
        current = stamp;
        currentRecorded = false;
        if (stamp >= live && stamp - live >= Capacity) {
            unrecord(stamp, Reason::Beyond);
            return;
        }
        const std::vector<Range> list(ranges.begin(), ranges.end());
        const bool listed = runsOn && list.size() <= StampedRanges::RunRanges;
        if (listed && recordInRun(list, stamp, live)) return;
        if (list.size() > StampedRanges::MaxRanges || blocksHeld(stamp) + blocksFor(list.size()) > StampedRanges::PoolBlocks) {
            unrecord(stamp, Reason::Pool);
            return;
        }
        recorded[stamp] = Record{list, NoRun};
        noteRecorded(stamp);
        if (listed) noteRecent(list, stamp);
    }

    Expected Before(std::span<const Range> asked, std::uint64_t live, std::uint64_t before) const {
        Expected expected;
        if (asked.empty() || live >= before) return expected;
        const std::uint64_t oldest = newest + 1 >= Capacity ? newest + 1 - Capacity : 0;
        const auto lo = std::max(live, oldest);
        if (lo >= before) return expected;
        if (lost > lo) {
            expected.answer = DeepAnswer::Unsure;
            return expected;
        }
        const auto hi = std::min(before, newest + 1);
        if (lo >= hi) return expected;
        if (Inverted(asked)) {
            expected.answer = DeepAnswer::Unsure;
            return expected;
        }
        bool runHit = false;
        bool runExact = false;
        std::uint64_t best = 0;
        std::vector<Range> exactRanges;
        std::vector<Range> conservativeRanges;
        for (const auto index : active) {
            const auto& run = runs[index];
            if (run.last < lo || run.first >= hi) continue;
            const auto overlap = FirstOverlap(run.list, asked);
            if (!overlap) continue;
            const auto stamp = std::min(run.last, hi - 1);
            if (!runHit || stamp > best) {
                runHit = true;
                runExact = false;
                best = stamp;
                exactRanges.clear();
                conservativeRanges.clear();
            }
            if (stamp != best) continue;
            if (stamp == run.last || stamp == run.first) {
                runExact = true;
                exactRanges.push_back(*overlap);
            } else {
                conservativeRanges.push_back(*overlap);
            }
        }
        const auto floor = runHit ? std::max(lo, best) : lo;
        for (auto at = recorded.lower_bound(hi); at != recorded.begin();) {
            --at;
            if (at->first < floor) break;
            const auto overlap = FirstOverlap(at->second.slot, asked);
            if (!overlap) continue;
            expected.answer = DeepAnswer::Hit;
            expected.stamp = at->first;
            expected.ranges = {*overlap};
            return expected;
        }
        if (!runHit) return expected;
        Require(!runExact || exactRanges.size() == 1, "the reference found " + std::to_string(exactRanges.size()) + " runs that carry number " + std::to_string(best));
        expected.answer = DeepAnswer::Hit;
        expected.stamp = best;
        expected.conservative = !runExact;
        expected.ranges = runExact ? exactRanges : conservativeRanges;
        if (expected.conservative) {
            const auto writer = TrueNewest(asked, lo, hi);
            Require(!writer || *writer <= best, "the reference named number " + std::to_string(best) + " below the newest writer " + std::to_string(writer ? *writer : 0));
            Require(best < before && InsideRun(asked, best), "the reference named number " + std::to_string(best) + " outside every run of an overlapping list");
        }
        return expected;
    }

    Watch Within(std::span<const Range> asked, std::uint64_t from, std::uint64_t to, bool runsOn) const {
        if (from >= to || asked.empty()) return Watch::False;
        if (lost > from) return Watch::True;
        const bool old = newest >= from && newest - from >= Capacity;
        if (old && !runsOn) return Watch::True;
        const auto hi = std::min(to, newest + 1);
        if (from >= hi) return Watch::False;
        if (Inverted(asked)) return Watch::True;
        for (const auto& run : runs) {
            if (run.last >= from && run.first < hi && FirstOverlap(run.list, asked)) return Watch::True;
        }
        for (auto at = recorded.lower_bound(from); at != recorded.end() && at->first < hi; ++at) {
            if (FirstOverlap(at->second.slot, asked)) return Watch::True;
        }
        return old ? Watch::Either : Watch::False;
    }

    std::uint64_t Overflows() const { return overflows; }
    std::uint64_t Unrecorded(Reason reason) const { return reasons[static_cast<std::size_t>(reason)]; }
    std::uint64_t Newest() const { return newest; }
    std::uint64_t RunsOpened() const { return runs.size(); }
    std::uint64_t RunsExtended() const { return extended; }
    std::uint64_t ClosedByLive() const { return closedLive; }
    std::uint64_t ClosedByGap() const { return closedGap; }
    std::uint64_t NoFreeRun() const { return noEntry; }
    std::uint64_t FreshSlots() const { return freshSlots; }

private:
    static constexpr std::size_t NoRun = ~std::size_t{0};

    struct Record {
        std::vector<Range> slot;
        std::size_t run = NoRun;
    };
    struct Run {
        std::vector<Range> list;
        std::uint64_t first = 0;
        std::uint64_t last = 0;
    };
    struct Recent {
        std::vector<Range> list;
        std::uint64_t stamp = 0;
    };

    static bool Inverted(std::span<const Range> asked) {
        return std::any_of(asked.begin(), asked.end(), [](const Range& range) { return range.second < range.first; });
    }

    static std::size_t blocksFor(std::size_t count) {
        return count > StampedRanges::Inline ? (count - StampedRanges::Inline + StampedRanges::BlockRanges - 1) / StampedRanges::BlockRanges : 0;
    }

    std::size_t blocksHeld(std::uint64_t anchor) const {
        std::size_t held = 0;
        const std::uint64_t from = anchor + 1 > Capacity ? anchor + 1 - Capacity : 0;
        for (auto at = recorded.lower_bound(from); at != recorded.end(); ++at) held += blocksFor(at->second.slot.size());
        return held;
    }

    std::optional<std::uint64_t> TrueNewest(std::span<const Range> asked, std::uint64_t lo, std::uint64_t hi) const {
        for (auto at = recorded.lower_bound(hi); at != recorded.begin();) {
            --at;
            if (at->first < lo) break;
            if (FirstOverlap(at->second.slot, asked)) return at->first;
            if (at->second.run != NoRun && FirstOverlap(runs[at->second.run].list, asked)) return at->first;
        }
        return std::nullopt;
    }

    bool InsideRun(std::span<const Range> asked, std::uint64_t stamp) const {
        return std::any_of(runs.begin(), runs.end(), [&](const Run& run) { return run.first <= stamp && stamp <= run.last && FirstOverlap(run.list, asked); });
    }

    void noteRecorded(std::uint64_t stamp) {
        newest = stamp;
        currentRecorded = true;
        active.erase(std::remove_if(active.begin(), active.end(), [&](std::size_t index) { return runs[index].last + Capacity <= stamp; }), active.end());
    }

    void noteRecent(const std::vector<Range>& list, std::uint64_t stamp) {
        const auto seen = std::find_if(recent.begin(), recent.end(), [&](const Recent& held) { return held.list == list; });
        if (seen != recent.end()) recent.erase(seen);
        recent.insert(recent.begin(), Recent{list, stamp});
        if (recent.size() > StampedRanges::RecentLists) recent.pop_back();
    }

    bool recordInRun(const std::vector<Range>& list, std::uint64_t stamp, std::uint64_t live) {
        if (const auto open = openRuns.find(list); open != openRuns.end()) {
            auto& run = runs[open->second];
            if (run.last >= live && stamp - run.last <= StampedRanges::RunGap) {
                run.last = stamp;
                recorded[stamp] = Record{{}, open->second};
                ++extended;
                noteRecorded(stamp);
                return true;
            }
            if (run.last + Capacity > stamp) ++(run.last < live ? closedLive : closedGap);
            openRuns.erase(open);
        }
        const auto seen = std::find_if(recent.begin(), recent.end(), [&](const Recent& held) { return held.list == list; });
        if (seen == recent.end() || seen->stamp < live || stamp - seen->stamp > StampedRanges::RunGap) return false;
        const auto inUse = static_cast<std::size_t>(std::count_if(active.begin(), active.end(), [&](std::size_t index) { return runs[index].last + Capacity > stamp; }));
        if (inUse >= StampedRanges::RunEntries) {
            ++noEntry;
            return false;
        }
        runs.push_back(Run{list, stamp, stamp});
        active.push_back(runs.size() - 1);
        openRuns[list] = runs.size() - 1;
        recorded[stamp] = Record{{}, runs.size() - 1};
        noteRecorded(stamp);
        return true;
    }

    void unrecord(std::uint64_t stamp, Reason reason) {
        lost = std::max(lost, stamp + 1);
        if (stamp == current) currentRecorded = false;
        ++overflows;
        ++reasons[static_cast<std::size_t>(reason)];
    }

    std::map<std::uint64_t, Record> recorded;
    std::vector<Run> runs;
    std::vector<std::size_t> active;
    std::map<std::vector<Range>, std::size_t> openRuns;
    std::vector<Recent> recent;
    std::uint64_t current = 0;
    bool started = false;
    bool currentRecorded = false;
    std::uint64_t newest = 0;
    std::uint64_t lost = 0;
    std::uint64_t overflows = 0;
    std::uint64_t extended = 0;
    std::uint64_t closedLive = 0;
    std::uint64_t closedGap = 0;
    std::uint64_t noEntry = 0;
    std::uint64_t freshSlots = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(Reason::Count)> reasons{};
};

class Stream {
public:
    explicit Stream(std::uint64_t seed) : random(seed) {
        const auto at = [](std::uint64_t offset) { return TargetBase + offset; };
        passes[0] = {{at(0), at(8 * MiB)}, {at(16 * MiB), at(32 * MiB)}, {at(64 * MiB), at(96 * MiB)}, {at(128 * MiB), at(132 * MiB)}};
        passes[1] = {{at(160 * MiB), at(224 * MiB)}};
        for (std::uint64_t index = 0; index < 8; ++index) passes[2].push_back({at(256 * MiB + index * 2 * MiB), at(256 * MiB + index * 2 * MiB + 2 * MiB)});
        passes[2].push_back({at(300 * MiB), at(300 * MiB + 0x4000)});
        for (std::uint64_t index = 0; index < StampedRanges::RunRanges; ++index) passes[3].push_back({at(320 * MiB + index * 0x10000), at(320 * MiB + index * 0x10000 + 0x8000)});
        for (std::uint64_t index = 0; index <= StampedRanges::RunRanges; ++index) passes[4].push_back({at(336 * MiB + index * 0x10000), at(336 * MiB + index * 0x10000 + 0x8000)});
        passes[5] = {{HugeBase, HugeBase + (std::uint64_t{3} << 30)}, {at(8 * MiB), at(9 * MiB)}};
        passes[6] = {{CommandBase + 0x1000, CommandBase + 0x1004}};
        passes[7] = {{at(40 * MiB), at(40 * MiB)}, {at(44 * MiB), at(46 * MiB)}, {at(48 * MiB + 0x40), at(48 * MiB + 0x80)}};
        passes[8] = {{at(52 * MiB + 0x100), at(52 * MiB)}, {at(56 * MiB), at(57 * MiB)}};
        passes[9] = {passes[0][3], passes[0][2], passes[0][1], passes[0][0]};
    }

    std::vector<Range> Write(bool wideLists, bool crowded, bool plain) {
        std::vector<Range> ranges;
        if (wideLists) {
            const auto count = 36 + random.Below(5);
            for (std::uint64_t index = 0; index < count; ++index) ranges.push_back(small());
            return ranges;
        }
        if (crowded) {
            const auto list = crowdNext++ / 3;
            const auto base = CrowdBase + (list % 4096) * 0x40000;
            ranges.push_back({base, base + 0x1000 + (list % 7) * 0x2000});
            if (list % 3 == 0) ranges.push_back({base + 0x20000, base + 0x20040});
            for (const auto& range : ranges) remember(range);
            return ranges;
        }
        if (random.Chance(55)) {
            if (random.Chance(3)) {
                other = pass;
                pass = random.Below(passes.size());
            }
            auto chosen = random.Chance(15) ? other : pass;
            if (plain && (chosen == 5 || chosen == 8)) chosen = 0;
            for (const auto& range : passes[chosen]) remember(range);
            return passes[chosen];
        }
        const auto roll = random.Below(100);
        if (roll < 45) {
            ranges.push_back(label());
        } else if (roll < 70) {
            ranges.push_back(small());
        } else if (roll < 78) {
            ranges.push_back(medium());
        } else if (roll < 84) {
            const auto base = TargetBase + random.Below(8) * 32 * MiB;
            ranges.push_back({base, base + (1 + random.Below(32)) * MiB});
            if (random.Chance(50)) ranges.push_back({base + 16 * MiB, base + 16 * MiB + (1 + random.Below(8)) * MiB});
        } else if (roll < 87) {
            const auto at = HugeBase + random.Below(std::uint64_t{1} << 36);
            if (plain) ranges.push_back(small());
            else ranges.push_back({at, at + 100 * MiB + random.Below(std::uint64_t{2} << 30)});
        } else if (roll < 90) {
            const auto at = CommandBase + random.Below(CommandSpan);
            ranges.push_back({at, at});
        } else if (roll < 94) {
            ranges.push_back(crossing());
        } else if (roll < 98) {
            const auto count = 3 + random.Below(18);
            for (std::uint64_t index = 0; index < count; ++index) ranges.push_back(random.Chance(50) ? label() : random.Chance(70) ? small() : medium());
        } else if (roll < 99) {
            const auto at = CommandBase + random.Below(CommandSpan);
            if (plain) ranges.push_back(label());
            else ranges.push_back({at + 16, at});
        } else {
            const auto count = 25 + random.Below(16);
            for (std::uint64_t index = 0; index < count; ++index) ranges.push_back(random.Chance(50) ? label() : small());
        }
        for (const auto& range : ranges) remember(range);
        return ranges;
    }

    std::vector<Range> Query() {
        std::vector<Range> ranges;
        const auto count = random.Chance(70) ? 1 : 2 + random.Below(2);
        for (std::uint64_t index = 0; index < count; ++index) ranges.push_back(asked());
        return ranges;
    }

    Random random;

private:
    Range label() {
        const auto at = CommandBase + cursor;
        cursor = (cursor + (random.Chance(50) ? 0x20 : 0xa8)) % CommandSpan;
        return {at, at + 4};
    }

    Range small() {
        const auto at = CommandBase + (cursor + CommandSpan - 0x200 + random.Below(0x400)) % CommandSpan;
        return {at, at + 16 + random.Below(65)};
    }

    Range medium() {
        auto at = CommandBase + random.Below(CommandSpan);
        if (random.Chance(50)) {
            at &= ~std::uint64_t{0xfff};
            return {at, at + 0x1000};
        }
        return {at, at + 64 + random.Below(4033)};
    }

    Range crossing() {
        static constexpr std::array<std::uint64_t, 6> grains{64, 1024, 16384, 262144, 4 * MiB, 64 * MiB};
        const auto grain = grains[random.Below(grains.size())];
        const auto edge = TargetBase + (1 + random.Below(64)) * grain;
        return {edge - 1 - random.Below(grain), edge + 1 + random.Below(grain)};
    }

    void remember(const Range& range) {
        recent[recentNext % recent.size()] = range;
        ++recentNext;
    }

    Range recentWrite() {
        if (recentNext == 0) return label();
        return recent[random.Below(std::min<std::uint64_t>(recentNext, recent.size()))];
    }

    Range asked() {
        const auto roll = random.Below(100);
        if (roll < 35) {
            const auto at = CommandBase + (cursor + CommandSpan - random.Below(0x800)) % CommandSpan;
            return {at, at + 16 + random.Below(65)};
        }
        if (roll < 50) {
            const auto write = recentWrite();
            if (random.Chance(50)) return {write.second, write.second + 1 + random.Below(16)};
            const auto start = write.first > 16 ? write.first - 1 - random.Below(16) : 0;
            return {start, write.first};
        }
        if (roll < 60) {
            const auto write = recentWrite();
            if (write.second <= write.first || random.Chance(30)) return write;
            const auto at = write.first + random.Below(write.second - write.first);
            return {at, at + 1 + random.Below(8)};
        }
        if (roll < 66) {
            const auto write = recentWrite();
            const auto choice = random.Below(3);
            const auto at = choice == 0 ? write.first : choice == 1 ? write.second : write.first + random.Below(write.second > write.first ? write.second - write.first : 1);
            return {at, at};
        }
        if (roll < 76) return medium();
        if (roll < 84) {
            const auto base = random.Chance(50) ? TargetBase + random.Below(256 * MiB) : CommandBase + random.Below(CommandSpan);
            return {base, base + 64 * 1024 + random.Below(64 * MiB)};
        }
        if (roll < 88) return random.Chance(50) ? Range{0, std::uint64_t{1} << 40} : Range{HugeBase, HugeBase + (std::uint64_t{1} << 37)};
        if (roll < 90) {
            const auto at = CommandBase + random.Below(CommandSpan);
            return {at + 8, at};
        }
        if (roll < 94) return crossing();
        if (roll < 98) {
            const auto at = 0x7000000000 + random.Below(64 * MiB);
            return {at, at + 1 + random.Below(256)};
        }
        const auto at = CommandBase + random.Below(CommandSpan);
        return {at, at + 1 + random.Below(256)};
    }

    std::array<std::vector<Range>, 10> passes;
    std::size_t pass = 0;
    std::size_t other = 1;
    std::uint64_t crowdNext = 0;
    std::uint64_t cursor = 0;
    std::array<Range, 256> recent{};
    std::uint64_t recentNext = 0;
};

struct Tally {
    std::uint64_t conservative = 0;
    std::uint64_t runAnswers = 0;
    std::uint64_t oldFalse = 0;
    std::uint64_t oldTrue = 0;
};

void CheckBefore(const StampedRanges& index, const Model& model, std::span<const Range> asked, std::uint64_t live, std::uint64_t before, const std::string& where, Tally& tally) {
    const auto got = before == Unbounded ? index.Newest(asked, live) : index.Before(asked, live, before);
    const auto expected = model.Before(asked, live, before);
    bool same = got.answer == expected.answer;
    if (same && got.answer == DeepAnswer::Hit) same = got.stamp == expected.stamp && got.conservative == expected.conservative && std::find(expected.ranges.begin(), expected.ranges.end(), got.range) != expected.ranges.end();
    std::string wanted = expected.answer == DeepAnswer::Unsure ? "unsure" : expected.answer == DeepAnswer::Clear ? "clear" : std::string(expected.conservative ? "conservative hit " : "hit ") + std::to_string(expected.stamp) + " over " + RangesText(expected.ranges);
    Require(same, where + ": " + (before == Unbounded ? std::string("newest") : "before " + std::to_string(before)) + " from live " + std::to_string(live) + " over " + RangesText(asked) + " answered " + FoundText(got) + ", the reference " + wanted + " (newest recorded " + std::to_string(model.Newest()) + ")");
    if (got.answer == DeepAnswer::Hit && got.conservative) ++tally.conservative;
}

void CheckWithin(const StampedRanges& index, const Model& model, std::span<const Range> asked, std::uint64_t from, std::uint64_t to, bool runsOn, const std::string& where, Tally& tally) {
    const auto got = index.Within(asked, from, to);
    const auto expected = model.Within(asked, from, to, runsOn);
    if (expected == Model::Watch::Either) {
        ++(got ? tally.oldTrue : tally.oldFalse);
        return;
    }
    const bool wanted = expected == Model::Watch::True;
    Require(got == wanted, where + ": within [" + std::to_string(from) + ", " + std::to_string(to) + ") over " + RangesText(asked) + " answered " + (got ? "true" : "false") + ", the reference " + (wanted ? "true" : "false") + " (newest recorded " + std::to_string(model.Newest()) + ", runs " + (runsOn ? "on" : "off") + ")");
}

void InsertBoth(StampedRanges& index, Model& model, std::span<const Range> ranges, std::uint64_t stamp, std::uint64_t live, bool runsOn, const std::string& where) {
    index.Insert(ranges, stamp, live);
    model.Insert(ranges, stamp, live, runsOn);
    Require(index.Overflows() == model.Overflows(), where + ": the index left " + std::to_string(index.Overflows()) + " inserts unrecorded, the reference " + std::to_string(model.Overflows()) + " (stamp " + std::to_string(stamp) + ", live " + std::to_string(live) + ", " + std::to_string(ranges.size()) + " ranges)");
}

void FixedCases() {
    SetRuns(false);
    auto index = std::make_unique<StampedRanges>();
    const std::array<Range, 1> label{{{0x1000, 0x1004}}};
    index->Insert(label, 1, 1);
    const std::array<Range, 1> touching{{{0x1004, 0x1008}}};
    const std::array<Range, 1> before{{{0x0ffc, 0x1000}}};
    const std::array<Range, 1> inside{{{0x1002, 0x1002}}};
    const std::array<Range, 1> edge{{{0x1004, 0x1004}}};
    const std::array<Range, 1> over{{{0x1003, 0x1004}}};
    Require(index->Newest(over, 1).answer == DeepAnswer::Hit && index->Newest(over, 1).stamp == 1, "a read over the last byte of a write was missed");
    Require(index->Newest(touching, 1).answer == DeepAnswer::Clear, "a read that starts at the end of a write overlapped");
    Require(index->Newest(before, 1).answer == DeepAnswer::Clear, "a read that ends at the start of a write overlapped");
    Require(index->Newest(inside, 1).answer == DeepAnswer::Hit, "an empty read strictly inside a write was not found, unlike the scans");
    Require(index->Newest(edge, 1).answer == DeepAnswer::Clear, "an empty read at the end of a write overlapped");
    Require(index->Newest(label, 2).answer == DeepAnswer::Clear, "a write older than live was found");
    Require(index->Before(label, 1, 1).answer == DeepAnswer::Clear, "an empty window found a write");
    const std::array<Range, 1> empty{{{0x2000, 0x2000}}};
    index->Insert(empty, 2, 1);
    const std::array<Range, 1> around{{{0x1ff0, 0x2010}}};
    const std::array<Range, 1> at{{{0x2000, 0x2010}}};
    Require(index->Newest(around, 1).answer == DeepAnswer::Hit && index->Newest(around, 1).stamp == 2, "an empty write strictly inside a read was missed");
    Require(index->Newest(at, 1).answer == DeepAnswer::Clear, "an empty write at the start of a read overlapped");
    const std::array<Range, 1> inverted{{{0x10, 0x8}}};
    Require(index->Newest(inverted, 1).answer == DeepAnswer::Unsure, "an inverted asked range was not answered unsure");
    Require(index->Within(inverted, 1, 3), "an inverted asked range was not answered written");
    const std::array<Range, 1> image{{{0x40000000, 0x42000000}}};
    index->Insert(image, 3, 1);
    const std::array<Range, 1> texel{{{0x41234560, 0x41234570}}};
    Require(index->Newest(texel, 1).stamp == 3 && index->Before(texel, 1, 3).answer == DeepAnswer::Clear, "a large write was not exact for a small read inside it");
    std::vector<Range> many;
    for (std::uint64_t offset = 0; offset < 40; ++offset) many.push_back({0x5000 + offset * 0x100, 0x5000 + offset * 0x100 + 8});
    index->Insert(many, 4, 1);
    const std::array<Range, 1> lastOfMany{{{0x5000 + 39 * 0x100 + 4, 0x5000 + 39 * 0x100 + 5}}};
    const std::array<Range, 1> gapOfMany{{{0x5000 + 20 * 0x100 + 8, 0x5000 + 21 * 0x100}}};
    Require(index->Newest(lastOfMany, 1).stamp == 4 && index->Newest(lastOfMany, 1).range == many.back(), "the last range of a long write list was missed");
    Require(index->Newest(gapOfMany, 1).answer == DeepAnswer::Clear, "a gap in a long write list overlapped");
    Require(index->Within(lastOfMany, 4, 5) && !index->Within(lastOfMany, 5, 9) && !index->Within(gapOfMany, 1, 9), "a watch over a long write list answered wrong");
    const std::array<Range, 1> added{{{0x9000, 0x9004}}};
    index->Insert(added, 4, 1);
    Require(index->Newest(added, 1).stamp == 4 && index->Newest(lastOfMany, 1).stamp == 4, "a write added to the newest stamp was missed");
    Require(index->Overflows() == 0, "the fixed cases left a write unrecorded");
    const std::array<Range, 1> late{{{0xa000, 0xa004}}};
    index->Insert(late, 4 + Capacity, 5);
    Require(index->Overflows() == 0 && index->Newest(late, 5).stamp == 4 + Capacity, "a stamp capacity - 1 ahead of live was not recorded");
    index->Insert(late, 5 + Capacity, 5);
    Require(index->Overflows() == 1, "a stamp capacity ahead of live was recorded");
    Require(index->Newest(label, 5).answer == DeepAnswer::Unsure, "a lookup whose window holds an unrecorded stamp was not unsure");
    Require(index->Within(label, 1, 2), "a watch whose window holds an unrecorded stamp answered not written");
    index->Insert(late, 3 + Capacity, 5);
    Require(index->Overflows() == 2, "a stamp older than the newest was recorded");
    index->Insert(late, 6 + Capacity, 7);
    const auto after = index->Newest(late, 6 + Capacity);
    Require(index->Overflows() == 2 && after.answer == DeepAnswer::Hit && after.stamp == 6 + Capacity, "a stamp within capacity after an unrecorded one was not recorded");
    Require(index->Newest(added, 6 + Capacity).answer == DeepAnswer::Clear, "a write older than the window was found");
    Require(index->Within(added, 4, 5), "a watch older than the last capacity stamps answered not written");
    const std::array<Range, 2> targets{{{0x40000000, 0x41000000}, {0x48000000, 0x48400000}}};
    const std::array<Range, 1> targetTexel{{{0x40800000, 0x40800010}}};
    auto plain = std::make_unique<StampedRanges>();
    for (std::uint64_t stamp = 1; stamp <= 10; ++stamp) plain->Insert(targets, stamp, 1);
    const auto plainHit = plain->Before(targetTexel, 1, 6);
    Require(plainHit.answer == DeepAnswer::Hit && plainHit.stamp == 5 && !plainHit.conservative, "with the switch off a repeated list was not recorded number by number");
}

void RunCases() {
    SetRuns(true);
    auto index = std::make_unique<StampedRanges>();
    const std::array<Range, 2> targets{{{0x40000000, 0x41000000}, {0x48000000, 0x48400000}}};
    const std::array<Range, 1> texel{{{0x40800000, 0x40800010}}};
    const std::array<Range, 1> beside{{{0x41000000, 0x41000010}}};
    for (std::uint64_t stamp = 1; stamp <= 10; ++stamp) index->Insert(targets, stamp, 1);
    auto found = index->Newest(texel, 1);
    Require(found.answer == DeepAnswer::Hit && found.stamp == 10 && !found.conservative && found.range == targets[0], "the newest writer of a run was not its last number, exactly: " + FoundText(found));
    found = index->Before(texel, 1, 6);
    Require(found.answer == DeepAnswer::Hit && found.stamp == 5 && found.conservative && found.range == targets[0], "a lookup inside a run did not name the number before it as conservative: " + FoundText(found));
    found = index->Before(texel, 1, 3);
    Require(found.answer == DeepAnswer::Hit && found.stamp == 2 && !found.conservative, "the first number of a run was not exact: " + FoundText(found));
    found = index->Before(texel, 1, 2);
    Require(found.answer == DeepAnswer::Hit && found.stamp == 1 && !found.conservative, "the number before a run was not found in its slot: " + FoundText(found));
    Require(index->Newest(beside, 1).answer == DeepAnswer::Clear, "a read that starts at the end of a run's range overlapped");
    Require(index->Within(texel, 4, 6), "a watch inside a run answered not written");
    Require(!index->Within(beside, 1, 11), "a watch beside a run answered written");
    Require(index->Newest(texel, 11).answer == DeepAnswer::Clear, "a run older than live was found");
    index->Insert(targets, 11 + StampedRanges::RunGap, 1);
    found = index->Before(texel, 1, 20);
    Require(found.answer == DeepAnswer::Hit && found.stamp == 10 && !found.conservative, "a run closed over a gap did not end at its last number: " + FoundText(found));
    index->Insert(targets, 12 + StampedRanges::RunGap, 1);
    found = index->Newest(texel, 1);
    Require(found.answer == DeepAnswer::Hit && found.stamp == 12 + StampedRanges::RunGap && !found.conservative, "the run opened after a gap was not found: " + FoundText(found));
    found = index->Before(texel, 1, 12 + StampedRanges::RunGap);
    Require(found.answer == DeepAnswer::Hit && found.stamp == 11 + StampedRanges::RunGap && !found.conservative, "the number recorded in a slot between two runs was not exact: " + FoundText(found));
    Require(!index->Within(texel, 11, 11 + StampedRanges::RunGap), "a watch over the gap between two runs answered written");
    const std::array<Range, 1> marker{{{0x9000, 0x9004}}};
    const auto runNumber = 12 + StampedRanges::RunGap;
    index->Insert(marker, runNumber, 1);
    found = index->Newest(marker, 1);
    Require(found.answer == DeepAnswer::Hit && found.stamp == runNumber && !found.conservative && found.range == marker[0], "a write added to a number recorded in a run was missed: " + FoundText(found));
    found = index->Newest(texel, 1);
    Require(found.answer == DeepAnswer::Hit && found.stamp == runNumber && !found.conservative, "a run number with a slot lost its run: " + FoundText(found));
    const std::array<Range, 1> huge{{{0x1000000000, 0x1080000000}}};
    const std::array<Range, 1> inHuge{{{0x1050000000, 0x1050000004}}};
    index->Insert(huge, runNumber + 1, 1);
    index->Insert(huge, runNumber + 2, 1);
    index->Insert(huge, runNumber + 3, 1);
    found = index->Newest(inHuge, 1);
    Require(found.answer == DeepAnswer::Hit && found.stamp == runNumber + 3 && !found.conservative, "a run of a range wider than the region filter was missed: " + FoundText(found));
    found = index->Before(inHuge, 1, runNumber + 3);
    Require(found.answer == DeepAnswer::Hit && found.stamp == runNumber + 2 && !found.conservative, "the first number of a wide run was not exact: " + FoundText(found));
    const auto quiet = runNumber + 3 + 2 * StampedRanges::RunGap + 64;
    std::uint64_t stamp = runNumber + 4;
    for (; stamp < quiet + Capacity + 64; ++stamp) {
        const Range write{0x100000 + (stamp * 0x20) % 0x10000, 0x100000 + (stamp * 0x20) % 0x10000 + 4};
        index->Insert({&write, 1}, stamp, stamp > 1000 ? stamp - 1000 : 1);
    }
    const auto newest = stamp - 1;
    const std::array<Range, 1> distant{{{0x7000000000, 0x7000000040}}};
    Require(!index->Within(distant, quiet, newest), "an old watch over memory nobody wrote was not proven unwritten");
    Require(!index->Within(texel, quiet, newest), "an old watch over ranges last written before it began was not proven unwritten");
    Require(index->Within(texel, 2, newest), "an old watch over the ranges of a reclaimed run answered not written");
    const std::array<Range, 1> labelPage{{{0x100000 + (newest * 0x20) % 0x10000, 0x100000 + (newest * 0x20) % 0x10000 + 4}}};
    Require(index->Within(labelPage, quiet, newest + 1), "an old watch over a recent label answered not written");
    const std::array<Range, 1> emptyPage{{{0x180000, 0x180010}}};
    Require(index->Newest(emptyPage, newest - 100).answer == DeepAnswer::Clear && index->Newest(labelPage, newest - 100).stamp == newest, "a small read next to the labels answered wrong");
    SetRuns(false);
    Require(index->Within(distant, quiet, newest), "an old watch answered not written with the switch off");
    SetRuns(true);
    Require(index->Overflows() == 0, "the run cases left a write unrecorded");
}

void DeterministicStream() {
    auto index = std::make_unique<StampedRanges>();
    auto model = std::make_unique<Model>();
    Stream stream(0x5eed1234abcdull);
    auto& random = stream.random;
    Tally tally;
    std::uint64_t next = 1;
    std::uint64_t front = 1;
    std::uint64_t last = 0;
    bool any = false;
    constexpr std::uint64_t Steps = 36000;
    std::uint64_t checks = 0;
    for (std::uint64_t step = 0; step < Steps; ++step) {
        const bool runsOn = (step / 2500) % 5 != 4;
        SetRuns(runsOn);
        const bool overrun = step >= 9000 && step < 10400;
        const bool wideLists = step >= 18000 && step < 20400;
        const bool crowded = step >= 26000 && step < 28000;
        const bool plain = step >= 30000 && step < 34000;
        const bool deep = step >= 14000 && step < 17000;
        const auto where = "step " + std::to_string(step);
        const auto action = random.Below(100);
        if (action < 55) {
            auto stamp = next;
            if (!wideLists && random.Chance(10)) stamp += random.Below(40);
            if (step % 4000 == 3999) stamp += 3 * Capacity + random.Below(Capacity);
            next = stamp + 1;
            if (!overrun && stamp >= front + Capacity) front = stamp + 1 - Capacity + random.Below(Capacity / 4);
            if (front > stamp) front = stamp;
            const auto ranges = stream.Write(wideLists, crowded, plain);
            InsertBoth(*index, *model, ranges, stamp, front, runsOn, where + " insert");
            last = stamp;
            any = true;
        } else if (action < 63 && any) {
            const auto ranges = stream.Write(false, false, plain);
            InsertBoth(*index, *model, ranges, last, front, runsOn, where + " append");
        } else if (action < 65 && any && last > 8 && !plain) {
            const auto ranges = stream.Write(false, false, plain);
            InsertBoth(*index, *model, ranges, last - 1 - random.Below(8), front, runsOn, where + " out of order");
        } else if (action < 82 && !overrun && !deep) {
            front = std::min(next, front + random.Below(64));
        } else if (action < 84 && !overrun && !deep) {
            front = std::min(next, front + random.Below(Capacity));
        }
        for (int query = 0; query < 3; ++query) {
            const auto asked = stream.Query();
            const auto kind = random.Below(100);
            if (kind < 30) {
                const auto before = front + random.Below(next - front + 3);
                CheckBefore(*index, *model, asked, front, before, where, tally);
            } else if (kind < 55) {
                CheckBefore(*index, *model, asked, front, Unbounded, where, tally);
            } else if (kind < 65) {
                const auto stale = front > 2 * Capacity ? front - random.Below(2 * Capacity) : random.Below(front + 1);
                const auto before = random.Chance(50) ? Unbounded : stale + random.Below(next - stale + 3);
                CheckBefore(*index, *model, asked, stale, before, where + " stale live", tally);
            } else {
                const auto from = next > 2 * Capacity ? next - 2 * Capacity + random.Below(2 * Capacity + 2) : random.Below(next + 2);
                const auto to = from + random.Below(next + 3 > from ? next + 3 - from : 1);
                CheckWithin(*index, *model, asked, from, to, runsOn, where, tally);
            }
            ++checks;
        }
    }
    SetRuns(true);
    Require(model->Unrecorded(Model::Reason::Beyond) != 0, "the stream never wrote beyond capacity");
    Require(model->Unrecorded(Model::Reason::Disorder) != 0, "the stream never wrote out of order");
    Require(model->Unrecorded(Model::Reason::Pool) != 0, "the stream never filled the spill pool");
    Require(model->RunsOpened() != 0 && model->RunsExtended() != 0, "the stream never opened and extended a run");
    Require(model->ClosedByLive() != 0 && model->ClosedByGap() != 0, "the stream never closed a run by live and by a gap");
    Require(model->NoFreeRun() != 0, "the stream never filled the run table");
    Require(model->FreshSlots() != 0, "the stream never added a write to a number recorded in a run");
    Require(tally.conservative != 0, "the stream never checked a conservative answer");
    Require(tally.oldFalse != 0, "the stream never proved an old watch unwritten");
    std::cout << "deterministic stream: " << checks << " answers equal to the reference, " << model->Overflows() << " writes unrecorded, " << model->RunsOpened() << " runs opened, " << model->RunsExtended() << " extended, " << tally.conservative << " conservative answers, old watches " << tally.oldFalse << " proven unwritten and " << tally.oldTrue << " kept written, newest stamp " << model->Newest() << '\n';
}

constexpr std::uint64_t MarkerBase = 0x100000000;
constexpr std::uint64_t NoiseBase = 0x200000000;
constexpr std::uint64_t NoiseSpan = 64 * MiB;
constexpr std::uint64_t MarkerCycle = 4096;
constexpr std::uint64_t RunBase = 0x300000000;
constexpr std::array<Range, 3> TargetList{{{RunBase, RunBase + 8 * MiB}, {RunBase + 16 * MiB, RunBase + 20 * MiB}, {RunBase + 32 * MiB, RunBase + 32 * MiB + 0x1000}}};

Range Marker(std::uint64_t stamp) {
    const auto at = MarkerBase + (stamp % MarkerCycle) * 0x40;
    return {at, at + 0x10};
}

bool CarriesTargets(std::uint64_t stamp) {
    return ((stamp >> 6) & 1u) == 0 && (stamp & 1u) == 0;
}

bool InTargetPhase(std::uint64_t stamp) {
    return ((stamp >> 6) & 1u) == 0;
}

bool PhaseMeets(std::uint64_t from, std::uint64_t to) {
    for (auto stamp = from; stamp < to; ++stamp) {
        if (InTargetPhase(stamp)) return true;
        stamp |= 63;
    }
    return false;
}

std::optional<std::uint64_t> NewestCarrier(std::uint64_t from, std::uint64_t to) {
    for (auto stamp = to; stamp-- > from;) {
        if (CarriesTargets(stamp)) return stamp;
    }
    return std::nullopt;
}

void RangesOf(std::uint64_t stamp, std::vector<Range>& first, std::vector<Range>& second) {
    first.clear();
    second.clear();
    const auto hash = Mix(stamp);
    const bool carries = CarriesTargets(stamp);
    if (carries) {
        first.assign(TargetList.begin(), TargetList.end());
        second.push_back(Marker(stamp));
    } else {
        first.push_back(Marker(stamp));
    }
    auto& extra = carries || ((hash >> 60) & 3) == 0 ? second : first;
    if ((hash & 3) == 0) {
        static constexpr std::array<std::uint64_t, 5> sizes{4, 64, 4096, MiB, 16 * MiB};
        const auto at = NoiseBase + (hash >> 8) % NoiseSpan;
        extra.push_back({at, at + sizes[(hash >> 40) % sizes.size()]});
    }
    if ((hash & 15) == 1) {
        auto spread = Mix(hash);
        for (int index = 0; index < 8; ++index) {
            const auto at = NoiseBase + spread % NoiseSpan;
            extra.push_back({at, at + 32});
            spread = Mix(spread);
        }
    }
}

bool WritesOverlap(std::uint64_t stamp, const Range& read, Range& hit, std::vector<Range>& first, std::vector<Range>& second) {
    RangesOf(stamp, first, second);
    for (const auto* list : {&first, &second}) {
        for (const auto& write : *list) {
            if (!Overlaps(write, read)) continue;
            hit = write;
            return true;
        }
    }
    return false;
}

void Concurrent() {
    SetRuns(true);
    auto index = std::make_unique<StampedRanges>();
    std::atomic<std::uint64_t> front{1};
    std::atomic<std::uint64_t> inserting{0};
    std::atomic<std::uint64_t> inserted{0};
    std::atomic<bool> failed{false};
    std::atomic<std::uint64_t> unsureAnswers{0};
    std::atomic<std::uint64_t> answers{0};
    std::atomic<std::uint64_t> conservativeAnswers{0};
    std::mutex failureMutex;
    std::string failure;
    const auto fail = [&](const std::string& message) {
        const std::lock_guard lock(failureMutex);
        if (failure.empty()) failure = message;
        failed.store(true, std::memory_order_relaxed);
    };
    constexpr std::uint64_t WriterStamps = 240000;
    constexpr std::uint64_t ReaderRounds = 30000;
    std::thread writer([&] {
        try {
            Random random(0x77ull);
            std::vector<Range> first;
            std::vector<Range> second;
            std::uint64_t live = 1;
            for (std::uint64_t stamp = 1; stamp <= WriterStamps && !failed.load(std::memory_order_relaxed); ++stamp) {
                if (stamp - live > Capacity - 2) live = stamp - (Capacity - 2);
                else if (random.Chance(40)) live = std::min(stamp, live + random.Below(3));
                front.store(live, std::memory_order_release);
                inserting.store(stamp, std::memory_order_seq_cst);
                RangesOf(stamp, first, second);
                index->Insert(first, stamp, live);
                if (!second.empty()) index->Insert(second, stamp, live);
                inserted.store(stamp, std::memory_order_release);
            }
        } catch (const std::exception& error) {
            fail(std::string("writer: ") + error.what());
        }
    });
    const auto reader = [&](std::uint64_t seed) {
        try {
            Random random(seed);
            std::vector<Range> first;
            std::vector<Range> second;
            std::uint64_t rounds = 0;
            while (rounds < ReaderRounds && !failed.load(std::memory_order_relaxed)) {
                const auto oldest = front.load(std::memory_order_acquire);
                const auto newest = inserted.load(std::memory_order_acquire);
                if (newest < oldest) {
                    std::this_thread::yield();
                    continue;
                }
                ++rounds;
                const auto target = oldest + random.Below(newest - oldest + 1);
                const auto marker = Marker(target);
                const Range inside{marker.first + 4, marker.first + 8};
                const Range gap{marker.first + 0x10, marker.first + 0x40};
                const auto settle = [&] {
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    return inserting.load(std::memory_order_seq_cst);
                };
                const auto context = [&](const char* lookup, const DeepFound& found) {
                    return std::string(lookup) + " of the marker of stamp " + std::to_string(target) + " from live " + std::to_string(oldest) + " with " + std::to_string(newest) + " inserted answered " + FoundText(found);
                };
                auto found = index->Before({&inside, 1}, oldest, target + 1);
                auto after = settle();
                if (after < target + Capacity) {
                    if (found.answer != DeepAnswer::Hit || found.stamp != target || found.range != marker) throw std::runtime_error(context("before", found) + " while its slot was not reused");
                } else if (found.answer == DeepAnswer::Hit && (found.stamp != target || found.range != marker)) {
                    throw std::runtime_error(context("before", found));
                }
                found = index->Newest({&inside, 1}, oldest);
                after = settle();
                if (found.answer == DeepAnswer::Unsure) unsureAnswers.fetch_add(1, std::memory_order_relaxed);
                if (found.answer == DeepAnswer::Hit && (found.stamp < target || (found.stamp - target) % MarkerCycle != 0 || found.range != marker)) throw std::runtime_error(context("newest", found));
                if (found.answer == DeepAnswer::Clear && after < target + Capacity) throw std::runtime_error(context("newest", found) + " while its slot was not reused");
                found = index->Newest({&gap, 1}, oldest);
                if (found.answer == DeepAnswer::Unsure) unsureAnswers.fetch_add(1, std::memory_order_relaxed);
                if (found.answer == DeepAnswer::Hit) throw std::runtime_error(context("newest of a gap", found));
                if (!index->Within({&inside, 1}, target, target + 1)) throw std::runtime_error("a watch of stamp " + std::to_string(target) + " over its own marker answered not written");
                const bool gapWritten = index->Within({&gap, 1}, oldest, newest + 1);
                after = settle();
                if (gapWritten && after < oldest + Capacity) throw std::runtime_error("a watch from " + std::to_string(oldest) + " over a range nobody writes answered written");
                const Range aimed{RunBase + 0x1000 + random.Below(4 * MiB), RunBase + 0x1000 + random.Below(4 * MiB) + 4 * MiB + 16};
                const auto bound = random.Chance(50) ? Unbounded : oldest + 1 + random.Below(newest - oldest + 1);
                found = bound == Unbounded ? index->Newest({&aimed, 1}, oldest) : index->Before({&aimed, 1}, oldest, bound);
                after = settle();
                const auto carrier = NewestCarrier(oldest, bound == Unbounded ? newest + 1 : bound);
                const auto targetContext = [&](const char* why) {
                    return std::string("a lookup of the repeated targets from live ") + std::to_string(oldest) + (bound == Unbounded ? std::string(" (newest)") : " before " + std::to_string(bound)) + " with " + std::to_string(newest) + " inserted answered " + FoundText(found) + ": " + why;
                };
                if (found.answer == DeepAnswer::Unsure) unsureAnswers.fetch_add(1, std::memory_order_relaxed);
                if (found.answer == DeepAnswer::Hit) {
                    if (found.conservative) conservativeAnswers.fetch_add(1, std::memory_order_relaxed);
                    if (found.stamp < oldest || (bound != Unbounded && found.stamp >= bound) || found.stamp > after) throw std::runtime_error(targetContext("outside the window"));
                    if (carrier && after < *carrier + Capacity && found.stamp < *carrier) throw std::runtime_error(targetContext("older than the newest writer"));
                    if (!InTargetPhase(found.stamp)) throw std::runtime_error(targetContext("a number outside every run of the targets"));
                    if (!found.conservative && !CarriesTargets(found.stamp)) throw std::runtime_error(targetContext("an exact answer that does not write the targets"));
                    if (found.range != TargetList[0]) throw std::runtime_error(targetContext("a range that is not the first target"));
                } else if (found.answer == DeepAnswer::Clear && carrier && after < *carrier + Capacity) {
                    throw std::runtime_error(targetContext("clear while a writer was in the window"));
                }
                const auto watchFrom = oldest + random.Below(newest - oldest + 1);
                const auto watchTo = watchFrom + 1 + random.Below(newest + 1 - watchFrom + 1);
                const bool watched = index->Within({&aimed, 1}, watchFrom, watchTo);
                after = settle();
                const auto watchedCarrier = NewestCarrier(watchFrom, std::min(watchTo, newest + 1));
                if (watchedCarrier && !watched) throw std::runtime_error("a watch of the repeated targets over [" + std::to_string(watchFrom) + ", " + std::to_string(watchTo) + ") answered not written although number " + std::to_string(*watchedCarrier) + " wrote them");
                if (watched && !watchedCarrier && !PhaseMeets(watchFrom, std::min(watchTo, after + 1)) && after < watchFrom + Capacity) throw std::runtime_error("a watch of the repeated targets over [" + std::to_string(watchFrom) + ", " + std::to_string(watchTo) + ") answered written without a writer or a run");
                if (rounds % 4 == 0) {
                    const auto at = NoiseBase + random.Below(NoiseSpan);
                    const Range noise{at, at + 1 + random.Below(MiB)};
                    found = index->Before({&noise, 1}, oldest, newest + 1);
                    after = settle();
                    if (after < oldest + Capacity) {
                        DeepFound expected;
                        for (auto stamp = newest + 1; stamp-- > oldest;) {
                            Range hit{};
                            if (!WritesOverlap(stamp, noise, hit, first, second)) continue;
                            expected = DeepFound{DeepAnswer::Hit, stamp, false, hit};
                            break;
                        }
                        if (!SameFound(found, expected)) throw std::runtime_error("before over noise " + RangeText(noise) + " from live " + std::to_string(oldest) + " to " + std::to_string(newest + 1) + " answered " + FoundText(found) + ", the scan " + FoundText(expected));
                    } else if (found.answer == DeepAnswer::Hit) {
                        Range hit{};
                        if (found.stamp < oldest || found.stamp > newest || !WritesOverlap(found.stamp, noise, hit, first, second)) throw std::runtime_error("before over noise " + RangeText(noise) + " named stamp " + std::to_string(found.stamp) + ", which does not write it");
                    }
                }
                answers.fetch_add(8, std::memory_order_relaxed);
            }
        } catch (const std::exception& error) {
            fail(error.what());
        }
    };
    std::thread readerOne(reader, 0x1001ull);
    std::thread readerTwo(reader, 0x2002ull);
    std::thread readerThree(reader, 0x3003ull);
    readerOne.join();
    readerTwo.join();
    readerThree.join();
    writer.join();
    Require(!failed.load(), "concurrent readers: " + failure);
    Require(inserted.load() == WriterStamps, "the writer stopped at stamp " + std::to_string(inserted.load()));
    Require(index->Overflows() == 0, "the writer kept the capacity yet " + std::to_string(index->Overflows()) + " writes went unrecorded");
    std::cout << "concurrent: " << answers.load() << " answers checked, " << unsureAnswers.load() << " unsure after torn reads, " << conservativeAnswers.load() << " conservative answers over the repeated targets\n";
}

}

int main() {
    const bool initial = RuntimeSwitches::Set(RuntimeSwitch::DeepRuns);
    try {
        FixedCases();
        RunCases();
        DeterministicStream();
        Concurrent();
        SetRuns(initial);
        std::cout << "deep index tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        SetRuns(initial);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
