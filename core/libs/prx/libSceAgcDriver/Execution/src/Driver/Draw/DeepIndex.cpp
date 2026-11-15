#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DeepIndex.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include <algorithm>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::uint64_t Golden = 0x9E3779B97F4A7C15ull;

std::size_t hashed(std::uint64_t value, unsigned bits) {
    return static_cast<std::size_t>((value * Golden) >> (64u - bits));
}

}

bool StampedRanges::Regions(const Range& range, std::uint64_t limit, std::uint64_t& first, std::uint64_t& last) {
    if (range.second < range.first) return false;
    first = range.first >> RegionShift;
    last = (range.second > range.first ? range.second - 1 : range.first) >> RegionShift;
    return last - first < limit;
}

StampedRanges::Bucket& StampedRanges::bucketAt(std::uint64_t region) {
    return buckets[hashed(region, BucketBits)];
}

const StampedRanges::Bucket& StampedRanges::bucketAt(std::uint64_t region) const {
    return buckets[hashed(region, BucketBits)];
}

void StampedRanges::bury(Bucket& bucket, const Entry& entry) {
    auto& grave = bucket.graves[bucket.nextGrave];
    bucket.nextGrave = (bucket.nextGrave + 1) % Graves;
    const auto buried = grave.last.load(std::memory_order_relaxed);
    if (buried > bucket.reclaimed.load(std::memory_order_relaxed)) bucket.reclaimed.store(buried, std::memory_order_release);
    const auto version = grave.version.load(std::memory_order_relaxed);
    grave.version.store(version + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    grave.begin.store(entry.begin.load(std::memory_order_relaxed), std::memory_order_relaxed);
    grave.end.store(entry.end.load(std::memory_order_relaxed), std::memory_order_relaxed);
    grave.first.store(entry.first.load(std::memory_order_relaxed), std::memory_order_relaxed);
    grave.last.store(entry.last.load(std::memory_order_relaxed), std::memory_order_relaxed);
    grave.version.store(version + 2, std::memory_order_release);
}

bool StampedRanges::place(Bucket& bucket, const Range& range, std::size_t index, std::uint64_t stamp, std::uint64_t live, Memo& memo) {
    Entry* free = nullptr;
    std::uint64_t freeLast = 0;
    for (auto& way : bucket.ways) {
        const auto last = way.last.load(std::memory_order_relaxed);
        if (last >= live) {
            if (way.begin.load(std::memory_order_relaxed) != range.first || way.end.load(std::memory_order_relaxed) != range.second) continue;
            way.last.store(stamp, std::memory_order_release);
            bucket.newest.store(stamp, std::memory_order_release);
            memo.entries.push_back({&way, &bucket, index});
            return true;
        }
        if (free == nullptr || last < freeLast) {
            free = &way;
            freeLast = last;
        }
    }
    if (free != nullptr) {
        if (freeLast != 0) bury(bucket, *free);
        const auto version = free->version.load(std::memory_order_relaxed);
        free->version.store(version + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        free->begin.store(range.first, std::memory_order_relaxed);
        free->end.store(range.second, std::memory_order_relaxed);
        free->first.store(stamp, std::memory_order_relaxed);
        free->last.store(stamp, std::memory_order_relaxed);
        free->version.store(version + 2, std::memory_order_release);
        bucket.newest.store(stamp, std::memory_order_release);
        memo.entries.push_back({free, &bucket, index});
        return true;
    }
    const auto overflowLast = bucket.overflowLast.load(std::memory_order_relaxed);
    if (overflowLast < live) {
        if (overflowLast > bucket.reclaimed.load(std::memory_order_relaxed)) bucket.reclaimed.store(overflowLast, std::memory_order_release);
        bucket.overflowFirst.store(stamp, std::memory_order_release);
    }
    bucket.overflowLast.store(stamp, std::memory_order_release);
    bucket.newest.store(stamp, std::memory_order_release);
    overflows.store(overflows.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    return false;
}

void StampedRanges::Insert(std::span<const Range> ranges, std::uint64_t stamp, std::uint64_t live) {
    if (ranges.empty()) return;
    for (auto& memo : memos) {
        if (!memo.valid || memo.ranges.size() != ranges.size() || !std::equal(ranges.begin(), ranges.end(), memo.ranges.begin())) continue;
        const bool fresh = std::all_of(memo.entries.begin(), memo.entries.end(), [&](const MemoEntry& held) {
            const auto& range = memo.ranges[held.range];
            return held.entry->last.load(std::memory_order_relaxed) >= live && held.entry->begin.load(std::memory_order_relaxed) == range.first && held.entry->end.load(std::memory_order_relaxed) == range.second;
        });
        if (!fresh) {
            memo.valid = false;
            break;
        }
        for (const auto& held : memo.entries) {
            held.entry->last.store(stamp, std::memory_order_release);
            held.bucket->newest.store(stamp, std::memory_order_release);
        }
        return;
    }
    auto& memo = memos[nextMemo];
    nextMemo = (nextMemo + 1) % memos.size();
    memo.valid = false;
    memo.ranges.assign(ranges.begin(), ranges.end());
    memo.entries.clear();
    bool placed = true;
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        const auto& range = ranges[index];
        std::uint64_t first = 0;
        std::uint64_t last = 0;
        if (!Regions(range, WideRegions, first, last)) {
            placed = place(wide, range, index, stamp, live, memo) && placed;
            continue;
        }
        for (auto region = first; region <= last; ++region) placed = place(bucketAt(region), range, index, stamp, live, memo) && placed;
    }
    memo.valid = placed;
}

bool StampedRanges::read(const Entry& entry, View& view) {
    for (unsigned attempt = 0; attempt < ReadAttempts; ++attempt) {
        const auto version = entry.version.load(std::memory_order_acquire);
        if ((version & 1u) != 0) {
            SpinPause();
            continue;
        }
        view.begin = entry.begin.load(std::memory_order_relaxed);
        view.end = entry.end.load(std::memory_order_relaxed);
        view.first = entry.first.load(std::memory_order_relaxed);
        view.last = entry.last.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (entry.version.load(std::memory_order_relaxed) == version) return true;
    }
    return false;
}

bool StampedRanges::scanBefore(const Bucket& bucket, const Range& range, std::uint64_t live, std::uint64_t before, DeepFound& found) {
    if (bucket.newest.load(std::memory_order_acquire) < live) return true;
    const auto overflowLast = bucket.overflowLast.load(std::memory_order_acquire);
    if (overflowLast >= live) {
        const auto overflowFirst = bucket.overflowFirst.load(std::memory_order_acquire);
        if (overflowFirst > overflowLast || overflowFirst < before) return false;
    }
    for (const auto& way : bucket.ways) {
        if (way.last.load(std::memory_order_acquire) < live) continue;
        View view;
        if (!read(way, view)) return false;
        if (view.last < live || range.first >= view.end || view.begin >= range.second) continue;
        std::uint64_t candidate = view.last;
        bool conservative = false;
        if (view.last >= before) {
            if (view.first >= before) continue;
            candidate = before - 1;
            conservative = true;
        }
        if (candidate < live) continue;
        if (found.answer != DeepAnswer::Hit || candidate > found.stamp || (candidate == found.stamp && found.conservative && !conservative)) found = {DeepAnswer::Hit, candidate, conservative, {view.begin, view.end}};
    }
    return true;
}

DeepFound StampedRanges::Before(std::span<const Range> ranges, std::uint64_t live, std::uint64_t before) const {
    DeepFound found;
    const bool wideLive = wide.newest.load(std::memory_order_acquire) >= live;
    for (const auto& range : ranges) {
        std::uint64_t first = 0;
        std::uint64_t last = 0;
        if (!Regions(range, QueryRegions, first, last)) return {DeepAnswer::Unsure};
        for (auto region = first; region <= last; ++region) {
            if (!scanBefore(bucketAt(region), range, live, before, found)) return {DeepAnswer::Unsure};
        }
        if (wideLive && !scanBefore(wide, range, live, before, found)) return {DeepAnswer::Unsure};
    }
    return found;
}

DeepFound StampedRanges::Newest(std::span<const Range> ranges, std::uint64_t live) const {
    return Before(ranges, live, ~std::uint64_t{0});
}

bool StampedRanges::scanWithin(const Bucket& bucket, const Range& range, std::uint64_t from, std::uint64_t to) {
    if (bucket.newest.load(std::memory_order_acquire) < from) return false;
    for (const auto& way : bucket.ways) {
        if (way.last.load(std::memory_order_acquire) < from) continue;
        View view;
        if (!read(way, view)) return true;
        if (view.last >= from && view.first < to && view.begin < range.second && range.first < view.end) return true;
    }
    for (const auto& grave : bucket.graves) {
        if (grave.last.load(std::memory_order_acquire) < from) continue;
        View view;
        if (!read(grave, view)) return true;
        if (view.last >= from && view.first < to && view.begin < range.second && range.first < view.end) return true;
    }
    const auto overflowLast = bucket.overflowLast.load(std::memory_order_acquire);
    if (overflowLast >= from) {
        const auto overflowFirst = bucket.overflowFirst.load(std::memory_order_acquire);
        if (overflowFirst < to || overflowFirst > overflowLast) return true;
    }
    return bucket.reclaimed.load(std::memory_order_acquire) >= from;
}

bool StampedRanges::Within(std::span<const Range> ranges, std::uint64_t from, std::uint64_t to) const {
    if (from >= to) return false;
    const bool wideSeen = wide.newest.load(std::memory_order_acquire) >= from;
    for (const auto& range : ranges) {
        std::uint64_t first = 0;
        std::uint64_t last = 0;
        if (!Regions(range, QueryRegions, first, last)) return true;
        for (auto region = first; region <= last; ++region) {
            if (scanWithin(bucketAt(region), range, from, to)) return true;
        }
        if (wideSeen && scanWithin(wide, range, from, to)) return true;
    }
    return false;
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
