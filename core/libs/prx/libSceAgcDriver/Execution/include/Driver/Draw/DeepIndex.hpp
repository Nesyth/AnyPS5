#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DEEPINDEX_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DEEPINDEX_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

enum class DeepAnswer : std::uint8_t { Clear, Hit, Unsure };

struct DeepFound {
    DeepAnswer answer = DeepAnswer::Clear;
    std::uint64_t stamp = 0;
    bool conservative = false;
    std::pair<std::uint64_t, std::uint64_t> range{};
};

class StampedRanges {
public:
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    static constexpr unsigned RegionShift = 12;
    static constexpr unsigned CoarseShift = 21;
    static constexpr std::size_t Buckets = 8192;
    static constexpr unsigned BucketBits = 13;
    static constexpr std::size_t Ways = 64;
    static constexpr std::size_t Graves = 4;
    static constexpr std::uint64_t WideRegions = 64;
    static constexpr std::uint64_t QueryRegions = 1024;
    static constexpr std::uint64_t CoarseQueryRegions = 64;
    static constexpr unsigned ReadAttempts = 8;
    static constexpr std::size_t Memos = 4;

    static bool Regions(const Range& range, std::uint64_t limit, std::uint64_t& first, std::uint64_t& last);
    static bool CoarseRegions(const Range& range, std::uint64_t limit, std::uint64_t& first, std::uint64_t& last);
    void Insert(std::span<const Range> ranges, std::uint64_t stamp, std::uint64_t live);
    DeepFound Before(std::span<const Range> ranges, std::uint64_t live, std::uint64_t before) const;
    DeepFound Newest(std::span<const Range> ranges, std::uint64_t live) const;
    bool Within(std::span<const Range> ranges, std::uint64_t from, std::uint64_t to) const;
    std::uint64_t Overflows() const { return overflows.load(std::memory_order_relaxed); }
    void Name(const char* name) { label = name; }

private:
    struct Entry {
        std::atomic<std::uint32_t> version{0};
        std::atomic<std::uint64_t> begin{0};
        std::atomic<std::uint64_t> end{0};
        std::atomic<std::uint64_t> first{0};
        std::atomic<std::uint64_t> last{0};
    };
    struct Census {
        std::atomic<std::uint64_t> overflows{0};
        std::array<std::atomic<std::uint64_t>, 5> placed{};
        std::array<std::atomic<std::uint64_t>, 5> occupants{};
        std::atomic<std::uint64_t> occupantsLive{0};
        std::atomic<std::uint64_t> occupantsInRegion{0};
        std::atomic<std::uint64_t> occupantPages{0};
        std::atomic<std::uint64_t> unsure{0};
        std::array<std::atomic<std::uint64_t>, 4> reasons{};
        std::array<std::atomic<std::uint64_t>, 5> asked{};
        std::atomic<std::uint64_t> askedRanges{0};
    };
    struct Bucket {
        std::atomic<std::uint64_t> newest{0};
        std::atomic<std::uint64_t> overflowFirst{0};
        std::atomic<std::uint64_t> overflowLast{0};
        std::atomic<std::uint64_t> reclaimed{0};
        std::array<Entry, Ways> ways{};
        std::array<Entry, Graves> graves{};
        std::size_t nextGrave = 0;
    };
    struct View {
        std::uint64_t begin = 0;
        std::uint64_t end = 0;
        std::uint64_t first = 0;
        std::uint64_t last = 0;
    };
    struct MemoEntry {
        Entry* entry = nullptr;
        Bucket* bucket = nullptr;
        std::size_t range = 0;
    };
    struct Memo {
        std::vector<Range> ranges;
        std::vector<MemoEntry> entries;
        bool valid = false;
    };
    Bucket& bucketAt(std::uint64_t region);
    const Bucket& bucketAt(std::uint64_t region) const;
    bool place(Bucket& bucket, const Range& range, std::size_t index, std::uint64_t stamp, std::uint64_t live, Memo& memo, std::uint64_t region);
    void noteOverflow(const Bucket& bucket, const Range& range, std::uint64_t region, std::uint64_t stamp, std::uint64_t live);
    void noteUnsure(unsigned reason, const Bucket* bucket, const Range& range, std::size_t count, std::uint64_t live, std::uint64_t before) const;
    void reportCensus() const;
    static void bury(Bucket& bucket, const Entry& entry);
    static bool read(const Entry& entry, View& view);
    static bool scanBefore(const Bucket& bucket, const Range& range, std::uint64_t live, std::uint64_t before, DeepFound& found);
    static bool scanWithin(const Bucket& bucket, const Range& range, std::uint64_t from, std::uint64_t to);

    std::array<Bucket, Buckets> buckets{};
    Bucket wide{};
    std::array<Memo, Memos> memos{};
    std::size_t nextMemo = 0;
    std::atomic<std::uint64_t> overflows{0};
    const char* label = "unnamed";
    mutable Census census;
};

class StampedKeys {
public:
    static constexpr std::size_t Buckets = 1024;
    static constexpr unsigned BucketBits = 10;
    static constexpr std::size_t Ways = 4;
    static constexpr unsigned ReadAttempts = 8;

    void Insert(std::uint64_t key, std::uint64_t stamp, bool inserts, std::uint64_t live);
    void InsertAny(std::uint64_t stamp, bool inserts, std::uint64_t live);
    DeepFound Before(std::uint64_t key, bool insertsOnly, std::uint64_t live, std::uint64_t before) const;
    DeepFound BeforeAny(bool insertsOnly, std::uint64_t live, std::uint64_t before) const;
    std::uint64_t Overflows() const { return overflows.load(std::memory_order_relaxed); }

private:
    struct Stamps {
        std::atomic<std::uint64_t> first{0};
        std::atomic<std::uint64_t> last{0};
        std::atomic<std::uint64_t> firstInserts{0};
        std::atomic<std::uint64_t> lastInserts{0};
    };
    struct Entry {
        std::atomic<std::uint32_t> version{0};
        std::atomic<std::uint64_t> key{0};
        Stamps stamps;
    };
    struct Bucket {
        Stamps overflow;
        std::array<Entry, Ways> ways{};
    };
    static void touch(Stamps& stamps, std::uint64_t stamp, bool inserts, std::uint64_t live);
    static void rewrite(Entry& entry, std::uint64_t key, std::uint64_t stamp, bool inserts);
    static void consider(std::uint64_t first, std::uint64_t last, std::uint64_t live, std::uint64_t before, DeepFound& found);
    Bucket& bucketOf(std::uint64_t key);
    const Bucket& bucketOf(std::uint64_t key) const;

    std::array<Bucket, Buckets> buckets{};
    Stamps any;
    Entry* recent = nullptr;
    std::uint64_t recentKey = 0;
    std::atomic<std::uint64_t> overflows{0};
};

}

#endif
