#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DEEPINDEX_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DEEPINDEX_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

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
    static constexpr std::uint64_t QueryRegions = 1024;
    static constexpr std::uint64_t CoarseQueryRegions = 64;
    static constexpr std::size_t Capacity = 1024;
    static constexpr std::size_t Inline = 2;
    static constexpr std::size_t BlockRanges = 7;
    static constexpr std::size_t PoolBlocks = 4096;
    static constexpr std::size_t MaxRanges = Inline + PoolBlocks * BlockRanges;
    static constexpr std::size_t Levels = 6;
    static constexpr std::array<unsigned, Levels> Shifts{6, 10, 14, 18, 22, 26};
    static constexpr std::array<unsigned, Levels> OwnBits{13, 12, 11, 10, 9, 9};
    static constexpr std::array<unsigned, Levels> DeepBits{0, 12, 11, 10, 9, 9};
    static constexpr std::uint64_t WriteBlocks = 2;
    static constexpr std::uint64_t QueryBlocks = 8;
    static constexpr unsigned NearShift = 22;
    static constexpr unsigned NearBits = 12;
    static constexpr std::uint64_t NearRegions = 16;
    static constexpr unsigned PageShift = 12;
    static constexpr unsigned PageBits = 13;
    static constexpr std::uint64_t PageQueryPages = 4;
    static constexpr std::size_t FineLevels = 2;
    static constexpr unsigned ReadAttempts = 8;
    static constexpr std::size_t RunEntries = 64;
    static constexpr std::size_t RunRanges = 10;
    static constexpr std::uint64_t RunGap = 16;
    static constexpr std::size_t RecentLists = 4;
    static constexpr std::uint64_t RunRegionLimit = 64;
    static constexpr std::uint64_t RunQueryRegions = 16;

    StampedRanges();
    ~StampedRanges();
    StampedRanges(const StampedRanges&) = delete;
    StampedRanges& operator=(const StampedRanges&) = delete;

    static bool Regions(const Range& range, std::uint64_t limit, std::uint64_t& first, std::uint64_t& last);
    static bool CoarseRegions(const Range& range, std::uint64_t limit, std::uint64_t& first, std::uint64_t& last);
    void Insert(std::span<const Range> ranges, std::uint64_t stamp, std::uint64_t live);
    DeepFound Before(std::span<const Range> ranges, std::uint64_t live, std::uint64_t before) const;
    DeepFound Newest(std::span<const Range> ranges, std::uint64_t live) const;
    bool Within(std::span<const Range> ranges, std::uint64_t from, std::uint64_t to) const;
    std::uint64_t Overflows() const { return overflows.load(std::memory_order_relaxed); }
    void Name(const char* name) { label = name; }

private:
    static constexpr std::size_t Words = Capacity / 64;
    static constexpr std::uint64_t SlotMask = Capacity - 1;
    static constexpr std::uint32_t NoBlock = 0xffffffffu;
    static constexpr std::uint32_t NoRun = 0xffffffffu;
    static constexpr std::size_t HugeLevel = Levels;
    static constexpr std::size_t Classes = Levels + 1;
    static constexpr std::size_t CensusStripes = 8;
    static constexpr std::size_t TestedBuckets = 7;
    static_assert(RunEntries <= 64, "the run masks hold one bit per run entry in a 64-bit word");
    static_assert(RunQueryRegions <= NearRegions, "a lookup that tests run masks reads only the region words its near filter reads");

    struct Cell {
        std::atomic<std::uint64_t> begin{0};
        std::atomic<std::uint64_t> end{0};
    };
    struct alignas(64) Slot {
        std::atomic<std::uint32_t> version{0};
        std::atomic<std::uint32_t> count{0};
        std::atomic<std::uint64_t> stamp{0};
        std::atomic<std::uint32_t> head{NoBlock};
        std::array<Cell, Inline> cells{};
    };
    struct alignas(64) Block {
        std::atomic<std::uint32_t> next{NoBlock};
        std::array<Cell, BlockRanges> cells{};
    };
    struct Meta {
        std::atomic<std::uint64_t> newest{0};
        std::atomic<std::uint64_t> summary{0};
    };
    struct alignas(64) Map {
        std::array<std::atomic<std::uint64_t>, Words> words{};
    };
    struct alignas(32) Near {
        std::atomic<std::uint64_t> coarse{0};
        std::atomic<std::uint64_t> fine{0};
        std::atomic<std::uint64_t> runs{0};
        std::atomic<std::uint64_t> runBound{0};
    };
    struct alignas(64) Front {
        std::atomic<std::uint64_t> newest{0};
        std::atomic<std::uint64_t> lost{0};
        std::atomic<std::uint64_t> wideRuns{0};
    };
    struct alignas(64) ClassStamps {
        std::array<std::atomic<std::uint64_t>, Classes> newest{};
    };
    struct alignas(64) RunSlot {
        std::atomic<std::uint32_t> version{0};
        std::atomic<std::uint32_t> count{0};
        std::atomic<std::uint64_t> first{0};
        std::atomic<std::uint64_t> last{0};
        std::array<Cell, RunRanges> cells{};
    };
    struct alignas(64) RunFront {
        std::atomic<std::uint64_t> held{0};
        std::atomic<std::uint64_t> wideBound{0};
        std::atomic<std::uint64_t> recycled{0};
    };
    struct RunState {
        std::uint64_t key = 0;
        std::uint64_t first = 0;
        std::uint64_t last = 0;
        std::uint32_t count = 0;
        bool wide = false;
        std::array<Range, RunRanges> ranges{};
    };
    struct RecentList {
        std::uint64_t key = 0;
        std::uint64_t stamp = 0;
        std::uint32_t count = 0;
        std::array<Range, RunRanges> ranges{};
    };
    struct RunFound {
        bool hit = false;
        bool conservative = false;
        std::uint64_t stamp = 0;
        Range range{};
    };
    struct ClassBounds {
        std::array<std::uint64_t, Classes> newest{};
        std::array<std::uint64_t, Levels> deep{};
    };
    struct Gates {
        bool every = false;
        bool fineSkipped = false;
        bool coarseSkipped = false;
    };
    enum class Lost : std::uint8_t { Capacity, Disorder, Pool, Count };
    enum class Doubt : std::uint8_t { Unrecorded, Torn, Inverted, Count };
    enum class Kept : std::uint8_t { Recycled, Unrecorded, Torn, Inverted, Count };
    enum class Closed : std::uint8_t { Live, Gap, Count };
    struct alignas(64) ReadCensus {
        std::array<std::atomic<std::uint64_t>, 3> lookups{};
        std::array<std::atomic<std::uint64_t>, TestedBuckets> tested{};
        std::atomic<std::uint64_t> testedTotal{0};
        std::atomic<std::uint64_t> everySlot{0};
        std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Doubt::Count)> unsure{};
        std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Kept::Count)> kept{};
        std::array<std::atomic<std::uint64_t>, 2> runAnswers{};
        std::atomic<std::uint64_t> runConservative{0};
        std::atomic<std::uint64_t> runWithin{0};
        std::atomic<std::uint64_t> runTested{0};
        std::atomic<std::uint64_t> runFiltered{0};
        std::atomic<std::uint64_t> fineSkips{0};
        std::atomic<std::uint64_t> coarseSkips{0};
        std::atomic<std::uint64_t> oldClear{0};
    };
    struct WriteCensus {
        std::uint64_t inserts = 0;
        std::uint64_t stamps = 0;
        std::uint64_t appends = 0;
        std::array<std::uint64_t, Classes> ranges{};
        std::array<std::uint64_t, static_cast<std::size_t>(Lost::Count)> unrecorded{};
        std::uint64_t evicted = 0;
        std::uint64_t bitsSet = 0;
        std::uint64_t bitsCleared = 0;
        std::uint64_t spillHighest = 0;
        std::uint64_t runStamps = 0;
        std::uint64_t runsOpened = 0;
        std::uint64_t runsExtended = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(Closed::Count)> runsClosed{};
        std::uint64_t runsReclaimed = 0;
        std::uint64_t runsHighest = 0;
        std::uint64_t runsNoEntry = 0;
        std::uint64_t runsLongLists = 0;
        std::uint64_t runFreshSlots = 0;
        std::int64_t reportedAt = 0;
    };
    struct Window {
        std::uint64_t lo = 0;
        std::uint64_t hi = 0;
        std::uint64_t first = 0;
        std::uint64_t length = 0;
        std::uint32_t words = 0;
    };
    struct Candidates {
        std::array<std::uint64_t, Words> bits{};
        std::uint32_t words = 0;
        bool every = false;
    };
    enum class Seen : std::uint8_t { Miss, Hit, Torn };

    static Window makeWindow(std::uint64_t lo, std::uint64_t hi);
    static std::uint64_t windowBits(const Window& window, std::size_t word);
    static std::size_t blocksFor(std::size_t count);
    bool occupiedAt(std::size_t slot) const;
    void setOccupied(std::size_t slot, bool held);
    void setBit(std::size_t map, std::size_t slot, std::uint64_t stamp);
    void clearBit(std::size_t map, std::size_t slot);
    void releaseSlot(std::size_t slot, bool keepMarked);
    void evictSlot(std::size_t slot);
    void evictSkipped(std::uint64_t first, std::uint64_t length);
    std::size_t heldSkipped(std::uint64_t first, std::uint64_t length) const;
    void writeCells(Slot& slot, std::size_t from, std::span<const Range> ranges);
    void raiseNears(std::atomic<std::uint64_t> Near::* word, const Range& range, std::uint64_t stamp);
    void placeRanges(std::span<const Range> ranges, std::size_t slot, std::uint64_t stamp);
    void append(std::span<const Range> ranges, std::uint64_t stamp, std::uint64_t live);
    void appendFresh(std::span<const Range> ranges, std::uint64_t stamp, std::uint64_t live);
    void unrecord(std::uint64_t stamp, std::uint64_t live, std::size_t count, Lost reason);
    bool recordInRun(std::span<const Range> ranges, std::uint64_t key, std::uint64_t stamp, std::uint64_t live);
    std::uint32_t findOpenRun(std::span<const Range> ranges, std::uint64_t key, std::uint64_t stamp);
    std::size_t findRecent(std::span<const Range> ranges, std::uint64_t key) const;
    void noteRecent(std::span<const Range> ranges, std::uint64_t key, std::uint64_t stamp);
    bool runAvailable(std::uint64_t stamp) const;
    void noteRunStamp(std::uint64_t stamp);
    void openRun(std::uint32_t entry, std::span<const Range> ranges, std::uint64_t key, std::uint64_t stamp);
    void closeRun(std::uint32_t entry, Closed reason);
    void reclaimRuns(std::uint64_t stamp);
    void noteInsert();
    void report() const;
    ClassBounds classBounds() const;
    template<typename TVisit>
    Gates probeMaps(const Range& range, std::uint64_t lo, const ClassBounds& bounds, TVisit&& visit) const;
    void gather(std::size_t map, const Window& window, Candidates& candidates) const;
    bool collect(std::span<const Range> ranges, const Window& window, Candidates& candidates, ReadCensus* census) const;
    bool slotsMayHold(std::span<const Range> ranges, std::uint64_t lo) const;
    bool oldWindowClear(std::span<const Range> ranges, std::uint64_t from) const;
    Seen probe(std::size_t slot, std::span<const Range> asked, const Window& window, std::uint64_t& stamp, Range& range) const;
    Seen runLookup(std::span<const Range> ranges, std::uint64_t lo, std::uint64_t hi, bool any, RunFound& found, ReadCensus* census) const;
    ReadCensus* readCensus() const;
    static void noteTested(ReadCensus* census, std::uint64_t tested);

    std::unique_ptr<Meta[]> metas;
    std::unique_ptr<Map[]> maps;
    std::unique_ptr<Slot[]> slots;
    std::unique_ptr<Block[]> blocks;
    std::unique_ptr<std::uint32_t[]> freeBlocks;
    std::unique_ptr<std::uint64_t[]> marks;
    std::unique_ptr<Near[]> nears;
    std::unique_ptr<std::atomic<std::uint64_t>[]> pages;
    std::unique_ptr<RunSlot[]> runs;
    std::unique_ptr<RunState[]> runStates;
    alignas(64) std::size_t freeCount = 0;
    std::uint64_t markSerial = 0;
    std::array<std::uint64_t, Words> occupied{};
    std::uint64_t current = 0;
    std::uint32_t currentTail = NoBlock;
    bool started = false;
    bool currentRecorded = false;
    bool currentInSlot = false;
    std::uint64_t runsUsed = 0;
    std::uint64_t runsOpen = 0;
    std::uint64_t runOldest = 0;
    std::uint32_t runHint = NoRun;
    std::uint32_t recentCount = 0;
    std::array<std::uint8_t, RecentLists> recentOrder{};
    std::array<RecentList, RecentLists> recent{};
    Front front;
    ClassStamps classes;
    RunFront runFront;
    std::atomic<std::uint64_t> overflows{0};
    const char* label = "unnamed";
    WriteCensus written;
    mutable std::array<ReadCensus, CensusStripes> reads{};
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
