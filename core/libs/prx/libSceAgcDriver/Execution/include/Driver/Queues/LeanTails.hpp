#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_QUEUES_LEANTAILS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_QUEUES_LEANTAILS_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/RegisterLists.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/Submission.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

using GuestRanges = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

struct KeptTail {
    const std::uint32_t* control = nullptr;
    std::size_t words = 0;
    std::uint64_t tracked = 0;
    std::uint64_t forgets = 0;
    std::uint64_t stamp = 0;
    Submission copy{};
    GuestRanges sources;
};

class RegisterListCache {
public:
    static constexpr std::uint32_t PairsMode = 0x80000000u;
    static constexpr std::uint32_t MaxPairs = 256;
    static constexpr std::size_t Sets = 2048;
    static constexpr std::size_t Ways = 4;
    static constexpr std::size_t BlockSlots = 4096;
    static constexpr std::uint32_t VolatileDeclines = 2;
    static constexpr std::uint64_t DeclineWindow = 4096;
    static constexpr std::size_t WideSets = 4096;
    static constexpr std::size_t WideWays = 8;
    static constexpr std::size_t GhostSlots = 16384;

    struct Entry {
        std::uint64_t address = 0;
        std::uint32_t mode = 0;
        std::uint32_t count = 0;
        std::uint64_t tracked = 0;
        std::uint64_t epoch = 0;
        std::uint64_t used = 0;
        bool live = false;
        bool admitted = false;
        std::vector<std::uint32_t> pairs;

        bool Is(std::uint64_t at, std::uint32_t listMode, std::uint32_t listCount) const { return live && address == at && mode == listMode && count == listCount; }
    };

    static bool Cacheable(std::uint32_t count) { return count != 0 && count <= MaxPairs; }
    Entry* Find(std::uint64_t address, std::uint32_t mode, std::uint32_t count);
    Entry& Place(std::uint64_t address, std::uint32_t mode, std::uint32_t count, bool& evicted);
    bool Volatile(std::uint64_t address) const;
    void Decline(std::uint64_t address, std::uint32_t weight = 1);
    void Tick();
    void Clear();
    std::uint64_t Clock() const { return clock; }
    bool Wide() const { return wide; }
    void Widen();
    bool Evicted(std::uint64_t address, std::uint32_t mode, std::uint32_t count) const;
    bool Refused(std::uint64_t address) const { return !refused.empty() && std::find(refused.begin(), refused.end(), address) != refused.end(); }
    void Refuse(std::uint64_t address);

    std::uint64_t stamp = 0;

private:
    struct Block {
        std::uint64_t block = ~std::uint64_t{0};
        std::uint32_t declines = 0;
    };
    static std::uint64_t mix(std::uint64_t address, std::uint32_t mode, std::uint32_t count);
    std::size_t setOf(std::uint64_t address, std::uint32_t mode, std::uint32_t count) const;
    std::vector<Entry> entries;
    std::vector<Block> blocks;
    std::vector<std::uint64_t> ghosts;
    std::vector<std::uint64_t> refused;
    std::uint64_t clock = 0;
    std::uint64_t declinesSince = 0;
    std::size_t sets = Sets;
    std::size_t ways = Ways;
    unsigned shift = 53;
    bool wide = false;
};

struct PendingListHit {
    std::uint32_t ref = 0;
    std::uint32_t cursor = 0;
    std::uint64_t address = 0;
    std::uint32_t mode = 0;
    std::uint32_t count = 0;
    std::uint64_t tracked = 0;
    RegisterListCache::Entry* entry = nullptr;
};

struct BatchedList {
    std::uint32_t cursor = 0;
    std::uint32_t query = 0;
    RegisterListCache::Entry* entry = nullptr;
    bool lean = false;
    bool due = false;
    bool looked = false;
    bool bypass = false;
};

class LeanTails {
public:
    KeptTail& Keep();
    void Unkeep();
    KeptTail* Front(const std::uint32_t* control);
    bool Kept() const { return keptFront != keptEnd; }
    void Pop();
    void ClearKept();
    void TrimKept();
    std::vector<std::uint32_t> TakeCommands();
    void GiveCommands(std::vector<std::uint32_t>&& commands);
    RegisterLists TakeLists();
    void GiveLists(RegisterLists&& lists);

    RegisterListCache cache;
    std::vector<PendingListHit> hits;
    std::vector<GuestMemory::UnchangedQuery> queries;
    std::vector<std::uint32_t> scratch;
    std::vector<BatchedList> plan;
    std::vector<GuestMemory::CollectRange> collects;
    std::vector<PendingListHit> served;
    std::vector<std::uint8_t> servedBatched;
    std::uint64_t verifyOrdinal = 0;
    GuestRanges spans;
    std::vector<std::uint8_t> spanIdle;
    std::uint64_t hookVerifyOrdinal = 0;

private:
    std::vector<KeptTail> kept;
    std::size_t keptFront = 0;
    std::size_t keptEnd = 0;
    std::vector<std::vector<std::uint32_t>> spareCommands;
    std::vector<RegisterLists> spareLists;
    std::size_t spareBytes = 0;
};

}

#endif
