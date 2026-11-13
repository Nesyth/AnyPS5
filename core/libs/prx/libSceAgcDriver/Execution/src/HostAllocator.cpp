#ifdef _WIN32

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <intrin.h>
#include <malloc.h>
#include <new>
#include <windows.h>

namespace {

constexpr std::size_t Granule = 16;
constexpr std::size_t LargestCached = 1024;
constexpr std::size_t Classes = LargestCached / Granule;
constexpr std::size_t LargeClasses = 6;
constexpr std::size_t LargestLarge = std::size_t{2048} << (LargeClasses - 1);
constexpr std::size_t AllClasses = Classes + LargeClasses;

constexpr std::uint16_t Depth(std::size_t sizeClass) {
    return sizeClass >= Classes ? 4 : sizeClass < 256 / Granule ? 64 : 16;
}

struct Cache {
    void* heads[AllClasses];
    std::uint16_t counts[AllClasses];
    std::ptrdiff_t untilSample;
};

constexpr std::uint16_t KeptLists = 8;
constexpr std::uint16_t KeptLargeLists = 2;

std::atomic<bool (*)()> largeCacheQuery{nullptr};
std::atomic<bool (*)()> prefetchQuery{nullptr};

bool LargeCache() noexcept {
    const auto query = largeCacheQuery.load(std::memory_order_acquire);
    return query != nullptr && query();
}

bool PrefetchNext() noexcept {
    const auto query = prefetchQuery.load(std::memory_order_acquire);
    return query != nullptr && query();
}

void PrefetchForWrite(const void* block) noexcept {
    asm volatile("prefetchw %0" : : "m"(*static_cast<const char*>(block)));
}

std::size_t LargeClassOf(std::size_t bytes) noexcept {
    return Classes + static_cast<std::size_t>(std::bit_width(bytes - 1)) - 11;
}

struct Depot {
    SRWLOCK lock = SRWLOCK_INIT;
    void* lists = nullptr;
    std::uint16_t count = 0;
};

Depot depots[AllClasses];

bool KeepList(std::size_t sizeClass, void* list) noexcept {
    auto& depot = depots[sizeClass];
    AcquireSRWLockExclusive(&depot.lock);
    const bool kept = depot.count < (sizeClass >= Classes ? KeptLargeLists : KeptLists);
    if (kept) {
        static_cast<void**>(list)[1] = depot.lists;
        depot.lists = list;
        ++depot.count;
    }
    ReleaseSRWLockExclusive(&depot.lock);
    return kept;
}

void* TakeList(std::size_t sizeClass) noexcept {
    auto& depot = depots[sizeClass];
    AcquireSRWLockExclusive(&depot.lock);
    void* list = depot.lists;
    if (list != nullptr) {
        depot.lists = static_cast<void**>(list)[1];
        --depot.count;
    }
    ReleaseSRWLockExclusive(&depot.lock);
    return list;
}

Cache* const Closed = reinterpret_cast<Cache*>(std::uintptr_t{1});

constexpr unsigned long TlsSlotsOffset = 0x1480;
constexpr DWORD DirectSlots = 64;

constexpr DWORD Unset = TLS_OUT_OF_INDEXES;
constexpr DWORD Failed = TLS_OUT_OF_INDEXES - 1;
std::atomic<DWORD> tlsSlot{Unset};
std::atomic<DWORD> flsSlot{FLS_OUT_OF_INDEXES};

void Drain(Cache* cache) noexcept {
    for (std::size_t sizeClass = 0; sizeClass < AllClasses; ++sizeClass) {
        while (void* block = cache->heads[sizeClass]) {
            cache->heads[sizeClass] = *static_cast<void**>(block);
            std::free(block);
        }
        cache->counts[sizeClass] = 0;
    }
}

void WINAPI ThreadEnded(void* data) {
    auto* cache = static_cast<Cache*>(data);
    if (cache == nullptr || cache == Closed) return;
    const auto slot = tlsSlot.load(std::memory_order_acquire);
    if (slot < Failed) TlsSetValue(slot, Closed);
    Drain(cache);
    std::free(cache);
}

DWORD Slot() noexcept {
    auto slot = tlsSlot.load(std::memory_order_acquire);
    if (slot != Unset) return slot;
    const DWORD error = GetLastError();
    const DWORD fls = FlsAlloc(ThreadEnded);
    DWORD tls = fls == FLS_OUT_OF_INDEXES ? TLS_OUT_OF_INDEXES : TlsAlloc();
    if (tls == TLS_OUT_OF_INDEXES || tls >= Failed) {
        if (tls != TLS_OUT_OF_INDEXES) TlsFree(tls);
        if (fls != FLS_OUT_OF_INDEXES) FlsFree(fls);
        tls = Failed;
    }
    DWORD expected = Unset;
    if (tlsSlot.compare_exchange_strong(expected, tls, std::memory_order_acq_rel)) {
        if (tls != Failed) flsSlot.store(fls, std::memory_order_release);
        SetLastError(error);
        return tls;
    }
    if (tls != Failed) {
        TlsFree(tls);
        FlsFree(fls);
    }
    SetLastError(error);
    return expected;
}

Cache* SlotValue(DWORD slot) noexcept {
    if (slot < DirectSlots) return reinterpret_cast<Cache*>(__readgsqword(TlsSlotsOffset + slot * sizeof(void*)));
    const DWORD error = GetLastError();
    auto* value = static_cast<Cache*>(TlsGetValue(slot));
    SetLastError(error);
    return value;
}

Cache* CreateCache(DWORD slot) noexcept {
    const auto fls = flsSlot.load(std::memory_order_acquire);
    if (fls == FLS_OUT_OF_INDEXES) return nullptr;
    auto* cache = static_cast<Cache*>(std::calloc(1, sizeof(Cache)));
    if (cache == nullptr) return nullptr;
    const DWORD error = GetLastError();
    if (!FlsSetValue(fls, cache) || !TlsSetValue(slot, cache)) {
        FlsSetValue(fls, nullptr);
        std::free(cache);
        cache = nullptr;
    }
    SetLastError(error);
    return cache;
}

Cache* CurrentCache() noexcept {
    const auto slot = Slot();
    if (slot == Failed) return nullptr;
    auto* cache = SlotValue(slot);
    if (cache == Closed) return nullptr;
    return cache != nullptr ? cache : CreateCache(slot);
}

constexpr std::size_t CensusInterval = std::size_t{256} << 10;
constexpr std::size_t CensusBuckets = std::size_t{1} << 16;
constexpr std::size_t CensusWays = 4;
constexpr std::size_t CensusSites = std::size_t{1} << 12;
constexpr int CensusPrinted = 60;

struct CensusEntry {
    void* site;
    std::size_t bytes;
};

struct CensusSite {
    void* site;
    std::uint64_t bytes;
    std::uint64_t samples;
};

std::atomic<void*> censusKeys[CensusBuckets][CensusWays];
CensusEntry censusEntries[CensusBuckets][CensusWays];
CensusSite censusSites[CensusSites];
SRWLOCK censusLock = SRWLOCK_INIT;
SRWLOCK censusReportLock = SRWLOCK_INIT;
std::atomic<std::uint64_t> censusDropped{0};
std::atomic<std::uint64_t> censusReportAt{0};
const bool censusOn = std::getenv("APS5_HEAP_CENSUS") != nullptr;

std::size_t CensusBucket(const void* block) noexcept {
    return static_cast<std::size_t>(((reinterpret_cast<std::uintptr_t>(block) >> 4) * 0x9e3779b97f4a7c15ull) >> 48);
}

void CensusReport() noexcept {
    AcquireSRWLockExclusive(&censusReportLock);
    for (auto& site : censusSites) site = {};
    std::uint64_t total = 0, samples = 0, beyond = 0;
    AcquireSRWLockShared(&censusLock);
    for (std::size_t bucket = 0; bucket < CensusBuckets; ++bucket) {
        for (std::size_t way = 0; way < CensusWays; ++way) {
            if (censusKeys[bucket][way].load(std::memory_order_relaxed) == nullptr) continue;
            const auto entry = censusEntries[bucket][way];
            total += entry.bytes;
            ++samples;
            auto at = static_cast<std::size_t>((reinterpret_cast<std::uintptr_t>(entry.site) * 0x9e3779b97f4a7c15ull) >> 52);
            std::size_t probes = 0;
            while (probes < CensusSites && censusSites[at].site != nullptr && censusSites[at].site != entry.site) {
                at = (at + 1) & (CensusSites - 1);
                ++probes;
            }
            if (probes == CensusSites) {
                beyond += entry.bytes;
                continue;
            }
            censusSites[at].site = entry.site;
            censusSites[at].bytes += entry.bytes;
            ++censusSites[at].samples;
        }
    }
    ReleaseSRWLockShared(&censusLock);
    char text[512];
    std::snprintf(text, sizeof(text), "[heap-census] %llu MiB live in %llu samples of the AGC driver's operator new (one sample per %llu KiB allocated, larger blocks at their size; %llu samples dropped, %llu MiB in sites beyond the table)\n", static_cast<unsigned long long>(total >> 20), static_cast<unsigned long long>(samples), static_cast<unsigned long long>(CensusInterval >> 10), static_cast<unsigned long long>(censusDropped.load(std::memory_order_relaxed)), static_cast<unsigned long long>(beyond >> 20));
    std::fputs(text, stderr);
    for (int rank = 0; rank < CensusPrinted; ++rank) {
        std::size_t best = CensusSites;
        for (std::size_t at = 0; at < CensusSites; ++at) {
            if (censusSites[at].bytes != 0 && (best == CensusSites || censusSites[at].bytes > censusSites[best].bytes)) best = at;
        }
        if (best == CensusSites) break;
        HMODULE module = nullptr;
        char path[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, static_cast<LPCSTR>(censusSites[best].site), &module) && module != nullptr) GetModuleFileNameA(module, path, sizeof(path));
        const char* name = path;
        for (const char* at = path; *at != 0; ++at) {
            if (*at == '\\' || *at == '/') name = at + 1;
        }
        std::snprintf(text, sizeof(text), "[heap-census] %2d. %8.1f MiB in %llu samples: %s+0x%llx\n", rank + 1, static_cast<double>(censusSites[best].bytes) / 1048576.0, static_cast<unsigned long long>(censusSites[best].samples), name, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(censusSites[best].site) - reinterpret_cast<std::uintptr_t>(module)));
        std::fputs(text, stderr);
        censusSites[best].bytes = 0;
    }
    ReleaseSRWLockExclusive(&censusReportLock);
}

void CensusAdd(void* block, std::size_t bytes, void* site) noexcept {
    const auto bucket = CensusBucket(block);
    bool placed = false;
    AcquireSRWLockExclusive(&censusLock);
    for (std::size_t way = 0; way < CensusWays && !placed; ++way) {
        if (censusKeys[bucket][way].load(std::memory_order_relaxed) != nullptr) continue;
        censusEntries[bucket][way] = {site, bytes > CensusInterval ? bytes : CensusInterval};
        censusKeys[bucket][way].store(block, std::memory_order_release);
        placed = true;
    }
    ReleaseSRWLockExclusive(&censusLock);
    if (!placed) censusDropped.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t now = GetTickCount64();
    auto due = censusReportAt.load(std::memory_order_relaxed);
    if (due == 0) censusReportAt.compare_exchange_strong(due, now + 30000, std::memory_order_relaxed);
    else if (now >= due && censusReportAt.compare_exchange_strong(due, now + 30000, std::memory_order_relaxed)) CensusReport();
}

void CensusRemove(void* block) noexcept {
    const auto bucket = CensusBucket(block);
    for (std::size_t way = 0; way < CensusWays; ++way) {
        if (censusKeys[bucket][way].load(std::memory_order_relaxed) != block) continue;
        AcquireSRWLockExclusive(&censusLock);
        if (censusKeys[bucket][way].load(std::memory_order_relaxed) == block) censusKeys[bucket][way].store(nullptr, std::memory_order_release);
        ReleaseSRWLockExclusive(&censusLock);
        return;
    }
}

void CensusNote(void* block, std::size_t bytes, void* site) noexcept {
    auto* cache = CurrentCache();
    if (cache == nullptr) return;
    cache->untilSample -= static_cast<std::ptrdiff_t>(bytes);
    if (cache->untilSample > 0) return;
    cache->untilSample = static_cast<std::ptrdiff_t>(CensusInterval);
    CensusAdd(block, bytes, site);
}

void* TryAllocate(std::size_t bytes) noexcept {
    if (bytes == 0) bytes = 1;
    const bool large = bytes > LargestCached && bytes <= LargestLarge && LargeCache();
    if (bytes <= LargestCached || large) {
        const auto sizeClass = large ? LargeClassOf(bytes) : (bytes - 1) / Granule;
        const auto blockBytes = large ? std::size_t{2048} << (sizeClass - Classes) : (sizeClass + 1) * Granule;
        if (auto* cache = CurrentCache()) {
            if (cache->heads[sizeClass] == nullptr) {
                if (void* list = TakeList(sizeClass)) {
                    cache->heads[sizeClass] = list;
                    cache->counts[sizeClass] = Depth(sizeClass);
                }
            }
            if (void* block = cache->heads[sizeClass]) {
                void* next = *static_cast<void**>(block);
                cache->heads[sizeClass] = next;
                --cache->counts[sizeClass];
                if (next != nullptr && PrefetchNext()) PrefetchForWrite(next);
                return block;
            }
        }
        return std::malloc(blockBytes);
    }
    return std::malloc(bytes);
}

void* Allocate(std::size_t bytes, void* site) {
    for (;;) {
        if (void* block = TryAllocate(bytes)) {
            if (censusOn) CensusNote(block, bytes, site);
            return block;
        }
        const auto handler = std::get_new_handler();
        if (handler == nullptr) throw std::bad_alloc();
        handler();
    }
}

void Release(void* block, std::size_t bytes = 0) noexcept {
    if (block == nullptr) return;
    if (censusOn) CensusRemove(block);
    if (auto* cache = CurrentCache()) {
        const bool large = LargeCache();
        const auto usable = bytes != 0 && bytes % Granule == 0 && (bytes <= LargestCached || !large) ? bytes : _msize(block);
        const bool largeBlock = large && usable > LargestCached && usable <= LargestLarge && std::has_single_bit(usable);
        if ((usable >= Granule && usable < LargestCached + Granule) || largeBlock) {
            const auto sizeClass = largeBlock ? LargeClassOf(usable) : usable / Granule - 1;
            if (cache->counts[sizeClass] == Depth(sizeClass) && KeepList(sizeClass, cache->heads[sizeClass])) {
                cache->heads[sizeClass] = nullptr;
                cache->counts[sizeClass] = 0;
            }
            if (cache->counts[sizeClass] < Depth(sizeClass)) {
                *static_cast<void**>(block) = cache->heads[sizeClass];
                cache->heads[sizeClass] = block;
                ++cache->counts[sizeClass];
                return;
            }
        }
    }
    std::free(block);
}

}

namespace AgcDriver {

void SetHostLargeCacheQuery(bool (*query)()) noexcept {
    largeCacheQuery.store(query, std::memory_order_release);
}

void SetHostPrefetchQuery(bool (*query)()) noexcept {
    prefetchQuery.store(query, std::memory_order_release);
}

}

void* operator new(std::size_t bytes) { return Allocate(bytes, __builtin_return_address(0)); }
void* operator new[](std::size_t bytes) { return Allocate(bytes, __builtin_return_address(0)); }
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return Allocate(bytes, __builtin_return_address(0));
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return Allocate(bytes, __builtin_return_address(0));
    } catch (...) {
        return nullptr;
    }
}
void operator delete(void* block) noexcept { Release(block); }
void operator delete[](void* block) noexcept { Release(block); }
void operator delete(void* block, std::size_t bytes) noexcept { Release(block, bytes); }
void operator delete[](void* block, std::size_t bytes) noexcept { Release(block, bytes); }
void operator delete(void* block, const std::nothrow_t&) noexcept { Release(block); }
void operator delete[](void* block, const std::nothrow_t&) noexcept { Release(block); }

#endif
