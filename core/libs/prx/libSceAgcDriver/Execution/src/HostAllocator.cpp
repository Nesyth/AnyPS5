#ifdef _WIN32

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
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
};

constexpr std::uint16_t KeptLists = 8;
constexpr std::uint16_t KeptLargeLists = 2;

std::atomic<bool (*)()> largeCacheQuery{nullptr};

bool LargeCache() noexcept {
    const auto query = largeCacheQuery.load(std::memory_order_acquire);
    return query != nullptr && query();
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
                cache->heads[sizeClass] = *static_cast<void**>(block);
                --cache->counts[sizeClass];
                return block;
            }
        }
        return std::malloc(blockBytes);
    }
    return std::malloc(bytes);
}

void* Allocate(std::size_t bytes) {
    for (;;) {
        if (void* block = TryAllocate(bytes)) return block;
        const auto handler = std::get_new_handler();
        if (handler == nullptr) throw std::bad_alloc();
        handler();
    }
}

void Release(void* block, std::size_t bytes = 0) noexcept {
    if (block == nullptr) return;
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

}

void* operator new(std::size_t bytes) { return Allocate(bytes); }
void* operator new[](std::size_t bytes) { return Allocate(bytes); }
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return Allocate(bytes);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return Allocate(bytes);
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
