#include "prx/libSceAgcDriver/Execution/include/DriverHostCensus.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

namespace AgcDriver {
namespace {

constexpr std::size_t Scopes = 5;
constexpr std::size_t Sites = 512;
constexpr std::size_t SampleBytes = std::size_t{256} << 10;
constexpr std::uint16_t NoSite = 0xffff;
constexpr std::uintptr_t NoFrame = 1;
constexpr int Printed = 12;
constexpr std::uint64_t ReportEveryMs = 30000;

struct Header {
    std::uint64_t bytes;
    std::uint32_t offset;
    std::uint16_t scope;
    std::uint16_t site;
};

struct ScopeCount {
    std::atomic<std::int64_t> bytes{0};
    std::atomic<std::int64_t> blocks{0};
};

struct Site {
    std::atomic<std::uintptr_t> address{0};
    std::atomic<std::int64_t> bytes{0};
    std::atomic<std::int64_t> samples{0};
};

ScopeCount scopeCounts[Scopes];
Site sites[Sites];
std::atomic<std::uint64_t> allocations{0};
std::atomic<std::uint64_t> frees{0};
std::atomic<std::uint64_t> sampleClock{0};
std::atomic<std::uint64_t> reportAt{0};
std::atomic<std::int64_t> beyondBytes{0};

std::int64_t Weight(std::uint64_t bytes) noexcept {
    return static_cast<std::int64_t>(bytes > SampleBytes ? bytes : SampleBytes);
}

std::uint64_t NowMs() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

#ifdef _WIN32
struct Section {
    std::uintptr_t low = 0;
    std::uintptr_t end = 0;
};

Section OwnSection() noexcept {
    static const char anchor = 0;
    Section section;
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, &anchor, &module) || module == nullptr) return section;
    const auto base = reinterpret_cast<std::uintptr_t>(module);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* headers = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + static_cast<std::uintptr_t>(dos->e_lfanew));
    const auto* table = IMAGE_FIRST_SECTION(headers);
    for (unsigned index = 0; index < headers->FileHeader.NumberOfSections && section.end == 0; ++index) {
        if ((table[index].Characteristics & IMAGE_SCN_MEM_EXECUTE) == 0) continue;
        section.low = base + table[index].VirtualAddress;
        section.end = section.low + table[index].Misc.VirtualSize;
    }
    return section;
}

std::uintptr_t CallerInOwnModule() noexcept {
    static const Section section = OwnSection();
    if (section.end == 0) return 0;
    void* frames[62];
    const USHORT count = RtlCaptureStackBackTrace(0, 62, frames, nullptr);
    bool left = false;
    for (USHORT index = 0; index < count; ++index) {
        const auto value = reinterpret_cast<std::uintptr_t>(frames[index]);
        if (value < section.low || value >= section.end) {
            left = true;
            continue;
        }
        if (left) return value;
    }
    return 0;
}
#else
std::uintptr_t CallerInOwnModule() noexcept {
    return 0;
}
#endif

std::uint16_t SiteOf(std::uintptr_t address) noexcept {
    auto at = static_cast<std::size_t>((static_cast<std::uint64_t>(address) * 0x9e3779b97f4a7c15ull) >> 55);
    for (std::size_t probes = 0; probes < Sites; ++probes) {
        auto held = sites[at].address.load(std::memory_order_acquire);
        if (held == 0 && sites[at].address.compare_exchange_strong(held, address, std::memory_order_acq_rel)) return static_cast<std::uint16_t>(at);
        if (held == address) return static_cast<std::uint16_t>(at);
        at = (at + 1) & (Sites - 1);
    }
    return NoSite;
}

void Report() noexcept {
    static const char* const names[Scopes] = {"command", "object", "cache", "device", "instance"};
    std::int64_t total = 0;
    std::int64_t blocks = 0;
    for (const auto& scope : scopeCounts) {
        total += scope.bytes.load(std::memory_order_relaxed);
        blocks += scope.blocks.load(std::memory_order_relaxed);
    }
    char text[768];
    int length = std::snprintf(text, sizeof(text), "[driver-host] %lld MiB live in %lld blocks of the card driver's host allocations (%llu allocations and %llu frees so far, %lld MiB in sampled blocks beyond the site table); by scope:", static_cast<long long>(total >> 20), static_cast<long long>(blocks), static_cast<unsigned long long>(allocations.load(std::memory_order_relaxed)), static_cast<unsigned long long>(frees.load(std::memory_order_relaxed)), static_cast<long long>(beyondBytes.load(std::memory_order_relaxed) >> 20));
    for (std::size_t scope = 0; scope < Scopes && length > 0 && static_cast<std::size_t>(length) < sizeof(text); ++scope) length += std::snprintf(text + length, sizeof(text) - static_cast<std::size_t>(length), " %s %lld MiB (%lld)%s", names[scope], static_cast<long long>(scopeCounts[scope].bytes.load(std::memory_order_relaxed) >> 20), static_cast<long long>(scopeCounts[scope].blocks.load(std::memory_order_relaxed)), scope + 1 < Scopes ? "," : "\n");
    std::fputs(text, stderr);
    static std::int64_t shown[Sites];
    for (std::size_t at = 0; at < Sites; ++at) shown[at] = sites[at].address.load(std::memory_order_relaxed) != 0 ? sites[at].bytes.load(std::memory_order_relaxed) : 0;
    for (int rank = 0; rank < Printed; ++rank) {
        std::size_t best = Sites;
        for (std::size_t at = 0; at < Sites; ++at) {
            if (shown[at] > 0 && (best == Sites || shown[at] > shown[best])) best = at;
        }
        if (best == Sites) break;
        const auto address = sites[best].address.load(std::memory_order_relaxed);
        const auto samples = sites[best].samples.load(std::memory_order_relaxed);
        if (address == NoFrame) {
            std::snprintf(text, sizeof(text), "[driver-host] %2d. %8.1f MiB in %lld samples: no-frame-of-ours+0x0\n", rank + 1, static_cast<double>(shown[best]) / 1048576.0, static_cast<long long>(samples));
        } else {
            const char* name = "?";
            std::uintptr_t base = 0;
#ifdef _WIN32
            HMODULE module = nullptr;
            char path[MAX_PATH] = "?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCSTR>(address), &module) && module != nullptr) GetModuleFileNameA(module, path, sizeof(path));
            name = path;
            for (const char* at = path; *at != 0; ++at) {
                if (*at == '\\' || *at == '/') name = at + 1;
            }
            base = reinterpret_cast<std::uintptr_t>(module);
#endif
            std::snprintf(text, sizeof(text), "[driver-host] %2d. %8.1f MiB in %lld samples: %.64s+0x%llx\n", rank + 1, static_cast<double>(shown[best]) / 1048576.0, static_cast<long long>(samples), name, static_cast<unsigned long long>(address - base));
        }
        std::fputs(text, stderr);
        shown[best] = 0;
    }
}

void MaybeReport() noexcept {
    const std::uint64_t now = NowMs();
    auto due = reportAt.load(std::memory_order_relaxed);
    if (due == 0) reportAt.compare_exchange_strong(due, now + ReportEveryMs, std::memory_order_relaxed);
    else if (now >= due && reportAt.compare_exchange_strong(due, now + ReportEveryMs, std::memory_order_relaxed)) Report();
}

void* Take(std::size_t size, std::size_t alignment, VkSystemAllocationScope scope) noexcept {
    if (size == 0) return nullptr;
    const std::size_t align = std::max<std::size_t>(alignment, 16);
    if (size > static_cast<std::size_t>(-1) - align - sizeof(Header)) return nullptr;
    auto* raw = static_cast<unsigned char*>(std::malloc(size + align + sizeof(Header)));
    if (raw == nullptr) return nullptr;
    const auto address = (reinterpret_cast<std::uintptr_t>(raw) + sizeof(Header) + align - 1) & ~(static_cast<std::uintptr_t>(align) - 1);
    Header header{size, static_cast<std::uint32_t>(address - reinterpret_cast<std::uintptr_t>(raw)), static_cast<std::uint16_t>(std::min<std::size_t>(static_cast<std::size_t>(scope), Scopes - 1)), NoSite};
    const auto before = sampleClock.fetch_add(size, std::memory_order_relaxed);
    if (size >= SampleBytes || before / SampleBytes != (before + size) / SampleBytes) {
        const auto caller = CallerInOwnModule();
        header.site = SiteOf(caller != 0 ? caller : NoFrame);
        if (header.site != NoSite) {
            sites[header.site].bytes.fetch_add(Weight(size), std::memory_order_relaxed);
            sites[header.site].samples.fetch_add(1, std::memory_order_relaxed);
        } else {
            beyondBytes.fetch_add(Weight(size), std::memory_order_relaxed);
        }
    }
    std::memcpy(reinterpret_cast<void*>(address - sizeof(Header)), &header, sizeof(header));
    scopeCounts[header.scope].bytes.fetch_add(static_cast<std::int64_t>(size), std::memory_order_relaxed);
    scopeCounts[header.scope].blocks.fetch_add(1, std::memory_order_relaxed);
    allocations.fetch_add(1, std::memory_order_relaxed);
    MaybeReport();
    return reinterpret_cast<void*>(address);
}

Header HeaderOf(const void* memory) noexcept {
    Header header;
    std::memcpy(&header, static_cast<const unsigned char*>(memory) - sizeof(Header), sizeof(header));
    return header;
}

void Give(void* memory) noexcept {
    if (memory == nullptr) return;
    const Header header = HeaderOf(memory);
    scopeCounts[header.scope].bytes.fetch_sub(static_cast<std::int64_t>(header.bytes), std::memory_order_relaxed);
    scopeCounts[header.scope].blocks.fetch_sub(1, std::memory_order_relaxed);
    if (header.site != NoSite) {
        sites[header.site].bytes.fetch_sub(Weight(header.bytes), std::memory_order_relaxed);
        sites[header.site].samples.fetch_sub(1, std::memory_order_relaxed);
    }
    frees.fetch_add(1, std::memory_order_relaxed);
    std::free(static_cast<unsigned char*>(memory) - header.offset);
}

void* VKAPI_CALL Allocation(void*, std::size_t size, std::size_t alignment, VkSystemAllocationScope scope) {
    return Take(size, alignment, scope);
}

void* VKAPI_CALL Reallocation(void*, void* original, std::size_t size, std::size_t alignment, VkSystemAllocationScope scope) {
    if (original == nullptr) return Take(size, alignment, scope);
    if (size == 0) {
        Give(original);
        return nullptr;
    }
    const Header header = HeaderOf(original);
    void* const fresh = Take(size, alignment, scope);
    if (fresh == nullptr) return nullptr;
    std::memcpy(fresh, original, std::min<std::size_t>(size, static_cast<std::size_t>(header.bytes)));
    Give(original);
    return fresh;
}

void VKAPI_CALL Free(void*, void* memory) {
    Give(memory);
}

}

const VkAllocationCallbacks* DriverHostCensusCallbacks() noexcept {
    static const bool on = std::getenv("APS5_DRIVER_HOST_CENSUS") != nullptr;
    static const VkAllocationCallbacks callbacks{nullptr, Allocation, Reallocation, Free, nullptr, nullptr};
    return on ? &callbacks : nullptr;
}

}
