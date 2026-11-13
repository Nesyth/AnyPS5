#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GPUZONES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GPUZONES_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>

namespace AgcDriver::Graphics::GpuZones {

inline constexpr std::uint64_t ZoneShift = 16;
inline constexpr std::uint64_t ZoneCount = std::uint64_t{1} << 24;
inline constexpr std::uint64_t Margin = 4096;

struct Map {
    std::unique_ptr<std::atomic<std::uint64_t>[]> words{new std::atomic<std::uint64_t>[ZoneCount / 64]()};
    std::atomic<bool> everything{false};
};

inline Map& Zones() {
    static Map map;
    return map;
}

template<typename TWord>
bool EachWord(std::uint64_t firstZone, std::uint64_t lastZone, TWord&& visit) {
    auto zone = firstZone;
    for (;;) {
        const auto bit = zone & (ZoneCount - 1);
        const auto inWord = bit & 63u;
        const auto left = lastZone - zone + 1;
        const auto span = std::min<std::uint64_t>(64u - inWord, left);
        const auto mask = (span == 64u ? ~std::uint64_t{0} : (std::uint64_t{1} << span) - 1u) << inWord;
        if (!visit(static_cast<std::size_t>(bit >> 6u), mask)) return false;
        if (span == left) return true;
        zone += span;
    }
}

inline void Mark(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return;
    auto& map = Zones();
    const auto last = bytes - 1 > std::numeric_limits<std::uint64_t>::max() - address ? std::numeric_limits<std::uint64_t>::max() : address + bytes - 1;
    const auto firstZone = address >> ZoneShift;
    const auto lastZone = last >> ZoneShift;
    if (lastZone - firstZone >= ZoneCount) {
        map.everything.store(true, std::memory_order_release);
        return;
    }
    static_cast<void>(EachWord(firstZone, lastZone, [&](std::size_t index, std::uint64_t mask) {
        auto& word = map.words[index];
        if ((word.load(std::memory_order_relaxed) & mask) != mask) word.fetch_or(mask, std::memory_order_acq_rel);
        return true;
    }));
}

inline bool Untouched(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return true;
    const auto& map = Zones();
    if (map.everything.load(std::memory_order_acquire)) return false;
    const auto begin = address > Margin ? address - Margin : 0;
    const auto reach = bytes > std::numeric_limits<std::uint64_t>::max() - 2 * Margin ? std::numeric_limits<std::uint64_t>::max() : bytes + 2 * Margin;
    const auto last = reach - 1 > std::numeric_limits<std::uint64_t>::max() - begin ? std::numeric_limits<std::uint64_t>::max() : begin + reach - 1;
    const auto firstZone = begin >> ZoneShift;
    const auto lastZone = last >> ZoneShift;
    if (lastZone - firstZone >= ZoneCount) return false;
    return EachWord(firstZone, lastZone, [&](std::size_t index, std::uint64_t mask) { return (map.words[index].load(std::memory_order_acquire) & mask) == 0; });
}

}

namespace AgcDriver::Graphics::PendingZones {

inline constexpr std::uint64_t ZoneShift = 16;
inline constexpr std::uint64_t ZoneCount = std::uint64_t{1} << 24;
inline constexpr std::uint16_t Stuck = 0xffff;

static_assert(sizeof(std::atomic<std::uint16_t>) == sizeof(std::uint16_t) && std::atomic<std::uint16_t>::is_always_lock_free);

inline std::atomic<std::uint16_t>* Counts() {
    static std::atomic<std::uint16_t>* const counts = static_cast<std::atomic<std::uint16_t>*>(std::calloc(static_cast<std::size_t>(ZoneCount), sizeof(std::atomic<std::uint16_t>)));
    return counts;
}

template<typename TVisit>
void EachZone(std::uint64_t address, std::uint64_t bytes, TVisit&& visit) {
    auto* const counts = Counts();
    if (bytes == 0 || counts == nullptr) return;
    const auto last = bytes - 1 > std::numeric_limits<std::uint64_t>::max() - address ? std::numeric_limits<std::uint64_t>::max() : address + bytes - 1;
    const auto firstZone = address >> ZoneShift;
    const auto zones = std::min<std::uint64_t>((last >> ZoneShift) - firstZone + 1, ZoneCount);
    for (std::uint64_t index = 0; index < zones; ++index) visit(counts[(firstZone + index) & (ZoneCount - 1)]);
}

inline void Add(std::uint64_t address, std::uint64_t bytes) {
    EachZone(address, bytes, [](std::atomic<std::uint16_t>& count) {
        auto value = count.load(std::memory_order_relaxed);
        while (value != Stuck && !count.compare_exchange_weak(value, static_cast<std::uint16_t>(value + 1), std::memory_order_relaxed)) {}
    });
}

inline void Remove(std::uint64_t address, std::uint64_t bytes) {
    EachZone(address, bytes, [](std::atomic<std::uint16_t>& count) {
        auto value = count.load(std::memory_order_relaxed);
        while (value != Stuck && value != 0 && !count.compare_exchange_weak(value, static_cast<std::uint16_t>(value - 1), std::memory_order_relaxed)) {}
    });
}

inline bool Clear(std::uint64_t address, std::uint64_t bytes) {
    if (Counts() == nullptr) return false;
    bool clear = true;
    EachZone(address, bytes, [&](std::atomic<std::uint16_t>& count) { clear = clear && count.load(std::memory_order_relaxed) == 0; });
    return clear;
}

}

#endif
