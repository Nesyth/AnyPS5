#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GUARDSPANS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GUARDSPANS_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

class GuardSpans {
public:
    std::uint64_t Generation() const { return generation; }

    template<typename TRanges>
    void Fill(std::uint64_t filled, const TRanges& ranges) {
        spans.clear();
        for (const auto& range : ranges) spans.emplace_back(range.address, range.address + range.bytes);
        std::ranges::sort(spans);
        std::uint64_t reach = 0;
        for (auto& span : spans) span.second = reach = std::max(reach, span.second);
        generation = filled;
    }

    bool Overlaps(std::uint64_t address, std::uint64_t bytes) const {
        const auto end = address + bytes;
        const auto after = std::ranges::lower_bound(spans, end, {}, &std::pair<std::uint64_t, std::uint64_t>::first);
        return after != spans.begin() && std::prev(after)->second > address;
    }

private:
    std::uint64_t generation = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
};

class GuardSpanBoard {
public:
    static constexpr std::size_t Capacity = 64;
    static constexpr std::uint32_t Overflow = static_cast<std::uint32_t>(Capacity) + 1u;
    struct View {
        std::uint64_t sequence = 1;
        std::uint32_t spans = 0;
    };

    template<typename TRanges>
    void Publish(std::uint64_t filled, const TRanges& ranges) {
        std::array<std::pair<std::uint64_t, std::uint64_t>, Capacity> sorted{};
        std::size_t count = 0;
        bool over = false;
        for (const auto& range : ranges) {
            if (count == Capacity) {
                over = true;
                break;
            }
            sorted[count++] = {range.address, range.address + range.bytes};
        }
        std::sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(count));
        const auto opened = sequence.load(std::memory_order_relaxed) + 1u;
        sequence.store(opened, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        std::uint64_t reach = 0;
        for (std::size_t index = 0; !over && index < count; ++index) {
            reach = std::max(reach, sorted[index].second);
            begins[index].store(sorted[index].first, std::memory_order_relaxed);
            reaches[index].store(reach, std::memory_order_relaxed);
        }
        generation.store(filled, std::memory_order_relaxed);
        spans.store(over ? Overflow : static_cast<std::uint32_t>(count), std::memory_order_relaxed);
        sequence.store(opened + 1u, std::memory_order_release);
    }

    bool Open(std::uint64_t expected, View& view) const {
        view.sequence = sequence.load(std::memory_order_acquire);
        if ((view.sequence & 1u) != 0 || generation.load(std::memory_order_relaxed) != expected) return false;
        view.spans = spans.load(std::memory_order_relaxed);
        return view.spans <= Capacity;
    }

    bool Overlaps(const View& view, std::uint64_t address, std::uint64_t bytes) const {
        const auto end = address + bytes;
        std::size_t low = 0;
        std::size_t high = view.spans;
        while (low < high) {
            const auto middle = low + (high - low) / 2u;
            if (begins[middle].load(std::memory_order_relaxed) < end) low = middle + 1u;
            else high = middle;
        }
        return low != 0 && reaches[low - 1u].load(std::memory_order_relaxed) > address;
    }

    bool Holds(const View& view) const {
        std::atomic_thread_fence(std::memory_order_acquire);
        return sequence.load(std::memory_order_relaxed) == view.sequence;
    }

private:
    std::atomic<std::uint64_t> sequence{0};
    std::atomic<std::uint64_t> generation{0};
    std::atomic<std::uint32_t> spans{0};
    std::array<std::atomic<std::uint64_t>, Capacity> begins{};
    std::array<std::atomic<std::uint64_t>, Capacity> reaches{};
};

struct GuardAnswer {
    static constexpr std::uint8_t DepthClear = 1;
    static constexpr std::uint8_t MultisampleClear = 2;
    static constexpr std::uint8_t InputsClear = 4;
    std::uint64_t depth = 0;
    std::uint64_t multisample = 0;
    std::uint8_t clear = 0;
    bool valid = false;
};

const GuardSpanBoard& DepthGuardBoard();
const GuardSpanBoard& MultisampleGuardBoard();

enum class GuardSpanEvent : std::size_t { Clear, Refill, Overlap, Verified, Missed, Count };

void NoteGuardSpans(GuardSpanEvent event);
bool VerifyGuardSpans();
void ReportGuardSpanMiss(const char* list, std::uint64_t address, std::uint64_t bytes);

}

#endif
