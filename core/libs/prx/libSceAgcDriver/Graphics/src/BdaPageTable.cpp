#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>

namespace AgcDriver::Graphics {

namespace {

namespace Abi = ShaderRecompiler::BdaAbi;

constexpr VkBufferUsageFlags TableUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
constexpr VkMemoryPropertyFlags TableMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr std::uint64_t PageBytes = std::uint64_t{1} << Abi::PageShift;
constexpr std::uint64_t WindowBytes = std::uint64_t{1} << Abi::WindowShift;

Abi::PageEntry Segment(const Abi::Range& range, std::uint64_t start) {
    const auto first = range.begin > start ? static_cast<std::uint32_t>(range.begin - start) : 0u;
    return Abi::SegmentEntry(range.deviceAddress - range.begin, first, std::min(range.end - start, Abi::LimitMax), range.permissions);
}

Abi::PageEntry Encode(std::vector<Abi::PageEntry>& entries, std::span<const Abi::Range> ranges, std::uint64_t start, std::uint32_t shift, const std::vector<std::size_t>& meeting) {
    if (meeting.size() == 1u) return Segment(ranges[meeting.front()], start);
    Require(shift >= Abi::SplitShift, "two BDA page ranges meet in one byte");
    const auto block = entries.size();
    entries.resize(block + Abi::SplitChildren);
    const auto childShift = shift - Abi::SplitShift;
    for (std::uint32_t child = 0; child < Abi::SplitChildren; ++child) {
        const auto childStart = start + (static_cast<std::uint64_t>(child) << childShift);
        const auto childEnd = childStart + (std::uint64_t{1} << childShift);
        std::vector<std::size_t> inside;
        for (const auto index : meeting) {
            if (ranges[index].begin < childEnd && childStart < ranges[index].end) inside.push_back(index);
        }
        if (inside.empty()) continue;
        const auto entry = Encode(entries, ranges, childStart, childShift, inside);
        entries[block + child] = entry;
    }
    return Abi::SplitEntry(block * sizeof(Abi::PageEntry));
}

struct Totals {
    std::mutex mutex;
    BdaPageTable::Counts counts;
};

Totals& Counted() {
    static Totals totals;
    return totals;
}

}

void ValidateBdaRangeList(std::span<const Abi::Range> ranges) {
    for (std::size_t index = 1; index < ranges.size(); ++index) Require(ranges[index - 1].end <= ranges[index].begin, "BDA ranges are unsorted or overlap");
}

void ValidateBdaPageRanges(std::span<const Abi::Range> ranges) {
    ValidateBdaRangeList(ranges);
    for (const auto& range : ranges) {
        Require(range.begin < range.end && range.end <= Abi::PageTableEnd, "a BDA page range is empty or ends past the page table");
        Require(range.deviceAddress != 0 && range.end - range.begin <= std::numeric_limits<std::uint64_t>::max() - range.deviceAddress, "a BDA page range has no device address or its device range wraps");
        Require(range.reserved == 0 && (range.permissions & ~(Abi::Read | Abi::Write)) == 0, "a BDA page range has unknown permissions or reserved bits");
    }
}

std::vector<Abi::PageEntry> EncodeBdaWindow(std::uint32_t window, std::span<const Abi::Range> ranges) {
    std::vector<Abi::PageEntry> entries(Abi::WindowPages);
    const auto start = static_cast<std::uint64_t>(window) << Abi::WindowShift;
    const auto end = start + WindowBytes;
    std::vector<std::pair<std::uint64_t, std::size_t>> partial;
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        const auto& range = ranges[index];
        Require(range.begin < end && start < range.end, "a BDA page range does not meet its window");
        const auto first = (std::max(range.begin, start) - start) >> Abi::PageShift;
        const auto last = (std::min(range.end, end) - 1u - start) >> Abi::PageShift;
        for (auto page = first; page <= last; ++page) {
            const auto pageStart = start + (page << Abi::PageShift);
            if (range.begin <= pageStart && pageStart + PageBytes <= range.end) entries[page] = Segment(range, pageStart);
            else partial.emplace_back(page, index);
        }
    }
    for (std::size_t at = 0; at < partial.size();) {
        const auto page = partial[at].first;
        std::vector<std::size_t> meeting;
        for (; at < partial.size() && partial[at].first == page; ++at) meeting.push_back(partial[at].second);
        const auto entry = Encode(entries, ranges, start + (page << Abi::PageShift), Abi::PageShift, meeting);
        entries[page] = entry;
    }
    return entries;
}

std::shared_ptr<const BdaPageTable> BdaPageTable::Build(const Context& context, std::span<const Abi::Range> ranges, std::span<const std::shared_ptr<const BdaPageTable>> shares) {
    ValidateBdaPageRanges(ranges);
    std::map<std::uint32_t, std::vector<Abi::Range>> held;
    for (const auto& range : ranges) {
        const auto last = static_cast<std::uint32_t>((range.end - 1u) >> Abi::WindowShift);
        for (auto window = static_cast<std::uint32_t>(range.begin >> Abi::WindowShift); window <= last; ++window) held[window].push_back(range);
    }
    std::shared_ptr<BdaPageTable> table(new BdaPageTable());
    std::vector<std::uint64_t> addresses(Abi::WindowCount, 0u);
    std::uint64_t shared = 0;
    for (auto& [window, inside] : held) {
        std::shared_ptr<const Leaf> old;
        for (const auto& candidate : shares) {
            auto leaf = candidate != nullptr ? candidate->find(window) : nullptr;
            if (leaf != nullptr && leaf->ranges.size() == inside.size() && std::memcmp(leaf->ranges.data(), inside.data(), inside.size() * sizeof(Abi::Range)) == 0) {
                old = std::move(leaf);
                break;
            }
        }
        if (old != nullptr) {
            table->leaves.push_back(std::move(old));
            ++shared;
        } else {
            auto leaf = std::make_shared<Leaf>();
            leaf->window = window;
            const auto entries = EncodeBdaWindow(window, inside);
            leaf->entries = upload(context, entries.data(), entries.size() * sizeof(Abi::PageEntry));
            leaf->ranges = std::move(inside);
            table->leaves.push_back(std::move(leaf));
        }
        const auto address = table->leaves.back()->entries->DeviceAddress();
        Require(address != 0 && address % sizeof(Abi::PageEntry) == 0, "a BDA page table leaf is not aligned for its entries");
        addresses[window] = address;
    }
    table->windows = upload(context, addresses.data(), addresses.size() * sizeof(std::uint64_t));
    Require(table->windows->DeviceAddress() % sizeof(std::uint64_t) == 0, "the BDA page table is not aligned for its entries");
    auto& totals = Counted();
    std::lock_guard lock(totals.mutex);
    ++totals.counts.tables;
    totals.counts.sharedWindows += shared;
    totals.counts.encodedWindows += held.size() - shared;
    return table;
}

BdaPageTable::Counts BdaPageTable::Totals() {
    auto& totals = Counted();
    std::lock_guard lock(totals.mutex);
    return totals.counts;
}

VkDeviceAddress BdaPageTable::Address() const {
    return windows->DeviceAddress();
}

std::shared_ptr<Buffer> BdaPageTable::upload(const Context& context, const void* bytes, std::size_t size) {
    auto buffer = std::make_shared<Buffer>(context, size, TableUsage, TableMemory);
    std::memcpy(buffer->Bytes().data(), bytes, size);
    return buffer;
}

std::optional<Abi::Range> BdaPageTable::Find(std::uint64_t address) const {
    if (address >= Abi::PageTableEnd) return std::nullopt;
    const auto leaf = find(static_cast<std::uint32_t>(address >> Abi::WindowShift));
    if (leaf == nullptr) return std::nullopt;
    const auto found = std::upper_bound(leaf->ranges.begin(), leaf->ranges.end(), address, [](std::uint64_t value, const Abi::Range& range) { return value < range.begin; });
    if (found == leaf->ranges.begin() || address >= std::prev(found)->end) return std::nullopt;
    return *std::prev(found);
}

std::shared_ptr<const BdaPageTable::Leaf> BdaPageTable::find(std::uint32_t window) const {
    const auto found = std::lower_bound(leaves.begin(), leaves.end(), window, [](const auto& leaf, std::uint32_t value) { return leaf->window < value; });
    return found != leaves.end() && (*found)->window == window ? *found : nullptr;
}

}
