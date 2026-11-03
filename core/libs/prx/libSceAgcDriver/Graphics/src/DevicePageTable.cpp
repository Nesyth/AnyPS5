#include "prx/libSceAgcDriver/Graphics/include/DevicePageTable.hpp"
#include <algorithm>

namespace AgcDriver::Graphics {

DevicePageTable::Snapshot DevicePageTable::Current(const Context& context) {
    std::lock_guard lock(mutex);
    if (snapshot.ranges != nullptr && proof.device == context.device && HostImportsUnchanged(context, proof)) {
        ++counts.reuses;
        return snapshot;
    }
    auto imports = HostImportRanges(context);
    if (imports.proof.device == VK_NULL_HANDLE) return {};
    std::shared_ptr<const BdaPageTable> previous = proof.device == imports.proof.device ? snapshot.table : nullptr;
    std::shared_ptr<const BdaPageTable> table;
    if (!imports.ranges.empty()) table = BdaPageTable::Build(context, imports.ranges, std::span(&previous, previous != nullptr ? 1u : 0u));
    counts.ranges = imports.ranges.size();
    ++counts.builds;
    snapshot = {std::move(table), std::make_shared<const std::vector<Range>>(std::move(imports.ranges))};
    proof = imports.proof;
    return snapshot;
}

void DevicePageTable::Release(VkDevice device) {
    std::lock_guard lock(mutex);
    if (proof.device != device) return;
    snapshot = {};
    proof = {};
}

DevicePageTable::Counts DevicePageTable::Totals() const {
    std::lock_guard lock(mutex);
    return counts;
}

DevicePageTable::Cover DevicePageTable::Compare(std::span<const Range> table, std::span<const Range> build) {
    for (const auto& range : build) {
        if (range.end <= range.begin) continue;
        auto cursor = range.begin;
        auto at = std::upper_bound(table.begin(), table.end(), cursor, [](std::uint64_t value, const Range& entry) { return value < entry.end; });
        while (cursor < range.end) {
            if (at == table.end() || at->begin > cursor) return Cover::Missing;
            if (at->deviceAddress + (cursor - at->begin) != range.deviceAddress + (cursor - range.begin)) return Cover::OtherAddress;
            if ((at->permissions & range.permissions) != range.permissions) return Cover::Narrower;
            cursor = std::min(range.end, at->end);
            ++at;
        }
    }
    return Cover::Covered;
}

DevicePageTable& GuestPageTable() {
    static auto* table = new DevicePageTable;
    return *table;
}

void ReleaseGuestPageTable(VkDevice device) {
    GuestPageTable().Release(device);
}

}
