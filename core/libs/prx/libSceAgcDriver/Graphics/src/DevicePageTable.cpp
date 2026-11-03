#include "prx/libSceAgcDriver/Graphics/include/DevicePageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <algorithm>
#include <cstring>

namespace AgcDriver::Graphics {

DevicePageTable::Snapshot DevicePageTable::Current(const Context& context) {
    std::lock_guard lock(mutex);
    if (snapshot.ranges != nullptr && proof.device == context.device && HostImportsUnchanged(context, proof)) {
        ++counts.reuses;
        return snapshot;
    }
    const auto now = std::chrono::steady_clock::now();
    if (snapshot.ranges != nullptr && proof.device == context.device && now - built < std::chrono::milliseconds(250)) {
        ++counts.deferred;
        return {};
    }
    auto imports = HostImportRanges(context);
    if (imports.proof.device == VK_NULL_HANDLE) return {};
    built = now;
    std::shared_ptr<const BdaPageTable> previous = proof.device == imports.proof.device ? snapshot.table : nullptr;
    std::shared_ptr<const BdaPageTable> table;
    if (!imports.ranges.empty()) table = BdaPageTable::Build(context, imports.ranges, std::span(&previous, previous != nullptr ? 1u : 0u));
    counts.ranges = imports.ranges.size();
    ++counts.builds;
    snapshot = finish(context, std::move(table), std::move(imports.ranges));
    proof = imports.proof;
    return snapshot;
}

DevicePageTable::Snapshot DevicePageTable::finish(const Context& context, std::shared_ptr<const BdaPageTable> table, std::vector<Range> ranges) {
    std::shared_ptr<Buffer> header;
    if (table != nullptr) {
        if (table->NeedsUploads(0)) {
            CommandBatch batch(context);
            table->RecordUploads(context, batch.Handle(), 0);
            const VkMemoryBarrier uploaded{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
            context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(batch.Handle(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &uploaded, 0, nullptr, 0, nullptr);
            batch.SubmitAndWait();
            ++counts.uploads;
        }
        const ShaderRecompiler::BdaAbi::PageTableHeader words{{ShaderRecompiler::BdaAbi::PageTableVersion, 0, sizeof(ShaderRecompiler::BdaAbi::PageEntry), 0}, table->Address()};
        header = std::make_shared<Buffer>(context, sizeof(words), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        std::memcpy(header->Bytes().data(), &words, sizeof(words));
    }
    return {std::move(table), std::make_shared<const std::vector<Range>>(std::move(ranges)), std::move(header)};
}

DevicePageTable::Snapshot DevicePageTable::ForRanges(const Context& context, std::uint64_t serial, std::uint64_t guard, std::span<const Range> ranges) {
    std::lock_guard lock(mutex);
    const auto found = std::find_if(spaces.begin(), spaces.end(), [&](const Space& space) { return space.device == context.device && space.serial == serial && space.guard == guard; });
    if (found != spaces.end()) {
        ++counts.spaceReuses;
        if (found != spaces.begin()) std::rotate(spaces.begin(), found, std::next(found));
        return spaces.front().snapshot;
    }
    if (ranges.empty()) return {};
    std::shared_ptr<const BdaPageTable> previous;
    if (!spaces.empty() && spaces.front().device == context.device) previous = spaces.front().snapshot.table;
    Snapshot built;
    try {
        built = finish(context, BdaPageTable::Build(context, ranges, std::span(&previous, previous != nullptr ? 1u : 0u)), std::vector<Range>(ranges.begin(), ranges.end()));
    } catch (const std::exception&) {
        ++counts.refused;
        return {};
    }
    ++counts.spaceBuilds;
    spaces.insert(spaces.begin(), Space{context.device, serial, guard, built});
    constexpr std::size_t keptSpaces = 4;
    if (spaces.size() > keptSpaces) spaces.resize(keptSpaces);
    return built;
}

void DevicePageTable::Release(VkDevice device) {
    std::lock_guard lock(mutex);
    std::erase_if(spaces, [&](const Space& space) { return space.device == device; });
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
