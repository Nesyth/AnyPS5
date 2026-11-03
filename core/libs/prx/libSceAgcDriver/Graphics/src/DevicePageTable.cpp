#include "prx/libSceAgcDriver/Graphics/include/DevicePageTable.hpp"
#include <algorithm>
#include <chrono>

namespace AgcDriver::Graphics {

DevicePageTable::~DevicePageTable() {
    {
        std::lock_guard lock(mutex);
        stopping = true;
        pending.reset();
    }
    wake.notify_all();
    if (worker.joinable()) worker.join();
}

DevicePageTable::Table DevicePageTable::Find(const Context& context, Key key, std::span<const Range> ranges) {
    std::lock_guard lock(mutex);
    const auto found = std::find_if(entries.begin(), entries.end(), [&](const Entry& entry) { return entry.device == context.device && entry.key == key; });
    if (found != entries.end()) {
        ++counts.served;
        if (found != entries.begin()) std::rotate(entries.begin(), found, std::next(found));
        return entries.front().table;
    }
    ++counts.missed;
    if (ranges.empty() || stopping) return {};
    const auto queued = [&](const std::optional<Job>& job) { return job.has_value() && job->context.device == context.device && job->key == key; };
    if (queued(pending) || queued(running)) return {};
    if (pending.has_value()) ++counts.superseded;
    pending.emplace(Job{context, key, std::vector<Range>(ranges.begin(), ranges.end())});
    if (!worker.joinable()) worker = std::thread([this] { work(); });
    wake.notify_one();
    return {};
}

void DevicePageTable::work() {
    std::unique_lock lock(mutex);
    while (true) {
        wake.wait(lock, [&] { return stopping || pending.has_value(); });
        if (stopping) return;
        running = std::move(pending);
        pending.reset();
        std::shared_ptr<const BdaPageTable> previous;
        const auto newest = std::find_if(entries.begin(), entries.end(), [&](const Entry& entry) { return entry.device == running->context.device; });
        if (newest != entries.end()) previous = newest->table;
        lock.unlock();
        const auto started = std::chrono::steady_clock::now();
        Table table;
        try {
            table = BdaPageTable::Build(running->context, running->ranges, std::span(&previous, previous != nullptr ? 1u : 0u));
            if (table->NeedsUploads(0)) table.reset();
        } catch (const std::exception&) {
            table.reset();
        }
        const auto elapsed = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        previous.reset();
        lock.lock();
        counts.buildNanoseconds += elapsed;
        if (table == nullptr) {
            ++counts.refused;
        } else {
            ++counts.built;
            entries.insert(entries.begin(), Entry{running->context.device, running->key, std::move(table)});
            if (entries.size() > KeptTables) entries.resize(KeptTables);
        }
        running.reset();
        idle.notify_all();
    }
}

std::vector<DevicePageTable::Range> DevicePageTable::Overlay(std::span<const Range> space, std::span<const Range> build) {
    std::vector<Range> overlay;
    const auto piece = [&](const Range& range, std::uint64_t begin, std::uint64_t end, std::uint32_t permissions) {
        overlay.push_back({begin, end, range.deviceAddress + (begin - range.begin), permissions, permissions != 0u ? range.reserved : 0u});
    };
    auto at = space.begin();
    for (const auto& range : build) {
        for (auto cursor = range.begin; cursor < range.end;) {
            while (at != space.end() && at->end <= cursor) ++at;
            if (at == space.end() || at->begin > cursor) {
                const auto stop = at == space.end() ? range.end : std::min(range.end, at->begin);
                piece(range, cursor, stop, range.permissions);
                cursor = stop;
                continue;
            }
            const auto stop = std::min(range.end, at->end);
            if (at->deviceAddress - at->begin != range.deviceAddress - range.begin || at->permissions != range.permissions || at->reserved != range.reserved) piece(range, cursor, stop, range.permissions);
            cursor = stop;
        }
    }
    auto held = build.begin();
    for (const auto& range : space) {
        for (auto cursor = range.begin; cursor < range.end;) {
            while (held != build.end() && held->end <= cursor) ++held;
            if (held != build.end() && held->begin <= cursor) {
                cursor = std::min(range.end, held->end);
                continue;
            }
            const auto stop = held == build.end() ? range.end : std::min(range.end, held->begin);
            piece(range, cursor, stop, 0u);
            cursor = stop;
        }
    }
    std::sort(overlay.begin(), overlay.end(), [](const Range& left, const Range& right) { return left.begin < right.begin; });
    return overlay;
}

void DevicePageTable::WaitIdle() {
    std::unique_lock lock(mutex);
    idle.wait(lock, [&] { return !pending.has_value() && !running.has_value(); });
}

void DevicePageTable::Release(VkDevice device) {
    std::unique_lock lock(mutex);
    if (pending.has_value() && pending->context.device == device) pending.reset();
    idle.wait(lock, [&] { return !running.has_value() || running->context.device != device; });
    std::erase_if(entries, [&](const Entry& entry) { return entry.device == device; });
}

DevicePageTable::Counts DevicePageTable::Totals() const {
    std::lock_guard lock(mutex);
    return counts;
}

DevicePageTable& GuestPageTable() {
    static auto* table = new DevicePageTable;
    return *table;
}

void ReleaseGuestPageTable(VkDevice device) {
    GuestPageTable().Release(device);
}

}
