#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <array>
#include <iostream>
#include <semaphore>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawFinisher;
using AgcDriver::DriverDetail::DrawPipeline;
using AgcDriver::DriverDetail::PublishedKeys;
using AgcDriver::DriverDetail::PublishedRanges;
using AgcDriver::Graphics::RuntimeSwitch;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;
using Range = PublishedRanges::Range;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

PublishedRanges::Answer Ask(const PublishedRanges& table, std::uint64_t begin, std::uint64_t end) {
    const Range range{begin, end};
    return table.Overlaps({&range, 1});
}

void PublishedRangesAnswer() {
    PublishedRanges table;
    Require(Ask(table, 0, ~0ull) == PublishedRanges::Answer::Clear, "an empty table reported a write");
    const std::vector<std::pair<Range, std::uint32_t>> entries{{{0x1000, 0x1100}, 2}, {{0x4000, 0x4010}, 1}};
    table.Publish(entries, 0);
    Require(Ask(table, 0x10ff, 0x1104) == PublishedRanges::Answer::Hit, "a range touching a published write was clear");
    Require(Ask(table, 0x1100, 0x1200) == PublishedRanges::Answer::Clear, "a range right after a published write overlapped");
    Require(Ask(table, 0x0f00, 0x1000) == PublishedRanges::Answer::Clear, "a range right before a published write overlapped");
    const std::array<Range, 2> both{{{0x2000, 0x3000}, {0x400f, 0x4011}}};
    Require(table.Overlaps(both) == PublishedRanges::Answer::Hit, "the second of two ranges was not checked");
    std::array<Range, PublishedRanges::Capacity> copy{};
    std::uint32_t size = 0;
    Require(table.Copy(copy, size) && size == 2 && copy[0] == Range{0x1000, 0x1100} && copy[1] == Range{0x4000, 0x4010}, "the copied table differs from the published one");
    table.Publish(entries, 1);
    Require(Ask(table, 0x9000, 0x9001) == PublishedRanges::Answer::Hit, "a held write of unknown range did not overlap everything");
    Require(!table.Copy(copy, size), "a table with unknown writes was copied");
    std::vector<std::pair<Range, std::uint32_t>> many;
    for (std::uint64_t index = 0; index <= PublishedRanges::Capacity; ++index) many.push_back({{index * 0x100, index * 0x100 + 0x10}, 1});
    table.Publish(many, 0);
    Require(Ask(table, 0x7000000, 0x7000001) == PublishedRanges::Answer::Unsure, "an overflowing table answered instead of falling back");
    Require(!table.Copy(copy, size), "an overflowing table was copied");
    table.Publish({}, 0);
    Require(Ask(table, 0, ~0ull) == PublishedRanges::Answer::Clear, "a cleared table still reported a write");
}

void PublishedKeysAnswer() {
    PublishedKeys keys;
    Require(keys.Find(7, 0, false) == PublishedKeys::Answer::Clear, "an empty key table found a key");
    keys.Push(1, 7, 70, false);
    keys.Push(2, 8, 80, true);
    Require(keys.Find(7, 0, false) == PublishedKeys::Answer::Hit, "a pending key was not found");
    Require(keys.Find(7, 0, true) == PublishedKeys::Answer::Clear, "a pending full hit was found by an inserts-only lookup");
    Require(keys.Find(9, 80, true) == PublishedKeys::Answer::Hit, "a pending shape was not found");
    Require(keys.Find(9, 0, false) == PublishedKeys::Answer::Clear, "a key that is not pending was found");
    keys.Retire(2);
    Require(keys.Find(7, 0, false) == PublishedKeys::Answer::Clear, "a retired key was still found");
    Require(keys.Find(8, 0, false) == PublishedKeys::Answer::Hit, "retiring the front lost the next key");
    for (std::uint64_t number = 3; number < 3 + PublishedKeys::Capacity - 1; ++number) keys.Push(number, 100 + number, 0, true);
    Require(keys.Find(8, 0, false) == PublishedKeys::Answer::Hit, "a full ring lost its oldest key");
    Require(keys.Find(100 + PublishedKeys::Capacity + 1, 0, false) == PublishedKeys::Answer::Hit, "a full ring lost its newest key");
    keys.Retire(3 + PublishedKeys::Capacity - 1);
    Require(keys.Find(8, 0, false) == PublishedKeys::Answer::Clear, "an emptied ring still found a key");
}

void PipelineQueriesMatch(bool lockless) {
    RuntimeSwitches::Assign(RuntimeSwitch::LocklessHazards, lockless);
    auto& pipeline = DrawPipeline::Queue0();
    std::binary_semaphore release{0};
    pipeline.Enqueue([&] { release.acquire(); }, {{0x10000, 0x10100}}, 0, {}, {}, true);
    pipeline.EnqueueStore([] {}, 0x20000, std::vector<std::byte>(8, std::byte{0x5a}));
    Require(pipeline.Overlaps(0x100f0, 0x20), "a queued draw write did not overlap");
    Require(!pipeline.Overlaps(0x10100, 0x20), "a range after a queued draw write overlapped");
    const std::array<Range, 2> ranges{{{0x30000, 0x30010}, {0x20004, 0x20008}}};
    Require(pipeline.OverlapsAny(ranges), "a queued store did not overlap a list of ranges");
    Require(pipeline.PendingLabel(0x20000, 4).value_or(0) == 0x5a5a5a5au, "a queued store's label value was lost");
    Require(!pipeline.PendingLabel(0x40000, 4).has_value(), "a label nobody queued had a value");
    std::array<std::byte, 4> stored{};
    Require(pipeline.StoredBytes(0x20000, stored.size(), stored) == DrawPipeline::QueuedWrite::Stored && stored[0] == std::byte{0x5a}, "a queued store's bytes were not read back");
    Require(pipeline.StoredBytes(0x40000, stored.size(), stored) == DrawPipeline::QueuedWrite::None, "a range nobody writes had queued bytes");
    const auto watch = pipeline.WatchWrites();
    Require(watch.items == 2, "the watch did not count both queued items");
    release.release();
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    Require(!pipeline.Overlaps(0x100f0, 0x20), "a retired draw write still overlapped");
    Require(pipeline.WrittenSince(watch, {{0x10080, 0x10090}}), "a write retired after the watch was missed");
    Require(!pipeline.WrittenSince(watch, {{0x50000, 0x50010}}), "a range nobody wrote was reported written");
    std::binary_semaphore later{0};
    const auto quiet = pipeline.WatchWrites();
    pipeline.Enqueue([&] { later.acquire(); }, {{0x60000, 0x60010}}, 0, {}, {}, true);
    pipeline.Enqueue([] {}, {{0x70000, 0x70010}}, 0, {}, {}, true);
    later.release();
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    Require(pipeline.WrittenSince(quiet, {{0x50000, 0x50010}}), "items queued after the watch retired without making the watch unsure");
}

void FinisherQueriesMatch(bool lockless) {
    RuntimeSwitches::Assign(RuntimeSwitch::LocklessHazards, lockless);
    auto& finisher = DrawFinisher::Get();
    std::binary_semaphore release{0};
    DrawFinisher::Pending held;
    held.key = 41;
    held.shape = 42;
    held.inserts = false;
    held.unknown = false;
    held.writes = {{0x80000, 0x80100}};
    finisher.Enqueue([&] { release.acquire(); }, held);
    Require(finisher.PendingOverlaps(0x800f0, 0x80110), "a pending finisher write did not overlap");
    Require(!finisher.PendingOverlaps(0x80100, 0x80110), "a range after a pending finisher write overlapped");
    const std::array<Range, 2> ranges{{{0x1, 0x2}, {0x80000, 0x80001}}};
    Require(finisher.PendingOverlapsAny(ranges), "a pending finisher write did not overlap a list of ranges");
    const bool precise = RuntimeSwitches::Set(RuntimeSwitch::PreciseLookupDrain);
    RuntimeSwitches::Assign(RuntimeSwitch::PreciseLookupDrain, false);
    Require(finisher.PendingKey(41, 0), "a pending finisher key was not found");
    Require(finisher.PendingKey(1, 42), "a pending finisher shape was not found");
    Require(!finisher.PendingKey(43, 0), "a key that is not pending was found");
    RuntimeSwitches::Assign(RuntimeSwitch::PreciseLookupDrain, true);
    Require(!finisher.PendingKey(41, 0), "a pending full hit was found by a precise lookup");
    RuntimeSwitches::Assign(RuntimeSwitch::PreciseLookupDrain, precise);
    std::uint64_t value = 0;
    Require(finisher.PendingLabel(0x90000, 4, value) == DrawFinisher::LabelLookup::NotHeld, "a label nobody holds was held");
    finisher.Enqueue([] {});
    Require(finisher.PendingOverlaps(0x1, 0x2), "a pending job with unknown writes did not overlap everything");
    release.release();
    finisher.Drain();
    Require(!finisher.PendingOverlaps(0x800f0, 0x80110), "a finished job's write still overlapped");
    Require(!finisher.PendingKey(41, 0), "a finished job's key was still found");
}

}

int main() {
    try {
        PublishedRangesAnswer();
        PublishedKeysAnswer();
        if (DrawPipeline::Depth() == 0) return 77;
        for (const bool lockless : {false, true}) {
            PipelineQueriesMatch(lockless);
            if (DrawFinisher::Overlap()) FinisherQueriesMatch(lockless);
        }
        std::cout << "lockless hazard tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
