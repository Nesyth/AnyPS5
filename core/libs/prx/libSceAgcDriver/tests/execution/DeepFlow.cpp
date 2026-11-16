#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanFront.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DeepSite;
using AgcDriver::DriverDetail::DrawFinisher;
using AgcDriver::DriverDetail::DrawPipeline;
using AgcDriver::DriverDetail::PublishedKeys;
using AgcDriver::Graphics::RuntimeSwitch;
namespace LeanFront = AgcDriver::DriverDetail::LeanFront;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;
using Range = DrawFinisher::Range;
using Pending = DrawFinisher::Pending;
using DeepCount = DrawFinisher::DeepCount;

constexpr std::uint64_t Stride = 0x400000;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void SetThreads(unsigned count) {
    RuntimeSwitches::Assign(RuntimeSwitch::SecondFinisher, count >= 2);
    RuntimeSwitches::Assign(RuntimeSwitch::ThirdFinisher, count >= 3);
    RuntimeSwitches::Assign(RuntimeSwitch::FourthFinisher, count >= 4);
    RuntimeSwitches::Assign(RuntimeSwitch::FifthFinisher, count >= 5);
}

bool WaitFor(const std::function<bool()>& ready) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

void Hold(const std::atomic<bool>& open) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!open.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::microseconds(200));
}

void Pause() {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
}

Pending Held(std::vector<Range> writes, std::uint64_t key = 0, std::uint64_t shape = 0, bool inserts = true) {
    Pending held;
    held.key = key;
    held.shape = shape;
    held.inserts = inserts;
    held.unknown = false;
    held.writes = std::move(writes);
    return held;
}

std::mutex orderMutex;
std::vector<std::uint32_t> order;

void ClearOrder() {
    const std::lock_guard lock(orderMutex);
    order.clear();
}

void RunPayload(const std::byte* payload) {
    std::uint32_t value = 0;
    std::memcpy(&value, payload, sizeof(value));
    DrawFinisher::Get().AwaitTurn();
    const std::lock_guard lock(orderMutex);
    order.push_back(value);
}

void RunNothing(const std::byte*) {}

void PublishValue(std::uint32_t value) {
    auto& finisher = DrawFinisher::Get();
    auto* payload = finisher.ReserveLean();
    Require(payload != nullptr, "the finisher ring refused a job while open");
    std::memcpy(payload, &value, sizeof(value));
    finisher.PublishLean(&RunPayload, {}, 0, 0);
}

void PublishWrite(const Range& range) {
    auto& finisher = DrawFinisher::Get();
    Require(finisher.ReserveLean() != nullptr, "the finisher ring refused a job while open");
    finisher.PublishLean(&RunNothing, {&range, 1}, 0, 0);
}

void BlockFront(const std::atomic<bool>& open) {
    DrawFinisher::Get().Enqueue([&open] { Hold(open); }, Held({}));
}

void RingWidens() {
    auto& finisher = DrawFinisher::Get();
    finisher.Drain();
    ClearOrder();
    std::atomic<bool> open{false};
    finisher.Enqueue([&open] {
        Hold(open);
        DrawFinisher::Get().AwaitTurn();
        const std::lock_guard lock(orderMutex);
        order.push_back(0);
    }, Held({}));
    Require(finisher.RingStore() && finisher.DeepRing(), "the finisher ring is not deep with the deep flow set");
    const auto waitsBefore = LeanFront::Total(LeanFront::Counter::RingRoomWaits);
    for (std::uint32_t index = 1; index < 600; ++index) PublishValue(index);
    Require(LeanFront::Total(LeanFront::Counter::RingRoomWaits) == waitsBefore, "jobs behind a blocked front waited for room in the deep ring");
    Require(finisher.Issued() - finisher.Front() == 600, "the deep ring does not hold 600 jobs behind its blocked front");
    open.store(true);
    finisher.Drain();
    {
        const std::lock_guard lock(orderMutex);
        Require(order.size() == 600, "the deep ring ran " + std::to_string(order.size()) + " of 600 jobs");
        for (std::uint32_t index = 0; index < 600; ++index) Require(order[index] == index, "the deep ring ran job " + std::to_string(order[index]) + " in turn " + std::to_string(index));
    }
    ClearOrder();
    RuntimeSwitches::Assign(RuntimeSwitch::DeepQueues, false);
    std::atomic<bool> release{false};
    BlockFront(release);
    Require(!finisher.DeepRing(), "the deep ring latch stayed on at an empty ring without deep queues");
    for (std::uint32_t index = 1; index < 16; ++index) PublishValue(index);
    const auto shallowBefore = LeanFront::Total(LeanFront::Counter::RingRoomWaits);
    std::thread releaser([&release] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        release.store(true);
    });
    PublishValue(16);
    releaser.join();
    finisher.Drain();
    RuntimeSwitches::Assign(RuntimeSwitch::DeepQueues, true);
    Require(LeanFront::Total(LeanFront::Counter::RingRoomWaits) == shallowBefore + 1, "a ring without deep queues did not wait for room at 16 jobs");
    ClearOrder();
}

void WorkerLookupsPastPublishedLists() {
    auto& finisher = DrawFinisher::Get();
    finisher.Drain();
    constexpr std::uint64_t Base = 0x200000000;
    std::atomic<bool> open{false};
    BlockFront(open);
    for (std::uint64_t index = 0; index < 100; ++index) PublishWrite({Base + index * Stride, Base + index * Stride + 0x100});
    Require(finisher.DeepRing(), "the deep ring latch is off behind 100 jobs");
    for (std::uint64_t index = 0; index < 100; ++index) {
        const auto at = Base + index * Stride;
        Require(finisher.PendingOverlaps(at + 0x80, at + 0x81), "the write of pending job " + std::to_string(index) + " did not overlap");
        Require(!finisher.PendingOverlaps(at + 0x100, at + 0x200), "a range right after pending write " + std::to_string(index) + " overlapped");
        Require(!finisher.PendingOverlaps(at - 0x100, at), "a range right before pending write " + std::to_string(index) + " overlapped");
    }
    const std::array<Range, 3> gaps{{{Base + 0x1000, Base + 0x2000}, {Base + Stride + 0x200, Base + Stride + 0x300}, {Base + 200 * Stride, Base + 201 * Stride}}};
    Require(!finisher.PendingOverlapsAny(gaps), "a list of gaps between pending writes overlapped");
    const std::array<Range, 2> mixed{{{Base + 0x1000, Base + 0x2000}, {Base + 57 * Stride + 0xff, Base + 57 * Stride + 0x100}}};
    Require(finisher.PendingOverlapsAny(mixed), "the second range of a list over a pending write was missed");
    finisher.Enqueue([] {});
    Require(finisher.PendingOverlaps(0x10, 0x20), "a pending job with unknown writes did not overlap everything");
    Require(finisher.PendingOverlapsAny(gaps), "a pending job with unknown writes did not overlap a list of gaps");
    open.store(true);
    finisher.Drain();
    Require(!finisher.PendingOverlaps(Base, Base + 0x100), "a finished job's write still overlapped");
}

void AheadWaitsForTheWriterOnly() {
    SetThreads(5);
    auto& finisher = DrawFinisher::Get();
    finisher.Drain();
    const Range written{0x300000000, 0x300000100};
    const Range later{0x310000000, 0x310000100};
    std::atomic<bool> writerOpen{false};
    std::atomic<bool> othersOpen{false};
    std::atomic<bool> writerDone{false};
    std::atomic<int> othersDone{0};
    std::atomic<bool> readerDone{false};
    std::atomic<bool> readerSawWriter{false};
    std::atomic<bool> laterReaderDone{false};
    finisher.Enqueue([&] {
        Hold(writerOpen);
        writerDone.store(true);
    }, Held({written}));
    for (int index = 0; index < 2; ++index) {
        finisher.Enqueue([&] {
            Hold(othersOpen);
            othersDone.fetch_add(1);
        }, Held({}));
    }
    finisher.Enqueue([&] {
        const DrawFinisher::AheadScope ahead;
        static_cast<void>(DrawFinisher::Get().PendingOverlaps(written.first + 0x10, written.first + 0x20));
        readerSawWriter.store(writerDone.load());
        readerDone.store(true);
    }, Held({}));
    finisher.Enqueue([&] {
        const DrawFinisher::AheadScope ahead;
        static_cast<void>(DrawFinisher::Get().PendingOverlaps(later.first, later.second));
        laterReaderDone.store(true);
    }, Held({}));
    finisher.Enqueue([] {}, Held({later}));
    Require(WaitFor([&] { return laterReaderDone.load(); }), "a job ahead of its turn waited for a later job's write");
    Pause();
    Require(!readerDone.load(), "a job ahead of its turn read a range an earlier job writes before that job finished");
    writerOpen.store(true);
    Require(WaitFor([&] { return readerDone.load(); }), "a job ahead of its turn did not go on once the earlier writer of its read finished");
    Require(readerSawWriter.load(), "a job ahead of its turn went on before the earlier writer of its read finished");
    Require(othersDone.load() == 0, "a job ahead of its turn waited for the unrelated jobs before it");
    othersOpen.store(true);
    finisher.Drain();
}

void UnknownJobs() {
    SetThreads(5);
    auto& finisher = DrawFinisher::Get();
    finisher.Drain();
    const Range read{0x320000000, 0x320000010};
    std::atomic<bool> unknownOpen{false};
    std::atomic<bool> unknownDone{false};
    std::atomic<bool> otherOpen{false};
    std::atomic<bool> otherDone{false};
    std::atomic<bool> readerDone{false};
    std::atomic<bool> readerSawUnknown{false};
    finisher.Enqueue([&] {
        Hold(unknownOpen);
        unknownDone.store(true);
    });
    finisher.Enqueue([&] {
        Hold(otherOpen);
        otherDone.store(true);
    }, Held({}));
    finisher.Enqueue([&] {
        const DrawFinisher::AheadScope ahead;
        static_cast<void>(DrawFinisher::Get().PendingOverlaps(read.first, read.second));
        readerSawUnknown.store(unknownDone.load());
        readerDone.store(true);
    }, Held({}));
    Pause();
    Require(!readerDone.load(), "a job ahead of its turn went on past an earlier job with unknown writes");
    unknownOpen.store(true);
    Require(WaitFor([&] { return readerDone.load(); }), "a job ahead of its turn did not go on once the earlier job with unknown writes finished");
    Require(readerSawUnknown.load() && !otherDone.load(), "a job ahead of its turn waited for more than the earlier job with unknown writes");
    otherOpen.store(true);
    finisher.Drain();
    std::atomic<bool> frontOpen{false};
    std::atomic<bool> laterDone{false};
    BlockFront(frontOpen);
    finisher.Enqueue([&] {
        const DrawFinisher::AheadScope ahead;
        static_cast<void>(DrawFinisher::Get().PendingOverlaps(read.first, read.second));
        laterDone.store(true);
    }, Held({}));
    finisher.Enqueue([] {});
    Require(WaitFor([&] { return laterDone.load(); }), "a job ahead of its turn waited for a later job with unknown writes");
    frontOpen.store(true);
    finisher.Drain();
}

void KeysWithManyOutstanding() {
    SetThreads(4);
    auto& finisher = DrawFinisher::Get();
    finisher.Drain();
    std::atomic<bool> open{false};
    BlockFront(open);
    finisher.Enqueue([] {}, Held({}, 501, 0, false));
    finisher.Enqueue([] {}, Held({}, 502, 77, true));
    for (std::uint64_t index = 0; index < 200; ++index) finisher.Enqueue([] {}, Held({}, 1000 + index % 50, 0, index % 2 == 0));
    std::atomic<int> answers{-1};
    finisher.Enqueue([&answers] {
        auto& self = DrawFinisher::Get();
        const auto held = [&](std::uint64_t key, std::uint64_t shape, bool insertsOnly) { return self.KeyBefore(key, shape, insertsOnly) != PublishedKeys::Answer::Clear; };
        int bits = 0;
        bits |= held(501, 0, false) ? 1 : 0;
        bits |= !held(501, 0, true) ? 2 : 0;
        bits |= !held(601, 0, false) ? 4 : 0;
        bits |= held(9, 77, false) ? 8 : 0;
        bits |= held(502, 0, true) ? 16 : 0;
        bits |= !held(9999, 0, false) ? 32 : 0;
        bits |= !held(1001, 0, true) ? 64 : 0;
        bits |= held(1001, 0, false) ? 128 : 0;
        bits |= held(1002, 0, true) ? 256 : 0;
        answers.store(bits);
    }, Held({}, 9998));
    finisher.Enqueue([] {}, Held({}, 601, 0, true));
    finisher.Enqueue([] {}, Held({}, 501, 0, true));
    Require(WaitFor([&] { return answers.load() != -1; }), "the job ahead of its turn never ran");
    Require(answers.load() == 511, "a job ahead of its turn found the wrong earlier keys (answers " + std::to_string(answers.load()) + ")");
    const bool precise = RuntimeSwitches::Set(RuntimeSwitch::PreciseLookupDrain);
    RuntimeSwitches::Assign(RuntimeSwitch::PreciseLookupDrain, false);
    Require(finisher.PendingKey(501, 0) && finisher.PendingKey(601, 0) && finisher.PendingKey(1, 77) && finisher.PendingKey(1001, 0), "a key held by an outstanding job was not found among 200 jobs");
    Require(!finisher.PendingKey(9999, 0), "a key no outstanding job holds was found");
    RuntimeSwitches::Assign(RuntimeSwitch::PreciseLookupDrain, true);
    Require(finisher.PendingKey(501, 0) && finisher.PendingKey(1002, 0) && finisher.PendingKey(601, 0), "a key an outstanding job inserts was not found by a precise lookup");
    Require(!finisher.PendingKey(1001, 0), "a key only jobs that insert nothing hold was found by a precise lookup");
    RuntimeSwitches::Assign(RuntimeSwitch::PreciseLookupDrain, precise);
    const auto keyed = finisher.Issued();
    finisher.Enqueue([] {}, Held({}, 701, 0, true));
    std::atomic<bool> tailOpen{false};
    std::atomic<bool> tailDone{false};
    finisher.Enqueue([&] {
        Hold(tailOpen);
        tailDone.store(true);
    }, Held({}));
    std::thread releaser([&open] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        open.store(true);
    });
    finisher.AwaitKeyed(701, 0, false);
    releaser.join();
    Require(finisher.Front() > keyed, "the worker's keyed wait returned before the job holding the key finished");
    Require(!tailDone.load(), "the worker's keyed wait drained the job after the one holding the key");
    tailOpen.store(true);
    finisher.Drain();
    Require(!finisher.PendingKey(501, 0), "a finished job's key was still found");
}

void OneRangeManyWriters() {
    SetThreads(4);
    auto& finisher = DrawFinisher::Get();
    finisher.Drain();
    const Range shared{0x330000000, 0x330000040};
    std::atomic<bool> open{false};
    std::atomic<bool> readerDone{false};
    BlockFront(open);
    for (int index = 0; index < 300; ++index) PublishWrite(shared);
    finisher.Enqueue([&readerDone, &shared] {
        const DrawFinisher::AheadScope ahead;
        static_cast<void>(DrawFinisher::Get().PendingOverlaps(shared.first, shared.second));
        readerDone.store(true);
    }, Held({}));
    Require(finisher.PendingOverlaps(shared.first + 0x20, shared.first + 0x30), "a range 300 outstanding jobs write did not overlap");
    Pause();
    Require(!readerDone.load(), "a job ahead of its turn read a range 300 earlier jobs write before they finished");
    open.store(true);
    Require(WaitFor([&] { return readerDone.load(); }), "a job ahead of its turn never went on after the writers of its read finished");
    finisher.Drain();
    const Range passed{0x340000000, 0x340000040};
    for (int index = 0; index < 300; ++index) PublishWrite(passed);
    finisher.Drain();
    std::atomic<bool> laterOpen{false};
    std::atomic<bool> laterDone{false};
    BlockFront(laterOpen);
    finisher.Enqueue([&laterDone, &passed] {
        const DrawFinisher::AheadScope ahead;
        static_cast<void>(DrawFinisher::Get().PendingOverlaps(passed.first, passed.second));
        laterDone.store(true);
    }, Held({}));
    Require(WaitFor([&] { return laterDone.load(); }), "a job ahead of its turn waited for writers that had all finished");
    Require(!finisher.PendingOverlaps(passed.first, passed.second), "a range whose writers all finished still overlapped");
    laterOpen.store(true);
    finisher.Drain();
}

void DenseWritesStayExact() {
    SetThreads(4);
    auto& finisher = DrawFinisher::Get();
    finisher.Drain();
    constexpr std::uint64_t Region = 0x350000000;
    constexpr std::uint64_t Writes = 600;
    constexpr std::uint64_t Step = 0x20;
    std::atomic<bool> open{false};
    BlockFront(open);
    const auto overflowsBefore = DrawFinisher::DeepTotal(DeepCount::WriteOverflows);
    const auto scansBefore = DrawFinisher::DeepTotal(DeepCount::WorkerFallbacks);
    const auto fullBefore = DrawFinisher::DeepTotal(DeepCount::IndexFull);
    for (std::uint64_t index = 0; index < Writes; ++index) PublishWrite({Region + index * Step, Region + index * Step + 4});
    Require(DrawFinisher::DeepTotal(DeepCount::WriteOverflows) == overflowsBefore, "a write among 600 dense writes was not recorded in the index");
    for (std::uint64_t index = 0; index < Writes; ++index) {
        const auto at = Region + index * Step;
        Require(finisher.PendingOverlaps(at + 2, at + 3), "a dense write was missed");
        Require(finisher.PendingOverlaps(at + 3, at + 0x10), "a read over the end of a dense write was missed");
        Require(!finisher.PendingOverlaps(at + 4, at + Step), "a gap between dense writes overlapped");
        Require(!finisher.PendingOverlaps(at + 4, at + 4), "an empty read at the end of a dense write overlapped");
    }
    const std::array<Range, 2> gaps{{{Region + 8, Region + Step}, {Region + 300 * Step + 4, Region + 301 * Step}}};
    Require(!finisher.PendingOverlapsAny(gaps), "a list of gaps between dense writes overlapped");
    Require(finisher.PendingOverlaps(Region, Region + Writes * Step), "a read over all dense writes was missed");
    Require(DrawFinisher::DeepTotal(DeepCount::WorkerFallbacks) == scansBefore, "the worker fell back to its scan among dense writes");
    const auto waitsBefore = LeanFront::Total(LeanFront::Counter::AheadWaits);
    std::atomic<bool> gapDone{false};
    finisher.Enqueue([&gapDone] {
        const DrawFinisher::AheadScope ahead;
        static_cast<void>(DrawFinisher::Get().PendingOverlaps(Region + 5 * Step + 8, Region + 5 * Step + 0x18));
        gapDone.store(true);
    }, Held({}));
    Require(WaitFor([&] { return gapDone.load(); }), "a job ahead of its turn waited to read a gap among dense earlier writes");
    Require(LeanFront::Total(LeanFront::Counter::AheadWaits) == waitsBefore, "a job ahead of its turn that reads a gap among dense writes counted a wait");
    Require(DrawFinisher::DeepTotal(DeepCount::IndexFull) == fullBefore, "a job ahead of its turn found the index full among dense writes");
    std::atomic<bool> readerDone{false};
    finisher.Enqueue([] {});
    finisher.Enqueue([&readerDone] {
        const DrawFinisher::AheadScope ahead;
        static_cast<void>(DrawFinisher::Get().PendingOverlaps(Region + 7 * Step + 8, Region + 7 * Step + 0x18));
        readerDone.store(true);
    }, Held({}));
    Pause();
    Require(!readerDone.load(), "a job ahead of its turn went on past an earlier job with unknown writes");
    open.store(true);
    Require(WaitFor([&] { return readerDone.load(); }), "a job ahead of its turn never got past the earlier job with unknown writes");
    finisher.Drain();
    Require(LeanFront::Total(LeanFront::Counter::AheadWaits) == waitsBefore + 1, "a job ahead of an earlier job with unknown writes did not count one wait");
    Require(DrawFinisher::DeepTotal(DeepCount::IndexFull) == fullBefore, "a wait for an earlier job with unknown writes was counted as a full index");
    Require(DrawFinisher::DeepTotal(DeepCount::WorkerFallbacks) == scansBefore, "the worker fell back to its scan among dense writes");
}

void CommitterIndexes() {
    auto& finisher = DrawFinisher::Get();
    finisher.Drain();
    auto& pipeline = DrawPipeline::Queue0();
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    constexpr std::uint64_t Base = 0x400000000;
    const auto at = [](std::uint64_t index) { return Base + index * Stride; };
    std::atomic<bool> firstOpen{false};
    std::atomic<bool> secondOpen{false};
    pipeline.Enqueue([&firstOpen] { Hold(firstOpen); }, {{Base - Stride, Base - Stride + 0x10}}, 0, {}, {}, true);
    Require(pipeline.DeepQueue(), "the committer queue latch is off with the deep flow set");
    for (std::uint64_t index = 0; index < 150; ++index) pipeline.Enqueue([] {}, {{at(index), at(index) + 0x100}}, 0, {}, {{at(index) + 0x200, at(index) + 0x300}}, true);
    pipeline.EnqueueStore([] {}, at(150), std::vector<std::byte>(8, std::byte{0x11}));
    pipeline.EnqueueStore([] {}, at(150), std::vector<std::byte>(8, std::byte{0x22}));
    pipeline.Enqueue([&secondOpen] { Hold(secondOpen); }, {{at(151), at(151) + 0x10}}, 0, {}, {}, true);
    for (std::uint64_t index = 152; index <= 300; ++index) pipeline.Enqueue([] {}, {{at(index), at(index) + 0x100}}, 0, {}, {}, true);
    pipeline.EnqueueStore([] {}, at(301), std::vector<std::byte>(8, std::byte{0x33}));
    pipeline.Enqueue([] {}, {{at(301), at(301) + 4}}, 0, {}, {}, true);
    Require(pipeline.Outstanding() == 305, "the committer queue does not hold 305 items behind its blocked front");
    Require(pipeline.Overlaps(at(10) + 0x80, 4), "a queued draw write did not overlap");
    Require(!pipeline.Overlaps(at(10) + 0x100, 0x100), "a range right after a queued draw write overlapped");
    const std::array<Range, 2> both{{{at(10) + 0x100, at(10) + 0x180}, {at(200) + 0xff, at(200) + 0x100}}};
    Require(pipeline.OverlapsAny(both), "the second range of a list over a queued write was missed");
    std::array<std::byte, 4> stored{};
    Require(pipeline.StoredBytes(at(150), stored.size(), stored) == DrawPipeline::QueuedWrite::Stored && stored[0] == std::byte{0x22}, "the newest queued store's bytes were not read back");
    Require(pipeline.StoredBytes(at(150), stored.size(), {}) == DrawPipeline::QueuedWrite::Stored, "queued stores were not reported");
    Require(pipeline.StoredBytes(at(20), stored.size(), stored) == DrawPipeline::QueuedWrite::Unknown, "a queued draw write was not unknown bytes");
    Require(pipeline.StoredBytes(at(20) + 0x100, stored.size(), stored) == DrawPipeline::QueuedWrite::None, "a range nobody writes had queued bytes");
    Require(pipeline.StoredBytes(at(301), stored.size(), stored) == DrawPipeline::QueuedWrite::Unknown, "a draw queued after a store over the same bytes did not make them unknown");
    Require(pipeline.StoredBytes(at(301), stored.size(), {}) == DrawPipeline::QueuedWrite::Unknown, "a draw queued after a store did not make the stored range unknown");
    Require(pipeline.PendingLabel(at(150), 4).value_or(0) == 0x22222222u, "the newest queued label value was not found");
    Require(!pipeline.PendingLabel(at(20), 4).has_value(), "a queued draw write answered as a label value");
    Require(!pipeline.PendingLabel(at(20) + 0x100, 4).has_value(), "a label nobody queued had a value");
    Require(DrawPipeline::HeldReads(at(30) + 0x280, 4), "a queued draw's read was not held");
    Require(!DrawPipeline::HeldReads(at(30) + 0x300, 0x100), "a range no queued item reads was held");
    const auto watch = pipeline.WatchWrites();
    Require(watch.items == 305, "the watch did not count the 305 queued items");
    firstOpen.store(true);
    Require(WaitFor([&] { return pipeline.Outstanding() == 152; }), "the committer did not stop at the second blocked item");
    Require(pipeline.WrittenSince(watch, {{at(5) + 0x10, at(5) + 0x20}}), "a write retired after the watch was missed");
    Require(pipeline.WrittenSince(watch, {{at(250), at(250) + 0x10}}), "a write still queued was missed by the watch");
    Require(!pipeline.WrittenSince(watch, {{at(5) + 0x100, at(5) + 0x180}}), "a range nobody wrote was reported written");
    Require(!DrawPipeline::HeldReads(at(30) + 0x280, 4), "a retired draw's read was still held");
    secondOpen.store(true);
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    Require(!pipeline.Overlaps(at(250), 4), "a retired draw write still overlapped");
}

void TargetedCommitterWait() {
    DrawFinisher::Get().Drain();
    auto& pipeline = DrawPipeline::Queue0();
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    const Range first{0x500000000, 0x500000100};
    const Range second{0x510000000, 0x510000100};
    std::atomic<bool> firstOpen{false};
    std::atomic<bool> secondOpen{false};
    std::atomic<bool> firstDone{false};
    std::atomic<bool> secondDone{false};
    pipeline.Enqueue([&] {
        Hold(firstOpen);
        firstDone.store(true);
    }, {first}, 0, {}, {}, true);
    pipeline.Enqueue([&] {
        Hold(secondOpen);
        secondDone.store(true);
    }, {second}, 0, {}, {}, true);
    const auto before = DrawPipeline::DrainRequestsByOpcode(0x77);
    std::thread releaser([&firstOpen] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        firstOpen.store(true);
    });
    pipeline.AwaitWriters({&first, 1}, DrawPipeline::DrainReason::Capture, 0x77, DeepSite::Decode);
    releaser.join();
    Require(firstDone.load(), "a targeted wait returned before the item writing its range retired");
    Require(!secondDone.load() && pipeline.Outstanding() == 1, "a targeted wait drained a later item");
    Require(DrawPipeline::DrainRequestsByOpcode(0x77) == before + 1, "a targeted wait did not count one drain request");
    Require(!pipeline.Overlaps(first.first, 0x100), "a retired item's write still overlapped");
    secondOpen.store(true);
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
}

void SwitchOffMidFlight() {
    SetThreads(4);
    auto& finisher = DrawFinisher::Get();
    finisher.Drain();
    constexpr std::uint64_t Base = 0x600000000;
    std::atomic<bool> open{false};
    BlockFront(open);
    Require(finisher.DeepRing(), "the deep ring latch is off with the deep flow set");
    for (std::uint64_t index = 0; index < 200; ++index) PublishWrite({Base + index * Stride, Base + index * Stride + 0x100});
    RuntimeSwitches::Assign(RuntimeSwitch::DeepFlow, false);
    Require(finisher.DeepRing(), "the deep ring latch changed with jobs outstanding");
    for (std::uint64_t index = 0; index < 200; index += 7) {
        const auto at = Base + index * Stride;
        Require(finisher.PendingOverlaps(at, at + 1), "a pending write was missed after the switch went off");
        Require(!finisher.PendingOverlaps(at + 0x100, at + 0x200), "a gap overlapped after the switch went off");
    }
    const auto waitsBefore = LeanFront::Total(LeanFront::Counter::RingRoomWaits);
    std::thread releaser([&open] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        open.store(true);
    });
    PublishWrite({Base + 200 * Stride, Base + 200 * Stride + 0x100});
    releaser.join();
    Require(LeanFront::Total(LeanFront::Counter::RingRoomWaits) == waitsBefore + 1, "switching the deep flow off did not bring back the room wait");
    finisher.Drain();
    PublishWrite({Base, Base + 0x10});
    Require(!finisher.DeepRing(), "the deep ring latch stayed on at an empty ring with the switch off");
    finisher.Drain();
    auto& pipeline = DrawPipeline::Queue0();
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    pipeline.Enqueue([] {}, {{Base, Base + 0x10}}, 0, {}, {}, true);
    Require(!pipeline.DeepQueue(), "the committer queue latch stayed on at an empty queue with the switch off");
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    RuntimeSwitches::Assign(RuntimeSwitch::DeepFlow, true);
    pipeline.Enqueue([] {}, {{Base, Base + 0x10}}, 0, {}, {}, true);
    Require(pipeline.DeepQueue(), "the committer queue latch did not come back at an empty queue");
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    finisher.Enqueue([] {}, Held({}));
    Require(finisher.DeepRing(), "the deep ring latch did not come back at an empty ring");
    finisher.Drain();
}

}

int main() {
    try {
        if (DrawPipeline::Depth() == 0 || !DrawFinisher::Enabled() || !DrawFinisher::Overlap()) return 77;
        RuntimeSwitches::Assign(RuntimeSwitch::XpFlow, false);
        RuntimeSwitches::Assign(RuntimeSwitch::WorkerHelps, false);
        RuntimeSwitches::Assign(RuntimeSwitch::DeepQueues, true);
        RuntimeSwitches::Assign(RuntimeSwitch::DeepFlow, true);
        SetThreads(4);
        auto& finisher = DrawFinisher::Get();
        finisher.Adopt(true);
        Require(finisher.RingStore(), "an idle finisher did not switch to its ring");
        Require(finisher.DeepRing(), "the finisher ring latch is off with the deep flow set");
        RingWidens();
        WorkerLookupsPastPublishedLists();
        AheadWaitsForTheWriterOnly();
        UnknownJobs();
        KeysWithManyOutstanding();
        OneRangeManyWriters();
        DenseWritesStayExact();
        CommitterIndexes();
        TargetedCommitterWait();
        SwitchOffMidFlight();
        finisher.Drain();
        DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
        finisher.Adopt(false);
        std::cout << "deep flow tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
