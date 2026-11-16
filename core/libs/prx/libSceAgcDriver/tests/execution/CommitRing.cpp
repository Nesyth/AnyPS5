#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawFinisher;
using AgcDriver::DriverDetail::DrawPipeline;
using AgcDriver::Graphics::RuntimeSwitch;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;
using Range = DrawPipeline::Range;
using RingCount = DrawPipeline::RingCount;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool WaitFor(const std::function<bool()>& ready) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

void Hold(const std::atomic<bool>& open) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!open.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::microseconds(200));
}

void Spin(std::uint32_t ns) {
    if (ns == 0) return;
    const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
    while (std::chrono::steady_clock::now() < until) {
    }
}

void SetThreads(unsigned count) {
    RuntimeSwitches::Assign(RuntimeSwitch::SecondFinisher, count >= 2);
    RuntimeSwitches::Assign(RuntimeSwitch::ThirdFinisher, count >= 3);
    RuntimeSwitches::Assign(RuntimeSwitch::FourthFinisher, count >= 4);
    RuntimeSwitches::Assign(RuntimeSwitch::FifthFinisher, count >= 5);
}

DrawPipeline& Pipeline() {
    return DrawPipeline::Queue0();
}

std::mutex committedMutex;
std::vector<std::uint64_t> committed;

void Record(std::uint64_t tag) {
    const std::lock_guard lock(committedMutex);
    committed.push_back(tag);
}

void ClearCommitted() {
    const std::lock_guard lock(committedMutex);
    committed.clear();
}

std::vector<std::uint64_t> Committed() {
    const std::lock_guard lock(committedMutex);
    return committed;
}

Range TagRange(std::uint64_t tag) {
    const std::uint64_t base = 0x700000000ull + (tag % 8191) * 0x40;
    return {base, base + 0x10};
}

void EnqueueTag(std::uint64_t tag) {
    Pipeline().Enqueue([tag] { Record(tag); }, {TagRange(tag)}, 0, {}, {}, true);
}

void DrainAll() {
    DrawFinisher::Get().Drain();
    Pipeline().Drain(DrawPipeline::DrainReason::Submission);
}

void RequireInOrder(const std::string& where, std::uint64_t count) {
    const auto seen = Committed();
    Require(seen.size() == count, where + ": the committer ran " + std::to_string(seen.size()) + " of " + std::to_string(count) + " items");
    for (std::size_t index = 0; index < seen.size(); ++index) Require(seen[index] == index, where + ": commit " + std::to_string(index) + " was the item enqueued as " + std::to_string(seen[index]));
}

struct JobSpec {
    std::uint64_t first = 0;
    std::uint32_t count = 0;
    std::uint32_t spinNs = 0;
};

void RunJob(const JobSpec& spec) {
    Spin(spec.spinNs);
    for (std::uint32_t index = 0; index < spec.count; ++index) EnqueueTag(spec.first + index);
}

void RunLeanJob(const std::byte* payload) {
    JobSpec spec;
    std::memcpy(&spec, payload, sizeof(spec));
    RunJob(spec);
}

void OrderAcrossEnqueuers() {
    auto& finisher = DrawFinisher::Get();
    DrainAll();
    ClearCommitted();
    const bool deposit = RuntimeSwitches::Set(RuntimeSwitch::FinisherDeposit);
    RuntimeSwitches::Assign(RuntimeSwitch::FinisherDeposit, true);
    SetThreads(4);
    const auto depositsBefore = DrawFinisher::Deposits();
    std::mt19937 random(5);
    std::uint64_t next = 0;
    for (int round = 0; round < 4000; ++round) {
        const auto choice = random() % 10;
        if (choice < 2) {
            EnqueueTag(next++);
            continue;
        }
        JobSpec spec;
        spec.first = next;
        spec.count = static_cast<std::uint32_t>(1 + random() % 3);
        spec.spinNs = static_cast<std::uint32_t>(random() % 4 == 0 ? 20000 + random() % 60000 : random() % 3000);
        next += spec.count;
        std::vector<Range> writes;
        for (std::uint32_t index = 0; index < spec.count; ++index) writes.push_back(TagRange(spec.first + index));
        if (choice < 6) {
            auto* payload = finisher.ReserveLean();
            Require(payload != nullptr, "the finisher ring refused a job while open");
            std::memcpy(payload, &spec, sizeof(spec));
            finisher.PublishLean(&RunLeanJob, writes, 0, 0);
        } else {
            DrawFinisher::Pending held;
            held.unknown = false;
            held.writes = writes;
            finisher.Enqueue([spec] { RunJob(spec); }, std::move(held));
        }
    }
    DrainAll();
    RequireInOrder("worker items and forwarded deposits", next);
    const auto depositsAfter = DrawFinisher::Deposits();
    Require(depositsAfter.forwarded > depositsBefore.forwarded, "no deposited item was forwarded by a finisher");
    if (RuntimeSwitches::Set(RuntimeSwitch::CommitRing)) Require(Pipeline().CommitRing(), "the committer ring latch is off with the switch set");
    RuntimeSwitches::Assign(RuntimeSwitch::FinisherDeposit, deposit);
    ClearCommitted();
}

void FullQueueWaits() {
    auto& pipeline = Pipeline();
    DrainAll();
    ClearCommitted();
    std::atomic<bool> open{false};
    pipeline.Enqueue([&open] {
        Hold(open);
        Record(0);
    }, {TagRange(0)}, 0, {}, {}, true);
    const std::size_t limit = pipeline.DeepQueue() ? DrawFinisher::LeanSlots : 32;
    static constexpr std::uint64_t Items = 1500;
    const auto waitsBefore = DrawPipeline::RingTotal(RingCount::RoomWaits);
    std::atomic<std::uint64_t> sent{0};
    std::thread producer([&sent] {
        for (std::uint64_t tag = 1; tag <= Items; ++tag) {
            EnqueueTag(tag);
            sent.store(tag);
        }
    });
    const bool filled = WaitFor([&] { return pipeline.Outstanding() >= limit; });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto held = pipeline.Outstanding();
    const auto through = sent.load();
    open.store(true);
    producer.join();
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    Require(filled, "the queue never filled to its room limit of " + std::to_string(limit));
    Require(held == limit, "a full queue held " + std::to_string(held) + " items, not its room limit of " + std::to_string(limit));
    Require(through == limit - 1, "the enqueuer got " + std::to_string(through) + " items into a queue of room " + std::to_string(limit) + " behind a blocked item");
    RequireInOrder("an enqueuer that waited for room", Items + 1);
    if (pipeline.CommitRing()) Require(DrawPipeline::RingTotal(RingCount::RoomWaits) > waitsBefore, "a full ring did not count a room wait");
    ClearCommitted();
}

void ScannersSeeWholeItems() {
    auto& pipeline = Pipeline();
    DrainAll();
    static constexpr std::uint64_t Base = 0x780000000ull;
    static constexpr std::uint64_t Slots = 64;
    static constexpr std::uint64_t Items = 40000;
    const auto labelAt = [](std::uint64_t slot) { return Base + slot * 0x40; };
    const auto drawnAt = [](std::uint64_t slot) { return Base + 0x200000 + slot * 0x40; };
    const auto readAt = [](std::uint64_t slot) { return Base + 0x400000 + slot * 0x40; };
    std::atomic<bool> done{false};
    std::atomic<std::uint64_t> scans{0};
    std::mutex failureMutex;
    std::string failure;
    const auto scansBefore = DrawPipeline::RingTotal(RingCount::SlowScans);
    std::thread scanner([&] {
        std::uint64_t slot = 0;
        try {
            while (!done.load()) {
                slot = (slot + 7) % Slots;
                const auto address = labelAt(slot);
                if (const auto value = pipeline.PendingLabel(address, 8)) Require((*value & 0xffu) == slot, "a queued label at slot " + std::to_string(slot) + " held the value of slot " + std::to_string(*value & 0xffu));
                std::array<std::byte, 8> stored{};
                if (pipeline.StoredBytes(address, stored.size(), stored) == DrawPipeline::QueuedWrite::Stored) {
                    std::uint64_t value = 0;
                    std::memcpy(&value, stored.data(), sizeof(value));
                    Require((value & 0xffu) == slot, "the stored bytes at slot " + std::to_string(slot) + " were those of slot " + std::to_string(value & 0xffu));
                }
                const auto code = pipeline.DescribeQueuedWriter(address, address + 8);
                Require(code == 6000u || code == 4000u + static_cast<std::uint32_t>(DrawPipeline::CommitKind::Store), "the queued writer of a label was described as " + std::to_string(code));
                Require(DrawPipeline::QueuedItems() <= DrawFinisher::LeanSlots, "the queue held more items than its room");
                static_cast<void>(DrawPipeline::HeldReads(readAt(slot), 4));
                const std::array<Range, 1> asked{{{address, address + 8}}};
                static_cast<void>(pipeline.QueuedClearForValidation(asked));
                static_cast<void>(pipeline.WrittenSince(pipeline.WatchWrites(), {{address, address + 8}}));
                scans.fetch_add(1);
            }
        } catch (const std::exception& error) {
            const std::lock_guard lock(failureMutex);
            failure = error.what();
            done.store(true);
        }
    });
    for (std::uint64_t tag = 1; tag <= Items && !done.load(); ++tag) {
        const auto slot = tag % Slots;
        const std::uint64_t value = (tag << 8) | slot;
        std::vector<std::byte> bytes(sizeof(value));
        std::memcpy(bytes.data(), &value, sizeof(value));
        pipeline.EnqueueStore([] {}, labelAt(slot), std::move(bytes));
        if (tag % 3 == 0) pipeline.Enqueue([] {}, {{drawnAt(slot), drawnAt(slot) + 0x10}}, 0, {}, {{readAt(slot), readAt(slot) + 0x10}}, true);
    }
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    done.store(true);
    scanner.join();
    {
        const std::lock_guard lock(failureMutex);
        Require(failure.empty(), "a scanner beside the committer: " + failure);
    }
    Require(scans.load() > 0, "the scanner never ran a round");
    if (pipeline.CommitRing()) Require(DrawPipeline::RingTotal(RingCount::SlowScans) > scansBefore, "no scan took the slow path while the ring was latched");
}

void DrainWaitsForEmpty() {
    auto& pipeline = Pipeline();
    DrainAll();
    ClearCommitted();
    std::atomic<bool> open{false};
    pipeline.Enqueue([&open] {
        Hold(open);
        Record(0);
    }, {TagRange(0)}, 0, {}, {}, true);
    for (std::uint64_t tag = 1; tag <= 300; ++tag) EnqueueTag(tag);
    std::atomic<bool> drained{false};
    std::thread drainer([&] {
        pipeline.Drain(DrawPipeline::DrainReason::Packet);
        drained.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    const bool early = drained.load();
    open.store(true);
    drainer.join();
    Require(!early, "a drain returned while a blocked item was queued");
    Require(pipeline.Outstanding() == 0 && !pipeline.Busy(), "the queue was not empty when the drain returned");
    RequireInOrder("a drain", 301);
    ClearCommitted();
}

void SwitchFlipsAtEmptyQueue() {
    auto& pipeline = Pipeline();
    DrainAll();
    ClearCommitted();
    const bool initial = RuntimeSwitches::Set(RuntimeSwitch::CommitRing);
    EnqueueTag(0);
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    const bool latched = pipeline.CommitRing();
    Require(latched == (initial && pipeline.DeepQueue()), "the committer ring latch does not follow the switch at an empty queue");
    std::atomic<bool> open{false};
    pipeline.Enqueue([&open] {
        Hold(open);
        Record(1);
    }, {TagRange(1)}, 0, {}, {}, true);
    for (std::uint64_t tag = 2; tag <= 100; ++tag) EnqueueTag(tag);
    RuntimeSwitches::Assign(RuntimeSwitch::CommitRing, !initial);
    for (std::uint64_t tag = 101; tag <= 200; ++tag) EnqueueTag(tag);
    const bool whileQueued = pipeline.CommitRing();
    open.store(true);
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    const bool afterDrain = pipeline.CommitRing();
    EnqueueTag(201);
    const bool flipped = pipeline.CommitRing();
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    RuntimeSwitches::Assign(RuntimeSwitch::CommitRing, initial);
    EnqueueTag(202);
    const bool restored = pipeline.CommitRing();
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    Require(whileQueued == latched, "the committer ring latch changed with items queued");
    Require(afterDrain == latched, "the committer ring latch changed without an enqueue at an empty queue");
    Require(flipped == (!initial && pipeline.DeepQueue()), "the committer ring latch did not follow the flipped switch at the next empty queue");
    Require(restored == latched, "the committer ring latch did not come back with the switch");
    RequireInOrder("a switch flipped with items queued", 203);
    ClearCommitted();
}

void FailureReachesTheEnqueuer() {
    auto& pipeline = Pipeline();
    DrainAll();
    ClearCommitted();
    const auto before = pipeline.Retired();
    pipeline.Enqueue([] { throw ProcessShutdown{}; }, {TagRange(1)}, 0, {}, {}, true);
    Require(WaitFor([&] { return pipeline.Retired() > before; }), "the failing item never retired");
    bool thrown = false;
    try {
        EnqueueTag(2);
    } catch (const ProcessShutdown&) {
        thrown = true;
    }
    Require(thrown, "a failure thrown by a commit did not reach the next enqueuer");
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    EnqueueTag(3);
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    Require(Committed().empty(), "a commit ran after the committer stopped on a failure");
}

}

int main() {
    try {
        if (DrawPipeline::Depth() == 0 || !DrawFinisher::Enabled() || !DrawFinisher::Overlap()) return 77;
        RuntimeSwitches::Assign(RuntimeSwitch::XpFlow, false);
        RuntimeSwitches::Assign(RuntimeSwitch::WorkerHelps, false);
        RuntimeSwitches::Assign(RuntimeSwitch::DeepShallow, false);
        RuntimeSwitches::Assign(RuntimeSwitch::DeepQueues, true);
        RuntimeSwitches::Assign(RuntimeSwitch::DeepFlow, true);
        SetThreads(4);
        auto& finisher = DrawFinisher::Get();
        finisher.Adopt(true);
        Require(finisher.RingStore(), "an idle finisher did not switch to its ring");
        const bool ring = RuntimeSwitches::Set(RuntimeSwitch::CommitRing);
        EnqueueTag(0);
        Pipeline().Drain(DrawPipeline::DrainReason::Submission);
        Require(Pipeline().DeepQueue(), "the committer queue latch is off with the deep flow set");
        Require(Pipeline().CommitRing() == ring, "the committer ring latch does not follow the switch");
        ClearCommitted();
        OrderAcrossEnqueuers();
        FullQueueWaits();
        ScannersSeeWholeItems();
        DrainWaitsForEmpty();
        SwitchFlipsAtEmptyQueue();
        const auto fast = DrawPipeline::RingTotal(RingCount::FastPops);
        const auto locked = DrawPipeline::RingTotal(RingCount::LockedPops);
        const auto scans = DrawPipeline::RingTotal(RingCount::SlowScans);
        FailureReachesTheEnqueuer();
        std::cout << "commit ring tests passed (ring " << (ring ? "on" : "off") << "; " << fast << " pops without the lock, " << locked << " with it, " << scans << " scans on the slow path)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const ProcessShutdown&) {
        std::cerr << "a failure of the committer reached the test outside its case\n";
        return 1;
    }
}
