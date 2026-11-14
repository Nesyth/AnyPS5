#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanFront.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawFinisher;
using AgcDriver::DriverDetail::DrawPipeline;
using AgcDriver::DriverDetail::PublishedKeys;
using AgcDriver::Graphics::RuntimeSwitch;
namespace LeanFront = AgcDriver::DriverDetail::LeanFront;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;
using Range = DrawFinisher::Range;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void SetThreads(unsigned count) {
    RuntimeSwitches::Assign(RuntimeSwitch::SecondFinisher, count >= 2);
    RuntimeSwitches::Assign(RuntimeSwitch::ThirdFinisher, count >= 3);
    RuntimeSwitches::Assign(RuntimeSwitch::FourthFinisher, count >= 4);
    RuntimeSwitches::Assign(RuntimeSwitch::FifthFinisher, count >= 5);
}

std::mutex orderMutex;
std::vector<std::uint32_t> order;

void Append(std::uint32_t value) {
    DrawFinisher::Get().AwaitTurn();
    const std::lock_guard lock(orderMutex);
    order.push_back(value);
}

void RunPayload(const std::byte* payload) {
    std::uint32_t value = 0;
    std::memcpy(&value, payload, sizeof(value));
    Append(value);
}

void PublishValue(std::uint32_t value, std::span<const Range> writes = {}, std::uint64_t key = 0) {
    auto& finisher = DrawFinisher::Get();
    auto* payload = finisher.ReserveLean();
    Require(payload != nullptr, "the finisher ring refused a job while open");
    std::memcpy(payload, &value, sizeof(value));
    finisher.PublishLean(&RunPayload, writes, key, 0);
}

void QueriesFromTheRing(bool lockless) {
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
    Require(finisher.RingStore(), "a job queued in ring mode switched the finisher store");
    Require(finisher.PendingOverlaps(0x800f0, 0x80110), "a pending ring write did not overlap");
    Require(!finisher.PendingOverlaps(0x80100, 0x80110), "a range after a pending ring write overlapped");
    const std::array<Range, 2> ranges{{{0x1, 0x2}, {0x80000, 0x80001}}};
    Require(finisher.PendingOverlapsAny(ranges), "a pending ring write did not overlap a list of ranges");
    const bool precise = RuntimeSwitches::Set(RuntimeSwitch::PreciseLookupDrain);
    RuntimeSwitches::Assign(RuntimeSwitch::PreciseLookupDrain, false);
    Require(finisher.PendingKey(41, 0), "a pending ring key was not found");
    Require(finisher.PendingKey(1, 42), "a pending ring shape was not found");
    Require(!finisher.PendingKey(43, 0), "a key that is not in the ring was found");
    RuntimeSwitches::Assign(RuntimeSwitch::PreciseLookupDrain, true);
    Require(!finisher.PendingKey(41, 0), "a pending full hit in the ring was found by a precise lookup");
    RuntimeSwitches::Assign(RuntimeSwitch::PreciseLookupDrain, precise);
    const std::array<Range, 1> lean{{{0x90000, 0x90010}}};
    PublishValue(7, lean, 51);
    Require(finisher.PendingOverlaps(0x90008, 0x90009), "a pending lean draw's write template did not overlap");
    Require(finisher.PendingKey(51, 0), "a pending lean draw's key was not found");
    std::uint64_t value = 0;
    Require(finisher.PendingLabel(0x90000, 4, value) == DrawFinisher::LabelLookup::Unknown, "a lean draw's held write answered as a label value");
    DrawFinisher::Pending label;
    label.unknown = false;
    label.writes = {{0xa0000, 0xa0004}};
    label.labelAddress = 0xa0000;
    label.labelBytes = {std::byte{0x11}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}};
    finisher.Enqueue([] {}, label);
    Require(finisher.PendingLabel(0xa0000, 4, value) == DrawFinisher::LabelLookup::Value && value == 0x44332211u, "a pending ring label did not answer its value");
    finisher.Enqueue([] {});
    Require(finisher.PendingOverlaps(0x1, 0x2), "a pending ring job with unknown writes did not overlap everything");
    Require(finisher.Issued() > finisher.Front(), "the ring issued nothing ahead of its front");
    release.release();
    finisher.Drain();
    Require(finisher.Front() == finisher.Issued(), "a drained ring left jobs before its front");
    Require(!finisher.PendingOverlaps(0x800f0, 0x80110), "a finished ring job's write still overlapped");
    Require(!finisher.PendingKey(41, 0) && !finisher.PendingKey(51, 0), "a finished ring job's key was still found");
    order.clear();
}

void OrderAcrossFinishers() {
    auto& finisher = DrawFinisher::Get();
    order.clear();
    constexpr std::uint32_t Jobs = 300;
    for (std::uint32_t index = 0; index < Jobs; ++index) {
        if (index % 3 == 0) {
            finisher.Enqueue([index] {
                if (index % 7 == 0) std::this_thread::sleep_for(std::chrono::microseconds(50));
                Append(index);
            }, DrawFinisher::Pending{0, 0, true, false, {}, 0, {}});
        } else {
            PublishValue(index);
        }
    }
    finisher.Drain();
    Require(order.size() == Jobs, "the ring ran " + std::to_string(order.size()) + " of " + std::to_string(Jobs) + " jobs");
    for (std::uint32_t index = 0; index < Jobs; ++index) Require(order[index] == index, "the ring ran job " + std::to_string(order[index]) + " in turn " + std::to_string(index));
    Require(finisher.Front() == finisher.Issued() && !DrawFinisher::Busy(), "the drained ring is still busy");
}

void FailureRethrown() {
    auto& finisher = DrawFinisher::Get();
    order.clear();
    finisher.Enqueue([] { throw std::runtime_error("ring job failure"); }, DrawFinisher::Pending{0, 0, true, false, {}, 0, {}});
    bool thrown = false;
    try {
        finisher.Drain();
    } catch (const std::runtime_error& error) {
        thrown = std::string(error.what()) == "ring job failure";
    }
    Require(thrown, "a failed ring job was not rethrown by the next drain");
    PublishValue(5);
    finisher.Drain();
    Require(order.size() == 1 && order[0] == 5, "a job queued after the failure was rethrown did not run");
}

void RoomWait() {
    auto& finisher = DrawFinisher::Get();
    const bool deep = RuntimeSwitches::Set(RuntimeSwitch::DeepQueues);
    RuntimeSwitches::Assign(RuntimeSwitch::DeepQueues, false);
    std::binary_semaphore release{0};
    finisher.Enqueue([&] { release.acquire(); }, DrawFinisher::Pending{0, 0, true, false, {}, 0, {}});
    order.clear();
    for (std::uint32_t index = 1; index < 16; ++index) PublishValue(index);
    const auto waitsBefore = LeanFront::Total(LeanFront::Counter::RingRoomWaits);
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        release.release();
    });
    const auto start = std::chrono::steady_clock::now();
    PublishValue(16);
    const auto waited = std::chrono::steady_clock::now() - start;
    releaser.join();
    finisher.Drain();
    RuntimeSwitches::Assign(RuntimeSwitch::DeepQueues, deep);
    Require(LeanFront::Total(LeanFront::Counter::RingRoomWaits) == waitsBefore + 1, "a job queued into a full ring did not wait for room");
    Require(waited >= std::chrono::milliseconds(20), "a job queued into a full ring did not wait for the front job");
    Require(order.size() == 16, "the ring lost jobs queued behind a room wait");
}

std::mutex helpedMutex;
std::vector<std::thread::id> helpedThreads;
std::atomic<bool> helpedOnFinisherThread{true};

void RunHelped(const std::byte*) {
    if (!DrawFinisher::OnFinisherThread()) helpedOnFinisherThread.store(false);
    const std::lock_guard lock(helpedMutex);
    helpedThreads.push_back(std::this_thread::get_id());
}

void PublishHelped() {
    auto& finisher = DrawFinisher::Get();
    Require(finisher.ReserveLean() != nullptr, "the finisher ring refused a job while open");
    finisher.PublishLean(&RunHelped, {}, 0, 0);
}

void WorkerHelps() {
    auto& finisher = DrawFinisher::Get();
    const bool deep = RuntimeSwitches::Set(RuntimeSwitch::DeepQueues);
    const bool helps = RuntimeSwitches::Set(RuntimeSwitch::WorkerHelps);
    RuntimeSwitches::Assign(RuntimeSwitch::DeepQueues, false);
    SetThreads(1);
    finisher.Drain();
    const auto self = std::this_thread::get_id();
    const auto ranHere = [&] {
        const std::lock_guard lock(helpedMutex);
        return static_cast<std::size_t>(std::count(helpedThreads.begin(), helpedThreads.end(), self));
    };
    const auto blockFront = [&](std::binary_semaphore& release, std::atomic<bool>& started) {
        finisher.Enqueue([&] {
            started.store(true);
            release.acquire();
        }, DrawFinisher::Pending{0, 0, true, false, {}, 0, {}});
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!started.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        Require(started.load(), "the only finisher thread never took the blocking job");
    };
    for (const bool on : {false, true}) {
        RuntimeSwitches::Assign(RuntimeSwitch::WorkerHelps, on);
        {
            const std::lock_guard lock(helpedMutex);
            helpedThreads.clear();
        }
        helpedOnFinisherThread.store(true);
        std::binary_semaphore release{0};
        std::atomic<bool> started{false};
        blockFront(release, started);
        for (int index = 0; index < 6; ++index) PublishHelped();
        std::thread releaser([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            release.release();
        });
        finisher.Drain(true);
        releaser.join();
        Require(!DrawFinisher::OnFinisherThread(), "the producer thread stayed a finisher thread after it drained");
        Require(helpedOnFinisherThread.load(), "a ring job ran without the finisher thread flag");
        {
            const std::lock_guard lock(helpedMutex);
            Require(helpedThreads.size() == 6, "a helped drain lost ring jobs");
        }
        Require(ranHere() == (on ? 6u : 0u), on ? "the draining producer did not run the untaken ring jobs" : "the producer ran ring jobs with the switch off");
        {
            const std::lock_guard lock(helpedMutex);
            helpedThreads.clear();
        }
        std::binary_semaphore releaseRoom{0};
        std::atomic<bool> startedRoom{false};
        blockFront(releaseRoom, startedRoom);
        for (int index = 1; index < 16; ++index) PublishHelped();
        std::thread roomReleaser([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            releaseRoom.release();
        });
        PublishHelped();
        roomReleaser.join();
        finisher.Drain();
        {
            const std::lock_guard lock(helpedMutex);
            Require(helpedThreads.size() == 16, "a helped room wait lost ring jobs");
        }
        Require(on ? ranHere() >= 15u : ranHere() == 0u, on ? "the producer waiting for ring room did not run the untaken ring jobs" : "the producer ran ring jobs in a room wait with the switch off");
        order.clear();
        PublishValue(9);
        finisher.Drain();
        Require(order.size() == 1 && order[0] == 9, "the producer could not queue a job after it helped");
    }
    RuntimeSwitches::Assign(RuntimeSwitch::WorkerHelps, helps);
    RuntimeSwitches::Assign(RuntimeSwitch::DeepQueues, deep);
}

std::atomic<bool> frontFinished{false};
std::atomic<bool> aheadStarted{false};
std::atomic<bool> aheadSawFront{false};
std::atomic<bool> keyHeldAhead{false};

void RunAhead(const std::byte*) {
    aheadStarted.store(true);
    auto& finisher = DrawFinisher::Get();
    keyHeldAhead.store(finisher.KeyBefore(77, 0, false) != PublishedKeys::Answer::Clear);
    const DrawFinisher::AheadScope ahead;
    Require(DrawPipeline::Queue0().Busy(), "an earlier unfinished job did not make the pipeline busy for a lean job");
    static_cast<void>(finisher.PendingOverlaps(0x1000, 0x1010));
    aheadSawFront.store(frontFinished.load());
}

void AheadWaits() {
    auto& finisher = DrawFinisher::Get();
    std::binary_semaphore release{0};
    frontFinished.store(false);
    aheadStarted.store(false);
    aheadSawFront.store(false);
    DrawFinisher::Pending held;
    held.key = 77;
    held.unknown = false;
    held.writes = {{0x1000, 0x1100}};
    finisher.Enqueue([&] {
        release.acquire();
        frontFinished.store(true);
    }, held);
    const auto waitsBefore = LeanFront::Total(LeanFront::Counter::AheadWaits);
    auto* payload = finisher.ReserveLean();
    Require(payload != nullptr, "the finisher ring refused a job while open");
    finisher.PublishLean(&RunAhead, {}, 78, 0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!aheadStarted.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    release.release();
    finisher.Drain();
    Require(aheadStarted.load(), "the lean job never ran");
    Require(keyHeldAhead.load(), "an earlier unfinished job's key was not found before the lean job's turn");
    Require(aheadSawFront.load(), "a lean job read a range an earlier job writes before that job finished");
    Require(LeanFront::Total(LeanFront::Counter::AheadWaits) == waitsBefore + 1, "the lean job did not count its wait for the earlier job");
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> answered;

void RunAnswering(const std::byte* payload) {
    std::uint32_t value = 0;
    std::memcpy(&value, payload, sizeof(value));
    if (value % 7 == 0) std::this_thread::sleep_for(std::chrono::microseconds(30));
    if (value % 3 != 0) Require(DrawFinisher::AnswerLean(std::make_shared<const std::uint32_t>(value * 2)), "a lean job with a retirer could not answer");
}

void RunSilent(const std::byte* payload) {
    Require(!DrawFinisher::AnswerLean(std::make_shared<const std::uint32_t>(1)), "a lean job without a retirer took an answer");
    static_cast<void>(payload);
}

void RetireAnswer(const std::byte* payload, std::shared_ptr<const void>&& answer) {
    std::uint32_t value = 0;
    std::memcpy(&value, payload, sizeof(value));
    answered.emplace_back(value, answer == nullptr ? 0u : *std::static_pointer_cast<const std::uint32_t>(answer));
}

void AnswersReachTheRetirer() {
    auto& finisher = DrawFinisher::Get();
    answered.clear();
    constexpr std::uint32_t Jobs = 300;
    std::uint32_t expected = 0;
    for (std::uint32_t index = 1; index <= Jobs; ++index) {
        if (index % 11 == 0) {
            finisher.Enqueue([] { Require(!DrawFinisher::AnswerLean(std::make_shared<const std::uint32_t>(1)), "a closure job took an answer"); }, DrawFinisher::Pending{0, 0, true, false, {}, 0, {}});
            continue;
        }
        auto* payload = finisher.ReserveLean();
        Require(payload != nullptr, "the finisher ring refused a job while open");
        std::memcpy(payload, &index, sizeof(index));
        if (index % 5 == 0) {
            finisher.PublishLean(&RunSilent, {}, 0, 0);
            continue;
        }
        finisher.PublishLean(&RunAnswering, {}, 0, 0, &RetireAnswer);
        ++expected;
    }
    finisher.Drain();
    Require(answered.size() == expected, "the retirer saw " + std::to_string(answered.size()) + " of " + std::to_string(expected) + " answered jobs");
    for (std::size_t index = 0; index < answered.size(); ++index) {
        const auto [value, answer] = answered[index];
        Require(index == 0 || answered[index - 1].first < value, "the retirer saw job " + std::to_string(value) + " out of order");
        Require(answer == (value % 3 != 0 ? value * 2 : 0u), "job " + std::to_string(value) + " retired with answer " + std::to_string(answer));
    }
    Require(!DrawFinisher::AnswerLean(std::make_shared<const std::uint32_t>(1)), "a thread outside the finisher ring took an answer");
}

void StoreSwitches() {
    auto& finisher = DrawFinisher::Get();
    finisher.Drain();
    finisher.Adopt(false);
    Require(!finisher.RingStore(), "an idle finisher did not switch back to its job queue");
    order.clear();
    std::binary_semaphore release{0};
    finisher.Enqueue([&] {
        release.acquire();
        Append(1);
    }, DrawFinisher::Pending{0, 0, true, false, {}, 0, {}});
    finisher.Adopt(true);
    Require(!finisher.RingStore(), "a busy finisher switched stores");
    release.release();
    finisher.Drain();
    finisher.Adopt(true);
    Require(finisher.RingStore(), "an idle finisher did not switch to its ring");
    PublishValue(2);
    finisher.Drain();
    Require(order.size() == 2 && order[0] == 1 && order[1] == 2, "jobs across a store switch ran out of order");
}

}

int main() {
    try {
        if (DrawPipeline::Depth() == 0 || !DrawFinisher::Enabled() || !DrawFinisher::Overlap()) return 77;
        auto& finisher = DrawFinisher::Get();
        finisher.Adopt(true);
        Require(finisher.RingStore(), "an idle finisher did not switch to its ring");
        for (const bool lockless : {false, true}) QueriesFromTheRing(lockless);
        const bool second = RuntimeSwitches::Set(RuntimeSwitch::SecondFinisher);
        const bool third = RuntimeSwitches::Set(RuntimeSwitch::ThirdFinisher);
        const bool fourth = RuntimeSwitches::Set(RuntimeSwitch::FourthFinisher);
        const bool fifth = RuntimeSwitches::Set(RuntimeSwitch::FifthFinisher);
        for (const unsigned threads : {1u, 3u, 4u, 5u}) {
            SetThreads(threads);
            OrderAcrossFinishers();
            FailureRethrown();
            RoomWait();
        }
        for (const unsigned threads : {3u, 5u}) {
            SetThreads(threads);
            AheadWaits();
        }
        for (const unsigned threads : {1u, 3u, 4u, 5u}) {
            SetThreads(threads);
            AnswersReachTheRetirer();
        }
        WorkerHelps();
        RuntimeSwitches::Assign(RuntimeSwitch::SecondFinisher, second);
        RuntimeSwitches::Assign(RuntimeSwitch::ThirdFinisher, third);
        RuntimeSwitches::Assign(RuntimeSwitch::FourthFinisher, fourth);
        RuntimeSwitches::Assign(RuntimeSwitch::FifthFinisher, fifth);
        StoreSwitches();
        finisher.Drain();
        finisher.Adopt(false);
        std::cout << "lean finisher ring tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
