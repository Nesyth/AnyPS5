#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
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
using AgcDriver::Graphics::RuntimeSwitch;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;
using Range = DrawFinisher::Range;
using Clock = std::chrono::steady_clock;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr std::uint16_t ThrowAfterItems = 1;
constexpr std::uint16_t LookUpOwnItems = 2;

struct JobSpec {
    std::uint32_t job = 0;
    std::uint16_t items = 0;
    std::uint16_t flags = 0;
    std::uint32_t spinNs = 0;
    std::uint32_t gapNs = 0;
    std::binary_semaphore* hold = nullptr;
    std::binary_semaphore* stall = nullptr;
};

std::mutex committedMutex;
std::vector<std::uint64_t> committed;
std::atomic<std::uint32_t> finishedJobs{0};
std::uint32_t nextJob = 1;
std::size_t expected = 0;

std::atomic<bool> lookupBusy{false};
std::atomic<bool> lookupOverlaps{false};
std::atomic<bool> lookupOverlapsAny{false};
std::atomic<bool> lookupStored{false};
std::atomic<bool> lookupWritten{false};
std::atomic<bool> lookupClearElsewhere{false};
std::atomic<bool> lookupDrained{false};
std::atomic<bool> lookupRan{false};
std::atomic<bool> lookupsDone{false};

std::uint64_t Tag(std::uint32_t job, std::uint32_t item) {
    return (std::uint64_t{job} << 8) | (item + 1);
}

Range ItemRange(std::uint64_t tag) {
    const std::uint64_t base = 0x40000000ull + (tag % 4093) * 64;
    return {base, base + 16};
}

void Record(std::uint64_t tag) {
    const std::lock_guard lock(committedMutex);
    committed.push_back(tag);
}

void Spin(std::uint32_t ns) {
    if (ns == 0) return;
    const auto until = Clock::now() + std::chrono::nanoseconds(ns);
    while (Clock::now() < until) {
    }
}

void Commit(std::uint64_t tag, std::binary_semaphore* stall = nullptr) {
    auto& pipeline = DrawPipeline::Queue0();
    if (stall == nullptr && tag % 8 == 5) {
        pipeline.EnqueueQuiet([tag] { Record(tag); });
        return;
    }
    const auto range = ItemRange(tag);
    pipeline.Enqueue([tag, stall] {
        if (stall != nullptr) stall->acquire();
        Record(tag);
    }, {range}, 0, {}, {}, true);
}

bool HasCommitted(std::uint64_t tag) {
    const std::lock_guard lock(committedMutex);
    for (const auto value : committed) {
        if (value == tag) return true;
    }
    return false;
}

void LookUp(const JobSpec& spec) {
    auto& pipeline = DrawPipeline::Queue0();
    const auto tag = Tag(spec.job, 0);
    const auto range = ItemRange(tag);
    const Range elsewhere{0x7f000000ull, 0x7f000010ull};
    lookupBusy.store(pipeline.Busy());
    lookupOverlaps.store(pipeline.Overlaps(range.first + 4, 8));
    lookupOverlapsAny.store(pipeline.OverlapsAny({&range, 1}));
    lookupClearElsewhere.store(!pipeline.Overlaps(elsewhere.first, 16) && !pipeline.OverlapsAny({&elsewhere, 1}));
    lookupStored.store(pipeline.StoredBytes(range.first, 8, {}) == DrawPipeline::QueuedWrite::Unknown);
    lookupWritten.store(pipeline.WrittenSince(pipeline.WatchWrites(), {range}));
    lookupsDone.store(true);
    pipeline.Drain(DrawPipeline::DrainReason::Packet);
    lookupDrained.store(HasCommitted(tag));
    lookupRan.store(true);
}

void RunSpec(const JobSpec& spec) {
    if (spec.hold != nullptr) spec.hold->acquire();
    Spin(spec.spinNs);
    for (std::uint32_t item = 0; item < spec.items; ++item) {
        if (item != 0) Spin(spec.gapNs);
        Commit(Tag(spec.job, item), item == 0 ? spec.stall : nullptr);
    }
    if ((spec.flags & LookUpOwnItems) != 0) LookUp(spec);
    finishedJobs.fetch_add(1);
    if ((spec.flags & ThrowAfterItems) != 0) throw std::runtime_error("deposit job failure");
}

void RunPayload(const std::byte* payload) {
    JobSpec spec;
    std::memcpy(&spec, payload, sizeof(spec));
    RunSpec(spec);
}

void Publish(const JobSpec& spec, bool lean) {
    auto& finisher = DrawFinisher::Get();
    if (!lean) {
        finisher.Enqueue([spec] { RunSpec(spec); }, DrawFinisher::Pending{0, 0, true, false, {}, 0, {}});
        return;
    }
    auto* payload = finisher.ReserveLean();
    Require(payload != nullptr, "the finisher ring refused a job while open");
    std::memcpy(payload, &spec, sizeof(spec));
    const auto range = ItemRange(Tag(spec.job, 0));
    finisher.PublishLean(&RunPayload, {&range, 1}, spec.job, 0);
}

JobSpec Job(std::uint16_t items, std::uint32_t spinNs = 0, std::uint32_t gapNs = 0) {
    JobSpec spec;
    spec.job = nextJob++;
    spec.items = items;
    spec.spinNs = spinNs;
    spec.gapNs = gapNs;
    expected += items;
    return spec;
}

void Reset() {
    const std::lock_guard lock(committedMutex);
    committed.clear();
    finishedJobs.store(0);
    expected = 0;
}

void DrainAll() {
    for (int attempt = 0; attempt < 4; ++attempt) {
        try {
            DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet);
            return;
        } catch (const std::runtime_error&) {
        }
    }
    DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet);
}

std::string Where(const char* test, unsigned threads, bool deposit) {
    return std::string(test) + " with " + std::to_string(threads) + " finisher threads" + (deposit ? " and deposits" : "");
}

void CheckOrder(const std::string& where, bool exact = true) {
    const std::lock_guard lock(committedMutex);
    if (exact) Require(committed.size() == expected, where + ": the committer saw " + std::to_string(committed.size()) + " of " + std::to_string(expected) + " items");
    for (std::size_t index = 1; index < committed.size(); ++index) {
        Require(committed[index - 1] < committed[index], where + ": item " + std::to_string(committed[index] >> 8) + "." + std::to_string(committed[index] & 0xff) + " reached the committer after item " + std::to_string(committed[index - 1] >> 8) + "." + std::to_string(committed[index - 1] & 0xff));
    }
}

bool WaitFor(const auto& condition, std::chrono::milliseconds limit = std::chrono::milliseconds(3000)) {
    const auto deadline = Clock::now() + limit;
    while (!condition()) {
        if (Clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    return true;
}

void SetThreads(unsigned count) {
    RuntimeSwitches::Assign(RuntimeSwitch::SecondFinisher, count >= 2);
    RuntimeSwitches::Assign(RuntimeSwitch::ThirdFinisher, count >= 3);
    RuntimeSwitches::Assign(RuntimeSwitch::FourthFinisher, count >= 4);
    RuntimeSwitches::Assign(RuntimeSwitch::FifthFinisher, count >= 5);
}

void SetDeposit(bool deposit) {
    RuntimeSwitches::Assign(RuntimeSwitch::FinisherDeposit, deposit);
}

unsigned RingRoom() {
    return RuntimeSwitches::Set(RuntimeSwitch::DeepQueues) ? 63u : 15u;
}

void StartsEveryFinisher() {
    SetThreads(3);
    Reset();
    Publish(Job(1), true);
    DrainAll();
    SetThreads(5);
    Publish(Job(1), true);
    Publish(Job(1), false);
    const bool started = WaitFor([] { return DrawFinisher::Finishers() >= 5; });
    DrainAll();
    Require(started, "switching on the fourth and fifth finisher after three ran started " + std::to_string(DrawFinisher::Finishers()) + " finisher threads");
    CheckOrder("thread start");
}

void SlowFront(unsigned threads, bool deposit) {
    const auto where = Where("a slow front job", threads, deposit);
    Reset();
    std::binary_semaphore hold{0};
    const auto before = DrawFinisher::Deposits();
    auto front = Job(1);
    front.hold = &hold;
    Publish(front, true);
    const auto behind = std::min(12u, RingRoom());
    for (std::uint32_t index = 0; index < behind; ++index) Publish(Job(static_cast<std::uint16_t>(1 + index % 3), 2000, 1000), index % 2 == 0);
    if (deposit && threads > 1) {
        const bool ranAhead = WaitFor([&] { return finishedJobs.load() == behind; });
        Require(ranAhead, where + ": only " + std::to_string(finishedJobs.load()) + " of " + std::to_string(behind) + " jobs behind a waiting front job finished");
        const std::lock_guard lock(committedMutex);
        Require(committed.empty(), where + ": a deposited item reached the committer before the front job's");
    }
    hold.release();
    DrainAll();
    CheckOrder(where);
    const auto after = DrawFinisher::Deposits();
    if (deposit && threads > 1) Require(after.deposited > before.deposited && after.forwarded > before.forwarded, where + ": no item was deposited");
    if (!deposit) Require(after.deposited == before.deposited, where + ": an item was deposited with the switch unset");
}

void ManyItems(unsigned threads, bool deposit) {
    const auto where = Where("several items per job", threads, deposit);
    Reset();
    for (std::uint32_t index = 0; index < 200; ++index) Publish(Job(static_cast<std::uint16_t>(index % 7), index % 5 == 0 ? 20000 : 500, index % 3 == 0 ? 3000 : 0), index % 3 != 0);
    DrainAll();
    CheckOrder(where);
}

void FullCommitter(unsigned threads, bool deposit) {
    const auto where = Where("a full committer queue", threads, deposit);
    Reset();
    std::binary_semaphore stall{0};
    auto first = Job(1);
    first.stall = &stall;
    Publish(first, false);
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        stall.release();
    });
    for (std::uint32_t index = 0; index < 150; ++index) Publish(Job(static_cast<std::uint16_t>(1 + index % 3), index % 4 == 0 ? 8000 : 0), index % 2 == 0);
    releaser.join();
    DrainAll();
    CheckOrder(where);
}

void DrainWhilePending(unsigned threads, bool deposit) {
    const auto where = Where("a drain while deposits wait", threads, deposit);
    Reset();
    std::binary_semaphore hold{0};
    auto front = Job(2);
    front.hold = &hold;
    Publish(front, false);
    const auto behind = std::min(10u, RingRoom());
    for (std::uint32_t index = 0; index < behind; ++index) Publish(Job(2, 1000), true);
    if (deposit && threads > 1) Require(WaitFor([&] { return finishedJobs.load() == behind; }), where + ": the jobs behind the front job did not finish");
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        hold.release();
    });
    DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet);
    {
        const std::lock_guard lock(committedMutex);
        Require(committed.size() == expected, where + ": a drain returned with " + std::to_string(expected - committed.size()) + " items not committed");
    }
    Require(!DrawFinisher::Busy() && !DrawPipeline::Queue0().Busy(), where + ": the pipeline was busy after a drain");
    releaser.join();
    CheckOrder(where);
}

void FailingJob(unsigned threads, bool deposit) {
    const auto where = Where("a failing job", threads, deposit);
    Reset();
    std::binary_semaphore hold{0};
    std::binary_semaphore gate{0};
    auto front = Job(1);
    front.hold = &hold;
    Publish(front, true);
    auto failing = Job(2);
    failing.flags = ThrowAfterItems;
    failing.hold = &gate;
    Publish(failing, true);
    for (std::uint32_t index = 0; index < 5; ++index) Publish(Job(2), index % 2 == 0);
    gate.release();
    if (deposit && threads > 1) Require(WaitFor([&] { return finishedJobs.load() >= 1; }), where + ": the failing job did not run ahead");
    hold.release();
    bool thrown = false;
    try {
        DrawFinisher::Get().Drain();
    } catch (const std::runtime_error& error) {
        thrown = std::string(error.what()) == "deposit job failure";
    }
    Require(thrown, where + ": the failed job was not rethrown by the next drain");
    DrainAll();
    CheckOrder(where, false);
    Require(HasCommitted(Tag(front.job, 0)) && HasCommitted(Tag(failing.job, 0)) && HasCommitted(Tag(failing.job, 1)), where + ": the items a failing job queued before it failed were not committed");
    Reset();
    for (std::uint32_t index = 0; index < 6; ++index) Publish(Job(2), index % 2 == 0);
    DrainAll();
    CheckOrder(where + " (after the failure)");
}

void OwnItemsVisible(unsigned threads, bool deposit) {
    const auto where = Where("lookups after a job's own items", threads, deposit);
    Reset();
    lookupRan.store(false);
    lookupsDone.store(false);
    std::binary_semaphore hold{0};
    std::binary_semaphore stall{0};
    auto front = Job(1);
    front.hold = &hold;
    front.stall = &stall;
    Publish(front, true);
    auto looking = Job(1);
    looking.flags = LookUpOwnItems;
    Publish(looking, true);
    Publish(Job(1), false);
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        hold.release();
        WaitFor([] { return lookupsDone.load(); });
        stall.release();
    });
    DrainAll();
    releaser.join();
    CheckOrder(where);
    Require(lookupRan.load(), where + ": the looking job did not run");
    Require(lookupBusy.load(), where + ": a job's own queued item did not make the pipeline busy");
    Require(lookupOverlaps.load() && lookupOverlapsAny.load(), where + ": a job's own queued write did not overlap");
    Require(lookupClearElsewhere.load(), where + ": a range nobody writes overlapped");
    Require(lookupStored.load(), where + ": a job's own queued draw write did not answer unknown bytes");
    Require(lookupWritten.load(), where + ": a job's own queued write was not written since a watch");
    Require(lookupDrained.load(), where + ": a job's committer drain returned before its own item was committed");
}

void Stress(std::uint32_t jobs, std::uint32_t seed, bool mixed, unsigned threads, bool deposit) {
    const auto where = std::string("the stress run (seed ") + std::to_string(seed) + (mixed ? ", mixed switches)" : ")") + " with " + std::to_string(threads) + " threads" + (deposit ? " and deposits" : "");
    Reset();
    std::mt19937 random(seed);
    const auto chance = [&](std::uint32_t perThousand) { return random() % 1000 < perThousand; };
    SetThreads(threads);
    SetDeposit(deposit);
    for (std::uint32_t index = 0; index < jobs; ++index) {
        if (mixed && chance(4)) SetThreads(static_cast<unsigned>(1 + random() % 5));
        if (mixed && chance(4)) SetDeposit(!RuntimeSwitches::Set(RuntimeSwitch::FinisherDeposit));
        if (mixed && chance(2)) RuntimeSwitches::Assign(RuntimeSwitch::ItemsByTurn, !RuntimeSwitches::Set(RuntimeSwitch::ItemsByTurn));
        if (chance(10)) {
            ++expected;
            Commit(Tag(nextJob++, 0));
            continue;
        }
        if (chance(3)) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet);
        const auto items = static_cast<std::uint16_t>(random() % 4);
        const auto spin = static_cast<std::uint32_t>(chance(80) ? 50000 + random() % 250000 : random() % 30000);
        const auto gap = static_cast<std::uint32_t>(random() % 4 == 0 ? random() % 10000 : 0);
        Publish(Job(items, spin, gap), random() % 3 != 0);
    }
    DrainAll();
    CheckOrder(where);
    RuntimeSwitches::Assign(RuntimeSwitch::ItemsByTurn, false);
}

void Shutdown() {
    SetThreads(5);
    SetDeposit(true);
    Reset();
    std::binary_semaphore hold{0};
    std::binary_semaphore stall{0};
    auto front = Job(1);
    front.hold = &hold;
    front.stall = &stall;
    Publish(front, true);
    const auto behind = std::min(15u, RingRoom());
    for (std::uint32_t index = 0; index < behind; ++index) Publish(Job(3), index % 2 == 0);
    Require(WaitFor([&] { return finishedJobs.load() == behind; }), "the jobs behind the front job did not finish before the shutdown");
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        hold.release();
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        stall.release();
    });
    DrawFinisher::Get().Stop();
    Require(DrawFinisher::Finishers() == 0, "the finisher shutdown left finisher threads running");
    releaser.join();
    DrawPipeline::Queue0().Stop();
    CheckOrder("the shutdown with deposits");
    const auto caller = std::this_thread::get_id();
    bool ranHere = false;
    const auto after = Job(1);
    DrawFinisher::Get().Enqueue([&] {
        ranHere = std::this_thread::get_id() == caller;
        RunSpec(after);
    });
    Require(ranHere, "a job queued after the shutdown did not run inline");
    CheckOrder("a job after the shutdown");
    DrawPipeline::Queue0().ReleaseRetired();
    DrawFinisher::Get().ReleaseRan();
}

}

int main(int argc, char** argv) {
    try {
        if (DrawPipeline::Depth() == 0 || !DrawFinisher::Enabled() || !DrawFinisher::Overlap()) return 77;
        auto& finisher = DrawFinisher::Get();
        finisher.Adopt(true);
        Require(finisher.RingStore(), "an idle finisher did not switch to its ring");
        if (argc > 1 && std::string(argv[1]) == "--shutdown") {
            Shutdown();
            std::cout << "draw finisher deposit shutdown test passed\n";
            return 0;
        }
        const bool deposit = RuntimeSwitches::Set(RuntimeSwitch::FinisherDeposit);
        const bool byTurn = RuntimeSwitches::Set(RuntimeSwitch::ItemsByTurn);
        StartsEveryFinisher();
        for (const unsigned threads : {1u, 2u, 3u, 4u, 5u}) {
            for (const bool deposits : {false, true}) {
                SetThreads(threads);
                SetDeposit(deposits);
                SlowFront(threads, deposits);
                ManyItems(threads, deposits);
                FullCommitter(threads, deposits);
                DrainWhilePending(threads, deposits);
                FailingJob(threads, deposits);
                OwnItemsVisible(threads, deposits);
                Stress(3000, 11 + threads, false, threads, deposits);
            }
        }
        Stress(20000, 7, true, 3, true);
        Stress(20000, 8, true, 5, false);
        SetThreads(1);
        SetDeposit(deposit);
        RuntimeSwitches::Assign(RuntimeSwitch::ItemsByTurn, byTurn);
        DrainAll();
        finisher.Adopt(false);
        const auto counts = DrawFinisher::Deposits();
        std::cout << "draw finisher deposit tests passed (" << counts.deposited << " items deposited, " << counts.forwarded << " forwarded, " << counts.direct << " queued at their turn)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
