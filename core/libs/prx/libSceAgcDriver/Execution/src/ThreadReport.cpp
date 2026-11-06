#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/HeapReplay.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <utility>

namespace AgcDriver::ThreadReport {

namespace {

constexpr std::int64_t WindowNs = 10'000'000'000;

struct alignas(64) Counter {
    std::atomic<std::uint64_t> value{0};
};

std::array<Counter, static_cast<std::size_t>(Count::Count)> counts{};
std::atomic<std::int64_t> windowStart{0};
std::mutex printMutex;
thread_local bool queue0Worker = false;
thread_local Clock::time_point lastSubmissionEnd{};
constexpr std::size_t StateCount = static_cast<std::size_t>(WorkerState::Count);
std::array<std::atomic<std::uint64_t>, StateCount> workerStateNs{};
std::atomic<std::uint32_t> workerState{0};
std::atomic<std::int64_t> workerStateSince{0};
std::atomic<std::uint64_t> finisherBusyNs{0};
std::atomic<std::int64_t> finisherBusySince{0};
std::array<Counter, StateCount + 1> committerIdleBy{};

std::int64_t sinceEpoch(Clock::time_point now) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

void print(std::int64_t now, std::int64_t since, const char* why) {
    std::lock_guard lock(printMutex);
    std::array<std::uint64_t, static_cast<std::size_t>(Count::Count)> taken{};
    for (std::size_t index = 0; index < taken.size(); ++index) taken[index] = counts[index].value.exchange(0, std::memory_order_relaxed);
    const auto value = [&](Count count) { return taken[static_cast<std::size_t>(count)]; };
    const auto items = [&](Count count) { return static_cast<unsigned long long>(value(count)); };
    const auto per = [&](Count ns, Count count) { return value(count) == 0 ? 0.0 : static_cast<double>(value(ns)) / 1e3 / static_cast<double>(value(count)); };
    const auto net = [&](Count ns, Count count, std::uint64_t waited) { return value(count) == 0 ? 0.0 : static_cast<double>(value(ns) - std::min(value(ns), waited)) / 1e3 / static_cast<double>(value(count)); };
    const auto ms = [&](Count ns) { return static_cast<double>(value(ns)) / 1e6; };
    const double seconds = since == 0 ? 0.0 : static_cast<double>(now - since) / 1e9;
    const auto finisherWaits = value(Count::FinisherRoomNs) + value(Count::FinisherCommitterDrainNs);
    const auto workerWaits = value(Count::WorkerFinisherDrainNs) + value(Count::WorkerCommitterDrainNs) + value(Count::WorkerFinisherRoomNs) + value(Count::WorkerCommitterRoomNs);
    AgcDriver::ProfilePrint_nid_no_patch(
        "[threads] %.1f s%s: committer %llu draws at %.2f us and %llu other items at %.2f us, idle %.1f ms (with the retire, the dequeue and the hand-off spin); finisher %llu jobs at %.2f us, %.2f us net of its waits for committer room (%llu, %.1f ms) and its drains of the committer (%llu, %.1f ms), idle %.1f ms; queue-0 worker %llu submissions at %.1f us, %.1f us net of its drains and room waits, idle %.1f ms, %llu draw packets at %.2f us with the waits inside them; worker drains of the finisher %llu (%.1f ms), of the committer %llu (%.1f ms); worker room waits for the finisher %llu (%.1f ms), for the committer %llu (%.1f ms); other threads' drains %llu (%.1f ms) and room waits %llu (%.1f ms)\n",
        seconds, why, items(Count::CommitterDraws), per(Count::CommitterDrawNs, Count::CommitterDraws), items(Count::CommitterOther), per(Count::CommitterOtherNs, Count::CommitterOther), ms(Count::CommitterIdleNs),
        items(Count::FinisherJobs), per(Count::FinisherJobNs, Count::FinisherJobs), net(Count::FinisherJobNs, Count::FinisherJobs, finisherWaits), items(Count::FinisherRoomWaits), ms(Count::FinisherRoomNs), items(Count::FinisherCommitterDrains), ms(Count::FinisherCommitterDrainNs), ms(Count::FinisherIdleNs),
        items(Count::WorkerSubmissions), per(Count::WorkerSubmissionNs, Count::WorkerSubmissions), net(Count::WorkerSubmissionNs, Count::WorkerSubmissions, workerWaits), ms(Count::WorkerIdleNs), items(Count::WorkerDrawPackets), per(Count::WorkerDrawPacketNs, Count::WorkerDrawPackets),
        items(Count::WorkerFinisherDrains), ms(Count::WorkerFinisherDrainNs), items(Count::WorkerCommitterDrains), ms(Count::WorkerCommitterDrainNs),
        items(Count::WorkerFinisherRoomWaits), ms(Count::WorkerFinisherRoomNs), items(Count::WorkerCommitterRoomWaits), ms(Count::WorkerCommitterRoomNs),
        items(Count::OtherDrains), ms(Count::OtherDrainNs), items(Count::OtherRoomWaits), ms(Count::OtherRoomNs));
    std::array<double, StateCount + 1> idle{};
    for (std::size_t index = 0; index < idle.size(); ++index) idle[index] = static_cast<double>(committerIdleBy[index].value.exchange(0, std::memory_order_relaxed)) / 1e6;
    const auto state = [&](WorkerState which) { return idle[static_cast<std::size_t>(which)]; };
    AgcDriver::ProfilePrint_nid_no_patch(
        "[idle] %.1f s%s: committer idle %.1f ms; meanwhile the queue-0 worker was executing packets %.1f ms, waiting on GPU memory %.1f ms, draining the finisher %.1f ms, draining the committer %.1f ms, waiting for finisher room %.1f ms, for committer room %.1f ms, waiting for a submission %.1f ms, ordered behind other queues %.1f ms, outside these %.1f ms; the finisher was running jobs %.1f ms of it\n",
        seconds, why, ms(Count::CommitterIdleNs), state(WorkerState::Packets), state(WorkerState::WaitMemory), state(WorkerState::DrainFinisher), state(WorkerState::DrainCommitter),
        state(WorkerState::RoomFinisher), state(WorkerState::RoomCommitter), state(WorkerState::NoSubmission), state(WorkerState::Order), state(WorkerState::Outside), idle[StateCount]);
}

void setWorkerState(WorkerState state) {
    const auto now = sinceEpoch(Clock::now());
    const auto old = workerState.load(std::memory_order_relaxed);
    const auto since = workerStateSince.load(std::memory_order_relaxed);
    if (since != 0 && now > since) workerStateNs[old].fetch_add(static_cast<std::uint64_t>(now - since), std::memory_order_relaxed);
    workerStateSince.store(now, std::memory_order_relaxed);
    workerState.store(static_cast<std::uint32_t>(state), std::memory_order_relaxed);
}

}

bool Enabled() {
    static const bool enabled = [] {
        const char* text = std::getenv("APS5_THREAD_REPORT");
        return text != nullptr && text[0] != '\0' && text[0] != '0';
    }();
    return enabled;
}

void Add(Count count, std::uint64_t value) {
    counts[static_cast<std::size_t>(count)].value.fetch_add(value, std::memory_order_relaxed);
}

std::uint64_t Nanoseconds(Clock::time_point from, Clock::time_point to) {
    return to > from ? static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count()) : 0u;
}

void Tick(Clock::time_point now) {
    if (!Enabled()) return;
    const auto at = sinceEpoch(now);
    auto start = windowStart.load(std::memory_order_relaxed);
    if (start == 0) {
        windowStart.compare_exchange_strong(start, at, std::memory_order_relaxed);
        return;
    }
    if (at - start < WindowNs || !windowStart.compare_exchange_strong(start, at, std::memory_order_relaxed)) return;
    print(at, start, "");
    Graphics::ReportHeapReplay(true);
}

void Phase() {
    if (Enabled()) {
        const auto at = sinceEpoch(Clock::now());
        print(at, windowStart.exchange(at, std::memory_order_relaxed), " to the A/B phase event");
    }
    Graphics::ReportHeapReplay(true);
}

bool OnQueue0Worker() {
    return queue0Worker;
}

bool EnterQueue0Worker(bool queue0) {
    return std::exchange(queue0Worker, queue0);
}

void LeaveQueue0Worker(bool previous) {
    queue0Worker = previous;
}

void WorkerSubmission(Clock::time_point start, Clock::time_point end) {
    Add(Count::WorkerSubmissions, 1);
    Add(Count::WorkerSubmissionNs, Nanoseconds(start, end));
    if (lastSubmissionEnd != Clock::time_point{}) Add(Count::WorkerIdleNs, Nanoseconds(lastSubmissionEnd, start));
    lastSubmissionEnd = end;
    Tick(end);
}

WorkerStateScope::WorkerStateScope(WorkerState state, bool active) : active(active && Enabled()) {
    if (!this->active) return;
    previous = static_cast<WorkerState>(workerState.load(std::memory_order_relaxed));
    setWorkerState(state);
}

WorkerStateScope::~WorkerStateScope() {
    if (active) setWorkerState(previous);
}

StateTimes SnapshotStates(Clock::time_point now) {
    StateTimes times;
    const auto at = sinceEpoch(now);
    for (std::size_t index = 0; index < StateCount; ++index) times.worker[index] = workerStateNs[index].load(std::memory_order_relaxed);
    const auto current = workerState.load(std::memory_order_relaxed);
    const auto since = workerStateSince.load(std::memory_order_relaxed);
    if (since != 0 && at > since && current < StateCount) times.worker[current] += static_cast<std::uint64_t>(at - since);
    times.finisherBusy = finisherBusyNs.load(std::memory_order_relaxed);
    const auto busySince = finisherBusySince.load(std::memory_order_relaxed);
    if (busySince != 0 && at > busySince) times.finisherBusy += static_cast<std::uint64_t>(at - busySince);
    return times;
}

void NoteCommitterIdle(const StateTimes& from, const StateTimes& to) {
    for (std::size_t index = 0; index < StateCount; ++index) {
        if (to.worker[index] > from.worker[index]) committerIdleBy[index].value.fetch_add(to.worker[index] - from.worker[index], std::memory_order_relaxed);
    }
    if (to.finisherBusy > from.finisherBusy) committerIdleBy[StateCount].value.fetch_add(to.finisherBusy - from.finisherBusy, std::memory_order_relaxed);
}

void FinisherBusy(bool busy, Clock::time_point now) {
    const auto at = sinceEpoch(now);
    if (busy) {
        finisherBusySince.store(at, std::memory_order_relaxed);
        return;
    }
    const auto since = finisherBusySince.exchange(0, std::memory_order_relaxed);
    if (since != 0 && at > since) finisherBusyNs.fetch_add(static_cast<std::uint64_t>(at - since), std::memory_order_relaxed);
}

void FlushForTest() {
    ProfileOutput_nid_no_patch().Flush();
}

}
