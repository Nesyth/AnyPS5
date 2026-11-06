#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>

namespace AgcDriver::ThreadReport {

namespace {

constexpr std::int64_t WindowNs = 10'000'000'000;

std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Count::Count)> counts{};
std::atomic<std::int64_t> windowStart{0};
std::mutex printMutex;

std::int64_t sinceEpoch(Clock::time_point now) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

void print(std::int64_t now, std::int64_t since, const char* why) {
    std::lock_guard lock(printMutex);
    std::array<std::uint64_t, static_cast<std::size_t>(Count::Count)> taken{};
    for (std::size_t index = 0; index < taken.size(); ++index) taken[index] = counts[index].exchange(0, std::memory_order_relaxed);
    const auto value = [&](Count count) { return taken[static_cast<std::size_t>(count)]; };
    const auto items = [&](Count count) { return static_cast<unsigned long long>(value(count)); };
    const auto per = [&](Count ns, Count count) { return value(count) == 0 ? 0.0 : static_cast<double>(value(ns)) / 1e3 / static_cast<double>(value(count)); };
    const auto ms = [&](Count ns) { return static_cast<double>(value(ns)) / 1e6; };
    const double seconds = since == 0 ? 0.0 : static_cast<double>(now - since) / 1e9;
    AgcDriver::ProfilePrint_nid_no_patch(
        "[threads] %.1f s%s: committer %llu draws at %.2f us and %llu other items at %.2f us, idle %.1f ms; finisher %llu jobs at %.2f us (waiting for committer room %llu times, %.1f ms of the jobs), idle %.1f ms; queue-0 worker %llu draw packets at %.2f us, %llu submissions at %.1f us; worker drains of the finisher %llu (%.1f ms), of the committer %llu (%.1f ms); worker room waits for the finisher %llu (%.1f ms), for the committer %llu (%.1f ms)\n",
        seconds, why, items(Count::CommitterDraws), per(Count::CommitterDrawNs, Count::CommitterDraws), items(Count::CommitterOther), per(Count::CommitterOtherNs, Count::CommitterOther), ms(Count::CommitterIdleNs),
        items(Count::FinisherJobs), per(Count::FinisherJobNs, Count::FinisherJobs), items(Count::FinisherRoomWaits), ms(Count::FinisherRoomNs), ms(Count::FinisherIdleNs),
        items(Count::WorkerDrawPackets), per(Count::WorkerDrawPacketNs, Count::WorkerDrawPackets), items(Count::WorkerSubmissions), per(Count::WorkerSubmissionNs, Count::WorkerSubmissions),
        items(Count::FinisherDrains), ms(Count::FinisherDrainNs), items(Count::CommitterDrains), ms(Count::CommitterDrainNs),
        items(Count::WorkerFinisherRoomWaits), ms(Count::WorkerFinisherRoomNs), items(Count::WorkerCommitterRoomWaits), ms(Count::WorkerCommitterRoomNs));
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
    counts[static_cast<std::size_t>(count)].fetch_add(value, std::memory_order_relaxed);
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
}

void Phase() {
    if (!Enabled()) return;
    const auto at = sinceEpoch(Clock::now());
    print(at, windowStart.exchange(at, std::memory_order_relaxed), " to the A/B phase event");
}

}
