#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_THREADREPORT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_THREADREPORT_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace AgcDriver::ThreadReport {

enum class Count : std::size_t {
    CommitterDraws,
    CommitterDrawNs,
    CommitterOther,
    CommitterOtherNs,
    CommitterIdleNs,
    FinisherJobs,
    FinisherJobNs,
    FinisherIdleNs,
    FinisherRoomWaits,
    FinisherRoomNs,
    WorkerDrawPackets,
    WorkerDrawPacketNs,
    WorkerSubmissions,
    WorkerSubmissionNs,
    FinisherDrains,
    FinisherDrainNs,
    CommitterDrains,
    CommitterDrainNs,
    WorkerFinisherRoomWaits,
    WorkerFinisherRoomNs,
    WorkerCommitterRoomWaits,
    WorkerCommitterRoomNs,
    Count
};

using Clock = std::chrono::steady_clock;

bool Enabled();
void Add(Count count, std::uint64_t value);
std::uint64_t Nanoseconds(Clock::time_point from, Clock::time_point to);
void Tick(Clock::time_point now);
void Phase();

}

#endif
