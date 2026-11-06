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
    FinisherCommitterDrains,
    FinisherCommitterDrainNs,
    WorkerDrawPackets,
    WorkerDrawPacketNs,
    WorkerSubmissions,
    WorkerSubmissionNs,
    WorkerIdleNs,
    WorkerFinisherDrains,
    WorkerFinisherDrainNs,
    WorkerCommitterDrains,
    WorkerCommitterDrainNs,
    WorkerFinisherRoomWaits,
    WorkerFinisherRoomNs,
    WorkerCommitterRoomWaits,
    WorkerCommitterRoomNs,
    OtherDrains,
    OtherDrainNs,
    OtherRoomWaits,
    OtherRoomNs,
    Count
};

using Clock = std::chrono::steady_clock;

bool Enabled();
void Add(Count count, std::uint64_t value);
std::uint64_t Nanoseconds(Clock::time_point from, Clock::time_point to);
void Tick(Clock::time_point now);
void Phase();

bool OnQueue0Worker();
bool EnterQueue0Worker(bool queue0);
void LeaveQueue0Worker(bool previous);
void WorkerSubmission(Clock::time_point start, Clock::time_point end);
void FlushForTest();

}

#endif
