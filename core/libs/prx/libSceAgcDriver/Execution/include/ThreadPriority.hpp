#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_THREADPRIORITY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_THREADPRIORITY_HPP

#include <cstdint>
#include <optional>
#include <string>

namespace AgcDriver {

enum class ThreadPriorityMode { Off, High, Realtime };

struct RttimeLimit {
    std::uint64_t soft = 0;
    std::uint64_t hard = 0;
    bool operator==(const RttimeLimit&) const = default;
};

struct RtkitLimits {
    std::int64_t maxPriority = 0;
    std::int64_t maxRttimeUs = 0;
};

struct RealtimePlan {
    std::string refusal;
    std::optional<RttimeLimit> limit;
};

ThreadPriorityMode ParseThreadPriorityMode(const char* text);
RealtimePlan PlanRealtime(std::uint32_t priority, const std::optional<RtkitLimits>& rtkit, RttimeLimit current, std::uint64_t requestedUs);
void RaiseWorkerThreadPriority(const char* role);
// Windows: reserve a physical core for the calling thread (the draw committer). The process default CPU
// set drops both of the core's logical processors and the thread selects one, so its SMT sibling stays
// idle and the title's spinning job workers cannot share it. By default the highest-numbered core on a
// host with at least six; APS5_COMMITTER_CPU=<logical processor> picks one, APS5_NO_COMMITTER_CORE=1
// turns it off. Threads pinned with their own mask (APS5_JOB_AFFINITY_MASK) keep it.
void PlaceReservedThread(const char* role);
// The next free physical core the same way for the draw finisher; APS5_NO_FINISHER_CORE=1 turns that off
// (APS5_NO_COMMITTER_CORE=1 turns both off).
void PlaceFinisherThread(const char* role);

}

#endif
