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
// APS5_XP_COMMITTER_CPU=<logical processor> (experiment, Windows): reserve that processor's physical core
// for the calling thread. The process default CPU set drops both of the core's logical processors and the
// thread selects the given one, so its SMT sibling stays idle. Pinned threads (job workers) need their own mask.
void PlaceReservedThread(const char* role);

}

#endif
