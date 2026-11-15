#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTALLOCATOR_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTALLOCATOR_HPP

#include <cstdint>

namespace AgcDriver {

void SetHostLargeCacheQuery(bool (*query)()) noexcept;
void SetHostPrefetchQuery(bool (*query)()) noexcept;
bool HostAllocationCountsOn() noexcept;
void HostAllocationCounts(std::uint64_t& allocations, std::uint64_t& frees) noexcept;

}

#endif
