#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTALLOCATOR_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTALLOCATOR_HPP

namespace AgcDriver {

void SetHostLargeCacheQuery(bool (*query)()) noexcept;
void SetHostPrefetchQuery(bool (*query)()) noexcept;

}

#endif
