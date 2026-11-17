#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVERHOSTCENSUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVERHOSTCENSUS_HPP

#include <vulkan/vulkan.h>

namespace AgcDriver {

const VkAllocationCallbacks* DriverHostCensusCallbacks() noexcept;

}

#endif
