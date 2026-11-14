#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_DRAW_DRAWREGISTERKEY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_DRIVER_DRAW_DRAWREGISTERKEY_HPP

#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace AgcDriver::DriverDetail {

struct RegisterStateKeyParts {
    std::uint64_t stableRegisters = 0;
    std::uint64_t userDataRegisters = 0;
    std::uint64_t auxiliaryRegisters = 0;
};

inline std::uint64_t UnorderedUserWordKey(std::uint64_t shape, std::uint64_t banks, std::uint64_t auxiliary) {
    return (((shape ^ banks) * 0x100000001b3ull) ^ auxiliary) * 0x100000001b3ull;
}

std::uint64_t RegisterStateKey(const QueueState& queue, bool allUserWords = false, bool incremental = true, std::uint64_t* shapeKey = nullptr, RegisterStateKeyParts* parts = nullptr, bool unordered = false);
std::uint64_t RegisterShapeKey(const QueueState& queue);
std::uint64_t DrawShaderSetKey(const QueueState& queue, const ShaderRegistry& registry);

struct DrawShaderSet {
    static constexpr std::array<std::uint32_t, 6> Bases{0x008u, 0x048u, 0x088u, 0x0c8u, 0x108u, 0x148u};
    std::array<bool, 6> present{};
    std::array<bool, 6> registered{};
    std::array<std::uint32_t, 6> low{};
    std::array<std::uint32_t, 6> high{};
    std::array<std::vector<std::uint64_t>, 6> mixes;
    bool operator==(const DrawShaderSet&) const = default;
};

DrawShaderSet ReadDrawShaderSet(const QueueState& queue, const ShaderRegistry& registry);

}

#endif
