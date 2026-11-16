#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DIAGNOSTICS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DIAGNOSTICS_HPP

#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace AgcDriver::DriverDetail {

double TraceMs();

bool SkippedShader(std::uint64_t address);

class RefusedDraw final : public std::runtime_error {
public:
    RefusedDraw(std::uint64_t address, const std::string& what, bool pending = false) : std::runtime_error(what), address(address), pending(pending) {}
    std::uint64_t address;
    bool pending;
};

bool SkipRefusedDraws();

void ReportRefusedDraw(const RefusedDraw& refused);
std::uint64_t PendingDrawsSkipped();
std::optional<std::size_t> RefusedShaderCount();

void require(bool condition, const char* reason);

std::uint32_t readRegister(const Registers& registers, std::uint32_t offset);

std::uint32_t readUserData(const Registers& shader, std::uint32_t offset);

}

#endif
