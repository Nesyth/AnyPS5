#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_PACKETS_PACKETRULES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_PACKETS_PACKETRULES_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace AgcDriver::DriverDetail {

struct Submission;

}

namespace AgcDriver::DriverDetail::PacketRules {

void noteXpStream(bool seal);
bool xpDropsDraw();
bool orderedFills();
bool keepsDrawFinisher(std::uint32_t header, std::uint32_t opcode);
bool waitTakesPendingLabel(std::span<const std::uint32_t> packet);
bool keepsDrawPipeline(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode);
std::size_t verdictRanges(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode, std::array<std::pair<std::uint64_t, std::uint64_t>, 3>& ranges);
bool awaitVerdictWriters(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode);
bool awaitVerdictFinisherWriters(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode);
void awaitPredicationWriters(std::uint64_t address, std::size_t bytes, std::uint32_t opcode);
enum class FlipForm : std::uint8_t { Header, EopLabel, EopNothing, Kept };
enum class DmaForm : std::uint8_t { Queues, Copy, Gds, Refused };
FlipForm flipForm(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode);
bool sealsWriteOnly(std::span<const std::uint32_t> packet, std::uint32_t opcode);
bool& drawnSinceSeal();
void dumpQueue(const Submission& submission);

}

#endif
