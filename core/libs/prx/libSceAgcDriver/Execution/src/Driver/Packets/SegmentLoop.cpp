#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/SegmentLoop.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Packets/DriverTimed.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Packets/PacketRules.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanFront.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/SubmissionRing.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceOverlay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

struct LeanRun {
    PacketProfile& packetProfile;
    PacketCounters& packetCounters;
    FlushGate gate;
    bool packetEpoch = false;
    bool report = false;
    std::uint64_t packets = 0;
};

namespace {

using LeanFront::Counter;
constexpr std::uint32_t NoTag = 0xffffffffu;

bool orderedWaits() {
    static const bool ordered = std::getenv("APS5_PIPELINE_DRAIN_LABELS") == nullptr;
    return ordered;
}

bool orderedCompletion() {
    static const bool ordered = std::getenv("APS5_PIPELINE_DRAIN_SUBMISSIONS") == nullptr && orderedWaits();
    return ordered;
}

bool epochEach() {
    static const bool each = std::getenv("APS5_SUBMISSION_EPOCH_EACH") != nullptr;
    return each;
}

bool traceGpu() {
    static const bool trace = std::getenv("APS5_TRACE_GPU") != nullptr;
    return trace;
}

bool profileDraws() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

bool traceDraws() {
    static const bool trace = std::getenv("APS5_TRACE_DRAWS") != nullptr;
    return trace;
}

bool drainAll() {
    static const bool all = std::getenv("APS5_DRAIN_ALL") != nullptr;
    return all;
}

bool drawDrain() {
    static const bool drain = std::getenv("APS5_DRAW_DRAIN") != nullptr;
    return drain;
}

bool skipInvalidLists() {
    static const bool skip = std::getenv("APS5_SKIP_INVALID_INDIRECT_REGISTERS") != nullptr;
    return skip;
}

bool everyPacketFlush() {
    static const bool every = !LabelBatchSubmit() || std::getenv("APS5_LABEL_OWN_LOCK") != nullptr || std::getenv("APS5_REAP_EACH_PACKET") != nullptr || std::getenv("APS5_LABEL_TRY_EACH_PACKET") != nullptr;
    return every;
}

bool timersWanted() {
    return APS5_ENABLE_TIMING_LOG || PerformanceTrace::Recording();
}

bool reportWanted() {
    return ThreadReport::Enabled() || profileDraws() || CaptureTrace::Enabled() || traceGpu() || traceDraws();
}

std::string hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

std::string where(const Submission& submission, std::size_t cursor) {
    return "serial " + std::to_string(submission.serial) + " " + Pm4::Name(submission.commands[cursor]) + " at DWORD " + std::to_string(cursor);
}

LeanPacket::Pipeline pipelineOf(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode) {
    using Pipeline = LeanPacket::Pipeline;
    if (header == FlipPacketHeader || header == RenderingWaitPacketHeader) return Pipeline::Drains;
    if (Pm4::DrawOpcode(opcode)) return Pipeline::Keeps;
    switch (opcode) {
        case 0x63: case 0x64: case 0x9f: return packet.size() >= 5 ? Pipeline::AsksLists : Pipeline::Drains;
        case 0x50:
            if (packet.size() < 7) return Pipeline::Drains;
            if (Pm4::DmaDestination(packet) == Pm4::DmaDestinationNowhere || Pm4::DmaBytes(packet) == 0) return Pipeline::Keeps;
            return Pipeline::Asks;
        case 0x3c: case 0x93: return packet.size() >= 5 ? Pipeline::Asks : Pipeline::Drains;
        default: return PacketRules::keepsDrawPipeline(packet, header, opcode) ? Pipeline::Keeps : Pipeline::Drains;
    }
}

LeanPacket classify(std::span<const std::uint32_t> packet, std::size_t cursor) {
    const auto header = packet[0];
    const auto opcode = (header >> 8u) & 0xffu;
    LeanPacket entry;
    entry.cursor = static_cast<std::uint32_t>(cursor);
    entry.words = static_cast<std::uint32_t>(packet.size());
    entry.header = header;
    entry.opcode = static_cast<std::uint8_t>(opcode);
    std::uint32_t flags = 0;
    if (Pm4::Predicated(header)) flags |= LeanPacket::Predicated;
    if (Pm4::DrawOpcode(opcode)) flags |= LeanPacket::Draw;
    if (Pm4::IndirectDrawOpcode(opcode)) flags |= LeanPacket::IndirectDraw;
    if (header == FlipPacketHeader) flags |= LeanPacket::FlipHeader;
    if (header == RenderingWaitPacketHeader) flags |= LeanPacket::RenderingWait;
    if (PacketRules::keepsDrawFinisher(header, opcode)) flags |= LeanPacket::KeepsFinisher;
    if (PacketRules::sealsWriteOnly(packet, opcode)) flags |= LeanPacket::Seal;
    if (opcode == 0x3c || opcode == 0x93) flags |= LeanPacket::Wait;
    if (Pm4::IndirectRegisterOpcode(opcode)) flags |= LeanPacket::IndirectRegisters;
    switch (opcode) {
        case 0x37: case 0x49: case 0x40: case 0x50: case 0x83: flags |= LeanPacket::LabelIssue; break;
        default: break;
    }
    switch (opcode) {
        case 0x37: case 0x49: case 0x50: case 0x83: case 0x46: flags |= LeanPacket::Enqueues; break;
        default: break;
    }
    switch (opcode) {
        case 0x37: case 0x49: case 0x3c: case 0x93: case 0x40: case 0x50: flags |= LeanPacket::Traced; break;
        default: break;
    }
    bool memory = drainAll() || header == FlipPacketHeader || Pm4::SampleDump(packet) || (Pm4::DrawOpcode(opcode) && drawDrain());
    switch (opcode) {
        case 0x37: case 0x49: case 0x40: case 0x45: case 0x50: case 0x83: memory = true; break;
        default: break;
    }
    if (memory) flags |= LeanPacket::Memory;
    constexpr std::uint32_t busy = LeanPacket::Predicated | LeanPacket::Draw | LeanPacket::FlipHeader | LeanPacket::RenderingWait | LeanPacket::Seal | LeanPacket::Wait | LeanPacket::LabelIssue | LeanPacket::Enqueues | LeanPacket::Memory | LeanPacket::Traced;
    bool plain = (flags & busy) == 0;
    switch (opcode) {
        case 0x15: case 0x16: case 0x22: case 0x91: case 0x3f: case 0x46: case 0x42: case 0x58: plain = false; break;
        default: break;
    }
    if (plain) flags |= LeanPacket::Plain;
    entry.flags = flags;
    entry.pipeline = pipelineOf(packet, header, opcode);
    return entry;
}

std::vector<std::uint64_t> labelWritesAt(const Submission& submission, std::span<const std::uint32_t> cursors) {
    std::vector<std::uint64_t> writes;
    for (const auto cursor : cursors) {
        const auto packet = std::span<const std::uint32_t>(submission.commands).subspan(cursor, std::min<std::size_t>(Pm4::PacketWords(submission.commands[cursor]), submission.commands.size() - cursor));
        const auto opcode = (packet[0] >> 8u) & 0xffu;
        if ((packet[0] >> 30u) != 3u || (opcode != 0x49 && opcode != 0x37)) continue;
        const auto label = Pm4::DecodeLabelWrite(packet);
        if (!label.has_value()) continue;
        const auto bytes = label->Bytes();
        if (label->address % 4 != 0 || bytes.size() > 64) continue;
        for (std::size_t offset = 0; offset < bytes.size(); offset += 4) writes.push_back(label->address + offset);
    }
    return writes;
}

bool newCounterWindow(std::atomic<std::int64_t>& window) {
    const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    return window.exchange(now, std::memory_order_relaxed) != now;
}

bool plainVerifyDue() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_PLAIN_REGISTER_PACKETS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    static std::atomic<std::uint64_t> ordinal{0};
    return every != 0 && ordinal.fetch_add(1, std::memory_order_relaxed) % every == 0;
}

void notePlainVerify(bool drains, bool flushDue, const Submission& submission, std::size_t cursor) {
    static std::atomic<std::uint64_t> checked{0}, deferred{0}, differing{0};
    static std::atomic<std::int64_t> window{-1};
    checked.fetch_add(1, std::memory_order_relaxed);
    if (flushDue) deferred.fetch_add(1, std::memory_order_relaxed);
    if (drains && differing.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[plain-register-verify] %s: the register packet took the short path, today's rules drain the finisher or the committer before it\n", where(submission, cursor).c_str());
    if (newCounterWindow(window)) std::fprintf(stderr, "[plain-register-verify] %llu packets checked, %llu with a flush due deferred to the next full packet, %llu differing\n", static_cast<unsigned long long>(checked.load(std::memory_order_relaxed)), static_cast<unsigned long long>(deferred.load(std::memory_order_relaxed)), static_cast<unsigned long long>(differing.load(std::memory_order_relaxed)));
}

std::uint64_t everyFrom(const char* name) {
    const char* text = std::getenv(name);
    return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
}

constexpr std::size_t LabelRunCap = 32;
constexpr std::size_t NoQueuedLabel = static_cast<std::size_t>(-1);

bool releasesInterrupt(std::span<const std::uint32_t> packet, std::uint32_t opcode) {
    if (opcode != 0x49 || packet.size() < 3) return false;
    const auto select = (packet[2] >> 24u) & 7u;
    return select != 0 && select != 3;
}

std::pair<LabelRunCensus::Gap, std::uint32_t> labelGap(const std::vector<LeanPacket>& packets, std::size_t previous, std::size_t index) {
    using Gap = LabelRunCensus::Gap;
    if (previous == NoQueuedLabel || previous >= index) return {Gap::First, 0u};
    if (previous + 1 == index) return {Gap::Adjacent, 0u};
    if (index - previous - 1 > 32) return {Gap::Far, 0u};
    bool draws = false;
    bool dispatches = false;
    bool waits = false;
    bool other = false;
    std::uint32_t otherOpcode = 0;
    for (auto at = previous + 1; at < index; ++at) {
        const auto& between = packets[at];
        if (between.Has(LeanPacket::Draw)) {
            draws = true;
        } else if (between.opcode == 0x15 || between.opcode == 0x16) {
            dispatches = true;
        } else if (between.Has(LeanPacket::Wait) || between.Has(LeanPacket::RenderingWait)) {
            waits = true;
        } else if (!between.Has(LeanPacket::Plain)) {
            if (!other) otherOpcode = between.opcode;
            other = true;
        }
    }
    if (draws) return {Gap::Draws, 0u};
    if (dispatches) return {Gap::Dispatches, 0u};
    if (other) return {Gap::Other, otherOpcode};
    if (waits) return {Gap::Waits, 0u};
    return {Gap::Registers, 0u};
}

LabelRunCensus::End labelRunEndAt(const LeanPacket& next) {
    using End = LabelRunCensus::End;
    if (next.Has(LeanPacket::Draw)) return End::Draw;
    if (next.opcode == 0x15 || next.opcode == 0x16) return End::Dispatch;
    if (next.Has(LeanPacket::Wait) || next.Has(LeanPacket::RenderingWait)) return End::Wait;
    if (next.Has(LeanPacket::Plain)) return End::Registers;
    return End::Other;
}

std::optional<std::uint32_t> firstRegisterDifference(const Registers& left, const Registers& right) {
    auto a = left.begin();
    auto b = right.begin();
    while (a != left.end() || b != right.end()) {
        if (a == left.end()) return (*b).first;
        if (b == right.end()) return (*a).first;
        const auto [leftOffset, leftValue] = *a;
        const auto [rightOffset, rightValue] = *b;
        if (leftOffset != rightOffset) return std::min(leftOffset, rightOffset);
        if (leftValue != rightValue) return leftOffset;
        ++a;
        ++b;
    }
    return std::nullopt;
}

std::uint32_t heldValue(const Registers& bank, std::uint32_t offset) {
    return bank.contains(offset) ? bank.at(offset) : 0u;
}

struct RegisterRunsScratch {};

void executeRegisters(std::span<const std::uint32_t> packet, const RegisterListView* list, QueueState& queue, const Submission& submission, std::size_t cursor) {
    static const std::uint64_t every = everyFrom("APS5_VERIFY_REGISTER_RUNS");
    static std::atomic<std::uint64_t> ordinal{0};
    if (every == 0 || ordinal.fetch_add(1, std::memory_order_relaxed) % every != 0) {
        if (list != nullptr) Pm4::ExecuteListedRegisters(packet, list->pairs, queue);
        else Pm4::ExecuteRegisterRun(packet, queue);
        return;
    }
    static std::atomic<std::uint64_t> packetsChecked{0}, listsChecked{0}, differing{0};
    static std::atomic<std::int64_t> window{-1};
    const auto opcode = (packet[0] >> 8u) & 0xffu;
    auto& scratch = HostThreadLocal<QueueState, RegisterRunsScratch>();
    auto& today = Pm4::RegistersFor(scratch, opcode);
    today = Pm4::RegistersFor(queue, opcode);
    scratch.indexType = queue.indexType;
    if (list != nullptr) Pm4::ExecuteIndirectRegisters(packet, list->pairs, scratch);
    else Pm4::Execute(packet, scratch);
    if (list != nullptr) Pm4::ExecuteListedRegisters(packet, list->pairs, queue);
    else Pm4::ExecuteRegisterRun(packet, queue);
    (list != nullptr ? listsChecked : packetsChecked).fetch_add(1, std::memory_order_relaxed);
    auto& fresh = Pm4::RegistersFor(queue, opcode);
    const auto at = firstRegisterDifference(fresh, today);
    const bool same = !at.has_value() && fresh.size() == today.size() && fresh.HeldFingerprint() == today.HeldFingerprint() && queue.indexType == scratch.indexType;
    if (!same) {
        if (differing.fetch_add(1, std::memory_order_relaxed) < 16) {
            const auto name = where(submission, cursor);
            if (at.has_value()) std::fprintf(stderr, "[register-runs-verify] %s: register 0x%x new 0x%08x (%s), today 0x%08x (%s)\n", name.c_str(), *at, heldValue(fresh, *at), fresh.contains(*at) ? "set" : "unset", heldValue(today, *at), today.contains(*at) ? "set" : "unset");
            else if (queue.indexType != scratch.indexType) std::fprintf(stderr, "[register-runs-verify] %s: index type new %u, today %u\n", name.c_str(), queue.indexType, scratch.indexType);
            else if (fresh.size() != today.size()) std::fprintf(stderr, "[register-runs-verify] %s: register count new %zu, today %zu\n", name.c_str(), fresh.size(), today.size());
            else std::fprintf(stderr, "[register-runs-verify] %s: fingerprint differs\n", name.c_str());
        }
        fresh = today;
        queue.indexType = scratch.indexType;
    }
    if (newCounterWindow(window)) std::fprintf(stderr, "[register-runs-verify] %llu register packets checked, %llu register lists checked, %llu differing\n", static_cast<unsigned long long>(packetsChecked.load(std::memory_order_relaxed)), static_cast<unsigned long long>(listsChecked.load(std::memory_order_relaxed)), static_cast<unsigned long long>(differing.load(std::memory_order_relaxed)));
}

bool listPacketVerifyDue() {
    static const std::uint64_t every = everyFrom("APS5_VERIFY_LIST_REGISTER_PACKETS");
    static std::atomic<std::uint64_t> ordinal{0};
    return every != 0 && ordinal.fetch_add(1, std::memory_order_relaxed) % every == 0;
}

void noteListPacketVerify(const char* drains, bool flushDue, const Submission& submission, std::size_t cursor) {
    static std::atomic<std::uint64_t> checked{0}, deferred{0}, moved{0};
    static std::atomic<std::int64_t> window{-1};
    checked.fetch_add(1, std::memory_order_relaxed);
    if (flushDue) deferred.fetch_add(1, std::memory_order_relaxed);
    if (drains != nullptr && moved.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[list-packets-verify] %s: kept on the short path, the verdict asked again drains (%s)\n", where(submission, cursor).c_str(), drains);
    if (newCounterWindow(window)) std::fprintf(stderr, "[list-packets-verify] %llu list packets checked, %llu with a flush due deferred to the next full packet, %llu with a verdict that moved\n", static_cast<unsigned long long>(checked.load(std::memory_order_relaxed)), static_cast<unsigned long long>(deferred.load(std::memory_order_relaxed)), static_cast<unsigned long long>(moved.load(std::memory_order_relaxed)));
}

struct PacketClass {
    std::uint32_t flags = 0;
    LeanPacket::Pipeline pipeline = LeanPacket::Pipeline::Drains;
    LeanPacket::Pipeline shortPipeline = LeanPacket::Pipeline::Drains;
    bool exact = false;
};

constexpr std::size_t ClassSizeLimit = 8;
constexpr std::size_t ClassSizeSplit = 5;

const std::array<PacketClass, 256>& packetClasses() {
    static const std::array<PacketClass, 256> classes = [] {
        std::array<PacketClass, 256> built{};
        std::array<std::uint32_t, ClassSizeLimit> words{};
        for (std::uint32_t opcode = 0; opcode < built.size(); ++opcode) {
            if (opcode == 0x10 || opcode == 0x46 || opcode == 0x50) continue;
            auto& known = built[opcode];
            bool exact = true;
            for (std::size_t size = 2; size <= ClassSizeLimit; ++size) {
                words.fill(0);
                words[0] = 0xc0000000u | (static_cast<std::uint32_t>(size - 2) << 16u) | (opcode << 8u);
                const auto entry = classify(std::span<const std::uint32_t>(words).first(size), 0);
                if (size == 2) {
                    known.flags = entry.flags;
                    known.shortPipeline = entry.pipeline;
                }
                if (size == ClassSizeSplit) known.pipeline = entry.pipeline;
                exact = exact && entry.flags == known.flags && entry.pipeline == (size >= ClassSizeSplit ? known.pipeline : known.shortPipeline);
            }
            known.exact = exact;
        }
        return built;
    }();
    return classes;
}

std::atomic<std::uint64_t> classPacketsChecked{0};
std::atomic<std::uint64_t> classTailsChecked{0};
std::atomic<std::uint64_t> classDiffering{0};
std::atomic<std::int64_t> classWindow{-1};

std::uint64_t packetClassesVerifyEvery() {
    static const std::uint64_t every = everyFrom("APS5_VERIFY_PACKET_CLASSES");
    return every;
}

void notePacketClassCounts() {
    if (!newCounterWindow(classWindow)) return;
    std::fprintf(stderr, "[packet-classes-verify] %llu packets checked, %llu tails checked, %llu differing\n", static_cast<unsigned long long>(classPacketsChecked.load(std::memory_order_relaxed)), static_cast<unsigned long long>(classTailsChecked.load(std::memory_order_relaxed)), static_cast<unsigned long long>(classDiffering.load(std::memory_order_relaxed)));
}

LeanPacket classifyByTable(const Submission& submission, std::span<const std::uint32_t> packet, std::size_t cursor) {
    const auto header = packet[0];
    const auto opcode = (header >> 8u) & 0xffu;
    const auto& known = packetClasses()[opcode];
    if ((header >> 30u) != 3u || !known.exact) return classify(packet, cursor);
    LeanPacket entry;
    entry.cursor = static_cast<std::uint32_t>(cursor);
    entry.words = static_cast<std::uint32_t>(packet.size());
    entry.header = header;
    entry.opcode = static_cast<std::uint8_t>(opcode);
    entry.flags = known.flags;
    entry.pipeline = packet.size() >= ClassSizeSplit ? known.pipeline : known.shortPipeline;
    if ((header & 1u) != 0) entry.flags = (entry.flags & ~LeanPacket::Plain) | LeanPacket::Predicated;
    if (const auto every = packetClassesVerifyEvery(); every != 0) {
        static std::atomic<std::uint64_t> ordinal{0};
        if (ordinal.fetch_add(1, std::memory_order_relaxed) % every == 0) {
            classPacketsChecked.fetch_add(1, std::memory_order_relaxed);
            const auto today = classify(packet, cursor);
            if ((today.flags != entry.flags || today.pipeline != entry.pipeline) && classDiffering.fetch_add(1, std::memory_order_relaxed) < 16) {
                std::fprintf(stderr, "[packet-classes-verify] %s: table flags 0x%x pipeline %u, today flags 0x%x pipeline %u\n", where(submission, cursor).c_str(), entry.flags, static_cast<unsigned>(entry.pipeline), today.flags, static_cast<unsigned>(today.pipeline));
            }
            if (today.flags != entry.flags || today.pipeline != entry.pipeline) entry = today;
            notePacketClassCounts();
        }
    }
    return entry;
}

void verifyTableLists(const Submission& submission, const PacketTable& table) {
    const auto every = packetClassesVerifyEvery();
    static std::atomic<std::uint64_t> ordinal{0};
    if (every == 0 || ordinal.fetch_add(1, std::memory_order_relaxed) % every != 0) return;
    classTailsChecked.fetch_add(1, std::memory_order_relaxed);
    const auto& commands = submission.commands;
    std::size_t index = 0;
    for (std::size_t cursor = 0; cursor < commands.size(); cursor += Pm4::PacketWords(commands[cursor])) {
        const auto header = commands[cursor];
        if ((header >> 30u) != 3u || !Pm4::IndirectRegisterOpcode((header >> 8u) & 0xffu)) continue;
        const bool listed = index < table.lists.size() && table.packets[table.lists[index]].cursor == cursor;
        if (!listed) {
            if (classDiffering.fetch_add(1, std::memory_order_relaxed) < 16) {
                if (index < table.lists.size()) std::fprintf(stderr, "[packet-classes-verify] serial %llu: today's walk finds a register list at DWORD %zu, the table's next list is at DWORD %u\n", static_cast<unsigned long long>(submission.serial), cursor, table.packets[table.lists[index]].cursor);
                else std::fprintf(stderr, "[packet-classes-verify] serial %llu: today's walk finds a register list at DWORD %zu, the table lists no more\n", static_cast<unsigned long long>(submission.serial), cursor);
            }
            notePacketClassCounts();
            return;
        }
        ++index;
    }
    if (index != table.lists.size() && classDiffering.fetch_add(1, std::memory_order_relaxed) < 16) {
        std::fprintf(stderr, "[packet-classes-verify] serial %llu: the table lists a register list at DWORD %u that today's walk does not find\n", static_cast<unsigned long long>(submission.serial), table.packets[table.lists[index]].cursor);
    }
    notePacketClassCounts();
}

}

std::size_t PacketTable::IndexAt(std::size_t cursor) const {
    const auto found = std::lower_bound(packets.begin(), packets.end(), cursor, [](const LeanPacket& entry, std::size_t at) { return entry.cursor < at; });
    return static_cast<std::size_t>(found - packets.begin());
}

SegmentSlot& SegmentSlots::Take() {
    for (auto& slot : slots) {
        if (slot->busy) continue;
        slot->busy = true;
        ++slot->generation;
        return *slot;
    }
    slots.push_back(std::make_unique<SegmentSlot>());
    LeanFront::AddOnWorker(Counter::SlotsCreated);
    auto& slot = *slots.back();
    slot.busy = true;
    slot.generation = 1;
    return slot;
}

void Driver::scanSegment(const Submission& submission, PacketTable& table, bool validating, const std::uint32_t* guest, bool byTable, bool quiet) {
    table.Clear();
    const auto& commands = submission.commands;
    std::vector<std::size_t> guarded;
    for (std::size_t cursor = 0; cursor < commands.size();) {
        if (validating) std::erase_if(guarded, [cursor](std::size_t end) { return end <= cursor; });
        const auto header = commands[cursor];
        if (Pm4::FillerPacket(header)) {
            ++cursor;
            continue;
        }
        const auto count = Pm4::PacketWords(header);
        const auto opcode = (header >> 8u) & 0xffu;
        if (validating) {
            bool valid = (header & 0xc0000000u) == 0xc0000000u && count <= commands.size() - cursor;
            if (valid) {
                try {
                    Pm4::Validate(std::span(commands).subspan(cursor, count), submission.queue);
                    if (header == FlipPacketHeader && !guarded.empty()) valid = false;
                    if (opcode == 0x22u) {
                        const auto end = submission.conditionalEnds.find(cursor);
                        if (end == submission.conditionalEnds.end()) valid = false;
                        else guarded.push_back(end->second);
                    }
                } catch (const std::exception&) {
                    valid = false;
                }
            }
            if (!valid) {
                if (!quiet) validate(submission, guest);
                throw std::runtime_error("AGC driver: " + Pm4::Name(header) + " at DWORD " + std::to_string(cursor) + ": the lean packet scan refused a packet today's validation accepts");
            }
        }
        require(count <= commands.size() - cursor, "truncated PM4 packet");
        if (byTable) {
            if (validating && (header >> 30u) == 3u && Pm4::IndirectRegisterOpcode(opcode)) table.lists.push_back(static_cast<std::uint32_t>(table.packets.size()));
            table.packets.push_back(classifyByTable(submission, std::span(commands).subspan(cursor, count), cursor));
        } else {
            table.packets.push_back(classify(std::span(commands).subspan(cursor, count), cursor));
        }
        const bool type3 = (header >> 30u) == 3u;
        if (header == FlipPacketHeader || header == RenderingWaitPacketHeader || (type3 && (opcode == 0x49 || opcode == 0x37 || opcode == 0x3c || opcode == 0x93))) table.outputs.push_back(static_cast<std::uint32_t>(cursor));
        if (header == FlipPacketHeader) ++table.flipHeaders;
        cursor += count;
    }
    for (auto& entry : table.packets) {
        if (entry.opcode != 0x22 && !(entry.opcode == 0x3f && entry.Has(LeanPacket::Predicated))) continue;
        const auto end = submission.conditionalEnds.find(entry.cursor);
        if (end != submission.conditionalEnds.end()) entry.skip = static_cast<std::uint32_t>(table.IndexAt(end->second));
    }
    if (byTable && validating) verifyTableLists(submission, table);
}

void Driver::markOutputs(const Submission& submission, PacketTable& table) {
    const auto mark = [&](std::size_t cursor, std::uint32_t flag) {
        const auto index = table.IndexAt(cursor);
        if (index < table.packets.size() && table.packets[index].cursor == cursor) {
            table.packets[index].flags |= flag;
            table.packets[index].flags &= ~LeanPacket::Plain;
        }
    };
    for (const auto& [cursor, request] : submission.flips) mark(cursor, LeanPacket::Flip);
    for (const auto cursor : submission.heldAtSubmit) mark(cursor, LeanPacket::Held);
    table.built = true;
}

void Driver::verifyOutputs(const Submission& submission, const PacketTable& table) {
    const auto listed = [&](std::size_t cursor) { return std::binary_search(table.outputs.begin(), table.outputs.end(), static_cast<std::uint32_t>(cursor)); };
    for (const auto& [cursor, request] : submission.flips) {
        if (!listed(cursor)) LeanFront::NoteDifference(LeanFront::Field::Flips, where(submission, cursor) + ": today's walk reserved a flip the packet table does not list");
    }
    for (const auto& [cursor, wait] : submission.renderingWaits) {
        if (!listed(cursor)) LeanFront::NoteDifference(LeanFront::Field::RenderingWaits, where(submission, cursor) + ": today's walk captured a rendering wait the packet table does not list");
    }
    for (const auto cursor : submission.heldAtSubmit) {
        if (!listed(cursor)) LeanFront::NoteDifference(LeanFront::Field::HeldWaits, where(submission, cursor) + ": today's walk held a wait at submission the packet table does not list");
    }
    const auto writes = labelWritesAt(submission, table.outputs);
    if (writes != submission.labelWrites) {
        LeanFront::NoteDifference(LeanFront::Field::LabelWrites, "serial " + std::to_string(submission.serial) + ": today's walk noted " + std::to_string(submission.labelWrites.size()) + " label write dwords, the packet table's positions " + std::to_string(writes.size()));
    }
}

void Driver::verifySegmentTable(const Submission& submission, const PacketTable& table) {
    PacketTable fresh;
    scanSegment(submission, fresh);
    markOutputs(submission, fresh);
    std::size_t index = 0;
    const auto& commands = submission.commands;
    for (std::size_t cursor = 0; cursor < commands.size();) {
        const auto header = commands[cursor];
        if (Pm4::FillerPacket(header)) {
            ++cursor;
            continue;
        }
        const auto count = Pm4::PacketWords(header);
        const auto packet = std::span(commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        if (index >= table.packets.size()) {
            LeanFront::NoteDifference(LeanFront::Field::PacketTable, where(submission, cursor) + ": the packet table ends before this packet");
            return;
        }
        const auto& entry = table.packets[index];
        const auto expect = [&](bool same, const char* what) {
            if (!same) LeanFront::NoteDifference(LeanFront::Field::PacketTable, where(submission, cursor) + ": the packet table's " + what + " differs from today's reading");
        };
        expect(entry.cursor == cursor && entry.words == count && entry.header == header && entry.opcode == opcode, "position");
        expect(entry.Has(LeanPacket::Predicated) == Pm4::Predicated(header), "predication");
        expect(entry.Has(LeanPacket::Draw) == Pm4::DrawOpcode(opcode) && entry.Has(LeanPacket::IndirectDraw) == Pm4::IndirectDrawOpcode(opcode), "draw kind");
        expect(entry.Has(LeanPacket::FlipHeader) == (header == FlipPacketHeader) && entry.Has(LeanPacket::RenderingWait) == (header == RenderingWaitPacketHeader), "flip header");
        expect(entry.Has(LeanPacket::Flip) == submission.flips.contains(cursor), "flip reservation");
        expect(entry.Has(LeanPacket::Held) == submission.heldAtSubmit.contains(cursor), "held wait");
        expect(entry.Has(LeanPacket::KeepsFinisher) == PacketRules::keepsDrawFinisher(header, opcode), "finisher verdict");
        expect(entry.Has(LeanPacket::Seal) == PacketRules::sealsWriteOnly(packet, opcode), "seal");
        expect(entry.Has(LeanPacket::IndirectRegisters) == Pm4::IndirectRegisterOpcode(opcode), "register list kind");
        if (entry.pipeline == LeanPacket::Pipeline::Keeps || entry.pipeline == LeanPacket::Pipeline::Drains) {
            expect((entry.pipeline == LeanPacket::Pipeline::Keeps) == PacketRules::keepsDrawPipeline(packet, header, opcode), "committer verdict");
        }
        if (opcode == 0x22 || (opcode == 0x3f && Pm4::Predicated(header))) {
            const auto end = submission.conditionalEnds.find(cursor);
            const auto skip = end == submission.conditionalEnds.end() ? LeanPacket::NoSkip : static_cast<std::uint32_t>(table.IndexAt(end->second));
            expect(entry.skip == skip && (skip == LeanPacket::NoSkip || skip >= table.packets.size() || table.packets[skip].cursor >= end->second), "conditional end");
        }
        if (index < fresh.packets.size()) expect(entry.flags == fresh.packets[index].flags && entry.pipeline == fresh.packets[index].pipeline, "classes");
        ++index;
        cursor += count;
    }
    if (index != table.packets.size()) LeanFront::NoteDifference(LeanFront::Field::PacketTable, "serial " + std::to_string(submission.serial) + ": the packet table lists " + std::to_string(table.packets.size()) + " packets, today's walk " + std::to_string(index));
    if (table.outputs != fresh.outputs) LeanFront::NoteDifference(LeanFront::Field::PacketTable, "serial " + std::to_string(submission.serial) + ": the packet table's output positions differ from a fresh scan");
}

bool Driver::flushDue(FlushGate& gate, FlushInputs& inputs) {
    inputs.given = false;
    if (gate.everyPacket || !deferredLabels().labels.empty()) return true;
    inputs.pending = Graphics::Recorder::PendingLabelSince();
    inputs.completions = completionsPending();
    inputs.work = Graphics::Recorder::RecordedWorkSinceSubmit();
    inputs.reap = lastReapTries[0];
    inputs.given = true;
    if (gate.cap != 0 && inputs.work >= gate.cap) return true;
    const auto& pending = inputs.pending;
    const bool completions = inputs.completions;
    if (!pending.has_value() && !completions) return false;
    if (pending.has_value() && (!gate.labelArmed || gate.label != *pending)) {
        gate.label = *pending;
        gate.labelArmed = true;
        gate.labelDue = FlushTicksAt(*pending + labelFlushDeadline());
    }
    if (completions && (!gate.reapArmed || gate.reap != inputs.reap)) {
        gate.reap = inputs.reap;
        gate.reapArmed = true;
        gate.reapDue = FlushTicksAt(gate.reap + ReapInterval);
    }
    const auto now = FlushTicks();
    return (pending.has_value() && now >= gate.labelDue) || (completions && now >= gate.reapDue);
}

void Driver::checkFlush(const Submission& submission, std::size_t cursor, const FlushGate& gate, bool gateDue, std::uint32_t decision, const FlushInputs& inputs) {
    using LeanFront::Allowed;
    using LeanFront::Field;
    const bool acts = (decision & FlushDecision::Acts) != 0;
    if (!acts) {
        if (gateDue && decision == 0) LeanFront::NoteAllowed(Allowed::EarlyFlushChecks);
        return;
    }
    if (gateDue) return;
    const auto late = [&](std::chrono::steady_clock::time_point due) { return std::chrono::duration_cast<std::chrono::nanoseconds>(inputs.now - due).count(); };
    const auto detail = [&](const char* what, std::int64_t lateNs) { return where(submission, cursor) + ": today's flush " + what + " where the lean deadline gate skipped the check" + (lateNs >= 0 ? " (" + std::to_string(lateNs) + " ns past the deadline)" : std::string()); };
    if ((decision & (FlushDecision::Record | FlushDecision::RecordAtLock)) != 0) {
        LeanFront::NoteDifference(Field::FlushRecords, detail("records queued labels", -1));
    } else if ((decision & FlushDecision::Capped) != 0) {
        if (Graphics::RuntimeSwitches::Generation() != gate.capGeneration) LeanFront::NoteAllowed(Allowed::CapMoved);
        else LeanFront::NoteDifference(Field::FlushCaps, detail("submits at the batch cap", -1));
    } else if ((decision & FlushDecision::Submit) != 0) {
        LeanFront::NoteDifference(Field::FlushSubmits, detail("submits a pending label", inputs.pending.has_value() ? late(*inputs.pending + labelFlushDeadline()) : -1));
    } else {
        LeanFront::NoteDifference(Field::FlushReaps, detail("reaps completions", late(inputs.reap + ReapInterval)));
    }
}

void Driver::releaseSegmentSlot(SegmentSlot& slot) {
    if (slot.prepared) {
        DrawFinisher::CheckRawReaders(&slot.own, "a lean segment slot released");
        recycleLeanTail(slot.own);
        auto labelWrites = std::move(slot.own.labelWrites);
        labelWrites.clear();
        slot.own = Submission{};
        slot.own.labelWrites = std::move(labelWrites);
        slot.prepared = false;
    }
    slot.table.Clear();
    slot.mark = 0;
    slot.busy = false;
}

void Driver::executeLean(const Submission& first) {
    struct ChainReport {
        bool reported = false;
        bool previous = false;
        ThreadReport::Clock::time_point start{};
        explicit ChainReport(bool enabled) : reported(enabled) {
            if (reported) previous = ThreadReport::EnterQueue0Worker(true);
        }
        ChainReport(const ChainReport&) = delete;
        ChainReport& operator=(const ChainReport&) = delete;
        void Segment() {
            if (!reported) return;
            const auto now = ThreadReport::Clock::now();
            if (start != ThreadReport::Clock::time_point{}) ThreadReport::WorkerSubmission(start, now);
            start = now;
        }
        ~ChainReport() {
            if (!reported) return;
            if (start != ThreadReport::Clock::time_point{}) ThreadReport::WorkerSubmission(start, ThreadReport::Clock::now());
            ThreadReport::LeaveQueue0Worker(previous);
        }
    } report{ThreadReport::Enabled()};
    const ThreadReport::WorkerStateScope executing(ThreadReport::WorkerState::Packets, true);
    FlipHold flipHold{*this, first.queue, first.holdsFlip, first.holdsFlip, first.flipWork};
    const LeanFront::SegmentNesting nesting(&frontLatch);
    DrawPipeline::Active() = true;
    struct PipelineDrain {
        ~PipelineDrain() {
            if (std::uncaught_exceptions() == 0) return;
            try {
                DrawFinisher::NoteDeepKept(DeepKept::Submission);
                DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
            } catch (...) {
            }
        }
    } pipelineDrain;
    QueueState* state = nullptr;
    {
        std::lock_guard lock(mutex);
        if (resetGraphics) {
            queues.erase(0);
            resetGraphics = false;
        }
        state = &queues[0];
    }
    struct ClearAhead {
        ~ClearAhead() { clearResolvedAhead(); }
    } clearAhead;
    struct HeldSlots {
        Driver& driver;
        std::vector<SegmentSlot*> slots;
        ~HeldSlots() {
            for (auto* slot : slots) driver.releaseSegmentSlot(*slot);
        }
    } held{*this, {}};
    struct SettleFinisher {
        const Submission* first;
        ~SettleFinisher() {
            if (DrawFinisher::FrameFlow()) {
                if (std::uncaught_exceptions() == 0) {
                    const auto start = DrawFinisher::FrameCensus() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    const bool waited = DrawFinisher::Get().AwaitRawReaders();
                    if (waited) LeanFront::AddOnWorker(Counter::SettlesWaited);
                    DrawFinisher::NoteFrameSite(waited ? FrameSite::ExitWait : FrameSite::ExitFree, !waited || start == std::chrono::steady_clock::time_point{} ? 0 : static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
                    DrawFinisher::CheckRawReaders(first, "the submission at the exit of the lean loop");
                    return;
                }
                DrawFinisher::NoteFrameKept(FrameKept::Exception);
            }
            if (DrawFinisher::Busy()) LeanFront::AddOnWorker(Counter::SettlesWaited);
            DrawFinisher::NoteDeepKept(DeepKept::Settle);
            DrawFinisher::Get().Settle();
        }
    } settleFinisher{&first};
    struct PacketProfileStorage {};
    LeanRun run{HostThreadLocal<PacketProfile, PacketProfileStorage>(), PacketCounters::ThisThread(), {}, packetEpoch(), false, 0};
    run.gate.everyPacket = everyPacketFlush();
    held.slots.reserve(8);
    auto* slot = &segmentSlots.Take();
    held.slots.push_back(slot);
    const Submission* segment = &first;
    bool verify = first.front == FrontMode::Lean ? first.verifySegment : LeanFront::Due(frontLatch, LeanFront::Item::Segment);
    bool head = true;
    for (;;) {
        report.Segment();
        FlipHold quiet{*this, segment->queue, false, false, nullptr};
        auto& hold = head ? flipHold : quiet;
        runLeanSegment(*segment, *state, *slot, hold, run, verify);
        const bool follows = endSegment<true>(*segment, hold, true, FrontMode::Lean);
        slot->mark = DrawFinisher::Get().Issued();
        if (!follows) break;
        const Submission& stalled = *segment;
        std::optional<PerformanceTimer> rewindTiming;
        if (timersWanted()) rewindTiming.emplace("Driver.Rewind");
        static const bool ring = ReadSubmissionRingSettings();
        if (ring) PrepareSubmissionRing(stalled.queue);
        const auto front = latchFront(stalled.queue, false);
        awaitRewindTail(stalled, front);
        if (front != FrontMode::Lean) {
            if (DrawFinisher::FrameFlowWanted() && DrawFinisher::Busy()) {
                DrawFinisher::NoteFrameKept(FrameKept::FrontChange);
                DrawFinisher::Get().Drain();
            }
            Submission tail{};
            const auto retiredTakes = prepareTail(stalled, tail, front, ring, nullptr, false);
            if (rewindTiming) rewindTiming->Finish();
            retireLabelTakes(tail.queue, retiredTakes);
            execute(tail);
            break;
        }
        auto& next = segmentSlots.Take();
        held.slots.push_back(&next);
        next.prepared = true;
        const bool verifyNext = LeanFront::Due(frontLatch, LeanFront::Item::Segment);
        const auto retiredTakes = prepareTail(stalled, next.own, front, ring, &next.table, verifyNext);
        next.own.slotGeneration = next.generation;
        if (rewindTiming) rewindTiming->Finish();
        retireLabelTakes(next.own.queue, retiredTakes);
        LeanFront::AddOnWorker(Counter::SegmentsLooped);
        const auto finished = DrawFinisher::Get().Front();
        std::size_t kept = 0;
        for (auto* heldSlot : held.slots) {
            if (heldSlot != &next && (!heldSlot->prepared || heldSlot->mark <= finished)) {
                releaseSegmentSlot(*heldSlot);
                continue;
            }
            held.slots[kept++] = heldSlot;
        }
        held.slots.resize(kept);
        if (kept > 1) LeanFront::AddOnWorker(Counter::SlotsHeld, kept - 1);
        segment = &next.own;
        slot = &next;
        verify = verifyNext;
        head = false;
    }
}

void Driver::runLeanSegment(const Submission& submission, QueueState& queue, SegmentSlot& slot, FlipHold& flipHold, LeanRun& run, bool verify) {
    includeTimingSubmission(submission, true);
    DrawPipeline::Active() = true;
    if (!orderedCompletion()) {
        DrawFinisher::NoteDeepKept(DeepKept::Submission);
        DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
    }
    if (const bool unordered = LeanFront::SwitchSet(frontLatch, Graphics::RuntimeSwitch::UnorderedDrawKeys); unordered != unorderedDrawKeys.load(std::memory_order_relaxed)) {
        DrawFinisher::NoteDeepKept(DeepKept::Submission);
        DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
        unorderedDrawKeys.store(unordered, std::memory_order_relaxed);
    }
    if (traceGpu()) std::fprintf(stderr, "[gpu] %.1f execute serial=%llu queue=0x%x dwords=%zu\n", TraceMs(), static_cast<unsigned long long>(submission.serial), submission.queue, submission.commands.size());
    PacketRules::dumpQueue(submission);
    ++run.packetProfile.submissions;
    if (epochEach() || submission.enqueuedAt == std::chrono::steady_clock::time_point{} || submission.enqueuedAt > lastEpochBump()) {
        if (!epochEach()) noteReleasedTails(submission, FrontMode::Lean);
        bumpEpoch(&EpochBumps::submissions);
    }
    clearResolvedAhead();
    gateOpened() = true;
    prepareLeanDraws();
    DrawFinisher::Get().LatchFrameFlow();
    if (!slot.table.built) {
        scanSegment(submission, slot.table, false, nullptr, LeanFront::SwitchSet(frontLatch, Graphics::RuntimeSwitch::PacketClassTable));
        markOutputs(submission, slot.table);
        LeanFront::AddOnWorker(Counter::PacketTables);
    }
    if (verify) {
        verifySegmentTable(submission, slot.table);
        if (submission.front != FrontMode::Lean) LeanFront::NoteChecked(LeanFront::Item::Segment);
    }
    run.gate.capGeneration = Graphics::RuntimeSwitches::Generation();
    run.gate.cap = batchCap();
    const auto skipped = run.gate.skipped;
    const auto calls = run.gate.calls;
    run.report = reportWanted();
    if (timersWanted()) runLeanPackets<true>(submission, queue, slot.table, flipHold, run);
    else runLeanPackets<false>(submission, queue, slot.table, flipHold, run);
    LeanFront::AddOnWorker(Counter::FlushChecksSkipped, run.gate.skipped - skipped);
    LeanFront::AddOnWorker(Counter::FlushCalls, run.gate.calls - calls);
}

template <bool Timers>
void Driver::runLeanPackets(const Submission& submission, QueueState& queue, const PacketTable& table, FlipHold& flipHold, LeanRun& run) {
    using Graphics::RuntimeSwitch;
    struct NoContext {
        explicit NoContext(FrameTiming*) {}
    };
    struct NoTimer {
        explicit NoTimer(const char*) {}
        void Mark(const char*, std::uint64_t = 0) {}
        void Finish() {}
        void FinishPacket(std::uint64_t, std::size_t, std::uint32_t, std::size_t) {}
    };
    using Context = std::conditional_t<Timers, PerformanceContext, NoContext>;
    using Timer = std::conditional_t<Timers, PerformanceTimer, NoTimer>;
    const bool report = run.report;
    const bool profilePackets = report && profileDraws();
    PacketHistory recent{submission.commands};
    auto& packetProfile = run.packetProfile;
    auto& packetCounters = run.packetCounters;
    struct SealState {
        bool drawn = PacketRules::drawnSinceSeal();
        ~SealState() { PacketRules::drawnSinceSeal() = drawn; }
    } seal;
    const auto timedWork = [report](double WorkerProfile::*bucket, auto&& work) {
        if (report) timed(bucket, work);
        else work();
    };
    std::size_t listHint = 0;
    std::uint32_t tagged = NoTag;
    const auto& packets = table.packets;
    const auto& commands = submission.commands;
    const auto skipTo = [&](const LeanPacket& entry) { return entry.skip != LeanPacket::NoSkip ? static_cast<std::size_t>(entry.skip) : table.IndexAt(submission.conditionalEnds.at(entry.cursor)); };
    struct LeanRunEnd {
        Driver& driver;
        ~LeanRunEnd() {
            try {
                driver.publishLeanRun();
            } catch (...) {
                driver.leanRun.count = 0;
                DrawFinisher::Get().SetOpenRun(nullptr);
            }
        }
    } leanRunEnd{*this};
    const bool plainRegisters = LeanFront::SwitchSet(frontLatch, RuntimeSwitch::PlainRegisterPackets);
    const bool registerRuns = LeanFront::SwitchSet(frontLatch, RuntimeSwitch::RegisterRuns);
    const bool listRegisters = LeanFront::SwitchSet(frontLatch, RuntimeSwitch::ListRegisterPackets);
    const bool labelRuns = LeanFront::SwitchSet(frontLatch, RuntimeSwitch::LabelRuns);
    std::size_t lastQueuedLabel = NoQueuedLabel;
    std::uint64_t plainDone = 0;
    struct PlainProgress {
        Driver& driver;
        std::uint64_t& done;
        ~PlainProgress() {
            if (done != 0) driver.packetsDone.fetch_add(done, std::memory_order_seq_cst);
        }
    } plainProgress{*this, plainDone};
    for (std::size_t index = 0; index < packets.size();) {
        const auto& entry = packets[index];
        if (run.packetEpoch) bumpEpoch(&EpochBumps::packets);
        if (failed.load(std::memory_order_acquire) || stopping.load(std::memory_order_acquire) || (++run.packets & 63u) == 0) CheckFailure();
        const std::size_t cursor = entry.cursor;
        const std::uint32_t header = entry.header;
        const std::size_t count = entry.words;
        const auto packet = std::span(commands).subspan(cursor, count);
        const std::uint32_t opcode = entry.opcode;
        auto next = index + 1;
        const bool predicated = entry.Has(LeanPacket::Predicated) && queue.predication.operation != 0;
        const bool flip = entry.Has(LeanPacket::Flip);
        const auto asksVerdict = [&] { return (entry.pipeline == LeanPacket::Pipeline::Asks || entry.pipeline == LeanPacket::Pipeline::AsksLists) && !entry.Has(LeanPacket::FlipHeader) && !entry.Has(LeanPacket::RenderingWait) && !flip; };
        const auto noteKept = [&] {
            if (!DrawFinisher::Deep()) return;
            const auto kept = entry.Has(LeanPacket::FlipHeader) || flip ? DeepKept::Flip : entry.Has(LeanPacket::RenderingWait) ? DeepKept::RenderingWait : entry.Has(LeanPacket::Wait) ? DeepKept::Wait : opcode == 0x49 || opcode == 0x37 ? DeepKept::WorkerLabel : DeepKept::PacketVerdict;
            DrawFinisher::NoteDeepKept(kept, opcode);
        };
        struct TagReset {
            std::uint32_t& tagged;
            bool plain;
            ~TagReset() {
                if (!plain) tagged = NoTag;
            }
        } tagReset{tagged, entry.Has(LeanPacket::Plain) && !predicated};
        if (entry.Has(LeanPacket::Draw) && !entry.Has(LeanPacket::IndirectDraw) && !predicated && LeanFront::SwitchSet(frontLatch, RuntimeSwitch::XpKeepDraws) && deferredLabels().labels.empty() && PacketRules::xpDropsDraw()) {
            recent.Record(cursor);
            index = next;
            continue;
        }
        if ((opcode == 0x15u || opcode == 0x16u) && !predicated && LeanFront::SwitchSet(frontLatch, RuntimeSwitch::XpSkipDispatches)) {
            recent.Record(cursor);
            index = next;
            continue;
        }
        const bool verifyPacket = LeanFront::Due(frontLatch, LeanFront::Item::Packet);
        if constexpr (!Timers) {
            if (plainRegisters && !verifyPacket && !profilePackets && entry.Has(LeanPacket::Plain) && entry.Has(LeanPacket::KeepsFinisher) && entry.pipeline == LeanPacket::Pipeline::Keeps && !entry.Has(LeanPacket::IndirectRegisters) && (opcode == 0x69 || opcode == 0x76 || opcode == 0x79) && !run.gate.everyPacket && deferredLabels().labels.empty()) {
                if (plainVerifyDue()) {
                    FlushInputs plainInputs;
                    const bool drains = (DrawFinisher::Busy() && !PacketRules::keepsDrawFinisher(header, opcode)) || (DrawPipeline::Queue0().Busy() && !PacketRules::keepsDrawPipeline(packet, header, opcode));
                    notePlainVerify(drains, flushDue(run.gate, plainInputs), submission, cursor);
                }
                if (opcode != tagged) {
                    GuestMemory::SetCurrentPacket(opcode, submission.queue);
                    tagged = opcode;
                }
                if (report) CaptureTrace::Log("packet submission=%llu queue=%x offset=%zu header=%08x words=%zu", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, header, packet.size());
                recent.Record(cursor);
                ++plainDone;
                if (registerRuns) executeRegisters(packet, nullptr, queue, submission, cursor);
                else Pm4::Execute(packet, queue);
                ++run.gate.skipped;
                index = next;
                continue;
            }
            if (listRegisters && !verifyPacket && !profilePackets && entry.Has(LeanPacket::Plain) && entry.Has(LeanPacket::IndirectRegisters) && entry.pipeline == LeanPacket::Pipeline::AsksLists && !predicated && submission.leanLists && !run.gate.everyPacket && deferredLabels().labels.empty()) {
                const auto* ref = submission.lists.Next(cursor, listHint);
                if (ref != nullptr && (ref->flags & RegisterLists::Valid) != 0) {
                    bool keeps = true;
                    bool asked = false;
                    if (DrawFinisher::Busy()) {
                        if (!DrawFinisher::Overlap()) {
                            keeps = false;
                        } else {
                            keeps = PacketRules::keepsDrawPipeline(packet, header, opcode);
                            asked = true;
                        }
                    }
                    if (keeps && !asked && DrawPipeline::Queue0().Busy()) keeps = PacketRules::keepsDrawPipeline(packet, header, opcode);
                    if (keeps) {
                        if (asked) DrawFinisher::NoteDrain(DrawFinisher::DrainSite::Packet, false);
                        if (listPacketVerifyDue()) {
                            FlushInputs listInputs;
                            const char* drains = nullptr;
                            if (DrawFinisher::Busy() && (!DrawFinisher::Overlap() || !PacketRules::keepsDrawPipeline(packet, header, opcode))) drains = "finisher";
                            else if (DrawPipeline::Queue0().Busy() && !PacketRules::keepsDrawPipeline(packet, header, opcode)) drains = "committer";
                            noteListPacketVerify(drains, flushDue(run.gate, listInputs), submission, cursor);
                        }
                        if (opcode != tagged) {
                            GuestMemory::SetCurrentPacket(opcode, submission.queue);
                            tagged = opcode;
                        }
                        if (report) CaptureTrace::Log("packet submission=%llu queue=%x offset=%zu header=%08x words=%zu", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, header, packet.size());
                        recent.Record(cursor);
                        ++plainDone;
                        const auto view = submission.lists.View(*ref);
                        if (registerRuns) executeRegisters(packet, &view, queue, submission, cursor);
                        else Pm4::ExecuteIndirectRegisters(packet, view.pairs, queue);
                        ++run.gate.skipped;
                        index = next;
                        continue;
                    }
                }
            }
        }
        if (plainDone != 0) packetsDone.fetch_add(std::exchange(plainDone, 0), std::memory_order_seq_cst);
        PacketRules::FlipForm flipKind = PacketRules::FlipForm::Kept;
        double abSeconds = 0;
        const bool orderedFlip = flip && flipOrdered(packet, header, opcode, predicated, flipKind, abSeconds);
        const bool orderedWait = entry.Has(LeanPacket::RenderingWait) && !predicated && DrawFinisher::FrameFlow();
        const bool ordered = orderedFlip || orderedWait;
        const auto pipelineVerdict = [&] {
            switch (entry.pipeline) {
                case LeanPacket::Pipeline::Keeps: return true;
                case LeanPacket::Pipeline::Drains: return false;
                default: return PacketRules::keepsDrawPipeline(packet, header, opcode);
            }
        };
        bool listVerdict = false;
        bool listVerdictKnown = false;
        bool dmaBeside = false;
        DrawPipeline::Range argumentRange{};
        const bool argumentWait = opcode == 0x16 && argumentWaitWanted(packet, queue, predicated, argumentRange);
        DrawPipeline::Range indirectRange{};
        const bool indirectQueued = opcode == 0x16 && !argumentWait && orderedIndirectWanted(packet, queue, predicated, indirectRange);
        if (DrawFinisher::Busy() && !entry.Has(LeanPacket::KeepsFinisher)) {
            const bool queuesLabel = !predicated && !entry.Has(LeanPacket::FlipHeader) && !entry.Has(LeanPacket::RenderingWait) && !flip && (opcode == 0x49 || opcode == 0x37) && labelPacketQueues(packet, opcode);
            const bool heldLabelWait = !predicated && orderedWaits() && entry.Has(LeanPacket::Wait) && !entry.Has(LeanPacket::RenderingWait) && PacketRules::waitTakesPendingLabel(packet);
            const bool queuesSampleDump = !predicated && opcode == 0x46 && (LeanFront::SwitchSet(frontLatch, RuntimeSwitch::BesideSampleDumps) || DrawFinisher::Flow() || DrawFinisher::Deep()) && sampleDumpQueues(packet, opcode, submission.queue);
            if (!predicated && opcode == 0x50 && DrawFinisher::Overlap() && DrawFinisher::FrameFlow()) {
                const auto form = dmaPacketForm(packet, opcode, submission.queue);
                dmaBeside = form == PacketRules::DmaForm::Queues;
                if (dmaBeside) DrawFinisher::NoteFrameSite(FrameSite::DmaBeside);
                else if (form == PacketRules::DmaForm::Copy) DrawFinisher::NoteFrameKept(FrameKept::DmaCopy);
                else if (form == PacketRules::DmaForm::Gds) DrawFinisher::NoteFrameKept(FrameKept::DmaGds);
            }
            bool keeps = false;
            const bool dispatchBeside = opcode == 0x15 && (LeanFront::SwitchSet(frontLatch, RuntimeSwitch::DispatchBeside) || DrawFinisher::Flow() || DrawFinisher::Deep());
            if (DrawFinisher::Overlap() && (opcode != 0x15 || dispatchBeside) && !queuesLabel && !heldLabelWait && !queuesSampleDump && !dmaBeside) {
                keeps = pipelineVerdict();
                const bool awaited = !keeps && DrawFinisher::Deep() && asksVerdict() && PacketRules::awaitVerdictFinisherWriters(packet, header, opcode);
                if (entry.pipeline == LeanPacket::Pipeline::AsksLists) {
                    listVerdict = keeps;
                    listVerdictKnown = !awaited;
                }
                keeps = keeps || awaited;
                if (!keeps && argumentWait) {
                    DrawFinisher::Get().AwaitExactPending({&argumentRange, 1}, DeepSite::IndirectArguments, FrameSite::ArgumentWait);
                    keeps = true;
                }
                if (!keeps && indirectQueued) keeps = true;
            }
            const bool beside = DrawFinisher::Overlap() && (opcode != 0x15 || dispatchBeside) && (queuesLabel || heldLabelWait || queuesSampleDump || dmaBeside || keeps);
            if (!beside && !ordered && !LeanFront::SwitchSet(frontLatch, RuntimeSwitch::XpNoPacketDrains) && !DrawFinisher::FlowSkips(DrawFinisher::FlowSite::PacketFinisher)) {
                if (opcode == 0x16 && DrawFinisher::FrameFlow()) DrawFinisher::NoteFrameKept(FrameKept::IndirectGpu);
                noteKept();
                DrawFinisher::Get().Drain(true);
                listVerdictKnown = false;
            }
            if (DrawFinisher::Overlap()) {
                DrawFinisher::NoteDrain(DrawFinisher::DrainSite::Packet, !beside && !ordered);
                if (!beside && !ordered) DrawFinisher::NoteDrainedOpcode(entry.Has(LeanPacket::FlipHeader) ? 0xffu : opcode, pipelineVerdict());
            }
        }
        if constexpr (Timers) {
            if (APS5_ENABLE_TIMING_LOG && pendingFrameTiming == nullptr) includeTimingSubmission(submission, false);
        }
        Context timingContext(pendingFrameTiming.get());
        const SampledHitch hitch("worker packet", opcode, !entry.Has(LeanPacket::Wait) && !entry.Has(LeanPacket::RenderingWait));
        Timer timing("Driver.Packet");
        if (!predicated && entry.Has(LeanPacket::LabelIssue)) noteLabelIssue(packet, submission.queue);
        if (!predicated && !entry.Has(LeanPacket::FlipHeader) && !entry.Has(LeanPacket::RenderingWait) && !flip && (entry.Has(LeanPacket::Enqueues) || verifyPacket)) {
            if constexpr (!Timers) {
                if (labelRuns && submission.queue == 0 && entry.Has(LeanPacket::Enqueues) && (opcode == 0x49 || opcode == 0x37) && DrawFinisher::Overlap() && labelPacketQueues(packet, opcode)) {
                    using End = LabelRunCensus::End;
                    std::optional<End> stop;
                    std::uint32_t stopOpcode = 0;
                    auto end = index + 1;
                    auto counted = run.packets;
                    while (!stop) {
                        if (end >= packets.size()) {
                            stop = End::SegmentEnd;
                            break;
                        }
                        if (end - index >= LabelRunCap) {
                            stop = End::Cap;
                            break;
                        }
                        const auto& follower = packets[end];
                        const auto words = std::span(commands).subspan(follower.cursor, follower.words);
                        if (follower.opcode != 0x49 && follower.opcode != 0x37) {
                            stop = labelRunEndAt(follower);
                            stopOpcode = follower.opcode;
                        } else if (run.packetEpoch || failed.load(std::memory_order_acquire) || stopping.load(std::memory_order_acquire) || ((counted + 1) & 63u) == 0) {
                            stop = End::Check;
                        } else if ((follower.Has(LeanPacket::Predicated) && queue.predication.operation != 0) || follower.Has(LeanPacket::Flip) || follower.Has(LeanPacket::FlipHeader) || follower.Has(LeanPacket::RenderingWait) || !follower.Has(LeanPacket::Enqueues) || follower.Has(LeanPacket::Plain) || follower.Has(LeanPacket::Draw) || follower.Has(LeanPacket::Wait) || follower.Has(LeanPacket::IndirectRegisters) || !DrawFinisher::Overlap() || !labelPacketQueues(words, follower.opcode)) {
                            stop = End::Unqueued;
                        } else if (releasesInterrupt(words, follower.opcode)) {
                            stop = End::Interrupt;
                        } else {
                            ++end;
                            ++counted;
                        }
                    }
                    if (end == index + 1) {
                        if (LabelRunCensus::Enabled()) LabelRunCensus::NoteRun(1, 1, stop.value_or(End::SegmentEnd), stopOpcode);
                    } else {
                        const auto interrupt = releaseInterrupt(packet, opcode, false);
                        auto labels = std::make_shared<LabelRun>();
                        std::vector<DrawPipeline::Range> writes;
                        const auto take = [&](std::span<const std::uint32_t> words, std::uint32_t code, std::uint32_t at, std::optional<std::uint32_t> raised) {
                            const auto label = Pm4::DecodeLabelWrite(words);
                            if (!label.has_value()) {
                                if (!raised.has_value()) {
                                    ++noOpLabels;
                                    return false;
                                }
                                labels->labels.push_back({0, static_cast<std::uint32_t>(labels->bytes.size()), 0, code, at, false, raised});
                                return true;
                            }
                            const auto stored = label->Bytes();
                            labels->labels.push_back({label->address, static_cast<std::uint32_t>(labels->bytes.size()), static_cast<std::uint32_t>(stored.size()), code, at, true, raised});
                            labels->bytes.insert(labels->bytes.end(), stored.begin(), stored.end());
                            writes.emplace_back(label->address, label->address + stored.size());
                            return true;
                        };
                        std::array<bool, LabelRunCap> items{};
                        std::array<bool, LabelRunCap> checks{};
                        items[0] = take(packet, opcode, cursor, interrupt);
                        if (interrupt.has_value()) bumpEpoch(&EpochBumps::drains);
                        for (auto at = index + 1; at < end; ++at) {
                            const auto& follower = packets[at];
                            const auto words = std::span(commands).subspan(follower.cursor, follower.words);
                            ++run.packets;
                            checks[at - index] = LeanFront::Due(frontLatch, LeanFront::Item::Packet);
                            if (DrawFinisher::Busy() && !follower.Has(LeanPacket::KeepsFinisher)) DrawFinisher::NoteDrain(DrawFinisher::DrainSite::Packet, false);
                            if (follower.Has(LeanPacket::LabelIssue)) noteLabelIssue(words, submission.queue);
                            items[at - index] = take(words, follower.opcode, follower.cursor, std::nullopt);
                        }
                        static const std::uint64_t verifyEvery = everyFrom("APS5_VERIFY_LABEL_RUNS");
                        static std::atomic<std::uint64_t> verifyOrdinal{0};
                        if (verifyEvery != 0 && verifyOrdinal.fetch_add(1, std::memory_order_relaxed) % verifyEvery == 0) LabelRunCensus::NoteVerify(end - index, verifyLabelRun(submission, queue, table, index, end, *labels, writes, interrupt));
                        const auto heldItems = labels->labels.size();
                        if (heldItems != 0) enqueueLabelRun(submission.queue, std::move(labels), std::move(writes));
                        if (verifyPacket) LeanFront::NoteChecked(LeanFront::Item::Packet);
                        traceLabel(packet, submission.queue);
                        recent.Record(cursor);
                        for (auto at = index + 1; at < end; ++at) {
                            const auto& follower = packets[at];
                            if (checks[at - index]) LeanFront::NoteChecked(LeanFront::Item::Packet);
                            traceLabel(std::span(commands).subspan(follower.cursor, follower.words), submission.queue);
                            recent.Record(follower.cursor);
                        }
                        if (LabelRunCensus::Enabled()) {
                            const auto [gap, otherOpcode] = labelGap(packets, lastQueuedLabel, index);
                            LabelRunCensus::NoteQueued(gap, otherOpcode, items[0]);
                            for (auto at = index + 1; at < end; ++at) LabelRunCensus::NoteQueued(LabelRunCensus::Gap::Adjacent, 0u, items[at - index]);
                            LabelRunCensus::NoteRun(end - index, heldItems, stop.value_or(End::SegmentEnd), stopOpcode);
                        }
                        lastQueuedLabel = end - 1;
                        index = end;
                        continue;
                    }
                }
            }
            std::optional<DrawPipeline::OrderProbe> dmaProbe;
            if (dmaBeside && DrawFinisher::FrameVerifyDue(FrameCheck::DmaOrder)) dmaProbe.emplace(FrameCheck::DmaOrder);
            if (enqueueLabelPacket(packet, opcode, submission.queue, releaseInterrupt(packet, opcode, false)) || enqueueDmaPacket(packet, opcode, submission.queue, queue) || enqueueConstantRamDump(packet, opcode, submission.queue, queue) || enqueueSampleDumpPacket(packet, opcode, submission.queue)) {
                if (!entry.Has(LeanPacket::Enqueues)) LeanFront::NoteDifference(LeanFront::Field::PacketTable, where(submission, cursor) + ": today's code queued the packet beside the pipeline, the packet table skips that check");
                if (verifyPacket) LeanFront::NoteChecked(LeanFront::Item::Packet);
                traceLabel(packet, submission.queue);
                recent.Record(cursor);
                if (opcode == 0x49 || opcode == 0x37) {
                    if (LabelRunCensus::Enabled()) {
                        const auto [gap, otherOpcode] = labelGap(packets, lastQueuedLabel, index);
                        LabelRunCensus::NoteQueued(gap, otherOpcode, Pm4::DecodeLabelWrite(packet).has_value() || releasesInterrupt(packet, opcode));
                    }
                    lastQueuedLabel = index;
                }
                index = next;
                continue;
            }
        }
        if (dmaBeside) {
            DrawFinisher::NoteFrameKept(FrameKept::DmaRefused);
            if (DrawFinisher::Busy() && !pipelineVerdict() && !(DrawFinisher::Deep() && asksVerdict() && PacketRules::awaitVerdictFinisherWriters(packet, header, opcode)) && !LeanFront::SwitchSet(frontLatch, RuntimeSwitch::XpNoPacketDrains)) {
                noteKept();
                DrawFinisher::Get().Drain(true);
            }
        }
        if ((opcode == 0x50 || (opcode == 0x46 && Pm4::SampleDump(packet))) && (DrawFinisher::Get().FlipsQueued() != DrawFinisher::Get().FlipsCommitted() || DrawFinisher::Get().IndirectsInFlight()) && DrawPipeline::Queue0().Busy()) {
            DrawFinisher::NoteFrameKept(DrawFinisher::Get().FlipsQueued() != DrawFinisher::Get().FlipsCommitted() ? FrameKept::WorkerBehindFlip : FrameKept::WorkerBehindIndirect);
            noteKept();
            DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, opcode);
        }
        if (!predicated && entry.Has(LeanPacket::Wait) && !entry.Has(LeanPacket::RenderingWait) && orderedWaits() && DrawPipeline::Queue0().Busy()) {
            const auto awaitedBytes = Pm4::WaitAwaitedBytes(packet);
            const std::uint64_t awaited = (packet[2] & (awaitedBytes == 8 ? ~7u : ~3u)) | (static_cast<std::uint64_t>(packet[3]) << 32u);
            const auto value = DrawPipeline::Queue0().PendingLabel(awaited, awaitedBytes);
            if (value.has_value() && Pm4::WaitComparesValue(packet, *value)) {
                ++waitOutcomes().fromRecorderSameQueue;
                gateOpened() = true;
                noteLabelPass(packet, submission.queue);
                if (verifyPacket) LeanFront::NoteChecked(LeanFront::Item::Packet);
                recent.Record(cursor);
                index = next;
                continue;
            }
            if (!DrawPipeline::Queue0().Overlaps(awaited, awaitedBytes) && Pm4::WaitSatisfiedUnchecked(packet) && !OwnStoreRejects(packet, awaited, awaitedBytes, submission.queue)) {
                bumpEpoch(&EpochBumps::waits);
                ++waitOutcomes().atEntry;
                gateOpened() = true;
                noteLabelPass(packet, submission.queue);
                if (verifyPacket) LeanFront::NoteChecked(LeanFront::Item::Packet);
                recent.Record(cursor);
                index = next;
                continue;
            }
        }
        timing.Mark("pipeline_prepare");
        if (DrawPipeline::Queue0().Busy()) {
            bool keeps = false;
            if (verifyPacket) {
                keeps = PacketRules::keepsDrawPipeline(packet, header, opcode);
                if ((entry.pipeline == LeanPacket::Pipeline::Keeps || entry.pipeline == LeanPacket::Pipeline::Drains) && keeps != (entry.pipeline == LeanPacket::Pipeline::Keeps)) {
                    LeanFront::NoteDifference(LeanFront::Field::PipelineVerdicts, where(submission, cursor) + ": today's committer verdict " + (keeps ? "keeps" : "drains") + " the pipeline, the packet table's static verdict does not");
                } else if (listVerdictKnown && keeps != listVerdict) {
                    LeanFront::NoteAllowed(LeanFront::Allowed::OverlapMoved);
                }
            } else {
                keeps = listVerdictKnown ? listVerdict : pipelineVerdict();
            }
            if (!keeps && DrawFinisher::Deep() && asksVerdict()) keeps = PacketRules::awaitVerdictWriters(packet, header, opcode);
            if (!keeps && argumentWait) keeps = DrawPipeline::Queue0().AwaitExactWriters({&argumentRange, 1}, DrawPipeline::DrainReason::Indirect, 0x16, DeepSite::IndirectArguments, FrameSite::ArgumentWait, FrameCheck::DispatchArguments);
            if (!keeps && indirectQueued) keeps = true;
            const bool ordersFrame = entry.Has(LeanPacket::FlipHeader) || entry.Has(LeanPacket::RenderingWait) || flip || entry.Has(LeanPacket::Wait) || opcode == 0x49 || opcode == 0x37;
            if (!keeps && !ordered && (ordersFrame || !DrawFinisher::FlowSkips(DrawFinisher::FlowSite::PacketCommitter))) {
                if (opcode == 0x16 && !argumentWait && !indirectQueued && DrawFinisher::FrameFlow()) DrawFinisher::NoteFrameKept(FrameKept::IndirectGpu);
                noteKept();
                DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, entry.Has(LeanPacket::FlipHeader) ? 0xffu : opcode);
            }
        }
        if (predicated && DrawPipeline::Queue0().Busy() && DrawPipeline::Queue0().Overlaps(queue.predication.address, Pm4::PredicationBytes(queue.predication)) && !DrawFinisher::FlowSkips(DrawFinisher::FlowSite::Predication)) PacketRules::awaitPredicationWriters(queue.predication.address, Pm4::PredicationBytes(queue.predication), opcode);
        timing.Mark("pipeline_drain");
        std::shared_lock deviceUse(deviceReplacement, std::defer_lock);
        if (!entry.Has(LeanPacket::Wait) && !entry.Has(LeanPacket::RenderingWait) && !entry.Has(LeanPacket::FlipHeader)) deviceUse.lock();
        timing.Mark("device_lock");
        const std::uint32_t tag = entry.Has(LeanPacket::FlipHeader) ? 0xffffu : opcode;
        if (tag != tagged) {
            GuestMemory::SetCurrentPacket(tag, submission.queue);
            tagged = tag;
        }
        if (report) CaptureTrace::Log("packet submission=%llu queue=%x offset=%zu header=%08x words=%zu", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, header, packet.size());
        if (predicated) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            if (!Pm4::PredicationPasses(queue)) {
                if (verifyPacket) LeanFront::NoteChecked(LeanFront::Item::Packet);
                index = opcode == 0x3f ? skipTo(entry) : next;
                continue;
            }
            if (entry.Has(LeanPacket::LabelIssue)) noteLabelIssue(packet, submission.queue);
        }
        if (opcode == 0x3f) {
            if (verifyPacket) LeanFront::NoteChecked(LeanFront::Item::Packet);
            index = next;
            continue;
        }

        const auto flushStart = profilePackets ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const bool labelPacket = opcode == 0x49 || opcode == 0x37;
        FlushInputs flushInputs;
        if (verifyPacket) {
            const auto before = std::chrono::steady_clock::now();
            const bool gateDue = flushDue(run.gate, flushInputs);
            flushInputs.now = before;
            const auto decision = flushBetweenPackets(submission.queue, header, labelPacket, flushInputs.given ? &flushInputs : nullptr);
            tagged = NoTag;
            ++run.gate.calls;
            checkFlush(submission, cursor, run.gate, gateDue, decision, flushInputs);
        } else if (flushDue(run.gate, flushInputs)) {
            if (flushInputs.given && (flushInputs.pending.has_value() || flushInputs.completions)) flushInputs.now = std::chrono::steady_clock::now();
            flushBetweenPackets(submission.queue, header, labelPacket, flushInputs.given ? &flushInputs : nullptr);
            tagged = NoTag;
            ++run.gate.calls;
        } else {
            ++run.gate.skipped;
        }
        timing.Mark("flush_between_packets");
        std::optional<PacketTimer> packetTimer;
        if (profilePackets) {
            packetTimer.emplace(true, entry.Has(LeanPacket::FlipHeader) ? 0xffffu : opcode, submission.queue, packetProfile, std::chrono::steady_clock::now());
            packetStartedAt() = packetTimer->start;
            pendingDispatchPhases() = {};
            pendingDrawPhases() = {};
            packetProfile.flushMs += std::chrono::duration<double, std::milli>(packetTimer->start - flushStart).count();
        }
        const auto finishDispatchPacket = [&](bool indirect) {
            if (!profilePackets) return;
            const auto now = std::chrono::steady_clock::now();
            auto& pending = pendingDispatchPhases();
            if (pending.phases) {
                pending.ms[PhaseEpilogue] = std::chrono::duration<double, std::milli>(now - pending.tailAt).count();
                addDriverPhases(indirect ? Queue0Indirect : Queue0Direct, pending.ms, pending.hit, pending.validated);
            }
            if (indirect) return;
            auto& row = packetProfile.dispatchOutcomes[static_cast<std::size_t>(pending.outcome)];
            ++row.first;
            row.second += std::chrono::duration<double, std::milli>(now - packetTimer->start).count();
        };
        const auto finishDrawPacket = [&](bool drawn) {
            if (!profilePackets) return;
            const auto now = std::chrono::steady_clock::now();
            auto& pending = pendingDrawPhases();
            std::array<double, DrawDriverPhaseCount> ms{};
            if (drawn && pending.phases) {
                ms = pending.ms;
                ms[DrawRowEpilogue] = std::chrono::duration<double, std::milli>(now - pending.tailAt).count();
            } else {
                ms[DrawRowSkipped] = std::chrono::duration<double, std::milli>(now - packetTimer->start).count();
            }
            addDrawPhases(ms, drawn && pending.phases, pending.captures);
        };

        bool wroteOnGpu = false, endOfPipeInterrupt = false, interruptDeferred = false, drawPacket = false, sampleDump = false, drains = false;
        if (orderedFlip) {
            wroteOnGpu = flipKind == PacketRules::FlipForm::EopLabel;
            if (flipKind == PacketRules::FlipForm::EopNothing) ++noOpLabels;
        } else if (entry.Has(LeanPacket::Memory) || verifyPacket) {
            drains = preparePacketMemory(submission, queue, packet, header, opcode, releaseInterrupt(packet, opcode, flip), wroteOnGpu, endOfPipeInterrupt, interruptDeferred, drawPacket, sampleDump);
            if (!entry.Has(LeanPacket::Memory) && (drains || wroteOnGpu || endOfPipeInterrupt || interruptDeferred || sampleDump || drawPacket != entry.Has(LeanPacket::Draw))) {
                LeanFront::NoteDifference(LeanFront::Field::PacketTable, where(submission, cursor) + ": today's packet memory step acted on a packet the packet table skips it for");
            }
        } else {
            drawPacket = entry.Has(LeanPacket::Draw);
        }
        if (entry.Has(LeanPacket::Traced)) traceLabel(packet, submission.queue);
        timing.Mark("prepare_memory");

        const bool waitPacket = entry.Has(LeanPacket::Wait) || entry.Has(LeanPacket::RenderingWait);
        struct Progress {
            Driver& driver;
            bool counted;
            ~Progress() {
                if (!counted) return;
                --driver.packetsInFlight;
                ++driver.packetsDone;
            }
        } progress{*this, !waitPacket};
        if (!waitPacket) ++packetsInFlight;
        recent.Record(cursor);
        if (entry.Has(LeanPacket::RenderingWait)) {
            const auto waitStart = orderedWait && DrawFinisher::FrameCensus() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (orderedWait) DrawFinisher::Get().FlushOpenRun();
            const DrawFinisher::TailScope standing(FrameTail::RenderingWait);
            timedWork(&WorkerProfile::waitMs, [&] { submission.renderingWaits.at(cursor)->Wait(); });
            if (orderedWait) DrawFinisher::NoteFrameSite(FrameSite::RenderingWait, waitStart == std::chrono::steady_clock::time_point{} ? 0 : static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStart).count()));
        } else if (orderedFlip) {
            if (deviceUse.owns_lock()) deviceUse.unlock();
            CheckFailure();
            ++flipsCounted;
            if (Graphics::AbConfigured()) {
                abAtFlip(abSeconds);
                run.gate.capGeneration = Graphics::RuntimeSwitches::Generation();
                run.gate.cap = batchCap();
            }
            timing.Mark("flip_flush");
            timing.FinishPacket(submission.serial, cursor, 0xffffu, count);
            const auto frameNumber = ++frameSerial;
            auto frame = std::make_shared<FrameTiming>(frameNumber);
            const auto now = FrameTiming::Clock::now();
            frame->IncludeSubmission(submission.serial, now, now, now, true);
            frame->SetFlip(submission.serial, cursor, now, now);
            FlipCommit commit;
            commit.request = submission.flips.at(cursor);
            commit.frame = std::move(frame);
            commit.frameNumber = frameNumber;
            commit.serial = submission.serial;
            commit.cursor = static_cast<std::uint32_t>(cursor);
            commit.header = flipKind == PacketRules::FlipForm::Header;
            commit.label = flipKind == PacketRules::FlipForm::EopLabel;
            std::copy_n(packet.begin(), std::min(packet.size(), commit.words.size()), commit.words.begin());
            commit.releases = std::exchange(flipHold.held, false);
            queueOrderedFlip(std::move(commit));
        } else if (flip) {
            if (opcode == 0x49) {
                if (!wroteOnGpu) {
                    Pm4::Execute(packet, queue);
                    if (const auto label = Pm4::DecodeLabelWrite(packet)) noteLabelStore(label->address, label->Bytes(), ++eventSerial, submission.queue);
                }
            }
            if (deviceUse.owns_lock()) deviceUse.unlock();
            CheckFailure();
            std::uint64_t batchesAtFlip = 0, unsignaledAtFlip = 0;
            Graphics::StorageTexture::NoteFlip();
            if (!drains) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                const auto localDevice = device.Load();
                if (localDevice != nullptr) static_cast<void>(Graphics::StorageTexture::StoreDrawnLinear("flip"));
                Graphics::StorageTexture::ReleaseIdleCopies();
                if (!recordLabelsForPacket(localDevice.get(), submission.queue) && localDevice != nullptr) localDevice->SubmitRecorded(true, Graphics::Recorder::SubmitReason::Flip);
                if (localDevice != nullptr) localDevice->FlipBatches(batchesAtFlip, unsignaledAtFlip);
            }
            ++flipsCounted;
            Graphics::NoteHostImportFrame();
            if (batchesAtFlip != 0) flipSerial = batchesAtFlip;
            flipBatchesUnsignaled += unsignaledAtFlip;
            if (Graphics::AbConfigured()) {
                abAtFlip();
                run.gate.capGeneration = Graphics::RuntimeSwitches::Generation();
                run.gate.cap = batchCap();
            }
            timing.Mark("flip_flush");
            timing.FinishPacket(submission.serial, cursor, 0xffffu, count);
            const bool queueTiming = APS5_ENABLE_TIMING_LOG;
            auto frame = queueTiming ? std::move(pendingFrameTiming) : std::make_shared<FrameTiming>(++frameSerial);
            require(frame != nullptr, "flip has no frame timing");
            const auto now = FrameTiming::Clock::now();
            if (!queueTiming) frame->IncludeSubmission(submission.serial, now, now, now, true);
            frame->SetFlip(submission.serial, cursor, queueTiming ? submission.receivedAt : now, now);
            if (queueTiming) frame->CollectBackground();
            frame->NoteFlipBatches(batchesAtFlip, unsignaledAtFlip);
            if (report) CaptureTrace::Log("flip frame=%llu submission=%llu offset=%zu batch=%llu unsignaled=%llu", static_cast<unsigned long long>(frameSerial.load()), static_cast<unsigned long long>(submission.serial), cursor, static_cast<unsigned long long>(batchesAtFlip), static_cast<unsigned long long>(unsignaledAtFlip));
            flipHold.Release();
            submission.flips.at(cursor)->GpuReady(frame);
            DrawFinisher::Get().LatchFrameFlow();
        } else if (opcode == 0x15 || opcode == 0x16) {
            packetCounters.CountDispatch();
            landPendingWaits(submission.queue);
            currentPacketOffset() = cursor;
            if (gateOpened()) {
                gateOpened() = false;
                if (!pipelineDispatch()) {
                    if (DrawPipeline::Queue0().Busy() && !DrawFinisher::FlowSkips(DrawFinisher::FlowSite::Dispatch)) {
                        DrawFinisher::NoteDeepKept(DeepKept::DispatchNotPipelined);
                        DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, opcode);
                    }
                    timedWork(&WorkerProfile::dispatchMs, [&] { resolveGroupAhead(submission, queue, cursor); });
                }
            }
            if (opcode == 0x15) timedWork(&WorkerProfile::dispatchMs, [&] { dispatch(queue, packet, submission); });
            else timedWork(&WorkerProfile::dispatchMs, [&] { dispatchIndirect(queue, packet, submission, indirectQueued); });
            Pm4::NoteDispatchRegisters(packet, queue);
            Graphics::Recorder::CountRecordedWork();
            finishDispatchPacket(opcode == 0x16);
        } else if (entry.Has(LeanPacket::Wait)) {
            const auto waitStart = traceGpu() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            timedWork(&WorkerProfile::waitMs, [&] { waitMemory(packet, submission.queue, recent, submission.received, entry.Has(LeanPacket::Held)); });
            noteLabelPass(packet, submission.queue);
            gateOpened() = true;
            if (traceGpu() && std::chrono::steady_clock::now() - waitStart > std::chrono::milliseconds(200)) {
                for (std::size_t following = cursor + count, shown = 0; following < commands.size() && shown < 48; ++shown) {
                    const auto followingHeader = commands[following];
                    const auto followingCount = Pm4::PacketWords(followingHeader);
                    const auto followingPacket = std::span(commands).subspan(following, followingCount);
                    std::fprintf(stderr, "[gpu]   then %s", Pm4::Name(followingHeader).c_str());
                    for (std::size_t i = 1; i < followingPacket.size() && i < 7; ++i) std::fprintf(stderr, " %08x", followingPacket[i]);
                    std::fprintf(stderr, "\n");
                    following += followingCount;
                }
            }
        } else if (drawPacket) {
            landPendingWaits(submission.queue);
            packetCounters.CountDraw();
            seal.drawn = true;
            if (report) {
                bool drawn = false;
                const auto drawStart = ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
                struct DrawPacketReport {
                    ThreadReport::Clock::time_point start;
                    ~DrawPacketReport() {
                        if (start == ThreadReport::Clock::time_point{}) return;
                        ThreadReport::Add(ThreadReport::Count::WorkerDrawPackets, 1);
                        ThreadReport::Add(ThreadReport::Count::WorkerDrawPacketNs, ThreadReport::Nanoseconds(start, ThreadReport::Clock::now()));
                    }
                } drawPacketReport{drawStart};
                timed(&WorkerProfile::drawMs, [&] {
                    const bool profile = profileDraws();
                    const auto color = (static_cast<std::uint64_t>(readRegister(queue.context, 0x390)) << 40u) | (static_cast<std::uint64_t>(readRegister(queue.context, 0x318)) << 8u);
                    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    const auto countSkip = [&](Graphics::DrawSkip kind) {
                        if (profile) Graphics::CountDrawSkip(kind, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count());
                    };
                    try {
                        std::string rejected;
                        const auto verdict = drawLean(queue, packet, submission, cursor, rejected);
                        drawn = verdict == DrawVerdict::Drawn;
                        CaptureTrace::Log("draw submission=%llu queue=%x offset=%zu target=%llx mask=%x verdict=%d reason=%.256s", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, static_cast<unsigned long long>(color), readRegister(queue.context, 0x8e), static_cast<int>(verdict), rejected.c_str());
                        if (verdict == DrawVerdict::Rejected) {
                            countSkip(Graphics::DrawSkip::Prechecked);
                            throw std::runtime_error(rejected);
                        } else if (verdict == DrawVerdict::Nothing) {
                            countSkip(Graphics::DrawSkip::Nothing);
                        } else if (traceDraws()) {
                            std::fprintf(stderr, "[draw] target 0x%llx mask 0x%x ok\n", static_cast<unsigned long long>(color), readRegister(queue.context, 0x8e));
                        }
                    } catch (const RefusedDraw& refused) {
                        CaptureTrace::Log("draw-refused submission=%llu offset=%zu reason=%.256s", static_cast<unsigned long long>(submission.serial), cursor, refused.what());
                        countSkip(Graphics::DrawSkip::Thrown);
                        ReportRefusedDraw(refused);
                    } catch (const std::exception& error) {
                        CaptureTrace::Log("draw-error submission=%llu offset=%zu reason=%.256s", static_cast<unsigned long long>(submission.serial), cursor, error.what());
                        countSkip(Graphics::DrawSkip::Thrown);
                        throw;
                    }
                });
                finishDrawPacket(drawn);
            } else {
                static_cast<void>(readRegister(queue.context, 0x390));
                static_cast<void>(readRegister(queue.context, 0x318));
                try {
                    std::string rejected;
                    const auto verdict = drawLean(queue, packet, submission, cursor, rejected);
                    static_cast<void>(readRegister(queue.context, 0x8e));
                    if (verdict == DrawVerdict::Rejected) throw std::runtime_error(rejected);
                } catch (const RefusedDraw& refused) {
                    ReportRefusedDraw(refused);
                }
            }
        } else if (opcode == 0x22) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            const auto condition = Pm4::ReadCondition(packet);
            if (condition == 0) next = skipTo(entry);
            if (traceGpu()) std::fprintf(stderr, "[gpu] %.1f queue 0x%x COND_EXEC at DWORD %zu reads 0x%x at 0x%llx: %s %zu dwords\n", TraceMs(), submission.queue, cursor, condition, static_cast<unsigned long long>(packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u)), condition == 0 ? "skips" : "executes", submission.conditionalEnds.at(cursor) - cursor - count);
        } else if (opcode == 0x91) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            Pm4::Execute(packet, queue);
        } else if (sampleDump && !wroteOnGpu) {
            dumpSampleCounters(Pm4::SampleDumpAddress(packet));
        } else if (entry.Has(LeanPacket::Seal)) {
            if (std::exchange(seal.drawn, false)) {
                const auto sealPass = [this] {
                    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
                    std::lock_guard gpuLock(GuestMemory::GpuMutex());
                    if (const auto localDevice = device.Load()) localDevice->SealPassWriteOnly();
                };
                const bool streamSeal = LeanFront::SwitchSet(frontLatch, RuntimeSwitch::XpStreamSegments);
                if (streamSeal && DrawFinisher::Overlap() && DrawFinisher::Busy()) {
                    DrawFinisher::Pending held;
                    held.unknown = false; held.origin = 11;
                    DrawFinisher::Get().Enqueue([sealPass] { DrawPipeline::Queue0().EnqueueQuiet(sealPass); }, std::move(held));
                    PacketRules::noteXpStream(true);
                } else if (streamSeal && DrawPipeline::Queue0().Busy()) {
                    DrawPipeline::Queue0().EnqueueQuiet(sealPass);
                    PacketRules::noteXpStream(true);
                } else if (DrawPipeline::Queue0().Busy()) {
                    DrawPipeline::Queue0().Enqueue(sealPass, {});
                } else {
                    sealPass();
                }
            }
        } else if (opcode != 0x42 && opcode != 0x46 && opcode != 0x58) {
            if (!wroteOnGpu) {
                if (entry.Has(LeanPacket::IndirectRegisters)) {
                    const auto list = RegisterListAt(submission, cursor, listHint);
                    const auto listed = list.pairs;
                    if (list.valid) {
                        if (registerRuns) executeRegisters(packet, &list, queue, submission, cursor);
                        else Pm4::ExecuteIndirectRegisters(packet, listed, queue);
                    } else {
                        landPendingWaits(submission.queue);
                        recordQueuedLabelsBeforeRead(submission.queue);
                        const auto current = Pm4::ReadIndirectRegisters(packet);
                        static std::atomic<bool> reported{false};
                        if (!reported.exchange(true)) {
                            std::uint32_t invalid = 0;
                            for (std::size_t i = 0; i < listed.size(); i += 2) {
                                if (!Pm4::IndirectRegistersValid(std::span(&listed[i], 1))) {
                                    invalid = listed[i];
                                    break;
                                }
                            }
                            std::fprintf(stderr, "[gpu] queue 0x%x %s at DWORD %zu: register list entry with offset dword 0x%08x read again where the packet executes%s\n", submission.queue, Pm4::Name(packet[0]).c_str(), cursor, invalid, skipInvalidLists() ? "; invalid entries are skipped (APS5_SKIP_INVALID_INDIRECT_REGISTERS)" : "");
                        }
                        Pm4::ExecuteIndirectRegisters(packet, current, queue, skipInvalidLists());
                    }
                } else if (registerRuns && (opcode == 0x69 || opcode == 0x76 || opcode == 0x79 || opcode == 0x7a)) {
                    executeRegisters(packet, nullptr, queue, submission, cursor);
                } else {
                    Pm4::Execute(packet, queue);
                }
                if (labelPacket) {
                    if (const auto label = Pm4::DecodeLabelWrite(packet)) noteLabelStore(label->address, label->Bytes(), ++eventSerial, submission.queue);
                }
            }
            if (endOfPipeInterrupt && !interruptDeferred) deliverGuestEopInterrupt(submission.queue, packet[7] & 0x0fffffffu);
        }
        if (drawPacket || (sampleDump && wroteOnGpu)) Graphics::Recorder::CountRecordedWork();
        timing.Mark(entry.Has(LeanPacket::RenderingWait) ? "rendering_wait" : entry.Has(LeanPacket::Wait) ? "memory_wait" : opcode == 0x15 || opcode == 0x16 ? "dispatch" : drawPacket ? "draw" : "execute");
        timing.FinishPacket(submission.serial, cursor, opcode, count);
        if (verifyPacket) LeanFront::NoteChecked(LeanFront::Item::Packet);
        index = next;
    }
}

template void Driver::runLeanPackets<true>(const Submission&, QueueState&, const PacketTable&, FlipHold&, LeanRun&);
template void Driver::runLeanPackets<false>(const Submission&, QueueState&, const PacketTable&, FlipHold&, LeanRun&);

std::size_t Driver::verifyLabelRun(const Submission& submission, const QueueState& queue, const PacketTable& table, std::size_t first, std::size_t end, const LabelRun& run, std::span<const std::pair<std::uint64_t, std::uint64_t>> writes, std::optional<std::uint32_t> leaderInterrupt) {
    static std::atomic<std::uint64_t> printed{0};
    std::size_t differing = 0;
    const auto differ = [&](std::size_t cursor, const char* what) {
        ++differing;
        const auto count = printed.fetch_add(1, std::memory_order_relaxed);
        if (count < 16 || count % 4096 == 0) std::fprintf(stderr, "[label-runs-verify] %s: %s (%llu differences so far)\n", where(submission, cursor).c_str(), what, static_cast<unsigned long long>(count + 1));
    };
    const auto leaderCursor = static_cast<std::size_t>(table.packets[first].cursor);
    std::size_t held = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> expected;
    for (auto at = first; at < end; ++at) {
        const auto& entry = table.packets[at];
        const auto cursor = static_cast<std::size_t>(entry.cursor);
        const auto words = std::span<const std::uint32_t>(submission.commands).subspan(cursor, entry.words);
        const auto header = submission.commands[cursor];
        const auto opcode = (header >> 8u) & 0xffu;
        const bool leader = at == first;
        if (opcode != entry.opcode || (opcode != 0x49 && opcode != 0x37)) differ(cursor, "the run took a packet that is not a label packet");
        if (Pm4::Predicated(header) && queue.predication.operation != 0) differ(cursor, "today's loop finds the packet predicated");
        if (header == FlipPacketHeader || header == RenderingWaitPacketHeader || submission.flips.contains(cursor)) differ(cursor, "today's loop handles the packet as a flip or a rendering wait");
        if (!labelPacketQueues(words, opcode)) differ(cursor, "today's rules do not queue the packet as a label item");
        std::optional<std::uint32_t> raised;
        try {
            raised = releaseInterrupt(words, opcode, false);
        } catch (...) {
            differ(cursor, "today's interrupt decode throws for the packet");
        }
        if (leader ? raised != leaderInterrupt : raised.has_value()) differ(cursor, leader ? "the leader's interrupt differs from today's" : "a follower raises an interrupt");
        const auto label = Pm4::DecodeLabelWrite(words);
        if (!label.has_value() && !raised.has_value()) {
            if (held < run.labels.size() && run.labels[held].cursor == cursor) differ(cursor, "the run holds a label for a packet today's code queues nothing for");
            continue;
        }
        if (held >= run.labels.size() || run.labels[held].cursor != cursor) {
            differ(cursor, "the run holds no label for a packet today's code queues an item for");
            continue;
        }
        const auto& kept = run.labels[held++];
        if (kept.opcode != opcode || kept.interrupt != raised) differ(cursor, "the run's label has another opcode or interrupt than today's item");
        if (!label.has_value()) {
            if (kept.stores || kept.size != 0) differ(cursor, "the run stores bytes for an interrupt today's item stores nothing for");
            continue;
        }
        const auto bytes = label->Bytes();
        expected.emplace_back(label->address, label->address + bytes.size());
        const auto stored = run.BytesOf(kept);
        const bool timestamp = opcode == 0x49 && (words[2] >> 29u) == 3u;
        if (!kept.stores || kept.address != label->address || kept.size != bytes.size()) differ(cursor, "the run's label address or size differs from today's item");
        else if (!timestamp && !std::equal(bytes.begin(), bytes.end(), stored.begin())) differ(cursor, "the run's label bytes differ from today's item");
    }
    if (held != run.labels.size()) differ(leaderCursor, "the run holds labels today's code would not queue");
    if (!std::equal(expected.begin(), expected.end(), writes.begin(), writes.end())) differ(leaderCursor, "the run's write ranges differ from today's items");
    if (!DrawFinisher::Overlap()) differ(leaderCursor, "the run was built with the finisher overlap off");
    return differing;
}

}
