#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Packets/DriverTimed.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceOverlay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

struct RingOrder {
    std::uint32_t queue = 0xffffffffu;
    std::uint64_t serial = 0;
    bool afterDispatch = false;
};

thread_local RingOrder ringOrder;

void noteRingPacket(std::uint32_t queueId, std::uint64_t serial, std::uint32_t opcode, std::uint32_t header) {
    if (ringOrder.queue != queueId || ringOrder.serial != serial) ringOrder = {queueId, serial, false};
    if (opcode == 0x15 || opcode == 0x16 || opcode == 0x76) return;
    if (opcode == 0x10) {
        const auto sub = (header >> 2u) & 0x3fu;
        if ((header & 0xfcu) == 0 || sub == 0x0b || sub == 0x0c) return;
    }
    ringOrder.afterDispatch = false;
}

}

namespace PacketRules {

void noteXpStream(bool seal) {
    static std::atomic<std::uint64_t> ends{0}, seals{0};
    static std::atomic<std::int64_t> lastReport{0};
    static std::uint64_t endsReported = 0, sealsReported = 0;
    (seal ? seals : ends).fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        lastReport.store(now, std::memory_order_relaxed);
    } else if (now - last >= 10000 && lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        const auto endsNow = ends.load(std::memory_order_relaxed), sealsNow = seals.load(std::memory_order_relaxed);
        AgcDriver::ProfilePrint_nid_no_patch("[xp-stream-segments] %.1f s: %llu released segment ends and %llu write-only seals queued without a drain, a submit or a proof break\n", static_cast<double>(now - last) / 1000.0, static_cast<unsigned long long>(endsNow - endsReported), static_cast<unsigned long long>(sealsNow - sealsReported));
        endsReported = endsNow;
        sealsReported = sealsNow;
    }
}

bool xpDropsDraw() {
    static const std::uint64_t keepOneIn = [] {
        const char* text = std::getenv("APS5_XP_KEEP_DRAWS_N");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    static std::atomic<std::uint64_t> seen{0}, dropped{0};
    static std::atomic<std::int64_t> lastReport{0};
    static std::uint64_t seenReported = 0, droppedReported = 0;
    const auto ordinal = seen.fetch_add(1, std::memory_order_relaxed);
    const bool drop = keepOneIn == 0 || ordinal % keepOneIn != 0;
    if (drop) dropped.fetch_add(1, std::memory_order_relaxed);
    if ((ordinal & 1023u) == 0) {
        const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        auto last = lastReport.load(std::memory_order_relaxed);
        if (last == 0) {
            lastReport.store(now, std::memory_order_relaxed);
        } else if (now - last >= 10000 && lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
            const auto seenNow = seen.load(std::memory_order_relaxed), droppedNow = dropped.load(std::memory_order_relaxed);
            AgcDriver::ProfilePrint_nid_no_patch("[xp-keep-draws] %.1f s: %llu direct draws reached the check, %llu dropped (keep one in %llu)\n", static_cast<double>(now - last) / 1000.0, static_cast<unsigned long long>(seenNow - seenReported), static_cast<unsigned long long>(droppedNow - droppedReported), static_cast<unsigned long long>(keepOneIn));
            seenReported = seenNow;
            droppedReported = droppedNow;
        }
    }
    return drop;
}

bool orderedFills() {
    static const bool ordered = std::getenv("APS5_PIPELINE_DRAIN_FILLS") == nullptr;
    return ordered;
}

bool keepsDrawFinisher(std::uint32_t header, std::uint32_t opcode) {
    if (header == FlipPacketHeader || header == RenderingWaitPacketHeader) return false;
    if (Pm4::DrawOpcode(opcode)) return true;
    switch (opcode) {
        case 0x13: case 0x26: case 0x2a: case 0x2f: case 0x69: case 0x76: case 0x79: return true;
        case 0x10: return ((header >> 2u) & 0x3fu) == 0;
        default: return false;
    }
}

bool waitTakesPendingLabel(std::span<const std::uint32_t> packet) {
    if (packet.size() < 5) return false;
    const auto awaitedBytes = Pm4::WaitAwaitedBytes(packet);
    const std::uint64_t awaited = (packet[2] & (awaitedBytes == 8 ? ~7u : ~3u)) | (static_cast<std::uint64_t>(packet[3]) << 32u);
    const auto value = DrawPipeline::Queue0().PendingLabel(awaited, awaitedBytes);
    return value.has_value() && Pm4::WaitComparesValue(packet, *value);
}

bool keepsDrawPipeline(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode) {
    if (header == FlipPacketHeader || header == RenderingWaitPacketHeader) return false;
    if (Pm4::DrawOpcode(opcode)) return true;
    switch (opcode) {
        case 0x11: case 0x12: case 0x13: case 0x20: case 0x26: case 0x2a: case 0x2f: case 0x3f: case 0x42: case 0x58: case 0x59: case 0x69: case 0x76: case 0x79: case 0x7a: case 0x81: return true;
        case 0x46: return packet.size() >= 2 && (packet[1] & 0x3fu) != 0x39u;
        case 0x15: return orderedFills();
        case 0x40:
            return packet.size() >= 6 && !Pm4::CopyDataToRegister(packet) && Pm4::CopyDataSource(packet) == Pm4::CopyDataGpuClockSource &&
                   !DrawPipeline::Queue0().Overlaps(packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u), Pm4::CopyDataBytes(packet));
        case 0x63: case 0x64: case 0x9f:
            return packet.size() >= 5 && !DrawPipeline::Queue0().Overlaps(packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u), static_cast<std::size_t>(packet[4]) * 8);
        case 0x50: {
            if (packet.size() < 7) return false;
            const auto destination = Pm4::DmaDestination(packet);
            if (destination == Pm4::DmaDestinationNowhere) return true;
            const auto bytes = Pm4::DmaBytes(packet);
            if (bytes == 0) return true;
            const auto source = Pm4::DmaSource(packet);
            if ((source == 0 || source == 3) && DrawPipeline::Queue0().Overlaps(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u), bytes)) return false;
            if (source == Pm4::DmaSelectGds && DrawPipeline::Queue0().Overlaps(Pm4::GdsAddress() + packet[2], bytes)) return false;
            const auto address = destination == Pm4::DmaSelectGds ? Pm4::GdsAddress() + packet[4] : packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
            return !DrawPipeline::Queue0().Overlaps(address, bytes);
        }
        case 0x3c: case 0x93: {
            if (packet.size() < 5) return false;
            const auto bytes = Pm4::WaitAwaitedBytes(packet);
            const std::uint64_t awaited = (packet[2] & (bytes == 8 ? ~7u : ~3u)) | (static_cast<std::uint64_t>(packet[3]) << 32u);
            if (DrawPipeline::Queue0().Overlaps(awaited, bytes)) return false;
            GuestMemory::CheckRange(reinterpret_cast<const void*>(awaited), bytes, bytes);
            return Pm4::WaitSatisfiedUnchecked(packet);
        }
        case 0x10: {
            const auto operation = (header >> 2u) & 0x3fu;
            return operation == 0 || operation == 0x09 || operation == 0x0b || operation == 0x0c || operation == 0x1a;
        }
        default: return false;
    }
}

std::size_t verdictRanges(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode, std::array<std::pair<std::uint64_t, std::uint64_t>, 3>& ranges) {
    if (header == FlipPacketHeader || header == RenderingWaitPacketHeader) return 0;
    switch (opcode) {
        case 0x63: case 0x64: case 0x9f: {
            if (packet.size() < 5) return 0;
            const auto address = packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u);
            ranges[0] = {address, address + static_cast<std::size_t>(packet[4]) * 8};
            return 1;
        }
        case 0x50: {
            if (packet.size() < 7) return 0;
            const auto destination = Pm4::DmaDestination(packet);
            if (destination == Pm4::DmaDestinationNowhere) return 0;
            const auto bytes = Pm4::DmaBytes(packet);
            if (bytes == 0) return 0;
            std::size_t count = 0;
            const auto source = Pm4::DmaSource(packet);
            if (source == 0 || source == 3) {
                const auto from = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
                ranges[count++] = {from, from + bytes};
            }
            if (source == Pm4::DmaSelectGds) {
                const auto from = Pm4::GdsAddress() + packet[2];
                ranges[count++] = {from, from + bytes};
            }
            const auto address = destination == Pm4::DmaSelectGds ? Pm4::GdsAddress() + packet[4] : packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
            ranges[count++] = {address, address + bytes};
            return count;
        }
        case 0x3c: case 0x93: {
            if (packet.size() < 5) return 0;
            const auto bytes = Pm4::WaitAwaitedBytes(packet);
            const std::uint64_t awaited = (packet[2] & (bytes == 8 ? ~7u : ~3u)) | (static_cast<std::uint64_t>(packet[3]) << 32u);
            ranges[0] = {awaited, awaited + bytes};
            return 1;
        }
        default: return 0;
    }
}

namespace {

bool registerLoad(std::uint32_t opcode) {
    return opcode == 0x63 || opcode == 0x64 || opcode == 0x9f;
}

}

bool awaitVerdictWriters(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode) {
    std::array<std::pair<std::uint64_t, std::uint64_t>, 3> ranges{};
    const auto count = verdictRanges(packet, header, opcode, ranges);
    if (count != 0) {
        auto& pipeline = DrawPipeline::Queue0();
        const std::span<const DrawPipeline::Range> asked(ranges.data(), count);
        if (registerLoad(opcode) && DrawFinisher::FrameFlow() && pipeline.AwaitExactWriters(asked, DrawPipeline::DrainReason::Packet, opcode, DeepSite::PacketVerdict, FrameSite::ListWait, FrameCheck::ListBytes)) return true;
        while (pipeline.Busy() && pipeline.OverlapsAny(asked)) pipeline.AwaitWriters(asked, DrawPipeline::DrainReason::Packet, opcode, DeepSite::PacketVerdict);
    }
    return keepsDrawPipeline(packet, header, opcode);
}

bool awaitVerdictFinisherWriters(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode) {
    std::array<std::pair<std::uint64_t, std::uint64_t>, 3> ranges{};
    const auto count = verdictRanges(packet, header, opcode, ranges);
    if (count == 0) return keepsDrawPipeline(packet, header, opcode);
    const std::span<const DrawFinisher::Range> asked(ranges.data(), count);
    if (registerLoad(opcode) && DrawFinisher::FrameFlow()) DrawFinisher::Get().AwaitExactPending(asked, DeepSite::PacketFinisher, FrameSite::ListWait);
    else DrawFinisher::Get().AwaitPending(asked, DeepSite::PacketFinisher);
    return true;
}

void awaitPredicationWriters(std::uint64_t address, std::size_t bytes, std::uint32_t opcode) {
    auto& pipeline = DrawPipeline::Queue0();
    if (!DrawFinisher::Deep()) {
        pipeline.Drain(DrawPipeline::DrainReason::Packet, opcode);
        return;
    }
    const DrawPipeline::Range range{address, address + bytes};
    do pipeline.AwaitWriters({&range, 1}, DrawPipeline::DrainReason::Packet, opcode, DeepSite::Predication);
    while (pipeline.Busy() && pipeline.Overlaps(address, bytes));
}

FlipForm flipForm(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode) {
    if (header == FlipPacketHeader) return FlipForm::Header;
    if (opcode != 0x49 || packet.size() != 8) return FlipForm::Kept;
    if (const auto label = Pm4::DecodeLabelWrite(packet)) {
        const auto bytes = label->Bytes();
        return DeferLabels() && bytes.size() <= DeferredLabel::Capacity && bytes.size() % 4 == 0 && label->address % 4 == 0 ? FlipForm::EopLabel : FlipForm::Kept;
    }
    const bool storesNothing = (packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0;
    return storesNothing ? FlipForm::EopNothing : FlipForm::Kept;
}

bool sealsWriteOnly(std::span<const std::uint32_t> packet, std::uint32_t opcode) {
    if (opcode == 0x58) return true;
    if (opcode != 0x46 || packet.size() < 2) return false;
    switch (packet[1] & 0x3fu) {
        case 0x07: case 0x0f: case 0x10: case 0x16: case 0x2a: case 0x2c: case 0x2e: case 0x31: return true;
        default: return false;
    }
}

bool& drawnSinceSeal() {
    static thread_local bool drawn = false;
    return drawn;
}

void dumpQueue(const Submission& submission) {
    static const long dumpQueue = [] { const char* text = std::getenv("APS5_DUMP_QUEUE"); return text ? std::strtol(text, nullptr, 16) : -1L; }();
    if (static_cast<long>(submission.queue) == dumpQueue) {

        static int dumped = 0;
        if (dumped++ < 40) {
            std::string text = "[queue] submission " + std::to_string(submission.serial) + ":\n";
            for (std::size_t cursor = 0; cursor < submission.commands.size();) {
                const auto header = submission.commands[cursor];
                const auto count = Pm4::PacketWords(header);
                char line[200];
                int length = std::snprintf(line, sizeof(line), "[queue]   %s", Pm4::Name(header).c_str());
                for (std::size_t i = 1; i < count && i < 10 && length < 180; ++i) length += std::snprintf(line + length, sizeof(line) - length, " %08x", submission.commands[cursor + i]);
                text += line;
                text += "\n";
                cursor += count;
            }
            std::fputs(text.c_str(), stderr);
        }
    }
}

}

using namespace PacketRules;

void Driver::deliverGuestEopInterrupt(std::uint32_t queue, std::uint32_t context) {
    AgcDriverDeliverEopInterrupt(queue, context);
}

void Driver::abAtFlip() {
    std::vector<std::string> lines;
    {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        lines = Graphics::AbAtPresent(flipsCounted.load());
    }
    for (const auto& line : lines) AgcDriver::ProfilePrint_nid_no_patch("%s", line.c_str());
    if (!lines.empty()) ThreadReport::Phase();
}

void Driver::abAtFlip(double seconds) {
    std::vector<std::string> lines;
    {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        lines = Graphics::AbAtPresent(flipsCounted.load(), seconds);
    }
    if (DrawFinisher::FrameVerifyDue(FrameCheck::AbPhase)) {
        const bool begins = std::any_of(lines.begin(), lines.end(), [](const std::string& line) { return line.find(" begins at present ") != std::string::npos || line.find("in-run A/B:") != std::string::npos; });
        DrawFinisher::NoteFrameCheck(FrameCheck::AbPhase, begins, "an A/B phase began at the ordered flip of present " + std::to_string(flipsCounted.load()) + ", where the pipeline is not quiescent");
    }
    for (const auto& line : lines) AgcDriver::ProfilePrint_nid_no_patch("%s", line.c_str());
    if (!lines.empty()) ThreadReport::Phase();
}

bool Driver::flipOrdered(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode, bool predicated, PacketRules::FlipForm& form, double& abSeconds) {
    form = PacketRules::FlipForm::Kept;
    abSeconds = 0;
    if (!DrawFinisher::FrameFlowWanted()) return false;
    static const bool syncFlip = std::getenv("APS5_SYNC_FLIP") != nullptr || std::getenv("APS5_DRAIN_ALL") != nullptr;
    const auto kept = [](FrameKept reason) {
        DrawFinisher::NoteFrameKept(reason);
        return false;
    };
    if (predicated) return kept(FrameKept::Predicated);
    if (syncFlip) return kept(FrameKept::SyncFlip);
    if (APS5_ENABLE_TIMING_LOG) return kept(FrameKept::TimingLog);
    if (!deferredLabels().labels.empty()) return kept(FrameKept::DeferredLabels);
    form = PacketRules::flipForm(packet, header, opcode);
    if (form == PacketRules::FlipForm::Kept) return kept(FrameKept::LabelForm);
    if (Graphics::AbConfigured()) {
        abSeconds = Graphics::AbClockSeconds();
        if (Graphics::AbBeginDue(flipsCounted.load() + 1, abSeconds)) return kept(FrameKept::AbPhase);
    }
    if (!DrawFinisher::FrameFlow()) DrawFinisher::Get().LatchFrameFlow();
    if (!DrawFinisher::FrameFlow()) return kept(FrameKept::NotLatched);
    return true;
}

void Driver::queueOrderedFlip(FlipCommit commit) {
    const auto start = DrawFinisher::FrameCensus() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<DrawPipeline::Range> writes;
    std::uint64_t labelAddress = 0;
    std::vector<std::byte> labelBytes;
    if (commit.label) {
        const auto label = Pm4::DecodeLabelWrite(commit.words);
        require(label.has_value(), "AGC driver: the label of an ordered flip decodes to no label write");
        const auto bytes = label->Bytes();
        writes.emplace_back(label->address, label->address + bytes.size());
        if ((commit.words[2] >> 29u) != 3u) {
            labelAddress = label->address;
            labelBytes.assign(bytes.begin(), bytes.end());
        }
    }
    auto& finisher = DrawFinisher::Get();
    auto& pipeline = DrawPipeline::Queue0();
    finisher.FlushOpenRun();
    if (start != std::chrono::steady_clock::time_point{}) DrawFinisher::NoteFlipDistance(DrawFinisher::OutstandingJobs(), pipeline.Outstanding(), finisher.FlipsQueued() - finisher.FlipsCommitted());
    const auto issued = finisher.Issued();
    if (DrawFinisher::FrameVerifyDue(FrameCheck::FlipOrder)) {
        auto probe = std::make_shared<FlipProbe>();
        probe->jobs = issued;
        probe->items = pipeline.Enqueued();
        commit.probe = std::move(probe);
    }
    DrawFinisher::NoteFlipJobs(issued);
    finisher.NoteFlipQueued();
    pipeline.EnqueueFlip([this, flip = std::move(commit)] { commitFlip(flip); }, std::move(writes), labelAddress, std::move(labelBytes));
    const auto after = finisher.Issued();
    finisher.NoteFlipSeq(after != issued ? after - 1 : 0);
    if (start != std::chrono::steady_clock::time_point{}) DrawFinisher::NoteFrameSite(FrameSite::OrderedFlip, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
}

void Driver::commitFlip(const FlipCommit& flip) {
    struct Settled {
        bool flipJobs = true;
        ~Settled() {
            if (flipJobs) DrawFinisher::RetireFlipJobs();
            DrawFinisher::Get().NoteFlipCommitted();
        }
    } settled;
    try {
        const DrawPipeline::FlipCommitScope committing;
        GuestMemory::SetCurrentPacket(flip.header ? 0xffffu : 0x49u, 0);
        if (flip.probe != nullptr) {
            const auto front = DrawFinisher::Get().Front();
            const auto retiredNow = DrawPipeline::Queue0().Retired();
            char text[256];
            std::snprintf(text, sizeof(text), "the flip of frame %llu queued with the finisher at job %llu and %llu committer items before it committed with the front at %llu and %llu items retired", static_cast<unsigned long long>(flip.frameNumber), static_cast<unsigned long long>(flip.probe->jobs), static_cast<unsigned long long>(flip.probe->items), static_cast<unsigned long long>(front), static_cast<unsigned long long>(retiredNow));
            DrawFinisher::NoteFrameCheck(FrameCheck::FlipOrder, front < flip.probe->jobs || retiredNow < flip.probe->items, text);
        }
        if (flip.header) {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            static_cast<void>(Graphics::StorageTexture::StoreSmallPending("flip"));
        }
        std::uint64_t batchesAtFlip = 0, unsignaledAtFlip = 0;
        Graphics::StorageTexture::NoteFlip();
        {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            const auto localDevice = device.Load();
            if (localDevice != nullptr) static_cast<void>(Graphics::StorageTexture::StoreDrawnLinear("flip"));
            Graphics::StorageTexture::ReleaseIdleCopies();
            if (flip.label) {
                const auto label = Pm4::DecodeLabelWrite(flip.words);
                require(label.has_value(), "AGC driver: the label of an ordered flip decodes to no label write");
                const auto bytes = label->Bytes();
                auto& deferred = deferredLabels();
                require(deferred.labels.empty(), "AGC driver: the committer holds deferred labels at an ordered flip");
                deferred.since = std::chrono::steady_clock::now();
                auto& entry = deferred.labels.emplace_back();
                entry.address = label->address;
                entry.size = bytes.size();
                std::memcpy(entry.bytes.data(), bytes.data(), bytes.size());
                ++queuedLabels;
            }
            if (!recordLabelsForPacket(localDevice.get(), 0) && localDevice != nullptr) localDevice->SubmitRecorded(true, Graphics::Recorder::SubmitReason::Flip);
            if (localDevice != nullptr) localDevice->FlipBatches(batchesAtFlip, unsignaledAtFlip);
        }
        Graphics::NoteHostImportFrame();
        DrawFinisher::RetireFlipJobs();
        settled.flipJobs = false;
        if (batchesAtFlip != 0) flipSerial = batchesAtFlip;
        flipBatchesUnsignaled += unsignaledAtFlip;
        flip.frame->NoteFlipBatches(batchesAtFlip, unsignaledAtFlip);
        CaptureTrace::Log("flip frame=%llu submission=%llu offset=%zu batch=%llu unsignaled=%llu", static_cast<unsigned long long>(flip.frameNumber), static_cast<unsigned long long>(flip.serial), static_cast<std::size_t>(flip.cursor), static_cast<unsigned long long>(batchesAtFlip), static_cast<unsigned long long>(unsignaledAtFlip));
        if (flip.releases) releaseFlipHold();
        flip.request->GpuReady(flip.frame);
    } catch (const ProcessShutdown&) {
        throw;
    } catch (...) {
        flip.request->Fail(std::current_exception());
        throw;
    }
}

bool Driver::suspendOrdered() {
    if (!DrawFinisher::OrderedSyncsWanted()) return false;
    static const bool suspendDrain = std::getenv("APS5_SUSPEND_DRAIN") != nullptr;
    static const bool orderedCompletion = std::getenv("APS5_PIPELINE_DRAIN_SUBMISSIONS") == nullptr && std::getenv("APS5_PIPELINE_DRAIN_LABELS") == nullptr;
    const auto kept = [](FrameKept reason) {
        DrawFinisher::NoteFrameKept(reason);
        return false;
    };
    if (APS5_ENABLE_TIMING_LOG) return kept(FrameKept::SuspendTimingLog);
    if (suspendDrain) return kept(FrameKept::SuspendDrainEnv);
    if (!deferredLabels().labels.empty()) return kept(FrameKept::SuspendDeferredLabels);
    if (!orderedCompletion) return kept(FrameKept::SuspendUnordered);
    if (!DrawFinisher::OrderedSyncs()) DrawFinisher::Get().LatchFrameFlow();
    if (!DrawFinisher::OrderedSyncs()) return kept(FrameKept::SuspendNotLatched);
    return true;
}

void Driver::queueOrderedSuspend(const Submission& submission) {
    const auto start = DrawFinisher::FrameCensus() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ++suspendPoints;
    {
        std::optional<DrawPipeline::OrderProbe> probe;
        if (DrawFinisher::FrameVerifyDue(FrameCheck::SuspendOrder)) probe.emplace(FrameCheck::SuspendOrder);
        DrawPipeline::Queue0().EnqueueSuspend([this, serial = submission.serial, after = submission.suspendAfter] { commitSuspend(serial, after); });
    }
    completionDeferred() = true;
    resetGraphics = true;
    DrawFinisher::NoteFrameSite(FrameSite::SuspendOrdered, start == std::chrono::steady_clock::time_point{} ? 0 : static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
}

void Driver::commitSuspend(std::uint64_t serial, std::uint64_t after) {
    struct Complete {
        Driver& driver;
        std::uint64_t serial;
        ~Complete() {
            bool notify = true;
            {
                std::lock_guard lock(driver.mutex);
                driver.markCompleted(serial);
                notify = driver.idleWaiters != 0 || driver.orderHolders.load(std::memory_order_acquire) != 0 || completionsPending();
            }
            if (notify) driver.changed.notify_all();
        }
    } complete{*this, serial};
    verifySuspendCommit(serial, after);
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    auto& costs = submissionCosts(0);
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    if (!Graphics::Recorder::PendingLabelSince().has_value() && Graphics::Recorder::RecordedWorkSinceSubmit() == 0) {
        ++costs.suspendsSkipped;
        return;
    }
    if (!GuestMemory::RangeProofs()) GuestMemory::BreakProofs(GuestMemory::ProofBreak::GpuLock);
    if (const auto localDevice = device.Load()) localDevice->SubmitRecorded(true, Graphics::Recorder::SubmitReason::Suspend);
    ++costs.suspends;
    if (profile) costs.suspendNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

void Driver::verifySuspendCommit(std::uint64_t serial, std::uint64_t after) {
    if (!DrawFinisher::FrameVerifyDue(FrameCheck::SuspendCompletion)) return;
    std::size_t open = 0;
    std::uint64_t first = 0;
    {
        std::lock_guard lock(mutex);
        for (const auto received : queue0Uncommitted) {
            if (received > after) continue;
            if (open++ == 0) first = received;
        }
    }
    char text[256];
    std::snprintf(text, sizeof(text), "the ordered suspend point of serial %llu committed with %zu queue-0 submissions received up to event %llu not yet completed, the first received at event %llu", static_cast<unsigned long long>(serial), open, static_cast<unsigned long long>(after), static_cast<unsigned long long>(first));
    DrawFinisher::NoteFrameCheck(FrameCheck::SuspendCompletion, open != 0, text);
}

std::optional<std::uint32_t> Driver::releaseInterrupt(std::span<const std::uint32_t> packet, std::uint32_t opcode, bool carriesFlip) {
    if (opcode != 0x49 || carriesFlip || packet.size() < 3) return std::nullopt;
    const auto interruptSelect = (packet[2] >> 24u) & 7u;
    if (interruptSelect == 0 || interruptSelect == 3) return std::nullopt;
    if (packet.size() < 8) throw std::runtime_error("AGC: an interrupting release has no interrupt context dword");
    const auto context = packet[7] & 0x0fffffffu;
    if ((context & 0x08000000u) != 0) throw std::runtime_error("AGC: a release asks for typed interrupt context " + std::to_string(context) + ", which is not delivered as a graphics end-of-pipe event; not implemented");
    return context;
}

template <bool Lean>
bool Driver::endSegment(const Submission& submission, FlipHold& flipHold, bool pipelined, FrontMode front) {
    const auto switchSet = [&](Graphics::RuntimeSwitch id) { return front == FrontMode::Lean ? LeanFront::SwitchSet(frontLatch, id) : Graphics::RuntimeSwitches::Set(id); };
    static const bool orderedCompletion = std::getenv("APS5_PIPELINE_DRAIN_SUBMISSIONS") == nullptr && std::getenv("APS5_PIPELINE_DRAIN_LABELS") == nullptr;
    const bool streamEnd = submission.queue == 0 && pipelined && orderedCompletion && !submission.progress && submission.rewindTail != nullptr && deferredLabels().labels.empty() && (std::atomic_ref<std::uint32_t>(*const_cast<std::uint32_t*>(submission.rewindTail - 1)).load(std::memory_order_acquire) & 0x80000000u) != 0 && switchSet(Graphics::RuntimeSwitch::XpStreamSegments);
    const bool frameFlow = submission.queue == 0 && DrawFinisher::FrameFlow();
    if (submission.queue == 0 && !streamEnd) {
        if (Lean && frameFlow) {
            DrawFinisher::NoteFrameSite(FrameSite::SegmentEnd);
        } else {
            if (frameFlow && DrawFinisher::Busy()) DrawFinisher::NoteFrameKept(FrameKept::FullLoopSegmentEnd);
            DrawFinisher::NoteDeepKept(DeepKept::SegmentEnd);
            DrawFinisher::Get().Drain();
        }
    }
    PerformanceContext endContext(submission.queue == 0 ? frameTiming() : FrameTiming::Async());
    PerformanceTimer endTiming("Driver.EndSubmission");
    static const bool submitAtEnd = std::getenv("APS5_SUBMIT_AT_END") != nullptr;
    if (streamEnd) {
        const bool holdsFlip = std::exchange(flipHold.working, false);
        auto enqueue = [this, holdsFlip, flipWork = flipHold.work] {
            DrawPipeline::Queue0().EnqueueQuiet([this, holdsFlip, flipWork] {
                if (holdsFlip) finishFlipWork(flipWork, submitFlipWork(0));
                else submitDueAfterCommit(device.Load().get());
                wakeQueueZeroForCompletions();
            });
        };
        if (DrawFinisher::Overlap() && DrawFinisher::Busy()) {
            DrawFinisher::Pending held;
            held.unknown = false; held.origin = 6;
            DrawFinisher::Get().Enqueue(std::move(enqueue), std::move(held));
        } else {
            enqueue();
        }
        noteXpStream(false);
        endTiming.Finish();
        if constexpr (!Lean) executeRewindTail(submission);
        return true;
    }
    if (pipelined && orderedCompletion && deferredLabels().labels.empty()) {
        std::optional<DrawPipeline::OrderProbe> probe;
        if (Lean && frameFlow && DrawFinisher::FrameVerifyDue(FrameCheck::SegmentEnd)) probe.emplace(FrameCheck::SegmentEnd);
        if (submission.rewindTail == nullptr) {
            flipHold.working = false;
            deferCompletion(submission);
            return false;
        }
        if ((std::atomic_ref<std::uint32_t>(*const_cast<std::uint32_t*>(submission.rewindTail - 1)).load(std::memory_order_acquire) & 0x80000000u) != 0) {
            const bool holdsFlip = std::exchange(flipHold.working, false);
            DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::SubmitEnd, [this, holdsFlip, flipWork = flipHold.work, progress = submission.progress] {
                if (holdsFlip) finishFlipWork(flipWork, submitFlipWork(0));
                else submitOpenWork(progress);
                wakeQueueZeroForCompletions();
            }, {});
            endTiming.Finish();
            if constexpr (!Lean) executeRewindTail(submission);
            return true;
        }
    }
    if (pipelined) {
        DrawFinisher::NoteDeepKept(DeepKept::Submission);
        DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
    }
    if (!deferredLabels().labels.empty() || Graphics::Recorder::PendingLabelSince().has_value() || Graphics::Recorder::RecordedWorkSinceSubmit() != 0) {
        auto& costs = submissionCosts(submission.queue);
        if (!submitAtEnd && submission.rewindTail == nullptr && submission.queue == 0 && workerQueued() != nullptr && workerQueued()->load(std::memory_order_acquire) != 0) {
            ++costs.endSkipped;
            flipHold.Finish();
            return false;
        }
        ++costs.endSubmits;
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::End);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        const auto localDevice = device.Load();

        recordDeferredLabels(localDevice.get(), submission.queue);
        if (localDevice != nullptr) localDevice->SubmitRecorded(submission.queue == 0, Graphics::Recorder::SubmitReason::End);
    }
    flipHold.Finish();
    endTiming.Finish();
    if (submission.rewindTail == nullptr) return false;
    if (submission.progress) submitOpenWork(submission.progress);
    if constexpr (!Lean) executeRewindTail(submission);
    return true;
}

template bool Driver::endSegment<false>(const Submission& submission, FlipHold& flipHold, bool pipelined, FrontMode front);
template bool Driver::endSegment<true>(const Submission& submission, FlipHold& flipHold, bool pipelined, FrontMode front);

void Driver::execute(Submission& submission) {
    const auto front = submission.front != FrontMode::Unlatched ? submission.front : latchFront(submission.queue, submission.suspend);
    if (front == FrontMode::Lean) {
        executeLean(submission);
        return;
    }
    struct SubmissionReport {
        bool reported = false;
        bool queue0 = false;
        bool previous = false;
        ThreadReport::Clock::time_point start;
        SubmissionReport(bool reported, bool queue0) : reported(reported), queue0(queue0) {
            if (!reported) return;
            previous = ThreadReport::EnterQueue0Worker(queue0);
            if (queue0) start = ThreadReport::Clock::now();
        }
        SubmissionReport(const SubmissionReport&) = delete;
        SubmissionReport& operator=(const SubmissionReport&) = delete;
        ~SubmissionReport() {
            if (!reported) return;
            if (queue0) ThreadReport::WorkerSubmission(start, ThreadReport::Clock::now());
            ThreadReport::LeaveQueue0Worker(previous);
        }
    } submissionReport{ThreadReport::Enabled(), submission.queue == 0};
    const ThreadReport::WorkerStateScope executing(ThreadReport::WorkerState::Packets, submission.queue == 0);
    auto* submissionTiming = includeTimingSubmission(submission, true);
    FlipHold flipHold{*this, submission.queue, submission.holdsFlip, submission.holdsFlip, submission.flipWork};
    const bool pipelined = submission.queue == 0 && DrawPipeline::Depth() != 0;
    const LeanFront::SegmentNesting nesting(submission.queue == 0 ? &frontLatch : nullptr);
    const auto switchSet = [&](Graphics::RuntimeSwitch id) { return front == FrontMode::Lean ? LeanFront::SwitchSet(frontLatch, id) : Graphics::RuntimeSwitches::Set(id); };
    static const bool orderedWaits = std::getenv("APS5_PIPELINE_DRAIN_LABELS") == nullptr;
    DrawPipeline::Active() = pipelined;
    if (pipelined && submission.suspend && suspendOrdered()) {
        queueOrderedSuspend(submission);
        return;
    }
    if (pipelined && DrawFinisher::FrameFlowWanted() && DrawFinisher::Busy()) {
        DrawFinisher::NoteFrameKept(FrameKept::FrontChange);
        DrawFinisher::Get().Drain();
    }
    if (pipelined && DrawFinisher::Get().Adopt(false)) leanDraws.ReleaseDevice();
    if (pipelined) DrawFinisher::Get().LatchFrameFlow();
    struct PipelineDrain {
        bool pipelined;
        ~PipelineDrain() {
            if (!pipelined || std::uncaught_exceptions() == 0) return;
            try {
                DrawFinisher::NoteDeepKept(DeepKept::Submission);
                DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
            } catch (...) {
            }
        }
    } pipelineDrain{pipelined};
    static const bool orderedCompletion = std::getenv("APS5_PIPELINE_DRAIN_SUBMISSIONS") == nullptr && orderedWaits;
    if (pipelined && (submission.suspend || !orderedCompletion)) {
        DrawFinisher::NoteDeepKept(DeepKept::Submission);
        DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
    }
    if (submission.suspend) {
        PerformanceContext timingContext(submissionTiming);
        PerformanceTimer timing("Driver.Suspend");

        static const bool suspendDrain = std::getenv("APS5_SUSPEND_DRAIN") != nullptr;
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        ++suspendPoints;
        auto& costs = submissionCosts(submission.queue);

        if (suspendDrain || !deferredLabels().labels.empty() || Graphics::Recorder::PendingLabelSince().has_value() || Graphics::Recorder::RecordedWorkSinceSubmit() != 0) {
            const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            const auto localDevice = device.Load();
            recordDeferredLabels(localDevice.get(), submission.queue);
            if (localDevice != nullptr) {
                if (suspendDrain) localDevice->WaitIdle();
                else localDevice->SubmitRecorded(submission.queue == 0, Graphics::Recorder::SubmitReason::Suspend);
            }
            ++costs.suspends;
            if (profile) costs.suspendNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        } else {
            ++costs.suspendsSkipped;
        }
        resetGraphics = true;
        return;
    }
    QueueState* state = nullptr;
    {
        std::lock_guard lock(mutex);
        if (submission.queue == 0 && resetGraphics) {
            queues.erase(0);
            resetGraphics = false;
        }
        state = &queues[submission.queue];
    }
    auto& queue = *state;
    static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
    if (traceGpu) std::fprintf(stderr, "[gpu] %.1f execute serial=%llu queue=0x%x dwords=%zu\n", TraceMs(), static_cast<unsigned long long>(submission.serial), submission.queue, submission.commands.size());

    dumpQueue(submission);
    PacketHistory recent{submission.commands};

    static const bool profilePackets = std::getenv("APS5_PROFILE_DRAW") != nullptr;

    struct PacketProfileStorage {};
    auto& packetProfile = HostThreadLocal<PacketProfile, PacketProfileStorage>();
    ++packetProfile.submissions;
    auto& packetCounters = PacketCounters::ThisThread();

    static const bool epochEach = std::getenv("APS5_SUBMISSION_EPOCH_EACH") != nullptr;
    const bool orderedAtSubmit = submission.rewindTail == nullptr && submission.enqueuedAt != std::chrono::steady_clock::time_point{};
    const auto submittedAt = orderedAtSubmit ? static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(submission.enqueuedAt.time_since_epoch()).count()) : 0;
    if (epochEach || submission.enqueuedAt == std::chrono::steady_clock::time_point{} || submission.enqueuedAt > lastEpochBump()) {
        if (!epochEach) noteReleasedTails(submission, front);
        bumpEpochSince(&EpochBumps::submissions, submittedAt);
    } else {
        raiseEpochSince(submittedAt);
    }
    clearResolvedAhead();
    gateOpened() = true;
    struct ClearAhead {
        ~ClearAhead() { clearResolvedAhead(); }
    } clearAhead;
    struct SettleFinisher {
        bool queue0;
        ~SettleFinisher() {
            if (!queue0) return;
            if (DrawFinisher::FrameFlow() && DrawFinisher::Busy()) DrawFinisher::NoteFrameKept(FrameKept::FullLoopSettle);
            DrawFinisher::NoteDeepKept(DeepKept::Settle);
            DrawFinisher::Get().Settle();
        }
    } settleFinisher{submission.queue == 0};
    std::size_t listHint = 0;
    std::size_t nextSegment = 0;
    auto* const executingPacket = workerPacket();
    for (std::size_t cursor = 0; cursor < submission.commands.size();) {
        if (packetEpoch()) bumpEpoch(&EpochBumps::packets);
        CheckFailure();
        while (nextSegment < submission.indirectSegments.size() && submission.indirectSegments[nextSegment].start <= cursor) {
            if (submission.indirectSegments[nextSegment].start == cursor && refreshIndirectSegment(submission, nextSegment)) {
                recent.commands = submission.commands;
                continue;
            }
            ++nextSegment;
        }
        if (cursor >= submission.commands.size()) break;
        const auto header = submission.commands[cursor];
        if (Pm4::FillerPacket(header)) { ++cursor; continue; }
        const auto count = Pm4::PacketWords(header);
        const auto packet = std::span(submission.commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        if (executingPacket != nullptr) executingPacket->store((static_cast<std::uint64_t>(cursor + 1) << 8u) | opcode, std::memory_order_relaxed);
        auto nextCursor = cursor + count;
        const bool predicated = Pm4::Predicated(header) && queue.predication.operation != 0;
        const auto asksVerdict = [&] { return header != FlipPacketHeader && header != RenderingWaitPacketHeader && !submission.flips.contains(cursor) && (opcode == 0x50 || opcode == 0x63 || opcode == 0x64 || opcode == 0x9f || opcode == 0x3c || opcode == 0x93); };
        const auto noteKept = [&] {
            if (!DrawFinisher::Deep()) return;
            const auto kept = header == FlipPacketHeader || submission.flips.contains(cursor) ? DeepKept::Flip : header == RenderingWaitPacketHeader ? DeepKept::RenderingWait : opcode == 0x3c || opcode == 0x93 ? DeepKept::Wait : opcode == 0x49 || opcode == 0x37 ? DeepKept::WorkerLabel : DeepKept::PacketVerdict;
            DrawFinisher::NoteDeepKept(kept, opcode);
        };
        if (submission.queue == 0 && Pm4::DrawOpcode(opcode) && !Pm4::IndirectDrawOpcode(opcode) && !predicated && switchSet(Graphics::RuntimeSwitch::XpKeepDraws) && deferredLabels().labels.empty() && xpDropsDraw()) {
            recent.Record(cursor);
            cursor = nextCursor;
            continue;
        }
        PacketRules::FlipForm flipKind = PacketRules::FlipForm::Kept;
        double abSeconds = 0;
        const bool orderedFlip = pipelined && (header == FlipPacketHeader || opcode == 0x49) && submission.flips.contains(cursor) && flipOrdered(packet, header, opcode, predicated, flipKind, abSeconds);
        const bool orderedWait = pipelined && header == RenderingWaitPacketHeader && !predicated && DrawFinisher::FrameFlow();
        const bool ordered = orderedFlip || orderedWait;
        bool dmaBeside = false;
        DrawPipeline::Range argumentRange{};
        const bool argumentWait = pipelined && opcode == 0x16 && argumentWaitWanted(packet, queue, predicated, argumentRange);
        DrawPipeline::Range indirectRange{};
        const bool indirectQueued = pipelined && opcode == 0x16 && !argumentWait && orderedIndirectWanted(packet, queue, predicated, indirectRange);
        if (submission.queue == 0 && DrawFinisher::Busy() && !keepsDrawFinisher(header, opcode)) {
            const bool queuesLabel = pipelined && !predicated && header != FlipPacketHeader && header != RenderingWaitPacketHeader && !submission.flips.contains(cursor) && labelPacketQueues(packet, opcode);
            const bool heldLabelWait = pipelined && !predicated && orderedWaits && (opcode == 0x3c || opcode == 0x93) && header != RenderingWaitPacketHeader && waitTakesPendingLabel(packet);
            const bool queuesSampleDump = pipelined && !predicated && (switchSet(Graphics::RuntimeSwitch::BesideSampleDumps) || DrawFinisher::Flow() || DrawFinisher::Deep()) && sampleDumpQueues(packet, opcode, submission.queue);
            if (pipelined && !predicated && opcode == 0x50 && DrawFinisher::Overlap() && DrawFinisher::FrameFlow()) {
                const auto form = dmaPacketForm(packet, opcode, submission.queue);
                dmaBeside = form == PacketRules::DmaForm::Queues;
                if (dmaBeside) DrawFinisher::NoteFrameSite(FrameSite::DmaBeside);
                else if (form == PacketRules::DmaForm::Copy) DrawFinisher::NoteFrameKept(FrameKept::DmaCopy);
                else if (form == PacketRules::DmaForm::Gds) DrawFinisher::NoteFrameKept(FrameKept::DmaGds);
            }
            bool beside = DrawFinisher::Overlap() && opcode != 0x15 && (queuesLabel || heldLabelWait || queuesSampleDump || dmaBeside || keepsDrawPipeline(packet, header, opcode));
            if (!beside && pipelined && DrawFinisher::Overlap() && opcode != 0x15 && DrawFinisher::Deep() && asksVerdict()) beside = awaitVerdictFinisherWriters(packet, header, opcode);
            if (!beside && argumentWait) {
                DrawFinisher::Get().AwaitExactPending({&argumentRange, 1}, DeepSite::IndirectArguments, FrameSite::ArgumentWait);
                beside = true;
            }
            if (!beside && indirectQueued) beside = true;
            if (!beside && !ordered && !DrawFinisher::FlowSkips(DrawFinisher::FlowSite::PacketFinisher)) {
                if (opcode == 0x16 && DrawFinisher::FrameFlow()) DrawFinisher::NoteFrameKept(FrameKept::IndirectGpu);
                noteKept();
                DrawFinisher::Get().Drain();
            }
            if (DrawFinisher::Overlap()) {
                DrawFinisher::NoteDrain(DrawFinisher::DrainSite::Packet, !beside && !ordered);
                if (!beside && !ordered) DrawFinisher::NoteDrainedOpcode(header == FlipPacketHeader ? 0xffu : opcode, keepsDrawPipeline(packet, header, opcode));
            }
        }
        if (APS5_ENABLE_TIMING_LOG && submission.queue == 0 && pendingFrameTiming == nullptr) includeTimingSubmission(submission, false);
        PerformanceContext timingContext(submission.queue == 0 ? pendingFrameTiming.get() : FrameTiming::Async());
        if (submission.queue != 0 && Graphics::Recorder::DispatchOverlapEnabled()) noteRingPacket(submission.queue, submission.serial, opcode, header);
        PerformanceTimer timing("Driver.Packet");
        if (!predicated) noteLabelIssue(packet, submission.queue);
        std::optional<DrawPipeline::OrderProbe> dmaProbe;
        if (dmaBeside && DrawFinisher::FrameVerifyDue(FrameCheck::DmaOrder)) dmaProbe.emplace(FrameCheck::DmaOrder);
        if (pipelined && !predicated && header != FlipPacketHeader && header != RenderingWaitPacketHeader && !submission.flips.contains(cursor) && (enqueueLabelPacket(packet, opcode, submission.queue, releaseInterrupt(packet, opcode, false)) || enqueueDmaPacket(packet, opcode, submission.queue, queue) || enqueueConstantRamDump(packet, opcode, submission.queue, queue) || enqueueSampleDumpPacket(packet, opcode, submission.queue))) {
            traceLabel(packet, submission.queue);
            recent.Record(cursor);
            cursor = nextCursor;
            continue;
        }
        dmaProbe.reset();
        if (dmaBeside) {
            DrawFinisher::NoteFrameKept(FrameKept::DmaRefused);
            if (DrawFinisher::Busy()) {
                bool beside = keepsDrawPipeline(packet, header, opcode);
                if (!beside && DrawFinisher::Deep() && asksVerdict()) beside = awaitVerdictFinisherWriters(packet, header, opcode);
                if (!beside) {
                    noteKept();
                    DrawFinisher::Get().Drain();
                }
            }
        }
        if (pipelined && (opcode == 0x50 || (opcode == 0x46 && Pm4::SampleDump(packet))) && (DrawFinisher::Get().FlipsQueued() != DrawFinisher::Get().FlipsCommitted() || DrawFinisher::Get().IndirectsInFlight()) && DrawPipeline::Queue0().Busy()) {
            DrawFinisher::NoteFrameKept(DrawFinisher::Get().FlipsQueued() != DrawFinisher::Get().FlipsCommitted() ? FrameKept::WorkerBehindFlip : FrameKept::WorkerBehindIndirect);
            noteKept();
            DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, opcode);
        }
        if (pipelined && !predicated && (opcode == 0x3c || opcode == 0x93) && header != RenderingWaitPacketHeader && orderedWaits && DrawPipeline::Queue0().Busy()) {
            const auto awaitedBytes = Pm4::WaitAwaitedBytes(packet);
            const std::uint64_t awaited = (packet[2] & (awaitedBytes == 8 ? ~7u : ~3u)) | (static_cast<std::uint64_t>(packet[3]) << 32u);
            const auto value = DrawPipeline::Queue0().PendingLabel(awaited, awaitedBytes);
            if (value.has_value() && Pm4::WaitComparesValue(packet, *value)) {
                ++waitOutcomes().fromRecorderSameQueue;
                gateOpened() = true;
                noteLabelPass(packet, submission.queue);
                recent.Record(cursor);
                cursor = nextCursor;
                continue;
            }
            if (!DrawPipeline::Queue0().Overlaps(awaited, awaitedBytes) && Pm4::WaitSatisfiedUnchecked(packet) && !OwnStoreRejects(packet, awaited, awaitedBytes, submission.queue)) {
                bumpWaitEpoch(awaited, awaitedBytes, storeOrderingSince(packet, awaited, awaitedBytes, submission.received, submission.queue));
                ++waitOutcomes().atEntry;
                gateOpened() = true;
                noteLabelPass(packet, submission.queue);
                recent.Record(cursor);
                cursor = nextCursor;
                continue;
            }
        }
        timing.Mark("pipeline_prepare");
        if (pipelined && !ordered && !indirectQueued && DrawPipeline::Queue0().Busy() && !keepsDrawPipeline(packet, header, opcode) && !(DrawFinisher::Deep() && asksVerdict() && awaitVerdictWriters(packet, header, opcode)) && !(argumentWait && DrawPipeline::Queue0().AwaitExactWriters({&argumentRange, 1}, DrawPipeline::DrainReason::Indirect, 0x16, DeepSite::IndirectArguments, FrameSite::ArgumentWait, FrameCheck::DispatchArguments))) {
            const bool ordersFrame = header == FlipPacketHeader || header == RenderingWaitPacketHeader || submission.flips.contains(cursor) || opcode == 0x3c || opcode == 0x93 || opcode == 0x49 || opcode == 0x37;
            if (ordersFrame || !DrawFinisher::FlowSkips(DrawFinisher::FlowSite::PacketCommitter)) {
                if (opcode == 0x16 && !argumentWait && DrawFinisher::FrameFlow()) DrawFinisher::NoteFrameKept(FrameKept::IndirectGpu);
                noteKept();
                DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, header == FlipPacketHeader ? 0xffu : opcode);
            }
        }
        if (pipelined && predicated && DrawPipeline::Queue0().Busy() && DrawPipeline::Queue0().Overlaps(queue.predication.address, Pm4::PredicationBytes(queue.predication)) && !DrawFinisher::FlowSkips(DrawFinisher::FlowSite::Predication)) awaitPredicationWriters(queue.predication.address, Pm4::PredicationBytes(queue.predication), opcode);
        timing.Mark("pipeline_drain");
        std::shared_lock deviceUse(deviceReplacement, std::defer_lock);
        if (opcode != 0x3c && opcode != 0x93 && header != RenderingWaitPacketHeader && header != FlipPacketHeader) deviceUse.lock();

        timing.Mark("device_lock");
        GuestMemory::SetCurrentPacket(header == FlipPacketHeader ? 0xffffu : opcode, submission.queue);
        CaptureTrace::Log("packet submission=%llu queue=%x offset=%zu header=%08x words=%zu", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, header, packet.size());
        if (predicated) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            const bool passes = Pm4::PredicationPasses(queue);
            publishReadPoint();
            if (!passes) {
                cursor = opcode == 0x3f ? submission.conditionalEnds.at(cursor) : nextCursor;
                continue;
            }
            noteLabelIssue(packet, submission.queue);
        }
        if (opcode == 0x3f) {
            cursor = nextCursor;
            continue;
        }

        const auto flushStart = profilePackets ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        flushBetweenPackets(submission.queue, header, opcode == 0x49 || opcode == 0x37);
        timing.Mark("flush_between_packets");
        PacketTimer packetTimer{profilePackets, header == FlipPacketHeader ? 0xffffu : opcode, submission.queue, packetProfile, profilePackets ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};

        if (profilePackets) {
            packetStartedAt() = packetTimer.start;
            pendingDispatchPhases() = {};
            pendingDrawPhases() = {};
        }
        const auto finishDispatchPacket = [&](bool indirect) {
            if (!profilePackets) return;
            const auto now = std::chrono::steady_clock::now();
            auto& pending = pendingDispatchPhases();
            if (pending.phases) {
                pending.ms[PhaseEpilogue] = std::chrono::duration<double, std::milli>(now - pending.tailAt).count();
                addDriverPhases(submission.queue != 0 ? OtherQueues : indirect ? Queue0Indirect : Queue0Direct, pending.ms, pending.hit, pending.validated);
            }
            if (indirect) return;
            auto& row = packetProfile.dispatchOutcomes[static_cast<std::size_t>(pending.outcome)];
            ++row.first;
            row.second += std::chrono::duration<double, std::milli>(now - packetTimer.start).count();
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
                ms[DrawRowSkipped] = std::chrono::duration<double, std::milli>(now - packetTimer.start).count();
            }
            addDrawPhases(ms, drawn && pending.phases, pending.captures);
        };
        if (profilePackets) packetProfile.flushMs += std::chrono::duration<double, std::milli>(packetTimer.start - flushStart).count();

        bool wroteOnGpu = false, endOfPipeInterrupt = false, interruptDeferred = false, drawPacket = false, sampleDump = false;
        bool drains = false;
        if (orderedFlip) {
            wroteOnGpu = flipKind == PacketRules::FlipForm::EopLabel;
            if (flipKind == PacketRules::FlipForm::EopNothing) ++noOpLabels;
        } else {
            drains = preparePacketMemory(submission, queue, packet, header, opcode, releaseInterrupt(packet, opcode, submission.flips.contains(cursor)), wroteOnGpu, endOfPipeInterrupt, interruptDeferred, drawPacket, sampleDump);
        }
        traceLabel(packet, submission.queue);
        timing.Mark("prepare_memory");

        const bool waitPacket = opcode == 0x3c || opcode == 0x93 || header == RenderingWaitPacketHeader;
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
        if (header == RenderingWaitPacketHeader) {
            const auto waitStart = orderedWait && DrawFinisher::FrameCensus() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (orderedWait) DrawFinisher::Get().FlushOpenRun();
            const DrawFinisher::TailScope standing(FrameTail::RenderingWait);
            timed(&WorkerProfile::waitMs, [&] { submission.renderingWaits.at(cursor)->Wait(); });
            if (orderedWait) DrawFinisher::NoteFrameSite(FrameSite::RenderingWait, waitStart == std::chrono::steady_clock::time_point{} ? 0 : static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStart).count()));
        } else if (orderedFlip) {
            if (deviceUse.owns_lock()) deviceUse.unlock();
            CheckFailure();
            ++flipsCounted;
            if (Graphics::AbConfigured()) abAtFlip(abSeconds);
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
        } else if (submission.flips.contains(cursor)) {
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

                if (!recordLabelsForPacket(localDevice.get(), submission.queue) && localDevice != nullptr) localDevice->SubmitRecorded(submission.queue == 0, Graphics::Recorder::SubmitReason::Flip);
                if (localDevice != nullptr) localDevice->FlipBatches(batchesAtFlip, unsignaledAtFlip);
            }
            ++flipsCounted;
            Graphics::NoteHostImportFrame();
            if (batchesAtFlip != 0) flipSerial = batchesAtFlip;
            flipBatchesUnsignaled += unsignaledAtFlip;
            if (Graphics::AbConfigured()) abAtFlip();

            timing.Mark("flip_flush");
            timing.FinishPacket(submission.serial, cursor, 0xffffu, count);
            const bool queueTiming = APS5_ENABLE_TIMING_LOG && submission.queue == 0;
            auto frame = queueTiming ? std::move(pendingFrameTiming) : std::make_shared<FrameTiming>(++frameSerial);
            require(frame != nullptr, "flip has no frame timing");
            const auto now = FrameTiming::Clock::now();
            if (!queueTiming) frame->IncludeSubmission(submission.serial, now, now, now, true);
            frame->SetFlip(submission.serial, cursor, queueTiming ? submission.receivedAt : now, now);
            if (queueTiming) frame->CollectBackground();
            frame->NoteFlipBatches(batchesAtFlip, unsignaledAtFlip);
            CaptureTrace::Log("flip frame=%llu submission=%llu offset=%zu batch=%llu unsignaled=%llu", static_cast<unsigned long long>(frameSerial.load()), static_cast<unsigned long long>(submission.serial), cursor, static_cast<unsigned long long>(batchesAtFlip), static_cast<unsigned long long>(unsignaledAtFlip));
            struct FlipReadying {
                Driver& driver;
                std::uint32_t handle;
                FlipReadying(Driver& owner, std::uint32_t flipHandle) : driver(owner), handle(flipHandle) { driver.beginFlipReady(handle); }
                FlipReadying(const FlipReadying&) = delete;
                FlipReadying& operator=(const FlipReadying&) = delete;
                ~FlipReadying() { driver.endFlipReady(handle); }
            } readying(*this, submission.flipHandles.at(cursor));
            flipHold.Release();
            submission.flips.at(cursor)->GpuReady(frame);
            if (pipelined) DrawFinisher::Get().LatchFrameFlow();
        } else if (opcode == 0x15 || opcode == 0x16) {
            packetCounters.CountDispatch();
            landPendingWaits(submission.queue);
            currentPacketOffset() = cursor;
            if (gateOpened()) {
                gateOpened() = false;
                if (!pipelineDispatch() || submission.queue != 0) timed(&WorkerProfile::dispatchMs, [&] { resolveGroupAhead(submission, queue, cursor); });
            }
            const bool ringOverlap = submission.queue != 0 && Graphics::Recorder::DispatchOverlapEnabled();
            if (ringOverlap) Graphics::Recorder::SetDispatchOrder(submission.queue + 1u, ringOrder.afterDispatch);
            struct ClearOrder {
                bool active;
                ~ClearOrder() {
                    if (active) Graphics::Recorder::SetDispatchOrder(0, false);
                }
            } clearOrder{ringOverlap};
            if (ringOverlap) ringOrder.afterDispatch = true;
            if (opcode == 0x15) timed(&WorkerProfile::dispatchMs, [&] { dispatch(queue, packet, submission); });
            else timed(&WorkerProfile::dispatchMs, [&] { dispatchIndirect(queue, packet, submission, indirectQueued); });
            Pm4::NoteDispatchRegisters(packet, queue);
            Graphics::Recorder::CountRecordedWork();
            finishDispatchPacket(opcode == 0x16);
        } else if (opcode == 0x3c || opcode == 0x93) {
            static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
            const auto waitStart = traceGpu ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            timed(&WorkerProfile::waitMs, [&] { waitMemory(packet, submission.queue, recent, submission.received, submission.heldAtSubmit.contains(cursor)); });
            noteLabelPass(packet, submission.queue);
            gateOpened() = true;
            if (traceGpu && std::chrono::steady_clock::now() - waitStart > std::chrono::milliseconds(200)) {

                for (std::size_t next = cursor + count, shown = 0; next < submission.commands.size() && shown < 48; ++shown) {
                    const auto nextHeader = submission.commands[next];
                    const auto nextCount = Pm4::PacketWords(nextHeader);
                    const auto nextPacket = std::span(submission.commands).subspan(next, nextCount);
                    std::fprintf(stderr, "[gpu]   then %s", Pm4::Name(nextHeader).c_str());
                    for (std::size_t i = 1; i < nextPacket.size() && i < 7; ++i) std::fprintf(stderr, " %08x", nextPacket[i]);
                    std::fprintf(stderr, "\n");
                    next += nextCount;
                }
            }
        } else if (drawPacket) {
            landPendingWaits(submission.queue);
            packetCounters.CountDraw();
            drawnSinceSeal() = true;
            bool drawn = false;
            const auto drawStart = submission.queue == 0 && ThreadReport::Enabled() ? ThreadReport::Clock::now() : ThreadReport::Clock::time_point{};
            struct DrawPacketReport {
                ThreadReport::Clock::time_point start;
                ~DrawPacketReport() {
                    if (start == ThreadReport::Clock::time_point{}) return;
                    ThreadReport::Add(ThreadReport::Count::WorkerDrawPackets, 1);
                    ThreadReport::Add(ThreadReport::Count::WorkerDrawPacketNs, ThreadReport::Nanoseconds(start, ThreadReport::Clock::now()));
                }
            } drawPacketReport{drawStart};
            timed(&WorkerProfile::drawMs, [&] {
                static const bool traceDraws = std::getenv("APS5_TRACE_DRAWS") != nullptr;
                static const bool profileDraws = std::getenv("APS5_PROFILE_DRAW") != nullptr;
                const auto color = (static_cast<std::uint64_t>(readRegister(queue.context, 0x390)) << 40u) | (static_cast<std::uint64_t>(readRegister(queue.context, 0x318)) << 8u);
                const auto started = profileDraws ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const auto countSkip = [&](Graphics::DrawSkip kind) {
                    if (profileDraws) Graphics::CountDrawSkip(kind, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count());
                };

                try {
                    std::string rejected;
                    const auto verdict = draw(queue, packet, submission, cursor, rejected);
                    drawn = verdict == DrawVerdict::Drawn;
                    CaptureTrace::Log("draw submission=%llu queue=%x offset=%zu target=%llx mask=%x verdict=%d reason=%.256s", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, static_cast<unsigned long long>(color), readRegister(queue.context, 0x8e), static_cast<int>(verdict), rejected.c_str());
                    if (verdict == DrawVerdict::Rejected) {
                        countSkip(Graphics::DrawSkip::Prechecked);
                        throw std::runtime_error(rejected);
                    } else if (verdict == DrawVerdict::Nothing) {
                        countSkip(Graphics::DrawSkip::Nothing);
                    } else if (traceDraws) {
                        std::fprintf(stderr, "[draw] target 0x%llx mask 0x%x ok\n",static_cast<unsigned long long>(color), readRegister(queue.context, 0x8e));
                    }
                } catch (const DrawCompilePending&) {
                    countSkip(Graphics::DrawSkip::Thrown);
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
        } else if (opcode == 0x22) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            const auto condition = Pm4::ReadCondition(packet);
            if (condition == 0) nextCursor = submission.conditionalEnds.at(cursor);
            if (traceGpu) std::fprintf(stderr, "[gpu] %.1f queue 0x%x COND_EXEC at DWORD %zu reads 0x%x at 0x%llx: %s %zu dwords\n", TraceMs(), submission.queue, cursor, condition, static_cast<unsigned long long>(packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u)), condition == 0 ? "skips" : "executes", submission.conditionalEnds.at(cursor) - cursor - count);
        } else if (opcode == 0x91) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            Pm4::Execute(packet, queue);
        } else if (sampleDump && !wroteOnGpu) {
            dumpSampleCounters(Pm4::SampleDumpAddress(packet));
        } else if (submission.queue == 0 && sealsWriteOnly(packet, opcode)) {
            if (std::exchange(drawnSinceSeal(), false)) {
                const auto seal = [this] {
                    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
                    std::lock_guard gpuLock(GuestMemory::GpuMutex());
                    if (const auto localDevice = device.Load()) localDevice->SealPassWriteOnly();
                };
                const bool streamSeal = pipelined && switchSet(Graphics::RuntimeSwitch::XpStreamSegments);
                if (streamSeal && DrawFinisher::Overlap() && DrawFinisher::Busy()) {
                    DrawFinisher::Pending held;
                    held.unknown = false; held.origin = 7;
                    DrawFinisher::Get().Enqueue([seal] { DrawPipeline::Queue0().EnqueueQuiet(seal); }, std::move(held));
                    noteXpStream(true);
                } else if (streamSeal && DrawPipeline::Queue0().Busy()) {
                    DrawPipeline::Queue0().EnqueueQuiet(seal);
                    noteXpStream(true);
                } else if (pipelined && DrawPipeline::Queue0().Busy()) {
                    DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Seal, seal, {});
                } else {
                    seal();
                }
            }
        } else if (opcode != 0x42 && opcode != 0x46 && opcode != 0x58) {
            if (!wroteOnGpu) {
                if (Pm4::IndirectRegisterOpcode(opcode)) {
                    const auto list = RegisterListAt(submission, cursor, listHint);
                    const auto listed = list.pairs;
                    if (list.valid) {
                        Pm4::ExecuteIndirectRegisters(packet, listed, queue);
                    } else {
                        landPendingWaits(submission.queue);
                        recordQueuedLabelsBeforeRead(submission.queue);
                        const auto current = Pm4::ReadIndirectRegisters(packet);
                        static const bool skipInvalid = std::getenv("APS5_SKIP_INVALID_INDIRECT_REGISTERS") != nullptr;
                        static std::atomic<bool> reported{false};
                        if (!reported.exchange(true)) {
                            std::uint32_t invalid = 0;
                            for (std::size_t i = 0; i < listed.size(); i += 2) {
                                if (!Pm4::IndirectRegistersValid(std::span(&listed[i], 1))) {
                                    invalid = listed[i];
                                    break;
                                }
                            }
                            std::fprintf(stderr, "[gpu] queue 0x%x %s at DWORD %zu: register list entry with offset dword 0x%08x read again where the packet executes%s\n", submission.queue, Pm4::Name(packet[0]).c_str(), cursor, invalid, skipInvalid ? "; invalid entries are skipped (APS5_SKIP_INVALID_INDIRECT_REGISTERS)" : "");
                        }
                        Pm4::ExecuteIndirectRegisters(packet, current, queue, skipInvalid);
                    }
                } else {
                    Pm4::Execute(packet, queue);
                }
                if (opcode == 0x49 || opcode == 0x37) {
                    if (const auto label = Pm4::DecodeLabelWrite(packet)) noteLabelStore(label->address, label->Bytes(), ++eventSerial, submission.queue);
                }
            }
            if (endOfPipeInterrupt && !interruptDeferred) deliverGuestEopInterrupt(submission.queue, packet[7] & 0x0fffffffu);
        }
        if (drawPacket || (sampleDump && wroteOnGpu)) Graphics::Recorder::CountRecordedWork();
        timing.Mark(header == RenderingWaitPacketHeader ? "rendering_wait" : opcode == 0x3c || opcode == 0x93 ? "memory_wait" : opcode == 0x15 || opcode == 0x16 ? "dispatch" : drawPacket ? "draw" : "execute");
        if (submission.queue == 0) timing.FinishPacket(submission.serial, cursor, opcode, count);
        else timing.Finish();
        cursor = nextCursor;
    }

    endSegment<false>(submission, flipHold, pipelined, front);
}

}
