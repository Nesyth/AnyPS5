#include "prx/libSceAgcDriver/Graphics/include/HeapReplay.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadReport.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace AgcDriver::Graphics {

namespace {

bool flag(const char* text) {
    return text != nullptr && text[0] != '\0' && text[0] != '0';
}

template<typename TEnum>
using Counters = std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(TEnum::Count)>;

Counters<ReplayEvent> events{};
Counters<ReplayEvent> eventTotals{};
Counters<ReplayReason> reasons{};
Counters<ReplayStageChange> stageChanges{};
Counters<ReplayField> mismatches{};
Counters<ReplayField> mismatchTotals{};
std::atomic<std::uint64_t> srtChanges{0}, programWords{0}, drawWords{0};
std::array<std::atomic<std::uint64_t>, 5> runChanges{};
std::atomic<std::uint64_t> ringSubmits{0}, keptChunks{0}, keptChunksMax{0};
std::array<std::atomic<std::uint64_t>, 4> generationMoves{};
std::atomic<std::uint64_t> checks{0}, checkTotal{0}, dumped{0};
std::atomic<std::uint64_t> notes{0};
std::atomic<std::int64_t> reportedAt{0};
std::mutex reportMutex;

bool reporting() {
    static const bool on = ThreadReport::Enabled() || ReplayCensusEnabled() || ReplayVerifyLevel() != 0;
    return on;
}

void tick() {
    if (!reporting()) return;
    if ((notes.fetch_add(1, std::memory_order_relaxed) & 4095u) == 4095u) ReportHeapReplay(false);
}

std::uint64_t take(std::atomic<std::uint64_t>& counter) {
    return counter.exchange(0, std::memory_order_relaxed);
}

unsigned long long taken(std::atomic<std::uint64_t>& counter) {
    return static_cast<unsigned long long>(take(counter));
}

}

bool ReplayCensusEnabled() {
    static const bool enabled = flag(std::getenv("APS5_REPLAY_CENSUS"));
    return enabled;
}

int ReplayVerifyLevel() {
    static const int level = [] {
        const char* text = std::getenv("APS5_VERIFY_REPLAY_HEAP_STAGE");
        if (text == nullptr) return 0;
        const auto value = std::strtol(text, nullptr, 10);
        return value <= 0 ? 0 : value == 1 ? 1 : 2;
    }();
    return level;
}

bool ReplayVerifyStrict() {
    static const bool strict = flag(std::getenv("APS5_VERIFY_REPLAY_STRICT"));
    return strict;
}

void NoteReplay(ReplayEvent event) {
    const auto index = static_cast<std::size_t>(event);
    events[index].fetch_add(1, std::memory_order_relaxed);
    eventTotals[index].fetch_add(1, std::memory_order_relaxed);
    if (event == ReplayEvent::Draws) tick();
}

std::uint64_t ReplayEvents(ReplayEvent event) {
    return eventTotals[static_cast<std::size_t>(event)].load(std::memory_order_relaxed);
}

const char* ReplayReasonName(ReplayReason reason) {
    static constexpr std::array<const char*, static_cast<std::size_t>(ReplayReason::Count)> names{
        "eligible", "not recorded", "draw transitions", "stencil replay", "indirect", "mesh", "tessellation", "quad list", "strip patches", "restart compaction",
        "multisampled, proxied, staged or partial-write target", "input capture", "draw profile", "not on the committer", "address-based", "lease", "not reusable",
        "space reuse", "copied writes", "not a ring stage"};
    return names[static_cast<std::size_t>(reason)];
}

void NoteReplayReason(ReplayReason reason) {
    reasons[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    tick();
}

void NoteReplayStageChange(ReplayStageChange change, bool srtChanged, bool wordsFromProgram) {
    stageChanges[static_cast<std::size_t>(change)].fetch_add(1, std::memory_order_relaxed);
    if (srtChanged) srtChanges.fetch_add(1, std::memory_order_relaxed);
    (wordsFromProgram ? programWords : drawWords).fetch_add(1, std::memory_order_relaxed);
}

void NoteReplayRunChange(std::size_t changed, bool resized) {
    const std::size_t bucket = resized ? 4u : changed == 0 ? 0u : changed == 1 ? 1u : changed <= 4 ? 2u : 3u;
    runChanges[bucket].fetch_add(1, std::memory_order_relaxed);
}

void NoteReplayRingSubmit(std::size_t kept) {
    ringSubmits.fetch_add(1, std::memory_order_relaxed);
    keptChunks.fetch_add(kept, std::memory_order_relaxed);
    auto peak = keptChunksMax.load(std::memory_order_relaxed);
    while (kept > peak && !keptChunksMax.compare_exchange_weak(peak, kept, std::memory_order_relaxed)) {}
}

void NoteReplayGenerations(bool allocations, bool imports, bool slots) {
    generationMoves[0].fetch_add(1, std::memory_order_relaxed);
    if (allocations) generationMoves[1].fetch_add(1, std::memory_order_relaxed);
    if (imports) generationMoves[2].fetch_add(1, std::memory_order_relaxed);
    if (slots) generationMoves[3].fetch_add(1, std::memory_order_relaxed);
}

const char* ReplayFieldName(ReplayField field) {
    static constexpr std::array<const char*, static_cast<std::size_t>(ReplayField::Count)> names{"slot key", "slots", "stage data", "inputs", "srt", "root offsets", "spans", "ring words", "descriptor runs", "table map"};
    return names[static_cast<std::size_t>(field)];
}

void NoteReplayCheck() {
    checks.fetch_add(1, std::memory_order_relaxed);
    checkTotal.fetch_add(1, std::memory_order_relaxed);
}

bool NoteReplayMismatch(ReplayField field, const std::string& detail) {
    const auto index = static_cast<std::size_t>(field);
    mismatches[index].fetch_add(1, std::memory_order_relaxed);
    mismatchTotals[index].fetch_add(1, std::memory_order_relaxed);
    if (dumped.fetch_add(1, std::memory_order_relaxed) < 8) std::fprintf(stderr, "[replay-verify] %s differs: %s\n", ReplayFieldName(field), detail.c_str());
    return ReplayVerifyStrict();
}

std::uint64_t ReplayChecks() {
    return checkTotal.load(std::memory_order_relaxed);
}

std::uint64_t ReplayMismatches(ReplayField field) {
    return mismatchTotals[static_cast<std::size_t>(field)].load(std::memory_order_relaxed);
}

void ReportHeapReplay(bool force) {
    if (!reporting()) return;
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard lock(reportMutex);
    const auto last = reportedAt.load(std::memory_order_relaxed);
    if (last == 0) {
        reportedAt.store(now, std::memory_order_relaxed);
        if (!force) return;
    }
    if (!force && now - last < 10000) return;
    reportedAt.store(now, std::memory_order_relaxed);
    const double seconds = last == 0 ? 0.0 : static_cast<double>(now - last) / 1000.0;
    const auto event = [](ReplayEvent which) { return taken(events[static_cast<std::size_t>(which)]); };
    const auto draws = event(ReplayEvent::Draws);
    if (draws != 0 || events[static_cast<std::size_t>(ReplayEvent::NonReplayWrites)].load(std::memory_order_relaxed) != 0) {
        const auto stages = event(ReplayEvent::Stages);
        const auto created = event(ReplayEvent::NewRecord);
        const auto writes = event(ReplayEvent::NonReplayWrites);
        const auto interrupted = event(ReplayEvent::Interrupted);
        const auto identity = event(ReplayEvent::HeapOrSwitches);
        const auto program = event(ReplayEvent::NewProgram);
        const auto slots = event(ReplayEvent::SlotsMoved);
        const auto shape = event(ReplayEvent::ShapeChanged);
        const auto masked = event(ReplayEvent::MaskedChange);
        const auto refresh = event(ReplayEvent::SubmitDuringRefresh);
        const auto pass = event(ReplayEvent::SubmitBeforePass);
        const auto dataSame = event(ReplayEvent::DataSame);
        const auto dataProven = event(ReplayEvent::DataProven);
        const auto dataHeld = event(ReplayEvent::DataHeld);
        const auto dataRewritten = event(ReplayEvent::DataRewritten);
        const auto dataPatched = event(ReplayEvent::DataPatched);
        const auto srtSame = event(ReplayEvent::SrtSame);
        const auto srtProven = event(ReplayEvent::SrtProven);
        const auto srtHeld = event(ReplayEvent::SrtHeld);
        const auto srtRewritten = event(ReplayEvent::SrtRewritten);
        const auto srtChanged = event(ReplayEvent::SrtChanged);
        const auto built = event(ReplayEvent::RunsBuilt);
        const auto skipped = event(ReplayEvent::RunsSkipped);
        AgcDriver::ProfilePrint_nid_no_patch(
            "[replay-heap-stage] %.1f s: %llu draws, %llu stages replayed; today's refresh: new record %llu, after another writer %llu, after a failed replay %llu, heap or switches changed %llu, new program %llu, slots moved %llu, words or shape changed %llu, patched dword changed %llu; submits: during the refresh %llu, before the pass %llu; stage data same %llu (span proven %llu, held %llu, rewritten %llu), patched %llu; srt same %llu (proven %llu, held %llu, rewritten %llu), changed %llu; template runs built %llu, skipped %llu\n",
            seconds, draws, stages, created, writes, interrupted, identity, program, slots, shape, masked, refresh, pass, dataSame, dataProven, dataHeld, dataRewritten, dataPatched, srtSame, srtProven, srtHeld, srtRewritten, srtChanged, built, skipped);
    }
    if (ReplayCensusEnabled()) {
        std::uint64_t hits = 0;
        std::array<unsigned long long, static_cast<std::size_t>(ReplayReason::Count)> reasonCounts{};
        for (std::size_t index = 0; index < reasonCounts.size(); ++index) {
            reasonCounts[index] = taken(reasons[index]);
            hits += reasonCounts[index];
        }
        std::string refusals;
        for (std::size_t index = 1; index < reasonCounts.size(); ++index) {
            if (reasonCounts[index] == 0) continue;
            char item[96];
            std::snprintf(item, sizeof(item), "%s %s %llu", refusals.empty() ? "" : ",", ReplayReasonName(static_cast<ReplayReason>(index)), reasonCounts[index]);
            refusals += item;
        }
        std::array<unsigned long long, static_cast<std::size_t>(ReplayStageChange::Count)> change{};
        for (std::size_t index = 0; index < change.size(); ++index) change[index] = taken(stageChanges[index]);
        std::array<unsigned long long, 5> runs{};
        for (std::size_t index = 0; index < runs.size(); ++index) runs[index] = taken(runChanges[index]);
        std::array<unsigned long long, 4> moves{};
        for (std::size_t index = 0; index < moves.size(); ++index) moves[index] = taken(generationMoves[index]);
        const auto submits = taken(ringSubmits);
        const auto chunks = taken(keptChunks);
        const auto peak = taken(keptChunksMax);
        AgcDriver::ProfilePrint_nid_no_patch(
            "[replay-census] %.1f s: heap template hits %llu, eligible %llu, refused by%s; stage refreshes: same %llu, user data only 1: %llu, 2-4: %llu, 5-16: %llu, 17+: %llu, words changed %llu, patched dwords changed %llu, slots or size changed %llu; srt changed %llu; words from the program %llu, per draw %llu; ring runs: unchanged %llu, 1 entry %llu, 2-4 %llu, 5+ %llu, resized %llu; between hits of a template (%llu): allocations generation moved %llu, import set moved %llu, slot serial moved %llu; submits %llu with %llu other chunks open (most %llu)\n",
            seconds, static_cast<unsigned long long>(hits), reasonCounts[0], refusals.empty() ? " nothing" : refusals.c_str(), change[0], change[1], change[2], change[3], change[4], change[5], change[6], change[7], taken(srtChanges), taken(programWords), taken(drawWords), runs[0], runs[1], runs[2], runs[3], runs[4], moves[0], moves[1], moves[2], moves[3], submits, chunks, peak);
    }
    if (ReplayVerifyLevel() != 0) {
        std::string fields;
        unsigned long long total = 0;
        for (std::size_t index = 0; index < mismatches.size(); ++index) {
            const auto count = taken(mismatches[index]);
            total += count;
            if (count == 0) continue;
            char item[64];
            std::snprintf(item, sizeof(item), "%s %s %llu", fields.empty() ? "" : ",", ReplayFieldName(static_cast<ReplayField>(index)), count);
            fields += item;
        }
        AgcDriver::ProfilePrint_nid_no_patch("[replay-verify] %.1f s: level %d, %llu stage checks, %llu mismatches%s%s\n", seconds, ReplayVerifyLevel(), taken(checks), total, fields.empty() ? "" : ":", fields.c_str());
    }
}

}
