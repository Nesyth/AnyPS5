#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_HEAPREPLAY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_HEAPREPLAY_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace AgcDriver::Graphics {

bool ReplayCensusEnabled();
int ReplayVerifyLevel();
bool ReplayVerifyStrict();

enum class ReplayEvent : std::size_t {
    Draws,
    Stages,
    NewRecord,
    NonReplayWrites,
    Interrupted,
    HeapOrSwitches,
    NewProgram,
    NotReplayable,
    SlotsMoved,
    ShapeChanged,
    MaskedChange,
    SubmitDuringRefresh,
    SubmitBeforePass,
    SubmitDuringFallback,
    OverlayMissing,
    DataSame,
    DataProven,
    DataHeld,
    DataRewritten,
    DataPatched,
    SrtSame,
    SrtProven,
    SrtHeld,
    SrtRewritten,
    SrtChanged,
    RunsBuilt,
    RunsSkipped,
    Count
};
void NoteReplay(ReplayEvent event);
std::uint64_t ReplayEvents(ReplayEvent event);

enum class ReplayReason : std::size_t {
    Eligible,
    NotRecordable,
    Transitions,
    Stencil,
    Indirect,
    Mesh,
    Tessellation,
    QuadList,
    StripPatches,
    Restart,
    Targets,
    Capture,
    Profile,
    Worker,
    AddressBased,
    Lease,
    NotReusable,
    SpaceReuse,
    CopiedWrites,
    NotRingStage,
    Count
};
const char* ReplayReasonName(ReplayReason reason);
void NoteReplayReason(ReplayReason reason);

enum class ReplayStageChange : std::size_t { Same, One, UpToFour, UpToSixteen, More, WordsChanged, Masked, Reshaped, Count };
void NoteReplayStageChange(ReplayStageChange change, bool srtChanged, bool wordsFromProgram, bool wordsUnderUserData);
void NoteReplayOpenSpans(bool dataUnchanged, bool dataOpen, bool srtUnchanged, bool srtOpen);
void NoteReplaySnapshots(std::size_t moved, std::size_t unmoved, std::size_t bytes, std::size_t memoHits, std::size_t uploads, std::size_t large);
void NoteReplayPass(bool continued, bool sameTemplate, bool sameShape);
void NoteReplayRunChange(std::size_t changed, bool resized);
void NoteReplayRingSubmit(std::size_t keptChunks);
void NoteReplayGenerations(bool allocations, bool imports, bool slots);

enum class ReplayField : std::size_t { Key, Slots, Block, Inputs, Srt, Offsets, Spans, Ring, Runs, Map, Count };
const char* ReplayFieldName(ReplayField field);
void NoteReplayCheck();
bool NoteReplayMismatch(ReplayField field, const std::string& detail);
std::uint64_t ReplayChecks();
std::uint64_t ReplayMismatches(ReplayField field);

void ReportHeapReplay(bool force);

void SetHeapReplayRefreshHookForTest(void (*hook)());
void RunHeapReplayRefreshHookForTest();

}

#endif
