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
    SlotsMoved,
    ShapeChanged,
    MaskedChange,
    SubmitDuringRefresh,
    SubmitBeforePass,
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
void NoteReplayStageChange(ReplayStageChange change, bool srtChanged, bool wordsFromProgram);
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

}

#endif
