#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_HEAPREPLAY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_HEAPREPLAY_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace AgcDriver::Graphics {

bool ReplayCensusEnabled();

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

void ReportHeapReplay(bool force);

}

#endif
