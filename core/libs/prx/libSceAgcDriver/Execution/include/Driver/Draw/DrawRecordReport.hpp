#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWRECORDREPORT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWRECORDREPORT_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanFront.hpp"
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace AgcDriver::DriverDetail {

namespace DrawRecords {

enum class Item : std::uint8_t { Lookup, Fetch, Commit, Count };

enum class Field : std::uint8_t {
    Keys,
    Shapes,
    Devices,
    Generations,
    ContentKeys,
    VertexLayouts,
    WriteTemplates,
    FragmentOutputs,
    Variants,
    PushConstants,
    Bindings,
    VertexAttributes,
    ImageViews,
    Parameters,
    ReadRanges,
    Snapshots,
    Templates,
    Pipelines,
    Framebuffers,
    TargetViews,
    Targets,
    PassKeys,
    MovedBuffers,
    CommitPushConstants,
    VertexInputs,
    ShaderStages,
    CommitFragmentOutputs,
    CommitBindings,
    PassAccess,
    Keeps,
    TargetNotes,
    Count,
};

enum class Allowed : std::uint8_t { FragmentOutputsOneSide, TodayNotQueued, CommitRecordMissed, CommitTodayUnrecorded, CommitTemplateRebuilt, Count };

enum class Counter : std::uint8_t {
    Draws,
    Hits,
    InputRefetches,
    HitsOnTodaysPath,
    MissNewIdentity,
    MissSecondSighting,
    MissBuildPending,
    MissShapeReplaced,
    MissRefused,
    MissContentDiffers,
    MissGenerationMoved,
    MissNotQueued,
    MissKeyless,
    MissNoFetchPlan,
    MissUserWord,
    MissFetchGate,
    MissImageViews,
    MissVertexLayout,
    MissBufferFields,
    MissMemoryChanged,
    MissFetchError,
    OffTodaysDraw,
    OffIndirect,
    OffLeanVerify,
    OffInline,
    RecordsBuilt,
    PlansBuilt,
    PlansRefused,
    BuildsRefused,
    RecordsInstalled,
    BuildsDropped,
    IdentitiesEvicted,
    RecordsCleared,
    RecordModeChanges,
    Demotions,
    CommitHits,
    CommitPushed,
    CommitUnpushed,
    CommitPassHeld,
    CommitKeptOnce,
    CommitNotesSkipped,
    CommitMovedTrusted,
    CommitBuilt,
    CommitVerified,
    CommitFirst,
    CommitNotRecordable,
    CommitTargets,
    CommitTemplate,
    CommitObjects,
    CommitProof,
    CommitMoved,
    CommitUnbuilt,
    CommitDemoted,
    Count,
};

constexpr std::size_t ItemCount = static_cast<std::size_t>(Item::Count);

std::uint64_t VerifyEvery();
void CorruptAnswers(bool corrupt);
bool CorruptingAnswers();
void CorruptCommits(bool corrupt);
bool CorruptingCommits();

inline bool Due(LeanFront::Latch& latch) {
    if (latch.recordsVerifyEvery == 0) return false;
    return ++latch.recordOrdinal % latch.recordsVerifyEvery == 0;
}

void NoteChecked(Item item);
void NoteDifference(Field field, std::string_view detail);
void NoteAllowed(Allowed kind);
void Add(Counter counter, std::uint64_t count = 1);
void AddOnWorker(Counter counter, std::uint64_t count = 1);
void NoteStore(std::size_t identities, std::size_t records, std::size_t bytes);
void Report(bool force);

const char* ItemName(Item item);
const char* FieldName(Field field);
std::uint64_t Checked(Item item);
std::uint64_t Differences();
std::uint64_t Differences(Field field);
std::uint64_t AllowedCount(Allowed kind);
std::uint64_t Total(Counter counter);

}

}

#endif
