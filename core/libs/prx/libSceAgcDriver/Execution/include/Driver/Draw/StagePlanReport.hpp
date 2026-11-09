#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_STAGEPLANREPORT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_STAGEPLANREPORT_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRecordReport.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/LeanFront.hpp"
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace AgcDriver::DriverDetail {

namespace StagePlans {

enum class Counter : std::uint8_t {
    Draws,
    Hits,
    HitsOnTodaysPath,
    InputRefetches,
    HitsRelocated,
    HitsBaseValue,
    HitsBuffersPatched,
    MissNoPlans,
    MissRefused,
    MissGenerationMoved,
    MissUserWordsUnread,
    MissNewLayout,
    MissRunsUnreadable,
    MissNewUserWords,
    MissNewBits,
    MissNewRunWords,
    MissNewCombination,
    MissImageViews,
    MissContentDiffers,
    MissInaccessible,
    MissQueuedLabel,
    MissFlushingImage,
    MissForgetMoved,
    MissPublishMoved,
    MissPendingMoved,
    MissDeviceImages,
    MissBufferFields,
    MissVertexLayout,
    MissMemoryChanged,
    MissFetchError,
    OffTodaysDraw,
    OffIndirect,
    OffLeanVerify,
    OffInline,
    Builds,
    BuildsNotQueued,
    BuildsGenerationMoved,
    BuildsDeferred,
    RefusedShape,
    RefusedIndirect,
    RefusedStages,
    RefusedGeometryBack,
    RefusedCapture,
    RefusedDirect,
    RefusedProgram,
    RefusedResult,
    RefusedPatched,
    RefusedLayout,
    BuildsDropped,
    DroppedUnanswered,
    DroppedStale,
    DroppedShape,
    DroppedRefused,
    DroppedOverCap,
    StagePlansAdded,
    StagePlansShared,
    StagePlansReplaced,
    StagePlansEvicted,
    LayoutsAdded,
    DrawPlansAdded,
    DrawPlansKnown,
    DrawPlansEvicted,
    SetsSwept,
    SetsCleared,
    ModeChanges,
    Demotions,
    ProofsSkipped,
    ProofsRequeried,
    ProofsVerified,
    ProofsDiffered,
    ProofFirst,
    ProofFailed,
    ProofNotReusable,
    ProofStorageImages,
    ProofDepthCopies,
    ProofOwnWrites,
    ProofNotClean,
    ProofViews,
    ProofSerials,
    ProofForgotten,
    ProofRewritten,
    ProofPendingWrites,
    ProofUnsettled,
    ProofsFull,
    TargetsHeld,
    TargetsFirst,
    TargetsPass,
    TargetsEpoch,
    TargetsProof,
    TargetsPending,
    TargetsDepth,
    TargetsChanged,
    MovedChecked,
    MovedTrusted,
    Count,
};

std::uint64_t VerifyEvery();

inline bool Due(LeanFront::Latch& latch) {
    if (latch.stagePlansVerifyEvery == 0) return false;
    return ++latch.stagePlanOrdinal % latch.stagePlansVerifyEvery == 0;
}

void NoteChecked(DrawRecords::Item item);
void NoteDifference(DrawRecords::Field field, std::string_view detail);
void NoteAllowed(DrawRecords::Allowed kind);
void Add(Counter counter, std::uint64_t count = 1);
void AddOnWorker(Counter counter, std::uint64_t count = 1);
void AddOnCommitter(Counter counter, std::uint64_t count = 1);
void NoteStore(std::size_t shapes, std::size_t stagePlans, std::size_t drawPlans, std::size_t bytes);
void Report(bool force);

const char* CounterName(Counter counter);
std::uint64_t Checked(DrawRecords::Item item);
std::uint64_t Differences();
std::uint64_t Differences(DrawRecords::Field field);
std::uint64_t AllowedCount(DrawRecords::Allowed kind);
std::uint64_t Total(Counter counter);

}

}

#endif
