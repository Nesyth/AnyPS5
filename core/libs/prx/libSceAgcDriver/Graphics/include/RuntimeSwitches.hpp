#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RUNTIMESWITCHES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RUNTIMESWITCHES_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace AgcDriver::Graphics {

enum class RuntimeSwitch : std::size_t {
    NoBarrierBatch,
    NoTextureProofStamps,
    NoRangeProofs,
    NoLeanRuns,
    NoPassAccessMemo,
    NoRangedDispatchBreaks,
    NoStageDataSideCars,
    NoDataRecipes,
    NoHeapRecipes,
    NoIntervalCollectMemo,
    NoInputMemo,
    NoPassAttachmentMemo,
    NoGuardSpans,
    NoHtileFixupCarry,
    XpWalksFree,
    NoArmResident,
    PipelineDispatch,
    NoHeapSetSkip,
    LargeBatches,
    ReplayHeapStage,
    XpReplay,
    XpProofAsIs,
    XpStagesAsIs,
    RearmQuiet,
    CleanCurrency,
    ArmWide,
    ArmSpans,
    ReplayBind,
    XpUploadNoSync,
    XpNoEagerStore,
    AsyncSubmit,
    PreciseLookupDrain,
    SecondFinisher,
    ThirdFinisher,
    LocklessHazards,
    XpHalfDraws,
    XpHalfWorkerDraws,
    SpinHandoff,
    BesideSampleDumps,
    DeepQueues,
    SpeculativeLookup,
    HostLargeCache,
    PipelinedGpuIndirect,
    HandoffGpuIndirect,
    InputScratch,
    CarryValidation,
    SingleIndexCheck,
    OwnDrawSets,
    GuardRangeMemo,
    MovedLargeUploads,
    UnlockedSnapshotRecheck,
    KeepBytesOnce,
    MsaaPassMemo,
    GpuSingleKeys,
    LeanRemark,
    KeepFastPath,
    LazyPassReads,
    RecycleCommandBuffers,
    PushMemcpy,
    ArmFlipReset,
    ResidentFirstCollect,
    SortedHostRegions,
    HostPrefetch,
    SkipHtileKeyPresync,
    XpKeepDraws,
    XpStreamSegments,
    LeanFront,
    DrawRecords,
    StagePlans,
    DirectDraws,
    DirectCleanProofs,
    DirectPassTargets,
    CleanCollects,
    DirectBuffers,
    PipelineDispatchMisses,
    XpCommitFloor,
    MsaaRecords,
    XpInputsAsIs,
    XpStampsByEpoch,
    MsaaRecordHold,
    TemplateTouch,
    XpBindingsAsIs,
    NoPlanCleanProofs,
    DeferredEncode,
    NoOwnedDrawInputs,
    DirectInputs,
    DirectRetry,
    ItemsByTurn,
    XpNoCommits,
    FourthFinisher,
    FifthFinisher,
    LeanListBatch,
    LeanListDirect,
    DirectRepeat,
    CleanResidentViews,
    CleanPendingRanges,
    RefusedCollectMemo,
    WorkerCore,
    DirectRepeatSets,
    DirectRepeatStores,
    DirectRepeatShared,
    FinisherDeposit,
    CommitBindings,
    CommitBindingsStamps,
    CommitBindingsSets,
    CommitBindingsDynamic,
    CommitFragments,
    ExpressCommits,
    LabelOrderedStores,
    DeeperRing,
    DispatchBeside,
    FaultFragments,
    StageMemo,
    KeepFaultSlots,
    SiblingFinishers,
    FastSettle,
    XpSlowFinisher,
    XpSlowCommitter,
    XpSlowWorker,
    XpExpressFloor,
    LeanExpress,
    XpAppendCommits,
    PreciseSettle,
    AppendCommits,
    XpNoBarriers,
    DrawRuns,
    LeanPrepare,
    PooledHits,
    ExpressTargetUploads,
    AliasesAtPrepare,
    StageMemoAdmitFirst,
    XpNoOcclusion,
    LabelReadbackMemo,
    StageMemoNoIndex,
    PassBarriersMerged,
    ExpressPassStamps,
    CommitStamps,
    PassReadsOnce,
    ReapThrottle,
    OcclusionWindows,
    StageMemoAdmitThird,
    TurnWaitsByJob,
    StageMemoFront,
    ExpressKeepState,
    RunFollow,
    StageMemoPatchVertex,
    StageMemoPatchBuffers,
    StageMemoPatchData,
    Count,
};

class SwitchSnapshot {
public:
    bool Set(RuntimeSwitch id) const {
        const auto index = static_cast<std::size_t>(id);
        return ((bits[index / 64] >> (index % 64)) & 1u) != 0;
    }

    void Assign(RuntimeSwitch id, bool set) {
        const auto index = static_cast<std::size_t>(id);
        const auto bit = std::uint64_t{1} << (index % 64);
        bits[index / 64] = set ? bits[index / 64] | bit : bits[index / 64] & ~bit;
    }

    bool operator==(const SwitchSnapshot&) const = default;

private:
    std::array<std::uint64_t, (static_cast<std::size_t>(RuntimeSwitch::Count) + 63) / 64> bits{};
};

namespace RuntimeSwitches {

const char* Name(RuntimeSwitch id);
bool Set(RuntimeSwitch id);
void Assign(RuntimeSwitch id, bool set);
std::uint64_t Generation();
SwitchSnapshot Snapshot();
std::optional<RuntimeSwitch> Find(std::string_view name);

}

class AbSchedule {
public:
    struct Config {
        double phaseSeconds = 0;
        double settleSeconds = 2;
        std::uint64_t startPresent = 0;
        std::optional<double> startSeconds;
        std::vector<RuntimeSwitch> switches;
        bool firstB = false;
    };

    enum class Event { None, Begin, Settled };

    struct Step {
        Event event = Event::None;
        std::uint64_t phase = 0;
        bool b = false;
    };

    static Config Parse(std::string_view ab, std::string_view start, std::string_view settle);

    explicit AbSchedule(Config config);

    Step AtPresent(std::uint64_t present, double seconds);

    const Config& Settings() const { return config; }

private:
    Config config;
    bool started = false;
    bool settled = false;
    std::uint64_t phase = 0;
    double phaseStart = 0;
    std::optional<double> firstSeconds;
    double lastSeconds = 0;
};

bool AbConfigured();
std::vector<std::string> AbAtPresent(std::uint64_t present, double seconds);
std::vector<std::string> AbAtPresent(std::uint64_t present);

}

#endif
