#include <string>
#include <optional>
#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAW_HPP

#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuardSpans.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

struct VkDrawCounts {
    std::uint64_t direct = 0;
    std::uint64_t indexed = 0;
    std::uint64_t indirect = 0;
    std::uint64_t mesh = 0;
};
VkDrawCounts TakeVkDrawCounts();

struct HostImport;

struct DrawTraceContext {
    std::uint64_t token = 0;
    std::uint64_t submission = 0;
    std::uint32_t queue = 0;
    std::size_t offset = 0;
    std::uint64_t target = 0;
};

// `recipe`, when given, receives the DrawRecipe a recorded, cacheable, reusable, non-indirect draw
// built for its draw-cache entry (design_cpu_final M8; null otherwise, and always under
// APS5_NO_DRAW_RECIPE=1).
void Draw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots = {}, std::shared_ptr<const DrawRecipe>* recipe = nullptr, const DrawTraceContext* trace = nullptr, std::uint64_t shape = 0, const std::vector<std::uint32_t>* contentKey = nullptr, const std::uint32_t* fragmentOutputs = nullptr, bool recordRecipe = false);
std::vector<std::uint32_t> DrawResourceContentKey(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders);

struct PixelFunctionLdsLayouts {
    VkDescriptorSetLayout empty = VK_NULL_HANDLE;
    VkDescriptorSetLayout buffer = VK_NULL_HANDLE;
};

PixelFunctionLdsLayouts PixelFunctionLdsSetLayouts(const Context& context);
void PreparePixelFunctionLds(const Context& context, VkCommandBuffer commands, std::uint32_t subgroupSize);
void BindPixelFunctionLds(const Context& context, VkCommandBuffer commands, VkPipelineLayout layout);
void CheckPixelFunctionLds(const Context& context);
void ReleasePixelFunctionLds(VkDevice device);
bool UsesMeshWindow(std::span<const CompiledShader> shaders);
bool ValidationKey(const Context& context, std::span<const CompiledShader> shaders, const State& state, std::vector<std::uint64_t>& key);
VkDescriptorSetLayout MeshWindowSetLayout(const Context& context);
void PrepareMeshWindow(const Context& context);
void BindMeshWindow(const Context& context, VkCommandBuffer commands, VkPipelineLayout layout);
void ReleaseMeshWindow(VkDevice device);
std::optional<std::string> KnownValidationFailure(const Context& context, std::span<const CompiledShader> shaders, const State& state, std::optional<std::uint32_t>* fragmentOutputs = nullptr);
std::uint64_t DrawRenderPassKey(const Context& context, const State& state, std::span<const VkImageView> targetViews);

struct DrawInputCopy {
    std::shared_ptr<Buffer> buffer;
    bool reused = false;
    std::uint32_t derived = 0;
    std::uint32_t lowest = 0;
    std::size_t from = 0;
    std::uint64_t generation = 0;
    std::uint64_t registryGeneration = 0;
    Recorder::OwnedDrawInputProof owned;
};
DrawInputCopy CopyDrawInput(const Context& context, Recorder* recorder, std::uint64_t address, std::size_t bytes, std::size_t alignment, Recorder::SnapshotUse use, std::size_t from = 0, bool hostOnly = false);
bool BindsIndexCopy(const State& state);
DrawInputCopy CopyZeroPaddedDrawInput(const Context& context, std::uint64_t address, std::size_t bytes, std::size_t validBytes, std::size_t stride = 0);
void KeepDrawInput(Recorder* recorder, std::uint64_t address, const DrawInputCopy& copy, Recorder::SnapshotUse use, std::uint32_t derived, std::uint32_t lowest = 0);
inline constexpr std::uint64_t InPlaceDrawInputMaxReads = 512;
const HostImport* InPlaceDrawInput(const Context& context, std::uint64_t address, std::size_t bytes, std::size_t alignment, bool* behindGpuWrite = nullptr, std::size_t fetched = std::numeric_limits<std::size_t>::max(), std::uint64_t reads = std::numeric_limits<std::uint64_t>::max());
struct InPlaceIndices {
    std::optional<std::uint32_t> highest;
    std::uint32_t lowest = 0;
    bool reused = false;
};
InPlaceIndices InPlaceIndexRange(Recorder& recorder, std::uint64_t address, std::size_t bytes, std::uint32_t indexSize, bool skipRestart);

struct PreparedInputs {
    enum class Refusal : std::uint8_t { None, NoRecipe, Shape, NotInPlace, Inaccessible, Thrown };
    struct Range {
        std::uint64_t begin = 0;
        std::uint64_t end = 0;
        std::uint32_t alignment = 1;
        const HostImport* import = nullptr;
        VkBuffer handle = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        // APS5_MIRRORED_INPUTS, where draw inputs are not bound in place: `source` is the published copy
        // bound instead of an import (handle and offset are its own) and `ticket` what the commit proves
        // it by (Recorder::PublishedDrawInputHolds); `small` marks an import bound under the
        // Context::drawInputsInPlaceBelow rule, which the commit applies again.
        std::shared_ptr<Buffer> source;
        Recorder::DrawInputTicket ticket;
        bool small = false;
    };
    Refusal refusal = Refusal::None;
    std::shared_ptr<const DrawRecipe> recipe;
    const State* state = nullptr;
    const Pm4::DrawParameters* draw = nullptr;
    const CompiledShader* shaders = nullptr;
    std::size_t stages = 0;
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t forget = 0;
    std::uint64_t imports = 0;
    std::uint64_t mapping = 0;
    std::uint64_t indexBytes = 0;
    std::uint32_t maxIndex = 0;
    VertexInputLayout vertexInput;
    std::vector<VkBuffer> vertexHandles;
    std::vector<VkDeviceSize> vertexOffsets;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> inPlaceReads;
    std::vector<Range> ranges;
    std::size_t indirectRanges = 0;
    bool exactCapacity = false;
    // Prepared under APS5_MIRRORED_INPUTS: every range is a published copy or a small import, and
    // inPlaceReads lists the imports only.
    bool mirrored = false;
    GuardAnswer guardAnswer;
};
bool PrepareDrawInputs(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::shared_ptr<const DrawRecipe> recipe, PreparedInputs& prepared);
enum class InputsMemoOutcome : std::uint8_t { None, Hit, MissKey, MissSerial, Indirect };
struct InputsMemoNote {
    InputsMemoOutcome outcome = InputsMemoOutcome::None;
    bool stored = false;
    bool verified = false;
    bool differed = false;
};
InputsMemoNote TakeInputsMemoNote();
DrawRecordInputs VerifyPreparedInputsForTest(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawRecipe& recipe, PreparedInputs& prepared, std::string& difference, DrawRecordInputs* taken = nullptr);

std::array<std::uint32_t, 4> MeshIndexBufferDescriptor(const Pm4::DrawParameters& draw);

struct MeshArguments {
    std::uint32_t groups;
    std::uint32_t instances;
    std::uint32_t layers;
    std::uint32_t indexCount;
    std::uint32_t firstIndex;
};
static_assert(sizeof(MeshArguments) == ShaderRecompiler::MeshArgumentBytes);
struct MeshArgumentRules {
    std::uint32_t indexCount;
    std::uint32_t inputSize;
    std::uint32_t step;
    std::uint32_t primitivesPerGroup;
    std::uint32_t maxGroups;
    std::uint32_t maxInstances;
    std::uint32_t maxTotal;
};
MeshArgumentRules MeshArgumentRulesFor(const Context& context, const ShaderRecompiler::MeshConfiguration& mesh, std::uint32_t indexCount);
struct MeshGrid {
    std::uint32_t groups;
    std::uint32_t instances;
    std::uint32_t layers;
    std::uint32_t layerStride;
};
std::optional<MeshGrid> MeshGridFor(const Context& context, std::uint32_t groups, std::uint32_t instances);
MeshArguments ResolveMeshArguments(const Pm4::DrawArguments& record, const MeshArgumentRules& rules);

// Why DrawWithRecipe did not record from the recipe (the caller then runs Draw): the draw is not
// recordable (no recorder, APS5_SYNC_DRAWS, APS5_DUMP_TARGETS), a resident target is gone from the
// storage cache, the template is gone, the pipeline is gone from the pipeline store, or the
// template's proof failed (ShaderResources::ProveCurrent; the template is removed from the
// resource cache and kept by the batch).
enum class DrawRecipeMiss : std::uint8_t { None, NotRecordable, TargetGone, TemplateGone, ObjectsGone, Proof, Moved, Buffers = Moved, Count };
const char* DrawRecipeMissName(DrawRecipeMiss miss);
struct DrawRecipeOutcome {
    bool recorded = false;
    DrawRecipeMiss miss = DrawRecipeMiss::None;
    ShaderResources::ProofReport proof;
    // APS5_PROFILE_DRAW: the proof's and the record's time.
    double proofUs = 0;
    double recordUs = 0;
};
// A draw-cache hit recorded from its recipe, under GuestMemory::GpuMutex after the packet's labels
// (design_cpu_final M8, step 8b): the validation with the recipe's stored outputs, the index and
// vertex work as Draw's, the resident-target proof on the stored images (Refresh while Cached()),
// the template's proof (ProveCurrent) with the alias checks repeated, the pipeline and framebuffer
// from the recipe, then the record as Draw's (pass continued or begun, Kept, MarkGpuWrites). A
// miss records nothing.
DrawRecipeOutcome DrawWithRecipe(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, const DrawRecipe& recipe, const DrawTraceContext* trace = nullptr);
void FailRecordChecks(DrawRecipeMiss check, std::uint32_t count);
// `prepared`: the buffers a finisher prepared for a direct hit (APS5_DIRECT_BUFFERS), bound in place
// of today's moved-buffer check and snapshot selection while they hold for the draw.
DrawRecipeOutcome DrawWithRecord(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawRecordCommit& record, const DrawTraceContext* trace = nullptr, DrawCommitView* answer = nullptr, const ShaderResources::PreparedBuffers* prepared = nullptr, PreparedInputs* preparedInputs = nullptr, DrawFragment* fragment = nullptr, std::span<const GuestMemorySnapshot> snapshots = {});
bool RecordDrawFragment(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawFragmentBlock& block, const ShaderResources::PreparedBuffers& prepared, const PreparedInputs& inputs, DrawFragment& out, DrawFragmentRefusal& refusal);
std::uint64_t CommitFragmentsVerifyEvery();
struct ExpressOutcome {
    bool taken = false;
    bool verify = false;
    bool reproved = false;
    ShaderResources::ExpressMiss reproveMiss = ShaderResources::ExpressMiss::None;
    std::uint32_t refreshed = 0;
    DrawExpressMiss miss = DrawExpressMiss::None;
    DrawRecipeOutcome result;
};
std::array<std::uint64_t, 8> ExpressTargetMissCauses();
bool DrawBufferAliasesClear(std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint64_t indexAddress, std::uint64_t indexBytes);
ExpressOutcome ExpressCommit(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawRecordCommit& record, const ShaderResources::PreparedBuffers* prepared, PreparedInputs* preparedInputs, DrawFragment* fragment, bool verifyDue);
enum class ExpressRunEnd : std::uint8_t { NotQueued, NotDraw, Cap, LockWaiter, IdleWaiter, Refusal, Switch, Stop, Count };
enum class ExpressRunRefusal : std::uint8_t { NotDirect, Verifies, Unlocked, Labels, Experiment, Device, State, Switches, Objects, Recorder, Generation, Key, OwnWrites, OwnImages, NotAppendable, Serials, Linear, Writers, Targets, Multisample, Depth, Template, Prepared, Inputs, PassWrites, Fenced, InputWrites, InputStores, Buffers, Stores, Verify, Selection, Staging, PassMoved, OwnTargets, OwnSerials, ImageHazard, Indirect, InPlace, Count };
void BeginExpressRun();
void EndExpressRun(ExpressRunEnd reason);
void BreakExpressRun();
bool ExpressRunLed();
void RefuseExpressFollow(ExpressRunRefusal why);
std::uint64_t ExpressFollowAttempts();
void NoteExpressFollowItem(std::uint64_t nanoseconds);
void NoteExpressRunHold(std::uint64_t nanoseconds);
void NoteExpressRunItem();
void NoteExpressNonFollower(bool attempted, std::uint64_t attemptNanoseconds, std::uint64_t commitNanoseconds);
void DropExpressFollowVerify();
void NoteExpressFollowDue(bool due);
bool ExpressFollowCommit(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawRecordCommit& record, const ShaderResources::PreparedBuffers& prepared, PreparedInputs& preparedInputs, DrawFragment& fragment);
void NoteCommitterTrusted(std::uint64_t count = 1);
std::uint64_t TakeCommitterTrusted();
std::uint64_t ExpressVerifyEvery();
inline constexpr std::uint32_t CommitFeedBlocks = 1;
inline constexpr std::uint32_t CommitFeedCards = 2;
inline constexpr std::uint32_t CommitFeedAhead = 4;
enum class CommitFeedCount : std::uint8_t { BlocksKept, BlocksWithdrawn, CardFollowerUses, CardLeaderUses, CardGuardUses, HitsWithoutCard, AheadEntries, AheadNoHint, AheadLevel0, AheadLevel1, AheadLevel2, AheadLevel3, AheadLevel4, AheadLevel5, LeadAttachments, LeadTargets, LeadShort, LeadPassClosed, LeadNotAppendable, LeadMultisample, NotDirectHit, NotDirectTraced, NotDirectRecipe, NotDirectWriters, NotDirectPlan, NotDirectInputsRecipe, NotDirectFragment, NotDirectBuffers, NotDirectInputs, Count };
bool CommitFeedOn(std::uint32_t part);
bool CommitFeedCensus();
std::uint64_t CommitFeedVerifyEvery();
void NoteCommitFeed(CommitFeedCount count, std::uint64_t amount = 1);
void NoteCommitFeedVerify(bool same, const char* what, const std::string& detail);
std::uint64_t CommitFeedTotal(CommitFeedCount count);
std::array<std::uint64_t, 2> CommitFeedVerifyTotals();
std::string TakeCommitFeedText();
inline constexpr std::uint32_t FollowAheadGuards = 1;
inline constexpr std::uint32_t FollowAheadRenotes = 2;
std::uint32_t FollowAheadParts();
bool FollowAheadOn(std::uint32_t part);
bool FollowAheadCensus();
std::uint64_t FollowAheadVerifyEvery();
enum class GuardAnswerMade : std::uint8_t { Off, Made, SpansOff, ProofPending, StampMoved, BoardStale, GenerationMoved };
struct GuardAnswerNote {
    GuardAnswerMade made = GuardAnswerMade::Off;
    std::uint8_t clear = 0;
    std::uint32_t depthRanges = 0;
    std::uint32_t multisampleRanges = 0;
};
GuardAnswerNote AnswerPreparedGuards(ShaderResources::PreparedBuffers& prepared);
GuardAnswerNote AnswerPreparedInputGuards(PreparedInputs& prepared);
enum class FollowAheadCount : std::uint8_t { FollowerDepth, FollowerMultisample, FollowerInputs, LeaderDepth, LeaderMultisample, LeaderInputs, NoAnswer, DepthMoved, MultisampleMoved, ProofPending, SpansOff, VerifyDiffers, Followers, Segments, SameTemplate, UnmovedEntries, QueuedStores, MergeableRenotes, Count };
std::uint64_t FollowAheadTotal(FollowAheadCount count);
std::array<std::uint64_t, 2> FollowAheadVerifyTotals();
std::string TakeFollowAheadText();
enum class RepeatRefusal : std::uint8_t { NoBlock, NotSimple, VerifyDue, Shape, Objects, Batch, Pass, Epoch, ProofSerial, Pending, Depth, Target, Access, Stores, Linear, Buffers, Selection, DepthGuards, Inputs, Changed, MovedHazard, UnmovedHazard, Count };
inline constexpr std::array<const char*, static_cast<std::size_t>(RepeatRefusal::Count)> RepeatRefusalNames{"no block", "not simple", "verify due", "shape", "objects", "another batch", "pass", "collect epoch", "proof serial", "pending stores", "depth surfaces", "target changed", "pass access", "queued stores", "drawn linear targets", "no prepared buffers", "selection", "depth guards", "inputs", "changed by the inputs", "moved-range hazard", "unmoved-range hazard"};
inline constexpr std::array<const char*, static_cast<std::size_t>(DrawRepeatClass::Count)> RepeatClassNames{"simple", "not recorded", "template", "stages", "pipeline", "targets", "not pushed", "sets", "buffers", "inputs", "keep", "pass", "access", "notes", "modes"};
inline constexpr std::array<const char*, static_cast<std::size_t>(DrawRepeatDetail::Count)> RepeatDetailNames{"none", "not reusable", "heap stage", "lease", "address table", "copied writes", "fault checks", "space reuse", "GPU writes", "GPU copies", "depth copies", "multisampled", "multisampled depth", "not the recipe's", "linear", "fmask"};
struct RepeatOutcome {
    bool taken = false;
    RepeatRefusal refusal = RepeatRefusal::Count;
    DrawRepeatClass simple = DrawRepeatClass::Simple;
    DrawRepeatDetail detail = DrawRepeatDetail::None;
    bool sets = false;
    bool stores = false;
    bool shared = false;
    DrawRecordInputs inputs = DrawRecordInputs::None;
    bool late = false;
    bool passBegun = false;
    PassBreak passBreak = PassBreak::Continued;
    bool proofSkipped = false;
    bool unmovedSkipped = false;
    const char* refuted = nullptr;
    ShaderResources::PreparedDifference pushDifference;
    DrawRecipeOutcome result;
};
RepeatOutcome RepeatCommit(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawRecordCommit& record, const ShaderResources::PreparedBuffers* prepared, PreparedInputs* preparedInputs, const DrawTraceContext* trace = nullptr, DrawCommitView* describe = nullptr, std::uint64_t verifyEvery = 0);
using RepeatHookForTest = void (*)(DrawRecordCommit& record);
void SetRepeatHookForTest(RepeatHookForTest hook);
std::uint64_t CommitBindingsVerifyEvery();
using CommitBindingsHookForTest = void (*)(DrawRecordCommit& record);
void SetCommitBindingsHookForTest(CommitBindingsHookForTest hook);
using ExpressHookForTest = void (*)(DrawRecordCommit& record);
void SetExpressHookForTest(ExpressHookForTest hook);
// Whether DrawWithRecord can record a draw at all (APS5_SYNC_DRAWS and APS5_DUMP_TARGETS unset).
bool DrawsRecordable();
// Why a full draw asked for a record's recipe built none, counted per reason since the start:
// the request was dropped early (stencil replay, quad list, strip patches, heap stages without
// data recipes, proxied targets), the resources are not cacheable, the draw waited, it is
// indirect, the template is not reusable or reuses its space, a target is multisampled, the
// pipeline reorders samples, or the draw left the recorded path.
enum class RecipeRefusal : std::uint8_t { Dropped, NotCacheable, Waited, Indirect, NotReusable, SpaceReuse, Multisampled, ReordersSamples, NotRecorded, Count };
inline constexpr std::array<const char*, static_cast<std::size_t>(RecipeRefusal::Count)> RecipeRefusalNames{"dropped early", "not cacheable", "waited", "indirect", "template not reusable", "space reuse", "multisampled", "reorders samples", "not recorded"};
void NoteRecipeRefusal(RecipeRefusal reason);
std::array<std::uint64_t, static_cast<std::size_t>(RecipeRefusal::Count)> RecipeRefusals();
// Whether recorded draws build recipes and draw-cache hits use them (APS5_NO_DRAW_RECIPE unset).
bool DrawRecipes();

// How an indirect draw's records were read, for the [draws] line (APS5_PROFILE_DRAW): Gpu is
// vkCmdDrawIndirect from the host import; every other value names why the CPU read them instead,
// either decided by the driver before the draw (its patched-SGPR fold rules, the shader path, the
// draw index, the vertex ranges, the device features, APS5_NO_GPU_INDIRECT_DRAW) or under the device
// lock in Draw (the memory state of the records). `readMs` is the CPU read's time (its sync);
// `rewritten` counts a GPU-side draw whose records were copied and patched with a constant.
enum class IndirectDrawPath : std::uint8_t { Gpu, NotFolded, FetchUnknown, NonVertexPath, IndxOffset, DrawIndex, VertexRange, FeatureGap, PendingImage, PendingLabelOrCopy, NotImported, Disabled, IndexRange, IndexLimitUnknown, Settled, Count };
void CountIndirectDraw(IndirectDrawPath path, double readMs, bool rewritten = false);
std::array<std::uint64_t, static_cast<std::size_t>(IndirectDrawPath::Count)> IndirectDrawCounts();
std::string IndirectDrawTotalsText();
void CountEmptyIndirectDraw(double readMs);
std::uint64_t EmptyIndirectDrawCount();

struct DrawPassCounts {
    std::uint64_t draws = 0;
    std::uint64_t passesBegun = 0;
    std::uint64_t passesContinued = 0;
    std::uint64_t keyStoreFlushes = 0;
    std::uint64_t storeFlushes = 0;
};
std::array<DrawPassCounts, 2> DrawPassTotals();
std::array<std::uint64_t, PassBreakCount> DrawPassBreaks();
const char* IndirectDrawPathName(IndirectDrawPath path);

// Draw packets that drew nothing, for the [draws] line: Nothing (an empty count, or no color writes
// and no pixel shader), Prechecked (State.hpp's DrawRejection refused it before the decode) and
// Thrown (the decode or the build threw); `us` is the packet's time in the driver.
enum class DrawSkip : std::uint8_t { Nothing, Prechecked, Thrown, Count };
void CountDrawSkip(DrawSkip kind, double us);

// Recorded draws whose written guest buffers were copied (their results reach guest memory by a CPU
// write-back when their batch completes), listed until that write-back ran. It is the draw
// counterpart of VulkanDevice's State::copiedWriters for dispatches, which VulkanDevice.cpp consults
// in two places: DispatchIndirect (an indirect dispatch whose arguments such a writer produced reads
// them on the CPU, behind the write-back, instead of from the host import on the GPU) and the fill
// HLE (a vkCmdFillBuffer over a range a listed write-back will land on must wait for it, or the
// write-back overwrites the fill). Draws are listed here only under APS5_RECORD_COPIED_DRAWS=1,
// which is safe only once BOTH of those checks also test this list, i.e. each extends its
// `std::any_of(state->copiedWriters ...)` with
// `|| std::any_of(Graphics::DrawCopiedWriters()->begin(), Graphics::DrawCopiedWriters()->end(), [&](const auto& writer) { return writer->WritesOverlap(<its range>); })`;
// cleaner still, State::copiedWriters can become this list (one registry for both producers), after
// which the switch can turn into APS5_NO_RECORD_COPIED_DRAWS. Until then such draws are recorded and
// waited for at once. Read and written under GuestMemory::GpuMutex, like the device's list.
std::shared_ptr<std::vector<std::shared_ptr<ShaderResources>>> DrawCopiedWriters();
void NoteCopiedWriter(bool draw);
std::array<std::uint64_t, 2> CopiedWriterTotals();
void NoteIdleStore();
struct ResidentCensusTotals {
    std::uint64_t cpuPublishes = 0;
    std::uint64_t idleStores = 0;
    std::array<std::uint64_t, 4> retires{};
    std::uint64_t templatesReplaced = 0;
};
ResidentCensusTotals ResidentCensus();
std::string TakeFullDrawLookupText();
std::string TakeTemplateBuildText();
std::string TakeTemplateKeepText();
std::string TakeViewTemplateKeepText();
enum class FullDrawCause : std::uint8_t { Other, TodaysFinisher, TodaysWorker, TodaysInline, RecordFirst, RecordFirstAddressed, RecordMissed, RecordVerify, SpaceSwitchOff, SpaceKeyMoved, SpaceRefused, SpaceVerify, Count };
class FullDrawCauseScope {
public:
    explicit FullDrawCauseScope(FullDrawCause cause);
    ~FullDrawCauseScope();
    FullDrawCauseScope(const FullDrawCauseScope&) = delete;
    FullDrawCauseScope& operator=(const FullDrawCauseScope&) = delete;

private:
    FullDrawCause previous = FullDrawCause::Other;
    bool set = false;
};
std::string TakeFullDrawStepText();
enum class RecordHeavyEvent : std::uint8_t { MissNotRecordable, MissTargets, MissTemplate, MissObjects, MissProof, MissMoved, SpaceInUse, SpaceChanged, SpacePartial, SpaceSnapshotChanged, SpaceOutside, FirstCommit, SwitchOff, KeyMoved, Refused, VerifyDue, Built, RefusedDropped, RefusedNotCacheable, RefusedWaited, RefusedIndirect, RefusedMoved, RefusedMultisampled, RefusedReorders, RefusedShape, Adopted, NotAdopted, VerifyChecked, VerifyDiffering, Count };
void NoteRecordHeavy(RecordHeavyEvent event);
void NoteRecordHeavyMiss(DrawRecipeMiss miss);
std::uint64_t RecordHeavyVerifyEvery();
void NoteRecordHeavyVerifyDifference(std::string_view field, std::string_view detail);
std::string TakeRecordHeavyText();

void RunColorMetadataPass(const Context& context, const ColorMetadataPass& pass);
std::vector<std::pair<std::uint64_t, std::uint64_t>> PipelinedColorMetadataPassWrites(const ColorMetadataPass& pass);
bool DrawPipelineEnabled();
bool CommitTrims();
struct FragmentOutputLookups {
    std::uint64_t last = 0;
    std::uint64_t memo = 0;
    std::uint64_t validated = 0;
};
FragmentOutputLookups ThreadFragmentOutputLookups();
enum class AddressDrawReuse : std::uint8_t { Reused, Built, NotReadOnly, NotSpaceOnly, SnapshotOutside, PartialOverlap, SnapshotChanged, MovedBuffer, PatchedData, SpaceChanged, InUse, Revalidation, Count };
using AddressDrawReuseCounts = std::array<std::uint64_t, static_cast<std::size_t>(AddressDrawReuse::Count)>;
AddressDrawReuseCounts AddressDrawReuseCounters();
const char* AddressDrawReuseName(AddressDrawReuse outcome);
std::uint64_t AddressDrawReuseVerifications();
struct VertexPlanScratchState {
    bool busy = false;
    std::size_t resting = 0;
    std::size_t capacity = 0;
};
VertexPlanScratchState ThreadVertexPlanScratch();
enum class DrawStopForTest { None = 0, ReturnAfterResolve = 1, ThrowAfterResolve = 2 };
void SetSpaceKeepFailuresForTest(std::uint32_t count);
void SetDrawStopAfterResolveForTest(DrawStopForTest stop);
enum class TargetMemoEvent : std::uint8_t {
    Lookup, Hit, OverlapCached, OverlapComputed, Accept, AcceptWithDepthMiss, DepthHit, AvoidedRefreshLookups,
    Invalid, ThreadChanged, Descriptor, ResidentGone, ResidentEvicted, DepthOverlap, Metadata,
    Epoch, Proofs, Pending, Unwatch, Allocations, Surfaces, Released, Claims, Recorder, Device,
    PostDepthInvalidated, DepthFallback, Admission, RecipeInvalidated, Build, VerifyChecked, Count
};
std::uint64_t TargetMemoEvents(TargetMemoEvent event);
const char* TargetMemoEventName(TargetMemoEvent event);
std::uint64_t TargetMemoBuildsForTest();
std::uint64_t TargetMemoHitNanosecondsForTest(const Context& context, const State& state, std::size_t repeats, std::size_t& hits);

}

#endif
