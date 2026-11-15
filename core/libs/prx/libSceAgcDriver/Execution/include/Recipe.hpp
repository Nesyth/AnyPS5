#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_RECIPE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_RECIPE_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "prx/libSceAgcDriver/Graphics/include/CommandStream.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include "prx/libc/include/general/AtomicSharedPtr.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace AgcDriver {

// A compute dispatch's Vulkan objects (VulkanDevice.cpp), shared by every dispatch of one compiled
// variant.
struct ComputePipelineObjects;

// What a value-equal dispatch-cache hit needs to record its dispatch without preparing, finding or
// comparing anything again (design_cpu_final M4): the template the resource cache served the
// entry's compiled result with, the pipeline objects of its variant, the push constants and data
// words the compiled result fixes, and the pre-sync answer. Built at the end of a successful device
// call of a reusable object and attached to the dispatch-cache variant; immutable after that. It
// holds weak references only: the resource cache owns the template (its 1024-entry bound stays
// what pins textures and storage images) and the device's pipeline map owns the objects, so a dead
// reference is a recipe miss, never a leak past the caches' budgets.
struct Recipe {
    // R10: unusable on any other device.
    VkDevice device = VK_NULL_HANDLE;
    std::weak_ptr<Graphics::ShaderResources> templateRef;
    // The template's content key: touches the cache's LRU on a hit.
    Graphics::ResourceCache::Key key;
    std::weak_ptr<ComputePipelineObjects> objects;
    std::array<std::byte, Graphics::PipelinePushConstantBytes> pushBytes{};
    bool pushes = false;
    // FNV over the compiled result's ShaderData/FlattenedSrt words (ShaderResources::DataWordsHash):
    // the template's data buffers need a refresh iff its own hash differs.
    std::uint64_t dataWordsHash = 0;
    // The template's PresyncSurfaces() as of the attach and the host-import identity they were
    // computed under (a surface is CPU-read only while its memory has no import).
    std::vector<Graphics::PresyncSurface> presyncSurfaces;
    Graphics::HostImportsProof presyncProof;
    // Both false for every recipe (only reusable objects get one); kept for the asserts.
    bool needsCompletion = false;
    bool holdsLease = false;
    bool moves = false;
};

// Recorded: the dispatch went into the open batch from the recipe; Rebuild: the proof failed or an
// object went away, the caller runs the ordinary path (the recipe is re-attached after it).
enum class RecipeOutcome { Recorded, Rebuild };

// What a recorded draw needs to record again from its draw-cache hit without looking anything up
// (design_cpu_final M8, step 8b): the resource-cache template the draw's stages were served with
// (weak, as a dispatch recipe's), its pipeline and framebuffer (weak, R10: the pipeline store owns
// the pipeline and destroys it with its device (ClearCachedPipelines), the pipeline owns its
// framebuffer list; a recipe outlives a device replacement in the draw cache, so an owning
// reference would destroy the objects against a dead VkDevice), the resident targets (weak: a hit
// proves them on the stored objects while they are still Cached()) with their attachment views,
// and the per-draw derivations of the compiled stages and the state: the vertex input layout, the
// push constant block, the state with the unwritten outputs masked, the fragment outputs and the
// pipeline stages. Built at the end of a recorded, cacheable, reusable, non-indirect draw and
// attached to the draw entry's matched stage variants; immutable.
struct DrawRecipe {
    VkDevice device = VK_NULL_HANDLE;
    std::weak_ptr<Graphics::ShaderResources> templateRef;
    Graphics::ResourceCache::Key key;
    std::weak_ptr<Graphics::Pipeline> pipeline;
    std::weak_ptr<Graphics::Framebuffer> framebuffer;
    std::vector<std::weak_ptr<Graphics::StorageTexture>> targets;
    std::vector<VkImageView> targetViews;
    std::uint64_t passKey = 0;
    Graphics::VertexInputLayout vertexInput;
    std::array<std::byte, Graphics::PipelinePushConstantBytes> pushBytes{};
    VkShaderStageFlags pushStages = 0;
    std::optional<Graphics::State> masked;
    std::uint32_t fragmentOutputs = 0;
    VkPipelineStageFlags shaderStages = 0;
    bool hasDepth = false;
    // One entry per color target when any of them is multisampled (APS5_MSAA_RECORDS), else empty.
    // A multisampled target has no resident image in `targets`: a hit asks for its view per draw.
    std::vector<std::uint8_t> multisampled;
    // Made while data recipes were on (APS5_NO_DATA_RECIPES unset): a hit may be a stage data hit
    // with its own data words and read-only buffer bases, which DrawWithRecipe moves per draw as a
    // template hit does.
    bool perDrawMoves = false;
    bool indirect = false;
    bool spaceReuse = false;
};

struct DrawRecordPass {
    bool valid = false;
    std::uint64_t recorder = 0;
    std::uint64_t generation = 0;
    std::uint64_t epoch = 0;
    std::uint64_t proof = 0;
    std::uint64_t pending = 0;
    std::uint64_t depthSerial = 0;
    std::uint64_t depthReleased = 0;
    std::vector<std::uint64_t> versions;
};

struct DrawRecordKeep {
    std::uint64_t recorder = 0;
    std::uint64_t batch = 0;
    const void* resources = nullptr;
    const void* pipeline = nullptr;
    const void* framebuffer = nullptr;
    std::vector<const void*> targets;
};

enum class DrawRecordHold : std::uint8_t { Held, First, Pass, Epoch, Proof, Pending, Depth, Target, Count };

// What a record commit did with a direct hit's prepared buffers (APS5_DIRECT_BUFFERS): bound them,
// why it took today's way instead, or how a verify compared them with today's way.
enum class DrawRecordBuffers : std::uint8_t { None, Used, PlanDiffers, MovedHazard, UnmovedHazard, DepthGuards, Checked, Differs, MemoryChanged, Unwatched, Count };

enum class DrawRecordInputs : std::uint8_t { None, Used, Rechecked, ImportChanged, Shape, Refused, Checked, Differs, Count };

struct DrawRecordBindings {
    bool timed = false;
    bool routed = false;
    bool verify = false;
    bool checked = false;
    bool todayTaken = false;
    bool written = false;
    bool pushed = false;
    Graphics::ShaderResources::BindingTaken taken = Graphics::ShaderResources::BindingTaken::None;
    std::uint32_t tested = 0;
    std::uint32_t stamped = 0;
    std::uint32_t refuted = 0;
    std::uint32_t objects = 0;
    std::uint32_t setsLent = 0;
    std::uint32_t setsWritten = 0;
    std::uint32_t guards = 0;
    std::uint32_t dynamicBound = 0;
    std::uint32_t dynamicHits = 0;
    std::uint32_t dynamicWritten = 0;
    std::uint32_t dynamicRefused = 0;
    std::uint8_t dynamicLeave = 0;
    std::size_t bytes = 0;
    std::uint64_t nanoseconds = 0;
    std::uint64_t keepNanoseconds = 0;
    std::vector<Graphics::ShaderResources::BoundBuffer> mine;
    std::vector<Graphics::ShaderResources::BoundBuffer> today;
};

struct DrawFragmentBlock {
    std::weak_ptr<const Graphics::ShaderResources> resources;
    std::weak_ptr<Graphics::Pipeline> pipeline;
    std::weak_ptr<Graphics::Pipeline> pushPipeline;
    const Graphics::ShaderResources::PushPlan* push = nullptr;
    VkDevice device = VK_NULL_HANDLE;
};

enum class DrawFragmentRefusal : std::uint8_t { None, NoBlock, Gone, Device, Modes, Stages, Pipeline, Selection, Thrown, Count };
inline constexpr std::array<const char*, static_cast<std::size_t>(DrawFragmentRefusal::Count)> DrawFragmentRefusalNames{"none", "no emit block", "objects gone", "another device", "modes", "stages", "pipeline", "selection", "thrown"};
enum class DrawFragmentDrop : std::uint8_t { None, Objects, Selection, Inputs, Bindings, Modes, Refused, Verify, Count };
inline constexpr std::array<const char*, static_cast<std::size_t>(DrawFragmentDrop::Count)> DrawFragmentDropNames{"none", "other objects", "selection", "inputs", "bindings", "modes", "record refused", "verify"};

struct DrawFragment {
    Graphics::CommandStream::Fragment commands;
    std::shared_ptr<Graphics::ShaderResources::DrawBindings> bindings;
    std::shared_ptr<const Graphics::ShaderResources> resources;
    std::shared_ptr<const Graphics::Pipeline> pipeline;
    const Graphics::ShaderResources::PushPlan* push = nullptr;
    bool pushing = false;
    bool written = false;
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t nanoseconds = 0;
    std::uint64_t faultGeneration = 0;
    bool indirect = false;
    VkBuffer arguments = VK_NULL_HANDLE;
    VkDeviceSize argumentOffset = 0;
    VkBuffer count = VK_NULL_HANDLE;
    VkDeviceSize countOffset = 0;
};

struct DrawExpressKeep {
    std::shared_ptr<Graphics::ShaderResources> resources;
    std::shared_ptr<const Graphics::Pipeline> pipeline;
    std::shared_ptr<Graphics::Framebuffer> framebuffer;
    std::vector<std::shared_ptr<Graphics::StorageTexture>> targets;
    std::vector<std::shared_ptr<void>> multisample;
};

struct DrawExpress {
    bool valid = false;
    const DrawRecipe* recipe = nullptr;
    const Graphics::State* state = nullptr;
    std::uint64_t switches = 0;
    VkDevice device = VK_NULL_HANDLE;
    std::shared_ptr<DrawExpressKeep> keep;
    std::vector<std::uint64_t> targetEpochs;
    std::vector<std::uint64_t> targetVersions;
    std::vector<std::uint64_t> dccAddresses;
    std::vector<VkImage> multisampleImages;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> slices;
    std::uint64_t passKey = 0;
    bool fenced = false;
    bool hasDepth = false;
    VkImage restartDepth = VK_NULL_HANDLE;
    std::uint64_t depthSerial = 0;
    std::uint64_t depthReleased = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writeOnly;
    std::vector<std::pair<VkImage, bool>> images;
    std::uint64_t keptRecorder = 0;
    std::uint64_t keptBatch = 0;
    std::uint64_t passSeenRecorder = 0;
    std::uint64_t passSeenGeneration = 0;
};

enum class DrawExpressMiss : std::uint8_t { None, NoState, Shape, Objects, Template, Targets, Depth, Stores, Linear, Buffers, Inputs, Moved, Selection, Staging, Verify, Count };
inline constexpr std::array<const char*, static_cast<std::size_t>(DrawExpressMiss::Count)> DrawExpressMissNames{"none", "no express state", "shape", "objects", "template", "targets", "depth", "queued stores", "drawn linear", "buffers", "inputs", "moved hazard", "selection", "staging", "verify"};

enum class DrawExpressCapture : std::uint8_t { None, Captured, NotSpliced, Objects, NotProved, Watch, Forget, Serials, Sources, Views, PendingCpu, Staging, Regions, Access, NotLean, Multisampled, Refused, FunctionLds, Fmask, Unreached, Count };

enum class DrawFragmentBlockMiss : std::uint8_t { None, NotReusable, SpaceReuse, Bda, HeapStage, Lease, CopiedWrites, GpuCopies, BufferWrites, StorageWrites, FaultBuffer, NoBuffers, Stencil, MeshWindow, Ineligible, Refused, Modes, Count };

struct DrawRecordFragment {
    DrawExpressCapture expressCapture = DrawExpressCapture::None;
    DrawFragmentBlockMiss blockMiss = DrawFragmentBlockMiss::None;
    bool offered = false;
    bool spliced = false;
    bool replayed = false;
    bool verified = false;
    bool differs = false;
    bool timed = false;
    DrawFragmentDrop drop = DrawFragmentDrop::None;
    const char* what = nullptr;
    std::uint64_t nanoseconds = 0;
};

struct DrawRecordEffects {
    bool pushed = false;
    bool unpushed = false;
    bool passHeld = false;
    bool keptOnce = false;
    std::uint32_t notesSkipped = 0;
    std::size_t movedChecked = 0;
    std::size_t movedTrusted = 0;
    DrawRecordHold hold = DrawRecordHold::Count;
    bool sharedHold = false;
    bool proved = false;
    std::optional<Graphics::ShaderResources::CleanProof> cleanProof;
    DrawRecordBuffers buffers = DrawRecordBuffers::None;
    Graphics::ShaderResources::PreparedDifference bufferDifference;
    DrawRecordInputs inputs = DrawRecordInputs::None;
    DrawRecordBindings bindings;
    DrawRecordFragment fragment;
    std::uint8_t bufferShortcuts = 0;
    bool inputsCapacity = false;
};

struct DrawRecordAccess {
    const void* recorder = nullptr;
    std::uint64_t generation = 0;
    const void* resources = nullptr;
    std::vector<const void*> targets;
    bool readsTarget = false;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writeOnly;
    std::vector<std::pair<VkImage, bool>> images;
};

enum class DrawCommitLookup : std::uint8_t { Unknown, Hit, Absent, Invalid, Unmoved, Uncacheable };

struct DrawCommitNote {
    bool present = false;
    bool linear = false;
    bool renote = false;
    bool storesKeys = false;
    std::uint32_t noteFirst = 0;
    std::uint32_t noteLayers = 0;
    std::uint32_t markedFirst = 0;
    std::uint32_t markedLayers = 0;
};

struct DrawCommitView {
    bool filled = false;
    const void* resources = nullptr;
    const void* pipeline = nullptr;
    const void* framebuffer = nullptr;
    bool wantPush = false;
    const void* pushPipeline = nullptr;
    const void* pushFramebuffer = nullptr;
    DrawCommitLookup lookup = DrawCommitLookup::Unknown;
    const void* found = nullptr;
    bool bound = false;
    std::vector<Graphics::ShaderResources::BoundBuffer> bindings;
    std::optional<DrawRecordAccess> access;
    bool keptOnce = false;
    DrawRecordKeep keep;
    std::size_t gpuWrites = 0;
    bool passNotes = false;
    std::vector<DrawCommitNote> notes;
    std::vector<std::shared_ptr<const void>> held;
    Graphics::ResourceCache::Key key;
    std::vector<VkImageView> views;
    std::vector<const void*> targets;
    std::uint64_t passKey = 0;
    std::vector<std::pair<std::size_t, std::uint64_t>> moved;
    std::vector<std::size_t> movedSizes;
    std::vector<std::vector<std::uint32_t>> movedWords;
    std::array<std::byte, Graphics::PipelinePushConstantBytes> push{};
    VkShaderStageFlags pushStages = 0;
    Graphics::VertexInputLayout vertexInput;
    std::uint32_t fragmentOutputs = 0;
    VkPipelineStageFlags shaderStages = 0;
    bool hasDepth = false;
    bool pushed = false;
};

struct DrawCommitViewTag {};
using DrawCommitViewSlot = HostThreadSlot<DrawCommitView*, DrawCommitViewTag>;

enum class DrawRepeatClass : std::uint8_t { Simple, NotRecorded, Template, Stages, Pipeline, Targets, NotPushed, Sets, Buffers, Inputs, Keep, Pass, Access, Notes, Modes, Count };
enum class DrawRepeatDetail : std::uint8_t { None, NotReusable, HeapStage, Lease, AddressTable, CopiedWrites, FaultChecks, SpaceReuse, GpuWrites, GpuCopies, DepthCopies, Multisampled, MultisampledDepth, NotRecipe, Linear, Fmask, Count };

struct DrawRepeatBlock {
    DrawRepeatClass simple = DrawRepeatClass::NotRecorded;
    DrawRepeatDetail detail = DrawRepeatDetail::None;
    bool sets = false;
    const DrawRecipe* recipe = nullptr;
    const Graphics::State* state = nullptr;
    std::uint64_t switches = 0;
    VkDevice device = VK_NULL_HANDLE;
    const void* descriptorCache = nullptr;
    std::array<const void*, 4> programs{};
    std::size_t stages = 0;
    Graphics::ShaderResources* resources = nullptr;
    Graphics::Pipeline* pipeline = nullptr;
    Graphics::Framebuffer* framebuffer = nullptr;
    const void* recipePipeline = nullptr;
    const void* recipeFramebuffer = nullptr;
    std::vector<Graphics::StorageTexture*> targets;
    std::vector<std::uint64_t> dccAddresses;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> slices;
    const Graphics::ShaderResources::PushPlan* push = nullptr;
    std::uint64_t passKey = 0;
    bool fenced = false;
    bool notesHeld = false;
    std::uint64_t writeNotes = 0;
    Graphics::ShaderResources::RepeatPush bindings;
    std::vector<bool> moved;
    std::uint64_t uploadRecorder = 0;
    std::uint64_t uploadBatch = 0;
    const void* upload = nullptr;
};

struct DrawRecordCommit {
    std::shared_ptr<const DrawRecipe> recipe;
    DrawRecordPass pass;
    DrawRecordKeep keep;
    Graphics::ShaderResources::MoveTicket ticket;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ticketRanges;
    DrawRecordAccess access;
    DrawRecordEffects effects;
    std::weak_ptr<Graphics::Pipeline> pushPipeline;
    std::weak_ptr<Graphics::Framebuffer> pushFramebuffer;
    const void* pushTemplate = nullptr;
    std::uint64_t touchedPresent = ~std::uint64_t{0};
    std::uint64_t builds = 0;
    std::uint8_t writers = 0;
    bool cleanProofs = false;
    // Set by the committer for the next commit: compare the hit's prepared buffers with today's way
    // (APS5_VERIFY_DIRECT_BUFFERS) and draw today's way.
    bool verifyBuffers = false;
    // The buffer plan of the recipe's template while APS5_DIRECT_BUFFERS is set, published by the
    // committer after a recorded commit: the only member a finisher thread reads. It owns the plan,
    // because the recipe holds its template weakly.
    AtomicSharedPtr<const Graphics::ShaderResources::BufferPlan> bufferPlan;
    bool bufferPlanAsked = false;
    bool verifyInputs = false;
    AtomicSharedPtr<const DrawRecipe> inputsRecipe;
    bool inputsRecipeAsked = false;
    std::shared_ptr<DrawRepeatBlock> repeat;
    bool bindingsRefused = false;
    std::weak_ptr<Graphics::Pipeline> dynamicPipeline;
    std::weak_ptr<Graphics::Framebuffer> dynamicFramebuffer;
    const void* dynamicTemplate = nullptr;
    std::weak_ptr<Graphics::Pipeline> pushDynamicPipeline;
    std::weak_ptr<Graphics::Framebuffer> pushDynamicFramebuffer;
    const void* pushDynamicTemplate = nullptr;
    bool secondSetWritten = false;
    DrawRepeatClass repeatSeen = DrawRepeatClass::NotRecorded;
    std::shared_ptr<DrawExpress> express;
    bool expressRefused = false;
    AtomicSharedPtr<const DrawFragmentBlock> fragmentBlock;
    const void* fragmentResources = nullptr;
    const void* fragmentPipeline = nullptr;
    const void* fragmentPushPipeline = nullptr;
    bool fragmentsRefused = false;
    bool spaceRefused = false;

    void Adopt(std::shared_ptr<const DrawRecipe> built) {
        recipe = std::move(built);
        repeat.reset();
        express.reset();
        if (fragmentResources != nullptr) fragmentBlock.store(nullptr);
        fragmentResources = nullptr;
        fragmentPipeline = nullptr;
        fragmentPushPipeline = nullptr;
        if (std::exchange(bufferPlanAsked, false)) bufferPlan.store(nullptr);
        if (std::exchange(inputsRecipeAsked, false)) inputsRecipe.store(nullptr);
        pass = {};
        keep = {};
        ticket = {};
        ticketRanges.clear();
        access = {};
        pushPipeline.reset();
        pushFramebuffer.reset();
        pushTemplate = nullptr;
        dynamicPipeline.reset();
        dynamicFramebuffer.reset();
        dynamicTemplate = nullptr;
        pushDynamicPipeline.reset();
        pushDynamicFramebuffer.reset();
        pushDynamicTemplate = nullptr;
        secondSetWritten = false;
        repeatSeen = DrawRepeatClass::NotRecorded;
        touchedPresent = ~std::uint64_t{0};
        ++builds;
    }
};

}

#endif
