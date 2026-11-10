#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_RECIPE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_RECIPE_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
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
    bool cleanProofs = false;
    // Set by the committer for the next commit: compare the hit's prepared buffers with today's way
    // (APS5_VERIFY_DIRECT_BUFFERS) and draw today's way.
    bool verifyBuffers = false;
    // The buffer plan of the recipe's template while APS5_DIRECT_BUFFERS is set, published by the
    // committer after a recorded commit: the only member a finisher thread reads. It owns the plan,
    // because the recipe holds its template weakly.
    AtomicSharedPtr<const Graphics::ShaderResources::BufferPlan> bufferPlan;
    bool bufferPlanAsked = false;

    void Adopt(std::shared_ptr<const DrawRecipe> built) {
        recipe = std::move(built);
        if (std::exchange(bufferPlanAsked, false)) bufferPlan.store(nullptr);
        pass = {};
        keep = {};
        ticket = {};
        ticketRanges.clear();
        access = {};
        pushPipeline.reset();
        pushFramebuffer.reset();
        pushTemplate = nullptr;
        touchedPresent = ~std::uint64_t{0};
        ++builds;
    }
};

}

#endif
