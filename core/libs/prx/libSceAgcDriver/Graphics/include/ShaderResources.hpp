#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHADERRESOURCES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHADERRESOURCES_HPP

#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BindlessHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/HeapReplay.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "Recompiler.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/WriteValidation.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuardSpans.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace AgcDriver { class VulkanDevice; }
namespace AgcDriver::GuestMemory { struct WriteSubscription; }

namespace AgcDriver::Graphics {


class Recorder;
class OwnDrawSets;
class RepeatSets;
class RepeatSetLease;
struct ShaderResourcesTestAccess;

void FlushCachedTextures(VkDevice device);
void ClearCachedTextures(VkDevice device);
struct WriteIndexOutcomes {
    std::uint64_t accepted = 0;
    std::uint64_t partial = 0;
    std::uint64_t fallback = 0;
    std::uint64_t bumped = 0;
    std::uint64_t lookedUp = 0;
    std::uint64_t rebound = 0;
    std::uint64_t compared = 0;
    std::uint64_t builds = 0;
};
WriteIndexOutcomes WriteIndexOutcomesCounted();
std::uint64_t ReleaseTextureCaches(std::uint64_t bytes, bool newImage = false);
struct StorageTextureBudgetSnapshot {
    std::optional<std::uint32_t> heap;
    std::uint64_t sampledBytes = 0;
    std::uint64_t storageBytes = 0;
    std::uint64_t reserved = 0;
    std::uint64_t available = 0;
    std::uint64_t ceiling = 0;
    std::uint64_t selectedBudget = 0;
};

StorageTextureBudgetSnapshot CalculateStorageTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t sampledBytes, std::uint64_t storageBytes);
std::uint64_t TextureCacheBudget(const VkPhysicalDeviceMemoryProperties& memory);
std::uint64_t StorageTextureCeiling(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported);
std::uint64_t StorageTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t sampledBytes, std::uint64_t storageBytes);
std::uint64_t SampledTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t textureBytes, std::uint64_t storageBytes = 0);
bool SampledBudgetReportDue(std::uint64_t reported, std::uint64_t budget, std::chrono::steady_clock::duration sinceReport);
std::optional<std::uint32_t> LargestDeviceLocalHeap(const VkPhysicalDeviceMemoryProperties& memory);
std::uint64_t StorageTextureCacheBudget(const Context& context);
inline constexpr std::uint64_t SpareSnapshotPresents = 16;
struct TextureCacheUse {
    std::size_t sampledEntries = 0;
    std::size_t spareEntries = 0;
    std::uint64_t sampledBytes = 0;
    std::uint64_t spareBytes = 0;
    std::uint64_t storageBytes = 0;
};
TextureCacheUse TextureCacheUsage();
void OverrideCacheBudgetForTest(std::uint64_t bytes);
void SampledFromStorageForTest(bool enabled);
std::uint64_t SampledTextureCacheBudget(const Context& context);
std::shared_ptr<Texture> CachedSampledTexture(const Context& context, std::span<const std::uint32_t> words, VkImageViewType multisampleViewType = VK_IMAGE_VIEW_TYPE_2D);
bool SampledTexturesShareEntry(std::span<const std::uint32_t> first, std::span<const std::uint32_t> second);
bool AnyPendingCpuWrites(const Context& context);
bool CompletePendingCpuWrites(const Context& context, std::uint64_t address, std::size_t bytes);
bool SpirvFaultReportOnly(std::span<const std::uint32_t> words, std::uint32_t set, std::uint32_t binding);
void LastRecordIndices(std::span<const void* const> logged, std::span<const void* const> objects, std::vector<std::size_t>& indices);

// The cached storage image of a surface (render targets use it as their resident image); brought up
// to date with guest memory before it is returned.
std::shared_ptr<StorageTexture> CachedStorageSurface(const Context& context, const GuestTextureResource& resource);
std::shared_ptr<StorageTexture> CachedStorageLayers(const Context& context, const GuestTextureResource& resource, std::uint32_t firstLayer, std::uint32_t layerCount);
// Whether `image` is still the storage cache's image of its surface (what CachedStorageSurface
// would return); a true answer counts as a use for the cache's eviction order, as a lookup would.
bool StorageImageCached(const Context& context, const StorageTexture* image);
bool StorageImageServesKeys(const StorageTexture& image, std::uint64_t dccAddress);
std::shared_ptr<StorageTexture> RepeatStorageSurface(const Context& context, const GuestTextureResource& resource, const std::shared_ptr<StorageTexture>& stored);

// Defined in Texture.cpp beside the pending-results registry, for the fast Revalidate below: whether
// a storage image other than `except` has results pending in [address, address + bytes).
bool PendingStorageOverlaps(std::uint64_t address, std::size_t bytes, const StorageTexture* except);

inline std::size_t HeapStageIndex(ShaderRecompiler::ShaderStage stage) {
    using ShaderRecompiler::ShaderStage;
    return static_cast<std::size_t>(ShaderRecompiler::RuntimeAbi::StageGroup(stage == ShaderStage::Fragment, stage == ShaderStage::TessellationControl, stage == ShaderStage::TessellationEvaluation));
}

struct PipelineSetLayouts {
    VkDescriptorSetLayout main = VK_NULL_HANDLE;
    VkDescriptorSetLayout fragment = VK_NULL_HANDLE;
    bool split = false;
    std::span<const std::uint32_t> mainKey;
    std::span<const std::uint32_t> fragmentKey;
    std::int32_t pushSet = -1;
    bool dynamic = false;
};

struct TableDescriptors {
    std::uint32_t binding = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> runs;
    std::vector<VkDescriptorImageInfo> images;
    std::shared_ptr<const std::vector<std::weak_ptr<Texture>>> textures;
};

class DescriptorCache {
public:
    explicit DescriptorCache(const Context& context);
    ~DescriptorCache();
    DescriptorCache(const DescriptorCache&) = delete;
    DescriptorCache& operator=(const DescriptorCache&) = delete;

    // The layout for `key` (binding, type, count, stage flags per binding, as ShaderResources builds
    // it from `bindings`), created on first use.
    VkDescriptorSetLayout Layout(std::span<const std::uint32_t> key, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind = false, std::span<const VkDescriptorBindingFlags> flags = {}, bool push = false);
    struct SetAllocation {
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::size_t poolIndex = 0;
        bool updateAfterBind = false;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        std::vector<VkDescriptorPoolSize> sizes;
        std::optional<std::uint32_t> variableCount;
        std::shared_ptr<const TableDescriptors> table;
    };
    SetAllocation Allocate(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind = false, std::optional<std::uint32_t> variableCount = std::nullopt);
    void Free(const SetAllocation& allocation, std::shared_ptr<const TableDescriptors> table = {}) noexcept;
    SetAllocation AllocateRewritten(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind = false, std::optional<std::uint32_t> variableCount = std::nullopt);
    void Recycle(VkDescriptorSetLayout layout, const SetAllocation& allocation) noexcept;
    // APS5_PROFILE_DRAW counters: layouts served from the map / created, sets allocated, pools opened.
    struct Stats {
        std::uint64_t layoutHits = 0;
        std::uint64_t layoutMisses = 0;
        std::uint64_t sets = 0;
        std::uint64_t pools = 0;
        std::uint64_t dedicatedPools = 0;
        std::uint64_t keptSets = 0;
        std::uint64_t reusedSets = 0;
    };
    Stats Counters() const;

private:
    Context context;
    PFN_vkDestroyDescriptorSetLayout destroyLayout;
    PFN_vkDestroyDescriptorPool destroyPool;
    PFN_vkResetDescriptorPool resetPool;
    PFN_vkFreeDescriptorSets freeDescriptorSets = nullptr;
    mutable AgcDriver::Mutex mutex;
    std::map<std::vector<std::uint32_t>, VkDescriptorSetLayout> layouts;
    struct Pool {
        VkDescriptorPool handle = VK_NULL_HANDLE;
        std::size_t liveSets = 0;
        bool exhausted = false;
    };
    std::vector<Pool> pools;
    std::vector<Pool> updateAfterBindPools;
    std::vector<VkDescriptorPool> dedicated;
    std::map<std::pair<VkDescriptorSetLayout, std::uint32_t>, std::vector<SetAllocation>> recycled;
    std::vector<SetAllocation> kept;
    Stats stats;
};

enum class ContentKeyComponent : std::size_t { Layout, BufferWord0, BufferWord1, BufferWord2, BufferWord3, Image, Sampler, ShaderData, Other, DrawTarget, Count };

using ContentKeyDiff = std::array<bool, static_cast<std::size_t>(ContentKeyComponent::Count)>;

struct ImageTableReuseCounts {
    std::uint64_t reused = 0;
    std::uint64_t resolved = 0;
    std::uint64_t replayed = 0;
    std::uint64_t writesKept = 0;
    std::uint64_t writesMade = 0;
    std::uint64_t variantSurfaces = 0;
    std::uint64_t variantSubstituted = 0;
    std::uint64_t variantRejected = 0;
    std::uint64_t arrayViewsShared = 0;
};
ImageTableReuseCounts ImageTableReuses();
std::uint64_t SampledTexturesCleared();
std::uint64_t SampledTexturesImported();
std::uint64_t SampledLookupKeyScans();
std::uint64_t SampledLookupKeyProofs();

struct PresyncSurface {
    std::uint64_t address;
    std::uint64_t bytes;
    bool dccKeys;
};

struct CommitterTrustState {
    std::uint32_t untilAudit = 0;
    std::uint32_t volatileUntil = 0;
};

enum class CommitterTrustFamily : std::uint8_t { Template, Buffers, Inputs, Count };
enum class CommitterTrustSite : std::uint8_t { FollowerTemplate, LeaderTemplate, Proof, FollowerBuffers, LeaderBuffers, RecordBuffers, FollowerInputs, LeaderInputs, RecordInputs, Count };
enum class CommitterTrustUse : std::uint8_t { Today, Trusted, Audit, Volatile };

inline std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(CommitterTrustSite::Count)> committerTrustTrusted{};

CommitterTrustUse CommitterTrustSlow(CommitterTrustState& state, CommitterTrustSite site);
void CommitterTrustFailed(CommitterTrustState& state, CommitterTrustSite site, CommitterTrustUse use);

inline CommitterTrustUse CommitterTrustDecide(CommitterTrustState& state, CommitterTrustSite site) {
    if (state.untilAudit > 1) {
        --state.untilAudit;
        auto& trusted = committerTrustTrusted[static_cast<std::size_t>(site)];
        trusted.store(trusted.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        return CommitterTrustUse::Trusted;
    }
    return CommitterTrustSlow(state, site);
}

struct CommitterTrustCheck {
    CommitterTrustState* state = nullptr;
    CommitterTrustSite site = CommitterTrustSite::Count;
    CommitterTrustUse use = CommitterTrustUse::Today;
    bool failed = false;
    bool Armed() const { return state != nullptr; }
    bool Skip() {
        if (state != nullptr && use == CommitterTrustUse::Today) use = CommitterTrustDecide(*state, site);
        return use == CommitterTrustUse::Trusted;
    }
    bool Audits() const { return use == CommitterTrustUse::Audit || use == CommitterTrustUse::Volatile; }
    void Fail() {
        if (!Audits() || failed) return;
        failed = true;
        CommitterTrustFailed(*state, site, use);
    }
};

class ShaderResources {
public:
    ShaderResources(const Context& context, const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& fragment, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes);
    ShaderResources(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes, std::span<const GuestMemorySnapshot> snapshots = {}, bool deviceStaging = false);
    ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots = {});
    // Two-stage build for dispatches (see build): with `deferred` the constructor runs stage A only,
    // which needs no device lock, and Complete() runs stage B under GuestMemory::GpuMutex; until
    // then no other member may be used. `compute` and `snapshots` must outlive Complete(). An
    // address-based shader (BDA tables) builds entirely in Complete(): its lease acquisition
    // reconciles imports and refreshes mirrors, which needs the lock.
    ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots, bool deferred, std::uint64_t dispatchThreads = 0, std::optional<std::pair<std::uint64_t, std::size_t>> htileClear = std::nullopt);
    void Complete();
    bool Completed() const { return completed; }
    ~ShaderResources();
    ShaderResources(const ShaderResources&) = delete;
    ShaderResources& operator=(const ShaderResources&) = delete;
    VkDescriptorSetLayout Layout() const;
    std::vector<VkDescriptorSetLayoutBinding> LayoutBindings() const;
    bool LayoutUpdateAfterBind() const;
    std::span<const VkDescriptorBindingFlags> LayoutBindingFlags() const { return layoutBindingFlags; }
    VkDescriptorSetLayout FragmentLayout() const { return fragment.layout; }
    bool SplitSets() const { return splitSets; }
    PipelineSetLayouts SetLayouts() const;
    bool HeapStage() const { return heapStage; }
    std::uint64_t CoverageSerial() const { return coverageSerial; }
    void NoteCoverageSerial(std::uint64_t serial) const { coverageSerial = serial; }
    void RefreshHeapStage(std::span<const CompiledShader> shaders);
    void RequireGuestBuffers() const { requireGuestBuffers(); }
    void RequireGuestBuffersAudited(CommitterTrustCheck& trust) const;
    void Bind(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout) const;
    void BindDescriptorHeaps(VkCommandBuffer commands) const;
    struct DrawBindings {
        DrawBindings() = default;
        DrawBindings(const DrawBindings&) = delete;
        DrawBindings& operator=(const DrawBindings&) = delete;
        struct Snapshot {
            std::uint64_t address;
            std::shared_ptr<Buffer> buffer;
            VkDeviceSize offset = 0;
            std::size_t bytes = 0;
            VkBuffer handle = VK_NULL_HANDLE;
            std::span<std::byte> Bytes() const { return buffer->Bytes().subspan(static_cast<std::size_t>(offset), bytes); }
            VkBuffer Handle() const { return buffer != nullptr ? buffer->Handle() : handle; }
        };
        DescriptorCache* cache = nullptr;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout fragmentLayout = VK_NULL_HANDLE;
        DescriptorCache::SetAllocation fragmentAllocation;
        DescriptorCache::SetAllocation allocation;
        std::vector<Snapshot> snapshots;
        bool addressOffsets = false;
        std::array<std::uint32_t, 2u * ShaderRecompiler::RuntimeAbi::StageCount> heapOffsets{};
        std::vector<VkDescriptorBufferInfo> addressOverrides;
        std::shared_ptr<OwnDrawSets> ownSets;
        std::array<bool, 2> owned{};
        std::int32_t pushedSet = -1;
        std::vector<std::size_t> pushSelected;
        std::array<bool, 2> lent{};
        std::array<std::shared_ptr<const DrawBindings>, 2> lenders;
        struct DynamicBound {
            std::array<std::shared_ptr<const void>, 2> sets;
            std::array<std::uint32_t, 2> counts{};
            std::array<std::uint32_t, 32> offsets{};
        };
        std::unique_ptr<DynamicBound> dynamic;
        std::uint32_t DynamicCount(std::size_t set) const { return dynamic != nullptr ? dynamic->counts[set] : 0u; }
        const std::uint32_t* DynamicOffsets(std::size_t set) const { return dynamic != nullptr ? dynamic->offsets.data() + (set == 0 ? 0u : dynamic->counts[0]) : nullptr; }
        ~DrawBindings();
        std::shared_ptr<OwnDrawSets> parkedSets;
        std::array<bool, 2> parkable{};
        std::array<std::uint64_t, 2> parkStamps{};
    };
    struct PushPlan {
        std::uint32_t set = 0;
        std::uint32_t descriptors = 0;
        PipelineSetLayouts layouts;
        std::vector<std::uint32_t> mainKey;
        std::vector<std::uint32_t> fragmentKey;
        std::vector<std::uint8_t> allocationSets;
        std::vector<std::vector<std::size_t>> slots;
        std::vector<std::size_t> writes;
    };
    void RefreshMovedAddresses(DrawBindings& rewritten) const;
    struct MovedBuffer {
        std::size_t allocation;
        std::uint64_t address;
        std::size_t size;
        std::span<const std::uint32_t> words;
        VkBuffer inPlace = VK_NULL_HANDLE;
        VkDeviceSize inPlaceOffset = 0;
        std::uint32_t adjustment = 0;
        VkDeviceSize importOffset = 0;
        std::uint64_t begin = 0;
        std::size_t bytes = 0;
        bool behind = false;
        bool written = false;
    };
    enum class MovedCount : std::uint8_t { Shape, Patched, NotGuest, Written, Oversize, Budget, Unmapped, TemplateWrites, PendingWrite, PendingStorage, Shadowed, Resized, ResizedClamped, PreparedResized, PreparedResizedClamped, InPlaceRanges, InPlaceHits, NotEligible, InPlaceStorage, InPlaceShadowed, CpuPublish, Label, QueuedStore, CopiedWriter, OverRange, NoImport, Misaligned, PreparedInPlaceHits, PreparedInPlaceRanges, PreparedInPlaceSerials, PreparedInPlaceSwitch, PreparedInPlaceWrites, Count };
    static const char* MovedCountName(MovedCount count);
    static constexpr std::size_t MovedCountArms = 4;
    static std::array<std::uint64_t, static_cast<std::size_t>(MovedCount::Count)> TakeMovedCounts(std::size_t arm);
    enum class InPlaceUser : std::uint8_t { DrawMoved, DispatchWritten, DrawPrepared, Count };
    MovedCount InPlaceRefusal(const Recorder& recorder, std::uint64_t begin, std::size_t bytes, InPlaceUser user = InPlaceUser::Count) const;
    void BindFragmentOwnSets(VkCommandBuffer commands, VkPipelineLayout layout) const;
    // What MovedReadOnlyBuffers and selectDrawSnapshots read from this object to decide a draw's
    // buffers, copied by value once so a finisher thread can make that decision from the plan, the
    // draw's stages and guest memory alone (PrepareDirectBuffers). Immutable once built; it points
    // into nothing, so it may outlive the object.
    struct BufferPlan {
        // One buffer binding element, in the order MovedReadOnlyBuffers walks them, with the shape
        // the draw's binding must still have (stage, index in the program, slot, role, count, and
        // whether the shader writes the element).
        struct Element {
            std::uint32_t stage = 0;
            std::uint32_t binding = 0;
            std::uint32_t slot = 0;
            std::uint32_t count = 0;
            std::uint32_t element = 0;
            ShaderRecompiler::DescriptorRole role = ShaderRecompiler::DescriptorRole::GuestBuffers;
            bool written = false;
            std::size_t allocation = 0;
            // Data roles: the words the allocation's buffer holds.
            std::vector<std::uint32_t> dataWords;
        };
        struct Allocation {
            std::uint64_t address = 0;
            std::size_t size = 0;
            std::uint32_t adjustment = 0;
            bool guest = false;
            bool written = false;
            // The range lies inside one of the object's in-place reads.
            bool inPlace = false;
        };
        struct Patch {
            std::size_t allocation = 0;
            std::uint32_t byte = 0;
            std::uint32_t adjustment = 0;
        };
        std::uint32_t stages = 0;
        std::vector<Element> elements;
        std::vector<Allocation> allocations;
        std::vector<Patch> patches;
        std::uint64_t maxStorageBufferRange = 0;
        std::size_t maxDrawSnapshotBytes = 0;
        std::size_t uploadLimit = 0;
        std::size_t movedUploadBytes = 0;
        std::size_t alignment = 16;
    };
    // Built on the first call from the stages drawn with, on the committer under the GPU mutex,
    // and never rebuilt; null when no draw's buffers can be decided from a plan (no set, a heap
    // stage, an address table, an object that is not reusable, or bindings MovedReadOnlyBuffers
    // refuses whatever the draw).
    const std::shared_ptr<const BufferPlan>& DirectBufferPlan(std::span<const CompiledShader> shaders) const;
    // A draw's buffers as MovedReadOnlyBuffers and selectDrawSnapshots (movedOnly false) decide
    // them, without their hazard tests: the moved list (its words point into the draw's stages),
    // the selected allocations in ascending order with their guest ranges (moved data words: begin
    // 0) and offsets into `blob`, which holds every selected byte, each slice aligned as
    // Recorder::DrawUpload aligns its slices. `plan` is the plan it was prepared from.
    struct PreparedBuffers {
        enum class Refusal : std::uint8_t { None, Classification, Inaccessible, UploadLimit, Budget, NotGuest, Written, Oversize, Count };
        // Whether the committer's selection took them, or today's selection ran instead: an
        // unmoved range no longer clear (or slices the device would not align), or a depth guard
        // that may have published samples into a selected range.
        enum class Selection : std::uint8_t { Taken, Hazard, DepthGuards };
        struct Selected {
            std::size_t allocation = 0;
            std::uint64_t begin = 0;
            std::size_t bytes = 0;
            std::size_t offset = 0;
            bool moved = false;
            bool inPlace = false;
            VkBuffer handle = VK_NULL_HANDLE;
            VkDeviceSize importOffset = 0;
        };
        std::shared_ptr<const BufferPlan> plan;
        std::vector<MovedBuffer> moved;
        std::vector<Selected> selected;
        std::vector<std::byte> blob;
        Refusal refusal = Refusal::None;
        // The tracker generation taken before the guest bytes were read.
        std::uint64_t generation = 0;
        // The depth guard generation taken before the guest bytes were read: a guard set up or
        // dropped since may have published depth samples into them.
        std::uint64_t depthGuards = 0;
        // GuestMemory::ForgetSerial taken before the moved ranges were found accessible.
        std::uint64_t forget = 0;
        std::uint64_t imports = 0;
        std::uint64_t mapping = 0;
        bool inPlace = false;
        mutable std::shared_ptr<DrawBindings> placed;
        VkDevice placedDevice = VK_NULL_HANDLE;
        bool aliasesClear = false;
        static constexpr std::uint8_t DeferredImports = 1;
        static constexpr std::uint8_t MergedSettle = 2;
        static constexpr std::uint8_t ExactCapacity = 4;
        std::uint8_t shortcuts = 0;
        std::uint32_t lookupsDeferred = 0;
        std::uint32_t lookupsForced = 0;
        struct ZoneCard {
            static constexpr std::size_t Capacity = 32;
            static constexpr std::uint64_t Moved = std::uint64_t{1} << 63u;
            const BufferPlan* plan = nullptr;
            std::uint8_t zones = 0;
            std::uint8_t guards = 0;
            bool valid = false;
            std::array<std::pair<std::uint64_t, std::uint64_t>, Capacity> entries{};
        };
        ZoneCard card;
        GuardAnswer guardAnswer;
    };
    enum class ZoneCardBuild : std::uint8_t { Made, OverCapacity, Refused };
    static ZoneCardBuild BuildZoneCard(PreparedBuffers& prepared);
    void PrefetchExpressHot() const;
    enum class BindingTaken : std::uint8_t { None, Placed, Copied, OtherDevice, Count };
    enum class DynamicLeave : std::uint8_t { None, Refused, Prepared, Stages, HeapPipeline, Heap, Address, AfterBind, Addressed, Faults, Shape, Uniform, OtherType, Flags, NoStorage, Limit, PushHeld, OtherSet, First, Repeats, Pipeline, Rebound, Count };
    struct BindingRoute {
        bool placed = false;
        bool stamps = false;
        bool sets = false;
        bool verifyStamps = false;
        std::uint32_t setsLent = 0;
        std::uint32_t setsWritten = 0;
        std::uint32_t guards = 0;
        const PreparedBuffers* selection = nullptr;
        bool dynamic = false;
        bool pushDynamic = false;
        std::uint32_t dynamicHits = 0;
        std::uint32_t dynamicBound = 0;
        std::uint32_t dynamicWritten = 0;
        std::uint32_t dynamicRefused = 0;
        DynamicLeave plainLeave = DynamicLeave::None;
        DynamicLeave pushedLeave = DynamicLeave::None;
        DynamicLeave leave = DynamicLeave::None;
        BindingTaken taken = BindingTaken::None;
        std::uint32_t tested = 0;
        std::uint32_t stamped = 0;
        std::uint32_t refuted = 0;
        std::size_t bytes = 0;
    };
    static bool PlaceDirectBuffers(const Context& context, PreparedBuffers& prepared);
    static std::size_t PlacedUploadChunks();
    static void SetUploadChunkBytesForTest(std::size_t bytes);
    struct UploadRingCounts {
        std::uint64_t reused = 0;
        std::uint64_t added = 0;
        std::uint64_t overflow = 0;
        std::uint64_t chunks = 0;
        std::uint64_t inFlight = 0;
    };
    static UploadRingCounts UploadRingTotals();
    // The first difference between prepared buffers and today's way of the same draw, compared as
    // data (APS5_VERIFY_DIRECT_BUFFERS): `what` names the part and stays null where they agree;
    // `memoryChanged`: only bytes differ, in a guest range written after the finisher read it (a
    // bound hit would draw the older bytes); `unwatched`: only bytes differ, in a guest range the
    // write tracker does not watch, so a write after the read and a fault look the same.
    struct PreparedDifference {
        const char* what = nullptr;
        std::string detail;
        bool memoryChanged = false;
        bool unwatched = false;
    };
    // Against today's MovedReadOnlyBuffers list: allocations, addresses, sizes and words.
    static PreparedDifference ComparePreparedMoves(const PreparedBuffers& prepared, std::span<const MovedBuffer> moved);
    // Against today's selection and the snapshots `bindings` holds for it: allocations in order,
    // guest ranges, then bytes.
    static PreparedDifference ComparePreparedSelection(const PreparedBuffers& prepared, const DrawBindings* bindings, std::span<const std::size_t> selected);
    // The selection the committer binds for `prepared` (one upload slice, built and dropped here)
    // against today's selection and the snapshots `bindings` holds for it: allocations in order,
    // guest ranges, aligned offsets, and the slice's bytes against the prepared blob. No difference
    // where the committer would take today's selection instead.
    PreparedDifference ComparePreparedBinding(Recorder& recorder, std::span<const MovedBuffer> moved, const PreparedBuffers& prepared, const DrawBindings* bindings, std::span<const std::size_t> selected) const;
    // Prepares the buffers of a draw with `shaders` from the plan, its stages' words and guest
    // memory alone, on any thread; `target` is the colour target MovedReadOnlyBuffers would get.
    // False, with the reason in `out.refusal`, where the draw's buffers cannot be prepared.
    static bool PrepareDirectBuffers(const BufferPlan& plan, std::span<const CompiledShader> shaders, const ColorTarget* target, PreparedBuffers& out, const Context* inPlace = nullptr);
    static bool TakePreparedInPlace(PreparedBuffers& prepared, std::size_t index);
    static bool FindPreparedImport(const Context& context, const BufferPlan& plan, PreparedBuffers& prepared, std::size_t index);
    const PushPlan* PushLayouts() const;
    struct DynamicPlan {
        std::array<bool, 2> written{};
        std::array<bool, 2> dynamicSet{};
        std::array<VkDescriptorSetLayout, 2> layouts{};
        std::array<std::vector<VkDescriptorPoolSize>, 2> sizes;
        std::array<std::optional<std::uint32_t>, 2> variable;
        std::vector<std::uint32_t> mainKey;
        std::vector<std::uint32_t> fragmentKey;
        PipelineSetLayouts pipelineLayouts;
        std::vector<std::uint8_t> writeSets;
        std::vector<VkDescriptorType> writeTypes;
        std::array<std::vector<std::size_t>, 2> dynamicSlots;
        std::vector<std::size_t> slotAllocations;
        std::size_t storageDynamic = 0;
        std::array<std::size_t, 2> setDynamic{};
        bool plainFits = false;
        bool pushedBuilt = false;
        bool pushedFits = false;
        std::vector<std::uint32_t> pushedMainKey;
        std::vector<std::uint32_t> pushedFragmentKey;
        PipelineSetLayouts pushedLayouts;
    };
    const DynamicPlan* DynamicLayouts() const;
    const PipelineSetLayouts* DynamicPushLayouts() const;
    DynamicLeave DynamicLeaveFor(bool pushed) const;
    static void SetDynamicLimitForTest(std::optional<std::uint32_t> limit);
    static std::uint64_t DynamicPlansRefused();
    // `prepared` (buffers prepared from this object's plan, with `moved` their moved list) takes
    // the place of today's snapshot selection while its unmoved ranges are still clear;
    // `preparedSelection` tells whether it did.
    std::shared_ptr<DrawBindings> PreparePushedDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved, const PushPlan& plan, std::vector<std::size_t>* selectedOut = nullptr, const PreparedBuffers* prepared = nullptr, PreparedBuffers::Selection* preparedSelection = nullptr, BindingRoute* route = nullptr) const;
    void PushDrawBindings(VkCommandBuffer commands, VkPipelineLayout layout, const DrawBindings& bindings, const PushPlan& plan) const;
    bool FragmentEligible() const { return reusable && !heapStage && (bda == nullptr || FaultBufferLean()) && !addressedBuffers && !rebound && context.descriptorCache != nullptr; }
    bool FaultBufferLean() const;
    bool TakeFaultCheck(std::uint64_t recorder, std::uint64_t batch) const;
    std::uint64_t FaultGeneration() const { return faultGeneration.load(std::memory_order_acquire); }
    enum class ExpressMiss : std::uint8_t { None, NotProved, Watch, Forget, Serials, Sources, Views, PendingCpu, Staging, Regions, Count };
    bool ExpressReady() const { return expressValid && cleanProved; }
    bool ExpressWatchable() const;
    ExpressMiss ExpressCurrent();
    ExpressMiss ExpressCurrentFollowing();
    ExpressMiss ExpressCurrentTrusted() const;
    ExpressMiss ExpressCurrentKept() const;
    ExpressMiss ExpressCurrentAudited(CommitterTrustCheck& check, bool following);
    CommitterTrustCheck TrustCheck(CommitterTrustFamily family, CommitterTrustSite site) const { return {&committerTrust[static_cast<std::size_t>(family)], site}; }
    void ExpressStage(Recorder& recorder);
    PreparedBuffers::Selection FragmentSelectionHolds(Recorder& recorder, std::span<const MovedBuffer> moved, const PreparedBuffers& prepared, BindingRoute* route, bool untouched = false, CommitterTrustCheck* trust = nullptr, std::uint8_t answered = 0) const;
    struct GuardTests {
        bool (*depth)(void* tests, std::uint64_t address, std::uint64_t bytes) = nullptr;
        bool (*multisample)(void* tests, std::uint64_t address, std::uint64_t bytes) = nullptr;
        void* tests = nullptr;
        bool (*writes)(void* tests, std::uint64_t address, std::uint64_t bytes) = nullptr;
    };
    PreparedBuffers::Selection FragmentSelectionHoldsBy(Recorder& recorder, std::span<const MovedBuffer> moved, const PreparedBuffers& prepared, bool untouched, const GuardTests* guards, CommitterTrustCheck* trust = nullptr) const;
    void PushFragmentBindings(VkCommandBuffer commands, VkPipelineLayout layout, const DrawBindings& bindings, const PushPlan& plan) const;
    void WriteFragmentSets(DrawBindings& result, std::span<const std::size_t> selected, std::int32_t pushedSet) const;
    struct FragmentPlanNote {
        std::uint32_t built = 0;
        std::uint32_t hit = 0;
        std::uint32_t pushed = 0;
        std::uint32_t checked = 0;
        std::uint32_t differing = 0;
    };
    static FragmentPlanNote TakeFragmentPlanNote();
    struct ParkedSetsNote {
        std::uint32_t taken = 0;
        std::uint32_t fresh = 0;
        std::uint32_t today = 0;
        std::uint32_t parkedBuffers = 0;
        std::uint32_t parkedOthers = 0;
        std::uint32_t freshBuffers = 0;
        std::uint32_t freshOthers = 0;
        std::uint32_t checked = 0;
        std::uint32_t differing = 0;
    };
    static ParkedSetsNote TakeParkedSetsNote();
    struct ParkedSetsCounts {
        std::uint64_t parked = 0;
        std::uint64_t refusedCapacity = 0;
        std::uint64_t refusedCap = 0;
        std::uint64_t refusedClosed = 0;
    };
    static ParkedSetsCounts TakeParkedSetsCounts();
    void RewriteFragmentFaults(const DrawBindings& placed) const;
    DescriptorCache* FragmentDescriptorCache() const { return context.descriptorCache; }
    struct RepeatPush {
        RepeatPush() = default;
        RepeatPush(const RepeatPush&) = delete;
        RepeatPush& operator=(const RepeatPush&) = delete;
        std::vector<VkDescriptorBufferInfo> buffers;
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<std::pair<std::size_t, std::size_t>> patches;
        std::vector<std::size_t> allocations;
        VkDescriptorSet boundSet = VK_NULL_HANDLE;
        std::uint32_t boundIndex = 0;
        std::uint32_t pushIndex = 0;
        RepeatSets* sets = nullptr;
        std::vector<std::size_t> outsideAt;
        RepeatSetLease* lease = nullptr;
        std::uint64_t leaseRecorder = 0;
        std::uint64_t leaseBatch = 0;
    };
    bool BuildRepeatPush(const PushPlan& plan, std::span<const PreparedBuffers::Selected> selected, RepeatPush& out, bool sets = false) const;
    void RepeatPushBindings(VkCommandBuffer commands, VkPipelineLayout layout, RepeatPush& push, std::span<const PreparedBuffers::Selected> selected, VkBuffer buffer, VkDeviceSize base, Recorder& recorder) const;
    PreparedDifference CompareRepeatPush(RepeatPush& push, const DrawBindings& bound, const PushPlan& plan) const;
    PreparedBuffers::Selection RepeatGuards(const PreparedBuffers& prepared) const;
    void RepeatRequire(std::span<const MovedBuffer> moved) const;
    bool RepeatUnmovedClear(const Recorder& recorder, const PreparedBuffers& prepared) const;
    std::shared_ptr<Buffer> RepeatUpload(Recorder& recorder, const PreparedBuffers& prepared, VkDeviceSize& offset) const;
    bool ProofHeld() const;
    bool HoldsDepthCopies() const;
    struct BoundBuffer {
        std::uint32_t set = 0;
        std::uint32_t binding = 0;
        std::uint32_t element = 0;
        VkDescriptorBufferInfo info{};
        bool snapshot = false;
        std::uint64_t address = 0;
        std::size_t bytes = 0;
        std::uint64_t content = 0;
    };
    void DescribeDrawBindings(const DrawBindings* bound, std::span<const std::size_t> selected, const PushPlan* plan, std::vector<BoundBuffer>& out, bool contents = false) const;
    void DescribeDynamicBindings(const DrawBindings& bound, std::span<const std::size_t> selected, const PushPlan* plan, std::vector<BoundBuffer>& out) const;
    void DescribeMovedInPlace(std::span<const MovedBuffer> moved, std::vector<BoundBuffer>& out) const;
    bool RecordLean() const;
    std::uint8_t LeanMiss() const;
    struct DrawUse {
        DrawUse() = default;
        DrawUse(const DrawUse&) = delete;
        DrawUse& operator=(const DrawUse&) = delete;
        ~DrawUse();
        std::shared_ptr<BdaResources> record;
        std::shared_ptr<DrawBindings> bindings;
        std::shared_ptr<std::atomic<bool>> ownRecord;
    };
    std::shared_ptr<DrawUse> BeginDrawUse(Recorder& recorder) const;
    void CompleteDrawUse(const DrawUse& use) const;
    void BindDrawUse(VkCommandBuffer commands, VkPipelineLayout layout, const DrawUse* use) const;
    // `prepared` and `preparedSelection` as for PreparePushedDrawBindings (not with `movedOnly`).
    std::shared_ptr<DrawBindings> PrepareDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved = {}, bool movedOnly = false, const BdaResources* faultRecord = nullptr, bool census = false, std::vector<std::size_t>* selectedOut = nullptr, const PreparedBuffers* prepared = nullptr, PreparedBuffers::Selection* preparedSelection = nullptr, BindingRoute* route = nullptr) const;
    ReplayReason HeapReplayRefusal(std::span<const CompiledShader> shaders) const;
    void NoteHeapReplayCensus();
    struct HeapReplayProof {
        std::uint64_t epoch = 0;
        bool overlay = false;
    };
    HeapReplayProof ReplayHeapStages(std::span<const CompiledShader> shaders, bool overlay);
    std::shared_ptr<DrawBindings> PrepareReplayDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved, std::span<const CompiledShader> shaders, const BdaResources* faultRecord, HeapReplayProof& proof, bool census = false, std::span<const std::uint32_t> contentKey = {});
    void ConfirmHeapReplay(Recorder& recorder, std::span<const CompiledShader> shaders, const std::shared_ptr<DrawBindings>& bindings, const HeapReplayProof& proof);
    std::uint64_t HeapStageWrites() const { return heapStageWrites; }
    struct MoveTicket {
        bool valid = false;
        std::uint64_t recorder = 0;
        std::uint64_t note = 0;
        std::uint64_t pending = 0;
        std::uint64_t allocations = 0;
        std::uint64_t epoch = 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>>* ranges = nullptr;
        std::size_t checked = 0;
        std::size_t trusted = 0;
    };
    std::optional<std::vector<MovedBuffer>> MovedReadOnlyBuffers(std::span<const CompiledShader> shaders, Recorder& recorder, const ColorTarget* target, bool* patchedData = nullptr, MoveTicket* ticket = nullptr) const;
    // The tests MovedReadOnlyBuffers applies to each moved guest range after its classification,
    // with the same ticket use, for the moved list of buffers prepared from this object's plan
    // (their memory was found accessible when they were prepared; it is tested again once a mapping
    // changed since). True when every range is clear; the ticket is then left as
    // MovedReadOnlyBuffers would leave it.
    bool PreparedMovesClear(const PreparedBuffers& prepared, Recorder& recorder, MoveTicket* ticket) const;
    bool PreparedUntouched(const PreparedBuffers& prepared) const;
    bool PreparedZonesClear(const PreparedBuffers& prepared, const Recorder& recorder) const;
    using ZoneClear = bool (*)(void* tests, std::uint64_t begin, std::size_t bytes, bool moved);
    bool PreparedZonesClearBy(const PreparedBuffers& prepared, ZoneClear clear, void* tests) const;
    bool PreparedZonesClearByWindows(const PreparedBuffers& prepared, const Recorder& recorder) const;
    std::optional<bool> PreparedMovesClearByWindows(const PreparedBuffers& prepared, Recorder& recorder, MoveTicket* ticket, WriteValidation::Test test) const;
    bool PreparedInPlaceHolds(const PreparedBuffers& prepared, const Recorder& recorder, bool& behind, CommitterTrustCheck* trust = nullptr, std::optional<WriteValidation::Test> validation = std::nullopt) const;
    bool StoresArmHolds(const Recorder& recorder) const;
    void ArmStores(const Recorder& recorder, std::uint64_t generationBefore) const;
    static void NotePreparedInPlaceTaken(const PreparedBuffers& prepared);
    static bool PreparedInPlaceOverlaps(const PreparedBuffers& prepared, std::uint64_t begin, std::uint64_t end);
    bool UnmovedRangesClear(const PreparedBuffers& prepared, Recorder& recorder) const;
    // The ticket PreparedMovesClear left (null: it refused the ranges) against the one today's
    // MovedReadOnlyBuffers left from the same ticket.
    static PreparedDifference ComparePreparedTickets(const MoveTicket* prepared, const MoveTicket& today);
    static std::uint64_t BytesUpToTarget(std::uint64_t address, std::uint64_t bytes, std::uint64_t stride, const ColorTarget* target);
    enum class MoveRefusal : std::size_t { None, Shape, NotGuest, Written, Limit, Inaccessible, Overlap, Bindings, WrittenTemplate, WrittenInPlace, WrittenAlias, Count };
    struct MoveProof {
        MoveRefusal refusal = MoveRefusal::None;
        std::size_t buffers = 0;
        std::size_t bytes = 0;
        std::size_t written = 0;
        std::uint8_t overlap = 0;
        MovedCount inPlace = MovedCount::Count;
        bool guarded = false;
        std::size_t readsInPlace = 0;
    };
    static const char* MoveRefusalName(MoveRefusal refusal);
    std::optional<std::vector<MovedBuffer>> MovedDispatchBuffers(const CompiledShader& shader, Recorder* recorder, std::size_t maxBytes, MoveProof& proof, bool writes = false, bool readsInPlace = false) const;
    bool MovableWrites() const;
    std::size_t MovableWritesRefusal() const;
    static const char* MovableWritesRefusalName(std::size_t refusal);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> MovedGpuWrites(std::span<const MovedBuffer> moved) const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> MovedInPlaceReads(std::span<const MovedBuffer> moved) const;
    std::vector<std::string> VerifyMovedWrites(const CompiledShader& shader, std::span<const MovedBuffer> moved, const HostImportsProof* recordedImports = nullptr, std::size_t* incomparable = nullptr) const;
    std::vector<std::string> VerifyInPlaceReads(std::span<const MovedBuffer> moved, const Recorder& recorder, const HostImportsProof* recordedImports = nullptr, std::size_t* incomparable = nullptr) const;
    enum class NotReusable : std::uint8_t { None, HtileClear, CopiedWrites, LeaseOrTable, MultisampleSources, PerUseTextures, DataBuffers, NoImportSerial, Count };
    NotReusable NotReusableReason() const { return notReusable; }
    static std::size_t MaxDrawSnapshotBytes();
    static void RereadMaxDrawSnapshotBytesForTest();
    static std::size_t DrawUploadLimit();
    struct MovedSnapshotCounts {
        std::uint64_t cached = 0;
        std::uint64_t reused = 0;
        std::uint64_t cachedBytes = 0;
        std::uint64_t uploaded = 0;
        std::uint64_t uploadedBytes = 0;
    };
    static MovedSnapshotCounts TakeMovedSnapshotCounts();
    void WriteBack();
    // Deferred completion: MarkGpuWrites registers the results the recorded work leaves on the GPU
    // (storage images stay there; buffer ranges are noted so CPU reads wait); WriteBackBuffers runs
    // once the work completed.
    void MarkGpuWrites(Recorder& recorder);
    void MarkGpuWrites(Recorder& recorder, std::span<const MovedBuffer> moved);
    void WriteBackBuffers();
    // Whether WriteBackBuffers has anything the CPU must see (copied written buffers, BDA faults).
    bool NeedsCompletion() const { return (bda != nullptr && faultChecks) || guestMemory.HasCopiedWrites(); }
    // Whether a written buffer was copied (its results reach guest memory by the CPU write-back).
    bool HasCopiedWrites() const { return guestMemory.HasCopiedWrites(); }
    bool HasGpuCopies() const { return guestMemory.HasGpuCopies(); }
    bool HasHostGpuCopies() const { return guestMemory.HasHostGpuCopies(); }
    bool HasStagedCopies() const { return guestMemory.HasStagedCopies(); }
    const void* MemoryOwner() const { return &guestMemory; }
    void CopyBacksAfterRun(std::shared_ptr<ShaderResources> self) { afterRunSelf = std::move(self); }
    void RunAccesses(std::vector<std::array<std::uint64_t, 3>>& out) const;
    bool MovedRunAccesses(std::span<const MovedBuffer> moved, std::vector<std::array<std::uint64_t, 3>>& out) const;
    std::vector<std::string> VerifyMovedRunAccesses(const CompiledShader& shader, std::span<const std::array<std::uint64_t, 3>> accesses) const;
    bool HoldsLease() const { return guestMemory.HoldsLease(); }
    bool SpaceReuse() const { return spaceReuse; }
    static bool SpaceReuseDisabled();
    bool BdaReadOnly() const { return usesBda && bdaReadOnly; }
    bool RetakeSpace() { return guestMemory.RetakeSpace(); }
    bool SpaceHeld() const { return guestMemory.SpaceHeld(); }
    SnapshotService SnapshotsServedInPlace(std::span<const GuestMemorySnapshot> snapshots) const { return guestMemory.SnapshotsServedInPlace(snapshots); }
    std::shared_ptr<const void> ReleaseSpace() { return guestMemory.ReleaseSpace(); }
    void HoldSpaceThrough(std::uint64_t recorder, std::uint64_t serial) { guestMemory.HoldSpaceThrough(recorder, serial); }
    bool ReleaseCompletedSpace(std::uint64_t recorder, std::uint64_t completedThrough) { return guestMemory.ReleaseCompletedSpace(recorder, completedThrough); }
    bool UsesAddressTable() const { return usesBda; }
    std::span<const std::byte> AddressTableBytes() const;
    bool AddressTableGuardCurrent();
    std::size_t CopiedBytes() const { return guestMemory.CopiedBytes(); }
    bool WritesOverlap(std::uint64_t address, std::size_t bytes) const { return guestMemory.WritesOverlap(address, bytes); }
    bool CopiedWritesOverlap(std::uint64_t address, std::size_t bytes) const { return guestMemory.CopiedWritesOverlap(address, bytes); }
    // Whether a region the recorded work reads in place through a host import overlaps the range.
    bool ReadsOverlap(std::uint64_t address, std::size_t bytes) const;
    // The recorder's hazard tracker inputs (Recorder::NoteAccess): the written elements' guest
    // ranges, the regions read in place, and the storage images with whether each is written.
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& GpuWrites() const { return guestMemory.Writes(); }
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& WriteOnlyRanges() const { return writeOnlyRanges; }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> InPlaceReads() const { return guestMemory.InPlaceReads(); }
    void InPlaceReads(std::vector<std::pair<std::uint64_t, std::uint64_t>>& out) const { guestMemory.InPlaceReads(out); }
    std::vector<std::pair<VkImage, bool>> StorageImages() const;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> DeviceReads() const { return guestMemory.DeviceReads(); }
    void DeviceReads(std::vector<std::pair<std::uint64_t, std::uint64_t>>& out) const { guestMemory.DeviceReads(out); }
    bool ReadsImage(const StorageTexture* image) const;
    bool HasCachedStorageImages() const;
    bool HoldsReleasedTexture() const;
    bool HoldsDiscardedTexture() const;
    bool KeptUse() const;
    bool RevivedUse() const;
    const char* KeptDifference(const ShaderResources& fresh, std::span<const MovedBuffer> moved) const;
    static std::string TakeTemplateKeepText();
    static std::string TakeViewTemplateKeepText();
    std::uint32_t TableElements() const { return tableElements[0] + tableElements[1]; }
    const std::vector<std::uint32_t>& LayoutKey() const { return layoutKey; }
    const std::vector<std::uint32_t>& FragmentLayoutKey() const { return fragmentLayoutKey; }
    // Debug aid: each bound guest resource with the fraction of sampled bytes that are nonzero.
    std::string Describe() const;

    // Reuse across dispatches of identical content (see the resource cache in VulkanDevice): the
    // key is everything a compute build consumes from its compiled shader (variant, and per binding
    // kind, role, slot, count, guest descriptor words, written images, depth-compare samplers); push
    // constants are not part of it because every dispatch pushes its own. Reusable says whether this
    // object can serve a later dispatch with the same key: nothing needs completion work and every
    // guest buffer is read in place through a host import. Revalidate brings a reusable object up to
    // date for another use, or says it cannot be reused any more (a buffer's import went away, or a
    // texture or storage image the set references is no longer the one its memory maps to).
    // Without `dataWords` the ShaderData and FlattenedSrt descriptor words stay out of the key
    // (their count and size remain): a compute template then serves dispatches whose constants
    // differ, and the hit refreshes its data buffers with the dispatch's words (RefreshData).
    static std::vector<std::uint32_t> ContentKey(const CompiledShader& shader, bool dataWords = true, bool movableBuffers = false, bool movableWrites = false);
    static void AppendContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers, std::vector<std::uint32_t>& key, bool movableWrites = false);
    static bool DiffContentKeys(std::span<const std::uint32_t> previous, std::span<const std::uint32_t> key, bool draw, ContentKeyDiff& diff);
    static bool ContentKeyMatches(const CompiledShader& shader, std::span<const std::uint32_t> key, bool dataWords, bool movableBuffers, bool movableWrites);
    // Records the shader's ShaderData and FlattenedSrt words into this object's data buffers
    // (vkCmdUpdateBuffer, a transfer write the caller's pre-dispatch barrier makes visible; a
    // buffer already holding the words is left alone). Returns whether anything was recorded. With
    // `recorder` the updates are one [gputime] range of class TemplateDataRefresh.
    bool RefreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder = nullptr);
    // Walks the write watch over every image surface of a completed object (no device lock: the
    // ranges are fixed by the build), so a Revalidate's collects on the same worker are memo hits.
    void PrecollectSurfaces() const;
    bool Reusable() const { return reusable; }
    void KeepStorageImageBytes(Recorder& recorder) const;
    const std::vector<std::uint64_t>& MultisampleSources() const { return multisampleSources; }
    // The present in which a record commit last touched this template's resource cache entry (APS5_TEMPLATE_TOUCH).
    std::uint64_t TouchedPresent() const { return touchedPresent; }
    void NoteTouchedPresent(std::uint64_t present) { touchedPresent = present; }
    void NoteCacheUse() const {
        if (!cacheUsed.load(std::memory_order_relaxed)) cacheUsed.store(true, std::memory_order_relaxed);
    }
    bool TakeCacheUse() const { return cacheUsed.exchange(false, std::memory_order_relaxed); }
    static bool NeverReusable(std::span<const CompiledShader> shaders);
    static std::uint64_t DepthGuardRefusals();
    const std::vector<std::shared_ptr<Texture>>& SampledTextures() const { return textures; }
    static std::size_t LookupLogCapacity();
    bool ChecksFaults() const { return bda != nullptr; }
    void CheckFaults() const;
    // `shaders` are the stages the object was built from, in build order (a recorded draw's vertex
    // and fragment stages, or one compute stage): their bindings are walked like the build did.
    // How a Revalidate proved (or refused) the object, for the [recipe] line: the proof path taken
    // and, on a refusal, what failed.
    enum class ProofPath : std::size_t { Fast, OwnRefreshed, Full, Count };
    enum class ProofFailure : std::size_t { None, Imports, Evicted, Pending, Changed, Keys, Other, Count };
    enum class ProofRefusal : std::uint8_t { None, Other, Views, NotReusable, WalkProgram, WalkSnapshot, WalkView, WalkDepth, WalkStorage, WalkTable, WalkCount, Imports, Staging, StagingCopies, AsIs, Injected, Count };
    struct ProofReport {
        ProofPath path = ProofPath::Fast;
        ProofFailure failure = ProofFailure::None;
        bool audited = false;
        ProofRefusal refusal = ProofRefusal::None;
        std::uint8_t fastFail = 0xffu;
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
    };
    static const char* ProofRefusalName(ProofRefusal refusal);
    static const char* FastFailName(std::size_t reason);
    enum class CleanProof : std::uint8_t { Skipped, Requeried, Verified, Differs, FirstProof, Failed, NotReusable, StorageImages, DepthCopies, OwnWrites, NotClean, Views, Serials, Forgotten, Rewritten, PendingWrites, Unsettled, Count };
    bool Revalidate(std::span<const CompiledShader> shaders, ProofReport* report = nullptr, CleanProof* clean = nullptr);
    struct CleanStamp {
        std::uint64_t forget = 0;
        std::uint64_t release = 0;
        std::uint64_t pending = 0;
        std::uint64_t imports = 0;
        std::uint64_t allocations = 0;
        std::uint64_t unwatched = 0;
        std::uint64_t depthSurfaces = 0;
        std::uint64_t generation = 0;
    };
    struct CleanRuleCounts {
        std::uint64_t viewHits = 0;
        std::uint64_t pendingHits = 0;
        std::uint64_t exemptRecords = 0;
        std::uint64_t viewDrops = 0;
        std::uint64_t sourceDrops = 0;
        std::uint64_t verified = 0;
        std::uint64_t differs = 0;
        std::uint64_t requiredNotClean = 0;
        std::uint64_t exemptUnsettled = 0;
        std::uint64_t noSources = 0;
        std::uint64_t snapshotOverView = 0;
        std::uint64_t unrecordedViews = 0;
    };
    static CleanRuleCounts CleanRuleCounted();
    bool AsIsCurrent();
    // Objects whose fast proof came from another template's stamp (FastProofStamp), so far.
    static std::uint64_t FastProofStampsTaken();
    static bool EpochProofs();
    // Why Revalidate's epoch shortcut missed since the last call (APS5_STAGE_STATS).
    static std::string TakeEpochMissReport();
    bool Revalidate(const CompiledShader& shader, ProofReport* report = nullptr) { return Revalidate(std::span<const CompiledShader>(&shader, 1), report); }
    // The proof a recipe hit runs on its template (design_cpu_final M4): Revalidate by that name.
    bool ProveCurrent(std::span<const CompiledShader> shaders, ProofReport* report = nullptr, CleanProof* clean = nullptr) { return Revalidate(shaders, report, clean); }
    bool ProveCurrent(const CompiledShader& shader, ProofReport* report = nullptr) { return Revalidate(shader, report); }
    // FNV over the ShaderData/FlattenedSrt words, in binding order: of this object's data buffers
    // as the recorded work will find them (maintained by addDataBuffer and RefreshData), and of a
    // compiled shader's descriptors. A recipe hit refreshes the template iff the two differ.
    std::uint64_t DataWordsHash() const { return dataWordsHash; }
    static std::uint64_t DataWordsHash(const CompiledShader& shader);
    void PatchPushConstants(std::span<std::byte, PipelinePushConstantBytes> bytes) const {
        for (const auto& [position, adjustment] : pushPatches) bytes[position] = static_cast<std::byte>(adjustment);
    }
    void PatchDataAddresses(std::span<std::byte, PipelinePushConstantBytes> bytes, std::span<const CompiledShader> shaders, const DrawBindings* rewritten) const;
    bool PatchMovedPushConstants(std::span<std::byte, PipelinePushConstantBytes> bytes, std::span<const MovedBuffer> moved) const {
        bool patched = false;
        for (const auto& entry : moved) {
            const auto& element = allocations[entry.allocation];
            if (entry.inPlace == VK_NULL_HANDLE || element.pushByte < 0 || entry.adjustment == element.adjustment) continue;
            bytes[static_cast<std::size_t>(element.pushByte)] = static_cast<std::byte>(entry.adjustment);
            patched = true;
        }
        return patched;
    }
    static bool MovedPushPatches(std::span<const MovedBuffer> moved) {
        return std::any_of(moved.begin(), moved.end(), [](const MovedBuffer& entry) { return entry.inPlace != VK_NULL_HANDLE; });
    }
    // Whether RefreshData would record anything for `shader` (the per-word compare; verification).
    bool DataWordsDiffer(const CompiledShader& shader) const;
    std::size_t BindingCount() const { return bindings.size(); }
    // Why the fast proof of a Revalidate left the object to the full walk (the [rescache]
    // revalidate line's reasons); Count: it did not.
    enum class FastFail : std::size_t { NoRecord, Collect, Pending, Evicted, Changed, Keys, ClearedView, StorageKeys, Depth, Replaced, Audit, Count };
    struct CollectRefusals {
        std::uint64_t sampled = 0;
        std::uint64_t storage = 0;
        std::uint64_t uncommitted = 0;
        std::uint64_t unwatched = 0;
        std::uint64_t other = 0;
        std::uint64_t surfaces = 0;
        std::uint64_t bytes = 0;
    };
    static CollectRefusals CollectRefusalCounts();
    struct UntrackedReproofs {
        std::uint64_t stamped = 0;
        std::uint64_t compared = 0;
    };
    static UntrackedReproofs UntrackedReproofCounts();
    struct PendingFails {
        std::uint64_t foreign = 0;
        std::uint64_t snapshot = 0;
        std::uint64_t t1Eligible = 0;
    };
    static PendingFails PendingFailCounts();
    static std::array<std::uint64_t, static_cast<std::size_t>(FastFail::Count)> FullWalkNanoseconds();
    // Why a Pending failure was left to the full walk instead of the own-object refresh (T1, see
    // refreshOwnObjects); Count: it was not.
    enum class OwnRefreshFallback : std::size_t { Disabled, Snapshot, Keys, ForeignView, SurfaceKey, NotImported, Uncached, Rerun, Count };
    // APS5_PROFILE_DRAW: the build's sub-phases ([resources] totals) and their times in
    // milliseconds (zero when not profiling), so the draw path can total them separately from
    // dispatches. bindingsMs covers the binding plan (stage A) and the image lookups (stage B);
    // prepareMs/completeMs are the two stages' totals.
    enum class BuildPhase : std::size_t { Bindings, Precollect, Upload, Descriptors, StageA, Images, Bda, StageB, Count };
    struct BuildTiming {
        double bindingsMs = 0;
        double uploadMs = 0;
        double descriptorsMs = 0;
        double prepareMs = 0;
        double completeMs = 0;
    };
    const BuildTiming& Timing() const { return timing; }
    // The image surfaces (address, bytes) whose stage-B lookup reads guest memory on the CPU and so
    // would wait, under the device lock, for recorded work writing them: a surface without a host
    // import, or a sampled one that stays a CPU snapshot (block compressed, or the storage-view path
    // is off). VulkanDevice::PrepareDispatch waits for that work before the lock (the pre-sync).
    // Usable right after the deferred constructor, for address-based builds too (it decodes the
    // compiled shader's bindings itself when stage A has not run), and on a completed object from
    // the resource cache (its Revalidate repeats the same lookups), without the device lock.
    std::vector<PresyncSurface> PresyncSurfaces() const;

private:
    friend class AgcDriver::VulkanDevice;
    friend struct ShaderResourcesTestAccess;
    bool refreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder);
    std::optional<std::size_t> shaderDataAllocation(const CompiledShader& shader) const;
    VkBuffer shaderDataBuffer(const CompiledShader& shader) const;
    std::pair<VkBuffer, VkDeviceSize> shaderDataLocation(const CompiledShader& shader, const DrawBindings* rewritten) const;
    void forgetShaderDataWords(const CompiledShader& shader);
    struct DescribedRange {
        const char* kind;
        std::uint64_t address;
        std::uint64_t bytes;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t format = 0;
        int tileMode = 0;
        std::uint64_t dccAddress = 0;
        bool imageTable = false;
        std::optional<std::array<std::uint32_t, 8>> storedWords;
        std::uint32_t storedMip = 0;
    };
    std::vector<DescribedRange> describedRanges;
    struct Allocation {
        std::uint64_t address;
        std::size_t size;
        bool guest;
        std::unique_ptr<Buffer> buffer;
        ShaderRecompiler::DescriptorRole role = ShaderRecompiler::DescriptorRole::ShaderData;
        // Guest buffers: whether the shader may store to the range (see addGuestBuffer).
        bool written = true;
        bool writeOnly = false;
        // Data buffers: the words the buffer holds once the recorded work ran (see RefreshData).
        std::vector<std::uint32_t> dataWords;
        std::uint32_t adjustment = 0;
        std::int32_t pushByte = -1;
        std::int64_t dataAllocation = -1;
        std::uint32_t dataByte = 0;
        bool oversized = false;
    };
    struct DataPatch {
        std::size_t allocation;
        std::uint32_t byte;
        std::uint32_t adjustment;
    };
    void writeDataWords(VkCommandBuffer commands, std::size_t allocation, std::span<const std::uint32_t> words) const;

    struct Binding {
        VkDescriptorSetLayoutBinding layout;
        std::vector<std::size_t> allocations;
        std::vector<std::size_t> imageAllocations;
        VkDescriptorBindingFlags flags = 0;
        std::int32_t set = -1;
        bool heap = false;
        bool addressed = false;
        std::uint32_t dataStage = 0;
        std::uint32_t addressDword = 0;
        ShaderRecompiler::DescriptorRole role = ShaderRecompiler::DescriptorRole::GuestBuffers;
    };

    struct TableStage {
        ShaderRecompiler::ShaderStage stage = ShaderRecompiler::ShaderStage::Compute;
        const ShaderRecompiler::RecompileResult* program = nullptr;
        const ShaderRecompiler::DescriptorBinding* images = nullptr;
        std::span<const std::uint32_t> imageWords;
        std::size_t mapAllocation = std::numeric_limits<std::size_t>::max();
        std::vector<std::uint32_t> map;
        std::vector<ShaderRecompiler::ImageTableEntryPoison> poison;
        std::vector<std::shared_ptr<const std::string>> messages;
        std::uint32_t elementBase = 0;
        std::uint32_t set = 0;
        std::size_t firstSampler = 0;
        std::size_t samplerCount = 0;
        bool sampled = false;
        bool storage = false;
        std::vector<std::size_t> elementTextures;
        std::vector<std::uint32_t> representatives;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> samplerRejections;
        bool heap = false;
        std::vector<std::size_t> elementStorage;
        std::vector<std::size_t> tableSamplers;
    };
    std::vector<TableStage> tableStages;
    std::array<std::size_t, 2> tableBindings{std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()};
    std::array<std::uint32_t, 2> tableElements{};
    std::array<std::size_t, 2> storageTableBindings{std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()};
    TableStage& tableStage(const CompiledShader& shader);
    static const ShaderRecompiler::DescriptorBinding* tableImages(std::span<const CompiledShader> shaders, ShaderRecompiler::ShaderStage stage, std::span<const std::uint32_t>& words);
    std::uint32_t rejectTableElement(TableStage& table, std::span<const std::uint32_t> words, const std::string& message);
    void addSamplerTable(const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> guest, VkShaderStageFlags flags, TableStage& table);
    void resolveImageTables();
    void finishImageTables();

    // A host-imported buffer region the set reads in place, with its import's identity at build time.
    struct DirectRegion {
        std::uint64_t begin;
        std::uint64_t end;
        std::uint64_t serial;
        std::uint64_t regionEnd = 0;
        std::uint64_t committedEnd = 0;
    };

    // The build is two stages. A (buildPrepare, no device lock): the binding plan with its layout
    // entries, samplers, data buffers, the upload's prepare stage (imports already serving the
    // regions, read-only copies), the set layout and the descriptor set. B (buildComplete, under
    // GuestMemory::GpuMutex): the texture and storage image lookups (they refresh, upload and flush
    // through the recorder), the rest of the upload, the BDA objects, the descriptor writes and the
    // reusability record. build runs both.
    void build(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes);
    void buildPrepare(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes);
    void buildComplete();
    // APS5_PROFILE_DRAW: closes the current sub-phase of the build into the [resources] totals.
    double phase(BuildPhase which);
    // `written` is the element's DescriptorBinding::bufferWritten: a read-only element binds the
    // same way but is left out of the write set (no write-back, no pending-write note). `atomic`
    // is its bufferAtomic (see GuestBufferMemory::AddWritable).
    std::size_t addGuestBuffer(std::span<const std::uint32_t> words, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes, bool written, bool atomic, bool read, std::uint32_t extent);
    std::uint64_t guestBufferBytes(std::span<const std::uint32_t> words) const;
    static std::uint64_t ClampedBufferBytes(std::span<const std::uint32_t> words, std::uint64_t maxStorageBufferRange);
    std::size_t addDataBuffer(std::span<const std::uint32_t> words);
    void requireGuestBuffers(std::span<const MovedBuffer> moved = {}, bool* found = nullptr) const;
    bool requireMovedGuestBuffers(std::span<const MovedBuffer> moved, const GuardTests* guards = nullptr, bool* found = nullptr, bool answered = false) const;
    bool movedInPlace(Recorder& recorder, const Allocation& item, std::uint64_t address, std::uint64_t size, MovedBuffer& entry) const;
    // Stage A: the layout entry of an image binding (samplers are taken at once, the sampler cache
    // locks itself); stage B looks the sampled textures and storage images up (resolveImageBinding).
    void addImageBinding(const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> guest, VkShaderStageFlags flags);
    // Stage A: one record per planned image element (ImageRecord), in plan order: the decoded
    // descriptor and its surface size (computed once for the build), the write watch walked so stage
    // B's collects are memo hits (APS5_NO_PRECOLLECT=1 skips the pass entirely), and for a sampled
    // element the cache entry it maps to, taken now so stage B needs only the fastRevalidate
    // predicate under the lock (APS5_NO_STAGE_A_IMAGES=1 leaves every element to the full lookup).
    // Whether it ran.
    bool precollectImages();
    struct ImageRecord {
        bool sampled = false;
        // The descriptor decoded and its surface described; false leaves the element to stage B's
        // own decode (which reports the error).
        bool decoded = false;
        std::array<std::uint32_t, 8> words{};
        GuestTextureResource resource{};
        std::uint64_t guestBytes = 0;
        VkComponentMapping components{};
        // Sampled elements: the keys read in stage A and the collect made BEFORE stage B's checks
        // (the generation the entry moves to when they pass).
        DccKeys keys = DccKeys::Uncompressed;
        std::uint64_t generation = 0;
        // The cache entry's object, storage source, keys and generation as of stage A (texture null:
        // no entry, the full lookup makes one).
        std::shared_ptr<Texture> texture;
        std::shared_ptr<StorageTexture> source;
        DccKeys entryKeys = DccKeys::Uncompressed;
        std::uint64_t entryGeneration = 0;
    };
    std::vector<ImageRecord> imageRecords;
    struct PreviousSampled {
        std::array<std::uint32_t, 8> words{};
        bool depthCompare = false;
        VkImageViewType multisampleViewType = VK_IMAGE_VIEW_TYPE_2D;
        bool firstSample = false;
        bool layerRows = false;
        std::shared_ptr<Texture> texture;
    };
    PreviousSampled previousSampled;
    // The next record resolveImageBinding consumes (records follow the deferredImages order).
    std::size_t nextImageRecord = 0;
    // Stage B: the record's texture when the fastRevalidate predicate proves it current under the
    // lock and the cache still holds it; null sends the element to cachedTexture.
    std::shared_ptr<Texture> fastTexture(ImageRecord& record);
    void resolveImageBinding(const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> guest, Binding& item, std::span<const std::shared_ptr<Sampler>> shaderSamplers);
    void applySamplerVariants();
    void forgetDeferredInputs();
    void release() noexcept;
    void prepareAddressBindings(std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots);
    VkDescriptorBufferInfo descriptor(Allocation& allocation) { return descriptor(allocation, allocation.adjustment); }
    VkDescriptorBufferInfo descriptor(const Allocation& allocation, std::uint32_t& adjustment) const;
    VkDescriptorBufferInfo wholeDwordRange(VkDescriptorBufferInfo info) const;
    void noteReusable();
    void reportDescriptorCaches() const;
    // What a sampled texture was proved current against when the build (or the last full Revalidate)
    // looked it up, so the next Revalidate can repeat the proof from write stamps and the DCC keys
    // alone instead of repeating the lookup (see fastRevalidate). One record per textures[] entry; an
    // element without a record always takes the full lookup. Storage images need no record: their
    // proof comes from the image itself (its own surface and generation).
    struct ValidatedSurface {
        GuestTextureResource resource{};
        std::uint64_t bytes = 0;
        // Keys read at the lookup (a snapshot texture holds its clear texels while they are unchanged).
        DccKeys keys = DccKeys::Uncompressed;
        // Snapshot textures: write generation the content was known current at, always from a
        // collect issued before the checks that proved it.
        std::uint64_t generation = 0;
        // The collect of the check in progress, moved into `generation` once every element passed.
        std::uint64_t collected = 0;
        // Storage-sourced textures: the image the view follows (kept alive by the texture).
        const StorageTexture* source = nullptr;
        bool depth = false;
        bool valid = false;
        std::uint64_t committedGeneration = 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> committed;
        bool committedProved = false;
        std::array<std::uint64_t, 4> untracked{};
    };
    bool keepsTemplateRecords() const;
    void captureValidation(bool spaceReusable);
    static void noteStampTaken();
    // A surface of this object the fast proof found a foreign image pending over: the element
    // (index into textures, or into storageTextures when `storage`) whose own object stands in for
    // the full walk's lookup (refreshOwnObjects). `viewUncompressed`: a storage-sourced view under
    // uncompressed keys, which the lookup serves by its own source whenever FindPending names it.
    // `sourceEligible` (set by refreshOwnObjects on a resolved view): the lookup views the cached
    // image of the surface's key, i.e. the view's source, when FindPending names nothing.
    struct PendingOverlap {
        std::size_t element;
        bool storage;
        bool viewUncompressed;
        bool sourceEligible = false;
        bool depthCompare = false;
        bool sourceRefreshed = false;
    };
    // `serialBefore` is StorageTexture::PendingSerial() loaded before any check of this Revalidate;
    // `refreshed` the surfaces refreshOwnObjects resolved earlier in the same Revalidate (empty on
    // the first run): a foreign image still pending over one of them (the own image's alias, an
    // image whose pending units lie outside the surface) is no failure, since the lookup's answer
    // for the surface is the own object its Refresh just brought current, whatever the registry
    // holds; `reason` receives why the proof failed (Count: it passed), `overlapping` the surfaces
    // of a Pending failure in element order (empty otherwise), `accepted` whether a foreign overlap
    // was accepted by either rule (the verify switch then runs the full walk beside the proof).
    std::uint64_t touchedPresent = ~std::uint64_t{0};
    mutable std::atomic<bool> cacheUsed{false};
    bool fastRevalidate(std::uint64_t serialBefore, std::span<const PendingOverlap> refreshed, FastFail& reason, std::vector<PendingOverlap>& overlapping, bool& accepted, const std::vector<std::size_t>* only = nullptr, bool withStorage = true, bool compares = true);
    enum class IndexOutcome : std::uint8_t { Accepted, Partial, Fallback };
    IndexOutcome indexRevalidate(std::span<const CompiledShader> shaders, std::uint64_t serialBefore);
    void subscribeIndex(std::span<const CompiledShader> shaders);
    void subscribeBuilt(std::span<const CompiledShader> shaders);
    void noteIndexProof(std::uint64_t generation, std::uint64_t serialBefore);
    bool reresolveElements(const std::vector<std::size_t>& elements, std::vector<std::size_t>& replacedElements);
    struct IndexElement {
        std::array<std::uint32_t, 8> words{};
        std::uint32_t count = 0;
        std::optional<ShaderRecompiler::DescriptorImageShape> shape;
        bool depthCompare = false;
        bool known = false;
    };
    std::shared_ptr<GuestMemory::WriteSubscription> writeSubscription;
    std::uint64_t subscribedGeneration = 0;
    std::uint64_t subscribedPendingSerial = 0;
    std::vector<std::pair<std::uint64_t, std::size_t>> subscribedBlocks;
    std::vector<std::size_t> perElementTextures;
    std::vector<std::size_t> comparedTextures;
    std::vector<std::size_t> depthCopyTextures;
    std::vector<std::pair<std::uint64_t, std::vector<std::byte>>> comparedTableSpans;
    std::vector<IndexElement> indexElements;
    struct ImagePosition {
        std::size_t texture;
        std::size_t position;
        bool table;
    };
    std::vector<ImagePosition> indexImagePositions;
    bool rebound = false;
    mutable std::atomic<std::uint64_t> keptSeen{0};
    mutable std::atomic<std::uint64_t> revivedSeen{0};
    // T1 (design_cpu_final M3): brings the overlapping surfaces' own objects up to date exactly as
    // the full walk's lookups would, in the walk's binding order, without the lookups: a storage
    // image's Refresh (which stores the foreign images over its surface first), a view's source's
    // Refresh when the lookup would refresh the cached image of the view's surface key, nothing
    // when FindPending names the source itself (the lookup hits on it). Count, or the fallback
    // reason: a surface whose lookup could return another object (a snapshot texture, a view under
    // fast-clear keys, a foreign pending image the lookup would view, a surface key mapping to
    // another cached image, a surface outside the host imports, an object no longer cached), which
    // leaves the whole object to the full walk. Under GuestMemory::GpuMutex, as Revalidate is.
    OwnRefreshFallback refreshOwnObjects(std::span<const CompiledShader> shaders, std::span<PendingOverlap> overlapping);
    // Today's per-element walk (APS5_NO_EPOCH_REVALIDATE=1).
    bool fastRevalidateEach();
    Context context;
    std::vector<std::uint32_t> layoutKey;
    GuestBufferMemory guestMemory;
    std::shared_ptr<ShaderResources> afterRunSelf;
    struct ReadNote {
        std::uint64_t batch = 0;
        std::uint8_t kind = 0;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    };
    ReadNote readNote;
    void notePendingReads(Recorder& recorder, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, std::uint8_t kind, bool lean);
    std::size_t maxDrawSnapshotBytes = MaxDrawSnapshotBytes();
    std::unique_ptr<BdaResources> bda;
    std::vector<BdaResources::SrtGuardFaults> srtGuardFaults;
    bool usesBda = false;
    mutable std::uint64_t faultCheckRecorder = 0;
    mutable std::uint64_t faultCheckBatch = 0;
    mutable std::atomic<std::uint64_t> faultGeneration{0};
    bool walksPageTables = true;
    bool bdaReadOnly = true;
    bool spaceReuse = false;
    bool faultReportOnly = true;
    std::vector<std::uint64_t> multisampleSources;
    bool usesFaultBuffer = false;
    bool faultChecks = false;
    VkDescriptorSetLayout _layout = VK_NULL_HANDLE;
    // Whether the layout is this object's own (no cache) and destroyed with it.
    bool ownsLayout = false;
    VkDescriptorSet _set = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    // The cache pool the set was allocated from, freed back to it on release.
    DescriptorCache::SetAllocation cacheAllocation;
    struct SetObjects {
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        bool ownsLayout = false;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        DescriptorCache::SetAllocation cacheAllocation;
    };
    bool splitSets = false;
    bool heapStage = false;
    std::array<std::uint32_t, 2u * ShaderRecompiler::RuntimeAbi::StageCount> heapOffsets{};
    void writeHeapStage(const CompiledShader& shader);
    struct RingDescriptors {
        std::vector<VkDeviceAddressRangeEXT> ranges;
        BindlessHeap::SpanMark mark;
        std::uint32_t first = 0;
    };
    struct HeapStageState {
        std::vector<std::uintptr_t> key;
        std::array<std::vector<std::uint32_t>, static_cast<std::size_t>(ShaderRecompiler::RuntimeAbi::Binding::Count)> slots;
        std::vector<std::uint32_t> slotGenerations;
        std::vector<std::uint32_t> samplerSlots;
        std::vector<std::uint32_t> samplerGenerations;
        std::uint64_t slotSerial = 0;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> dataPatches;
        struct KeptSpan {
            std::vector<std::uint32_t> words;
            BindlessHeap::SpanMark mark;
        };
        KeptSpan data, flat, map;
        std::vector<std::uint32_t> dataInput;
        RingDescriptors descriptors;
    };
    std::array<HeapStageState, ShaderRecompiler::RuntimeAbi::StageCount> heapStages;
    mutable std::uint64_t coverageSerial = 0;
    std::uint32_t heapSpan(HeapStageState::KeptSpan& kept, std::span<const std::uint32_t> words);
    struct HeapReplayStage {
        const ShaderRecompiler::RecompileResult* program = nullptr;
        bool ready = false;
        bool replayable = false;
        bool srt = false;
        bool wordsFromProgram = false;
        std::size_t dataBinding = 0;
        std::size_t srtBinding = 0;
        std::size_t userDwords = 0;
        std::size_t dataDwords = 0;
        std::uint64_t spanEpoch = 0;
        std::vector<std::uint64_t> mask;
        std::vector<VkDeviceAddressRangeEXT> censusRun;
    };
    struct HeapReplay {
        std::uint64_t writes = 0;
        std::uint64_t keyHash = 0;
        const BindlessHeap* heap = nullptr;
        std::uint64_t heapIdentity = 0;
        std::uint64_t switches = 0;
        bool open = false;
        std::vector<bool> addressed;
        std::array<std::uint64_t, 3> censusGenerations{};
        std::array<HeapReplayStage, ShaderRecompiler::RuntimeAbi::StageCount> stages;
    };
    std::unique_ptr<HeapReplay> heapReplay;
    std::uint64_t heapStageWrites = 0;
    bool heapReplaying = false;
    HeapReplay& heapReplayRecord();
    void heapStageMask(const CompiledShader& shader, std::size_t dataDwords, std::vector<std::uint64_t>& mask) const;
    void seedHeapReplayStage(HeapReplayStage& stage, const CompiledShader& shader) const;
    void reseedHeapReplay(std::span<const CompiledShader> shaders);
    void refreshHeapReplayFallback(Recorder& recorder, std::span<const CompiledShader> shaders, const std::shared_ptr<DrawBindings>& bindings, std::uint64_t provenEpoch);
    bool replayHeapStage(const CompiledShader& shader, HeapReplayStage& stage, std::uint64_t epoch, bool overlay);
    bool keepReplaySpan(HeapStageState::KeptSpan& kept, bool proven, bool srt);
    void writeReplaySpan(HeapStageState::KeptSpan& kept, std::span<const std::uint32_t> words);
    void noteHeapStageCensus(const CompiledShader& shader, bool current) const;
    void buildHeapStageWords(const CompiledShader& shader, const HeapStageState& state, std::span<const std::uint32_t> words, std::span<const std::uint32_t> userData, std::vector<std::uint32_t>& patched) const;
    void verifyHeapReplay(Recorder& recorder, std::span<const CompiledShader> shaders, const std::shared_ptr<DrawBindings>& bindings, const HeapReplayProof& proof, bool overlaid);
    // The hazard tests of moved read-only (begin, bytes) ranges and the ticket's use of them (a range
    // the ticket trusts needs only the shadow test); `accessible`: the caller found every range mapped.
    struct InPlaceRange {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
    };
    bool movedRangesClear(Recorder& recorder, std::span<const std::pair<std::uint64_t, std::size_t>> ranges, MoveTicket* ticket, bool accessible, std::vector<InPlaceRange>* inPlace = nullptr, MovedCount* why = nullptr, std::vector<std::uint8_t>* behind = nullptr) const;
    bool takeMovedInPlace(const Recorder& recorder, std::uint64_t begin, std::size_t bytes, InPlaceRange& out) const;
    std::optional<MovedCount> inPlaceRefusalByWindows(const Recorder& recorder, std::uint64_t begin, std::size_t bytes, WriteValidation::Test test) const;
    std::shared_ptr<DrawBindings> selectDrawSnapshots(Recorder& recorder, std::span<const MovedBuffer> moved, bool movedOnly, std::vector<std::size_t>& selected, bool census) const;
    // The selection of prepared buffers, their blob in one upload slice; null, before any upload,
    // when an unmoved range is no longer clear or a resident depth surface may hold a selected
    // range's bytes (selectDrawSnapshots then selects), with the reason in `selection`. `binds`:
    // the draw binds it, so its snapshots count in the moved-upload tally and the capture trace.
    std::shared_ptr<DrawBindings> selectPreparedSnapshots(Recorder& recorder, std::span<const MovedBuffer> moved, const PreparedBuffers& prepared, std::vector<std::size_t>& selected, bool binds, PreparedBuffers::Selection& selection, BindingRoute* route = nullptr) const;
    bool unmovedClear(const Recorder& recorder, const PreparedBuffers& prepared, BindingRoute* route) const;
    mutable std::uint64_t clearRecorder = 0;
    mutable std::uint64_t clearNote = 0;
    mutable std::size_t clearWrites = 0;
    mutable std::vector<bool> clearAllocations;
    std::shared_ptr<DrawBindings> finishDrawBindings(Recorder& recorder, std::shared_ptr<DrawBindings> result, std::span<const std::size_t> selected, const BdaResources* faultRecord, std::int32_t pushedSet = -1, BindingRoute* route = nullptr) const;
    struct SetLender {
        std::weak_ptr<const DrawBindings> bindings;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        std::vector<std::size_t> allocations;
        std::vector<std::size_t> snapshots;
        std::vector<VkDescriptorBufferInfo> buffers;
        std::vector<std::byte> bytes;
        std::uint32_t misses = 0;
        std::uint32_t resting = 0;
    };
    mutable std::array<SetLender, 2> setLenders;
    bool lendsSets(std::int32_t pushedSet, const BdaResources* faultRecord) const;
    void lendDrawSets(DrawBindings& result, std::span<const std::size_t> selected, std::int32_t pushedSet, BindingRoute& route) const;
    void noteSetLenders(const std::shared_ptr<DrawBindings>& result, std::span<const std::size_t> selected, BindingRoute& route) const;
    bool ownsHeapBinding(const Binding& binding, VkShaderStageFlags stageBit) const;
    VkImageView heapPlaceholderView(std::uint32_t kind) const;
    VkImageView heapSampledView(std::size_t index) const;
    VkImageView heapStorageView(std::size_t index, std::uint32_t binding) const;
    VkImageView heapMultisampleStorageView(std::size_t index) const;
    const HeapSlotOwner* heapMultisampleStorageOwner(std::size_t index) const;
    void appendHeapSlotKey(VkShaderStageFlags stageBit, std::vector<std::uintptr_t>& key) const;
    void takeHeapSlots(BindlessHeap& heap, HeapStageState& state, VkShaderStageFlags stageBit, std::span<const std::uintptr_t> key);
    static void noteHeapSlotGenerations(const BindlessHeap& heap, HeapStageState& state);
    void verifyHeapSlots(const BindlessHeap& heap, const HeapStageState& state, VkShaderStageFlags stageBit) const;
    void patchHeapSlotWords(const CompiledShader& shader, const HeapStageState& stageState, std::vector<std::uint32_t>& patched, bool current) const;
    static constexpr std::int64_t HeapRingDataFirst = -2;
    static constexpr std::int64_t HeapRingData(std::size_t stage) { return HeapRingDataFirst - static_cast<std::int64_t>(stage); }
    static constexpr bool IsHeapRingData(std::int64_t allocation) { return allocation <= HeapRingDataFirst && allocation > HeapRingDataFirst - static_cast<std::int64_t>(ShaderRecompiler::RuntimeAbi::StageCount); }
    static constexpr std::size_t HeapRingStage(std::int64_t allocation) { return static_cast<std::size_t>(HeapRingDataFirst - allocation); }
    std::vector<std::uint32_t> fragmentLayoutKey;
    SetObjects fragment;
    struct TableWrites {
        bool recorded = false;
        bool written = false;
        std::uint32_t binding = 0;
        std::size_t firstImage = 0;
        std::size_t images = 0;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> runs;
    };
    std::array<TableWrites, 2> tableWrites;
    std::shared_ptr<const TableDescriptors> keptTable(std::size_t set, std::shared_ptr<const std::vector<std::weak_ptr<Texture>>>& held) noexcept;
    std::uint32_t setOf(const Binding& binding) const;
    std::vector<Allocation> allocations;
    mutable std::vector<VkDescriptorBufferInfo> descriptorBuffers;
    std::vector<VkDescriptorImageInfo> descriptorImages;
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    bool addressedBuffers = false;
    std::size_t gdsAllocation = std::numeric_limits<std::size_t>::max();
    std::array<std::vector<std::uint32_t>, ShaderRecompiler::RuntimeAbi::StageCount> stageDataWords;
    std::array<std::size_t, ShaderRecompiler::RuntimeAbi::StageCount> stageDataIndex{};
    std::array<std::uint32_t, ShaderRecompiler::RuntimeAbi::StageCount> heapSrtDwords{};
    std::array<std::uint32_t, ShaderRecompiler::RuntimeAbi::StageCount> heapTableMaps{};
    std::array<bool, ShaderRecompiler::RuntimeAbi::StageCount> heapTableMapSet{};
    void writeHeapTableMap(const CompiledShader& shader, std::size_t stageIndex);
    void writeBufferAddresses(std::span<std::uint32_t> words, std::uint32_t dataStage, const std::vector<VkDescriptorBufferInfo>* overrides,
                              std::vector<VkDeviceAddressRangeEXT>* ranges = nullptr) const;
    VkDeviceAddressRangeEXT srtRange(std::size_t stage) const;
    std::uint32_t ringDescriptors(std::size_t stage, const std::vector<VkDescriptorBufferInfo>* overrides, RingDescriptors& kept) const;
    std::array<std::uint32_t, ShaderRecompiler::RuntimeAbi::StageCount> heapSrtOffsets{};
    std::array<bool, ShaderRecompiler::RuntimeAbi::StageCount> ringDescriptorStages{};
    std::array<std::uint32_t, 2u * ShaderRecompiler::RuntimeAbi::StageCount> addressedOffsets(const std::vector<VkDescriptorBufferInfo>* overrides, bool census = true) const;
    struct FaultDescriptor {
        std::size_t write;
        std::size_t buffer;
        std::uint32_t element;
    };
    std::vector<FaultDescriptor> faultDescriptors;
    void prepareFaultSlot() const;
    mutable bool bindingsChecked = false;
    mutable std::uint64_t faultStampRecorder = 0;
    mutable std::uint64_t faultStampBatch = 0;
    mutable std::atomic<std::uint64_t> depthGuardsClear{std::numeric_limits<std::uint64_t>::max()};
    mutable std::atomic<std::uint64_t> multisampleGuardsClear{std::numeric_limits<std::uint64_t>::max()};
    void writeDrawSets(DrawBindings& result, std::span<const std::size_t> selected, const BdaResources* faultRecord, std::int32_t pushedSet = -1) const;
    void writeFreshDrawSets(DrawBindings& result, std::span<const std::size_t> selected, const BdaResources* faultRecord, std::int32_t pushedSet) const;
    void writePlannedFragmentSets(DrawBindings& result, std::span<const std::size_t> selected, std::int32_t pushedSet) const;
    void pushPlannedBindings(VkCommandBuffer commands, VkPipelineLayout layout, const DrawBindings& result, const PushPlan& plan) const;
    static std::uint64_t NextPlanSerial();
    std::uint64_t planSerial = NextPlanSerial();
    bool preparedSelectionHolds(Recorder& recorder, std::span<const MovedBuffer> moved, const PreparedBuffers& prepared, PreparedBuffers::Selection& selection, BindingRoute* route, bool untouched = false, const GuardTests* guards = nullptr, bool trusted = false, CommitterTrustCheck* trust = nullptr, std::uint8_t answered = 0) const;
    enum class ZoneCardUse : std::uint8_t { Follower, Leader, Guards };
    bool zoneCardUsable(const PreparedBuffers& prepared, ZoneCardUse use) const;
    bool zoneCardHolds(const PreparedBuffers& prepared, bool guards) const;
    void pushBindings(VkCommandBuffer commands, VkPipelineLayout layout, const DrawBindings& result, const PushPlan& plan) const;
    void writeOwnDrawSets(DrawBindings& result, std::span<const std::size_t> selected, const BdaResources* faultRecord, std::int32_t pushedSet = -1) const;
    mutable std::unique_ptr<PushPlan> pushPlan;
    void overlayPushedBuffers(const DrawBindings& result, const PushPlan& plan, std::span<VkDescriptorBufferInfo> buffers, std::span<std::size_t> owners = {}) const;
    mutable bool pushPlanBuilt = false;
    mutable std::unique_ptr<DynamicPlan> dynamicPlan;
    mutable bool dynamicPlanBuilt = false;
    mutable DynamicLeave dynamicPlanMiss = DynamicLeave::None;
    mutable DynamicLeave pushedPlanMiss = DynamicLeave::None;
    struct DynamicSetEntry;
    mutable std::array<std::vector<std::shared_ptr<DynamicSetEntry>>, 2> dynamicSets;
    mutable std::vector<VkDescriptorBufferInfo> dynamicBuffers;
    bool writeDynamicSets(DrawBindings& result, std::span<const std::size_t> selected, BindingRoute& route, std::int32_t pushedSet) const;
    mutable std::shared_ptr<const BufferPlan> bufferPlan;
    mutable bool bufferPlanBuilt = false;
    std::shared_ptr<OwnDrawSets> makeOwnDrawSets(bool fragmentStore = false) const;
    mutable std::shared_ptr<OwnDrawSets> ownDrawSets;
    void writeParkedFragmentSets(DrawBindings& result, std::span<const std::size_t> selected, std::int32_t pushedSet) const;
    mutable std::once_flag fragmentSetsOnce;
    mutable std::atomic<bool> fragmentSetsReady{false};
    mutable std::shared_ptr<OwnDrawSets> fragmentSets;
    mutable std::vector<std::shared_ptr<RepeatSets>> repeatSets;
    void writeAddressedOverlay(DrawBindings& result, std::span<const std::size_t> selected) const;
    std::shared_ptr<std::atomic<bool>> ownRecordBusy = std::make_shared<std::atomic<bool>>(false);
    mutable std::vector<VkDescriptorPoolSize> drawPoolSizes;
    // Guest buffer elements bound read-only: each use of this object skips that many pending-write
    // notes (counted in MarkGpuWrites for the [buffers] line).
    std::size_t readOnlyBuffers = 0;
    std::chrono::steady_clock::time_point lastFullWalk = std::chrono::steady_clock::now();
    std::uint64_t walkImportGeneration = HostImportSetGeneration();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writeOnlyRanges;
    bool drawBuild = false;
    std::uint64_t dispatchThreads = 0;
    std::optional<std::pair<std::uint64_t, std::size_t>> htileClear;
    std::vector<std::shared_ptr<Texture>> textures;
    std::vector<bool> textureFirstLayer;
    std::unordered_set<std::size_t> arrayViewTextures;
    std::vector<bool> textureBaseLevel;
    std::vector<bool> textureSrgb;
    std::vector<std::shared_ptr<StorageTexture>> storageTextures;
    std::vector<MultisampleStorage> multisampleStorage;
    std::vector<std::uint32_t> storageMips;
    std::vector<std::uint64_t> storageKeys;
    std::vector<bool> storageFirstLayer;
    std::vector<bool> storageWritten;
    std::vector<bool> storageAtomic;
    std::vector<bool> storageAtomic64;
    std::vector<bool> storageUint;
    std::vector<std::shared_ptr<Sampler>> samplers;
    std::shared_ptr<Sampler> paddingSampler;
    struct SamplerSource {
        std::array<std::uint32_t, 4> words{};
        bool compareEnable = false;
        bool unnormalized = false;
        bool integerBorder = false;
        bool singleLevelImage = false;
        bool mipmappedImage = false;
        std::optional<std::uint32_t> pinnedLod;
    };
    std::vector<SamplerSource> samplerSources;
    bool reusable = false;
    NotReusable notReusable = NotReusable::None;
    std::vector<DirectRegion> directRegions;
    std::vector<ValidatedSurface> validatedTextures;
    std::vector<std::array<std::uint64_t, 4>> storageUntracked;
    std::vector<ShaderRecompiler::RegisteredImageView> registeredImageViews;
    // The pending registry's serial at the last Revalidate that proved this object, taken before
    // its checks and kept only when unchanged after them (0: none): while it is still the serial,
    // no image other than the object's own sources has results pending over its surfaces and
    // regions, so those checks are skipped (see fastRevalidate).
    std::uint64_t pendingSerialSeen = 0;
    bool revalidateProof(std::span<const CompiledShader> shaders, ProofReport* report);
    CleanStamp cleanStampNow() const;
    bool cleanCurrent(const CleanStamp& stamp, CleanProof& outcome);
    void recordClean(bool proved, const CleanStamp& stamp);
    bool cleanEligible();
    bool cleanUnchangedSince(std::uint64_t generation) const;
    bool cleanUnchangedSince(std::uint64_t generation, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) const;
    void noteCleanSources();
    std::uint8_t cleanViewsKept(const CleanStamp& stamp);
    bool cleanSourcesCurrent(FastFail& reason);
    bool cleanProved = false;
    struct ExpressBefore {
        std::uint64_t generation = 0;
        bool watched = false;
    };
    ExpressBefore expressBefore();
    void expressAfter(bool proved, const ExpressBefore& before);
    bool expressEligible() const;
    ExpressMiss expressCurrent(bool memo);
    ExpressMiss expressTest(bool memo, bool keeps, bool& kept);
    ExpressMiss expressRest(bool memo);
    bool proofKeepHeld();
    void noteProofKeepMiss(bool keeps) const;
    void noteProofKept();
    bool revalidateClean(std::span<const CompiledShader> shaders, ProofReport* report, CleanProof* clean);
    std::shared_ptr<GuestMemory::WriteSubscription> expressWatch;
    std::uint64_t expressGeneration = 0;
    CleanStamp expressAt;
    bool expressValid = false;
    mutable std::array<CommitterTrustState, static_cast<std::size_t>(CommitterTrustFamily::Count)> committerTrust{};
    std::atomic<std::uint64_t> expressViewsAt{0};
    std::atomic<std::uint64_t> expressRegionsAt{0};
    std::vector<std::pair<const StorageTexture*, std::uint64_t>> expressSources;
    std::vector<std::pair<const StorageTexture*, std::uint64_t>> expressSourcesBefore;
    std::vector<std::uint64_t> expressProofEpochs;
    std::vector<std::uint64_t> expressProofEpochsBefore;
    std::uint64_t proofKeptSum = 0;
    bool proofKeepVerify = false;
    struct StoresArm {
        std::uint64_t session = 0;
        std::uint64_t switches = 0;
        std::uint64_t recorder = 0;
        std::uint64_t stores = 0;
        std::uint64_t proof = 0;
    };
    mutable StoresArm wvStores;
    std::uint64_t wvProofSerial = 0;
    std::uint8_t cleanEligibility = 0;
    std::uint32_t cleanMisses = 0;
    CleanProof cleanRefusal = CleanProof::FirstProof;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> cleanRangeList;
    CleanStamp cleanAt;
    std::vector<std::size_t> cleanSourced;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> cleanRequired;
    bool cleanSourcesEligible = false;
    bool cleanExempt = false;
    std::uint8_t cleanHitRule = 0;
    std::uint64_t provedEpoch = 0;
    std::uint64_t provedProof = 0;
    std::uint64_t provedPending = 0;
    // Range proofs (on unless APS5_NO_RANGE_PROOFS is set): the guest ranges the proof read and the release serial it saw.
    std::uint64_t provedRelease = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> proofRanges;
    bool proofRangesBuilt = false;
    bool proofRangesKnown = false;
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>* rangedProofRanges();
    std::uint64_t provedRetired = 0;
    bool revalidateFull(std::span<const CompiledShader> shaders, ProofReport* report);
    // The import table's identity when the direct regions' serials were last proved.
    HostImportsProof importsProof;
    bool importsReconcilable = false;
    // FNV-1a offset basis: the hash of no data buffers (DataWordsHash).
    std::uint64_t dataWordsHash = 14695981039346656037ull;
    void rehashDataWords();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pushPatches;
    std::vector<DataPatch> dataPatches;
    BuildTiming timing;
    // Build state carried from stage A to stage B: the bindings in plan order, the image bindings
    // still to look up (index into `bindings`; the DescriptorBinding lives in the compiled shader),
    // the descriptor counts the set was sized for, and the compute stage of a deferred build.
    std::vector<Binding> bindings;
    std::vector<VkDescriptorBindingFlags> layoutBindingFlags;
    std::vector<VkDescriptorBindingFlags> fragmentBindingFlags;
    std::vector<std::uint32_t> refreshResourceKey;
    struct DeferredImages {
        const ShaderRecompiler::DescriptorBinding* binding;
        std::span<const std::uint32_t> words;
        std::size_t index;
        std::size_t firstSampler = 0;
        std::size_t samplerCount = 0;
    };
    std::vector<DeferredImages> deferredImages;
    std::uint64_t storageBuffers = 0;
    bool updateAfterBind = false;
    std::uint32_t plannedSampledImages = 0;
    std::uint32_t plannedStorageImages = 0;
    // The compute constructor's shader and captured regions: the caller's objects, valid only until
    // the build (Complete() for a deferred one) is done, and reset then (forgetDeferredInputs).
    CompiledShader deferredCompute{};
    std::span<const GuestMemorySnapshot> deferredSnapshots;
    // The deferred build is address-based: everything runs in Complete().
    bool lockedBuild = false;
    // Stage A runs without the device lock (a deferred, non-address-based build).
    bool unlockedPrepare = false;
    bool completed = false;
    std::chrono::steady_clock::time_point phaseStart;
};

// Built ShaderResources by the content the build consumed (see ShaderResources::ContentKey), so a
// dispatch or recorded draw repeating an earlier one binds the earlier descriptor set instead of
// building one (import lookups, sampler and descriptor objects, guest buffer uploads). Only reusable
// objects (nothing to complete, every buffer read in place) are kept, and an entry serves a later
// use only after ShaderResources::Revalidate brought it up to date. Entries hold their object alive;
// the recorder keeps it as well while its batch runs. One LRU bound covers dispatch and draw entries
// together (APS5_RESOURCE_CACHE_ENTRIES, default 1024); the map has its own mutex and locks itself:
// VulkanDevice::PrepareDispatch probes it without GuestMemory::GpuMutex (a hit there is revalidated
// under the lock by the dispatch; inserts and Revalidate stay under it). Compute keys start with the
// stage, draw keys with a marker word, so the two never meet, and both name the device handle: the driver replaces
// the headless device with the windowed one while workers may still hold the old one, so a replaced
// device's entries (whose descriptor sets and pooled buffers belong to it) are unreachable by key
// and dropped by that device's Clear(). APS5_NO_RESOURCE_CACHE=1 (dispatches) and
// APS5_NO_DRAW_RESOURCE_CACHE=1 (draws) build every object as before; the device clears the cache at
// teardown before its descriptor caches go.
class ResourceCache {
public:
    using Key = std::vector<std::uint32_t>;
    std::shared_ptr<ShaderResources> Find(const Key& key);
    // `evicted`, when given, receives the objects the insert displaces (the entry replaced under the
    // key, the ones over the bound) instead of their being destroyed here: a caller under the GPU
    // mutex hands them to the recorder so the destruction runs off the lock (VulkanDevice::dispatch).
    bool Insert(const Key& key, std::shared_ptr<ShaderResources> resources, std::vector<std::shared_ptr<ShaderResources>>* evicted = nullptr);
    // With `object`, erases the entry only while it still holds that object (a replacement made
    // meanwhile by another worker stays).
    void Remove(const Key& key, const ShaderResources* object = nullptr);
    void RemoveImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted = nullptr);
    bool TryRemoveImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted = nullptr);
    // Moves the entry to the front of the LRU (a recipe hit uses its template without Find, and a
    // hot template must not age out under the recipes that depend on it); false when no entry.
    bool Touch(const Key& key);
    bool TouchObject(const Key& key, const ShaderResources& object);
    // Find and Touch calls so far (the [rescache] line: a recipe hit touches instead of finding).
    static std::uint64_t Finds();
    static std::uint64_t Touches();
    enum class Fate : std::uint8_t { None, Built, NotRecorded, NotReusable, InsertRefused, Inserted, LookupProof, LookupSpace, RecordProof, Capacity, Tables, Image, ReleasedTexture, Count };
    struct FateNote {
        Fate fate = Fate::None;
        std::uint8_t detail = 0;
        std::uint64_t present = 0;
    };
    static bool FateCensus();
    static void NoteFate(const Key& key, Fate fate, std::uint8_t detail = 0);
    static FateNote FateOf(const Key& key);
    void Clear();
    std::size_t Size() const;
    std::size_t Capacity() const;
    void SetBase(std::size_t entries, std::size_t shrinkWindow);
    std::uint64_t TableElements() const;
    static std::uint64_t TableElementBudget();
    static constexpr std::size_t InitialEntries = 1024;
    static constexpr std::size_t MaximumEntries = 16384;

private:
    void erase(const Key& key);
    void noteMiss(const Key& key);
    void dropReleasedTextures(std::vector<std::shared_ptr<ShaderResources>>* evicted);
    void noteEvicted(const Key& key);
    void noteReturn(const Key& key);
    void noteUse(const Key& key);
    void removeImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted);
    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            constexpr std::uint64_t Prime = 0x9e3779b97f4a7c15ull;
            std::uint64_t lanes[4] = {0x243f6a8885a308d3ull, 0x13198a2e03707344ull, 0xa4093822299f31d0ull, 0x082efa98ec4e6c89ull};
            const auto* words = key.data();
            const auto count = key.size();
            std::size_t at = 0;
            for (; at + 8u <= count; at += 8u) {
                for (std::size_t lane = 0; lane < 4u; ++lane) {
                    const auto value = static_cast<std::uint64_t>(words[at + lane * 2u]) | static_cast<std::uint64_t>(words[at + lane * 2u + 1u]) << 32u;
                    lanes[lane] = (lanes[lane] ^ value) * Prime;
                    lanes[lane] ^= lanes[lane] >> 29u;
                }
            }
            std::uint64_t hash = lanes[0] ^ (lanes[1] << 17u | lanes[1] >> 47u) ^ (lanes[2] << 31u | lanes[2] >> 33u) ^ (lanes[3] << 47u | lanes[3] >> 17u) ^ count;
            for (; at < count; ++at) hash = (hash ^ words[at]) * Prime;
            hash ^= hash >> 33u;
            hash *= 0xff51afd7ed558ccdull;
            hash ^= hash >> 33u;
            hash *= 0xc4ceb9fe1a85ec53ull;
            hash ^= hash >> 33u;
            return static_cast<std::size_t>(hash);
        }
    };
    mutable AgcDriver::Mutex mutex;
    std::list<std::pair<Key, std::shared_ptr<ShaderResources>>> entries;
    std::unordered_map<Key, decltype(entries)::iterator, KeyHash> index;
    std::size_t capacity = 0;
    bool fixedCapacity = false;
    std::atomic<bool> lazyTouch{false};
    std::list<std::size_t> evictedOrder;
    std::unordered_map<std::size_t, decltype(evictedOrder)::iterator> evictedKeys;
    std::size_t evictionsSinceCheck = 0;
    std::size_t returnsSinceCheck = 0;
    std::unordered_set<std::size_t> usedKeys;
    std::size_t usesSinceCheck = 0;
    static constexpr std::size_t InitialShrinkWindow = 8;
    static constexpr std::size_t MaximumShrinkWindow = 1024;
    std::size_t baseEntries = InitialEntries;
    std::size_t baseShrinkWindow = InitialShrinkWindow;
    std::size_t shrinkWindow = InitialShrinkWindow;
    bool shrankLast = false;
};

// The process-wide cache the device and the draw path share (keys name the device; a device clears
// it when it goes). Never destroyed: entries belong to the device, not to static teardown.
ResourceCache& SharedResourceCache();

}

#endif
