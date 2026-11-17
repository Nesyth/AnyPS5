#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "ThreadOwned.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuardSpans.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuZones.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BindlessHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeviceImageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MemoryAccounting.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ResidencyCensus.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <memory>
#include <mutex>
#include <array>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <stdexcept>
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Optimization/include/Optimization/ShaderStageInputInfo.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaDescriptorFormat.hpp"
#include "IntermediateRepresentation/IrMetadata/ShaderStage.hpp"
#include "ImageTableAbi.hpp"
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "prx/libc/include/GuestAllocations.hpp"

namespace AgcDriver::Graphics {
namespace {

std::uint32_t takenSlot(std::optional<std::uint32_t> slot, const char* what) {
    Require(slot.has_value(), what);
    return *slot;
}

bool UintStorageBinding(std::uint32_t binding) {
    const auto kind = binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    return ShaderRecompiler::RuntimeAbi::UintStorageImageKind(kind);
}

bool overlap(std::uint64_t first, std::size_t firstSize, std::uint64_t second, std::size_t secondSize) {
    return first < second + secondSize && second < first + firstSize;
}

bool registeredImageViewsCurrent(std::span<const ShaderRecompiler::RegisteredImageView> views) {
    if (views.empty()) return true;
    std::vector<std::uintptr_t> addresses;
    addresses.reserve(views.size());
    for (const auto& view : views) addresses.push_back(static_cast<std::uintptr_t>(view.address));
    std::vector<std::shared_ptr<const GuestAllocations::Range>> ranges(views.size());
    GuestAllocations::GuestAllocationsLookupEach_nid_postfix(addresses.data(), addresses.size(), ranges.data());
    for (std::size_t index = 0; index < views.size(); ++index) {
        const auto& view = views[index];
        const auto& range = ranges[index];
        if (view.hasRange != (range != nullptr)) return false;
        if (range != nullptr && (range->address != view.rangeAddress || range->bytes != view.rangeBytes || range->allocationAddress != view.allocationAddress || range->allocationBytes != view.allocationBytes || range->backingSerial != view.backingSerial || range->readable != view.readable || range->writable != view.writable || range->gpu != view.gpu)) return false;
    }
    return true;
}

VkComponentSwizzle ComponentSwizzleFor(std::uint8_t dstSel) {
    switch (dstSel) {
        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
        case 1: return VK_COMPONENT_SWIZZLE_ONE;
        case 4: return VK_COMPONENT_SWIZZLE_R;
        case 5: return VK_COMPONENT_SWIZZLE_G;
        case 6: return VK_COMPONENT_SWIZZLE_B;
        case 7: return VK_COMPONENT_SWIZZLE_A;
        default: throw std::runtime_error("AGC graphics: guest texture descriptor has an invalid destination channel selector " + std::to_string(dstSel));
    }
}

VkComponentMapping ViewComponents(const GuestTextureResource& resource) {
    if (IsConvertedTextureFormat(resource.format) && !IsRepackedTextureFormat(resource.format) && resource.format != static_cast<std::uint32_t>(ShaderRecompiler::IrBufferFormat::Format11_11_10Float)) return {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    const auto channel = [&](std::uint8_t dstSel) { return TextureComponentChannel(resource.format, ComponentSwizzleFor(dstSel)); };
    return {channel(resource.dstSelX), channel(resource.dstSelY), channel(resource.dstSelZ), channel(resource.dstSelW)};
}

// Sampled textures are reused across draws and dispatches while their guest bytes are unchanged; a
// byte copy of the guest surface validates each reuse, so CPU or GPU writes to it force a re-upload.
// The key is everything a lookup matches: the device, the eight descriptor words and the swizzle.
struct AllocationIdentity {
    std::uint64_t rangeAddress = 0;
    std::uint64_t rangeBytes = 0;
    std::uint64_t allocationAddress = 0;
    std::uint64_t allocationBytes = 0;
    std::uint64_t backingSerial = 0;
    std::uint32_t flags = 0;
    bool operator==(const AllocationIdentity&) const = default;
};

AllocationIdentity LookupAllocationIdentity(std::uintptr_t address) {
    const auto range = GuestAllocations::GuestAllocationsLookup_nid_postfix(address, false);
    if (range == nullptr) return {};
    return {range->address, range->bytes, range->allocationAddress, range->allocationBytes, range->backingSerial, (range->readable ? 1u : 0u) | (range->writable ? 2u : 0u) | (range->gpu ? 4u : 0u)};
}

struct AllocationIdentityMemo {
    std::uintptr_t address = 0;
    std::uint64_t generation = 0;
    bool valid = false;
    AllocationIdentity identity;
};

constexpr std::size_t AllocationIdentitySlots = 4096;
struct AllocationIdentityTag {};

AllocationIdentity ImageAllocationIdentity(std::span<const std::uint32_t> words) {
    const auto address = TextureResourceAllocationAddress(words);
    if (!address) return {};
    const auto key = static_cast<std::uintptr_t>(*address);
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    auto& memos = HostThreadLocal<std::unique_ptr<std::array<AllocationIdentityMemo, AllocationIdentitySlots>>, AllocationIdentityTag>();
    if (memos == nullptr) memos = std::make_unique<std::array<AllocationIdentityMemo, AllocationIdentitySlots>>();
    auto& memo = (*memos)[((static_cast<std::uint64_t>(key) * 0x9e3779b97f4a7c15ull) >> 32u) % AllocationIdentitySlots];
    if (memo.valid && memo.address == key && memo.generation == generation) return memo.identity;
    memo.identity = LookupAllocationIdentity(key);
    memo.address = key;
    memo.generation = generation;
    memo.valid = true;
    return memo.identity;
}

struct TextureKey {
    VkDevice device;
    std::array<std::uint32_t, 8> words;
    std::array<std::uint32_t, 4> components;
    bool depthCompare = false;
    AllocationIdentity allocation;
    bool operator==(const TextureKey&) const = default;
};

std::uint64_t hashWords(std::uint64_t hash, const std::uint32_t* words, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) hash = (hash ^ words[i]) * 1099511628211ull;
    return hash;
}

struct TextureKeyHash {
    std::size_t operator()(const TextureKey& key) const noexcept {
        auto hash = 14695981039346656037ull ^ reinterpret_cast<std::uintptr_t>(key.device);
        hash = hashWords(hash, key.words.data(), key.words.size()) ^ static_cast<std::uint64_t>(key.depthCompare);
        for (const auto value : {key.allocation.rangeAddress, key.allocation.rangeBytes, key.allocation.allocationAddress, key.allocation.allocationBytes, key.allocation.backingSerial, static_cast<std::uint64_t>(key.allocation.flags)}) hash = (hash ^ value) * 1099511628211ull;
        return static_cast<std::size_t>(hashWords(hash, key.components.data(), key.components.size()));
    }
};

TextureKey MakeTextureKey(VkDevice device, std::span<const std::uint32_t> words, VkComponentMapping components, bool depthCompare = false) {
    TextureKey key{device, {}, {static_cast<std::uint32_t>(components.r), static_cast<std::uint32_t>(components.g), static_cast<std::uint32_t>(components.b), static_cast<std::uint32_t>(components.a)}};
    key.depthCompare = depthCompare;
    key.allocation = ImageAllocationIdentity(words);
    std::copy(words.begin(), words.end(), key.words.begin());
    key.words[5] &= ~((0xfffu << 8u) | (1u << 25u));
    key.words[6] &= ~0xffu;
    return key;
}

struct CommittedSlabs {
    std::vector<std::uint32_t> slabs;
    std::uint64_t slabBytes = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    bool operator==(const CommittedSlabs& other) const { return slabs == other.slabs && slabBytes == other.slabBytes; }
};

enum class KeepForm : std::uint8_t { None, View, Clear, KeyedClear, Snapshot, Count };
constexpr std::array<const char*, static_cast<std::size_t>(KeepForm::Count)> KeepFormNames{"other", "view of a storage image", "clear-colour texture", "keyed clear snapshot", "snapshot of guest memory"};

struct ParkedTexture {
    std::shared_ptr<Texture> texture;
    KeepForm form = KeepForm::None;
    GuestTextureResource made{};
    VkClearColorValue clear{};
    std::array<std::byte, 16> tail{};
    bool layerRows = false;
    DccKeys keys = DccKeys::Uncompressed;
    std::uint64_t bytes = 0;
    std::shared_ptr<StorageTexture> source;
    std::uint64_t address = 0;
    std::uint64_t present = 0;
    std::uint64_t accounted = 0;
};

struct CachedTexture {
    TextureKey key;
    std::uint64_t address;
    std::vector<std::byte> bytes;
    std::shared_ptr<Texture> texture;
    // A fast-cleared surface is cached as its clear texels; it stays valid while the keys are unchanged.
    DccKeys keys = DccKeys::Uncompressed;
    // Write generation the snapshot is known current at (see GuestMemory::CollectWrites).
    std::uint64_t generation = 0;
    // A texture viewing a storage image on the GPU has no snapshot; it stays valid while that image
    // is still its surface's cached image and nothing wrote the guest memory since it matched.
    std::shared_ptr<StorageTexture> source;
    std::uint64_t sourceVersion = 0;
    std::uint64_t accounted = 0;
    std::uint64_t present = 0;
    std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> committed;
    // The untrackedKey the snapshot was last read or compared equal at, for a surface outside the
    // write watch; zero when none was taken.
    std::array<std::uint64_t, 4> untracked{};
    std::array<std::uint64_t, 4> proved{};
    bool refuted = false;
    std::uint64_t surfaceBytes = 0;
    std::optional<CommittedSlabs> slabs;
    KeepForm form = KeepForm::None;
    GuestTextureResource made{};
    VkClearColorValue madeClear{};
    std::array<std::byte, 16> madeTail{};
    bool madeLayerRows = false;
};

// Entries in use order (front = most recent) with a hash index by key: a lookup is O(1) and the
// eviction takes the back. APS5_NO_TEXTURE_HASH=1 finds entries by scanning the list (the index is
// still kept), the linear lookup of before.
struct TextureCache {
    AgcDriver::Mutex mutex;
    std::list<CachedTexture> entries;
    std::unordered_map<TextureKey, std::list<CachedTexture>::iterator, TextureKeyHash> index;
    std::unordered_multimap<const StorageTexture*, std::list<CachedTexture>::iterator> views;
    std::unordered_map<const Texture*, std::list<CachedTexture>::iterator> owners;
    std::vector<std::pair<std::vector<std::byte>, std::uint64_t>> spare;
    std::uint64_t spareBytes = 0;
    std::uint64_t bytes = 0;
    std::size_t passed = 0;
    std::chrono::steady_clock::time_point passedSince{};
    std::uint64_t firstUsesPresent = 0;
    std::unordered_set<TextureKey, TextureKeyHash> firstUses;
    struct Counts {
        std::uint64_t inserted = 0;
        std::uint64_t uncached = 0;
        std::uint64_t evicted = 0;
        std::uint64_t passed = 0;
        std::uint64_t stale = 0;
    } counts;
    std::chrono::steady_clock::time_point countsReported = std::chrono::steady_clock::now();
    VkPhysicalDevice budgetDevice = VK_NULL_HANDLE;
    std::chrono::steady_clock::time_point budgetRead{};
    std::uint64_t budget = 0;
    std::uint64_t reportedBudget = 0;
    std::uint64_t reportedStorage = 0;
    std::chrono::steady_clock::time_point reportedAt{};
    std::uint64_t erasures = 0;
    std::unordered_map<TextureKey, ParkedTexture, TextureKeyHash> parked;
    std::uint64_t parkedBytes = 0;
    std::uint64_t parkedSweep = 0;
};

TextureCache& Textures() {
    static TextureCache cache;
    return cache;
}

thread_local unsigned sampledCacheHeld = 0;
thread_local unsigned storageCacheHeld = 0;
thread_local unsigned sampledCacheReclaimable = 0;
thread_local unsigned storageCacheReclaimable = 0;

struct CacheHold {
    unsigned& depth;
    explicit CacheHold(unsigned& depth) : depth(depth) { ++depth; }
    ~CacheHold() { --depth; }
    CacheHold(const CacheHold&) = delete;
    CacheHold& operator=(const CacheHold&) = delete;
};

struct UncachedImages {
    AgcDriver::Mutex mutex;
    std::vector<std::weak_ptr<StorageTexture>> images;
    std::atomic<bool> any{false};
};

UncachedImages& Uncached() {
    static UncachedImages uncached;
    return uncached;
}

void noteUncached(const std::shared_ptr<StorageTexture>& image) {
    auto& uncached = Uncached();
    std::lock_guard lock(uncached.mutex);
    uncached.images.push_back(image);
    uncached.any.store(true, std::memory_order_release);
}

std::atomic<bool>& TexturesReleased() {
    static std::atomic<bool> released{false};
    return released;
}

// Moves at every release of a cached texture (the fast proof stamps compare it).
std::atomic<std::uint64_t>& TextureReleaseSerial() {
    static std::atomic<std::uint64_t> serial{1};
    return serial;
}

enum IndexCount : std::size_t { IndexAccepted, IndexPartial, IndexFallback, IndexBumped, IndexLookedUp, IndexRebound, IndexCompared, IndexBuilds, IndexCountCount };

std::array<std::atomic<std::uint64_t>, IndexCountCount>& IndexCounts() {
    static std::array<std::atomic<std::uint64_t>, IndexCountCount> counts{};
    return counts;
}

void countIndex(IndexCount which, std::uint64_t amount = 1) {
    IndexCounts()[which].fetch_add(amount, std::memory_order_relaxed);
}

void noteReleased(const std::shared_ptr<Texture>& texture, std::uint64_t address) {
    texture->MarkReleased();
    TexturesReleased().store(true, std::memory_order_release);
    TextureReleaseSerial().fetch_add(1, std::memory_order_acq_rel);
    GuestMemory::NoteTextureReleased(address);
}

void noteRefilled(std::uint64_t address) {
    TextureReleaseSerial().fetch_add(1, std::memory_order_acq_rel);
    GuestMemory::NoteTextureReleased(address);
}

enum class ReleaseCause : std::uint8_t { Other, UncachedView, Evicted, StaleView, Rewritten, KeysChanged, NowPending, Shape, NotAdmitted, Cleared, Parked, Count };
constexpr std::array<const char*, static_cast<std::size_t>(ReleaseCause::Count)> ReleaseCauseNames{"other", "view of an uncached image", "budget eviction", "stale view", "snapshot rewritten", "DCC keys changed", "now pending in a storage image", "surface size or refuted proof", "not admitted", "device cleared", "parked, then taken back by a refill"};
enum class ReleaseKind : std::uint8_t { Snapshot, Keyed, View, Copy, Count };
constexpr std::array<const char*, static_cast<std::size_t>(ReleaseKind::Count)> ReleaseKindNames{"snapshot", "keyed snapshot or clear", "view", "per-use copy"};
constexpr std::size_t RefillRefusals = static_cast<std::size_t>(Texture::RefillRefusal::Count) + 2u;
constexpr std::size_t RefillRefusedByView = static_cast<std::size_t>(Texture::RefillRefusal::Count);
constexpr std::size_t RefillThrew = RefillRefusedByView + 1u;
constexpr std::array<const char*, RefillRefusals> RefillRefusalNames{"none", "no origin", "descriptor or kind differs", "sparse", "no recorder", "replaced by a view", "refill threw"};

struct TemplateKeepCensus {
    using Counter = std::atomic<std::uint64_t>;
    std::array<std::array<Counter, static_cast<std::size_t>(ReleaseKind::Count)>, static_cast<std::size_t>(ReleaseCause::Count)> released{};
    std::array<Counter, static_cast<std::size_t>(ReleaseCause::Count)> refilled{};
    std::array<Counter, RefillRefusals> refused{};
    Counter refillsVerified{0};
    Counter refillsDiffering{0};
};

TemplateKeepCensus& templateKeepCensus() {
    static auto* census = new TemplateKeepCensus();
    return *census;
}

ReleaseKind releaseKind(const CachedTexture& entry) {
    if (entry.source != nullptr) return entry.texture != nullptr && entry.texture->CopyKind() != Texture::PerUseCopy::None ? ReleaseKind::Copy : ReleaseKind::View;
    return entry.keys != DccKeys::Uncompressed ? ReleaseKind::Keyed : ReleaseKind::Snapshot;
}

void noteReleaseCause(ReleaseCause cause, ReleaseKind kind) {
    if (!ResourceCache::FateCensus()) return;
    templateKeepCensus().released[static_cast<std::size_t>(cause)][static_cast<std::size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

void noteRefill(ReleaseCause cause, std::size_t refusal) {
    if (!ResourceCache::FateCensus()) return;
    auto& census = templateKeepCensus();
    if (refusal == static_cast<std::size_t>(Texture::RefillRefusal::None)) census.refilled[static_cast<std::size_t>(cause)].fetch_add(1, std::memory_order_relaxed);
    else census.refused[std::min(refusal, RefillRefusals - 1u)].fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t TemplateKeepsVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_TEMPLATE_KEEPS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

enum class NotParked : std::uint8_t { OtherForm, SameSide, Full, Count };
constexpr std::array<const char*, static_cast<std::size_t>(NotParked::Count)> NotParkedNames{"a per-use copy or made while no form was recorded", "replaced by the same side (a view by a view, a clear or snapshot by a clear or snapshot)", "park limit reached"};
enum class ReviveRefusal : std::uint8_t { None, OtherForm, Descriptor, OtherImage, ClearValue, Keys, Levels, Tail, Count };
constexpr std::array<const char*, static_cast<std::size_t>(ReviveRefusal::Count)> ReviveRefusalNames{"none", "the other form parked", "descriptor, size or layer rows differ", "another storage image or a per-use copy", "clear value differs", "keys differ", "levels below the minimum LOD", "snapshot tail differs"};
enum class ParkedDrop : std::uint8_t { Superseded, Aged, Uncached, SwitchUnset, Cleared, Unwound, Count };
constexpr std::array<const char*, static_cast<std::size_t>(ParkedDrop::Count)> ParkedDropNames{"superseded by a newer park of its key", "not taken back within 8 presents", "its storage image left the cache", "switch unset", "device cleared", "the lookup threw"};
constexpr std::uint64_t ParkedPresents = 8;
constexpr std::size_t ParkedLimit = 64;
constexpr std::uint64_t ParkedByteLimit = 256ull << 20u;

struct ViewKeepCensus {
    using Counter = std::atomic<std::uint64_t>;
    std::array<Counter, static_cast<std::size_t>(KeepForm::Count)> parked{};
    std::array<Counter, static_cast<std::size_t>(KeepForm::Count)> parkable{};
    std::array<Counter, static_cast<std::size_t>(KeepForm::Count)> revived{};
    std::array<Counter, static_cast<std::size_t>(KeepForm::Count)> refilledBack{};
    std::array<Counter, static_cast<std::size_t>(NotParked::Count)> notParked{};
    std::array<Counter, static_cast<std::size_t>(ReviveRefusal::Count)> refused{};
    std::array<Counter, static_cast<std::size_t>(ParkedDrop::Count)> dropped{};
    Counter revivedNotAdmitted{0};
    Counter revivedUnwound{0};
    Counter verified{0};
    Counter differing{0};
    Counter parkedNow{0};
    Counter parkedNowBytes{0};
    std::atomic<std::uint32_t> namedParks{0};
    std::atomic<std::uint32_t> namedRevivals{0};
};

ViewKeepCensus& viewKeepCensus() {
    static auto* census = new ViewKeepCensus();
    return *census;
}

void countViewKeep(std::atomic<std::uint64_t>& counter) {
    if (ResourceCache::FateCensus()) counter.fetch_add(1, std::memory_order_relaxed);
}

std::atomic<std::uint64_t>& textureRevivals() {
    static std::atomic<std::uint64_t> revivals{0};
    return revivals;
}

std::uint64_t ViewTemplateKeepsVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_VIEW_TEMPLATE_KEEPS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

bool SamplerCacheDisabled() {
    static const bool disabled = std::getenv("APS5_NO_SAMPLER_CACHE") != nullptr;
    return disabled;
}

bool TextureHashEnabled() {
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_HASH") != nullptr;
    return !disabled;
}

std::list<CachedTexture>::iterator findTexture(TextureCache& cache, const TextureKey& key) {
    if (TextureHashEnabled()) {
        const auto found = cache.index.find(key);
        return found == cache.index.end() ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) return it;
    }
    return cache.entries.end();
}

void eraseTexture(TextureCache& cache, std::list<CachedTexture>::iterator it, ReleaseCause cause = ReleaseCause::Other, bool release = true) {
    cache.bytes -= it->accounted;
    if (it->source != nullptr) {
        for (auto [view, last] = cache.views.equal_range(it->source.get()); view != last; ++view) {
            if (view->second == it) {
                cache.views.erase(view);
                break;
            }
        }
    }
    if (release && it->texture.use_count() > 1) {
        noteReleaseCause(cause, releaseKind(*it));
        noteReleased(it->texture, it->address);
    }
    cache.index.erase(it->key);
    cache.owners.erase(it->texture.get());
    cache.entries.erase(it);
    ++cache.erasures;
}

void noteParkedSize(const TextureCache& cache) {
    auto& census = viewKeepCensus();
    census.parkedNow.store(cache.parked.size(), std::memory_order_relaxed);
    census.parkedNowBytes.store(cache.parkedBytes, std::memory_order_relaxed);
}

void dropParked(TextureCache& cache, std::unordered_map<TextureKey, ParkedTexture, TextureKeyHash>::iterator it, ParkedDrop why) {
    auto parked = std::move(it->second);
    cache.parked.erase(it);
    cache.parkedBytes -= std::min(cache.parkedBytes, parked.accounted);
    noteParkedSize(cache);
    countViewKeep(viewKeepCensus().dropped[static_cast<std::size_t>(why)]);
    if (parked.texture == nullptr) return;
    if (parked.texture.use_count() > 1) noteReleased(parked.texture, parked.address);
    parked.texture->MarkParked(false);
}

template<typename Drop>
void dropParkedWhere(TextureCache& cache, ParkedDrop why, Drop&& drop) {
    for (auto it = cache.parked.begin(); it != cache.parked.end();) {
        const auto next = std::next(it);
        if (drop(it->first, it->second)) dropParked(cache, it, why);
        it = next;
    }
}

void dropUncachedViews(TextureCache& cache) {
    auto& uncached = Uncached();
    if (!uncached.any.load(std::memory_order_acquire)) return;
    std::vector<std::weak_ptr<StorageTexture>> images;
    {
        std::lock_guard lock(uncached.mutex);
        images.swap(uncached.images);
        uncached.any.store(false, std::memory_order_release);
    }
    for (const auto& weak : images) {
        const auto image = weak.lock();
        if (image == nullptr) continue;
        for (auto view = cache.views.find(image.get()); view != cache.views.end(); view = cache.views.find(image.get())) eraseTexture(cache, view->second, ReleaseCause::UncachedView);
        if (!cache.parked.empty()) dropParkedWhere(cache, ParkedDrop::Uncached, [&](const TextureKey&, const ParkedTexture& parked) { return parked.source == image; });
    }
}

std::array<std::byte, 16> snapshotTail(std::span<const std::byte> bytes) {
    std::array<std::byte, 16> tail{};
    const auto count = std::min<std::size_t>(bytes.size(), tail.size());
    std::copy(bytes.end() - static_cast<std::ptrdiff_t>(count), bytes.end(), tail.begin());
    return tail;
}

bool noLevelsBelowMinLod(const GuestTextureResource& resource) {
    return resource.baseLevel != 0u || std::min(resource.lastLevel, resource.mipCount - 1u) == 0u || (resource.dimension != TextureDimension::k2D && resource.dimension != TextureDimension::k2DArray);
}

std::uint64_t parkedAccount(const CachedTexture& entry) {
    return entry.form == KeepForm::View ? 0u : entry.texture->AllocationBytes();
}

std::optional<NotParked> parkRefusal(const TextureCache& cache, const CachedTexture& entry, bool viewNext) {
    if (entry.form == KeepForm::None || (entry.form == KeepForm::View && entry.texture->CopyKind() != Texture::PerUseCopy::None)) return NotParked::OtherForm;
    if ((entry.form == KeepForm::View) == viewNext) return NotParked::SameSide;
    const auto existing = cache.parked.find(entry.key);
    const bool replaces = existing != cache.parked.end();
    const std::uint64_t freed = replaces ? existing->second.accounted : 0u;
    if (cache.parked.size() - (replaces ? 1u : 0u) >= ParkedLimit || cache.parkedBytes - std::min(cache.parkedBytes, freed) + parkedAccount(entry) > ParkedByteLimit) return NotParked::Full;
    return std::nullopt;
}

ParkedTexture parkedFrom(const CachedTexture& entry) {
    ParkedTexture parked;
    parked.texture = entry.texture;
    parked.form = entry.form;
    parked.made = entry.made;
    parked.clear = entry.madeClear;
    parked.tail = entry.madeTail;
    parked.layerRows = entry.madeLayerRows;
    parked.keys = entry.keys;
    parked.bytes = entry.surfaceBytes;
    parked.source = entry.source;
    parked.address = entry.address;
    parked.present = Recorder::Presents();
    parked.accounted = parkedAccount(entry);
    return parked;
}

ReviveRefusal reviveRefusal(const ParkedTexture& parked, KeepForm form, const GuestTextureResource& resource, bool layerRows, std::uint64_t bytes, DccKeys keys, const StorageTexture* source, const VkClearColorValue* clear, std::span<const std::byte> snapshot) {
    if (parked.texture == nullptr || parked.form != form) return ReviveRefusal::OtherForm;
    if (!(parked.made == resource) || parked.layerRows != layerRows || parked.bytes != bytes) return ReviveRefusal::Descriptor;
    if (form == KeepForm::View) return parked.source.get() == source && parked.texture->StorageSource() == source && parked.texture->CopyKind() == Texture::PerUseCopy::None ? ReviveRefusal::None : ReviveRefusal::OtherImage;
    if (form == KeepForm::Clear) return clear != nullptr && std::memcmp(&parked.clear, clear, sizeof(VkClearColorValue)) == 0 ? ReviveRefusal::None : ReviveRefusal::ClearValue;
    if (parked.keys != keys) return ReviveRefusal::Keys;
    if (!noLevelsBelowMinLod(resource)) return ReviveRefusal::Levels;
    return snapshotTail(snapshot) == parked.tail ? ReviveRefusal::None : ReviveRefusal::Tail;
}

struct ParkCommit {
    TextureCache& cache;
    const TextureKey& key;
    std::optional<ParkedTexture> stash;
    std::shared_ptr<Texture> revived;
    std::uint64_t revivedAddress = 0;
    int exceptions = -1;
    ParkCommit(TextureCache& cache, const TextureKey& key) : cache(cache), key(key) {}
    ParkCommit(const ParkCommit&) = delete;
    ParkCommit& operator=(const ParkCommit&) = delete;
    void Arm() {
        if (exceptions < 0) exceptions = std::uncaught_exceptions();
    }
    ~ParkCommit() {
        if (exceptions < 0) return;
        if (std::uncaught_exceptions() > exceptions) {
            if (revived != nullptr) {
                noteReleased(revived, revivedAddress);
                countViewKeep(viewKeepCensus().revivedUnwound);
            }
            if (stash.has_value() && stash->texture != nullptr) {
                if (stash->texture.use_count() > 1) noteReleased(stash->texture, stash->address);
                stash->texture->MarkParked(false);
                countViewKeep(viewKeepCensus().dropped[static_cast<std::size_t>(ParkedDrop::Unwound)]);
            }
            return;
        }
        if (!stash.has_value()) return;
        if (const auto found = cache.parked.find(key); found != cache.parked.end()) dropParked(cache, found, ParkedDrop::Superseded);
        cache.parkedBytes += stash->accounted;
        cache.parked.insert_or_assign(key, std::move(*stash));
        noteParkedSize(cache);
    }
};

void sweepParked(TextureCache& cache, bool on) {
    if (cache.parked.empty()) return;
    if (!on) {
        dropParkedWhere(cache, ParkedDrop::SwitchUnset, [](const TextureKey&, const ParkedTexture&) { return true; });
        return;
    }
    const auto present = Recorder::Presents();
    if (cache.parkedSweep == present) return;
    cache.parkedSweep = present;
    dropParkedWhere(cache, ParkedDrop::Aged, [&](const TextureKey&, const ParkedTexture& parked) { return parked.present + ParkedPresents < present; });
}

void nameParking(const GuestTextureResource& resource, KeepForm form, ReleaseCause cause, DccKeys keys, const char* why) {
    if (!ResourceCache::FateCensus()) return;
    auto& census = viewKeepCensus();
    if (census.namedParks.load(std::memory_order_relaxed) >= 8u || census.namedParks.fetch_add(1, std::memory_order_relaxed) >= 8u) return;
    std::fprintf(stderr, "[view-template-keeps] parked the %s of 0x%llx (%ux%u format %u, dcc 0x%llx) at a lookup that found it %s: keys %s, %s\n", KeepFormNames[static_cast<std::size_t>(form)], static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<unsigned long long>(resource.dccAddress), ReleaseCauseNames[static_cast<std::size_t>(cause)], DccKeysName(keys), why);
}

void nameRevival(const GuestTextureResource& resource, KeepForm form, DccKeys keys, std::uint64_t age) {
    if (!ResourceCache::FateCensus()) return;
    auto& census = viewKeepCensus();
    if (census.namedRevivals.load(std::memory_order_relaxed) >= 8u || census.namedRevivals.fetch_add(1, std::memory_order_relaxed) >= 8u) return;
    std::fprintf(stderr, "[view-template-keeps] took back the %s of 0x%llx (%ux%u format %u, dcc 0x%llx, keys %s) parked %llu presents before\n", KeepFormNames[static_cast<std::size_t>(form)], static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<unsigned long long>(resource.dccAddress), DccKeysName(keys), static_cast<unsigned long long>(age));
}

// Moves an entry to the front (most recently used).
void touchTexture(TextureCache& cache, std::list<CachedTexture>::iterator it) {
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
    it->present = Recorder::Presents();
}

constexpr std::uint64_t SpareSnapshotBytes = 256ull << 20;

void expireSpareSnapshots(TextureCache& cache, std::uint64_t present) {
    while (!cache.spare.empty() && present - cache.spare.front().second > SpareSnapshotPresents) {
        cache.spareBytes -= cache.spare.front().first.size();
        cache.spare.erase(cache.spare.begin());
    }
}

void parkSnapshot(TextureCache& cache, std::vector<std::byte>& bytes) {
    const auto size = bytes.size();
    if (size == 0) return;
    const auto present = Recorder::Presents();
    expireSpareSnapshots(cache, present);
    while (!cache.spare.empty() && cache.spareBytes + size > SpareSnapshotBytes) {
        cache.spareBytes -= cache.spare.front().first.size();
        cache.spare.erase(cache.spare.begin());
    }
    if (cache.spareBytes + size <= SpareSnapshotBytes) {
        cache.spareBytes += size;
        cache.spare.emplace_back(std::move(bytes), present);
    }
    bytes = std::vector<std::byte>{};
}

std::vector<std::byte> spareSnapshot(TextureCache& cache, std::size_t bytes) {
    expireSpareSnapshots(cache, Recorder::Presents());
    for (auto spare = cache.spare.begin(); spare != cache.spare.end(); ++spare) {
        if (spare->first.size() != bytes) continue;
        cache.spareBytes -= bytes;
        std::vector<std::byte> taken = std::move(spare->first);
        cache.spare.erase(spare);
        return taken;
    }
    return {};
}

bool admitTexture(TextureCache& cache, const TextureKey& key, std::uint64_t incoming, std::uint64_t budget) {
    if (cache.bytes + incoming <= budget) return true;
    if (const auto now = std::chrono::steady_clock::now(); now - cache.passedSince >= std::chrono::seconds(1)) {
        cache.passed = 0;
        cache.passedSince = now;
    }
    const auto present = Recorder::Presents();
    if (cache.firstUsesPresent != present) {
        cache.firstUses.clear();
        cache.firstUsesPresent = present;
    }
    const bool scan = incoming != 0 && present != 0;
    const bool firstUse = scan && !cache.firstUses.contains(key);
    const auto ceiling = budget + budget / 4u;
    while (!cache.entries.empty() && cache.bytes + incoming > budget) {
        const auto back = std::prev(cache.entries.end());
        const bool over = cache.bytes + incoming > ceiling;
        if (scan && !over && (firstUse ? back->present + 1u >= present : back->present == present)) {
            cache.firstUses.insert(key);
            ++cache.counts.uncached;
            return false;
        }
        const bool keep = back->accounted == 0 || (!over && back->texture.use_count() > 1);
        if (keep && cache.passed < cache.entries.size()) {
            cache.entries.splice(cache.entries.begin(), cache.entries, back);
            ++cache.passed;
            ++cache.counts.passed;
        } else if (keep && !over) {
            break;
        } else {
            eraseTexture(cache, back, ReleaseCause::Evicted);
            ++cache.counts.evicted;
        }
    }
    return true;
}

void reportCacheCounts(TextureCache& cache, std::uint64_t budget) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - cache.countsReported < std::chrono::seconds(10)) return;
    cache.countsReported = now;
    const auto counts = std::exchange(cache.counts, {});
    AgcDriver::ProfilePrint_nid_no_patch("[texture-cache] (10 s): %llu textures cached, %llu used uncached (first use this frame with the least recently used entry used this frame or the last, or a later use with it used this frame), %llu evicted, %llu held ones passed over, %llu found changed; %llu MiB in %zu entries, budget %llu MiB, frame %llu\n", static_cast<unsigned long long>(counts.inserted), static_cast<unsigned long long>(counts.uncached), static_cast<unsigned long long>(counts.evicted), static_cast<unsigned long long>(counts.passed), static_cast<unsigned long long>(counts.stale), static_cast<unsigned long long>(cache.bytes >> 20u), cache.entries.size(), static_cast<unsigned long long>(budget >> 20u), static_cast<unsigned long long>(Recorder::Presents()));
}

struct ImageMemoEntry {
    VkDevice device = VK_NULL_HANDLE;
    std::array<std::uint32_t, 8> words{};
    GuestTextureResource resource{};
    std::uint64_t guestBytes = 0;
    AllocationIdentity allocation;
    VkComponentMapping components{};
    bool hasEntry = false;
    std::list<CachedTexture>::iterator entry{};
    std::uint64_t erasures = 0;
};

constexpr std::size_t ImageMemoSlots = 1024;
struct ImageMemoTag {};

ImageMemoEntry& imageMemoSlot(std::span<const std::uint32_t> words) {
    auto& imageMemo = HostThreadLocal<std::unique_ptr<std::array<ImageMemoEntry, ImageMemoSlots>>, ImageMemoTag>();
    if (imageMemo == nullptr) imageMemo = std::make_unique<std::array<ImageMemoEntry, ImageMemoSlots>>();
    const auto hash = hashWords(14695981039346656037ull, words.data(), words.size());
    return (*imageMemo)[(hash ^ (hash >> 29u)) % ImageMemoSlots];
}

struct SampledKeyProofEntry {
    std::uint64_t dccAddress = 0;
    std::uint64_t guestBytes = 0;
    std::uint32_t format = 0;
    bool dccAlphaOnMsb = false;
    DccKeyProof proof;
};

constexpr std::size_t SampledKeyProofSlots = 1024;
struct SampledKeyProofTag {};

SampledKeyProofEntry& sampledKeyProofSlot(std::uint64_t dccAddress, std::uint64_t guestBytes, std::uint32_t format, bool dccAlphaOnMsb) {
    auto& proofs = HostThreadLocal<std::unique_ptr<std::array<SampledKeyProofEntry, SampledKeyProofSlots>>, SampledKeyProofTag>();
    if (proofs == nullptr) proofs = std::make_unique<std::array<SampledKeyProofEntry, SampledKeyProofSlots>>();
    const auto hash = dccAddress ^ (guestBytes * 0x9e3779b97f4a7c15ull) ^ (static_cast<std::uint64_t>(format) << 32u) ^ (dccAlphaOnMsb ? 0x94d049bb133111ebull : 0ull);
    return (*proofs)[(hash ^ (hash >> 29u)) % SampledKeyProofSlots];
}

const ImageMemoEntry& decodedImage(VkDevice device, std::span<const std::uint32_t> words) {
    auto& slot = imageMemoSlot(words);
    const auto allocation = ImageAllocationIdentity(words);
    if (slot.device == device && slot.allocation == allocation && std::equal(words.begin(), words.end(), slot.words.begin())) return slot;
    slot.device = VK_NULL_HANDLE;
    slot.resource = DecodeTextureResource(words);
    slot.guestBytes = SurfaceGuestBytes(slot.resource);
    slot.allocation = allocation;
    slot.components = ViewComponents(slot.resource);
    std::copy(words.begin(), words.end(), slot.words.begin());
    slot.hasEntry = false;
    slot.device = device;
    return slot;
}

std::list<CachedTexture>::iterator findSampledEntry(TextureCache& cache, VkDevice device, const std::array<std::uint32_t, 8>& words, VkComponentMapping components) {
    auto& slot = imageMemoSlot(words);
    const bool same = slot.device == device && slot.allocation == ImageAllocationIdentity(words) && slot.words == words && slot.components.r == components.r && slot.components.g == components.g && slot.components.b == components.b && slot.components.a == components.a;
    if (same && slot.hasEntry && slot.erasures == cache.erasures) return slot.entry;
    const auto it = findTexture(cache, MakeTextureKey(device, words, components));
    if (same) {
        slot.hasEntry = it != cache.entries.end();
        slot.entry = it;
        slot.erasures = cache.erasures;
    }
    return it;
}

// APS5_PROFILE_DRAW: what the sampled-texture and storage-image lookups did, printed as [textures]
// every 10 s (cumulative). Storage-sourced sampled textures need no CPU read; snapshots do, and a
// pending read is a snapshot compare or read made while recorded work still wrote the surface (the
// flush hook may find that work already signaled, so this bounds the hook syncs from above; the
// [hooksync] line attributes the actual waits).
struct TextureCounters {
    std::atomic<std::uint64_t> fromStorage{0};
    std::atomic<std::uint64_t> snapshots{0};
    std::atomic<std::uint64_t> cleared{0};
    std::atomic<std::uint64_t> imported{0};
    std::atomic<std::uint64_t> pendingReads{0};
    std::atomic<std::uint64_t> storageFallbacks{0};
    std::atomic<std::uint64_t> records{0};
    std::atomic<std::uint64_t> fastHits{0};
    std::atomic<std::uint64_t> fastMisses{0};
    std::atomic<std::uint64_t> storageHits{0};
    std::atomic<std::uint64_t> storageCreated{0};
    // Storage images a Revalidate refreshed directly instead of through the lookups (T1, see
    // ShaderResources::refreshOwnObjects).
    std::atomic<std::uint64_t> ownRefreshes{0};
    std::atomic<std::uint64_t> tablesReused{0};
    std::atomic<std::uint64_t> tablesReplayed{0};
    std::atomic<std::uint64_t> tablesResolved{0};
    std::atomic<std::uint64_t> tableWritesKept{0};
    std::atomic<std::uint64_t> tableWritesMade{0};
    std::atomic<std::uint64_t> variantSurfaces{0};
    std::atomic<std::uint64_t> variantSubstituted{0};
    std::atomic<std::uint64_t> variantRejected{0};
    std::atomic<std::uint64_t> arrayViewsShared{0};
    // Snapshot hits over memory outside the write watch: kept by an unchanged UntrackedHint key, or
    // by a byte compare.
    std::atomic<std::uint64_t> untrackedKept{0};
    std::atomic<std::uint64_t> untrackedCompared{0};
    std::atomic<std::int64_t> lastReport{0};
};

TextureCounters& TextureCounts() {
    static TextureCounters counters;
    return counters;
}

thread_local std::uint64_t sampledLookupKeyScans = 0;
thread_local std::uint64_t sampledLookupKeyProofs = 0;

bool TextureCountersReported() {
    static const bool reported = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return reported;
}

void reportTextureCounters() {
    if (!TextureCountersReported()) return;
    auto& counters = TextureCounts();
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto count = [](const std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.load(std::memory_order_relaxed)); };
    AgcDriver::ProfilePrint_nid_no_patch("[textures] sampled created: %llu from storage images, %llu cleared on the GPU, %llu uploaded on the GPU from host imports, %llu snapshots (%llu snapshot reads over recorded writes inside cachedTexture, %llu storage-path fallbacks); stage-A records %llu: %llu fast hits, %llu full lookups; storage images %llu hits, %llu created, %llu own-object refreshes; image tables %llu reused, %llu resolved; dynamic-resolution variants: %llu surfaces, %llu entries substituted, %llu entries rejected; %llu one-layer 2D array entries bound to the 2D texture of their surface; untracked snapshots %llu kept by the write-watch hint, %llu compared\n", count(counters.fromStorage), count(counters.cleared), count(counters.imported), count(counters.snapshots), count(counters.pendingReads), count(counters.storageFallbacks), count(counters.records), count(counters.fastHits), count(counters.fastMisses), count(counters.storageHits), count(counters.storageCreated), count(counters.ownRefreshes), count(counters.tablesReused), count(counters.tablesResolved), count(counters.variantSurfaces), count(counters.variantSubstituted), count(counters.variantRejected), count(counters.arrayViewsShared), count(counters.untrackedKept), count(counters.untrackedCompared));
}

// What the sampled-texture lookups on this thread proved their returned objects current against,
// taken by object into the ShaderResources being built or revalidated (captureValidation), which
// then repeats the proof from write stamps on its next Revalidate. The lookups run inside one build
// on one thread; records nobody takes (a build that threw, a resident target looked up outside a
// build, a thread that never captures) are dropped by the next capture or by the bound below.
struct LookupRecord {
    const void* object;
    GuestTextureResource resource;
    std::uint64_t bytes;
    DccKeys keys;
    std::uint64_t generation;
    const StorageTexture* source;
    bool depth = false;
    std::uint64_t committedGeneration = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> committed;
};

thread_local std::vector<LookupRecord>* lookupLogSlot = nullptr;
constexpr std::size_t LookupLogBound = 1024;
constexpr std::size_t NoBuildLookups = std::numeric_limits<std::size_t>::max();
thread_local std::size_t buildLookupsStart = NoBuildLookups;
thread_local bool buildLookupsPastBound = false;
std::atomic<std::uint64_t> lookupBuildsPastBound{0};

std::vector<LookupRecord>& lookupLog() {
    return ShaderRecompiler::ThreadOwned(lookupLogSlot);
}

void logLookup(const LookupRecord& record) {
    auto& log = lookupLog();
    if (log.size() >= LookupLogBound) {
        const auto dropped = std::min(LookupLogBound / 2, buildLookupsStart);
        if (dropped < LookupLogBound / 2) buildLookupsPastBound = true;
        log.erase(log.begin(), log.begin() + static_cast<std::ptrdiff_t>(dropped));
        if (buildLookupsStart != NoBuildLookups) buildLookupsStart -= dropped;
    }
    log.push_back(record);
}

class BuildLookups {
public:
    BuildLookups() : outermost(buildLookupsStart == NoBuildLookups) { buildLookupsStart = std::min(buildLookupsStart, lookupLog().size()); }
    ~BuildLookups() {
        if (!outermost) return;
        buildLookupsStart = NoBuildLookups;
        if (buildLookupsPastBound) lookupBuildsPastBound.fetch_add(1, std::memory_order_relaxed);
        buildLookupsPastBound = false;
        auto& log = lookupLog();
        if (log.size() > LookupLogBound) log.erase(log.begin(), log.end() - static_cast<std::ptrdiff_t>(LookupLogBound));
        if (log.capacity() > LookupLogBound + LookupLogBound / 2) log.shrink_to_fit();
    }
    BuildLookups(const BuildLookups&) = delete;
    BuildLookups& operator=(const BuildLookups&) = delete;

private:
    bool outermost;
};

void refreshForView(StorageTexture& source, const GuestTextureResource& resource) {
    if (resource.dimension == TextureDimension::k2D && source.ImageLayers() > 1) source.RefreshLayers(resource.baseArray, 1);
    else source.Refresh();
}

std::shared_ptr<StorageTexture> cachedStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip, std::uint64_t guestBytes = 0, bool written = false, std::optional<std::pair<std::uint32_t, std::uint32_t>> layers = std::nullopt);

std::uint64_t heldBytes(const StorageTexture& image) {
    return image.BudgetBytes();
}

bool MetadataMoved(const StorageTexture& image, const GuestTextureResource& resource) {
    return resource.dccAddress != 0 && image.Descriptor().dccAddress != resource.dccAddress && !image.ServesKeysAt(resource.dccAddress);
}

// Whether a sampled texture over `resource` can be a view of the surface's cached storage image
// instead of a CPU snapshot (see cachedTexture): the format has a storage form and is not block
// compressed, and the surface lives in host-imported memory, where the image uploads and refreshes
// GPU-direct (a snapshot of such a surface reads and compares its bytes on the CPU, waiting for
// the recorded work that wrote them). APS5_NO_SAMPLED_FROM_STORAGE=1 keeps every sampled texture a
// snapshot as before.
std::atomic<bool> sampledFromStorageForTest{true};

bool SampledFromStorageEligible(const Context& context, std::uint32_t format, std::uint64_t address, std::uint64_t guestBytes) {
    static const bool disabled = std::getenv("APS5_NO_SAMPLED_FROM_STORAGE") != nullptr;
    if (disabled || !sampledFromStorageForTest.load(std::memory_order_relaxed) || IsBlockCompressed(format) || !StorageFormatAvailable(context, format)) return false;
    return HostImportCovers(context, address, static_cast<std::size_t>(guestBytes));
}

bool SampledFromStorageEligible(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    return SampledFromStorageEligible(context, resource.format, resource.baseAddress, guestBytes);
}

// Whether a sampled texture over a fast-cleared, storage-eligible surface views the cleared storage
// image (see cachedTexture). APS5_NO_CLEARED_VIEW=1 keeps such surfaces snapshots, as before.
bool ClearedViewEnabled() {
    static const bool disabled = std::getenv("APS5_NO_CLEARED_VIEW") != nullptr;
    return !disabled;
}

std::optional<VkClearColorValue> SampledClearColor(const Context& context, const GuestTextureResource& resource, DccKeys keys, bool depthCompare) {
    if (depthCompare || !IsDccClear(keys) || keys == DccKeys::ClearRegister) return std::nullopt;
    if (IsBlockCompressed(resource.format) || IsRepackedTextureFormat(resource.format) || SrgbExpandsToRgba8(context, resource.format)) return std::nullopt;
    const auto format = SampledTextureFormat(context, resource.format);
    if (format != DccClearTextureFormat(resource.format)) return std::nullopt;
    std::array<std::uint32_t, 4> pattern{};
    if (!FillDccClear(format, keys, resource.dccAlphaOnMsb, std::as_writable_bytes(std::span(pattern)))) return std::nullopt;
    VkClearColorValue clear{};
    if (!ClearColorForTexel(format, BytesPerElement(resource.format), pattern, clear)) return std::nullopt;
    return clear;
}

std::uint64_t DepthCopyMinBytes() {
    static const std::uint64_t bytes = [] {
        const char* text = std::getenv("APS5_DEPTH_COPY_MIN_KIB");
        return (text != nullptr ? std::strtoull(text, nullptr, 10) : 1024ull) << 10u;
    }();
    return bytes;
}

// Surfaces whose storage image could not be made (no writable committed pages, say): remembered
// so the sampled lookup does not throw and fall back on every use. A failure is keyed by the surface
// (address and size: the heap reuses addresses) and holds only while the guest mappings are what
// they were when it failed (the allocation generation): pages committed later, or another surface
// at the address, get a fresh attempt.
struct StorageFailures {
    AgcDriver::Mutex mutex;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> generations;
};

StorageFailures& StorageFailed() {
    static StorageFailures failures;
    return failures;
}

// The cached storage image for a sampled descriptor, or null when it cannot be made (then the
// snapshot path serves the descriptor, as before). Every null return counts as a fallback.
std::shared_ptr<StorageTexture> sampledStorageSource(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    auto& counters = TextureCounts();
    auto& failures = StorageFailed();
    const auto surface = std::make_pair(resource.baseAddress, guestBytes);
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    {
        std::lock_guard lock(failures.mutex);
        if (const auto it = failures.generations.find(surface); it != failures.generations.end()) {
            if (it->second == generation) {
                counters.storageFallbacks.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            }
            failures.generations.erase(it);
        }
    }
    try {
        return cachedStorageTexture(context, {}, resource, 0, guestBytes);
    } catch (const std::exception& error) {
        counters.storageFallbacks.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lock(failures.mutex);
        if (failures.generations.size() < 4096 && failures.generations.emplace(surface, generation).second) std::fprintf(stderr, "[textures] sampled texture 0x%llx (%ux%u format %u) keeps the snapshot path: %s\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, error.what());
        return nullptr;
    }
}

std::uint64_t storageCacheBytes();

std::optional<std::uint32_t> largestDeviceLocalHeap(const VkPhysicalDeviceMemoryProperties& memory) {
    std::optional<std::uint32_t> heap;
    for (std::uint32_t i = 0; i < std::min<std::uint32_t>(memory.memoryHeapCount, VK_MAX_MEMORY_HEAPS); ++i) {
        if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 && (!heap || memory.memoryHeaps[i].size > memory.memoryHeaps[*heap].size)) heap = i;
    }
    return heap;
}

std::atomic<std::uint64_t> budgetOverride{0};

std::uint64_t forcedCacheBudget() {
    if (const auto overridden = budgetOverride.load(std::memory_order_relaxed); overridden != 0) return overridden;
    static const std::uint64_t forced = [] {
        const char* text = std::getenv("APS5_TEXTURE_CACHE_MIB");
        return text != nullptr ? std::strtoull(text, nullptr, 10) << 20u : 0u;
    }();
    return forced;
}

std::uint64_t storageBudget(const Context& context, std::uint64_t required = 0, std::optional<std::chrono::steady_clock::time_point>* calculatedAt = nullptr) {
    if (calculatedAt != nullptr) calculatedAt->reset();
    const auto forced = forcedCacheBudget();
    if (forced != 0) return forced;
    if (context.memoryProperties2 == nullptr || context.physical == VK_NULL_HANDLE) return TextureCacheBudget(context.memory);
    static struct {
        AgcDriver::Mutex mutex;
        VkPhysicalDevice device{VK_NULL_HANDLE};
        std::chrono::steady_clock::time_point read{};
        std::uint64_t budget{0};
        std::uint64_t ceiling{0};
        std::chrono::steady_clock::time_point calculatedAt{};
    } cached;
    const auto now = std::chrono::steady_clock::now();
    std::optional<VkPhysicalDeviceMemoryBudgetPropertiesEXT> reported;
    StorageTextureBudgetSnapshot snapshot;
    std::uint64_t budget;
    {
        std::lock_guard lock(cached.mutex);
        const auto read = [&] {
            reported.emplace(VkPhysicalDeviceMemoryBudgetPropertiesEXT{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT});
            VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
            properties.pNext = &*reported;
            context.memoryProperties2(context.physical, &properties);
            cached.device = context.physical;
            cached.read = now;
            const auto sampled = SampledTextureMemory();
            const auto storage = storageCacheBytes();
            const auto owned = UsageWithoutHostImports(context.memory, *reported, HostImportLiveBytes(), context.hostImportsInHeapUsage);
            snapshot = CalculateStorageTextureBudget(context.memory, &owned, sampled, storage);
            cached.budget = snapshot.selectedBudget;
            cached.ceiling = snapshot.ceiling;
            cached.calculatedAt = std::chrono::steady_clock::now();
        };
        if (cached.device != context.physical || now - cached.read >= std::chrono::seconds(1)) read();
        const auto needed = std::min(required, cached.ceiling);
        if (needed > cached.budget && context.bufferPool != nullptr && context.bufferPool->ReclaimIdleDevice(needed - cached.budget, now) != 0) read();
        budget = cached.budget;
        if (calculatedAt != nullptr) *calculatedAt = cached.calculatedAt;
    }
    if (reported) MemoryAccounting::Report(context, *reported, snapshot);
    return budget;
}

std::size_t storageEntries() {
    static const std::size_t entries = [] {
        const char* text = std::getenv("APS5_STORAGE_CACHE_ENTRIES");
        if (text == nullptr || *text == '\0') return std::size_t{2048};
        const auto parsed = std::strtoull(text, nullptr, 10);
        return parsed != 0 ? static_cast<std::size_t>(parsed) : std::numeric_limits<std::size_t>::max();
    }();
    return entries;
}

std::uint64_t sampledBudget(const Context& context, TextureCache& cache) {
    if (const auto forced = forcedCacheBudget(); forced != 0) return forced;
    if (context.memoryProperties2 == nullptr || context.physical == VK_NULL_HANDLE) return TextureCacheBudget(context.memory);
    const auto now = std::chrono::steady_clock::now();
    if (cache.budgetDevice == context.physical && now - cache.budgetRead < std::chrono::seconds(1)) return cache.budget;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    context.memoryProperties2(context.physical, &properties);
    const auto sampled = SampledTextureMemory();
    const auto storage = storageCacheBytes();
    const auto owned = UsageWithoutHostImports(context.memory, reported, HostImportLiveBytes(), context.hostImportsInHeapUsage);
    const auto budget = SampledTextureBudget(context.memory, &owned, sampled + storage, storage);
    const bool fresh = cache.budgetDevice != context.physical;
    std::optional<std::chrono::steady_clock::time_point> storageCalculatedAt;
    const auto storageLimit = storageBudget(context, 0, &storageCalculatedAt);
    if (const auto heap = largestDeviceLocalHeap(context.memory); heap && (fresh || SampledBudgetReportDue(cache.reportedBudget, budget, now - cache.reportedAt) || SampledBudgetReportDue(cache.reportedStorage, storageLimit, now - cache.reportedAt))) {
        char was[48] = "";
        char wasStorage[48] = "";
        char storageLimitAge[32] = "unavailable";
        if (storageCalculatedAt) {
            const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - *storageCalculatedAt);
            std::snprintf(storageLimitAge, sizeof(storageLimitAge), "%llu", static_cast<unsigned long long>(age.count()));
        }
        if (!fresh) {
            std::snprintf(was, sizeof(was), " (was %llu MiB)", static_cast<unsigned long long>(cache.reportedBudget >> 20u));
            std::snprintf(wasStorage, sizeof(wasStorage), " (was %llu MiB)", static_cast<unsigned long long>(cache.reportedStorage >> 20u));
        }
        std::fprintf(stderr, "[gpu] sampled texture cache budget %llu MiB%s: device-local heap %u has a VK_EXT_memory_budget budget of %llu MiB and uses %llu MiB, %llu MiB of it sampled textures (%llu MiB in the cache) and %llu MiB cached storage images; active storage image cache budget %llu MiB%s; storage_limit_age_ms=%s\n", static_cast<unsigned long long>(budget >> 20u), was, *heap, static_cast<unsigned long long>(reported.heapBudget[*heap] >> 20u), static_cast<unsigned long long>(reported.heapUsage[*heap] >> 20u), static_cast<unsigned long long>(sampled >> 20u), static_cast<unsigned long long>(cache.bytes >> 20u), static_cast<unsigned long long>(storage >> 20u), static_cast<unsigned long long>(storageLimit >> 20u), wasStorage, storageLimitAge);
        cache.reportedBudget = budget;
        cache.reportedStorage = storageLimit;
        cache.reportedAt = now;
    }
    cache.budgetDevice = context.physical;
    cache.budgetRead = now;
    cache.budget = budget;
    return budget;
}

// A snapshot over memory outside the write watch is compared with guest memory again only when this
// key changed: a write-watch mark in the range, a driver store, a mapping change or a recorded GPU
// write (as StorageTexture::compareUntracked). Taken before the snapshot is read or compared, so a
// CPU store racing with that leaves a mark for the next key. Zero when the range has no hint.
std::array<std::uint64_t, 4> untrackedKey(std::uint64_t address, std::size_t bytes) {
    if (GuestMemory::CollectEpoch() == 0) return {};
    const auto hint = GuestMemory::UntrackedHint(address, bytes);
    if (hint == 0) return {};
    const auto* recorder = Recorder::Active();
    const std::array<std::uint64_t, 4> key{hint, GuestMemory::NewestStamp(address, bytes), GuestMemory::ForgetSerial(), recorder != nullptr ? recorder->NewestWriteNote(address, bytes) : 0};
    return (key[2] & 1u) == 0 ? key : std::array<std::uint64_t, 4>{};
}

std::array<std::uint64_t, 4> untrackedRangeStamp(std::uint64_t address, std::size_t bytes);

bool UntrackedProofsVerified() {
    static const bool verify = std::getenv("APS5_VERIFY_UNTRACKED_PROOFS") != nullptr;
    return verify;
}

bool UntrackedReproofEnabled() {
    static const bool disabled = std::getenv("APS5_NO_UNTRACKED_REPROOF") != nullptr;
    return !disabled;
}

struct UntrackedReproofCounters {
    std::atomic<std::uint64_t> stamped{0};
    std::atomic<std::uint64_t> compared{0};
};

UntrackedReproofCounters& ReproofCounters() {
    static UntrackedReproofCounters counters;
    return counters;
}

std::span<const std::uint32_t> firstLayerWords(std::span<const std::uint32_t> words, std::optional<ShaderRecompiler::DescriptorImageShape> shape, std::array<std::uint32_t, 8>& narrowed, bool report) {
    if (shape != ShaderRecompiler::DescriptorImageShape::Image2D || words.size() != narrowed.size() || IsNullSampledTextureResource(words)) return words;
    try {
        if (!FirstLayerDescriptor(words, narrowed)) return words;
    } catch (const std::runtime_error&) {
        return words;
    }
    if (report) {
        static std::atomic<std::uint64_t> narrowings{0};
        const auto count = narrowings.fetch_add(1) + 1;
        if (count <= 5 || (count & (count - 1)) == 0) {
            const auto resource = DecodeTextureResource(words);
            std::fprintf(stderr, "[gpu] 2D read of the 2D array 0x%llx (%ux%u format %u, slices %u..%u) built from slice %u alone; %llu such reads\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, resource.baseArray, resource.depthOrLastArray, resource.baseArray, static_cast<unsigned long long>(count));
        }
    }
    return narrowed;
}

bool layerRowsBinding(std::optional<ShaderRecompiler::DescriptorImageShape> shape, const GuestTextureResource& resource, bool depthCompare) {
    return !depthCompare && shape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k1DArray;
}

void reportGuestAllocation(const GuestTextureResource& resource, std::uint64_t guestBytes) {
    const auto range = GuestAllocations::GuestAllocationsLookup_nid_postfix(resource.baseAddress, false);
    const auto geometry = DescribeSurface(resource);
    if (range == nullptr) {
        std::fprintf(stderr, "[gpu] view 0x%llx of %llu MiB lies in no registered guest allocation\n", static_cast<unsigned long long>(resource.baseAddress), static_cast<unsigned long long>(guestBytes >> 20u));
        return;
    }
    const auto end = range->address + range->bytes;
    const auto fitting = geometry.thick || geometry.layerBytes == 0 ? 0ull : std::min<std::uint64_t>(geometry.layers, (end - resource.baseAddress) / geometry.layerBytes);
    std::fprintf(stderr, "[gpu] view 0x%llx of %llu MiB lies in the guest allocation 0x%llx..0x%llx (%llu MiB, range 0x%llx..0x%llx, %s%s%s): %llu of its %u slices fit in the range\n", static_cast<unsigned long long>(resource.baseAddress), static_cast<unsigned long long>(guestBytes >> 20u), static_cast<unsigned long long>(range->allocationAddress), static_cast<unsigned long long>(range->allocationAddress + range->allocationBytes), static_cast<unsigned long long>(range->allocationBytes >> 20u), static_cast<unsigned long long>(range->address), static_cast<unsigned long long>(end), range->readable ? "r" : "-", range->writable ? "w" : "-", range->gpu ? "g" : "-", static_cast<unsigned long long>(fitting), geometry.layers);
}

std::optional<CommittedSlabs> committedSlabs(const GuestTextureResource& resource, std::uint64_t guestBytes, const std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>>& committed) {
    if (resource.dimension != TextureDimension::k3D) return std::nullopt;
    const auto geometry = DescribeSurface(resource);
    const auto slabBytes = geometry.layerBytes;
    if (slabBytes == 0 || geometry.layers == 0 || geometry.guestBytes != guestBytes || geometry.GuestLayerOffset(geometry.layers - 1u) + slabBytes != guestBytes) return std::nullopt;
    const auto described = committed.has_value() ? GuestMemory::Commitment{} : GuestMemory::DescribeCommitted(resource.baseAddress, static_cast<std::size_t>(guestBytes));
    const auto& ranges = committed.has_value() ? *committed : described.ranges;
    CommittedSlabs result{{}, slabBytes};
    const auto count = guestBytes / slabBytes;
    auto range = ranges.begin();
    for (std::uint64_t slab = 0; slab < count; ++slab) {
        const auto begin = resource.baseAddress + slab * slabBytes;
        while (range != ranges.end() && range->second <= begin) ++range;
        if (range != ranges.end() && range->first < begin + slabBytes) result.slabs.push_back(static_cast<std::uint32_t>(slab));
    }
    if (result.slabs.size() == count) return std::nullopt;
    const auto end = resource.baseAddress + guestBytes;
    for (const auto& [first, last] : ranges) {
        if (last > resource.baseAddress && first < end) result.ranges.emplace_back(std::max(first, resource.baseAddress) - resource.baseAddress, std::min(last, end) - resource.baseAddress);
    }
    return result;
}

bool stillCommitted(const std::optional<CommittedSlabs>& now, const std::optional<CommittedSlabs>& snapshot) {
    if (!now.has_value() || !snapshot.has_value()) return now.has_value() == snapshot.has_value();
    auto range = now->ranges.begin();
    for (const auto& [begin, end] : snapshot->ranges) {
        while (range != now->ranges.end() && range->second <= begin) ++range;
        if (range == now->ranges.end() || range->first > begin || range->second < end) return false;
    }
    return true;
}

std::size_t snapshotBytes(const std::optional<CommittedSlabs>& slabs, std::size_t guestBytes) {
    return slabs.has_value() ? static_cast<std::size_t>(slabs->slabs.size() * slabs->slabBytes) : guestBytes;
}

std::optional<std::span<const std::uint32_t>> packedSlabs(const std::optional<CommittedSlabs>& slabs) {
    if (!slabs.has_value()) return std::nullopt;
    return std::span<const std::uint32_t>(slabs->slabs);
}

std::vector<std::array<std::uint64_t, 3>> slabRuns(const CommittedSlabs& committed) {
    std::vector<std::array<std::uint64_t, 3>> runs;
    for (std::size_t index = 0; index < committed.slabs.size(); ++index) {
        const auto guest = std::uint64_t{committed.slabs[index]} * committed.slabBytes;
        if (!runs.empty() && runs.back()[0] + runs.back()[2] == guest) runs.back()[2] += committed.slabBytes;
        else runs.push_back({guest, index * committed.slabBytes, committed.slabBytes});
    }
    return runs;
}

void readSnapshot(const GuestTextureResource& resource, DccKeys keys, const std::optional<CommittedSlabs>& slabs, std::span<std::byte> bytes) {
    if (!slabs.has_value()) return ReadTextureSurface(resource, keys, bytes);
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::TextureRead);
    for (const auto& [guest, packed, size] : slabRuns(*slabs)) GuestMemory::ReadCommitted(resource.baseAddress + guest, bytes.subspan(static_cast<std::size_t>(packed), static_cast<std::size_t>(size)));
}

bool equalsCommittedSlabs(std::uint64_t address, const CommittedSlabs& slabs, std::span<const std::byte> bytes) {
    for (const auto& [guest, packed, size] : slabRuns(slabs)) {
        if (!GuestMemory::EqualsCommitted(address + guest, bytes.subspan(static_cast<std::size_t>(packed), static_cast<std::size_t>(size)))) return false;
    }
    return true;
}

void reportCommittedSlabs(const GuestTextureResource& resource, const CommittedSlabs& slabs, std::uint64_t guestBytes) {
    static std::atomic<std::uint64_t> views{0};
    const auto count = views.fetch_add(1) + 1;
    if (count > 5 && (count & (count - 1)) != 0) return;
    const auto geometry = DescribeSurface(resource);
    std::uint32_t zeroed = 0;
    for (std::uint32_t layer = 0; layer < geometry.layers; ++layer) zeroed += std::ranges::binary_search(slabs.slabs, geometry.GuestLayerOffset(layer) / slabs.slabBytes) ? 0u : 1u;
    std::fprintf(stderr, "[gpu] sampled view 0x%llx (%ux%u format %u, dimension %d, tile %d, slices %u..%u) uploads the %zu of its %llu slabs holding committed guest memory (%llu of %llu MiB) and %u zero-filled slices; %llu such views\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.dimension), static_cast<int>(resource.tileMode), resource.baseArray, resource.depthOrLastArray, slabs.slabs.size(), static_cast<unsigned long long>(guestBytes / slabs.slabBytes), static_cast<unsigned long long>((slabs.slabs.size() * slabs.slabBytes) >> 20u), static_cast<unsigned long long>(guestBytes >> 20u), zeroed, static_cast<unsigned long long>(count));
}

struct PendingLookup {
    std::shared_ptr<StorageTexture> source;
    bool depthCopy = false;
    bool srgbExpand = false;
    bool blockCopy = false;
    bool repackCopy = false;
};

PendingLookup pendingLookupSource(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes, bool depthCompare, bool layerRows = false) {
    PendingLookup lookup;
    lookup.depthCopy = depthCompare && guestBytes >= DepthCopyMinBytes() && Texture::DepthCopyFormat(ResolveTextureFormat(resource.format)) != VK_FORMAT_UNDEFINED && resource.dimension != TextureDimension::k3D;
    lookup.srgbExpand = !depthCompare && SrgbExpandsToRgba8(context, resource.format);
    lookup.source = StorageTexture::FindPending(resource.baseAddress, guestBytes);
    if (layerRows) lookup.source.reset();
    auto& source = lookup.source;
    lookup.blockCopy = source != nullptr && !depthCompare && !lookup.srgbExpand && Texture::CanBlockCopyFrom(*source, resource);
    lookup.repackCopy = source != nullptr && !depthCompare && Texture::CanRepackCopyFrom(*source, resource);
    if (source != nullptr && !lookup.blockCopy && !lookup.repackCopy && (depthCompare ? !lookup.depthCopy || !Texture::CanDepthCopyFrom(*source, resource) : lookup.srgbExpand ? !Texture::CanSrgbExpandFrom(context, *source, resource) : !Texture::CanCopyFrom(*source, resource))) source.reset();
    if (source != nullptr && MetadataMoved(*source, resource)) source.reset();
    return lookup;
}

// `guestBytes` is the surface size when the caller described the surface already (0: described here).
std::shared_ptr<Texture> cachedTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& requested, VkComponentMapping components, std::uint64_t guestBytes = 0, bool depthCompare = false, VkImageViewType multisampleViewType = VK_IMAGE_VIEW_TYPE_2D, bool layerRows = false) {
    Require(!layerRows || (requested.dimension == TextureDimension::k1DArray && !depthCompare), "only a sampled 1D array is laid out as rows of a 2D image");
    const auto resource = layerRows ? requested : ArrayLayerWindow(requested, context.limits.maxImageArrayLayers);
    if (resource.baseAddress != requested.baseAddress) guestBytes = 0;
    if (resource.dimension == TextureDimension::k2DMsaa) {
        Require(!depthCompare, "native multisampled depth comparison sampling is not implemented");
        if (auto depth = DepthSurfaceTexture(context, words, resource, components, multisampleViewType)) {
            logLookup({depth.get(), resource, 0, DccKeys::Uncompressed, 0, nullptr, true});
            return depth;
        }
        auto target = MultisampleTargetTexture(context, resource, components, multisampleViewType);
        logLookup({target.get(), resource, 0, DccKeys::Uncompressed, 0, nullptr, true});
        return target;
    }
    if (resource.fmaskCmaskAddress != 0u) {
        Require(!depthCompare, "FMASK depth comparison sampling is not implemented");
        Require(resource.baseArray == 0u && resource.depthOrLastArray == 0u, "native FMASK array sampling is not implemented");
        CompletePendingCpuWrites(context, resource.fmaskCmaskAddress, CmaskBytes(resource.width, resource.height));
        auto native = MultisampleTargetFmaskTexture_nid_no_patch(context, resource, components);
        Require(native != nullptr, "compressed FMASK sampling without a matching resident native target is not implemented");
        return native;
    }
    CaptureTrace::Log("sampled-lookup address=%llx width=%u height=%u dcc=%llx", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, static_cast<unsigned long long>(resource.dccAddress));
    RequireNoMultisampleTarget(resource.baseAddress, guestBytes != 0 ? static_cast<std::size_t>(guestBytes) : 1u, "a texture read");
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->BoundKeptBytes();
    Require(!layerRows || !DepthSurfaceAt(resource.baseAddress), "sampling a native depth surface as a 1D array past maxImageArrayLayers is not implemented");
    if (auto depth = layerRows ? nullptr : DepthSurfaceTexture(context, words, resource, components)) {
        logLookup({depth.get(), resource, 0, DccKeys::Uncompressed, 0, nullptr, true});
        return depth;
    }
    const auto depthBitsWidth = words.size() >= 4 ? ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) : 0u;
    constexpr auto float32 = static_cast<std::uint32_t>(ShaderRecompiler::IrBufferFormat::Format32Float);
    if (depthBitsWidth == 32u && resource.format != float32) {
        auto bits = resource;
        bits.format = float32;
        return cachedTexture(context, words, bits, components, guestBytes, depthCompare, VK_IMAGE_VIEW_TYPE_2D, layerRows);
    }
    constexpr auto unorm16 = static_cast<std::uint32_t>(ShaderRecompiler::IrBufferFormat::Format16UNorm);
    if (depthBitsWidth == 16u && resource.format != unorm16) {
        auto normalized = resource;
        normalized.format = unorm16;
        return cachedTexture(context, words, normalized, components, guestBytes, depthCompare, VK_IMAGE_VIEW_TYPE_2D, layerRows);
    }
    const auto viewedWindow = [&] {
        const auto window = ViewedArrayWindow(resource, false);
        if (window.baseAddress != resource.baseAddress || resource.dccAddress == 0) return window;
        if (ViewedArrayWindow(resource, true).baseAddress == resource.baseAddress) return window;
        if (TextureClearKeys(resource, guestBytes != 0 ? guestBytes : SurfaceGuestBytes(resource)) != DccKeys::Uncompressed) return window;
        return ViewedArrayWindow(resource, true);
    };
    if (const auto window = viewedWindow(); window.baseAddress != resource.baseAddress) {
        static std::atomic<std::uint64_t> windows{0};
        const auto count = windows.fetch_add(1) + 1;
        if (count <= 5 || (count & (count - 1)) == 0) std::fprintf(stderr, "[gpu] sampled array view 0x%llx (%ux%u format %u, dimension %d, slices %u..%u) served from its viewed slices at 0x%llx; %llu such views\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.dimension), resource.baseArray, resource.depthOrLastArray, static_cast<unsigned long long>(window.baseAddress), static_cast<unsigned long long>(count));
        return cachedTexture(context, words, window, components, 0, depthCompare, VK_IMAGE_VIEW_TYPE_2D, layerRows);
    }
    if (guestBytes == 0) guestBytes = SurfaceGuestBytes(resource);
    if (guestBytes >= (1ull << 30u)) {
        static std::atomic<std::uint64_t> large{0};
        const auto count = large.fetch_add(1) + 1;
        if (count <= 5 || (count & (count - 1)) == 0) {
            std::fprintf(stderr, "[gpu] sampled view 0x%llx (%ux%u format %u, dimension %d, tile %d, slices %u..%u) covers %llu MiB; %llu such views\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.dimension), static_cast<int>(resource.tileMode), resource.baseArray, resource.depthOrLastArray, static_cast<unsigned long long>(guestBytes >> 20u), static_cast<unsigned long long>(count));
            reportGuestAllocation(resource, guestBytes);
        }
    }
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    const bool profile = LookupOutcomes::Profiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::optional<DccKeys> scanned;
    const auto scanKeys = [&] {
        if (scanned.has_value()) return *scanned;
        if (resource.dccAddress == 0) {
            scanned = DccKeys::Uncompressed;
            return *scanned;
        }
        const auto scanStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        auto& memo = sampledKeyProofSlot(resource.dccAddress, guestBytes, resource.format, resource.dccAlphaOnMsb);
        if (memo.dccAddress != resource.dccAddress || memo.guestBytes != guestBytes || memo.format != resource.format || memo.dccAlphaOnMsb != resource.dccAlphaOnMsb) {
            memo.dccAddress = resource.dccAddress;
            memo.guestBytes = guestBytes;
            memo.format = resource.format;
            memo.dccAlphaOnMsb = resource.dccAlphaOnMsb;
            memo.proof = {};
        }
        const auto scans = KeyProofCounts().scanned;
        const auto proofs = KeyProofCounts().proved;
        scanned = ProvedClearKeys(resource, guestBytes, memo.proof);
        if (KeyProofCounts().scanned != scans) ++sampledLookupKeyScans;
        if (KeyProofCounts().proved != proofs) ++sampledLookupKeyProofs;
        if (profile) LookupOutcomes::Add(LookupOutcomes::DccScan, scanStart);
        return *scanned;
    };
    auto& counters = TextureCounts();
    const auto address = resource.baseAddress;
    const auto bytes = static_cast<std::size_t>(guestBytes);
    static_cast<void>(FlushDepthPlanes(address, bytes));
    // A storage image whose results for this surface (or for a mip chain containing it) are still on
    // the GPU supplies the texture by a view of it; anything else needs those results in guest
    // memory first.
    auto lookup = pendingLookupSource(context, resource, guestBytes, depthCompare, layerRows);
    const bool depthCopy = lookup.depthCopy;
    const bool srgbExpand = lookup.srgbExpand;
    auto source = std::move(lookup.source);
    const bool blockCopy = lookup.blockCopy;
    std::optional<DccKeys> keys;
    bool clearThroughKeys = false;
    if (source != nullptr && resource.dccAddress != 0 && IsDccClear(source->FilledKeys())) {
        keys = scanKeys();
        if (*keys == source->FilledKeys()) {
            std::array<std::byte, 16> probe{};
            if (!FillDccClear(DccClearTextureFormat(resource.format), *keys, resource.dccAlphaOnMsb, probe)) {
                char text[256];
                std::snprintf(text, sizeof(text), "AGC graphics: sampled texture 0x%llx (%ux%u format %u, dcc 0x%llx) reads %s DCC keys filled over its pending image, a clear value the format has no encoding for", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<unsigned long long>(resource.dccAddress), DccKeysName(*keys));
                throw std::runtime_error(text);
            }
            static const bool traceKeys = std::getenv("APS5_TRACE_DCC_KEYS") != nullptr;
            if (traceKeys) std::fprintf(stderr, "[dcc-keys] sampled 0x%llx through keys 0x%llx reads the %s fill, not the pending image\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(resource.dccAddress), DccKeysName(*keys));
            source.reset();
            clearThroughKeys = true;
        }
    }
    if (source != nullptr && resource.dccAddress != 0 && (keys.has_value() ? *keys : scanKeys()) == DccKeys::ClearSingle) source.reset();
    if (source != nullptr && resource.dccAddress != 0 && source->Descriptor().dccAddress != resource.dccAddress && (keys.has_value() ? IsDccClear(*keys) : IsDccClear(scanKeys()))) source.reset();

    // Otherwise a surface in host-imported memory is viewed through its cached storage image (made
    // here when there is none): its refresh after a CPU or GPU write is a GPU-direct detile from the
    // import, recorded behind the producer, so no bytes are read or compared on the CPU and nothing
    // waits for the producer. A fast-cleared surface (keys) is viewed only through an image whose own
    // descriptor carries the DCC address (below); it stays a snapshot otherwise, its texels not read.
    if ((!depthCompare || depthCopy) && !layerRows && source == nullptr && !clearThroughKeys && SampledFromStorageEligible(context, resource, guestBytes)) {

        keys = scanKeys();
        if (*keys == DccKeys::Uncompressed) {
            source = sampledStorageSource(context, resource, guestBytes);
            if (source != nullptr && depthCompare && !Texture::CanDepthCopyFrom(*source, resource)) source.reset();
            if (source != nullptr && srgbExpand && !Texture::CanSrgbExpandFrom(context, *source, resource)) source.reset();
        } else if (!depthCompare && ClearedViewEnabled() && StorageClearAvailable(context, resource.format, *keys)) {
            // A fast-cleared surface is viewed as well, through its image cleared on the GPU (the
            // lookup's Refresh, see StorageTexture::upload), when the image's own descriptor names
            // the same DCC metadata so the refresh sees the keys. A snapshot of the clear texels
            // (filled and uploaded on the CPU, then replaced by a view once results are pending: two
            // 4K uploads per clear at the movie stage) serves the surface otherwise, as before.
            auto candidate = sampledStorageSource(context, resource, guestBytes);
            if (candidate != nullptr && candidate->Descriptor().dccAddress == resource.dccAddress) source = std::move(candidate);
        }
        scanned.reset();
        // The lookup's Refresh may have flushed another image's results over the memory (which
        // marks the keys uncompressed): a surface that stays a snapshot re-reads them behind that
        // flush, as it does behind its own. APS5_NO_KEYS_RESCAN=1 keeps the keys scanned above.
        if (source != nullptr && IsRepackedTextureFormat(resource.format) && !Texture::CanRepackCopyFrom(*source, resource)) source.reset();
        static const bool rescan = std::getenv("APS5_NO_KEYS_RESCAN") == nullptr;
        if (rescan && source == nullptr) keys.reset();
    }
    // Pages of the surface written since they were last collected are stamped now, before any
    // UnchangedSince (here and on a cache hit) looks at them: the checks only compare stamped blocks,
    // so a write the game made since the last walk is invisible until a collect stamps its page. The
    // walk is memoized per packet, so a texture viewed from a storage image repeats it for free.
    auto generation = GuestMemory::CollectWrites(address, bytes);
    if (source != nullptr && !GuestMemory::UnchangedSince(address, bytes, source->GenerationFor(address, bytes))) {
        // The CPU wrote the memory while results were pending: Refresh merges them the usual way.
        refreshForView(*source, resource);
        scanned.reset();
    }
    // Results stored to guest memory just now are stamped newer than `generation`; a snapshot read
    // after them is current at the generation of a second (memoized) collect. The store marks the
    // surface's DCC keys uncompressed, so the keys are read after it.
    const auto flushStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool stagedClear = source != nullptr && source->AnyLayerPending() && resource.dccAddress != 0 && source->Descriptor().dccAddress == resource.dccAddress && [&] {
        const auto observedKeys = keys.has_value() ? *keys : scanKeys();
        return IsDccClear(observedKeys) || observedKeys == DccKeys::ClearSingle;
    }();
    bool stagedFlushed = false;
    const bool flushAsked = (source == nullptr || stagedClear) && !clearThroughKeys;
    if (flushAsked && StorageTexture::FlushPending(address, bytes, nullptr, "sampled texture")) {
        if (profile) LookupOutcomes::Add(LookupOutcomes::PendingFlush, flushStart);
        generation = GuestMemory::CollectWrites(address, bytes);
        if (stagedClear) {
            stagedFlushed = true;
            keys = DccKeys::Uncompressed;
        } else {
            keys.reset();
        }
        if (stagedFlushed) source.reset();
    }
    if (flushAsked) scanned.reset();
    if (!keys.has_value()) keys = scanKeys();
    const bool slabbed = source == nullptr && generation == 0 && *keys == DccKeys::Uncompressed && resource.dccAddress == 0 && !depthCompare && !srgbExpand && resource.dimension == TextureDimension::k3D;
    if (disabled) {
        if (source != nullptr && depthCompare) {
            auto texture = std::make_shared<Texture>(context, source, resource, components, Texture::DepthCopyTag{});
            texture->RefreshDepthCopy();
            return texture;
        }
        if (source != nullptr && blockCopy) {
            auto texture = std::make_shared<Texture>(context, source, resource, components, Texture::BlockCopyTag{});
            texture->RefreshDepthCopy();
            return texture;
        }
        if (source != nullptr && IsRepackedTextureFormat(resource.format)) {
            auto texture = std::make_shared<Texture>(context, source, resource, components, Texture::RepackCopyTag{});
            texture->RefreshDepthCopy();
            return texture;
        }
        if (source != nullptr && srgbExpand) {
            auto texture = std::make_shared<Texture>(context, source, resource, components, Texture::SrgbExpandTag{});
            texture->RefreshDepthCopy();
            return texture;
        }
        if (source != nullptr) return std::make_shared<Texture>(context, source, resource, components);
        if (const auto clear = SampledClearColor(context, resource, *keys, depthCompare)) return std::make_shared<Texture>(context, *context.detiler, resource, components, std::span<const std::byte>{}, false, layerRows, &*clear);
        const auto slabs = slabbed ? committedSlabs(resource, guestBytes, std::nullopt) : std::nullopt;
        if (slabs.has_value()) reportCommittedSlabs(resource, *slabs, guestBytes);
        std::vector<std::byte> snapshot(snapshotBytes(slabs, bytes));
        readSnapshot(resource, *keys, slabs, snapshot);
        return std::make_shared<Texture>(context, *context.detiler, resource, components, snapshot, depthCompare, layerRows, nullptr, packedSlabs(slabs));
    }
    auto snapshotGeneration = generation;
    std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>> committed;
    if (source == nullptr && generation == 0) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        if (const auto partial = GuestMemory::CollectCommittedWrites(address, bytes, ranges); partial != 0) {
            snapshotGeneration = partial;
            committed = std::move(ranges);
        }
    }
    const bool keyed = source == nullptr && generation == 0 && !committed.has_value() && *keys == DccKeys::Uncompressed;
    const auto untracked = keyed ? untrackedKey(address, bytes) : std::array<std::uint64_t, 4>{};
    const auto epochStamp = keyed && untracked[0] == 0 && UntrackedReproofEnabled() ? untrackedRangeStamp(address, bytes) : std::array<std::uint64_t, 4>{};
    auto slabs = slabbed ? committedSlabs(resource, guestBytes, committed) : std::nullopt;
    const auto snapshotSize = snapshotBytes(slabs, bytes);
    auto& cache = Textures();
    const auto key = MakeTextureKey(context.device, words, components, depthCompare);
    std::lock_guard lock(cache.mutex);
    const CacheHold held(sampledCacheHeld);
    dropUncachedViews(cache);
    const bool viewKeeps = RuntimeSwitches::Set(RuntimeSwitch::ViewTemplateKeeps) && !RuntimeSwitches::Set(RuntimeSwitch::TrustAudit) && !RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter) && !RuntimeSwitches::Set(RuntimeSwitch::XpTrustFinisher);
    sweepParked(cache, viewKeeps);
    ParkCommit parking{cache, key};
    std::vector<std::byte> recycled;
    std::shared_ptr<Texture> refill;
    auto refillCause = ReleaseCause::Other;
    auto refillKind = ReleaseKind::Snapshot;
    if (auto it = findTexture(cache, key); it != cache.entries.end()) {
        auto staleCause = ReleaseCause::StaleView;
        if (it->source != nullptr) {
            // The view follows the storage image, whatever the GPU wrote to it since; guest memory
            // written meanwhile is taken in by refreshing the image. A fast clear the image cannot
            // see (keys, and no pending results to prefer) ends the view: a snapshot holds the clear.
            if (!stagedFlushed && (source == nullptr || source == it->source) && (source != nullptr || *keys == DccKeys::Uncompressed) && StorageImageServesKeys(*it->source, resource.dccAddress)) {
                if (!GuestMemory::UnchangedSince(address, bytes, it->source->GenerationFor(address, bytes))) refreshForView(*it->source, resource);
                if (it->texture->IsDepthCopy()) it->texture->RefreshDepthCopy();
                it->keys = *keys;
                touchTexture(cache, it);
                logLookup({it->texture.get(), resource, guestBytes, *keys, 0, it->source.get()});
                reportTextureCounters();
                if (profile) LookupOutcomes::Add(*keys != DccKeys::Uncompressed ? LookupOutcomes::SampledHitClearedView : LookupOutcomes::SampledHitView, start);
                return it->texture;
            }
        } else if (source == nullptr && it->surfaceBytes == guestBytes && it->keys == *keys && !it->refuted && it->slabs == slabs && stillCommitted(slabs, it->slabs)) {
            staleCause = ReleaseCause::Rewritten;
            // Unwritten pages need no comparison; partially resident textures compare only committed
            // pages. The compare goes through the flush hook: it waits for recorded work over the
            // surface (counted, and named for the [hooksync] line).
            const auto equalsCommitted = [&] {
                if (it->slabs.has_value() ? it->bytes.size() != snapshotBytes(it->slabs, bytes) : it->bytes.empty()) return false;
                if (Recorder::SnapshotWriteOverlaps(address, bytes)) counters.pendingReads.fetch_add(1, std::memory_order_relaxed);
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::TextureCompare);
                return it->slabs.has_value() ? equalsCommittedSlabs(address, *it->slabs, it->bytes) : GuestMemory::EqualsCommitted(address, it->bytes);
            };
            const auto unchanged = [&] {
                if (it->committed != committed) return false;
                return committed.has_value() ? !Recorder::SnapshotWriteOverlaps(address, static_cast<std::size_t>(it->surfaceBytes)) && GuestMemory::UnchangedCommittedSince(*committed, it->generation) : GuestMemory::UnchangedSince(address, static_cast<std::size_t>(it->surfaceBytes), it->generation);
            };
            const auto kept = [&] {
                if (untracked[0] == 0) return false;
                const bool same = it->untracked == untracked && !Recorder::SnapshotWriteOverlaps(address, bytes);
                (same ? counters.untrackedKept : counters.untrackedCompared).fetch_add(1, std::memory_order_relaxed);
                return same;
            };
            const auto proven = [&] {
                const bool same = epochStamp[0] != 0 && it->proved == epochStamp && !UntrackedProofsVerified() && !Recorder::SnapshotWriteOverlaps(address, bytes);
                if (same) ReproofCounters().stamped.fetch_add(1, std::memory_order_relaxed);
                return same;
            };
            if ((*keys != DccKeys::Uncompressed && *keys != DccKeys::ClearSingle) || unchanged() || kept() || proven() || equalsCommitted()) {
                it->generation = snapshotGeneration;
                it->committed = std::move(committed);
                it->untracked = untracked;
                it->proved = epochStamp;
                touchTexture(cache, it);
                LookupRecord record{it->texture.get(), resource, guestBytes, *keys, generation, nullptr};
                if (generation == 0 && it->committed.has_value()) {
                    record.committedGeneration = snapshotGeneration;
                    record.committed = *it->committed;
                }
                logLookup(record);
                reportTextureCounters();
                if (profile) LookupOutcomes::Add(LookupOutcomes::SampledHitSnapshot, start);
                return it->texture;
            }
        } else {
            staleCause = source != nullptr ? ReleaseCause::NowPending : it->keys != *keys ? ReleaseCause::KeysChanged : ReleaseCause::Shape;
        }
        if (it->bytes.size() == snapshotSize) recycled = std::move(it->bytes);
        else parkSnapshot(cache, it->bytes);
        bool parks = false;
        if ((viewKeeps || ResourceCache::FateCensus()) && it->texture.use_count() > 1) {
            if (const auto refusal = parkRefusal(cache, *it, source != nullptr)) countViewKeep(viewKeepCensus().notParked[static_cast<std::size_t>(*refusal)]);
            else if (viewKeeps) parks = true;
            else countViewKeep(viewKeepCensus().parkable[static_cast<std::size_t>(it->form)]);
        }
        const bool refillable = !parks && it->source == nullptr && it->texture.use_count() > 1 && RuntimeSwitches::Set(RuntimeSwitch::TemplateKeeps);
        if (parks) {
            parking.Arm();
            parking.stash = parkedFrom(*it);
            it->texture->MarkParked(true);
            noteRefilled(it->address);
            countViewKeep(viewKeepCensus().parked[static_cast<std::size_t>(it->form)]);
            nameParking(resource, it->form, staleCause, *keys, clearThroughKeys ? "the pending image holds the fill of these clear keys" : stagedFlushed ? "its pending results were flushed under clear keys" : source != nullptr ? "results now pending in a storage image" : "no pending image and no image that shows the keys");
        }
        if (refillable) {
            refill = it->texture;
            refillCause = staleCause;
            refillKind = releaseKind(*it);
        }
        eraseTexture(cache, it, staleCause, !refillable && !parks);
        ++cache.counts.stale;
    }
    const auto releaseRefill = [&](std::size_t refusal) {
        if (refill == nullptr) return;
        noteRefill(refillCause, refusal);
        noteReleaseCause(refillCause, refillKind);
        noteReleased(refill, address);
        refill.reset();
    };
    struct RefillUnwind {
        const decltype(releaseRefill)& release;
        ~RefillUnwind() { release(RefillThrew); }
    } refillUnwind{releaseRefill};
    const auto clear = source == nullptr ? SampledClearColor(context, resource, *keys, depthCompare) : std::nullopt;
    static const bool sampledImport = std::getenv("APS5_NO_SAMPLED_IMPORT") == nullptr;
    const bool gpuWritten = sampledImport && source == nullptr && !clear && !depthCompare && !layerRows && !slabs.has_value() && *keys == DccKeys::Uncompressed && Recorder::SnapshotWriteOverlaps(address, bytes);
    const HostImport* gpuSource = gpuWritten ? ExistingHostImportFor(context, address, bytes) : nullptr;
    if (source != nullptr || clear.has_value() || gpuSource != nullptr) parkSnapshot(cache, recycled);
    CachedTexture entry{key, address, std::vector<std::byte>{}, nullptr, *keys, snapshotGeneration};
    entry.committed = std::move(committed);
    entry.surfaceBytes = bytes;
    entry.slabs = std::move(slabs);
    const auto createTexture = [&](auto&&... args) {
        const CacheHold reclaimable(sampledCacheReclaimable);
        return std::make_shared<Texture>(context, std::forward<decltype(args)>(args)...);
    };
    const auto takeRefill = [&](std::span<const std::byte> snapshot, bool refillDepthCompare, const VkClearColorValue* refillClear) -> std::shared_ptr<Texture> {
        if (refill == nullptr) return nullptr;
        auto refusal = Texture::RefillRefusal::None;
        try {
            const CacheHold reclaimable(sampledCacheReclaimable);
            refusal = refill->Refill(*context.detiler, resource, components, snapshot, refillDepthCompare, layerRows, refillClear);
        } catch (...) {
            releaseRefill(RefillThrew);
            throw;
        }
        if (refusal != Texture::RefillRefusal::None) {
            releaseRefill(static_cast<std::size_t>(refusal));
            return nullptr;
        }
        noteRefill(refillCause, static_cast<std::size_t>(refusal));
        noteRefilled(address);
        auto kept = std::move(refill);
        static std::atomic<std::uint64_t> refillOrdinal{0};
        if (const auto every = TemplateKeepsVerifyEvery(); every != 0 && refillOrdinal.fetch_add(1, std::memory_order_relaxed) % every == 0) {
            auto& census = templateKeepCensus();
            const char* what = nullptr;
            try {
                const auto fresh = createTexture(*context.detiler, resource, components, snapshot, refillDepthCompare, layerRows, refillClear);
                what = kept->RefillDifference(*fresh);
            } catch (const std::exception&) {
                what = "a fresh texture that threw";
            }
            static std::atomic<std::uint64_t> checks{0};
            static std::atomic<std::uint64_t> differences{0};
            census.refillsVerified.fetch_add(1, std::memory_order_relaxed);
            if (what != nullptr) census.refillsDiffering.fetch_add(1, std::memory_order_relaxed);
            const auto checked = checks.fetch_add(1, std::memory_order_relaxed) + 1u;
            const auto differing = what != nullptr ? differences.fetch_add(1, std::memory_order_relaxed) + 1u : differences.load(std::memory_order_relaxed);
            if (what != nullptr && (differing <= 16u || differing % 4096u == 0u)) std::fprintf(stderr, "[template-keeps-verify] a refilled texture 0x%llx (%ux%u format %u) differs from a fresh one in its %s; %llu refills checked, %llu differing\n", static_cast<unsigned long long>(address), resource.width, resource.height, resource.format, what, static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing));
        }
        return kept;
    };
    const auto revive = [&](KeepForm form, const VkClearColorValue* reviveClear, std::span<const std::byte> snapshot) -> std::shared_ptr<Texture> {
        if (!viewKeeps || cache.parked.empty()) return nullptr;
        const auto found = cache.parked.find(key);
        if (found == cache.parked.end()) return nullptr;
        if (const auto refusal = reviveRefusal(found->second, form, resource, layerRows, bytes, *keys, source.get(), reviveClear, snapshot); refusal != ReviveRefusal::None) {
            countViewKeep(viewKeepCensus().refused[static_cast<std::size_t>(refusal)]);
            return nullptr;
        }
        auto parked = std::move(found->second);
        cache.parked.erase(found);
        cache.parkedBytes -= std::min(cache.parkedBytes, parked.accounted);
        noteParkedSize(cache);
        parked.texture->Revive(textureRevivals().fetch_add(1, std::memory_order_acq_rel) + 1u);
        parking.Arm();
        parking.revived = parked.texture;
        parking.revivedAddress = address;
        countViewKeep(viewKeepCensus().revived[static_cast<std::size_t>(form)]);
        nameRevival(resource, form, *keys, Recorder::Presents() - parked.present);
        static std::atomic<std::uint64_t> reviveOrdinal{0};
        if (const auto every = ViewTemplateKeepsVerifyEvery(); every != 0 && reviveOrdinal.fetch_add(1, std::memory_order_relaxed) % every == 0) {
            const char* what = nullptr;
            try {
                const auto fresh = form == KeepForm::View ? createTexture(source, resource, components) : form == KeepForm::Clear ? createTexture(*context.detiler, resource, components, std::span<const std::byte>{}, false, layerRows, reviveClear) : createTexture(*context.detiler, resource, components, snapshot, depthCompare, layerRows);
                what = parked.texture->ReviveDifference(*fresh);
            } catch (const std::exception&) {
                what = "a fresh texture that threw";
            }
            auto& census = viewKeepCensus();
            census.verified.fetch_add(1, std::memory_order_relaxed);
            if (what != nullptr) census.differing.fetch_add(1, std::memory_order_relaxed);
            static std::atomic<std::uint64_t> checks{0};
            static std::atomic<std::uint64_t> differences{0};
            const auto checked = checks.fetch_add(1, std::memory_order_relaxed) + 1u;
            const auto differing = what != nullptr ? differences.fetch_add(1, std::memory_order_relaxed) + 1u : differences.load(std::memory_order_relaxed);
            if (what != nullptr && (differing <= 16u || differing % 4096u == 0u)) std::fprintf(stderr, "[view-template-keeps-verify] a taken-back %s 0x%llx (%ux%u format %u) differs from a fresh one in its %s; %llu take-backs checked, %llu differing\n", KeepFormNames[static_cast<std::size_t>(form)], static_cast<unsigned long long>(address), resource.width, resource.height, resource.format, what, static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing));
        }
        return parked.texture;
    };
    const auto refillFromPark = [&] {
        if (refill != nullptr || !viewKeeps || cache.parked.empty() || !RuntimeSwitches::Set(RuntimeSwitch::TemplateKeeps)) return;
        const auto found = cache.parked.find(key);
        if (found == cache.parked.end() || found->second.texture == nullptr || found->second.form == KeepForm::View || found->second.texture->ViewsStorageImage()) return;
        auto parked = std::move(found->second);
        cache.parked.erase(found);
        cache.parkedBytes -= std::min(cache.parkedBytes, parked.accounted);
        noteParkedSize(cache);
        countViewKeep(viewKeepCensus().refilledBack[static_cast<std::size_t>(parked.form)]);
        parked.texture->MarkParked(false);
        refillCause = ReleaseCause::Parked;
        refillKind = parked.keys != DccKeys::Uncompressed ? ReleaseKind::Keyed : ReleaseKind::Snapshot;
        refill = std::move(parked.texture);
    };
    if (source != nullptr) {
        releaseRefill(RefillRefusedByView);
        entry.source = source;
        entry.sourceVersion = source->Version();
        if (depthCompare) {
            entry.texture = createTexture(source, resource, components, Texture::DepthCopyTag{});
            entry.texture->RefreshDepthCopy();
        } else if (blockCopy) {
            entry.texture = createTexture(source, resource, components, Texture::BlockCopyTag{});
            entry.texture->RefreshDepthCopy();
        } else if (IsRepackedTextureFormat(resource.format)) {
            entry.texture = createTexture(source, resource, components, Texture::RepackCopyTag{});
            entry.texture->RefreshDepthCopy();
        } else if (srgbExpand) {
            entry.texture = createTexture(source, resource, components, Texture::SrgbExpandTag{});
            entry.texture->RefreshDepthCopy();
        } else {
            entry.texture = revive(KeepForm::View, nullptr, {});
            if (entry.texture == nullptr) entry.texture = createTexture(source, resource, components);
            entry.form = KeepForm::View;
            entry.made = resource;
            entry.madeLayerRows = layerRows;
        }
        entry.accounted = source->Cached() ? 0u : heldBytes(*source);
        counters.fromStorage.fetch_add(1, std::memory_order_relaxed);
    } else if (clear) {
        entry.texture = takeRefill(std::span<const std::byte>{}, false, &*clear);
        if (entry.texture == nullptr) entry.texture = revive(KeepForm::Clear, &*clear, {});
        if (entry.texture == nullptr) {
            refillFromPark();
            entry.texture = takeRefill(std::span<const std::byte>{}, false, &*clear);
        }
        if (entry.texture == nullptr) entry.texture = createTexture(*context.detiler, resource, components, std::span<const std::byte>{}, false, layerRows, &*clear);
        entry.form = KeepForm::Clear;
        entry.made = resource;
        entry.madeClear = *clear;
        entry.madeLayerRows = layerRows;
        entry.accounted = entry.texture->AllocationBytes();
        counters.cleared.fetch_add(1, std::memory_order_relaxed);
    } else if (gpuSource != nullptr) {
        entry.texture = createTexture(*context.detiler, resource, components, std::span<const std::byte>{}, false, false, nullptr, std::nullopt, gpuSource);
        entry.accounted = entry.texture->AllocationBytes();
        counters.imported.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Snapshot before the upload so a write racing with it is caught by the next comparison.
        const bool pending = *keys == DccKeys::Uncompressed && Recorder::SnapshotWriteOverlaps(address, bytes);
        if (pending) counters.pendingReads.fetch_add(1, std::memory_order_relaxed);
        if (entry.slabs.has_value()) reportCommittedSlabs(resource, *entry.slabs, guestBytes);
        constexpr std::size_t SnapshotCompareBytes = 256u << 10u;
        const bool dropped = *keys == DccKeys::Uncompressed && snapshotSize > SnapshotCompareBytes && (entry.generation != 0 || (!pending && untracked[0] != 0));
        std::shared_ptr<Buffer> staging;
        if (dropped) {
            parkSnapshot(cache, recycled);
            expireSpareSnapshots(cache, Recorder::Presents());
            staging = std::make_shared<Buffer>(context, snapshotSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        } else {
            if (recycled.empty()) recycled = spareSnapshot(cache, snapshotSize);
            entry.bytes = recycled.empty() ? std::vector<std::byte>(snapshotSize) : std::move(recycled);
        }
        const auto read = staging != nullptr ? staging->Bytes() : std::span<std::byte>(entry.bytes);
        readSnapshot(resource, *keys, entry.slabs, read);
        if (!pending) {
            entry.untracked = untracked;
            entry.proved = epochStamp;
        }
        static const bool traceTextures = std::getenv("APS5_TRACE_TEXTURES") != nullptr;
        if (traceTextures) {
            std::size_t nonzero = 0;
            for (std::size_t i = 0; i < read.size(); i += 64) nonzero += read[i] != std::byte{0};
            std::fprintf(stderr, "[texture] 0x%llx %ux%u format %u tile %d: %zu of %zu sampled bytes nonzero\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), nonzero, read.size() / 64);
        }
        const bool keyedClear = IsDccClear(*keys);
        if (!entry.slabs.has_value()) {
            entry.form = keyedClear ? KeepForm::KeyedClear : KeepForm::Snapshot;
            entry.made = resource;
            entry.madeTail = snapshotTail(read);
            entry.madeLayerRows = layerRows;
            entry.texture = takeRefill(read, depthCompare, nullptr);
            if (entry.texture == nullptr && keyedClear) entry.texture = revive(KeepForm::KeyedClear, nullptr, read);
            if (entry.texture == nullptr) {
                refillFromPark();
                entry.texture = takeRefill(read, depthCompare, nullptr);
            }
        } else {
            releaseRefill(static_cast<std::size_t>(Texture::RefillRefusal::Sparse));
        }
        if (entry.texture == nullptr && staging != nullptr) entry.texture = createTexture(*context.detiler, resource, components, std::move(staging), depthCompare, layerRows, packedSlabs(entry.slabs));
        else if (entry.texture == nullptr) entry.texture = createTexture(*context.detiler, resource, components, entry.bytes, depthCompare, layerRows, nullptr, packedSlabs(entry.slabs));
        entry.accounted = entry.texture->AllocationBytes();
        if (entry.bytes.size() > SnapshotCompareBytes && (entry.generation != 0 || entry.untracked[0] != 0)) parkSnapshot(cache, entry.bytes);
        counters.snapshots.fetch_add(1, std::memory_order_relaxed);
    }
    const auto budget = sampledBudget(context, cache);
    auto texture = entry.texture;
    const auto madeRecord = [&] {
        LookupRecord record{texture.get(), resource, guestBytes, *keys, generation, source.get()};
        if (source == nullptr && generation == 0 && entry.committed.has_value()) {
            record.committedGeneration = entry.generation;
            record.committed = *entry.committed;
        }
        return record;
    };
    if (!admitTexture(cache, key, entry.accounted, budget)) {
        if (parking.revived != nullptr && parking.revived == texture) countViewKeep(viewKeepCensus().revivedNotAdmitted);
        noteReleaseCause(ReleaseCause::NotAdmitted, releaseKind(entry));
        noteReleased(texture, resource.baseAddress);
        logLookup(madeRecord());
        reportCacheCounts(cache, budget);
        reportTextureCounters();
        if (profile) LookupOutcomes::Add(source != nullptr ? LookupOutcomes::SampledMadeView : LookupOutcomes::SampledMadeSnapshot, start);
        return texture;
    }
    cache.bytes += entry.accounted;
    entry.present = Recorder::Presents();
    ++cache.counts.inserted;
    logLookup(madeRecord());
    cache.entries.push_front(std::move(entry));
    cache.index[key] = cache.entries.begin();
    cache.owners[texture.get()] = cache.entries.begin();
    if (source != nullptr) cache.views.emplace(source.get(), cache.entries.begin());
    reportCacheCounts(cache, budget);
    reportTextureCounters();
    if (profile) LookupOutcomes::Add(source != nullptr ? LookupOutcomes::SampledMadeView : LookupOutcomes::SampledMadeSnapshot, start);
    return texture;
}

enum class SnapshotProof { Refused, Stamped, Unproved };

CachedTexture* untrackedSnapshotEntry(TextureCache& cache, const Texture& texture, std::uint64_t address, std::size_t bytes) {
    const auto found = cache.owners.find(&texture);
    if (found == cache.owners.end()) return nullptr;
    auto& entry = *found->second;
    if (entry.source != nullptr || entry.address != address || entry.surfaceBytes != bytes || entry.bytes.size() != bytes || entry.keys != DccKeys::Uncompressed || entry.committed.has_value() || entry.refuted) return nullptr;
    return &entry;
}

SnapshotProof untrackedSnapshotStamped(const Texture& texture, std::uint64_t address, std::size_t bytes, const std::array<std::uint64_t, 4>& stamp) {
    if (stamp[0] == 0 || Recorder::SnapshotWriteOverlaps(address, bytes)) return SnapshotProof::Refused;
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    const auto* entry = untrackedSnapshotEntry(cache, texture, address, bytes);
    if (entry == nullptr) return SnapshotProof::Refused;
    if (entry->proved != stamp) return SnapshotProof::Unproved;
    ReproofCounters().stamped.fetch_add(1, std::memory_order_relaxed);
    return SnapshotProof::Stamped;
}

bool untrackedSnapshotCompared(const Texture& texture, std::uint64_t address, std::size_t bytes, const std::array<std::uint64_t, 4>& stamp) {
    if (Recorder::SnapshotWriteOverlaps(address, bytes) || PendingStorageOverlaps(address, bytes, nullptr)) return false;
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    auto* entry = untrackedSnapshotEntry(cache, texture, address, bytes);
    if (entry == nullptr) return false;
    if (entry->proved == stamp) return true;
    ReproofCounters().compared.fetch_add(1, std::memory_order_relaxed);
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::TextureCompare);
    if (!GuestMemory::EqualsCommitted(address, entry->bytes)) {
        entry->refuted = true;
        return false;
    }
    entry->proved = stamp;
    return true;
}

// Storage images stay on the GPU between dispatches: while guest memory still holds what an image was
// last uploaded from or written back as, its next use skips the upload and detile.
struct StorageKey {
    VkDevice device;
    std::array<std::uint32_t, 8> words;
    bool operator==(const StorageKey&) const = default;
};

struct StorageKeyHash {
    std::size_t operator()(const StorageKey& key) const noexcept {
        return static_cast<std::size_t>(hashWords(14695981039346656037ull ^ reinterpret_cast<std::uintptr_t>(key.device), key.words.data(), key.words.size()));
    }
};

struct CachedStorageTexture {
    StorageKey key;
    std::uint32_t mip;
    std::shared_ptr<StorageTexture> texture;
    std::uint64_t accounted = 0;
};

// As TextureCache: use order with a hash index by key, plus one by image for StorageImageCached.
struct StorageTextureCache {
    AgcDriver::Mutex mutex;
    std::list<CachedStorageTexture> entries;
    std::unordered_map<StorageKey, std::list<CachedStorageTexture>::iterator, StorageKeyHash> index;
    std::unordered_map<const StorageTexture*, std::list<CachedStorageTexture>::iterator> byImage;
    std::atomic<std::uint64_t> bytes{0};
};

StorageTextureCache& StorageTextures() {
    static auto& cache = *new StorageTextureCache();
    return cache;
}

std::uint64_t storageCacheBytes() {
    return StorageTextures().bytes.load(std::memory_order_relaxed);
}

std::list<CachedStorageTexture>::iterator findStorage(StorageTextureCache& cache, const StorageKey& key) {
    if (TextureHashEnabled()) {
        const auto found = cache.index.find(key);
        return found == cache.index.end() ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) return it;
    }
    return cache.entries.end();
}

std::list<CachedStorageTexture>::iterator findStorageByImage(StorageTextureCache& cache, VkDevice device, const StorageTexture* image) {
    if (TextureHashEnabled()) {
        const auto found = cache.byImage.find(image);
        return found == cache.byImage.end() || found->second->key.device != device ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key.device == device && it->texture.get() == image) return it;
    }
    return cache.entries.end();
}

// Evicts an entry: its pending results go to guest memory first (the image may die with the entry).
std::shared_ptr<StorageTexture> evictStorage(StorageTextureCache& cache, std::list<CachedStorageTexture>::iterator it, bool store = true, const char* reason = "cache eviction") {
    auto texture = it->texture;
    it->texture->SetCached(false);
    noteUncached(it->texture);
    if (store) it->texture->Flush(reason);
    else it->texture->Detach();
    cache.bytes -= it->accounted;
    cache.index.erase(it->key);
    cache.byImage.erase(it->texture.get());
    cache.entries.erase(it);
    return texture;
}

// Storage images are shared by every descriptor of one surface (address, extent, layers, format, tile
// mode): the image holds the whole mip chain, and render targets in the same memory attach to it.
std::array<std::uint32_t, 8> SurfaceKey(const Context& context, const GuestTextureResource& resource) {
    // Guest formats that store in the same Vulkan format share the image (views carry the difference).
    return {static_cast<std::uint32_t>(resource.baseAddress | resource.pipeBankXor), static_cast<std::uint32_t>(resource.baseAddress >> 32u), resource.width, resource.height, (resource.depthOrLastArray << 16u) | (resource.mipCount & 0xffffu), (static_cast<std::uint32_t>(resource.tileMode) << 12u) | (static_cast<std::uint32_t>(resource.dimension) << 20u), resource.baseArray, static_cast<std::uint32_t>(StorageFormatForGuest(context, resource.format))};
}

struct ExtendedSurfaces {
    AgcDriver::Mutex mutex;
    std::map<std::array<std::uint32_t, 8>, std::uint32_t> levels;
    std::atomic<bool> any{false};
};

ExtendedSurfaces& ExtendedChains() {
    static ExtendedSurfaces surfaces;
    return surfaces;
}

std::uint32_t AllocatedLevels(const GuestTextureResource& resource) {
    return resource.allocatedMipCount != 0 ? resource.allocatedMipCount : resource.mipCount;
}

GuestTextureResource StorageSurface(const Context& context, const GuestTextureResource& viewed) {
    auto& surfaces = ExtendedChains();
    const bool extended = viewed.mipCount > AllocatedLevels(viewed);
    if (!extended && !surfaces.any.load(std::memory_order_acquire)) return viewed;
    auto surface = viewed;
    surface.mipCount = AllocatedLevels(viewed);
    const auto identity = SurfaceKey(context, surface);
    std::lock_guard lock(surfaces.mutex);
    if (extended) {
        auto& levels = surfaces.levels[identity];
        levels = std::max(levels, viewed.mipCount);
        surfaces.any.store(true, std::memory_order_release);
        surface.mipCount = levels;
    } else if (const auto it = surfaces.levels.find(identity); it != surfaces.levels.end()) {
        surface.mipCount = it->second;
    } else {
        surface.mipCount = viewed.mipCount;
    }
    return surface;
}

struct WrittenStorage {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> bytes;
    std::vector<WrittenStorageExtent> extents;
};

static WrittenStorage WrittenStorageBytes(const GuestTextureResource& viewed, std::uint32_t mip) {
    WrittenStorage written;
    const auto geometry = DescribeSurface(viewed);
    if (geometry.thick || mip >= geometry.mips.size()) return written;
    const auto& level = geometry.mips[mip];
    const auto bytesPerElement = BytesPerElement(viewed.format);
    const auto covered = CoveredMipBytes(viewed.tileMode, bytesPerElement, level);
    const bool blocks = viewed.tileMode != TextureTileMode::kLinear && !level.tail;
    const auto first = geometry.layers > 1u && geometry.imageDepth <= 1u ? std::min(viewed.baseArray, geometry.layers) : 0u;
    for (auto layer = first; layer < geometry.layers; ++layer) {
        if (!geometry.HasLayer(mip, layer)) continue;
        const auto begin = viewed.baseAddress + geometry.GuestOffset(layer, level);
        for (const auto& [rangeBegin, rangeEnd] : covered) {
            if (!written.bytes.empty() && written.bytes.back().second == begin + rangeBegin) written.bytes.back().second = begin + rangeEnd;
            else written.bytes.emplace_back(begin + rangeBegin, begin + rangeEnd);
        }
        if (blocks) written.extents.push_back({begin, level.width, level.height, bytesPerElement, ThinBlockLayout(viewed.tileMode, bytesPerElement), level.blocksPerRow});
    }
    return written;
}

static void claimStoredMultisampleMemory(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes, std::uint32_t mip) {
    if (!MultisampleColorTargetOverlaps(resource.baseAddress, guestBytes)) return;
    for (const auto& [begin, end] : WrittenStorageBytes(ArrayLayerWindow(resource, context.limits.maxImageArrayLayers), mip).bytes) ClaimMultisampleTargetStore(begin, end - begin);
}

// `guestBytes` is the surface size when the caller described the surface already (0: described here).
std::shared_ptr<StorageTexture> cachedStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& requested, std::uint32_t mip, std::uint64_t guestBytes, bool written, std::optional<std::pair<std::uint32_t, std::uint32_t>> layers) {
    const auto viewed = ArrayLayerWindow(requested, context.limits.maxImageArrayLayers);
    if (viewed.baseAddress != requested.baseAddress) guestBytes = 0;
    Require(viewed.dimension != TextureDimension::k2DMsaa, "storage image access to a multisampled texture is not implemented");
    const auto viewedBytes = guestBytes != 0 ? guestBytes : SurfaceGuestBytes(viewed);
    RequireNoMultisampleTarget(viewed.baseAddress, static_cast<std::size_t>(viewedBytes), "a storage image access");
    if (auto depth = DepthSurfaceStorage(context, words, viewed, mip)) return depth;
    if (viewedBytes >= (1ull << 30u)) {
        static std::atomic<std::uint64_t> large{0};
        const auto count = large.fetch_add(1) + 1;
        if (count <= 5 || (count & (count - 1)) == 0) {
            std::fprintf(stderr, "[gpu] storage view 0x%llx (%ux%u format %u, dimension %d, tile %d, slices %u..%u, mip %u) covers %llu MiB; %llu such views\n", static_cast<unsigned long long>(viewed.baseAddress), viewed.width, viewed.height, viewed.format, static_cast<int>(viewed.dimension), static_cast<int>(viewed.tileMode), viewed.baseArray, viewed.depthOrLastArray, mip, static_cast<unsigned long long>(viewedBytes >> 20u), static_cast<unsigned long long>(count));
            reportGuestAllocation(viewed, viewedBytes);
        }
    }
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    if ((DepthPlanesStale() || written) && (written || !layers || disabled)) static_cast<void>(FlushDepthPlanes(viewed.baseAddress, static_cast<std::size_t>(viewedBytes), written ? DepthPlaneScope::Whole : DepthPlaneScope::Overlapped));
    if (written && DepthSurfaceOverlaps(context.device, viewed.baseAddress, viewedBytes)) {
        const auto storage = WrittenStorageBytes(viewed, mip);
        RetireCoveredDepthSurfaces(context.device, storage.bytes, storage.extents);
    }
    if (disabled) return StorageTexture::Create(context, *context.detiler, viewed, mip);
    static_cast<void>(words);
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->BoundKeptBytes();
    const bool profile = LookupOutcomes::Profiled();
    auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& counters = TextureCounts();
    const auto resource = StorageSurface(context, viewed);
    const StorageKey key{context.device, SurfaceKey(context, resource)};
    auto& cache = StorageTextures();
    std::vector<std::shared_ptr<StorageTexture>> evicted;
    std::shared_ptr<StorageTexture> texture;
    {
        std::lock_guard lock(cache.mutex);
        const CacheHold held(storageCacheHeld);
        if (resource.mipCount > AllocatedLevels(resource)) {
            auto allocated = resource;
            allocated.mipCount = AllocatedLevels(resource);
            if (const auto found = findStorage(cache, {context.device, SurfaceKey(context, allocated)}); found != cache.entries.end()) evicted.push_back(evictStorage(cache, found));
        }
        auto it = findStorage(cache, key);
        std::shared_ptr<StorageTexture> detached;
        if (it != cache.entries.end() && MetadataMoved(*it->texture, resource)) {
            detached = evictStorage(cache, it, false);
            evicted.push_back(detached);
            it = cache.entries.end();
        }
        if (it != cache.entries.end()) {
            if (layers) it->texture->RefreshLayers(layers->first, layers->second);
            else it->texture->Refresh();
            cache.entries.splice(cache.entries.begin(), cache.entries, it);
            texture = it->texture;
            if (TextureCountersReported()) counters.storageHits.fetch_add(1, std::memory_order_relaxed);
            if (profile) LookupOutcomes::Add(LookupOutcomes::StorageHit, start);
        } else {
            if (guestBytes == 0) guestBytes = SurfaceGuestBytes(resource);
            bool flushed = false;
            auto handOver = StorageTexture::PrepareCreation(context, resource, guestBytes, std::move(detached), flushed);
            if (flushed && profile) start = LookupOutcomes::Add(LookupOutcomes::PendingFlush, start);
            std::shared_ptr<StorageTexture> created;
            {
                const CacheHold reclaimable(storageCacheReclaimable);
                created = StorageTexture::Create(context, *context.detiler, resource, mip);
            }
            created->AdoptHandOver(std::move(handOver));
            CachedStorageTexture entry{key, mip, std::move(created)};
            static const bool keepNew = std::getenv("APS5_NO_KEEP_NEW_STORAGE") == nullptr;
            if (auto* recorder = Recorder::Active(); keepNew && recorder != nullptr && GuestMemory::GpuMutex().HeldByThisThread() && recorder->Recording()) recorder->Keep(entry.texture);
            entry.accounted = heldBytes(*entry.texture);
            const auto budget = storageBudget(context, cache.bytes + entry.accounted);
            while (!cache.entries.empty()) {
                const bool overBudget = cache.bytes + entry.accounted > budget;
                if (!overBudget && cache.entries.size() < storageEntries()) break;
                const auto back = std::prev(cache.entries.end());
                if (back->texture->TakeUsed()) cache.entries.splice(cache.entries.begin(), cache.entries, back);
                else evicted.push_back(evictStorage(cache, back, true, overBudget ? "cache eviction" : "entry cap"));
            }
            cache.bytes += entry.accounted;
            texture = entry.texture;
            cache.entries.push_front(std::move(entry));
            cache.index[key] = cache.entries.begin();
            cache.byImage[texture.get()] = cache.entries.begin();
            texture->SetCached(true);
            counters.storageCreated.fetch_add(1, std::memory_order_relaxed);
            if (profile) LookupOutcomes::Add(LookupOutcomes::StorageMade, start);
        }
    }
    for (const auto& image : evicted) {
        std::vector<std::shared_ptr<ShaderResources>> resources;
        SharedResourceCache().RemoveImage(image.get(), &resources);
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording()) {
            if (image.use_count() > 1) recorder->KeepBytes(image.get(), static_cast<std::size_t>(heldBytes(*image)));
            for (auto& resource : resources) {
                resource->KeepStorageImageBytes(*recorder);
                recorder->Keep(std::move(resource));
            }
        }
    }
    if (!evicted.empty()) {
        auto& sampled = Textures();
        std::lock_guard lock(sampled.mutex);
        dropUncachedViews(sampled);
    }
    return texture;
}

}

std::uint64_t TextureCacheBudget(const VkPhysicalDeviceMemoryProperties& memory) {
    const auto heap = largestDeviceLocalHeap(memory);
    return std::max<std::uint64_t>(2048ull << 20u, heap ? memory.memoryHeaps[*heap].size / 4u : 0u);
}

std::uint64_t StorageTextureCeiling(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported) {
    return CalculateStorageTextureBudget(memory, reported, 0, 0).ceiling;
}

std::uint64_t StorageTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t sampledBytes, std::uint64_t storageBytes) {
    return CalculateStorageTextureBudget(memory, reported, sampledBytes, storageBytes).selectedBudget;
}

StorageTextureBudgetSnapshot CalculateStorageTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t sampledBytes, std::uint64_t storageBytes) {
    StorageTextureBudgetSnapshot snapshot;
    snapshot.sampledBytes = sampledBytes;
    snapshot.storageBytes = storageBytes;
    snapshot.ceiling = TextureCacheBudget(memory);
    snapshot.selectedBudget = snapshot.ceiling;
    const auto heap = largestDeviceLocalHeap(memory);
    if (reported == nullptr || !heap || reported->heapBudget[*heap] == 0) return snapshot;
    snapshot.heap = heap;
    const auto heapBudget = reported->heapBudget[*heap];
    const auto usage = reported->heapUsage[*heap];
    snapshot.reserved = usage - std::min<std::uint64_t>(usage, sampledBytes + storageBytes) + sampledBytes + heapBudget / 8u;
    snapshot.available = heapBudget > snapshot.reserved ? heapBudget - snapshot.reserved : 0u;
    snapshot.ceiling = std::max<std::uint64_t>(512ull << 20u, std::min<std::uint64_t>(memory.memoryHeaps[*heap].size / 2u, heapBudget / 2u));
    snapshot.selectedBudget = std::min<std::uint64_t>(snapshot.ceiling, std::max<std::uint64_t>(512ull << 20u, snapshot.available));
    return snapshot;
}

std::uint64_t SampledTextureBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, std::uint64_t textureBytes, std::uint64_t storageBytes) {
    const auto heap = largestDeviceLocalHeap(memory);
    if (reported == nullptr || !heap || reported->heapBudget[*heap] == 0) return TextureCacheBudget(memory);
    const auto budget = reported->heapBudget[*heap];
    const auto usage = reported->heapUsage[*heap];
    const auto reserved = usage - std::min<std::uint64_t>(usage, textureBytes) + std::max(TextureCacheBudget(memory), storageBytes) + budget / 8u;
    return std::max<std::uint64_t>(2048ull << 20u, budget > reserved ? budget - reserved : 0u);
}

bool SampledBudgetReportDue(std::uint64_t reported, std::uint64_t budget, std::chrono::steady_clock::duration sinceReport) {
    const auto change = budget > reported ? budget - reported : reported - budget;
    return change > reported / 10u && sinceReport >= std::chrono::seconds(10);
}

std::optional<std::uint32_t> LargestDeviceLocalHeap(const VkPhysicalDeviceMemoryProperties& memory) {
    return largestDeviceLocalHeap(memory);
}

std::uint64_t StorageTextureCacheBudget(const Context& context) {
    return storageBudget(context);
}

void SampledFromStorageForTest(bool enabled) {
    sampledFromStorageForTest.store(enabled, std::memory_order_relaxed);
}

void OverrideCacheBudgetForTest(std::uint64_t bytes) {
    budgetOverride.store(bytes, std::memory_order_relaxed);
}

TextureCacheUse TextureCacheUsage() {
    TextureCacheUse use;
    {
        auto& sampled = Textures();
        std::lock_guard lock(sampled.mutex);
        dropUncachedViews(sampled);
        use.sampledEntries = sampled.entries.size();
        use.sampledBytes = sampled.bytes;
        use.spareEntries = sampled.spare.size();
        use.spareBytes = sampled.spareBytes;
    }
    use.storageBytes = storageCacheBytes();
    return use;
}

std::uint64_t SampledTextureCacheBudget(const Context& context) {
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    return sampledBudget(context, cache);
}

std::shared_ptr<Texture> CachedSampledTexture(const Context& context, std::span<const std::uint32_t> words, VkImageViewType multisampleViewType) {
    const auto resource = DecodeTextureResource(words);
    return cachedTexture(context, words, resource, ViewComponents(resource), 0, false, multisampleViewType);
}

bool SampledTexturesShareEntry(std::span<const std::uint32_t> first, std::span<const std::uint32_t> second) {
    Require(first.size() == 8 && second.size() == 8, "sampled texture keys take eight descriptor words");
    const VkComponentMapping components{VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
    const auto firstKey = MakeTextureKey(VK_NULL_HANDLE, first, components);
    const auto secondKey = MakeTextureKey(VK_NULL_HANDLE, second, components);
    return firstKey == secondKey && TextureKeyHash{}(firstKey) == TextureKeyHash{}(secondKey);
}

void FlushCachedTextures(VkDevice device) {
    Require(device != VK_NULL_HANDLE, "cannot flush textures without a Vulkan device");
    GuestMemory::AssertGpuLockHeld("FlushCachedTextures");
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    const CacheHold held(storageCacheHeld);
    for (const auto& entry : cache.entries) {
        if (entry.key.device == device) entry.texture->Flush();
    }
}

WriteIndexOutcomes WriteIndexOutcomesCounted() {
    const auto& counts = IndexCounts();
    const auto at = [&](IndexCount which) { return counts[which].load(std::memory_order_relaxed); };
    return {at(IndexAccepted), at(IndexPartial), at(IndexFallback), at(IndexBumped), at(IndexLookedUp), at(IndexRebound), at(IndexCompared), at(IndexBuilds)};
}

void ClearCachedTextures(VkDevice device) {
    Require(device != VK_NULL_HANDLE, "cannot clear textures without a Vulkan device");
    auto& sampled = Textures();
    {
        std::lock_guard lock(sampled.mutex);
        const CacheHold held(sampledCacheHeld);
        for (auto it = sampled.entries.begin(); it != sampled.entries.end();) {
            if (it->key.device == device) eraseTexture(sampled, it++, ReleaseCause::Cleared);
            else ++it;
        }
        sampled.spare.clear();
        sampled.spareBytes = 0;
        if (!sampled.parked.empty()) dropParkedWhere(sampled, ParkedDrop::Cleared, [&](const TextureKey& key, const ParkedTexture&) { return key.device == device; });
    }
    auto& storage = StorageTextures();
    std::vector<std::shared_ptr<StorageTexture>> cleared;
    {
        std::lock_guard lock(storage.mutex);
        const CacheHold held(storageCacheHeld);
        for (auto it = storage.entries.begin(); it != storage.entries.end();) {
            if (it->key.device != device) {
                ++it;
                continue;
            }
            cleared.push_back(it->texture);
            it->texture->SetCached(false);
            noteUncached(it->texture);
            storage.bytes -= it->accounted;
            storage.index.erase(it->key);
            storage.byImage.erase(it->texture.get());
            it = storage.entries.erase(it);
        }
    }
    for (const auto& image : cleared) {
        std::vector<std::shared_ptr<ShaderResources>> resources;
        SharedResourceCache().RemoveImage(image.get(), &resources);
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording()) {
            for (auto& resource : resources) {
                resource->KeepStorageImageBytes(*recorder);
                recorder->Keep(std::move(resource));
            }
        }
    }
}

std::uint64_t ReleaseTextureCaches(std::uint64_t bytes, bool newImage) {
    thread_local bool releasing = false;
    if (releasing) return 0;
    releasing = true;
    struct Done {
        bool& flag;
        ~Done() { flag = false; }
    } done{releasing};
    const auto releasedBefore = ImageMemoryReleasedBytes();
    std::uint64_t freed = 0;
    std::size_t sampledEntries = 0;
    std::size_t storageEntries = 0;
    std::size_t storageKept = 0;
    bool sampledSkipped = false;
    bool storageSkipped = false;
    {
        auto& cache = Textures();
        std::unique_lock lock(cache.mutex, std::defer_lock);
        const bool inherited = sampledCacheHeld != 0;
        if ((inherited && (!newImage || sampledCacheReclaimable == 0)) || (!inherited && !lock.try_lock())) {
            sampledSkipped = true;
        } else {
            for (auto it = cache.entries.end(); it != cache.entries.begin() && freed < bytes;) {
                --it;
                if (it->texture.use_count() != 1 || (it->source != nullptr && it->source->Dirty())) continue;
                const auto next = std::next(it);
                eraseTexture(cache, it);
                ++cache.counts.evicted;
                ++sampledEntries;
                freed = ImageMemoryReleasedBytes() - releasedBefore;
                it = next;
            }
        }
    }
    if (freed < bytes) {
        auto& cache = StorageTextures();
        std::unique_lock<GuestMemory::GpuMutexType> gpu(GuestMemory::GpuMutex(), std::defer_lock);
        const bool inherited = storageCacheHeld != 0;
        if ((inherited && (!newImage || storageCacheReclaimable == 0)) || !gpu.try_lock()) {
            storageSkipped = true;
        } else {
            try {
                while (freed < bytes) {
                    std::shared_ptr<StorageTexture> image;
                    {
                        std::unique_lock lock(cache.mutex, std::defer_lock);
                        if (!inherited && !lock.try_lock()) {
                            storageSkipped = true;
                            break;
                        }
                        storageKept = 0;
                        bool spared = false;
                        for (auto it = cache.entries.end(); it != cache.entries.begin();) {
                            --it;
                            if (it->texture->Dirty()) {
                                ++storageKept;
                                continue;
                            }
                            if (it->texture->TakeUsed()) {
                                spared = true;
                                continue;
                            }
                            image = evictStorage(cache, it);
                            break;
                        }
                        if (image == nullptr) {
                            if (spared) continue;
                            break;
                        }
                    }
                    std::vector<std::shared_ptr<ShaderResources>> resources;
                    const bool removed = SharedResourceCache().TryRemoveImage(image.get(), &resources);
                    if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording()) {
                        for (auto& resource : resources) recorder->Keep(std::move(resource));
                    }
                    resources.clear();
                    ++storageEntries;
                    image.reset();
                    freed = ImageMemoryReleasedBytes() - releasedBefore;
                    if (!removed) {
                        storageSkipped = true;
                        break;
                    }
                }
            } catch (const std::exception& error) {
                std::fprintf(stderr, "[gpu] texture cache release stopped: %s\n", error.what());
            }
        }
    }
    freed = ImageMemoryReleasedBytes() - releasedBefore;
    std::fprintf(stderr, "[gpu] texture caches released %llu MiB of the %llu MiB asked for: %zu sampled entries%s, %zu storage images%s, %zu with pending results kept\n", static_cast<unsigned long long>(freed >> 20u), static_cast<unsigned long long>(bytes >> 20u), sampledEntries, sampledSkipped ? " (cache busy, skipped)" : "", storageEntries, storageSkipped ? " (cache or device busy, skipped)" : "", storageKept);
    return freed;
}

bool StorageImageCached(const Context& context, const StorageTexture* image) {
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    const auto it = findStorageByImage(cache, context.device, image);
    if (it == cache.entries.end()) return false;
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
    return true;
}

namespace {

// StorageImageCached for several images under one acquisition of the cache mutex.
bool StorageImagesCached(const Context& context, std::span<const StorageTexture* const> images) {
    if (images.empty()) return true;
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    for (const auto* image : images) {
        const auto it = findStorageByImage(cache, context.device, image);
        if (it == cache.entries.end()) return false;
        cache.entries.splice(cache.entries.begin(), cache.entries, it);
    }
    return true;
}

}

std::shared_ptr<StorageTexture> CachedStorageSurface(const Context& context, const GuestTextureResource& resource) {
    return cachedStorageTexture(context, {}, resource, 0);
}

std::shared_ptr<StorageTexture> CachedStorageLayers(const Context& context, const GuestTextureResource& resource, std::uint32_t firstLayer, std::uint32_t layerCount) {
    return cachedStorageTexture(context, {}, resource, 0, 0, false, std::pair{firstLayer, layerCount});
}

bool StorageImageServesKeys(const StorageTexture& image, std::uint64_t dccAddress) {
    GuestTextureResource resource{};
    resource.dccAddress = dccAddress;
    return !MetadataMoved(image, resource);
}

std::shared_ptr<StorageTexture> RepeatStorageSurface(const Context& context, const GuestTextureResource& requested, const std::shared_ptr<StorageTexture>& stored) {
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    if (disabled || stored == nullptr) return nullptr;
    const auto viewed = ArrayLayerWindow(requested, context.limits.maxImageArrayLayers);
    Require(viewed.dimension != TextureDimension::k2DMsaa, "storage image access to a multisampled texture is not implemented");
    const auto viewedBytes = SurfaceGuestBytes(viewed);
    RequireNoMultisampleTarget(viewed.baseAddress, static_cast<std::size_t>(viewedBytes), "a storage image access");
    if (DepthSurfaceStorage(context, {}, viewed, 0) != nullptr) return nullptr;
    if (DepthPlanesStale()) static_cast<void>(FlushDepthPlanes(viewed.baseAddress, static_cast<std::size_t>(viewedBytes), DepthPlaneScope::Overlapped));
    if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->BoundKeptBytes();
    const bool profile = LookupOutcomes::Profiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto resource = StorageSurface(context, viewed);
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    const CacheHold held(storageCacheHeld);
    if (resource.mipCount > AllocatedLevels(resource)) return nullptr;
    const auto it = findStorageByImage(cache, context.device, stored.get());
    if (it == cache.entries.end() || MetadataMoved(*it->texture, resource)) return nullptr;
    it->texture->Refresh();
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
    TextureCounts().storageHits.fetch_add(1, std::memory_order_relaxed);
    if (profile) LookupOutcomes::Add(LookupOutcomes::StorageHit, start);
    return it->texture;
}

namespace {

const char* roleName(ShaderRecompiler::DescriptorRole role) {
    switch (role) {
        case ShaderRecompiler::DescriptorRole::GuestBuffers: return "GuestBuffers";
        case ShaderRecompiler::DescriptorRole::GuestImages: return "GuestImages";
        case ShaderRecompiler::DescriptorRole::GuestSamplers: return "GuestSamplers";
        case ShaderRecompiler::DescriptorRole::Gds: return "Gds";
        case ShaderRecompiler::DescriptorRole::BdaPagetable: return "BdaPagetable";
        case ShaderRecompiler::DescriptorRole::FaultBuffer: return "FaultBuffer";
        case ShaderRecompiler::DescriptorRole::FlattenedSrt: return "FlattenedSrt";
        case ShaderRecompiler::DescriptorRole::ShaderData: return "ShaderData";
        case ShaderRecompiler::DescriptorRole::ImageTable: return "ImageTable";
        case ShaderRecompiler::DescriptorRole::SamplerTable: return "SamplerTable";
        case ShaderRecompiler::DescriptorRole::ImageTableMap: return "ImageTableMap";
        case ShaderRecompiler::DescriptorRole::ImageStorageTable: return "ImageStorageTable";
    }
    throw std::runtime_error("AGC graphics: unknown descriptor role");
}

const char* kindName(ShaderRecompiler::DescriptorKind kind) {
    switch (kind) {
        case ShaderRecompiler::DescriptorKind::UniformBuffer: return "UniformBuffer";
        case ShaderRecompiler::DescriptorKind::StorageBuffer: return "StorageBuffer";
        case ShaderRecompiler::DescriptorKind::UniformTexelBuffer: return "UniformTexelBuffer";
        case ShaderRecompiler::DescriptorKind::StorageTexelBuffer: return "StorageTexelBuffer";
        case ShaderRecompiler::DescriptorKind::SampledImage: return "SampledImage";
        case ShaderRecompiler::DescriptorKind::StorageImage: return "StorageImage";
        case ShaderRecompiler::DescriptorKind::Sampler: return "Sampler";
    }
    throw std::runtime_error("AGC graphics: unknown descriptor kind");
}

// Consecutive identical storage descriptors address successive mips of one texture (dynamic-mip
// storage writes), so the second and later ones share the first one's image lookup.
// APS5_NO_STORAGE_DEDUPE=1 looks each one up (and refreshes the image) separately as before.
bool StorageDedupeEnabled() {
    static const bool disabled = std::getenv("APS5_NO_STORAGE_DEDUPE") != nullptr;
    return !disabled;
}

// Whether storage element `element` of `binding` repeats the previous element's eight words.
bool SameAsPreviousStorageElement(std::span<const std::uint32_t> guest, std::uint32_t element) {
    if (element == 0) return false;
    const auto words = guest.begin() + static_cast<std::size_t>(element) * 8u;
    return std::equal(words, words + 8, words - 8);
}

}

ShaderResources::ShaderResources(const Context& context, const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& fragment, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes) : ShaderResources(context, std::array<CompiledShader, 2>{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, {ShaderRecompiler::ShaderStage::Fragment, &fragment, static_cast<std::uint32_t>(vertex.pushConstants.size())}}}, target, 1, indexAddress, indexBytes) {}

void ShaderResources::KeepStorageImageBytes(Recorder& recorder) const {
    const auto keep = [&](const std::shared_ptr<StorageTexture>& image) {
        if (image != nullptr && !image->Cached()) recorder.KeepBytes(image.get(), static_cast<std::size_t>(image->BudgetBytes()));
    };
    for (const auto& image : storageTextures) keep(image);
    for (const auto& texture : textures) {
        if (texture != nullptr) keep(texture->SharedStorageSource());
    }
}

ShaderResources::ShaderResources(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes, std::span<const GuestMemorySnapshot> snapshots, bool deviceStaging) : context(context), guestMemory(context) {
    drawBuild = true;
    const bool addressTables = std::ranges::any_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && std::ranges::any_of(shader.program->bindings, [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; }); });
    if (deviceStaging && !addressTables) guestMemory.AllowDeviceStaging();
    prepareAddressBindings(shaders, snapshots);
    build(shaders, &target, colorAttachments, indexAddress, indexBytes);
}

ShaderResources::ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots) : ShaderResources(context, compute, snapshots, false) {}

namespace {

// Whether a compute stage maps registered guest memory through BDA tables (see prepareAddressBindings).
bool UsesAddressTables(const CompiledShader& compute) {
    Require(compute.program != nullptr, "missing compiled shader");
    return std::any_of(compute.program->bindings.begin(), compute.program->bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; });
}

}

ShaderResources::ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots, bool deferred, std::uint64_t dispatchThreads, std::optional<std::pair<std::uint64_t, std::size_t>> htileClear) : context(context), guestMemory(context), dispatchThreads(dispatchThreads), htileClear(htileClear), deferredCompute(compute), deferredSnapshots(snapshots) {
    Require(compute.stage == ShaderRecompiler::ShaderStage::Compute, "compute resources require a compute shader");
    // Every use of a compute build is a recorded dispatch that calls MarkGpuWrites, which staged
    // buffers need (a synchronous draw's use would not).
    guestMemory.AllowDeviceStaging();
    static const std::uint64_t addressStagingMinThreads = [] {
        const char* value = std::getenv("APS5_BDA_STAGING_MIN_THREADS");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 512ull;
    }();
    if (dispatchThreads != 0 && dispatchThreads < addressStagingMinThreads) guestMemory.DisallowAddressStaging();
    const std::span<const CompiledShader> shaders(&deferredCompute, 1);
    if (!deferred) {
        prepareAddressBindings(shaders, snapshots);
        build(shaders, nullptr, 0, 0, 0);
        forgetDeferredInputs();
        return;
    }
    // Address-based shaders pin and mirror registered memory in prepareAddressBindings (reconciling
    // imports retires buffers to the recorder, refreshing mirrors waits for recorded work): that is
    // device-lock work, so their whole build waits for Complete(). Without tables the call only
    // checks the bindings.
    lockedBuild = UsesAddressTables(compute);
    if (lockedBuild) return;
    unlockedPrepare = true;
    prepareAddressBindings(shaders, snapshots);
    buildPrepare(shaders, nullptr, 0, 0, 0);
}

void ShaderResources::Complete() {
    Require(!completed, "shader resources were already completed");
    const std::span<const CompiledShader> shaders(&deferredCompute, 1);
    if (lockedBuild) {
        prepareAddressBindings(shaders, deferredSnapshots);
        buildPrepare(shaders, nullptr, 0, 0, 0);
    }
    buildComplete();
    if (heapStage) RefreshHeapStage(shaders);
    forgetDeferredInputs();
}

void ShaderResources::forgetDeferredInputs() {
    // The compiled shader and the captured regions belong to the dispatch, which returns while this
    // object lives on in the resource cache and the recorder: nothing may reach for them after the
    // build, so they are dropped rather than left dangling.
    deferredCompute = {};
    deferredSnapshots = {};
}

void ShaderResources::build(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes) {
    buildPrepare(shaders, target, colorAttachments, indexAddress, indexBytes);
    buildComplete();
    if (heapStage) RefreshHeapStage(shaders);
    if (!GuestMemory::WriteIndexEnabled()) return;
    countIndex(IndexBuilds);
    subscribeBuilt(shaders);
}

namespace {

VkDescriptorSetLayout CreateSetLayout(const Context& context, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind, std::span<const VkDescriptorBindingFlags> flags, bool push = false) {
    Require(flags.size() == bindings.size(), "descriptor binding flag count differs from layout");
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bindingFlags.bindingCount = static_cast<std::uint32_t>(flags.size());
    bindingFlags.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.pNext = std::ranges::any_of(flags, [](auto value) { return value != 0; }) ? &bindingFlags : nullptr;
    info.flags = (updateAfterBind ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0) | (push ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0);
    info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    info.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &layout), "vkCreateDescriptorSetLayout");
    return layout;
}

std::vector<VkDescriptorPoolSize> PoolSizes(std::span<const VkDescriptorSetLayoutBinding> bindings) {
    std::map<VkDescriptorType, std::uint32_t> counts;
    for (const auto& binding : bindings) counts[binding.descriptorType] += binding.descriptorCount;
    std::vector<VkDescriptorPoolSize> sizes;
    for (const auto& [type, count] : counts) sizes.push_back({type, count});
    return sizes;
}

VkDescriptorPool CreateSetPool(const Context& context, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind) {
    VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    info.flags = updateAfterBind ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0;
    info.maxSets = 1;
    info.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    info.pPoolSizes = sizes.data();
    VkDescriptorPool pool = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &info, nullptr, &pool), "vkCreateDescriptorPool");
    return pool;
}

// APS5_PROFILE_DRAW: the [resources] phase totals. Each thread accumulates its builds' phases in
// arrays of its own and merges them into the shared totals every 1000 of its builds (and when it
// ends), so no build takes the shared mutex per phase; the totals lag by up to 999 builds per
// worker.
constexpr std::size_t BuildPhaseCount = static_cast<std::size_t>(ShaderResources::BuildPhase::Count);
constexpr std::array<const char*, BuildPhaseCount> BuildPhaseNames{"bindings", "precollect", "guest memory upload", "descriptors", "stage A", "images", "bda", "stage B"};

struct BuildProfile {
    AgcDriver::Mutex mutex;
    std::array<double, BuildPhaseCount> ms{};
    std::uint64_t builds = 0;
};

BuildProfile& Builds() {
    static BuildProfile profile;
    return profile;
}

bool BuildProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

struct ThreadBuildProfile {
    std::array<double, BuildPhaseCount> ms{};
    std::uint64_t builds = 0;
    ~ThreadBuildProfile() { merge(); }
    void merge() {
        auto& profile = Builds();
        std::lock_guard lock(profile.mutex);
        for (std::size_t i = 0; i < BuildPhaseCount; ++i) profile.ms[i] += ms[i];
        profile.builds += builds;
        ms = {};
        if (builds == 0) return;
        builds = 0;
        std::string report;
        for (std::size_t i = 0; i < BuildPhaseCount; ++i) report += " " + std::string(BuildPhaseNames[i]) + "=" + std::to_string(static_cast<long long>(profile.ms[i])) + "ms";
        AgcDriver::ProfilePrint_nid_no_patch("[resources] %llu builds, phase totals:%s\n", static_cast<unsigned long long>(profile.builds), report.c_str());
    }
};

ThreadBuildProfile& ThreadBuilds() {
    struct ProfileStorage {};
    auto& profile = HostThreadLocal<ThreadBuildProfile, ProfileStorage>();
    return profile;
}

void addBuildPhase(ShaderResources::BuildPhase which, double ms) {
    ThreadBuilds().ms[static_cast<std::size_t>(which)] += ms;
}

// APS5_PROFILE_DRAW: guest buffer elements bound by descriptor (addGuestBuffer) and how many the
// recompiler proved read-only, plus the pending-write notes their uses skipped (MarkGpuWrites),
// printed as [buffers] every 10 s from MarkGpuWrites. Cumulative.
struct BufferWriteCounters {
    std::atomic<std::uint64_t> elements{0};
    std::atomic<std::uint64_t> readOnly{0};
    std::atomic<std::uint64_t> notesSkipped{0};
    std::atomic<std::int64_t> lastReport{0};
};

BufferWriteCounters& BufferWrites() {
    static BufferWriteCounters counters;
    return counters;
}

struct MovedInPlaceCounters {
    std::atomic<std::uint64_t> taken{0};
    std::atomic<std::uint64_t> refused{0};
};

MovedInPlaceCounters& MovedInPlace() {
    static MovedInPlaceCounters counters;
    return counters;
}

bool MovedBuffersInPlace() {
    static const bool enabled = std::getenv("APS5_NO_MOVED_BUFFERS_IN_PLACE") == nullptr;
    return enabled;
}

}

double ShaderResources::phase(BuildPhase which) {
    if (!BuildProfiled()) return 0.0;
    const auto now = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration<double, std::milli>(now - phaseStart).count();
    addBuildPhase(which, ms);
    if (ms > 50) AgcDriver::ProfilePrint_nid_no_patch("[resources] %s took %.0f ms (%zu textures, %zu storage images, %zu buffers, bda %d)\n", BuildPhaseNames[static_cast<std::size_t>(which)], ms, textures.size(), storageTextures.size(), allocations.size(), usesBda ? 1 : 0);
    phaseStart = now;
    return ms;
}

namespace {

std::uint32_t IrStage(ShaderRecompiler::ShaderStage stage) {
    using Ir = ShaderRecompiler::IrShaderStage;
    switch (stage) {
        case ShaderRecompiler::ShaderStage::Compute: return static_cast<std::uint32_t>(Ir::Compute);
        case ShaderRecompiler::ShaderStage::Vertex: return static_cast<std::uint32_t>(Ir::Vertex);
        case ShaderRecompiler::ShaderStage::Fragment: return static_cast<std::uint32_t>(Ir::Pixel);
        case ShaderRecompiler::ShaderStage::Mesh: return static_cast<std::uint32_t>(Ir::Mesh);
        case ShaderRecompiler::ShaderStage::Local: return static_cast<std::uint32_t>(Ir::Local);
        case ShaderRecompiler::ShaderStage::TessellationControl: return static_cast<std::uint32_t>(Ir::TessellationControl);
        case ShaderRecompiler::ShaderStage::TessellationEvaluation: return static_cast<std::uint32_t>(Ir::TessellationEvaluation);
        case ShaderRecompiler::ShaderStage::Geometry: break;
    }
    return static_cast<std::uint32_t>(Ir::Unknown);
}

bool TableReuseEnabled() {
    static const bool disabled = std::getenv("APS5_NO_TABLE_REUSE") != nullptr;
    return !disabled;
}

bool TableReplay() {
    static const bool disabled = std::getenv("APS5_NO_TABLE_REPLAY") != nullptr;
    return !disabled;
}

struct TableState {
    std::uint64_t epoch = 0;
    std::uint64_t unwatched = 0;
    std::uint64_t allocations = 0;
    std::uint64_t depthSurfaces = 0;
    bool operator==(const TableState&) const = default;
};

TableState CurrentTableState() {
    if (!TableReuseEnabled()) return {};
    return {GuestMemory::ThreadCollectEpoch(), GuestMemory::UnwatchSerial(), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), DepthSurfacesSerial()};
}

constexpr auto NoRepresentative = std::numeric_limits<std::uint32_t>::max();

std::vector<std::uint32_t> variantRepresentatives(VkDevice device, const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> imageWords, std::uint64_t elementBase, std::uint64_t capacity);

struct ReusedTableElement {
    TextureKey key{};
    std::weak_ptr<Texture> texture;
    GuestTextureResource resource{};
    std::uint64_t guestBytes = 0;
    bool firstLayer = false;
    bool rejected = false;
    std::string message;
    bool live = false;
    bool substitute = false;
    std::uint32_t representative = NoRepresentative;
    GuestTextureResource representativeResource{};
};

struct ReusedTable {
    VkDevice device = VK_NULL_HANDLE;
    TableState state;
    std::uint64_t pendingSerial = 0;
    std::vector<std::uint32_t> words;
    std::vector<ShaderRecompiler::DescriptorImageShape> shapes;
    ShaderRecompiler::ElementFlags depthCompare;
    std::vector<ReusedTableElement> elements;
    std::vector<GuestMemory::UnchangedQuery> queries;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::vector<std::uint64_t> imageSamplers;
    std::vector<std::uint32_t> samplers;
    std::vector<std::string> bound;
    bool replayable = false;
    std::vector<std::uint32_t> representatives;
};

constexpr std::size_t ReusedTableCount = 4;

std::vector<ReusedTable>& ReusedTables() {
    struct ReusedTablesStorage {};
    return HostThreadLocal<std::vector<ReusedTable>, ReusedTablesStorage>();
}

bool SameTable(const ReusedTable& table, VkDevice device, const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> words) {
    return table.device == device && std::ranges::equal(table.words, words) && table.shapes == binding.imageShapes && table.depthCompare == binding.imageDepthCompare;
}

ReusedTable* FindReusedTable(VkDevice device, const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> words, const TableState& state) {
    if (state.epoch == 0) return nullptr;
    auto& tables = ReusedTables();
    std::erase_if(tables, [&](const ReusedTable& table) { return table.state != state; });
    for (auto& table : tables) {
        if (SameTable(table, device, binding, words)) return &table;
    }
    return nullptr;
}

bool TableCurrent(ReusedTable& table) {
    const auto serial = StorageTexture::PendingSerial();
    if (serial != table.pendingSerial) {
        if (StorageTexture::AnyPendingOverlaps(table.ranges) || StorageTexture::PendingSerial() != serial) return false;
        table.pendingSerial = serial;
    }
    return GuestMemory::UnchangedSinceAll(table.queries);
}

bool CachedTableTextures(const ReusedTable& table, std::vector<std::shared_ptr<Texture>>& cached) {
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    std::vector<std::list<CachedTexture>::iterator> entries;
    entries.reserve(table.queries.size());
    for (const auto& element : table.elements) {
        if (element.rejected || element.live || element.substitute) continue;
        const auto it = findTexture(cache, element.key);
        if (it == cache.entries.end() || it->texture != element.texture.lock()) return false;
        entries.push_back(it);
    }
    cached.reserve(entries.size());
    for (const auto it : entries) {
        cached.push_back(it->texture);
        touchTexture(cache, it);
    }
    return true;
}

void logTableLookups(const ReusedTable& table, const std::vector<std::shared_ptr<Texture>>& cached) {
    if (RuntimeSwitches::Set(RuntimeSwitch::NoExtendedFastProofs)) return;
    std::size_t kept = 0;
    for (const auto& element : table.elements) {
        if (element.rejected || element.live || element.substitute) continue;
        if (kept >= cached.size() || kept >= table.queries.size()) return;
        logLookup({cached[kept].get(), element.resource, element.guestBytes, DccKeys::Uncompressed, table.queries[kept].generation, nullptr});
        ++kept;
    }
}

void RememberTable(ReusedTable&& table, const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> words) {
    auto& tables = ReusedTables();
    std::erase_if(tables, [&](const ReusedTable& entry) { return SameTable(entry, table.device, binding, words); });
    table.words.assign(words.begin(), words.end());
    table.shapes = binding.imageShapes;
    table.depthCompare = binding.imageDepthCompare;
    tables.insert(tables.begin(), std::move(table));
    if (tables.size() > ReusedTableCount) tables.resize(ReusedTableCount);
}

std::string HexText(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(value));
    return text;
}

std::string DescriptorLimitError(std::string_view description, std::uint64_t required, std::uint64_t limit) {
    return std::string(description) + ": required " + std::to_string(required) + ", device limit " + std::to_string(limit);
}

void ValidateDescriptorLayout(const Context& context, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind) {
    const auto& indexed = context.descriptorIndexingLimits;
    const bool sampledUpdateAfterBind = updateAfterBind && context.sampledImageUpdateAfterBind;
    const auto maxStageResources = updateAfterBind ? indexed.maxPerStageUpdateAfterBindResources : context.limits.maxPerStageResources;
    const auto maxStageSampledImages = sampledUpdateAfterBind ? indexed.maxPerStageDescriptorUpdateAfterBindSampledImages : context.limits.maxPerStageDescriptorSampledImages;
    const auto maxStageSamplers = sampledUpdateAfterBind ? indexed.maxPerStageDescriptorUpdateAfterBindSamplers : context.limits.maxPerStageDescriptorSamplers;
    const auto maxSetSampledImages = sampledUpdateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSampledImages : context.limits.maxDescriptorSetSampledImages;
    const auto maxSetSamplers = sampledUpdateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSamplers : context.limits.maxDescriptorSetSamplers;
    std::map<VkDescriptorType, std::uint64_t> heapTotals;
    std::map<std::pair<VkShaderStageFlags, VkDescriptorType>, std::uint64_t> stageTotals;
    std::map<VkShaderStageFlags, std::uint64_t> stageResources;
    for (const auto& layout : bindings) {
        heapTotals[layout.descriptorType] += layout.descriptorCount;
        for (std::uint32_t bit = 0u; bit < 32u; ++bit) {
            const auto stage = VkShaderStageFlags{1u} << bit;
            if ((layout.stageFlags & stage) == 0u) continue;
            const auto count = stageTotals[{stage, layout.descriptorType}] += layout.descriptorCount;
            const auto all = stageResources[stage] += layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ? 0u : layout.descriptorCount;
            Require(all <= maxStageResources, "typed heaps exceed per-stage resource capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) Require(count <= maxStageSampledImages, "typed heaps exceed per-stage sampled image capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) Require(count <= context.limits.maxPerStageDescriptorStorageImages, "typed heaps exceed per-stage storage image capacity");
            if (layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER && count > maxStageSamplers) throw std::runtime_error(DescriptorLimitError("typed heaps exceed per-stage sampler capacity", count, maxStageSamplers));
        }
    }
    Require(heapTotals[VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] <= maxSetSampledImages && heapTotals[VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] <= context.limits.maxDescriptorSetStorageImages, "typed heaps exceed descriptor set capacity");
    if (heapTotals[VK_DESCRIPTOR_TYPE_SAMPLER] > maxSetSamplers) throw std::runtime_error(DescriptorLimitError("typed heaps exceed descriptor set sampler capacity", heapTotals[VK_DESCRIPTOR_TYPE_SAMPLER], maxSetSamplers));
}

void ValidateRuntimeResources(const CompiledShader& shader, std::span<const std::uint32_t> words) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    const auto& descriptors = program.bindings;
    constexpr auto bindingCount = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    std::array<const ShaderRecompiler::DescriptorBinding*, bindingCount> heaps{};
    if (program.runtimeImageCount != 0u) {
        for (const auto& binding : descriptors) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
            auto& heap = heaps[binding.binding % bindingCount];
            Require(heap == nullptr, "duplicate runtime image binding");
            heap = &binding;
        }
    }
    Require(words.size() == program.shaderDataDwords, "invalid compact shader data size");
    constexpr auto metadataWords = sizeof(ShaderRecompiler::RuntimeAbi::ResourceMetadata) / sizeof(std::uint32_t);
    Require(program.imageMetadataDword <= words.size() && program.runtimeImageCount <= (words.size() - program.imageMetadataDword) / metadataWords, "runtime image metadata exceeds shader data");
    for (const auto index : program.runtimeImageResources) {
        Require(index < program.runtimeImageCount, "runtime image resource exceeds its compact layout");
        const auto offset = program.imageMetadataDword + index * metadataWords;
        const auto kind = words[offset];
        const auto first = words[offset + 1u];
        const auto count = words[offset + 2u];
        Require(count != 0u, "runtime image metadata has no descriptor elements");
        Require(kind < heaps.size() && heaps[kind] != nullptr, "runtime metadata references an unbound image heap");
        Require(first < heaps[kind]->count && count <= heaps[kind]->count - first, "runtime metadata exceeds its bound heap");
    }
}

enum class HeapBindingClass { SharedImage, ShaderData, PerDraw, Unsupported };

HeapBindingClass ClassifyHeapBinding(const ShaderRecompiler::DescriptorBinding& binding) {
    using Role = ShaderRecompiler::DescriptorRole;
    switch (binding.role) {
        case Role::GuestImages:
            return binding.kind == ShaderRecompiler::DescriptorKind::StorageImage ? HeapBindingClass::PerDraw : HeapBindingClass::SharedImage;
        case Role::GuestSamplers:
            return HeapBindingClass::SharedImage;
        case Role::BdaPagetable:
        case Role::FaultBuffer:
        case Role::Gds:
        case Role::GuestBuffers:
            return HeapBindingClass::PerDraw;
        case Role::ShaderData:
        case Role::FlattenedSrt:
            return HeapBindingClass::ShaderData;
        default:
            return HeapBindingClass::Unsupported;
    }
}

}

void ShaderResources::buildPrepare(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint32_t colorAttachments, std::uint64_t indexAddress, std::size_t indexBytes) {
    expressViewsAt.store(0, std::memory_order_relaxed);
    registeredImageViews.clear();
    for (const auto& shader : shaders) if (shader.registeredImageViews != nullptr) registeredImageViews.insert(registeredImageViews.end(), shader.registeredImageViews->begin(), shader.registeredImageViews->end());
    Require(registeredImageViewsCurrent(registeredImageViews), "AGC graphics: registered image range changed after resource capture");
    PerformanceTimer frameTiming("Resources.Prepare");
    const auto stageStart = std::chrono::steady_clock::now();
    phaseStart = stageStart;
    if (BuildProfiled()) {
        auto& profile = ThreadBuilds();
        if (++profile.builds % 1000 == 0) profile.merge();
    }
    try {
        Require(!shaders.empty() && context.limits.maxBoundDescriptorSets >= 1, "shader descriptor set exceeds device limits");
        if (shaders.size() == 1u) refreshResourceKey = ContentKey(shaders.front(), false, true);
        std::size_t plannedBindings = 0;
        std::size_t plannedAllocations = 0;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) continue;
            plannedBindings += shader.program->bindings.size();
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages && binding.role != ShaderRecompiler::DescriptorRole::GuestSamplers) plannedAllocations += binding.count;
            }
        }
        bindings.reserve(plannedBindings);
        allocations.reserve(plannedAllocations);
        std::vector<std::uint32_t> occupied;
        occupied.reserve(plannedBindings);
        for (const auto& shader : shaders) {
            Require(shader.program != nullptr, "missing compiled shader");
            ShaderRecompiler::RuntimeAbi::RequireVersion(shader.program->runtimeAbiVersion);
            if (shader.program->poisonedSrtReads != 0) {
                const auto& programBindings = shader.program->bindings;
                const auto flat = std::find_if(programBindings.begin(), programBindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt; });
                if (flat != programBindings.end()) srtGuardFaults.push_back({IrStage(shader.stage), DrawGuestWords(shader, *flat)});
            }
            const VkShaderStageFlags flags = VulkanStage(shader.stage);
            std::uint64_t stageStorageBuffers = 0;
            std::uint64_t stageResources = shader.stage == ShaderRecompiler::ShaderStage::Fragment ? colorAttachments : 0;
            Require(stageResources <= context.limits.maxPerStageResources, "shader descriptors exceed per-stage limits");
            std::vector<std::size_t> offsetsInData;
            std::int64_t shaderData = -1;
            const auto firstSampler = samplers.size();
            const auto firstDeferred = deferredImages.size();
            const auto requireStageLimits = [&] {
                if (stageStorageBuffers <= context.limits.maxPerStageDescriptorStorageBuffers && stageResources <= context.limits.maxPerStageResources) return;
                const auto& limits = context.descriptorIndexingLimits;
                Require(stageStorageBuffers <= limits.maxPerStageDescriptorUpdateAfterBindStorageBuffers && stageResources <= limits.maxPerStageUpdateAfterBindResources, "shader descriptors exceed per-stage limits");
                updateAfterBind = true;
            };
            const bool heapShader = HeapShader(shader);
            if (heapShader) {
                Require(DescriptorHeapEnabled(), "a GPU-mode stage needs VK_EXT_descriptor_heap (APS5_GPU_HEAP)");
                heapStage = true;
                if (shader.stage == ShaderRecompiler::ShaderStage::Compute) NoteHeapComputeStage();
            }
            for (const auto& binding : shader.program->bindings) {
                Require(binding.descriptorSet == ShaderRecompiler::RuntimeAbi::DescriptorSet || ((shader.stage == ShaderRecompiler::ShaderStage::Fragment || heapShader) && binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet) || (heapShader && binding.role == ShaderRecompiler::DescriptorRole::Gds && binding.descriptorSet == ShaderRecompiler::RuntimeAbi::GpuGdsSet), "unexpected descriptor set: a shader resource must use its stage's descriptor set");
                if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable || binding.role == ShaderRecompiler::DescriptorRole::ImageStorageTable) {
                    const bool storage = binding.role == ShaderRecompiler::DescriptorRole::ImageStorageTable;
                    Require(binding.binding == (storage ? ShaderRecompiler::RuntimeAbi::ImageStorageTableBinding : ShaderRecompiler::RuntimeAbi::ImageTableBinding) && binding.kind == (storage ? ShaderRecompiler::DescriptorKind::StorageImage : ShaderRecompiler::DescriptorKind::SampledImage) && binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 8u && binding.imageShapes.size() == binding.count && binding.imageDepthCompare.size() == binding.count && binding.imageSamplers.size() == binding.count, "image table binding is malformed");
                    NoteImageTableBuild();
                    auto& table = tableStage(shader);
                    Require(storage ? !table.storage : !table.sampled, "a stage binds two image tables");
                    Require(table.images == nullptr || table.images->count == binding.count, "a stage's image tables disagree on their elements");
                    (storage ? table.storage : table.sampled) = true;
                    table.images = &binding;
                    table.imageWords = DrawGuestWords(shader, binding);
                    Require(table.imageWords.size() == static_cast<std::size_t>(binding.count) * 8u, "a draw's image table words differ in size from its binding");
                    table.firstSampler = firstSampler;
                    continue;
                }
                const bool heapStorage = heapShader && binding.role == ShaderRecompiler::DescriptorRole::GuestImages && binding.kind == ShaderRecompiler::DescriptorKind::StorageImage && binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet;
                const bool sharedHeapArray = heapShader && (binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers || (binding.role == ShaderRecompiler::DescriptorRole::GuestImages && binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) || heapStorage);
                const bool fresh = std::find(occupied.begin(), occupied.end(), binding.binding) == occupied.end();
                Require(fresh || sharedHeapArray, "duplicate shader binding");
                if (fresh) occupied.push_back(binding.binding);
                if (binding.role == ShaderRecompiler::DescriptorRole::SamplerTable) {
                    addSamplerTable(binding, DrawGuestWords(shader, binding), flags, tableStage(shader));
                    continue;
                }
                if (binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap) {
                    const auto& mapWords = DrawGuestWords(shader, binding);
                    Require(binding.kind == ShaderRecompiler::DescriptorKind::StorageBuffer && binding.count == 1 && mapWords.size() >= ShaderRecompiler::ImageTableAbi::HeaderWords && !binding.readOnly && mapWords[ShaderRecompiler::ImageTableAbi::HeaderVersion] == ShaderRecompiler::ImageTableAbi::Version, "image table map binding is malformed");
                    const auto size = mapWords.size() * sizeof(std::uint32_t);
                    Require(size <= context.limits.maxStorageBufferRange, "image table map exceeds the storage buffer range limit");
                    stageStorageBuffers += 1;
                    stageResources += 1;
                    storageBuffers += 1;
                    requireStageLimits();
                    auto& table = tableStage(shader);
                    Require(table.mapAllocation == std::numeric_limits<std::size_t>::max(), "a stage binds two image table maps");
                    table.map = mapWords;
                    allocations.push_back({0, size, false, std::make_unique<Buffer>(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT), ShaderRecompiler::DescriptorRole::ImageTableMap});
                    table.mapAllocation = allocations.size() - 1u;
                    bindings.push_back({{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flags, nullptr}, {allocations.size() - 1u}});
                    bindings.back().heap = table.heap;
                    continue;
                }
                if (heapShader) {
                    if (heapStorage) {
                        Require(ShaderRecompiler::RuntimeAbi::GpuHeapStorageKind(binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count)), "a heap storage image binding has an atomic or unknown kind");
                        addImageBinding(binding, DrawGuestWords(shader, binding), flags);
                        bindings.back().heap = true;
                        continue;
                    }
                    const auto heapClass = ClassifyHeapBinding(binding);
                    if (heapClass == HeapBindingClass::Unsupported) Require(false, std::string("a GPU-mode fragment stage has an unsupported ") + roleName(binding.role) + " binding");
                    if (heapClass == HeapBindingClass::SharedImage) {
                        Require(binding.kind == ShaderRecompiler::DescriptorKind::SampledImage || binding.kind == ShaderRecompiler::DescriptorKind::Sampler, "a GPU-mode stage binds sampled images and samplers only");
                        addImageBinding(binding, DrawGuestWords(shader, binding), flags);
                        bindings.back().heap = true;
                        continue;
                    }
                    if (heapClass == HeapBindingClass::ShaderData) {
                        Require(binding.count == 1 && !DrawGuestWords(shader, binding).empty(), "invalid GPU-mode shader data binding");
                        if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) ValidateRuntimeResources(shader, DrawGuestWords(shader, binding));
                        Binding item{{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, flags, nullptr}, {}, {}};
                        item.heap = true;
                        item.role = binding.role;
                        bindings.push_back(std::move(item));
                        continue;
                    }
                }
                const bool addressRole = binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable || binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer;
                const bool bufferRole = addressRole || binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers || binding.role == ShaderRecompiler::DescriptorRole::ShaderData || binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt || binding.role == ShaderRecompiler::DescriptorRole::Gds;
                const bool imageRole = binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers;
                if (imageRole) {
                    if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage || binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) stageResources += binding.count;
                    requireStageLimits();
                    addImageBinding(binding, DrawGuestWords(shader, binding), flags);
                    continue;
                }
                if (!bufferRole) Require(false, std::string("unsupported descriptor role ") + roleName(binding.role));
                if (binding.kind != ShaderRecompiler::DescriptorKind::StorageBuffer) Require(false, std::string("unsupported descriptor kind ") + kindName(binding.kind) + " for role " + roleName(binding.role) + ": only StorageBuffer is supported");
                Require(!binding.readOnly, "read-only descriptors are unsupported because the recompiler emits no NonWritable decoration");
                Require(binding.count != 0, "empty descriptor binding");
                stageStorageBuffers += binding.count;
                stageResources += binding.count;
                storageBuffers += binding.count;
                requireStageLimits();
                Binding item{{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding.count, flags, nullptr}, {}};
                if (binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers) {
                    const auto& guestWords = DrawGuestWords(shader, binding);
                    Require(guestWords.size() == static_cast<std::uint64_t>(binding.count) * 4, "guest buffer descriptor must contain four DWORDs per array element");
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        // An element the recompiler did not classify (a producer without the
                        // vector) counts as written, like imageWritten below.
                        const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                        const bool atomic = element < binding.bufferAtomic.size() && binding.bufferAtomic[element];
                        const bool read = element < binding.bufferRead.size() && binding.bufferRead[element];
                        const std::uint32_t extent = element < binding.bufferExtent.size() ? binding.bufferExtent[element] : 0u;
                        const auto index = addGuestBuffer(std::span<const std::uint32_t>(guestWords).subspan(static_cast<std::size_t>(element) * 4, 4), target, indexAddress, indexBytes, written, atomic, read, extent);
                        const bool classified = element < binding.bufferWritten.size() && element < binding.bufferRead.size() && element < binding.bufferAtomic.size();
                        allocations[index].writeOnly = classified && allocations[index].guest && allocations[index].written && binding.bufferWritten[element] && !read && !atomic;
                        const auto& push = shader.program->pushConstants;
                        const auto pushDataBytes = shader.program->stencilReferences.empty() ? push.size() : shader.program->stencilReferenceDword * sizeof(std::uint32_t);
                        if (pushDataBytes != 0u && !shader.program->dataAddressPush) {
                            const auto position = shader.program->memoryOffsetDword * 4u + element;
                            Require(position < pushDataBytes, "guest buffer offset lies outside the shader's push constants");
                            allocations[index].pushByte = static_cast<std::int32_t>(shader.pushConstantOffset + position);
                        } else {
                            allocations[index].dataByte = shader.program->memoryOffsetDword * 4u + element;
                            offsetsInData.push_back(index);
                        }
                        item.allocations.push_back(index);
                    }
                    if (heapShader && shader.program->bufferAddressCount != 0u) {
                        Require(shader.program->bufferAddressCount == binding.count, "a GPU-mode stage's buffer address words disagree with its buffer binding");
                        item.heap = true;
                        item.addressed = true;
                        item.dataStage = static_cast<std::uint32_t>(HeapStageIndex(shader.stage));
                        item.addressDword = shader.program->bufferAddressDword;
                        addressedBuffers = true;
                    }
                } else if (addressRole) {
                    item.allocations.push_back(allocations.size());
                    allocations.push_back({0, 0, false, nullptr, binding.role});
                    item.heap = (heapShader && shader.program->dataAddressPush) || (heapStage && !heapShader && binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer);
                } else if (binding.role == ShaderRecompiler::DescriptorRole::Gds) {
                    Require(binding.count == 1 && binding.guestDescriptor.empty(), "invalid GDS descriptor contract");
                    static_assert(Pm4::GdsBytes == ShaderRecompiler::RuntimeAbi::GdsBytes);
                    const auto gds = Pm4::GdsAddress();
                    guestMemory.AddWritable(gds, Pm4::GdsBytes, true);
                    item.allocations.push_back(allocations.size());
                    allocations.push_back({gds, Pm4::GdsBytes, true, nullptr, ShaderRecompiler::DescriptorRole::ShaderData, true});
                    if (heapShader && shader.program->dataAddressPush) {
                        item.heap = true;
                        if (gdsAllocation == std::numeric_limits<std::size_t>::max()) gdsAllocation = item.allocations.back();
                    }
                } else {
                    Require(binding.count == 1, "shader data and flattened SRT descriptors must not be arrays");
                    const auto& guestWords = DrawGuestWords(shader, binding);
                    Require(!guestWords.empty(), "empty shader data descriptor");
                    if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) {
                        ValidateRuntimeResources(shader, guestWords);
                    }
                    item.allocations.push_back(addDataBuffer(guestWords));
                    if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) shaderData = static_cast<std::int64_t>(item.allocations.back());
                }
                bindings.push_back(std::move(item));
            }
            for (const auto index : offsetsInData) allocations[index].dataAllocation = heapShader ? HeapRingData(HeapStageIndex(shader.stage)) : shaderData;
            for (auto index = firstDeferred; index < deferredImages.size(); ++index) {
                deferredImages[index].firstSampler = firstSampler;
                deferredImages[index].samplerCount = samplers.size() - firstSampler;
            }
            for (auto& table : tableStages) {
                if (table.program == shader.program && table.images != nullptr) table.samplerCount = samplers.size() - table.firstSampler;
            }
        }
        splitSets = SplitsSets(context, shaders);
        std::array<VkShaderStageFlags, 2> tableFlags{};
        std::array<VkShaderStageFlags, 2> storageTableFlags{};
        for (auto& table : tableStages) {
            Require(table.mapAllocation != std::numeric_limits<std::size_t>::max(), "an image table stage has no map");
            const auto& map = table.map;
            Require(map.size() >= ShaderRecompiler::ImageTableAbi::HeaderWords, "image table map is truncated");
            const auto elements = table.images != nullptr ? table.images->count : 0u;
            Require(map[ShaderRecompiler::ImageTableAbi::HeaderElementCount] == elements, "image table map disagrees with its image binding");
            const auto faults = map[ShaderRecompiler::ImageTableAbi::HeaderElementFaults];
            const auto samplerFaults = map[ShaderRecompiler::ImageTableAbi::HeaderSamplerFaults];
            Require(static_cast<std::uint64_t>(faults) + elements <= map.size() && static_cast<std::uint64_t>(samplerFaults) + map[ShaderRecompiler::ImageTableAbi::HeaderSamplerCount] / 4u <= map.size(), "image table map is truncated");
            if (table.images == nullptr || table.heap) continue;
            table.set = splitSets && !heapStage && table.stage == ShaderRecompiler::ShaderStage::Fragment ? 1u : 0u;
            table.elementBase = tableElements[table.set];
            table.map[ShaderRecompiler::ImageTableAbi::HeaderElementBase] = table.elementBase;
            tableElements[table.set] += elements;
            if (table.sampled) tableFlags[table.set] |= VulkanStage(table.stage);
            if (table.storage) storageTableFlags[table.set] |= VulkanStage(table.stage);
        }
        for (std::uint32_t set = 0; set < storageTableFlags.size(); ++set) {
            if (storageTableFlags[set] == 0) continue;
            bindings.push_back({{ShaderRecompiler::RuntimeAbi::ImageStorageTableBinding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, StorageTableCapacity(context), storageTableFlags[set], nullptr}, {}, {}, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, static_cast<std::int32_t>(set)});
            storageTableBindings[set] = bindings.size() - 1u;
        }
        for (std::uint32_t set = 0; set < tableFlags.size(); ++set) {
            if (tableFlags[set] == 0) continue;
            Require(context.imageTableCapacity != 0, "image tables need runtimeDescriptorArray, descriptorBindingPartiallyBound and descriptorBindingVariableDescriptorCount, which the device lacks");
            const VkDescriptorBindingFlags imageTableFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | (context.imageTableUpdateAfterBind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0);
            bindings.push_back({{ShaderRecompiler::RuntimeAbi::ImageTableBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, context.imageTableCapacity, tableFlags[set], nullptr}, {}, {}, imageTableFlags, static_cast<std::int32_t>(set)});
            tableBindings[set] = bindings.size() - 1u;
        }
        if (storageBuffers > context.limits.maxDescriptorSetStorageBuffers) {
            Require(storageBuffers <= context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindStorageBuffers, "pipeline descriptors exceed device limits");
            updateAfterBind = true;
        }
        timing.bindingsMs = phase(BuildPhase::Bindings);
        // For every build, locked ones included: their stage B then takes the fast path too, and the
        // collects cost the same wherever they run.
        if (precollectImages()) phase(BuildPhase::Precollect);
        const auto heapDataBytes = [&](std::int64_t ring) -> std::uint64_t {
            const auto stage = HeapRingStage(ring);
            for (const auto& shader : shaders) {
                if (HeapStageIndex(shader.stage) != stage || !HeapShader(shader)) continue;
                for (const auto& binding : shader.program->bindings) {
                    if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) return binding.guestDescriptor.size() * sizeof(std::uint32_t);
                }
            }
            return 0;
        };
        const auto patchable = [&](const Allocation& allocation) {
            if (!allocation.guest || allocation.pushByte >= 0) return true;
            if (IsHeapRingData(allocation.dataAllocation)) return allocation.dataByte < heapDataBytes(allocation.dataAllocation);
            return allocation.dataAllocation >= 0 && allocation.dataByte < allocations[static_cast<std::size_t>(allocation.dataAllocation)].size;
        };
        if (drawBuild && !usesBda && std::all_of(allocations.begin(), allocations.end(), patchable)) guestMemory.AllowAdjustedRegions();
        requireGuestBuffers();
        guestMemory.UploadPrepare(usesBda);
        timing.uploadMs = phase(BuildPhase::Upload);
        splitSets = SplitsSets(context, shaders);
        std::array<std::vector<VkDescriptorSetLayoutBinding>, 2> descriptions;
        bool tablesUpdateAfterBind = false;
        for (const auto& binding : bindings) {
            if (binding.heap) continue;
            const auto set = setOf(binding);
            descriptions[set].push_back(binding.layout);
            (set == 0 ? layoutBindingFlags : fragmentBindingFlags).push_back(binding.flags | (updateAfterBind && binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0));
            tablesUpdateAfterBind |= (binding.flags & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT) != 0;
            auto& key = set == 0 ? layoutKey : fragmentLayoutKey;
            key.insert(key.end(), {binding.layout.binding, static_cast<std::uint32_t>(binding.layout.descriptorType), binding.layout.descriptorCount, binding.layout.stageFlags});
        }
        updateAfterBind = updateAfterBind || tablesUpdateAfterBind;
        if (updateAfterBind) {
            layoutKey.push_back(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
            if (splitSets) fragmentLayoutKey.push_back(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
        }
        static const bool noLayoutCache = std::getenv("APS5_NO_LAYOUT_CACHE") != nullptr;
        static const bool noPoolCache = std::getenv("APS5_NO_POOL_CACHE") != nullptr;
        const auto tableDescriptorCount = [&](std::size_t set) { return std::max(1u, std::min(tableElements[set], context.imageTableCapacity)); };
        const auto tableVariableCount = [&](std::size_t set) { return tableBindings[set] != std::numeric_limits<std::size_t>::max() ? std::optional<std::uint32_t>(tableDescriptorCount(set)) : std::nullopt; };
        const auto splitPoolSizes = [&](std::size_t set) {
            std::map<VkDescriptorType, std::uint32_t> counts;
            for (std::size_t index = 0; index < bindings.size(); ++index) {
                if (setOf(bindings[index]) != set) continue;
                counts[bindings[index].layout.descriptorType] += index == tableBindings[set] ? tableDescriptorCount(set) : bindings[index].layout.descriptorCount;
            }
            std::vector<VkDescriptorPoolSize> sizes;
            for (const auto& [type, count] : counts) sizes.push_back({type, count});
            return sizes;
        };
        const auto allocateVariableSet = [&](VkDescriptorPool from, VkDescriptorSetLayout layout, std::optional<std::uint32_t> variableCount) {
            const std::uint32_t variable = variableCount.value_or(0u);
            VkDescriptorSetVariableDescriptorCountAllocateInfoEXT variableInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT};
            variableInfo.descriptorSetCount = 1;
            variableInfo.pDescriptorCounts = &variable;
            VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocation.pNext = variableCount.has_value() ? &variableInfo : nullptr;
            allocation.descriptorPool = from;
            allocation.descriptorSetCount = 1;
            allocation.pSetLayouts = &layout;
            VkDescriptorSet set = VK_NULL_HANDLE;
            Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &set), "vkAllocateDescriptorSets");
            return set;
        };
        const auto createLayout = [&](const std::vector<std::uint32_t>& key, const std::vector<VkDescriptorSetLayoutBinding>& description, const std::vector<VkDescriptorBindingFlags>& flags, bool& owns) {
            if (context.descriptorCache != nullptr && !noLayoutCache) return context.descriptorCache->Layout(key, description, updateAfterBind, flags);
            ValidateDescriptorLayout(context, description, updateAfterBind);
            owns = true;
            return CreateSetLayout(context, description, updateAfterBind, flags);
        };
        if (heapStage) {
            Require(splitSets && descriptions[1].empty(), "a GPU-mode fragment stage needs split sets and no per-draw fragment bindings");
            fragmentLayoutKey = {HeapLayoutKeyTag};
        }
        _layout = createLayout(layoutKey, descriptions[0], layoutBindingFlags, ownsLayout);
        if (splitSets && !heapStage) {
            fragment.layout = createLayout(fragmentLayoutKey, descriptions[1], fragmentBindingFlags, fragment.ownsLayout);
            {
                const auto sizes = splitPoolSizes(1);
                const auto variableCount = tableVariableCount(1);
                if (context.descriptorCache != nullptr && !noPoolCache) {
                    fragment.cacheAllocation = context.descriptorCache->Allocate(fragment.layout, sizes, updateAfterBind, variableCount);
                    fragment.set = fragment.cacheAllocation.set;
                }
                if (fragment.set == VK_NULL_HANDLE) {
                    fragment.pool = CreateSetPool(context, sizes, updateAfterBind);
                    fragment.set = allocateVariableSet(fragment.pool, fragment.layout, variableCount);
                }
            }
        }
        if (splitSets) {
            const auto sizes = splitPoolSizes(0);
            const auto variableCount = tableVariableCount(0);
            if (context.descriptorCache != nullptr && !noPoolCache) {
                cacheAllocation = context.descriptorCache->Allocate(_layout, sizes, updateAfterBind, variableCount);
                _set = cacheAllocation.set;
            }
            if (_set == VK_NULL_HANDLE) {
                pool = CreateSetPool(context, sizes, updateAfterBind);
                _set = allocateVariableSet(pool, _layout, variableCount);
            }
        }
        if (!splitSets && !bindings.empty()) {
            // The set is sized from the plan: every image element becomes exactly one descriptor
            // when stage B looks it up.
            std::vector<VkDescriptorPoolSize> sizes;
            if (storageBuffers != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<std::uint32_t>(storageBuffers)});
            const auto tableDescriptors = tableBindings[0] != std::numeric_limits<std::size_t>::max() ? tableDescriptorCount(0) : 0u;
            if (plannedSampledImages + tableDescriptors != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, plannedSampledImages + tableDescriptors});
            const auto storageTableDescriptors = storageTableBindings[0] != std::numeric_limits<std::size_t>::max() ? bindings[storageTableBindings[0]].layout.descriptorCount : 0u;
            if (plannedStorageImages + storageTableDescriptors != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, plannedStorageImages + storageTableDescriptors});
            std::uint32_t samplerDescriptors = 0;
            for (const auto& binding : bindings) {
                if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) samplerDescriptors += binding.layout.descriptorCount;
            }
            if (samplerDescriptors != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLER, samplerDescriptors});
            const auto variableCount = tableVariableCount(0);
            if (context.descriptorCache != nullptr && !noPoolCache) {
                const auto allocated = context.descriptorCache->Allocate(_layout, sizes, updateAfterBind, variableCount);
                _set = allocated.set;
                cacheAllocation = allocated;
            }
            if (_set == VK_NULL_HANDLE) {
                pool = CreateSetPool(context, sizes, updateAfterBind);
                const std::uint32_t variable = variableCount.value_or(0u);
                VkDescriptorSetVariableDescriptorCountAllocateInfoEXT variableInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT};
                variableInfo.descriptorSetCount = 1;
                variableInfo.pDescriptorCounts = &variable;
                VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                allocation.pNext = variableCount.has_value() ? &variableInfo : nullptr;
                allocation.descriptorPool = pool;
                allocation.descriptorSetCount = 1;
                allocation.pSetLayouts = &_layout;
                Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &_set), "vkAllocateDescriptorSets");
            }
        }
        writeOnlyRanges.clear();
        for (const auto& item : allocations) {
            if (!item.writeOnly || item.size == 0) continue;
            const auto begin = item.address;
            const auto end = item.address + item.size;
            const bool shared = std::any_of(allocations.begin(), allocations.end(), [&](const Allocation& other) { return !other.writeOnly && other.size != 0 && other.address < end && begin < other.address + other.size; });
            if (!shared) writeOnlyRanges.emplace_back(begin, end);
        }
        timing.descriptorsMs = phase(BuildPhase::Descriptors);
        if (BuildProfiled()) {
            timing.prepareMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            addBuildPhase(BuildPhase::StageA, timing.prepareMs);
        }
    } catch (...) {
        release();
        throw;
    }
}

bool SplitsSets(const Context& context, std::span<const CompiledShader> shaders) {
    const auto fragmentSet = [](const CompiledShader& shader) { return shader.program != nullptr && std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const auto& binding) { return binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet; }); };
    const auto fragmentStage = [](const CompiledShader& shader) { return shader.stage == ShaderRecompiler::ShaderStage::Fragment; };
    return std::any_of(shaders.begin(), shaders.end(), fragmentSet) || AnyHeapShader(shaders) || (context.graphicsPipelineLibrary && std::any_of(shaders.begin(), shaders.end(), fragmentStage));
}

DescribedSetLayouts DescribeSetLayouts(const Context& context, std::span<const CompiledShader> shaders) {
    Require(context.descriptorCache != nullptr, "described set layouts need the descriptor cache");
    DescribedSetLayouts result;
    std::array<std::vector<VkDescriptorSetLayoutBinding>, 2> descriptions;
    const bool split = SplitsSets(context, shaders);
    std::array<std::vector<VkDescriptorBindingFlags>, 2> bindingFlags;
    std::array<VkShaderStageFlags, 2> tableFlags{};
    std::array<VkShaderStageFlags, 2> storageTableFlags{};
    bool updateAfterBind = false;
    bool tablesUpdateAfterBind = false;
    std::uint64_t storageBuffers = 0;
    const bool heapStage = AnyHeapShader(shaders);
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        const VkShaderStageFlags flags = VulkanStage(shader.stage);
        const bool gpuMode = HeapShader(shader);
        std::uint64_t stageStorageBuffers = 0;
        std::uint64_t stageResources = 0;
        const bool addressedStage = gpuMode && shader.program->dataAddressPush;
        for (const auto& binding : shader.program->bindings) {
            const auto set = !heapStage && binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet ? 1 : 0;
            if (heapStage && !gpuMode && binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer) continue;
            if (addressedStage) {
                using Role = ShaderRecompiler::DescriptorRole;
                const bool tableRole = binding.role == Role::ImageTable || binding.role == Role::ImageStorageTable || binding.role == Role::SamplerTable || binding.role == Role::ImageTableMap;
                const bool addressRole = binding.role == Role::BdaPagetable || binding.role == Role::FaultBuffer || (binding.role == Role::GuestBuffers && shader.program->bufferAddressCount != 0u) || (binding.role == Role::Gds && shader.program->dataAddressPush);
                const bool heapStorage = binding.role == Role::GuestImages && binding.kind == ShaderRecompiler::DescriptorKind::StorageImage && binding.descriptorSet == ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet;
                if (tableRole || addressRole || heapStorage) continue;
            }
            if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable) {
                tableFlags[split && !heapStage && shader.stage == ShaderRecompiler::ShaderStage::Fragment ? 1 : 0] |= flags;
                continue;
            }
            if (binding.role == ShaderRecompiler::DescriptorRole::ImageStorageTable) {
                storageTableFlags[split && shader.stage == ShaderRecompiler::ShaderStage::Fragment ? 1 : 0] |= flags;
                continue;
            }
            if (gpuMode) {
                const auto heapClass = ClassifyHeapBinding(binding);
                if (heapClass == HeapBindingClass::SharedImage || heapClass == HeapBindingClass::ShaderData) continue;
            }
            if (binding.role == ShaderRecompiler::DescriptorRole::SamplerTable) {
                descriptions[set].push_back({binding.binding, VK_DESCRIPTOR_TYPE_SAMPLER, context.samplerTableCapacity, flags, nullptr});
                bindingFlags[set].push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | (context.samplerTableUpdateAfterBind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0));
                tablesUpdateAfterBind |= context.samplerTableUpdateAfterBind;
                continue;
            }
            const bool image = binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers;
            const auto type = !image ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : binding.kind == ShaderRecompiler::DescriptorKind::StorageImage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : binding.kind == ShaderRecompiler::DescriptorKind::SampledImage ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER;
            if (type != VK_DESCRIPTOR_TYPE_SAMPLER) stageResources += binding.count;
            if (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                stageStorageBuffers += binding.count;
                storageBuffers += binding.count;
            }
            if (stageStorageBuffers > context.limits.maxPerStageDescriptorStorageBuffers || stageResources > context.limits.maxPerStageResources) updateAfterBind = true;
            descriptions[set].push_back({binding.binding, type, binding.count, flags, nullptr});
            bindingFlags[set].push_back(0);
        }
    }
    for (std::size_t set = 0; set < storageTableFlags.size(); ++set) {
        if (storageTableFlags[set] == 0) continue;
        descriptions[set].push_back({ShaderRecompiler::RuntimeAbi::ImageStorageTableBinding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, StorageTableCapacity(context), storageTableFlags[set], nullptr});
        bindingFlags[set].push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    }
    for (std::size_t set = 0; set < tableFlags.size(); ++set) {
        if (tableFlags[set] == 0) continue;
        descriptions[set].push_back({ShaderRecompiler::RuntimeAbi::ImageTableBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, context.imageTableCapacity, tableFlags[set], nullptr});
        bindingFlags[set].push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT | (context.imageTableUpdateAfterBind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0));
        tablesUpdateAfterBind |= context.imageTableUpdateAfterBind;
    }
    if (storageBuffers > context.limits.maxDescriptorSetStorageBuffers) updateAfterBind = true;
    for (std::size_t set = 0; set < descriptions.size(); ++set) {
        for (std::size_t index = 0; index < descriptions[set].size(); ++index) {
            if (updateAfterBind && descriptions[set][index].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) bindingFlags[set][index] |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        }
    }
    updateAfterBind = updateAfterBind || tablesUpdateAfterBind;
    for (std::size_t set = 0; set < descriptions.size(); ++set) {
        auto& key = set == 0 ? result.mainKey : result.fragmentKey;
        for (const auto& binding : descriptions[set]) key.insert(key.end(), {binding.binding, static_cast<std::uint32_t>(binding.descriptorType), binding.descriptorCount, binding.stageFlags});
        if (updateAfterBind && (set == 0 || split)) key.push_back(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT);
    }
    result.layouts.split = split;
    result.layouts.main = context.descriptorCache->Layout(result.mainKey, descriptions[0], updateAfterBind, bindingFlags[0]);
    if (heapStage) {
        Require(split && descriptions[1].empty(), "a GPU-mode stage needs split sets and no per-draw fragment bindings");
        result.fragmentKey = {HeapLayoutKeyTag};
        return result;
    }
    if (split) result.layouts.fragment = context.descriptorCache->Layout(result.fragmentKey, descriptions[1], updateAfterBind, bindingFlags[1]);
    return result;
}

ShaderRecompiler::PipelineIdentity DescribedPipelineIdentity(const ShaderRecompiler::RecompileResult& shader) {
    const auto hash = shader.spirv.ContentHash();
    if (hash == ShaderRecompiler::SpirvContentHash{}) return {};
    ShaderRecompiler::PipelineIdentity identity{hash.low, hash.high};
    const auto mix = [&](std::uint64_t value) {
        identity.low = (identity.low ^ value) * 0x9e3779b97f4a7c15ull;
        identity.low ^= identity.low >> 29u;
        identity.high = (identity.high ^ (value + 0x7f4a7c159e3779b9ull)) * 0xc2b2ae3d27d4eb4full;
        identity.high ^= identity.high >> 31u;
    };
    mix(shader.specialization.size());
    for (const auto& constant : shader.specialization) mix(constant.id | (static_cast<std::uint64_t>(constant.value) << 32u));
    mix(shader.hostSubgroupSize | (shader.fullSubgroups ? 0x10000u : 0u) | (static_cast<std::uint64_t>(shader.workgroupMemoryDwords) << 32u));
    mix(shader.bindings.size());
    for (const auto& binding : shader.bindings) {
        const bool fixedCapacity = binding.role == ShaderRecompiler::DescriptorRole::ImageTable || binding.role == ShaderRecompiler::DescriptorRole::ImageStorageTable || binding.role == ShaderRecompiler::DescriptorRole::SamplerTable;
        const std::uint64_t count = fixedCapacity ? 0u : binding.count;
        mix(binding.descriptorSet | (static_cast<std::uint64_t>(binding.binding) << 16u) | (static_cast<std::uint64_t>(binding.kind) << 40u) | (count << 48u));
    }
    if (identity.Empty()) identity.low = 1;
    return identity;
}

void ShaderResources::buildComplete() {
    PerformanceTimer frameTiming("Resources.Complete");
    BuildLookups lookups;
    Require(!completed, "cannot update a published descriptor heap");
    Require(registeredImageViewsCurrent(registeredImageViews), "AGC graphics: registered image range changed after resource preparation");
    const auto stageStart = std::chrono::steady_clock::now();
    phaseStart = stageStart;
    try {
        // The lookups run in plan order: Revalidate walks the bindings the same way, and consecutive
        // storage elements of one mip chain share the previous element's image.
        previousSampled = {};
        for (auto& source : samplerSources) {
            source.singleLevelImage = source.mipmappedImage = false;
            source.pinnedLod.reset();
        }
        for (const auto& deferred : deferredImages) resolveImageBinding(*deferred.binding, deferred.words, bindings[deferred.index], std::span<const std::shared_ptr<Sampler>>(samplers).subspan(deferred.firstSampler, deferred.samplerCount));
        previousSampled = {};
        deferredImages.clear();
        applySamplerVariants();
        // The records served their purpose: each holds the cache entry's objects as of stage A,
        // which would otherwise keep a replaced texture or an evicted storage image (and its device
        // memory) alive, outside the caches' budgets, for as long as this object is cached.
        imageRecords.clear();
        imageRecords.shrink_to_fit();
        nextImageRecord = 0;
        Require(textures.size() == plannedSampledImages && storageTextures.size() == plannedStorageImages, "image lookups disagree with the descriptor plan");
        resolveImageTables();
        Require(registeredImageViewsCurrent(registeredImageViews), "AGC graphics: registered image range changed during resource materialization");
        timing.bindingsMs += phase(BuildPhase::Images);
        requireGuestBuffers();
        guestMemory.UploadFinish(usesBda, htileClear);
        timing.uploadMs += phase(BuildPhase::Upload);
        if (usesBda) bda = std::make_unique<BdaResources>(context, guestMemory, faultReportOnly, walksPageTables);
        else if (usesFaultBuffer) bda = std::make_unique<BdaResources>(context, faultReportOnly);
        if (bda) bda->SetSrtGuards(std::move(srtGuardFaults));
        finishImageTables();
        phase(BuildPhase::Bda);
        for (const auto& binding : bindings) {
            if (!binding.addressed) continue;
            for (const auto index : binding.allocations) static_cast<void>(descriptor(allocations[index]));
        }
        if (_set != VK_NULL_HANDLE || fragment.set != VK_NULL_HANDLE) {
            // One update call for the whole set: the info arrays are sized up front so every write's
            // pointer into them stays valid until the call.
            std::size_t bufferCount = 0;
            std::size_t imageCount = 0;
            for (const auto& binding : bindings) {
                bufferCount += binding.allocations.size();
                if (binding.layout.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) imageCount += binding.flags != 0 ? binding.imageAllocations.size() : binding.layout.descriptorCount;
            }
            auto& buffers = descriptorBuffers;
            auto& images = descriptorImages;
            buffers.reserve(bufferCount);
            images.reserve(imageCount);
            auto& writes = descriptorWrites;
            writes.reserve(bindings.size());
            std::vector<std::pair<std::size_t, std::size_t>> keptWrites;
            const auto absentImage = [this](std::uint32_t binding) {
                constexpr auto bindingCount = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
                const auto view = context.nullDescriptors ? VK_NULL_HANDLE : context.placeholderImages->View(binding % bindingCount);
                return VkDescriptorImageInfo{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
            };
            for (const auto& binding : bindings) {
                if (binding.heap) continue;
                if (binding.flags != 0) {
                    const auto set = setOf(binding);
                    const bool sampled = binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    const auto& allocation = set == 0 ? cacheAllocation : fragment.cacheAllocation;
                    const bool recorded = sampled && set < tableWrites.size() && allocation.layout != VK_NULL_HANDLE;
                    std::vector<std::pair<std::uint32_t, std::uint32_t>> runs;
                    const auto firstWrite = writes.size();
                    const auto firstImage = images.size();
                    for (std::size_t first = 0; first < binding.imageAllocations.size();) {
                        if (binding.imageAllocations[first] == std::numeric_limits<std::size_t>::max()) {
                            ++first;
                            continue;
                        }
                        auto last = first;
                        while (last < binding.imageAllocations.size() && binding.imageAllocations[last] != std::numeric_limits<std::size_t>::max()) ++last;
                        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                        write.dstSet = set == 0 ? _set : fragment.set;
                        write.dstBinding = binding.layout.binding;
                        write.dstArrayElement = static_cast<std::uint32_t>(first);
                        write.descriptorCount = static_cast<std::uint32_t>(last - first);
                        write.descriptorType = binding.layout.descriptorType;
                        write.pImageInfo = images.data() + images.size();
                        for (auto element = first; element < last; ++element) {
                            const auto index = binding.imageAllocations[element];
                            if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) images.push_back({samplers[index]->Handle(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
                            else if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) images.push_back({VK_NULL_HANDLE, storageAtomic64[index] ? storageTextures[index]->Atomic64View(storageMips[index], storageFirstLayer[index]) : storageAtomic[index] ? storageTextures[index]->AtomicView(storageMips[index], storageFirstLayer[index]) : storageTextures[index]->StorageView(storageMips[index], storageFirstLayer[index], ShaderRecompiler::RuntimeAbi::UintStorageImageBinding(binding.layout.binding) || storageUint[index]), VK_IMAGE_LAYOUT_GENERAL});
                            else images.push_back({VK_NULL_HANDLE, textureFirstLayer[index] ? textures[index]->FirstLayerView() : arrayViewTextures.contains(index) ? textures[index]->ArrayView() : textures[index]->View(), textures[index]->Layout()});
                        }
                        if (recorded) runs.emplace_back(write.dstArrayElement, write.descriptorCount);
                        writes.push_back(write);
                        first = last;
                    }
                    if (!recorded) continue;
                    const auto& held = allocation.table;
                    auto& counters = TextureCounts();
                    const bool heldAlive = held != nullptr && held->textures != nullptr && std::none_of(held->textures->begin(), held->textures->end(), [](const std::weak_ptr<Texture>& texture) { return texture.expired(); });
                    if (heldAlive && held->binding == binding.layout.binding && held->runs == runs && std::equal(held->images.begin(), held->images.end(), images.begin() + static_cast<std::ptrdiff_t>(firstImage), images.end(), [](const VkDescriptorImageInfo& a, const VkDescriptorImageInfo& b) { return a.sampler == b.sampler && a.imageView == b.imageView && a.imageLayout == b.imageLayout; })) {
                        keptWrites.emplace_back(firstWrite, writes.size());
                        counters.tableWritesKept.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        counters.tableWritesMade.fetch_add(1, std::memory_order_relaxed);
                    }
                    tableWrites[set] = {true, false, binding.layout.binding, firstImage, images.size() - firstImage, std::move(runs)};
                    continue;
                }
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = setOf(binding) == 0 ? _set : fragment.set;
                write.dstBinding = binding.layout.binding;
                write.descriptorCount = binding.layout.descriptorCount;
                write.descriptorType = binding.layout.descriptorType;
                switch (binding.layout.descriptorType) {
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                        write.pBufferInfo = buffers.data() + buffers.size();
                        for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
                            const auto index = binding.allocations[element];
                            if (allocations[index].role == ShaderRecompiler::DescriptorRole::FaultBuffer) faultDescriptors.push_back({writes.size(), buffers.size(), static_cast<std::uint32_t>(element)});
                            buffers.push_back(descriptor(allocations[index]));
                        }
                        break;
                    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) {
                            if (index == std::numeric_limits<std::size_t>::max()) images.push_back(absentImage(binding.layout.binding));
                            else images.push_back({VK_NULL_HANDLE, textureSrgb[index] ? textures[index]->SrgbView(textureFirstLayer[index], textureBaseLevel[index]) : textureBaseLevel[index] ? textures[index]->BaseLevelView() : textureFirstLayer[index] ? textures[index]->FirstLayerView() : textures[index]->View(), textures[index]->Layout()});
                        }
                        Require(binding.imageAllocations.size() == binding.layout.descriptorCount, "descriptor allocations disagree with compact binding");
                        break;
                    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) {
                            if (index == std::numeric_limits<std::size_t>::max()) images.push_back(absentImage(binding.layout.binding));
                            else if (storageTextures[index] == nullptr) images.push_back({VK_NULL_HANDLE, multisampleStorage[index].view, VK_IMAGE_LAYOUT_GENERAL});
                            else images.push_back({VK_NULL_HANDLE, storageAtomic64[index] ? storageTextures[index]->Atomic64View(storageMips[index], storageFirstLayer[index]) : storageAtomic[index] ? storageTextures[index]->AtomicView(storageMips[index], storageFirstLayer[index]) : storageTextures[index]->StorageView(storageMips[index], storageFirstLayer[index], ShaderRecompiler::RuntimeAbi::UintStorageImageBinding(binding.layout.binding) || storageUint[index]), VK_IMAGE_LAYOUT_GENERAL});
                        }
                        Require(binding.imageAllocations.size() == binding.layout.descriptorCount, "descriptor allocations disagree with compact binding");
                        break;
                    case VK_DESCRIPTOR_TYPE_SAMPLER:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) images.push_back({samplers[index]->Handle(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
                        Require(binding.imageAllocations.size() == binding.layout.descriptorCount, "descriptor allocations disagree with compact binding");
                        break;
                    default: throw std::runtime_error("AGC graphics: ShaderResources encountered an unknown descriptor type while writing the descriptor set");
                }
                writes.push_back(write);
            }
            if (keptWrites.empty()) {
                context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            } else {
                std::vector<VkWriteDescriptorSet> made;
                made.reserve(writes.size());
                std::size_t next = 0;
                for (std::size_t index = 0; index < writes.size(); ++index) {
                    if (next < keptWrites.size() && index == keptWrites[next].first) {
                        index = keptWrites[next++].second - 1u;
                        continue;
                    }
                    made.push_back(writes[index]);
                }
                if (!made.empty()) context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(made.size()), made.data(), 0, nullptr);
            }
            for (auto& table : tableWrites) table.written = table.recorded;
        }
        for (const auto& allocation : allocations) {
            if (allocation.adjustment == 0) continue;
            if (allocation.pushByte >= 0) {
                pushPatches.emplace_back(static_cast<std::uint32_t>(allocation.pushByte), allocation.adjustment);
                continue;
            }
            if (IsHeapRingData(allocation.dataAllocation)) {
                heapStages[HeapRingStage(allocation.dataAllocation)].dataPatches.emplace_back(allocation.dataByte, allocation.adjustment);
                continue;
            }
            Require(allocation.dataAllocation >= 0, "a guest buffer off the storage buffer offset alignment in a shader without shader data is not implemented");
            auto& data = allocations[static_cast<std::size_t>(allocation.dataAllocation)];
            Require(data.buffer != nullptr && allocation.dataByte < data.size, "guest buffer offset lies outside the shader's data buffer");
            data.buffer->Bytes()[allocation.dataByte] = static_cast<std::byte>(allocation.adjustment);
            dataPatches.push_back({static_cast<std::size_t>(allocation.dataAllocation), allocation.dataByte, allocation.adjustment});
        }
        timing.descriptorsMs += phase(BuildPhase::Descriptors);
        noteReusable();
        keptSeen.store(Texture::Refills(), std::memory_order_relaxed);
        completed = true;
        if (BuildProfiled()) {
            timing.completeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            addBuildPhase(BuildPhase::StageB, timing.completeMs);
            reportDescriptorCaches();
        }
    } catch (...) {
        release();
        throw;
    }
}

// Whether this build can serve later dispatches of the same content (see ContentKey): nothing to do
// once the GPU completed (no BDA, no copied written buffers, no lease) and every guest buffer bound
// in place through a host import (or staged in device memory from one, see
// GuestBufferMemory::DirectRegions) whose identity is recorded for Revalidate.
namespace {

// APS5_NO_TEMPLATE_DATA_REFRESH=1: compute keys keep the ShaderData/FlattenedSrt words and no
// template refreshes its data buffers (see ShaderResources::ContentKey).
bool TemplateDataRefresh() {
    static const bool enabled = std::getenv("APS5_NO_TEMPLATE_DATA_REFRESH") == nullptr;
    return enabled;
}

bool DataRole(ShaderRecompiler::DescriptorRole role) {
    return role == ShaderRecompiler::DescriptorRole::ShaderData || role == ShaderRecompiler::DescriptorRole::FlattenedSrt;
}

// The largest data buffer vkCmdUpdateBuffer refreshes; a template with a bigger one is not reused.
constexpr std::size_t MaxRefreshBytes = 65536;

// FNV-1a over one data buffer's words, its count first (DataWordsHash).
constexpr std::uint64_t FnvOffset = 14695981039346656037ull;
constexpr std::uint64_t FnvPrime = 1099511628211ull;
void mixDataWords(std::uint64_t& hash, std::span<const std::uint32_t> words) {
    hash = (hash ^ static_cast<std::uint64_t>(words.size())) * FnvPrime;
    for (const auto word : words) hash = (hash ^ word) * FnvPrime;
}

// ResourceCache::Find and Touch calls (the [rescache] line).
std::atomic<std::uint64_t> resourceCacheFinds{0};
std::atomic<std::uint64_t> resourceCacheTouches{0};
std::atomic<std::uint64_t> depthGuardRefusals{0};
thread_local bool indexProof = false;

}

namespace {

bool LoopGuardEnabled() {
    static const bool enabled = std::getenv("APS5_LOOP_GUARD") != nullptr;
    return enabled;
}

}

void ShaderResources::noteReusable() {
    const bool spaceReusable = !htileClear.has_value() && drawBuild && usesBda && bdaReadOnly && !SpaceReuseDisabled() && guestMemory.SpaceOnly() && !guestMemory.HasCopiedWrites() && multisampleSources.empty() && !(bda != nullptr && LoopGuardEnabled());
    // Taken whether or not the object turns out reusable: the lookups' records are this build's.
    captureValidation(spaceReusable);
    reusable = false;
    spaceReuse = false;
    importsReconcilable = false;
    spaceReuse = false;
    expressRegionsAt.store(0, std::memory_order_relaxed);
    directRegions.clear();
    notReusable = NotReusable::None;
    if (!spaceReusable && (htileClear.has_value() || guestMemory.HasCopiedWrites() || HoldsLease() || usesBda || !multisampleSources.empty() || (bda != nullptr && LoopGuardEnabled()))) {
        notReusable = htileClear.has_value() ? NotReusable::HtileClear : guestMemory.HasCopiedWrites() ? NotReusable::CopiedWrites : !multisampleSources.empty() ? NotReusable::MultisampleSources : NotReusable::LeaseOrTable;
        return;
    }
    if (std::any_of(textures.begin(), textures.end(), [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->RefreshedPerUse(); })) {
        notReusable = NotReusable::PerUseTextures;
        return;
    }
    if (TemplateDataRefresh() && std::any_of(allocations.begin(), allocations.end(), [](const Allocation& allocation) { return allocation.buffer != nullptr && !allocation.guest && allocation.role != ShaderRecompiler::DescriptorRole::ImageTableMap && allocation.size > MaxRefreshBytes; })) {
        notReusable = NotReusable::DataBuffers;
        return;
    }
    if (spaceReusable) {
        spaceReuse = true;
        reusable = true;
        return;
    }
    notReusable = NotReusable::NoImportSerial;
    const auto regions = guestMemory.DirectRegions();
    if (!regions.has_value()) return;
    bool mirrored = false;
    for (const auto& [begin, end] : *regions) {
        // No reconcile here: the upload just took these imports, and the set is about to be recorded
        // against them.
        auto serial = HostImportSerial(context, begin, static_cast<std::size_t>(end - begin), false);
        // Read-only image mirrors (exe ranges) are as stable as imports; their serials have the top
        // bit set, so the two spaces never collide.
        const bool imported = serial != 0;
        if (serial == 0) serial = ImageMirrorSerial(context, begin, static_cast<std::size_t>(end - begin));
        if (serial == 0) return;
        directRegions.push_back({begin, end, serial});
        mirrored = mirrored || !imported;
    }
    bool bounded = false;
    for (const auto& cut : guestMemory.CutRegions()) {
        for (auto& region : directRegions) {
            if (region.begin != cut.begin) continue;
            region.regionEnd = cut.end;
            region.committedEnd = cut.committedEnd;
            bounded = true;
        }
    }
    importsReconcilable = !mirrored && !bounded;
    notReusable = NotReusable::None;
    reusable = true;
}

std::uint64_t ShaderResources::DepthGuardRefusals() {
    return depthGuardRefusals.load(std::memory_order_relaxed);
}

bool ShaderResources::NeverReusable(std::span<const CompiledShader> shaders) {
    const bool loopGuard = LoopGuardEnabled();
    return std::any_of(shaders.begin(), shaders.end(), [loopGuard](const CompiledShader& shader) {
        return shader.program != nullptr && std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [&](const ShaderRecompiler::DescriptorBinding& binding) { return (binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable && (SpaceReuseDisabled() || shader.program->bdaStores)) || (loopGuard && binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer); });
    });
}

bool ShaderResources::SpaceReuseDisabled() {
    static const bool disabled = std::getenv("APS5_NO_ADDRESS_DRAW_REUSE") != nullptr;
    return disabled;
}

// APS5_PROFILE_DRAW: the per-device descriptor caches' counters, every 10 s.
void ShaderResources::reportDescriptorCaches() const {
    static AgcDriver::Mutex reportMutex;
    static auto lastReport = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(reportMutex);
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    const auto descriptors = context.descriptorCache != nullptr ? context.descriptorCache->Counters() : DescriptorCache::Stats{};
    const auto samplerHits = context.samplerCache != nullptr ? context.samplerCache->Hits() : 0;
    const auto samplerMisses = context.samplerCache != nullptr ? context.samplerCache->Misses() : 0;
    AgcDriver::ProfilePrint_nid_no_patch("[descriptors] layouts %llu hits / %llu created, sets %llu from %llu pools and %llu dedicated pools, samplers %llu hits / %llu created\n", static_cast<unsigned long long>(descriptors.layoutHits), static_cast<unsigned long long>(descriptors.layoutMisses), static_cast<unsigned long long>(descriptors.sets), static_cast<unsigned long long>(descriptors.pools), static_cast<unsigned long long>(descriptors.dedicatedPools), static_cast<unsigned long long>(samplerHits), static_cast<unsigned long long>(samplerMisses));
}

constexpr std::size_t CompactTableKeyWords = 8192;

struct TableKeyHash {
    static constexpr std::uint64_t P1 = 0x9e3779b185ebca87ull;
    static constexpr std::uint64_t P2 = 0xc2b2ae3d27d4eb4full;
    std::uint64_t lanes[4] = {P1 + P2, P2, 0, 0 - P1};
    std::uint64_t total = 0;
    static std::uint64_t round(std::uint64_t lane, std::uint64_t value) {
        lane += value * P2;
        lane = (lane << 31u) | (lane >> 33u);
        return lane * P1;
    }
    void bytes(const void* data, std::size_t size) {
        const auto* at = static_cast<const unsigned char*>(data);
        std::size_t i = 0;
        for (; i + 32 <= size; i += 32) {
            std::uint64_t v[4];
            std::memcpy(v, at + i, sizeof(v));
            lanes[0] = round(lanes[0], v[0]);
            lanes[1] = round(lanes[1], v[1]);
            lanes[2] = round(lanes[2], v[2]);
            lanes[3] = round(lanes[3], v[3]);
        }
        for (; i < size; ++i) lanes[i % 4] = round(lanes[i % 4], at[i] | (static_cast<std::uint64_t>(i) << 8u));
        total += size;
        add(size);
    }
    void add(std::uint64_t value) {
        const auto lane = total++ % 4;
        lanes[lane] = round(lanes[lane], value);
    }
    void bits(const ShaderRecompiler::ElementFlags& set) {
        add(set.size());
        set.ForEachSet([&](std::size_t index) { add(index); });
        add(~0ull);
    }
    std::uint64_t a() const { return round(lanes[0], lanes[1]) ^ round(lanes[2], lanes[3]); }
    std::uint64_t b() const { return round(lanes[3] ^ total, lanes[0]) + round(lanes[1], lanes[2] ^ P1); }
};

template<typename TVisitor>
void VisitContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers, const TVisitor& visit, bool movableWrites = false) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    visit(dataWords ? 1u : 0u);
    visit(static_cast<std::uint32_t>(shader.stage));
    visit(static_cast<std::uint32_t>(program.PipelineVariantId()));
    visit(static_cast<std::uint32_t>(program.PipelineVariantId() >> 32u));
    if (shader.heap != nullptr && program.heapTemplateId != 0) {
        visit(0u);
        visit(HeapLayoutKeyTag);
        visit(static_cast<std::uint32_t>(program.heapTemplateId));
        visit(static_cast<std::uint32_t>(program.heapTemplateId >> 32u));
        for (const auto& binding : program.bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages && binding.role != ShaderRecompiler::DescriptorRole::GuestSamplers) continue;
            visit(binding.binding);
            for (const auto word : DrawGuestWords(shader, binding)) visit(word);
        }
        return;
    }
    visit(static_cast<std::uint32_t>(program.bindings.size()));
    const auto packBits = [&](const ShaderRecompiler::ElementFlags& bits) {
        visit(static_cast<std::uint32_t>(bits.size()));
        std::uint32_t word = 0;
        for (std::size_t i = 0; i < bits.size(); ++i) {
            if (bits[i]) word |= 1u << (i % 32u);
            if (i % 32u == 31u || i + 1 == bits.size()) {
                visit(word);
                word = 0;
            }
        }
    };
    for (const auto& binding : program.bindings) {
        for (const auto word : {static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.descriptorSet, binding.binding, binding.count, binding.readOnly ? 1u : 0u, binding.imageShape.has_value() ? static_cast<std::uint32_t>(*binding.imageShape) + 1u : 0u, static_cast<std::uint32_t>(DrawGuestWords(shader, binding).size())}) visit(word);
        if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable && DrawGuestWords(shader, binding).size() > CompactTableKeyWords) {
            TableKeyHash hash;
            hash.bytes(DrawGuestWords(shader, binding).data(), DrawGuestWords(shader, binding).size() * sizeof(std::uint32_t));
            hash.bytes(binding.imageSamplers.data(), binding.imageSamplers.size() * sizeof(binding.imageSamplers[0]));
            hash.bytes(binding.imageShapes.data(), binding.imageShapes.size() * sizeof(binding.imageShapes[0]));
            for (const auto* set : {&binding.imageWritten, &binding.samplerDepthCompare, &binding.imageDepthCompare, &binding.imageAtomic, &binding.samplerUnnormalized, &binding.samplerIntegerBorder, &binding.samplerCoordinatesIgnored, &binding.imageUnnormalized, &binding.imageUnnormalizedBaseLevel, &binding.bufferWritten}) hash.bits(*set);
            const auto first = hash.a();
            const auto second = hash.b();
            for (const auto word : {0xc0de7ab1u, static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(first >> 32u), static_cast<std::uint32_t>(second), static_cast<std::uint32_t>(second >> 32u)}) visit(word);
            continue;
        }
        if (movableBuffers && binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers && DrawGuestWords(shader, binding).size() == static_cast<std::size_t>(binding.count) * 4u) {
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = DrawGuestWords(shader, binding).data() + static_cast<std::size_t>(element) * 4u;
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                const bool empty = words[2] == 0 || (words[0] == 0 && (words[1] & 0xffffu) == 0);
                if ((written && !movableWrites) || empty) { for (std::uint32_t index = 0; index < 4u; ++index) visit(words[index]); }
                else { for (const auto word : {0u, words[1] & 0xffff0000u, 0u, words[3]}) visit(word); }
            }
        } else if (binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap && DrawGuestWords(shader, binding).size() > CompactTableKeyWords) {
            TableKeyHash hash;
            hash.bytes(DrawGuestWords(shader, binding).data(), DrawGuestWords(shader, binding).size() * sizeof(std::uint32_t));
            const auto first = hash.a();
            const auto second = hash.b();
            for (const auto word : {0xc0de7ab2u, static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(first >> 32u), static_cast<std::uint32_t>(second), static_cast<std::uint32_t>(second >> 32u)}) visit(word);
        } else if ((dataWords && !movableBuffers) || !DataRole(binding.role)) {
            for (const auto word : DrawGuestWords(shader, binding)) visit(word);
        }
        packBits(binding.imageWritten);
        packBits(binding.samplerDepthCompare);
        packBits(binding.imageDepthCompare);
        packBits(binding.imageAtomic);
        packBits(binding.samplerUnnormalized);
        packBits(binding.samplerIntegerBorder);
        packBits(binding.samplerCoordinatesIgnored);
        packBits(binding.imageUnnormalized);
        packBits(binding.imageUnnormalizedBaseLevel);
        // Read-only elements are bound without a write set: an object built for one written set
        // must not serve a build with another (the variant implies it, this makes it explicit).
        packBits(binding.bufferWritten);
        visit(static_cast<std::uint32_t>(binding.imageSamplers.size()));
        for (const auto sampler : binding.imageSamplers) {
            visit(static_cast<std::uint32_t>(sampler));
            visit(static_cast<std::uint32_t>(sampler >> 32u));
        }
        visit(static_cast<std::uint32_t>(binding.imageShapes.size()));
        for (const auto shape : binding.imageShapes) visit(static_cast<std::uint32_t>(shape));
    }
}

std::vector<std::uint32_t> ShaderResources::ContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers, bool movableWrites) {
    std::vector<std::uint32_t> key;
    AppendContentKey(shader, dataWords, movableBuffers, key, movableWrites);
    return key;
}

void ShaderResources::AppendContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers, std::vector<std::uint32_t>& key, bool movableWrites) {
    Require(shader.program != nullptr, "missing compiled shader");
    std::size_t words = key.size() + 5;
    const auto addWords = [&](std::size_t count) {
        if (count > key.max_size() - words) throw std::length_error("resource content key exceeds vector limits");
        words += count;
    };
    for (const auto& binding : shader.program->bindings) {
        addWords(20);
        if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable && binding.guestDescriptor.size() > CompactTableKeyWords) continue;
        addWords(binding.imageSamplers.size() * 2u);
        addWords(binding.imageShapes.size());
        if (binding.role == ShaderRecompiler::DescriptorRole::ImageTableMap && binding.guestDescriptor.size() > CompactTableKeyWords) addWords(5);
        else if ((dataWords && !movableBuffers) || !DataRole(binding.role)) addWords(binding.guestDescriptor.size());
        for (const auto* bits : {&binding.imageWritten, &binding.samplerDepthCompare, &binding.imageDepthCompare, &binding.imageAtomic, &binding.samplerUnnormalized, &binding.samplerIntegerBorder, &binding.samplerCoordinatesIgnored, &binding.imageUnnormalized, &binding.imageUnnormalizedBaseLevel, &binding.bufferWritten}) addWords(bits->size() / 32u + (bits->size() % 32u != 0 ? 1u : 0u));
    }
    key.reserve(words);
    VisitContentKey(shader, dataWords, movableBuffers, [&](std::uint32_t word) { key.push_back(word); }, movableBuffers && movableWrites);
}

namespace {

using Component = ContentKeyComponent;

bool LabelShaderKey(std::span<const std::uint32_t> key, std::size_t at, std::size_t end, bool movable, std::vector<std::uint8_t>& labels) {
    const auto label = [&](std::size_t first, std::size_t count, Component component) {
        if (first + count > end) return false;
        std::fill(labels.begin() + static_cast<std::ptrdiff_t>(first), labels.begin() + static_cast<std::ptrdiff_t>(first + count), static_cast<std::uint8_t>(component));
        return true;
    };
    if (!label(at, 5, Component::Layout)) return false;
    const bool dataWords = key[at] != 0;
    const auto bindings = key[at + 4];
    at += 5;
    for (std::uint32_t binding = 0; binding < bindings; ++binding) {
        if (!label(at, 8, Component::Layout)) return false;
        const auto role = static_cast<ShaderRecompiler::DescriptorRole>(key[at + 1]);
        const auto size = key[at + 7];
        at += 8;
        const auto words = DataRole(role) && !(dataWords && !movable) ? 0u : size;
        if (at + words > end) return false;
        for (std::uint32_t i = 0; i < words; ++i) {
            auto component = Component::Other;
            if (role == ShaderRecompiler::DescriptorRole::GuestBuffers) component = static_cast<Component>(static_cast<std::size_t>(Component::BufferWord0) + i % 4u);
            else if (role == ShaderRecompiler::DescriptorRole::GuestImages || role == ShaderRecompiler::DescriptorRole::ImageTable) component = Component::Image;
            else if (role == ShaderRecompiler::DescriptorRole::GuestSamplers || role == ShaderRecompiler::DescriptorRole::SamplerTable) component = Component::Sampler;
            else if (DataRole(role)) component = Component::ShaderData;
            labels[at + i] = static_cast<std::uint8_t>(component);
        }
        at += words;
        for (int flags = 0; flags < 10; ++flags) {
            if (at >= end) return false;
            const auto count = 1 + (static_cast<std::size_t>(key[at]) + 31) / 32;
            if (!label(at, count, Component::Other)) return false;
            at += count;
        }
        if (at >= end) return false;
        const auto samplers = static_cast<std::size_t>(key[at]);
        if (!label(at, 1 + samplers, Component::Sampler)) return false;
        at += 1 + samplers;
        if (at >= end) return false;
        const auto shapes = static_cast<std::size_t>(key[at]);
        if (!label(at, 1 + shapes, Component::Image)) return false;
        at += 1 + shapes;
    }
    return at == end;
}

bool LabelContentKey(std::span<const std::uint32_t> key, bool draw, bool movable, std::vector<std::uint8_t>& labels) {
    labels.assign(key.size(), static_cast<std::uint8_t>(Component::Layout));
    if (!draw) {
        if (LabelShaderKey(key, 0, key.size(), movable, labels)) return true;
        if (key.size() < 2) return false;
        labels.assign(key.size(), static_cast<std::uint8_t>(Component::Layout));
        if (!LabelShaderKey(key, 0, key.size() - 2, movable, labels)) return false;
        labels[key.size() - 2] = static_cast<std::uint8_t>(Component::Other);
        labels[key.size() - 1] = static_cast<std::uint8_t>(Component::Other);
        return true;
    }
    if (key.size() < 4) return false;
    std::fill(labels.begin() + 1, labels.begin() + 3, static_cast<std::uint8_t>(Component::Other));
    std::size_t at = 4;
    for (std::uint32_t stage = 0; stage < key[3]; ++stage) {
        if (at >= key.size()) return false;
        const auto size = static_cast<std::size_t>(key[at]);
        ++at;
        if (at + size > key.size() || !LabelShaderKey(key, at, at + size, movable, labels)) return false;
        at += size;
    }
    if (at >= key.size()) return false;
    for (auto tail = at + 1; tail < key.size(); ++tail) labels[tail] = static_cast<std::uint8_t>(Component::DrawTarget);
    return true;
}

}

bool ShaderResources::DiffContentKeys(std::span<const std::uint32_t> previous, std::span<const std::uint32_t> key, bool draw, ContentKeyDiff& diff) {
    diff = {};
    std::vector<std::uint8_t> labels;
    bool labelled = false;
    for (const bool movable : {draw, false}) {
        labelled = LabelContentKey(key, draw, movable, labels);
        if (labelled) break;
    }
    if (!labelled || previous.size() != key.size()) {
        diff[static_cast<std::size_t>(Component::Layout)] = true;
        return false;
    }
    for (std::size_t i = 0; i < key.size(); ++i) {
        if (previous[i] != key[i]) diff[labels[i]] = true;
    }
    return true;
}

bool MatchesContentKey(const CompiledShader& shader, std::span<const std::uint32_t> key) {
    std::size_t cursor = 0;
    bool equal = true;
    VisitContentKey(shader, false, true, [&](std::uint32_t word) {
        equal &= cursor < key.size() && key[cursor] == word;
        ++cursor;
    });
    return equal && cursor == key.size();
}

bool ShaderResources::ContentKeyMatches(const CompiledShader& shader, std::span<const std::uint32_t> key, bool dataWords, bool movableBuffers, bool movableWrites) {
    std::size_t cursor = 0;
    bool equal = true;
    VisitContentKey(shader, dataWords, movableBuffers, [&](std::uint32_t word) {
        equal &= cursor < key.size() && key[cursor] == word;
        ++cursor;
    }, movableBuffers && movableWrites);
    return equal && cursor == key.size();
}

namespace {

// APS5_PROFILE_DRAW: Revalidate outcomes and time, printed as [rescache] every 10 s next to the
// device's hit/miss line: failed = the object could not be reused (whichever image path it took);
// of the reused ones, fast = every image proved current from stamps, full = the lookups were repeated.
// Why the fast path left an object to the full walk (the "fast-fail by reason" counts).
using FastFail = ShaderResources::FastFail;
constexpr std::array<const char*, static_cast<std::size_t>(FastFail::Count)> FastFailNames{"no record", "collect", "pending image", "evicted image", "memory changed", "keys", "cleared view", "storage keys", "depth surface", "replaced texture", "audit"};
using OwnRefreshFallback = ShaderResources::OwnRefreshFallback;
constexpr std::array<const char*, static_cast<std::size_t>(OwnRefreshFallback::Count)> OwnRefreshFallbackNames{"disabled", "snapshot texture", "cleared view", "foreign view", "surface key", "not imported", "uncached", "re-run failed"};

struct RevalidateProfile {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> nanoseconds{0};
    std::atomic<std::uint64_t> fast{0};
    std::atomic<std::uint64_t> full{0};
    std::atomic<std::uint64_t> failed{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FastFail::Count)> fastFails{};
    // Epoch gate: registry scans skipped because the pending serial was unchanged, and import
    // serial loops skipped because the table's identity was unchanged.
    std::atomic<std::uint64_t> serialSkips{0};
    std::atomic<std::uint64_t> importSkips{0};
    // Elements with DCC keys the fast path accepted through the images' key proofs.
    std::atomic<std::uint64_t> keysProven{0};
    std::atomic<std::uint64_t> storageKeysProven{0};
    // The 'pending image' fast-fails by what overlapped: a surface with an own object (the query
    // excepts it, so the overlapping image is foreign to the surface) or a snapshot texture (no own
    // object); and the fails whose overlapping surfaces all have an own object (the T1 trim of
    // design_cpu_final applies: flush the foreign image, refresh the own object, no full walk).
    std::atomic<std::uint64_t> pendingForeign{0};
    std::atomic<std::uint64_t> pendingSnapshot{0};
    std::atomic<std::uint64_t> pendingT1Eligible{0};
    // The full walks by the reason of the fast proof that preceded them (fastFails also counts the
    // Pending failures the own-object refresh resolved), the Pending failures it resolved (with the
    // objects it refreshed: storage images, view sources) and those it left to the walk by reason;
    // view surfaces a foreign image overlapped that the proof accepted because FindPending names
    // their own source; surfaces a foreign image was still pending over after their own object's
    // refresh (its alias, a straddling image) that the re-run accepted; proofs the verify switch
    // checked against the full walk.
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FastFail::Count)> fullByReason{};
    std::atomic<std::uint64_t> ownRefreshed{0};
    std::atomic<std::uint64_t> ownStorageRefreshes{0};
    std::atomic<std::uint64_t> ownViewRefreshes{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(OwnRefreshFallback::Count)> ownFallbacks{};
    std::atomic<std::uint64_t> ownSourceViews{0};
    std::atomic<std::uint64_t> refreshedOverlaps{0};
    std::atomic<std::uint64_t> proofsVerified{0};
    std::atomic<std::uint64_t> indexNanoseconds{0};
    std::atomic<std::int64_t> lastReport{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FastFail::Count)> fullNanoseconds{};
    std::atomic<std::uint64_t> refusedSampled{0};
    std::atomic<std::uint64_t> refusedStorage{0};
    std::atomic<std::uint64_t> refusedUncommitted{0};
    std::atomic<std::uint64_t> refusedUnwatched{0};
    std::atomic<std::uint64_t> refusedOther{0};
    std::mutex refusedMutex;
    std::unordered_map<std::uint64_t, std::uint64_t> refusedSurfaces;
};

RevalidateProfile& Revalidations() {
    static RevalidateProfile profile;
    return profile;
}

void countFastFail(FastFail reason) {
    if (BuildProfiled() && !indexProof) Revalidations().fastFails[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countPendingFail(std::span<const StorageTexture::PendingQuery> pending) {
    if (!BuildProfiled() || indexProof) return;
    auto& profile = Revalidations();
    bool eligible = true;
    for (const auto& query : pending) {
        if (!query.overlaps) continue;
        if (query.except != nullptr) {
            profile.pendingForeign.fetch_add(1, std::memory_order_relaxed);
        } else {
            profile.pendingSnapshot.fetch_add(1, std::memory_order_relaxed);
            eligible = false;
        }
    }
    if (eligible) profile.pendingT1Eligible.fetch_add(1, std::memory_order_relaxed);
}

void countKeysProven(bool storage) {
    if (BuildProfiled()) (storage ? Revalidations().storageKeysProven : Revalidations().keysProven).fetch_add(1, std::memory_order_relaxed);
}

void countFullWalk(FastFail reason) {
    if (BuildProfiled() && reason != FastFail::Count) Revalidations().fullByReason[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countFullWalkTime(FastFail reason, std::chrono::steady_clock::time_point start) {
    if (!BuildProfiled() || reason == FastFail::Count) return;
    const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    Revalidations().fullNanoseconds[static_cast<std::size_t>(reason)].fetch_add(static_cast<std::uint64_t>(nanoseconds), std::memory_order_relaxed);
}

struct UntrackedVerification {
    std::atomic<std::uint64_t> checked{0}, disagreed{0};
};

UntrackedVerification& UntrackedVerified() {
    static UntrackedVerification verification;
    static const bool registered = [] {
        std::atexit([] {
            const auto& total = UntrackedVerified();
            if (total.checked.load() != 0) std::fprintf(stderr, "[verify-untracked] at exit: %llu fast proofs over untracked memory checked by the full walk, %llu disagreed\n", static_cast<unsigned long long>(total.checked.load()), static_cast<unsigned long long>(total.disagreed.load()));
        });
        return true;
    }();
    (void)registered;
    return verification;
}

constexpr std::size_t RefusedSurfaceLimit = 65536;

void countCollectRefusal(bool storage, std::uint64_t address, std::size_t bytes) {
    if (!BuildProfiled() || indexProof) return;
    auto& profile = Revalidations();
    (storage ? profile.refusedStorage : profile.refusedSampled).fetch_add(1, std::memory_order_relaxed);
    const bool accessible = GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes);
    const bool watched = accessible && GuestMemory::Watched(address, bytes);
    (!accessible ? profile.refusedUncommitted : !watched ? profile.refusedUnwatched : profile.refusedOther).fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(profile.refusedMutex);
    if (profile.refusedSurfaces.size() < RefusedSurfaceLimit || profile.refusedSurfaces.contains(address)) profile.refusedSurfaces[address] = bytes;
}

void countOwnRefreshFallback(OwnRefreshFallback reason) {
    if (BuildProfiled()) Revalidations().ownFallbacks[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countOwnRefresh(bool storage) {
    TextureCounts().ownRefreshes.fetch_add(1, std::memory_order_relaxed);
    if (BuildProfiled()) (storage ? Revalidations().ownStorageRefreshes : Revalidations().ownViewRefreshes).fetch_add(1, std::memory_order_relaxed);
}

// APS5_NO_OWN_IMAGE_REFRESH=1: a Pending failure of the fast proof takes the full walk, as before
// the own-object refresh (T1).
bool OwnImageRefreshEnabled() {
    static const bool disabled = std::getenv("APS5_NO_OWN_IMAGE_REFRESH") != nullptr;
    return !disabled;
}

// APS5_VERIFY_PROOFS=1: after every own-object refresh the full walk runs beside the proof and the
// process aborts when it would return another object or upload again (a different decision).
bool VerifyProofs() {
    static const bool enabled = std::getenv("APS5_VERIFY_PROOFS") != nullptr;
    return enabled;
}

// Whether a sampled surface's keys are the very keys of the image the view follows (the same
// metadata, extent, format and alpha placement: the same scan), so the image's proof serves it.
bool SameKeySurface(const StorageTexture* source, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    if (source == nullptr) return false;
    const auto& own = source->Descriptor();
    return own.dccAddress == resource.dccAddress && source->GuestBytes() == guestBytes && own.format == resource.format && own.dccAlphaOnMsb == resource.dccAlphaOnMsb;
}

void countRevalidate(bool fast, bool ok, std::chrono::steady_clock::time_point start) {
    auto& profile = Revalidations();
    const auto now = std::chrono::steady_clock::now();
    profile.calls.fetch_add(1, std::memory_order_relaxed);
    profile.nanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count()), std::memory_order_relaxed);
    (!ok ? profile.failed : fast ? profile.fast : profile.full).fetch_add(1, std::memory_order_relaxed);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto last = profile.lastReport.load();
    if (nowMs - last < 10000 || !profile.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto byReason = [](auto& counts, const auto& names) {
        std::string reasons;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto count = counts[i].exchange(0, std::memory_order_relaxed);
            if (count == 0) continue;
            reasons += " " + std::string(names[i]) + " " + std::to_string(count);
        }
        return reasons;
    };
    const auto reasons = byReason(profile.fastFails, FastFailNames);
    const auto fullReasons = byReason(profile.fullByReason, FastFailNames);
    const auto fallbacks = byReason(profile.ownFallbacks, OwnRefreshFallbackNames);
    std::string walkMs;
    for (std::size_t i = 0; i < FastFailNames.size(); ++i) {
        const auto nanoseconds = profile.fullNanoseconds[i].exchange(0, std::memory_order_relaxed);
        if (nanoseconds == 0) continue;
        char text[64];
        std::snprintf(text, sizeof(text), " %s %.1f", FastFailNames[i], static_cast<double>(nanoseconds) / 1e6);
        walkMs += text;
    }
    std::uint64_t refusedSurfaces = 0;
    std::uint64_t refusedBytes = 0;
    {
        std::lock_guard lock(profile.refusedMutex);
        refusedSurfaces = profile.refusedSurfaces.size();
        for (const auto& surface : profile.refusedSurfaces) refusedBytes += surface.second;
        profile.refusedSurfaces.clear();
    }
    AgcDriver::ProfilePrint_nid_no_patch("[rescache] revalidate (10 s): %llu calls %.1f ms: fast %llu, full %llu, failed %llu; fast-fail by reason:%s; full walks: T1 refreshed %llu (%llu storage images, %llu view sources, %llu views served by their own pending source, %llu overlaps left by the refresh accepted), full by reason:%s, T1 fallback by reason:%s, proofs verified %llu; fast-fail pending image: own 0 (the query excepts the own object), foreign %llu, snapshot %llu, T1 eligible %llu; keys proven: %llu sampled, %llu storage; epoch gate: %llu registry scans skipped, %llu import loops skipped; lookup log: %llu builds and full walks kept their records past the bound; full walk ms by reason:%s; collect refusals: sampled %llu, storage %llu (uncommitted pages %llu, unwatched %llu, other %llu) over %llu surfaces, %.1f MiB\n", static_cast<unsigned long long>(profile.calls.exchange(0, std::memory_order_relaxed)), profile.nanoseconds.exchange(0, std::memory_order_relaxed) / 1e6, static_cast<unsigned long long>(profile.fast.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.full.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.failed.exchange(0, std::memory_order_relaxed)), reasons.c_str(), static_cast<unsigned long long>(profile.ownRefreshed.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.ownStorageRefreshes.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.ownViewRefreshes.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.ownSourceViews.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.refreshedOverlaps.exchange(0, std::memory_order_relaxed)), fullReasons.c_str(), fallbacks.c_str(), static_cast<unsigned long long>(profile.proofsVerified.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.pendingForeign.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.pendingSnapshot.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.pendingT1Eligible.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.keysProven.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.storageKeysProven.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.serialSkips.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.importSkips.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(lookupBuildsPastBound.exchange(0, std::memory_order_relaxed)), walkMs.c_str(), static_cast<unsigned long long>(profile.refusedSampled.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.refusedStorage.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.refusedUncommitted.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.refusedUnwatched.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(profile.refusedOther.exchange(0, std::memory_order_relaxed)), static_cast<unsigned long long>(refusedSurfaces), static_cast<double>(refusedBytes) / 1048576.0);
    if (!GuestMemory::WriteIndexEnabled()) return;
    const auto index = GuestMemory::WriteIndexCounted();
    GuestMemory::ResetWriteIndexCounts();
    std::string sources;
    constexpr std::array<const char*, static_cast<std::size_t>(GuestMemory::WriteBumpSource::Count)> SourceNames{"cpu", "driver", "pending", "unwatch", "forget", "release", "import"};
    for (std::size_t i = 0; i < SourceNames.size(); ++i) sources += " " + std::string(SourceNames[i]) + " " + std::to_string(index.bySource[i]);
    static std::array<std::uint64_t, IndexCountCount> reported{};
    std::array<std::uint64_t, IndexCountCount> counts{};
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const auto total = IndexCounts()[i].load(std::memory_order_relaxed);
        counts[i] = total - reported[i];
        reported[i] = total;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[rescache] write index (10 s): %llu proofs %.1f ms: %llu accepted, %llu partial (%llu bumped elements, %llu looked up again, %llu re-bound), %llu unwatched elements compared, %llu fallbacks, %llu builds subscribed; %llu subscriptions over %llu blocks, %llu bumps by source:%s, %llu sweeps (%llu failed), %llu bump lists overflowed, writers' index lookups %llu in %.1f ms\n", static_cast<unsigned long long>(counts[IndexAccepted] + counts[IndexPartial] + counts[IndexFallback]), profile.indexNanoseconds.exchange(0, std::memory_order_relaxed) / 1e6, static_cast<unsigned long long>(counts[IndexAccepted]), static_cast<unsigned long long>(counts[IndexPartial]), static_cast<unsigned long long>(counts[IndexBumped]), static_cast<unsigned long long>(counts[IndexLookedUp]), static_cast<unsigned long long>(counts[IndexRebound]), static_cast<unsigned long long>(counts[IndexCompared]), static_cast<unsigned long long>(counts[IndexFallback]), static_cast<unsigned long long>(counts[IndexBuilds]), static_cast<unsigned long long>(index.subscriptions), static_cast<unsigned long long>(index.blocks), static_cast<unsigned long long>(index.bumps), sources.c_str(), static_cast<unsigned long long>(index.sweeps), static_cast<unsigned long long>(index.sweepFailures), static_cast<unsigned long long>(index.overflows), static_cast<unsigned long long>(index.lookups), static_cast<double>(index.lookupNanoseconds) / 1e6);
}

// APS5_NO_EPOCH_REVALIDATE=1: the per-element registry, cache and stamp checks and the per-region
// flush and import lookups of every Revalidate, as before the epoch gate.
bool EpochRevalidate() {
    static const bool enabled = std::getenv("APS5_NO_EPOCH_REVALIDATE") == nullptr;
    return enabled;
}

std::chrono::steady_clock::duration TemplateAuditInterval() {
    static const auto interval = [] {
        const char* text = std::getenv("APS5_TEMPLATE_AUDIT_MS");
        return std::chrono::milliseconds(text != nullptr ? std::strtoull(text, nullptr, 10) : 1000ull);
    }();
    return interval;
}

bool AuditEveryInterval() {
    static const bool always = std::getenv("APS5_TEMPLATE_AUDIT_ALWAYS") != nullptr;
    return always;
}

bool TemplateAuditDue(std::chrono::steady_clock::time_point last) {
    const auto interval = TemplateAuditInterval();
    return interval.count() != 0 && std::chrono::steady_clock::now() - last >= interval;
}

std::uint32_t ImportAuditBudget() {
    static const auto budget = [] {
        const char* text = std::getenv("APS5_TEMPLATE_AUDIT_BUDGET");
        return static_cast<std::uint32_t>(text != nullptr ? std::strtoul(text, nullptr, 10) : 8ul);
    }();
    return budget;
}

bool ImportAuditAdmitted() {
    static std::atomic<std::int64_t> slot{-1};
    static std::atomic<std::uint32_t> admitted{0};
    const auto budget = ImportAuditBudget();
    if (budget == 0) return true;
    const auto now = std::chrono::steady_clock::now().time_since_epoch() / std::chrono::milliseconds(4);
    auto seen = slot.load(std::memory_order_relaxed);
    if (seen != now && slot.compare_exchange_strong(seen, now, std::memory_order_relaxed)) admitted.store(0, std::memory_order_relaxed);
    return admitted.fetch_add(1, std::memory_order_relaxed) < budget;
}

}

void LastRecordIndices(std::span<const void* const> logged, std::span<const void* const> objects, std::vector<std::size_t>& indices) {
    indices.assign(objects.size(), logged.size());
    struct WantedStorage {};
    auto& wanted = HostThreadLocal<std::vector<std::pair<const void*, std::size_t>>, WantedStorage>();
    wanted.clear();
    for (std::size_t i = 0; i < objects.size(); ++i) wanted.emplace_back(objects[i], i);
    const auto byObject = [](const std::pair<const void*, std::size_t>& left, const std::pair<const void*, std::size_t>& right) { return std::less<const void*>{}(left.first, right.first); };
    std::sort(wanted.begin(), wanted.end(), byObject);
    std::size_t open = wanted.size();
    for (auto at = logged.size(); at-- > 0 && open != 0;) {
        const auto [first, last] = std::equal_range(wanted.begin(), wanted.end(), std::pair<const void*, std::size_t>{logged[at], 0}, byObject);
        if (first == last || indices[first->second] != logged.size()) continue;
        for (auto it = first; it != last; ++it) indices[it->second] = at;
        open -= static_cast<std::size_t>(last - first);
    }
    if (wanted.capacity() > LookupLogBound + LookupLogBound / 2) {
        wanted.clear();
        wanted.shrink_to_fit();
    }
}

bool ShaderResources::keepsTemplateRecords() const {
    static const bool always = std::getenv("APS5_KEEP_TEMPLATE_RECORDS") != nullptr;
    return always || !(usesBda || (usesFaultBuffer && LoopGuardEnabled()));
}

namespace {

std::array<std::uint64_t, 4> untrackedRangeStamp(std::uint64_t address, std::size_t bytes) {
    const auto epoch = GuestMemory::CollectEpoch();
    if (epoch == 0 || bytes == 0 || (GuestMemory::ForgetSerial() & 1u) != 0) return {};
    const auto* recorder = Recorder::Active();
    return {epoch, GuestMemory::NewestStamp(address, bytes), GuestMemory::ForgetStamp(address, bytes), recorder != nullptr ? recorder->NewestWriteNote(address, bytes) : 0};
}

}

// Turns the lookups' records into this object's per-texture validation records (see
// ValidatedSurface); the last record of an object is the freshest.
void ShaderResources::captureValidation(bool spaceReusable) {
    if (!keepsTemplateRecords() && !(spaceReusable && !RuntimeSwitches::Set(RuntimeSwitch::NoSpaceReuseRecords))) {
        validatedTextures.clear();
        lookupLog().clear();
        return;
    }
    const auto& log = lookupLog();
    struct LoggedStorage {};
    auto& logged = HostThreadLocal<std::vector<const void*>, LoggedStorage>();
    struct ObjectsStorage {};
    auto& objects = HostThreadLocal<std::vector<const void*>, ObjectsStorage>();
    struct IndicesStorage {};
    auto& indices = HostThreadLocal<std::vector<std::size_t>, IndicesStorage>();
    logged.clear();
    for (const auto& entry : log) logged.push_back(entry.object);
    objects.clear();
    for (const auto& texture : textures) objects.push_back(texture.get());
    LastRecordIndices(logged, objects, indices);
    validatedTextures.assign(textures.size(), {});
    for (std::size_t i = 0; i < textures.size(); ++i) {
        if (indices[i] == log.size()) continue;
        const auto& found = log[indices[i]];
        validatedTextures[i] = {found.resource, found.bytes, found.keys, found.generation, 0, found.source, found.depth, true, found.committedGeneration, found.committed};
        auto& surface = validatedTextures[i];
        if (!surface.depth && GuestMemory::CollectWrites(surface.resource.baseAddress, static_cast<std::size_t>(surface.bytes)) == 0) surface.untracked = untrackedRangeStamp(surface.resource.baseAddress, static_cast<std::size_t>(surface.bytes));
    }
    storageUntracked.assign(storageTextures.size(), {});
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        const auto* image = storageTextures[i].get();
        if (image == nullptr) continue;
        const auto address = image->Descriptor().baseAddress;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        if (GuestMemory::CollectWrites(address, bytes) == 0) storageUntracked[i] = untrackedRangeStamp(address, bytes);
    }
    lookupLog().clear();
    const auto trim = [](auto& scratch) {
        if (scratch.capacity() <= LookupLogBound + LookupLogBound / 2) return;
        scratch.clear();
        scratch.shrink_to_fit();
    };
    trim(logged);
    trim(objects);
    trim(indices);
}

ShaderResources::CollectRefusals ShaderResources::CollectRefusalCounts() {
    auto& profile = Revalidations();
    CollectRefusals counts;
    counts.sampled = profile.refusedSampled.load(std::memory_order_relaxed);
    counts.storage = profile.refusedStorage.load(std::memory_order_relaxed);
    counts.uncommitted = profile.refusedUncommitted.load(std::memory_order_relaxed);
    counts.unwatched = profile.refusedUnwatched.load(std::memory_order_relaxed);
    counts.other = profile.refusedOther.load(std::memory_order_relaxed);
    std::lock_guard lock(profile.refusedMutex);
    counts.surfaces = profile.refusedSurfaces.size();
    for (const auto& surface : profile.refusedSurfaces) counts.bytes += surface.second;
    return counts;
}

ShaderResources::UntrackedReproofs ShaderResources::UntrackedReproofCounts() {
    auto& counters = ReproofCounters();
    return {counters.stamped.load(std::memory_order_relaxed), counters.compared.load(std::memory_order_relaxed)};
}

ShaderResources::PendingFails ShaderResources::PendingFailCounts() {
    auto& profile = Revalidations();
    PendingFails counts;
    counts.foreign = profile.pendingForeign.load(std::memory_order_relaxed);
    counts.snapshot = profile.pendingSnapshot.load(std::memory_order_relaxed);
    counts.t1Eligible = profile.pendingT1Eligible.load(std::memory_order_relaxed);
    return counts;
}

std::array<std::uint64_t, static_cast<std::size_t>(ShaderResources::FastFail::Count)> ShaderResources::FullWalkNanoseconds() {
    std::array<std::uint64_t, static_cast<std::size_t>(FastFail::Count)> result{};
    auto& profile = Revalidations();
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = profile.fullNanoseconds[i].load(std::memory_order_relaxed);
    return result;
}

std::size_t ShaderResources::LookupLogCapacity() {
    return lookupLog().capacity();
}

// Proves every texture and storage image still current without repeating its lookup: exactly the
// "unchanged" branches of cachedTexture and StorageTexture::Refresh, evaluated from write stamps,
// the pending-results registry and the DCC keys. Per element: collect first (stamps come only from
// collects, and the generation a record moves to must predate the checks, so a write landing between
// them is stamped newer: the rule of Driver's dispatch cache), then no other image may have results
// pending over the memory (a lookup would flush them into it), a storage image must still be the
// cache's image of its surface, the memory must be unchanged since the object's content matched it,
// and the DCC keys must be what the content was made under whenever the surface has any (a fast
// clear touches only the keys). Anything else, including a failed collect (memory outside the arena,
// uncommitted pages), is left to the full walk.
// The epoch gate: while the pending registry's serial is what this object's last proof saw
// (pendingSerialSeen), the registry is what it was then (every change bumps the serial under the
// registry mutex), so the pending-image checks are skipped; otherwise one scan answers them for
// every surface at once. The stamp checks go through one tracker lock, the cache checks through
// the images' cached flags and one touch of the cache.
namespace {
// APS5_VERIFY_TEMPLATE_PROOFS: every fast proof that took a stamp proves again without stamps, and
// every epoch shortcut that held only through range proofs proves in full; the draw takes that
// answer, and the ones that disagree are counted ([template-proof-verify] on stderr).
bool TemplateProofVerify() {
    static const bool on = std::getenv("APS5_VERIFY_TEMPLATE_PROOFS") != nullptr;
    return on;
}

thread_local bool stampVerifying = false;

void noteTemplateProofCheck(bool stamp, bool failed) {
    static std::atomic<std::uint64_t> stampChecks{0}, stampFailures{0}, rangeChecks{0}, rangeFailures{0};
    if (failed) (stamp ? stampFailures : rangeFailures).fetch_add(1, std::memory_order_relaxed);
    (stamp ? stampChecks : rangeChecks).fetch_add(1, std::memory_order_relaxed);
    const auto count = stampChecks.load(std::memory_order_relaxed) + rangeChecks.load(std::memory_order_relaxed);
    if (failed || count % 262144u == 0u || (count & (count - 1u)) == 0u) std::fprintf(stderr, "[template-proof-verify] stamps: %llu checked, %llu failed; range proofs: %llu checked, %llu failed\n", static_cast<unsigned long long>(stampChecks.load()), static_cast<unsigned long long>(stampFailures.load()), static_cast<unsigned long long>(rangeChecks.load()), static_cast<unsigned long long>(rangeFailures.load()));
}
}

bool ShaderResources::fastRevalidate(std::uint64_t serialBefore, std::span<const PendingOverlap> refreshed, FastFail& reason, std::vector<PendingOverlap>& overlapping, bool& accepted, const std::vector<std::size_t>* only, bool withStorage, bool compares) {
    reason = FastFail::Count;
    overlapping.clear();
    accepted = false;
    if (!EpochRevalidate()) return fastRevalidateEach();
    const auto elementCount = only != nullptr ? only->size() : textures.size();
    const auto elementAt = [only](std::size_t k) { return only != nullptr ? (*only)[k] : k; };
    const auto fail = [&reason, &overlapping](FastFail why) {
        if (why != FastFail::StorageKeys && why != FastFail::Pending) overlapping.clear();
        reason = why;
        countFastFail(why);
        return false;
    };
    if (validatedTextures.size() != textures.size()) return fail(FastFail::NoRecord);
    const bool unchanged = pendingSerialSeen != 0 && pendingSerialSeen == serialBefore;
    const bool keyProofs = KeyFastPath();
    struct QueriesTag {};
    auto& queries = ThreadScratch<std::vector<GuestMemory::UnchangedQuery>, QueriesTag>();
    struct PendingTag {};
    auto& pending = ThreadScratch<std::vector<StorageTexture::PendingQuery>, PendingTag>();
    // The element each pending query stands for, in query order.
    struct OwnersTag {};
    auto& owners = ThreadScratch<std::vector<PendingOverlap>, OwnersTag>();
    struct ImagesTag {};
    auto& images = ThreadScratch<std::vector<const StorageTexture*>, ImagesTag>();
    struct ScannedKeysTag {};
    auto& scannedKeys = ThreadScratch<std::vector<DccKeys>, ScannedKeysTag>();
    struct ReproofsStorage {};
    auto& reproofs = HostThreadLocal<std::vector<std::pair<std::size_t, std::array<std::uint64_t, 4>>>, ReproofsStorage>();
    queries.clear();
    pending.clear();
    owners.clear();
    images.clear();
    reproofs.clear();
    scannedKeys.assign(textures.size(), DccKeys::Uncompressed);
    const auto query = [&](std::uint64_t begin, std::uint64_t end, const StorageTexture* except, const StorageTexture* identity, PendingOverlap owner) {
        pending.push_back({begin, end, except, identity, false});
        owners.push_back(owner);
    };
    // An object another template proved cleanly under the same collect epoch and serials
    // (FastProofStamp) is taken as proved. The T1 rerun (`refreshed`) proves everything itself, and
    // untracked surfaces keep their per-template stamps (untrackedRangeStamp).
    const bool stamps = !stampVerifying && !RuntimeSwitches::Set(RuntimeSwitch::NoTextureProofStamps) && refreshed.empty() && only == nullptr;
    FastProofStamp now{};
    if (stamps) {
        now.epoch = GuestMemory::CollectEpoch();
        now.proof = GuestMemory::ProofSerial();
        now.pending = serialBefore;
        now.release = TextureReleaseSerial().load(std::memory_order_acquire);
        now.unwatched = GuestMemory::UnwatchSerial();
        now.mapping = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        now.depthSurfaces = DepthSurfacesSerial();
        now.depthReleased = DepthSurfacesReleased();
        now.stalePlanes = StaleDepthPlanesSerial();
    }
    const auto stamped = [&](const FastProofStamp& stamp, std::uint64_t address, std::uint64_t bytes, const void* source) {
        return stamps && now.epoch != 0 && stamp.epoch == now.epoch && (RuntimeSwitches::Set(RuntimeSwitch::XpStampsByEpoch) || (stamp.proof == now.proof && stamp.pending == now.pending)) && stamp.release == now.release && stamp.unwatched == now.unwatched && stamp.mapping == now.mapping && stamp.depthSurfaces == now.depthSurfaces && stamp.depthReleased == now.depthReleased && stamp.stalePlanes == now.stalePlanes && stamp.address == address && stamp.bytes == bytes && stamp.source == source;
    };
    // Per texture, then per storage image: 0 proved here, 1 taken from a stamp, 2 untracked.
    struct TakenStorage {};
    auto& taken = HostThreadLocal<std::vector<std::uint8_t>, TakenStorage>();
    taken.assign(textures.size() + storageTextures.size(), 0);
    for (std::size_t k = 0; k < elementCount; ++k) {
        const auto i = elementAt(k);
        auto& surface = validatedTextures[i];
        if (!surface.valid) return fail(FastFail::NoRecord);
        surface.committedProved = false;
        if (surface.depth) {
            if (!textures[i]->HasReuseLookup() && !DepthSurfaceHolds(context, surface.resource, textures[i].get())) return fail(FastFail::Depth);
            continue;
        }
        if (textures[i]->Released()) return fail(FastFail::Replaced);
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        if (const auto& stamp = textures[i]->FastProof(); stamped(stamp, address, bytes, surface.source)) {
            if (surface.source != nullptr && !surface.source->Cached()) return fail(FastFail::Evicted);
            surface.collected = stamp.generation;
            scannedKeys[i] = stamp.keys;
            taken[i] = 1;
            noteStampTaken();
            continue;
        }
        if (!FlushDepthPlanes(address, bytes).empty()) return fail(FastFail::Changed);
        surface.collected = GuestMemory::CollectWrites(address, bytes);
        bool untracked = false;
        if (surface.collected == 0 && surface.generation == 0 && surface.committedGeneration != 0 && surface.source == nullptr && !RuntimeSwitches::Set(RuntimeSwitch::NoExtendedFastProofs)) {
            struct CommittedTag {};
            auto& ranges = ThreadScratch<std::vector<std::pair<std::uint64_t, std::uint64_t>>, CommittedTag>();
            const auto partial = GuestMemory::CollectCommittedWrites(address, bytes, ranges);
            if (partial != 0 && ranges == surface.committed) {
                if (Recorder::SnapshotWriteOverlaps(address, bytes) || !GuestMemory::UnchangedCommittedSince(surface.committed, surface.committedGeneration)) return fail(FastFail::Changed);
                surface.collected = partial;
                surface.committedProved = true;
            }
        }
        if (surface.collected == 0) {
            const bool recorded = surface.untracked[0] != 0;
            auto stamp = recorded ? untrackedRangeStamp(address, bytes) : std::array<std::uint64_t, 4>{};
            if (!recorded || stamp != surface.untracked) {
                const bool reprovable = surface.source == nullptr && UntrackedReproofEnabled() && !DepthSurfaceAt(address);
                if (reprovable && !recorded) stamp = untrackedRangeStamp(address, bytes);
                const auto proof = reprovable ? untrackedSnapshotStamped(*textures[i], address, bytes, stamp) : SnapshotProof::Refused;
                if (proof == SnapshotProof::Stamped) {
                    surface.untracked = stamp;
                } else if (proof == SnapshotProof::Unproved && compares && recorded && surface.untracked[2] == stamp[2]) {
                    reproofs.emplace_back(i, stamp);
                } else {
                    countCollectRefusal(false, address, bytes);
                    return fail(FastFail::Collect);
                }
            }
            untracked = true;
            taken[i] = 2;
        }
        const auto* source = surface.source;
        if (!keyProofs) {
            // Without proofs the keys are compared with the record's (a cleared view's identity
            // below is the record's too): the scan repeats on every call, as before.
            if (surface.resource.dccAddress != 0 && TextureClearKeys(surface.resource, surface.bytes) != surface.keys) return fail(FastFail::Keys);
            scannedKeys[i] = surface.keys;
            if (!unchanged) query(address, address + bytes, source, source != nullptr && surface.keys != DccKeys::Uncompressed ? source : nullptr, {i, false, source != nullptr && surface.keys == DccKeys::Uncompressed});
        } else if (source != nullptr) {
            // The view follows its image, so it stays valid while the image would Refresh as
            // unchanged: the image's own keys must be what its content was uploaded under (its
            // proof answers without a scan while they are unstamped), and its memory unchanged
            // since its generation (the query below). The surface's own keys come from the same
            // proof when they are the image's, else from the texture's.
            const auto& own = source->Descriptor();
            const auto sourceKeys = source->ProvedKeys();
            if (sourceKeys != source->UploadedKeys()) return fail(FastFail::Keys);
            const auto keys = surface.resource.dccAddress == 0 ? DccKeys::Uncompressed : SameKeySurface(source, surface.resource, surface.bytes) ? sourceKeys : ProvedClearKeys(surface.resource, surface.bytes, textures[i]->KeyProof());
            if (surface.resource.dccAddress != 0 && surface.resource.dccAddress != own.dccAddress && (keys != DccKeys::Uncompressed || !StorageImageServesKeys(*source, surface.resource.dccAddress))) return fail(FastFail::Keys);
            scannedKeys[i] = keys;
            // A view of a fast-cleared surface stays one only while its image still has results
            // pending over it (cachedTexture's hit rule), unless the image's own descriptor names
            // the surface's metadata (the lookup views the cleared image then). The identity is
            // asked whatever the registry's serial did: the keys may have moved since the last
            // proof while the registry did not.
            if (keys != DccKeys::Uncompressed && !(own.dccAddress == surface.resource.dccAddress && ClearedViewEnabled() && StorageClearAvailable(context, surface.resource.format, keys))) query(address, address + bytes, source, source, {i, false, false});
            else if (!unchanged) query(address, address + bytes, source, nullptr, {i, false, keys == DccKeys::Uncompressed});
            if (compares && (surface.resource.dccAddress != 0 || own.dccAddress != 0)) countKeysProven(false);
        } else {
            // A snapshot holds its clear texels or the guest bytes: the keys must be what it was made under.
            const auto keys = surface.resource.dccAddress != 0 ? ProvedClearKeys(surface.resource, surface.bytes, textures[i]->KeyProof()) : DccKeys::Uncompressed;
            if (keys != surface.keys) return fail(FastFail::Keys);
            scannedKeys[i] = keys;
            if (!unchanged) query(address, address + bytes, nullptr, nullptr, {i, false, false});
            if (compares && surface.resource.dccAddress != 0) countKeysProven(false);
        }
        if (source != nullptr) {
            if (!source->Cached()) return fail(FastFail::Evicted);
            if (!untracked) queries.push_back({address, bytes, source->GenerationFor(address, bytes)});
            images.push_back(source);
        } else if (!untracked && !surface.committedProved) {
            queries.push_back({address, bytes, surface.generation});
        }
    }
    bool keysMoved = false;
    for (std::size_t i = 0; withStorage && i < storageTextures.size(); ++i) {
        if (storageTextures[i] != nullptr && (i >= storageKeys.size() || !StorageImageServesKeys(*storageTextures[i], storageKeys[i]))) return fail(FastFail::StorageKeys);
        if (i != 0 && storageTextures[i] == storageTextures[i - 1]) continue;
        const auto* image = storageTextures[i].get();
        if (image == nullptr) return fail(FastFail::NoRecord);
        const auto& own = image->Descriptor();
        const auto address = own.baseAddress;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        if (stamped(image->FastProof(), address, bytes, image)) {
            if (!image->Cached()) return fail(FastFail::Evicted);
            taken[textures.size() + i] = 1;
            noteStampTaken();
            continue;
        }
        if (DepthSurfaceOverlaps(context.device, address, bytes)) {
            if (!indexProof) depthGuardRefusals.fetch_add(1, std::memory_order_relaxed);
            return fail(FastFail::Depth);
        }
        bool untracked = false;
        if (GuestMemory::CollectWrites(address, bytes) == 0) {
            if (i >= storageUntracked.size() || storageUntracked[i][0] == 0 || untrackedRangeStamp(address, bytes) != storageUntracked[i]) {
                countCollectRefusal(true, address, bytes);
                return fail(FastFail::Collect);
            }
            untracked = true;
            taken[textures.size() + i] = 2;
        }
        if (own.dccAddress != 0) {
            // Refresh's unchanged branch (its key compare against the keys the content was
            // uploaded under, through the image's proof; the memory query below), minus its
            // byte-compare fallback; the image's generation is left where it is.
            if (!keyProofs) return fail(FastFail::StorageKeys);
            if (image->ProvedKeys() != image->UploadedKeys()) {
                if (!OwnImageRefreshEnabled()) return fail(FastFail::StorageKeys);
                keysMoved = true;
                overlapping.push_back({i, true, false});
                continue;
            }
            if (compares) countKeysProven(true);
        }
        if (!unchanged) query(address, address + bytes, image, nullptr, {i, true, false});
        if (!image->Cached()) return fail(FastFail::Evicted);
        if (!untracked) queries.push_back({address, bytes, image->GenerationFor(address, bytes)});
        images.push_back(image);
    }
    if (keysMoved) return fail(FastFail::StorageKeys);
    if (!pending.empty()) {
        if (!StorageTexture::ScanPending(pending)) return fail(FastFail::ClearedView);
        // A foreign image over a view under uncompressed keys is no failure while FindPending
        // names the view's own source: the lookup views the source (cachedTexture's hit rule)
        // and touches nothing else. A surface whose own object T1 refreshed in this Revalidate
        // (`refreshed`) is no failure either: a storage element's lookup is that Refresh and
        // returns the image whatever stays registered over it; a view's lookup hits on the source
        // when FindPending names it, and views the cached image of its key (the source, proved by
        // refreshOwnObjects) when FindPending names nothing. Every other overlap is the full
        // walk's, or T1's (Revalidate).
        const auto refreshedOwner = [&](const PendingOverlap& owner) {
            return std::find_if(refreshed.begin(), refreshed.end(), [&](const PendingOverlap& entry) { return entry.element == owner.element && entry.storage == owner.storage; });
        };
        for (std::size_t k = 0; k < pending.size(); ++k) {
            if (!pending[k].overlaps) continue;
            if (OwnImageRefreshEnabled() && owners[k].viewUncompressed && pending[k].found == pending[k].except) {
                if (compares && BuildProfiled()) Revalidations().ownSourceViews.fetch_add(1, std::memory_order_relaxed);
                accepted = true;
                continue;
            }
            const auto lookupAccepts = [&](const PendingOverlap& entry) {
                const auto& surface = validatedTextures[entry.element];
                if (surface.resource.dccAddress != 0) return false;
                const auto viewed = pendingLookupSource(context, surface.resource, surface.bytes, entry.depthCompare).source;
                return viewed.get() == pending[k].except || (viewed == nullptr && entry.sourceEligible && entry.sourceRefreshed);
            };
            if (const auto entry = refreshedOwner(owners[k]); entry != refreshed.end() && (entry->storage || pending[k].found == pending[k].except || (pending[k].found == nullptr && entry->sourceEligible) || lookupAccepts(*entry))) {
                if (compares && BuildProfiled()) Revalidations().refreshedOverlaps.fetch_add(1, std::memory_order_relaxed);
                accepted = true;
                continue;
            }
            overlapping.push_back(owners[k]);
        }
        if (!overlapping.empty()) {
            countPendingFail(pending);
            return fail(FastFail::Pending);
        }
    } else if (compares && unchanged && BuildProfiled()) {
        Revalidations().serialSkips.fetch_add(1, std::memory_order_relaxed);
    }
    if (!GuestMemory::UnchangedSinceAll(queries)) return fail(FastFail::Changed);
    if (!StorageImagesCached(context, images)) return fail(FastFail::Evicted);
    if (!reproofs.empty()) {
        for (const auto& [i, stamp] : reproofs) {
            auto& surface = validatedTextures[i];
            const auto address = surface.resource.baseAddress;
            const auto bytes = static_cast<std::size_t>(surface.bytes);
            if (!untrackedSnapshotCompared(*textures[i], address, bytes, stamp)) {
                countCollectRefusal(false, address, bytes);
                return fail(FastFail::Collect);
            }
            surface.untracked = stamp;
        }
        return fastRevalidate(serialBefore, refreshed, reason, overlapping, accepted, only, withStorage, false);
    }
    for (const auto* image : images) image->NoteProved();
    for (std::size_t k = 0; k < elementCount; ++k) {
        const auto i = elementAt(k);
        auto& surface = validatedTextures[i];
        if (surface.source == nullptr) {
            if (surface.committedProved) surface.committedGeneration = surface.collected;
            else surface.generation = surface.collected;
        }
        surface.keys = scannedKeys[i];
    }
    if (stamps && now.epoch != 0 && !accepted) {
        for (std::size_t i = 0; i < textures.size(); ++i) {
            const auto& surface = validatedTextures[i];
            if (surface.depth || taken[i] != 0 || surface.collected == 0 || surface.committedProved) continue;
            auto stamp = now;
            stamp.address = surface.resource.baseAddress;
            stamp.bytes = surface.bytes;
            stamp.source = surface.source;
            stamp.generation = surface.collected;
            stamp.keys = scannedKeys[i];
            textures[i]->FastProof() = stamp;
        }
        for (std::size_t i = 0; i < storageTextures.size(); ++i) {
            const auto* image = storageTextures[i].get();
            if (image == nullptr || taken[textures.size() + i] != 0 || (i != 0 && storageTextures[i] == storageTextures[i - 1])) continue;
            auto stamp = now;
            stamp.address = image->Descriptor().baseAddress;
            stamp.bytes = image->GuestBytes();
            stamp.source = image;
            image->FastProof() = stamp;
        }
    }
    if (stamps && TemplateProofVerify() && std::find(taken.begin(), taken.end(), std::uint8_t{1}) != taken.end()) {
        struct Verifying {
            Verifying() { stampVerifying = true; }
            ~Verifying() { stampVerifying = false; }
        } verifying;
        const bool ok = fastRevalidate(serialBefore, refreshed, reason, overlapping, accepted, only, withStorage, compares);
        noteTemplateProofCheck(true, !ok);
        return ok;
    }
    return true;
}

bool ShaderResources::fastRevalidateEach() {
    if (validatedTextures.size() != textures.size()) return false;
    for (std::size_t i = 0; i < textures.size(); ++i) {
        auto& surface = validatedTextures[i];
        if (!surface.valid) return false;
        if (surface.depth) {
            if (!textures[i]->HasReuseLookup() && !DepthSurfaceHolds(context, surface.resource, textures[i].get())) return false;
            continue;
        }
        if (textures[i]->Released()) return false;
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        if (!FlushDepthPlanes(address, bytes).empty()) return false;
        surface.collected = GuestMemory::CollectWrites(address, bytes);
        if (surface.collected == 0) return false;
        if (PendingStorageOverlaps(address, bytes, surface.source)) return false;
        if (surface.source != nullptr && surface.resource.dccAddress != 0 && surface.resource.dccAddress != surface.source->Descriptor().dccAddress) return false;
        if (surface.source != nullptr) {
            // The view follows the image: it needs the image current with guest memory, as the
            // lookup's Refresh would make it.
            if (!StorageImageCached(context, surface.source) || !GuestMemory::UnchangedSince(address, bytes, surface.source->GenerationFor(address, bytes))) return false;
        } else if (!GuestMemory::UnchangedSince(address, bytes, surface.generation)) {
            return false;
        }
        if (surface.resource.dccAddress != 0 && TextureClearKeys(surface.resource, surface.bytes) != surface.keys) return false;
        // A view made under fast-clear keys stays one only while its image still has results pending
        // over the surface (the lookup prefers them to the clear); once they are flushed the clear
        // the image cannot see wins and the lookup makes a snapshot (cachedTexture's hit rule).
        if (surface.source != nullptr && surface.keys != DccKeys::Uncompressed && StorageTexture::FindPending(address, surface.bytes).get() != surface.source) return false;
    }
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        if (storageTextures[i] != nullptr && (i >= storageKeys.size() || (storageKeys[i] != 0 && storageKeys[i] != storageTextures[i]->Descriptor().dccAddress))) return false;
        // Consecutive elements of one mip chain share the image (see addStorageImageBinding).
        if (i != 0 && storageTextures[i] == storageTextures[i - 1]) continue;
        const auto* image = storageTextures[i].get();
        if (image == nullptr) return false;
        // StorageTexture::Refresh checks the image's own surface, not the element's descriptor (one
        // image serves every descriptor of its memory, with or without a DCC address), and compares
        // its keys with the keys it was uploaded under, which only the image knows (they move to
        // Uncompressed on every write-back; a fast clear after it puts the key bytes back to what a
        // record saw while the image holds last frame's results), so a surface with keys is left to
        // the full walk. Without a DCC address both sides of that compare are Uncompressed whatever
        // happened, and the memory checks below are Refresh's.
        const auto& own = image->Descriptor();
        if (own.dccAddress != 0) return false;
        const auto address = own.baseAddress;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        if (DepthSurfaceOverlaps(context.device, address, bytes)) {
            depthGuardRefusals.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (GuestMemory::CollectWrites(address, bytes) == 0) return false;
        if (PendingStorageOverlaps(address, bytes, image)) return false;
        if (!StorageImageCached(context, image)) return false;
        if (!GuestMemory::UnchangedSince(address, bytes, image->GenerationFor(address, bytes))) return false;
    }
    // Every check passed: snapshot textures are current at the collects issued before them. Storage
    // images keep their own generation: nothing was stamped above it, so the blocks a later write
    // stamps are the same set either way.
    for (auto& surface : validatedTextures) {
        if (surface.source == nullptr) surface.generation = surface.collected;
    }
    return true;
}

ShaderResources::OwnRefreshFallback ShaderResources::refreshOwnObjects(std::span<const CompiledShader> shaders, std::span<PendingOverlap> overlapping) {
    const auto listed = [&](std::size_t element, bool storage) {
        return std::find_if(overlapping.begin(), overlapping.end(), [&](const PendingOverlap& overlap) { return overlap.element == element && overlap.storage == storage; });
    };
    // A sampled view's lookup (cachedTexture): the pending image FindPending serves the surface
    // by, if it fits the view, else the cached storage image of the surface's key, refreshed; the
    // view stays the lookup's answer only when that image is its own source. `sourceEligible` is
    // recorded whichever branch answers: a later refresh of this call may flush the source, and
    // the re-run then judges the surface by the other branch.
    const auto refreshView = [&](std::size_t i, PendingOverlap& overlap, bool depthCompare) {
        const auto& surface = validatedTextures[i];
        const auto& source = textures[i]->SharedStorageSource();
        if (source == nullptr || source.get() != surface.source) return OwnRefreshFallback::Snapshot;
        if (!source->Cached()) return OwnRefreshFallback::Uncached;
        if (!StorageImageServesKeys(*source, surface.resource.dccAddress)) return OwnRefreshFallback::Keys;
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        const bool imported = SampledFromStorageEligible(context, surface.resource, surface.bytes);
        overlap.sourceEligible = imported && SurfaceKey(context, source->Descriptor()) == SurfaceKey(context, surface.resource);
        overlap.depthCompare = depthCompare;
        overlap.sourceRefreshed = false;
        const auto found = pendingLookupSource(context, surface.resource, surface.bytes, depthCompare).source;
        if (found != nullptr) {
            if (found != source) return OwnRefreshFallback::ForeignView;
            if (!GuestMemory::UnchangedSince(address, bytes, source->GenerationFor(address, bytes))) {
                countOwnRefresh(false);
                refreshForView(*source, surface.resource);
            }
            return OwnRefreshFallback::Count;
        }
        if (!imported) return OwnRefreshFallback::NotImported;
        if (!overlap.sourceEligible) return OwnRefreshFallback::SurfaceKey;
        countOwnRefresh(false);
        // The cache's image of the key is this source while it is cached (an entry is replaced
        // only after its image left the cache), so the lookup would refresh and return it.
        source->Refresh();
        overlap.sourceRefreshed = true;
        return OwnRefreshFallback::Count;
    };
    // A storage element's lookup (cachedStorageTexture): the cached image of its surface key,
    // refreshed; the key is the element's words', which mapped to this image at the build.
    const auto refreshStorage = [&](std::size_t i) {
        auto& image = storageTextures[i];
        if (image == nullptr || !image->Cached()) return OwnRefreshFallback::Uncached;
        if (i >= storageKeys.size() || !StorageImageServesKeys(*image, storageKeys[i])) return OwnRefreshFallback::Keys;
        countOwnRefresh(true);
        image->Refresh();
        return OwnRefreshFallback::Count;
    };
    std::size_t textureIndex = 0;
    std::size_t storageIndex = 0;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) return OwnRefreshFallback::Uncached;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
            if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto words = std::span<const std::uint32_t>(DrawGuestWords(shader, binding)).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                    if (IsNullSampledTextureResource(words)) continue;
                    if (textureIndex >= textures.size()) return OwnRefreshFallback::Uncached;
                    const auto index = textureIndex++;
                    const auto overlap = listed(index, false);
                    if (overlap == overlapping.end()) continue;
                    if (const auto fallback = refreshView(index, *overlap, !binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element)); fallback != OwnRefreshFallback::Count) return fallback;
                }
            } else if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
                for (std::uint32_t element = 0; element < binding.count; ++element) {
                    const auto words = std::span<const std::uint32_t>(DrawGuestWords(shader, binding)).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                    if (IsNullTextureResource(words)) continue;
                    if (storageIndex >= storageTextures.size()) return OwnRefreshFallback::Uncached;
                    const auto index = storageIndex++;
                    if (listed(index, true) == overlapping.end()) continue;
                    if (const auto fallback = refreshStorage(index); fallback != OwnRefreshFallback::Count) return fallback;
                }
            }
        }
    }
    for (const auto& table : tableStages) {
        std::span<const std::uint32_t> imageWords;
        const auto* images = table.elementTextures.empty() ? nullptr : tableImages(shaders, table.stage, imageWords);
        for (std::size_t element = 0; element < table.elementTextures.size(); ++element) {
            const auto index = table.elementTextures[element];
            if (index == std::numeric_limits<std::size_t>::max()) continue;
            if (!table.representatives.empty() && table.representatives[element] != NoRepresentative) continue;
            if (index != textureIndex++ || index >= textures.size()) return OwnRefreshFallback::Uncached;
            const auto overlap = listed(index, false);
            if (overlap == overlapping.end()) continue;
            if (images == nullptr || images->count != table.elementTextures.size()) return OwnRefreshFallback::Uncached;
            if (const auto fallback = refreshView(index, *overlap, images->imageDepthCompare.at(element)); fallback != OwnRefreshFallback::Count) return fallback;
        }
    }
    return OwnRefreshFallback::Count;
}

bool AnyPendingCpuWrites(const Context& context) {
    if (Recorder::InCompletion() || Recorder::Active() == nullptr) return false;
    return (context.copiedWriters != nullptr && !context.copiedWriters->empty()) || !DrawCopiedWriters()->empty() || Recorder::PendingCompletionLabels() != 0;
}

bool CompletePendingCpuWrites(const Context& context, std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || Recorder::InCompletion()) return false;
    auto* recorder = Recorder::Active();
    if (recorder == nullptr) return false;
    const auto drawWriters = DrawCopiedWriters();
    const bool labels = Recorder::PendingCompletionLabels() != 0;
    if ((context.copiedWriters == nullptr || context.copiedWriters->empty()) && drawWriters->empty() && !labels) return false;
    if (!recorder->PendingWriteOverlaps(address, bytes)) return false;
    const auto overlaps = [&](const auto& writers) {
        return std::any_of(writers.begin(), writers.end(), [&](const auto& writer) { return writer->CopiedWritesOverlap(address, bytes); });
    };
    if ((context.copiedWriters != nullptr && overlaps(*context.copiedWriters)) || overlaps(*drawWriters) || (labels && recorder->CompletionLabelIn(address, bytes))) {
        Recorder::CountSync(4);
        recorder->SyncThrough(address, bytes);
        return true;
    }
    return false;
}

const ShaderRecompiler::DescriptorBinding* ShaderResources::tableImages(std::span<const CompiledShader> shaders, ShaderRecompiler::ShaderStage stage, std::span<const std::uint32_t>& words) {
    for (const auto& shader : shaders) {
        if (shader.stage != stage || shader.program == nullptr) continue;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::ImageTable) continue;
            words = DrawGuestWords(shader, binding);
            return &binding;
        }
    }
    return nullptr;
}

bool ShaderResources::EpochProofs() {
    return DrawPipelineEnabled();
}

namespace {
std::atomic<std::uint64_t> stampsTaken{0};
}

void ShaderResources::noteStampTaken() {
    stampsTaken.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t ShaderResources::FastProofStampsTaken() {
    return stampsTaken.load(std::memory_order_relaxed);
}

namespace {
std::array<std::atomic<std::uint64_t>, 6> epochMisses{};

void noteEpochMiss(std::uint64_t provedEpoch, std::uint64_t provedProof, std::uint64_t provedPending, std::uint64_t epoch, std::uint64_t proof, std::uint64_t pending) {
    static const bool stats = std::getenv("APS5_STAGE_STATS") != nullptr;
    if (!stats) return;
    std::size_t reason = 0;
    if (provedEpoch == 0) reason = 0;
    else if (epoch == 0) reason = 1;
    else if (provedEpoch != epoch) reason = 2;
    else if (provedProof != proof && provedPending != pending) reason = 3;
    else if (provedProof != proof) reason = 4;
    else reason = 5;
    epochMisses[reason].fetch_add(1, std::memory_order_relaxed);
}
}

std::string ShaderResources::TakeEpochMissReport() {
    static constexpr std::array<const char*, 6> names{"unproved", "no epoch", "epoch moved", "proof and pending moved", "proof serial moved", "pending serial moved"};
    std::string text;
    char item[64];
    for (std::size_t index = 0; index < names.size(); ++index) {
        std::snprintf(item, sizeof(item), "%s %s %llu", index == 0 ? "" : ",", names[index], static_cast<unsigned long long>(epochMisses[index].exchange(0, std::memory_order_relaxed)));
        text += item;
    }
    return text;
}

namespace {
void noteEpochProof(bool reused, bool verified, bool failed) {
    static std::atomic<std::uint64_t> reuses{0}, proofs{0}, checks{0}, failures{0};
    if (verified) checks.fetch_add(1, std::memory_order_relaxed);
    if (failed) failures.fetch_add(1, std::memory_order_relaxed);
    const auto count = (reused ? reuses : proofs).fetch_add(1, std::memory_order_relaxed) + 1u;
    static const bool report = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_VERIFY_DRAW_PIPELINE") != nullptr;
    if (!report) return;
    if (count % 262144u == 0u || (count & (count - 1u)) == 0u) std::fprintf(stderr, "[epoch-proofs] resource proofs: %llu reused the epoch's, %llu proved; verify: %llu checked, %llu failed; storage refreshes reused %llu\n", static_cast<unsigned long long>(reuses.load()), static_cast<unsigned long long>(proofs.load()), static_cast<unsigned long long>(checks.load()), static_cast<unsigned long long>(failures.load()), static_cast<unsigned long long>(StorageTexture::epochRefreshes.load()));
}
}

namespace {

constexpr std::size_t TableSpanElement = std::numeric_limits<std::size_t>::max();

bool IndexBroken(const char* part) {
    static const std::string broken = [] {
        const char* text = std::getenv("APS5_TABLE_INDEX_BREAK");
        return std::string(text != nullptr ? text : "");
    }();
    return !broken.empty() && broken.find(part) != std::string::npos;
}

}

void ShaderResources::subscribeIndex(std::span<const CompiledShader> shaders) {
    writeSubscription.reset();
    subscribedGeneration = 0;
    subscribedPendingSerial = 0;
    subscribedBlocks.clear();
    perElementTextures.clear();
    comparedTextures.clear();
    comparedTableSpans.clear();
    depthCopyTextures.clear();
    indexElements.assign(textures.size(), {});
    for (std::size_t i = 0; i < textures.size(); ++i) {
        if (textures[i] != nullptr && (textures[i]->IsDepthCopy() || textures[i]->HasReuseLookup())) depthCopyTextures.push_back(i);
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(textures.size() + 4);
    for (std::size_t i = 0; i < textures.size(); ++i) {
        if (i >= validatedTextures.size()) {
            perElementTextures.push_back(i);
            continue;
        }
        const auto& surface = validatedTextures[i];
        const auto begin = surface.resource.baseAddress;
        const auto bytes = surface.bytes;
        if (!surface.valid || surface.depth || bytes == 0) {
            perElementTextures.push_back(i);
            continue;
        }
        if (!GuestMemory::Watched(begin, static_cast<std::size_t>(bytes))) {
            comparedTextures.push_back(i);
            continue;
        }
        if (surface.source != nullptr || surface.resource.dccAddress != 0) {
            perElementTextures.push_back(i);
            continue;
        }
        ranges.emplace_back(begin, begin + bytes);
        for (auto block = GuestMemory::WriteBlockOf(begin); block <= GuestMemory::WriteBlockOf(begin + bytes - 1); ++block) subscribedBlocks.emplace_back(block, i);
    }
    for (const auto& table : tableStages) {
        std::span<const std::uint32_t> tableWords;
        if (tableImages(shaders, table.stage, tableWords) == nullptr || table.map.size() < ShaderRecompiler::ImageTableAbi::HeaderWords) continue;
        const auto tables = table.map[ShaderRecompiler::ImageTableAbi::HeaderTableCount];
        for (std::uint32_t index = 0; index < tables; ++index) {
            const auto header = ShaderRecompiler::ImageTableAbi::TableHeader(index);
            if (static_cast<std::size_t>(header) + ShaderRecompiler::ImageTableAbi::TableWords > table.map.size()) break;
            const auto base = static_cast<std::uint64_t>(table.map[header + ShaderRecompiler::ImageTableAbi::TableBaseLow]) | (static_cast<std::uint64_t>(table.map[header + ShaderRecompiler::ImageTableAbi::TableBaseHigh]) << 32);
            const auto size = static_cast<std::uint64_t>(table.map[header + ShaderRecompiler::ImageTableAbi::TableSizeLow]) | (static_cast<std::uint64_t>(table.map[header + ShaderRecompiler::ImageTableAbi::TableSizeHigh]) << 32);
            if (base == 0 || size == 0 || size > (std::uint64_t{64} << 20) || size > std::numeric_limits<std::uint64_t>::max() - base) continue;
            if (GuestMemory::Watched(base, static_cast<std::size_t>(size))) {
                ranges.emplace_back(base, base + size);
                for (auto block = GuestMemory::WriteBlockOf(base); block <= GuestMemory::WriteBlockOf(base + size - 1); ++block) subscribedBlocks.emplace_back(block, TableSpanElement);
            } else if (GuestMemory::Accessible(reinterpret_cast<const void*>(base), static_cast<std::size_t>(size))) {
                std::vector<std::byte> copy(static_cast<std::size_t>(size));
                std::memcpy(copy.data(), reinterpret_cast<const void*>(base), copy.size());
                comparedTableSpans.emplace_back(base, std::move(copy));
            }
        }
    }
    std::sort(subscribedBlocks.begin(), subscribedBlocks.end());
    std::size_t textureIndex = 0;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) break;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.kind != ShaderRecompiler::DescriptorKind::SampledImage) continue;
            const auto elementWords = binding.count != 0 ? binding.guestDescriptor.size() / binding.count : 0;
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
                if (IsNullTextureResource(words)) continue;
                if (textureIndex < indexElements.size() && words.size() <= 8) {
                    auto& entry = indexElements[textureIndex];
                    std::copy(words.begin(), words.end(), entry.words.begin());
                    entry.count = static_cast<std::uint32_t>(words.size());
                    entry.shape = binding.imageShape;
                    entry.depthCompare = !binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element);
                    entry.known = true;
                }
                ++textureIndex;
            }
        }
    }
    for (const auto& table : tableStages) {
        std::span<const std::uint32_t> tableWords;
        const auto* images = tableImages(shaders, table.stage, tableWords);
        if (images == nullptr) continue;
        for (std::size_t element = 0; element < table.elementTextures.size(); ++element) {
            const auto index = table.elementTextures[element];
            if (index >= indexElements.size() || (element + 1) * 8u > tableWords.size()) continue;
            auto& entry = indexElements[index];
            const auto words = tableWords.subspan(element * 8u, 8u);
            std::copy(words.begin(), words.end(), entry.words.begin());
            entry.count = 8;
            entry.shape = element < images->imageShapes.size() ? std::optional(images->imageShapes[element]) : std::nullopt;
            entry.depthCompare = element < images->imageDepthCompare.size() && images->imageDepthCompare[element];
            entry.known = true;
        }
    }
    writeSubscription = GuestMemory::SubscribeWrites(ranges);
}

void ShaderResources::subscribeBuilt(std::span<const CompiledShader> shaders) {
    if (!reusable || !EpochRevalidate() || std::getenv("APS5_NO_FAST_REVALIDATE") != nullptr) return;
    subscribeIndex(shaders);
    if (writeSubscription == nullptr) return;
    const auto generation = GuestMemory::SubscriptionGeneration(*writeSubscription);
    std::vector<GuestMemory::UnchangedQuery> queries;
    queries.reserve(textures.size());
    std::size_t last = std::numeric_limits<std::size_t>::max();
    for (const auto& [block, element] : subscribedBlocks) {
        if (element == TableSpanElement || element == last) continue;
        last = element;
        const auto& surface = validatedTextures[element];
        if (surface.generation == 0) return;
        queries.push_back({surface.resource.baseAddress, static_cast<std::size_t>(surface.bytes), surface.generation});
    }
    if (!queries.empty() && !GuestMemory::UnchangedSinceAll(queries)) return;
    subscribedGeneration = generation;
    subscribedPendingSerial = 0;
}

void ShaderResources::noteIndexProof(std::uint64_t generation, std::uint64_t serialBefore) {
    if (writeSubscription == nullptr) return;
    subscribedGeneration = generation;
    subscribedPendingSerial = StorageTexture::PendingSerial() == serialBefore ? serialBefore : 0;
    GuestMemory::TrimSubscriptionBumps(*writeSubscription, generation);
}

bool ShaderResources::reresolveElements(const std::vector<std::size_t>& elements, std::vector<std::size_t>& replacedElements) {
    auto* recorder = Recorder::Active();
    if (elements.empty()) return true;
    if (recorder == nullptr) return false;
    if (indexImagePositions.empty()) {
        std::size_t position = 0;
        for (const auto& binding : bindings) {
            if (binding.flags != 0) {
                for (const auto index : binding.imageAllocations) {
                    if (index == std::numeric_limits<std::size_t>::max()) continue;
                    if (binding.layout.descriptorType != VK_DESCRIPTOR_TYPE_SAMPLER) indexImagePositions.push_back({index, position, true});
                    ++position;
                }
                continue;
            }
            if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) continue;
            for (const auto index : binding.imageAllocations) {
                if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE && index != std::numeric_limits<std::size_t>::max()) indexImagePositions.push_back({index, position, false});
                ++position;
            }
        }
        if (position != descriptorImages.size()) {
            indexImagePositions.clear();
            return false;
        }
        std::sort(indexImagePositions.begin(), indexImagePositions.end(), [](const ImagePosition& left, const ImagePosition& right) { return left.texture < right.texture; });
    }
    lookupLog().clear();
    BuildLookups lookups;
    std::vector<std::pair<std::size_t, std::shared_ptr<Texture>>> replaced;
    for (const auto i : elements) {
        if (i >= indexElements.size() || i >= textures.size() || !indexElements[i].known) return false;
        const auto& element = indexElements[i];
        std::array<std::uint32_t, 8> narrowed{};
        const auto words = firstLayerWords(std::span<const std::uint32_t>(element.words.data(), element.count), element.shape, narrowed, false);
        const auto resource = DecodeTextureResource(words);
        const VkImageViewType multisampleViewType = element.shape == ShaderRecompiler::DescriptorImageShape::Image2DMultisampleArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        auto texture = cachedTexture(context, words, resource, ViewComponents(resource), 0, element.depthCompare, multisampleViewType);
        if (texture == nullptr) return false;
        if (texture != textures[i]) replaced.emplace_back(i, std::move(texture));
    }
    if (!replaced.empty() && (!drawBuild || usesBda)) {
        lookupLog().clear();
        return false;
    }
    for (auto& [i, texture] : replaced) {
        replacedElements.push_back(i);
        recorder->Keep(textures[i]);
        textures[i] = std::move(texture);
        depthCopyTextures.erase(std::remove(depthCopyTextures.begin(), depthCopyTextures.end(), i), depthCopyTextures.end());
        if (textures[i]->IsDepthCopy() || textures[i]->HasReuseLookup()) depthCopyTextures.insert(std::lower_bound(depthCopyTextures.begin(), depthCopyTextures.end(), i), i);
        const auto first = std::lower_bound(indexImagePositions.begin(), indexImagePositions.end(), i, [](const ImagePosition& entry, std::size_t texture) { return entry.texture < texture; });
        for (auto entry = first; entry != indexImagePositions.end() && entry->texture == i; ++entry) {
            const auto& current = textures[i];
            const auto view = entry->table ? (textureFirstLayer[i] ? current->FirstLayerView() : current->View()) : textureSrgb[i] ? current->SrgbView(textureFirstLayer[i], textureBaseLevel[i]) : textureBaseLevel[i] ? current->BaseLevelView() : textureFirstLayer[i] ? current->FirstLayerView() : current->View();
            descriptorImages[entry->position] = {VK_NULL_HANDLE, view, current->Layout()};
        }
        rebound = true;
        countIndex(IndexRebound);
    }
    const auto& log = lookupLog();
    for (const auto i : elements) {
        const auto found = std::find_if(log.rbegin(), log.rend(), [&](const LookupRecord& record) { return record.object == textures[i].get(); });
        if (found == log.rend()) {
            validatedTextures[i].valid = false;
            lookupLog().clear();
            return false;
        }
        validatedTextures[i] = {found->resource, found->bytes, found->keys, found->generation, 0, found->source, found->depth, true, found->committedGeneration, found->committed};
    }
    lookupLog().clear();
    return true;
}

ShaderResources::IndexOutcome ShaderResources::indexRevalidate(std::span<const CompiledShader> shaders, std::uint64_t serialBefore) {
    static_cast<void>(shaders);
    if (writeSubscription == nullptr || subscribedGeneration == 0) return IndexOutcome::Fallback;
    const bool profiled = BuildProfiled();
    const auto start = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto outcome = [&](IndexOutcome result) {
        countIndex(result == IndexOutcome::Accepted ? IndexAccepted : result == IndexOutcome::Partial ? IndexPartial : IndexFallback);
        if (!profiled) return result;
        Revalidations().indexNanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
        return result;
    };
    const auto fallback = [&] { return outcome(IndexOutcome::Fallback); };
    auto& subscription = *writeSubscription;
    for (const auto& [begin, end] : GuestMemory::SubscriptionSpans(subscription)) {
        if (!FlushDepthPlanes(begin, static_cast<std::size_t>(end - begin)).empty()) return fallback();
    }
    if (!GuestMemory::SweepSubscription(subscription)) return fallback();
    if ((subscribedPendingSerial == 0 || subscribedPendingSerial != serialBefore) && !IndexBroken("pending")) {
        thread_local std::vector<StorageTexture::PendingQuery>* regionsSlot = nullptr;
        auto& regions = ShaderRecompiler::ThreadOwned(regionsSlot);
        regions.clear();
        for (const auto& [begin, end] : GuestMemory::SubscriptionSpans(subscription)) regions.push_back({begin, end, nullptr, nullptr, false});
        if (!regions.empty()) {
            StorageTexture::ScanPending(regions);
            if (std::any_of(regions.begin(), regions.end(), [](const StorageTexture::PendingQuery& region) { return region.overlaps; })) return fallback();
        }
    }
    for (const auto& [address, bytes] : comparedTableSpans) {
        if (GuestMemory::CompareMapped(address, bytes) != GuestMemory::Compare::Equal) return fallback();
    }
    const auto generation = GuestMemory::SubscriptionGeneration(subscription);
    thread_local std::vector<std::size_t>* checkSlot = nullptr;
    auto& check = ShaderRecompiler::ThreadOwned(checkSlot);
    thread_local std::vector<std::size_t>* bumpedSlot = nullptr;
    auto& bumped = ShaderRecompiler::ThreadOwned(bumpedSlot);
    thread_local std::vector<std::size_t>* lookedSlot = nullptr;
    auto& looked = ShaderRecompiler::ThreadOwned(lookedSlot);
    thread_local std::vector<std::uint64_t>* blocksSlot = nullptr;
    auto& blocks = ShaderRecompiler::ThreadOwned(blocksSlot);
    thread_local std::vector<std::uint64_t>* lookupBlocksSlot = nullptr;
    auto& lookupBlocks = ShaderRecompiler::ThreadOwned(lookupBlocksSlot);
    check = perElementTextures;
    bumped.clear();
    looked.clear();
    const auto elementsOf = [&](const std::vector<std::uint64_t>& from, std::vector<std::size_t>& into) {
        for (const auto block : from) {
            auto entry = std::lower_bound(subscribedBlocks.begin(), subscribedBlocks.end(), std::pair<std::uint64_t, std::size_t>{block, 0});
            for (; entry != subscribedBlocks.end() && entry->first == block; ++entry) {
                if (entry->second == TableSpanElement) return false;
                into.push_back(entry->second);
            }
        }
        std::sort(into.begin(), into.end());
        into.erase(std::unique(into.begin(), into.end()), into.end());
        return true;
    };
    const bool partial = generation != subscribedGeneration;
    if (partial) {
        if (!GuestMemory::SubscriptionBumpsSince(subscription, subscribedGeneration, blocks, lookupBlocks)) return fallback();
        if (!elementsOf(blocks, bumped) || !elementsOf(lookupBlocks, looked)) return fallback();
        countIndex(IndexBumped, bumped.size());
        check.insert(check.end(), bumped.begin(), bumped.end());
        check.insert(check.end(), looked.begin(), looked.end());
        std::sort(check.begin(), check.end());
        check.erase(std::unique(check.begin(), check.end()), check.end());
    }
    thread_local std::vector<std::size_t>* replacedSlot = nullptr;
    auto& replaced = ShaderRecompiler::ThreadOwned(replacedSlot);
    replaced.clear();
    if (!looked.empty()) {
        countIndex(IndexLookedUp, looked.size());
        if (!reresolveElements(looked, replaced)) return fallback();
    }
    if (!comparedTextures.empty() && !IndexBroken("compared")) {
        countIndex(IndexCompared, comparedTextures.size());
        if (!reresolveElements(comparedTextures, replaced)) return fallback();
    }
    if (!check.empty() || !storageTextures.empty()) {
        thread_local std::vector<PendingOverlap>* overlappingSlot = nullptr;
        auto& overlapping = ShaderRecompiler::ThreadOwned(overlappingSlot);
        FastFail reason = FastFail::Count;
        bool accepted = false;
        indexProof = true;
        struct Done {
            bool& flag;
            ~Done() { flag = false; }
        } done{indexProof};
        if (!fastRevalidate(serialBefore, {}, reason, overlapping, accepted, &check, true)) {
            if (!partial || bumped.empty() || reason != FastFail::Changed) return fallback();
            if (!reresolveElements(bumped, replaced)) return fallback();
            if (!fastRevalidate(serialBefore, {}, reason, overlapping, accepted, &check, true)) return fallback();
        }
    }
    auto proved = generation;
    if (!replaced.empty()) {
        std::sort(replaced.begin(), replaced.end());
        const auto now = GuestMemory::SubscriptionGeneration(subscription);
        thread_local std::vector<std::size_t>* absorbedSlot = nullptr;
        auto& absorbed = ShaderRecompiler::ThreadOwned(absorbedSlot);
        absorbed.clear();
        const auto own = [&](std::size_t element) { return std::binary_search(replaced.begin(), replaced.end(), element) && !textures[element]->Released(); };
        if (now != generation && GuestMemory::SubscriptionBumpsSince(subscription, generation, blocks, lookupBlocks) && blocks.empty() && elementsOf(lookupBlocks, absorbed) && std::all_of(absorbed.begin(), absorbed.end(), own)) proved = now;
    }
    noteIndexProof(proved, serialBefore);
    return outcome(partial || !replaced.empty() ? IndexOutcome::Partial : IndexOutcome::Accepted);
}

const std::vector<std::pair<std::uint64_t, std::uint64_t>>* ShaderResources::rangedProofRanges() {
    if (!proofRangesBuilt) {
        proofRangesBuilt = true;
        // Address-based and table-driven builds read memory the template does not list.
        proofRangesKnown = bda == nullptr && tableStages.empty() && !UsesAddressTable();
        if (proofRangesKnown) {
            for (const auto& region : directRegions) proofRanges.emplace_back(region.begin, region.regionEnd != 0 ? std::max(region.end, region.regionEnd) : region.end);
            for (const auto& range : describedRanges) {
                proofRanges.emplace_back(range.address, range.address + range.bytes);
                if (range.dccAddress != 0) proofRanges.emplace_back(range.dccAddress, range.dccAddress + DccKeyBytes(range.bytes));
            }
            for (const auto& image : storageTextures) {
                proofRanges.emplace_back(image->Descriptor().baseAddress, image->Descriptor().baseAddress + image->GuestBytes());
                if (image->Descriptor().dccAddress != 0) proofRanges.emplace_back(image->Descriptor().dccAddress, image->Descriptor().dccAddress + DccKeyBytes(image->GuestBytes()));
            }
        }
    }
    return proofRangesKnown ? &proofRanges : nullptr;
}

namespace {

enum class CleanEvent { Hit, Requeried, Verified, Differs, NoRecord, Views, Serials, Forgotten, Rewritten, PendingWrites, NotReusable, StorageImages, DepthCopies, OwnWrites, NotClean, Unsettled, Count };

std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(CleanEvent::Count)>& cleanEvents() {
    static std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(CleanEvent::Count)> events{};
    return events;
}

void noteClean(CleanEvent event) {
    auto& events = cleanEvents();
    events[static_cast<std::size_t>(event)].fetch_add(1, std::memory_order_relaxed);
    static std::atomic<std::uint64_t> calls{0};
    if ((calls.fetch_add(1, std::memory_order_relaxed) & 0x7ffffu) != 0x7ffffu) return;
    constexpr std::array<const char*, static_cast<std::size_t>(CleanEvent::Count)> names{"hits", "of them re-queried", "verified", "verify differs", "no record", "views moved", "serials moved", "a range forgotten", "rewritten since the record", "pending writes", "not reusable", "storage images", "depth copies", "own writes", "not all clean", "written during the proof"};
    std::string line = "[clean-currency] template proofs (last 524288 events):";
    for (std::size_t index = 0; index < names.size(); ++index) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu", names[index], static_cast<unsigned long long>(events[index].exchange(0, std::memory_order_relaxed)));
        line += item;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
}

bool verifyCleanCurrency() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_CLEAN_CURRENCY");
        return text != nullptr ? std::max<std::uint64_t>(1u, std::strtoull(text, nullptr, 10)) : 0u;
    }();
    if (every == 0) return false;
    thread_local std::uint64_t hits = 0;
    return ++hits % every == 0;
}

}

namespace {

// APS5_PIPELINE_STATS: which ranges keep templates from a clean proof. Every not-clean template
// is counted by how many of its ranges are not clean, and each such range by address, so a line
// every 10 s names the ranges most templates fail on.
void noteUncleanRanges(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!stats) return;
    struct Census {
        std::mutex mutex;
        std::unordered_map<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>> byAddress;
        std::uint64_t templates = 0;
        std::uint64_t rangesSeen = 0;
        std::uint64_t rangesUnclean = 0;
        std::uint64_t oneUnclean = 0;
        std::int64_t reportedAt = 0;
    };
    static auto* census = new Census;
    std::lock_guard lock(census->mutex);
    ++census->templates;
    std::uint64_t unclean = 0;
    for (const auto& range : ranges) {
        ++census->rangesSeen;
        if (GuestMemory::AllClean(std::span(&range, 1))) continue;
        ++unclean;
        auto& entry = census->byAddress[range.first];
        ++entry.first;
        entry.second = range.second - range.first;
    }
    census->rangesUnclean += unclean;
    if (unclean == 1) ++census->oneUnclean;
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (census->reportedAt == 0) census->reportedAt = now;
    if (now - census->reportedAt < 10) return;
    census->reportedAt = now;
    std::vector<std::pair<std::uint64_t, std::pair<std::uint64_t, std::uint64_t>>> top(census->byAddress.begin(), census->byAddress.end());
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    std::string text;
    for (std::size_t index = 0; index < top.size() && index < 12; ++index) {
        char item[160];
        const auto address = top[index].first;
        const auto bytes = static_cast<std::size_t>(top[index].second.second);
        // What lives there: a live resident surface (L), a pending one (P), a depth surface (D), and
        // how many tracker generations ago the range was last stamped as written.
        const auto newest = GuestMemory::NewestStamp(address, bytes);
        const auto generation = GuestMemory::TrackerGeneration();
        std::snprintf(item, sizeof(item), " 0x%llx+%llu x%llu %s%s%s age %llu", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(top[index].second.first), StorageTexture::FindLive(address, bytes) != nullptr ? "L" : "-", StorageTexture::FindPending(address, bytes) != nullptr ? "P" : "-", DepthSurfaceAt(address) ? "D" : "-", static_cast<unsigned long long>(generation >= newest ? generation - newest : 0));
        text += item;
    }
    std::fprintf(stderr, "[clean-ranges] %llu not-clean templates checked: %llu of %llu ranges not clean, %llu templates with exactly one; %llu distinct addresses; top:%s\n", static_cast<unsigned long long>(census->templates), static_cast<unsigned long long>(census->rangesUnclean), static_cast<unsigned long long>(census->rangesSeen), static_cast<unsigned long long>(census->oneUnclean), static_cast<unsigned long long>(census->byAddress.size()), text.c_str());
    census->byAddress.clear();
    census->templates = census->rangesSeen = census->rangesUnclean = census->oneUnclean = 0;
}

}

namespace {

// APS5_PIPELINE_STATS: which serial of the clean stamp moved when a kept clean proof is dropped
// (several can move at once), and how often only the pending-store serial did.
std::atomic<std::uint64_t> cleanSerialsKept{0};

void noteCleanSerialsKept() {
    cleanSerialsKept.fetch_add(1, std::memory_order_relaxed);
}

void noteCleanSerials(std::array<bool, 6> moved) {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!stats) return;
    static std::array<std::atomic<std::uint64_t>, 6> counts{};
    static std::atomic<std::uint64_t> drops{0}, pendingOnly{0};
    static std::atomic<std::int64_t> reportedAt{0};
    for (std::size_t index = 0; index < moved.size(); ++index) {
        if (moved[index]) counts[index].fetch_add(1, std::memory_order_relaxed);
    }
    drops.fetch_add(1, std::memory_order_relaxed);
    if (moved[1] && !moved[0] && !moved[2] && !moved[3] && !moved[4] && !moved[5]) pendingOnly.fetch_add(1, std::memory_order_relaxed);
    const auto tenSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (reportedAt.exchange(tenSeconds, std::memory_order_relaxed) == tenSeconds) return;
    std::fprintf(stderr, "[clean-serials] %llu kept clean proofs dropped for a moved serial: release %llu, pending %llu (pending only %llu), imports %llu, allocations %llu, unwatched %llu, depth surfaces %llu; kept through a release or pending change elsewhere so far %llu\n", static_cast<unsigned long long>(drops.exchange(0)), static_cast<unsigned long long>(counts[0].exchange(0)), static_cast<unsigned long long>(counts[1].exchange(0)), static_cast<unsigned long long>(pendingOnly.exchange(0)), static_cast<unsigned long long>(counts[2].exchange(0)), static_cast<unsigned long long>(counts[3].exchange(0)), static_cast<unsigned long long>(counts[4].exchange(0)), static_cast<unsigned long long>(counts[5].exchange(0)), static_cast<unsigned long long>(cleanSerialsKept.load(std::memory_order_relaxed)));
}

}

namespace {

enum class CleanRuleEvent : std::size_t { ViewHits, PendingHits, BothHits, ExemptRecords, ExemptOverPending, RequiredNotClean, ExemptUnsettled, NoSources, SnapshotOverView, UnrecordedViews, ViewsReleased, ViewsPending, ViewsShadowed, ViewsRewritten, SourcesFailed, SourcesAccepted, SourcesRewritten, ExemptForgotten, ViewsVerified, ViewsDiffers, PendingVerified, PendingDiffers, Count };

constexpr std::size_t CleanRuleEvents = static_cast<std::size_t>(CleanRuleEvent::Count);
constexpr std::size_t CleanSourceReasons = static_cast<std::size_t>(FastFail::Count) + 1;

struct CleanRuleCensus {
    std::array<std::atomic<std::uint64_t>, CleanRuleEvents> events{};
    std::array<std::atomic<std::uint64_t>, CleanSourceReasons> failed{};
    std::array<std::uint64_t, CleanRuleEvents> eventsReported{};
    std::array<std::uint64_t, CleanSourceReasons> failedReported{};
    std::atomic<std::int64_t> reportedAt{0};
    std::mutex mutex;
};

CleanRuleCensus& cleanRuleCensus() {
    static auto* census = new CleanRuleCensus;
    return *census;
}

void reportCleanRules() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!stats) return;
    auto& census = cleanRuleCensus();
    const auto tenSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    const auto previous = census.reportedAt.exchange(tenSeconds, std::memory_order_relaxed);
    if (previous == tenSeconds) return;
    std::array<std::uint64_t, CleanRuleEvents> events{};
    std::array<std::uint64_t, CleanSourceReasons> failed{};
    {
        std::lock_guard lock(census.mutex);
        for (std::size_t index = 0; index < CleanRuleEvents; ++index) {
            const auto now = census.events[index].load(std::memory_order_relaxed);
            events[index] = now - census.eventsReported[index];
            census.eventsReported[index] = now;
        }
        for (std::size_t index = 0; index < CleanSourceReasons; ++index) {
            const auto now = census.failed[index].load(std::memory_order_relaxed);
            failed[index] = now - census.failedReported[index];
            census.failedReported[index] = now;
        }
    }
    if (previous == 0) return;
    const auto at = [&](CleanRuleEvent event) { return static_cast<unsigned long long>(events[static_cast<std::size_t>(event)]); };
    std::string reasons;
    for (std::size_t index = 0; index < CleanSourceReasons; ++index) {
        if (failed[index] == 0) continue;
        char item[64];
        std::snprintf(item, sizeof(item), "%s%s %llu", reasons.empty() ? "" : ", ", index < FastFailNames.size() ? FastFailNames[index] : "other", static_cast<unsigned long long>(failed[index]));
        reasons += item;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[clean-rules] 10 s: hits through resident views %llu, through pending ranges %llu, through both %llu; records with pending ranges %llu (%llu over a pending surface), refused for other ranges not clean %llu, written during the proof %llu; not eligible: no viewed surface %llu, a snapshot over a viewed surface %llu, unrecorded elements %llu; resident-view keeps dropped: released %llu, pending over other ranges %llu, shadowed regions %llu, rewritten %llu; element proofs failed %llu (%s), with an accepted overlap %llu; rewritten after the element proof %llu; pending-range records forgotten %llu; verify: resident views %llu agreed, %llu differed, pending ranges %llu agreed, %llu differed\n",
        at(CleanRuleEvent::ViewHits), at(CleanRuleEvent::PendingHits), at(CleanRuleEvent::BothHits), at(CleanRuleEvent::ExemptRecords), at(CleanRuleEvent::ExemptOverPending), at(CleanRuleEvent::RequiredNotClean), at(CleanRuleEvent::ExemptUnsettled),
        at(CleanRuleEvent::NoSources), at(CleanRuleEvent::SnapshotOverView), at(CleanRuleEvent::UnrecordedViews),
        at(CleanRuleEvent::ViewsReleased), at(CleanRuleEvent::ViewsPending), at(CleanRuleEvent::ViewsShadowed), at(CleanRuleEvent::ViewsRewritten),
        at(CleanRuleEvent::SourcesFailed), reasons.empty() ? "none" : reasons.c_str(), at(CleanRuleEvent::SourcesAccepted), at(CleanRuleEvent::SourcesRewritten), at(CleanRuleEvent::ExemptForgotten),
        at(CleanRuleEvent::ViewsVerified), at(CleanRuleEvent::ViewsDiffers), at(CleanRuleEvent::PendingVerified), at(CleanRuleEvent::PendingDiffers));
}

void noteCleanRule(CleanRuleEvent event) {
    cleanRuleCensus().events[static_cast<std::size_t>(event)].fetch_add(1, std::memory_order_relaxed);
    reportCleanRules();
}

void noteCleanSourceFailure(FastFail reason) {
    const auto index = std::min(static_cast<std::size_t>(reason), CleanSourceReasons - 1);
    cleanRuleCensus().failed[index].fetch_add(1, std::memory_order_relaxed);
    noteCleanRule(CleanRuleEvent::SourcesFailed);
}

}

ShaderResources::CleanRuleCounts ShaderResources::CleanRuleCounted() {
    auto& census = cleanRuleCensus();
    const auto at = [&](CleanRuleEvent event) { return census.events[static_cast<std::size_t>(event)].load(std::memory_order_relaxed); };
    CleanRuleCounts counts;
    counts.viewHits = at(CleanRuleEvent::ViewHits) + at(CleanRuleEvent::BothHits);
    counts.pendingHits = at(CleanRuleEvent::PendingHits) + at(CleanRuleEvent::BothHits);
    counts.exemptRecords = at(CleanRuleEvent::ExemptRecords);
    counts.viewDrops = at(CleanRuleEvent::ViewsReleased) + at(CleanRuleEvent::ViewsPending) + at(CleanRuleEvent::ViewsShadowed) + at(CleanRuleEvent::ViewsRewritten);
    counts.sourceDrops = at(CleanRuleEvent::SourcesFailed) + at(CleanRuleEvent::SourcesAccepted) + at(CleanRuleEvent::SourcesRewritten) + at(CleanRuleEvent::ExemptForgotten);
    counts.verified = at(CleanRuleEvent::ViewsVerified) + at(CleanRuleEvent::PendingVerified);
    counts.differs = at(CleanRuleEvent::ViewsDiffers) + at(CleanRuleEvent::PendingDiffers);
    counts.requiredNotClean = at(CleanRuleEvent::RequiredNotClean);
    counts.exemptUnsettled = at(CleanRuleEvent::ExemptUnsettled);
    counts.noSources = at(CleanRuleEvent::NoSources);
    counts.snapshotOverView = at(CleanRuleEvent::SnapshotOverView);
    counts.unrecordedViews = at(CleanRuleEvent::UnrecordedViews);
    return counts;
}

ShaderResources::CleanStamp ShaderResources::cleanStampNow() const {
    return {GuestMemory::CleanForgetSerial(), TextureReleaseSerial().load(std::memory_order_acquire), StorageTexture::PendingSerial(), HostImportSetGeneration(), GuestAllocations::GuestAllocationsGeneration_nid_postfix(), GuestMemory::UnwatchSerial(), DepthSurfacesSerial()};
}

bool ShaderResources::cleanEligible() {
    if (cleanEligibility == 0) {
        const bool depthCopies = std::any_of(textures.begin(), textures.end(), [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->IsDepthCopy(); });
        cleanEligibility = !reusable ? 2 : !storageTextures.empty() ? 3 : depthCopies ? 4 : 1;
        if (cleanEligibility == 1) {
            for (const auto& range : describedRanges) {
                cleanRangeList.emplace_back(range.address, range.address + range.bytes);
                if (range.dccAddress != 0) cleanRangeList.emplace_back(range.dccAddress, range.dccAddress + DccKeyBytes(range.bytes));
            }
        }
    }
    if (cleanEligibility != 1) {
        noteClean(cleanEligibility == 2 ? CleanEvent::NotReusable : cleanEligibility == 3 ? CleanEvent::StorageImages : CleanEvent::DepthCopies);
        cleanRefusal = cleanEligibility == 2 ? CleanProof::NotReusable : cleanEligibility == 3 ? CleanProof::StorageImages : CleanProof::DepthCopies;
        return false;
    }
    if (!guestMemory.Writes().empty() || std::find(storageWritten.begin(), storageWritten.end(), true) != storageWritten.end()) {
        noteClean(CleanEvent::OwnWrites);
        cleanRefusal = CleanProof::OwnWrites;
        return false;
    }
    return true;
}

bool ShaderResources::cleanUnchangedSince(std::uint64_t generation) const {
    if (generation == 0) return false;
    struct CleanQueriesStorage {};
    auto& queries = HostThreadLocal<std::vector<GuestMemory::UnchangedQuery>, CleanQueriesStorage>();
    queries.clear();
    for (const auto& [begin, end] : cleanRangeList) queries.push_back({begin, static_cast<std::size_t>(end - begin), generation});
    return queries.empty() || GuestMemory::UnchangedSinceAll(queries);
}

bool ShaderResources::cleanUnchangedSince(std::uint64_t generation, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) const {
    if (generation == 0) return false;
    struct CleanQueriesStorage {};
    auto& queries = HostThreadLocal<std::vector<GuestMemory::UnchangedQuery>, CleanQueriesStorage>();
    queries.clear();
    for (const auto& [begin, end] : ranges) queries.push_back({begin, static_cast<std::size_t>(end - begin), generation});
    return queries.empty() || GuestMemory::UnchangedSinceAll(queries);
}

void ShaderResources::noteCleanSources() {
    cleanSourced.clear();
    cleanRequired.clear();
    cleanSourcesEligible = false;
    if (validatedTextures.size() != textures.size()) {
        noteCleanRule(CleanRuleEvent::UnrecordedViews);
        return;
    }
    for (std::size_t i = 0; i < validatedTextures.size(); ++i) {
        const auto& surface = validatedTextures[i];
        if (!surface.valid) {
            noteCleanRule(CleanRuleEvent::UnrecordedViews);
            cleanSourced.clear();
            return;
        }
        if (surface.source != nullptr && !surface.depth) cleanSourced.push_back(i);
    }
    if (cleanSourced.empty()) {
        noteCleanRule(CleanRuleEvent::NoSources);
        return;
    }
    const auto viewed = [&](std::uint64_t begin, std::uint64_t end) {
        return std::any_of(cleanSourced.begin(), cleanSourced.end(), [&](std::size_t i) {
            const auto& surface = validatedTextures[i];
            return begin < surface.resource.baseAddress + surface.bytes && surface.resource.baseAddress < end;
        });
    };
    for (const auto& surface : validatedTextures) {
        if (surface.source != nullptr && !surface.depth) continue;
        const auto begin = surface.resource.baseAddress;
        const auto end = begin + std::max<std::uint64_t>(surface.bytes, 1u);
        const auto keys = surface.resource.dccAddress;
        if (viewed(begin, end) || (keys != 0 && viewed(keys, keys + std::max<std::uint64_t>(DccKeyBytes(surface.bytes), 1u)))) {
            noteCleanRule(CleanRuleEvent::SnapshotOverView);
            cleanSourced.clear();
            return;
        }
    }
    const auto ownSurface = [&](const std::pair<std::uint64_t, std::uint64_t>& range) {
        return std::any_of(cleanSourced.begin(), cleanSourced.end(), [&](std::size_t i) {
            const auto& surface = validatedTextures[i];
            return range.first == surface.resource.baseAddress && range.second == surface.resource.baseAddress + surface.bytes;
        });
    };
    for (const auto& range : cleanRangeList) {
        if (!ownSurface(range)) cleanRequired.push_back(range);
    }
    cleanSourcesEligible = true;
}

std::uint8_t ShaderResources::cleanViewsKept(const CleanStamp& stamp) {
    if (std::any_of(textures.begin(), textures.end(), [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->Released(); })) {
        noteCleanRule(CleanRuleEvent::ViewsReleased);
        return 1;
    }
    struct PendingRangesStorage {};
    auto& ranges = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, PendingRangesStorage>();
    ranges.assign(cleanRequired.begin(), cleanRequired.end());
    for (const auto& region : directRegions) ranges.emplace_back(region.begin, region.end);
    if (StorageTexture::AnyPendingOverlaps(ranges)) {
        noteCleanRule(CleanRuleEvent::ViewsPending);
        return 2;
    }
    ranges.clear();
    for (const auto& region : directRegions) ranges.emplace_back(region.begin, region.end);
    if (!ranges.empty() && AnyShadowedOverlaps(ranges)) {
        noteCleanRule(CleanRuleEvent::ViewsShadowed);
        return 3;
    }
    if (!cleanUnchangedSince(cleanAt.generation, cleanExempt ? cleanRequired : cleanRangeList)) {
        noteCleanRule(CleanRuleEvent::ViewsRewritten);
        return 4;
    }
    cleanAt.release = stamp.release;
    cleanAt.pending = stamp.pending;
    return 0;
}

bool ShaderResources::cleanSourcesCurrent(FastFail& reason) {
    const auto serialBefore = StorageTexture::PendingSerial();
    struct CleanOverlapStorage {};
    auto& overlapping = ThreadScratch<std::vector<PendingOverlap>, CleanOverlapStorage>();
    bool accepted = false;
    reason = FastFail::Count;
    if (!fastRevalidate(serialBefore, {}, reason, overlapping, accepted, &cleanSourced, false)) {
        noteCleanSourceFailure(reason);
        return false;
    }
    if (accepted) {
        noteCleanRule(CleanRuleEvent::SourcesAccepted);
        return false;
    }
    return true;
}

bool ShaderResources::cleanCurrent(const CleanStamp& stamp, CleanProof& outcome) {
    const auto refuse = [&](CleanEvent event, CleanProof why) {
        cleanProved = false;
        noteClean(event);
        outcome = why;
        return false;
    };
    if (!cleanProved) {
        noteClean(CleanEvent::NoRecord);
        outcome = cleanRefusal;
        return false;
    }
    if (!registeredImageViewsCurrent(registeredImageViews)) return refuse(CleanEvent::Views, CleanProof::Views);
    cleanHitRule = 0;
    if (cleanExempt && !RuntimeSwitches::Set(RuntimeSwitch::CleanPendingRanges)) return refuse(CleanEvent::NotClean, CleanProof::NotClean);
    const bool storeSerialsOnly = (stamp.release != cleanAt.release || stamp.pending != cleanAt.pending) && stamp.imports == cleanAt.imports && stamp.allocations == cleanAt.allocations && stamp.unwatched == cleanAt.unwatched && stamp.depthSurfaces == cleanAt.depthSurfaces;
    bool proveSources = cleanExempt;
    bool viewsKept = false;
    // APS5_CLEAN_PRECISE_SERIALS: the release and pending-store serials are global, and one of them
    // moves between most uses of a template. What they stand for here is narrower: one of this
    // template's own textures was released, or a pending surface now lies over one of its ranges
    // (its snapshot texture would be stale against it). When only those two moved and neither is
    // the case, the kept proof stays and adopts the new serials.
    static const bool preciseSerials = std::getenv("APS5_CLEAN_PRECISE_SERIALS") != nullptr;
    if (storeSerialsOnly && cleanSourcesEligible && RuntimeSwitches::Set(RuntimeSwitch::CleanResidentViews)) {
        const auto dropped = cleanViewsKept(stamp);
        if (dropped == 4) return refuse(CleanEvent::Rewritten, CleanProof::Rewritten);
        viewsKept = dropped == 0;
        proveSources = proveSources || viewsKept;
    } else if (preciseSerials && storeSerialsOnly) {
        const bool released = std::any_of(textures.begin(), textures.end(), [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->Released(); });
        // A view of a resident image follows that image's own state (its keys and pending results),
        // which these checks do not see: a template with such a view keeps the plain serial rule.
        const bool sourced = std::any_of(validatedTextures.begin(), validatedTextures.end(), [](const auto& surface) { return surface.source != nullptr; });
        // A surface that was pending over a range and has stored since is gone from the registry
        // but stamped the range as written, so the ranges must also be unwritten since the proof.
        if (!released && !sourced && !StorageTexture::AnyPendingOverlaps(cleanRangeList) && cleanUnchangedSince(cleanAt.generation)) {
            cleanAt.release = stamp.release;
            cleanAt.pending = stamp.pending;
            noteCleanSerialsKept();
        }
    }
    if (stamp.release != cleanAt.release || stamp.pending != cleanAt.pending || stamp.imports != cleanAt.imports || stamp.allocations != cleanAt.allocations || stamp.unwatched != cleanAt.unwatched || stamp.depthSurfaces != cleanAt.depthSurfaces) {
        noteCleanSerials({stamp.release != cleanAt.release, stamp.pending != cleanAt.pending, stamp.imports != cleanAt.imports, stamp.allocations != cleanAt.allocations, stamp.unwatched != cleanAt.unwatched, stamp.depthSurfaces != cleanAt.depthSurfaces});
        return refuse(CleanEvent::Serials, CleanProof::Serials);
    }
    outcome = CleanProof::Skipped;
    const auto* ranges = cleanExempt ? &cleanRequired : &cleanRangeList;
    if (stamp.forget != cleanAt.forget) {
        if (!GuestMemory::AllClean(*ranges)) {
            if (cleanExempt) noteCleanRule(CleanRuleEvent::ExemptForgotten);
            return refuse(CleanEvent::Forgotten, CleanProof::Forgotten);
        }
        if (!cleanUnchangedSince(cleanAt.generation, *ranges)) return refuse(CleanEvent::Rewritten, CleanProof::Rewritten);
        cleanAt.forget = stamp.forget;
        noteClean(CleanEvent::Requeried);
        outcome = CleanProof::Requeried;
    }
    const bool pendingCpu = [&] {
        if (Recorder::InCompletion() || Recorder::Active() == nullptr) return false;
        return (context.copiedWriters != nullptr && !context.copiedWriters->empty()) || !DrawCopiedWriters()->empty() || Recorder::PendingCompletionLabels() != 0;
    }();
    for (const auto& [begin, end] : cleanRangeList) {
        if (!pendingCpu) break;
        if (CompletePendingCpuWrites(context, begin, static_cast<std::size_t>(end - begin))) return refuse(CleanEvent::PendingWrites, CleanProof::PendingWrites);
    }
    for (const auto& region : directRegions) {
        static_cast<void>(FlushDepthPlanes(region.begin, static_cast<std::size_t>(region.end - region.begin)));
        if (pendingCpu && CompletePendingCpuWrites(context, region.begin, static_cast<std::size_t>(region.end - region.begin))) return refuse(CleanEvent::PendingWrites, CleanProof::PendingWrites);
    }
    if (proveSources) {
        auto reason = FastFail::Count;
        if (!cleanSourcesCurrent(reason)) return reason == FastFail::Changed ? refuse(CleanEvent::Rewritten, CleanProof::Rewritten) : refuse(CleanEvent::Serials, CleanProof::Serials);
        if (cleanExempt && !cleanUnchangedSince(cleanAt.generation, cleanRequired)) {
            noteCleanRule(CleanRuleEvent::SourcesRewritten);
            return refuse(CleanEvent::Rewritten, CleanProof::Rewritten);
        }
        cleanHitRule = static_cast<std::uint8_t>((viewsKept ? 1u : 0u) | (cleanExempt ? 2u : 0u));
    }
    if (!guestMemory.StagingDecisionsCurrent()) return refuse(CleanEvent::PendingWrites, CleanProof::PendingWrites);
    if (auto* recorder = Recorder::Active(); recorder != nullptr) {
        try {
            guestMemory.RecordStagingCopies(*recorder);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[resources] staging copies of a clean-currency hit failed: %s\n", error.what());
            cleanProved = false;
            outcome = CleanProof::PendingWrites;
            return false;
        }
    }
    if (cleanHitRule != 0) noteCleanRule(cleanHitRule == 1 ? CleanRuleEvent::ViewHits : cleanHitRule == 2 ? CleanRuleEvent::PendingHits : CleanRuleEvent::BothHits);
    return true;
}

void ShaderResources::recordClean(bool proved, const CleanStamp& stamp) {
    cleanProved = false;
    if (!proved) {
        cleanRefusal = CleanProof::Failed;
        return;
    }
    if (!cleanEligible()) return;
    const bool pendingRule = RuntimeSwitches::Set(RuntimeSwitch::CleanPendingRanges);
    if (pendingRule || RuntimeSwitches::Set(RuntimeSwitch::CleanResidentViews)) noteCleanSources();
    else if (cleanSourcesEligible) cleanSourcesEligible = false;
    if (!(pendingRule && cleanSourcesEligible) && cleanMisses != 0 && (++cleanMisses & 15u) != 0) {
        cleanRefusal = CleanProof::NotClean;
        return;
    }
    bool exempt = false;
    if (!GuestMemory::AllClean(cleanRangeList)) {
        exempt = pendingRule && cleanSourcesEligible && GuestMemory::AllClean(cleanRequired);
        if (!exempt) {
            if (pendingRule && cleanSourcesEligible) noteCleanRule(CleanRuleEvent::RequiredNotClean);
            if (cleanMisses == 0) cleanMisses = 1;
            noteClean(CleanEvent::NotClean);
            noteUncleanRanges(pendingRule && cleanSourcesEligible ? cleanRequired : cleanRangeList);
            cleanRefusal = CleanProof::NotClean;
            return;
        }
    }
    cleanMisses = 0;
    if (!cleanUnchangedSince(stamp.generation, exempt ? cleanRequired : cleanRangeList)) {
        if (exempt) noteCleanRule(CleanRuleEvent::ExemptUnsettled);
        noteClean(CleanEvent::Unsettled);
        cleanRefusal = CleanProof::Unsettled;
        return;
    }
    cleanProved = true;
    cleanAt = stamp;
    cleanExempt = exempt;
    if (!exempt) return;
    noteCleanRule(CleanRuleEvent::ExemptRecords);
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (stats && std::any_of(cleanSourced.begin(), cleanSourced.end(), [&](std::size_t i) { return StorageTexture::FindPending(validatedTextures[i].resource.baseAddress, validatedTextures[i].bytes).get() == validatedTextures[i].source; })) noteCleanRule(CleanRuleEvent::ExemptOverPending);
}

bool ShaderResources::expressEligible() const {
    return reusable && cleanEligibility <= 1 && storageTextures.empty() && !heapStage && (bda == nullptr || FaultBufferLean()) && !addressedBuffers && !rebound && !GuestMemory::WriteIndexEnabled() && RuntimeSwitches::Set(RuntimeSwitch::ExpressCommits);
}

bool ShaderResources::ExpressWatchable() const {
    return expressEligible() && std::none_of(textures.begin(), textures.end(), [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->IsDepthCopy(); });
}

ShaderResources::ExpressBefore ShaderResources::expressBefore() {
    ExpressBefore before;
    expressSourcesBefore.clear();
    expressProofEpochsBefore.clear();
    if (!expressEligible()) return before;
    if (expressWatch == nullptr) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        ranges.reserve(describedRanges.size() * 2u);
        for (const auto& range : describedRanges) {
            ranges.emplace_back(range.address, range.address + range.bytes);
            if (range.dccAddress != 0) ranges.emplace_back(range.dccAddress, range.dccAddress + DccKeyBytes(range.bytes));
        }
        expressWatch = GuestMemory::SubscribeWriteGeneration(ranges);
    }
    before.generation = GuestMemory::SubscriptionGeneration(*expressWatch);
    before.watched = true;
    for (const auto& surface : validatedTextures) {
        if (surface.source == nullptr) continue;
        expressProofEpochsBefore.push_back(surface.source->ProofEpoch());
        expressSourcesBefore.emplace_back(surface.source, surface.source->StateEpoch());
    }
    return before;
}

void ShaderResources::expressAfter(bool proved, const ExpressBefore& before) {
    ++wvProofSerial;
    expressValid = false;
    if (!proved || !before.watched || !cleanProved) return;
    std::size_t at = 0;
    for (const auto& surface : validatedTextures) {
        if (surface.source == nullptr) continue;
        if (at >= expressSourcesBefore.size() || expressSourcesBefore[at].first != surface.source) return;
        ++at;
    }
    if (at != expressSourcesBefore.size()) return;
    expressSources.swap(expressSourcesBefore);
    expressProofEpochs.swap(expressProofEpochsBefore);
    expressGeneration = before.generation;
    expressAt = cleanAt;
    expressValid = true;
}

ShaderResources::ExpressMiss ShaderResources::ExpressCurrent() {
    return expressCurrent(false);
}

ShaderResources::ExpressMiss ShaderResources::ExpressCurrentFollowing() {
    return expressCurrent(true);
}

namespace {

constexpr std::size_t CommitterTrustSites = static_cast<std::size_t>(CommitterTrustSite::Count);

constexpr std::array<const char*, CommitterTrustSites> CommitterTrustSiteNames{
    "follower, template current (watched writes, sources, views, pending CPU writes, staging, shadowed regions)",
    "leader, template current (the same tests)",
    "template proof (record, repeat and reproving commits)",
    "follower, buffer hazards (prepared buffer zones, pass reads and images, queued stores, selection and guest buffer guards)",
    "leader, buffer hazards (prepared buffer zones, moved and in-place ranges, queued stores, selection and guest buffer guards)",
    "record draw, prepared buffers (moved and in-place ranges)",
    "follower, vertex inputs (pending writes, labels and queued stores over vertex ranges, input read guards)",
    "leader, vertex inputs (the same tests)",
    "record draw, input read guards",
};

constexpr std::array<const char*, CommitterTrustSites> CommitterTrustObjectNames{
    "the template, shared by its three currency sites",
    "the template, shared by its three currency sites",
    "the template, shared by its three currency sites",
    "the template, shared by its three buffer sites",
    "the template, shared by its three buffer sites",
    "the template, shared by its three buffer sites",
    "the template, shared by its three input sites",
    "the template, shared by its three input sites",
    "the template, shared by its three input sites",
};

struct CommitterTrustCensus {
    std::array<std::atomic<std::uint64_t>, CommitterTrustSites> audits{};
    std::array<std::atomic<std::uint64_t>, CommitterTrustSites> auditFailures{};
    std::array<std::atomic<std::uint64_t>, CommitterTrustSites> volatileUses{};
    std::array<std::atomic<std::uint64_t>, CommitterTrustSites> volatileFailures{};
    std::array<std::atomic<std::uint64_t>, CommitterTrustSites> demoted{};
    std::atomic<std::uint64_t> checkedPresent{0};
    std::atomic<std::uint64_t> reportedPresent{0};
    std::atomic<std::int64_t> reportedAt{0};
    std::atomic<std::uint32_t> seed{0x9e3779b9u};
};

CommitterTrustCensus committerTrustCensus;

void countCommitterTrust(std::atomic<std::uint64_t>& counter) {
    counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

std::uint32_t committerTrustEvery() {
    static const std::uint32_t every = [] {
        const char* text = std::getenv("APS5_TRUST_AUDIT_EVERY");
        if (text == nullptr || *text == 0) return std::uint32_t{32};
        return static_cast<std::uint32_t>(std::clamp<unsigned long long>(std::strtoull(text, nullptr, 10), 2ull, 0x7fffffffull));
    }();
    return every;
}

std::uint32_t committerTrustHold() {
    static const std::uint32_t hold = [] {
        const char* text = std::getenv("APS5_TRUST_AUDIT_HOLD");
        if (text == nullptr || *text == 0) return std::uint32_t{600};
        return static_cast<std::uint32_t>(std::clamp<unsigned long long>(std::strtoull(text, nullptr, 10), 1ull, 0x3fffffffull));
    }();
    return hold;
}

std::uint32_t committerTrustPeriod() {
    const auto every = committerTrustEvery();
    auto& seed = committerTrustCensus.seed;
    const auto next = seed.load(std::memory_order_relaxed) * 1664525u + 1013904223u;
    seed.store(next, std::memory_order_relaxed);
    return every - (next >> 16) % ((every + 1u) / 2u);
}

struct CommitterTrustRow {
    std::uint64_t trusted = 0;
    std::uint64_t audits = 0;
    std::uint64_t auditFailures = 0;
    std::uint64_t volatileUses = 0;
    std::uint64_t volatileFailures = 0;
    std::uint64_t demoted = 0;
};

std::string committerTrustText(const CommitterTrustRow& row) {
    const double rate = row.audits != 0 ? static_cast<double>(row.auditFailures) / static_cast<double>(row.audits) : 0.0;
    std::array<char, 512> text{};
    std::snprintf(text.data(), text.size(), ": trusted %llu, audited %llu, audits failed %llu (%.2f%% of audits, about %.0f stale trusted uses at that rate), tested while volatile %llu (%llu failed), demoted %llu",
        static_cast<unsigned long long>(row.trusted), static_cast<unsigned long long>(row.audits), static_cast<unsigned long long>(row.auditFailures), 100.0 * rate, rate * static_cast<double>(row.trusted),
        static_cast<unsigned long long>(row.volatileUses), static_cast<unsigned long long>(row.volatileFailures), static_cast<unsigned long long>(row.demoted));
    return text.data();
}

void reportCommitterTrust(std::uint64_t presents) {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!stats) return;
    auto& census = committerTrustCensus;
    if (census.checkedPresent.load(std::memory_order_relaxed) == presents) return;
    census.checkedPresent.store(presents, std::memory_order_relaxed);
    const auto tenSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (census.reportedAt.exchange(tenSeconds, std::memory_order_relaxed) == tenSeconds) return;
    const auto interval = presents - census.reportedPresent.exchange(presents, std::memory_order_relaxed);
    std::array<CommitterTrustRow, CommitterTrustSites> rows{};
    CommitterTrustRow all;
    for (std::size_t at = 0; at < CommitterTrustSites; ++at) {
        auto& row = rows[at];
        row.trusted = committerTrustTrusted[at].exchange(0, std::memory_order_relaxed);
        row.audits = census.audits[at].exchange(0, std::memory_order_relaxed);
        row.auditFailures = census.auditFailures[at].exchange(0, std::memory_order_relaxed);
        row.volatileUses = census.volatileUses[at].exchange(0, std::memory_order_relaxed);
        row.volatileFailures = census.volatileFailures[at].exchange(0, std::memory_order_relaxed);
        row.demoted = census.demoted[at].exchange(0, std::memory_order_relaxed);
        all.trusted += row.trusted;
        all.audits += row.audits;
        all.auditFailures += row.auditFailures;
        all.volatileUses += row.volatileUses;
        all.volatileFailures += row.volatileFailures;
        all.demoted += row.demoted;
    }
    if (all.trusted == 0 && all.audits == 0 && all.volatileUses == 0) return;
    const auto every = committerTrustEvery();
    AgcDriver::ProfilePrint_nid_no_patch("[trust-audit] committer, %llu presents, not exact: each state of a template runs the real test once in %u to %u uses (drawn again after each audit), a failed test has that state tested at every use for %u presents; all sites%s\n", static_cast<unsigned long long>(interval), every - (every + 1u) / 2u + 1u, every, committerTrustHold(), committerTrustText(all).c_str());
    for (std::size_t at = 0; at < CommitterTrustSites; ++at) {
        const auto& row = rows[at];
        if (row.trusted == 0 && row.audits == 0 && row.volatileUses == 0) continue;
        AgcDriver::ProfilePrint_nid_no_patch("[trust-audit] committer, %s, state on %s%s\n", CommitterTrustSiteNames[at], CommitterTrustObjectNames[at], committerTrustText(row).c_str());
    }
}

}

CommitterTrustUse CommitterTrustSlow(CommitterTrustState& state, CommitterTrustSite site) {
    const auto at = static_cast<std::size_t>(site);
    auto& census = committerTrustCensus;
    const auto presents = Recorder::Presents();
    reportCommitterTrust(presents);
    if (state.untilAudit == 0 && static_cast<std::int32_t>(state.volatileUntil - static_cast<std::uint32_t>(presents)) > 0) {
        countCommitterTrust(census.volatileUses[at]);
        return CommitterTrustUse::Volatile;
    }
    state.untilAudit = committerTrustPeriod();
    countCommitterTrust(census.audits[at]);
    return CommitterTrustUse::Audit;
}

void CommitterTrustFailed(CommitterTrustState& state, CommitterTrustSite site, CommitterTrustUse use) {
    const auto at = static_cast<std::size_t>(site);
    auto& census = committerTrustCensus;
    if (use == CommitterTrustUse::Audit) {
        countCommitterTrust(census.auditFailures[at]);
        countCommitterTrust(census.demoted[at]);
    } else {
        countCommitterTrust(census.volatileFailures[at]);
    }
    state.untilAudit = 0;
    state.volatileUntil = static_cast<std::uint32_t>(Recorder::Presents()) + committerTrustHold();
}

namespace {

constexpr std::size_t ProofKeepChanges = static_cast<std::size_t>(StorageTexture::StateChange::Count);

struct ProofKeepCensus {
    std::atomic<std::uint64_t> kept{0};
    std::atomic<std::uint64_t> keptNew{0};
    std::atomic<std::uint64_t> held{0};
    std::atomic<std::uint64_t> contentOff{0};
    std::atomic<std::uint64_t> contentLater{0};
    std::array<std::atomic<std::uint64_t>, ProofKeepChanges> reproved{};
    std::atomic<std::uint64_t> checked{0};
    std::atomic<std::uint64_t> differing{0};
    std::atomic<std::uint64_t> failed{0};
    std::atomic<std::uint64_t> otherObjects{0};
    std::atomic<std::uint64_t> otherKeys{0};
    std::atomic<std::uint64_t> versionMoved{0};
    std::atomic<std::uint64_t> notComparable{0};
    std::atomic<std::uint64_t> noExpressState{0};
    std::atomic<std::uint64_t> checkedPresent{0};
    std::atomic<std::uint64_t> reportedPresent{0};
    std::atomic<std::int64_t> reportedAt{0};
};

ProofKeepCensus proofKeepCensus;

std::uint64_t proofKeepsVerifyEvery() {
    static const std::uint64_t every = [] {
        const char* text = std::getenv("APS5_VERIFY_PROOF_KEEPS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    return every;
}

bool proofKeepsCensus() {
    static const bool on = std::getenv("APS5_PIPELINE_STATS") != nullptr || proofKeepsVerifyEvery() != 0;
    return on;
}

void countProofKeep(std::atomic<std::uint64_t>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

std::string proofKeepChangesText(const std::array<std::uint64_t, ProofKeepChanges>& counts, bool content, std::uint64_t& total) {
    std::string text;
    total = 0;
    for (std::size_t at = 0; at < ProofKeepChanges; ++at) {
        const auto change = static_cast<StorageTexture::StateChange>(at);
        if (counts[at] == 0 || StorageTexture::ContentOnly(change) != content) continue;
        total += counts[at];
        char item[96];
        std::snprintf(item, sizeof(item), "%s%s %llu", text.empty() ? "" : ", ", StorageTexture::StateChangeName(change), static_cast<unsigned long long>(counts[at]));
        text += item;
    }
    return text.empty() ? std::string("none") : text;
}

void reportProofKeeps() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!stats) return;
    auto& census = proofKeepCensus;
    const auto presents = Recorder::Presents();
    if (census.checkedPresent.load(std::memory_order_relaxed) == presents) return;
    census.checkedPresent.store(presents, std::memory_order_relaxed);
    const auto tenSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (census.reportedAt.exchange(tenSeconds, std::memory_order_relaxed) == tenSeconds) return;
    const auto interval = presents - census.reportedPresent.exchange(presents, std::memory_order_relaxed);
    const auto take = [](std::atomic<std::uint64_t>& counter) { return counter.exchange(0, std::memory_order_relaxed); };
    const auto kept = take(census.kept);
    const auto keptNew = take(census.keptNew);
    const auto held = take(census.held);
    const auto contentOff = take(census.contentOff);
    const auto contentLater = take(census.contentLater);
    std::array<std::uint64_t, ProofKeepChanges> reproved{};
    for (std::size_t at = 0; at < ProofKeepChanges; ++at) reproved[at] = take(census.reproved[at]);
    const auto changes = StorageTexture::TakeStateChanges();
    std::uint64_t reprovedTotal = 0;
    std::uint64_t contentChanges = 0;
    std::uint64_t proofChanges = 0;
    const auto reprovedText = proofKeepChangesText(reproved, false, reprovedTotal);
    const auto contentText = proofKeepChangesText(changes, true, contentChanges);
    const auto proofText = proofKeepChangesText(changes, false, proofChanges);
    const auto left = contentOff + contentLater + reprovedTotal;
    if (kept == 0 && held == 0 && left == 0 && contentChanges == 0 && proofChanges == 0) return;
    const auto load = [](const std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.load(std::memory_order_relaxed)); };
    AgcDriver::ProfilePrint_nid_no_patch("[proof-keeps] %llu presents: express template tests that found a viewed storage image moved %llu; kept current %llu (reproofs not run %llu), held back for the verify %llu; left to today's reproof %llu: images changed by content alone with the switch off or a trust mode set %llu, by content alone with a later test missing %llu, by the last proof-relevant change of the first such image: %s; storage image changes: content alone %llu (%s), proof-relevant %llu (%s); verify since start %llu checked, %llu differing (reproof failed %llu, other objects %llu, other keys %llu, content version moved %llu), %llu not comparable, %llu reproofs that left no express state\n",
        static_cast<unsigned long long>(interval), static_cast<unsigned long long>(kept + held + left), static_cast<unsigned long long>(kept), static_cast<unsigned long long>(keptNew), static_cast<unsigned long long>(held),
        static_cast<unsigned long long>(left), static_cast<unsigned long long>(contentOff), static_cast<unsigned long long>(contentLater), reprovedText.c_str(),
        static_cast<unsigned long long>(contentChanges), contentText.c_str(), static_cast<unsigned long long>(proofChanges), proofText.c_str(),
        load(census.checked), load(census.differing), load(census.failed), load(census.otherObjects), load(census.otherKeys), load(census.versionMoved), load(census.notComparable), load(census.noExpressState));
}

void noteProofKeepVerify(bool proved, bool sameObjects, bool sameKeys, bool unmoved, bool expressLeft) {
    auto& census = proofKeepCensus;
    const auto checked = census.checked.fetch_add(1, std::memory_order_relaxed) + 1u;
    if (proved && !expressLeft) countProofKeep(census.noExpressState);
    const char* what = !proved ? "the reproof failed" : !sameObjects ? "the reproof holds other objects" : !sameKeys ? "the reproof read other keys" : !unmoved ? "a viewed image's content version moved during the reproof" : nullptr;
    if (what != nullptr) {
        countProofKeep(!proved ? census.failed : !sameObjects ? census.otherObjects : !sameKeys ? census.otherKeys : census.versionMoved);
        const auto differing = census.differing.fetch_add(1, std::memory_order_relaxed) + 1u;
        if (differing <= 16u || differing % 4096u == 0u) std::fprintf(stderr, "[proof-keeps-verify] a template kept current across content-only changes of its viewed storage images differs from its reproof: %s; %llu checked, %llu differing\n", what, static_cast<unsigned long long>(checked), static_cast<unsigned long long>(differing));
    }
    reportProofKeeps();
}

}

ShaderResources::ExpressMiss ShaderResources::ExpressCurrentTrusted() const {
    const auto kept = ExpressCurrentKept();
    if (kept == ExpressMiss::None) NoteCommitterTrusted();
    return kept;
}

ShaderResources::ExpressMiss ShaderResources::ExpressCurrentKept() const {
    if (!expressValid || !cleanProved || expressWatch == nullptr) return ExpressMiss::NotProved;
    if (GuestMemory::CleanForgetSerial() != expressAt.forget) return ExpressMiss::Forget;
    if (HostImportSetGeneration() != expressAt.imports || GuestAllocations::GuestAllocationsGeneration_nid_postfix() != expressAt.allocations || GuestMemory::UnwatchSerial() != expressAt.unwatched || DepthSurfacesSerial() != expressAt.depthSurfaces) return ExpressMiss::Serials;
    return ExpressMiss::None;
}

ShaderResources::ExpressMiss ShaderResources::ExpressCurrentAudited(CommitterTrustCheck& check, bool following) {
    const bool candidate = ExpressCurrentKept() == ExpressMiss::None;
    if (candidate && check.Skip()) return ExpressMiss::None;
    const auto missed = expressCurrent(following);
    if (candidate && missed != ExpressMiss::None) check.Fail();
    return missed;
}

ShaderResources::ExpressMiss ShaderResources::expressCurrent(bool memo) {
    const bool keeps = RuntimeSwitches::Set(RuntimeSwitch::ProofKeeps) && !RuntimeSwitches::Set(RuntimeSwitch::TrustAudit) && !RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter);
    bool kept = false;
    const auto missed = expressTest(memo, keeps, kept);
    if (missed == ExpressMiss::Sources && proofKeepsCensus()) noteProofKeepMiss(keeps);
    if (!kept) return missed;
    if (proofKeepVerify) return ExpressMiss::Sources;
    if (const auto every = proofKeepsVerifyEvery(); every != 0) {
        thread_local std::uint64_t uses = 0;
        if (++uses % every == 0) {
            proofKeepVerify = true;
            if (proofKeepsCensus()) {
                countProofKeep(proofKeepCensus.held);
                reportProofKeeps();
            }
            return ExpressMiss::Sources;
        }
    }
    if (proofKeepsCensus()) noteProofKept();
    return ExpressMiss::None;
}

ShaderResources::ExpressMiss ShaderResources::expressTest(bool memo, bool keeps, bool& kept) {
    kept = false;
    if (!expressValid || !cleanProved || expressWatch == nullptr) return ExpressMiss::NotProved;
    if (GuestMemory::SubscriptionGeneration(*expressWatch) != expressGeneration) return ExpressMiss::Watch;
    if (GuestMemory::CleanForgetSerial() != expressAt.forget) return ExpressMiss::Forget;
    if (HostImportSetGeneration() != expressAt.imports || GuestAllocations::GuestAllocationsGeneration_nid_postfix() != expressAt.allocations || GuestMemory::UnwatchSerial() != expressAt.unwatched || DepthSurfacesSerial() != expressAt.depthSurfaces) return ExpressMiss::Serials;
    bool moved = false;
    for (std::size_t at = 0; at < expressSources.size(); ++at) {
        const auto& [source, epoch] = expressSources[at];
        if (source->StateEpoch() == epoch) continue;
        if (!keeps || at >= expressProofEpochs.size() || source->ProofEpoch() != expressProofEpochs[at]) return ExpressMiss::Sources;
        moved = true;
    }
    const auto rest = expressRest(memo);
    if (moved && rest != ExpressMiss::None) return ExpressMiss::Sources;
    kept = moved;
    return rest;
}

bool ShaderResources::proofKeepHeld() {
    bool kept = false;
    return expressTest(false, true, kept) == ExpressMiss::None && kept;
}

void ShaderResources::noteProofKeepMiss(bool keeps) const {
    auto change = StorageTexture::StateChange::Count;
    for (std::size_t at = 0; at < expressSources.size(); ++at) {
        const auto& [source, epoch] = expressSources[at];
        if (source->StateEpoch() == epoch) continue;
        if (at >= expressProofEpochs.size()) {
            change = StorageTexture::StateChange::Other;
            break;
        }
        if (source->ProofEpoch() != expressProofEpochs[at]) {
            change = source->LastProofChange();
            break;
        }
    }
    auto& census = proofKeepCensus;
    if (change == StorageTexture::StateChange::Count) countProofKeep(keeps ? census.contentLater : census.contentOff);
    else countProofKeep(census.reproved[std::min(static_cast<std::size_t>(change), ProofKeepChanges - 1u)]);
    reportProofKeeps();
}

void ShaderResources::noteProofKept() {
    thread_local std::uint64_t uses = 0;
    std::uint64_t sum = 0;
    for (const auto& entry : expressSources) sum += entry.first->StateEpoch();
    const bool fresh = sum != proofKeptSum;
    if (fresh) {
        proofKeptSum = sum;
        countProofKeep(proofKeepCensus.keptNew);
    }
    if (++uses < 256u && !fresh) return;
    proofKeepCensus.kept.fetch_add(uses, std::memory_order_relaxed);
    uses = 0;
    reportProofKeeps();
}

ShaderResources::ExpressMiss ShaderResources::expressRest(bool memo) {
    if (!memo || expressViewsAt.load(std::memory_order_relaxed) != expressAt.allocations) {
        if (!registeredImageViewsCurrent(registeredImageViews)) return ExpressMiss::Views;
        if (memo) expressViewsAt.store(expressAt.allocations, std::memory_order_relaxed);
    }
    if (!Recorder::InCompletion() && Recorder::Active() != nullptr && ((context.copiedWriters != nullptr && !context.copiedWriters->empty()) || !DrawCopiedWriters()->empty() || Recorder::PendingCompletionLabels() != 0)) return ExpressMiss::PendingCpu;
    if (!guestMemory.StagingDecisionsCurrent()) return ExpressMiss::Staging;
    if (!directRegions.empty()) {
        const auto shadows = memo ? ShadowLiveSerial() + 1u : 0u;
        if (!memo || expressRegionsAt.load(std::memory_order_relaxed) != shadows) {
            struct ExpressRegionsStorage {};
            auto& ranges = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, ExpressRegionsStorage>();
            ranges.clear();
            for (const auto& region : directRegions) ranges.emplace_back(region.begin, region.end);
            if (AnyShadowedOverlaps(ranges)) return ExpressMiss::Regions;
            if (memo) expressRegionsAt.store(shadows, std::memory_order_relaxed);
        }
    }
    return ExpressMiss::None;
}

void ShaderResources::ExpressStage(Recorder& recorder) {
    for (const auto& region : directRegions) static_cast<void>(FlushDepthPlanes(region.begin, static_cast<std::size_t>(region.end - region.begin)));
    guestMemory.RecordStagingCopies(recorder);
}

const char* ShaderResources::ProofRefusalName(ProofRefusal refusal) {
    constexpr std::array<const char*, static_cast<std::size_t>(ProofRefusal::Count)> names{"none", "other", "registered views", "not reusable", "walk: no program", "walk: snapshot texture", "walk: view of a GPU image", "walk: depth surface", "walk: storage image", "walk: image table", "walk: element count", "imports", "staging decisions", "staging copies", "as-is check", "injected"};
    const auto index = static_cast<std::size_t>(refusal);
    return index < names.size() ? names[index] : "unnamed";
}

const char* ShaderResources::FastFailName(std::size_t reason) {
    return reason < FastFailNames.size() ? FastFailNames[reason] : "none";
}

bool ShaderResources::Revalidate(std::span<const CompiledShader> shaders, ProofReport* report, CleanProof* clean) {
    ++wvProofSerial;
    if (!RuntimeSwitches::Set(RuntimeSwitch::ExpressCommits)) {
        if (expressValid || expressWatch != nullptr) {
            expressValid = false;
            expressWatch.reset();
        }
        return revalidateClean(shaders, report, clean);
    }
    const bool keepFlagged = proofKeepVerify;
    const bool keepCheck = keepFlagged && proofKeepHeld();
    if (clean != nullptr && expressValid && !verifyCleanCurrency() && ExpressCurrent() == ExpressMiss::None) {
        bool staged = true;
        if (auto* recorder = Recorder::Active(); recorder != nullptr) {
            try {
                ExpressStage(*recorder);
            } catch (const std::exception&) {
                staged = false;
            }
        }
        if (staged) {
            proofKeepVerify = false;
            noteClean(CleanEvent::Hit);
            if (report != nullptr) *report = {ProofPath::Fast, ProofFailure::None};
            *clean = CleanProof::Skipped;
            return true;
        }
    }
    proofKeepVerify = false;
    if (keepFlagged && !keepCheck) countProofKeep(proofKeepCensus.notComparable);
    std::vector<std::shared_ptr<Texture>> keptTextures;
    std::vector<DccKeys> keptKeys;
    std::vector<std::pair<const StorageTexture*, std::uint64_t>> keptVersions;
    if (keepCheck) {
        keptTextures = textures;
        for (const auto& surface : validatedTextures) {
            keptKeys.push_back(surface.keys);
            if (surface.source != nullptr) keptVersions.emplace_back(surface.source, surface.source->Version());
        }
    }
    const auto before = expressBefore();
    const bool proved = revalidateClean(shaders, report, clean);
    expressAfter(proved, before);
    if (keepCheck) {
        const bool sameKeys = keptKeys.size() == validatedTextures.size() && std::equal(keptKeys.begin(), keptKeys.end(), validatedTextures.begin(), [](DccKeys keys, const ValidatedSurface& surface) { return keys == surface.keys; });
        const bool unmoved = std::none_of(keptVersions.begin(), keptVersions.end(), [](const std::pair<const StorageTexture*, std::uint64_t>& entry) { return entry.first->Version() != entry.second; });
        noteProofKeepVerify(proved, textures == keptTextures, sameKeys, unmoved, expressValid && cleanProved);
    }
    return proved;
}

bool ShaderResources::revalidateClean(std::span<const CompiledShader> shaders, ProofReport* report, CleanProof* clean) {
    if (clean == nullptr && !RuntimeSwitches::Set(RuntimeSwitch::CleanCurrency)) {
        const bool ok = revalidateProof(shaders, report);
        if (!ok) cleanProved = false;
        return ok;
    }
    auto stamp = cleanStampNow();
    auto outcome = CleanProof::FirstProof;
    if (cleanCurrent(stamp, outcome)) {
        if (!verifyCleanCurrency()) {
            noteClean(CleanEvent::Hit);
            if (report != nullptr) *report = {ProofPath::Fast, ProofFailure::None};
            if (clean != nullptr) *clean = outcome;
            return true;
        }
        stamp.generation = GuestMemory::TrackerGeneration();
        const auto rule = cleanHitRule;
        std::vector<std::pair<const StorageTexture*, std::uint64_t>> versions;
        for (const auto& surface : validatedTextures) {
            if (surface.source != nullptr) versions.emplace_back(surface.source, surface.source->Version());
        }
        const auto before = textures;
        const bool ok = revalidateProof(shaders, report);
        const bool moved = std::any_of(versions.begin(), versions.end(), [](const auto& entry) { return entry.first->Version() != entry.second; });
        const bool same = ok && !moved && textures == before;
        noteClean(same ? CleanEvent::Verified : CleanEvent::Differs);
        if ((rule & 1u) != 0) noteCleanRule(same ? CleanRuleEvent::ViewsVerified : CleanRuleEvent::ViewsDiffers);
        if ((rule & 2u) != 0) noteCleanRule(same ? CleanRuleEvent::PendingVerified : CleanRuleEvent::PendingDiffers);
        static std::atomic<int> reports{0};
        if (!same && reports.fetch_add(1, std::memory_order_relaxed) < 20) {
            std::fprintf(stderr, "[clean-currency] verify: a clean-currency hit disagrees with the full proof (proved %d, an upload %d, other objects %d, %zu ranges)\n", ok ? 1 : 0, moved ? 1 : 0, textures == before ? 0 : 1, cleanRangeList.size());
            if (rule != 0) std::fprintf(stderr, "[clean-currency] verify: the disagreeing hit came through the %s rule\n", rule == 1 ? "resident views" : rule == 2 ? "pending ranges" : "resident views and pending ranges");
        }
        if (clean != nullptr) *clean = same ? CleanProof::Verified : CleanProof::Differs;
        recordClean(ok, stamp);
        return ok;
    }
    stamp.generation = GuestMemory::TrackerGeneration();
    const bool ok = revalidateProof(shaders, report);
    recordClean(ok, stamp);
    if (clean != nullptr) *clean = outcome;
    return ok;
}

bool ShaderResources::revalidateProof(std::span<const CompiledShader> shaders, ProofReport* report) {
    if (!registeredImageViewsCurrent(registeredImageViews)) {
        if (report != nullptr) *report = {ProofPath::Full, ProofFailure::Other, false, ProofRefusal::Views};
        provedEpoch = 0;
        return false;
    }
    if (EpochProofs()) {
        const auto epoch = GuestMemory::CollectEpoch();
        const auto proof = GuestMemory::ProofSerial();
        const auto pending = StorageTexture::PendingSerial();
        const auto release = TextureReleaseSerial().load(std::memory_order_acquire);
        const auto retired = RetiredImportSerial();
        const auto ranged = [&] {
            if (provedProof == proof || !GuestMemory::RangeProofs() || provedRelease != release) return false;
            const auto* ranges = rangedProofRanges();
            return ranges != nullptr && GuestMemory::RangeProofHolds(provedProof, *ranges);
        };
        if (epoch != 0 && provedEpoch == epoch && provedRetired == retired && (RuntimeSwitches::Set(RuntimeSwitch::XpStampsByEpoch) || (provedPending == pending && (provedProof == proof || ranged())))) {
            const bool throughRanges = provedProof != proof;
            provedProof = proof;
            for (const auto& texture : textures) {
                if (texture != nullptr && texture->IsDepthCopy()) texture->RefreshDepthCopy();
            }
            if (throughRanges && TemplateProofVerify()) {
                const bool ok = revalidateFull(shaders, report);
                noteTemplateProofCheck(false, !ok);
                if (!ok) provedEpoch = 0;
                return ok;
            }
            static const bool verify = std::getenv("APS5_VERIFY_DRAW_PIPELINE") != nullptr;
            thread_local std::uint64_t reuses = 0;
            if (verify && ++reuses % 64u == 0u) {
                const bool ok = revalidateFull(shaders, report);
                noteEpochProof(true, true, !ok);
                return ok;
            }
            if (report != nullptr) *report = {ProofPath::Full, ProofFailure::None};
            noteEpochProof(true, false, false);
            return true;
        }
        noteEpochMiss(provedEpoch, provedProof, provedPending, epoch, proof, pending);
        const bool ok = revalidateFull(shaders, report);
        noteEpochProof(false, false, false);
        if (ok) {
            provedEpoch = std::any_of(textures.begin(), textures.end(), [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->HasReuseLookup(); }) ? 0 : epoch;
            provedProof = proof;
            provedPending = pending;
            provedRelease = release;
            provedRetired = retired;
        } else {
            provedEpoch = 0;
        }
        return ok;
    }
    return revalidateFull(shaders, report);
}

bool ShaderResources::AsIsCurrent() {
    if (!reusable || !registeredImageViewsCurrent(registeredImageViews) || !guestMemory.Writes().empty() || guestMemory.HasGpuCopies() || std::find(storageWritten.begin(), storageWritten.end(), true) != storageWritten.end()) return false;
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->IsDepthCopy()) texture->RefreshDepthCopy();
    }
    if (importsReconcilable ? HostImportsReconciledUnchanged(context, importsProof) : HostImportsUnchanged(context, importsProof)) return true;
    for (const auto& region : directRegions) {
        auto serial = HostImportSerial(context, region.begin, static_cast<std::size_t>(region.end - region.begin), true);
        if (serial == 0) serial = ImageMirrorSerial(context, region.begin, static_cast<std::size_t>(region.end - region.begin));
        if (serial != region.serial) return false;
        if (region.regionEnd != 0) {
            const auto committed = GuestMemory::DescribeCommitted(region.begin, static_cast<std::size_t>(region.regionEnd - region.begin));
            if (committed.ranges.size() != 1 || committed.ranges.front().first != region.begin || committed.ranges.front().second != region.committedEnd) return false;
        }
    }
    importsProof = HostImportsIdentity(context);
    return true;
}

bool ShaderResources::revalidateFull(std::span<const CompiledShader> shaders, ProofReport* report) {
    if (report != nullptr) *report = {ProofPath::Full, ProofFailure::Other, false, ProofRefusal::NotReusable};
    if (!reusable || shaders.empty()) return false;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool cpuWriters = !GuestMemory::WriteIndexEnabled() || AnyPendingCpuWrites(context);
    for (const auto& region : directRegions) {
        static_cast<void>(FlushDepthPlanes(region.begin, static_cast<std::size_t>(region.end - region.begin)));
        if (cpuWriters) CompletePendingCpuWrites(context, region.begin, static_cast<std::size_t>(region.end - region.begin));
    }
    for (const auto& range : describedRanges) {
        if (!range.imageTable) {
            if (range.storedWords) claimStoredMultisampleMemory(context, DecodeTextureResource(*range.storedWords), range.bytes, range.storedMip);
            RequireNoMultisampleColorTarget(range.address, static_cast<std::size_t>(range.bytes), "a cached image access");
        }
        if (!cpuWriters) continue;
        CompletePendingCpuWrites(context, range.address, static_cast<std::size_t>(range.bytes));
        if (range.dccAddress != 0) CompletePendingCpuWrites(context, range.dccAddress, DccKeyBytes(range.bytes));
    }
    for (const auto& image : storageTextures) {
        if (!cpuWriters) break;
        CompletePendingCpuWrites(context, image->Descriptor().baseAddress, static_cast<std::size_t>(image->GuestBytes()));
        if (image->Descriptor().dccAddress != 0) CompletePendingCpuWrites(context, image->Descriptor().dccAddress, DccKeyBytes(image->GuestBytes()));
    }
    // APS5_NO_FAST_REVALIDATE=1 always repeats the lookups.
    static const bool noFast = std::getenv("APS5_NO_FAST_REVALIDATE") != nullptr;
    auto refusal = ProofRefusal::Other;
    std::uint64_t refusedAddress = 0;
    std::uint64_t refusedBytes = 0;
    const auto refuse = [&](ProofRefusal why, std::uint64_t address = 0, std::uint64_t bytes = 0) {
        refusal = why;
        refusedAddress = address;
        refusedBytes = bytes;
        return false;
    };
    const auto refuseTexture = [&](std::size_t index, std::uint64_t address) {
        if (index >= validatedTextures.size()) return refuse(ProofRefusal::WalkCount, address);
        const auto& surface = validatedTextures[index];
        return refuse(surface.depth ? ProofRefusal::WalkDepth : surface.source != nullptr ? ProofRefusal::WalkView : ProofRefusal::WalkSnapshot, address, surface.bytes);
    };
    const auto finish = [&](bool fast, bool ok, ProofFailure failure = ProofFailure::Other, ProofRefusal why = ProofRefusal::Count) {
        if (profile) countRevalidate(fast, ok, start);
        if (report != nullptr) {
            report->failure = ok ? ProofFailure::None : failure;
            report->refusal = ok ? ProofRefusal::None : why != ProofRefusal::Count ? why : refusal;
            report->address = ok || why != ProofRefusal::Count ? 0 : refusedAddress;
            report->bytes = ok || why != ProofRefusal::Count ? 0 : refusedBytes;
        }
        return ok;
    };
    // Loaded before any check (the memo rule of fastRevalidate): kept as the memo only when the
    // registry did not move through the whole proof, flushes of this call included.
    const auto serialBefore = StorageTexture::PendingSerial();
    // (1) Textures and storage images, from stamps when every element allows it (fastRevalidate),
    // else by the same lookups as the build (which refresh or replace them as guest memory changed):
    // they must hand back the very objects the set's views belong to. The build appended them stage
    // by stage, binding by binding, so the walk repeats that order.
    const auto fullWalk = [&] {
        lastFullWalk = std::chrono::steady_clock::now();
        walkImportGeneration = HostImportSetGeneration();
        lookupLog().clear();
        BuildLookups lookups;
        std::size_t textureIndex = 0;
        std::size_t storageIndex = 0;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) return refuse(ProofRefusal::WalkProgram);
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
                if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
                    const auto elementWords = binding.count != 0 ? DrawGuestWords(shader, binding).size() / binding.count : 0;
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        const auto raw = std::span<const std::uint32_t>(DrawGuestWords(shader, binding)).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
                        std::array<std::uint32_t, 8> narrowed{};
                        const auto words = firstLayerWords(raw, binding.imageShape, narrowed, false);
                        if (IsNullSampledTextureResource(words)) continue;
                        const auto resource = DecodeTextureResource(words);
                        const VkComponentMapping components = ViewComponents(resource);
                        const VkImageViewType multisampleViewType = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2DMultisampleArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
                        const bool depthCompare = !binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element);
                        if (textureIndex >= textures.size() || cachedTexture(context, words, resource, components, 0, depthCompare, multisampleViewType, layerRowsBinding(binding.imageShape, resource, depthCompare)) != textures[textureIndex]) return refuseTexture(textureIndex, resource.baseAddress);
                        ++textureIndex;
                    }
                } else if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        const auto words = std::span<const std::uint32_t>(DrawGuestWords(shader, binding)).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                        if (IsNullTextureResource(words)) continue;
                        if (storageIndex >= storageTextures.size()) return refuse(ProofRefusal::WalkCount);
                        std::shared_ptr<StorageTexture> expected;
                        if (SameAsPreviousStorageElement(DrawGuestWords(shader, binding), element) && StorageDedupeEnabled()) expected = storageTextures[storageIndex - 1];
                        else expected = cachedStorageTexture(context, words, DecodeTextureResource(words), storageMips[storageIndex], 0, storageWritten[storageIndex]);
                        if (expected != storageTextures[storageIndex]) return refuse(ProofRefusal::WalkStorage, storageTextures[storageIndex] != nullptr ? storageTextures[storageIndex]->Descriptor().baseAddress : 0, storageTextures[storageIndex] != nullptr ? storageTextures[storageIndex]->GuestBytes() : 0);
                        ++storageIndex;
                    }
                }
            }
        }
        const auto currentTableEntry = [&](const ShaderRecompiler::DescriptorBinding& images, std::span<const std::uint32_t> imageWords, std::uint32_t element) {
            if (textureIndex >= textures.size()) return false;
            const auto raw = imageWords.subspan(static_cast<std::size_t>(element) * 8u, 8u);
            std::array<std::uint32_t, 8> narrowed{};
            const auto words = firstLayerWords(raw, images.imageShapes.at(element), narrowed, false);
            const auto resource = DecodeTextureResource(words);
            const VkImageViewType multisampleViewType = images.imageShapes.at(element) == ShaderRecompiler::DescriptorImageShape::Image2DMultisampleArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            return cachedTexture(context, words, resource, ViewComponents(resource), 0, images.imageDepthCompare.at(element), multisampleViewType) == textures[textureIndex];
        };
        for (const auto& table : tableStages) {
            if (table.elementTextures.empty()) continue;
            std::span<const std::uint32_t> imageWords;
            const auto* images = tableImages(shaders, table.stage, imageWords);
            if (images == nullptr || images->count != table.elementTextures.size() || imageWords.size() != static_cast<std::size_t>(images->count) * 8u) return refuse(ProofRefusal::WalkTable);
            std::vector<std::shared_ptr<Texture>> cached;
            if (auto* reused = FindReusedTable(context.device, *images, imageWords, CurrentTableState()); reused != nullptr && TableCurrent(*reused) && CachedTableTextures(*reused, cached)) {
                if (!reused->representatives.empty() && reused->representatives != variantRepresentatives(context.device, *images, imageWords, table.elementBase, context.imageTableCapacity)) return refuse(ProofRefusal::WalkTable);
                logTableLookups(*reused, cached);
                auto next = cached.begin();
                bool same = true;
                for (std::uint32_t element = 0; same && element < images->count; ++element) {
                    const auto& entry = reused->elements[element];
                    const auto index = table.elementTextures[element];
                    if (entry.rejected) {
                        same = index == std::numeric_limits<std::size_t>::max();
                        continue;
                    }
                    if (entry.substitute) continue;
                    if (entry.live) {
                        if (index == std::numeric_limits<std::size_t>::max()) continue;
                        same = index == textureIndex && currentTableEntry(*images, imageWords, element);
                        ++textureIndex;
                        continue;
                    }
                    const auto& texture = *next++;
                    if (index == std::numeric_limits<std::size_t>::max()) continue;
                    same = index == textureIndex && textureIndex < textures.size() && textures[textureIndex] == texture;
                    ++textureIndex;
                }
                if (!same) return refuse(ProofRefusal::WalkTable);
                TextureCounts().tablesReused.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (!table.representatives.empty() && table.representatives != variantRepresentatives(context.device, *images, imageWords, table.elementBase, context.imageTableCapacity)) return refuse(ProofRefusal::WalkTable);
            for (std::uint32_t element = 0; element < images->count; ++element) {
                if (table.elementTextures[element] == std::numeric_limits<std::size_t>::max()) continue;
                if (!table.representatives.empty() && table.representatives[element] != NoRepresentative) continue;
                if (table.elementTextures[element] != textureIndex || !currentTableEntry(*images, imageWords, element)) return refuse(ProofRefusal::WalkTable);
                ++textureIndex;
            }
        }
        if (textureIndex != textures.size() || storageIndex != storageTextures.size()) return refuse(ProofRefusal::WalkCount);
        // The walk proved every object current again: the records move to what it proved, or one
        // spurious stamp (a label sharing a 64 KiB block with a surface's edge) would keep this
        // object on the full walk for good.
        captureValidation(spaceReuse);
        return true;
    };
    struct OverlappingTag {};
    auto& overlapping = ThreadScratch<std::vector<PendingOverlap>, OverlappingTag>();
    struct RefreshedTag {};
    auto& refreshed = ThreadScratch<std::vector<PendingOverlap>, RefreshedTag>();
    refreshed.clear();
    FastFail reason = FastFail::Count;
    bool accepted = false;
    bool indexProved = false;
    std::uint64_t indexBaseline = 0;
    if (GuestMemory::WriteIndexEnabled() && !noFast && EpochRevalidate()) {
        if (indexRevalidate(shaders, serialBefore) != IndexOutcome::Fallback) indexProved = true;
        else {
            if (writeSubscription == nullptr) subscribeIndex(shaders);
            if (writeSubscription != nullptr) indexBaseline = GuestMemory::SubscriptionGeneration(*writeSubscription);
        }
    }
    bool fast = indexProved || (!noFast && fastRevalidate(serialBefore, refreshed, reason, overlapping, accepted));
    {
        static const bool verifyUntracked = std::getenv("APS5_VERIFY_UNTRACKED_PROOFS") != nullptr;
        const bool anyUntracked = std::any_of(validatedTextures.begin(), validatedTextures.end(), [](const ValidatedSurface& surface) { return surface.untracked[0] != 0; }) || std::any_of(storageUntracked.begin(), storageUntracked.end(), [](const auto& stamp) { return stamp[0] != 0; });
        if (verifyUntracked && fast && anyUntracked) {
            auto& [checked, disagreed] = UntrackedVerified();
            const auto saved = validatedTextures;
            const auto savedStorage = storageUntracked;
            if (!fullWalk()) disagreed.fetch_add(1, std::memory_order_relaxed);
            validatedTextures = saved;
            storageUntracked = savedStorage;
            if (checked.fetch_add(1, std::memory_order_relaxed) % 100000 == 99999) std::fprintf(stderr, "[verify-untracked] %llu fast proofs over untracked memory checked by the full walk, %llu disagreed%c", static_cast<unsigned long long>(checked.load()), static_cast<unsigned long long>(disagreed.load()), 10);
        }
    }
    // T1 (design_cpu_final M3, rule RT1): a Pending failure whose overlapping images are foreign
    // to the surfaces is resolved by the own objects' refresh (what the walk's lookups would do to
    // them) and the fast proof run again, which is then authoritative (it accepts what stays
    // pending over the refreshed surfaces, see fastRevalidate); a fallback, or a second failure,
    // takes the full walk as before.
    bool ownRefreshed = false;
    if (!fast && !overlapping.empty()) {
        auto fallback = OwnRefreshFallback::Count;
        if (!OwnImageRefreshEnabled()) fallback = OwnRefreshFallback::Disabled;
        else {
            refreshed = overlapping;
            fallback = refreshOwnObjects(shaders, refreshed);
        }
        if (fallback == OwnRefreshFallback::Count) {
            fast = fastRevalidate(serialBefore, refreshed, reason, overlapping, accepted);
            if (fast) ownRefreshed = true;
            else fallback = OwnRefreshFallback::Rerun;
        }
        if (fallback != OwnRefreshFallback::Count) countOwnRefreshFallback(fallback);
    }
    if (fast && TemplateAuditDue(lastFullWalk) && (AuditEveryInterval() || (walkImportGeneration != HostImportSetGeneration() && ImportAuditAdmitted()))) {
        fast = false;
        ownRefreshed = false;
        reason = FastFail::Audit;
    }
    if (report != nullptr) {
        report->path = !fast ? ProofPath::Full : ownRefreshed ? ProofPath::OwnRefreshed : ProofPath::Fast;
        report->audited = !fast && reason == FastFail::Audit;
        report->fastFail = static_cast<std::uint8_t>(fast ? FastFail::Count : reason);
    }
    if (!fast) {
        countFullWalk(reason);
        const auto walkStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const bool walked = fullWalk();
        countFullWalkTime(reason, walkStart);
        if (!walked) {
            const auto failure = [&] {
                switch (reason) {
                    case FastFail::Pending: return ProofFailure::Pending;
                    case FastFail::Evicted: case FastFail::Replaced: return ProofFailure::Evicted;
                    case FastFail::Changed: return ProofFailure::Changed;
                    case FastFail::Keys: case FastFail::ClearedView: case FastFail::StorageKeys: return ProofFailure::Keys;
                    default: return ProofFailure::Other;
                }
            }();
            return finish(false, false, failure);
        }
    } else if (ownRefreshed || accepted) {
        if (profile && ownRefreshed) Revalidations().ownRefreshed.fetch_add(1, std::memory_order_relaxed);
        if (VerifyProofs()) {
            // The walk after the refresh, or beside a proof that accepted a foreign overlap, must
            // find every object in place and current: another object, or an upload (a moved
            // content version), is a decision the proof got wrong.
            struct VersionsTag {};
            auto& versions = ThreadScratch<std::vector<std::pair<const StorageTexture*, std::uint64_t>>, VersionsTag>();
            versions.clear();
            for (const auto& surface : validatedTextures) {
                if (surface.source != nullptr) versions.emplace_back(surface.source, surface.source->Version());
            }
            for (const auto& image : storageTextures) versions.emplace_back(image.get(), image->Version());
            const bool same = fullWalk();
            const auto moved = std::find_if(versions.begin(), versions.end(), [](const auto& entry) { return entry.first->Version() != entry.second; });
            if (!same || moved != versions.end()) {
                std::fprintf(stderr, "[rescache] APS5_VERIFY_PROOFS: the T1 proof (%s) disagrees with the full walk (%s)\n", ownRefreshed ? "own-object refresh" : "accepted overlap", !same ? "another object" : "an upload");
                std::fflush(stderr);
                std::abort();
            }
            if (profile) Revalidations().proofsVerified.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // (2) The imported buffers the set reads in place: results of storage images pending in them go
    // to guest memory first (as an upload does), then each import must still be the one the set was
    // written against. This comes last because the lookups and flushes above can reconcile imports
    // themselves; nothing else touches them between here and the dispatch being recorded. Under the
    // epoch gate the flush runs only for regions one registry scan finds pending images over (none
    // while the serial is the memo's), and the serial loop only while the import table's identity
    // moved since the last proof (a retire bumps its epoch, a registry change its generation).
    const auto serialLoop = [&] {
        for (const auto& region : directRegions) {
            auto serial = HostImportSerial(context, region.begin, static_cast<std::size_t>(region.end - region.begin), true);
            if (serial == 0) serial = ImageMirrorSerial(context, region.begin, static_cast<std::size_t>(region.end - region.begin));
            if (serial != region.serial) return refuse(ProofRefusal::Imports, region.begin, region.end - region.begin);
            if (region.regionEnd != 0) {
                const auto committed = GuestMemory::DescribeCommitted(region.begin, static_cast<std::size_t>(region.regionEnd - region.begin));
                if (committed.ranges.size() != 1 || committed.ranges.front().first != region.begin || committed.ranges.front().second != region.committedEnd) return refuse(ProofRefusal::Imports, region.begin, region.regionEnd - region.begin);
            }
        }
        return true;
    };
    if (EpochRevalidate()) {
        if ((!directRegions.empty() || !guestMemory.Writes().empty()) && !(pendingSerialSeen != 0 && pendingSerialSeen == StorageTexture::PendingSerial())) {
            struct RegionsTag {};
            auto& regions = ThreadScratch<std::vector<StorageTexture::PendingQuery>, RegionsTag>();
            regions.clear();
            for (const auto& region : directRegions) regions.push_back({region.begin, region.end, nullptr, nullptr, false});
            for (const auto& [begin, end] : guestMemory.Writes()) regions.push_back({begin, end, nullptr, nullptr, false});
            StorageTexture::ScanPending(regions);
            // A region over a unit shadow's fresh results (a retile bumped the serial) is
            // published by the flush, as one over a pending image is stored.
            for (const auto& region : regions) {
                if (region.overlaps || AnyShadowedOverlaps(region.begin, static_cast<std::size_t>(region.end - region.begin))) StorageTexture::FlushPending(region.begin, static_cast<std::size_t>(region.end - region.begin), nullptr, "imported buffer region");
            }
        }
        if (importsReconcilable ? HostImportsReconciledUnchanged(context, importsProof) : HostImportsUnchanged(context, importsProof)) {
            if (profile) Revalidations().importSkips.fetch_add(1, std::memory_order_relaxed);
        } else {
            if (!serialLoop()) return finish(fast, false, ProofFailure::Imports);
            importsProof = HostImportsIdentity(context);
        }
    } else {
        for (const auto& region : directRegions) StorageTexture::FlushPending(region.begin, static_cast<std::size_t>(region.end - region.begin), nullptr, "imported buffer region");
        for (const auto& [begin, end] : guestMemory.Writes()) StorageTexture::FlushPending(begin, static_cast<std::size_t>(end - begin), nullptr, "written buffer range");
        if (!serialLoop()) return finish(fast, false, ProofFailure::Imports);
    }
    // (3) Buffers staged in device memory (GuestBufferMemory::AllowDeviceStaging) are copied in
    // from their imports anew for this use, after the flushes above and before the work is
    // recorded; a failure to record leaves the object unusable for this dispatch, not the batch.
    if (!guestMemory.StagingDecisionsCurrent()) return finish(fast, false, ProofFailure::Changed, ProofRefusal::Staging);
    if (auto* recorder = Recorder::Active(); recorder != nullptr) {
        try {
            guestMemory.RecordStagingCopies(*recorder);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[resources] staging copies of a reused build failed: %s\n", error.what());
            return finish(fast, false, ProofFailure::Other, ProofRefusal::StagingCopies);
        }
    }
    pendingSerialSeen = EpochRevalidate() && StorageTexture::PendingSerial() == serialBefore ? serialBefore : 0;
    if (indexBaseline != 0) {
        noteIndexProof(indexBaseline, serialBefore);
        for (auto& [address, copy] : comparedTableSpans) {
            if (GuestMemory::Accessible(reinterpret_cast<const void*>(address), copy.size())) std::memcpy(copy.data(), reinterpret_cast<const void*>(address), copy.size());
            else copy.clear();
        }
        comparedTableSpans.erase(std::remove_if(comparedTableSpans.begin(), comparedTableSpans.end(), [](const auto& span) { return span.second.empty(); }), comparedTableSpans.end());
    }
    if (writeSubscription != nullptr) {
        for (const auto index : depthCopyTextures) {
            if (textures[index]->HasReuseLookup()) {
                if (!textures[index]->ReuseLookupHolds()) return finish(fast, false, ProofFailure::Changed);
            } else {
                textures[index]->RefreshDepthCopy();
            }
        }
    } else {
        for (const auto& texture : textures) {
            if (texture == nullptr) continue;
            if (texture->HasReuseLookup()) {
                if (!texture->ReuseLookupHolds()) return finish(fast, false, ProofFailure::Changed);
            } else if (texture->IsDepthCopy()) {
                texture->RefreshDepthCopy();
            }
        }
    }
    return finish(fast, true);
}

namespace {

// APS5_PROFILE_DRAW: what makes content keys miss. On a miss the key is compared with the last key
// seen for the same shader variant(s) and the first differing word is charged to its binding's role
// (and to its index within the V# or T# element), printed as [rescache] every 10 s: it says whether
// V# bases of ring allocations, SRT words or image descriptors churn.
struct KeyMissWindow {
    std::uint64_t compared = 0;
    std::uint64_t noCandidate = 0;
    std::uint64_t unlabelled = 0;
    std::uint64_t identical = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(ContentKeyComponent::Count)> differing{};
};

struct ChurnCounts {
    std::array<std::uint64_t, 8> roles{};
    std::array<std::uint64_t, 4> bufferWords{};
    std::array<std::uint64_t, 8> imageWords{};
    std::uint64_t firstSeen = 0;
    std::uint64_t sameKey = 0;
    std::uint64_t layout = 0;
    std::uint64_t drawWords = 0;
    KeyMissWindow window;
};

// Dispatch keys and draw keys are counted apart: what churns a draw's key decides the draw steps.
struct ChurnProfile {
    AgcDriver::Mutex mutex;
    std::unordered_map<std::uint64_t, ResourceCache::Key> lastByVariant;
    ChurnCounts dispatch;
    ChurnCounts draws;
    std::unordered_map<std::uint64_t, std::uint32_t> tracedByVariant;
    std::size_t tracedLines = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

constexpr std::size_t MaxTracedLines = 256;
constexpr std::uint32_t MaxTracedLinesPerVariant = 4;

const char* componentName(std::size_t component) {
    constexpr std::array<const char*, static_cast<std::size_t>(ContentKeyComponent::Count)> names{"layout", "V# word 0", "V# word 1", "V# word 2", "V# word 3", "image", "sampler", "shader-data", "other", "draw target/index"};
    return names[component];
}

void traceMiss(ChurnProfile& churn, bool draw, std::uint64_t variants, const ContentKeyDiff* parts);

ChurnProfile& Churn() {
    static ChurnProfile profile;
    return profile;
}

// Charges word `diff` of `key` to the binding of the ContentKey that starts at `at` (the layout of
// ShaderResources::ContentKey), moving `at` past that key; false when `diff` lies beyond it.
bool chargeShaderKey(ChurnCounts& profile, const ResourceCache::Key& key, std::size_t& at, std::size_t end, std::size_t diff) {
    if (at + 5 > end || diff < at + 5) {
        ++profile.layout;
        return true;
    }
    const bool dataWords = key[at] != 0;
    const auto bindings = key[at + 4];
    at += 5;
    for (std::uint32_t binding = 0; binding < bindings; ++binding) {
        if (at + 8 > end || diff < at + 8) {
            ++profile.layout;
            return true;
        }
        const auto kind = key[at];
        const auto role = key[at + 1];
        const auto descriptorWords = dataWords || !DataRole(static_cast<ShaderRecompiler::DescriptorRole>(role)) ? key[at + 7] : 0u;
        at += 8;
        if (diff < at + descriptorWords) {
            if (role < profile.roles.size()) ++profile.roles[role];
            const auto index = diff - at;
            if (role == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorRole::GuestBuffers)) ++profile.bufferWords[index % 4];
            else if (role == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorRole::GuestImages) && (kind == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorKind::SampledImage) || kind == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorKind::StorageImage))) ++profile.imageWords[index % 8];
            return true;
        }
        at += descriptorWords;
        for (int flags = 0; flags < 10; ++flags) {
            if (at >= end) {
                ++profile.layout;
                return true;
            }
            const auto words = 1 + (static_cast<std::size_t>(key[at]) + 31) / 32;
            if (diff < at + words) {
                ++profile.layout;
                return true;
            }
            at += words;
        }
        for (const auto listRole : {ShaderRecompiler::DescriptorRole::GuestSamplers, ShaderRecompiler::DescriptorRole::GuestImages}) {
            if (at >= end) {
                ++profile.layout;
                return true;
            }
            const auto words = 1 + static_cast<std::size_t>(key[at]);
            if (diff < at + words) {
                ++profile.roles[static_cast<std::size_t>(listRole)];
                return true;
            }
            at += words;
        }
    }
    return false;
}

}

std::shared_ptr<ShaderResources> ResourceCache::Find(const Key& key) {
    resourceCacheFinds.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(mutex);
    const auto found = index.find(key);
    if (found == index.end()) {
        noteReturn(key);
        noteMiss(key);
        return nullptr;
    }
    entries.splice(entries.begin(), entries, found->second);
    noteUse(key);
    return found->second->second;
}

void ResourceCache::noteEvicted(const Key& key) {
    if (fixedCapacity) return;
    const auto hash = KeyHash{}(key);
    if (const auto found = evictedKeys.find(hash); found != evictedKeys.end()) evictedOrder.erase(found->second);
    evictedOrder.push_front(hash);
    evictedKeys[hash] = evictedOrder.begin();
    while (evictedOrder.size() > capacity) {
        evictedKeys.erase(evictedOrder.back());
        evictedOrder.pop_back();
    }
    if (++evictionsSinceCheck < capacity / 4u) return;
    const bool churning = returnsSinceCheck * 2u > evictionsSinceCheck;
    evictionsSinceCheck = 0;
    returnsSinceCheck = 0;
    if (!churning || capacity >= MaximumEntries) return;
    const auto previous = capacity;
    capacity = std::min(capacity * 2u, MaximumEntries);
    if (shrankLast) shrinkWindow = std::min(shrinkWindow * 2u, MaximumShrinkWindow);
    shrankLast = false;
    usedKeys.clear();
    usesSinceCheck = 0;
    std::fprintf(stderr, "[rescache] capacity %zu -> %zu entries: keys it evicted keep coming back\n", previous, capacity);
}

void ResourceCache::noteUse(const Key& key) {
    if (fixedCapacity || capacity <= baseEntries) return;
    usedKeys.insert(KeyHash{}(key));
    if (++usesSinceCheck < capacity * shrinkWindow) return;
    const auto used = usedKeys.size();
    const auto uses = usesSinceCheck;
    usedKeys.clear();
    usesSinceCheck = 0;
    if (used * 4u > capacity) return;
    const auto previous = capacity;
    capacity = std::max(capacity / 2u, baseEntries);
    shrankLast = true;
    evictionsSinceCheck = 0;
    returnsSinceCheck = 0;
    std::fprintf(stderr, "[rescache] capacity %zu -> %zu entries: %zu keys served the last %zu uses\n", previous, capacity, used, uses);
}

void ResourceCache::noteReturn(const Key& key) {
    if (fixedCapacity) return;
    const auto found = evictedKeys.find(KeyHash{}(key));
    if (found == evictedKeys.end()) return;
    evictedOrder.erase(found->second);
    evictedKeys.erase(found);
    ++returnsSinceCheck;
}

namespace {

void traceMiss(ChurnProfile& churn, bool draw, std::uint64_t variants, const ContentKeyDiff* parts) {
    if (churn.tracedLines >= MaxTracedLines || ++churn.tracedByVariant[variants] > MaxTracedLinesPerVariant) return;
    ++churn.tracedLines;
    std::string changed = parts == nullptr ? "first-seen" : "";
    for (std::size_t i = 0; parts != nullptr && i < parts->size(); ++i) {
        if (!(*parts)[i]) continue;
        if (!changed.empty()) changed += ", ";
        changed += componentName(i);
    }
    if (changed.empty()) changed = "no-component-change";
    const auto hashed = (variants ^ 0x9e3779b97f4a7c15ull) * 0xff51afd7ed558ccdull;
    AgcDriver::ProfilePrint_nid_no_patch("[keymiss] resource-content %s shader set %016llx: %s\n", draw ? "draw" : "dispatch", static_cast<unsigned long long>(hashed ^ (hashed >> 33u)), changed.c_str());
}

}

void ResourceCache::noteMiss(const Key& key) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static const bool trace = std::getenv("APS5_TRACE_RESOURCE_KEY_MISSES") != nullptr;
    if ((!profile && !trace) || key.size() < 5) return;
    // The variant(s) the key belongs to: a compute key starts with the data-words flag, the stage
    // and its variant, a draw key with a marker, the device and its stages' keys (Draw.cpp
    // DrawResourceKey).
    const bool draw = key[0] == 0xffffffffu;
    std::uint64_t variants = 0;
    if (draw) {
        std::size_t at = 4;
        for (std::uint32_t stage = 0; stage < key[3] && at + 4 < key.size(); ++stage) {
            variants = variants * 1000003ull ^ (key[at + 3] | (static_cast<std::uint64_t>(key[at + 4]) << 32u));
            at += 1 + key[at];
        }
    } else {
        variants = key[2] | (static_cast<std::uint64_t>(key[3]) << 32u);
    }
    auto& churn = Churn();
    std::lock_guard lock(churn.mutex);
    auto& counts = draw ? churn.draws : churn.dispatch;
    const auto found = churn.lastByVariant.find(variants);
    if (found == churn.lastByVariant.end()) {
        ++counts.firstSeen;
        ++counts.window.noCandidate;
        if (trace) traceMiss(churn, draw, variants, nullptr);
        churn.lastByVariant.emplace(variants, key);
    } else {
        const auto& previous = found->second;
        const auto common = std::min(key.size(), previous.size());
        std::size_t diff = 0;
        while (diff < common && key[diff] == previous[diff]) ++diff;
        if (diff == common && key.size() == previous.size()) {
            ++counts.sameKey;
        } else if (draw) {
            std::size_t at = 4;
            bool charged = diff < at;
            if (charged) ++counts.layout;
            for (std::uint32_t stage = 0; stage < key[3] && at < key.size() && !charged; ++stage) {
                const auto end = std::min(key.size(), at + 1 + static_cast<std::size_t>(key[at]));
                if (diff == at) {
                    ++counts.layout;
                    charged = true;
                } else if (diff < end) {
                    ++at;
                    if (!chargeShaderKey(counts, key, at, end, diff)) ++counts.layout;
                    charged = true;
                }
                at = end;
            }
            if (!charged) ++counts.drawWords;
        } else {
            std::size_t at = 0;
            if (!chargeShaderKey(counts, key, at, key.size(), diff)) ++counts.layout;
        }
        if (trace) {
            ContentKeyDiff parts;
            ++counts.window.compared;
            if (!ShaderResources::DiffContentKeys(previous, key, draw, parts)) ++counts.window.unlabelled;
            bool anyPart = false;
            for (std::size_t i = 0; i < parts.size(); ++i) {
                if (!parts[i]) continue;
                ++counts.window.differing[i];
                anyPart = true;
            }
            if (!anyPart) ++counts.window.identical;
            traceMiss(churn, draw, variants, &parts);
        }
        found->second = key;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - churn.lastReport < std::chrono::seconds(10)) return;
    churn.lastReport = now;
    const auto report = [](const char* what, const ChurnCounts& churn) {
        std::string line = std::string("[rescache] miss churn (") + what + "), first differing word by role:";
        char item[256];
        for (std::size_t role = 0; role < churn.roles.size(); ++role) {
            if (churn.roles[role] == 0) continue;
            std::snprintf(item, sizeof(item), " %s %llu", roleName(static_cast<ShaderRecompiler::DescriptorRole>(role)), static_cast<unsigned long long>(churn.roles[role]));
            line += item;
            if (role == static_cast<std::size_t>(ShaderRecompiler::DescriptorRole::GuestBuffers)) {
                std::snprintf(item, sizeof(item), " (V# word %llu/%llu/%llu/%llu)", static_cast<unsigned long long>(churn.bufferWords[0]), static_cast<unsigned long long>(churn.bufferWords[1]), static_cast<unsigned long long>(churn.bufferWords[2]), static_cast<unsigned long long>(churn.bufferWords[3]));
                line += item;
            } else if (role == static_cast<std::size_t>(ShaderRecompiler::DescriptorRole::GuestImages)) {
                std::snprintf(item, sizeof(item), " (T# word %llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu)", static_cast<unsigned long long>(churn.imageWords[0]), static_cast<unsigned long long>(churn.imageWords[1]), static_cast<unsigned long long>(churn.imageWords[2]), static_cast<unsigned long long>(churn.imageWords[3]), static_cast<unsigned long long>(churn.imageWords[4]), static_cast<unsigned long long>(churn.imageWords[5]), static_cast<unsigned long long>(churn.imageWords[6]), static_cast<unsigned long long>(churn.imageWords[7]));
                line += item;
            }
        }
        std::snprintf(item, sizeof(item), "; layout %llu, draw target/index %llu, same key %llu (not inserted or evicted), first seen %llu", static_cast<unsigned long long>(churn.layout), static_cast<unsigned long long>(churn.drawWords), static_cast<unsigned long long>(churn.sameKey), static_cast<unsigned long long>(churn.firstSeen));
        line += item;
        AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    };
    if (profile) {
        report("dispatch", churn.dispatch);
        report("draws", churn.draws);
    }
    const auto keyMiss = [](const char* what, KeyMissWindow& window) {
        const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
        const auto part = [&](ContentKeyComponent component) { return count(window.differing[static_cast<std::size_t>(component)]); };
        AgcDriver::ProfilePrint_nid_no_patch("[keymiss] resource-content %s (10 s): misses %llu; differing: layout %llu, V# word0 %llu, V# word1 %llu, V# word2 %llu, V# word3 %llu, image %llu, sampler %llu, shader-data %llu, other %llu, draw target/index %llu; no-candidate %llu; identical %llu; not comparable %llu\n", what, count(window.compared + window.noCandidate), part(ContentKeyComponent::Layout), part(ContentKeyComponent::BufferWord0), part(ContentKeyComponent::BufferWord1), part(ContentKeyComponent::BufferWord2), part(ContentKeyComponent::BufferWord3), part(ContentKeyComponent::Image), part(ContentKeyComponent::Sampler), part(ContentKeyComponent::ShaderData), part(ContentKeyComponent::Other), part(ContentKeyComponent::DrawTarget), count(window.noCandidate), count(window.identical), count(window.unlabelled));
        window = KeyMissWindow{};
    };
    if (trace) {
        keyMiss("dispatch", churn.dispatch.window);
        keyMiss("draw", churn.draws.window);
    }
}

bool ResourceCache::Insert(const Key& key, std::shared_ptr<ShaderResources> resources, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    std::lock_guard lock(mutex);
    dropReleasedTextures(evicted);
    if (!resources->HasCachedStorageImages() || resources->HoldsReleasedTexture()) {
        NoteFate(key, Fate::InsertRefused);
        return false;
    }
    bool replaced = false;
    if (const auto found = index.find(key); found != index.end()) {
        if (evicted != nullptr) evicted->push_back(std::move(found->second->second));
        entries.erase(found->second);
        index.erase(found);
        replaced = true;
    }
    entries.emplace_front(key, std::move(resources));
    index.emplace(key, entries.begin());
    NoteFate(key, Fate::Inserted);
    if (capacity == 0) {
        const char* value = std::getenv("APS5_RESOURCE_CACHE_ENTRIES");
        const auto parsed = value ? std::strtoull(value, nullptr, 10) : 0ull;
        fixedCapacity = parsed != 0;
        capacity = fixedCapacity ? static_cast<std::size_t>(parsed) : baseEntries;
    }
    lazyTouch.store(fixedCapacity || capacity <= baseEntries, std::memory_order_relaxed);
    noteUse(key);
    std::size_t spared = 0;
    while (entries.size() > capacity) {
        if (spared < entries.size() && entries.back().second->TakeCacheUse()) {
            entries.splice(entries.begin(), entries, std::prev(entries.end()));
            ++spared;
            continue;
        }
        NoteFate(entries.back().first, Fate::Capacity);
        noteEvicted(entries.back().first);
        if (evicted != nullptr) evicted->push_back(std::move(entries.back().second));
        index.erase(entries.back().first);
        entries.pop_back();
    }
    if (entries.front().second->TableElements() == 0) return replaced;
    std::uint64_t tables = 0;
    for (const auto& entry : entries) tables += entry.second->TableElements();
    for (auto it = std::prev(entries.end()); tables > TableElementBudget() && it != entries.begin();) {
        const auto elements = it->second->TableElements();
        if (elements == 0 || it->second->TakeCacheUse()) {
            --it;
            continue;
        }
        tables -= elements;
        if (evicted != nullptr) evicted->push_back(std::move(it->second));
        NoteFate(it->first, Fate::Tables);
        index.erase(it->first);
        it = std::prev(entries.erase(it));
    }
    return replaced;
}

void ResourceCache::Remove(const Key& key, const ShaderResources* object) {
    std::lock_guard lock(mutex);
    if (object != nullptr) {
        const auto found = index.find(key);
        if (found == index.end() || found->second->second.get() != object) return;
    }
    erase(key);
}

void ResourceCache::RemoveImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    std::lock_guard lock(mutex);
    removeImage(image, evicted);
}

bool ResourceCache::TryRemoveImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    std::unique_lock lock(mutex, std::try_to_lock);
    if (!lock.owns_lock()) return false;
    removeImage(image, evicted);
    return true;
}

void ResourceCache::removeImage(const StorageTexture* image, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    for (auto it = entries.begin(); it != entries.end();) {
        if (!it->second->ReadsImage(image)) {
            ++it;
            continue;
        }
        if (evicted != nullptr) evicted->push_back(std::move(it->second));
        NoteFate(it->first, Fate::Image);
        index.erase(it->first);
        it = entries.erase(it);
    }
}

void ResourceCache::dropReleasedTextures(std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    if (!TexturesReleased().exchange(false, std::memory_order_acq_rel)) return;
    for (auto it = entries.begin(); it != entries.end();) {
        if (!it->second->HoldsDiscardedTexture()) {
            ++it;
            continue;
        }
        if (evicted != nullptr) evicted->push_back(std::move(it->second));
        NoteFate(it->first, Fate::ReleasedTexture);
        index.erase(it->first);
        it = entries.erase(it);
    }
}

bool ResourceCache::Touch(const Key& key) {
    resourceCacheTouches.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(mutex);
    const auto found = index.find(key);
    if (found == index.end()) return false;
    entries.splice(entries.begin(), entries, found->second);
    noteUse(key);
    return true;
}

bool ResourceCache::TouchObject(const Key& key, const ShaderResources& object) {
    if (!lazyTouch.load(std::memory_order_relaxed)) return Touch(key);
    resourceCacheTouches.fetch_add(1, std::memory_order_relaxed);
    object.NoteCacheUse();
    return true;
}

std::uint64_t ResourceCache::Finds() {
    return resourceCacheFinds.load(std::memory_order_relaxed);
}

std::uint64_t ResourceCache::Touches() {
    return resourceCacheTouches.load(std::memory_order_relaxed);
}

void ResourceCache::Clear() {
    std::lock_guard lock(mutex);
    index.clear();
    entries.clear();
    evictedKeys.clear();
    evictedOrder.clear();
    evictionsSinceCheck = 0;
    returnsSinceCheck = 0;
    usedKeys.clear();
    usesSinceCheck = 0;
    shrinkWindow = baseShrinkWindow;
    shrankLast = false;
    if (!fixedCapacity) capacity = 0;
}

void ResourceCache::SetBase(std::size_t entries, std::size_t window) {
    std::lock_guard lock(mutex);
    baseEntries = entries;
    baseShrinkWindow = window;
}

std::size_t ResourceCache::Size() const {
    std::lock_guard lock(mutex);
    return entries.size();
}

std::size_t ResourceCache::Capacity() const {
    std::lock_guard lock(mutex);
    return capacity == 0 ? baseEntries : capacity;
}

std::uint64_t ResourceCache::TableElementBudget() {
    static const std::uint64_t elements = [] {
        const char* value = std::getenv("APS5_RESOURCE_CACHE_TABLE_ELEMENTS");
        const auto parsed = value ? std::strtoull(value, nullptr, 10) : 65536ull;
        return static_cast<std::uint64_t>(parsed != 0 ? parsed : 65536ull);
    }();
    return elements;
}

std::uint64_t ResourceCache::TableElements() const {
    std::lock_guard lock(mutex);
    std::uint64_t tables = 0;
    for (const auto& entry : entries) tables += entry.second->TableElements();
    return tables;
}

void ResourceCache::erase(const Key& key) {
    const auto found = index.find(key);
    if (found == index.end()) return;
    entries.erase(found->second);
    index.erase(found);
}

ResourceCache& SharedResourceCache() {
    static auto* cache = new ResourceCache();
    return *cache;
}

namespace {
struct FateTable {
    AgcDriver::Mutex mutex;
    std::unordered_map<std::size_t, ResourceCache::FateNote> notes;
};

FateTable& fateTable() {
    static auto* table = new FateTable();
    return *table;
}
}

bool ResourceCache::FateCensus() {
    static const bool census = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return census;
}

void ResourceCache::NoteFate(const Key& key, Fate fate, std::uint8_t detail) {
    if (!FateCensus() || key.empty() || key[0] != 0xffffffffu) return;
    const auto hash = KeyHash{}(key);
    const auto present = Recorder::Presents();
    auto& table = fateTable();
    std::lock_guard lock(table.mutex);
    if (table.notes.size() >= 65536u) std::erase_if(table.notes, [&](const auto& item) { return item.second.present + 600u < present; });
    if (table.notes.size() >= 65536u) table.notes.clear();
    table.notes[hash] = {fate, detail, present};
}

ResourceCache::FateNote ResourceCache::FateOf(const Key& key) {
    if (!FateCensus() || key.empty()) return {};
    const auto hash = KeyHash{}(key);
    auto& table = fateTable();
    std::lock_guard lock(table.mutex);
    const auto found = table.notes.find(hash);
    return found != table.notes.end() ? found->second : FateNote{};
}

namespace {
std::size_t KeptSets() {
    static const std::size_t count = [] {
        const char* text = std::getenv("APS5_KEPT_DESCRIPTOR_SETS");
        return text != nullptr ? static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) : std::size_t{8};
    }();
    return count;
}

}

DescriptorCache::DescriptorCache(const Context& context) : context(context), destroyLayout(context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")), destroyPool(context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")), resetPool(context.Function<PFN_vkResetDescriptorPool>("vkResetDescriptorPool")), freeDescriptorSets(context.individuallyFreeDescriptorSets ? context.Function<PFN_vkFreeDescriptorSets>("vkFreeDescriptorSets") : nullptr) {
    kept.reserve(KeptSets() + 1u);
}

DescriptorCache::~DescriptorCache() {
    for (const auto& pool : pools) destroyPool(context.device, pool.handle, nullptr);
    for (const auto& pool : updateAfterBindPools) destroyPool(context.device, pool.handle, nullptr);
    for (const auto pool : dedicated) destroyPool(context.device, pool, nullptr);
    for (const auto& set : kept) destroyPool(context.device, set.pool, nullptr);
    for (const auto& [key, layout] : layouts) destroyLayout(context.device, layout, nullptr);
}

VkDescriptorSetLayout DescriptorCache::Layout(std::span<const std::uint32_t> key, std::span<const VkDescriptorSetLayoutBinding> bindings, bool updateAfterBind, std::span<const VkDescriptorBindingFlags> flags, bool push) {
    std::vector<VkDescriptorBindingFlags> defaults;
    if (flags.empty()) {
        for (const auto& binding : bindings) defaults.push_back(updateAfterBind && binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0);
        flags = defaults;
    }
    Require(flags.size() == bindings.size(), "descriptor binding flag count differs from layout");
    std::lock_guard lock(mutex);
    std::vector<std::uint32_t> keyCopy(key.begin(), key.end());
    keyCopy.push_back(static_cast<std::uint32_t>(flags.size()));
    keyCopy.insert(keyCopy.end(), flags.begin(), flags.end());
    if (push) keyCopy.push_back(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
    if (const auto found = layouts.find(keyCopy); found != layouts.end()) {
        ++stats.layoutHits;
        return found->second;
    }
    ValidateDescriptorLayout(context, bindings, updateAfterBind);
    ++stats.layoutMisses;
    const auto layout = CreateSetLayout(context, bindings, updateAfterBind, flags, push);
    try {
        layouts.emplace(std::move(keyCopy), layout);
    } catch (...) {
        destroyLayout(context.device, layout, nullptr);
        throw;
    }
    return layout;
}

namespace {
std::uint32_t SetDescriptorLimit(const Context& context, VkDescriptorType type, bool updateAfterBind) {
    const auto& limits = context.limits;
    const auto& indexed = context.descriptorIndexingLimits;
    const bool sampledUpdateAfterBind = updateAfterBind && context.sampledImageUpdateAfterBind;
    switch (type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return updateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindStorageBuffers : limits.maxDescriptorSetStorageBuffers;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: return limits.maxDescriptorSetStorageBuffersDynamic;
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: return sampledUpdateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSampledImages : limits.maxDescriptorSetSampledImages;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return limits.maxDescriptorSetStorageImages;
        case VK_DESCRIPTOR_TYPE_SAMPLER: return sampledUpdateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSamplers : limits.maxDescriptorSetSamplers;
        default: throw std::runtime_error("AGC graphics: descriptor set uses an unsupported descriptor type " + std::to_string(type));
    }
}

// What one chain pool holds; a set needing more of any type gets a dedicated pool.
constexpr std::uint32_t ChainPoolSets = 1024;
constexpr std::array<VkDescriptorPoolSize, 4> ChainPoolSizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 512}}};
constexpr std::array<VkDescriptorPoolSize, 4> ChainPoolCreateSizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096 + 2}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 512}}};
}

DescriptorCache::SetAllocation DescriptorCache::Allocate(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind, std::optional<std::uint32_t> variableCount) {
    bool fits = true;
    for (const auto& size : sizes) {
        const auto limit = SetDescriptorLimit(context, size.type, updateAfterBind);
        Require(size.descriptorCount <= limit, "descriptor set exceeds the device's per-set descriptor limit");
        const auto capacity = std::find_if(ChainPoolSizes.begin(), ChainPoolSizes.end(), [&](const auto& item) { return item.type == size.type; });
        if (capacity == ChainPoolSizes.end() || size.descriptorCount > capacity->descriptorCount) fits = false;
    }
    std::lock_guard lock(mutex);
    const auto allocate = context.Resolved(&DeviceFunctions::allocateDescriptorSets, "vkAllocateDescriptorSets");
    const std::uint32_t variable = variableCount.value_or(0u);
    VkDescriptorSetVariableDescriptorCountAllocateInfoEXT variableInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO_EXT};
    variableInfo.descriptorSetCount = 1;
    variableInfo.pDescriptorCounts = &variable;
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.pNext = variableCount.has_value() ? &variableInfo : nullptr;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &layout;
    if (!fits) {
        dedicated.reserve(dedicated.size() + 1);
        const auto same = std::find_if(kept.begin(), kept.end(), [&](const SetAllocation& set) {
            return set.layout == layout && set.updateAfterBind == updateAfterBind && set.variableCount == variableCount && std::equal(set.sizes.begin(), set.sizes.end(), sizes.begin(), sizes.end(), [](const VkDescriptorPoolSize& a, const VkDescriptorPoolSize& b) { return a.type == b.type && a.descriptorCount == b.descriptorCount; });
        });
        if (same != kept.end()) {
            auto reused = std::move(*same);
            kept.erase(same);
            dedicated.push_back(reused.pool);
            ++stats.reusedSets;
            ++stats.sets;
            return reused;
        }
        const auto pool = CreateSetPool(context, sizes, updateAfterBind);
        allocation.descriptorPool = pool;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const auto result = allocate(context.device, &allocation, &set);
        if (result != VK_SUCCESS) {
            destroyPool(context.device, pool, nullptr);
            Check(result, "vkAllocateDescriptorSets");
        }
        dedicated.push_back(pool);
        ++stats.dedicatedPools;
        ++stats.sets;
        return {set, pool, 0, updateAfterBind, layout, std::vector<VkDescriptorPoolSize>(sizes.begin(), sizes.end()), variableCount, {}};
    }
    auto& chain = updateAfterBind ? updateAfterBindPools : pools;
    for (std::size_t index = chain.size(); index != 0; --index) {
        auto& pool = chain[index - 1];
        if (pool.exhausted) continue;
        allocation.descriptorPool = pool.handle;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const auto result = allocate(context.device, &allocation, &set);
        if (result == VK_SUCCESS) {
            ++stats.sets;
            ++pool.liveSets;
            return {set, pool.handle, index - 1, updateAfterBind, VK_NULL_HANDLE, {}, variableCount, {}};
        }
        if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL) Check(result, "vkAllocateDescriptorSets");
        pool.exhausted = true;
    }
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = updateAfterBind ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0;
    if (context.individuallyFreeDescriptorSets) poolInfo.flags |= VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = ChainPoolSets;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(ChainPoolCreateSizes.size());
    poolInfo.pPoolSizes = ChainPoolCreateSizes.data();
    chain.reserve(chain.size() + 1);
    VkDescriptorPool pool = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
    chain.push_back({pool});
    ++stats.pools;
    allocation.descriptorPool = pool;
    VkDescriptorSet set = VK_NULL_HANDLE;
    Check(allocate(context.device, &allocation, &set), "vkAllocateDescriptorSets");
    chain.back().liveSets = 1;
    ++stats.sets;
    return {set, pool, chain.size() - 1, updateAfterBind, VK_NULL_HANDLE, {}, variableCount, {}};
}

DescriptorCache::SetAllocation DescriptorCache::AllocateRewritten(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes, bool updateAfterBind, std::optional<std::uint32_t> variableCount) {
    {
        std::lock_guard lock(mutex);
        if (const auto found = recycled.find({layout, variableCount.value_or(0u)}); found != recycled.end() && !found->second.empty()) {
            const auto allocation = found->second.back();
            found->second.pop_back();
            return allocation;
        }
    }
    return Allocate(layout, sizes, updateAfterBind, variableCount);
}

void DescriptorCache::Recycle(VkDescriptorSetLayout layout, const SetAllocation& allocation) noexcept {
    if (allocation.set == VK_NULL_HANDLE || allocation.pool == VK_NULL_HANDLE) return;
    {
        std::lock_guard lock(mutex);
        try {
            auto& list = recycled[{layout, allocation.variableCount.value_or(0u)}];
            if (list.size() < 4096) {
                list.push_back(allocation);
                return;
            }
        } catch (...) {
        }
    }
    Free(allocation);
}

void DescriptorCache::Free(const SetAllocation& allocation, std::shared_ptr<const TableDescriptors> table) noexcept {
    if (allocation.set == VK_NULL_HANDLE || allocation.pool == VK_NULL_HANDLE) return;
    std::lock_guard lock(mutex);
    if (const auto found = std::find(dedicated.begin(), dedicated.end(), allocation.pool); found != dedicated.end()) {
        dedicated.erase(found);
        if (KeptSets() == 0u || allocation.layout == VK_NULL_HANDLE) {
            destroyPool(context.device, allocation.pool, nullptr);
            return;
        }
        if (kept.size() >= KeptSets()) {
            destroyPool(context.device, kept.back().pool, nullptr);
            kept.pop_back();
        }
        kept.insert(kept.begin(), allocation);
        kept.front().table = std::move(table);
        ++stats.keptSets;
        return;
    }
    auto& chain = allocation.updateAfterBind ? updateAfterBindPools : pools;
    if (allocation.poolIndex >= chain.size()) std::terminate();
    auto& pool = chain[allocation.poolIndex];
    if (pool.handle != allocation.pool || pool.liveSets == 0) std::terminate();
    if (freeDescriptorSets != nullptr && freeDescriptorSets(context.device, allocation.pool, 1, &allocation.set) != VK_SUCCESS) std::terminate();
    if (--pool.liveSets != 0) return;
    if (resetPool(context.device, pool.handle, 0) != VK_SUCCESS) std::terminate();
    pool.exhausted = false;
}

DescriptorCache::Stats DescriptorCache::Counters() const {
    std::lock_guard lock(mutex);
    return stats;
}

namespace {

std::atomic<std::uint64_t> fragmentParkedSets{0};
std::atomic<std::uint64_t> fragmentParksRefusedCapacity{0};
std::atomic<std::uint64_t> fragmentParksRefusedCap{0};
std::atomic<std::uint64_t> fragmentParksRefusedClosed{0};

}

class OwnDrawSets {
public:
    static constexpr std::size_t Capacity = 32;
    static constexpr std::size_t FragmentCapacity = 8;
    static constexpr std::uint64_t FragmentParkedCap = 16384;
    struct Plan {
        std::array<bool, 2> described{};
        std::array<std::vector<VkDescriptorPoolSize>, 2> sizes;
        std::array<std::optional<std::uint32_t>, 2> variable;
        std::vector<std::uint8_t> sets;
        std::vector<std::size_t> buffers;
        std::vector<std::vector<std::size_t>> slots;
        std::vector<std::uint8_t> allocationSets;
        std::array<bool, 2> lendable{};
        std::vector<std::pair<std::size_t, std::size_t>> variableBindings;
        std::size_t bufferCount = 0;
    };

    OwnDrawSets(DescriptorCache& cache, std::array<VkDescriptorSetLayout, 2> layouts, Plan plan, bool fragmentStore = false) : plan(std::move(plan)), cache(cache), layouts(layouts), capacity(fragmentStore ? FragmentCapacity : Capacity), counted(fragmentStore) {
        for (auto& list : parked) list.reserve(capacity);
        if (counted) {
            for (auto& list : stamps) list.reserve(capacity);
        }
    }

    bool Take(std::size_t set, DescriptorCache::SetAllocation& allocation) {
        std::uint64_t stamp = 0;
        return Take(set, allocation, stamp);
    }

    bool Take(std::size_t set, DescriptorCache::SetAllocation& allocation, std::uint64_t& stamp) {
        if (counted && count.load(std::memory_order_relaxed) == 0) return false;
        std::lock_guard lock(mutex);
        auto& list = parked[set];
        if (list.empty()) return false;
        allocation = std::move(list.back());
        list.pop_back();
        count.fetch_sub(1, std::memory_order_relaxed);
        if (counted) {
            stamp = stamps[set].back();
            stamps[set].pop_back();
            fragmentParkedSets.fetch_sub(1, std::memory_order_relaxed);
        }
        return true;
    }

    bool Park(std::size_t set, DescriptorCache::SetAllocation& allocation) noexcept {
        return Park(set, allocation, 0);
    }

    bool Park(std::size_t set, DescriptorCache::SetAllocation& allocation, std::uint64_t stamp) noexcept {
        std::lock_guard lock(mutex);
        auto& list = parked[set];
        if (closed || list.size() >= capacity) {
            if (counted) (closed ? fragmentParksRefusedClosed : fragmentParksRefusedCapacity).fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (counted && fragmentParkedSets.fetch_add(1, std::memory_order_relaxed) >= FragmentParkedCap) {
            fragmentParkedSets.fetch_sub(1, std::memory_order_relaxed);
            fragmentParksRefusedCap.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        list.push_back(std::move(allocation));
        if (counted) stamps[set].push_back(stamp);
        count.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void Drain(bool close) noexcept {
        std::lock_guard lock(mutex);
        closed = closed || close;
        for (std::size_t set = 0; set < parked.size(); ++set) {
            for (const auto& allocation : parked[set]) cache.Recycle(layouts[set], allocation);
            if (counted) fragmentParkedSets.fetch_sub(parked[set].size(), std::memory_order_relaxed);
            parked[set].clear();
            stamps[set].clear();
        }
        count.store(0, std::memory_order_relaxed);
    }

    std::size_t Parked() const noexcept { return count.load(std::memory_order_relaxed); }

    const Plan plan;

private:
    DescriptorCache& cache;
    const std::array<VkDescriptorSetLayout, 2> layouts;
    const std::size_t capacity;
    const bool counted;
    AgcDriver::Mutex mutex;
    std::array<std::vector<DescriptorCache::SetAllocation>, 2> parked;
    std::array<std::vector<std::uint64_t>, 2> stamps;
    bool closed = false;
    std::atomic<std::size_t> count{0};
};

class RepeatSets : public std::enable_shared_from_this<RepeatSets> {
public:
    static constexpr std::size_t Capacity = 256;

    RepeatSets(DescriptorCache& cache, VkDescriptorSetLayout layout, std::vector<VkDescriptorPoolSize> sizes, std::optional<std::uint32_t> variable) : cache(cache), layout(layout), sizes(std::move(sizes)), variable(variable) {}

    RepeatSets(const RepeatSets&) = delete;
    RepeatSets& operator=(const RepeatSets&) = delete;

    ~RepeatSets() {
        for (const auto& allocation : free) cache.Recycle(layout, allocation);
    }

    DescriptorCache::SetAllocation Take(bool& fresh) {
        {
            std::lock_guard lock(mutex);
            if (!free.empty()) {
                auto allocation = std::move(free.back());
                free.pop_back();
                fresh = false;
                return allocation;
            }
        }
        fresh = true;
        auto allocation = cache.AllocateRewritten(layout, sizes, false, variable);
        Require(allocation.set != VK_NULL_HANDLE, "repeat descriptor allocation failed");
        return allocation;
    }

    void Return(std::vector<DescriptorCache::SetAllocation>& used) noexcept {
        std::lock_guard lock(mutex);
        for (auto& allocation : used) {
            if (free.size() < Capacity) free.push_back(std::move(allocation));
            else cache.Recycle(layout, allocation);
        }
        used.clear();
    }

    std::uint32_t set = 0;
    std::vector<std::size_t> allocations;
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> patchInfos;
    std::vector<VkWriteDescriptorSet> patches;
    std::vector<std::size_t> patchSelected;
    std::vector<std::size_t> patchSlots;

private:
    DescriptorCache& cache;
    const VkDescriptorSetLayout layout;
    const std::vector<VkDescriptorPoolSize> sizes;
    const std::optional<std::uint32_t> variable;
    AgcDriver::Mutex mutex;
    std::vector<DescriptorCache::SetAllocation> free;
};

class RepeatSetLease {
public:
    RepeatSetLease() = default;
    RepeatSetLease(const RepeatSetLease&) = delete;
    RepeatSetLease& operator=(const RepeatSetLease&) = delete;
    ~RepeatSetLease() {
        if (owner != nullptr) owner->Return(sets);
    }

    std::shared_ptr<RepeatSets> owner;
    std::vector<DescriptorCache::SetAllocation> sets;
};

std::uint64_t ShaderResources::guestBufferBytes(std::span<const std::uint32_t> words) const {
    return ClampedBufferBytes(words, context.limits.maxStorageBufferRange);
}

std::uint64_t ShaderResources::ClampedBufferBytes(std::span<const std::uint32_t> words, std::uint64_t maxStorageBufferRange) {
    const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1] & ~0x40000000u, words[2], words[3]}};
    const auto address = descriptor.Base48();
    auto byteSize = descriptor.GetSize();
    if (byteSize > maxStorageBufferRange) {
        const auto mappedEnd = RegisteredReadableEnd(address);
        if (mappedEnd <= address) byteSize = 0;
        else if (mappedEnd - address < maxStorageBufferRange) byteSize = mappedEnd - address;
    }
    if (byteSize == 0 || address == 0) return 0;
    if (const std::uint64_t stride = descriptor.Stride(); stride != 0 && byteSize <= maxStorageBufferRange) {
        constexpr std::uint64_t GuestPageBytes = 0x4000;
        const auto end = address + byteSize;
        const auto overhang = end % GuestPageBytes;
        if (overhang != 0 && overhang < stride) {
            const auto allocationEnd = RegisteredReadableEnd(address);
            if (allocationEnd > address && allocationEnd < end && end - allocationEnd < stride) byteSize = allocationEnd - address;
        }
    }
    return byteSize;
}

std::uint64_t ShaderResources::BytesUpToTarget(std::uint64_t address, std::uint64_t bytes, std::uint64_t stride, const ColorTarget* target) {
    if (target == nullptr || stride == 0 || address >= target->MemoryAddress() || address + bytes <= target->MemoryAddress() || address + bytes - target->MemoryAddress() >= stride) return bytes;
    return target->MemoryAddress() - address;
}

std::size_t ShaderResources::addGuestBuffer(std::span<const std::uint32_t> words, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes, bool written, bool atomic, bool read, std::uint32_t extent) {
    Require(words.size() == 4, "buffer descriptor must contain four DWORDs");
    const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1] & ~0x40000000u, words[2], words[3]}};
    Require(descriptor.Type() == 0u, "buffer descriptor uses an unsupported type");
    const auto address = descriptor.Base48();
    const auto byteSize = guestBufferBytes(words);
    if (byteSize == 0 || address == 0) {
        allocations.push_back({0, EmptyBufferBytes, false, nullptr, ShaderRecompiler::DescriptorRole::GuestBuffers, false});
        allocations.back().oversized = address != 0 && descriptor.GetSize() != 0;
        return allocations.size() - 1;
    }
    Require(byteSize <= context.limits.maxStorageBufferRange, "shader buffer exceeds descriptor range limit");
    Require(byteSize <= std::numeric_limits<std::size_t>::max(), "shader buffer size exceeds host address space");
    const auto size = static_cast<std::size_t>(BytesUpToTarget(address, byteSize, descriptor.Stride(), written ? nullptr : target));
    Require(target == nullptr || !overlap(address, size, target->MemoryAddress(), target->bytes), "shader buffer aliases the render target");
    // APS5_ALL_BUFFERS_WRITTEN=1: every element is noted as written, as before bufferWritten existed.
    static const bool allWritten = std::getenv("APS5_ALL_BUFFERS_WRITTEN") != nullptr;
    Require(!written || !overlap(address, size, indexAddress, indexBytes), "writable shader buffer aliases the index buffer");
    written = written || allWritten;
    const std::uint64_t recordBytes = descriptor.Stride() != 0 ? descriptor.Stride() : extent;
    const bool swept = (read || (context.discreteGpu && descriptor.Stride() != 0)) && dispatchThreads != 0 && recordBytes != 0 && dispatchThreads * recordBytes * 2u >= byteSize;
    if (written) guestMemory.AddWritable(address, size, atomic, swept);
    else {
        guestMemory.AddReadable(address, size);
        ++readOnlyBuffers;
    }
    if (BuildProfiled()) {
        auto& counters = BufferWrites();
        counters.elements.fetch_add(1, std::memory_order_relaxed);
        if (!written) counters.readOnly.fetch_add(1, std::memory_order_relaxed);
    }
    allocations.push_back({address, size, true, nullptr, ShaderRecompiler::DescriptorRole::ShaderData, written});
    return allocations.size() - 1;
}

std::string ShaderResources::Describe() const {
    const auto sample = [](std::uint64_t address, std::uint64_t bytes) {
        // Heaps bound whole can include uncommitted pages; those ranges are not sampled.
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes))) return -1.0;
        std::size_t nonzero = 0;
        std::size_t samples = 0;
        const auto* data = reinterpret_cast<const std::uint32_t*>(address);
        const auto words = bytes / 4;
        const auto step = std::max<std::uint64_t>(1, words / 4096);
        for (std::uint64_t i = 0; i < words; i += step, ++samples) nonzero += data[i] != 0;
        return samples == 0 ? 0.0 : static_cast<double>(nonzero) / samples;
    };
    std::string text;
    char line[160];
    for (const auto& range : describedRanges) {
        std::snprintf(line, sizeof(line), " %s 0x%llx+0x%llx(%ux%u f%u t%d) nz=%.2f", range.kind, static_cast<unsigned long long>(range.address), static_cast<unsigned long long>(range.bytes), range.width, range.height, range.format, range.tileMode, sample(range.address, range.bytes));
        text += line;
        if (range.dccAddress != 0) {
            const auto keys = ReadDccKeys(range.dccAddress, range.bytes);
            std::snprintf(line, sizeof(line), " dcc=%s@0x%llx", DccKeysName(keys), static_cast<unsigned long long>(range.dccAddress));
            text += line;
            if (keys == DccKeys::Mixed) {
                // Where the keys change: first key, how many leading keys match it, and the next key.
                const auto* bytes = reinterpret_cast<const std::uint8_t*>(range.dccAddress);
                const auto count = static_cast<std::size_t>(range.bytes / 256u);
                std::size_t run = 1;
                while (run < count && bytes[run] == bytes[0]) ++run;
                std::snprintf(line, sizeof(line), "(%02x x%zu then %02x of %zu)", bytes[0], run, run < count ? bytes[run] : 0u, count);
                text += line;
            }
        }
    }
    // APS5_TRACE_DISPATCH_IO=2 also lists the words of small buffers (constants).
    static const bool words = [] { const char* value = std::getenv("APS5_TRACE_DISPATCH_IO"); return value != nullptr && value[0] == '2'; }();
    const auto appendWords = [&](const std::uint32_t* data, std::size_t bytes) {
        if (!words || bytes > 0x200) return;
        text += " [";
        for (std::size_t i = 0; i < bytes / 4; ++i) {
            char word[12];
            std::snprintf(word, sizeof(word), "%s%08x", i == 0 ? "" : " ", data[i]);
            text += word;
        }
        text += "]";
    };
    for (const auto& allocation : allocations) {
        if (allocation.guest) {
            std::snprintf(line, sizeof(line), " buffer%s 0x%llx+0x%zx nz=%.2f", allocation.written ? "" : "(ro)", static_cast<unsigned long long>(allocation.address), allocation.size, sample(allocation.address, allocation.size));
            text += line;
            if (GuestMemory::Accessible(reinterpret_cast<const void*>(allocation.address), allocation.size)) appendWords(reinterpret_cast<const std::uint32_t*>(allocation.address), allocation.size);
        } else if (allocation.buffer) {
            const auto bytes = allocation.buffer->Bytes();
            std::snprintf(line, sizeof(line), " data+0x%zx nz=%.2f", allocation.size, sample(reinterpret_cast<std::uint64_t>(bytes.data()), allocation.size));
            text += line;
            appendWords(reinterpret_cast<const std::uint32_t*>(bytes.data()), allocation.size);
        }
    }
    return text;
}

std::size_t ShaderResources::addDataBuffer(std::span<const std::uint32_t> words) {
    const auto size = words.size() * sizeof(std::uint32_t);
    Require(size <= context.limits.maxStorageBufferRange, "shader data buffer exceeds descriptor range limit");
    const bool refreshable = TemplateDataRefresh() && size <= MaxRefreshBytes;
    auto buffer = std::make_unique<Buffer>(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::memcpy(buffer->Bytes().data(), words.data(), size);
    Allocation allocation{0, size, false, std::move(buffer)};
    if (refreshable && keepsTemplateRecords()) allocation.dataWords.assign(words.begin(), words.end());
    allocations.push_back(std::move(allocation));
    mixDataWords(dataWordsHash, allocations.back().dataWords);
    return allocations.size() - 1;
}

void ShaderResources::rehashDataWords() {
    dataWordsHash = FnvOffset;
    // Data buffers are the non-guest allocations with a buffer (an address-role allocation has
    // none), appended in binding order by buildPrepare: the order DataWordsHash(shader) hashes.
    for (const auto& allocation : allocations) {
        if (!allocation.guest && allocation.buffer != nullptr && allocation.role != ShaderRecompiler::DescriptorRole::ImageTableMap) mixDataWords(dataWordsHash, allocation.dataWords);
    }
}

std::uint64_t ShaderResources::DataWordsHash(const CompiledShader& shader) {
    Require(shader.program != nullptr, "missing compiled shader");
    std::uint64_t hash = FnvOffset;
    for (const auto& binding : shader.program->bindings) {
        if (DataRole(binding.role)) mixDataWords(hash, DrawGuestWords(shader, binding));
    }
    return hash;
}

bool ShaderResources::DataWordsDiffer(const CompiledShader& shader) const {
    Require(shader.program != nullptr, "missing compiled shader");
    if (heapStage) return false;
    const auto& program = *shader.program;
    for (const auto& binding : program.bindings) {
        if (!DataRole(binding.role)) continue;
        const auto held = std::ranges::find_if(bindings, [&](const Binding& candidate) { return candidate.layout.binding == binding.binding; });
        if (held == bindings.end() || held->allocations.size() != 1) return true;
        const auto& allocation = allocations[held->allocations.front()];
        if (allocation.dataWords.size() != DrawGuestWords(shader, binding).size() || !std::equal(allocation.dataWords.begin(), allocation.dataWords.end(), DrawGuestWords(shader, binding).begin())) return true;
    }
    return false;
}

bool ShaderResources::RefreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder) {
    Require(shader.program != nullptr, "missing compiled shader");
    Require(!refreshResourceKey.empty() && MatchesContentKey(shader, refreshResourceKey), "runtime data refresh cannot replace bound resources");
    return refreshData(commands, shader, recorder);
}

bool ShaderResources::refreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder) {
    Require(shader.program != nullptr, "missing compiled shader");
    if (heapStage) return false;
    const auto& program = *shader.program;
    bool recorded = false;
    // One [gputime] range of class TemplateDataRefresh around the updates, begun at the first one
    // (a refresh that finds every word equal records nothing and times nothing).
    auto timing = Recorder::NoTiming;
    std::uint64_t refreshedBytes = 0;
    for (const auto& binding : program.bindings) {
        if (!DataRole(binding.role)) continue;
        const auto held = std::ranges::find_if(bindings, [&](const Binding& candidate) { return candidate.layout.binding == binding.binding; });
        Require(held != bindings.end(), "template bindings disagree with the shader");
        Require(held->allocations.size() == 1, "data binding without its buffer");
        auto& allocation = allocations[held->allocations.front()];
        const auto size = DrawGuestWords(shader, binding).size() * sizeof(std::uint32_t);
        Require(allocation.buffer != nullptr && !allocation.guest && allocation.size == size && size <= MaxRefreshBytes, "template data buffer cannot take the dispatch's words");
        if (allocation.dataWords.size() == DrawGuestWords(shader, binding).size() && std::equal(allocation.dataWords.begin(), allocation.dataWords.end(), DrawGuestWords(shader, binding).begin())) continue;
        if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) ValidateRuntimeResources(shader, DrawGuestWords(shader, binding));
        if (recorder != nullptr && timing == Recorder::NoTiming) timing = recorder->BeginGpuTiming(Recorder::CommandClass::TemplateDataRefresh);
        writeDataWords(commands, held->allocations.front(), DrawGuestWords(shader, binding));
        allocation.dataWords.assign(DrawGuestWords(shader, binding).begin(), DrawGuestWords(shader, binding).end());
        refreshedBytes += size;
        recorded = true;
    }
    if (timing != Recorder::NoTiming) recorder->EndGpuTiming(timing, refreshedBytes);
    if (recorded) rehashDataWords();
    return recorded;
}

std::optional<std::size_t> ShaderResources::shaderDataAllocation(const CompiledShader& shader) const {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    if (program.bindings.size() != bindings.size()) return std::nullopt;
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        if (program.bindings[index].role != ShaderRecompiler::DescriptorRole::ShaderData || bindings[index].allocations.size() != 1) continue;
        const auto allocation = bindings[index].allocations.front();
        if (allocations[allocation].guest || allocations[allocation].buffer == nullptr) return std::nullopt;
        return allocation;
    }
    return std::nullopt;
}

VkBuffer ShaderResources::shaderDataBuffer(const CompiledShader& shader) const {
    const auto allocation = shaderDataAllocation(shader);
    return allocation.has_value() ? allocations[*allocation].buffer->Handle() : VK_NULL_HANDLE;
}

void ShaderResources::forgetShaderDataWords(const CompiledShader& shader) {
    if (HeapShader(shader)) {
        ++heapStageWrites;
        auto& stageState = heapStages[HeapStageIndex(shader.stage)];
        stageState.data.mark.entry = 0;
        stageState.data.words.clear();
        stageState.dataInput.clear();
    }
    if (const auto allocation = shaderDataAllocation(shader)) {
        allocations[*allocation].dataWords.clear();
        rehashDataWords();
    }
}

void ShaderResources::writeDataWords(VkCommandBuffer commands, std::size_t allocation, std::span<const std::uint32_t> words) const {
    const auto& buffer = *allocations[allocation].buffer;
    const auto size = words.size() * sizeof(std::uint32_t);
    if (std::none_of(dataPatches.begin(), dataPatches.end(), [&](const DataPatch& patch) { return patch.allocation == allocation; })) {
        context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, buffer.Handle(), 0, size, words.data());
        return;
    }
    std::vector<std::uint32_t> patched(words.begin(), words.end());
    auto* bytes = reinterpret_cast<std::byte*>(patched.data());
    for (const auto& patch : dataPatches) {
        if (patch.allocation == allocation && patch.byte < size) bytes[patch.byte] = static_cast<std::byte>(patch.adjustment);
    }
    context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, buffer.Handle(), 0, size, patched.data());
}

void ShaderResources::PrecollectSurfaces() const {
    for (const auto& range : describedRanges) GuestMemory::CollectWrites(range.address, static_cast<std::size_t>(range.bytes));
}

void ShaderResources::addImageBinding(const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> guest, VkShaderStageFlags flags) {
    Require(binding.count != 0, "empty descriptor binding");
    const auto heap = static_cast<ShaderRecompiler::RuntimeAbi::Binding>(binding.binding % static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count));
    const auto capacity = binding.count;
    Require(binding.count <= ShaderRecompiler::RuntimeAbi::HeapCapacity(heap), "typed descriptor heap capacity exceeded");
    const auto localBinding = static_cast<std::uint32_t>(heap);
    const bool sampledHeap = localBinding >= ShaderRecompiler::RuntimeAbi::FirstImageBinding && localBinding < ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding;
    const bool storageHeap = localBinding >= ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding && localBinding < static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers);
    Require((binding.kind == ShaderRecompiler::DescriptorKind::SampledImage && sampledHeap) || (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage && storageHeap) || (binding.kind == ShaderRecompiler::DescriptorKind::Sampler && heap == ShaderRecompiler::RuntimeAbi::Binding::Samplers), "resource class disagrees with typed descriptor heap");
    Require(binding.kind == ShaderRecompiler::DescriptorKind::Sampler || context.nullDescriptors || context.placeholderImages != nullptr, "typed image heaps require nullDescriptor or placeholder images");
    if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
        // Storage images are guest textures the shader writes (looked up in stage B).
        Require(binding.role == ShaderRecompiler::DescriptorRole::GuestImages, "storage image binding has a non-image role");
        Require(guest.size() == static_cast<std::size_t>(binding.count) * 8u, "guest storage image descriptors must contain 8 dwords each");
        Require(context.detiler != nullptr, "device texture detiler is unavailable");
        Require(capacity <= context.limits.maxPerStageDescriptorStorageImages, "shader storage-image descriptors exceed per-stage limits");
        bindings.push_back({{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, capacity, flags, nullptr}, {}, {}});
        plannedStorageImages += binding.count;
        deferredImages.push_back({&binding, guest, bindings.size() - 1});
        return;
    }
    const bool sampledImage = binding.kind == ShaderRecompiler::DescriptorKind::SampledImage;
    const bool samplerKind = binding.kind == ShaderRecompiler::DescriptorKind::Sampler;
    if (!(sampledImage || samplerKind)) Require(false, std::string("unsupported descriptor kind ") + kindName(binding.kind) + " for role " + roleName(binding.role));
    Require((sampledImage && binding.role == ShaderRecompiler::DescriptorRole::GuestImages) || (samplerKind && binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers), "guest image descriptor role disagrees with its kind");
    Require(guest.size() % binding.count == 0, "guest image descriptor size is not a multiple of the binding count");
    const auto elementWords = guest.size() / binding.count;

    Binding item{{binding.binding, sampledImage ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER, capacity, flags, nullptr}, {}, {}};

    if (sampledImage) {
        Require(elementWords == 8, "guest texture descriptor must contain 8 dwords");
        Require(binding.imageShape.has_value(), "guest image binding is missing an image shape");
        Require(context.detiler != nullptr, "device texture detiler is unavailable");
        Require(context.textureCache != nullptr, "device texture cache is unavailable");
        Require(capacity <= context.limits.maxPerStageDescriptorSampledImages, "shader sampled-image descriptors exceed per-stage limits");
        plannedSampledImages += binding.count;
        Require(plannedSampledImages <= context.limits.maxDescriptorSetSampledImages, "pipeline sampled-image descriptors exceed device limits");
    } else {
        Require(elementWords == 4, "guest sampler descriptor must contain 4 dwords");
        if (capacity > context.limits.maxPerStageDescriptorSamplers) throw std::runtime_error(DescriptorLimitError("shader sampler descriptors exceed per-stage limits", capacity, context.limits.maxPerStageDescriptorSamplers));
        Require(binding.samplerDepthCompare.size() == binding.count, "guest sampler binding is missing depth comparison metadata");
        Require(binding.samplerUnnormalized.empty() || binding.samplerUnnormalized.size() == binding.count, "guest sampler binding has unnormalized coordinate metadata of another size");
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            auto guestWords = std::span<const std::uint32_t>(guest).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            std::array<std::uint32_t, 4> normalizedWords{};
            if (element < binding.samplerCoordinatesIgnored.size() && binding.samplerCoordinatesIgnored[element]) {
                std::copy(guestWords.begin(), guestWords.end(), normalizedWords.begin());
                normalizedWords[0] &= ~(1u << 15u);
                guestWords = normalizedWords;
            }
            const auto words = guestWords;
            const bool compareEnable = binding.samplerDepthCompare.at(element);
            const bool unnormalized = element < binding.samplerUnnormalized.size() && binding.samplerUnnormalized[element];
            const bool integerBorder = element < binding.samplerIntegerBorder.size() && binding.samplerIntegerBorder[element];
            if (context.samplerCache != nullptr && !SamplerCacheDisabled()) {
                samplers.push_back(context.samplerCache->Get(context, words, compareEnable, unnormalized, true, integerBorder));
            } else {
                auto resource = DecodeSamplerResource(words, unnormalized, true, integerBorder);
                resource.compareEnable = compareEnable;
                samplers.push_back(std::make_shared<Sampler>(context, resource));
            }
            samplerSources.push_back({{words[0], words[1], words[2], words[3]}, compareEnable, unnormalized, integerBorder});
            item.imageAllocations.push_back(samplers.size() - 1);
        }
        Require(samplers.size() <= context.limits.maxDescriptorSetSamplers, "pipeline sampler descriptors exceed device limits");
    }

    bindings.push_back(std::move(item));
    if (sampledImage) deferredImages.push_back({&binding, guest, bindings.size() - 1});
}

namespace {

// Calls `visit(binding, element, words)` for every sampled and storage image element of a compiled
// shader's bindings, in plan order (the order addImageBinding and resolveImageBinding walk).
template <typename Visit>
void forEachImageElement(const ShaderRecompiler::RecompileResult& program, Visit&& visit) {
    for (const auto& binding : program.bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.count == 0) continue;
        if (binding.kind != ShaderRecompiler::DescriptorKind::SampledImage && binding.kind != ShaderRecompiler::DescriptorKind::StorageImage) continue;
        const auto elementWords = binding.guestDescriptor.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) visit(binding, element, std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords));
    }
}

}

bool ShaderResources::precollectImages() {
    // The image lookups of stage B (cachedTexture, StorageTexture::Refresh) each start with a
    // GuestMemory::CollectWrites of their surface, the GetWriteWatch walk that is most of the
    // lookups' time under the lock. The walk is memoized per thread epoch, and the worker thread
    // passes no ordering point between the stages (see GuestMemory::BumpCollectEpoch), so walking
    // every surface here makes stage B's collects memo hits: the walk leaves the lock, the lookups'
    // checks stay where they were. A memo miss (the ring overflowed, an uncommitted page) just walks
    // under the lock as before; a descriptor stage B rejects is left to it. APS5_NO_PRECOLLECT=1
    // disables the pass.
    // The pass also leaves one ImageRecord per element: the decode and the surface description are
    // made once for the build, and a sampled element's cache entry is taken here (a hash lookup
    // under the cache's own mutex) so that stage B, under the device lock, runs only the
    // fastRevalidate predicate on it (see fastTexture). The collect comes before the entry is read,
    // as the predicate's rule demands (the generation the entry moves to must predate the checks).
    static const bool disabled = std::getenv("APS5_NO_PRECOLLECT") != nullptr;
    static const bool noRecords = std::getenv("APS5_NO_STAGE_A_IMAGES") != nullptr;
    if (disabled || deferredImages.empty()) return false;
    imageRecords.clear();
    nextImageRecord = 0;
    auto& counters = TextureCounts();
    for (const auto& deferred : deferredImages) {
        const auto& binding = *deferred.binding;
        const auto elementWords = deferred.words.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const bool sampled = binding.kind == ShaderRecompiler::DescriptorKind::SampledImage;
            const auto raw = std::span<const std::uint32_t>(deferred.words).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            std::array<std::uint32_t, 8> narrowed{};
            const auto words = firstLayerWords(raw, sampled ? binding.imageShape : std::nullopt, narrowed, false);
            if (sampled && !imageRecords.empty()) {
                const auto& previous = imageRecords.back();
                if (previous.sampled && previous.decoded && words.size() == 8 && std::equal(words.begin(), words.end(), previous.words.begin()) && (binding.imageDepthCompare.empty() || !binding.imageDepthCompare.at(element))) {
                    ImageRecord repeated;
                    repeated.sampled = true;
                    repeated.decoded = previous.decoded;
                    repeated.words = previous.words;
                    repeated.resource = previous.resource;
                    repeated.guestBytes = previous.guestBytes;
                    repeated.components = previous.components;
                    imageRecords.push_back(repeated);
                    continue;
                }
            }
            ImageRecord record;
            record.sampled = sampled;
            if (sampled ? !IsNullSampledTextureResource(words) : !IsNullTextureResource(words)) {
                const auto* memo = words.size() == 8 ? &decodedImage(context.device, words) : nullptr;
                record.resource = memo != nullptr ? memo->resource : DecodeTextureResource(words);
                record.guestBytes = memo != nullptr ? memo->guestBytes : SurfaceGuestBytes(record.resource);
                record.generation = GuestMemory::CollectWrites(record.resource.baseAddress, static_cast<std::size_t>(record.guestBytes));
                record.decoded = true;
                if (record.sampled && !noRecords && words.size() == 8 && (binding.imageDepthCompare.empty() || !binding.imageDepthCompare.at(element))) {
                    std::copy(words.begin(), words.end(), record.words.begin());
                    record.components = memo->components;
                    record.keys = TextureClearKeys(record.resource, record.guestBytes);
                    auto& cache = Textures();
                    std::lock_guard lock(cache.mutex);
                    if (const auto it = findSampledEntry(cache, context.device, record.words, record.components); it != cache.entries.end()) {
                        record.texture = it->texture;
                        record.source = it->source;
                        record.entryKeys = it->keys;
                        record.entryGeneration = it->generation;
                        if (TextureCountersReported()) counters.records.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            imageRecords.push_back(std::move(record));
        }
    }
    return true;
}

std::shared_ptr<Texture> ShaderResources::fastTexture(ImageRecord& record) {
    if (record.texture == nullptr || record.texture->RefreshedPerUse() || record.texture->IsDepthCopy()) return nullptr;
    struct Outcome {
        bool profile;
        std::chrono::steady_clock::time_point start;
        bool hit = false;
        ~Outcome() {
            if (profile) LookupOutcomes::Add(hit ? LookupOutcomes::SampledFast : LookupOutcomes::SampledFastMiss, start);
        }
    } outcome{LookupOutcomes::Profiled(), LookupOutcomes::Profiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};
    // Exactly the "unchanged" branches of cachedTexture, from write stamps, the pending-results
    // registry and the DCC keys, the way fastRevalidate proves a built object current: the collect
    // first (a memo hit: stage A walked the range), then no other image may have results pending
    // over the memory (a lookup would flush them into it, or view them instead), a storage-sourced
    // view needs its image to be the cache's image of the surface and current with guest memory,
    // a snapshot needs the memory unchanged since its content matched, and the keys must be what
    // the content was made under whenever the surface has any (re-read under the lock: a flush
    // between the stages marks them uncompressed).
    const auto address = record.resource.baseAddress;
    const auto bytes = static_cast<std::size_t>(record.guestBytes);
    RequireNoMultisampleColorTarget(address, bytes != 0 ? bytes : 1u, "a texture read");
    if (DepthSurfaceAt(address) || !FlushDepthPlanes(address, bytes).empty()) return nullptr;
    if (GuestMemory::CollectWrites(address, bytes) == 0) return nullptr;
    if (PendingStorageOverlaps(address, bytes, record.source.get())) return nullptr;
    auto keys = record.keys;
    if (record.resource.dccAddress != 0) {
        const auto scanStart = outcome.profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // Through the proof of the image whose keys they are (the view's source when the surface
        // is the image's, else the texture's): a scan only when the key range was stamped.
        keys = ProvedClearKeys(record.resource, record.guestBytes, SameKeySurface(record.source.get(), record.resource, record.guestBytes) ? record.source->KeyProof() : record.texture->KeyProof());
        if (outcome.profile) LookupOutcomes::Add(LookupOutcomes::DccScan, scanStart);
    }
    // The keys the entry was made under (a view of a pending render target was made under its clear
    // keys and stays valid while they are unchanged); a change is the full lookup's to judge.
    if (keys != record.entryKeys) return nullptr;
    if (record.source != nullptr) {
        if (!StorageImageCached(context, record.source.get()) || !GuestMemory::UnchangedSince(address, bytes, record.source->GenerationFor(address, bytes))) return nullptr;
        if (record.resource.dccAddress != 0 && IsDccClear(record.source->FilledKeys())) return nullptr;
        // Under fast-clear keys the view holds only while its image's results are still pending over
        // the surface; flushed, the clear the image cannot see makes the lookup take a snapshot.
        if (keys != DccKeys::Uncompressed && StorageTexture::FindPending(address, record.guestBytes) != record.source) return nullptr;
    } else if ((keys == DccKeys::Uncompressed || keys == DccKeys::ClearSingle) && !GuestMemory::UnchangedSince(address, bytes, record.entryGeneration)) {
        return nullptr;
    }
    // The entry must still be the cache's, holding this object (an evicted object is not reused: the
    // next lookup would make another), and it takes the stage-A generation like a hit would.
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    const auto it = findSampledEntry(cache, context.device, record.words, record.components);
    if (it == cache.entries.end() || it->texture != record.texture || it->committed.has_value()) return nullptr;
    if (it->source == nullptr) it->generation = record.generation;
    touchTexture(cache, it);
    logLookup({record.texture.get(), record.resource, record.guestBytes, keys, record.source != nullptr ? 0 : record.generation, record.source.get()});
    outcome.hit = true;
    return std::move(record.texture);
}

void ShaderResources::applySamplerVariants() {
    for (std::size_t index = 0; index < samplerSources.size(); ++index) {
        const auto& source = samplerSources[index];
        const auto variant = SingleLevelSamplerWords(source.words, source.singleLevelImage, source.mipmappedImage);
        if (!variant.has_value() && !source.pinnedLod.has_value()) continue;
        const auto words = source.pinnedLod.has_value() ? PinnedLodSamplerWords(variant.has_value() ? *variant : source.words, *source.pinnedLod) : *variant;
        if (context.samplerCache != nullptr && !SamplerCacheDisabled()) {
            samplers[index] = context.samplerCache->Get(context, words, source.compareEnable, source.unnormalized, true, source.integerBorder);
        } else {
            auto resource = DecodeSamplerResource(words, source.unnormalized, true, source.integerBorder);
            resource.compareEnable = source.compareEnable;
            samplers[index] = std::make_shared<Sampler>(context, resource);
        }
    }
}

void ShaderResources::resolveImageBinding(const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> guest, Binding& item, std::span<const std::shared_ptr<Sampler>> shaderSamplers) {
    auto& counters = TextureCounts();
    // The element's stage-A record, when the pass ran (records follow the plan order exactly).
    const auto nextRecord = [&]() -> ImageRecord* { return nextImageRecord < imageRecords.size() ? &imageRecords[nextImageRecord++] : nullptr; };
    if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
        Require(binding.imageSamplers.size() == binding.count, "guest sampled image binding is missing its image-sampler pairs");
        const auto elementWords = guest.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto raw = std::span<const std::uint32_t>(guest).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            std::array<std::uint32_t, 8> narrowed{};
            const auto words = firstLayerWords(raw, binding.imageShape, narrowed, true);
            auto* record = nextRecord();
            if (IsNullSampledTextureResource(words)) {
                item.imageAllocations.push_back(std::numeric_limits<std::size_t>::max());
                --plannedSampledImages;
                continue;
            }
            const auto resource = record != nullptr && record->decoded ? record->resource : DecodeTextureResource(words);
            const auto minLod = context.shaderImageMinLod && resource.minLod > resource.baseLevel * 256u && resource.lastLevel > resource.baseLevel ? std::min(resource.minLod - resource.baseLevel * 256u, (resource.lastLevel - resource.baseLevel) * 256u) : 0u;
            if (minLod != 0u && !shaderSamplers.empty()) {
                const auto firstSampler = static_cast<std::size_t>(shaderSamplers.data() - samplers.data());
                for (std::uint32_t sampler = 0; sampler < shaderSamplers.size() && sampler < 32u; ++sampler) {
                    auto& source = samplerSources.at(firstSampler + sampler);
                    if (((binding.imageSamplers[element] >> sampler) & 1u) == 0 || !HoldsLodAtZero(source.words)) continue;
                    Require(!source.pinnedLod.has_value() || *source.pinnedLod == minLod, "an S# that holds the LOD at 0 is paired with images of different minimum LODs");
                    source.pinnedLod = minLod;
                }
            }
            const bool forcedSrgb = ForcesDegammaOn(resource.format, binding.imageSamplers[element], shaderSamplers);
            RequireNonSeamlessCube(resource.dimension == TextureDimension::kCube, binding.imageSamplers[element], shaderSamplers);
            RequireDisablesDegammaOn(resource.format, binding.imageSamplers[element], shaderSamplers);
            const bool depthCompare = !binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element);
            const bool firstLayer = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && (resource.dimension == TextureDimension::k2DArray || (depthCompare && resource.dimension == TextureDimension::k1DArray));
            const bool depth1D = depthCompare && ((binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k1D) || (binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2DArray && resource.dimension == TextureDimension::k1DArray));
            const bool firstSample = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DMsaa;
            const bool layerRows = layerRowsBinding(binding.imageShape, resource, depthCompare);
            Require(!layerRows || (resource.baseArray <= resource.depthOrLastArray && resource.depthOrLastArray - resource.baseArray >= context.limits.maxImageArrayLayers), "a 1D array within maxImageArrayLayers was bound as rows of a 2D image");
            if (!firstLayer && !depth1D && !firstSample && !layerRows && !MatchesGuestDimension(*binding.imageShape, resource.dimension)) throw std::runtime_error("AGC graphics: guest texture dimension disagrees with the shader's declared image shape (shape " + std::to_string(static_cast<int>(*binding.imageShape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
            const VkComponentMapping components = ViewComponents(resource);
            const auto guestBytes = record != nullptr && record->decoded ? record->guestBytes : SurfaceGuestBytes(resource);
            CompletePendingCpuWrites(context, resource.baseAddress, static_cast<std::size_t>(guestBytes));
            if (resource.dccAddress != 0) CompletePendingCpuWrites(context, resource.dccAddress, DccKeyBytes(guestBytes));
            std::shared_ptr<Texture> texture;
            const VkImageViewType multisampleViewType = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2DMultisampleArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            if (previousSampled.texture != nullptr && previousSampled.depthCompare == depthCompare && previousSampled.multisampleViewType == multisampleViewType && previousSampled.firstSample == firstSample && previousSampled.layerRows == layerRows && words.size() == 8 && std::equal(words.begin(), words.end(), previousSampled.words.begin())) texture = previousSampled.texture;
            if (texture == nullptr && !firstSample && record != nullptr && record->texture != nullptr) {
                texture = fastTexture(*record);
                if (TextureCountersReported()) (texture != nullptr ? counters.fastHits : counters.fastMisses).fetch_add(1, std::memory_order_relaxed);
            }
            if (texture == nullptr && firstSample) {
                Require(!depthCompare, "reading sample 0 of a multisampled texture with a depth comparison through a 2D image is not implemented");
                texture = DepthSurfaceAt(resource.baseAddress) ? DepthSurfaceFirstSample(context, words, resource, components) : MultisampleTargetFirstSample(context, resource, components);
            }
            if (texture == nullptr) texture = cachedTexture(context, words, resource, components, guestBytes, depthCompare, multisampleViewType, layerRows);
            if (resource.dimension == TextureDimension::k2DMsaa) multisampleSources.push_back(resource.baseAddress);
            bool baseLevel = false;
            if (element < binding.imageUnnormalized.size() && binding.imageUnnormalized[element]) {
                const auto range = texture->SampledViewRange(firstLayer);
                const bool exactAtBase = element < binding.imageUnnormalizedBaseLevel.size() && binding.imageUnnormalizedBaseLevel[element];
                const bool singleLevel = range.levels == 1u && resource.baseLevel == 0u;
                const bool supported = range.layers == 1u && EffectiveMinLod(resource) == 0.0f && (singleLevel || exactAtBase);
                if (!supported || (range.type != VK_IMAGE_VIEW_TYPE_1D && range.type != VK_IMAGE_VIEW_TYPE_2D)) throw std::runtime_error("AGC graphics: guest texture sampled with unnormalized coordinates is not a single-layer 1D or 2D view without a minimum LOD clamp, and is neither a single-level view at mip 0 nor read only by image_sample_lz through an S# without minimum LOD or bias, which is not implemented (base level " + std::to_string(resource.baseLevel) + ", levels " + std::to_string(range.levels) + ", layers " + std::to_string(range.layers) + ", view type " + std::to_string(static_cast<int>(range.type)) + ")");
                baseLevel = range.levels > 1u;
            }
            RequireFilterMinmax(context, texture->ViewFormat(), binding.imageSamplers[element], shaderSamplers);
            RequireBorderSwizzle(resource.bcSwizzle, binding.imageSamplers[element], shaderSamplers);
            if (forcedSrgb && !texture->HasSrgbView()) Require(false, "a texture of guest format " + std::to_string(resource.format) + ", which is not sRGB and has no sRGB view, is sampled through a sampler that forces sRGB decoding, which is not implemented");
            if (!shaderSamplers.empty()) {
                const bool singleLevel = DescriptorSingleLevel(words);
                const auto firstSampler = static_cast<std::size_t>(shaderSamplers.data() - samplers.data());
                for (auto mask = binding.imageSamplers[element]; mask != 0u; mask &= mask - 1u) {
                    const auto sampler = static_cast<std::uint32_t>(std::countr_zero(mask));
                    if (sampler >= shaderSamplers.size()) break;
                    auto& source = samplerSources.at(firstSampler + sampler);
                    (singleLevel ? source.singleLevelImage : source.mipmappedImage) = true;
                }
            }
            if (words.size() == 8) {
                std::copy(words.begin(), words.end(), previousSampled.words.begin());
                previousSampled.depthCompare = depthCompare;
                previousSampled.multisampleViewType = multisampleViewType;
                previousSampled.firstSample = firstSample;
                previousSampled.layerRows = layerRows;
                previousSampled.texture = texture;
            }
            textures.push_back(std::move(texture));
            textureFirstLayer.push_back(firstLayer);
            textureBaseLevel.push_back(baseLevel);
            textureSrgb.push_back(forcedSrgb);
            describedRanges.push_back({"texture", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress});
            item.imageAllocations.push_back(textures.size() - 1);
        }
        // The line is due even when every element took the fast path (cachedTexture reports too).
        reportTextureCounters();
        return;
    }
    previousSampled = {};
    // Consecutive identical storage descriptors address successive mips of one texture (dynamic-mip
    // storage writes).
    std::uint32_t mipOffset = 0;
    for (std::uint32_t element = 0; element < binding.count; ++element) {
        const auto words = std::span<const std::uint32_t>(guest).subspan(static_cast<std::size_t>(element) * 8u, 8u);
        const bool sameAsPrevious = SameAsPreviousStorageElement(guest, element);
        if (sameAsPrevious) ++mipOffset;
        else mipOffset = 0;
        const auto* record = nextRecord();
        if (IsNullTextureResource(words)) {
            item.imageAllocations.push_back(std::numeric_limits<std::size_t>::max());
            --plannedStorageImages;
            continue;
        }
        const auto resource = record != nullptr && record->decoded ? record->resource : DecodeTextureResource(words);
        const bool firstLayer = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
        if (binding.imageShape.has_value() && !firstLayer && !MatchesGuestDimension(*binding.imageShape, resource.dimension)) throw std::runtime_error("AGC graphics: guest storage texture dimension disagrees with the shader's declared image shape (shape " + std::to_string(static_cast<int>(*binding.imageShape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
        if (resource.dimension == TextureDimension::k2DMsaa) {
            Require(!(element < binding.imageAtomic.size() && binding.imageAtomic[element]) && !(element < binding.imageAtomic64.size() && binding.imageAtomic64[element]), "storage image atomics on a multisampled texture are not implemented");
            storageTextures.push_back(nullptr);
            multisampleStorage.resize(storageTextures.size());
            const bool uintStorage = ShaderRecompiler::RuntimeAbi::UintStorageImageBinding(binding.binding);
            multisampleStorage.back() = MultisampleTargetStorageView(context, resource, uintStorage);
            multisampleSources.push_back(resource.baseAddress);
            storageMips.push_back(0);
            storageKeys.push_back(0);
            storageFirstLayer.push_back(false);
            storageWritten.push_back(false);
            storageAtomic.push_back(false);
            storageAtomic64.push_back(false);
            storageUint.push_back(false);
            item.imageAllocations.push_back(storageTextures.size() - 1);
            continue;
        }
        const auto mip = std::min(resource.baseLevel + mipOffset, resource.mipCount - 1u);
        Require(resource.minLod <= mip * 256u, "guest storage texture descriptor clamps its minimum LOD above the level it addresses, which is not implemented");
        const auto guestBytes = record != nullptr && record->decoded ? record->guestBytes : SurfaceGuestBytes(resource);
        CompletePendingCpuWrites(context, resource.baseAddress, static_cast<std::size_t>(guestBytes));
        if (resource.dccAddress != 0) CompletePendingCpuWrites(context, resource.dccAddress, DccKeyBytes(guestBytes));
        // Images the shader only reads have nothing to store back.
        const bool written = element >= binding.imageWritten.size() || binding.imageWritten[element];
        const bool stored = element < binding.imageWritten.size() && binding.imageWritten[element];
        if (stored) claimStoredMultisampleMemory(context, resource, guestBytes, mip);
        // The same surface as the previous element: its image was just looked up and refreshed.
        if (sameAsPrevious && StorageDedupeEnabled()) storageTextures.push_back(storageTextures.back());
        else storageTextures.push_back(cachedStorageTexture(context, words, resource, mip, guestBytes, written));
        storageMips.push_back(mip);
        storageKeys.push_back(resource.dccAddress);
        storageFirstLayer.push_back(firstLayer);
        storageWritten.push_back(written);
        storageAtomic.push_back(element < binding.imageAtomic.size() && binding.imageAtomic[element]);
        storageAtomic64.push_back(element < binding.imageAtomic64.size() && binding.imageAtomic64[element]);
        storageUint.push_back(false);
        describedRanges.push_back({"storage", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress});
        if (stored) {
            describedRanges.back().storedWords.emplace();
            std::copy(words.begin(), words.end(), describedRanges.back().storedWords->begin());
            describedRanges.back().storedMip = mip;
        }
        item.imageAllocations.push_back(storageTextures.size() - 1);
    }
}

std::uint32_t StorageTableCapacity(const Context& context) {
    return std::min<std::uint32_t>({context.imageTableCapacity, 4096u, context.limits.maxPerStageDescriptorStorageImages / 4u});
}

namespace {

std::uint64_t Room(std::uint64_t limit, std::uint64_t reserve) {
    return limit > reserve ? limit - reserve : 0u;
}

std::uint32_t ImageTableCapacityFor(const Context& context, bool updateAfterBind) {
    if (!context.runtimeDescriptorArray || !context.partiallyBound || !context.variableDescriptorCount || !context.descriptorIndexing) return 0u;
    if (updateAfterBind && !context.sampledImageUpdateAfterBind) return 0u;
    constexpr std::uint64_t sampledHeaps = static_cast<std::uint64_t>(ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding - ShaderRecompiler::RuntimeAbi::FirstImageBinding) * ShaderRecompiler::RuntimeAbi::SampledHeapCapacity;
    constexpr std::uint64_t storageHeaps = static_cast<std::uint64_t>(static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Samplers) - ShaderRecompiler::RuntimeAbi::FirstStorageImageBinding) * ShaderRecompiler::RuntimeAbi::StorageHeapCapacity;
    constexpr std::uint64_t buffers = ShaderRecompiler::RuntimeAbi::BufferCapacity + 8u;
    const auto& indexed = context.descriptorIndexingLimits;
    const std::uint64_t stageImages = updateAfterBind ? indexed.maxPerStageDescriptorUpdateAfterBindSampledImages : context.limits.maxPerStageDescriptorSampledImages;
    const std::uint64_t stageResources = updateAfterBind ? indexed.maxPerStageUpdateAfterBindResources : context.limits.maxPerStageResources;
    const std::uint64_t setImages = updateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSampledImages : context.limits.maxDescriptorSetSampledImages;
    std::uint64_t capacity = static_cast<std::uint64_t>(ShaderRecompiler::RuntimeAbi::StageCount) * ShaderRecompiler::ImageTableAbi::ImageElementBudget;
    capacity = std::min(capacity, Room(stageImages, sampledHeaps));
    capacity = std::min(capacity, Room(stageResources, sampledHeaps + storageHeaps + buffers));
    capacity = std::min(capacity, Room(setImages, sampledHeaps * ShaderRecompiler::RuntimeAbi::StageCount));
    return static_cast<std::uint32_t>(capacity);
}

std::uint32_t SamplerTableCapacityFor(const Context& context, bool updateAfterBind) {
    if (!context.runtimeDescriptorArray || !context.partiallyBound) return 0u;
    if (updateAfterBind && !context.sampledImageUpdateAfterBind) return 0u;
    const auto& indexed = context.descriptorIndexingLimits;
    const std::uint64_t stageSamplers = updateAfterBind ? indexed.maxPerStageDescriptorUpdateAfterBindSamplers : context.limits.maxPerStageDescriptorSamplers;
    const std::uint64_t setSamplers = updateAfterBind ? indexed.maxDescriptorSetUpdateAfterBindSamplers : context.limits.maxDescriptorSetSamplers;
    std::uint64_t capacity = ShaderRecompiler::ImageTableAbi::SamplerElementBudget;
    capacity = std::min(capacity, Room(stageSamplers, ShaderRecompiler::RuntimeAbi::SamplerHeapCapacity));
    capacity = std::min(capacity, Room(setSamplers / ShaderRecompiler::RuntimeAbi::StageCount, ShaderRecompiler::RuntimeAbi::SamplerHeapCapacity));
    return static_cast<std::uint32_t>(capacity - capacity % ShaderRecompiler::ImageTableAbi::SamplerEntryElements);
}

}

std::uint64_t& DeviceProcLookups() {
    thread_local std::uint64_t count = 0;
    return count;
}

std::uint32_t ImageTableCapacity(const Context& context) {
    const auto capacity = ImageTableCapacityFor(context, false);
    return capacity != 0u ? capacity : ImageTableCapacityFor(context, true);
}

std::uint32_t SamplerTableCapacity(const Context& context) {
    const auto capacity = SamplerTableCapacityFor(context, false);
    return capacity != 0u ? capacity : SamplerTableCapacityFor(context, true);
}

bool ImageTableUpdateAfterBind(const Context& context) {
    return ImageTableCapacityFor(context, false) == 0u && ImageTableCapacityFor(context, true) != 0u;
}

bool SamplerTableUpdateAfterBind(const Context& context) {
    return SamplerTableCapacityFor(context, false) == 0u && SamplerTableCapacityFor(context, true) != 0u;
}

ShaderResources::TableStage& ShaderResources::tableStage(const CompiledShader& shader) {
    for (auto& table : tableStages) {
        if (table.program == shader.program) return table;
    }
    auto& table = tableStages.emplace_back();
    table.stage = shader.stage;
    table.program = shader.program;
    table.poison = shader.program->imageTablePoison;
    table.messages.assign(table.poison.size(), nullptr);
    table.heap = HeapShader(shader) && shader.program->dataAddressPush;
    return table;
}

std::uint32_t ShaderResources::rejectTableElement(TableStage& table, std::span<const std::uint32_t> words, const std::string& message) {
    ShaderRecompiler::ImageTableEntryPoison poison{};
    std::copy(words.begin(), words.end(), poison.words.begin());
    poison.dwordCount = static_cast<std::uint32_t>(words.size());
    poison.resource = ShaderRecompiler::NoTable;
    poison.reason = static_cast<std::uint32_t>(ShaderRecompiler::ImageTableAbi::PoisonReason::DriverRejected);
    table.poison.push_back(poison);
    const bool repeated = !table.messages.empty() && table.messages.back() != nullptr && *table.messages.back() == message;
    table.messages.push_back(repeated ? table.messages.back() : std::make_shared<const std::string>(message));
    faultChecks = true;
    return ShaderRecompiler::ImageTableAbi::PoisonCode(static_cast<std::uint32_t>(table.poison.size() - 1u));
}

void ShaderResources::addSamplerTable(const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> guest, VkShaderStageFlags flags, TableStage& table) {
    namespace Abi = ShaderRecompiler::ImageTableAbi;
    Require(binding.kind == ShaderRecompiler::DescriptorKind::Sampler && guest.size() == static_cast<std::size_t>(binding.count) * 4u && binding.samplerDepthCompare.size() == binding.count && binding.count % Abi::SamplerEntryElements == 0u, "sampler table binding is malformed");
    Require(table.heap || context.samplerTableCapacity != 0, "sampler tables need runtimeDescriptorArray and descriptorBindingPartiallyBound, which the device lacks");
    const VkDescriptorBindingFlags samplerTableFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | (context.samplerTableUpdateAfterBind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0);
    Binding item{{binding.binding, VK_DESCRIPTOR_TYPE_SAMPLER, context.samplerTableCapacity, flags, nullptr}, {}, {}, samplerTableFlags};
    static const bool noSamplerCache = std::getenv("APS5_NO_SAMPLER_CACHE") != nullptr;
    for (std::uint32_t entry = 0; entry < binding.count / Abi::SamplerEntryElements; ++entry) {
        const auto first = entry * Abi::SamplerEntryElements;
        const auto words = guest.subspan(static_cast<std::size_t>(first) * 4u, 4u);
        std::string rejected;
        if (!table.heap && first + Abi::SamplerEntryElements > context.samplerTableCapacity) rejected = "the sampler table exceeds the device's capacity of " + std::to_string(context.samplerTableCapacity) + " samplers";
        for (std::uint32_t element = first; rejected.empty() && element < first + Abi::SamplerEntryElements; ++element) {
            try {
                const auto elementWords = std::span<const std::uint32_t, 4>(guest.data() + static_cast<std::size_t>(element) * 4u, 4u);
                auto resource = DecodeSamplerResource(elementWords);
                resource.compareEnable = binding.samplerDepthCompare.at(element);
                Sampler::Validate(context, resource);
                Require(!Sampler::ShaderAppliesLodBias(resource), "a sampler table entry with a LOD bias beyond the sampler limit needs a per-entry shader bias, which is not implemented");
                Require(!Sampler::ReadsOpaqueBlackBorder(resource), "a sampler table entry with an opaque black border needs dynamic texture BC-swizzle pairing, which is not implemented");
                Require(resource.addressModeU != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER && resource.addressModeV != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER && resource.addressModeW != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, "a sampler table entry with a border needs dynamic texture numeric-class pairing, which is not implemented");
                Require(!resource.nonSeamlessCube || context.nonSeamlessCubeMap, "a sampler table entry that disables seamless cube filtering needs VK_EXT_non_seamless_cube_map because dynamic texture/sampler pairing is not implemented");
                Require(!SingleLevelSamplerWords(elementWords, true, false).has_value(), "a sampler table entry with anisotropic ANISO_OVERRIDE needs dynamic texture mip-level pairing, which is not implemented");
            } catch (const std::runtime_error& error) {
                rejected = error.what();
            }
        }
        if (!rejected.empty()) {
            table.samplerRejections.emplace_back(entry, rejectTableElement(table, words, rejected));
            item.imageAllocations.insert(item.imageAllocations.end(), Abi::SamplerEntryElements, std::numeric_limits<std::size_t>::max());
            continue;
        }
        for (std::uint32_t element = first; element < first + Abi::SamplerEntryElements; ++element) {
            const auto elementWords = guest.subspan(static_cast<std::size_t>(element) * 4u, 4u);
            const bool compareEnable = binding.samplerDepthCompare.at(element);
            if (context.samplerCache != nullptr && !noSamplerCache) {
                samplers.push_back(context.samplerCache->Get(context, elementWords, compareEnable, false));
            } else {
                auto resource = DecodeSamplerResource(elementWords, false);
                resource.compareEnable = compareEnable;
                samplers.push_back(std::make_shared<Sampler>(context, resource));
            }
            samplerSources.push_back({{elementWords[0], elementWords[1], elementWords[2], elementWords[3]}, compareEnable, false});
            item.imageAllocations.push_back(samplers.size() - 1u);
        }
    }
    const auto setSamplers = context.samplerTableUpdateAfterBind ? context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindSamplers : context.limits.maxDescriptorSetSamplers;
    Require(samplers.size() <= setSamplers, "pipeline sampler descriptors exceed device limits");
    if (table.heap) {
        table.tableSamplers = std::move(item.imageAllocations);
        return;
    }
    bindings.push_back(std::move(item));
}

namespace {

void noteVariantSurface(std::uint64_t address, std::size_t members, const GuestTextureResource& chosen, const char* rule) {
    static std::mutex mutex;
    static std::unordered_set<std::uint64_t> seen;
    {
        std::lock_guard lock(mutex);
        if (seen.size() >= 4096u || !seen.insert(address).second) return;
    }
    TextureCounts().variantSurfaces.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "[textures] surface 0x%llx has %zu dynamic-resolution variants in one image table; they share the %ux%u texture (chosen by %s)\n", static_cast<unsigned long long>(address), members, chosen.width, chosen.height, rule);
}

std::vector<std::uint32_t> variantRepresentatives(VkDevice device, const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> imageWords, std::uint64_t elementBase, std::uint64_t capacity) {
    struct Member {
        std::uint32_t element;
        GuestTextureResource resource;
    };
    using FamilyKey = std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>;
    std::map<FamilyKey, std::vector<Member>> families;
    for (std::uint32_t element = 0; element < binding.count && elementBase + element < capacity; ++element) {
        if (element < binding.imageWritten.size() && binding.imageWritten[element]) continue;
        const auto words = imageWords.subspan(static_cast<std::size_t>(element) * 8u, 8u);
        GuestTextureResource resource{};
        try {
            if (IsNullTextureResource(words)) continue;
            resource = DecodeTextureResource(words);
        } catch (const std::runtime_error&) {
            continue;
        }
        const FamilyKey key{resource.baseAddress, resource.pipeBankXor, resource.format, static_cast<std::uint32_t>(resource.tileMode), static_cast<std::uint32_t>(resource.dimension), resource.mipCount, resource.baseLevel, resource.lastLevel, resource.depthOrLastArray, resource.baseArray};
        families[key].push_back({element, resource});
    }
    std::vector<std::uint32_t> representatives;
    for (const auto& [key, members] : families) {
        const auto& first = members.front().resource;
        const auto otherExtent = [&](const Member& member) { return member.resource.width != first.width || member.resource.height != first.height; };
        if (std::none_of(members.begin(), members.end(), otherExtent)) continue;
        const auto withExtent = [&](std::uint32_t width, std::uint32_t height) {
            return std::find_if(members.begin(), members.end(), [&](const Member& member) { return member.resource.width == width && member.resource.height == height; });
        };
        auto chosen = members.end();
        const char* rule = "newest write";
        if (const auto record = StorageTexture::LastWrite(first.baseAddress); record && record->tileMode == first.tileMode && record->dimension == first.dimension && BytesPerElement(record->format) == BytesPerElement(first.format)) chosen = withExtent(record->width, record->height);
        if (chosen == members.end()) {
            rule = "active depth surface";
            if (const auto depth = ActiveDepthExtent(device, first.baseAddress)) chosen = withExtent(depth->width, depth->height);
        }
        if (chosen == members.end()) {
            rule = "largest accessible";
            std::uint64_t chosenBytes = 0;
            std::uint64_t chosenExtent = 0;
            for (auto member = members.begin(); member != members.end(); ++member) {
                try {
                    const auto bytes = DescribeSurface(member->resource).guestBytes;
                    const auto extent = static_cast<std::uint64_t>(member->resource.width) * member->resource.height;
                    if (bytes < chosenBytes || (bytes == chosenBytes && extent <= chosenExtent) || !GuestMemory::Accessible(reinterpret_cast<const void*>(member->resource.baseAddress), static_cast<std::size_t>(bytes))) continue;
                    chosen = member;
                    chosenBytes = bytes;
                    chosenExtent = extent;
                } catch (const std::runtime_error&) {
                }
            }
            if (chosen == members.end()) continue;
        }
        for (const auto& member : members) {
            if (member.resource.width == chosen->resource.width && member.resource.height == chosen->resource.height) continue;
            if (binding.imageShapes.at(member.element) != binding.imageShapes.at(chosen->element) || binding.imageDepthCompare.at(member.element) != binding.imageDepthCompare.at(chosen->element)) continue;
            if (representatives.empty()) representatives.assign(binding.count, NoRepresentative);
            representatives[member.element] = chosen->element;
        }
        noteVariantSurface(first.baseAddress, members.size(), chosen->resource, rule);
    }
    return representatives;
}

std::vector<std::uint32_t> arrayPartners(const ShaderRecompiler::DescriptorBinding& binding, std::span<const std::uint32_t> imageWords, std::uint64_t elementBase, std::uint64_t capacity) {
    struct Candidate {
        std::uint32_t element;
        std::array<std::uint32_t, 8> key;
    };
    std::map<std::array<std::uint32_t, 8>, std::uint32_t> flat;
    std::vector<Candidate> arrays;
    for (std::uint32_t element = 0; element < binding.count && elementBase + element < capacity; ++element) {
        if (element < binding.imageWritten.size() && binding.imageWritten[element]) continue;
        const auto words = imageWords.subspan(static_cast<std::size_t>(element) * 8u, 8u);
        GuestTextureResource resource{};
        try {
            if (IsNullSampledTextureResource(words)) continue;
            resource = DecodeTextureResource(words);
        } catch (const std::runtime_error&) {
            continue;
        }
        std::array<std::uint32_t, 8> key{};
        std::copy(words.begin(), words.end(), key.begin());
        key[3] &= ~(0xfu << 28u);
        const auto shape = binding.imageShapes.at(element);
        if (shape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2D) flat.emplace(key, element);
        else if (shape == ShaderRecompiler::DescriptorImageShape::Image2DArray && resource.dimension == TextureDimension::k2DArray && resource.baseArray == 0u && resource.depthOrLastArray == 0u) arrays.push_back({element, key});
    }
    std::vector<std::uint32_t> partners;
    for (const auto& candidate : arrays) {
        const auto found = flat.find(candidate.key);
        if (found == flat.end() || binding.imageDepthCompare.at(candidate.element) != binding.imageDepthCompare.at(found->second)) continue;
        if (partners.empty()) partners.assign(binding.count, NoRepresentative);
        partners[candidate.element] = found->second;
    }
    return partners;
}

}

void ShaderResources::resolveImageTables() {
    const auto none = [](const std::array<std::size_t, 2>& set) { return set[0] == std::numeric_limits<std::size_t>::max() && set[1] == std::numeric_limits<std::size_t>::max(); };
    if (none(tableBindings) && none(storageTableBindings) && std::none_of(tableStages.begin(), tableStages.end(), [](const TableStage& table) { return table.heap && table.images != nullptr; })) return;
    auto& counters = TextureCounts();
    for (std::size_t set = 0; set < tableBindings.size(); ++set) {
        if (storageTableBindings[set] != std::numeric_limits<std::size_t>::max()) bindings[storageTableBindings[set]].imageAllocations.assign(std::min(tableElements[set], StorageTableCapacity(context)), std::numeric_limits<std::size_t>::max());
        if (tableBindings[set] != std::numeric_limits<std::size_t>::max()) bindings[tableBindings[set]].imageAllocations.assign(std::min(tableElements[set], context.imageTableCapacity), std::numeric_limits<std::size_t>::max());
    }
    for (auto& table : tableStages) {
        if (table.images == nullptr) continue;
        auto* item = !table.heap && tableBindings[table.set] != std::numeric_limits<std::size_t>::max() ? &bindings[tableBindings[table.set]] : nullptr;
        const auto& binding = *table.images;
        const auto faults = table.map[ShaderRecompiler::ImageTableAbi::HeaderElementFaults];
        const auto stageSamplers = std::span<const std::shared_ptr<Sampler>>(samplers).subspan(table.firstSampler, table.samplerCount);
        table.elementTextures.assign(binding.count, std::numeric_limits<std::size_t>::max());
        if (table.heap) table.elementStorage.assign(binding.count, std::numeric_limits<std::size_t>::max());
        const auto capacityMessage = [&] { return "the image table exceeds the device's capacity of " + std::to_string(context.imageTableCapacity) + " images"; };
        std::vector<std::uint32_t> samplerKey;
        for (std::uint32_t sampler = 0; sampler < table.samplerCount; ++sampler) {
            const auto& source = samplerSources.at(table.firstSampler + sampler);
            samplerKey.insert(samplerKey.end(), source.words.begin(), source.words.end());
            samplerKey.push_back((source.compareEnable ? 1u : 0u) | (source.unnormalized ? 2u : 0u) | (source.integerBorder ? 4u : 0u));
        }
        const auto place = [&](std::uint32_t element, std::shared_ptr<Texture> texture, const GuestTextureResource& resource, std::uint64_t guestBytes, bool firstLayer) {
            textures.push_back(std::move(texture));
            textureFirstLayer.push_back(firstLayer);
            describedRanges.push_back({"texture", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress, true});
            table.elementTextures[element] = textures.size() - 1u;
            if (item != nullptr) item->imageAllocations[static_cast<std::size_t>(table.elementBase) + element] = textures.size() - 1u;
        };
        const auto refusal = [&](std::uint32_t element, std::span<const std::uint32_t> words, const std::shared_ptr<Texture>& texture, const GuestTextureResource& resource) -> std::string {
            try {
                RequireFilterMinmax(context, texture->ViewFormat(), binding.imageSamplers.at(element), stageSamplers);
                RequireBorderSwizzle(resource.bcSwizzle, binding.imageSamplers.at(element), stageSamplers);
                RequireNonSeamlessCube(resource.dimension == TextureDimension::kCube, binding.imageSamplers.at(element), stageSamplers);
                RequireDisablesDegammaOn(resource.format, binding.imageSamplers.at(element), stageSamplers);
                const auto numeric = ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) != 0u ? ShaderRecompiler::IrTextureNumericClass::Float : ShaderRecompiler::SampledTextureNumericClass(static_cast<ShaderRecompiler::IrBufferFormat>(resource.format));
                const bool integerImage = numeric == ShaderRecompiler::IrTextureNumericClass::Uint || numeric == ShaderRecompiler::IrTextureNumericClass::Sint;
                for (auto mask = binding.imageSamplers.at(element); mask != 0u; mask &= mask - 1u) {
                    const auto sampler = static_cast<std::uint32_t>(std::countr_zero(mask));
                    if (sampler >= stageSamplers.size()) break;
                    const auto& source = samplerSources.at(table.firstSampler + sampler);
                    const auto descriptor = DecodeSamplerResource(source.words, source.unnormalized, true, source.integerBorder);
                    const bool border = descriptor.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || descriptor.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || descriptor.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                    Require(!border || source.integerBorder == integerImage, "an image table entry's sampler border color numeric class disagrees with its sampled image class, which needs dynamic pairing and is not implemented");
                    Require(!SingleLevelSamplerWords(source.words, true, false).has_value(), "an image table entry paired with anisotropic ANISO_OVERRIDE needs dynamic texture mip-level pairing, which is not implemented");
                }
            } catch (const std::runtime_error& error) {
                return error.what();
            }
            return {};
        };
        const auto bind = [&](std::uint32_t element, std::span<const std::uint32_t> words, std::shared_ptr<Texture> texture, const GuestTextureResource& resource, std::uint64_t guestBytes, bool firstLayer) -> std::string {
            auto message = refusal(element, words, texture, resource);
            if (!message.empty()) {
                table.map[faults + element] = rejectTableElement(table, words, message);
                return message;
            }
            place(element, std::move(texture), resource, guestBytes, firstLayer);
            return {};
        };
        struct DecodedEntry {
            GuestTextureResource resource{};
            std::uint64_t guestBytes = 0;
            bool firstLayer = false;
            std::string rejected;
        };
        const auto decodeEntry = [&](std::uint32_t element, std::span<const std::uint32_t> words) {
            DecodedEntry decoded;
            decoded.resource = DecodeTextureResource(words);
            const auto& resource = decoded.resource;
            const auto shape = binding.imageShapes.at(element);
            const bool depthCompare = element < binding.imageDepthCompare.size() && binding.imageDepthCompare[element];
            decoded.firstLayer = shape == ShaderRecompiler::DescriptorImageShape::Image2D && (resource.dimension == TextureDimension::k2DArray || (depthCompare && resource.dimension == TextureDimension::k1DArray));
            const bool depth1D = depthCompare && ((shape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k1D) || (shape == ShaderRecompiler::DescriptorImageShape::Image2DArray && resource.dimension == TextureDimension::k1DArray));
            if (!decoded.firstLayer && !depth1D && !MatchesGuestDimension(shape, resource.dimension)) {
                decoded.rejected = "AGC graphics: guest texture dimension disagrees with the table entry's image shape (shape " + std::to_string(static_cast<int>(shape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")";
                return decoded;
            }
            decoded.guestBytes = SurfaceGuestBytes(resource);
            if (!GuestMemory::Accessible(reinterpret_cast<const void*>(resource.baseAddress), static_cast<std::size_t>(decoded.guestBytes)) && GuestMemory::CommittedRanges(resource.baseAddress, static_cast<std::size_t>(decoded.guestBytes), false).empty()) {
                decoded.rejected = "AGC graphics: the table entry's surface at 0x" + HexText(resource.baseAddress) + " (0x" + HexText(decoded.guestBytes) + " bytes) is not mapped";
                return decoded;
            }
            static_cast<void>(ResolveTextureFormat(resource.format));
            return decoded;
        };
        struct SampledEntry {
            std::array<std::uint32_t, 8> words{};
            GuestTextureResource resource{};
            std::uint64_t guestBytes = 0;
            VkComponentMapping components{};
            bool depthCompare = false;
        };
        const auto prepareSampled = [&](std::uint32_t element, std::span<const std::uint32_t> words, const DecodedEntry& decoded) {
            SampledEntry sampled;
            sampled.resource = decoded.resource;
            sampled.guestBytes = decoded.guestBytes;
            std::array<std::uint32_t, 8> narrowed{};
            const auto sampledWords = firstLayerWords(words, binding.imageShapes.at(element), narrowed, true);
            std::copy(sampledWords.begin(), sampledWords.end(), sampled.words.begin());
            if (sampledWords.data() != words.data()) {
                sampled.resource = DecodeTextureResource(sampledWords);
                sampled.guestBytes = SurfaceGuestBytes(sampled.resource);
            }
            CompletePendingCpuWrites(context, sampled.resource.baseAddress, static_cast<std::size_t>(sampled.guestBytes));
            if (sampled.resource.dccAddress != 0) CompletePendingCpuWrites(context, sampled.resource.dccAddress, DccKeyBytes(sampled.guestBytes));
            sampled.components = ViewComponents(sampled.resource);
            sampled.depthCompare = binding.imageDepthCompare.at(element);
            return sampled;
        };
        const auto lookupSampled = [&](std::uint32_t element, const SampledEntry& sampled) {
            const VkImageViewType multisampleViewType = binding.imageShapes.at(element) == ShaderRecompiler::DescriptorImageShape::Image2DMultisampleArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            Require(context.imageViewMinLod || EffectiveMinLod(sampled.resource) == 0.0f, "a sampled image table entry has a minimum LOD clamp, which needs VK_EXT_image_view_min_lod");
            return cachedTexture(context, sampled.words, sampled.resource, sampled.components, sampled.guestBytes, sampled.depthCompare, multisampleViewType);
        };
        const auto resolveLive = [&](std::uint32_t element, std::span<const std::uint32_t> words) -> std::string {
            DecodedEntry decoded;
            try {
                decoded = decodeEntry(element, words);
            } catch (const std::runtime_error& error) {
                decoded.rejected = error.what();
            }
            if (!decoded.rejected.empty()) {
                table.map[faults + element] = rejectTableElement(table, words, decoded.rejected);
                return decoded.rejected;
            }
            const auto sampled = prepareSampled(element, words, decoded);
            std::shared_ptr<Texture> texture;
            try {
                texture = lookupSampled(element, sampled);
            } catch (const std::runtime_error& error) {
                table.map[faults + element] = rejectTableElement(table, words, error.what());
                return error.what();
            }
            return bind(element, sampled.words, std::move(texture), sampled.resource, sampled.guestBytes, decoded.firstLayer);
        };
        const auto state = CurrentTableState();
        std::vector<std::shared_ptr<Texture>> cached;
        const bool storageTable = std::find(binding.imageWritten.begin(), binding.imageWritten.end(), true) != binding.imageWritten.end();
        auto* reused = storageTable || item == nullptr ? nullptr : FindReusedTable(context.device, binding, table.imageWords, state);
        if (reused != nullptr && AnyPendingCpuWrites(context)) {
            bool landed = false;
            for (const auto& entry : reused->elements) {
                if (!entry.rejected && CompletePendingCpuWrites(context, entry.resource.baseAddress, static_cast<std::size_t>(entry.guestBytes))) landed = true;
            }
            if (landed && CurrentTableState() != state) reused = nullptr;
        }
        if (reused != nullptr && !reused->representatives.empty() && reused->representatives != variantRepresentatives(context.device, binding, table.imageWords, table.elementBase, context.imageTableCapacity)) reused = nullptr;
        if (reused != nullptr && TableCurrent(*reused) && CachedTableTextures(*reused, cached)) {
            logTableLookups(*reused, cached);
            table.representatives = reused->representatives;
            const bool replay = TableReplay() && reused->replayable && reused->imageSamplers == binding.imageSamplers && reused->samplers == samplerKey;
            if (!replay) reused->bound.assign(binding.count, {});
            auto next = cached.begin();
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto words = table.imageWords.subspan(static_cast<std::size_t>(element) * 8u, 8u);
                const auto& entry = reused->elements[element];
                if (!table.heap && static_cast<std::uint64_t>(table.elementBase) + element >= context.imageTableCapacity) {
                    if (!entry.rejected && !entry.live && !entry.substitute) ++next;
                    table.map[faults + element] = rejectTableElement(table, words, capacityMessage());
                } else if (entry.rejected) {
                    table.map[faults + element] = rejectTableElement(table, words, entry.message);
                } else if (entry.substitute) {
                    continue;
                } else if (entry.live) {
                    auto message = resolveLive(element, words);
                    if (!replay) reused->bound[element] = std::move(message);
                } else {
                    if (!replay) {
                        reused->bound[element] = bind(element, words, std::move(*next++), entry.resource, entry.guestBytes, entry.firstLayer);
                    } else if (!reused->bound[element].empty()) {
                        ++next;
                        table.map[faults + element] = rejectTableElement(table, words, reused->bound[element]);
                    } else {
                        place(element, std::move(*next++), entry.resource, entry.guestBytes, entry.firstLayer);
                    }
                }
            }
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto& entry = reused->elements[element];
                if (!entry.substitute || static_cast<std::uint64_t>(table.elementBase) + element >= context.imageTableCapacity) continue;
                const auto words = table.imageWords.subspan(static_cast<std::size_t>(element) * 8u, 8u);
                const auto representativeWords = table.imageWords.subspan(static_cast<std::size_t>(entry.representative) * 8u, 8u);
                const auto index = table.elementTextures[entry.representative];
                const auto message = index == std::numeric_limits<std::size_t>::max() ? std::string("the dynamic-resolution variant this entry borrows from was rejected") : refusal(element, representativeWords, textures[index], entry.representativeResource);
                if (!message.empty()) {
                    table.map[faults + element] = rejectTableElement(table, words, message);
                    counters.variantRejected.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                table.elementTextures[element] = index;
                item->imageAllocations[static_cast<std::size_t>(table.elementBase) + element] = index;
                counters.variantSubstituted.fetch_add(1, std::memory_order_relaxed);
            }
            if (!replay) {
                reused->imageSamplers = binding.imageSamplers;
                reused->samplers = samplerKey;
                reused->replayable = static_cast<std::uint64_t>(table.elementBase) + binding.count <= context.imageTableCapacity;
            }
            (replay ? counters.tablesReplayed : counters.tablesReused).fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        ReusedTable fresh;
        fresh.bound.assign(binding.count, {});
        bool reusable = state.epoch != 0;
        if (binding.count > 1) table.representatives = variantRepresentatives(context.device, binding, table.imageWords, table.elementBase, context.imageTableCapacity);
        const auto& representatives = table.representatives;
        std::vector<bool> representativeNeeded(representatives.size(), false);
        for (const auto representative : representatives) {
            if (representative != NoRepresentative) representativeNeeded[representative] = true;
        }
        std::unordered_map<std::uint32_t, GuestTextureResource> representativeSources;
        std::vector<std::uint32_t> substitutes;
        const auto partners = binding.count > 1 ? arrayPartners(binding, table.imageWords, table.elementBase, context.imageTableCapacity) : std::vector<std::uint32_t>{};
        std::vector<std::uint32_t> order;
        if (!partners.empty()) {
            reusable = false;
            order.reserve(binding.count);
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                if (partners[element] == NoRepresentative) order.push_back(element);
            }
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                if (partners[element] != NoRepresentative) order.push_back(element);
            }
        }
        for (std::uint32_t position = 0; position < binding.count; ++position) {
            const auto element = order.empty() ? position : order[position];
            const auto words = table.imageWords.subspan(static_cast<std::size_t>(element) * 8u, 8u);
            const auto absolute = static_cast<std::uint64_t>(table.elementBase) + element;
            if (!table.heap && absolute >= context.imageTableCapacity) {
                table.map[faults + element] = rejectTableElement(table, words, capacityMessage());
                reusable = false;
                continue;
            }
            if (!representatives.empty() && representatives[element] != NoRepresentative) {
                if (reusable) {
                    ReusedTableElement substitute;
                    substitute.substitute = true;
                    substitute.representative = representatives[element];
                    fresh.elements.push_back(std::move(substitute));
                }
                substitutes.push_back(element);
                continue;
            }
            if (!partners.empty() && partners[element] != NoRepresentative) {
                const auto partner = partners[element];
                const auto index = table.elementTextures[partner];
                if (index != std::numeric_limits<std::size_t>::max() && textures[index] != nullptr && textures[index]->HasArrayView()) {
                    const auto partnerWords = table.imageWords.subspan(static_cast<std::size_t>(partner) * 8u, 8u);
                    const auto partnerResource = DecodeTextureResource(partnerWords);
                    reusable = false;
                    fresh.bound[element] = bind(element, words, textures[index], partnerResource, DescribeSurface(partnerResource).guestBytes, false);
                    if (fresh.bound[element].empty()) {
                        arrayViewTextures.insert(textures.size() - 1u);
                        counters.arrayViewsShared.fetch_add(1, std::memory_order_relaxed);
                    }
                    continue;
                }
            }
            DecodedEntry decoded;
            try {
                decoded = decodeEntry(element, words);
            } catch (const std::runtime_error& error) {
                decoded.rejected = error.what();
            }
            if (!decoded.rejected.empty()) {
                table.map[faults + element] = rejectTableElement(table, words, decoded.rejected);
                if (reusable) fresh.elements.push_back({{}, {}, {}, 0, false, true, decoded.rejected});
                continue;
            }
            const auto& resource = decoded.resource;
            const auto guestBytes = decoded.guestBytes;
            const bool firstLayer = decoded.firstLayer;
            if (element < binding.imageWritten.size() && binding.imageWritten[element]) {
                reusable = false;
                const auto storageBinding = storageTableBindings[table.set];
                if (!table.heap && storageBinding == std::numeric_limits<std::size_t>::max()) {
                    table.map[faults + element] = rejectTableElement(table, words, "set " + std::to_string(table.set) + " has no image storage table, but the shader stores to its table element " + std::to_string(absolute));
                    continue;
                }
                if (!table.heap && absolute >= StorageTableCapacity(context)) {
                    table.map[faults + element] = rejectTableElement(table, words, "the stored table element " + std::to_string(absolute) + " of set " + std::to_string(table.set) + " is past the image storage table's capacity of " + std::to_string(StorageTableCapacity(context)) + " images");
                    continue;
                }
                CompletePendingCpuWrites(context, resource.baseAddress, static_cast<std::size_t>(guestBytes));
                if (resource.dccAddress != 0) CompletePendingCpuWrites(context, resource.dccAddress, DccKeyBytes(guestBytes));
                const auto mip = std::min(resource.baseLevel, resource.mipCount - 1u);
                if (resource.minLod > mip * 256u) {
                    table.map[faults + element] = rejectTableElement(table, words, "the table entry's storage T# clamps its minimum LOD above the level it addresses, which is not implemented");
                    continue;
                }
                const bool uintStorage = element < binding.imageUintStorage.size() && binding.imageUintStorage[element];
                try {
                    auto texture = cachedStorageTexture(context, words, resource, mip, guestBytes, true);
                    if (uintStorage && texture != nullptr) texture->StorageView(mip, firstLayer, true);
                    storageTextures.push_back(std::move(texture));
                } catch (const std::runtime_error& error) {
                    table.map[faults + element] = rejectTableElement(table, words, error.what());
                    continue;
                }
                storageMips.push_back(mip);
                storageKeys.push_back(resource.dccAddress);
                storageFirstLayer.push_back(firstLayer);
                storageWritten.push_back(true);
                storageAtomic.push_back(element < binding.imageAtomic.size() && binding.imageAtomic[element]);
                storageAtomic64.push_back(element < binding.imageAtomic64.size() && binding.imageAtomic64[element]);
                storageUint.push_back(uintStorage);
                describedRanges.push_back({"storage", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress, true});
                if (table.heap) table.elementStorage[element] = storageTextures.size() - 1u;
                else bindings[storageBinding].imageAllocations[absolute] = storageTextures.size() - 1u;
                continue;
            }
            if (item == nullptr && !table.heap) {
                table.map[faults + element] = rejectTableElement(table, words, "a sampled image table entry has no sampled image table binding");
                reusable = false;
                continue;
            }
            const auto sampled = prepareSampled(element, words, decoded);
            std::shared_ptr<Texture> texture;
            try {
                texture = lookupSampled(element, sampled);
            } catch (const std::runtime_error& error) {
                table.map[faults + element] = rejectTableElement(table, words, error.what());
                if (reusable) fresh.elements.push_back({{}, {}, {}, 0, false, true, error.what()});
                continue;
            }
            if (reusable) {
                const auto& log = lookupLog();
                const auto* looked = log.empty() ? nullptr : &log.back();
                if (looked != nullptr && looked->object == texture.get() && looked->source == nullptr && looked->generation != 0 && looked->keys == DccKeys::Uncompressed && sampled.resource.dccAddress == 0) {
                    fresh.elements.push_back({MakeTextureKey(context.device, sampled.words, sampled.components, sampled.depthCompare), texture, sampled.resource, sampled.guestBytes, firstLayer, false, {}});
                    fresh.queries.push_back({sampled.resource.baseAddress, static_cast<std::size_t>(sampled.guestBytes), looked->generation});
                    fresh.ranges.emplace_back(sampled.resource.baseAddress, sampled.resource.baseAddress + sampled.guestBytes);
                } else {
                    fresh.elements.push_back({{}, {}, {}, 0, firstLayer, false, {}, true});
                }
            }
            if (!representativeNeeded.empty() && representativeNeeded[element]) representativeSources.emplace(element, sampled.resource);
            fresh.bound[element] = bind(element, sampled.words, std::move(texture), sampled.resource, sampled.guestBytes, firstLayer);
        }
        for (const auto element : substitutes) {
            const auto representative = representatives[element];
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(representative) * 8u, 8u);
            const auto index = table.elementTextures[representative];
            const auto source = representativeSources.find(representative);
            const auto message = index == std::numeric_limits<std::size_t>::max() || source == representativeSources.end() ? std::string("the dynamic-resolution variant this entry borrows from was rejected") : refusal(element, words, textures[index], source->second);
            if (reusable) {
                if (fresh.elements[representative].rejected || source == representativeSources.end()) reusable = false;
                else fresh.elements[element].representativeResource = source->second;
            }
            if (!message.empty()) {
                table.map[faults + element] = rejectTableElement(table, words, message);
                counters.variantRejected.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            table.elementTextures[element] = index;
            item->imageAllocations[static_cast<std::size_t>(table.elementBase) + element] = index;
            counters.variantSubstituted.fetch_add(1, std::memory_order_relaxed);
        }
        counters.tablesResolved.fetch_add(1, std::memory_order_relaxed);
        const bool anyKept = std::any_of(fresh.elements.begin(), fresh.elements.end(), [](const ReusedTableElement& element) { return !element.live; });
        if (reusable && anyKept && CurrentTableState() == state) {
            const auto serial = StorageTexture::PendingSerial();
            if (!StorageTexture::AnyPendingOverlaps(fresh.ranges) && StorageTexture::PendingSerial() == serial) {
                fresh.device = context.device;
                fresh.state = state;
                fresh.pendingSerial = serial;
                fresh.imageSamplers = binding.imageSamplers;
                fresh.samplers = std::move(samplerKey);
                fresh.replayable = static_cast<std::uint64_t>(table.elementBase) + binding.count <= context.imageTableCapacity;
                fresh.representatives = table.representatives;
                RememberTable(std::move(fresh), binding, table.imageWords);
            }
        }
    }
    for (auto& table : tableStages) table.imageWords = {};
    reportTextureCounters();
}

std::uint64_t SampledLookupKeyScans() {
    return sampledLookupKeyScans;
}

std::uint64_t SampledLookupKeyProofs() {
    return sampledLookupKeyProofs;
}

DeviceTableTexture LookupDeviceTableTexture(const Context& context, std::span<const std::uint32_t, 8> descriptor, const DeviceTableView& view, std::uint32_t samplerClass) {
    namespace Abi = ShaderRecompiler::DeviceImageTableAbi;
    std::array<std::uint32_t, 8> words{};
    std::copy(descriptor.begin(), descriptor.end(), words.begin());
    if (view.identitySwizzle) words[3] = (words[3] & ~0xfffu) | ShaderRecompiler::ShaderImageIdentitySwizzle;
    const auto resource = DecodeTextureResource(words);
    const bool firstLayer = view.shape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
    if (!firstLayer && !MatchesGuestDimension(view.shape, resource.dimension)) throw std::runtime_error("AGC graphics: guest texture dimension disagrees with the table entry's image shape (shape " + std::to_string(static_cast<int>(view.shape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
    const auto guestBytes = DescribeSurface(resource).guestBytes;
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(resource.baseAddress), static_cast<std::size_t>(guestBytes)) && GuestMemory::CommittedRanges(resource.baseAddress, static_cast<std::size_t>(guestBytes), false).empty()) throw std::runtime_error("AGC graphics: the table entry's surface at 0x" + HexText(resource.baseAddress) + " (0x" + HexText(guestBytes) + " bytes) is not mapped");
    static_cast<void>(ResolveTextureFormat(resource.format));
    CompletePendingCpuWrites(context, resource.baseAddress, static_cast<std::size_t>(guestBytes));
    if (resource.dccAddress != 0) CompletePendingCpuWrites(context, resource.dccAddress, DccKeyBytes(guestBytes));
    DeviceTableTexture result;
    result.address = resource.baseAddress;
    result.guestBytes = guestBytes;
    result.generation = GuestMemory::CollectWrites(resource.baseAddress, static_cast<std::size_t>(guestBytes));
    auto texture = cachedTexture(context, words, resource, ViewComponents(resource), guestBytes, view.depthCompare);
    const auto& log = lookupLog();
    const auto* looked = !log.empty() && log.back().object == texture.get() ? &log.back() : nullptr;
    if (looked != nullptr && looked->depth) {
        result.unservable = "the surface at 0x" + HexText(resource.baseAddress) + " is a depth surface, whose layout transitions per use the device image table does not make";
        return result;
    }
    if (texture->CopyKind() != Texture::PerUseCopy::None) {
        result.unservable = "the texture at 0x" + HexText(resource.baseAddress) + " is a copy remade or refreshed per use, which the device image table does not make";
        return result;
    }
    if ((samplerClass & Abi::SamplerFilterMinmax) != 0u) {
        VkFormatProperties properties{};
        context.formatProperties(context.physical, texture->ViewFormat(), &properties);
        if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT_EXT) == 0) throw std::runtime_error("a sampled texture whose format " + std::to_string(texture->ViewFormat()) + " does not support min/max filtering is sampled through a min or max reduction sampler with linear filtering");
    }
    if ((samplerClass & Abi::SamplerOpaqueBlackBorder) != 0u && resource.bcSwizzle != 0u && resource.bcSwizzle != 4u) throw std::runtime_error("a sampled texture whose BC swizzle " + std::to_string(resource.bcSwizzle) + " moves the alpha channel is sampled through an opaque black border, which is not implemented");
    if ((samplerClass & Abi::SamplerNonSeamlessCube) != 0u && resource.dimension == TextureDimension::kCube && !context.nonSeamlessCubeMap) throw std::runtime_error("a cube texture is sampled through a sampler that disables seamless cube filtering, which needs VK_EXT_non_seamless_cube_map");
    const auto numeric = ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) != 0u ? ShaderRecompiler::IrTextureNumericClass::Float : ShaderRecompiler::SampledTextureNumericClass(static_cast<ShaderRecompiler::IrBufferFormat>(resource.format));
    const bool integerImage = numeric == ShaderRecompiler::IrTextureNumericClass::Uint || numeric == ShaderRecompiler::IrTextureNumericClass::Sint;
    if (((samplerClass & Abi::SamplerFloatBorder) != 0u && integerImage) || ((samplerClass & Abi::SamplerIntegerBorder) != 0u && !integerImage)) throw std::runtime_error("an image table entry's sampler border color numeric class disagrees with its sampled image class, which needs dynamic pairing and is not implemented");
    if ((samplerClass & Abi::SamplerAnisoOverride) != 0u) throw std::runtime_error("an image table entry paired with anisotropic ANISO_OVERRIDE needs dynamic texture mip-level pairing, which is not implemented");
    result.view = firstLayer ? texture->FirstLayerView() : texture->View();
    result.layout = texture->Layout();
    result.texture = std::move(texture);
    result.words = words;
    result.depthCompare = view.depthCompare;
    return result;
}

bool DeviceTableTextureCurrent(const Context& context, const std::array<std::uint32_t, 8>& words, bool depthCompare, const Texture& texture) {
    const auto key = MakeTextureKey(context.device, words, ViewComponents(DecodeTextureResource(words)), depthCompare);
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    const auto it = findTexture(cache, key);
    if (it == cache.entries.end() || it->texture.get() != &texture || it->source != nullptr || it->keys != DccKeys::Uncompressed || it->committed.has_value()) return false;
    if (Recorder::SnapshotWriteOverlaps(it->address, it->bytes.size())) return false;
    const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::TextureCompare);
    return GuestMemory::EqualsCommitted(it->address, it->bytes);
}

std::uint64_t SampledTexturesCleared() {
    return TextureCounts().cleared.load(std::memory_order_relaxed);
}

std::uint64_t SampledTexturesImported() {
    return TextureCounts().imported.load(std::memory_order_relaxed);
}

ImageTableReuseCounts ImageTableReuses() {
    const auto& counters = TextureCounts();
    return {counters.tablesReused.load(std::memory_order_relaxed) + counters.tablesReplayed.load(std::memory_order_relaxed), counters.tablesResolved.load(std::memory_order_relaxed), counters.tablesReplayed.load(std::memory_order_relaxed), counters.tableWritesKept.load(std::memory_order_relaxed), counters.tableWritesMade.load(std::memory_order_relaxed), counters.variantSurfaces.load(std::memory_order_relaxed), counters.variantSubstituted.load(std::memory_order_relaxed), counters.variantRejected.load(std::memory_order_relaxed), counters.arrayViewsShared.load(std::memory_order_relaxed)};
}

void ShaderResources::finishImageTables() {
    namespace Abi = ShaderRecompiler::ImageTableAbi;
    if (tableStages.empty()) return;
    const auto overlaps = [](std::uint64_t base, std::uint64_t size, std::uint64_t address, std::uint64_t bytes) { return size != 0 && bytes != 0 && (base <= address ? address - base < size : base - address < bytes); };
    std::vector<BdaResources::ImageTableFaults> faults;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (auto& table : tableStages) {
        auto& map = table.map;
        for (const auto& [entry, code] : table.samplerRejections) map[map[Abi::HeaderSamplerFaults] + entry] = code;
        for (std::uint32_t index = 0; index < map[Abi::HeaderTableCount]; ++index) {
            const auto header = Abi::TableHeader(index);
            if (map[header + Abi::TableFault] != 0u) continue;
            const auto base = static_cast<std::uint64_t>(map[header + Abi::TableBaseLow]) | (static_cast<std::uint64_t>(map[header + Abi::TableBaseHigh]) << 32u);
            const auto size = static_cast<std::uint64_t>(map[header + Abi::TableSizeLow]) | (static_cast<std::uint64_t>(map[header + Abi::TableSizeHigh]) << 32u);
            const auto tableOverlaps = [&](std::uint64_t address, std::uint64_t bytes) {
                if (overlaps(base, size, address, bytes)) return true;
                if (index >= table.program->imageTableReadRanges.size()) return false;
                const auto& reads = table.program->imageTableReadRanges[index];
                return std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return overlaps(range.first, range.second, address, bytes); });
            };
            bool written = false;
            for (const auto& allocation : allocations) written = written || (allocation.guest && allocation.written && tableOverlaps(allocation.address, allocation.size));
            for (std::size_t image = 0; image < storageTextures.size(); ++image) written = written || (storageWritten[image] && storageTextures[image] != nullptr && tableOverlaps(storageTextures[image]->Descriptor().baseAddress, storageTextures[image]->GuestBytes()));
            if (!written) continue;
            ShaderRecompiler::ImageTableEntryPoison poison{};
            poison.resource = index;
            poison.reason = static_cast<std::uint32_t>(Abi::PoisonReason::HeapWritten);
            table.poison.push_back(poison);
            table.messages.push_back(std::make_shared<const std::string>("the table at 0x" + HexText(base) + " is written by its own dispatch"));
            map[header + Abi::TableFault] = Abi::PoisonCode(static_cast<std::uint32_t>(table.poison.size() - 1u));
            faultChecks = true;
        }
        map[Abi::HeaderPoisonCount] = static_cast<std::uint32_t>(table.poison.size());
        auto& allocation = allocations.at(table.mapAllocation);
        Require(allocation.buffer != nullptr && allocation.size == map.size() * sizeof(std::uint32_t), "image table map buffer disagrees with the map");
        std::memcpy(allocation.buffer->Bytes().data(), map.data(), allocation.size);
        faults.push_back({IrStage(table.stage), table.program->imageTableShader, table.poison, table.messages});
        ranges.insert(ranges.end(), table.program->imageTableRanges.begin(), table.program->imageTableRanges.end());
    }
    for (auto& table : tableStages) {
        table.program = nullptr;
        table.images = nullptr;
    }
    if (bda == nullptr) return;
    bda->SetImageTables(std::move(faults), std::move(ranges));
}

std::vector<PresyncSurface> ShaderResources::PresyncSurfaces() const {
    std::vector<PresyncSurface> surfaces;
    // A surface's lookup reads guest memory on the CPU unless it is served GPU-direct from a host
    // import: a storage image of imported memory uploads and refreshes from the import, and a
    // sampled texture of one views that image (SampledFromStorageEligible), so neither waits for
    // the producer.
    const auto consider = [&](std::uint32_t format, std::uint64_t address, std::uint64_t guestBytes, bool sampled) {
        const bool gpuDirect = sampled ? SampledFromStorageEligible(context, format, address, guestBytes) : HostImportCovers(context, address, static_cast<std::size_t>(guestBytes));
        if (!gpuDirect) surfaces.push_back({address, guestBytes, false});
    };
    static const bool keyRanges = std::getenv("APS5_NO_DCC_PRESYNC") == nullptr;
    const auto considerKeys = [&](std::uint64_t dccAddress, std::uint64_t guestBytes) {
        const auto keyBytes = DccKeyBytes(guestBytes);
        if (keyRanges && dccAddress != 0 && keyBytes != 0) surfaces.push_back({dccAddress, keyBytes, true});
    };
    if (completed) {
        // A cached object (the resource cache): Revalidate repeats the lookups of the objects the
        // build made, which never change afterwards (a lookup returning another object fails the
        // revalidation instead), so the surfaces come from the build's own record of them. A
        // sampled texture viewing a storage image refreshes like that image; a snapshot texture
        // over an eligible surface is replaced by a view without reading. Storage images are
        // checked on their own surface (a mip chain can exceed the element's descriptor). Read
        // without the device lock: these members are fixed once the build completed.
        std::size_t textureIndex = 0;
        for (const auto& range : describedRanges) {
            if (std::strcmp(range.kind, "texture") == 0 && textureIndex < textures.size()) {
                const auto& texture = textures[textureIndex++];
                if (texture->ViewsStorageImage()) consider(range.format, range.address, range.bytes, false);
                else consider(range.format, range.address, range.bytes, true);
                considerKeys(range.dccAddress, range.bytes);
            }
        }
        for (std::size_t index = 0; index < storageTextures.size(); ++index) {
            if (storageTextures[index] == nullptr || (index != 0 && storageTextures[index] == storageTextures[index - 1])) continue;
            const auto& image = *storageTextures[index];
            consider(image.Descriptor().format, image.Descriptor().baseAddress, image.GuestBytes(), false);
            considerKeys(image.Descriptor().dccAddress, image.GuestBytes());
        }
        return surfaces;
    }
    if (!imageRecords.empty()) {
        for (const auto& record : imageRecords) {
            if (!record.decoded) continue;
            consider(record.resource.format, record.resource.baseAddress, record.guestBytes, record.sampled);
            considerKeys(record.resource.dccAddress, record.guestBytes);
        }
        return surfaces;
    }
    // Stage A has not run (an address-based build, or the pass is off): decoded from the shader.
    if (deferredCompute.program == nullptr) return surfaces;
    forEachImageElement(*deferredCompute.program, [&](const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t, std::span<const std::uint32_t> words) {
        if (IsNullTextureResource(words)) return;
        const auto resource = DecodeTextureResource(words);
        const auto guestBytes = SurfaceGuestBytes(resource);
        consider(resource.format, resource.baseAddress, guestBytes, binding.kind == ShaderRecompiler::DescriptorKind::SampledImage);
        considerKeys(resource.dccAddress, guestBytes);
    });
    for (const auto& binding : deferredCompute.program->bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::ImageTable) continue;
        for (std::size_t element = 0; element < binding.guestDescriptor.size() / 8u; ++element) {
            try {
                const auto resource = DecodeTextureResource(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(element * 8u, 8u));
                const auto guestBytes = SurfaceGuestBytes(resource);
                consider(resource.format, resource.baseAddress, guestBytes, true);
                considerKeys(resource.dccAddress, guestBytes);
            } catch (const std::runtime_error&) {
            }
        }
    }
    return surfaces;
}

ShaderResources::~ShaderResources() {
    release();
}

std::shared_ptr<const TableDescriptors> ShaderResources::keptTable(std::size_t set, std::shared_ptr<const std::vector<std::weak_ptr<Texture>>>& held) noexcept {
    const auto& table = tableWrites[set];
    if (!table.written || KeptSets() == 0u || table.firstImage + table.images > descriptorImages.size()) return nullptr;
    try {
        if (held == nullptr) {
            std::vector<std::weak_ptr<Texture>> weak;
            weak.reserve(textures.size());
            for (const auto& texture : textures) {
                if (texture != nullptr) weak.push_back(texture);
            }
            held = std::make_shared<const std::vector<std::weak_ptr<Texture>>>(std::move(weak));
        }
        auto kept = std::make_shared<TableDescriptors>();
        kept->binding = table.binding;
        kept->runs = table.runs;
        kept->images.assign(descriptorImages.begin() + static_cast<std::ptrdiff_t>(table.firstImage), descriptorImages.begin() + static_cast<std::ptrdiff_t>(table.firstImage + table.images));
        kept->textures = held;
        return kept;
    } catch (...) {
        return nullptr;
    }
}

void ShaderResources::release() noexcept {
    // A set from the cache's pool chain goes back to it; a dedicated pool dies with its set.
    if (ownDrawSets != nullptr) ownDrawSets->Drain(true);
    if (fragmentSetsReady.load(std::memory_order_acquire) && fragmentSets != nullptr) fragmentSets->Drain(true);
    std::shared_ptr<const std::vector<std::weak_ptr<Texture>>> held;
    if (cacheAllocation.pool && context.descriptorCache != nullptr) context.descriptorCache->Free(cacheAllocation, keptTable(0, held));
    if (pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    if (_layout && ownsLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, _layout, nullptr);
    if (fragment.cacheAllocation.pool && context.descriptorCache != nullptr) context.descriptorCache->Free(fragment.cacheAllocation, keptTable(1, held));
    if (fragment.pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, fragment.pool, nullptr);
    if (fragment.layout && fragment.ownsLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, fragment.layout, nullptr);
    cacheAllocation = {};
    pool = VK_NULL_HANDLE;
    _set = VK_NULL_HANDLE;
    _layout = VK_NULL_HANDLE;
    fragment = {};
    tableWrites = {};
}

std::uint32_t ShaderResources::setOf(const Binding& binding) const {
    constexpr auto count = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    if (binding.set >= 0) return static_cast<std::uint32_t>(binding.set);
    if (heapStage) return 0u;
    return splitSets && binding.layout.binding / count == static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Stage::Fragment) ? 1u : 0u;
}

VkDescriptorSetLayout ShaderResources::Layout() const {
    return _layout;
}

std::vector<VkDescriptorSetLayoutBinding> ShaderResources::LayoutBindings() const {
    std::vector<VkDescriptorSetLayoutBinding> result;
    for (const auto& binding : bindings) result.push_back(binding.layout);
    return result;
}

bool ShaderResources::LayoutUpdateAfterBind() const {
    return updateAfterBind;
}

ShaderResources::DrawBindings::~DrawBindings() {
    if (cache == nullptr) return;
    const bool park = ownSets != nullptr && RuntimeSwitches::Set(RuntimeSwitch::OwnDrawSets);
    const bool parkFragment = parkedSets != nullptr && RuntimeSwitches::Set(RuntimeSwitch::FragmentParkedSets);
    const auto release = [&](std::size_t index, VkDescriptorSetLayout setLayout, DescriptorCache::SetAllocation& set) {
        if (set.set == VK_NULL_HANDLE || lent[index]) return;
        if (park && owned[index] && ownSets->Park(index, set)) return;
        if (parkFragment && parkable[index] && parkedSets->Park(index, set, parkStamps[index])) return;
        if (setLayout != VK_NULL_HANDLE) cache->Recycle(setLayout, set);
        else cache->Free(set);
    };
    release(0, layout, allocation);
    release(1, fragmentLayout, fragmentAllocation);
}

namespace {

bool movedCensus() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return stats;
}

std::array<std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ShaderResources::MovedCount::Count)>, ShaderResources::MovedCountArms> movedCounts{};

void noteMoved(ShaderResources::MovedCount what) {
    if (!movedCensus() || what == ShaderResources::MovedCount::Count) return;
    const auto arm = (RuntimeSwitches::Set(RuntimeSwitch::MovedInPlace) ? 1u : 0u) | (RuntimeSwitches::Set(RuntimeSwitch::DirectBuffersInPlace) ? 2u : 0u);
    movedCounts[arm][static_cast<std::size_t>(what)].fetch_add(1, std::memory_order_relaxed);
}

constexpr std::array<const char*, static_cast<std::size_t>(ShaderResources::MovedCount::Count)> MovedCountNames{
    "bindings differ", "patched data", "template buffer not in guest memory", "written or emptied", "over the range limit", "snapshot budget", "unmapped", "template writes", "pending write", "pending storage image", "shadowed",
    "same address, raw size differs and stays unequal", "same address, raw size differs but equal once clamped", "finisher: same address, raw size differs and stays unequal", "finisher: same address, raw size differs but equal once clamped",
    "ranges bound in place", "hits with a range in place",
    "not in place: template not eligible", "not in place: pending storage image", "not in place: shadowed", "not in place: CPU publish pending", "not in place: label", "not in place: queued store", "not in place: copied writer", "not in place: over the range", "not in place: no import", "not in place: misaligned",
    "prepared hits committed with a range in place", "prepared ranges committed in place", "prepared in place refused: forget, import or mapping serial moved", "prepared in place refused: switch off at the commit", "prepared in place refused: template writes"};
static_assert(MovedCountNames.back() != nullptr, "every moved-buffer count needs a name");

}

const char* ShaderResources::MovedCountName(MovedCount count) {
    return count < MovedCount::Count ? MovedCountNames[static_cast<std::size_t>(count)] : "?";
}

std::array<std::uint64_t, static_cast<std::size_t>(ShaderResources::MovedCount::Count)> ShaderResources::TakeMovedCounts(std::size_t arm) {
    std::array<std::uint64_t, static_cast<std::size_t>(MovedCount::Count)> taken{};
    if (arm >= MovedCountArms) return taken;
    for (std::size_t index = 0; index < taken.size(); ++index) taken[index] = movedCounts[arm][index].exchange(0, std::memory_order_relaxed);
    return taken;
}

std::optional<std::vector<ShaderResources::MovedBuffer>> ShaderResources::MovedReadOnlyBuffers(std::span<const CompiledShader> shaders, Recorder& recorder, const ColorTarget* target, bool* patchedData, MoveTicket* ticket) const {
    std::vector<MovedBuffer> moved;
    std::size_t movedBytes = 0;
    struct RangesStorage {};
    auto& ranges = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::size_t>>, RangesStorage>();
    ranges.clear();
    struct RangeMovesStorage {};
    auto& rangeMoves = HostThreadLocal<std::vector<std::size_t>, RangeMovesStorage>();
    rangeMoves.clear();
    const bool inPlaceRule = RuntimeSwitches::Set(RuntimeSwitch::MovedInPlace);
    const bool placeBehind = !inPlaceRule && MovedBuffersInPlace() && context.hostImportAlignment != 0;
    const auto refuse = [](MovedCount why) {
        noteMoved(why);
        return std::optional<std::vector<MovedBuffer>>{};
    };
    if ((_set == VK_NULL_HANDLE && fragment.set == VK_NULL_HANDLE && !heapStage) || (usesBda && !spaceReuse)) return moved;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) return refuse(MovedCount::Shape);
        const bool heapShader = HeapShader(shader);
        for (const auto& binding : shader.program->bindings) {
            if (heapShader && (DataRole(binding.role) || binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers)) continue;
            if (DataRole(binding.role)) {
                const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
                if (kept == bindings.end() || kept->allocations.size() != 1) return refuse(MovedCount::Shape);
                const auto index = kept->allocations.front();
                const auto& item = allocations[index];
                const auto& guestWords = DrawGuestWords(shader, binding);
                if (item.guest || item.buffer == nullptr || item.size != guestWords.size() * sizeof(std::uint32_t)) return refuse(MovedCount::Shape);
                const bool patched = std::any_of(dataPatches.begin(), dataPatches.end(), [&](const DataPatch& patch) { return patch.allocation == index; });
                const bool same = !item.dataWords.empty() ? item.dataWords == guestWords : !patched && std::memcmp(item.buffer->Bytes().data(), guestWords.data(), item.size) == 0;
                if (same) continue;
                if (item.dataWords.empty() && patched) {
                    // A moved range walked before the patch that is not clear refuses the draw first.
                    if (patchedData != nullptr && movedRangesClear(recorder, ranges, nullptr, false)) *patchedData = true;
                    return refuse(MovedCount::Patched);
                }
                if (moved.empty()) moved.reserve(allocations.size());
                moved.push_back({index, 0, item.size, guestWords});
                continue;
            }
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
            const auto& guestWords = DrawGuestWords(shader, binding);
            const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
            if (kept == bindings.end() || kept->allocations.size() != binding.count || guestWords.size() != static_cast<std::size_t>(binding.count) * 4u) return refuse(MovedCount::Shape);
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = guestWords.data() + static_cast<std::size_t>(element) * 4u;
                const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
                const auto address = descriptor.Base48();
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                const auto rawSize = BytesUpToTarget(address, descriptor.GetSize(), descriptor.Stride(), written ? nullptr : target);
                const auto index = kept->allocations[element];
                const auto& item = allocations[index];
                const bool resized = item.guest && item.address == address && item.size != rawSize;
                auto size = rawSize;
                if (inPlaceRule && !(item.guest && item.address == address && item.size == size)) size = BytesUpToTarget(address, guestBufferBytes(std::span<const std::uint32_t>(words, 4)), descriptor.Stride(), written ? nullptr : target);
                const bool empty = size == 0 || address == 0;
                if (!item.guest) {
                    if (empty) continue;
                    return refuse(MovedCount::NotGuest);
                }
                if (resized) noteMoved(item.size == size ? MovedCount::ResizedClamped : MovedCount::Resized);
                if (item.address == address && item.size == size) continue;
                if (written || empty || item.written) return refuse(MovedCount::Written);
                if (size > context.limits.maxStorageBufferRange) return refuse(MovedCount::Oversize);
                const auto begin = address - item.adjustment;
                const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(size, item.adjustment));
                if (!inPlaceRule && !placeBehind && bytes > maxDrawSnapshotBytes - movedBytes) return refuse(MovedCount::Budget);
                ranges.emplace_back(begin, bytes);
                if (moved.empty()) moved.reserve(allocations.size());
                rangeMoves.push_back(moved.size());
                moved.push_back({index, address, static_cast<std::size_t>(size)});
                movedBytes += bytes;
            }
        }
    }
    auto why = MovedCount::Count;
    if (!inPlaceRule && !placeBehind) {
        if (!movedRangesClear(recorder, ranges, ticket, false, nullptr, &why)) return refuse(why);
        return moved;
    }
    if (!inPlaceRule) {
        struct BehindStorage {};
        auto& behind = HostThreadLocal<std::vector<std::uint8_t>, BehindStorage>();
        if (!movedRangesClear(recorder, ranges, ticket, false, nullptr, &why, &behind)) return refuse(why);
        std::size_t copied = 0;
        for (std::size_t at = 0; at < ranges.size(); ++at) {
            if (ranges[at].second > maxDrawSnapshotBytes - copied) return refuse(MovedCount::Budget);
            auto& entry = moved[rangeMoves[at]];
            if (behind[at] == 0) {
                copied += ranges[at].second;
                continue;
            }
            if (!movedInPlace(recorder, allocations[entry.allocation], entry.address, entry.size, entry)) return refuse(MovedCount::PendingWrite);
        }
        return moved;
    }
    struct PlacedStorage {};
    auto& placed = HostThreadLocal<std::vector<InPlaceRange>, PlacedStorage>();
    if (!movedRangesClear(recorder, ranges, ticket, false, &placed, &why)) return refuse(why);
    std::size_t copied = 0;
    for (std::size_t at = 0; at < ranges.size(); ++at) {
        if (placed[at].buffer == VK_NULL_HANDLE) copied += ranges[at].second;
    }
    if (copied > maxDrawSnapshotBytes) return refuse(MovedCount::Budget);
    bool any = false;
    for (std::size_t at = 0; at < ranges.size(); ++at) {
        if (placed[at].buffer == VK_NULL_HANDLE) continue;
        auto& entry = moved[rangeMoves[at]];
        entry.inPlace = placed[at].buffer;
        entry.importOffset = placed[at].offset;
        entry.inPlaceOffset = placed[at].offset;
        entry.adjustment = allocations[entry.allocation].adjustment;
        entry.begin = ranges[at].first;
        entry.bytes = ranges[at].second;
        entry.behind = true;
        noteMoved(MovedCount::InPlaceRanges);
        any = true;
    }
    if (any) noteMoved(MovedCount::InPlaceHits);
    return moved;
}

namespace {

bool zoneCensus() {
    static const bool on = std::getenv("APS5_ZONE_CENSUS") != nullptr;
    return on;
}

void noteZoneCensus(bool countsClear, bool testsBusy) {
    static std::atomic<std::uint64_t> counts[4];
    static std::atomic<std::uint64_t> seen{0};
    counts[(countsClear ? 0u : 2u) + (testsBusy ? 1u : 0u)].fetch_add(1, std::memory_order_relaxed);
    if ((seen.fetch_add(1, std::memory_order_relaxed) + 1u) % (std::uint64_t{1} << 20) != 0) return;
    const auto wide = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    std::fprintf(stderr, "[zone-census] per 2^20 moved ranges: %llu clear by the zone counts and by the tests, %llu clear by the counts but busy by the tests (unsound), %llu busy by the counts but clear by the tests, %llu busy by both\n", wide(counts[0]), wide(counts[1]), wide(counts[2]), wide(counts[3]));
}

struct InPlaceExactCensus {
    static constexpr std::size_t Users = static_cast<std::size_t>(ShaderResources::InPlaceUser::Count);
    std::array<std::atomic<std::uint64_t>, Users> passed{};
    std::array<std::atomic<std::uint64_t>, Users> refused{};
    std::atomic<std::int64_t> window{-1};
};

void noteInPlaceExact(ShaderResources::InPlaceUser user, bool refused) {
    if (!movedCensus() || user >= ShaderResources::InPlaceUser::Count) return;
    static InPlaceExactCensus census;
    const auto index = static_cast<std::size_t>(user);
    census.passed[index].fetch_add(1, std::memory_order_relaxed);
    if (refused) census.refused[index].fetch_add(1, std::memory_order_relaxed);
    const std::int64_t window = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (census.window.exchange(window, std::memory_order_relaxed) == window) return;
    std::array<unsigned long long, InPlaceExactCensus::Users> tested{};
    std::array<unsigned long long, InPlaceExactCensus::Users> stricter{};
    for (std::size_t at = 0; at < InPlaceExactCensus::Users; ++at) {
        tested[at] = static_cast<unsigned long long>(census.passed[at].exchange(0, std::memory_order_relaxed));
        stricter[at] = static_cast<unsigned long long>(census.refused[at].exchange(0, std::memory_order_relaxed));
    }
    std::fprintf(stderr, "[in-place-exact] 10 s, APS5_IN_PLACE_EXACT set: ranges today's in-place test passed, and of them refused by the stricter test for a storage image whose write-back to guest memory was in progress on another thread: moved buffers of draw template hits (APS5_MOVED_IN_PLACE) %llu, refused %llu; written buffers of dispatches (APS5_DISPATCH_MOVED_WRITES) %llu, refused %llu; ranges of prepared draws committed in place %llu, refused %llu\n", tested[0], stricter[0], tested[1], stricter[1], tested[2], stricter[2]);
}

}

bool ShaderResources::movedRangesClear(Recorder& recorder, std::span<const std::pair<std::uint64_t, std::size_t>> ranges, MoveTicket* ticket, bool accessible, std::vector<InPlaceRange>* inPlace, MovedCount* why, std::vector<std::uint8_t>* behind) const {
    const bool ticketed = ticket != nullptr && ticket->ranges != nullptr;
    const bool stamped = ticketed && ticket->valid && ticket->recorder == recorder.Id() && ticket->pending == StorageTexture::PendingSerial() && ticket->allocations == GuestAllocations::GuestAllocationsGeneration_nid_postfix() && ticket->epoch == GuestMemory::CollectEpoch();
    struct FreshStorage {};
    auto& fresh = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, FreshStorage>();
    fresh.clear();
    if (inPlace != nullptr) inPlace->assign(ranges.size(), InPlaceRange{});
    if (behind != nullptr) behind->assign(ranges.size(), 0);
    const auto refuse = [why](MovedCount cause) {
        if (why != nullptr) *why = cause;
        return false;
    };
    for (std::size_t at = 0; at < ranges.size(); ++at) {
        const auto begin = ranges[at].first;
        const auto bytes = ranges[at].second;
        const auto range = std::pair<std::uint64_t, std::uint64_t>{begin, begin + bytes};
        if (zoneCensus()) noteZoneCensus(PendingZones::Clear(begin, bytes), recorder.PendingWriteOverlaps(begin, bytes) || PendingStorageOverlaps(begin, bytes, nullptr));
        if (ticketed) ++ticket->checked;
        if (stamped && std::find(ticket->ranges->begin(), ticket->ranges->end(), range) != ticket->ranges->end() && !recorder.WriteNotedSince(ticket->note, begin, bytes)) {
            ++ticket->trusted;
            if (AnyShadowedOverlaps(begin, bytes)) return refuse(MovedCount::Shadowed);
        } else {
            if (!accessible && !GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return refuse(MovedCount::Unmapped);
            if (guestMemory.WritesOverlap(begin, bytes)) return refuse(MovedCount::TemplateWrites);
            if (recorder.PendingWriteOverlaps(begin, bytes)) {
                if (behind != nullptr) {
                    if (PendingStorageOverlaps(begin, bytes, nullptr)) return refuse(MovedCount::PendingStorage);
                    if (AnyShadowedOverlaps(begin, bytes)) return refuse(MovedCount::Shadowed);
                    (*behind)[at] = 1;
                    continue;
                }
                if (inPlace == nullptr || !takeMovedInPlace(recorder, begin, bytes, (*inPlace)[at])) return refuse(MovedCount::PendingWrite);
                continue;
            }
            if (PendingStorageOverlaps(begin, bytes, nullptr)) return refuse(MovedCount::PendingStorage);
            if (AnyShadowedOverlaps(begin, bytes)) return refuse(MovedCount::Shadowed);
        }
        if (ticketed) fresh.push_back(range);
    }
    if (ticketed) {
        ticket->ranges->assign(fresh.begin(), fresh.end());
        ticket->valid = true;
        ticket->recorder = recorder.Id();
        ticket->note = recorder.WriteNoteCount();
        ticket->pending = StorageTexture::PendingSerial();
        ticket->allocations = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        ticket->epoch = GuestMemory::CollectEpoch();
    }
    return true;
}

ShaderResources::MovedCount ShaderResources::InPlaceRefusal(const Recorder& recorder, std::uint64_t begin, std::size_t bytes, InPlaceUser user) const {
    const auto today = [&] {
        if (heapStage || addressedBuffers || usesBda || rebound) return MovedCount::NotEligible;
        if (PendingStorageOverlaps(begin, bytes, nullptr)) return MovedCount::InPlaceStorage;
        if (AnyShadowedOverlaps(begin, bytes)) return MovedCount::InPlaceShadowed;
        if (CpuPublishPendingOverlaps(begin, bytes)) return MovedCount::CpuPublish;
        if (recorder.PendingLabelIn(begin, bytes) || recorder.CompletionLabelIn(begin, bytes)) return MovedCount::Label;
        if (recorder.QueuedStoreOverlaps(begin, bytes) || recorder.QueuedKeyStoreOverlaps(begin, bytes)) return MovedCount::QueuedStore;
        const auto overlaps = [&](const std::shared_ptr<ShaderResources>& writer) { return writer != nullptr && writer->CopiedWritesOverlap(begin, bytes); };
        if (context.copiedWriters != nullptr && std::any_of(context.copiedWriters->begin(), context.copiedWriters->end(), overlaps)) return MovedCount::CopiedWriter;
        const auto drawWriters = DrawCopiedWriters();
        if (std::any_of(drawWriters->begin(), drawWriters->end(), overlaps)) return MovedCount::CopiedWriter;
        if (bytes > context.limits.maxStorageBufferRange) return MovedCount::OverRange;
        return MovedCount::Count;
    }();
    if (today != MovedCount::Count || !RuntimeSwitches::Set(RuntimeSwitch::InPlaceExact)) return today;
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> span{{{begin, begin + bytes}}};
    const bool flushing = StorageTexture::AnyPendingOverlaps(span);
    noteInPlaceExact(user, flushing);
    return flushing ? MovedCount::InPlaceStorage : MovedCount::Count;
}

bool ShaderResources::takeMovedInPlace(const Recorder& recorder, std::uint64_t begin, std::size_t bytes, InPlaceRange& out) const {
    const auto refuse = [](MovedCount why) {
        noteMoved(why);
        return false;
    };
    if (const auto why = InPlaceRefusal(recorder, begin, bytes, InPlaceUser::DrawMoved); why != MovedCount::Count) return refuse(why);
    const auto* import = HostImportFor(context, begin, bytes);
    if (import == nullptr) return refuse(MovedCount::NoImport);
    if (begin < import->base || begin + bytes > import->base + import->bytes) return refuse(MovedCount::OverRange);
    const auto offset = HostImportOffset(*import, begin);
    if (offset % std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4) != 0) return refuse(MovedCount::Misaligned);
    static_cast<void>(FlushDepthPlanes(begin, bytes));
    out = {import->buffer, offset};
    return true;
}

bool ShaderResources::PreparedMovesClear(const PreparedBuffers& prepared, Recorder& recorder, MoveTicket* ticket) const {
    struct RangesStorage {};
    auto& ranges = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::size_t>>, RangesStorage>();
    ranges.clear();
    for (const auto& entry : prepared.moved) {
        const auto& item = allocations[entry.allocation];
        if (!item.guest || entry.inPlace != VK_NULL_HANDLE) continue;
        ranges.emplace_back(entry.address - item.adjustment, static_cast<std::size_t>(GuestBufferMemory::ViewBytes(entry.size, item.adjustment)));
    }
    // The finisher's accessibility answers hold while no mapping changed since.
    const bool accessible = (prepared.forget & 1u) == 0 && GuestMemory::ForgetSerial() == prepared.forget;
    if (accessible && RuntimeSwitches::Set(RuntimeSwitch::XpCommitFloor)) return true;
    return movedRangesClear(recorder, ranges, ticket, accessible);
}

bool ShaderResources::PreparedZonesClear(const PreparedBuffers& prepared, const Recorder& recorder) const {
    if ((prepared.forget & 1u) != 0 || GuestMemory::ForgetSerial() != prepared.forget) return false;
    if (zoneCardUsable(prepared, ZoneCardUse::Leader)) {
        const auto& card = prepared.card;
        for (std::size_t index = 0; index < card.zones; ++index) {
            const auto begin = card.entries[index].first;
            const auto packed = card.entries[index].second;
            const auto bytes = static_cast<std::size_t>(packed & ~PreparedBuffers::ZoneCard::Moved);
            if ((packed & PreparedBuffers::ZoneCard::Moved) != 0) {
                if (!PendingZones::Clear(begin, bytes) && (recorder.PendingWriteOverlaps(begin, bytes) || PendingStorageOverlaps(begin, bytes, nullptr))) return false;
                if (guestMemory.WritesOverlap(begin, bytes) || AnyShadowedOverlaps(begin, bytes)) return false;
            } else {
                if (!PendingZones::Clear(begin, bytes) && recorder.PendingWriteOverlaps(begin, bytes)) return false;
                if (guestMemory.WritesOverlap(begin, bytes)) return false;
            }
        }
        return true;
    }
    for (const auto& entry : prepared.moved) {
        const auto& item = allocations[entry.allocation];
        if (!item.guest || entry.inPlace != VK_NULL_HANDLE) continue;
        const auto begin = entry.address - item.adjustment;
        const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(entry.size, item.adjustment));
        if (!PendingZones::Clear(begin, bytes) && (recorder.PendingWriteOverlaps(begin, bytes) || PendingStorageOverlaps(begin, bytes, nullptr))) return false;
        if (guestMemory.WritesOverlap(begin, bytes) || AnyShadowedOverlaps(begin, bytes)) return false;
    }
    for (const auto& item : prepared.selected) {
        if (item.moved) continue;
        const auto& allocation = allocations[item.allocation];
        if (!PendingZones::Clear(allocation.address, allocation.size) && recorder.PendingWriteOverlaps(allocation.address, allocation.size)) return false;
        if (guestMemory.WritesOverlap(allocation.address, allocation.size)) return false;
    }
    return true;
}

bool ShaderResources::PreparedZonesClearByWindows(const PreparedBuffers& prepared, const Recorder& recorder) const {
    constexpr auto test = WriteValidation::Test::LeaderZones;
    if ((prepared.forget & 1u) != 0 || GuestMemory::ForgetSerial() != prepared.forget) return false;
    bool generations = false;
    const auto answered = [&](std::uint64_t begin, std::size_t bytes, std::uint16_t facts) {
        if (bytes == 0) return true;
        const auto answer = WriteValidation::CommitterClear(recorder, begin, bytes, facts, test);
        generations = generations || answer == WriteValidation::Answer::Generations;
        return answer != WriteValidation::Answer::Today;
    };
    for (const auto& entry : prepared.moved) {
        const auto& item = allocations[entry.allocation];
        if (!item.guest || entry.inPlace != VK_NULL_HANDLE) continue;
        const auto begin = entry.address - item.adjustment;
        const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(entry.size, item.adjustment));
        if (guestMemory.WritesOverlap(begin, bytes)) return false;
        if (!answered(begin, bytes, WriteValidation::Fact::Moved)) return PreparedZonesClear(prepared, recorder);
    }
    for (const auto& item : prepared.selected) {
        if (item.moved) continue;
        const auto& allocation = allocations[item.allocation];
        if (guestMemory.WritesOverlap(allocation.address, allocation.size)) return false;
        if (!answered(allocation.address, static_cast<std::size_t>(allocation.size), WriteValidation::Fact::Writes)) return PreparedZonesClear(prepared, recorder);
    }
    if (generations && WriteValidation::VerifyDue(test)) {
        const bool held = PreparedZonesClear(prepared, recorder);
        WriteValidation::NoteVerify(test, !held, "a prepared buffer has a pending write, a pending storage image or a live shadow");
        return held;
    }
    return true;
}

std::optional<bool> ShaderResources::PreparedMovesClearByWindows(const PreparedBuffers& prepared, Recorder& recorder, MoveTicket* ticket, WriteValidation::Test test) const {
    if ((prepared.forget & 1u) != 0 || GuestMemory::ForgetSerial() != prepared.forget || RuntimeSwitches::Set(RuntimeSwitch::XpCommitFloor) || WriteValidation::Session() == 0) return std::nullopt;
    struct WindowRangesStorage {};
    auto& ranges = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::size_t>>, WindowRangesStorage>();
    ranges.clear();
    for (const auto& entry : prepared.moved) {
        const auto& item = allocations[entry.allocation];
        if (!item.guest || entry.inPlace != VK_NULL_HANDLE) continue;
        ranges.emplace_back(entry.address - item.adjustment, static_cast<std::size_t>(GuestBufferMemory::ViewBytes(entry.size, item.adjustment)));
    }
    bool generations = false;
    for (const auto& [begin, bytes] : ranges) {
        if (bytes == 0 || guestMemory.WritesOverlap(begin, bytes)) return std::nullopt;
        const auto answer = WriteValidation::CommitterClear(recorder, begin, bytes, WriteValidation::Fact::Moved, test);
        if (answer == WriteValidation::Answer::Today) return std::nullopt;
        generations = generations || answer == WriteValidation::Answer::Generations;
    }
    if (generations && WriteValidation::VerifyDue(test)) {
        const bool held = movedRangesClear(recorder, ranges, ticket, true);
        WriteValidation::NoteVerify(test, !held, "a moved range has a pending write, a pending storage image or a live shadow");
        return held;
    }
    if (ticket != nullptr && ticket->ranges != nullptr) {
        ticket->checked += ranges.size();
        ticket->ranges->clear();
        for (const auto& [begin, bytes] : ranges) ticket->ranges->emplace_back(begin, begin + bytes);
        ticket->valid = true;
        ticket->recorder = recorder.Id();
        ticket->note = recorder.WriteNoteCount();
        ticket->pending = StorageTexture::PendingSerial();
        ticket->allocations = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        ticket->epoch = GuestMemory::CollectEpoch();
    }
    return true;
}

bool ShaderResources::StoresArmHolds(const Recorder& recorder) const {
    const auto session = WriteValidation::Session();
    return session != 0 && wvStores.session == session && wvStores.switches == RuntimeSwitches::Generation() && wvStores.recorder == recorder.Id() && wvStores.stores == Recorder::StoreGenerationForValidation() && wvStores.proof == wvProofSerial;
}

void ShaderResources::ArmStores(const Recorder& recorder, std::uint64_t generationBefore) const {
    const auto session = WriteValidation::Session();
    if (session == 0) return;
    wvStores.session = session;
    wvStores.switches = RuntimeSwitches::Generation();
    wvStores.recorder = recorder.Id();
    wvStores.stores = generationBefore;
    wvStores.proof = wvProofSerial;
}

std::optional<ShaderResources::MovedCount> ShaderResources::inPlaceRefusalByWindows(const Recorder& recorder, std::uint64_t begin, std::size_t bytes, WriteValidation::Test test) const {
    if (heapStage || addressedBuffers || usesBda || rebound) return MovedCount::NotEligible;
    const auto answer = WriteValidation::CommitterClear(recorder, begin, bytes, WriteValidation::Fact::InPlace, test);
    if (answer == WriteValidation::Answer::Today) return std::nullopt;
    if (answer == WriteValidation::Answer::Generations && WriteValidation::VerifyDue(test)) {
        const auto why = InPlaceRefusal(recorder, begin, bytes, InPlaceUser::DrawPrepared);
        const bool differs = why == MovedCount::InPlaceStorage || why == MovedCount::InPlaceShadowed || why == MovedCount::CpuPublish || why == MovedCount::Label || why == MovedCount::QueuedStore;
        WriteValidation::NoteVerify(test, differs, "an in-place range has a pending storage image, a live shadow, a CPU publish, a label or a queued store");
        return why;
    }
    const auto overlaps = [&](const std::shared_ptr<ShaderResources>& writer) { return writer != nullptr && writer->CopiedWritesOverlap(begin, bytes); };
    if (context.copiedWriters != nullptr && std::any_of(context.copiedWriters->begin(), context.copiedWriters->end(), overlaps)) return MovedCount::CopiedWriter;
    const auto drawWriters = DrawCopiedWriters();
    if (std::any_of(drawWriters->begin(), drawWriters->end(), overlaps)) return MovedCount::CopiedWriter;
    if (bytes > context.limits.maxStorageBufferRange) return MovedCount::OverRange;
    return MovedCount::Count;
}

bool ShaderResources::PreparedZonesClearBy(const PreparedBuffers& prepared, ZoneClear clear, void* tests) const {
    if ((prepared.forget & 1u) != 0 || GuestMemory::ForgetSerial() != prepared.forget) return false;
    if (zoneCardUsable(prepared, ZoneCardUse::Follower)) {
        const auto& card = prepared.card;
        for (std::size_t index = 0; index < card.zones; ++index) {
            const auto begin = card.entries[index].first;
            const auto packed = card.entries[index].second;
            const auto bytes = static_cast<std::size_t>(packed & ~PreparedBuffers::ZoneCard::Moved);
            if (guestMemory.WritesOverlap(begin, bytes) || !clear(tests, begin, bytes, (packed & PreparedBuffers::ZoneCard::Moved) != 0)) return false;
        }
        return true;
    }
    for (const auto& entry : prepared.moved) {
        const auto& item = allocations[entry.allocation];
        if (!item.guest || entry.inPlace != VK_NULL_HANDLE) continue;
        const auto begin = entry.address - item.adjustment;
        const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(entry.size, item.adjustment));
        if (guestMemory.WritesOverlap(begin, bytes) || !clear(tests, begin, bytes, true)) return false;
    }
    for (const auto& item : prepared.selected) {
        if (item.moved) continue;
        const auto& allocation = allocations[item.allocation];
        if (guestMemory.WritesOverlap(allocation.address, allocation.size) || !clear(tests, allocation.address, static_cast<std::size_t>(allocation.size), false)) return false;
    }
    return true;
}

ShaderResources::ZoneCardBuild ShaderResources::BuildZoneCard(PreparedBuffers& prepared) {
    using Card = PreparedBuffers::ZoneCard;
    auto& card = prepared.card;
    card.valid = false;
    card.zones = 0;
    card.guards = 0;
    card.plan = prepared.plan.get();
    if (card.plan == nullptr) return ZoneCardBuild::Refused;
    const auto& planned = card.plan->allocations;
    std::size_t count = 0;
    auto outcome = ZoneCardBuild::Made;
    const auto add = [&](std::uint64_t begin, std::uint64_t bytes, bool moved) {
        if ((bytes & Card::Moved) != 0) {
            outcome = ZoneCardBuild::Refused;
            return false;
        }
        if (count == Card::Capacity) {
            outcome = ZoneCardBuild::OverCapacity;
            return false;
        }
        card.entries[count++] = {begin, moved ? bytes | Card::Moved : bytes};
        return true;
    };
    std::size_t zones = 0;
    try {
        for (const auto& entry : prepared.moved) {
            if (entry.allocation >= planned.size()) return ZoneCardBuild::Refused;
            const auto& item = planned[entry.allocation];
            if (!item.guest || entry.inPlace != VK_NULL_HANDLE) continue;
            const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(entry.size, item.adjustment));
            if (!add(entry.address - item.adjustment, static_cast<std::uint64_t>(bytes), true)) return outcome;
        }
        for (const auto& item : prepared.selected) {
            if (item.moved) continue;
            if (item.allocation >= planned.size()) return ZoneCardBuild::Refused;
            const auto& allocation = planned[item.allocation];
            if (!add(allocation.address, static_cast<std::uint64_t>(allocation.size), false)) return outcome;
        }
        zones = count;
        for (const auto& item : prepared.selected) {
            if (item.begin == 0 || item.inPlace) continue;
            if (!add(item.begin, static_cast<std::uint64_t>(item.bytes), false)) return outcome;
        }
    } catch (const std::exception&) {
        return ZoneCardBuild::Refused;
    }
    card.zones = static_cast<std::uint8_t>(zones);
    card.guards = static_cast<std::uint8_t>(count - zones);
    card.valid = true;
    return ZoneCardBuild::Made;
}

namespace {

std::uint64_t zoneCardVerifyOrdinal = 0;

}

bool ShaderResources::zoneCardUsable(const PreparedBuffers& prepared, ZoneCardUse use) const {
    if (!CommitFeedOn(CommitFeedCards)) return false;
    const auto& card = prepared.card;
    const bool usable = card.valid && card.plan != nullptr && card.plan == prepared.plan.get() && card.plan == bufferPlan.get();
    if (CommitFeedCensus()) NoteCommitFeed(!usable ? CommitFeedCount::HitsWithoutCard : use == ZoneCardUse::Follower ? CommitFeedCount::CardFollowerUses : use == ZoneCardUse::Leader ? CommitFeedCount::CardLeaderUses : CommitFeedCount::CardGuardUses);
    if (!usable) return false;
    const auto every = CommitFeedVerifyEvery();
    if (every == 0 || ++zoneCardVerifyOrdinal % every != 0) return true;
    return zoneCardHolds(prepared, use == ZoneCardUse::Guards);
}

bool ShaderResources::zoneCardHolds(const PreparedBuffers& prepared, bool guards) const {
    using Card = PreparedBuffers::ZoneCard;
    const auto& card = prepared.card;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> today;
    if (guards) {
        for (const auto& item : prepared.selected) {
            if (item.begin == 0 || item.inPlace) continue;
            today.emplace_back(item.begin, static_cast<std::uint64_t>(item.bytes));
        }
    } else {
        for (const auto& entry : prepared.moved) {
            const auto& item = allocations[entry.allocation];
            if (!item.guest || entry.inPlace != VK_NULL_HANDLE) continue;
            const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(entry.size, item.adjustment));
            today.emplace_back(entry.address - item.adjustment, static_cast<std::uint64_t>(bytes) | Card::Moved);
        }
        for (const auto& item : prepared.selected) {
            if (item.moved) continue;
            const auto& allocation = allocations[item.allocation];
            today.emplace_back(allocation.address, static_cast<std::uint64_t>(allocation.size));
        }
    }
    const std::size_t first = guards ? card.zones : 0u;
    const std::size_t count = guards ? card.guards : card.zones;
    std::string detail;
    if (today.size() != count) detail = std::to_string(count) + " entries on the card, " + std::to_string(today.size()) + " in today's walk";
    for (std::size_t index = 0; detail.empty() && index < count; ++index) {
        const auto& mine = card.entries[first + index];
        const auto& theirs = today[index];
        if (mine.first == theirs.first && mine.second == theirs.second) continue;
        detail = "entry " + std::to_string(index) + ": the card holds " + std::to_string(mine.first) + " for " + std::to_string(mine.second & ~Card::Moved) + (((mine.second & Card::Moved) != 0) ? " bytes moved" : " bytes") + ", today's walk " + std::to_string(theirs.first) + " for " + std::to_string(theirs.second & ~Card::Moved) + (((theirs.second & Card::Moved) != 0) ? " bytes moved" : " bytes");
    }
    NoteCommitFeedVerify(detail.empty(), guards ? "guard list" : "zone card", detail.empty() ? detail : detail + "; today's walk answers this commit");
    return detail.empty();
}

void ShaderResources::PrefetchExpressHot() const {
    __builtin_prefetch(this);
    __builtin_prefetch(&expressValid);
    __builtin_prefetch(&expressWatch);
    __builtin_prefetch(&expressAt);
    __builtin_prefetch(&expressSources);
    __builtin_prefetch(&expressViewsAt);
    __builtin_prefetch(&directRegions);
    __builtin_prefetch(&bufferPlan);
    __builtin_prefetch(&touchedPresent);
    __builtin_prefetch(&faultGeneration);
    __builtin_prefetch(&guestMemory);
}

bool ShaderResources::PreparedInPlaceHolds(const PreparedBuffers& prepared, const Recorder& recorder, bool& behind, CommitterTrustCheck* trust, std::optional<WriteValidation::Test> validation) const {
    if (!prepared.inPlace) return true;
    const auto refuse = [](MovedCount why) {
        noteMoved(why);
        return false;
    };
    if (!RuntimeSwitches::Set(RuntimeSwitch::DirectBuffersInPlace)) return refuse(MovedCount::PreparedInPlaceSwitch);
    if ((prepared.forget & 1u) != 0 || GuestMemory::ForgetSerial() != prepared.forget || HostImportSetGeneration() != prepared.imports || GuestAllocations::GuestAllocationsGeneration_nid_postfix() != prepared.mapping) return refuse(MovedCount::PreparedInPlaceSerials);
    const bool trusted = RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter);
    if (trusted) NoteCommitterTrusted();
    const bool skipped = trusted || (trust != nullptr && trust->Skip());
    const bool windowed = validation.has_value() && !skipped && !RuntimeSwitches::Set(RuntimeSwitch::InPlaceExact) && WriteValidation::Session() != 0;
    for (const auto& entry : prepared.moved) {
        if (entry.inPlace == VK_NULL_HANDLE) continue;
        if (skipped) {
            if (heapStage || addressedBuffers || usesBda || rebound) return refuse(MovedCount::NotEligible);
            if (entry.bytes > context.limits.maxStorageBufferRange) return refuse(MovedCount::OverRange);
            continue;
        }
        if (guestMemory.WritesOverlap(entry.begin, entry.bytes)) {
            if (trust != nullptr) trust->Fail();
            return refuse(MovedCount::PreparedInPlaceWrites);
        }
        if (const auto answered = windowed ? inPlaceRefusalByWindows(recorder, entry.begin, entry.bytes, *validation) : std::optional<MovedCount>{}; answered.has_value()) {
            if (*answered == MovedCount::Count) continue;
            if (trust != nullptr && *answered != MovedCount::NotEligible && *answered != MovedCount::OverRange) trust->Fail();
            return refuse(*answered);
        }
        if (const auto why = InPlaceRefusal(recorder, entry.begin, entry.bytes, InPlaceUser::DrawPrepared); why != MovedCount::Count) {
            if (trust != nullptr && why != MovedCount::NotEligible && why != MovedCount::OverRange) trust->Fail();
            return refuse(why);
        }
    }
    for (const auto& entry : prepared.moved) {
        if (entry.inPlace == VK_NULL_HANDLE) continue;
        static_cast<void>(FlushDepthPlanes(entry.begin, entry.bytes));
        behind = behind || recorder.PendingWriteOverlaps(entry.begin, entry.bytes);
    }
    return true;
}

void ShaderResources::NotePreparedInPlaceTaken(const PreparedBuffers& prepared) {
    if (!prepared.inPlace) return;
    for (const auto& entry : prepared.moved) {
        if (entry.inPlace != VK_NULL_HANDLE) noteMoved(MovedCount::PreparedInPlaceRanges);
    }
    noteMoved(MovedCount::PreparedInPlaceHits);
}

bool ShaderResources::PreparedInPlaceOverlaps(const PreparedBuffers& prepared, std::uint64_t begin, std::uint64_t end) {
    if (!prepared.inPlace) return false;
    return std::ranges::any_of(prepared.moved, [&](const MovedBuffer& entry) { return entry.inPlace != VK_NULL_HANDLE && begin < entry.begin + entry.bytes && entry.begin < end; });
}

bool ShaderResources::PreparedUntouched(const PreparedBuffers& prepared) const {
    for (const auto& entry : prepared.moved) {
        const auto& item = allocations[entry.allocation];
        if (!item.guest) continue;
        if (!GpuZones::Untouched(entry.address - item.adjustment, GuestBufferMemory::ViewBytes(entry.size, item.adjustment))) return false;
    }
    for (const auto& item : prepared.selected) {
        if (item.moved) continue;
        const auto& allocation = allocations[item.allocation];
        if (!GpuZones::Untouched(allocation.address, allocation.size)) return false;
    }
    return true;
}

bool ShaderResources::UnmovedRangesClear(const PreparedBuffers& prepared, Recorder& recorder) const {
    for (const auto& item : prepared.selected) {
        if (item.moved) continue;
        const auto& allocation = allocations[item.allocation];
        if (guestMemory.WritesOverlap(allocation.address, allocation.size) || recorder.PendingWriteOverlaps(allocation.address, allocation.size)) return false;
    }
    return true;
}

bool ShaderResources::movedInPlace(Recorder& recorder, const Allocation& item, std::uint64_t address, std::uint64_t size, MovedBuffer& entry) const {
    auto& counters = MovedInPlace();
    const auto refuse = [&] {
        counters.refused.fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    if (!MovedBuffersInPlace() || context.hostImportAlignment == 0) return false;
    const auto* import = HostImportFor(context, address, static_cast<std::size_t>(size));
    if (import == nullptr) return refuse();
    const auto offset = HostImportOffset(*import, address);
    if (offset % 4 != 0) return refuse();
    const auto adjustment = static_cast<std::uint32_t>(offset % std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4));
    const bool inData = item.pushByte < 0 && item.dataAllocation >= 0;
    const auto* data = inData ? &allocations[static_cast<std::size_t>(item.dataAllocation)] : nullptr;
    const bool told = item.pushByte >= 0 || (data != nullptr && !data->guest && data->buffer != nullptr && item.dataByte < data->size);
    if (adjustment != item.adjustment && !told) return refuse();
    const auto begin = address - adjustment;
    const auto bytes = GuestBufferMemory::ViewBytes(size, adjustment);
    if (bytes > context.limits.maxStorageBufferRange || begin + bytes > import->base + import->bytes) return refuse();
    const auto length = static_cast<std::size_t>(bytes);
    if (guestMemory.WritesOverlap(begin, length) || PendingStorageOverlaps(begin, length, nullptr) || AnyShadowedOverlaps(begin, length)) return refuse();
    if (recorder.PendingLabelIn(begin, length) || recorder.QueuedStoreOverlaps(begin, length)) return refuse();
    const auto overlaps = [&](const auto& writer) { return writer->WritesOverlap(begin, length); };
    if (context.copiedWriters != nullptr && std::any_of(context.copiedWriters->begin(), context.copiedWriters->end(), overlaps)) return refuse();
    const auto drawWriters = DrawCopiedWriters();
    if (std::any_of(drawWriters->begin(), drawWriters->end(), overlaps)) return refuse();
    entry.inPlace = import->buffer;
    entry.inPlaceOffset = offset - adjustment;
    entry.adjustment = adjustment;
    entry.importOffset = entry.inPlaceOffset;
    entry.begin = begin;
    entry.bytes = length;
    entry.behind = true;
    counters.taken.fetch_add(1, std::memory_order_relaxed);
    return true;
}

const char* ShaderResources::MoveRefusalName(MoveRefusal refusal) {
    switch (refusal) {
        case MoveRefusal::None: return "none";
        case MoveRefusal::Shape: return "bindings differ";
        case MoveRefusal::NotGuest: return "not read in place";
        case MoveRefusal::Written: return "written or emptied";
        case MoveRefusal::Limit: return "over the limit";
        case MoveRefusal::Inaccessible: return "unmapped";
        case MoveRefusal::Overlap: return "pending write";
        case MoveRefusal::Bindings: return "no bindings made";
        case MoveRefusal::WrittenTemplate: return "written, template not movable";
        case MoveRefusal::WrittenInPlace: return "written, not bindable in place";
        case MoveRefusal::WrittenAlias: return "written, aliased";
        case MoveRefusal::Count: break;
    }
    return "?";
}

std::optional<std::vector<ShaderResources::MovedBuffer>> ShaderResources::MovedDispatchBuffers(const CompiledShader& shader, Recorder* recorder, std::size_t maxBytes, MoveProof& proof, bool writes, bool readsInPlace) const {
    proof = {};
    const auto refuse = [&](MoveRefusal why) {
        proof.refusal = why;
        return std::optional<std::vector<MovedBuffer>>{};
    };
    std::vector<MovedBuffer> moved;
    if (shader.program == nullptr || usesBda) return refuse(MoveRefusal::Shape);
    if (_set == VK_NULL_HANDLE && !heapStage) return moved;
    std::optional<bool> movable;
    const auto movedWritten = [&](std::size_t allocation, std::uint64_t address, std::uint64_t size) -> std::optional<MoveRefusal> {
        if (recorder != nullptr && !movable.has_value()) movable = MovableWrites();
        if (recorder != nullptr && !*movable) return MoveRefusal::WrittenTemplate;
        const auto& item = allocations[allocation];
        if (size > context.limits.maxStorageBufferRange) return MoveRefusal::Limit;
        if (address < item.adjustment) return MoveRefusal::WrittenInPlace;
        const auto begin = address - item.adjustment;
        const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(size, item.adjustment));
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return MoveRefusal::Inaccessible;
        MovedBuffer entry{allocation, address, static_cast<std::size_t>(size)};
        entry.begin = begin;
        entry.bytes = bytes;
        entry.written = true;
        if (recorder != nullptr) {
            if (InPlaceRefusal(*recorder, begin, bytes, InPlaceUser::DispatchWritten) != MovedCount::Count) return MoveRefusal::WrittenInPlace;
            const auto* import = HostImportFor(context, begin, bytes);
            if (import == nullptr || begin < import->base || begin + bytes > import->base + import->bytes) return MoveRefusal::WrittenInPlace;
            const auto offset = HostImportOffset(*import, begin);
            if (offset % std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4) != 0) return MoveRefusal::WrittenInPlace;
            if (MayOverlapDepthGuards_nid_no_patch(begin, bytes) || MayOverlapMultisampleGuards(begin, bytes)) return MoveRefusal::WrittenAlias;
            entry.inPlace = import->buffer;
            entry.importOffset = offset;
            entry.inPlaceOffset = offset;
            entry.adjustment = item.adjustment;
            entry.behind = recorder->PendingWriteOverlaps(begin, bytes);
        }
        moved.push_back(entry);
        ++proof.written;
        return std::nullopt;
    };
    const auto placeRead = [&](std::size_t allocation, std::uint64_t address, std::uint64_t size, std::uint64_t begin, std::size_t bytes) {
        if (const auto why = InPlaceRefusal(*recorder, begin, bytes); why != MovedCount::Count) {
            proof.inPlace = why;
            return false;
        }
        const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> span{{{begin, begin + bytes}}};
        if (StorageTexture::AnyPendingOverlaps(span)) {
            proof.inPlace = MovedCount::InPlaceStorage;
            return false;
        }
        const auto* import = HostImportFor(context, begin, bytes);
        if (import == nullptr) {
            proof.inPlace = MovedCount::NoImport;
            return false;
        }
        if (begin < import->base || begin + bytes > import->base + import->bytes) {
            proof.inPlace = MovedCount::OverRange;
            return false;
        }
        const auto offset = HostImportOffset(*import, begin);
        if (offset % std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4) != 0) {
            proof.inPlace = MovedCount::Misaligned;
            return false;
        }
        if (MayOverlapDepthGuards_nid_no_patch(begin, bytes) || MayOverlapMultisampleGuards(begin, bytes)) {
            proof.guarded = true;
            return false;
        }
        static_cast<void>(FlushDepthPlanes(begin, bytes));
        MovedBuffer entry{allocation, address, static_cast<std::size_t>(size)};
        entry.begin = begin;
        entry.bytes = bytes;
        entry.inPlace = import->buffer;
        entry.importOffset = offset;
        entry.inPlaceOffset = offset;
        entry.adjustment = static_cast<std::uint32_t>(address - begin);
        entry.behind = true;
        moved.push_back(entry);
        ++proof.buffers;
        ++proof.readsInPlace;
        return true;
    };
    for (const auto& binding : shader.program->bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
        const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
        if (kept == bindings.end() || kept->allocations.size() != binding.count || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) return refuse(MoveRefusal::Shape);
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
            const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
            const auto address = descriptor.Base48();
            const auto size = guestBufferBytes(std::span<const std::uint32_t>(words, 4));
            const auto& item = allocations[kept->allocations[element]];
            const bool empty = size == 0 || address == 0;
            if (!item.guest) {
                if (empty) continue;
                return refuse(MoveRefusal::NotGuest);
            }
            if (item.address == address && item.size == size) continue;
            const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
            if (writes && written && !empty && item.written) {
                if (const auto why = movedWritten(kept->allocations[element], address, size)) return refuse(*why);
                continue;
            }
            if (written || empty || item.written) return refuse(MoveRefusal::Written);
            if (size > context.limits.maxStorageBufferRange) return refuse(MoveRefusal::Limit);
            const auto begin = address - item.adjustment;
            const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(size, item.adjustment));
            if (maxBytes != 0 && proof.bytes + bytes > maxBytes) return refuse(MoveRefusal::Limit);
            if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return refuse(MoveRefusal::Inaccessible);
            const bool pending = recorder != nullptr ? recorder->PendingWriteOverlaps(begin, bytes) : Recorder::SnapshotWriteOverlaps(begin, bytes);
            const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{begin, begin + bytes}}};
            std::uint8_t overlap = 0;
            if (guestMemory.WritesOverlap(begin, bytes)) overlap = 1;
            else if (pending) overlap = 2;
            else if (StorageTexture::AnyPendingOverlaps(range)) overlap = 3;
            else if (AnyShadowedOverlaps(begin, bytes)) overlap = 4;
            if (overlap == 2 && readsInPlace) {
                if (recorder == nullptr) {
                    if (heapStage || addressedBuffers || rebound || StorageTexture::AnyPendingOverlaps(range) || AnyShadowedOverlaps(begin, bytes)) {
                        proof.overlap = overlap;
                        return refuse(MoveRefusal::Overlap);
                    }
                    moved.push_back({kept->allocations[element], address, static_cast<std::size_t>(size)});
                    ++proof.buffers;
                    ++proof.readsInPlace;
                    continue;
                }
                if (!placeRead(kept->allocations[element], address, size, begin, bytes)) {
                    proof.overlap = overlap;
                    return refuse(MoveRefusal::Overlap);
                }
                continue;
            }
            if (overlap != 0) {
                proof.overlap = overlap;
                return refuse(MoveRefusal::Overlap);
            }
            moved.push_back({kept->allocations[element], address, static_cast<std::size_t>(size)});
            ++proof.buffers;
            proof.bytes += bytes;
        }
    }
    if (proof.written == 0) return moved;
    const auto viewOf = [&](std::size_t index, std::uint64_t address, std::uint64_t size) {
        const auto& item = allocations[index];
        const auto first = address - item.adjustment;
        return std::pair<std::uint64_t, std::uint64_t>{first, first + GuestBufferMemory::ViewBytes(size, item.adjustment)};
    };
    for (const auto& entry : moved) {
        if (!entry.written) continue;
        const auto end = entry.begin + entry.bytes;
        for (std::size_t index = 0; index < allocations.size(); ++index) {
            const auto& item = allocations[index];
            if (index == entry.allocation || !item.guest) continue;
            const auto [oldBegin, oldEnd] = viewOf(index, item.address, item.size);
            if (oldBegin < end && entry.begin < oldEnd) return refuse(MoveRefusal::WrittenAlias);
            const auto other = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& candidate) { return candidate.allocation == index; });
            if (other == moved.end()) continue;
            const auto [newBegin, newEnd] = viewOf(index, other->address, other->size);
            if (newBegin < end && entry.begin < newEnd) return refuse(MoveRefusal::WrittenAlias);
        }
    }
    std::sort(moved.begin(), moved.end(), [](const MovedBuffer& a, const MovedBuffer& b) { return a.allocation < b.allocation; });
    return moved;
}

bool ShaderResources::MovableWrites() const {
    if (!reusable || NeedsCompletion() || usesBda || HoldsLease() || heapStage || addressedBuffers || rebound || htileClear.has_value() || spaceReuse) return false;
    const auto& writes = guestMemory.Writes();
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& item = allocations[index];
        if (!item.guest || !item.written) continue;
        const std::pair<std::uint64_t, std::uint64_t> range{item.address, item.address + item.size};
        if (std::count(writes.begin(), writes.end(), range) != 1) return false;
        for (std::size_t other = 0; other < allocations.size(); ++other) {
            const auto& alias = allocations[other];
            if (other != index && alias.guest && alias.address < range.second && range.first < alias.address + alias.size) return false;
        }
        if (!guestMemory.BoundInPlace(range.first, range.second)) return false;
    }
    return true;
}

std::size_t ShaderResources::MovableWritesRefusal() const {
    if (!reusable) return 1;
    if (NeedsCompletion()) return 2;
    if (usesBda) return 3;
    if (HoldsLease()) return 4;
    if (heapStage) return 5;
    if (addressedBuffers) return 6;
    if (rebound) return 7;
    if (htileClear.has_value()) return 8;
    if (spaceReuse) return 9;
    const auto& writes = guestMemory.Writes();
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& item = allocations[index];
        if (!item.guest || !item.written) continue;
        const std::pair<std::uint64_t, std::uint64_t> range{item.address, item.address + item.size};
        if (std::count(writes.begin(), writes.end(), range) != 1) return 10;
        for (std::size_t other = 0; other < allocations.size(); ++other) {
            const auto& alias = allocations[other];
            if (other != index && alias.guest && alias.address < range.second && range.first < alias.address + alias.size) return 11;
        }
        if (!guestMemory.BoundInPlace(range.first, range.second)) return 12;
    }
    return 0;
}

const char* ShaderResources::MovableWritesRefusalName(std::size_t refusal) {
    static constexpr std::array<const char*, 13> names{"movable", "not reusable", "needs completion", "address-based", "lease", "heap stage", "address table", "rebound selection", "HTILE clear", "address space reuse", "written range not listed once", "written range shared with another buffer", "written range not bound in place"};
    return refusal < names.size() ? names[refusal] : "?";
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ShaderResources::MovedGpuWrites(std::span<const MovedBuffer> moved) const {
    auto writes = guestMemory.Writes();
    for (const auto& entry : moved) {
        if (!entry.written || entry.allocation >= allocations.size()) continue;
        const auto& item = allocations[entry.allocation];
        const std::pair<std::uint64_t, std::uint64_t> old{item.address, item.address + item.size};
        const std::pair<std::uint64_t, std::uint64_t> range{entry.address, entry.address + entry.size};
        const auto found = std::find(writes.begin(), writes.end(), old);
        if (found != writes.end()) *found = range;
        else writes.push_back(range);
    }
    return writes;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ShaderResources::MovedInPlaceReads(std::span<const MovedBuffer> moved) const {
    auto reads = guestMemory.InPlaceReads();
    for (const auto& entry : moved) {
        if (!entry.written || entry.allocation >= allocations.size()) continue;
        const auto& item = allocations[entry.allocation];
        const auto first = item.address;
        const auto last = item.address + item.size;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> kept;
        kept.reserve(reads.size() + 1u);
        for (const auto& [begin, end] : reads) {
            if (end <= first || last <= begin) {
                kept.emplace_back(begin, end);
                continue;
            }
            if (begin < first) kept.emplace_back(begin, first);
            if (last < end) kept.emplace_back(last, end);
        }
        reads = std::move(kept);
    }
    for (const auto& entry : moved) {
        if (entry.inPlace != VK_NULL_HANDLE) reads.emplace_back(entry.begin, entry.begin + entry.bytes);
    }
    return reads;
}

std::vector<std::string> ShaderResources::VerifyMovedWrites(const CompiledShader& shader, std::span<const MovedBuffer> moved, const HostImportsProof* recordedImports, std::size_t* incomparable) const {
    std::vector<std::string> differences;
    const auto note = [&](const char* what, std::uint32_t binding, std::uint32_t element, std::uint64_t recordedBegin, std::uint64_t recordedEnd, std::uint64_t expectedBegin, std::uint64_t expectedEnd) {
        char text[256];
        std::snprintf(text, sizeof(text), "%s: binding %u element %u, recorded 0x%llx-0x%llx, expected 0x%llx-0x%llx", what, binding, element, static_cast<unsigned long long>(recordedBegin), static_cast<unsigned long long>(recordedEnd), static_cast<unsigned long long>(expectedBegin), static_cast<unsigned long long>(expectedEnd));
        differences.emplace_back(text);
    };
    if (shader.program == nullptr) {
        differences.emplace_back("no compiled shader");
        return differences;
    }
    const auto writes = MovedGpuWrites(moved);
    const auto reads = MovedInPlaceReads(moved);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> expected;
    for (const auto& binding : shader.program->bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) continue;
        const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
        if (kept == bindings.end() || kept->allocations.size() != binding.count) {
            note("binding shape", binding.binding, 0, 0, 0, 0, 0);
            continue;
        }
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
            const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
            const auto address = descriptor.Base48();
            const auto size = guestBufferBytes(std::span<const std::uint32_t>(words, 4));
            const auto index = kept->allocations[element];
            const auto& item = allocations[index];
            if (!item.guest || size == 0 || address == 0) continue;
            const auto entry = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& candidate) { return candidate.allocation == index; });
            const auto boundAddress = entry != moved.end() ? entry->address : item.address;
            const auto boundSize = entry != moved.end() ? static_cast<std::uint64_t>(entry->size) : static_cast<std::uint64_t>(item.size);
            if (boundAddress != address || boundSize != size) note("bound range", binding.binding, element, boundAddress, boundAddress + boundSize, address, address + size);
            const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
            if (written) {
                expected.emplace_back(address, address + size);
                if (std::find(writes.begin(), writes.end(), std::pair<std::uint64_t, std::uint64_t>{address, address + size}) == writes.end()) note("write list", binding.binding, element, 0, 0, address, address + size);
            }
            if (entry == moved.end() || entry->inPlace == VK_NULL_HANDLE) continue;
            const auto begin = address - item.adjustment;
            const auto end = begin + GuestBufferMemory::ViewBytes(size, item.adjustment);
            if (entry->begin != begin || entry->begin + entry->bytes != end) note("in-place view", binding.binding, element, entry->begin, entry->begin + entry->bytes, begin, end);
            if (std::find(reads.begin(), reads.end(), std::pair<std::uint64_t, std::uint64_t>{entry->begin, entry->begin + entry->bytes}) == reads.end()) note("read list", binding.binding, element, 0, 0, begin, end);
            const auto* import = ExistingHostImportFor(context, entry->begin, entry->bytes);
            if (import != nullptr && import->buffer == entry->inPlace && HostImportOffset(*import, entry->begin) == entry->importOffset) continue;
            if (recordedImports != nullptr && !HostImportsUnchanged(context, *recordedImports)) {
                if (incomparable != nullptr) ++*incomparable;
                continue;
            }
            if (import == nullptr) note("import: no import serves the in-place view, offsets", binding.binding, element, entry->importOffset, 0, 0, 0);
            else if (import->buffer != entry->inPlace) note("import: another import serves the in-place view, offsets", binding.binding, element, entry->importOffset, 0, HostImportOffset(*import, entry->begin), 0);
            else note("import: the recorded import at another offset", binding.binding, element, entry->importOffset, 0, HostImportOffset(*import, entry->begin), 0);
        }
    }
    for (const auto& entry : moved) {
        if (!entry.written || entry.allocation >= allocations.size()) continue;
        const auto& item = allocations[entry.allocation];
        const std::pair<std::uint64_t, std::uint64_t> old{item.address, item.address + item.size};
        if (std::find(expected.begin(), expected.end(), old) == expected.end() && std::find(writes.begin(), writes.end(), old) != writes.end()) note("old range still written", 0, static_cast<std::uint32_t>(entry.allocation), old.first, old.second, 0, 0);
    }
    return differences;
}

std::vector<std::string> ShaderResources::VerifyInPlaceReads(std::span<const MovedBuffer> moved, const Recorder& recorder, const HostImportsProof* recordedImports, std::size_t* incomparable) const {
    std::vector<std::string> differences;
    const auto note = [&](const char* what, const MovedBuffer& entry, std::uint64_t recorded, std::uint64_t fresh) {
        char text[256];
        std::snprintf(text, sizeof(text), "read-only buffer 0x%llx+0x%llx bound in place: %s, recorded 0x%llx, a fresh build 0x%llx", static_cast<unsigned long long>(entry.address), static_cast<unsigned long long>(entry.size), what, static_cast<unsigned long long>(recorded), static_cast<unsigned long long>(fresh));
        differences.emplace_back(text);
    };
    const auto premise = [&](const char* what, const char* detail, const MovedBuffer& entry) {
        char text[256];
        std::snprintf(text, sizeof(text), "read-only buffer 0x%llx+0x%llx bound in place: %s%s", static_cast<unsigned long long>(entry.address), static_cast<unsigned long long>(entry.size), what, detail);
        differences.emplace_back(text);
    };
    const auto writes = MovedGpuWrites(moved);
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4);
    for (const auto& entry : moved) {
        if (entry.inPlace == VK_NULL_HANDLE || entry.written) continue;
        if (entry.allocation >= allocations.size() || !allocations[entry.allocation].guest || allocations[entry.allocation].written) {
            premise("not a read-only guest buffer of the template", "", entry);
            continue;
        }
        const auto& item = allocations[entry.allocation];
        const auto first = entry.address & ~std::uint64_t{3};
        const auto last = first == entry.address ? entry.address + entry.size : (entry.address + entry.size + 3) & ~std::uint64_t{3};
        const auto* import = ExistingHostImportFor(context, first, static_cast<std::size_t>(last - first));
        const bool importsMoved = (import == nullptr || import->buffer != entry.inPlace) && recordedImports != nullptr && !HostImportsUnchanged(context, *recordedImports);
        if (importsMoved) {
            if (incomparable != nullptr) ++*incomparable;
        } else if (import == nullptr) {
            premise("no import serves the range a fresh build binds", "", entry);
        } else {
            const auto offset = HostImportOffset(*import, entry.address);
            const auto adjustment = static_cast<std::uint32_t>(offset % alignment);
            auto range = GuestBufferMemory::ViewBytes(entry.size, adjustment);
            if (entry.size <= context.limits.maxStorageBufferRange) range = std::min<std::uint64_t>(range, context.limits.maxStorageBufferRange & ~std::uint64_t{3});
            if (import->buffer != entry.inPlace) note("bound to another buffer than the import's, offsets", entry, entry.importOffset, offset - adjustment);
            if (entry.importOffset != offset - adjustment) note("descriptor offset", entry, entry.importOffset, offset - adjustment);
            if (entry.bytes != range) note("descriptor range", entry, entry.bytes, range);
            if (item.adjustment != adjustment) note("view adjustment", entry, item.adjustment, adjustment);
            if (entry.begin != entry.address - adjustment) note("view begin", entry, entry.begin, entry.address - adjustment);
            if (entry.address - adjustment + range > import->base + import->bytes) note("view end past the import's end", entry, entry.address - adjustment + range, import->base + import->bytes);
        }
        bool ownImage = false;
        for (std::size_t index = 0; index < storageTextures.size() && !ownImage; ++index) {
            const auto& texture = storageTextures[index];
            if (texture == nullptr || index >= storageWritten.size() || !storageWritten[index]) continue;
            const auto base = texture->Descriptor().baseAddress;
            ownImage = entry.begin < base + texture->GuestBytes() && base < entry.begin + entry.bytes;
        }
        if (const auto why = InPlaceRefusal(recorder, entry.begin, entry.bytes); why != MovedCount::Count && !(ownImage && why == MovedCount::InPlaceStorage)) premise("a fresh build would sync, flush or copy first: ", MovedCountName(why), entry);
        const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> span{{{entry.begin, entry.begin + entry.bytes}}};
        if (!ownImage && StorageTexture::AnyPendingOverlaps(span)) premise("a fresh build would flush first: storage image results pending or being written back over the range", "", entry);
        if (MayOverlapDepthGuards_nid_no_patch(entry.begin, entry.bytes) || MayOverlapMultisampleGuards(entry.begin, entry.bytes)) premise("a depth or multisample guard covers the range", "", entry);
        if (std::ranges::any_of(writes, [&](const std::pair<std::uint64_t, std::uint64_t>& written) { return written.first < entry.begin + entry.bytes && entry.begin < written.second; })) premise("the dispatch writes the range it reads in place", "", entry);
    }
    return differences;
}

namespace {

std::size_t readDrawSnapshotCap() {
    const char* text = std::getenv("APS5_DRAW_SNAPSHOT_MAX_MIB");
    const auto mib = text != nullptr ? std::strtoull(text, nullptr, 10) : 64ull;
    return static_cast<std::size_t>(mib * 1024ull * 1024ull);
}

std::atomic<std::size_t>& drawSnapshotCap() {
    static std::atomic<std::size_t> bytes{readDrawSnapshotCap()};
    return bytes;
}

}

std::size_t ShaderResources::MaxDrawSnapshotBytes() {
    return drawSnapshotCap().load(std::memory_order_relaxed);
}

void ShaderResources::RereadMaxDrawSnapshotBytesForTest() {
    drawSnapshotCap().store(readDrawSnapshotCap(), std::memory_order_relaxed);
}

std::size_t ShaderResources::DrawUploadLimit() {
    static const std::size_t limit = [] {
        const char* value = std::getenv("APS5_DRAW_UPLOAD_MAX");
        return value != nullptr ? static_cast<std::size_t>(std::strtoull(value, nullptr, 10)) : std::size_t{4096};
    }();
    return limit;
}

namespace {

std::uint64_t HeapReplayKeyHash(std::span<const std::uint32_t> key) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const auto word : key) {
        hash ^= word;
        hash *= 0x100000001b3ull;
    }
    return hash == 0u ? 1u : hash;
}

constexpr std::size_t MovedUploadBytes = Recorder::DrawUploadChunk / 16u;

bool MovedSnapshotStats() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return stats;
}

struct MovedSnapshotTally {
    std::atomic<std::uint64_t> cached{0};
    std::atomic<std::uint64_t> reused{0};
    std::atomic<std::uint64_t> cachedBytes{0};
    std::atomic<std::uint64_t> uploaded{0};
    std::atomic<std::uint64_t> uploadedBytes{0};
};

MovedSnapshotTally& movedSnapshotTally() {
    static MovedSnapshotTally tally;
    return tally;
}

}

ShaderResources::MovedSnapshotCounts ShaderResources::TakeMovedSnapshotCounts() {
    auto& tally = movedSnapshotTally();
    return {tally.cached.exchange(0, std::memory_order_relaxed), tally.reused.exchange(0, std::memory_order_relaxed), tally.cachedBytes.exchange(0, std::memory_order_relaxed), tally.uploaded.exchange(0, std::memory_order_relaxed), tally.uploadedBytes.exchange(0, std::memory_order_relaxed)};
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::PrepareDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved, bool movedOnly, const BdaResources* faultRecord, bool census, std::vector<std::size_t>* selectedOut, const PreparedBuffers* prepared, PreparedBuffers::Selection* preparedSelection, BindingRoute* route) const {
    struct SelectedStorage {};
    auto& selected = HostThreadLocal<std::vector<std::size_t>, SelectedStorage>();
    auto selection = PreparedBuffers::Selection::Hazard;
    auto result = prepared != nullptr && !movedOnly ? selectPreparedSnapshots(recorder, moved, *prepared, selected, true, selection, route) : nullptr;
    if (result != nullptr && route != nullptr && route->taken == BindingTaken::Placed) result->pushSelected.clear();
    if (preparedSelection != nullptr) *preparedSelection = selection;
    if (result == nullptr) result = selectDrawSnapshots(recorder, moved, movedOnly, selected, census);
    if (selectedOut != nullptr) selectedOut->assign(selected.begin(), selected.end());
    return finishDrawBindings(recorder, std::move(result), selected, faultRecord, -1, route);
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::PrepareReplayDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved, std::span<const CompiledShader> shaders, const BdaResources* faultRecord, HeapReplayProof& proof, bool census, std::span<const std::uint32_t> contentKey) {
    struct SelectedStorage {};
    auto& selected = HostThreadLocal<std::vector<std::size_t>, SelectedStorage>();
    auto result = selectDrawSnapshots(recorder, moved, false, selected, census);
    auto& record = heapReplayRecord();
    if (record.keyHash == 0u && !contentKey.empty()) record.keyHash = HeapReplayKeyHash(contentKey);
    const bool overlay = result != nullptr && addressedBuffers && std::ranges::any_of(selected, [&](std::size_t index) { return index < record.addressed.size() && record.addressed[index]; });
    proof = ReplayHeapStages(shaders, overlay);
    return finishDrawBindings(recorder, std::move(result), selected, faultRecord);
}

bool ShaderResources::lendsSets(std::int32_t pushedSet, const BdaResources* faultRecord) const {
    static_cast<void>(pushedSet);
    return faultRecord == nullptr && completed && !heapStage && !addressedBuffers && !rebound && context.descriptorCache != nullptr && RuntimeSwitches::Set(RuntimeSwitch::OwnDrawSets);
}

namespace {

constexpr std::size_t LentSetBytes = 16384;
constexpr std::uint32_t LenderMisses = 8;
constexpr std::uint32_t LenderRest = 64;

}

void ShaderResources::lendDrawSets(DrawBindings& result, std::span<const std::size_t> selected, std::int32_t pushedSet, BindingRoute& route) const {
    const auto* prepared = route.selection;
    if (prepared == nullptr || prepared->inPlace || prepared->selected.size() != selected.size() || selected.size() > result.snapshots.size()) return;
    if (ownDrawSets == nullptr) ownDrawSets = makeOwnDrawSets();
    const auto& plan = ownDrawSets->plan;
    for (std::size_t set = 0; set < 2u; ++set) {
        if (static_cast<std::int32_t>(set) == pushedSet || !plan.lendable[set] || !(plan.described[set] || splitSets)) continue;
        auto& lender = setLenders[set];
        if (lender.resting != 0) {
            --lender.resting;
            continue;
        }
        if (lender.set == VK_NULL_HANDLE) continue;
        const auto held = lender.bindings.lock();
        if (held == nullptr) continue;
        const auto missed = [&] {
            if (++lender.misses < LenderMisses) return;
            lender.misses = 0;
            lender.resting = LenderRest;
        };
        if (lender.buffers.size() != descriptorBuffers.size() || std::memcmp(lender.buffers.data(), descriptorBuffers.data(), descriptorBuffers.size() * sizeof(VkDescriptorBufferInfo)) != 0) {
            missed();
            continue;
        }
        std::size_t matched = 0;
        std::size_t at = 0;
        bool same = true;
        for (std::size_t index = 0; same && index < selected.size(); ++index) {
            const auto allocationSet = selected[index] < plan.allocationSets.size() ? plan.allocationSets[selected[index]] : std::uint8_t{0xffu};
            if (allocationSet == 0xfeu) same = false;
            if (allocationSet != set) continue;
            if (matched >= lender.allocations.size() || lender.allocations[matched] != selected[index] || lender.snapshots[matched] >= held->snapshots.size()) {
                same = false;
                continue;
            }
            const auto& mine = result.snapshots[index];
            const auto& theirs = held->snapshots[lender.snapshots[matched]];
            const auto& item = prepared->selected[index];
            same = item.allocation == selected[index] && item.bytes == mine.bytes && mine.address == theirs.address && mine.bytes == theirs.bytes && at + mine.bytes <= lender.bytes.size() && (mine.bytes == 0 || std::memcmp(prepared->blob.data() + item.offset, lender.bytes.data() + at, mine.bytes) == 0);
            at += mine.bytes;
            ++matched;
        }
        if (!same || matched != lender.allocations.size()) {
            missed();
            continue;
        }
        lender.misses = 0;
        matched = 0;
        for (std::size_t index = 0; index < selected.size(); ++index) {
            if (selected[index] < plan.allocationSets.size() && plan.allocationSets[selected[index]] == set) result.snapshots[index] = held->snapshots[lender.snapshots[matched++]];
        }
        result.lent[set] = true;
        if (set == 0) {
            result.allocation.set = lender.set;
            result.layout = lender.layout;
        } else {
            result.fragmentAllocation.set = lender.set;
            result.fragmentLayout = lender.layout;
        }
        result.lenders[set] = held;
        ++route.setsLent;
    }
}

void ShaderResources::noteSetLenders(const std::shared_ptr<DrawBindings>& result, std::span<const std::size_t> selected, BindingRoute& route) const {
    if (ownDrawSets == nullptr) return;
    const auto& plan = ownDrawSets->plan;
    const auto* prepared = route.selection;
    const bool copied = prepared != nullptr && !prepared->inPlace && prepared->selected.size() == selected.size();
    for (std::size_t set = 0; set < 2u; ++set) {
        const auto handle = set == 0 ? result->allocation.set : result->fragmentAllocation.set;
        if (handle == VK_NULL_HANDLE || result->lent[set]) continue;
        ++route.setsWritten;
        auto& lender = setLenders[set];
        lender.bindings.reset();
        lender.set = VK_NULL_HANDLE;
        if (!plan.lendable[set] || !copied || lender.resting != 0) continue;
        lender.allocations.clear();
        lender.snapshots.clear();
        lender.bytes.clear();
        bool lendable = selected.size() <= result->snapshots.size();
        for (std::size_t index = 0; lendable && index < selected.size(); ++index) {
            const auto allocationSet = selected[index] < plan.allocationSets.size() ? plan.allocationSets[selected[index]] : std::uint8_t{0xffu};
            if (allocationSet == 0xfeu) lendable = false;
            if (allocationSet != set) continue;
            const auto& item = prepared->selected[index];
            const auto bytes = result->snapshots[index].bytes;
            if (item.allocation != selected[index] || item.bytes != bytes || lender.bytes.size() + bytes > LentSetBytes) {
                lendable = false;
                continue;
            }
            lender.allocations.push_back(selected[index]);
            lender.snapshots.push_back(index);
            lender.bytes.insert(lender.bytes.end(), prepared->blob.begin() + static_cast<std::ptrdiff_t>(item.offset), prepared->blob.begin() + static_cast<std::ptrdiff_t>(item.offset + bytes));
        }
        if (!lendable) continue;
        lender.bindings = result;
        lender.set = handle;
        lender.layout = set == 0 ? result->layout : result->fragmentLayout;
        lender.buffers.assign(descriptorBuffers.begin(), descriptorBuffers.end());
    }
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::finishDrawBindings(Recorder& recorder, std::shared_ptr<DrawBindings> result, std::span<const std::size_t> selected, const BdaResources* faultRecord, std::int32_t pushedSet, BindingRoute* route) const {
    if (result == nullptr || (selected.empty() && !rebound)) return {};
    const bool lends = route != nullptr && route->sets && lendsSets(pushedSet, faultRecord);
    if (pushedSet < 0) {
        const auto refusedBefore = route != nullptr ? route->dynamicRefused : 0u;
        const bool dynamic = route != nullptr && route->dynamic && faultRecord == nullptr && writeDynamicSets(*result, selected, *route, -1);
        if (route != nullptr && !dynamic) route->leave = !route->dynamic ? route->plainLeave : faultRecord != nullptr ? DynamicLeave::Faults : route->dynamicRefused != refusedBefore ? DynamicLeave::None : DynamicLeave::Rebound;
        if (!dynamic) {
            if (lends) lendDrawSets(*result, selected, pushedSet, *route);
            writeDrawSets(*result, selected, faultRecord);
            if (lends) noteSetLenders(result, selected, *route);
        }
    } else {
        const auto& sets = pushPlan->allocationSets;
        if (std::ranges::any_of(selected, [&](std::size_t index) { return index >= sets.size() || sets[index] != static_cast<std::uint8_t>(pushedSet); })) {
            const auto refusedBefore = route != nullptr ? route->dynamicRefused : 0u;
            const bool dynamic = route != nullptr && route->pushDynamic && faultRecord == nullptr && writeDynamicSets(*result, selected, *route, pushedSet);
            if (route != nullptr && !dynamic) route->leave = !route->pushDynamic ? route->pushedLeave : faultRecord != nullptr ? DynamicLeave::Faults : route->dynamicRefused != refusedBefore ? DynamicLeave::None : DynamicLeave::Rebound;
            if (!dynamic) {
                if (lends) lendDrawSets(*result, selected, pushedSet, *route);
                writeDrawSets(*result, selected, faultRecord, pushedSet);
                if (lends) noteSetLenders(result, selected, *route);
            }
        } else {
            result->cache = context.descriptorCache;
        }
    }
    if (heapStage && RuntimeSwitches::Set(RuntimeSwitch::ReplayBind) && !RuntimeSwitches::Set(RuntimeSwitch::NoHeapSetSkip)) {
        for (const auto& snapshot : result->snapshots) {
            if (snapshot.buffer != nullptr) recorder.KeepOnce(snapshot.buffer, snapshot.buffer->Bytes().size());
        }
        return result;
    }
    recorder.Keep(result);
    if (RuntimeSwitches::Set(RuntimeSwitch::KeepBytesOnce)) {
        for (const auto& snapshot : result->snapshots) {
            if (snapshot.buffer != nullptr) recorder.KeepSnapshotBytes(snapshot.buffer.get(), snapshot.buffer->Bytes().size());
        }
        return result;
    }
    for (const auto& snapshot : result->snapshots) {
        if (snapshot.buffer != nullptr) recorder.KeepBytes(snapshot.buffer.get(), snapshot.buffer->Bytes().size());
    }
    return result;
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::selectDrawSnapshots(Recorder& recorder, std::span<const MovedBuffer> moved, bool movedOnly, std::vector<std::size_t>& selected, bool census) const {
    selected.clear();
    requireGuestBuffers(moved);
    if ((_set == VK_NULL_HANDLE && fragment.set == VK_NULL_HANDLE && !heapStage) || (usesBda && !spaceReuse)) return {};
    prepareFaultSlot();
    bindingsChecked = heapStage && RuntimeSwitches::Set(RuntimeSwitch::ReplayBind);
    auto split = guestMemory.SplitInPlaceReads();
    const bool sharedOnly = split && split->own.empty();
    struct ReadsStorage {};
    auto& reads = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, ReadsStorage>();
    if (!sharedOnly) guestMemory.InPlaceReads(reads);
    const auto maxSnapshotBytes = maxDrawSnapshotBytes;
    const auto uploadLimit = DrawUploadLimit();
    std::size_t totalSnapshotBytes = 0;
    auto result = std::make_shared<DrawBindings>();
    std::array<std::uint64_t, 3> stamp{};
    bool stamped = false;
    struct SnapshotCensus {
        std::size_t moved = 0;
        std::size_t unmoved = 0;
        std::size_t bytes = 0;
        std::size_t memoHits = 0;
        std::size_t uploads = 0;
        std::size_t large = 0;
    } counted;
    const auto keep = [&](DrawBindings::Snapshot snapshot) {
        if (result->snapshots.empty()) result->snapshots.reserve(allocations.size());
        result->snapshots.push_back(std::move(snapshot));
    };
    const auto toldInData = [&](const MovedBuffer& entry, std::size_t data) {
        const auto& element = allocations[entry.allocation];
        return entry.inPlace != VK_NULL_HANDLE && element.pushByte < 0 && element.dataAllocation == static_cast<std::int64_t>(data) && entry.adjustment != element.adjustment;
    };
    const bool inPlaceData = std::any_of(moved.begin(), moved.end(), [&](const MovedBuffer& entry) { return entry.inPlace != VK_NULL_HANDLE && allocations[entry.allocation].pushByte < 0 && entry.adjustment != allocations[entry.allocation].adjustment; });
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& item = allocations[index];
        const auto override = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& entry) { return entry.allocation == index; });
        const bool words = override != moved.end() && !override->words.empty();
        const bool offsets = inPlaceData && !item.guest && item.buffer != nullptr && std::any_of(moved.begin(), moved.end(), [&](const MovedBuffer& entry) { return toldInData(entry, index); });
        if (words || (offsets && override == moved.end())) {
            const auto size = words ? override->size : item.size;
            auto slice = recorder.DrawUpload(size);
            if (words) std::memcpy(slice.bytes, override->words.data(), size);
            else if (!item.dataWords.empty()) std::memcpy(slice.bytes, item.dataWords.data(), std::min(size, item.dataWords.size() * sizeof(std::uint32_t)));
            else std::memcpy(slice.bytes, item.buffer->Bytes().data(), size);
            for (const auto& patch : dataPatches) {
                if (patch.allocation == index && patch.byte < size) slice.bytes[patch.byte] = static_cast<std::byte>(patch.adjustment);
            }
            if (offsets) {
                for (const auto& entry : moved) {
                    if (toldInData(entry, index) && allocations[entry.allocation].dataByte < size) slice.bytes[allocations[entry.allocation].dataByte] = static_cast<std::byte>(entry.adjustment);
                }
            }
            selected.push_back(index);
            keep({0, std::move(slice.buffer), slice.offset, size});
            if (census) {
                if (words) ++counted.moved;
                else ++counted.unmoved;
                ++counted.uploads;
                counted.bytes += size;
            }
            continue;
        }
        if (override != moved.end() && override->inPlace != VK_NULL_HANDLE) {
            const auto begin = override->address - override->adjustment;
            const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(override->size, override->adjustment));
            if (!RuntimeSwitches::Set(RuntimeSwitch::MovedInPlace)) recorder.NotePendingRead(begin, bytes, Recorder::ReadKind::DispatchElement);
            selected.push_back(index);
            keep({begin, nullptr, override->inPlaceOffset, bytes, override->inPlace});
            if (census) ++counted.moved;
            continue;
        }
        std::uint64_t address = item.address;
        std::size_t size = item.size;
        if (override != moved.end()) {
            address = override->address;
            size = override->size;
        } else {
            if (movedOnly || !item.guest || item.written || guestMemory.WritesOverlap(item.address, item.size)) continue;
            const bool direct = [&] {
                if (sharedOnly) {
                    const auto& ranges = *split->space;
                    const auto found = std::partition_point(ranges.begin(), ranges.end(), [&](const auto& range) { return range.second <= item.address; });
                    return found != ranges.end() && found->first <= item.address && item.address < found->second && item.size <= found->second - item.address;
                }
                return std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return item.address >= range.first && item.address < range.second && item.size <= range.second - item.address; });
            }();
            if (!direct || recorder.PendingWriteOverlaps(item.address, item.size)) continue;
        }
        const auto begin = address - item.adjustment;
        const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(size, item.adjustment));
        if (override == moved.end() && (bytes > maxSnapshotBytes || totalSnapshotBytes + bytes > maxSnapshotBytes)) continue;
        const auto registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        if (!stamped) {
            stamp = {GuestMemory::TrackerGeneration(), GuestMemory::CollectEpoch(), registryGeneration};
            stamped = true;
        }
        VkDeviceSize offset = 0;
        auto buffer = recorder.UnwrittenDrawSnapshot(begin, bytes, stamp, &offset);
        if (census && buffer != nullptr) ++counted.memoHits;
        static const bool verifySnapshots = std::getenv("APS5_VERIFY_DRAW_SNAPSHOTS") != nullptr;
        if (verifySnapshots && buffer != nullptr) {
            static std::atomic<std::uint64_t> checked{0}, differing{0};
            const auto fast = buffer->Bytes().subspan(static_cast<std::size_t>(offset), bytes);
            VkDeviceSize slowOffset = 0;
            const auto slow = recorder.ReusableDrawSnapshot(begin, bytes, Recorder::SnapshotUse::Storage, nullptr, GuestMemory::CollectWrites(begin, bytes), &slowOffset);
            const bool same = (slow == buffer && slowOffset == offset) || (slow != nullptr && slow->Bytes().size() >= slowOffset + fast.size() && std::memcmp(slow->Bytes().data() + slowOffset, fast.data(), fast.size()) == 0) || (slow == nullptr && std::memcmp(fast.data(), reinterpret_cast<const void*>(begin), bytes) == 0);
            if (!same) differing.fetch_add(1);
            if (checked.fetch_add(1) % 200000 == 199999) std::fprintf(stderr, "[draw-snapshots] verify: %llu fast answers checked, %llu differing%c", static_cast<unsigned long long>(checked.load()), static_cast<unsigned long long>(differing.load()), 10);
        }
        if (buffer == nullptr) {
            const bool movedUpload = override != moved.end() && bytes > uploadLimit && bytes <= MovedUploadBytes && RuntimeSwitches::Set(RuntimeSwitch::MovedLargeUploads);
            const bool uploaded = bytes != 0 && (bytes <= uploadLimit || movedUpload);
            if (census) {
                if (uploaded) ++counted.uploads;
                else ++counted.large;
            }
            if (movedUpload && MovedSnapshotStats()) {
                movedSnapshotTally().uploaded.fetch_add(1, std::memory_order_relaxed);
                movedSnapshotTally().uploadedBytes.fetch_add(bytes, std::memory_order_relaxed);
            }
            if (uploaded) {
                auto slice = recorder.DrawUpload(bytes);
                std::memcpy(slice.bytes, reinterpret_cast<const void*>(begin), bytes);
                buffer = std::move(slice.buffer);
                offset = slice.offset;
            } else {
                const auto generation = GuestMemory::CollectWrites(begin, bytes);
                buffer = recorder.ReusableDrawSnapshot(begin, bytes, Recorder::SnapshotUse::Storage, nullptr, generation, &offset);
                if (override != moved.end() && bytes <= MovedUploadBytes && MovedSnapshotStats()) {
                    movedSnapshotTally().cached.fetch_add(1, std::memory_order_relaxed);
                    if (buffer != nullptr) movedSnapshotTally().reused.fetch_add(1, std::memory_order_relaxed);
                    movedSnapshotTally().cachedBytes.fetch_add(bytes, std::memory_order_relaxed);
                }
                if (buffer == nullptr) {
                    buffer = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                    std::memcpy(buffer->Bytes().data(), reinterpret_cast<const void*>(begin), bytes);
                    recorder.KeepDrawSnapshot(begin, bytes, generation, registryGeneration, buffer);
                }
            }
            const auto current = RuntimeSwitches::Set(RuntimeSwitch::UnlockedSnapshotRecheck) ? GuestMemory::WriteGeneration() : GuestMemory::TrackerGeneration();
            if (current != stamp[0]) stamped = false;
            else if (!uploaded || stamp[1] != 0) recorder.NoteUnwrittenDrawSnapshot(begin, bytes, stamp, buffer, offset);
        }
        selected.push_back(index);
        keep({begin, std::move(buffer), offset, bytes});
        totalSnapshotBytes += bytes;
        if (census) {
            if (override != moved.end()) ++counted.moved;
            else ++counted.unmoved;
            counted.bytes += bytes;
        }
        CaptureTrace::Log("draw-snapshot batch=%llu address=%llx bytes=%zu", static_cast<unsigned long long>(recorder.Submissions() + 1), static_cast<unsigned long long>(begin), bytes);
    }
    if (census) NoteReplaySnapshots(counted.moved, counted.unmoved, counted.bytes, counted.memoHits, counted.uploads, counted.large);
    return result;
}

bool ShaderResources::unmovedClear(const Recorder& recorder, const PreparedBuffers& prepared, BindingRoute* route) const {
    const bool stamps = route != nullptr && route->stamps;
    if (stamps && (clearRecorder != recorder.Id() || clearNote != recorder.WriteNoteCount() || clearWrites != guestMemory.Writes().size() || clearAllocations.size() != allocations.size())) {
        clearRecorder = recorder.Id();
        clearNote = recorder.WriteNoteCount();
        clearWrites = guestMemory.Writes().size();
        clearAllocations.assign(allocations.size(), false);
    }
    for (const auto& item : prepared.selected) {
        if (item.moved || RuntimeSwitches::Set(RuntimeSwitch::XpCommitFloor) || RuntimeSwitches::Set(RuntimeSwitch::XpExpressFloor)) continue;
        const auto& allocation = allocations[item.allocation];
        if (route != nullptr) ++route->tested;
        if (stamps && clearAllocations[item.allocation]) {
            ++route->stamped;
            if (route->verifyStamps && (guestMemory.WritesOverlap(allocation.address, allocation.size) || recorder.PendingWriteOverlaps(allocation.address, allocation.size))) {
                ++route->refuted;
                return false;
            }
            continue;
        }
        if (guestMemory.WritesOverlap(allocation.address, allocation.size) || recorder.PendingWriteOverlaps(allocation.address, allocation.size)) return false;
        if (stamps) clearAllocations[item.allocation] = true;
    }
    return true;
}

namespace {

std::atomic<std::size_t> placedUploadChunks{0};
std::atomic<std::size_t> uploadChunkBytes{0};
constexpr std::size_t SharedUploadChunk = std::size_t{16} << 20u;

constexpr std::size_t UploadRingChunks = 8;

struct SharedUpload {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    std::size_t chunkBytes = 0;
    std::vector<std::shared_ptr<Buffer>> ring;
    std::size_t current = 0;
    std::shared_ptr<Buffer> overflow;
    std::size_t used = 0;
    VkMemoryPropertyFlags properties = 0;
};

struct UploadRingTally {
    std::atomic<std::uint64_t> reused{0};
    std::atomic<std::uint64_t> added{0};
    std::atomic<std::uint64_t> overflow{0};
    std::atomic<std::uint64_t> chunks{0};
    std::atomic<std::uint64_t> inFlight{0};
};

UploadRingTally& uploadRingTally() {
    static UploadRingTally tally;
    return tally;
}

SharedUpload& sharedUpload() {
    static auto* kept = new SharedUpload;
    return *kept;
}

}

ShaderResources::UploadRingCounts ShaderResources::UploadRingTotals() {
    auto& tally = uploadRingTally();
    return {tally.reused.load(std::memory_order_relaxed), tally.added.load(std::memory_order_relaxed), tally.overflow.load(std::memory_order_relaxed), tally.chunks.load(std::memory_order_relaxed), tally.inFlight.load(std::memory_order_relaxed)};
}

void ShaderResources::SetUploadChunkBytesForTest(std::size_t bytes) {
    uploadChunkBytes.store(bytes, std::memory_order_relaxed);
}

std::size_t ShaderResources::PlacedUploadChunks() {
    return placedUploadChunks.load(std::memory_order_acquire);
}

bool ShaderResources::PlaceDirectBuffers(const Context& context, PreparedBuffers& prepared) {
    prepared.placed.reset();
    prepared.placedDevice = VK_NULL_HANDLE;
    if (prepared.plan == nullptr) return false;
    struct Upload {
        std::weak_ptr<Buffer> chunk;
        VkDevice device = VK_NULL_HANDLE;
        std::size_t used = 0;
    };
    struct UploadStorage {};
    auto& upload = HostThreadLocal<Upload, UploadStorage>();
    const auto bytes = prepared.blob.size();
    std::shared_ptr<Buffer> buffer;
    std::size_t offset = 0;
    if (!prepared.selected.empty()) {
        const auto alignment = prepared.plan->alignment;
        const bool shared = RuntimeSwitches::Set(RuntimeSwitch::CommitBindingsDynamic);
        const auto chosen = uploadChunkBytes.load(std::memory_order_relaxed);
        const auto chunkBytes = chosen != 0 ? chosen : shared ? SharedUploadChunk : Recorder::DrawUploadChunk;
        const auto properties = UploadMemoryProperties(context);
        const auto make = [&] {
            auto* made = new Buffer(context, std::max(chunkBytes, bytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, properties);
            placedUploadChunks.fetch_add(1, std::memory_order_acq_rel);
            return std::shared_ptr<Buffer>(made, [](Buffer* held) {
                delete held;
                placedUploadChunks.fetch_sub(1, std::memory_order_acq_rel);
            });
        };
        if (shared) {
            auto& common = sharedUpload();
            auto& tally = uploadRingTally();
            std::lock_guard lock(common.mutex);
            if (common.device != context.device || common.chunkBytes != chunkBytes || common.properties != properties) {
                common.ring.clear();
                common.overflow.reset();
                common.device = context.device;
                common.chunkBytes = chunkBytes;
                common.properties = properties;
                common.current = 0;
                common.used = 0;
                tally.chunks.store(0, std::memory_order_relaxed);
            }
            const auto* held = common.overflow != nullptr ? &common.overflow : common.current < common.ring.size() ? &common.ring[common.current] : nullptr;
            offset = (common.used + alignment - 1) & ~(alignment - 1);
            if (held == nullptr || offset + bytes > (*held)->Bytes().size()) {
                common.overflow.reset();
                bool found = false;
                std::uint64_t busy = 0;
                for (std::size_t step = 1; step <= common.ring.size(); ++step) {
                    const auto index = (common.current + step) % common.ring.size();
                    if (common.ring[index].use_count() != 1 || common.ring[index]->Bytes().size() < bytes) {
                        ++busy;
                        continue;
                    }
                    if (found) continue;
                    std::atomic_thread_fence(std::memory_order_acquire);
                    common.current = index;
                    found = true;
                    tally.reused.fetch_add(1, std::memory_order_relaxed);
                }
                if (busy > tally.inFlight.load(std::memory_order_relaxed)) tally.inFlight.store(busy, std::memory_order_relaxed);
                if (!found && common.ring.size() < UploadRingChunks) {
                    common.ring.push_back(make());
                    common.current = common.ring.size() - 1;
                    tally.added.fetch_add(1, std::memory_order_relaxed);
                    tally.chunks.store(common.ring.size(), std::memory_order_relaxed);
                } else if (!found) {
                    common.overflow = make();
                    tally.overflow.fetch_add(1, std::memory_order_relaxed);
                }
                held = common.overflow != nullptr ? &common.overflow : &common.ring[common.current];
                offset = 0;
            }
            buffer = *held;
            common.used = offset + bytes;
        } else {
            buffer = upload.device == context.device ? upload.chunk.lock() : nullptr;
            if (buffer != nullptr && buffer->Properties() != properties) buffer = nullptr;
            offset = (upload.used + alignment - 1) & ~(alignment - 1);
            if (buffer == nullptr || offset + bytes > buffer->Bytes().size()) {
                buffer = make();
                upload.chunk = buffer;
                upload.device = context.device;
                offset = 0;
            }
            upload.used = offset + bytes;
        }
        std::memcpy(buffer->Bytes().data() + offset, prepared.blob.data(), bytes);
        if ((properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 && Recorder::GpuSideStats()) Recorder::CountGpuSide(Recorder::GpuSide::DeviceUploadBytes, bytes);
    }
    auto placed = std::make_shared<DrawBindings>();
    placed->snapshots.reserve(prepared.selected.size());
    placed->pushSelected.reserve(prepared.selected.size());
    for (const auto& item : prepared.selected) {
        if (item.inPlace) placed->snapshots.push_back({item.begin, nullptr, item.importOffset, item.bytes, item.handle});
        else placed->snapshots.push_back({item.begin, buffer, offset + item.offset, item.bytes});
        placed->pushSelected.push_back(item.allocation);
    }
    prepared.placed = std::move(placed);
    prepared.placedDevice = context.device;
    return true;
}

ShaderResources::PreparedBuffers::Selection ShaderResources::FragmentSelectionHolds(Recorder& recorder, std::span<const MovedBuffer> moved, const PreparedBuffers& prepared, BindingRoute* route, bool untouched, CommitterTrustCheck* trust, std::uint8_t answered) const {
    auto selection = PreparedBuffers::Selection::Hazard;
    static_cast<void>(preparedSelectionHolds(recorder, moved, prepared, selection, route, untouched, nullptr, RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter), trust, answered));
    return selection;
}

ShaderResources::PreparedBuffers::Selection ShaderResources::FragmentSelectionHoldsBy(Recorder& recorder, std::span<const MovedBuffer> moved, const PreparedBuffers& prepared, bool untouched, const GuardTests* guards, CommitterTrustCheck* trust) const {
    auto selection = PreparedBuffers::Selection::Hazard;
    static_cast<void>(preparedSelectionHolds(recorder, moved, prepared, selection, nullptr, untouched, guards, RuntimeSwitches::Set(RuntimeSwitch::XpTrustCommitter), trust));
    return selection;
}

bool ShaderResources::preparedSelectionHolds(Recorder& recorder, std::span<const MovedBuffer> moved, const PreparedBuffers& prepared, PreparedBuffers::Selection& selection, BindingRoute* route, bool untouched, const GuardTests* guards, bool trusted, CommitterTrustCheck* trust, std::uint8_t answered) const {
    using Selection = PreparedBuffers::Selection;
    selection = Selection::Hazard;
    const auto& plan = *prepared.plan;
    const auto alignment = context.limits.minStorageBufferOffsetAlignment;
    if (alignment != 0 && plan.alignment % alignment != 0) return false;
    // requireGuestBuffers publishes the host-written samples of a resident depth surface into the
    // memory it overlaps, after the finisher read that memory; a guard set up or dropped since the
    // read may have published them already.
    if (trusted) NoteCommitterTrusted(3);
    const bool skipped = trusted || (trust != nullptr && trust->Skip());
    const auto failed = [trust] {
        if (trust != nullptr) trust->Fail();
        return false;
    };
    const bool guardsMoved = !skipped && DepthGuardsGeneration_nid_no_patch() != prepared.depthGuards;
    const auto depthGuardHit = [&](std::uint64_t begin, std::uint64_t bytes) { return guardsMoved || (guards != nullptr ? guards->depth(guards->tests, begin, bytes) : MayOverlapDepthGuards_nid_no_patch(begin, static_cast<std::size_t>(bytes))); };
    const auto depthGuarded = [&]() -> bool {
        if (!zoneCardUsable(prepared, ZoneCardUse::Guards)) return std::ranges::any_of(prepared.selected, [&](const PreparedBuffers::Selected& item) { return item.begin != 0 && !item.inPlace && depthGuardHit(item.begin, item.bytes); });
        const auto& card = prepared.card;
        const std::size_t end = static_cast<std::size_t>(card.zones) + static_cast<std::size_t>(card.guards);
        for (std::size_t index = card.zones; index < end; ++index) {
            if (depthGuardHit(card.entries[index].first, card.entries[index].second)) return true;
        }
        return false;
    };
    const bool depthAnswered = (answered & GuardAnswer::DepthClear) != 0 && guards == nullptr && !guardsMoved;
    if (!skipped && (guardsMoved || HasResidentDepthGuards_nid_no_patch()) && !depthAnswered && depthGuarded()) {
        selection = Selection::DepthGuards;
        return failed();
    }
    bool guarded = false;
    bool* const found = trust != nullptr && trust->Audits() ? &guarded : nullptr;
    const struct GuardFails {
        CommitterTrustCheck* check;
        const bool& seen;
        ~GuardFails() {
            if (seen) check->Fail();
        }
    } guardFails{trust, guarded};
    if (skipped) {
    } else if (route != nullptr && route->stamps && requireMovedGuestBuffers(moved, nullptr, found)) {
        ++route->guards;
        if (route->verifyStamps && std::ranges::any_of(moved, [&](const MovedBuffer& entry) { return allocations[entry.allocation].guest && MayOverlapDepthGuards_nid_no_patch(entry.address, entry.size); })) {
            ++route->refuted;
            requireGuestBuffers(moved, found);
        }
    } else if (route == nullptr && RuntimeSwitches::Set(RuntimeSwitch::CommitStamps) && requireMovedGuestBuffers(moved, guards, found, (answered & GuardAnswer::MultisampleClear) != 0)) {
    } else {
        requireGuestBuffers(moved, found);
    }
    prepareFaultSlot();
    bindingsChecked = heapStage && RuntimeSwitches::Set(RuntimeSwitch::ReplayBind);
    if (skipped) {
    } else if (route != nullptr) {
        if (!unmovedClear(recorder, prepared, route)) return failed();
    } else {
        for (const auto& item : prepared.selected) {
            if (item.moved || untouched || RuntimeSwitches::Set(RuntimeSwitch::XpCommitFloor) || RuntimeSwitches::Set(RuntimeSwitch::XpExpressFloor)) continue;
            const auto& allocation = allocations[item.allocation];
            if (guestMemory.WritesOverlap(allocation.address, allocation.size) || (guards != nullptr && guards->writes != nullptr ? guards->writes(guards->tests, allocation.address, allocation.size) : recorder.PendingWriteOverlaps(allocation.address, allocation.size))) return failed();
        }
    }
    selection = Selection::Taken;
    if (route != nullptr) route->selection = &prepared;
    return true;
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::selectPreparedSnapshots(Recorder& recorder, std::span<const MovedBuffer> moved, const PreparedBuffers& prepared, std::vector<std::size_t>& selected, bool binds, PreparedBuffers::Selection& selection, BindingRoute* route) const {
    selected.clear();
    if (!preparedSelectionHolds(recorder, moved, prepared, selection, route)) return {};
    const auto& plan = *prepared.plan;
    if (route != nullptr && route->placed && binds && prepared.placed != nullptr) {
        if (prepared.placedDevice != context.device) {
            route->taken = BindingTaken::OtherDevice;
        } else {
            route->taken = BindingTaken::Placed;
            route->bytes = prepared.blob.size();
            auto result = std::move(prepared.placed);
            for (const auto& item : prepared.selected) {
                selected.push_back(item.allocation);
                if (item.begin == 0) continue;
                if (item.moved && item.bytes > plan.uploadLimit && MovedSnapshotStats()) {
                    movedSnapshotTally().uploaded.fetch_add(1, std::memory_order_relaxed);
                    movedSnapshotTally().uploadedBytes.fetch_add(item.bytes, std::memory_order_relaxed);
                }
                CaptureTrace::Log("draw-snapshot batch=%llu address=%llx bytes=%zu", static_cast<unsigned long long>(recorder.Submissions() + 1), static_cast<unsigned long long>(item.begin), item.bytes);
            }
            return result;
        }
    }
    if (route != nullptr && route->taken == BindingTaken::None) route->taken = BindingTaken::Copied;
    auto result = std::make_shared<DrawBindings>();
    if (prepared.selected.empty()) return result;
    auto slice = recorder.DrawUpload(prepared.blob.size());
    std::memcpy(slice.bytes, prepared.blob.data(), prepared.blob.size());
    result->snapshots.reserve(prepared.selected.size());
    for (const auto& item : prepared.selected) {
        selected.push_back(item.allocation);
        if (item.inPlace) result->snapshots.push_back({item.begin, nullptr, item.importOffset, item.bytes, item.handle});
        else result->snapshots.push_back({item.begin, slice.buffer, slice.offset + item.offset, item.bytes});
        if (!binds || item.begin == 0) continue;
        if (item.moved && item.bytes > plan.uploadLimit && MovedSnapshotStats()) {
            movedSnapshotTally().uploaded.fetch_add(1, std::memory_order_relaxed);
            movedSnapshotTally().uploadedBytes.fetch_add(item.bytes, std::memory_order_relaxed);
        }
        CaptureTrace::Log("draw-snapshot batch=%llu address=%llx bytes=%zu", static_cast<unsigned long long>(recorder.Submissions() + 1), static_cast<unsigned long long>(item.begin), item.bytes);
    }
    return result;
}

void ShaderResources::writeDrawSets(DrawBindings& result, std::span<const std::size_t> selected, const BdaResources* faultRecord, std::int32_t pushedSet) const {
    Require(context.descriptorCache != nullptr, "draw snapshots require a descriptor cache");
    if (heapStage && !RuntimeSwitches::Set(RuntimeSwitch::NoHeapSetSkip)) {
        result.cache = context.descriptorCache;
        writeAddressedOverlay(result, selected);
        return;
    }
    if (completed && RuntimeSwitches::Set(RuntimeSwitch::OwnDrawSets)) {
        writeOwnDrawSets(result, selected, faultRecord, pushedSet);
        return;
    }
    result.lent = {};
    if (ownDrawSets != nullptr && ownDrawSets->Parked() != 0) ownDrawSets->Drain(false);
    writeFreshDrawSets(result, selected, faultRecord, pushedSet);
}

void ShaderResources::WriteFragmentSets(DrawBindings& result, std::span<const std::size_t> selected, std::int32_t pushedSet) const {
    Require(context.descriptorCache != nullptr && !heapStage, "a fragment's sets need a descriptor cache and no heap stage");
    result.lent = {};
    if (RuntimeSwitches::Set(RuntimeSwitch::FragmentParkedSets)) {
        writeParkedFragmentSets(result, selected, pushedSet);
        return;
    }
    if (fragmentSetsReady.load(std::memory_order_acquire) && fragmentSets->Parked() != 0) fragmentSets->Drain(false);
    if (RuntimeSwitches::Set(RuntimeSwitch::FragmentPlans)) {
        writePlannedFragmentSets(result, selected, pushedSet);
        return;
    }
    writeFreshDrawSets(result, selected, nullptr, pushedSet);
}

namespace {

struct FragmentSetPlan {
    struct Write {
        std::size_t index = 0;
        std::size_t binding = 0;
        std::size_t offset = 0;
        std::uint8_t set = 0;
        bool patched = false;
    };
    std::uint64_t serial = 0;
    std::int32_t pushedSet = -2;
    std::size_t writeCount = 0;
    std::size_t bufferCount = 0;
    std::vector<std::pair<std::size_t, std::size_t>> variable;
    std::array<std::vector<VkDescriptorPoolSize>, 2> sizes;
    std::array<bool, 2> described{};
    std::array<std::optional<std::uint32_t>, 2> variableCounts;
    std::vector<Write> writes;
};

struct FragmentSetPlans {
    static constexpr std::size_t Slots = 1024;
    std::vector<FragmentSetPlan> slots;
    ShaderResources::FragmentPlanNote note;
    std::uint64_t verifyOrdinal = 0;
};

struct FragmentSetPlansStorage {};

std::atomic<std::uint64_t> planSerials{0};

const char* plannedWritesDifference(std::span<const VkWriteDescriptorSet> mine, std::span<const VkWriteDescriptorSet> today) {
    if (mine.size() != today.size()) return "the write count";
    for (std::size_t at = 0; at < mine.size(); ++at) {
        const auto& a = mine[at];
        const auto& b = today[at];
        if (a.dstSet != b.dstSet) return "a write's set";
        if (a.dstBinding != b.dstBinding || a.dstArrayElement != b.dstArrayElement || a.descriptorCount != b.descriptorCount || a.descriptorType != b.descriptorType) return "a write's slot";
        if (a.pNext != b.pNext || a.pImageInfo != b.pImageInfo || a.pTexelBufferView != b.pTexelBufferView || (a.pBufferInfo == nullptr) != (b.pBufferInfo == nullptr)) return "a write's infos";
        if (a.pBufferInfo == nullptr) continue;
        for (std::uint32_t element = 0; element < a.descriptorCount; ++element) {
            const auto& x = a.pBufferInfo[element];
            const auto& y = b.pBufferInfo[element];
            if (x.buffer != y.buffer || x.offset != y.offset || x.range != y.range) return "a buffer info";
        }
    }
    return nullptr;
}

void notePlanDifference(const char* what, std::size_t planned, std::size_t today, std::int32_t pushedSet) {
    static std::atomic<std::uint32_t> printed{0};
    if (printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[fragment-plans-verify] %s differs from today's: %zu writes planned, %zu today, pushed set %d\n", what, planned, today, static_cast<int>(pushedSet));
}

}

std::uint64_t ShaderResources::NextPlanSerial() {
    return planSerials.fetch_add(1, std::memory_order_relaxed) + 1u;
}

ShaderResources::FragmentPlanNote ShaderResources::TakeFragmentPlanNote() {
    auto& plans = HostThreadLocal<FragmentSetPlans, FragmentSetPlansStorage>();
    return std::exchange(plans.note, {});
}

void ShaderResources::writePlannedFragmentSets(DrawBindings& result, std::span<const std::size_t> selected, std::int32_t pushedSet) const {
    auto& plans = HostThreadLocal<FragmentSetPlans, FragmentSetPlansStorage>();
    if (plans.slots.empty()) plans.slots.resize(FragmentSetPlans::Slots);
    auto mixed = planSerial * 0x9e3779b97f4a7c15ull ^ static_cast<std::uint64_t>(pushedSet + 2);
    mixed ^= mixed >> 29u;
    auto& plan = plans.slots[static_cast<std::size_t>(mixed % FragmentSetPlans::Slots)];
    bool valid = plan.serial == planSerial && plan.pushedSet == pushedSet && plan.writeCount == descriptorWrites.size() && plan.bufferCount == descriptorBuffers.size();
    for (std::size_t at = 0; valid && at < plan.variable.size(); ++at) valid = plan.variable[at].first < bindings.size() && bindings[plan.variable[at].first].imageAllocations.size() == plan.variable[at].second;
    if (valid) {
        ++plans.note.hit;
    } else {
        ++plans.note.built;
        plan = {};
        plan.serial = planSerial;
        plan.pushedSet = pushedSet;
        plan.writeCount = descriptorWrites.size();
        plan.bufferCount = descriptorBuffers.size();
        for (std::size_t index = 0; index < bindings.size(); ++index) {
            const auto& binding = bindings[index];
            if (binding.heap) continue;
            const auto set = setOf(binding);
            plan.described[set] = true;
            const bool variable = (binding.flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) != 0;
            if (variable) plan.variable.emplace_back(index, binding.imageAllocations.size());
            const auto count = variable ? std::max(1u, static_cast<std::uint32_t>(binding.imageAllocations.size())) : binding.layout.descriptorCount;
            if (count == 0) continue;
            auto& sizes = plan.sizes[set];
            const auto held = std::find_if(sizes.begin(), sizes.end(), [&](const VkDescriptorPoolSize& entry) { return entry.type == binding.layout.descriptorType; });
            if (held != sizes.end()) held->descriptorCount += count;
            else sizes.push_back({binding.layout.descriptorType, count});
        }
        for (auto& sizes : plan.sizes) std::sort(sizes.begin(), sizes.end(), [](const VkDescriptorPoolSize& a, const VkDescriptorPoolSize& b) { return a.type < b.type; });
        for (std::size_t set = 0; set < plan.variableCounts.size(); ++set) {
            if (tableBindings[set] == std::numeric_limits<std::size_t>::max()) continue;
            plan.variableCounts[set] = std::max(1u, static_cast<std::uint32_t>(bindings[tableBindings[set]].imageAllocations.size()));
            plan.variable.emplace_back(tableBindings[set], bindings[tableBindings[set]].imageAllocations.size());
        }
        std::size_t next = 0;
        for (std::size_t index = 0; index < bindings.size(); ++index) {
            const auto& binding = bindings[index];
            if (binding.heap) continue;
            const auto original = setOf(binding) == 0 ? _set : fragment.set;
            while (next < descriptorWrites.size() && descriptorWrites[next].dstBinding == binding.layout.binding && descriptorWrites[next].dstSet == original) {
                const bool patched = !binding.allocations.empty();
                plan.writes.push_back({next, index, patched ? static_cast<std::size_t>(descriptorWrites[next].pBufferInfo - descriptorBuffers.data()) : 0u, static_cast<std::uint8_t>(setOf(binding)), patched});
                ++next;
            }
        }
        if (next != descriptorWrites.size()) {
            plan = {};
            Require(false, "draw snapshot writes disagree with their descriptor bindings");
        }
    }
    result.cache = context.descriptorCache;
    if ((plan.described[0] || splitSets) && pushedSet != 0) {
        result.allocation = result.cache->AllocateRewritten(_layout, plan.sizes[0], updateAfterBind, plan.variableCounts[0]);
        result.layout = _layout;
        Require(result.allocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    }
    if (!heapStage && (plan.described[1] || splitSets) && pushedSet != 1) {
        result.fragmentAllocation = result.cache->AllocateRewritten(fragment.layout, plan.sizes[1], updateAfterBind, plan.variableCounts[1]);
        result.fragmentLayout = fragment.layout;
        Require(result.fragmentAllocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    }
    struct BuffersStorage {};
    auto& buffers = HostThreadLocal<std::vector<VkDescriptorBufferInfo>, BuffersStorage>();
    if (buffers.size() != descriptorBuffers.size()) buffers.resize(descriptorBuffers.size());
    struct IndicesStorage {};
    auto& snapshotIndices = HostThreadLocal<std::vector<std::size_t>, IndicesStorage>();
    snapshotIndices.assign(allocations.size(), selected.size());
    for (std::size_t index = 0; index < selected.size(); ++index) snapshotIndices[selected[index]] = index;
    for (const auto& planned : plan.writes) {
        if (!planned.patched || planned.offset >= buffers.size()) continue;
        const auto reach = std::max<std::size_t>(descriptorWrites[planned.index].descriptorCount, bindings[planned.binding].allocations.size());
        const auto end = std::min(planned.offset + reach, buffers.size());
        std::copy(descriptorBuffers.begin() + static_cast<std::ptrdiff_t>(planned.offset), descriptorBuffers.begin() + static_cast<std::ptrdiff_t>(end), buffers.begin() + static_cast<std::ptrdiff_t>(planned.offset));
    }
    for (const auto& planned : plan.writes) {
        if (!planned.patched) continue;
        const auto& binding = bindings[planned.binding];
        for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
            const auto snapshotIndex = snapshotIndices[binding.allocations[element]];
            if (snapshotIndex == selected.size() || planned.offset + element >= buffers.size()) continue;
            const auto& snapshot = result.snapshots[snapshotIndex];
            buffers[planned.offset + element] = wholeDwordRange({snapshot.Handle(), snapshot.offset, snapshot.bytes});
        }
    }
    struct WritesStorage {};
    auto& writes = HostThreadLocal<std::vector<VkWriteDescriptorSet>, WritesStorage>();
    writes.clear();
    for (const auto& planned : plan.writes) {
        const auto set = planned.set == 0 ? result.allocation.set : result.fragmentAllocation.set;
        if (pushedSet >= 0 && set == VK_NULL_HANDLE) continue;
        auto& write = writes.emplace_back(descriptorWrites[planned.index]);
        write.dstSet = set;
        if (planned.patched) write.pBufferInfo = buffers.data() + planned.offset;
    }
    if (const auto every = CommitFragmentsVerifyEvery(); every != 0 && ++plans.verifyOrdinal % every == 0) {
        ++plans.note.checked;
        std::array<bool, 2> described{};
        std::array<std::vector<VkDescriptorPoolSize>, 2> sizes;
        for (const auto& binding : bindings) {
            if (binding.heap) continue;
            const auto set = setOf(binding);
            described[set] = true;
            const auto count = (binding.flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) != 0 ? std::max(1u, static_cast<std::uint32_t>(binding.imageAllocations.size())) : binding.layout.descriptorCount;
            if (count == 0) continue;
            const auto held = std::find_if(sizes[set].begin(), sizes[set].end(), [&](const VkDescriptorPoolSize& entry) { return entry.type == binding.layout.descriptorType; });
            if (held != sizes[set].end()) held->descriptorCount += count;
            else sizes[set].push_back({binding.layout.descriptorType, count});
        }
        std::vector<VkDescriptorBufferInfo> todayBuffers = descriptorBuffers;
        std::vector<VkWriteDescriptorSet> todayWrites = descriptorWrites;
        auto nextWrite = todayWrites.begin();
        for (const auto& binding : bindings) {
            if (binding.heap) continue;
            const auto original = setOf(binding) == 0 ? _set : fragment.set;
            while (nextWrite != todayWrites.end() && nextWrite->dstBinding == binding.layout.binding && nextWrite->dstSet == original) {
                auto& write = *nextWrite++;
                write.dstSet = setOf(binding) == 0 ? result.allocation.set : result.fragmentAllocation.set;
                if (binding.allocations.empty()) continue;
                const auto offset = static_cast<std::size_t>(write.pBufferInfo - descriptorBuffers.data());
                write.pBufferInfo = todayBuffers.data() + offset;
                for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
                    const auto snapshotIndex = snapshotIndices[binding.allocations[element]];
                    if (snapshotIndex == selected.size()) continue;
                    const auto& snapshot = result.snapshots[snapshotIndex];
                    todayBuffers[offset + element] = wholeDwordRange({snapshot.Handle(), snapshot.offset, snapshot.bytes});
                }
            }
        }
        const char* what = nextWrite != todayWrites.end() ? "the write shape" : nullptr;
        if (pushedSet >= 0) std::erase_if(todayWrites, [](const VkWriteDescriptorSet& write) { return write.dstSet == VK_NULL_HANDLE; });
        const auto samePool = [](const VkDescriptorPoolSize& a, const VkDescriptorPoolSize& b) { return a.type == b.type && a.descriptorCount == b.descriptorCount; };
        for (std::size_t set = 0; what == nullptr && set < sizes.size(); ++set) {
            std::sort(sizes[set].begin(), sizes[set].end(), [](const VkDescriptorPoolSize& a, const VkDescriptorPoolSize& b) { return a.type < b.type; });
            const auto variable = tableBindings[set] != std::numeric_limits<std::size_t>::max() ? std::optional<std::uint32_t>(std::max(1u, static_cast<std::uint32_t>(bindings[tableBindings[set]].imageAllocations.size()))) : std::nullopt;
            if (described[set] != plan.described[set] || !std::ranges::equal(sizes[set], plan.sizes[set], samePool) || variable != plan.variableCounts[set]) what = "the pool sizes";
        }
        if (what == nullptr) what = plannedWritesDifference(writes, todayWrites);
        if (what != nullptr) {
            ++plans.note.differing;
            notePlanDifference(what, writes.size(), todayWrites.size(), pushedSet);
            plan = {};
            Require(false, "a planned fragment set update differs from today's");
        }
    }
    writeAddressedOverlay(result, selected);
    context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void ShaderResources::writeFreshDrawSets(DrawBindings& result, std::span<const std::size_t> selected, const BdaResources* faultRecord, std::int32_t pushedSet) const {
    std::array<bool, 2> described{};
    struct CountsStorage {};
    auto& counts = HostThreadLocal<std::array<std::vector<VkDescriptorPoolSize>, 2>, CountsStorage>();
    for (auto& sizes : counts) sizes.clear();
    for (const auto& binding : bindings) {
        if (binding.heap) continue;
        const auto set = setOf(binding);
        described[set] = true;
        const auto count = (binding.flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) != 0 ? std::max(1u, static_cast<std::uint32_t>(binding.imageAllocations.size())) : binding.layout.descriptorCount;
        if (count == 0) continue;
        auto& sizes = counts[set];
        const auto held = std::find_if(sizes.begin(), sizes.end(), [&](const VkDescriptorPoolSize& entry) { return entry.type == binding.layout.descriptorType; });
        if (held != sizes.end()) held->descriptorCount += count;
        else sizes.push_back({binding.layout.descriptorType, count});
    }
    for (auto& sizes : counts) std::sort(sizes.begin(), sizes.end(), [](const VkDescriptorPoolSize& a, const VkDescriptorPoolSize& b) { return a.type < b.type; });
    const auto sizesOf = [&](std::size_t set) { return std::span<const VkDescriptorPoolSize>(counts[set]); };
    const auto variableOf = [&](std::size_t set) { return tableBindings[set] != std::numeric_limits<std::size_t>::max() ? std::optional<std::uint32_t>(std::max(1u, static_cast<std::uint32_t>(bindings[tableBindings[set]].imageAllocations.size()))) : std::nullopt; };
    result.cache = context.descriptorCache;
    if ((described[0] || splitSets) && pushedSet != 0) {
        result.allocation = result.cache->AllocateRewritten(_layout, sizesOf(0), updateAfterBind, variableOf(0));
        result.layout = _layout;
        Require(result.allocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    }
    if (!heapStage && (described[1] || splitSets) && pushedSet != 1) {
        result.fragmentAllocation = result.cache->AllocateRewritten(fragment.layout, sizesOf(1), updateAfterBind, variableOf(1));
        result.fragmentLayout = fragment.layout;
        Require(result.fragmentAllocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    }
    struct BuffersStorage {};
    auto& buffers = HostThreadLocal<std::vector<VkDescriptorBufferInfo>, BuffersStorage>();
    buffers = descriptorBuffers;
    if (faultRecord != nullptr) {
        for (const auto& fault : faultDescriptors) buffers[fault.buffer] = faultRecord->Fault();
    }
    struct WritesStorage {};
    auto& writes = HostThreadLocal<std::vector<VkWriteDescriptorSet>, WritesStorage>();
    writes = descriptorWrites;
    struct IndicesStorage {};
    auto& snapshotIndices = HostThreadLocal<std::vector<std::size_t>, IndicesStorage>();
    snapshotIndices.assign(allocations.size(), selected.size());
    for (std::size_t index = 0; index < selected.size(); ++index) snapshotIndices[selected[index]] = index;
    auto nextWrite = writes.begin();
    for (const auto& binding : bindings) {
        if (binding.heap) continue;
        const auto original = setOf(binding) == 0 ? _set : fragment.set;
        while (nextWrite != writes.end() && nextWrite->dstBinding == binding.layout.binding && nextWrite->dstSet == original) {
            auto& write = *nextWrite++;
            write.dstSet = setOf(binding) == 0 ? result.allocation.set : result.fragmentAllocation.set;
            if (binding.allocations.empty()) continue;
            const auto offset = static_cast<std::size_t>(write.pBufferInfo - descriptorBuffers.data());
            write.pBufferInfo = buffers.data() + offset;
            for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
                const auto snapshotIndex = snapshotIndices[binding.allocations[element]];
                if (snapshotIndex == selected.size()) continue;
                const auto& snapshot = result.snapshots[snapshotIndex];
                buffers[offset + element] = wholeDwordRange({snapshot.Handle(), snapshot.offset, snapshot.bytes});
            }
        }
    }
    Require(nextWrite == writes.end(), "draw snapshot writes disagree with their descriptor bindings");
    if (pushedSet >= 0) std::erase_if(writes, [](const VkWriteDescriptorSet& write) { return write.dstSet == VK_NULL_HANDLE; });
    writeAddressedOverlay(result, selected);
    context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

std::shared_ptr<OwnDrawSets> ShaderResources::makeOwnDrawSets(bool fragmentStore) const {
    OwnDrawSets::Plan plan;
    for (std::size_t index = 0; fragmentStore && index < bindings.size(); ++index) {
        const auto& binding = bindings[index];
        if (!binding.heap && (binding.flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) != 0) plan.variableBindings.emplace_back(index, binding.imageAllocations.size());
    }
    if (fragmentStore) plan.bufferCount = descriptorBuffers.size();
    for (const auto& binding : bindings) {
        if (binding.heap) continue;
        const auto set = setOf(binding);
        plan.described[set] = true;
        const auto count = (binding.flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) != 0 ? std::max(1u, static_cast<std::uint32_t>(binding.imageAllocations.size())) : binding.layout.descriptorCount;
        if (count == 0) continue;
        auto& sizes = plan.sizes[set];
        const auto held = std::find_if(sizes.begin(), sizes.end(), [&](const VkDescriptorPoolSize& entry) { return entry.type == binding.layout.descriptorType; });
        if (held != sizes.end()) held->descriptorCount += count;
        else sizes.push_back({binding.layout.descriptorType, count});
    }
    for (auto& sizes : plan.sizes) std::sort(sizes.begin(), sizes.end(), [](const VkDescriptorPoolSize& a, const VkDescriptorPoolSize& b) { return a.type < b.type; });
    for (std::size_t set = 0; set < plan.variable.size(); ++set) {
        if (tableBindings[set] != std::numeric_limits<std::size_t>::max()) plan.variable[set] = std::max(1u, static_cast<std::uint32_t>(bindings[tableBindings[set]].imageAllocations.size()));
    }
    plan.sets.assign(descriptorWrites.size(), 0);
    plan.buffers.assign(descriptorWrites.size(), std::numeric_limits<std::size_t>::max());
    plan.slots.resize(allocations.size());
    plan.allocationSets.assign(allocations.size(), 0xffu);
    plan.lendable = {true, true};
    std::size_t next = 0;
    for (const auto& binding : bindings) {
        if (binding.heap) continue;
        const auto set = setOf(binding);
        const auto original = set == 0 ? _set : fragment.set;
        for (const auto index : binding.allocations) {
            if (index < plan.allocationSets.size()) plan.allocationSets[index] = plan.allocationSets[index] == 0xffu || plan.allocationSets[index] == set ? static_cast<std::uint8_t>(set) : 0xfeu;
        }
        while (next < descriptorWrites.size() && descriptorWrites[next].dstBinding == binding.layout.binding && descriptorWrites[next].dstSet == original) {
            if (descriptorWrites[next].pTexelBufferView != nullptr || (!binding.allocations.empty() && descriptorWrites[next].descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)) plan.lendable[set] = false;
            plan.sets[next] = static_cast<std::uint8_t>(set);
            if (!binding.allocations.empty()) {
                const auto offset = static_cast<std::size_t>(descriptorWrites[next].pBufferInfo - descriptorBuffers.data());
                plan.buffers[next] = offset;
                for (std::size_t element = 0; element < binding.allocations.size(); ++element) plan.slots[binding.allocations[element]].push_back(offset + element);
            }
            ++next;
        }
    }
    Require(next == descriptorWrites.size(), "draw snapshot writes disagree with their descriptor bindings");
    return std::make_shared<OwnDrawSets>(*context.descriptorCache, std::array<VkDescriptorSetLayout, 2>{_layout, fragment.layout}, std::move(plan), fragmentStore);
}

void ShaderResources::writeOwnDrawSets(DrawBindings& result, std::span<const std::size_t> selected, const BdaResources* faultRecord, std::int32_t pushedSet) const {
    if (ownDrawSets == nullptr) ownDrawSets = makeOwnDrawSets();
    const auto& plan = ownDrawSets->plan;
    result.cache = context.descriptorCache;
    result.ownSets = ownDrawSets;
    if ((result.lent[0] || result.lent[1]) && !((plan.described[0] || splitSets) && pushedSet != 0 && !result.lent[0]) && !(!heapStage && (plan.described[1] || splitSets) && pushedSet != 1 && !result.lent[1])) {
        writeAddressedOverlay(result, selected);
        result.owned = {};
        return;
    }
    std::array<bool, 2> pooled{};
    if ((plan.described[0] || splitSets) && pushedSet != 0 && !result.lent[0]) {
        pooled[0] = ownDrawSets->Take(0, result.allocation);
        if (!pooled[0]) result.allocation = result.cache->AllocateRewritten(_layout, plan.sizes[0], updateAfterBind, plan.variable[0]);
        result.layout = _layout;
        Require(result.allocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    }
    if (!heapStage && (plan.described[1] || splitSets) && pushedSet != 1 && !result.lent[1]) {
        pooled[1] = ownDrawSets->Take(1, result.fragmentAllocation);
        if (!pooled[1]) result.fragmentAllocation = result.cache->AllocateRewritten(fragment.layout, plan.sizes[1], updateAfterBind, plan.variable[1]);
        result.fragmentLayout = fragment.layout;
        Require(result.fragmentAllocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    }
    struct BuffersStorage {};
    auto& buffers = HostThreadLocal<std::vector<VkDescriptorBufferInfo>, BuffersStorage>();
    buffers = descriptorBuffers;
    if (faultRecord != nullptr) {
        for (const auto& fault : faultDescriptors) buffers[fault.buffer] = faultRecord->Fault();
    }
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const auto& snapshot = result.snapshots[index];
        for (const auto slot : plan.slots[selected[index]]) buffers[slot] = wholeDwordRange({snapshot.Handle(), snapshot.offset, snapshot.bytes});
    }
    struct WritesStorage {};
    auto& writes = HostThreadLocal<std::vector<VkWriteDescriptorSet>, WritesStorage>();
    writes.clear();
    for (std::size_t index = 0; index < descriptorWrites.size(); ++index) {
        const auto set = plan.sets[index];
        if (static_cast<std::int32_t>(set) == pushedSet) continue;
        if (result.lent[set]) continue;
        if (pooled[set] && descriptorWrites[index].descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) continue;
        auto& write = writes.emplace_back(descriptorWrites[index]);
        write.dstSet = set == 0 ? result.allocation.set : result.fragmentAllocation.set;
        if (plan.buffers[index] != std::numeric_limits<std::size_t>::max()) write.pBufferInfo = buffers.data() + plan.buffers[index];
    }
    writeAddressedOverlay(result, selected);
    if (!writes.empty() || (!result.lent[0] && !result.lent[1])) context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    result.owned = {result.allocation.set != VK_NULL_HANDLE && !result.lent[0], result.fragmentAllocation.set != VK_NULL_HANDLE && !result.lent[1]};
}

namespace {

struct ParkedSetsNotes {
    ShaderResources::ParkedSetsNote note;
    std::uint64_t verifyOrdinal = 0;
};

struct ParkedSetsNotesStorage {};

bool imageWrite(const VkWriteDescriptorSet& write) {
    if (write.pImageInfo == nullptr) return false;
    switch (write.descriptorType) {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return true;
    default: return false;
    }
}

void mixImageWrite(std::uint64_t& hash, const VkWriteDescriptorSet& write) {
    const auto mix = [&](std::uint64_t value) { hash = (hash ^ value) * 0x100000001b3ull; };
    mix(write.dstBinding);
    mix(write.dstArrayElement);
    mix(write.descriptorCount);
    mix(static_cast<std::uint64_t>(write.descriptorType));
    for (std::uint32_t element = 0; element < write.descriptorCount; ++element) {
        const auto& info = write.pImageInfo[element];
        mix(reinterpret_cast<std::uint64_t>(info.sampler));
        mix(reinterpret_cast<std::uint64_t>(info.imageView));
        mix(static_cast<std::uint64_t>(info.imageLayout));
    }
}

void noteParkedDifference(const char* what, std::size_t mine, std::size_t today) {
    static std::atomic<std::uint32_t> printed{0};
    if (printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[fragment-parked-sets-verify] %s differs from today's: %zu writes on the parked route, %zu today; parking is off for the template\n", what, mine, today);
}

}

ShaderResources::ParkedSetsNote ShaderResources::TakeParkedSetsNote() {
    auto& notes = HostThreadLocal<ParkedSetsNotes, ParkedSetsNotesStorage>();
    return std::exchange(notes.note, {});
}

ShaderResources::ParkedSetsCounts ShaderResources::TakeParkedSetsCounts() {
    ParkedSetsCounts counts;
    counts.parked = fragmentParkedSets.load(std::memory_order_relaxed);
    counts.refusedCapacity = fragmentParksRefusedCapacity.exchange(0, std::memory_order_relaxed);
    counts.refusedCap = fragmentParksRefusedCap.exchange(0, std::memory_order_relaxed);
    counts.refusedClosed = fragmentParksRefusedClosed.exchange(0, std::memory_order_relaxed);
    return counts;
}

void ShaderResources::writeParkedFragmentSets(DrawBindings& result, std::span<const std::size_t> selected, std::int32_t pushedSet) const {
    auto& notes = HostThreadLocal<ParkedSetsNotes, ParkedSetsNotesStorage>();
    std::call_once(fragmentSetsOnce, [&] {
        fragmentSets = makeOwnDrawSets(true);
        fragmentSetsReady.store(true, std::memory_order_release);
    });
    auto& store = *fragmentSets;
    const auto& plan = store.plan;
    const auto variableOf = [&](std::size_t set) { return tableBindings[set] != std::numeric_limits<std::size_t>::max() ? std::optional<std::uint32_t>(std::max(1u, static_cast<std::uint32_t>(bindings[tableBindings[set]].imageAllocations.size()))) : std::nullopt; };
    bool planned = plan.sets.size() == descriptorWrites.size() && plan.buffers.size() == descriptorWrites.size() && plan.slots.size() == allocations.size() && plan.bufferCount == descriptorBuffers.size() && variableOf(0) == plan.variable[0] && variableOf(1) == plan.variable[1];
    for (const auto& [at, held] : plan.variableBindings) planned = planned && at < bindings.size() && bindings[at].imageAllocations.size() == held;
    if (!planned) {
        ++notes.note.today;
        writeFreshDrawSets(result, selected, nullptr, pushedSet);
        return;
    }
    const auto every = CommitFragmentsVerifyEvery();
    const auto imageHash = [&](std::size_t set) {
        std::uint64_t hash = 0xcbf29ce484222325ull;
        for (std::size_t index = 0; index < descriptorWrites.size(); ++index) {
            if (static_cast<std::size_t>(plan.sets[index]) == set && imageWrite(descriptorWrites[index])) mixImageWrite(hash, descriptorWrites[index]);
        }
        return hash;
    };
    result.cache = context.descriptorCache;
    std::array<bool, 2> taken{};
    std::array<std::uint64_t, 2> stamps{};
    const auto allocate = [&](std::size_t set, VkDescriptorSetLayout layout, DescriptorCache::SetAllocation& allocation) {
        if (!rebound && store.Take(set, allocation, stamps[set])) {
            if (allocation.variableCount == plan.variable[set]) {
                taken[set] = true;
                return;
            }
            result.cache->Recycle(layout, allocation);
            stamps[set] = 0;
        }
        allocation = result.cache->AllocateRewritten(layout, plan.sizes[set], updateAfterBind, plan.variable[set]);
    };
    if ((plan.described[0] || splitSets) && pushedSet != 0) {
        allocate(0, _layout, result.allocation);
        result.layout = _layout;
        Require(result.allocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    }
    if (!heapStage && (plan.described[1] || splitSets) && pushedSet != 1) {
        allocate(1, fragment.layout, result.fragmentAllocation);
        result.fragmentLayout = fragment.layout;
        Require(result.fragmentAllocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    }
    struct ParkedBuffersStorage {};
    auto& buffers = HostThreadLocal<std::vector<VkDescriptorBufferInfo>, ParkedBuffersStorage>();
    buffers.assign(descriptorBuffers.begin(), descriptorBuffers.end());
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const auto& snapshot = result.snapshots[index];
        for (const auto slot : plan.slots[selected[index]]) buffers[slot] = wholeDwordRange({snapshot.Handle(), snapshot.offset, snapshot.bytes});
    }
    struct ParkedWritesStorage {};
    auto& writes = HostThreadLocal<std::vector<VkWriteDescriptorSet>, ParkedWritesStorage>();
    writes.clear();
    std::array<std::uint32_t, 2> bufferDescriptors{};
    std::array<std::uint32_t, 2> otherDescriptors{};
    for (std::size_t index = 0; index < descriptorWrites.size(); ++index) {
        const auto set = plan.sets[index];
        const auto target = set == 0 ? result.allocation.set : result.fragmentAllocation.set;
        if (pushedSet >= 0 && target == VK_NULL_HANDLE) continue;
        const auto& original = descriptorWrites[index];
        if (taken[set] && imageWrite(original)) continue;
        auto& write = writes.emplace_back(original);
        write.dstSet = target;
        if (plan.buffers[index] != std::numeric_limits<std::size_t>::max()) write.pBufferInfo = buffers.data() + plan.buffers[index];
        (original.pBufferInfo != nullptr ? bufferDescriptors : otherDescriptors)[set] += original.descriptorCount;
    }
    bool parking = true;
    if (every != 0 && (taken[0] || taken[1]) && ++notes.verifyOrdinal % every == 0) {
        ++notes.note.checked;
        struct TodayBuffersStorage {};
        auto& todayBuffers = HostThreadLocal<std::vector<VkDescriptorBufferInfo>, TodayBuffersStorage>();
        todayBuffers.assign(descriptorBuffers.begin(), descriptorBuffers.end());
        std::vector<VkWriteDescriptorSet> todayWrites = descriptorWrites;
        std::vector<std::size_t> snapshotIndices(allocations.size(), selected.size());
        for (std::size_t index = 0; index < selected.size(); ++index) snapshotIndices[selected[index]] = index;
        auto nextWrite = todayWrites.begin();
        for (const auto& binding : bindings) {
            if (binding.heap) continue;
            const auto original = setOf(binding) == 0 ? _set : fragment.set;
            while (nextWrite != todayWrites.end() && nextWrite->dstBinding == binding.layout.binding && nextWrite->dstSet == original) {
                auto& write = *nextWrite++;
                write.dstSet = setOf(binding) == 0 ? result.allocation.set : result.fragmentAllocation.set;
                if (binding.allocations.empty()) continue;
                const auto offset = static_cast<std::size_t>(write.pBufferInfo - descriptorBuffers.data());
                write.pBufferInfo = todayBuffers.data() + offset;
                for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
                    const auto snapshotIndex = snapshotIndices[binding.allocations[element]];
                    if (snapshotIndex == selected.size()) continue;
                    const auto& snapshot = result.snapshots[snapshotIndex];
                    todayBuffers[offset + element] = wholeDwordRange({snapshot.Handle(), snapshot.offset, snapshot.bytes});
                }
            }
        }
        const bool shaped = nextWrite == todayWrites.end();
        const char* what = !shaped ? "the write shape" : nullptr;
        if (pushedSet >= 0) std::erase_if(todayWrites, [](const VkWriteDescriptorSet& write) { return write.dstSet == VK_NULL_HANDLE; });
        std::vector<VkWriteDescriptorSet> kept;
        kept.reserve(todayWrites.size());
        for (const auto& write : todayWrites) {
            const bool parked = (write.dstSet == result.allocation.set && taken[0]) || (write.dstSet == result.fragmentAllocation.set && taken[1]);
            if (!parked || !imageWrite(write)) kept.push_back(write);
        }
        if (what == nullptr) what = plannedWritesDifference(writes, kept);
        for (std::size_t set = 0; what == nullptr && set < taken.size(); ++set) {
            if (taken[set] && stamps[set] != 0 && stamps[set] != imageHash(set)) what = "a parked set's image descriptors";
        }
        if (what != nullptr) {
            ++notes.note.differing;
            noteParkedDifference(what, writes.size(), todayWrites.size());
            store.Drain(true);
            Require(shaped, "draw snapshot writes disagree with their descriptor bindings");
            writes.assign(todayWrites.begin(), todayWrites.end());
            parking = false;
        }
    }
    writeAddressedOverlay(result, selected);
    context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    Require(!rebound, "a fragment's parked sets met a rebound template");
    std::array<bool, 2> allocated{result.allocation.set != VK_NULL_HANDLE, result.fragmentAllocation.set != VK_NULL_HANDLE};
    for (std::size_t set = 0; set < allocated.size(); ++set) {
        if (!allocated[set]) continue;
        if (taken[set]) {
            ++notes.note.taken;
            notes.note.parkedBuffers += bufferDescriptors[set];
            notes.note.parkedOthers += otherDescriptors[set];
            result.parkStamps[set] = stamps[set];
        } else {
            ++notes.note.fresh;
            notes.note.freshBuffers += bufferDescriptors[set];
            notes.note.freshOthers += otherDescriptors[set];
            result.parkStamps[set] = every != 0 ? imageHash(set) : 0u;
        }
    }
    result.parkedSets = fragmentSets;
    result.parkable = {allocated[0] && parking, allocated[1] && parking};
}

void ShaderResources::writeAddressedOverlay(DrawBindings& result, std::span<const std::size_t> selected) const {
    if (!addressedBuffers) return;
    std::vector<VkDescriptorBufferInfo> overrides(allocations.size(), VkDescriptorBufferInfo{VK_NULL_HANDLE, 0, 0});
    bool overridden = false;
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const auto& snapshot = result.snapshots[index];
        overrides[selected[index]] = {snapshot.Handle(), snapshot.offset, snapshot.bytes};
    }
    for (const auto& binding : bindings) {
        if (!binding.addressed) continue;
        for (const auto allocation : binding.allocations) overridden = overridden || overrides[allocation].buffer != VK_NULL_HANDLE;
    }
    if (overridden) {
        result.heapOffsets = addressedOffsets(&overrides);
        result.addressOffsets = true;
        result.addressOverrides = std::move(overrides);
    }
}

ShaderResources::DrawUse::~DrawUse() {
    if (ownRecord != nullptr) ownRecord->store(false, std::memory_order_release);
}

std::shared_ptr<ShaderResources::DrawUse> ShaderResources::BeginDrawUse(Recorder& recorder) const {
    if (!spaceReuse || bda == nullptr || !faultChecks || faultDescriptors.empty()) return {};
    auto use = std::make_shared<DrawUse>();
    if (!ownRecordBusy->exchange(true, std::memory_order_acq_rel)) {
        use->ownRecord = ownRecordBusy;
        return use;
    }
    use->record = std::make_shared<BdaResources>(context, faultReportOnly);
    use->record->NoteUse(recorder.Device() == context.device ? recorder.OpenSerial() : std::numeric_limits<std::uint64_t>::max());
    use->bindings = std::make_shared<DrawBindings>();
    writeDrawSets(*use->bindings, {}, use->record.get());
    recorder.Keep(use->bindings);
    return use;
}

void ShaderResources::CompleteDrawUse(const DrawUse& use) const {
    if (bda != nullptr) bda->CheckUse(use.record.get());
}

void ShaderResources::BindDrawUse(VkCommandBuffer commands, VkPipelineLayout layout, const DrawUse* use) const {
    if (use == nullptr || use->bindings == nullptr) {
        Bind(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout);
        return;
    }
    requireGuestBuffers();
    const auto set = use->bindings->allocation.set;
    const auto fragmentSet = use->bindings->fragmentAllocation.set;
    if (set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, SetBindSite::Draw);
    if (fragmentSet != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet, 1, &fragmentSet, SetBindSite::Draw);
}

namespace {

std::size_t DataOffsetIndex(std::size_t stage, ShaderRecompiler::RuntimeAbi::Binding binding) {
    Require(binding == ShaderRecompiler::RuntimeAbi::Binding::ShaderData || binding == ShaderRecompiler::RuntimeAbi::Binding::FlattenedSrt, "only shader data and the flattened SRT have ring offsets");
    return stage * 2u + (binding == ShaderRecompiler::RuntimeAbi::Binding::FlattenedSrt ? 1u : 0u);
}

}

void ShaderResources::RefreshMovedAddresses(DrawBindings& rewritten) const {
    if (rewritten.addressOffsets) rewritten.heapOffsets = addressedOffsets(&rewritten.addressOverrides, false);
}

std::array<std::uint32_t, 2u * ShaderRecompiler::RuntimeAbi::StageCount> ShaderResources::addressedOffsets(const std::vector<VkDescriptorBufferInfo>* overrides, bool census) const {
    auto offsets = heapOffsets;
    auto& heap = BindlessHeap::For(context);
    for (std::size_t stage = 0; stage < stageDataWords.size(); ++stage) {
        if (stageDataWords[stage].empty()) continue;
        auto words = stageDataWords[stage];
        writeBufferAddresses(words, static_cast<std::uint32_t>(stage), overrides);
        auto span = heap.Allocate(words.size());
        std::memcpy(span.words.data(), words.data(), words.size() * sizeof(std::uint32_t));
        offsets[stageDataIndex[stage]] = span.offset;
    }
    for (std::size_t stage = 0; stage < ringDescriptorStages.size(); ++stage) {
        if (!ringDescriptorStages[stage]) continue;
        RingDescriptors moved;
        offsets[DataOffsetIndex(stage, ShaderRecompiler::RuntimeAbi::Binding::FlattenedSrt)] = ringDescriptors(stage, overrides, moved);
        if (census && ReplayCensusEnabled() && heapReplay != nullptr) {
            auto& last = heapReplay->stages[stage].censusRun;
            if (!last.empty()) {
                const bool resized = last.size() != moved.ranges.size();
                std::size_t changed = 0;
                for (std::size_t entry = 0; !resized && entry < last.size(); ++entry) changed += last[entry].address != moved.ranges[entry].address || last[entry].size != moved.ranges[entry].size ? 1u : 0u;
                NoteReplayRunChange(changed, resized);
            }
            last = moved.ranges;
        }
    }
    return offsets;
}

void ShaderResources::RewriteFragmentFaults(const DrawBindings& placed) const {
    if (bda == nullptr || faultDescriptors.empty()) return;
    struct FaultWritesStorage {};
    auto& writes = HostThreadLocal<std::vector<VkWriteDescriptorSet>, FaultWritesStorage>();
    writes.clear();
    for (const auto& fault : faultDescriptors) {
        auto write = descriptorWrites[fault.write];
        write.dstSet = write.dstSet == _set ? placed.allocation.set : placed.fragmentAllocation.set;
        if (write.dstSet == VK_NULL_HANDLE) continue;
        write.dstArrayElement = fault.element;
        write.descriptorCount = 1;
        write.pBufferInfo = &descriptorBuffers[fault.buffer];
        writes.push_back(write);
    }
    if (!writes.empty()) context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void ShaderResources::prepareFaultSlot() const {
    if (bda == nullptr) return;
    const auto* recorder = Recorder::Active();
    const bool stamped = recorder != nullptr && recorder->Device() == context.device && RuntimeSwitches::Set(RuntimeSwitch::CommitStamps);
    if (stamped && faultStampRecorder == recorder->Id() && faultStampBatch == recorder->OpenSerial()) return;
    bda->NoteUse(recorder != nullptr && recorder->Device() == context.device ? recorder->OpenSerial() : std::numeric_limits<std::uint64_t>::max());
    if (stamped) {
        faultStampRecorder = recorder->Id();
        faultStampBatch = recorder->OpenSerial();
    }
    if (!bda->TakeFaultSlot(context)) return;
    faultGeneration.fetch_add(1, std::memory_order_acq_rel);
    struct Settle {
        std::atomic<std::uint64_t>& generation;
        ~Settle() { generation.fetch_add(1, std::memory_order_acq_rel); }
    } settle{faultGeneration};
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(faultDescriptors.size());
    for (const auto& fault : faultDescriptors) {
        descriptorBuffers[fault.buffer] = bda->Fault();
        auto write = descriptorWrites[fault.write];
        write.dstArrayElement = fault.element;
        write.descriptorCount = 1;
        write.pBufferInfo = &descriptorBuffers[fault.buffer];
        writes.push_back(write);
    }
    if (!writes.empty()) context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void ShaderResources::RequireGuestBuffersAudited(CommitterTrustCheck& trust) const {
    bool found = false;
    const struct FoundFails {
        CommitterTrustCheck& check;
        const bool& seen;
        ~FoundFails() {
            if (seen) check.Fail();
        }
    } fails{trust, found};
    requireGuestBuffers({}, &found);
}

void ShaderResources::requireGuestBuffers(std::span<const MovedBuffer> moved, bool* found) const {
    if (!completed || !RuntimeSwitches::Set(RuntimeSwitch::GuardRangeMemo) || RuntimeSwitches::Set(RuntimeSwitch::NoGuardSpans) || VerifyGuardSpans()) {
        for (std::size_t index = 0; index < allocations.size(); ++index) {
            const auto& item = allocations[index];
            if (!item.guest) continue;
            const auto override = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& entry) { return entry.allocation == index; });
            const auto address = override != moved.end() ? override->address : item.address;
            const auto bytes = override != moved.end() ? override->size : item.size;
            if (found != nullptr && !*found && (!DepthReadGuardIdle_nid_no_patch(address, bytes) || MayOverlapMultisampleGuards(address, bytes))) *found = true;
            RequireNoResidentDepthRead_nid_no_patch(address, bytes, htileClear);
            RequireNoMultisampleColorTarget(address, bytes, "a shader buffer access");
        }
        return;
    }
    const auto depthGeneration = DepthGuardsGeneration_nid_no_patch();
    const auto multisampleGeneration = MultisampleGuardsGeneration();
    const bool depthHeld = depthGuardsClear.load(std::memory_order_relaxed) == depthGeneration;
    const bool multisampleHeld = multisampleGuardsClear.load(std::memory_order_relaxed) == multisampleGeneration;
    if (moved.empty() && depthHeld && multisampleHeld) return;
    bool depthClear = moved.empty();
    bool multisampleClear = moved.empty();
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& item = allocations[index];
        if (!item.guest) continue;
        const auto override = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& entry) { return entry.allocation == index; });
        const bool own = override == moved.end();
        const auto address = own ? item.address : override->address;
        const auto bytes = own ? item.size : override->size;
        if ((!own || !depthHeld) && MayOverlapDepthGuards_nid_no_patch(address, bytes)) {
            depthClear = false;
            if (found != nullptr) *found = true;
            RequireNoResidentDepthRead_nid_no_patch(address, bytes, htileClear);
        }
        if ((!own || !multisampleHeld) && MayOverlapMultisampleGuards(address, bytes)) {
            multisampleClear = false;
            if (found != nullptr) *found = true;
            RequireNoMultisampleColorTarget(address, bytes, "a shader buffer access");
        }
    }
    if (depthClear && !depthHeld) depthGuardsClear.store(depthGeneration, std::memory_order_relaxed);
    if (multisampleClear && !multisampleHeld) multisampleGuardsClear.store(multisampleGeneration, std::memory_order_relaxed);
}

bool ShaderResources::requireMovedGuestBuffers(std::span<const MovedBuffer> moved, const GuardTests* guards, bool* found, bool answered) const {
    if (moved.empty() || !completed || !RuntimeSwitches::Set(RuntimeSwitch::GuardRangeMemo) || RuntimeSwitches::Set(RuntimeSwitch::NoGuardSpans) || VerifyGuardSpans()) return false;
    if (depthGuardsClear.load(std::memory_order_relaxed) != DepthGuardsGeneration_nid_no_patch() || multisampleGuardsClear.load(std::memory_order_relaxed) != MultisampleGuardsGeneration()) return false;
    for (std::size_t index = 0; index < moved.size(); ++index) {
        if (moved[index].allocation >= allocations.size() || (index != 0 && moved[index - 1].allocation >= moved[index].allocation)) return false;
    }
    if (answered && guards == nullptr && found == nullptr) return true;
    for (const auto& entry : moved) {
        if (!allocations[entry.allocation].guest) continue;
        if (guards != nullptr ? guards->multisample(guards->tests, entry.address, entry.size) : MayOverlapMultisampleGuards(entry.address, entry.size)) {
            if (found != nullptr) *found = true;
            RequireNoMultisampleColorTarget(entry.address, entry.size, "a shader buffer access");
        }
    }
    return true;
}

void ShaderResources::BindDescriptorHeaps(VkCommandBuffer commands) const {
    if (!std::exchange(bindingsChecked, false)) {
        requireGuestBuffers();
        prepareFaultSlot();
    }
    BindlessHeap::For(context).BindDescriptorHeaps(commands);
}

std::uint8_t ShaderResources::LeanMiss() const {
    if (!reusable) return 1;
    if (spaceReuse) return 2;
    if (usesBda) return 3;
    if (bda != nullptr && !FaultBufferLean()) return 10;
    if (heapStage) return 4;
    if (guestMemory.HoldsLease()) return 5;
    if (guestMemory.HasCopiedWrites()) return 6;
    if (guestMemory.HasGpuCopies()) return 7;
    if (!guestMemory.Writes().empty()) return 8;
    if (std::ranges::any_of(storageWritten, [](bool written) { return written; })) return 9;
    return 0;
}

bool ShaderResources::FaultBufferLean() const {
    return bda != nullptr && !usesBda && usesFaultBuffer && RuntimeSwitches::Set(RuntimeSwitch::FaultFragments);
}

bool ShaderResources::TakeFaultCheck(std::uint64_t recorder, std::uint64_t batch) const {
    if (bda == nullptr || (faultCheckRecorder == recorder && faultCheckBatch == batch)) return false;
    faultCheckRecorder = recorder;
    faultCheckBatch = batch;
    return true;
}

bool ShaderResources::RecordLean() const {
    return reusable && !spaceReuse && !usesBda && !heapStage && (bda == nullptr || FaultBufferLean()) && !guestMemory.HoldsLease() && !guestMemory.HasCopiedWrites() && !guestMemory.HasGpuCopies() && guestMemory.Writes().empty() && std::ranges::none_of(storageWritten, [](bool written) { return written; });
}

const ShaderResources::PushPlan* ShaderResources::PushLayouts() const {
    if (pushPlanBuilt) return pushPlan.get();
    pushPlanBuilt = true;
    if (!context.pushDescriptors || context.maxPushDescriptors == 0 || context.descriptorCache == nullptr || heapStage || usesBda || updateAfterBind || addressedBuffers || !faultDescriptors.empty()) return nullptr;
    constexpr auto none = std::numeric_limits<std::size_t>::max();
    constexpr VkDescriptorBindingFlags refused = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
    std::array<std::uint32_t, 2> descriptors{};
    std::array<std::uint32_t, 2> movable{};
    std::array<bool, 2> eligible{true, splitSets};
    for (const auto& binding : bindings) {
        if (binding.heap) return nullptr;
        const auto set = setOf(binding);
        if (set > 1u) return nullptr;
        const auto type = binding.layout.descriptorType;
        if ((binding.flags & refused) != 0 || binding.addressed || type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC || type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) eligible[set] = false;
        descriptors[set] += binding.layout.descriptorCount;
        for (const auto index : binding.allocations) {
            if (index < allocations.size() && (DataRole(allocations[index].role) || (allocations[index].guest && !allocations[index].written))) ++movable[set];
        }
    }
    std::optional<std::uint32_t> chosen;
    for (std::uint32_t set = 0; set < 2u; ++set) {
        if (!eligible[set] || descriptors[set] == 0 || descriptors[set] > context.maxPushDescriptors || tableBindings[set] != none || storageTableBindings[set] != none || movable[set] == 0) continue;
        if (!chosen || movable[set] > movable[*chosen]) chosen = set;
    }
    if (!chosen) return nullptr;
    auto plan = std::make_unique<PushPlan>();
    plan->set = *chosen;
    plan->descriptors = descriptors[*chosen];
    std::vector<VkDescriptorSetLayoutBinding> description;
    std::vector<VkDescriptorBindingFlags> flags;
    for (const auto& binding : bindings) {
        if (binding.heap || setOf(binding) != *chosen) continue;
        description.push_back(binding.layout);
        flags.push_back(binding.flags);
    }
    plan->mainKey.assign(layoutKey.begin(), layoutKey.end());
    plan->fragmentKey.assign(fragmentLayoutKey.begin(), fragmentLayoutKey.end());
    auto& key = *chosen == 0 ? plan->mainKey : plan->fragmentKey;
    key.push_back(PushLayoutKeyTag);
    const auto layout = context.descriptorCache->Layout(key, description, false, flags, true);
    plan->layouts = SetLayouts();
    if (*chosen == 0) plan->layouts.main = layout;
    else plan->layouts.fragment = layout;
    plan->layouts.mainKey = plan->mainKey;
    plan->layouts.fragmentKey = plan->fragmentKey;
    plan->layouts.pushSet = static_cast<std::int32_t>(*chosen);
    plan->allocationSets.assign(allocations.size(), 0xffu);
    plan->slots.resize(allocations.size());
    std::size_t next = 0;
    for (const auto& binding : bindings) {
        if (binding.heap) continue;
        const auto set = setOf(binding);
        for (const auto index : binding.allocations) {
            if (index < plan->allocationSets.size()) plan->allocationSets[index] = static_cast<std::uint8_t>(set);
        }
        const auto original = set == 0 ? _set : fragment.set;
        while (next < descriptorWrites.size() && descriptorWrites[next].dstBinding == binding.layout.binding && descriptorWrites[next].dstSet == original) {
            if (set == *chosen) {
                plan->writes.push_back(next);
                if (!binding.allocations.empty() && descriptorWrites[next].pBufferInfo != nullptr) {
                    const auto offset = static_cast<std::size_t>(descriptorWrites[next].pBufferInfo - descriptorBuffers.data());
                    for (std::size_t element = 0; element < binding.allocations.size(); ++element) plan->slots[binding.allocations[element]].push_back(offset + element);
                }
            }
            ++next;
        }
    }
    if (next != descriptorWrites.size()) return nullptr;
    pushPlan = std::move(plan);
    return pushPlan.get();
}

namespace {

constexpr std::uint32_t DynamicLayoutKeyTag = 0x44594e41u;
constexpr std::size_t DynamicSetsPerSet = 8;
std::atomic<std::uint32_t> dynamicLimitOverride{~0u};
std::atomic<std::uint64_t> dynamicPlansRefused{0};

}

struct ShaderResources::DynamicSetEntry {
    DescriptorCache* cache = nullptr;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    DescriptorCache::SetAllocation allocation;
    std::vector<std::weak_ptr<Buffer>> owners;
    std::vector<VkDescriptorBufferInfo> infos;
    DynamicSetEntry() = default;
    DynamicSetEntry(const DynamicSetEntry&) = delete;
    DynamicSetEntry& operator=(const DynamicSetEntry&) = delete;
    ~DynamicSetEntry() {
        if (cache != nullptr) cache->Recycle(layout, allocation);
    }
};

void ShaderResources::SetDynamicLimitForTest(std::optional<std::uint32_t> limit) {
    dynamicLimitOverride.store(limit.value_or(~0u), std::memory_order_relaxed);
}

std::uint64_t ShaderResources::DynamicPlansRefused() {
    return dynamicPlansRefused.load(std::memory_order_relaxed);
}

const ShaderResources::DynamicPlan* ShaderResources::DynamicLayouts() const {
    if (dynamicPlanBuilt) return dynamicPlan.get();
    dynamicPlanBuilt = true;
    const auto missed = [&](DynamicLeave why) -> const DynamicPlan* {
        dynamicPlanMiss = why;
        return nullptr;
    };
    if (context.descriptorCache == nullptr || !completed) return missed(DynamicLeave::Shape);
    if (heapStage) return missed(DynamicLeave::Heap);
    if (usesBda || bda != nullptr) return missed(DynamicLeave::Address);
    if (updateAfterBind) return missed(DynamicLeave::AfterBind);
    if (addressedBuffers) return missed(DynamicLeave::Addressed);
    if (!faultDescriptors.empty()) return missed(DynamicLeave::Faults);
    constexpr auto none = std::numeric_limits<std::size_t>::max();
    auto plan = std::make_unique<DynamicPlan>();
    std::array<std::vector<VkDescriptorSetLayoutBinding>, 2> description;
    std::array<std::vector<VkDescriptorBindingFlags>, 2> flags;
    std::array<std::vector<std::pair<std::uint32_t, std::size_t>>, 2> dynamicBindings;
    std::array<bool, 2> described{};
    plan->writeSets.assign(descriptorWrites.size(), 0xffu);
    plan->writeTypes.resize(descriptorWrites.size());
    for (std::size_t index = 0; index < descriptorWrites.size(); ++index) plan->writeTypes[index] = descriptorWrites[index].descriptorType;
    plan->slotAllocations.assign(descriptorBuffers.size(), none);
    std::size_t next = 0;
    for (const auto& binding : bindings) {
        if (binding.heap) return missed(DynamicLeave::Heap);
        const auto set = setOf(binding);
        if (set > 1u) return missed(DynamicLeave::Shape);
        described[set] = true;
        const auto original = set == 0 ? _set : fragment.set;
        const bool selectable = std::ranges::any_of(binding.allocations, [&](std::size_t index) { return index < allocations.size() && (DataRole(allocations[index].role) || (allocations[index].guest && !allocations[index].written)); });
        const bool dynamic = selectable && binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && binding.flags == 0;
        if (selectable && !dynamic) return missed(binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? DynamicLeave::Flags : binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ? DynamicLeave::Uniform : DynamicLeave::OtherType);
        auto layout = binding.layout;
        if (dynamic) layout.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        description[set].push_back(layout);
        flags[set].push_back(binding.flags);
        std::size_t writes = 0;
        while (next < descriptorWrites.size() && descriptorWrites[next].dstBinding == binding.layout.binding && descriptorWrites[next].dstSet == original) {
            const auto& write = descriptorWrites[next];
            plan->writeSets[next] = static_cast<std::uint8_t>(set);
            if (dynamic) {
                if (write.pBufferInfo == nullptr || write.dstArrayElement != 0 || write.descriptorCount != binding.layout.descriptorCount || binding.allocations.size() != write.descriptorCount) return missed(DynamicLeave::Shape);
                plan->writeTypes[next] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
                const auto offset = static_cast<std::size_t>(write.pBufferInfo - descriptorBuffers.data());
                for (std::size_t element = 0; element < binding.allocations.size(); ++element) plan->slotAllocations[offset + element] = binding.allocations[element];
                dynamicBindings[set].emplace_back(binding.layout.binding, offset);
                plan->storageDynamic += write.descriptorCount;
                plan->setDynamic[set] += write.descriptorCount;
            }
            ++writes;
            ++next;
        }
        if (dynamic && writes != 1) return missed(DynamicLeave::Shape);
    }
    if (next != descriptorWrites.size()) return missed(DynamicLeave::Shape);
    if (plan->storageDynamic == 0) return missed(DynamicLeave::NoStorage);
    constexpr auto capacity = std::tuple_size_v<decltype(DrawBindings::DynamicBound::offsets)>;
    const auto fits = [&](std::size_t count) { return count <= context.limits.maxDescriptorSetStorageBuffersDynamic && count <= capacity; };
    plan->plainFits = fits(plan->storageDynamic);
    if (!plan->plainFits && !fits(plan->setDynamic[0]) && !fits(plan->setDynamic[1])) {
        dynamicPlansRefused.fetch_add(1, std::memory_order_relaxed);
        return missed(DynamicLeave::Limit);
    }
    if (!plan->plainFits) dynamicPlansRefused.fetch_add(1, std::memory_order_relaxed);
    plan->mainKey.assign(layoutKey.begin(), layoutKey.end());
    plan->fragmentKey.assign(fragmentLayoutKey.begin(), fragmentLayoutKey.end());
    for (std::size_t set = 0; set < 2u; ++set) {
        plan->written[set] = (described[set] || splitSets) && (set == 0 || !heapStage);
        std::ranges::sort(dynamicBindings[set]);
        for (const auto& [number, offset] : dynamicBindings[set]) {
            const auto found = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& binding) { return !binding.heap && setOf(binding) == set && binding.layout.binding == number; });
            for (std::uint32_t element = 0; element < found->layout.descriptorCount; ++element) plan->dynamicSlots[set].push_back(offset + element);
        }
        plan->dynamicSet[set] = !plan->dynamicSlots[set].empty();
        auto& key = set == 0 ? plan->mainKey : plan->fragmentKey;
        if (plan->dynamicSet[set]) {
            key.push_back(DynamicLayoutKeyTag);
            for (const auto& entry : dynamicBindings[set]) key.push_back(entry.first);
            plan->layouts[set] = context.descriptorCache->Layout(key, description[set], false, flags[set], false);
        } else {
            plan->layouts[set] = set == 0 ? _layout : fragment.layout;
        }
        auto& sizes = plan->sizes[set];
        for (const auto& layout : description[set]) {
            const auto count = layout.descriptorCount;
            if (count == 0) continue;
            const auto held = std::find_if(sizes.begin(), sizes.end(), [&](const VkDescriptorPoolSize& entry) { return entry.type == layout.descriptorType; });
            if (held != sizes.end()) held->descriptorCount += count;
            else sizes.push_back({layout.descriptorType, count});
        }
        std::ranges::sort(sizes, [](const VkDescriptorPoolSize& a, const VkDescriptorPoolSize& b) { return a.type < b.type; });
        if (tableBindings[set] != std::numeric_limits<std::size_t>::max()) plan->variable[set] = std::max(1u, static_cast<std::uint32_t>(bindings[tableBindings[set]].imageAllocations.size()));
    }
    plan->pipelineLayouts = SetLayouts();
    if (plan->dynamicSet[0]) plan->pipelineLayouts.main = plan->layouts[0];
    if (plan->dynamicSet[1]) plan->pipelineLayouts.fragment = plan->layouts[1];
    plan->pipelineLayouts.mainKey = plan->mainKey;
    plan->pipelineLayouts.fragmentKey = plan->fragmentKey;
    plan->pipelineLayouts.dynamic = true;
    dynamicPlan = std::move(plan);
    return dynamicPlan.get();
}

const PipelineSetLayouts* ShaderResources::DynamicPushLayouts() const {
    const auto* plan = DynamicLayouts();
    if (plan == nullptr) return nullptr;
    auto& built = *dynamicPlan;
    if (built.pushedBuilt) return built.pushedFits ? &built.pushedLayouts : nullptr;
    built.pushedBuilt = true;
    const auto* push = PushLayouts();
    const auto missed = [&](DynamicLeave why) -> const PipelineSetLayouts* {
        pushedPlanMiss = why;
        return nullptr;
    };
    if (push == nullptr || push->set > 1u) return missed(DynamicLeave::Shape);
    const auto other = push->set == 0 ? 1u : 0u;
    if (!plan->dynamicSet[other]) return missed(DynamicLeave::OtherSet);
    if (plan->setDynamic[other] > std::min<std::size_t>(context.limits.maxDescriptorSetStorageBuffersDynamic, std::tuple_size_v<decltype(DrawBindings::DynamicBound::offsets)>)) return missed(DynamicLeave::Limit);
    built.pushedMainKey = other == 0 ? plan->mainKey : push->mainKey;
    built.pushedFragmentKey = other == 1 ? plan->fragmentKey : push->fragmentKey;
    built.pushedLayouts = push->layouts;
    if (other == 0) built.pushedLayouts.main = plan->layouts[0];
    else built.pushedLayouts.fragment = plan->layouts[1];
    built.pushedLayouts.mainKey = built.pushedMainKey;
    built.pushedLayouts.fragmentKey = built.pushedFragmentKey;
    built.pushedLayouts.dynamic = true;
    built.pushedFits = true;
    return &built.pushedLayouts;
}

ShaderResources::DynamicLeave ShaderResources::DynamicLeaveFor(bool pushed) const {
    const auto* plan = DynamicLayouts();
    if (plan == nullptr) return dynamicPlanMiss;
    if (!pushed) return plan->plainFits ? DynamicLeave::None : DynamicLeave::Limit;
    return DynamicPushLayouts() != nullptr ? DynamicLeave::None : pushedPlanMiss;
}

bool ShaderResources::writeDynamicSets(DrawBindings& result, std::span<const std::size_t> selected, BindingRoute& route, std::int32_t pushedSet) const {
    const auto* plan = DynamicLayouts();
    if (plan == nullptr || rebound || selected.size() > result.snapshots.size()) return false;
    if (pushedSet >= 0 && DynamicPushLayouts() == nullptr) return false;
    const auto limit = std::min({context.limits.maxDescriptorSetStorageBuffersDynamic, dynamicLimitOverride.load(std::memory_order_relaxed), static_cast<std::uint32_t>(std::tuple_size_v<decltype(DrawBindings::DynamicBound::offsets)>)});
    const auto counted = pushedSet < 0 ? plan->storageDynamic : plan->setDynamic[pushedSet == 0 ? 1 : 0];
    if (counted > limit || (pushedSet < 0 && !plan->plainFits)) {
        ++route.dynamicRefused;
        return false;
    }
    if (dynamicBuffers.size() != descriptorBuffers.size() || std::memcmp(dynamicBuffers.data(), descriptorBuffers.data(), descriptorBuffers.size() * sizeof(VkDescriptorBufferInfo)) != 0) {
        for (auto& list : dynamicSets) list.clear();
        dynamicBuffers.assign(descriptorBuffers.begin(), descriptorBuffers.end());
    }
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 1);
    struct KeysStorage {};
    auto& keys = HostThreadLocal<std::array<std::vector<std::pair<const std::shared_ptr<Buffer>*, VkDescriptorBufferInfo>>, 2>, KeysStorage>();
    auto bound = std::make_unique<DrawBindings::DynamicBound>();
    std::uint32_t at = 0;
    for (std::size_t set = 0; set < 2u; ++set) {
        auto& key = keys[set];
        key.clear();
        if (!plan->written[set] || static_cast<std::int32_t>(set) == pushedSet) continue;
        for (const auto slot : plan->dynamicSlots[set]) {
            const auto allocation = plan->slotAllocations[slot];
            const auto chosen = std::find(selected.begin(), selected.end(), allocation);
            if (chosen == selected.end()) {
                key.emplace_back(nullptr, descriptorBuffers[slot]);
                bound->offsets[at++] = 0;
                continue;
            }
            const auto& snapshot = result.snapshots[static_cast<std::size_t>(chosen - selected.begin())];
            if (snapshot.buffer == nullptr || snapshot.offset % alignment != 0 || snapshot.offset > std::numeric_limits<std::uint32_t>::max()) {
                ++route.dynamicRefused;
                return false;
            }
            key.emplace_back(&snapshot.buffer, wholeDwordRange({snapshot.buffer->Handle(), 0, snapshot.bytes}));
            bound->offsets[at++] = static_cast<std::uint32_t>(snapshot.offset);
        }
    }
    const auto sameInfo = [](const VkDescriptorBufferInfo& a, const VkDescriptorBufferInfo& b) { return a.buffer == b.buffer && a.offset == b.offset && a.range == b.range; };
    const std::weak_ptr<Buffer> unowned;
    for (std::size_t set = 0; set < 2u; ++set) {
        if (!plan->written[set] || static_cast<std::int32_t>(set) == pushedSet) continue;
        const auto& key = keys[set];
        auto& list = dynamicSets[set];
        std::shared_ptr<DynamicSetEntry> entry;
        for (auto candidate = list.rbegin(); candidate != list.rend() && entry == nullptr; ++candidate) {
            const auto& held = **candidate;
            if (held.infos.size() != key.size()) continue;
            bool same = true;
            for (std::size_t index = 0; same && index < key.size(); ++index) {
                const auto* owner = key[index].first;
                const auto& mine = held.owners[index];
                same = sameInfo(held.infos[index], key[index].second) && (owner == nullptr ? !mine.owner_before(unowned) && !unowned.owner_before(mine) : !mine.owner_before(*owner) && !owner->owner_before(mine));
            }
            if (same) entry = *candidate;
        }
        if (entry != nullptr) ++route.dynamicHits;
        if (entry == nullptr) {
            auto made = std::make_shared<DynamicSetEntry>();
            made->layout = plan->layouts[set];
            made->allocation = context.descriptorCache->AllocateRewritten(plan->layouts[set], plan->sizes[set], false, plan->variable[set]);
            Require(made->allocation.set != VK_NULL_HANDLE, "dynamic draw set allocation failed");
            made->cache = context.descriptorCache;
            made->owners.reserve(key.size());
            made->infos.reserve(key.size());
            for (const auto& [owner, info] : key) {
                made->owners.push_back(owner != nullptr ? std::weak_ptr<Buffer>(*owner) : std::weak_ptr<Buffer>{});
                made->infos.push_back(info);
            }
            struct BuffersStorage {};
            auto& buffers = HostThreadLocal<std::vector<VkDescriptorBufferInfo>, BuffersStorage>();
            buffers = descriptorBuffers;
            for (std::size_t index = 0; index < key.size(); ++index) buffers[plan->dynamicSlots[set][index]] = key[index].second;
            struct WritesStorage {};
            auto& writes = HostThreadLocal<std::vector<VkWriteDescriptorSet>, WritesStorage>();
            writes.clear();
            for (std::size_t index = 0; index < descriptorWrites.size(); ++index) {
                if (plan->writeSets[index] != set) continue;
                auto& write = writes.emplace_back(descriptorWrites[index]);
                write.dstSet = made->allocation.set;
                write.descriptorType = plan->writeTypes[index];
                if (write.pBufferInfo != nullptr) write.pBufferInfo = buffers.data() + (descriptorWrites[index].pBufferInfo - descriptorBuffers.data());
            }
            context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
            if (list.size() >= DynamicSetsPerSet) list.erase(list.begin());
            list.push_back(made);
            entry = std::move(made);
            ++route.dynamicWritten;
        }
        if (set == 0) {
            result.allocation.set = entry->allocation.set;
            result.layout = entry->layout;
        } else {
            result.fragmentAllocation.set = entry->allocation.set;
            result.fragmentLayout = entry->layout;
        }
        result.lent[set] = true;
        bound->sets[set] = entry;
        bound->counts[set] = static_cast<std::uint32_t>(key.size());
    }
    result.cache = context.descriptorCache;
    result.owned = {};
    result.dynamic = std::move(bound);
    ++route.dynamicBound;
    return true;
}


void ShaderResources::DescribeDynamicBindings(const DrawBindings& bound, std::span<const std::size_t> selected, const PushPlan* push, std::vector<BoundBuffer>& out) const {
    out.clear();
    const auto* plan = dynamicPlan.get();
    if (plan == nullptr) return;
    constexpr auto none = std::numeric_limits<std::size_t>::max();
    std::vector<VkDescriptorBufferInfo> buffers = dynamicBuffers.size() == descriptorBuffers.size() ? dynamicBuffers : descriptorBuffers;
    std::vector<std::size_t> owners(buffers.size(), none);
    std::vector<std::uint64_t> contents(buffers.size(), 0);
    const bool pushing = bound.pushedSet >= 0 && push != nullptr;
    std::vector<std::size_t> pushedOwners(buffers.size(), none);
    if (pushing) overlayPushedBuffers(bound, *push, buffers, pushedOwners);
    for (std::size_t slot = 0; slot < pushedOwners.size(); ++slot) {
        if (pushedOwners[slot] == none) continue;
        owners[slot] = pushedOwners[slot];
        std::uint64_t hash = 0xcbf29ce484222325ull;
        for (const auto byte : bound.snapshots[pushedOwners[slot]].Bytes()) hash = (hash ^ std::to_integer<std::uint8_t>(byte)) * 0x100000001b3ull;
        contents[slot] = hash;
    }
    std::uint32_t at = 0;
    for (std::size_t set = 0; set < 2u; ++set) {
        const auto* entry = bound.dynamic != nullptr ? static_cast<const DynamicSetEntry*>(bound.dynamic->sets[set].get()) : nullptr;
        if (entry == nullptr) continue;
        const auto& slots = plan->dynamicSlots[set];
        for (std::size_t index = 0; index < slots.size() && index < entry->infos.size(); ++index) {
            auto info = entry->infos[index];
            info.offset += bound.dynamic->offsets[at + index];
            const auto slot = slots[index];
            buffers[slot] = info;
            const auto chosen = std::find(selected.begin(), selected.end(), plan->slotAllocations[slot]);
            if (chosen == selected.end()) continue;
            const auto snapshotIndex = static_cast<std::size_t>(chosen - selected.begin());
            owners[slot] = snapshotIndex;
            const auto& snapshot = bound.snapshots[snapshotIndex];
            const auto owner = entry->owners[index].lock();
            if (owner == nullptr || owner != snapshot.buffer || info.offset + snapshot.bytes > owner->Bytes().size()) continue;
            std::uint64_t hash = 0xcbf29ce484222325ull;
            for (const auto byte : owner->Bytes().subspan(static_cast<std::size_t>(info.offset), snapshot.bytes)) hash = (hash ^ std::to_integer<std::uint8_t>(byte)) * 0x100000001b3ull;
            contents[slot] = hash;
        }
        at += bound.DynamicCount(set);
    }
    const auto emit = [&](std::uint32_t set, const VkWriteDescriptorSet& write) {
        if (write.pBufferInfo == nullptr) return;
        const auto offset = static_cast<std::size_t>(write.pBufferInfo - descriptorBuffers.data());
        for (std::uint32_t element = 0; element < write.descriptorCount; ++element) {
            BoundBuffer entry{set, write.dstBinding, write.dstArrayElement + element, buffers[offset + element]};
            if (const auto owner = owners[offset + element]; owner != none) {
                const auto& snapshot = bound.snapshots[owner];
                entry.snapshot = true;
                entry.address = snapshot.address;
                entry.bytes = snapshot.bytes;
                entry.content = contents[offset + element];
            }
            out.push_back(entry);
        }
    };
    for (std::size_t index = 0; index < descriptorWrites.size(); ++index) {
        if (plan->writeSets[index] == 0xffu || (pushing && static_cast<std::int32_t>(plan->writeSets[index]) == bound.pushedSet)) continue;
        emit(plan->writeSets[index], descriptorWrites[index]);
    }
    if (pushing) {
        for (const auto index : push->writes) emit(push->set, descriptorWrites[index]);
    }
    std::ranges::sort(out, [](const BoundBuffer& a, const BoundBuffer& b) { return std::tie(a.set, a.binding, a.element) < std::tie(b.set, b.binding, b.element); });
}


const std::shared_ptr<const ShaderResources::BufferPlan>& ShaderResources::DirectBufferPlan(std::span<const CompiledShader> shaders) const {
    if (bufferPlanBuilt) return bufferPlan;
    bufferPlanBuilt = true;
    if ((_set == VK_NULL_HANDLE && fragment.set == VK_NULL_HANDLE) || heapStage || usesBda || !reusable) return bufferPlan;
    auto plan = std::make_shared<BufferPlan>();
    plan->stages = static_cast<std::uint32_t>(shaders.size());
    for (std::size_t stage = 0; stage < shaders.size(); ++stage) {
        const auto& shader = shaders[stage];
        if (shader.program == nullptr || HeapShader(shader)) return bufferPlan;
        const auto& programBindings = shader.program->bindings;
        for (std::size_t index = 0; index < programBindings.size(); ++index) {
            const auto& binding = programBindings[index];
            const bool data = DataRole(binding.role);
            if (!data && binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
            const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
            if (kept == bindings.end()) return bufferPlan;
            BufferPlan::Element entry;
            entry.stage = static_cast<std::uint32_t>(stage);
            entry.binding = static_cast<std::uint32_t>(index);
            entry.slot = binding.binding;
            entry.count = binding.count;
            entry.role = binding.role;
            if (data) {
                if (kept->allocations.size() != 1) return bufferPlan;
                entry.allocation = kept->allocations.front();
                const auto& item = allocations[entry.allocation];
                if (item.guest || item.buffer == nullptr) return bufferPlan;
                if (!item.dataWords.empty()) {
                    entry.dataWords = item.dataWords;
                } else {
                    const bool patched = std::any_of(dataPatches.begin(), dataPatches.end(), [&](const DataPatch& patch) { return patch.allocation == entry.allocation; });
                    const auto bytes = item.buffer->Bytes();
                    if (patched || item.size % sizeof(std::uint32_t) != 0 || bytes.size() < item.size) return bufferPlan;
                    entry.dataWords.resize(item.size / sizeof(std::uint32_t));
                    std::memcpy(entry.dataWords.data(), bytes.data(), item.size);
                }
                plan->elements.push_back(std::move(entry));
                continue;
            }
            if (kept->allocations.size() != binding.count) return bufferPlan;
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                entry.element = element;
                entry.written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                entry.allocation = kept->allocations[element];
                plan->elements.push_back(entry);
            }
        }
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;
    guestMemory.InPlaceReads(reads);
    plan->allocations.reserve(allocations.size());
    for (const auto& item : allocations) {
        const bool inPlace = std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return item.address >= range.first && item.address < range.second && item.size <= range.second - item.address; });
        plan->allocations.push_back({item.address, item.size, item.adjustment, item.guest, item.written, inPlace});
    }
    plan->patches.reserve(dataPatches.size());
    for (const auto& patch : dataPatches) plan->patches.push_back({patch.allocation, patch.byte, patch.adjustment});
    plan->maxStorageBufferRange = context.limits.maxStorageBufferRange;
    plan->maxDrawSnapshotBytes = maxDrawSnapshotBytes;
    plan->uploadLimit = DrawUploadLimit();
    plan->movedUploadBytes = MovedUploadBytes;
    plan->alignment = static_cast<std::size_t>(std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 16));
    bufferPlan = std::move(plan);
    return bufferPlan;
}

bool ShaderResources::TakePreparedInPlace(PreparedBuffers& prepared, std::size_t index) {
    if (index >= prepared.selected.size()) return false;
    auto& item = prepared.selected[index];
    if (!item.moved || item.begin == 0 || item.handle == VK_NULL_HANDLE) return false;
    const auto entry = std::find_if(prepared.moved.begin(), prepared.moved.end(), [&](const MovedBuffer& candidate) { return candidate.allocation == item.allocation; });
    if (entry == prepared.moved.end() || !entry->words.empty()) return false;
    item.inPlace = true;
    entry->inPlace = item.handle;
    entry->importOffset = item.importOffset;
    entry->inPlaceOffset = item.importOffset;
    entry->adjustment = static_cast<std::uint32_t>(entry->address - item.begin);
    entry->begin = item.begin;
    entry->bytes = item.bytes;
    entry->behind = false;
    prepared.inPlace = true;
    return true;
}

bool ShaderResources::FindPreparedImport(const Context& context, const BufferPlan& plan, PreparedBuffers& prepared, std::size_t index) {
    if (index >= prepared.selected.size()) return false;
    auto& item = prepared.selected[index];
    if (item.handle != VK_NULL_HANDLE) return true;
    if (!item.moved || item.begin == 0 || item.bytes == 0 || item.bytes > plan.maxStorageBufferRange) return false;
    if (HostImportSetGeneration() != prepared.imports || GuestAllocations::GuestAllocationsGeneration_nid_postfix() != prepared.mapping) return false;
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{item.begin, item.begin + item.bytes}}};
    std::array<ExistingHostImport, 1> found{};
    if (!ExistingHostImportsFor(context, range, found) || found[0].offset % std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 4) != 0) return false;
    if (HostImportSetGeneration() != prepared.imports || GuestAllocations::GuestAllocationsGeneration_nid_postfix() != prepared.mapping) return false;
    item.handle = found[0].buffer;
    item.importOffset = found[0].offset;
    return true;
}

bool ShaderResources::PrepareDirectBuffers(const BufferPlan& plan, std::span<const CompiledShader> shaders, const ColorTarget* target, PreparedBuffers& out, const Context* inPlace) {
    using Refusal = PreparedBuffers::Refusal;
    out.moved.clear();
    out.selected.clear();
    out.blob.clear();
    if (out.blob.capacity() == 0 && !RuntimeSwitches::Set(RuntimeSwitch::BufferBlobAppend)) out.blob.reserve(8192);
    out.refusal = Refusal::None;
    out.inPlace = false;
    out.imports = inPlace != nullptr ? HostImportSetGeneration() : 0;
    out.mapping = inPlace != nullptr ? GuestAllocations::GuestAllocationsGeneration_nid_postfix() : 0;
    out.depthGuards = DepthGuardsGeneration_nid_no_patch();
    out.forget = GuestMemory::ForgetSerial();
    const bool lazyImports = inPlace != nullptr && RuntimeSwitches::Set(RuntimeSwitch::LazyInPlaceImports);
    const bool capacity = RuntimeSwitches::Set(RuntimeSwitch::PreparedCapacity);
    out.shortcuts = static_cast<std::uint8_t>((lazyImports ? PreparedBuffers::DeferredImports : 0u) | (capacity ? PreparedBuffers::ExactCapacity : 0u));
    out.lookupsDeferred = 0;
    out.lookupsForced = 0;
    struct FirstMovedStorage {};
    auto* const firstMoved = capacity ? &HostThreadLocal<std::vector<std::size_t>, FirstMovedStorage>() : nullptr;
    if (capacity) {
        out.selected.reserve(plan.allocations.size());
        firstMoved->assign(plan.allocations.size(), std::numeric_limits<std::size_t>::max());
    }
    const auto noteMovedEntry = [&](std::size_t allocation) {
        if (firstMoved != nullptr && allocation < firstMoved->size() && (*firstMoved)[allocation] == std::numeric_limits<std::size_t>::max()) (*firstMoved)[allocation] = out.moved.size() - 1;
    };
    const auto refuse = [&](Refusal why) {
        out.refusal = why;
        return false;
    };
    if (shaders.size() != plan.stages) return refuse(Refusal::Classification);
    const bool clamped = RuntimeSwitches::Set(RuntimeSwitch::MovedInPlace);
    std::size_t next = 0;
    std::size_t movedBytes = 0;
    const auto reserveMoved = [&] {
        if (capacity && out.moved.empty()) out.moved.reserve(plan.elements.size() - next + 1);
    };
    for (std::size_t stage = 0; stage < shaders.size(); ++stage) {
        const auto& shader = shaders[stage];
        if (shader.program == nullptr || HeapShader(shader)) return refuse(Refusal::Classification);
        const auto& programBindings = shader.program->bindings;
        for (std::size_t index = 0; index < programBindings.size(); ++index) {
            const auto& binding = programBindings[index];
            const bool data = DataRole(binding.role);
            if (!data && binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
            const auto planned = [&](std::uint32_t element) {
                if (next >= plan.elements.size()) return false;
                const auto& entry = plan.elements[next];
                return entry.stage == stage && entry.binding == index && entry.slot == binding.binding && entry.count == binding.count && entry.role == binding.role && entry.element == element;
            };
            const auto& guestWords = DrawGuestWords(shader, binding);
            if (data) {
                if (!planned(0)) return refuse(Refusal::Classification);
                const auto& entry = plan.elements[next++];
                const auto size = plan.allocations[entry.allocation].size;
                if (size != guestWords.size() * sizeof(std::uint32_t)) return refuse(Refusal::Classification);
                if (entry.dataWords == guestWords) continue;
                reserveMoved();
                out.moved.push_back({entry.allocation, 0, size, guestWords});
                noteMovedEntry(entry.allocation);
                continue;
            }
            if (guestWords.size() != static_cast<std::size_t>(binding.count) * 4u) return refuse(Refusal::Classification);
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                if (!planned(element) || plan.elements[next].written != written) return refuse(Refusal::Classification);
                const auto& entry = plan.elements[next++];
                const auto* words = guestWords.data() + static_cast<std::size_t>(element) * 4u;
                const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
                const auto address = descriptor.Base48();
                const auto rawSize = BytesUpToTarget(address, descriptor.GetSize(), descriptor.Stride(), written ? nullptr : target);
                const auto& item = plan.allocations[entry.allocation];
                const bool resized = item.guest && item.address == address && item.size != rawSize;
                auto size = rawSize;
                if (clamped && !(item.guest && item.address == address && item.size == size)) size = BytesUpToTarget(address, ClampedBufferBytes(std::span<const std::uint32_t>(words, 4), plan.maxStorageBufferRange), descriptor.Stride(), written ? nullptr : target);
                const bool empty = size == 0 || address == 0;
                if (!item.guest) {
                    if (empty) continue;
                    return refuse(Refusal::NotGuest);
                }
                if (resized) noteMoved(item.size == size ? MovedCount::PreparedResizedClamped : MovedCount::PreparedResized);
                if (item.address == address && item.size == size) continue;
                if (written || empty || item.written) return refuse(Refusal::Written);
                if (size > plan.maxStorageBufferRange) return refuse(Refusal::Oversize);
                const auto begin = address - item.adjustment;
                const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(size, item.adjustment));
                if (bytes > plan.maxDrawSnapshotBytes - movedBytes) return refuse(Refusal::Budget);
                if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return refuse(Refusal::Inaccessible);
                reserveMoved();
                out.moved.push_back({entry.allocation, address, static_cast<std::size_t>(size)});
                noteMovedEntry(entry.allocation);
                movedBytes += bytes;
            }
        }
    }
    if (next != plan.elements.size()) return refuse(Refusal::Classification);
    const bool movedUploads = RuntimeSwitches::Set(RuntimeSwitch::MovedLargeUploads);
    struct Part {
        std::size_t offset;
        const std::byte* source;
        std::size_t bytes;
        std::size_t allocation;
        bool words;
    };
    struct PartsStorage {};
    auto* const parts = RuntimeSwitches::Set(RuntimeSwitch::BufferBlobAppend) ? &HostThreadLocal<std::vector<Part>, PartsStorage>() : nullptr;
    if (parts != nullptr) parts->clear();
    std::size_t total = 0;
    const auto place = [&](std::size_t bytes) {
        const auto offset = ((parts != nullptr ? total : out.blob.size()) + plan.alignment - 1) & ~(plan.alignment - 1);
        if (parts != nullptr) total = offset + bytes;
        else out.blob.resize(offset + bytes);
        return offset;
    };
    const auto patchWords = [&](std::size_t index, std::size_t offset, std::size_t bytes) {
        for (const auto& patch : plan.patches) {
            if (patch.allocation == index && patch.byte < bytes) out.blob[offset + patch.byte] = static_cast<std::byte>(patch.adjustment);
        }
    };
    const auto copy = [&](std::size_t index, std::size_t offset, const void* source, std::size_t bytes, bool words) {
        if (parts != nullptr) {
            parts->push_back({offset, static_cast<const std::byte*>(source), bytes, index, words});
            return;
        }
        std::memcpy(out.blob.data() + offset, source, bytes);
        if (words) patchWords(index, offset, bytes);
    };
    std::size_t totalSnapshotBytes = 0;
    for (std::size_t index = 0; index < plan.allocations.size(); ++index) {
        const auto& item = plan.allocations[index];
        const auto override = firstMoved != nullptr ? ((*firstMoved)[index] == std::numeric_limits<std::size_t>::max() ? out.moved.end() : out.moved.begin() + static_cast<std::ptrdiff_t>((*firstMoved)[index])) : std::find_if(out.moved.begin(), out.moved.end(), [&](const MovedBuffer& entry) { return entry.allocation == index; });
        const bool moved = override != out.moved.end();
        if (moved && !override->words.empty()) {
            const auto offset = place(override->size);
            copy(index, offset, override->words.data(), override->size, true);
            out.selected.push_back({index, 0, override->size, offset, true});
            continue;
        }
        if (!moved && (!item.guest || item.written || !item.inPlace)) continue;
        const auto begin = (moved ? override->address : item.address) - item.adjustment;
        const auto bytes = static_cast<std::size_t>(GuestBufferMemory::ViewBytes(moved ? override->size : item.size, item.adjustment));
        if (!moved && (bytes > plan.maxDrawSnapshotBytes || totalSnapshotBytes + bytes > plan.maxDrawSnapshotBytes)) continue;
        const bool movedUpload = moved && bytes > plan.uploadLimit && bytes <= plan.movedUploadBytes && movedUploads;
        VkBuffer handle = VK_NULL_HANDLE;
        VkDeviceSize importOffset = 0;
        const bool lookup = moved && inPlace != nullptr && bytes != 0 && bytes <= plan.maxStorageBufferRange;
        const bool forced = bytes == 0 || (bytes > plan.uploadLimit && !movedUpload);
        if (lookup && lazyImports && !forced) ++out.lookupsDeferred;
        else if (lookup && lazyImports) ++out.lookupsForced;
        if (lookup && (!lazyImports || forced)) {
            const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{begin, begin + bytes}}};
            std::array<ExistingHostImport, 1> found{};
            if (ExistingHostImportsFor(*inPlace, range, found) && found[0].offset % std::max<VkDeviceSize>(inPlace->limits.minStorageBufferOffsetAlignment, 4) == 0) {
                handle = found[0].buffer;
                importOffset = found[0].offset;
            }
        }
        if (forced) {
            if (handle == VK_NULL_HANDLE) return refuse(Refusal::UploadLimit);
            out.selected.push_back({index, begin, bytes, 0, true, false, handle, importOffset});
            if (!TakePreparedInPlace(out, out.selected.size() - 1)) return refuse(Refusal::UploadLimit);
            continue;
        }
        if (!moved && !GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return refuse(Refusal::Inaccessible);
        const auto offset = place(bytes);
        copy(index, offset, reinterpret_cast<const void*>(begin), bytes, false);
        out.selected.push_back({index, begin, bytes, offset, moved, false, handle, importOffset});
        totalSnapshotBytes += bytes;
    }
    if (parts == nullptr) return true;
    if (out.blob.capacity() < total) out.blob.reserve(total);
    for (const auto& part : *parts) {
        if (out.blob.size() < part.offset) out.blob.resize(part.offset);
        out.blob.insert(out.blob.end(), part.source, part.source + part.bytes);
        if (part.words) patchWords(part.allocation, part.offset, part.bytes);
    }
    parts->clear();
    return true;
}

ShaderResources::PreparedDifference ShaderResources::ComparePreparedMoves(const PreparedBuffers& prepared, std::span<const MovedBuffer> moved) {
    const auto same = [](const MovedBuffer& a, const MovedBuffer& b) { return a.allocation == b.allocation && a.address == b.address && a.size == b.size && std::ranges::equal(a.words, b.words) && (a.inPlace == VK_NULL_HANDLE || b.inPlace == VK_NULL_HANDLE || (a.inPlace == b.inPlace && a.importOffset == b.importOffset && a.begin == b.begin && a.bytes == b.bytes)); };
    std::size_t index = 0;
    while (index < prepared.moved.size() && index < moved.size() && same(prepared.moved[index], moved[index])) ++index;
    if (index == prepared.moved.size() && index == moved.size()) return {};
    const auto describe = [&](std::span<const MovedBuffer> list) { return index < list.size() ? "allocation " + std::to_string(list[index].allocation) + " at 0x" + HexText(list[index].address) + " (" + std::to_string(list[index].size) + " bytes, " + std::to_string(list[index].words.size()) + " words" + (list[index].inPlace != VK_NULL_HANDLE ? ", in place at import offset " + std::to_string(list[index].importOffset) + " for " + std::to_string(list[index].bytes) + " bytes" : std::string()) + ")" : std::string("nothing"); };
    return {"moved buffers", "the finisher moves " + std::to_string(prepared.moved.size()) + " buffers, today's way " + std::to_string(moved.size()) + "; at " + std::to_string(index) + " the finisher has " + describe(prepared.moved) + ", today's way " + describe(moved)};
}

ShaderResources::PreparedDifference ShaderResources::ComparePreparedSelection(const PreparedBuffers& prepared, const DrawBindings* bindings, std::span<const std::size_t> selected) {
    const auto snapshots = bindings != nullptr ? std::span<const DrawBindings::Snapshot>(bindings->snapshots) : std::span<const DrawBindings::Snapshot>{};
    const auto count = std::min(selected.size(), snapshots.size());
    std::size_t index = 0;
    while (index < prepared.selected.size() && index < count && prepared.selected[index].allocation == selected[index] && prepared.selected[index].begin == snapshots[index].address && prepared.selected[index].bytes == snapshots[index].bytes) ++index;
    if (index != prepared.selected.size() || index != count || count != selected.size()) {
        const auto mine = index < prepared.selected.size() ? "allocation " + std::to_string(prepared.selected[index].allocation) + " (" + std::to_string(prepared.selected[index].bytes) + " bytes at 0x" + HexText(prepared.selected[index].begin) + ")" : std::string("nothing");
        const auto today = index < count ? "allocation " + std::to_string(selected[index]) + " (" + std::to_string(snapshots[index].bytes) + " bytes at 0x" + HexText(snapshots[index].address) + ")" : std::string("nothing");
        return {"selection", "the finisher selects " + std::to_string(prepared.selected.size()) + " allocations, today's way " + std::to_string(selected.size()) + " with " + std::to_string(snapshots.size()) + " snapshots; at " + std::to_string(index) + " the finisher has " + mine + ", today's way " + today};
    }
    for (index = 0; index < count; ++index) {
        const auto& item = prepared.selected[index];
        if (snapshots[index].buffer == nullptr || item.inPlace) continue;
        const auto held = snapshots[index].Bytes();
        const auto* blob = prepared.blob.data() + item.offset;
        if (std::memcmp(held.data(), blob, item.bytes) == 0) continue;
        std::size_t at = 0;
        while (at < item.bytes && held[at] == blob[at]) ++at;
        bool written = false;
        const bool unwatched = item.begin != 0 && !GuestMemory::Watched(item.begin, item.bytes);
        if (item.begin != 0 && !unwatched && prepared.generation != 0) {
            GuestMemory::CollectWritesUncached(item.begin, item.bytes);
            written = GuestMemory::NewestStamp(item.begin, item.bytes) > prepared.generation;
        }
        const auto range = item.begin != 0 ? std::to_string(item.bytes) + " bytes at 0x" + HexText(item.begin) : std::to_string(item.bytes) + " data bytes";
        return {"bytes", "allocation " + std::to_string(item.allocation) + ": " + range + " first differ at byte " + std::to_string(at) + (written ? ", written since the finisher read them" : unwatched ? ", in memory the write tracker does not watch" : ""), written, unwatched};
    }
    return {};
}

ShaderResources::PreparedDifference ShaderResources::ComparePreparedTickets(const MoveTicket* prepared, const MoveTicket& today) {
    if (prepared == nullptr) return {"moved buffers", "the committer's tests refuse the prepared moved ranges, which today's way takes"};
    const auto ranges = [](const MoveTicket& ticket) { return ticket.ranges != nullptr ? ticket.ranges->size() : std::size_t{0}; };
    const bool stamps = prepared->valid == today.valid && prepared->recorder == today.recorder && prepared->note == today.note && prepared->pending == today.pending && prepared->allocations == today.allocations && prepared->epoch == today.epoch;
    const bool sameRanges = (prepared->ranges == nullptr) == (today.ranges == nullptr) && (today.ranges == nullptr || *prepared->ranges == *today.ranges);
    if (stamps && sameRanges && prepared->checked == today.checked && prepared->trusted == today.trusted) return {};
    const auto describe = [&](const MoveTicket& ticket) { return std::string(ticket.valid ? "a valid" : "an invalid") + " ticket of " + std::to_string(ranges(ticket)) + " ranges, " + std::to_string(ticket.checked) + " checked, " + std::to_string(ticket.trusted) + " trusted"; };
    return {"ticket", "the prepared moved-range check leaves " + describe(*prepared) + (stamps ? "" : " with other stamps") + (sameRanges ? "" : " with other ranges") + ", today's check " + describe(today)};
}

ShaderResources::PreparedDifference ShaderResources::ComparePreparedBinding(Recorder& recorder, std::span<const MovedBuffer> moved, const PreparedBuffers& prepared, const DrawBindings* bindings, std::span<const std::size_t> selected) const {
    std::vector<std::size_t> taken;
    auto selection = PreparedBuffers::Selection::Hazard;
    const auto bound = selectPreparedSnapshots(recorder, moved, prepared, taken, false, selection);
    if (bound == nullptr) return {};
    const auto snapshots = bindings != nullptr ? std::span<const DrawBindings::Snapshot>(bindings->snapshots) : std::span<const DrawBindings::Snapshot>{};
    const auto alignment = std::max<VkDeviceSize>(context.limits.minStorageBufferOffsetAlignment, 1);
    const bool sized = taken.size() == selected.size() && bound->snapshots.size() == taken.size() && snapshots.size() >= taken.size() && prepared.selected.size() == taken.size();
    const auto same = [&](std::size_t index) {
        const auto& mine = bound->snapshots[index];
        const auto& theirs = snapshots[index];
        return taken[index] == selected[index] && mine.address == theirs.address && mine.bytes == theirs.bytes && mine.offset % alignment == 0;
    };
    std::size_t index = 0;
    while (sized && index < taken.size() && same(index)) ++index;
    if (!sized || index != taken.size()) {
        const auto at = !sized ? std::string() : "; at " + std::to_string(index) + " the committer binds allocation " + std::to_string(taken[index]) + " (" + std::to_string(bound->snapshots[index].bytes) + " bytes at 0x" + HexText(bound->snapshots[index].address) + ", slice offset " + std::to_string(bound->snapshots[index].offset) + "), today's way allocation " + std::to_string(selected[index]) + " (" + std::to_string(snapshots[index].bytes) + " bytes at 0x" + HexText(snapshots[index].address) + ")";
        return {"bound selection", "the committer binds " + std::to_string(taken.size()) + " prepared allocations with " + std::to_string(bound->snapshots.size()) + " snapshots, today's way " + std::to_string(selected.size()) + " with " + std::to_string(snapshots.size()) + at};
    }
    for (index = 0; index < taken.size(); ++index) {
        const auto& item = prepared.selected[index];
        if (bound->snapshots[index].buffer == nullptr) continue;
        const auto held = bound->snapshots[index].Bytes();
        if (held.size() == item.bytes && std::memcmp(held.data(), prepared.blob.data() + item.offset, item.bytes) == 0) continue;
        return {"bound bytes", "allocation " + std::to_string(item.allocation) + ": the committer's slice holds other bytes than the finisher's block at offset " + std::to_string(item.offset)};
    }
    return {};
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::PreparePushedDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved, const PushPlan& plan, std::vector<std::size_t>* selectedOut, const PreparedBuffers* prepared, PreparedBuffers::Selection* preparedSelection, BindingRoute* route) const {
    struct SelectedStorage {};
    auto& selected = HostThreadLocal<std::vector<std::size_t>, SelectedStorage>();
    auto selection = PreparedBuffers::Selection::Hazard;
    auto result = prepared != nullptr ? selectPreparedSnapshots(recorder, moved, *prepared, selected, true, selection, route) : nullptr;
    const bool placed = result != nullptr && route != nullptr && route->taken == BindingTaken::Placed;
    if (preparedSelection != nullptr) *preparedSelection = selection;
    if (result == nullptr) result = selectDrawSnapshots(recorder, moved, false, selected, false);
    if (selectedOut != nullptr) selectedOut->assign(selected.begin(), selected.end());
    if (result == nullptr || (selected.empty() && !rebound)) return {};
    const bool pushes = std::ranges::any_of(selected, [&](std::size_t index) { return index < plan.allocationSets.size() && plan.allocationSets[index] == plan.set; });
    if (!pushes) {
        if (placed) result->pushSelected.clear();
        return finishDrawBindings(recorder, std::move(result), selected, nullptr, -1, route);
    }
    const auto pushed = static_cast<std::int32_t>(plan.set);
    result->pushedSet = pushed;
    if (!placed) result->pushSelected.assign(selected.begin(), selected.end());
    return finishDrawBindings(recorder, std::move(result), selected, nullptr, pushed, route);
}

void ShaderResources::PushDrawBindings(VkCommandBuffer commands, VkPipelineLayout layout, const DrawBindings& result, const PushPlan& plan) const {
    requireGuestBuffers();
    prepareFaultSlot();
    pushBindings(commands, layout, result, plan);
}

void ShaderResources::PushFragmentBindings(VkCommandBuffer commands, VkPipelineLayout layout, const DrawBindings& result, const PushPlan& plan) const {
    if (RuntimeSwitches::Set(RuntimeSwitch::FragmentPlans)) {
        pushPlannedBindings(commands, layout, result, plan);
        return;
    }
    pushBindings(commands, layout, result, plan);
}

void ShaderResources::pushPlannedBindings(VkCommandBuffer commands, VkPipelineLayout layout, const DrawBindings& result, const PushPlan& plan) const {
    const auto fragmentSet = ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet;
    if (plan.set == 1u) {
        const auto set = result.allocation.set != VK_NULL_HANDLE ? result.allocation.set : _set;
        if (set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, SetBindSite::Draw, result.DynamicCount(0), result.DynamicOffsets(0));
    } else if (splitSets) {
        const auto set = result.fragmentAllocation.set != VK_NULL_HANDLE ? result.fragmentAllocation.set : fragment.set;
        if (set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, fragmentSet, 1, &set, SetBindSite::Draw, result.DynamicCount(1), result.DynamicOffsets(1));
    }
    struct PushBuffersStorage {};
    auto& buffers = HostThreadLocal<std::vector<VkDescriptorBufferInfo>, PushBuffersStorage>();
    if (buffers.size() != descriptorBuffers.size()) buffers.resize(descriptorBuffers.size());
    for (const auto index : plan.writes) {
        const auto& write = descriptorWrites[index];
        if (write.pBufferInfo == nullptr) continue;
        const auto offset = static_cast<std::size_t>(write.pBufferInfo - descriptorBuffers.data());
        const auto end = std::min<std::size_t>(offset + write.descriptorCount, descriptorBuffers.size());
        if (offset < end) std::copy(descriptorBuffers.begin() + static_cast<std::ptrdiff_t>(offset), descriptorBuffers.begin() + static_cast<std::ptrdiff_t>(end), buffers.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    overlayPushedBuffers(result, plan, buffers);
    struct PushWritesStorage {};
    auto& writes = HostThreadLocal<std::vector<VkWriteDescriptorSet>, PushWritesStorage>();
    writes.clear();
    for (const auto index : plan.writes) {
        auto& write = writes.emplace_back(descriptorWrites[index]);
        write.dstSet = VK_NULL_HANDLE;
        if (write.pBufferInfo != nullptr) write.pBufferInfo = buffers.data() + (descriptorWrites[index].pBufferInfo - descriptorBuffers.data());
    }
    auto& plans = HostThreadLocal<FragmentSetPlans, FragmentSetPlansStorage>();
    ++plans.note.pushed;
    if (const auto every = CommitFragmentsVerifyEvery(); every != 0 && ++plans.verifyOrdinal % every == 0) {
        ++plans.note.checked;
        std::vector<VkDescriptorBufferInfo> today = descriptorBuffers;
        overlayPushedBuffers(result, plan, today);
        bool same = true;
        for (const auto& write : writes) {
            if (write.pBufferInfo == nullptr) continue;
            const auto offset = static_cast<std::size_t>(write.pBufferInfo - buffers.data());
            for (std::uint32_t element = 0; same && element < write.descriptorCount && offset + element < today.size(); ++element) {
                const auto& a = write.pBufferInfo[element];
                const auto& b = today[offset + element];
                same = a.buffer == b.buffer && a.offset == b.offset && a.range == b.range;
            }
        }
        if (!same) {
            ++plans.note.differing;
            notePlanDifference("a pushed buffer info", writes.size(), writes.size(), static_cast<std::int32_t>(plan.set));
            Require(false, "a planned push differs from today's");
        }
    }
    CmdPushDescriptorSet(context.Resolved(&DeviceFunctions::cmdPushDescriptorSet, "vkCmdPushDescriptorSetKHR"), commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, plan.set == 0 ? 0u : fragmentSet, static_cast<std::uint32_t>(writes.size()), writes.data(), SetBindSite::Draw);
}

void ShaderResources::pushBindings(VkCommandBuffer commands, VkPipelineLayout layout, const DrawBindings& result, const PushPlan& plan) const {
    const auto fragmentSet = ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet;
    if (plan.set == 1u) {
        const auto set = result.allocation.set != VK_NULL_HANDLE ? result.allocation.set : _set;
        if (set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, SetBindSite::Draw, result.DynamicCount(0), result.DynamicOffsets(0));
    } else if (splitSets) {
        const auto set = result.fragmentAllocation.set != VK_NULL_HANDLE ? result.fragmentAllocation.set : fragment.set;
        if (set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, fragmentSet, 1, &set, SetBindSite::Draw, result.DynamicCount(1), result.DynamicOffsets(1));
    }
    struct PushBuffersStorage {};
    auto& buffers = HostThreadLocal<std::vector<VkDescriptorBufferInfo>, PushBuffersStorage>();
    buffers = descriptorBuffers;
    overlayPushedBuffers(result, plan, buffers);
    struct PushWritesStorage {};
    auto& writes = HostThreadLocal<std::vector<VkWriteDescriptorSet>, PushWritesStorage>();
    writes.clear();
    for (const auto index : plan.writes) {
        auto& write = writes.emplace_back(descriptorWrites[index]);
        write.dstSet = VK_NULL_HANDLE;
        if (write.pBufferInfo != nullptr) write.pBufferInfo = buffers.data() + (descriptorWrites[index].pBufferInfo - descriptorBuffers.data());
    }
    CmdPushDescriptorSet(context.Resolved(&DeviceFunctions::cmdPushDescriptorSet, "vkCmdPushDescriptorSetKHR"), commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, plan.set == 0 ? 0u : fragmentSet, static_cast<std::uint32_t>(writes.size()), writes.data(), SetBindSite::Draw);
}

bool ShaderResources::BuildRepeatPush(const PushPlan& plan, std::span<const PreparedBuffers::Selected> selected, RepeatPush& out, bool sets) const {
    out.allocations.clear();
    out.patches.clear();
    out.writes.clear();
    out.sets = nullptr;
    if (bda != nullptr || selected.empty()) return false;
    const std::uint32_t other = plan.set == 0 ? 1u : 0u;
    std::vector<std::size_t> outside;
    std::vector<std::size_t> outsideAt;
    for (std::size_t index = 0; index < selected.size(); ++index) {
        const auto allocation = selected[index].allocation;
        if (allocation >= plan.allocationSets.size() || allocation >= plan.slots.size()) return false;
        out.allocations.push_back(allocation);
        if (plan.allocationSets[allocation] != plan.set) {
            if (!sets || plan.allocationSets[allocation] != other) return false;
            outside.push_back(allocation);
            outsideAt.push_back(index);
            continue;
        }
        for (const auto slot : plan.slots[allocation]) out.patches.emplace_back(slot, index);
    }
    if (!outside.empty()) {
        if (!splitSets || heapStage || updateAfterBind || context.descriptorCache == nullptr) return false;
        RepeatSets* found = nullptr;
        for (const auto& pool : repeatSets) {
            if (pool->set == other && pool->allocations == outside) found = pool.get();
        }
        if (found == nullptr) {
            if (repeatSets.size() >= 8) return false;
            std::vector<VkDescriptorPoolSize> sizes;
            for (const auto& binding : bindings) {
                if (binding.heap || setOf(binding) != other) continue;
                const auto count = (binding.flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) != 0 ? std::max(1u, static_cast<std::uint32_t>(binding.imageAllocations.size())) : binding.layout.descriptorCount;
                if (count == 0) continue;
                const auto held = std::find_if(sizes.begin(), sizes.end(), [&](const VkDescriptorPoolSize& entry) { return entry.type == binding.layout.descriptorType; });
                if (held != sizes.end()) held->descriptorCount += count;
                else sizes.push_back({binding.layout.descriptorType, count});
            }
            std::sort(sizes.begin(), sizes.end(), [](const VkDescriptorPoolSize& a, const VkDescriptorPoolSize& b) { return a.type < b.type; });
            std::optional<std::uint32_t> variable;
            if (tableBindings[other] != std::numeric_limits<std::size_t>::max()) variable = std::max(1u, static_cast<std::uint32_t>(bindings[tableBindings[other]].imageAllocations.size()));
            auto pool = std::make_shared<RepeatSets>(*context.descriptorCache, other == 0 ? _layout : fragment.layout, std::move(sizes), variable);
            pool->set = other;
            pool->allocations = outside;
            pool->buffers = descriptorBuffers;
            std::vector<std::size_t> setWrites;
            std::vector<std::vector<std::size_t>> slotsOf(allocations.size());
            std::size_t next = 0;
            for (const auto& binding : bindings) {
                if (binding.heap) continue;
                const auto set = setOf(binding);
                const auto original = set == 0 ? _set : fragment.set;
                while (next < descriptorWrites.size() && descriptorWrites[next].dstBinding == binding.layout.binding && descriptorWrites[next].dstSet == original) {
                    if (set == other) {
                        setWrites.push_back(next);
                        if (!binding.allocations.empty() && descriptorWrites[next].pBufferInfo != nullptr) {
                            const auto offset = static_cast<std::size_t>(descriptorWrites[next].pBufferInfo - descriptorBuffers.data());
                            for (std::size_t element = 0; element < binding.allocations.size(); ++element) slotsOf[binding.allocations[element]].push_back(offset + element);
                        }
                    }
                    ++next;
                }
            }
            if (next != descriptorWrites.size()) return false;
            for (const auto index : setWrites) {
                auto& write = pool->writes.emplace_back(descriptorWrites[index]);
                if (write.pBufferInfo != nullptr) write.pBufferInfo = pool->buffers.data() + (descriptorWrites[index].pBufferInfo - descriptorBuffers.data());
            }
            for (std::size_t at = 0; at < outside.size(); ++at) {
                if (slotsOf[outside[at]].empty()) return false;
                for (const auto slot : slotsOf[outside[at]]) {
                    const auto write = std::find_if(setWrites.begin(), setWrites.end(), [&](std::size_t index) {
                        const auto& candidate = descriptorWrites[index];
                        if (candidate.pBufferInfo == nullptr) return false;
                        const auto offset = static_cast<std::size_t>(candidate.pBufferInfo - descriptorBuffers.data());
                        return slot >= offset && slot < offset + candidate.descriptorCount;
                    });
                    if (write == setWrites.end() || descriptorWrites[*write].descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) return false;
                    auto patch = descriptorWrites[*write];
                    patch.dstArrayElement += static_cast<std::uint32_t>(slot - static_cast<std::size_t>(patch.pBufferInfo - descriptorBuffers.data()));
                    patch.descriptorCount = 1;
                    patch.pImageInfo = nullptr;
                    patch.pTexelBufferView = nullptr;
                    pool->patches.push_back(patch);
                    pool->patchInfos.push_back(descriptorBuffers[slot]);
                    pool->patchSlots.push_back(slot);
                    pool->patchSelected.push_back(at);
                }
            }
            for (std::size_t at = 0; at < pool->patches.size(); ++at) pool->patches[at].pBufferInfo = &pool->patchInfos[at];
            found = pool.get();
            repeatSets.push_back(std::move(pool));
        }
        out.sets = found;
        out.outsideAt = outsideAt;
    }
    out.buffers = descriptorBuffers;
    out.writes.reserve(plan.writes.size());
    for (const auto index : plan.writes) {
        auto& write = out.writes.emplace_back(descriptorWrites[index]);
        write.dstSet = VK_NULL_HANDLE;
        if (write.pBufferInfo != nullptr) write.pBufferInfo = out.buffers.data() + (descriptorWrites[index].pBufferInfo - descriptorBuffers.data());
    }
    const auto fragmentSet = ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet;
    out.boundSet = VK_NULL_HANDLE;
    out.boundIndex = 0;
    if (plan.set == 1u) {
        out.boundSet = _set;
    } else if (splitSets) {
        out.boundSet = fragment.set;
        out.boundIndex = fragmentSet;
    }
    out.pushIndex = plan.set == 0 ? 0u : fragmentSet;
    return true;
}

void ShaderResources::RepeatPushBindings(VkCommandBuffer commands, VkPipelineLayout layout, RepeatPush& push, std::span<const PreparedBuffers::Selected> selected, VkBuffer buffer, VkDeviceSize base, Recorder& recorder) const {
    requireGuestBuffers();
    prepareFaultSlot();
    auto bound = push.boundSet;
    if (push.sets != nullptr) {
        auto& pool = *push.sets;
        bool fresh = false;
        auto allocation = pool.Take(fresh);
        bound = allocation.set;
        if (push.lease == nullptr || push.leaseRecorder != recorder.Id() || push.leaseBatch != recorder.OpenSerial()) {
            auto lease = std::make_shared<RepeatSetLease>();
            lease->owner = pool.shared_from_this();
            push.lease = lease.get();
            push.leaseRecorder = recorder.Id();
            push.leaseBatch = recorder.OpenSerial();
            recorder.Keep(std::move(lease));
        }
        push.lease->sets.push_back(std::move(allocation));
        const auto update = context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets");
        if (fresh) {
            for (auto& write : pool.writes) write.dstSet = bound;
            update(context.device, static_cast<std::uint32_t>(pool.writes.size()), pool.writes.data(), 0, nullptr);
        }
        for (std::size_t at = 0; at < pool.patches.size(); ++at) {
            const auto& item = selected[push.outsideAt[pool.patchSelected[at]]];
            pool.patchInfos[at] = wholeDwordRange({buffer, base + item.offset, item.bytes});
            pool.patches[at].dstSet = bound;
        }
        update(context.device, static_cast<std::uint32_t>(pool.patches.size()), pool.patches.data(), 0, nullptr);
    }
    if (bound != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, push.boundIndex, 1, &bound, SetBindSite::Draw);
    for (const auto& [slot, index] : push.patches) push.buffers[slot] = wholeDwordRange({buffer, base + selected[index].offset, selected[index].bytes});
    CmdPushDescriptorSet(context.Resolved(&DeviceFunctions::cmdPushDescriptorSet, "vkCmdPushDescriptorSetKHR"), commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, push.pushIndex, static_cast<std::uint32_t>(push.writes.size()), push.writes.data(), SetBindSite::Draw);
}

ShaderResources::PreparedDifference ShaderResources::CompareRepeatPush(RepeatPush& push, const DrawBindings& bound, const PushPlan& plan) const {
    std::vector<VkDescriptorBufferInfo> buffers = descriptorBuffers;
    overlayPushedBuffers(bound, plan, buffers);
    for (std::size_t index = 0; index < push.allocations.size(); ++index) {
        if (index >= bound.snapshots.size() || index >= bound.pushSelected.size() || bound.pushSelected[index] != push.allocations[index]) return {"push writes", "the block's selection of " + std::to_string(push.allocations.size()) + " allocations is not the one today's way pushes (" + std::to_string(bound.pushSelected.size()) + ")"};
    }
    for (const auto& [slot, index] : push.patches) {
        const auto& snapshot = bound.snapshots[index];
        push.buffers[slot] = wholeDwordRange({snapshot.Handle(), snapshot.offset, snapshot.bytes});
    }
    const VkDescriptorSet today = plan.set == 1u ? _set : splitSets ? fragment.set : VK_NULL_HANDLE;
    if (push.sets == nullptr && push.boundSet != today) return {"push writes", "the block binds another set beside the pushed one than today's way"};
    if (push.sets != nullptr) {
        const auto& pool = *push.sets;
        const std::uint32_t other = plan.set == 0 ? 1u : 0u;
        if (pool.set != other || push.boundIndex != (other == 0 ? 0u : static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet))) return {"set writes", "the block writes set " + std::to_string(pool.set) + " at index " + std::to_string(push.boundIndex) + ", today's way the set beside the pushed one"};
        std::vector<VkDescriptorBufferInfo> expected = descriptorBuffers;
        std::vector<std::size_t> setWrites;
        std::size_t next = 0;
        for (const auto& binding : bindings) {
            if (binding.heap) continue;
            const auto set = setOf(binding);
            const auto original = set == 0 ? _set : fragment.set;
            while (next < descriptorWrites.size() && descriptorWrites[next].dstBinding == binding.layout.binding && descriptorWrites[next].dstSet == original) {
                if (set == other) {
                    setWrites.push_back(next);
                    if (!binding.allocations.empty() && descriptorWrites[next].pBufferInfo != nullptr) {
                        const auto offset = static_cast<std::size_t>(descriptorWrites[next].pBufferInfo - descriptorBuffers.data());
                        for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
                            const auto chosen = std::find(bound.pushSelected.begin(), bound.pushSelected.end(), binding.allocations[element]);
                            if (chosen == bound.pushSelected.end()) continue;
                            const auto& snapshot = bound.snapshots[static_cast<std::size_t>(chosen - bound.pushSelected.begin())];
                            expected[offset + element] = wholeDwordRange({snapshot.Handle(), snapshot.offset, snapshot.bytes});
                        }
                    }
                }
                ++next;
            }
        }
        std::vector<VkDescriptorBufferInfo> content = pool.buffers;
        for (std::size_t at = 0; at < pool.patches.size(); ++at) {
            const auto& snapshot = bound.snapshots[push.outsideAt[pool.patchSelected[at]]];
            const auto slot = pool.patchSlots[at];
            content[slot] = wholeDwordRange({snapshot.Handle(), snapshot.offset, snapshot.bytes});
            const auto write = std::find_if(setWrites.begin(), setWrites.end(), [&](std::size_t index) {
                const auto& candidate = descriptorWrites[index];
                if (candidate.pBufferInfo == nullptr) return false;
                const auto offset = static_cast<std::size_t>(candidate.pBufferInfo - descriptorBuffers.data());
                return slot >= offset && slot < offset + candidate.descriptorCount;
            });
            if (write == setWrites.end()) return {"set writes", "the block rewrites a buffer entry outside the set beside the pushed one"};
            const auto& theirs = descriptorWrites[*write];
            const auto element = theirs.dstArrayElement + static_cast<std::uint32_t>(slot - static_cast<std::size_t>(theirs.pBufferInfo - descriptorBuffers.data()));
            const auto& mine = pool.patches[at];
            if (mine.dstBinding != theirs.dstBinding || mine.dstArrayElement != element || mine.descriptorCount != 1 || mine.descriptorType != theirs.descriptorType) return {"set writes", "the block's rewrite of binding " + std::to_string(theirs.dstBinding) + "[" + std::to_string(element) + "] has another shape than the entry it replaces"};
        }
        if (pool.writes.size() != setWrites.size()) return {"set writes", "the block writes " + std::to_string(pool.writes.size()) + " entries of the set beside the pushed one, today's way " + std::to_string(setWrites.size())};
        for (std::size_t at = 0; at < setWrites.size(); ++at) {
            const auto& mine = pool.writes[at];
            const auto& theirs = descriptorWrites[setWrites[at]];
            if (mine.dstBinding != theirs.dstBinding || mine.dstArrayElement != theirs.dstArrayElement || mine.descriptorCount != theirs.descriptorCount || mine.descriptorType != theirs.descriptorType || mine.pImageInfo != theirs.pImageInfo || mine.pTexelBufferView != theirs.pTexelBufferView || mine.pNext != theirs.pNext || (mine.pBufferInfo == nullptr) != (theirs.pBufferInfo == nullptr)) return {"set writes", "write " + std::to_string(at) + " (binding " + std::to_string(theirs.dstBinding) + ") of the set beside the pushed one has another shape than today's"};
            if (theirs.pBufferInfo == nullptr) continue;
            const auto offset = static_cast<std::size_t>(theirs.pBufferInfo - descriptorBuffers.data());
            for (std::uint32_t element = 0; element < theirs.descriptorCount; ++element) {
                const auto& a = content[offset + element];
                const auto& b = expected[offset + element];
                if (a.buffer != b.buffer || a.offset != b.offset || a.range != b.range) return {"set writes", "binding " + std::to_string(theirs.dstBinding) + "[" + std::to_string(theirs.dstArrayElement + element) + "]: the block's set holds range " + std::to_string(a.range) + " at " + std::to_string(a.offset) + ", today's way range " + std::to_string(b.range) + " at " + std::to_string(b.offset) + (a.buffer == b.buffer ? "" : " of another buffer")};
            }
        }
    }
    if (push.writes.size() != plan.writes.size()) return {"push writes", "the block holds " + std::to_string(push.writes.size()) + " writes, today's way pushes " + std::to_string(plan.writes.size())};
    for (std::size_t at = 0; at < plan.writes.size(); ++at) {
        const auto& mine = push.writes[at];
        const auto& theirs = descriptorWrites[plan.writes[at]];
        if (mine.dstBinding != theirs.dstBinding || mine.dstArrayElement != theirs.dstArrayElement || mine.descriptorCount != theirs.descriptorCount || mine.descriptorType != theirs.descriptorType || mine.pImageInfo != theirs.pImageInfo || mine.pTexelBufferView != theirs.pTexelBufferView || mine.pNext != theirs.pNext || (mine.pBufferInfo == nullptr) != (theirs.pBufferInfo == nullptr)) return {"push writes", "write " + std::to_string(at) + " (binding " + std::to_string(theirs.dstBinding) + ") has another shape than today's"};
        if (theirs.pBufferInfo == nullptr) continue;
        const auto offset = static_cast<std::size_t>(theirs.pBufferInfo - descriptorBuffers.data());
        for (std::uint32_t element = 0; element < theirs.descriptorCount; ++element) {
            const auto& a = mine.pBufferInfo[element];
            const auto& b = buffers[offset + element];
            if (a.buffer != b.buffer || a.offset != b.offset || a.range != b.range) return {"push writes", "binding " + std::to_string(theirs.dstBinding) + "[" + std::to_string(theirs.dstArrayElement + element) + "]: the block pushes range " + std::to_string(a.range) + " at " + std::to_string(a.offset) + ", today's way range " + std::to_string(b.range) + " at " + std::to_string(b.offset) + (a.buffer == b.buffer ? "" : " of another buffer")};
        }
    }
    return {};
}

ShaderResources::PreparedBuffers::Selection ShaderResources::RepeatGuards(const PreparedBuffers& prepared) const {
    using Selection = PreparedBuffers::Selection;
    const auto alignment = context.limits.minStorageBufferOffsetAlignment;
    if (prepared.plan == nullptr || (alignment != 0 && prepared.plan->alignment % alignment != 0)) return Selection::Hazard;
    const bool guardsMoved = DepthGuardsGeneration_nid_no_patch() != prepared.depthGuards;
    if ((guardsMoved || HasResidentDepthGuards_nid_no_patch()) && std::ranges::any_of(prepared.selected, [&](const PreparedBuffers::Selected& item) { return item.begin != 0 && (guardsMoved || MayOverlapDepthGuards_nid_no_patch(item.begin, item.bytes)); })) return Selection::DepthGuards;
    return Selection::Taken;
}

void ShaderResources::RepeatRequire(std::span<const MovedBuffer> moved) const {
    requireGuestBuffers(moved);
    prepareFaultSlot();
    bindingsChecked = heapStage && RuntimeSwitches::Set(RuntimeSwitch::ReplayBind);
}

bool ShaderResources::RepeatUnmovedClear(const Recorder& recorder, const PreparedBuffers& prepared) const {
    for (const auto& item : prepared.selected) {
        if (item.moved || RuntimeSwitches::Set(RuntimeSwitch::XpCommitFloor) || RuntimeSwitches::Set(RuntimeSwitch::XpExpressFloor)) continue;
        const auto& allocation = allocations[item.allocation];
        if (guestMemory.WritesOverlap(allocation.address, allocation.size) || recorder.PendingWriteOverlaps(allocation.address, allocation.size)) return false;
    }
    return true;
}

std::shared_ptr<Buffer> ShaderResources::RepeatUpload(Recorder& recorder, const PreparedBuffers& prepared, VkDeviceSize& offset) const {
    auto slice = recorder.DrawUpload(prepared.blob.size());
    std::memcpy(slice.bytes, prepared.blob.data(), prepared.blob.size());
    const auto& plan = *prepared.plan;
    for (const auto& item : prepared.selected) {
        if (item.begin == 0) continue;
        if (item.moved && item.bytes > plan.uploadLimit && MovedSnapshotStats()) {
            movedSnapshotTally().uploaded.fetch_add(1, std::memory_order_relaxed);
            movedSnapshotTally().uploadedBytes.fetch_add(item.bytes, std::memory_order_relaxed);
        }
        CaptureTrace::Log("draw-snapshot batch=%llu address=%llx bytes=%zu", static_cast<unsigned long long>(recorder.Submissions() + 1), static_cast<unsigned long long>(item.begin), item.bytes);
    }
    offset = slice.offset;
    return std::move(slice.buffer);
}

bool ShaderResources::ProofHeld() const {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_PIPELINE") != nullptr;
    if (!EpochProofs() || verify) return false;
    const auto epoch = GuestMemory::CollectEpoch();
    if (epoch == 0 || provedEpoch != epoch || provedPending != StorageTexture::PendingSerial() || provedProof != GuestMemory::ProofSerial() || HoldsDepthCopies()) return false;
    return registeredImageViewsCurrent(registeredImageViews);
}

bool ShaderResources::HoldsDepthCopies() const {
    return std::ranges::any_of(textures, [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->IsDepthCopy(); });
}

void ShaderResources::overlayPushedBuffers(const DrawBindings& result, const PushPlan& plan, std::span<VkDescriptorBufferInfo> buffers, std::span<std::size_t> owners) const {
    for (std::size_t index = 0; index < result.pushSelected.size(); ++index) {
        const auto allocation = result.pushSelected[index];
        if (allocation >= plan.slots.size() || index >= result.snapshots.size()) continue;
        const auto& snapshot = result.snapshots[index];
        for (const auto slot : plan.slots[allocation]) {
            buffers[slot] = wholeDwordRange({snapshot.Handle(), snapshot.offset, snapshot.bytes});
            if (!owners.empty() && snapshot.buffer != nullptr) owners[slot] = index;
        }
    }
}

void ShaderResources::DescribeDrawBindings(const DrawBindings* bound, std::span<const std::size_t> selected, const PushPlan* plan, std::vector<BoundBuffer>& out, bool contents) const {
    out.clear();
    constexpr auto none = std::numeric_limits<std::size_t>::max();
    std::vector<VkDescriptorBufferInfo> buffers = descriptorBuffers;
    std::vector<std::size_t> owners(buffers.size(), none);
    const bool pushing = bound != nullptr && bound->pushedSet >= 0 && plan != nullptr;
    const auto pushedSet = pushing ? bound->pushedSet : -1;
    const auto chosen = pushing ? std::span<const std::size_t>(bound->pushSelected) : selected;
    if (pushing) overlayPushedBuffers(*bound, *plan, buffers, owners);
    const bool rewritten = bound != nullptr && (!pushing || std::ranges::any_of(chosen, [&](std::size_t index) { return index >= plan->allocationSets.size() || plan->allocationSets[index] != static_cast<std::uint8_t>(pushedSet); }));
    std::vector<std::size_t> snapshotOf(allocations.size(), none);
    if (rewritten) {
        for (std::size_t index = 0; index < chosen.size() && index < bound->snapshots.size(); ++index) {
            if (chosen[index] < snapshotOf.size()) snapshotOf[chosen[index]] = index;
        }
    }
    std::vector<std::uint8_t> writeSets(descriptorWrites.size(), 0xffu);
    std::size_t next = 0;
    for (const auto& binding : bindings) {
        if (binding.heap) continue;
        const auto set = setOf(binding);
        const auto original = set == 0 ? _set : fragment.set;
        while (next < descriptorWrites.size() && descriptorWrites[next].dstBinding == binding.layout.binding && descriptorWrites[next].dstSet == original) {
            const auto& write = descriptorWrites[next];
            writeSets[next++] = static_cast<std::uint8_t>(set);
            if (!rewritten || static_cast<std::int32_t>(set) == pushedSet || binding.allocations.empty() || write.pBufferInfo == nullptr) continue;
            const auto offset = static_cast<std::size_t>(write.pBufferInfo - descriptorBuffers.data());
            for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
                const auto snapshot = snapshotOf[binding.allocations[element]];
                if (snapshot == none) continue;
                const auto& held = bound->snapshots[snapshot];
                buffers[offset + element] = wholeDwordRange({held.Handle(), held.offset, held.bytes});
                if (held.buffer != nullptr) owners[offset + element] = snapshot;
            }
        }
    }
    const auto emit = [&](std::uint32_t set, const VkWriteDescriptorSet& write) {
        if (write.pBufferInfo == nullptr) return;
        const auto offset = static_cast<std::size_t>(write.pBufferInfo - descriptorBuffers.data());
        for (std::uint32_t element = 0; element < write.descriptorCount; ++element) {
            BoundBuffer entry{set, write.dstBinding, write.dstArrayElement + element, buffers[offset + element]};
            if (const auto owner = owners[offset + element]; owner != none) {
                const auto& snapshot = bound->snapshots[owner];
                entry.snapshot = true;
                entry.address = snapshot.address;
                entry.bytes = snapshot.bytes;
                if (snapshot.address == 0 || contents) {
                    std::uint64_t hash = 0xcbf29ce484222325ull;
                    for (const auto byte : snapshot.Bytes()) hash = (hash ^ std::to_integer<std::uint8_t>(byte)) * 0x100000001b3ull;
                    entry.content = hash;
                }
            }
            out.push_back(entry);
        }
    };
    for (std::size_t index = 0; index < descriptorWrites.size(); ++index) {
        if (writeSets[index] != 0xffu && static_cast<std::int32_t>(writeSets[index]) != pushedSet) emit(writeSets[index], descriptorWrites[index]);
    }
    if (pushing) {
        for (const auto index : plan->writes) emit(plan->set, descriptorWrites[index]);
    }
    std::ranges::sort(out, [](const BoundBuffer& a, const BoundBuffer& b) { return std::tie(a.set, a.binding, a.element) < std::tie(b.set, b.binding, b.element); });
}

void ShaderResources::DescribeMovedInPlace(std::span<const MovedBuffer> moved, std::vector<BoundBuffer>& out) const {
    out.clear();
    std::size_t next = 0;
    for (const auto& binding : bindings) {
        if (binding.heap) continue;
        const auto set = setOf(binding);
        const auto original = set == 0 ? _set : fragment.set;
        while (next < descriptorWrites.size() && descriptorWrites[next].dstBinding == binding.layout.binding && descriptorWrites[next].dstSet == original) {
            const auto& write = descriptorWrites[next++];
            if (binding.allocations.empty() || write.pBufferInfo == nullptr) continue;
            for (std::size_t element = 0; element < binding.allocations.size() && element < write.descriptorCount; ++element) {
                const auto entry = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& candidate) { return candidate.allocation == binding.allocations[element] && candidate.inPlace != VK_NULL_HANDLE; });
                if (entry == moved.end()) continue;
                BoundBuffer bound{static_cast<std::uint32_t>(set), write.dstBinding, write.dstArrayElement + static_cast<std::uint32_t>(element), wholeDwordRange({entry->inPlace, entry->importOffset, entry->bytes})};
                bound.address = entry->begin;
                bound.bytes = entry->bytes;
                out.push_back(bound);
            }
        }
    }
    std::ranges::sort(out, [](const BoundBuffer& a, const BoundBuffer& b) { return std::tie(a.set, a.binding, a.element) < std::tie(b.set, b.binding, b.element); });
}

void ShaderResources::BindFragmentOwnSets(VkCommandBuffer commands, VkPipelineLayout layout) const {
    if (_set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &_set, SetBindSite::Draw);
    if (fragment.set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet, 1, &fragment.set, SetBindSite::Draw);
}

void ShaderResources::Bind(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout) const {
    requireGuestBuffers();
    prepareFaultSlot();
    if (_set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, bindPoint, layout, 0, 1, &_set, bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE ? SetBindSite::Dispatch : SetBindSite::Draw);
    if (fragment.set != VK_NULL_HANDLE) CmdBindDescriptorSets(context, commands, bindPoint, layout, ShaderRecompiler::RuntimeAbi::FragmentDescriptorSet, 1, &fragment.set, bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE ? SetBindSite::Dispatch : SetBindSite::Draw);
}

PipelineSetLayouts ShaderResources::SetLayouts() const {
    if (!heapStage) return {_layout, fragment.layout, splitSets, layoutKey, fragmentLayoutKey};
    return {_layout, VK_NULL_HANDLE, true, layoutKey, fragmentLayoutKey};
}

void ShaderResources::RefreshHeapStage(std::span<const CompiledShader> shaders) {
    for (const auto& shader : shaders) {
        if (HeapShader(shader)) writeHeapStage(shader);
    }
}

void ShaderResources::writeBufferAddresses(std::span<std::uint32_t> words, std::uint32_t dataStage, const std::vector<VkDescriptorBufferInfo>* overrides,
                                           std::vector<VkDeviceAddressRangeEXT>* ranges) const {
    for (const auto& binding : bindings) {
        if (!binding.addressed || binding.dataStage != dataStage) continue;
        for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
            const auto index = binding.allocations[element];
            const auto& allocation = allocations[index];
            const bool overridden = overrides != nullptr && (*overrides)[index].buffer != VK_NULL_HANDLE;
            const bool owned = !overridden && !allocation.guest && allocation.buffer != nullptr && allocation.role != ShaderRecompiler::DescriptorRole::GuestBuffers && allocation.role != ShaderRecompiler::DescriptorRole::BdaPagetable && allocation.role != ShaderRecompiler::DescriptorRole::FaultBuffer;
            VkDeviceAddress address = 0;
            VkDeviceSize range = 0;
            if (owned) {
                address = allocation.buffer->DeviceAddress();
                range = allocation.size;
            } else {
                auto adjustment = allocation.adjustment;
                auto info = overridden ? (*overrides)[index] : descriptor(allocation, adjustment);
                if (info.buffer == VK_NULL_HANDLE) info = {context.emptyBuffer, 0, EmptyBufferBytes};
                const auto base = BufferDeviceAddress(context, info.buffer);
                Require(base != 0, "an addressed buffer has no device address");
                address = base + info.offset;
                range = info.range;
            }
            if (ranges != nullptr) ranges->push_back({address, std::min<VkDeviceSize>(range, context.limits.maxStorageBufferRange)});
            if (words.empty()) continue;
            const auto at = static_cast<std::size_t>(binding.addressDword) + element * 4u;
            Require(at + 3u < words.size(), "buffer address words lie outside the shader data");
            words[at] = static_cast<std::uint32_t>(address);
            words[at + 1u] = static_cast<std::uint32_t>(address >> 32u);
            words[at + 2u] = static_cast<std::uint32_t>(std::min<VkDeviceSize>(range, 0xffffffffu));
            words[at + 3u] = 0u;
        }
    }
}

VkDeviceAddressRangeEXT ShaderResources::srtRange(std::size_t stage) const {
    const auto bytes = static_cast<VkDeviceSize>(heapSrtDwords[stage]) * sizeof(std::uint32_t);
    if (bytes == 0u || bytes > context.limits.maxStorageBufferRange) {
        const auto empty = BufferDeviceAddress(context, context.emptyBuffer);
        Require(empty != 0, "the empty buffer has no device address");
        return {empty, EmptyBufferBytes};
    }
    return {BindlessHeap::For(context).RingAddress() + heapSrtOffsets[stage], bytes};
}

std::uint32_t ShaderResources::ringDescriptors(std::size_t stage, const std::vector<VkDescriptorBufferInfo>* overrides, RingDescriptors& kept) const {
    struct RangeStorage {};
    auto& ranges = HostThreadLocal<std::vector<VkDeviceAddressRangeEXT>, RangeStorage>();
    ranges.clear();
    ranges.push_back(srtRange(stage));
    writeBufferAddresses({}, static_cast<std::uint32_t>(stage), overrides, &ranges);
    auto& heap = BindlessHeap::For(context);
    const auto equal = [](const VkDeviceAddressRangeEXT& a, const VkDeviceAddressRangeEXT& b) { return a.address == b.address && a.size == b.size; };
    if (!std::ranges::equal(ranges, kept.ranges, equal) || !heap.Held(kept.mark)) {
        kept.first = heap.RingBufferDescriptors(ranges, kept.mark);
        kept.ranges.assign(ranges.begin(), ranges.end());
    }
    return kept.first;
}

std::pair<VkBuffer, VkDeviceSize> ShaderResources::shaderDataLocation(const CompiledShader& shader, const DrawBindings* rewritten) const {
    if (!heapStage || !HeapShader(shader)) return {shaderDataBuffer(shader), 0};
    const auto& offsets = rewritten != nullptr && rewritten->addressOffsets ? rewritten->heapOffsets : heapOffsets;
    const auto& heap = BindlessHeap::For(context);
    return {heap.RingBuffer(), heap.RingOffset() + offsets[DataOffsetIndex(HeapStageIndex(shader.stage), ShaderRecompiler::RuntimeAbi::Binding::ShaderData)]};
}

void ShaderResources::PatchDataAddresses(std::span<std::byte, PipelinePushConstantBytes> bytes, std::span<const CompiledShader> shaders, const DrawBindings* rewritten) const {
    namespace Abi = ShaderRecompiler::RuntimeAbi;
    if (!heapStage) return;
    const auto& offsets = rewritten != nullptr && rewritten->addressOffsets ? rewritten->heapOffsets : heapOffsets;
    VkDeviceAddress ring = 0;
    VkDeviceAddress table = 0;
    VkDeviceAddress fault = 0;
    std::uint32_t tableDwords = 0;
    const auto binds = [&](ShaderRecompiler::DescriptorRole role) {
        return std::any_of(shaders.begin(), shaders.end(), [&](const CompiledShader& shader) { return shader.program != nullptr && std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [&](const auto& binding) { return binding.role == role; }); });
    };
    const bool tableBound = binds(ShaderRecompiler::DescriptorRole::BdaPagetable);
    const bool faultBound = binds(ShaderRecompiler::DescriptorRole::FaultBuffer);
    if (bda != nullptr && (tableBound || faultBound)) {
        const auto of = [&](const VkDescriptorBufferInfo& info) {
            const auto base = BufferDeviceAddress(context, info.buffer);
            Require(base != 0, "a BDA buffer has no device address");
            return base + info.offset;
        };
        if (tableBound) {
            const auto tableInfo = bda->Table();
            table = of(tableInfo);
            tableDwords = static_cast<std::uint32_t>(tableInfo.range / sizeof(std::uint32_t));
        }
        if (faultBound) fault = of(bda->Fault());
    }
    std::array<std::uint32_t, Abi::RootDwords> root{};
    bool rooted = false;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr || !shader.program->dataAddressPush) continue;
        if (ring == 0) ring = BindlessHeap::For(context).RingAddress();
        const auto stage = HeapStageIndex(shader.stage);
        const auto record = [&](Abi::RootStageField field) -> std::uint32_t& { return root[Abi::RootStageDword(static_cast<Abi::Stage>(stage), field)]; };
        record(Abi::RootStageField::ShaderData) = offsets[DataOffsetIndex(stage, Abi::Binding::ShaderData)];
        record(Abi::RootStageField::FlattenedSrt) = offsets[DataOffsetIndex(stage, Abi::Binding::FlattenedSrt)];
        record(Abi::RootStageField::SrtDwords) = heapSrtDwords[stage];
        record(Abi::RootStageField::TableMap) = heapTableMapSet[stage] ? heapTableMaps[stage] : 0u;
        rooted = true;
    }
    if (!rooted) return;
    const auto split = [&](std::uint32_t at, VkDeviceAddress address) {
        root[at] = static_cast<std::uint32_t>(address);
        root[at + 1u] = static_cast<std::uint32_t>(address >> 32u);
    };
    split(Abi::RootRingAddress, ring);
    split(Abi::RootPagetableAddress, table);
    split(Abi::RootFaultAddress, fault);
    root[Abi::RootPagetableDwords] = tableDwords;
    if (gdsAllocation != std::numeric_limits<std::size_t>::max()) {
        const auto* overrides = rewritten != nullptr && rewritten->addressOffsets ? &rewritten->addressOverrides : nullptr;
        auto adjustment = allocations[gdsAllocation].adjustment;
        const auto info = overrides != nullptr && (*overrides)[gdsAllocation].buffer != VK_NULL_HANDLE ? (*overrides)[gdsAllocation] : descriptor(allocations[gdsAllocation], adjustment);
        Require(info.buffer != VK_NULL_HANDLE, "the GDS buffer is missing");
        const auto base = BufferDeviceAddress(context, info.buffer);
        Require(base != 0, "the GDS buffer has no device address");
        split(Abi::RootGdsAddress, base + info.offset);
    }
    std::memcpy(&root[Abi::RootStencilReference], bytes.data() + (Abi::RootDword + Abi::RootStencilReference) * sizeof(std::uint32_t), sizeof(std::uint32_t));
    static_assert((Abi::RootDword + Abi::RootDwords) * sizeof(std::uint32_t) <= PipelinePushConstantBytes);
    std::memcpy(bytes.data() + Abi::RootDword * sizeof(std::uint32_t), root.data(), sizeof(root));
    NoteDataAddressPush();
}

VkImageView ShaderResources::heapPlaceholderView(std::uint32_t kind) const {
    return context.nullDescriptors ? VK_NULL_HANDLE : context.placeholderImages->View(kind);
}

void ShaderResources::writeHeapTableMap(const CompiledShader& shader, std::size_t stageIndex) {
    constexpr auto noElement = std::numeric_limits<std::size_t>::max();
    namespace Abi = ShaderRecompiler::RuntimeAbi;
    namespace TableAbi = ShaderRecompiler::ImageTableAbi;
    heapTableMapSet[stageIndex] = false;
    const auto table = std::find_if(tableStages.begin(), tableStages.end(),
                                    [&](const TableStage& candidate) { return candidate.stage == shader.stage && candidate.heap; });
    if (table == tableStages.end() || table->mapAllocation == noElement) return;
    const ShaderRecompiler::DescriptorBinding* images = nullptr;
    for (const auto& binding : shader.program->bindings) {
        if (binding.role == ShaderRecompiler::DescriptorRole::ImageTable || binding.role == ShaderRecompiler::DescriptorRole::ImageStorageTable)
            images = &binding;
    }
    auto& heap = BindlessHeap::For(context);
    struct MapStorage {};
    auto& map = HostThreadLocal<std::vector<std::uint32_t>, MapStorage>();
    map.assign(table->map.begin(), table->map.end());
    const auto elements = images != nullptr ? static_cast<std::size_t>(images->count) : std::size_t{0};
    TableAbi::Trailer trailer;
    if (shader.program->deviceImageTables) {
        auto* deviceTable = DeviceImageTable::Existing(context);
        auto* recorder = Recorder::Active();
        Require(deviceTable != nullptr, "a stage reading images through the device image table needs the device's table");
        Require(recorder != nullptr, "a stage reading images through the device image table records into the device's recorder");
        const std::array<ShaderRecompiler::PipelineIdentity, 1> programs{shader.program->Identity()};
        deviceTable->Prepare(context, *recorder, programs);
        trailer = deviceTable->Trailer(*recorder);
    }
    const auto base = TableAbi::AppendTrailer(map, trailer);
    const auto sampledAbsent = heap.AbsentSlot(Abi::FirstImageBinding, heapPlaceholderView(Abi::FirstImageBinding));
    const auto storageAbsent = heap.AbsentSlot(Abi::FirstStorageImageBinding, heapPlaceholderView(Abi::FirstStorageImageBinding));
    for (std::size_t element = 0; element < elements; ++element) {
        const auto index = element < table->elementTextures.size() ? table->elementTextures[element] : noElement;
        if (index == noElement) {
            map.push_back(sampledAbsent);
            continue;
        }
        const auto view = textureFirstLayer[index] ? textures[index]->FirstLayerView() : textures[index]->View();
        map.push_back(takenSlot(heap.TextureSlot(Abi::FirstImageBinding, textures[index].get(), view, textures[index]->Layout()), "the global image heap is full"));
    }
    const auto appendStorageSlots = [&](std::uint32_t kind, std::uint32_t absent, const char* full, const auto& wanted, const auto& viewOf) {
        for (std::size_t element = 0; element < elements; ++element) {
            const auto index = element < table->elementStorage.size() ? table->elementStorage[element] : noElement;
            if (!wanted(element) || index == noElement || storageTextures[index] == nullptr) {
                map.push_back(absent);
                continue;
            }
            map.push_back(takenSlot(heap.StorageSlot(kind, storageTextures[index].get(), viewOf(*storageTextures[index], index)), full));
        }
    };
    appendStorageSlots(
        Abi::FirstStorageImageBinding, storageAbsent, "the global storage image heap is full", [](std::size_t) { return true; },
        [&](StorageTexture& image, std::size_t index) { return image.StorageView(storageMips[index], storageFirstLayer[index]); });
    const auto samplerSlotOf = [&](std::size_t index) { return takenSlot(heap.SamplerSlot(samplers[index]), "the global sampler heap is full"); };
    const auto firstSampler = std::find_if(table->tableSamplers.begin(), table->tableSamplers.end(), [&](const auto index) { return index != noElement; });
    const std::uint32_t fallbackSampler = firstSampler != table->tableSamplers.end() ? samplerSlotOf(*firstSampler) : 0u;
    for (const auto index : table->tableSamplers) map.push_back(index == noElement ? fallbackSampler : samplerSlotOf(index));
    const auto atomicElement = [&](std::size_t element, bool wide) {
        if (images == nullptr) return false;
        const auto& flags = wide ? images->imageAtomic64 : images->imageAtomic;
        return element < flags.size() && flags[element];
    };
    const auto anyAtomic = [&](bool wide) {
        for (std::size_t element = 0; element < elements; ++element) {
            if (atomicElement(element, wide)) return true;
        }
        return false;
    };
    const auto samplerSpan = std::max<std::size_t>(elements, map[TableAbi::HeaderSamplerCount]);
    Require(table->tableSamplers.size() <= samplerSpan, "an image table map has more table samplers than its header counts");
    const auto extraBase = base + 2u * elements + samplerSpan;
    if (anyAtomic(false) || anyAtomic(true)) {
        map.resize(extraBase, fallbackSampler);
        for (const bool wide : {false, true}) {
            const auto kind = wide ? Abi::FirstAtomic64ImageBinding : Abi::FirstAtomicImageBinding;
            appendStorageSlots(
                kind, heap.AbsentSlot(kind, heapPlaceholderView(kind)), "the global atomic image heap is full",
                [&](std::size_t element) { return atomicElement(element, wide); },
                [&](StorageTexture& image, std::size_t index) {
                    return wide ? image.Atomic64View(storageMips[index], storageFirstLayer[index])
                                : image.AtomicView(storageMips[index], storageFirstLayer[index]);
                });
        }
    }
    const bool storageTable = std::ranges::any_of(shader.program->bindings, [](const ShaderRecompiler::DescriptorBinding& binding) {
        return binding.role == ShaderRecompiler::DescriptorRole::ImageStorageTable;
    });
    if (storageTable) {
        map.resize(extraBase + 2u * elements, fallbackSampler);
        const auto kind = Abi::FirstUintStorageImageBinding;
        appendStorageSlots(
            kind, heap.AbsentSlot(kind, heapPlaceholderView(kind)), "the global storage image heap is full", [](std::size_t) { return true; },
            [&](StorageTexture& image, std::size_t index) { return image.UintStorageViewOrDefault(storageMips[index], storageFirstLayer[index]); });
    }
    if (map.size() == base) map.push_back(sampledAbsent);
    heapTableMaps[stageIndex] = heapSpan(heapStages[stageIndex].map, map);
    heapTableMapSet[stageIndex] = true;
}

std::uint32_t ShaderResources::heapSpan(HeapStageState::KeptSpan& kept, std::span<const std::uint32_t> words) {
    auto& heap = BindlessHeap::For(context);
    if (std::ranges::equal(kept.words, words) && heap.Held(kept.mark)) return kept.mark.offset;
    auto span = heap.Allocate(words.size(), kept.mark);
    std::memcpy(span.words.data(), words.data(), words.size() * sizeof(std::uint32_t));
    kept.words.assign(words.begin(), words.end());
    return span.offset;
}

bool ShaderResources::ownsHeapBinding(const Binding& binding, VkShaderStageFlags stageBit) const {
    return binding.heap && (binding.layout.stageFlags & stageBit) != 0u;
}

VkImageView ShaderResources::heapSampledView(std::size_t index) const {
    const auto& texture = textures[index];
    if (textureSrgb[index]) return texture->SrgbView(textureFirstLayer[index], textureBaseLevel[index]);
    if (textureBaseLevel[index]) return texture->BaseLevelView();
    if (textureFirstLayer[index]) return texture->FirstLayerView();
    return texture->View();
}

VkImageView ShaderResources::heapMultisampleStorageView(std::size_t index) const {
    return index < multisampleStorage.size() ? multisampleStorage[index].view : VK_NULL_HANDLE;
}

const HeapSlotOwner* ShaderResources::heapMultisampleStorageOwner(std::size_t index) const {
    return index < multisampleStorage.size() ? multisampleStorage[index].owner : nullptr;
}

VkImageView ShaderResources::heapStorageView(std::size_t index, std::uint32_t binding) const {
    const auto& texture = storageTextures[index];
    if (storageAtomic64[index]) return texture->Atomic64View(storageMips[index], storageFirstLayer[index]);
    if (storageAtomic[index]) return texture->AtomicView(storageMips[index], storageFirstLayer[index]);
    return texture->StorageView(storageMips[index], storageFirstLayer[index], UintStorageBinding(binding));
}

void ShaderResources::appendHeapSlotKey(VkShaderStageFlags stageBit, std::vector<std::uintptr_t>& key) const {
    constexpr auto absent = std::numeric_limits<std::size_t>::max();
    key.clear();
    for (const auto& binding : bindings) {
        if (!ownsHeapBinding(binding, stageBit)) continue;
        if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
            for (const auto index : binding.imageAllocations) {
                if (index == absent) {
                    key.push_back(0u);
                    continue;
                }
                key.insert(key.end(), {reinterpret_cast<std::uintptr_t>(textures[index].get()),
                                       reinterpret_cast<std::uintptr_t>(heapSampledView(index)),
                                       static_cast<std::uintptr_t>(textures[index]->Layout())});
            }
        } else if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            for (const auto index : binding.imageAllocations) {
                if (index != absent && storageTextures[index] == nullptr && heapMultisampleStorageView(index) != VK_NULL_HANDLE) {
                    key.insert(key.end(), {reinterpret_cast<std::uintptr_t>(heapMultisampleStorageOwner(index)), reinterpret_cast<std::uintptr_t>(heapMultisampleStorageView(index)),
                                           static_cast<std::uintptr_t>(VK_IMAGE_LAYOUT_GENERAL)});
                    continue;
                }
                if (index == absent || storageTextures[index] == nullptr) {
                    key.push_back(0u);
                    continue;
                }
                key.insert(key.end(), {reinterpret_cast<std::uintptr_t>(storageTextures[index].get()),
                                       reinterpret_cast<std::uintptr_t>(heapStorageView(index, binding.layout.binding)),
                                       static_cast<std::uintptr_t>(VK_IMAGE_LAYOUT_GENERAL)});
            }
        } else if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) {
            for (const auto index : binding.imageAllocations) key.push_back(reinterpret_cast<std::uintptr_t>(samplers[index].get()));
        }
    }
}

void ShaderResources::takeHeapSlots(BindlessHeap& heap, HeapStageState& state, VkShaderStageFlags stageBit,
                                    std::span<const std::uintptr_t> key) {
    namespace Abi = ShaderRecompiler::RuntimeAbi;
    constexpr auto bindingCount = static_cast<std::uint32_t>(Abi::Binding::Count);
    constexpr auto absent = std::numeric_limits<std::size_t>::max();
    for (auto& kindSlots : state.slots) kindSlots.clear();
    state.samplerSlots.clear();
    std::size_t cursor = 0;
    for (const auto& binding : bindings) {
        if (!ownsHeapBinding(binding, stageBit)) continue;
        const auto kind = binding.layout.binding % bindingCount;
        if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
            auto& kindSlots = state.slots[kind];
            for (const auto index : binding.imageAllocations) {
                if (index == absent) {
                    ++cursor;
                    kindSlots.push_back(heap.AbsentSlot(kind, heapPlaceholderView(kind)));
                    continue;
                }
                const auto view = reinterpret_cast<VkImageView>(key[cursor + 1]);
                cursor += 3;
                kindSlots.push_back(takenSlot(heap.TextureSlot(kind, textures[index].get(), view, textures[index]->Layout()), "the global image heap is full"));
            }
        } else if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            auto& kindSlots = state.slots[kind];
            for (const auto index : binding.imageAllocations) {
                if (index != absent && storageTextures[index] == nullptr && heapMultisampleStorageView(index) != VK_NULL_HANDLE) {
                    const auto view = reinterpret_cast<VkImageView>(key[cursor + 1]);
                    cursor += 3;
                    kindSlots.push_back(takenSlot(heap.ViewSlot(kind, heapMultisampleStorageOwner(index), view), "the global storage image heap is full"));
                    continue;
                }
                if (index == absent || storageTextures[index] == nullptr) {
                    ++cursor;
                    kindSlots.push_back(heap.AbsentSlot(kind, heapPlaceholderView(kind)));
                    continue;
                }
                const auto view = reinterpret_cast<VkImageView>(key[cursor + 1]);
                cursor += 3;
                kindSlots.push_back(takenSlot(heap.StorageSlot(kind, storageTextures[index].get(), view), "the global storage image heap is full"));
            }
        } else if (binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) {
            for (const auto index : binding.imageAllocations) {
                ++cursor;
                state.samplerSlots.push_back(takenSlot(heap.SamplerSlot(samplers[index]), "the global sampler heap is full"));
            }
        }
    }
}

void ShaderResources::noteHeapSlotGenerations(const BindlessHeap& heap, HeapStageState& state) {
    state.slotGenerations.clear();
    for (std::uint32_t kind = 0; kind < state.slots.size(); ++kind) {
        for (const auto slot : state.slots[kind]) state.slotGenerations.push_back(heap.SlotGeneration(slot));
    }
    state.samplerGenerations.clear();
    for (const auto slot : state.samplerSlots) state.samplerGenerations.push_back(heap.SamplerGeneration(slot));
}

void ShaderResources::verifyHeapSlots(const BindlessHeap& heap, const HeapStageState& state, VkShaderStageFlags stageBit) const {
    constexpr auto bindingCount = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
    static std::atomic<std::uint64_t> checked{0}, wrong{0}, storageChecked{0}, storageWrong{0};
    for (const auto& binding : bindings) {
        if (!ownsHeapBinding(binding, stageBit)) continue;
        const bool sampled = binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        const bool storage = binding.layout.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (!sampled && !storage) continue;
        const auto kind = binding.layout.binding % bindingCount;
        for (std::size_t element = 0; element < binding.imageAllocations.size(); ++element) {
            const auto index = binding.imageAllocations[element];
            if (index == std::numeric_limits<std::size_t>::max()) continue;
            const bool bound = element < state.slots[kind].size();
            if (sampled) {
                ++checked;
                if (!bound || heap.SlotTexture(kind, state.slots[kind][element]) != textures[index].get()) ++wrong;
                continue;
            }
            if (storageTextures[index] == nullptr) {
                const auto view = heapMultisampleStorageView(index);
                if (view == VK_NULL_HANDLE) continue;
                ++storageChecked;
                if (!bound || !heap.SlotHoldsStorage(kind, state.slots[kind][element], heapMultisampleStorageOwner(index), view)) ++storageWrong;
                continue;
            }
            ++storageChecked;
            if (!bound || !heap.SlotHoldsStorage(kind, state.slots[kind][element], storageTextures[index].get(), heapStorageView(index, binding.layout.binding))) ++storageWrong;
        }
    }
    if ((checked.load() + storageChecked.load()) % 100000u < 8u) {
        std::fprintf(stderr, "[heap-slots] verify: %llu image slots checked, %llu hold another texture; %llu storage slots checked, %llu hold another image or view\n",
                     static_cast<unsigned long long>(checked.load()), static_cast<unsigned long long>(wrong.load()),
                     static_cast<unsigned long long>(storageChecked.load()), static_cast<unsigned long long>(storageWrong.load()));
    }
}

namespace {

[[noreturn]] void ThrowHeapSlotMismatch(const char* format, ...) {
    char text[320];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(text, sizeof(text), format, arguments);
    va_end(arguments);
    throw std::runtime_error(text);
}

}

void ShaderResources::patchHeapSlotWords(const CompiledShader& shader, const HeapStageState& stageState,
                                         std::vector<std::uint32_t>& patched, bool current) const {
    namespace Abi = ShaderRecompiler::RuntimeAbi;
    constexpr auto bindingCount = static_cast<std::uint32_t>(Abi::Binding::Count);
    constexpr auto metadataWords = sizeof(Abi::ResourceMetadata) / sizeof(std::uint32_t);
    const auto& program = *shader.program;
    const auto& slots = stageState.slots;
    const auto& samplerSlots = stageState.samplerSlots;
    for (std::uint32_t image = 0; image < program.runtimeImageCount; ++image) {
        const auto offset = program.imageMetadataDword + image * metadataWords;
        const auto kind = patched[offset + offsetof(Abi::ResourceMetadata, binding) / 4u] % bindingCount;
        const bool storageKind = kind >= Abi::FirstStorageImageBinding && kind < static_cast<std::uint32_t>(Abi::Binding::Samplers);
        if (storageKind && (!Abi::GpuHeapStorageKind(kind) || slots[kind].empty())) continue;
        auto& first = patched[offset + offsetof(Abi::ResourceMetadata, firstElement) / 4u];
        if (kind < Abi::FirstImageBinding) continue;
        if (!(kind < slots.size() && first < slots[kind].size())) {
            ThrowHeapSlotMismatch(
                "AGC graphics: GPU-mode image metadata names an element the stage does not bind (stage %u, image %u of %u, kind %u, first %u, slots of the kind %zu, flags 0x%x, element count %u, bindings %zu, stage state reused %d)",
                static_cast<unsigned>(shader.stage), image, program.runtimeImageCount, kind, first,
                kind < slots.size() ? slots[kind].size() : std::size_t{0}, patched[offset + offsetof(Abi::ResourceMetadata, flags) / 4u],
                patched[offset + offsetof(Abi::ResourceMetadata, elementCount) / 4u], bindings.size(), current ? 1 : 0);
        }
        first = slots[kind][first];
    }
    for (const auto& word : program.heapSlotWords) {
        Require(word.dword < patched.size(), "a GPU-mode slot word lies outside the shader data");
        if (word.kind == static_cast<std::uint32_t>(Abi::Binding::Samplers)) {
            if (word.element >= samplerSlots.size()) {
                ThrowHeapSlotMismatch("AGC graphics: a GPU-mode slot word names a sampler the stage does not bind (stage %u, element %u, sampler slots %zu, bindings %zu)",
                                      static_cast<unsigned>(shader.stage), word.element, samplerSlots.size(), bindings.size());
            }
            patched[word.dword] = samplerSlots[word.element];
            continue;
        }
        if (!(word.kind < slots.size() && word.element < slots[word.kind].size())) {
            ThrowHeapSlotMismatch(
                "AGC graphics: a GPU-mode slot word names an image the stage does not bind (stage %u, kind %u, element %u, slots of the kind %zu, bindings %zu)",
                static_cast<unsigned>(shader.stage), word.kind, word.element,
                word.kind < slots.size() ? slots[word.kind].size() : std::size_t{0}, bindings.size());
        }
        patched[word.dword] = slots[word.kind][word.element];
    }
    const bool slotWords = !program.registrationModule;
    Require(slotWords || program.samplerSlotDword + samplerSlots.size() <= patched.size(), "GPU-mode sampler slots exceed the shader data");
    if (!slotWords) std::copy(samplerSlots.begin(), samplerSlots.end(), patched.begin() + program.samplerSlotDword);
}

namespace {

void noteHeapInputChange(std::span<const std::uint32_t> before, std::span<const std::uint32_t> words, std::span<const std::uint32_t> userData, bool current, bool same) {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!stats) return;
    enum : std::size_t { Same, One, UpToFour, UpToSixteen, More, Reshaped, Count };
    static std::array<std::atomic<std::uint64_t>, Count> counts{};
    static std::atomic<std::int64_t> reportedAt{0};
    std::size_t bucket = Same;
    if (!same) {
        if (!current || before.size() != words.size() + userData.size()) {
            bucket = Reshaped;
        } else {
            std::size_t changed = 0;
            for (std::size_t index = 0; index < words.size(); ++index) changed += before[index] != words[index] ? 1u : 0u;
            for (std::size_t index = 0; index < userData.size(); ++index) changed += before[words.size() + index] != userData[index] ? 1u : 0u;
            bucket = changed <= 1 ? One : changed <= 4 ? UpToFour : changed <= 16 ? UpToSixteen : More;
        }
    }
    counts[bucket].fetch_add(1, std::memory_order_relaxed);
    const auto tenSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (reportedAt.exchange(tenSeconds, std::memory_order_relaxed) == tenSeconds) return;
    std::array<unsigned long long, Count> taken{};
    for (std::size_t index = 0; index < Count; ++index) taken[index] = counts[index].exchange(0, std::memory_order_relaxed);
    std::fprintf(stderr, "[heap-inputs] stage data refreshes (10 s): same %llu; words changed 1: %llu, 2-4: %llu, 5-16: %llu, 17+: %llu; slots or size changed %llu\n", taken[Same], taken[One], taken[UpToFour], taken[UpToSixteen], taken[More], taken[Reshaped]);
}

}

void ShaderResources::writeHeapStage(const CompiledShader& shader) {
    namespace Abi = ShaderRecompiler::RuntimeAbi;
    ++heapStageWrites;
    HeapCostScope slotsCost(HeapCost::Slots);
    auto& heap = BindlessHeap::For(context);
    const VkShaderStageFlags stageBit = VulkanStage(shader.stage);
    const auto stageIndex = HeapStageIndex(shader.stage);
    auto& stageState = heapStages[stageIndex];
    struct HeapKeyStorage {};
    auto& key = HostThreadLocal<std::vector<std::uintptr_t>, HeapKeyStorage>();
    appendHeapSlotKey(stageBit, key);
    bool current = key == stageState.key;
    const auto slotSerial = heap.SlotSerial();
    const bool unmoved = current && EpochProofs() && stageState.slotSerial == slotSerial;
    if (!unmoved && current)
        current =
            heap.GenerationsCurrent(stageState.slots, stageState.slotGenerations, stageState.samplerSlots, stageState.samplerGenerations);
    if (!current) {
        stageState.key = key;
        takeHeapSlots(heap, stageState, stageBit, key);
        noteHeapSlotGenerations(heap, stageState);
    }
    stageState.slotSerial = slotSerial;
    static const bool verifySlots = std::getenv("APS5_VERIFY_GPU_HEAP") != nullptr;
    if (verifySlots) verifyHeapSlots(heap, stageState, stageBit);
    slotsCost.Stop();
    if (ReplayCensusEnabled() && heapReplay != nullptr && !heapReplaying) noteHeapStageCensus(shader, current);

    HeapCostScope dataCost(HeapCost::StageData);
    const std::size_t dataIndex = DataOffsetIndex(stageIndex, Abi::Binding::ShaderData);
    const std::size_t flatIndex = DataOffsetIndex(stageIndex, Abi::Binding::FlattenedSrt);
    const auto& program = *shader.program;
    if (program.dataAddressPush) writeHeapTableMap(shader, stageIndex);
    const bool ringStage = context.heapBufferDescriptors && program.dataAddressPush;
    ringDescriptorStages[stageIndex] = ringStage;
    if (ringStage) stageDataWords[stageIndex].clear();
    bool dataKept = false;
    bool inputsKept = false;
    bool flattened = false;
    for (const auto& binding : program.bindings) {
        const bool shaderData = binding.role == ShaderRecompiler::DescriptorRole::ShaderData;
        if (!shaderData && binding.role != ShaderRecompiler::DescriptorRole::FlattenedSrt) continue;
        const bool heapFlattened = !shaderData && shader.heap != nullptr && !shader.heap->flattenedSrt.empty();
        const auto& words = heapFlattened ? shader.heap->flattenedSrt : DrawGuestWords(shader, binding);
        const auto userData =
            shader.heap != nullptr ? std::span<const std::uint32_t>(shader.heap->userData) : std::span<const std::uint32_t>{};
        const bool sameInput =
            binding.role == ShaderRecompiler::DescriptorRole::ShaderData && current &&
            stageState.dataInput.size() == words.size() + userData.size() &&
            std::equal(words.begin(), words.end(), stageState.dataInput.begin()) &&
            std::equal(userData.begin(), userData.end(), stageState.dataInput.begin() + static_cast<std::ptrdiff_t>(words.size()));
        if (shaderData) noteHeapInputChange(stageState.dataInput, words, userData, current, sameInput);
        if (sameInput && verifySlots && heap.Held(stageState.data.mark)) inputsKept = true;
        if (sameInput && !verifySlots) {
            if (heap.Held(stageState.data.mark)) {
                heapOffsets[dataIndex] = stageState.data.mark.offset;
                dataKept = true;
                continue;
            }
            stageState.data.mark.entry = 0;
        }
        if (!shaderData) {
            (ringStage ? heapSrtOffsets[stageIndex] : heapOffsets[flatIndex]) = heapSpan(stageState.flat, words);
            heapSrtDwords[stageIndex] = static_cast<std::uint32_t>(words.size());
            flattened = true;
            continue;
        }
        ValidateRuntimeResources(shader, words);
        struct PatchedStorage {};
        auto& patched = HostThreadLocal<std::vector<std::uint32_t>, PatchedStorage>();
        patched.assign(words.begin(), words.end());
        if (shader.heap != nullptr)
            std::copy_n(shader.heap->userData.begin(), std::min(shader.heap->userData.size(), patched.size()), patched.begin());
        for (const auto& [byte, adjustment] : stageState.dataPatches) {
            Require(byte < patched.size() * sizeof(std::uint32_t), "a GPU-mode guest buffer offset lies outside the shader data");
            reinterpret_cast<std::byte*>(patched.data())[byte] = static_cast<std::byte>(adjustment);
        }
        patchHeapSlotWords(shader, stageState, patched, current);
        if (addressedBuffers && !ringStage) {
            HeapCostScope bufferCost(HeapCost::BufferAddresses);
            writeBufferAddresses(patched, static_cast<std::uint32_t>(stageIndex), nullptr);
            bufferCost.Stop();
            stageDataWords[stageIndex].assign(patched.begin(), patched.end());
            stageDataIndex[stageIndex] = dataIndex;
        }
        if (verifySlots && sameInput && !std::ranges::equal(stageState.data.words, patched))
            throw std::runtime_error("AGC graphics: a GPU-mode stage's shader data changed although its inputs and slots did not");
        heapOffsets[dataIndex] = heapSpan(stageState.data, patched);
        stageState.dataInput.assign(words.begin(), words.end());
        stageState.dataInput.insert(stageState.dataInput.end(), userData.begin(), userData.end());
    }
    if (!ringStage) return;
    if (!flattened) heapSrtDwords[stageIndex] = 0u;
    auto& kept = stageState.descriptors;
    const auto srt = srtRange(stageIndex);
    const bool keepable = !kept.ranges.empty() && kept.ranges.front().address == srt.address && kept.ranges.front().size == srt.size && heap.Held(kept.mark);
    if (dataKept && keepable) {
        heapOffsets[flatIndex] = kept.first;
        return;
    }
    if (!(inputsKept && keepable)) {
        heapOffsets[flatIndex] = ringDescriptors(stageIndex, nullptr, kept);
        return;
    }
    const std::vector<VkDeviceAddressRangeEXT> before(kept.ranges.begin(), kept.ranges.end());
    const auto first = kept.first;
    heapOffsets[flatIndex] = ringDescriptors(stageIndex, nullptr, kept);
    if (kept.first != first || !std::ranges::equal(before, kept.ranges, [](const auto& a, const auto& b) { return a.address == b.address && a.size == b.size; }))
        throw std::runtime_error("AGC graphics: a GPU-mode stage's kept buffer descriptor run would bind other buffers although its inputs did not change");
}

ReplayReason ShaderResources::HeapReplayRefusal(std::span<const CompiledShader> shaders) const {
    if (usesBda || bda != nullptr) return ReplayReason::AddressBased;
    if (guestMemory.HoldsLease()) return ReplayReason::Lease;
    if (!reusable) return ReplayReason::NotReusable;
    if (spaceReuse) return ReplayReason::SpaceReuse;
    if (guestMemory.HasCopiedWrites()) return ReplayReason::CopiedWrites;
    for (const auto& shader : shaders) {
        if (HeapShader(shader) && !(shader.program->dataAddressPush && context.heapBufferDescriptors)) return ReplayReason::NotRingStage;
    }
    return ReplayReason::Eligible;
}

ShaderResources::HeapReplay& ShaderResources::heapReplayRecord() {
    if (heapReplay == nullptr) {
        heapReplay = std::make_unique<HeapReplay>();
        heapReplay->addressed.assign(allocations.size(), false);
        for (const auto& binding : bindings) {
            if (!binding.addressed) continue;
            for (const auto allocation : binding.allocations) {
                if (allocation < heapReplay->addressed.size()) heapReplay->addressed[allocation] = true;
            }
        }
    }
    return *heapReplay;
}

void ShaderResources::NoteHeapReplayCensus() {
    auto& record = heapReplayRecord();
    const std::array<std::uint64_t, 3> now{GuestAllocations::GuestAllocationsGeneration_nid_postfix(), HostImportSetGeneration(), BindlessHeap::For(context).SlotSerial()};
    if (record.censusGenerations != std::array<std::uint64_t, 3>{}) NoteReplayGenerations(now[0] != record.censusGenerations[0], now[1] != record.censusGenerations[1], now[2] != record.censusGenerations[2]);
    record.censusGenerations = now;
}

void ShaderResources::heapStageMask(const CompiledShader& shader, std::size_t dataDwords, std::vector<std::uint64_t>& mask) const {
    namespace Abi = ShaderRecompiler::RuntimeAbi;
    const auto& program = *shader.program;
    const auto& state = heapStages[HeapStageIndex(shader.stage)];
    mask.assign((dataDwords + 63u) / 64u, 0u);
    const auto mark = [&](std::size_t dword) {
        if (dword < dataDwords) mask[dword / 64u] |= std::uint64_t{1} << (dword % 64u);
    };
    for (const auto& patch : state.dataPatches) mark(patch.first / sizeof(std::uint32_t));
    constexpr std::size_t metadataWords = sizeof(Abi::ResourceMetadata) / sizeof(std::uint32_t);
    for (std::uint32_t image = 0; image < program.runtimeImageCount; ++image) {
        for (std::size_t word = 0; word < metadataWords; ++word) mark(program.imageMetadataDword + image * metadataWords + word);
    }
    for (const auto& word : program.heapSlotWords) mark(word.dword);
    if (program.registrationModule) {
        for (std::size_t index = 0; index < state.samplerSlots.size(); ++index) mark(program.samplerSlotDword + index);
    }
}

void ShaderResources::noteHeapStageCensus(const CompiledShader& shader, bool current) const {
    const auto& program = *shader.program;
    const auto& state = heapStages[HeapStageIndex(shader.stage)];
    const ShaderRecompiler::DescriptorBinding* data = nullptr;
    const ShaderRecompiler::DescriptorBinding* srt = nullptr;
    for (const auto& binding : program.bindings) {
        if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) data = &binding;
        else if (binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt) srt = &binding;
    }
    if (data == nullptr) return;
    const auto& words = DrawGuestWords(shader, *data);
    const auto userData = shader.heap != nullptr ? std::span<const std::uint32_t>(shader.heap->userData) : std::span<const std::uint32_t>{};
    bool srtChanged = false;
    if (srt != nullptr) {
        const auto& srtWords = shader.heap != nullptr && !shader.heap->flattenedSrt.empty() ? shader.heap->flattenedSrt : DrawGuestWords(shader, *srt);
        srtChanged = !std::ranges::equal(state.flat.words, srtWords);
    }
    const auto size = words.size();
    auto change = ReplayStageChange::Same;
    bool underUserData = false;
    if (!current || state.dataInput.size() != size + userData.size()) {
        change = ReplayStageChange::Reshaped;
    } else if (const auto copied = std::min(userData.size(), size); !std::equal(words.begin() + static_cast<std::ptrdiff_t>(copied), words.end(), state.dataInput.begin() + static_cast<std::ptrdiff_t>(copied))) {
        change = ReplayStageChange::WordsChanged;
    } else {
        underUserData = !std::equal(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(copied), state.dataInput.begin());
        struct MaskStorage {};
        auto& mask = HostThreadLocal<std::vector<std::uint64_t>, MaskStorage>();
        heapStageMask(shader, size, mask);
        std::size_t changed = 0;
        bool masked = false;
        for (std::size_t index = 0; index < copied; ++index) {
            if (userData[index] == state.dataInput[size + index]) continue;
            ++changed;
            masked = masked || ((mask[index / 64u] >> (index % 64u)) & 1u) != 0u;
        }
        change = masked ? ReplayStageChange::Masked : changed == 0 ? ReplayStageChange::Same : changed == 1 ? ReplayStageChange::One : changed <= 4 ? ReplayStageChange::UpToFour : changed <= 16 ? ReplayStageChange::UpToSixteen : ReplayStageChange::More;
    }
    if (current) {
        const auto& heap = BindlessHeap::For(context);
        const bool dataSame = change == ReplayStageChange::Same;
        const bool srtSame = srt != nullptr && !srtChanged;
        NoteReplayOpenSpans(dataSame, dataSame && heap.SpanCurrent(state.data.mark), srtSame, srtSame && heap.SpanCurrent(state.flat.mark));
    }
    NoteReplayStageChange(change, srtChanged, &words == &data->guestDescriptor, underUserData);
}

void ShaderResources::seedHeapReplayStage(HeapReplayStage& stage, const CompiledShader& shader) const {
    const auto& program = *shader.program;
    stage.program = shader.program;
    stage.ready = true;
    stage.replayable = false;
    stage.spanEpoch = 0;
    std::size_t dataBindings = 0;
    std::size_t srtBindings = 0;
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        const auto role = program.bindings[index].role;
        if (role == ShaderRecompiler::DescriptorRole::ShaderData) {
            ++dataBindings;
            stage.dataBinding = index;
        } else if (role == ShaderRecompiler::DescriptorRole::FlattenedSrt) {
            ++srtBindings;
            stage.srtBinding = index;
        }
    }
    stage.srt = srtBindings == 1;
    if (!program.dataAddressPush || !context.heapBufferDescriptors || dataBindings != 1 || srtBindings > 1) return;
    const auto& data = program.bindings[stage.dataBinding];
    const auto& words = DrawGuestWords(shader, data);
    stage.userDwords = shader.heap != nullptr ? shader.heap->userData.size() : 0u;
    stage.dataDwords = words.size();
    stage.wordsFromProgram = &words == &data.guestDescriptor;
    heapStageMask(shader, words.size(), stage.mask);
    stage.replayable = true;
}

bool ShaderResources::keepReplaySpan(HeapStageState::KeptSpan& kept, bool proven, bool srt) {
    if (kept.mark.entry == 0u || kept.mark.words == 0u) return false;
    if (proven) {
        NoteReplay(srt ? ReplayEvent::SrtProven : ReplayEvent::DataProven);
        return true;
    }
    if (!BindlessHeap::For(context).Held(kept.mark)) return false;
    NoteReplay(srt ? ReplayEvent::SrtHeld : ReplayEvent::DataHeld);
    return true;
}

void ShaderResources::writeReplaySpan(HeapStageState::KeptSpan& kept, std::span<const std::uint32_t> words) {
    BindlessHeap::SpanMark mark;
    const auto span = BindlessHeap::For(context).Allocate(words.size(), mark);
    if (!words.empty()) std::memcpy(span.words.data(), words.data(), words.size_bytes());
    if (words.data() != kept.words.data()) kept.words.assign(words.begin(), words.end());
    kept.mark = mark;
}

bool ShaderResources::replayHeapStage(const CompiledShader& shader, HeapReplayStage& stage, std::uint64_t epoch, bool overlay) {
    namespace Abi = ShaderRecompiler::RuntimeAbi;
    auto& heap = BindlessHeap::For(context);
    const auto stageIndex = HeapStageIndex(shader.stage);
    auto& state = heapStages[stageIndex];
    const auto& program = *shader.program;
    const auto rebuild = [&](ReplayEvent why) {
        NoteReplay(why);
        writeHeapStage(shader);
        seedHeapReplayStage(stage, shader);
        return false;
    };
    const auto slotSerial = heap.SlotSerial();
    if (!(EpochProofs() && state.slotSerial == slotSerial)) {
        if (!heap.GenerationsCurrent(state.slots, state.slotGenerations, state.samplerSlots, state.samplerGenerations)) {
            if (ReplayCensusEnabled()) noteHeapStageCensus(shader, false);
            return rebuild(ReplayEvent::SlotsMoved);
        }
        state.slotSerial = slotSerial;
    }
    if (ReplayCensusEnabled()) noteHeapStageCensus(shader, true);
    static const bool verifySlots = std::getenv("APS5_VERIFY_GPU_HEAP") != nullptr;
    if (verifySlots) verifyHeapSlots(heap, state, VulkanStage(shader.stage));
    const auto& dataBinding = program.bindings[stage.dataBinding];
    const auto& words = DrawGuestWords(shader, dataBinding);
    const auto userData = shader.heap != nullptr ? std::span<const std::uint32_t>(shader.heap->userData) : std::span<const std::uint32_t>{};
    const auto size = words.size();
    if (size != stage.dataDwords || userData.size() != stage.userDwords || state.dataInput.size() != size + userData.size() || state.data.words.size() != size) return rebuild(ReplayEvent::ShapeChanged);
    const bool fromProgram = &words == &dataBinding.guestDescriptor;
    if (!(fromProgram && stage.wordsFromProgram) && !std::equal(words.begin(), words.end(), state.dataInput.begin())) return rebuild(ReplayEvent::ShapeChanged);
    struct ChangedStorage {};
    auto& changed = HostThreadLocal<std::vector<std::uint32_t>, ChangedStorage>();
    changed.clear();
    const auto copied = std::min(userData.size(), size);
    const auto* recorded = state.dataInput.data() + size;
    for (std::size_t index = 0; index < copied; ++index) {
        if (userData[index] == recorded[index]) continue;
        if (((stage.mask[index / 64u] >> (index % 64u)) & 1u) != 0u) return rebuild(ReplayEvent::MaskedChange);
        changed.push_back(static_cast<std::uint32_t>(index));
    }
    if (auto& overlayWords = stageDataWords[stageIndex]; !overlayWords.empty() && RuntimeSwitches::Set(RuntimeSwitch::ReplayBind)) {
        if (overlayWords.size() != size) return rebuild(ReplayEvent::ShapeChanged);
        for (const auto index : changed) overlayWords[index] = userData[index];
    }
    if (fromProgram) stage.wordsFromProgram = true;
    const bool proven = stage.spanEpoch == epoch;
    if (stage.srt) {
        const auto& srtWords = shader.heap != nullptr && !shader.heap->flattenedSrt.empty() ? shader.heap->flattenedSrt : DrawGuestWords(shader, program.bindings[stage.srtBinding]);
        if (std::ranges::equal(state.flat.words, srtWords)) {
            NoteReplay(ReplayEvent::SrtSame);
            if (!keepReplaySpan(state.flat, proven, true)) {
                NoteReplay(ReplayEvent::SrtRewritten);
                writeReplaySpan(state.flat, state.flat.words);
            }
        } else {
            NoteReplay(ReplayEvent::SrtChanged);
            writeReplaySpan(state.flat, srtWords);
        }
        heapSrtOffsets[stageIndex] = state.flat.mark.offset;
        heapSrtDwords[stageIndex] = static_cast<std::uint32_t>(srtWords.size());
    } else {
        heapSrtDwords[stageIndex] = 0u;
    }
    const bool overlayWrites = overlay && !stageDataWords[stageIndex].empty() && RuntimeSwitches::Set(RuntimeSwitch::ReplayBind) && !RuntimeSwitches::Set(RuntimeSwitch::NoHeapSetSkip);
    if (overlayWrites) {
        NoteReplay(changed.empty() ? ReplayEvent::DataSame : ReplayEvent::DataPatched);
        for (const auto index : changed) state.data.words[index] = userData[index];
        state.data.mark = {};
    } else if (!changed.empty()) {
        NoteReplay(ReplayEvent::DataPatched);
        BindlessHeap::SpanMark mark;
        const auto span = heap.Allocate(size, mark);
        for (const auto index : changed) state.data.words[index] = userData[index];
        std::memcpy(span.words.data(), state.data.words.data(), size * sizeof(std::uint32_t));
        state.data.mark = mark;
    } else {
        NoteReplay(ReplayEvent::DataSame);
        if (!keepReplaySpan(state.data, proven, false)) {
            NoteReplay(ReplayEvent::DataRewritten);
            writeReplaySpan(state.data, state.data.words);
        }
    }
    std::copy(userData.begin(), userData.end(), state.dataInput.begin() + static_cast<std::ptrdiff_t>(size));
    if (verifySlots) {
        struct RebuiltStorage {};
        auto& rebuilt = HostThreadLocal<std::vector<std::uint32_t>, RebuiltStorage>();
        buildHeapStageWords(shader, state, words, userData, rebuilt);
        if (!std::ranges::equal(state.data.words, rebuilt)) throw std::runtime_error("AGC graphics: a replayed GPU-mode stage's shader data differs from the block built anew from its inputs and slots");
    }
    heapOffsets[DataOffsetIndex(stageIndex, Abi::Binding::ShaderData)] = state.data.mark.offset;
    if (overlay) {
        NoteReplay(ReplayEvent::RunsSkipped);
    } else {
        NoteReplay(ReplayEvent::RunsBuilt);
        heapOffsets[DataOffsetIndex(stageIndex, Abi::Binding::FlattenedSrt)] = ringDescriptors(stageIndex, nullptr, state.descriptors);
    }
    return true;
}

ShaderResources::HeapReplayProof ShaderResources::ReplayHeapStages(std::span<const CompiledShader> shaders, bool overlay) {
    auto& heap = BindlessHeap::For(context);
    auto& record = heapReplayRecord();
    const auto switches = RuntimeSwitches::Generation();
    if (record.open || record.heap != &heap || record.heapIdentity != heap.Identity() || record.switches != switches || record.writes != heapStageWrites) {
        NoteReplay(record.heap == nullptr ? ReplayEvent::NewRecord : record.open ? ReplayEvent::Interrupted : record.writes != heapStageWrites ? ReplayEvent::NonReplayWrites : ReplayEvent::HeapOrSwitches);
        for (auto& stage : record.stages) stage.ready = false;
        record.heap = &heap;
        record.heapIdentity = heap.Identity();
        record.switches = switches;
    }
    record.open = true;
    NoteReplay(ReplayEvent::Draws);
    struct Replaying {
        bool& active;
        explicit Replaying(bool& active) : active(active) { active = true; }
        ~Replaying() { active = false; }
        Replaying(const Replaying&) = delete;
        Replaying& operator=(const Replaying&) = delete;
    } replaying{heapReplaying};
    const auto before = heap.SubmitEpoch();
    std::array<bool, ShaderRecompiler::RuntimeAbi::StageCount> current{};
    for (const auto& shader : shaders) {
        if (!HeapShader(shader)) continue;
        const auto stageIndex = HeapStageIndex(shader.stage);
        auto& stage = record.stages[stageIndex];
        if (stage.ready && stage.replayable && stage.program == shader.program) {
            if (shader.program->dataAddressPush) writeHeapTableMap(shader, stageIndex);
            current[stageIndex] = true;
        } else {
            if (stage.ready) NoteReplay(stage.program != shader.program ? ReplayEvent::NewProgram : ReplayEvent::NotReplayable);
            writeHeapStage(shader);
            seedHeapReplayStage(stage, shader);
        }
        RunHeapReplayRefreshHookForTest();
    }
    const auto epoch = heap.SubmitEpoch();
    if (epoch != before) {
        NoteReplay(ReplayEvent::SubmitDuringRefresh);
        RefreshHeapStage(shaders);
        reseedHeapReplay(shaders);
        record.writes = heapStageWrites;
        record.open = false;
        return {epoch, overlay};
    }
    for (const auto& shader : shaders) {
        if (!HeapShader(shader)) continue;
        const auto stageIndex = HeapStageIndex(shader.stage);
        auto& stage = record.stages[stageIndex];
        if (current[stageIndex] && replayHeapStage(shader, stage, epoch, overlay)) NoteReplay(ReplayEvent::Stages);
        stage.spanEpoch = epoch;
    }
    record.writes = heapStageWrites;
    record.open = false;
    return {epoch, overlay};
}

void ShaderResources::reseedHeapReplay(std::span<const CompiledShader> shaders) {
    auto& record = heapReplayRecord();
    for (auto& stage : record.stages) stage.spanEpoch = 0;
    for (const auto& shader : shaders) {
        if (HeapShader(shader)) seedHeapReplayStage(record.stages[HeapStageIndex(shader.stage)], shader);
    }
}

void ShaderResources::refreshHeapReplayFallback(Recorder& recorder, std::span<const CompiledShader> shaders, const std::shared_ptr<DrawBindings>& bindings, std::uint64_t provenEpoch) {
    auto& heap = BindlessHeap::For(context);
    for (auto epoch = heap.SubmitEpoch();; epoch = heap.SubmitEpoch()) {
        RefreshHeapStage(shaders);
        if (bindings != nullptr) RefreshMovedAddresses(*bindings);
        if (heap.SubmitEpoch() == epoch) break;
        NoteReplay(ReplayEvent::SubmitDuringFallback);
    }
    if (bindings == nullptr || heap.SubmitEpoch() == provenEpoch) return;
    recorder.Keep(bindings);
    for (const auto& snapshot : bindings->snapshots) {
        if (snapshot.buffer != nullptr) recorder.KeepBytes(snapshot.buffer.get(), snapshot.buffer->Bytes().size());
    }
}

void ShaderResources::ConfirmHeapReplay(Recorder& recorder, std::span<const CompiledShader> shaders, const std::shared_ptr<DrawBindings>& bindings, const HeapReplayProof& proof) {
    namespace Abi = ShaderRecompiler::RuntimeAbi;
    if (BindlessHeap::For(context).SubmitEpoch() != proof.epoch) {
        NoteReplay(ReplayEvent::SubmitBeforePass);
        refreshHeapReplayFallback(recorder, shaders, bindings, proof.epoch);
        return;
    }
    const bool overlaid = bindings != nullptr && bindings->addressOffsets;
    if (proof.overlay && !overlaid) {
        NoteReplay(ReplayEvent::OverlayMissing);
        for (const auto& shader : shaders) {
            if (!HeapShader(shader)) continue;
            const auto stageIndex = HeapStageIndex(shader.stage);
            if (ringDescriptorStages[stageIndex]) heapOffsets[DataOffsetIndex(stageIndex, Abi::Binding::FlattenedSrt)] = ringDescriptors(stageIndex, nullptr, heapStages[stageIndex].descriptors);
        }
    }
    if (ReplayVerifyLevel() != 0) verifyHeapReplay(recorder, shaders, bindings, proof, overlaid);
}

void ShaderResources::buildHeapStageWords(const CompiledShader& shader, const HeapStageState& state, std::span<const std::uint32_t> words, std::span<const std::uint32_t> userData, std::vector<std::uint32_t>& patched) const {
    ValidateRuntimeResources(shader, words);
    patched.assign(words.begin(), words.end());
    std::copy_n(userData.begin(), std::min(userData.size(), patched.size()), patched.begin());
    for (const auto& [byte, adjustment] : state.dataPatches) {
        Require(byte < patched.size() * sizeof(std::uint32_t), "a GPU-mode guest buffer offset lies outside the shader data");
        reinterpret_cast<std::byte*>(patched.data())[byte] = static_cast<std::byte>(adjustment);
    }
    patchHeapSlotWords(shader, state, patched, true);
}

void ShaderResources::verifyHeapReplay(Recorder& recorder, std::span<const CompiledShader> shaders, const std::shared_ptr<DrawBindings>& bindings, const HeapReplayProof& proof, bool overlaid) {
    namespace Abi = ShaderRecompiler::RuntimeAbi;
    auto& heap = BindlessHeap::For(context);
    const bool shadow = ReplayVerifyLevel() == 1;
    const auto keyHash = heapReplay != nullptr ? heapReplay->keyHash : 0u;
    std::string failure;
    const auto mismatch = [&](ReplayField field, std::size_t stageIndex, const std::string& what) {
        char prefix[128];
        std::snprintf(prefix, sizeof(prefix), "template key %016llx (%p) stage %zu: ", static_cast<unsigned long long>(keyHash), static_cast<const void*>(this), stageIndex);
        if (NoteReplayMismatch(field, prefix + what) && failure.empty()) failure = std::string("APS5_VERIFY_REPLAY_STRICT: the heap stage replay's ") + ReplayFieldName(field) + " differs from today's: " + prefix + what;
    };
    const auto difference = [](std::span<const std::uint32_t> found, std::span<const std::uint32_t> expected) -> std::string {
        if (found.size() != expected.size()) return std::to_string(found.size()) + " dwords, expected " + std::to_string(expected.size());
        for (std::size_t index = 0; index < found.size(); ++index) {
            if (found[index] == expected[index]) continue;
            char text[96];
            std::snprintf(text, sizeof(text), "dword %zu is 0x%08x, expected 0x%08x", index, found[index], expected[index]);
            return text;
        }
        return {};
    };
    const auto sameRanges = [](std::span<const VkDeviceAddressRangeEXT> a, std::span<const VkDeviceAddressRangeEXT> b) {
        return std::ranges::equal(a, b, [](const VkDeviceAddressRangeEXT& left, const VkDeviceAddressRangeEXT& right) { return left.address == right.address && left.size == right.size; });
    };
    const auto spanHolds = [&](const HeapStageState::KeptSpan& kept, std::size_t stageIndex, const char* what) {
        if (kept.words.empty()) return;
        if (!heap.SpanCurrent(kept.mark)) mismatch(ReplayField::Spans, stageIndex, std::string(what) + " span is not in an open chunk of this batch");
        else if (const auto text = difference(heap.RingWords(kept.mark), kept.words); !text.empty()) mismatch(ReplayField::Ring, stageIndex, std::string(what) + " ring words differ from the CPU copy: " + text);
    };
    struct Facts {
        std::size_t stageIndex = 0;
        std::vector<std::uintptr_t> key;
        std::array<std::vector<std::uint32_t>, static_cast<std::size_t>(Abi::Binding::Count)> slots;
        std::vector<std::uint32_t> samplerSlots;
        std::vector<std::uint32_t> block;
        std::vector<std::uint32_t> srt;
        std::uint32_t srtDwords = 0;
        std::vector<std::uint32_t> map;
        std::vector<VkDeviceAddressRangeEXT> run;
    };
    std::vector<Facts> replayed;
    if (proof.overlay != overlaid) {
        const auto first = std::ranges::find_if(shaders, [](const CompiledShader& shader) { return HeapShader(shader); });
        mismatch(ReplayField::Runs, first != shaders.end() ? HeapStageIndex(first->stage) : 0u, proof.overlay ? "the replay predicted an overlay the draw's bindings do not have" : "the draw's bindings have an overlay the replay did not predict");
    }
    for (const auto& shader : shaders) {
        if (!HeapShader(shader)) continue;
        NoteReplayCheck();
        const auto stageIndex = HeapStageIndex(shader.stage);
        const auto stageBit = static_cast<VkShaderStageFlags>(VulkanStage(shader.stage));
        auto& state = heapStages[stageIndex];
        const auto dataIndex = DataOffsetIndex(stageIndex, Abi::Binding::ShaderData);
        const auto flatIndex = DataOffsetIndex(stageIndex, Abi::Binding::FlattenedSrt);
        if (heapOffsets[dataIndex] != state.data.mark.offset) mismatch(ReplayField::Offsets, stageIndex, "the root's shader data offset is not the stage data span");
        spanHolds(state.data, stageIndex, "the stage data");
        const ShaderRecompiler::DescriptorBinding* data = nullptr;
        const ShaderRecompiler::DescriptorBinding* srt = nullptr;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) data = &binding;
            else if (binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt) srt = &binding;
        }
        std::span<const std::uint32_t> srtWords;
        if (srt != nullptr) {
            srtWords = shader.heap != nullptr && !shader.heap->flattenedSrt.empty() ? std::span<const std::uint32_t>(shader.heap->flattenedSrt) : std::span<const std::uint32_t>(DrawGuestWords(shader, *srt));
            if (heapSrtOffsets[stageIndex] != state.flat.mark.offset || heapSrtDwords[stageIndex] != srtWords.size()) mismatch(ReplayField::Offsets, stageIndex, "the root's srt offset or dwords are not the srt span's");
            spanHolds(state.flat, stageIndex, "the srt");
        } else if (heapSrtDwords[stageIndex] != 0u) {
            mismatch(ReplayField::Offsets, stageIndex, "a stage without an srt has srt dwords");
        }
        if (heapTableMapSet[stageIndex]) {
            if (heapTableMaps[stageIndex] != state.map.mark.offset) mismatch(ReplayField::Map, stageIndex, "the root's table map offset is not the table map span");
            spanHolds(state.map, stageIndex, "the table map");
        }
        if (!overlaid) {
            if (heapOffsets[flatIndex] != state.descriptors.first) mismatch(ReplayField::Offsets, stageIndex, "the root's descriptor index is not the template run's");
            if (!heap.SpanCurrent(state.descriptors.mark)) mismatch(ReplayField::Spans, stageIndex, "the template run is not in an open chunk of this batch");
            const std::array<VkDeviceAddressRangeEXT, 1> srtEntry{srtRange(stageIndex)};
            if (state.descriptors.ranges.empty() || !sameRanges(std::span(state.descriptors.ranges).first(1), srtEntry)) mismatch(ReplayField::Runs, stageIndex, "the template run's first entry is not the srt span");
        }
        if (overlaid && bindings->heapOffsets[dataIndex] != heapOffsets[dataIndex]) mismatch(ReplayField::Offsets, stageIndex, "the overlay's shader data offset is not the replayed one");
        if (data == nullptr) continue;
        const auto& words = DrawGuestWords(shader, *data);
        const auto userData = shader.heap != nullptr ? std::span<const std::uint32_t>(shader.heap->userData) : std::span<const std::uint32_t>{};
        if (shadow) {
            Facts facts;
            facts.stageIndex = stageIndex;
            facts.key = state.key;
            facts.slots = state.slots;
            facts.samplerSlots = state.samplerSlots;
            facts.block = state.data.words;
            facts.srt = state.flat.words;
            facts.srtDwords = heapSrtDwords[stageIndex];
            if (heapTableMapSet[stageIndex]) facts.map = state.map.words;
            if (!overlaid) facts.run = state.descriptors.ranges;
            replayed.push_back(std::move(facts));
            continue;
        }
        std::vector<std::uintptr_t> key;
        appendHeapSlotKey(stageBit, key);
        if (key != state.key) mismatch(ReplayField::Key, stageIndex, "the slot key built anew differs from the record's");
        HeapStageState reference;
        reference.dataPatches = state.dataPatches;
        takeHeapSlots(heap, reference, stageBit, key);
        if (reference.slots != state.slots || reference.samplerSlots != state.samplerSlots || !heap.GenerationsCurrent(state.slots, state.slotGenerations, state.samplerSlots, state.samplerGenerations)) mismatch(ReplayField::Slots, stageIndex, "the slots taken anew differ from the record's or their generations moved");
        std::vector<std::uint32_t> block;
        buildHeapStageWords(shader, reference, words, userData, block);
        if (const auto text = difference(state.data.words, block); !text.empty()) mismatch(ReplayField::Block, stageIndex, "the stage data: " + text);
        const bool inputs = state.dataInput.size() == words.size() + userData.size() && std::equal(words.begin(), words.end(), state.dataInput.begin()) && std::equal(userData.begin(), userData.end(), state.dataInput.begin() + static_cast<std::ptrdiff_t>(words.size()));
        if (!inputs) mismatch(ReplayField::Inputs, stageIndex, "the recorded inputs are not this draw's");
        if (srt != nullptr) {
            if (const auto text = difference(state.flat.words, srtWords); !text.empty()) mismatch(ReplayField::Srt, stageIndex, "the srt words: " + text);
        }
        if (!overlaid) {
            std::vector<VkDeviceAddressRangeEXT> ranges{srtRange(stageIndex)};
            writeBufferAddresses({}, static_cast<std::uint32_t>(stageIndex), nullptr, &ranges);
            if (!sameRanges(ranges, state.descriptors.ranges)) mismatch(ReplayField::Runs, stageIndex, "the template run's ranges differ from the ranges built anew");
        }
    }
    if (shadow) {
        for (const auto& shader : shaders) {
            if (!HeapShader(shader)) continue;
            auto& state = heapStages[HeapStageIndex(shader.stage)];
            state.key.clear();
            state.data.mark.entry = 0;
            state.flat.mark.entry = 0;
            state.map.mark.entry = 0;
            state.descriptors.mark.entry = 0;
        }
        refreshHeapReplayFallback(recorder, shaders, bindings, proof.epoch);
        for (const auto& facts : replayed) {
            const auto stageIndex = facts.stageIndex;
            const auto& state = heapStages[stageIndex];
            if (facts.key != state.key) mismatch(ReplayField::Key, stageIndex, "the record's slot key differs from the one today's refresh built");
            if (facts.slots != state.slots || facts.samplerSlots != state.samplerSlots) mismatch(ReplayField::Slots, stageIndex, "the record's slots differ from the ones today's refresh took");
            if (const auto text = difference(facts.block, state.data.words); !text.empty()) mismatch(ReplayField::Block, stageIndex, "the replayed stage data against today's refresh: " + text);
            if (const auto text = difference(facts.srt, state.flat.words); !text.empty() || facts.srtDwords != heapSrtDwords[stageIndex]) mismatch(ReplayField::Srt, stageIndex, "the replayed srt against today's refresh: " + text);
            if (heapTableMapSet[stageIndex]) {
                if (const auto text = difference(facts.map, state.map.words); !text.empty()) mismatch(ReplayField::Map, stageIndex, "the table map against today's refresh: " + text);
            }
            if (!facts.run.empty()) {
                std::vector<VkDeviceAddressRangeEXT> expected(state.descriptors.ranges.begin(), state.descriptors.ranges.end());
                if (!expected.empty()) expected.front() = facts.run.front();
                if (!sameRanges(facts.run, expected)) mismatch(ReplayField::Runs, stageIndex, "the replayed template run binds other buffers than today's");
            }
        }
        auto& record = heapReplayRecord();
        const auto epoch = heap.SubmitEpoch();
        for (const auto& shader : shaders) {
            if (!HeapShader(shader)) continue;
            auto& stage = record.stages[HeapStageIndex(shader.stage)];
            seedHeapReplayStage(stage, shader);
            stage.spanEpoch = epoch == proof.epoch ? epoch : 0u;
        }
        record.writes = heapStageWrites;
    }
    if (!failure.empty()) throw std::runtime_error(failure);
}

namespace {
// Debug aid: APS5_GPU_NO_WRITEBACK=1 keeps GPU results out of guest memory.
bool SkipWriteBack() {
    static const bool skip = std::getenv("APS5_GPU_NO_WRITEBACK") != nullptr;
    return skip;
}
}

void ShaderResources::WriteBack() {
    std::exception_ptr unaligned;
    try {
        WriteBackBuffers();
    } catch (const BdaUnalignedAccess&) {
        unaligned = std::current_exception();
    }
    if (!guestMemory.Writes().empty() || std::find(storageWritten.begin(), storageWritten.end(), true) != storageWritten.end() || (usesBda && !bdaReadOnly)) GuestMemory::BreakProofs(GuestMemory::ProofBreak::BuildWrites);
    if (!SkipWriteBack()) {
        for (std::size_t index = 0; index < storageTextures.size(); ++index) {
            if (!storageWritten[index] || storageTextures[index] == nullptr) continue;
            storageTextures[index]->MarkDirty();
            static_cast<void>(DepthSurfaceStoreBack(context, *storageTextures[index]));
        }
    }
    if (unaligned) std::rethrow_exception(unaligned);
}

void ShaderResources::notePendingReads(Recorder& recorder, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, std::uint8_t kind, bool lean) {
    const auto readKind = static_cast<Recorder::ReadKind>(kind);
    if (!lean) {
        recorder.NotePendingReads(ranges, readKind);
        return;
    }
    const auto batch = recorder.OpenBatchGeneration();
    if (batch != 0 && readNote.batch == batch && readNote.kind == kind && std::ranges::equal(readNote.ranges, ranges)) return;
    recorder.NotePendingReads(ranges, readKind);
    readNote.batch = 0;
    try {
        readNote.ranges.assign(ranges.begin(), ranges.end());
        readNote.kind = kind;
        readNote.batch = recorder.OpenBatchGeneration();
    } catch (const std::bad_alloc&) {
    }
}

void ShaderResources::MarkGpuWrites(Recorder& recorder) {
    if (bda != nullptr && recorder.Device() == context.device) bda->NoteUse(recorder.OpenSerial());
    // The ranges this use reads in place through their host imports (read-only and written elements
    // alike, and an address-based build's whole leased heaps), before the writes: a CPU store into
    // one of them (the copy HLE) must not land before the recorded work read it.
    static const bool readOnlyLeaseSkip = std::getenv("APS5_NO_READ_ONLY_LEASE_SKIP") == nullptr;
    const auto kind = !guestMemory.HoldsLease() ? Recorder::ReadKind::DispatchElement : readOnlyLeaseSkip && usesBda && bdaReadOnly ? Recorder::ReadKind::AddressBasedReadOnly : Recorder::ReadKind::AddressBased;
    const bool lean = RuntimeSwitches::Set(RuntimeSwitch::KeepFastPath) && !Recorder::FlipReadCheck() && !CaptureTrace::Enabled();
    if (auto split = guestMemory.SplitInPlaceReads()) {
        recorder.NotePendingReadList(std::move(split->space), kind);
        notePendingReads(recorder, split->own, static_cast<std::uint8_t>(kind), lean);
    } else if (lean) {
        struct InPlaceReadsStorage {};
        auto& reads = HostThreadLocal<std::vector<std::pair<std::uint64_t, std::uint64_t>>, InPlaceReadsStorage>();
        guestMemory.InPlaceReads(reads);
        notePendingReads(recorder, reads, static_cast<std::uint8_t>(kind), lean);
    } else {
        recorder.NotePendingReads(guestMemory.InPlaceReads(), kind);
    }
    const auto inRun = std::exchange(afterRunSelf, nullptr);
    if (SkipWriteBack()) return;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (!storageWritten[index] || storageTextures[index] == nullptr) continue;
        storageTextures[index]->MarkDirty();
        static_cast<void>(DepthSurfaceStoreBack(context, *storageTextures[index]));
    }
    // Written sub-ranges of buffers the GPU copied out of a host import go back into it by the GPU,
    // recorded here after the work: those regions then need no CPU write-back (HasCopiedWrites),
    // and the note and mark below cover them like direct writes.
    const bool deferCopyBacks = inRun != nullptr && recorder.RunOpen() && guestMemory.HasGpuCopies();
    if (deferCopyBacks) {
        recorder.AfterRun([inRun](Recorder& after) { inRun->guestMemory.RecordCopyBacks(after); });
        guestMemory.MarkStagedWrites();
    } else {
        guestMemory.RecordCopyBacks(recorder);
    }
    // Only the written elements' ranges (AddWritable): a read-only element is neither noted here
    // nor marked as a direct write, so CPU reads of its memory never wait for this work.
    recorder.NotePendingWrites(guestMemory.Writes());
    if (usesBda && !bdaReadOnly) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> reachable;
        guestMemory.AddressStores(reachable);
        recorder.NotePendingReach(reachable);
    }
    guestMemory.MarkDirectWrites();
    if (ResidencyCensus::On()) guestMemory.CensusArm(deferCopyBacks);
    guestMemory.NoteCarriedShadows(recorder);
    if (!BuildProfiled()) return;
    auto& counters = BufferWrites();
    // Recorded uses only (dispatches and recorded draws): a synchronous draw writes back in
    // WriteBack() and publishes no pending writes, so it has no notes to skip.
    counters.notesSkipped.fetch_add(readOnlyBuffers, std::memory_order_relaxed);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto elements = counters.elements.load();
    const auto readOnly = counters.readOnly.load();
    AgcDriver::ProfilePrint_nid_no_patch("[buffers] descriptor elements bound (cumulative): %llu total, %llu written, %llu read-only; %llu pending-write notes skipped\n", static_cast<unsigned long long>(elements), static_cast<unsigned long long>(elements - readOnly), static_cast<unsigned long long>(readOnly), static_cast<unsigned long long>(counters.notesSkipped.load()));
    AgcDriver::ProfilePrint_nid_no_patch("[buffers] moved read-only elements under a recorded GPU write (cumulative): %llu read in place on a cached set, %llu refused and rebuilt\n", static_cast<unsigned long long>(MovedInPlace().taken.load()), static_cast<unsigned long long>(MovedInPlace().refused.load()));
}

void ShaderResources::MarkGpuWrites(Recorder& recorder, std::span<const MovedBuffer> moved) {
    if (std::none_of(moved.begin(), moved.end(), [](const MovedBuffer& entry) { return entry.inPlace != VK_NULL_HANDLE; })) {
        MarkGpuWrites(recorder);
        return;
    }
    if (bda != nullptr && recorder.Device() == context.device) bda->NoteUse(recorder.OpenSerial());
    static const bool readOnlyLeaseSkip = std::getenv("APS5_NO_READ_ONLY_LEASE_SKIP") == nullptr;
    const auto kind = !guestMemory.HoldsLease() ? Recorder::ReadKind::DispatchElement : readOnlyLeaseSkip && usesBda && bdaReadOnly ? Recorder::ReadKind::AddressBasedReadOnly : Recorder::ReadKind::AddressBased;
    recorder.NotePendingReads(MovedInPlaceReads(moved), kind);
    const auto inRun = std::exchange(afterRunSelf, nullptr);
    if (SkipWriteBack()) return;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (!storageWritten[index] || storageTextures[index] == nullptr) continue;
        storageTextures[index]->MarkDirty();
        static_cast<void>(DepthSurfaceStoreBack(context, *storageTextures[index]));
    }
    const bool deferCopyBacks = inRun != nullptr && recorder.RunOpen() && guestMemory.HasGpuCopies();
    if (deferCopyBacks) {
        recorder.AfterRun([inRun](Recorder& after) { inRun->guestMemory.RecordCopyBacks(after); });
        guestMemory.MarkStagedWrites();
    } else {
        guestMemory.RecordCopyBacks(recorder);
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> dropped;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> added;
    for (const auto& entry : moved) {
        if (!entry.written || entry.allocation >= allocations.size()) continue;
        const auto& item = allocations[entry.allocation];
        dropped.emplace_back(item.address, item.address + item.size);
        added.emplace_back(entry.address, entry.address + entry.size);
    }
    recorder.NotePendingWrites(MovedGpuWrites(moved));
    if (usesBda && !bdaReadOnly) {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> reachable;
        guestMemory.AddressStores(reachable);
        recorder.NotePendingReach(reachable);
    }
    guestMemory.MarkDirectWrites(dropped, added);
    if (ResidencyCensus::On()) guestMemory.CensusArm(deferCopyBacks);
}

void ShaderResources::CheckFaults() const {
    if (bda) bda->CheckFault();
}

void ShaderResources::WriteBackBuffers() {
    std::exception_ptr unaligned;
    if (bda) {
        try {
            bda->CheckFault();
        } catch (const BdaUnalignedAccess&) {
            unaligned = std::current_exception();
        }
    }
    if (!SkipWriteBack()) guestMemory.WriteBack();
    if (unaligned) std::rethrow_exception(unaligned);
}

std::span<const std::byte> ShaderResources::AddressTableBytes() const {
    return bda != nullptr ? bda->TableBytes() : std::span<const std::byte>{};
}

bool ShaderResources::AddressTableGuardCurrent() {
    if (bda == nullptr) return true;
    if (const auto cached = guestMemory.CachedAddressTable(); cached.has_value()) return bda->TableGuardCurrent(context.device, *cached->ranges);
    return bda->TableGuardCurrent(context.device, guestMemory.AddressRanges());
}

bool ShaderResources::ReadsOverlap(std::uint64_t address, std::size_t bytes) const {
    static const bool walk = std::getenv("APS5_NO_READS_OVERLAP_WALK") == nullptr;
    if (walk) return guestMemory.InPlaceReadsOverlap(address, bytes);
    const auto reads = guestMemory.InPlaceReads();
    return std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return address < range.second && range.first < address + bytes; });
}

void ShaderResources::RunAccesses(std::vector<std::array<std::uint64_t, 3>>& out) const {
    out.clear();
    guestMemory.RunAccesses(out);
    if (usesBda) out.push_back({0, std::numeric_limits<std::uint64_t>::max(), Recorder::RunAccessWritten});
    for (const auto& [image, written] : StorageImages()) {
        const auto handle = reinterpret_cast<std::uint64_t>(image);
        out.push_back({handle, handle + 1, Recorder::RunAccessImage | (written ? Recorder::RunAccessWritten : 0u)});
    }
}

namespace {

std::pair<std::uint64_t, std::uint64_t> DescriptorRegion(std::uint64_t address, std::uint64_t bytes) {
    const auto begin = address & ~std::uint64_t{3};
    const auto end = begin == address ? address + bytes : (address + bytes + 3u) & ~std::uint64_t{3};
    return {begin, end};
}

void CoverRanges(std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
    std::erase_if(ranges, [](const std::pair<std::uint64_t, std::uint64_t>& range) { return range.second <= range.first; });
    std::sort(ranges.begin(), ranges.end());
    std::size_t kept = 0;
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        if (kept != 0 && ranges[index].first <= ranges[kept - 1].second) ranges[kept - 1].second = std::max(ranges[kept - 1].second, ranges[index].second);
        else ranges[kept++] = ranges[index];
    }
    ranges.resize(kept);
}

}

bool ShaderResources::MovedRunAccesses(std::span<const MovedBuffer> moved, std::vector<std::array<std::uint64_t, 3>>& out) const {
    RunAccesses(out);
    if (moved.empty() || usesBda || guestMemory.HoldsLease()) return false;
    const auto region = DescriptorRegion;
    const auto cover = CoverRanges;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> held;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> own;
    std::vector<std::array<std::uint64_t, 3>> live;
    for (const auto& access : out) {
        if ((access[2] & Recorder::RunAccessImage) != 0) live.push_back(access);
        else if ((access[2] & Recorder::RunAccessWritten) == 0) held.emplace_back(access[0], access[1]);
    }
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& item = allocations[index];
        if (!item.guest) continue;
        own.push_back(region(item.address, item.size));
        const auto entry = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& candidate) { return candidate.allocation == index; });
        const auto range = entry != moved.end() ? region(entry->address, entry->size) : own.back();
        live.push_back({range.first, range.second, 0});
    }
    cover(held);
    cover(own);
    if (held != own) {
        for (const auto& entry : moved) {
            const auto range = region(entry.address, entry.size);
            out.push_back({range.first, range.second, 0});
            if (entry.written) out.push_back({entry.address, entry.address + entry.size, Recorder::RunAccessWritten});
        }
        return false;
    }
    for (const auto& [begin, end] : MovedGpuWrites(moved)) live.push_back({begin, end, Recorder::RunAccessWritten});
    out = std::move(live);
    return true;
}

std::vector<std::string> ShaderResources::VerifyMovedRunAccesses(const CompiledShader& shader, std::span<const std::array<std::uint64_t, 3>> accesses) const {
    std::vector<std::string> differences;
    if (shader.program == nullptr) {
        differences.emplace_back("no compiled shader");
        return differences;
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> bound(allocations.size());
    for (std::size_t index = 0; index < allocations.size(); ++index) bound[index] = {allocations[index].address, allocations[index].address + allocations[index].size};
    for (const auto& binding : shader.program->bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) continue;
        const auto kept = std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
        if (kept == bindings.end() || kept->allocations.size() != binding.count) {
            char text[96];
            std::snprintf(text, sizeof(text), "binding %u has another shape than the template's", binding.binding);
            differences.emplace_back(text);
            continue;
        }
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
            const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
            const auto address = descriptor.Base48();
            const auto size = guestBufferBytes(std::span<const std::uint32_t>(words, 4));
            const auto index = kept->allocations[element];
            if (index >= allocations.size() || !allocations[index].guest) continue;
            bound[index] = size == 0 || address == 0 ? std::pair<std::uint64_t, std::uint64_t>{0, 0} : std::pair<std::uint64_t, std::uint64_t>{address, address + size};
        }
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> expectedAll, expectedWrites, recordedAll, recordedWrites;
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        if (!allocations[index].guest || bound[index].second <= bound[index].first) continue;
        expectedAll.push_back(DescriptorRegion(bound[index].first, bound[index].second - bound[index].first));
        if (allocations[index].written) expectedWrites.push_back(bound[index]);
    }
    for (const auto& access : accesses) {
        if ((access[2] & Recorder::RunAccessImage) != 0) continue;
        recordedAll.emplace_back(access[0], access[1]);
        if ((access[2] & Recorder::RunAccessWritten) != 0) recordedWrites.emplace_back(access[0], access[1]);
    }
    expectedAll.insert(expectedAll.end(), expectedWrites.begin(), expectedWrites.end());
    CoverRanges(expectedAll);
    CoverRanges(expectedWrites);
    CoverRanges(recordedAll);
    CoverRanges(recordedWrites);
    const auto compare = [&](const char* what, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& recorded, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& expected) {
        if (recorded == expected) return;
        std::size_t at = 0;
        while (at < recorded.size() && at < expected.size() && recorded[at] == expected[at]) ++at;
        const auto one = [](const std::vector<std::pair<std::uint64_t, std::uint64_t>>& list, std::size_t index) { return index < list.size() ? list[index] : std::pair<std::uint64_t, std::uint64_t>{0, 0}; };
        char text[224];
        std::snprintf(text, sizeof(text), "the %s differ (%zu ranges recorded, %zu expected), first at %zu: recorded 0x%llx-0x%llx, expected 0x%llx-0x%llx", what, recorded.size(), expected.size(), at, static_cast<unsigned long long>(one(recorded, at).first), static_cast<unsigned long long>(one(recorded, at).second), static_cast<unsigned long long>(one(expected, at).first), static_cast<unsigned long long>(one(expected, at).second));
        differences.emplace_back(text);
    };
    compare("accessed ranges", recordedAll, expectedAll);
    compare("written ranges", recordedWrites, expectedWrites);
    return differences;
}

std::vector<std::pair<VkImage, bool>> ShaderResources::StorageImages() const {
    std::vector<std::pair<VkImage, bool>> images;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (storageTextures[index] != nullptr) images.emplace_back(storageTextures[index]->Image(), storageWritten[index]);
    }
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->StorageSource() != nullptr) images.emplace_back(texture->StorageSource()->Image(), false);
    }
    return images;
}

bool ShaderResources::HasCachedStorageImages() const {
    if (std::any_of(storageTextures.begin(), storageTextures.end(), [](const auto& image) { return image != nullptr && !image->Cached(); })) return false;
    return std::none_of(textures.begin(), textures.end(), [](const auto& texture) { return texture != nullptr && texture->StorageSource() != nullptr && !texture->StorageSource()->Cached(); });
}

bool ShaderResources::HoldsReleasedTexture() const {
    return std::any_of(textures.begin(), textures.end(), [](const auto& texture) { return texture != nullptr && texture->Released(); });
}

bool ShaderResources::HoldsDiscardedTexture() const {
    return std::any_of(textures.begin(), textures.end(), [](const auto& texture) { return texture != nullptr && texture->Discarded(); });
}

bool ShaderResources::RevivedUse() const {
    const auto revivals = textureRevivals().load(std::memory_order_acquire);
    if (revivals == revivedSeen.load(std::memory_order_relaxed)) return false;
    const auto seen = revivedSeen.exchange(revivals, std::memory_order_relaxed);
    if (seen >= revivals) return false;
    return std::any_of(textures.begin(), textures.end(), [&](const auto& texture) { return texture != nullptr && texture->RevivedAt() > seen; });
}

bool ShaderResources::KeptUse() const {
    const auto refills = Texture::Refills();
    if (refills == keptSeen.load(std::memory_order_relaxed)) return false;
    const auto seen = keptSeen.exchange(refills, std::memory_order_relaxed);
    if (seen >= refills) return false;
    return std::any_of(textures.begin(), textures.end(), [&](const auto& texture) { return texture != nullptr && texture->RefillSerial() > seen; });
}

const char* ShaderResources::KeptDifference(const ShaderResources& fresh, std::span<const MovedBuffer> moved) const {
    if (textures != fresh.textures) return "sampled textures";
    if (StorageImages() != fresh.StorageImages()) return "storage images";
    if (layoutKey != fresh.layoutKey || fragmentLayoutKey != fresh.fragmentLayoutKey) return "set layouts";
    if (descriptorWrites.size() != fresh.descriptorWrites.size()) return "descriptor writes";
    for (std::size_t index = 0; index < descriptorWrites.size(); ++index) {
        const auto& mine = descriptorWrites[index];
        const auto& other = fresh.descriptorWrites[index];
        if (mine.dstBinding != other.dstBinding || mine.dstArrayElement != other.dstArrayElement || mine.descriptorCount != other.descriptorCount || mine.descriptorType != other.descriptorType || (mine.dstSet == _set) != (other.dstSet == fresh._set) || (mine.pImageInfo != nullptr) != (other.pImageInfo != nullptr) || (mine.pBufferInfo != nullptr) != (other.pBufferInfo != nullptr)) return "descriptor writes";
    }
    const bool sharedSamplers = context.samplerCache != nullptr && !SamplerCacheDisabled();
    if (descriptorImages.size() != fresh.descriptorImages.size()) return "image descriptors";
    for (std::size_t index = 0; index < descriptorImages.size(); ++index) {
        const auto& mine = descriptorImages[index];
        const auto& other = fresh.descriptorImages[index];
        if (mine.imageView != other.imageView) return "image views";
        if (mine.imageLayout != other.imageLayout) return "image layouts";
        if (sharedSamplers && mine.sampler != other.sampler) return "samplers";
    }
    if (sharedSamplers) {
        if (samplers.size() != fresh.samplers.size()) return "samplers";
        for (std::size_t index = 0; index < samplers.size(); ++index) {
            const auto& mine = samplers[index];
            const auto& other = fresh.samplers[index];
            if ((mine == nullptr) != (other == nullptr) || (mine != nullptr && mine->Handle() != other->Handle())) return "samplers";
        }
    }
    if (allocations.size() != fresh.allocations.size()) return "bound ranges";
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& mine = allocations[index];
        const auto& other = fresh.allocations[index];
        if (mine.guest != other.guest) return "bound ranges";
        if (!mine.guest) continue;
        auto address = mine.address;
        auto size = mine.size;
        if (const auto entry = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& candidate) { return candidate.allocation == index; }); entry != moved.end()) {
            address = entry->address;
            size = entry->size;
        }
        if (address != other.address || size != other.size) return "bound ranges";
    }
    return nullptr;
}

std::string ShaderResources::TakeTemplateKeepText() {
    auto& census = templateKeepCensus();
    const auto take = [](std::atomic<std::uint64_t>& counter) { return counter.exchange(0, std::memory_order_relaxed); };
    const auto join = [](std::string& text, const std::string& item) {
        if (!text.empty()) text += ", ";
        text += item;
    };
    std::string released;
    for (std::size_t cause = 0; cause < census.released.size(); ++cause) {
        for (std::size_t kind = 0; kind < census.released[cause].size(); ++kind) {
            const auto count = take(census.released[cause][kind]);
            if (count != 0) join(released, std::string(ReleaseCauseNames[cause]) + " (" + ReleaseKindNames[kind] + ") " + std::to_string(count));
        }
    }
    std::string refilled;
    for (std::size_t cause = 0; cause < census.refilled.size(); ++cause) {
        if (const auto count = take(census.refilled[cause]); count != 0) join(refilled, std::string(ReleaseCauseNames[cause]) + " " + std::to_string(count));
    }
    std::string refused;
    for (std::size_t reason = 0; reason < census.refused.size(); ++reason) {
        if (const auto count = take(census.refused[reason]); count != 0) join(refused, std::string(RefillRefusalNames[reason]) + " " + std::to_string(count));
    }
    const auto verified = take(census.refillsVerified);
    const auto differing = take(census.refillsDiffering);
    std::string text = "textures released while held: " + (released.empty() ? std::string("none") : released);
    text += "; refilled in place: " + (refilled.empty() ? std::string("none") : refilled);
    text += "; refills refused: " + (refused.empty() ? std::string("none") : refused);
    if (verified != 0) text += "; refills verified " + std::to_string(verified) + ", differing " + std::to_string(differing);
    return text;
}

std::string ShaderResources::TakeViewTemplateKeepText() {
    auto& census = viewKeepCensus();
    const auto take = [](std::atomic<std::uint64_t>& counter) { return counter.exchange(0, std::memory_order_relaxed); };
    const auto list = [&](auto& counters, const auto& names, std::size_t first) {
        std::string text;
        for (std::size_t index = first; index < counters.size(); ++index) {
            const auto count = take(counters[index]);
            if (count == 0) continue;
            if (!text.empty()) text += ", ";
            text += std::string(names[index]) + " " + std::to_string(count);
        }
        return text.empty() ? std::string("none") : text;
    };
    std::string text = "parked: " + list(census.parked, KeepFormNames, 1u);
    text += "; taken back: " + list(census.revived, KeepFormNames, 1u);
    text += "; taken back by a refill (APS5_TEMPLATE_KEEPS): " + list(census.refilledBack, KeepFormNames, 1u);
    text += "; parkable while unset: " + list(census.parkable, KeepFormNames, 1u);
    text += "; held stale entries released as today: " + list(census.notParked, NotParkedNames, 0u);
    text += "; take-backs refused (left parked): " + list(census.refused, ReviveRefusalNames, 1u);
    text += "; parked textures released: " + list(census.dropped, ParkedDropNames, 0u);
    const auto notAdmitted = take(census.revivedNotAdmitted);
    const auto unwound = take(census.revivedUnwound);
    if (notAdmitted != 0 || unwound != 0) text += "; taken back then released: not admitted " + std::to_string(notAdmitted) + ", the lookup threw " + std::to_string(unwound);
    text += "; parked now " + std::to_string(census.parkedNow.load(std::memory_order_relaxed)) + " (" + std::to_string(census.parkedNowBytes.load(std::memory_order_relaxed) >> 20u) + " MiB)";
    const auto verified = take(census.verified);
    const auto differing = take(census.differing);
    if (verified != 0) text += "; take-backs verified " + std::to_string(verified) + ", differing " + std::to_string(differing);
    return text;
}

bool ShaderResources::ReadsImage(const StorageTexture* image) const {
    if (image == nullptr) return false;
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->StorageSource() == image) return true;
    }
    return std::any_of(storageTextures.begin(), storageTextures.end(), [&](const auto& storage) { return storage.get() == image; });
}

}
