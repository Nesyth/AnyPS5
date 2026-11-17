#include "prx/libSceAgcDriver/Graphics/include/MemoryAccounting.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

namespace {

thread_local std::uint64_t imageMemoryReleasedBytes = 0;

class MemoryAllocationError : public std::runtime_error {
public:
    MemoryAllocationError(VkResult result, const char* details) : std::runtime_error(std::string("AGC graphics: ") + details + ": Vulkan result " + std::to_string(result)), result(result) {}
    VkResult result;
};

// APS5_BUFFER_POOL_SHARED=1: one tier for every size, as before the split.
bool sharedTiers() {
    static const bool shared = std::getenv("APS5_BUFFER_POOL_SHARED") != nullptr;
    return shared;
}

bool exactSizes() {
    static const bool exact = std::getenv("APS5_NO_BUFFER_CLASSES") != nullptr;
    return exact;
}

bool systemMemoryHeap(const VkPhysicalDeviceMemoryProperties& memory, std::uint32_t heap) {
    bool typed = false;
    for (std::uint32_t type = 0; type < std::min<std::uint32_t>(memory.memoryTypeCount, VK_MAX_MEMORY_TYPES); ++type) {
        if (memory.memoryTypes[type].heapIndex != heap) continue;
        if ((memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) return false;
        typed = true;
    }
    return typed;
}

constexpr VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

std::uint64_t spreadValue(const char* name, std::uint64_t fallback, bool zeroAllowed) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') return fallback;
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || (value == 0 && !zeroAllowed)) return fallback;
    return value;
}

bool spreadStats() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    return stats;
}

void noteHigh(std::atomic<std::uint64_t>& value, std::uint64_t seen) noexcept {
    auto current = value.load(std::memory_order_relaxed);
    while (seen > current && !value.compare_exchange_weak(current, seen, std::memory_order_relaxed)) {
    }
}

struct OwnerBatch {
    std::vector<std::shared_ptr<const void>> owners;
    std::size_t next = 0;
    std::size_t bytes = 0;
};

struct SpreadState {
    std::atomic<void (*)()> wake{nullptr};
    std::atomic<std::uint64_t> allocations{0};
    std::atomic<std::uint64_t> allocationBytes{0};
    std::atomic<std::uint64_t> owners{0};
    std::mutex registryMutex;
    std::vector<BufferPool*> pools;
    std::size_t cursor = 0;
    std::recursive_mutex stepMutex;
    bool ownersTurn = false;
    std::mutex ownerMutex;
    std::deque<OwnerBatch> batches;
    std::size_t ownerBytes = 0;
    std::atomic<std::uint64_t> queued{0};
    std::atomic<std::uint64_t> queuedBytes{0};
    std::atomic<std::uint64_t> destroyed{0};
    std::atomic<std::uint64_t> destroyedBytes{0};
    std::atomic<std::uint64_t> portions{0};
    std::atomic<std::uint64_t> overBound{0};
    std::atomic<std::uint64_t> overBoundBytes{0};
    std::atomic<std::uint64_t> drained{0};
    std::atomic<std::uint64_t> drainedBytes{0};
    std::atomic<std::uint64_t> longestPortionUs{0};
    std::atomic<std::uint64_t> largestPending{0};
    std::atomic<std::uint64_t> largestPendingBytes{0};
    std::atomic<std::uint64_t> ownerBatches{0};
    std::atomic<std::uint64_t> ownersQueued{0};
    std::atomic<std::uint64_t> ownersReleased{0};
    std::atomic<std::uint64_t> ownerPortions{0};
    std::atomic<std::uint64_t> ownersOverBound{0};
    std::atomic<std::uint64_t> ownersDrained{0};
    std::mutex reportMutex;
    std::chrono::steady_clock::time_point reported{};
};

SpreadState& spreadState() {
    static auto& state = *new SpreadState();
    return state;
}

void wakeDrain(SpreadState& state) noexcept {
    if (const auto wake = state.wake.load(std::memory_order_acquire); wake != nullptr) wake();
}

std::size_t destroyAllocations(SpreadState& state, std::size_t count, std::chrono::steady_clock::time_point deadline) noexcept {
    std::lock_guard registry(state.registryMutex);
    const auto pools = state.pools.size();
    std::size_t done = 0;
    for (std::size_t visited = 0; visited < pools && done < count; ++visited) {
        done += state.pools[(state.cursor + visited) % pools]->DestroyPending(count - done, deadline);
        if (done != 0 && std::chrono::steady_clock::now() >= deadline) break;
    }
    if (pools != 0) state.cursor = (state.cursor + 1) % pools;
    return done;
}

std::size_t releaseOwners(SpreadState& state, std::size_t count, std::chrono::steady_clock::time_point deadline) noexcept {
    std::size_t done = 0;
    while (done < count) {
        std::shared_ptr<const void> owner;
        std::vector<std::shared_ptr<const void>> spent;
        {
            std::lock_guard lock(state.ownerMutex);
            if (state.batches.empty()) break;
            auto& batch = state.batches.front();
            owner = std::move(batch.owners[batch.next++]);
            state.owners.fetch_sub(1);
            if (batch.next == batch.owners.size()) {
                state.ownerBytes -= batch.bytes;
                spent = std::move(batch.owners);
                state.batches.pop_front();
            }
        }
        owner.reset();
        ++done;
        state.ownersReleased.fetch_add(1, std::memory_order_relaxed);
        if (std::chrono::steady_clock::now() >= deadline) break;
    }
    return done;
}

void reportSpread(SpreadState& state) noexcept {
    if (!spreadStats()) return;
    try {
        const auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard lock(state.reportMutex);
            if (state.reported == std::chrono::steady_clock::time_point{}) {
                state.reported = now;
                return;
            }
            if (now - state.reported < std::chrono::seconds(10)) return;
            state.reported = now;
        }
        const bool on = RuntimeSwitches::Set(RuntimeSwitch::SpreadDestruction);
        const auto counts = SpreadDestruction::TakeWindow();
        if (!on && counts.queued == 0 && counts.ownersQueued == 0) return;
        constexpr double MiBytes = 1048576.0;
        std::fprintf(stderr, "[spread-destruction] switch %s, drain thread %s; allocations: %llu queued (%.1f MiB), %llu destroyed in %llu portions (%.1f MiB), %llu destroyed by the caller over the bound (%.1f MiB), %llu destroyed by drains (%.1f MiB), %llu pending (%.1f MiB); owners: %llu handed to the reaper in %llu batches, %llu released in %llu portions, %llu released by the caller over the bound, %llu released by drains, %llu pending; this window: longest portion %llu us, largest pending list %llu allocations (%.1f MiB); portion %zu objects or %lld us, bound %llu MiB\n",
            on ? "on" : "off", state.wake.load(std::memory_order_acquire) != nullptr ? "on" : "off",
            static_cast<unsigned long long>(counts.queued), counts.queuedBytes / MiBytes, static_cast<unsigned long long>(counts.destroyed), static_cast<unsigned long long>(counts.portions), counts.destroyedBytes / MiBytes,
            static_cast<unsigned long long>(counts.overBound), counts.overBoundBytes / MiBytes, static_cast<unsigned long long>(counts.drained), counts.drainedBytes / MiBytes, static_cast<unsigned long long>(counts.pending), counts.pendingBytes / MiBytes,
            static_cast<unsigned long long>(counts.ownersQueued), static_cast<unsigned long long>(counts.ownerBatches), static_cast<unsigned long long>(counts.ownersReleased), static_cast<unsigned long long>(counts.ownerPortions),
            static_cast<unsigned long long>(counts.ownersOverBound), static_cast<unsigned long long>(counts.ownersDrained), static_cast<unsigned long long>(counts.ownersPending),
            static_cast<unsigned long long>(counts.longestPortionUs), static_cast<unsigned long long>(counts.largestPending), counts.largestPendingBytes / MiBytes,
            SpreadDestruction::PerStep(), static_cast<long long>(SpreadDestruction::Budget().count()), static_cast<unsigned long long>(SpreadDestruction::MaxBytes() >> 20u));
    } catch (...) {
    }
}

}

namespace SpreadDestruction {

bool Queueing() {
    return RuntimeSwitches::Set(RuntimeSwitch::SpreadDestruction) && spreadState().wake.load(std::memory_order_acquire) != nullptr;
}

void SetWake(void (*wake)()) {
    spreadState().wake.store(wake, std::memory_order_release);
}

bool Pending() {
    auto& state = spreadState();
    return state.allocations.load() != 0 || state.owners.load() != 0;
}

std::chrono::milliseconds Gap() {
    return std::chrono::milliseconds(1);
}

std::size_t PerStep() {
    static const auto count = static_cast<std::size_t>(spreadValue("APS5_SPREAD_DESTRUCTION_PER_STEP", 64, false));
    return count;
}

std::chrono::microseconds Budget() {
    static const auto budget = std::chrono::microseconds(static_cast<std::int64_t>(std::min<std::uint64_t>(spreadValue("APS5_SPREAD_DESTRUCTION_US", 300, true), 1000000)));
    return budget;
}

VkDeviceSize MaxBytes() {
    static const VkDeviceSize bytes = [] {
        const auto mib = spreadValue("APS5_SPREAD_DESTRUCTION_MAX_MIB", 512, true);
        return mib > (std::numeric_limits<VkDeviceSize>::max() >> 20u) ? std::numeric_limits<VkDeviceSize>::max() : static_cast<VkDeviceSize>(mib) << 20u;
    }();
    return bytes;
}

std::size_t Step() noexcept {
    return Step(PerStep(), Budget());
}

std::size_t Step(std::size_t count, std::chrono::microseconds budget) noexcept {
    auto& state = spreadState();
    reportSpread(state);
    if (count == 0 || !Pending()) return 0;
    std::lock_guard step(state.stepMutex);
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + budget;
    const bool ownersWaiting = state.owners.load() != 0;
    const bool allocationsWaiting = state.allocations.load() != 0;
    const bool owners = ownersWaiting && (!allocationsWaiting || state.ownersTurn);
    state.ownersTurn = !owners;
    const auto done = owners ? releaseOwners(state, count, deadline) : destroyAllocations(state, count, deadline);
    if (done == 0) return 0;
    (owners ? state.ownerPortions : state.portions).fetch_add(1, std::memory_order_relaxed);
    noteHigh(state.longestPortionUs, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count()));
    return done;
}

void DrainAll() noexcept {
    auto& state = spreadState();
    std::lock_guard step(state.stepMutex);
    for (;;) {
        std::vector<std::shared_ptr<const void>> owners;
        std::size_t left = 0;
        {
            std::lock_guard lock(state.ownerMutex);
            if (state.batches.empty()) break;
            auto& batch = state.batches.front();
            left = batch.owners.size() - batch.next;
            owners = std::move(batch.owners);
            state.ownerBytes -= batch.bytes;
            state.owners.fetch_sub(left);
            state.batches.pop_front();
        }
        state.ownersDrained.fetch_add(left, std::memory_order_relaxed);
        owners.clear();
    }
    std::lock_guard registry(state.registryMutex);
    for (auto* pool : state.pools) pool->DrainPending();
}

bool Release(std::vector<std::shared_ptr<const void>>& owners, std::size_t bytes) noexcept {
    if (owners.empty()) return true;
    if (!Queueing()) return false;
    auto& state = spreadState();
    const auto count = owners.size();
    const auto bound = MaxBytes();
    bool queued = false;
    std::uint64_t before = 0;
    try {
        std::lock_guard lock(state.ownerMutex);
        if (bytes <= bound && state.ownerBytes <= bound - bytes) {
            state.batches.emplace_back();
            auto& batch = state.batches.back();
            batch.owners = std::move(owners);
            batch.bytes = bytes;
            state.ownerBytes += bytes;
            before = state.owners.fetch_add(count);
            queued = true;
        }
    } catch (...) {
    }
    if (!queued) {
        state.ownersOverBound.fetch_add(count, std::memory_order_relaxed);
        return false;
    }
    owners.clear();
    state.ownerBatches.fetch_add(1, std::memory_order_relaxed);
    state.ownersQueued.fetch_add(count, std::memory_order_relaxed);
    if (before == 0) wakeDrain(state);
    return true;
}

SpreadDestructionCounts Counts() {
    auto& state = spreadState();
    SpreadDestructionCounts counts;
    counts.queued = state.queued.load(std::memory_order_relaxed);
    counts.queuedBytes = state.queuedBytes.load(std::memory_order_relaxed);
    counts.destroyed = state.destroyed.load(std::memory_order_relaxed);
    counts.destroyedBytes = state.destroyedBytes.load(std::memory_order_relaxed);
    counts.portions = state.portions.load(std::memory_order_relaxed);
    counts.overBound = state.overBound.load(std::memory_order_relaxed);
    counts.overBoundBytes = state.overBoundBytes.load(std::memory_order_relaxed);
    counts.drained = state.drained.load(std::memory_order_relaxed);
    counts.drainedBytes = state.drainedBytes.load(std::memory_order_relaxed);
    counts.pending = state.allocations.load();
    counts.pendingBytes = state.allocationBytes.load();
    counts.longestPortionUs = state.longestPortionUs.load(std::memory_order_relaxed);
    counts.largestPending = state.largestPending.load(std::memory_order_relaxed);
    counts.largestPendingBytes = state.largestPendingBytes.load(std::memory_order_relaxed);
    counts.ownerBatches = state.ownerBatches.load(std::memory_order_relaxed);
    counts.ownersQueued = state.ownersQueued.load(std::memory_order_relaxed);
    counts.ownersReleased = state.ownersReleased.load(std::memory_order_relaxed);
    counts.ownerPortions = state.ownerPortions.load(std::memory_order_relaxed);
    counts.ownersOverBound = state.ownersOverBound.load(std::memory_order_relaxed);
    counts.ownersDrained = state.ownersDrained.load(std::memory_order_relaxed);
    counts.ownersPending = state.owners.load();
    return counts;
}

SpreadDestructionCounts TakeWindow() {
    auto& state = spreadState();
    auto counts = Counts();
    counts.longestPortionUs = state.longestPortionUs.exchange(0, std::memory_order_relaxed);
    counts.largestPending = state.largestPending.exchange(0, std::memory_order_relaxed);
    counts.largestPendingBytes = state.largestPendingBytes.exchange(0, std::memory_order_relaxed);
    return counts;
}

}

BufferPool::BufferPool(const Context& context) : maxSlots(std::max<std::size_t>(1, std::min<std::size_t>(MaxSlots(), context.limits.maxMemoryAllocationCount / 8u))), cacheReclaim(context.cacheReclaim), device(context.device), unmap(context.Resolved(&DeviceFunctions::unmapMemory, "vkUnmapMemory")), destroyBuffer(context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")), freeMemory(context.Resolved(&DeviceFunctions::freeMemory, "vkFreeMemory")), allocateMemory(context.Resolved(&DeviceFunctions::allocateMemory, "vkAllocateMemory")), mapMemory(context.Resolved(&DeviceFunctions::mapMemory, "vkMapMemory")), physical(context.physical), memoryProperties(context.memory), memoryProperties2(context.memoryProperties2) {
    smallTier.budget = smallBudget;
    largeTier.budget = budget;
    deviceTier.budget = deviceBudget(context);
    smallDeviceTier.budget = std::max<VkDeviceSize>(smallBudget, deviceTier.budget / 8u);
    memoryProperties = context.memory;
    physical = context.physical;
    memoryProperties2 = context.memoryProperties2;
    imageBudget = ImageBudget(deviceTier.budget);
    pendingBound = SpreadDestruction::MaxBytes();
    auto& state = spreadState();
    std::lock_guard registry(state.registryMutex);
    state.pools.push_back(this);
}

BufferPool::~BufferPool() {
    {
        auto& state = spreadState();
        std::lock_guard registry(state.registryMutex);
        std::erase(state.pools, this);
        state.cursor = 0;
    }
    DrainPending();
    for (auto* tier : {&smallTier, &largeTier, &deviceTier, &smallDeviceTier}) {
        for (const auto& [key, slots] : tier->free) {
            for (const auto& slot : slots) destroy(slot.allocation, true);
        }
    }
    for (auto& [memory, block] : slabBlocks) freeBlock(*block);
    for (const auto& [memory, block] : imageBlocks) MemoryAccounting::Free(freeMemory, device, memory);
}

VkDeviceMemory BufferPool::AllocateMemory(const VkMemoryAllocateInfo& allocation, const char* operation, std::optional<std::uint32_t> hostType, std::uint32_t* allocatedType, bool newImage) {
    return allocateMemoryWithReuse(allocation, operation, hostType, allocatedType, newImage, {});
}

VkDeviceMemory BufferPool::allocateMemoryWithReuse(const VkMemoryAllocateInfo& allocation, const char* operation, std::optional<std::uint32_t> hostType, std::uint32_t* allocatedType, bool newImage, const std::function<bool(std::uint32_t)>& reuse) {
    if (allocatedType != nullptr) *allocatedType = allocation.memoryTypeIndex;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    auto result = allocateMemory(device, &allocation, nullptr, &memory);
    const auto initialResult = result;
    unsigned attempts = 1;
    VkDeviceSize reclaimed = 0;
    VkDeviceSize released = 0;
    const auto exhausted = [](VkResult value) { return value == VK_ERROR_OUT_OF_DEVICE_MEMORY || value == VK_ERROR_OUT_OF_HOST_MEMORY || value == VK_ERROR_TOO_MANY_OBJECTS; };
    if (exhausted(result)) {
        if (reuse && reuse(allocation.memoryTypeIndex)) return VK_NULL_HANDLE;
        reclaimed = trim();
        if (reuse && reuse(allocation.memoryTypeIndex)) return VK_NULL_HANDLE;
        if (reclaimed != 0) {
            memory = VK_NULL_HANDLE;
            result = allocateMemory(device, &allocation, nullptr, &memory);
            ++attempts;
        }
    }
    if (exhausted(result)) {
        if (cacheReclaim != nullptr) released = cacheReclaim(ReclaimTarget(allocation.allocationSize), newImage);
        if (reuse && reuse(allocation.memoryTypeIndex)) return VK_NULL_HANDLE;
        if (released != 0) {
            reclaimed += trim();
            memory = VK_NULL_HANDLE;
            result = allocateMemory(device, &allocation, nullptr, &memory);
            ++attempts;
        }
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && hostType.has_value() && *hostType != allocation.memoryTypeIndex) {
        if (reuse && reuse(*hostType)) return VK_NULL_HANDLE;
        auto host = allocation;
        host.memoryTypeIndex = *hostType;
        memory = VK_NULL_HANDLE;
        result = allocateMemory(device, &host, nullptr, &memory);
        ++attempts;
        if (result == VK_SUCCESS) {
            if (allocatedType != nullptr) *allocatedType = *hostType;
            MemoryAccounting::Record(device, memory, host, memoryProperties, operation);
            static std::atomic<std::uint64_t> placements{0}, placedBytes{0};
            const auto count = placements.fetch_add(1, std::memory_order_relaxed) + 1;
            const auto total = placedBytes.fetch_add(allocation.allocationSize, std::memory_order_relaxed) + allocation.allocationSize;
            if (count <= 5 || (count & (count - 1)) == 0) std::fprintf(stderr, "[bufferpool] %s: memory type %u is out of device memory for %llu bytes after reclaiming %llu and releasing %llu; placed in host memory type %u until it is released; %llu such allocations, %llu MiB\n", operation, allocation.memoryTypeIndex, static_cast<unsigned long long>(allocation.allocationSize), static_cast<unsigned long long>(reclaimed), static_cast<unsigned long long>(released), *hostType, static_cast<unsigned long long>(count), static_cast<unsigned long long>(total >> 20u));
            return memory;
        }
    }
    if (result == VK_SUCCESS) {
        MemoryAccounting::Record(device, memory, allocation, memoryProperties, operation);
        if (attempts > 1) std::fprintf(stderr, "[bufferpool] %s recovered: allocation=%llu reclaimed=%llu released=%llu initialResult=%d attempts=%u\n", operation, static_cast<unsigned long long>(allocation.allocationSize), static_cast<unsigned long long>(reclaimed), static_cast<unsigned long long>(released), static_cast<int>(initialResult), attempts);
        return memory;
    }
    char details[1536];
    std::snprintf(details, sizeof(details), "%s: allocation=%llu memoryType=%u reclaimed=%llu released=%llu initialResult=%d attempts=%u", operation, static_cast<unsigned long long>(allocation.allocationSize), allocation.memoryTypeIndex, static_cast<unsigned long long>(reclaimed), static_cast<unsigned long long>(released), static_cast<int>(initialResult), attempts);
    if (memoryProperties2 != nullptr && physical != VK_NULL_HANDLE) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &budget};
        memoryProperties2(physical, &properties);
        auto used = std::strlen(details);
        for (std::uint32_t heap = 0; heap < memoryProperties.memoryHeapCount && heap < VK_MAX_MEMORY_HEAPS; ++heap) {
            const auto remaining = sizeof(details) - used;
            const auto appended = std::snprintf(details + used, remaining, "; heap %u %llu/%llu MiB", heap, static_cast<unsigned long long>(budget.heapUsage[heap] >> 20u), static_cast<unsigned long long>(budget.heapBudget[heap] >> 20u));
            if (appended < 0 || static_cast<std::size_t>(appended) >= remaining) break;
            used += static_cast<std::size_t>(appended);
        }
    }
    if (result == VK_ERROR_DEVICE_LOST) ReportDeviceLost();
    throw MemoryAllocationError(result, details);
}

VkDeviceSize BufferPool::ImageBudget(VkDeviceSize deviceBudget) {
    static const char* value = std::getenv("APS5_IMAGE_POOL_MIB");
    if (value != nullptr) return static_cast<VkDeviceSize>(std::strtoull(value, nullptr, 10)) << 20u;
    return deviceBudget / 4u;
}

bool BufferPool::imagePooled(VkDeviceSize bytes, std::uint32_t memoryType) const {
    return imageBudget != 0 && bytes != 0 && bytes <= imageBudget && bytes <= deviceTier.budget && memoryType < memoryProperties.memoryTypeCount && DeviceTiered(memoryProperties.memoryTypes[memoryType].propertyFlags);
}

bool BufferPool::heapOverBudget() {
    if (memoryProperties2 == nullptr || physical == VK_NULL_HANDLE) return false;
    const auto now = std::chrono::steady_clock::now();
    if (heapRead != std::chrono::steady_clock::time_point{} && now - heapRead < std::chrono::milliseconds(100)) return heapOver;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    memoryProperties2(physical, &properties);
    heapRead = now;
    heapOver = false;
    for (std::uint32_t heap = 0; heap < memoryProperties.memoryHeapCount && heap < VK_MAX_MEMORY_HEAPS; ++heap) {
        if ((memoryProperties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 && reported.heapBudget[heap] != 0 && reported.heapUsage[heap] > reported.heapBudget[heap]) heapOver = true;
    }
    return heapOver;
}

VkDeviceMemory BufferPool::TakeImageMemory(const VkMemoryAllocateInfo& allocation, const char* operation, std::optional<std::uint32_t> hostType, std::uint32_t* allocatedType, bool newImage) {
    const auto type = allocation.memoryTypeIndex;
    if (imagePooled(allocation.allocationSize, type)) {
        std::lock_guard lock(mutex);
        ++clock;
        const auto found = deviceTier.free.find({static_cast<std::size_t>(allocation.allocationSize), 0, memoryProperties.memoryTypes[type].propertyFlags, type});
        if (found != deviceTier.free.end() && !found->second.empty()) {
            auto& slots = found->second;
            const auto memory = slots.back().allocation.memory;
            imageOrder.erase(slots.back().order);
            slots.pop_back();
            if (slots.empty()) deviceTier.free.erase(found);
            --deviceTier.slots;
            deviceTier.retainedBytes -= allocation.allocationSize;
            --imageSlots;
            imageBytes -= allocation.allocationSize;
            ++imageHits;
            MemoryAccounting::Use(device, memory, operation);
            if (allocatedType != nullptr) *allocatedType = type;
            return memory;
        }
        ++imageMisses;
    }
    return AllocateMemory(allocation, operation, hostType, allocatedType, newImage);
}

void BufferPool::PutImageMemory(VkDeviceMemory memory, VkDeviceSize bytes, std::uint32_t memoryType) noexcept {
    if (memory == VK_NULL_HANDLE) return;
    imageMemoryReleasedBytes += bytes;
    if (!imagePooled(bytes, memoryType)) {
        MemoryAccounting::Free(freeMemory, device, memory);
        return;
    }
    BufferAllocation allocation{VK_NULL_HANDLE, memory, nullptr, 0, bytes, static_cast<std::size_t>(bytes), 0, memoryProperties.memoryTypes[memoryType].propertyFlags};
    allocation.memoryType = memoryType;
    std::vector<BufferAllocation> evicted;
    bool urgent = false;
    try {
        std::lock_guard lock(mutex);
        if (heapOverBudget() || bytes > imageBudget - std::min(imageBudget, retainedImageSlabBytes.load(std::memory_order_relaxed))) {
            urgent = true;
            while (imageSlots != 0 && evictOldestImage(evicted)) {
            }
            evicted.push_back(allocation);
        } else {
            while (imageSlots != 0 && imageBytes + bytes + retainedImageSlabBytes.load(std::memory_order_relaxed) > imageBudget && evictOldestImage(evicted)) {
            }
            while (deviceTier.slots != 0 && (deviceTier.retainedBytes + bytes > deviceTier.budget || deviceTier.slots >= maxSlots)) evictOldest(deviceTier, evicted);
            const SlotKey imageKey{allocation.bytes, 0, allocation.properties, memoryType};
            auto& slots = deviceTier.free[imageKey];
            slots.push_back({allocation, ++clock, {}});
            try {
                slots.back().order = imageOrder.insert(imageOrder.end(), &slots.back());
            } catch (...) {
                slots.pop_back();
                if (slots.empty()) deviceTier.free.erase(imageKey);
                throw;
            }
            ++deviceTier.slots;
            deviceTier.retainedBytes += bytes;
            ++imageSlots;
            imageBytes += bytes;
            MemoryAccounting::Retain(device, memory, bytes);
        }
    } catch (...) {
        destroy(allocation);
    }
    for (const auto& gone : evicted) {
        if (urgent) destroy(gone, true);
        else retire(gone, true);
    }
}

std::size_t BufferPool::RetainedImages() {
    std::lock_guard lock(mutex);
    return imageSlots;
}

VkDeviceSize BufferPool::ReclaimTarget(VkDeviceSize bytes) {
    const auto reserve = std::max<VkDeviceSize>(bytes, VkDeviceSize{256} << 20u);
    const auto limit = std::numeric_limits<VkDeviceSize>::max();
    return bytes > limit - reserve ? limit : bytes + reserve;
}

bool BufferPool::SlabEligible(std::size_t capacity, VkDeviceSize alignment, VkDeviceSize size, VkDeviceSize atom) {
    if (capacity >= classLimit || size > capacity || !std::has_single_bit(capacity)) return false;
    const auto alignmentOk = alignment == 0 || (std::has_single_bit(alignment) && alignment <= capacity);
    const auto atomOk = atom <= 1 || (std::has_single_bit(atom) && atom <= capacity);
    return alignmentOk && atomOk;
}

VkDeviceSize BufferPool::SlabBlockBytes(std::size_t capacity) {
    return std::max<VkDeviceSize>(VkDeviceSize{2} << 20u, static_cast<VkDeviceSize>(capacity) * 8u);
}

std::uint64_t BufferPool::SlabKey(std::uint32_t memoryType, std::size_t capacity, bool addressable) {
    return (static_cast<std::uint64_t>(memoryType) << 40u) | (static_cast<std::uint64_t>(addressable ? 1u : 0u) << 39u) | static_cast<std::uint64_t>(capacity);
}

SlabSlot BufferPool::TakeSlot(const Context& context, std::uint32_t memoryType, std::size_t capacity, bool addressable) {
    const auto key = SlabKey(memoryType, capacity, addressable);
    const auto takeFrom = [&](SlabBlock& block, Slab& slab) {
        const auto index = block.free.back();
        block.free.pop_back();
        if (block.used++ == 0) --slab.emptyBlocks;
        if (block.free.empty()) slab.available.erase(std::find(slab.available.begin(), slab.available.end(), &block));
        const auto offset = static_cast<VkDeviceSize>(index) * block.slotBytes;
        MemoryAccounting::RetainPart(device, block.memory, block.slotBytes, false);
        return SlabSlot{block.memory, offset, block.mapping != nullptr ? block.mapping + offset : nullptr};
    };
    {
        std::lock_guard lock(slabMutex);
        auto& slab = slabs[key];
        if (!slab.available.empty()) return takeFrom(*slab.available.back(), slab);
    }
    const auto blockBytes = SlabBlockBytes(capacity);
    const auto release = [this](SlabBlock* block) noexcept {
        if (block->memory != VK_NULL_HANDLE) freeBlock(*block);
        delete block;
    };
    std::unique_ptr<SlabBlock, decltype(release)> block(new SlabBlock, release);
    block->slotBytes = capacity;
    block->slab = key;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    if (addressable) allocation.pNext = &flags;
    allocation.allocationSize = blockBytes;
    allocation.memoryTypeIndex = memoryType;
    block->memory = AllocateMemory(allocation, "vkAllocateMemory buffer slab");
    if ((context.memory.memoryTypes[memoryType].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        void* mapping = nullptr;
        const auto mapped = mapMemory(device, block->memory, 0, VK_WHOLE_SIZE, 0, &mapping);
        Check(mapped, "vkMapMemory buffer slab");
        block->mapping = static_cast<std::byte*>(mapping);
    }
    const auto slots = static_cast<std::uint32_t>(blockBytes / capacity);
    block->free.reserve(slots);
    for (std::uint32_t index = slots; index-- != 0;) block->free.push_back(index);
    MemoryAccounting::Retain(device, block->memory, blockBytes);
    std::lock_guard lock(slabMutex);
    auto& slab = slabs[key];
    if (slab.blocks == slab.available.capacity()) slab.available.reserve(std::max<std::size_t>(1, slab.available.capacity() * 2));
    auto& inserted = *block;
    const auto owner = slabBlocks.try_emplace(inserted.memory).first;
    owner->second.reset(block.release());
    slab.available.push_back(&inserted);
    ++slab.blocks;
    ++slab.emptyBlocks;
    return takeFrom(*slab.available.back(), slab);
}

void BufferPool::freeBlock(SlabBlock& block) noexcept {
    if (block.mapping != nullptr) unmap(device, block.memory);
    MemoryAccounting::Free(freeMemory, device, block.memory);
}

VkDeviceSize BufferPool::PutSlot(VkDeviceMemory memory, VkDeviceSize offset) noexcept {
    std::unique_ptr<SlabBlock> released;
    {
        std::lock_guard lock(slabMutex);
        const auto found = slabBlocks.find(memory);
        if (found == slabBlocks.end()) return 0;
        auto& block = *found->second;
        auto& slab = slabs[block.slab];
        if (block.free.empty()) slab.available.push_back(&block);
        block.free.push_back(static_cast<std::uint32_t>(offset / block.slotBytes));
        MemoryAccounting::RetainPart(device, memory, block.slotBytes, true);
        if (--block.used != 0) return 0;
        if (slab.emptyBlocks == 0) {
            ++slab.emptyBlocks;
            return 0;
        }
        slab.available.erase(std::find(slab.available.begin(), slab.available.end(), &block));
        --slab.blocks;
        released = std::move(found->second);
        slabBlocks.erase(found);
    }
    freeBlock(*released);
    return SlabBlockBytes(released->slotBytes);
}

std::size_t BufferPool::SlabBlocks() {
    std::lock_guard lock(slabMutex);
    return slabBlocks.size();
}

bool BufferPool::DeviceTierEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_STAGING_POOL_MIB");
        return value == nullptr || std::strtoull(value, nullptr, 10) != 0;
    }();
    return enabled;
}

VkDeviceSize BufferPool::DeviceBudget(const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceMemoryBudgetPropertiesEXT* reported, bool integratedGpu) {
    static const char* value = std::getenv("APS5_STAGING_POOL_MIB");
    if (value != nullptr) return static_cast<VkDeviceSize>(std::strtoull(value, nullptr, 10)) << 20u;
    std::optional<std::uint32_t> largest;
    for (std::uint32_t heap = 0; heap < memory.memoryHeapCount && heap < VK_MAX_MEMORY_HEAPS; ++heap) {
        if ((memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 && (!largest || memory.memoryHeaps[heap].size > memory.memoryHeaps[*largest].size)) largest = heap;
    }
    VkDeviceSize size = largest ? memory.memoryHeaps[*largest].size : 0;
    if (largest && reported != nullptr && reported->heapBudget[*largest] != 0) size = std::min(size, reported->heapBudget[*largest]);
    if (largest && (integratedGpu || systemMemoryHeap(memory, *largest))) return std::max<VkDeviceSize>(VkDeviceSize{128} << 20u, size / 64u);
    return std::max<VkDeviceSize>(VkDeviceSize{512} << 20u, size / 8u);
}

VkDeviceSize BufferPool::deviceBudget(const Context& context) {
    if (!DeviceTierEnabled()) return 0;
    if (context.memoryProperties2 == nullptr || context.physical == VK_NULL_HANDLE) return DeviceBudget(context.memory, nullptr, context.integratedGpu);
    VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    properties.pNext = &reported;
    context.memoryProperties2(context.physical, &properties);
    return DeviceBudget(context.memory, &reported, context.integratedGpu);
}

VkDeviceSize BufferPool::destroy(const BufferAllocation& allocation, bool retained) noexcept {
    if (allocation.slab) {
        if (retained) MemoryAccounting::RetainPart(device, allocation.memory, allocation.allocationBytes, false);
        destroyBuffer(device, allocation.buffer, nullptr);
        return PutSlot(allocation.memory, allocation.offset);
    }
    // Device-local allocations (see DeviceBuffer) are never mapped.
    if (allocation.mapping != nullptr) unmap(device, allocation.memory);
    if (allocation.buffer != VK_NULL_HANDLE) destroyBuffer(device, allocation.buffer, nullptr);
    MemoryAccounting::Free(freeMemory, device, allocation.memory);
    return allocation.allocationBytes;
}

bool BufferPool::DeviceTiered(VkMemoryPropertyFlags properties) {
    return (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 && (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0 && DeviceTierEnabled();
}

std::size_t BufferPool::Capacity(std::size_t bytes) {
    constexpr std::size_t smallest = 256;
    if (exactSizes()) return bytes;
    if (bytes < classLimit) return std::max(smallest, std::bit_ceil(bytes));
    const auto step = std::bit_floor(bytes) >> deviceClassBits;
    return (bytes + step - 1) / step * step;
}

std::size_t BufferPool::NextCapacity(std::size_t capacity) {
    return capacity < classLimit ? capacity * 2 : capacity + (std::bit_floor(capacity) >> deviceClassBits);
}

namespace {

std::atomic<bool> forceAddresses{false};

}

bool BufferPool::ForceAddresses() {
    return forceAddresses.load(std::memory_order_relaxed);
}

void BufferPool::SetForceAddresses(bool force) {
    forceAddresses.store(force, std::memory_order_relaxed);
}

VkBufferUsageFlags BufferPool::Usage(VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    if (ForceAddresses() && (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    return DeviceTiered(properties) ? usage | deviceUsage : usage;
}

BufferPool::Tier& BufferPool::tierFor(std::size_t capacity, VkMemoryPropertyFlags properties) {
    if (DeviceTiered(properties)) return sharedTiers() || capacity >= classLimit ? deviceTier : smallDeviceTier;
    return !sharedTiers() && capacity < classLimit ? smallTier : largeTier;
}

std::optional<BufferAllocation> BufferPool::Take(std::size_t bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, const char* category, bool hostPlaced) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    const auto capacity = Capacity(bytes);
    std::lock_guard lock(mutex);
    ++clock;
    if (profile) {
        static auto lastReport = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (now - lastReport > std::chrono::seconds(10)) {
            lastReport = now;
            AgcDriver::ProfilePrint_nid_no_patch("[bufferpool] small: %llu hits, %llu misses, %llu evictions, %zu retained (%.0f MiB); large: %llu hits, %llu misses, %llu evictions, %zu retained (%.0f MiB)%s; device: %llu hits, %llu misses, %llu evictions, %zu retained (%.0f MiB of %llu); small device: %llu hits, %llu misses, %llu evictions, %zu retained (%.0f MiB of %llu); images: %llu hits, %llu misses, %llu evictions, %zu retained (%.0f MiB of %llu)\n", static_cast<unsigned long long>(smallTier.hits), static_cast<unsigned long long>(smallTier.misses), static_cast<unsigned long long>(smallTier.evictions), smallTier.slots, smallTier.retainedBytes / 1048576.0, static_cast<unsigned long long>(largeTier.hits), static_cast<unsigned long long>(largeTier.misses), static_cast<unsigned long long>(largeTier.evictions), largeTier.slots, largeTier.retainedBytes / 1048576.0, sharedTiers() ? " (shared)" : "", static_cast<unsigned long long>(deviceTier.hits), static_cast<unsigned long long>(deviceTier.misses), static_cast<unsigned long long>(deviceTier.evictions), deviceTier.slots, deviceTier.retainedBytes / 1048576.0, static_cast<unsigned long long>(deviceTier.budget >> 20u), static_cast<unsigned long long>(smallDeviceTier.hits), static_cast<unsigned long long>(smallDeviceTier.misses), static_cast<unsigned long long>(smallDeviceTier.evictions), smallDeviceTier.slots, smallDeviceTier.retainedBytes / 1048576.0, static_cast<unsigned long long>(smallDeviceTier.budget >> 20u), static_cast<unsigned long long>(imageHits), static_cast<unsigned long long>(imageMisses), static_cast<unsigned long long>(imageEvictions), imageSlots, imageBytes / 1048576.0, static_cast<unsigned long long>(imageBudget >> 20u));
        }
    }
    auto& tier = tierFor(capacity, properties);
    const auto largest = (&tier == &deviceTier || &tier == &smallDeviceTier) && !exactSizes() && bytes <= std::numeric_limits<std::size_t>::max() / 2 ? std::max(capacity, bytes * 2) : capacity;
    auto found = tier.free.end();
    for (auto size = capacity; size <= largest && found == tier.free.end(); size = NextCapacity(size)) found = tier.free.find({size, usage, properties, 0, hostPlaced});
    if (found == tier.free.end() || found->second.empty()) {
        if (!hostPlaced) ++tier.misses;
        return std::nullopt;
    }
    auto& slots = found->second;
    const auto newest = std::prev(slots.end());
    auto chosen = newest;
    for (std::size_t scanned = 1; chosen->allocation.bufferBytes != bytes && scanned < exactSizeScan && chosen != slots.begin(); ++scanned) --chosen;
    if (chosen->allocation.bufferBytes != bytes) chosen = newest;
    auto result = chosen->allocation;
    slots.erase(chosen);
    if (slots.empty()) tier.free.erase(found);
    --tier.slots;
    tier.retainedBytes -= result.allocationBytes;
    ++tier.hits;
    if (result.slab) MemoryAccounting::RetainPart(device, result.memory, result.allocationBytes, false);
    else MemoryAccounting::Use(device, result.memory, category);
    return result;
}

void BufferPool::evictOldest(Tier& tier, std::vector<BufferAllocation>& evicted) {
    const auto oldest = std::min_element(tier.free.begin(), tier.free.end(), [](const auto& left, const auto& right) { return left.second.front().lastUse < right.second.front().lastUse; });
    // First: a throw here leaves the slot retained and counted.
    evicted.push_back(oldest->second.front().allocation);
    tier.retainedBytes -= oldest->second.front().allocation.allocationBytes;
    if (oldest->second.front().allocation.buffer == VK_NULL_HANDLE) {
        imageOrder.erase(oldest->second.front().order);
        imageBytes -= oldest->second.front().allocation.allocationBytes;
        --imageSlots;
        ++imageEvictions;
    }
    oldest->second.pop_front();
    if (oldest->second.empty()) tier.free.erase(oldest);
    --tier.slots;
    ++tier.evictions;
}

bool BufferPool::evictOldestImage(std::vector<BufferAllocation>& evicted) {
    if (imageOrder.empty()) return false;
    const Slot* oldest = imageOrder.front();
    const SlotKey key{oldest->allocation.bytes, 0, oldest->allocation.properties, oldest->allocation.memoryType};
    const auto found = deviceTier.free.find(key);
    evicted.push_back(oldest->allocation);
    imageOrder.pop_front();
    const auto bytes = oldest->allocation.allocationBytes;
    deviceTier.retainedBytes -= bytes;
    imageBytes -= bytes;
    found->second.pop_front();
    if (found->second.empty()) deviceTier.free.erase(found);
    --deviceTier.slots;
    --imageSlots;
    ++deviceTier.evictions;
    ++imageEvictions;
    return true;
}

VkDeviceSize BufferPool::ReclaimIdleDevice(VkDeviceSize bytes, std::chrono::steady_clock::time_point now) {
    if (bytes == 0) return 0;
    const auto heaps = std::min<std::uint32_t>(memoryProperties.memoryHeapCount, VK_MAX_MEMORY_HEAPS);
    if (std::count_if(memoryProperties.memoryHeaps, memoryProperties.memoryHeaps + heaps, [](const auto& heap) { return (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0; }) != 1) return 0;
    std::vector<BufferAllocation> evicted;
    {
        std::lock_guard lock(mutex);
        if (!idleRead) {
            idleRead = now;
            idleClock = clock;
            return 0;
        }
        if (now < *idleRead) return 0;
        if (now - *idleRead >= std::chrono::seconds(1)) {
            reclaimBefore = idleClock;
            idleClock = clock;
            idleRead = now;
        }
        if (reclaimBefore <= scannedIdleClock) return 0;
        VkDeviceSize selected = 0;
        while (selected < bytes) {
            auto oldest = deviceTier.free.end();
            for (auto it = deviceTier.free.begin(); it != deviceTier.free.end(); ++it) {
                if (it->second.empty()) continue;
                const auto& slot = it->second.front();
                if (slot.allocation.buffer == VK_NULL_HANDLE || slot.allocation.slab || slot.allocation.hostPlaced || slot.lastUse > reclaimBefore) continue;
                if (oldest == deviceTier.free.end() || slot.lastUse < oldest->second.front().lastUse) oldest = it;
            }
            if (oldest == deviceTier.free.end()) {
                scannedIdleClock = reclaimBefore;
                break;
            }
            if (evicted.empty()) evicted.reserve(deviceTier.slots);
            const auto allocation = oldest->second.front().allocation;
            evicted.push_back(allocation);
            selected += allocation.allocationBytes;
            deviceTier.retainedBytes -= allocation.allocationBytes;
            oldest->second.pop_front();
            if (oldest->second.empty()) deviceTier.free.erase(oldest);
            --deviceTier.slots;
            ++deviceTier.evictions;
        }
    }
    VkDeviceSize released = 0;
    for (const auto& allocation : evicted) released += destroy(allocation);
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (profile && released != 0) AgcDriver::ProfilePrint_nid_no_patch("[bufferpool] idle device reclaim: %llu bytes released for %llu bytes of storage-cache pressure\n", static_cast<unsigned long long>(released), static_cast<unsigned long long>(bytes));
    return released;
}

std::size_t BufferPool::MaxSlots() {
    // APS5_BUFFER_POOL_SLOTS=<n> bounds the retained allocations; 64 is the capacity the pool had
    // before least-recently-used retention, for A/B runs.
    static const std::size_t slots = [] {
        const char* value = std::getenv("APS5_BUFFER_POOL_SLOTS");
        const auto parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
        return parsed != 0 ? static_cast<std::size_t>(parsed) : defaultSlots;
    }();
    return slots;
}

VkDeviceSize BufferPool::trim() {
    Tier retiredSmall;
    Tier retiredLarge;
    Tier retiredDevice;
    Tier retiredSmallDevice;
    {
        std::lock_guard lock(mutex);
        for (const auto& [tier, retired] : {std::pair{&smallTier, &retiredSmall}, std::pair{&largeTier, &retiredLarge}, std::pair{&deviceTier, &retiredDevice}, std::pair{&smallDeviceTier, &retiredSmallDevice}}) {
            tier->free.swap(retired->free);
            tier->evictions += tier->slots;
            tier->retainedBytes = 0;
            tier->slots = 0;
        }
        imageEvictions += imageSlots;
        imageBytes = 0;
        imageSlots = 0;
        imageOrder.clear();
    }
    VkDeviceSize bytes = DrainPending();
    for (const auto* tier : {&retiredSmall, &retiredLarge, &retiredDevice, &retiredSmallDevice}) {
        for (const auto& [key, slots] : tier->free) {
            for (const auto& slot : slots) bytes += destroy(slot.allocation, true);
        }
    }
    for (;;) {
        std::unique_ptr<SlabBlock> released;
        {
            std::lock_guard lock(slabMutex);
            const auto found = std::find_if(slabBlocks.begin(), slabBlocks.end(), [](const auto& entry) { return entry.second->used == 0; });
            if (found == slabBlocks.end()) break;
            auto& slab = slabs.at(found->second->slab);
            slab.available.erase(std::find(slab.available.begin(), slab.available.end(), found->second.get()));
            --slab.blocks;
            --slab.emptyBlocks;
            released = std::move(found->second);
            slabBlocks.erase(found);
        }
        freeBlock(*released);
        bytes += SlabBlockBytes(released->slotBytes);
    }
    return bytes + trimImageBlocks();
}

void BufferPool::Put(const BufferAllocation& allocation) noexcept {
    // Evicted allocations are destroyed after the mutex is released (see evictOldest). The vector
    // may throw on growth; a Put that cannot retain simply destroys, as noexcept requires.
    std::vector<BufferAllocation> evicted;
    bool urgent = false;
    try {
        std::lock_guard lock(mutex);
        auto& tier = tierFor(allocation.bytes, allocation.properties);
        if (allocation.allocationBytes > tier.budget) {
            evicted.push_back(allocation);
        } else {
            while (tier.slots != 0 && (tier.retainedBytes + allocation.allocationBytes > tier.budget || tier.slots >= maxSlots)) evictOldest(tier, evicted);
            tier.free[{allocation.bytes, allocation.usage, allocation.properties, 0, allocation.hostPlaced}].push_back({allocation, ++clock, {}});
            ++tier.slots;
            tier.retainedBytes += allocation.allocationBytes;
            if (allocation.slab) MemoryAccounting::RetainPart(device, allocation.memory, allocation.allocationBytes, true);
            else MemoryAccounting::Retain(device, allocation.memory, allocation.allocationBytes);
        }
        if (!evicted.empty() && SpreadDestruction::Queueing()) urgent = heapOverBudget();
    } catch (...) {
        destroy(allocation);
    }
    for (const auto& gone : evicted) {
        if (urgent) destroy(gone, true);
        else retire(gone, true);
    }
}

void BufferPool::retire(const BufferAllocation& allocation, bool retained) noexcept {
    if (!SpreadDestruction::Queueing()) {
        destroy(allocation, retained);
        return;
    }
    if (allocation.slab) {
        if (retained) MemoryAccounting::RetainPart(device, allocation.memory, allocation.allocationBytes, false);
    } else {
        MemoryAccounting::Use(device, allocation.memory, "pending destruction");
    }
    auto& state = spreadState();
    bool queued = false;
    std::uint64_t before = 0;
    std::uint64_t held = 0;
    try {
        std::lock_guard lock(pendingMutex);
        if (pending.size() < maxSlots && allocation.allocationBytes <= pendingBound && pendingBytes <= pendingBound - allocation.allocationBytes) {
            pending.push_back(allocation);
            pendingBytes += allocation.allocationBytes;
            before = state.allocations.fetch_add(1);
            held = state.allocationBytes.fetch_add(allocation.allocationBytes) + allocation.allocationBytes;
            queued = true;
        }
    } catch (...) {
    }
    if (!queued) {
        state.overBound.fetch_add(1, std::memory_order_relaxed);
        state.overBoundBytes.fetch_add(allocation.allocationBytes, std::memory_order_relaxed);
        destroy(allocation);
        return;
    }
    state.queued.fetch_add(1, std::memory_order_relaxed);
    state.queuedBytes.fetch_add(allocation.allocationBytes, std::memory_order_relaxed);
    noteHigh(state.largestPending, before + 1);
    noteHigh(state.largestPendingBytes, held);
    if (before == 0) wakeDrain(state);
}

std::size_t BufferPool::DestroyPending(std::size_t count, std::chrono::steady_clock::time_point deadline) noexcept {
    auto& state = spreadState();
    std::size_t done = 0;
    while (done < count) {
        BufferAllocation allocation{};
        {
            std::lock_guard lock(pendingMutex);
            if (pending.empty()) break;
            allocation = pending.front();
            pending.pop_front();
            pendingBytes -= allocation.allocationBytes;
            state.allocations.fetch_sub(1);
            state.allocationBytes.fetch_sub(allocation.allocationBytes);
        }
        destroy(allocation);
        ++done;
        state.destroyed.fetch_add(1, std::memory_order_relaxed);
        state.destroyedBytes.fetch_add(allocation.allocationBytes, std::memory_order_relaxed);
        if (std::chrono::steady_clock::now() >= deadline) break;
    }
    return done;
}

VkDeviceSize BufferPool::DrainPending() noexcept {
    auto& state = spreadState();
    VkDeviceSize released = 0;
    for (;;) {
        BufferAllocation allocation{};
        {
            std::lock_guard lock(pendingMutex);
            if (pending.empty()) break;
            allocation = pending.front();
            pending.pop_front();
            pendingBytes -= allocation.allocationBytes;
            state.allocations.fetch_sub(1);
            state.allocationBytes.fetch_sub(allocation.allocationBytes);
        }
        released += destroy(allocation);
        state.drained.fetch_add(1, std::memory_order_relaxed);
        state.drainedBytes.fetch_add(allocation.allocationBytes, std::memory_order_relaxed);
    }
    return released;
}

std::size_t BufferPool::PendingAllocations() {
    std::lock_guard lock(pendingMutex);
    return pending.size();
}

VkDeviceSize BufferPool::PendingBytes() {
    std::lock_guard lock(pendingMutex);
    return pendingBytes;
}

void BufferPool::SetPendingBound(VkDeviceSize bytes) {
    std::lock_guard lock(pendingMutex);
    pendingBound = bytes;
}

ImageAllocation::~ImageAllocation() {
    Reset();
}

ImageAllocation::ImageAllocation(ImageAllocation&& other) noexcept {
    *this = std::move(other);
}

ImageAllocation& ImageAllocation::operator=(ImageAllocation&& other) noexcept {
    if (this == &other) return *this;
    Reset();
    owner = std::move(other.owner);
    memory = std::exchange(other.memory, VK_NULL_HANDLE);
    offset = std::exchange(other.offset, 0);
    bytes = std::exchange(other.bytes, 0);
    memoryType = std::exchange(other.memoryType, 0);
    slab = std::exchange(other.slab, false);
    dedicated = std::exchange(other.dedicated, false);
    return *this;
}

void ImageAllocation::Reset() noexcept {
    auto pool = std::move(owner);
    if (memory != VK_NULL_HANDLE) pool->releaseImageAllocation(*this);
    memory = VK_NULL_HANDLE;
    offset = 0;
    bytes = 0;
    memoryType = 0;
    slab = false;
    dedicated = false;
}

std::optional<std::uint32_t> BufferPool::imageRange(const ImageBlock& block, std::uint32_t units, std::uint32_t alignment) {
    if (block.freeUnits < units) return std::nullopt;
    const auto search = [&](std::uint32_t first, std::uint32_t stop) -> std::optional<std::uint32_t> {
        auto start = (first + alignment - 1u) & ~(alignment - 1u);
        while (start < stop && start <= imageBlockUnits - units) {
            const auto end = start + units;
            auto position = start;
            while (position < end) {
                const auto word = position / 64u;
                const auto bit = position % 64u;
                const auto count = std::min(64u - bit, end - position);
                const auto mask = (~std::uint64_t{0} >> (64u - count)) << bit;
                const auto occupied = block.occupied[word] & mask;
                if (occupied != 0) {
                    const auto after = word * 64u + 64u - static_cast<std::uint32_t>(std::countl_zero(occupied));
                    start = (after + alignment - 1u) & ~(alignment - 1u);
                    break;
                }
                position += count;
            }
            if (position == end) return start;
        }
        return std::nullopt;
    };
    if (const auto found = search(block.hint, imageBlockUnits)) return found;
    return search(0, block.hint);
}

void BufferPool::markImageRange(ImageBlock& block, std::uint32_t first, std::uint32_t units, bool occupied) noexcept {
    const auto end = first + units;
    for (auto position = first; position < end;) {
        const auto word = position / 64u;
        const auto bit = position % 64u;
        const auto count = std::min(64u - bit, end - position);
        const auto mask = (~std::uint64_t{0} >> (64u - count)) << bit;
        if (occupied) block.occupied[word] |= mask;
        else block.occupied[word] &= ~mask;
        position += count;
    }
}

void BufferPool::linkImageBlock(ImageBlock& block) noexcept {
    auto& first = imageAvailable[block.memoryType];
    block.previous = nullptr;
    block.next = first;
    if (first != nullptr) first->previous = &block;
    first = &block;
}

void BufferPool::unlinkImageBlock(ImageBlock& block) noexcept {
    if (block.previous != nullptr) block.previous->next = block.next;
    else imageAvailable[block.memoryType] = block.next;
    if (block.next != nullptr) block.next->previous = block.previous;
    block.previous = nullptr;
    block.next = nullptr;
}

ImageAllocation BufferPool::takeImageRange(const std::shared_ptr<BufferPool>& pool, std::uint32_t memoryType, VkDeviceSize bytes, VkDeviceSize alignment) {
    const auto units = static_cast<std::uint32_t>((bytes + imageUnitBytes - 1u) / imageUnitBytes);
    const auto aligned = static_cast<std::uint32_t>(std::max(alignment, imageUnitBytes) / imageUnitBytes);
    std::lock_guard lock(imageMutex);
    for (auto* block = imageAvailable[memoryType]; block != nullptr; block = block->next) {
        const auto first = imageRange(*block, units, aligned);
        if (!first) continue;
        if (block->retained) {
            block->retained = false;
            retainedImageSlabBytes.fetch_sub(imageBlockBytes, std::memory_order_relaxed);
        }
        markImageRange(*block, *first, units, true);
        block->freeUnits -= units;
        ++block->allocations;
        block->hint = *first + units;
        MemoryAccounting::RetainPart(device, block->memory, static_cast<VkDeviceSize>(units) * imageUnitBytes, false);
        if (block->freeUnits == 0) unlinkImageBlock(*block);
        ImageAllocation allocation;
        allocation.owner = pool;
        allocation.memory = block->memory;
        allocation.offset = static_cast<VkDeviceSize>(*first) * imageUnitBytes;
        allocation.bytes = bytes;
        allocation.memoryType = memoryType;
        allocation.slab = true;
        return allocation;
    }
    return {};
}

ImageAllocation BufferPool::AllocateImage(const Context& context, VkImage image, const VkImageCreateInfo& imageInfo, const char* operation, bool newImage, bool budgeted) {
    Require(image != VK_NULL_HANDLE, "an image allocation requires an image");
    constexpr auto unsupported = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT | VK_IMAGE_CREATE_SPARSE_ALIASED_BIT | VK_IMAGE_CREATE_DISJOINT_BIT | VK_IMAGE_CREATE_PROTECTED_BIT;
    Require((imageInfo.flags & unsupported) == 0, "sparse, disjoint and protected images require separate image memory binding");
    for (auto* next = static_cast<const VkBaseInStructure*>(imageInfo.pNext); next != nullptr; next = next->pNext) {
        Require(next->sType != VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO && next->sType != VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR, "external and swapchain images require separate image memory binding");
    }
    VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dedicated};
    VkImageMemoryRequirementsInfo2 imageRequirements{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
    imageRequirements.image = image;
    context.Function<PFN_vkGetImageMemoryRequirements2>("vkGetImageMemoryRequirements2")(device, &imageRequirements, &requirements);
    const auto& needed = requirements.memoryRequirements;
    Require(needed.size != 0 && std::has_single_bit(needed.alignment), "invalid image memory requirements");
    auto pool = GetBufferPool(context);
    Require(pool.get() == this, "an image allocation must use its context's buffer pool");
    const bool separate = dedicated.requiresDedicatedAllocation != VK_FALSE || dedicated.prefersDedicatedAllocation != VK_FALSE;
    static const bool disabled = std::getenv("APS5_NO_IMAGE_SLABS") != nullptr;
    const bool eligible = !disabled && !separate && imageInfo.tiling == VK_IMAGE_TILING_OPTIMAL && imageInfo.pNext == nullptr && needed.size <= (VkDeviceSize{1} << 20u) && needed.alignment < imageBlockBytes;
    const auto preferredType = context.MemoryType(needed.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (eligible) {
        if (auto existing = takeImageRange(pool, preferredType, needed.size, needed.alignment)) return existing;
        for (std::uint32_t type = 0; type < memoryProperties.memoryTypeCount; ++type) {
            if (type == preferredType || (needed.memoryTypeBits & (1u << type)) == 0 || (memoryProperties.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) continue;
            if (auto existing = takeImageRange(pool, type, needed.size, needed.alignment)) return existing;
        }
    }
    const auto memoryType = budgeted ? context.DeviceMemoryType(needed.memoryTypeBits, needed.size, operation) : preferredType;
    const auto standalone = [&] {
        ImageAllocation result;
        result.owner = pool;
        result.bytes = needed.size;
        result.memoryType = memoryType;
        result.dedicated = separate;
        VkMemoryDedicatedAllocateInfo imageAllocation{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        imageAllocation.image = image;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.pNext = separate ? &imageAllocation : nullptr;
        allocation.allocationSize = needed.size;
        allocation.memoryTypeIndex = memoryType;
        const auto fallback = context.RefusedDeviceMemoryFallback(needed.memoryTypeBits, memoryType);
        result.memory = separate ? AllocateMemory(allocation, operation, fallback, &result.memoryType, newImage) : TakeImageMemory(allocation, operation, fallback, &result.memoryType, newImage);
        return result;
    };
    if (!eligible) return standalone();
    if (auto existing = takeImageRange(pool, memoryType, needed.size, needed.alignment)) return existing;
    if (context.memoryProperties2 != nullptr && context.physical != VK_NULL_HANDLE) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT reported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &reported};
        context.memoryProperties2(context.physical, &properties);
        const auto heap = memoryProperties.memoryTypes[memoryType].heapIndex;
        const auto budget = reported.heapBudget[heap];
        const auto usage = reported.heapUsage[heap];
        if (budget != 0 && (usage >= budget || imageBlockBytes > budget - usage)) return standalone();
    }
    const auto dispose = [this](ImageBlock* block) noexcept {
        if (block->memory != VK_NULL_HANDLE) MemoryAccounting::Free(freeMemory, device, block->memory);
        delete block;
    };
    std::unique_ptr<ImageBlock, decltype(dispose)> block(new ImageBlock, dispose);
    block->memoryType = memoryType;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = imageBlockBytes;
    allocation.memoryTypeIndex = memoryType;
    ImageAllocation reused;
    const auto reuse = [&](std::uint32_t type) {
        reused = takeImageRange(pool, type, needed.size, needed.alignment);
        return static_cast<bool>(reused);
    };
    try {
        block->memory = allocateMemoryWithReuse(allocation, operation, std::nullopt, &block->memoryType, newImage, reuse);
    } catch (const MemoryAllocationError& error) {
        if (error.result != VK_ERROR_OUT_OF_DEVICE_MEMORY && error.result != VK_ERROR_OUT_OF_HOST_MEMORY && error.result != VK_ERROR_TOO_MANY_OBJECTS) throw;
        return standalone();
    }
    if (reused) return reused;
    if (auto existing = takeImageRange(pool, block->memoryType, needed.size, needed.alignment)) return existing;
    const auto units = static_cast<std::uint32_t>((needed.size + imageUnitBytes - 1u) / imageUnitBytes);
    markImageRange(*block, 0, units, true);
    block->freeUnits -= units;
    block->allocations = 1;
    block->hint = units;
    MemoryAccounting::Retain(device, block->memory, imageBlockBytes - static_cast<VkDeviceSize>(units) * imageUnitBytes);
    ImageAllocation result;
    {
        std::lock_guard lock(imageMutex);
        auto* created = block.get();
        const auto entry = imageBlocks.try_emplace(created->memory).first;
        entry->second.reset(block.release());
        linkImageBlock(*created);
        result.owner = pool;
        result.memory = created->memory;
        result.memoryType = created->memoryType;
        result.bytes = needed.size;
        result.slab = true;
    }
    return result;
}

void BufferPool::releaseImageAllocation(const ImageAllocation& allocation) noexcept {
    if (!allocation.slab) {
        if (allocation.dedicated) {
            MemoryAccounting::Free(freeMemory, device, allocation.memory);
            imageMemoryReleasedBytes += allocation.bytes;
        }
        else PutImageMemory(allocation.memory, allocation.bytes, allocation.memoryType);
        return;
    }
    std::unique_ptr<ImageBlock> released;
    {
        std::lock_guard budgetLock(mutex);
        std::lock_guard lock(imageMutex);
        const auto found = imageBlocks.find(allocation.memory);
        if (found == imageBlocks.end()) std::terminate();
        auto& block = *found->second;
        if (block.freeUnits == 0) linkImageBlock(block);
        const auto first = static_cast<std::uint32_t>(allocation.offset / imageUnitBytes);
        const auto units = static_cast<std::uint32_t>((allocation.bytes + imageUnitBytes - 1u) / imageUnitBytes);
        markImageRange(block, first, units, false);
        block.freeUnits += units;
        block.hint = std::min(block.hint, first);
        MemoryAccounting::RetainPart(device, block.memory, static_cast<VkDeviceSize>(units) * imageUnitBytes, true);
        if (--block.allocations != 0) return;
        imageMemoryReleasedBytes += imageBlockBytes;
        if (retainedImageSlabBytes.load(std::memory_order_relaxed) == 0 && imageBudget >= imageBlockBytes && imageBytes <= imageBudget - imageBlockBytes && !heapOverBudget()) {
            block.retained = true;
            retainedImageSlabBytes.store(imageBlockBytes, std::memory_order_relaxed);
            return;
        }
        unlinkImageBlock(block);
        released = std::move(found->second);
        imageBlocks.erase(found);
    }
    MemoryAccounting::Free(freeMemory, device, released->memory);
}

VkDeviceSize BufferPool::trimImageBlocks() {
    VkDeviceSize bytes = 0;
    for (;;) {
        std::unique_ptr<ImageBlock> released;
        {
            std::lock_guard lock(imageMutex);
            const auto found = std::find_if(imageBlocks.begin(), imageBlocks.end(), [](const auto& entry) { return entry.second->allocations == 0; });
            if (found == imageBlocks.end()) break;
            auto& block = *found->second;
            unlinkImageBlock(block);
            if (block.retained) retainedImageSlabBytes.fetch_sub(imageBlockBytes, std::memory_order_relaxed);
            released = std::move(found->second);
            imageBlocks.erase(found);
        }
        MemoryAccounting::Free(freeMemory, device, released->memory);
        bytes += imageBlockBytes;
    }
    return bytes;
}

std::size_t BufferPool::ImageSlabBlocks() {
    std::lock_guard lock(imageMutex);
    return imageBlocks.size();
}

ImageAllocation AllocateImageMemory(const Context& context, VkImage image, const VkImageCreateInfo& imageInfo, const char* operation, bool newImage, bool budgeted) {
    return GetBufferPool(context)->AllocateImage(context, image, imageInfo, operation, newImage, budgeted);
}

std::uint64_t ImageMemoryReleasedBytes() noexcept {
    return imageMemoryReleasedBytes;
}

std::shared_ptr<BufferPool> GetBufferPool(const Context& context) {
    if (!context.bufferPool) context.bufferPool = std::make_shared<BufferPool>(context);
    return context.bufferPool;
}

VkDeviceMemory AllocateImageMemory(const Context& context, const VkMemoryAllocateInfo& allocation, std::uint32_t mask, const char* operation, std::uint32_t* allocatedType, bool newImage) {
    return GetBufferPool(context)->TakeImageMemory(allocation, operation, context.RefusedDeviceMemoryFallback(mask, allocation.memoryTypeIndex), allocatedType, newImage);
}

void ReleaseImageMemory(const Context& context, VkDeviceMemory memory, VkDeviceSize bytes, std::uint32_t memoryType) noexcept {
    if (memory == VK_NULL_HANDLE) return;
    if (context.bufferPool) {
        context.bufferPool->PutImageMemory(memory, bytes, memoryType);
        return;
    }
    MemoryAccounting::Free(context, memory);
    imageMemoryReleasedBytes += bytes;
}

}
